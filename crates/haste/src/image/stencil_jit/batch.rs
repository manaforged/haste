use std::sync::Arc;

use anyhow::{Result, bail};
use haste_codegen::Code;

use crate::image::stencil::aot_relocs_valid;
use crate::image::workers::{bucket_work, lpt_shards};
use crate::image::{Image, Pass};
use crate::lower::{self, FootprintId, RelocPlan, RelocVec, Slots};
use crate::sema::Context;
use crate::sema::reach::Reached;

use super::{Lowered, Ready};

impl Image {
    pub(in crate::image) fn compile_batch(
        &mut self,
        cx: &mut Context<'_>,
        reached: &Reached,
        work: &[usize],
    ) -> Vec<(usize, usize)> {
        let mut work = work.to_vec();
        work.sort_unstable();
        let buckets = bucket_work(reached, &work);
        self.pass.ready = if buckets.len() <= 1 || crate::jobs() <= 1 {
            buckets
                .iter()
                .flat_map(|bucket| self.ready_bucket(cx, reached, bucket))
                .collect()
        } else {
            let shards = lpt_shards(buckets, crate::jobs());
            let forks = shards.iter().map(|_| cx.fork()).collect();
            let made = self.run_shards(forks, &shards, |mut fork, shard| {
                shard
                    .iter()
                    .flat_map(|bucket| self.ready_bucket(&mut fork, reached, bucket))
                    .collect::<Vec<_>>()
            });
            made.into_iter().filter_map(Result::ok).flatten().collect()
        };
        work.retain(|index| self.pass.ready.contains_key(index));
        let compiled = work
            .into_iter()
            .filter_map(|index| {
                self.compile_one(cx, reached, index)
                    .ok()
                    .map(|(code, _)| (index, code))
            })
            .collect();
        self.pass.ready.clear();
        self.take_wanted();
        compiled
    }

    pub(super) fn thaw_lazy_stencil(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        fp: FootprintId,
        wanted0: usize,
    ) -> Result<Option<Lowered>> {
        if !self.backend.is_lazy_jit() {
            return Ok(None);
        }
        let Some((_, key, fp_bits)) = self.frozen_stencil_key(cx, reached, index, fp) else {
            return Ok(None);
        };
        let Some(substrate) = cx.store().substrate().cloned() else {
            return Ok(None);
        };
        let Some(slice) = substrate.stencil(key, fp_bits) else {
            return Ok(None);
        };
        let Ok(code) = slice.to_code().map(Arc::new) else {
            return Ok(None);
        };
        if code.relocs().iter().all(|r| r.symbol_index().is_none())
            && code.unwind_drops().is_empty()
        {
            return Ok(Some((code, RelocVec::new(), Arc::default())));
        }
        let Some(plan) = self
            .decode_lazy_reloc_plan(cx, reached, slice.plan())
            .map(Arc::new)
        else {
            if self.wanted_len() > wanted0 {
                bail!("unreached inline callee");
            }
            return Ok(None);
        };
        if !aot_relocs_valid(&code, plan.reloc_len()) {
            return Ok(None);
        }
        let relocs = match lower::eval_relocs(cx, reached, self, &plan, index) {
            Ok(r) => r,
            Err(_) if self.wanted_len() > wanted0 => bail!("unreached inline callee"),
            Err(_) => return Ok(None),
        };
        if self.wanted_len() > wanted0 {
            bail!("unreached inline callee");
        }
        Ok(Some((code, relocs, plan)))
    }

    fn ready_bucket(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        bucket: &[usize],
    ) -> Vec<(usize, Ready)> {
        let mut plans: rustc_hash::FxHashMap<FootprintId, Arc<RelocPlan>> =
            rustc_hash::FxHashMap::default();
        let mut made = Vec::new();
        for &index in bucket {
            let item = reached.list[index].0.0;
            let footprint = self.quietly(cx, |cx| {
                lower::eval_footprint(cx, reached, self, index).ok()
            });
            let plan = footprint.and_then(|fp| {
                self.pass
                    .stencils
                    .jit_templates
                    .get(&(item, fp))
                    .map(|template| Arc::clone(&template.reloc_plan))
                    .or_else(|| plans.get(&fp).cloned())
            });
            let ready = match plan {
                Some(plan) => Ready {
                    footprint,
                    code: None,
                    relocs: self
                        .quietly(cx, |cx| {
                            lower::eval_relocs(cx, reached, self, &plan, index).ok()
                        })
                        .map(|relocs| (plan, relocs)),
                    thawed: false,
                },
                None => {
                    let wanted_before = self.wanted_len();
                    let thawed = footprint.and_then(|fp| {
                        self.quietly(cx, |cx| {
                            self.thaw_lazy_stencil(cx, reached, index, fp, wanted_before)
                                .ok()
                                .flatten()
                        })
                    });
                    match thawed {
                        Some((code, relocs, plan)) => {
                            if let Some(fp) = footprint {
                                plans.insert(fp, Arc::clone(&plan));
                            }
                            Ready {
                                footprint,
                                code: Some(code),
                                relocs: Some((plan, relocs)),
                                thawed: true,
                            }
                        }
                        None if self.wanted_len() > wanted_before => continue,
                        None => match self
                            .quietly(cx, |cx| lower::lower_stencil(cx, reached, self, index).ok())
                            .and_then(|out| {
                                Some((
                                    super::super::stencil::compile_traced(
                                        cx,
                                        reached,
                                        index,
                                        out.func,
                                        &self.compile_target_for(cx, item),
                                    )
                                    .ok()?,
                                    out.relocs,
                                    out.reloc_plan,
                                ))
                            }) {
                            Some((code, relocs, plan)) => {
                                if let Some(fp) = footprint {
                                    plans.insert(fp, Arc::clone(&plan));
                                }
                                Ready {
                                    footprint,
                                    code: Some(Arc::new(code)),
                                    relocs: Some((plan, relocs)),
                                    thawed: false,
                                }
                            }
                            None => continue,
                        },
                    }
                }
            };
            if ready.relocs.is_some() {
                made.push((index, ready));
            }
        }
        self.bodies.trim_transient();
        made
    }

    fn quietly<'a, T>(
        &self,
        cx: &mut Context<'a>,
        run: impl FnOnce(&mut Context<'a>) -> Option<T>,
    ) -> Option<T> {
        let mark = (self.wanted_len(), cx.quiet_mark());
        run(cx).filter(|_| (self.wanted_len(), cx.quiet_mark()) == mark)
    }

    pub(super) fn footprint(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        pass: &Pass,
    ) -> Result<FootprintId> {
        match pass.ready.get(&index).and_then(|ready| ready.footprint) {
            Some(footprint) => Ok(footprint),
            None => lower::eval_footprint(cx, reached, self, index),
        }
    }

    pub(super) fn relocs(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        plan: &Arc<RelocPlan>,
        index: usize,
        pass: &Pass,
    ) -> Result<RelocVec> {
        match pass
            .ready
            .get(&index)
            .and_then(|ready| ready.relocs.as_ref())
            .filter(|(made, _)| Arc::ptr_eq(made, plan))
        {
            Some((_, relocs)) => Ok(relocs.clone()),
            None => lower::eval_relocs(cx, reached, self, plan, index),
        }
    }

    pub(super) fn lowered(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        wanted0: usize,
        pass: &Pass,
    ) -> Result<(Lowered, bool)> {
        if let Some(Ready {
            code: Some(code),
            relocs: Some((plan, relocs)),
            thawed,
            ..
        }) = pass.ready.get(&index)
        {
            return Ok((
                (Arc::clone(code), relocs.clone(), Arc::clone(plan)),
                *thawed,
            ));
        }
        let out = lower::lower_stencil(cx, reached, self, index)?;
        if self.wanted_len() > wanted0 {
            bail!("unreached inline callee");
        }
        Ok((
            (
                Arc::new(super::super::stencil::compile_traced(
                    cx,
                    reached,
                    index,
                    out.func,
                    &self.compile_target_for(cx, reached.list[index].0.0),
                )?),
                out.relocs,
                out.reloc_plan,
            ),
            false,
        ))
    }

    pub(super) fn freeze_lazy(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        index: usize,
        fp: FootprintId,
        (code, plan): (&Arc<Code>, &RelocPlan),
        pass: &mut Pass,
    ) {
        if self.backend.is_lazy_jit()
            && !pass.stencils.cfg.verify
            && cx.outline.is_fixed(reached.list[index].0.0.krate)
        {
            self.freeze_stencil(cx, reached, index, fp, false, code, Some(plan), pass);
        }
    }
}
