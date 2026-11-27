use std::sync::Arc;

use haste_codegen::Code;
use rustc_hash::FxHashMap;

use crate::image::aot::{AotCode, AotFn, CachedAotTemplate};
use crate::image::stencil::{
    DictConfig, DistinctInst, GroupWork, aot_relocs_valid, item_err, mode_dict_key,
};
use crate::image::{FrozenStencil, Image, Pass};
use crate::lower::{self, FootprintId, RelocPlan, SymbolSource};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::substrate::tally::{Tally, clock, tally};

struct PartitionCx<'a, 'b> {
    cx: &'a mut Context<'b>,
    reached: &'a Reached,
}

#[derive(Clone, Copy)]
struct PartitionGroup<'a> {
    item: ItemRef,
    footprint: FootprintId,
    rep: usize,
    rest: &'a [usize],
    indices: &'a [usize],
}

impl PartitionGroup<'_> {
    fn cache_key(self) -> (ItemRef, FootprintId, usize) {
        (self.item, self.footprint, self.rep)
    }
}

#[derive(PartialEq, Eq, Hash)]
enum InstanceKey {
    Relocs(Vec<SymbolSource>),
    Syms(Vec<u64>),
}

impl Image {
    pub(in crate::image) fn compile_partitioned(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        work: &[usize],
        cfg: DictConfig,
        record_cache: bool,
        pass: &mut Pass,
    ) -> Vec<(usize, String)> {
        let mut errors = Vec::new();
        let aot_fast = self.backend.is_aot() && !cfg.verify;
        let has_bundle = aot_fast && self.aot_cache().is_some_and(|c| c.has_loaded_bundle());
        let remaining = if aot_fast {
            self.drain_aot_cache_hits(cx, reached, work, pass)
        } else {
            work.to_vec()
        };
        if remaining.is_empty() {
            return errors;
        }
        let (order, mut by_fp) =
            self.partition_work(cx, reached, &remaining, has_bundle, &mut errors);
        let mut verify_hits = Vec::new();
        for (item, footprint) in order {
            let Some(indices) = by_fp.remove(&(item, footprint)) else {
                continue;
            };
            if record_cache
                && let Some(reloc_plan) = pass
                    .stencils
                    .templates
                    .get(&(item, footprint))
                    .map(|t| Arc::clone(t.reloc_plan()))
            {
                for index in indices {
                    let relocs = match lower::eval_relocs(cx, reached, self, &reloc_plan, index) {
                        Ok(r) => r,
                        Err(err) => {
                            errors.push((index, item_err(cx, item, &err)));
                            continue;
                        }
                    };
                    let symbols = self.symbols(&relocs);
                    if let Some(hit) = self.reuse_cached(
                        (item, footprint),
                        index,
                        &symbols,
                        self.backend.is_aot().then_some(&relocs),
                        pass,
                    ) && cfg.verify
                    {
                        verify_hits.push((index, hit));
                    }
                }
                continue;
            }
            if aot_fast && let Some(cached) = self.cached_aot_template(item, footprint) {
                errors.extend(self.apply_cached_aot_template(
                    &mut PartitionCx { cx, reached },
                    item,
                    &indices,
                    &cached,
                    pass,
                ));
                continue;
            }
            let Some((&rep, rest)) = indices.split_first() else {
                continue;
            };
            let group_meta = PartitionGroup {
                item,
                footprint,
                rep,
                rest,
                indices: &indices,
            };
            let skip_frozen = footprint == FootprintId(((rep as u128) << 1) | 1);
            let frozen = if skip_frozen {
                false
            } else {
                let lookup_clock = std::time::Instant::now();
                let hit = self.try_frozen_partitioned(
                    &mut PartitionCx { cx, reached },
                    group_meta,
                    cfg,
                    pass,
                );
                clock(Tally::LookupUs, lookup_clock);
                hit
            };
            if frozen {
                continue;
            }
            let lower_clock = std::time::Instant::now();
            let lowered = lower::lower_stencil(cx, reached, self, rep);
            clock(Tally::LowerUs, lower_clock);
            let out = match lowered {
                Ok(out) => out,
                Err(err) => {
                    let msg = item_err(cx, item, &err);
                    errors.extend(indices.iter().map(|&idx| (idx, msg.clone())));
                    continue;
                }
            };
            let distinct = self.group_instances(
                &mut PartitionCx { cx, reached },
                group_meta,
                &out,
                aot_fast,
                &mut errors,
            );
            let leader_plan = Arc::clone(&out.reloc_plan);
            let mut group = GroupWork {
                item,
                footprint,
                distinct,
                leader_out: Some(out),
            };
            if let Err(err) = self.compile_group(cx, reached, &mut group, cfg, record_cache, pass) {
                errors.extend(group.all_indices().map(|idx| (idx, err.clone())));
            } else {
                let plan = self
                    .cached_aot_template(item, footprint)
                    .map(|c| Arc::clone(c.reloc_plan()))
                    .unwrap_or(leader_plan);
                self.record_aot_inlined_indices(cx, reached, &plan, &indices);
            }
            drop(group);
            self.bodies.trim_transient();
        }
        if cfg.verify {
            errors.extend(self.verify_records(cx, reached, &verify_hits));
        }
        errors
    }

    fn apply_cached_aot_template(
        &self,
        env: &mut PartitionCx<'_, '_>,
        item: ItemRef,
        indices: &[usize],
        cached: &CachedAotTemplate,
        pass: &mut Pass,
    ) -> Vec<(usize, String)> {
        let mut errors = Vec::new();
        let mut evaluated = Vec::with_capacity(indices.len());
        for &index in indices {
            match lower::eval_relocs(env.cx, env.reached, self, cached.reloc_plan(), index) {
                Ok(relocs) => evaluated.push((index, relocs)),
                Err(err) => errors.push((index, item_err(env.cx, item, &err))),
            }
        }
        if !env.cx.deferred_evals.is_empty() {
            return errors;
        }
        self.commit_aot_groups(&evaluated, cached, 0, pass);
        self.record_aot_inlined_evals(env.cx, env.reached, cached.reloc_plan(), &evaluated);
        errors
    }

    fn try_direct_frozen_slice(
        &self,
        env: &mut PartitionCx<'_, '_>,
        group: PartitionGroup<'_>,
        slice: &crate::substrate::StencilSlice<'_>,
        pass: &mut Pass,
    ) -> Result<(), Tally> {
        if self.backend.is_check() {
            let len = slice.code().len() as u32;
            for &idx in group.indices {
                pass.record_size(idx, len, 0);
            }
            return Ok(());
        }
        if !self.backend.is_aot() {
            return Err(Tally::MissOther);
        }
        let Ok(code) = slice.to_code().map(Arc::new) else {
            return Err(Tally::MissOther);
        };
        if code.relocs().iter().all(|r| r.symbol_index().is_none())
            && code.unwind_drops().is_empty()
        {
            let len = code.bytes().len() as u32;
            let aot_code = AotCode::from_code(&code);
            self.cache_aot_template(
                env.cx,
                env.reached,
                group.cache_key(),
                CachedAotTemplate::Direct {
                    code: Arc::clone(&aot_code),
                    reloc_plan: Arc::default(),
                },
            );
            self.record_aot(
                group.rep,
                AotFn::Direct {
                    code: aot_code,
                    relocs: Vec::new(),
                },
                group.rest,
            );
            for &idx in group.indices {
                pass.record_size(idx, len, 0);
            }
            return Ok(());
        }
        let Some(reloc_plan) = self
            .decode_aot_reloc_plan(
                env.cx.outline,
                &env.cx.types,
                env.reached,
                false,
                slice.plan(),
            )
            .map(Arc::new)
        else {
            return Err(Tally::MissPlan);
        };
        if aot_relocs_valid(&code, reloc_plan.reloc_len())
            && self.try_commit_frozen_template(
                env,
                group,
                CachedAotTemplate::Direct {
                    code: AotCode::from_code(&code),
                    reloc_plan,
                },
                0,
                pass,
            )
        {
            return Ok(());
        }
        Err(Tally::MissOther)
    }

    fn try_dict_frozen_slice(
        &self,
        env: &mut PartitionCx<'_, '_>,
        group: PartitionGroup<'_>,
        slice: &crate::substrate::StencilSlice<'_>,
        pass: &mut Pass,
    ) -> Result<(), Tally> {
        let Ok(code) = slice.to_code().map(Arc::new) else {
            return Err(Tally::MissOther);
        };
        let Some(reloc_plan) = self
            .decode_aot_reloc_plan(
                env.cx.outline,
                &env.cx.types,
                env.reached,
                false,
                slice.plan(),
            )
            .map(Arc::new)
        else {
            return Err(Tally::MissPlan);
        };
        let dict_len = code.bytes().len() as u32;
        if self.try_commit_frozen_template(
            env,
            group,
            CachedAotTemplate::Dict {
                body: AotCode::from_code(&code),
                reloc_plan,
            },
            dict_len,
            pass,
        ) {
            Ok(())
        } else {
            Err(Tally::MissOther)
        }
    }

    fn try_frozen_partitioned(
        &self,
        env: &mut PartitionCx<'_, '_>,
        group: PartitionGroup<'_>,
        cfg: DictConfig,
        pass: &mut Pass,
    ) -> bool {
        if cfg.verify {
            return false;
        }
        tally(Tally::Lookup);
        let Some((_, direct_key, fp_bits)) =
            self.frozen_stencil_key(env.cx, env.reached, group.rep, group.footprint)
        else {
            tally(if group.footprint.is_portable() {
                Tally::MissUnkeyed
            } else {
                Tally::MissUnportable
            });
            return false;
        };
        let Some(substrate) = env.cx.store().substrate().cloned() else {
            tally(Tally::MissOther);
            return false;
        };
        let prefer_dict =
            self.backend.is_aot() && cfg.min_bytes == 0 && group.indices.len() >= cfg.min_fanout;
        let mut reason = Tally::MissAbsent;
        if let Some(slice) = (!prefer_dict)
            .then(|| substrate.stencil(direct_key, fp_bits))
            .flatten()
        {
            match self.try_direct_frozen_slice(env, group, &slice, pass) {
                Ok(()) => {
                    tally(Tally::Hit);
                    return true;
                }
                Err(why) => reason = why,
            }
        }
        let dict_slice = (self.backend.is_aot() && group.indices.len() >= cfg.min_fanout)
            .then(|| substrate.stencil(mode_dict_key(direct_key), fp_bits))
            .flatten()
            .filter(|slice| prefer_dict || slice.code().len() > cfg.min_bytes);
        if let Some(slice) = dict_slice {
            match self.try_dict_frozen_slice(env, group, &slice, pass) {
                Ok(()) => {
                    tally(Tally::Hit);
                    return true;
                }
                Err(why) => reason = why,
            }
        }
        tally(reason);
        false
    }

    fn try_commit_frozen_template(
        &self,
        env: &mut PartitionCx<'_, '_>,
        group: PartitionGroup<'_>,
        cached: CachedAotTemplate,
        dict_len: u32,
        pass: &mut Pass,
    ) -> bool {
        let mut evaluated = Vec::with_capacity(group.indices.len());
        for &index in group.indices {
            let Ok(relocs) =
                lower::eval_relocs(env.cx, env.reached, self, cached.reloc_plan(), index)
            else {
                return !env.cx.deferred_evals.is_empty();
            };
            evaluated.push((index, relocs));
        }
        if !env.cx.deferred_evals.is_empty() {
            return true;
        }
        self.commit_aot_groups(&evaluated, &cached, dict_len, pass);
        self.record_aot_inlined_evals(env.cx, env.reached, cached.reloc_plan(), &evaluated);
        self.cache_aot_template(env.cx, env.reached, group.cache_key(), cached);
        true
    }

    fn commit_aot_groups(
        &self,
        evaluated: &[(usize, Vec<SymbolSource>)],
        cached: &CachedAotTemplate,
        dict_len: u32,
        pass: &mut Pass,
    ) {
        let mut reloc_pos: FxHashMap<&[SymbolSource], usize> = FxHashMap::default();
        let mut groups: Vec<(usize, Vec<usize>, Vec<SymbolSource>)> = Vec::new();
        let code_len = match cached {
            CachedAotTemplate::Direct { code, .. } => code.bytes().len() as u32,
            CachedAotTemplate::Dict { .. } => 0,
        };
        for (i, (index, relocs)) in evaluated.iter().enumerate() {
            if let Some(&pos) = reloc_pos.get(relocs.as_slice()) {
                groups[pos].1.push(*index);
            } else {
                reloc_pos.insert(relocs.as_slice(), groups.len());
                groups.push((*index, Vec::new(), relocs.clone()));
            }
            let size = if i == 0 && dict_len > 0 {
                dict_len
            } else {
                code_len
            };
            pass.record_size(*index, size, 0);
        }
        for (leader, followers, relocs) in groups {
            self.register_aot_relocs(&relocs);
            let entry = match cached {
                CachedAotTemplate::Direct { code, .. } => AotFn::Direct {
                    code: Arc::clone(code),
                    relocs,
                },
                CachedAotTemplate::Dict { body, .. } => AotFn::Dict {
                    body: Arc::clone(body),
                    relocs,
                },
            };
            self.record_aot(leader, entry, &followers);
        }
    }

    pub(in crate::image) fn try_frozen_dict_group(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        group: &GroupWork,
        pass: &mut Pass,
    ) -> Option<Result<(), String>> {
        let rep = group.representative();
        let (_, direct_key, fp_bits) =
            self.frozen_stencil_key(cx, reached, rep, group.footprint)?;
        let slice = cx
            .store()
            .substrate()?
            .stencil(mode_dict_key(direct_key), fp_bits)?;
        let code = Arc::new(slice.to_code().ok()?);
        let reloc_plan = Arc::new(self.decode_aot_reloc_plan(
            cx.outline,
            &cx.types,
            reached,
            false,
            slice.plan(),
        )?);
        let mut evaluated = Vec::with_capacity(group.distinct.len());
        for inst in &group.distinct {
            let relocs = lower::eval_relocs(cx, reached, self, &reloc_plan, inst.leader).ok()?;
            if !cx.deferred_evals.is_empty() {
                return Some(Ok(()));
            }
            evaluated.push((inst.leader, &inst.followers, relocs));
        }
        let body = AotCode::from_code(&code);
        self.record_aot_inlined_groups(cx, reached, &reloc_plan, &evaluated);
        for (leader, followers, relocs) in evaluated {
            self.register_aot_relocs(&relocs);
            self.record_aot(
                leader,
                AotFn::Dict {
                    body: Arc::clone(&body),
                    relocs,
                },
                followers,
            );
            pass.record_size(leader, 0, 0);
        }
        pass.record_size(rep, code.bytes().len() as u32, 0);
        self.cache_aot_template(
            cx,
            reached,
            (group.item, group.footprint, rep),
            CachedAotTemplate::Dict { body, reloc_plan },
        );
        tally(Tally::HitLowered);
        Some(Ok(()))
    }

    fn encode_frozen_plan(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        is_sysroot: bool,
        plan: Option<&RelocPlan>,
    ) -> Option<(bool, Vec<u8>)> {
        let p = plan?;
        if let Some(bytes) =
            self.encode_aot_reloc_plan(cx.outline, &cx.types, reached, is_sysroot, p)
        {
            return Some((is_sysroot, bytes));
        }
        is_sysroot
            .then(|| self.encode_aot_reloc_plan(cx.outline, &cx.types, reached, false, p))
            .flatten()
            .map(|bytes| (false, bytes))
    }

    pub(in crate::image) fn freeze_stencil(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        index: usize,
        fp: FootprintId,
        is_dict: bool,
        code: &Arc<Code>,
        plan: Option<&RelocPlan>,
        pass: &mut Pass,
    ) {
        let valid_relocs = if is_dict {
            code.relocs().iter().all(|r| r.as_runtime().is_some())
        } else {
            code.relocs().iter().all(|r| {
                r.symbol_index().is_some()
                    || ((self.backend.is_aot() || self.backend.is_lazy_jit())
                        && r.as_runtime().is_some())
            })
        };
        if !cx.deferred_evals.is_empty() {
            return tally(Tally::FreezeDeferred);
        }
        let Some((is_sysroot, direct_key, fp_bits)) = valid_relocs
            .then(|| self.frozen_stencil_key(cx, reached, index, fp))
            .flatten()
        else {
            return tally(Tally::FreezeSkipped);
        };
        let needs_plan = is_dict
            || code.relocs().iter().any(|r| r.symbol_index().is_some())
            || !code.unwind_drops().is_empty();
        let (store_sysroot, plan_bytes) =
            if (self.backend.is_aot() || self.backend.is_lazy_jit()) && needs_plan {
                let Some(pair) = self.encode_frozen_plan(cx, reached, is_sysroot, plan) else {
                    return tally(if is_dict {
                        Tally::FreezeSkipped
                    } else {
                        Tally::FreezePlanless
                    });
                };
                pair
            } else {
                (is_sysroot, Vec::new())
            };
        let key = if is_dict {
            mode_dict_key(direct_key)
        } else {
            direct_key
        };
        pass.frozen_stencils.push(FrozenStencil {
            is_sysroot: store_sysroot,
            key,
            fp: fp_bits,
            code: Arc::clone(code),
            plan: plan_bytes,
        });
    }

    fn group_instances(
        &self,
        env: &mut PartitionCx<'_, '_>,
        group: PartitionGroup<'_>,
        out: &lower::StencilOutput,
        aot_fast: bool,
        errors: &mut Vec<(usize, String)>,
    ) -> Vec<DistinctInst> {
        let mut sym_pos: FxHashMap<InstanceKey, usize> = FxHashMap::default();
        let mut distinct: Vec<DistinctInst> = Vec::new();
        let (rep_key, rep_inst) = self.build_inst(group.rep, out.relocs.clone(), aot_fast);
        sym_pos.insert(rep_key, 0);
        distinct.push(rep_inst);
        for &index in group.rest {
            let relocs = match lower::eval_relocs(env.cx, env.reached, self, &out.reloc_plan, index)
            {
                Ok(r) => r,
                Err(err) => {
                    errors.push((index, item_err(env.cx, group.item, &err)));
                    continue;
                }
            };
            let (key, inst) = self.build_inst(index, relocs, aot_fast);
            if let Some(&pos) = sym_pos.get(&key) {
                distinct[pos].followers.push(index);
            } else {
                sym_pos.insert(key, distinct.len());
                distinct.push(inst);
            }
        }
        distinct
    }

    fn build_inst(
        &self,
        index: usize,
        relocs: Vec<SymbolSource>,
        aot_fast: bool,
    ) -> (InstanceKey, DistinctInst) {
        if aot_fast {
            self.register_aot_relocs(&relocs);
            let inst = DistinctInst {
                leader: index,
                followers: Vec::new(),
                symbols: Vec::new(),
                relocs: Some(relocs.clone()),
            };
            (InstanceKey::Relocs(relocs), inst)
        } else {
            let symbols = self.symbols(&relocs);
            let key = InstanceKey::Syms(symbols.clone());
            let inst = DistinctInst {
                leader: index,
                followers: Vec::new(),
                symbols,
                relocs: self.backend.is_aot().then_some(relocs),
            };
            (key, inst)
        }
    }

    fn partition_work(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        remaining: &[usize],
        has_bundle: bool,
        errors: &mut Vec<(usize, String)>,
    ) -> (
        Vec<(ItemRef, FootprintId)>,
        FxHashMap<(ItemRef, FootprintId), Vec<usize>>,
    ) {
        let mut order: Vec<(ItemRef, FootprintId)> = Vec::new();
        let mut by_fp: FxHashMap<(ItemRef, FootprintId), Vec<usize>> = FxHashMap::default();
        let need_fp: Vec<usize> = if has_bundle {
            let mut item_order: Vec<ItemRef> = Vec::new();
            let mut by_item: FxHashMap<ItemRef, Vec<usize>> = FxHashMap::default();
            for &index in remaining {
                let item = reached.list[index].0.0;
                by_item
                    .entry(item)
                    .or_insert_with(|| {
                        item_order.push(item);
                        Vec::new()
                    })
                    .push(index);
            }
            let mut rest = Vec::new();
            for item in item_order {
                let Some(indices) = by_item.remove(&item) else {
                    continue;
                };
                if indices.len() == 1 {
                    let fp = FootprintId(((indices[0] as u128) << 1) | 1);
                    order.push((item, fp));
                    by_fp.insert((item, fp), indices);
                } else {
                    rest.extend(indices);
                }
            }
            rest
        } else {
            remaining.to_vec()
        };
        if !need_fp.is_empty() {
            let footprint_clock = std::time::Instant::now();
            for index in need_fp {
                let item = reached.list[index].0.0;
                let footprint = match lower::eval_footprint(cx, reached, self, index) {
                    Ok(fp) => fp,
                    Err(err) => {
                        errors.push((index, item_err(cx, item, &err)));
                        continue;
                    }
                };
                by_fp
                    .entry((item, footprint))
                    .or_insert_with(|| {
                        order.push((item, footprint));
                        Vec::new()
                    })
                    .push(index);
            }
            clock(Tally::FootprintUs, footprint_clock);
        }
        (order, by_fp)
    }
}
