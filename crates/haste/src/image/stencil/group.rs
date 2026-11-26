use std::sync::Arc;

use anyhow::{Result, bail};
use haste_codegen::Code;

use super::{
    CachedTemplate, DICT_STUB_SPACE, DictConfig, DistinctInst, GroupWork, VerifyRecord,
    aot_relocs_valid, compile_traced, item_err, target,
};
use crate::image::aot::{AotCode, AotFn, CachedAotTemplate};
use crate::image::{Image, Pass};
use crate::lower::{self, FootprintId, RelocPlan, Slots};
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::substrate::tally::{Tally, tally};

impl Image {
    fn apply_frozen_aot_group(
        &self,
        (cx, reached): (&mut Context<'_>, &Reached),
        group: &GroupWork,
        rep: usize,
        slice: &crate::substrate::StencilSlice<'_>,
        pass: &mut Pass,
    ) -> Option<Result<(), String>> {
        let code = Arc::new(slice.to_code().ok()?);
        if !group.distinct.iter().all(|inst| {
            inst.relocs
                .as_ref()
                .is_some_and(|r| aot_relocs_valid(&code, r.len()))
        }) {
            return None;
        }
        let aot_code = AotCode::from_code(&code);
        if let Some(out) = &group.leader_out {
            self.cache_aot_template(
                cx,
                reached,
                (group.item, group.footprint, rep),
                CachedAotTemplate::Direct {
                    code: Arc::clone(&aot_code),
                    reloc_plan: Arc::clone(&out.reloc_plan),
                },
            );
        }
        let code_len = code.bytes().len() as u32;
        for inst in &group.distinct {
            if let Some(relocs) = &inst.relocs {
                self.record_aot(
                    inst.leader,
                    AotFn::Direct {
                        code: Arc::clone(&aot_code),
                        relocs: relocs.clone(),
                    },
                    &inst.followers,
                );
            }
            pass.record_size(inst.leader, code_len, 0);
        }
        tally(Tally::HitLowered);
        Some(Ok(()))
    }

    fn try_frozen_group(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        group: &GroupWork,
        cfg: DictConfig,
        pass: &mut Pass,
    ) -> Option<Result<(), String>> {
        if cfg.verify {
            return None;
        }
        let rep = group.representative();
        let (_, stencil_key, fp_bits) =
            self.frozen_stencil_key(cx, reached, rep, group.footprint)?;
        let substrate = cx.store().substrate().cloned()?;
        let Some(slice) = substrate.stencil(stencil_key, fp_bits) else {
            return (self.backend.is_aot() && group.distinct.len() >= cfg.min_fanout)
                .then(|| self.try_frozen_dict_group(cx, reached, group, pass))
                .flatten();
        };
        if self.backend.is_aot() {
            if cfg.shares_dict(slice.code().len(), group.distinct.len()) {
                return self.try_frozen_dict_group(cx, reached, group, pass);
            }
            return self.apply_frozen_aot_group((cx, reached), group, rep, &slice, pass);
        }
        let unwind_drops = slice.decode_unwind_drops().ok()?;
        let sites = slice.decode_sites().ok()?;
        for inst in &group.distinct {
            if self.backend.is_check() {
                pass.record_size(inst.leader, slice.code().len() as u32, 0);
                continue;
            }
            let patched = crate::substrate::patch_stencil(&slice, &inst.symbols).ok()?;
            let exec = match pass.memory.write(&patched) {
                Ok(exec) => exec,
                Err(e) => return Some(Err(item_err(cx, group.item, e))),
            };
            self.register_jit_drops(
                (exec, patched.len()),
                (&unwind_drops, &sites, slice.saved_regs()),
                &inst.symbols,
            );
            self.store_instance(inst, exec);
            pass.record_size(inst.leader, patched.len() as u32, 0);
        }
        tally(Tally::HitLowered);
        Some(Ok(()))
    }

    fn emit_direct_inst(
        &self,
        (cx, group, inst): (&Context<'_>, &GroupWork, &DistinctInst),
        (code, aot_code): (&Arc<Code>, &Option<Arc<AotCode>>),
        (record_cache, verify): (bool, bool),
        (pass, records): (&mut Pass, &mut Vec<(usize, VerifyRecord)>),
    ) -> Result<(), String> {
        let exec = if !self.backend.is_jit() {
            0
        } else {
            let e = self
                .write_jit_code(&mut pass.memory, code, &inst.symbols)
                .map_err(|e| item_err(cx, group.item, e))?;
            self.store_instance(inst, e);
            e
        };
        if let (Some(aot_code), Some(relocs)) = (aot_code, &inst.relocs) {
            self.record_aot(
                inst.leader,
                AotFn::Direct {
                    code: Arc::clone(aot_code),
                    relocs: relocs.clone(),
                },
                &inst.followers,
            );
        }
        pass.record_size(inst.leader, code.bytes().len() as u32, 0);
        if record_cache && !self.backend.is_check() {
            let stored = if inst.relocs.is_some() {
                inst.leader
            } else {
                exec
            };
            pass.stencils.aliases.insert(
                (group.item, group.footprint, inst.symbols.clone()),
                (stored, false),
            );
        }
        if verify {
            for idx in std::iter::once(inst.leader).chain(inst.followers.iter().copied()) {
                let rec = if self.backend.is_jit() {
                    VerifyRecord::Patched {
                        exec,
                        len: code.bytes().len(),
                        sites: code.sites().to_vec(),
                        unwind_drops: code.unwind_drops().to_vec(),
                    }
                } else {
                    VerifyRecord::Compiled {
                        code: Arc::clone(code),
                        symbols: inst.symbols.clone(),
                    }
                };
                records.push((idx, rec));
            }
        }
        Ok(())
    }

    fn finish_direct_group(
        &self,
        (cx, reached): (&mut Context<'_>, &Reached),
        group: &GroupWork,
        (code, reloc_plan): (Arc<Code>, Arc<RelocPlan>),
        (record_cache, verify, elapsed): (bool, bool, u64),
        pass: &mut Pass,
    ) -> Result<(), String> {
        let rep = group.representative();
        if group
            .distinct
            .first()
            .is_some_and(|inst| aot_relocs_valid(&code, inst.reloc_count()))
            && lower::eval_footprint(cx, reached, self, rep).ok() == Some(group.footprint)
        {
            self.freeze_stencil(
                cx,
                reached,
                rep,
                group.footprint,
                false,
                &code,
                Some(&reloc_plan),
                pass,
            );
        }
        let aot_code = self.backend.is_aot().then(|| AotCode::from_code(&code));
        let mut records = Vec::new();
        for inst in &group.distinct {
            self.emit_direct_inst(
                (cx, group, inst),
                (&code, &aot_code),
                (record_cache, verify),
                (pass, &mut records),
            )?;
        }
        pass.record_size(rep, 0, elapsed);
        if record_cache {
            pass.stencils.templates.insert(
                (group.item, group.footprint),
                CachedTemplate::Stencil { code, reloc_plan },
            );
        } else if !verify && let Some(code) = aot_code {
            self.cache_aot_template(
                cx,
                reached,
                (group.item, group.footprint, rep),
                CachedAotTemplate::Direct { code, reloc_plan },
            );
        }
        self.finish_verify(cx, reached, &records)
    }

    pub(in crate::image) fn compile_group(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        group: &mut GroupWork,
        cfg: DictConfig,
        record_cache: bool,
        pass: &mut Pass,
    ) -> Result<(), String> {
        let rep = group.representative();
        if !self.backend.is_check() && cfg.min_bytes == 0 && group.distinct.len() >= cfg.min_fanout
        {
            return self.compile_dict_group(cx, reached, group, cfg, record_cache, pass);
        }
        if !self.backend.is_lazy_jit()
            && let Some(exec_ok) = self.try_frozen_group(cx, reached, group, cfg, pass)
        {
            return exec_ok;
        }
        let started = std::time::Instant::now();
        let out = match group.leader_out.take() {
            Some(out) => out,
            None => lower::lower_stencil(cx, reached, self, rep)
                .map_err(|e| item_err(cx, group.item, e))?,
        };
        if self.backend.is_check() && !cfg.verify {
            group
                .distinct
                .iter()
                .for_each(|inst| pass.record_size(inst.leader, 0, 0));
            pass.record_size(rep, 0, started.elapsed().as_nanos() as u64);
            return Ok(());
        }
        let reloc_plan = out.reloc_plan;
        let target = self.compile_target_for(cx, group.item);
        let code = Arc::new(
            compile_traced(cx, reached, rep, out.func, &target)
                .map_err(|e| item_err(cx, group.item, e))?,
        );
        if cfg.shares_dict(code.bytes().len(), group.distinct.len()) {
            pass.record_size(rep, 0, started.elapsed().as_nanos() as u64);
            return self.compile_dict_group(cx, reached, group, cfg, record_cache, pass);
        }
        self.finish_direct_group(
            (cx, reached),
            group,
            (code, reloc_plan),
            (
                record_cache,
                cfg.verify,
                started.elapsed().as_nanos() as u64,
            ),
            pass,
        )
    }

    pub(in crate::image) fn write_dict_body(&self, code: &Code, pass: &mut Pass) -> Result<usize> {
        if !code
            .relocs()
            .iter()
            .all(|reloc| reloc.as_runtime().is_some())
        {
            bail!("dictionary body has non-runtime relocations");
        }
        let exec = pass
            .memory
            .write_patched(code, &[])
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        self.register_jit_drops(
            (exec, code.bytes().len()),
            (code.unwind_drops(), code.sites(), code.saved_regs()),
            &[],
        );
        Ok(exec)
    }

    fn emit_dict_inst(
        &self,
        (cx, reached): (&mut Context<'_>, &Reached),
        (group, inst, reloc_plan): (&GroupWork, &DistinctInst, &RelocPlan),
        (body_exec, code, aot_code, record_cache, verify): (
            usize,
            &Arc<Code>,
            &Option<Arc<AotCode>>,
            bool,
            bool,
        ),
        (pass, records): (&mut Pass, &mut Vec<(usize, VerifyRecord)>),
    ) -> Result<(), String> {
        let relocs = lower::eval_relocs(cx, reached, self, reloc_plan, inst.leader)
            .map_err(|e| item_err(cx, group.item, e))?;
        let syms = if self.backend.is_jit() || record_cache {
            self.symbols(&relocs)
        } else {
            Vec::new()
        };
        let (stub_exec, stub_bytes) = if !self.backend.is_jit() {
            (0, Vec::new())
        } else {
            let (e, b) = self
                .emit_dict_stub(body_exec, &syms, pass)
                .map_err(|e| item_err(cx, group.item, e))?;
            self.store_instance(inst, e);
            (e, b)
        };
        if let Some(aot_code) = aot_code {
            self.register_aot_relocs(&relocs);
            self.record_aot(
                inst.leader,
                AotFn::Dict {
                    body: Arc::clone(aot_code),
                    relocs,
                },
                &inst.followers,
            );
        }
        pass.record_size(inst.leader, stub_bytes.len() as u32, 0);
        if record_cache && !self.backend.is_check() {
            pass.stencils.aliases.insert(
                (group.item, group.footprint, syms),
                (
                    if aot_code.is_some() {
                        inst.leader
                    } else {
                        stub_exec
                    },
                    true,
                ),
            );
        }
        if verify {
            for idx in std::iter::once(inst.leader).chain(inst.followers.iter().copied()) {
                records.push((
                    idx,
                    VerifyRecord::Dict {
                        body_exec,
                        stub_exec,
                        stub_bytes: stub_bytes.clone(),
                        code: Arc::clone(code),
                    },
                ));
            }
        }
        Ok(())
    }

    fn store_dict_template(
        &self,
        (cx, reached): (&Context<'_>, &Reached),
        (group, rep, verify, record_cache): (&GroupWork, usize, bool, bool),
        (body_exec, code, aot_code, reloc_plan): (
            usize,
            Arc<Code>,
            Option<Arc<AotCode>>,
            Arc<RelocPlan>,
        ),
        pass: &mut Pass,
    ) {
        if record_cache {
            pass.stencils.templates.insert(
                (group.item, group.footprint),
                CachedTemplate::Dict {
                    body_exec,
                    code,
                    reloc_plan,
                },
            );
        } else if !verify && let Some(body) = aot_code {
            self.cache_aot_template(
                cx,
                reached,
                (group.item, group.footprint, rep),
                CachedAotTemplate::Dict { body, reloc_plan },
            );
        }
    }

    fn compile_dict_group(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        group: &GroupWork,
        cfg: DictConfig,
        record_cache: bool,
        pass: &mut Pass,
    ) -> Result<(), String> {
        let (rep, started) = (group.representative(), std::time::Instant::now());
        if !cfg.verify
            && self.backend.is_aot()
            && let Some(ok) = self.try_frozen_dict_group(cx, reached, group, pass)
        {
            return ok;
        }
        let (func, _, reloc_plan) =
            lower::lower_dict(cx, reached, self, rep).map_err(|e| item_err(cx, group.item, e))?;
        let code = Arc::new(
            haste_codegen::compile(func, &target()).map_err(|e| item_err(cx, group.item, e))?,
        );
        if !cfg.verify
            && lower::eval_footprint(cx, reached, self, rep).ok() == Some(group.footprint)
        {
            self.freeze_stencil(
                cx,
                reached,
                rep,
                group.footprint,
                true,
                &code,
                Some(&reloc_plan),
                pass,
            );
        }
        let body_exec = if !self.backend.is_jit() {
            0
        } else {
            self.write_dict_body(&code, pass)
                .map_err(|e| item_err(cx, group.item, e))?
        };
        let (aot_code, mut records) = (
            self.backend.is_aot().then(|| AotCode::from_code(&code)),
            Vec::new(),
        );
        for inst in &group.distinct {
            self.emit_dict_inst(
                (cx, reached),
                (group, inst, &reloc_plan),
                (body_exec, &code, &aot_code, record_cache, cfg.verify),
                (pass, &mut records),
            )?;
        }
        pass.record_size(
            rep,
            code.bytes().len() as u32,
            started.elapsed().as_nanos() as u64,
        );
        self.store_dict_template(
            (cx, reached),
            (group, rep, cfg.verify, record_cache),
            (body_exec, code, aot_code, reloc_plan),
            pass,
        );
        self.finish_verify(cx, reached, &records)
    }

    fn store_instance(&self, inst: &DistinctInst, addr: usize) {
        self.publish(inst.leader, addr);
        for &follower in &inst.followers {
            self.publish(follower, addr);
        }
    }

    fn alloc_dict(&self, symbols: &[u64]) -> usize {
        if symbols.is_empty() {
            return 0;
        }
        let bytes: Vec<u8> = symbols.iter().flat_map(|word| word.to_le_bytes()).collect();
        self.data(&bytes) as usize
    }

    pub(in crate::image) fn frozen_stencil_key(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        index: usize,
        fp: FootprintId,
    ) -> Option<(bool, u64, u128)> {
        if !fp.is_portable() {
            return None;
        }
        let ((item, _), Ok(typed)) = &reached.list[index] else {
            return None;
        };
        if matches!(typed.shape, crate::sema::infer::Shape::Anon(_)) || typed.has_awaits() {
            return None;
        }
        let (is_sysroot, item_key) = crate::substrate::layer_item_key(cx.outline, *item)?;
        Some((is_sysroot, item_key ^ self.backend.stencil_tag(), fp.0))
    }

    pub(super) fn emit_dict_stub(
        &self,
        body_exec: usize,
        symbols: &[u64],
        pass: &mut Pass,
    ) -> Result<(usize, Vec<u8>)> {
        self.emit_dict_stub_with_tier(body_exec, symbols, None, pass)
    }

    pub(in crate::image) fn emit_dict_stub_with_tier(
        &self,
        body_exec: usize,
        symbols: &[u64],
        tier_up: Option<haste_codegen::DictTierUp>,
        pass: &mut Pass,
    ) -> Result<(usize, Vec<u8>)> {
        let dict_addr = self.alloc_dict(symbols);
        let target = target();
        let space = if tier_up.is_some() {
            128
        } else {
            DICT_STUB_SPACE
        };
        let mut written = Vec::new();
        let exec = pass.memory.write_with(space, |stub_exec| {
            let bytes = target.dict_entry_stub(stub_exec, body_exec, dict_addr, tier_up)?;
            written = bytes.clone();
            Ok(bytes)
        })?;
        Ok((exec, written))
    }
}
