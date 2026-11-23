use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use haste_codegen::{Code, CodeUnwindDrop, DictTierUp, Site};

use crate::lower::{self, FootprintId, RelocPlan, RelocVec, Slots, SymbolSource};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sync::lock;

use super::stencil::{DictConfig, FirstJit, FirstJitState, FreshSlots, JitTemplate, TierUpEntry, VerifyRecord, aot_relocs_valid};
use super::workers::{bucket_work, lpt_shards};
use super::{Image, Pass, stub};

/// What a worker made ahead for one function of a lazy batch, each part only when its making asked for nothing: its footprint, its encoded stencil, and its relocations under the plan they follow.
pub(super) struct Ready {
    footprint: Option<FootprintId>,
    code: Option<Arc<Code>>,
    relocs: Option<(Arc<RelocPlan>, RelocVec)>,
    thawed: bool,
}

/// A function's encoded stencil, its relocations, and their plan.
type Lowered = (Arc<Code>, RelocVec, Arc<RelocPlan>);

impl Image {
    pub(super) fn register_jit_drops(&self, exec: usize, len: usize, drops: &[CodeUnwindDrop], symbols: &[u64]) {
        crate::host::unwind::register_shared_drops(self.unwind_head, |bytes| lock(&self.shared).alloc(bytes).ok(), exec, len, drops, symbols);
    }

    pub(super) fn write_jit_code(&self, memory: &mut haste_codegen::Memory, code: &Code, symbols: &[u64]) -> Result<usize, haste_codegen::Error> {
        let exec = memory.write_patched(code, symbols)?;
        self.register_jit_drops(exec, code.bytes().len(), code.unwind_drops(), symbols);
        Ok(exec)
    }
    /// Compiles the inferred functions `work` of a lazy image, lowering and encoding them on every core first, and gives the code of each that compiled; one that did not keeps its stub and its slot.
    pub(super) fn compile_batch(&mut self, cx: &mut Context<'_>, reached: &Reached, work: &[usize]) -> Vec<(usize, usize)> {
        let mut work = work.to_vec();
        work.sort_unstable();
        let buckets = bucket_work(reached, &work);
        self.pass.ready = if buckets.len() <= 1 || crate::jobs() <= 1 {
            buckets.iter().flat_map(|bucket| self.ready_bucket(cx, reached, bucket)).collect()
        } else {
            let shards = lpt_shards(buckets, crate::jobs());
            let forks = shards.iter().map(|_| cx.fork()).collect();
            let made = self.run_shards(forks, &shards, |mut fork, shard| shard.iter().flat_map(|bucket| self.ready_bucket(&mut fork, reached, bucket)).collect::<Vec<_>>());
            made.into_iter().filter_map(Result::ok).flatten().collect()
        };
        work.retain(|index| self.pass.ready.contains_key(index));
        let compiled = work.into_iter().filter_map(|index| self.compile_one(cx, reached, index).ok().map(|code| (index, code))).collect();
        self.pass.ready.clear();
        self.take_wanted();
        compiled
    }

    fn thaw_lazy_stencil(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, fp: FootprintId, wanted0: usize) -> Result<Option<Lowered>> {
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
        if code.relocs().iter().all(|r| r.symbol_index().is_none()) && code.unwind_drops().is_empty() {
            return Ok(Some((code, RelocVec::new(), Arc::default())));
        }
        let Some(plan) = self.decode_lazy_reloc_plan(cx, reached, slice.plan()).map(Arc::new) else {
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

    /// What a worker makes ahead for the functions `bucket` of one item: relocations under a plan that a template or an earlier function of the bucket already has, else a thawed substrate stencil, else a lowered and encoded stencil.
    fn ready_bucket(&self, cx: &mut Context<'_>, reached: &Reached, bucket: &[usize]) -> Vec<(usize, Ready)> {
        let mut plans: rustc_hash::FxHashMap<FootprintId, Arc<RelocPlan>> = rustc_hash::FxHashMap::default();
        let mut made = Vec::new();
        for &index in bucket {
            let item = reached.list[index].0.0;
            let footprint = self.quietly(cx, |cx| lower::eval_footprint(cx, reached, self, index).ok());
            let plan = footprint.and_then(|fp| self.pass.stencils.jit_templates.get(&(item, fp)).map(|template| Arc::clone(&template.reloc_plan)).or_else(|| plans.get(&fp).cloned()));
            let ready = match plan {
                Some(plan) => Ready { footprint, code: None, relocs: self.quietly(cx, |cx| lower::eval_relocs(cx, reached, self, &plan, index).ok()).map(|relocs| (plan, relocs)), thawed: false },
                None => {
                    let wanted_before = self.wanted_len();
                    let thawed = footprint.and_then(|fp| self.quietly(cx, |cx| self.thaw_lazy_stencil(cx, reached, index, fp, wanted_before).ok().flatten()));
                    match thawed {
                        Some((code, relocs, plan)) => {
                            if let Some(fp) = footprint {
                                plans.insert(fp, Arc::clone(&plan));
                            }
                            Ready { footprint, code: Some(code), relocs: Some((plan, relocs)), thawed: true }
                        }
                        None if self.wanted_len() > wanted_before => continue,
                        None => match self.quietly(cx, |cx| lower::lower_stencil(cx, reached, self, index).ok()).and_then(|out| Some((haste_codegen::compile(out.func, &self.compile_target_for(cx, item)).ok()?, out.relocs, out.reloc_plan))) {
                            Some((code, relocs, plan)) => {
                                if let Some(fp) = footprint {
                                    plans.insert(fp, Arc::clone(&plan));
                                }
                                Ready { footprint, code: Some(Arc::new(code)), relocs: Some((plan, relocs)), thawed: false }
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

    /// What `run` gives on `cx`, when it noted no unreached function, asked for nothing and deferred no constant evaluation.
    fn quietly<'a, T>(&self, cx: &mut Context<'a>, run: impl FnOnce(&mut Context<'a>) -> Option<T>) -> Option<T> {
        let mark = (self.wanted_len(), cx.quiet_mark());
        run(cx).filter(|_| (self.wanted_len(), cx.quiet_mark()) == mark)
    }

    /// The values of the relocations `relocs` in this image.
    pub(super) fn symbols(&self, relocs: &[SymbolSource]) -> Vec<u64> {
        relocs.iter().map(|sym| self.symbol_value(sym)).collect()
    }

    /// The footprint of function `index`: the one a worker made ahead, or a fresh one.
    fn footprint(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, pass: &Pass) -> Result<FootprintId> {
        match pass.ready.get(&index).and_then(|ready| ready.footprint) {
            Some(footprint) => Ok(footprint),
            None => lower::eval_footprint(cx, reached, self, index),
        }
    }

    /// The relocations of function `index` under `plan`: the ones a worker made ahead under that same plan, or fresh ones.
    fn relocs(&self, cx: &mut Context<'_>, reached: &Reached, plan: &Arc<RelocPlan>, index: usize, pass: &Pass) -> Result<RelocVec> {
        match pass.ready.get(&index).and_then(|ready| ready.relocs.as_ref()).filter(|(made, _)| Arc::ptr_eq(made, plan)) {
            Some((_, relocs)) => Ok(relocs.clone()),
            None => lower::eval_relocs(cx, reached, self, plan, index),
        }
    }

    /// Function `index` lowered and encoded: what a worker made ahead, else a fresh lowering that fails when it noted an unreached function since `wanted0`, plus whether it was thawed from substrate.
    fn lowered(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, wanted0: usize, pass: &Pass) -> Result<(Lowered, bool)> {
        if let Some(Ready { code: Some(code), relocs: Some((plan, relocs)), thawed, .. }) = pass.ready.get(&index) {
            return Ok(((Arc::clone(code), relocs.clone(), Arc::clone(plan)), *thawed));
        }
        let out = lower::lower_stencil(cx, reached, self, index)?;
        if self.wanted_len() > wanted0 {
            bail!("unreached inline callee");
        }
        Ok(((Arc::new(haste_codegen::compile(out.func, &self.compile_target_for(cx, reached.list[index].0.0))?), out.relocs, out.reloc_plan), false))
    }

    fn freeze_lazy(&self, cx: &Context<'_>, reached: &Reached, index: usize, fp: FootprintId, (code, plan): (&Arc<Code>, &RelocPlan), pass: &mut Pass) {
        if self.backend.is_lazy_jit() && !pass.stencils.cfg.verify && cx.outline.is_fixed(reached.list[index].0.0.krate) {
            self.freeze_stencil(cx, reached, index, fp, false, code, Some(plan), pass);
        }
    }

    pub(super) fn compile_jit_one(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, pass: &mut Pass) -> Result<usize> {
        let cfg = pass.stencils.cfg;
        let item = reached.list[index].0.0;
        let wanted0 = self.wanted_len();
        if !cfg.verify
            && cfg.min_fanout > 1
            && let Some(exec) = self.try_first_jit(cx, reached, index, item, wanted0, pass)?
        {
            return Ok(exec);
        }
        self.compile_jit_cached(cx, reached, index, item, wanted0, cfg, pass)
    }

    fn try_first_jit(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, item: ItemRef, wanted0: usize, pass: &mut Pass) -> Result<Option<usize>> {
        match pass.stencils.first_jit.get(&item).cloned() {
            None => {
                let fp = if self.backend.is_lazy_jit() && cx.outline.is_fixed(item.krate) {
                    let fp = self.footprint(cx, reached, index, pass)?;
                    if self.wanted_len() > wanted0 {
                        bail!("unreached inline callee");
                    }
                    Some(fp)
                } else {
                    None
                };
                let ((code, relocs, reloc_plan), thawed) = if let Some(fp) = fp
                    && pass.ready.get(&index).is_none_or(|r| r.code.is_none())
                    && let Some(hit) = self.thaw_lazy_stencil(cx, reached, index, fp, wanted0)?
                {
                    (hit, true)
                } else {
                    self.lowered(cx, reached, index, wanted0, pass)?
                };
                if !thawed && let Some(fp) = fp {
                    self.freeze_lazy(cx, reached, index, fp, (&code, &reloc_plan), pass);
                }
                let symbols = self.symbols(&relocs);
                let exec = self.write_jit_code(&mut pass.memory, &code, &symbols)?;
                pass.stencils.first_jit.insert(item, FirstJitState::First(FirstJit { index, code, symbols, exec, reloc_plan }));
                Ok(Some(exec))
            }
            Some(FirstJitState::First(first)) => {
                let fp1 = self.footprint(cx, reached, first.index, pass)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                pass.stencils.jit_templates.insert((item, fp1), JitTemplate { stencil: first.code, dict: None, fanout: 1, reloc_plan: first.reloc_plan });
                pass.stencils.aliases.insert((item, fp1, first.symbols), (first.exec, false));
                pass.stencils.first_jit.insert(item, FirstJitState::Promoted);
                Ok(None)
            }
            Some(FirstJitState::Promoted) => Ok(None),
        }
    }

    fn compile_jit_cached(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, item: ItemRef, wanted0: usize, cfg: DictConfig, pass: &mut Pass) -> Result<usize> {
        let footprint = self.footprint(cx, reached, index, pass)?;
        if self.wanted_len() > wanted0 {
            bail!("unreached inline callee");
        }
        let existing_plan = pass.stencils.jit_templates.get(&(item, footprint)).map(|jt| Arc::clone(&jt.reloc_plan));
        let (symbols, prelowered) = match existing_plan {
            Some(plan) => {
                let relocs = self.relocs(cx, reached, &plan, index, pass)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                (self.symbols(&relocs), None)
            }
            None => {
                let ((code, relocs, plan), thawed) = if !cfg.verify
                    && pass.ready.get(&index).is_none_or(|r| r.code.is_none())
                    && let Some(hit) = self.thaw_lazy_stencil(cx, reached, index, footprint, wanted0)?
                {
                    (hit, true)
                } else {
                    self.lowered(cx, reached, index, wanted0, pass)?
                };
                if !thawed {
                    self.freeze_lazy(cx, reached, index, footprint, (&code, &plan), pass);
                }
                (self.symbols(&relocs), Some((code, plan)))
            }
        };
        let alias_key = (item, footprint, symbols.clone());
        if let Some(&(addr, is_dict)) = pass.stencils.aliases.get(&alias_key) {
            if cfg.verify
                && let Some(jt) = pass.stencils.jit_templates.get(&(item, footprint))
            {
                let rec = match (is_dict, &jt.dict) {
                    (true, Some((body_exec, dict_code, _))) => VerifyRecord::Dict { body_exec: *body_exec, stub_exec: addr, stub_bytes: Vec::new(), code: dict_code.clone() },
                    _ => VerifyRecord::Patched { exec: addr, len: jt.stencil.bytes().len(), sites: jt.stencil.sites().to_vec(), unwind_drops: jt.stencil.unwind_drops().to_vec() },
                };
                self.verify_single(cx, reached, index, &rec)?;
            }
            return Ok(addr);
        }
        let stencil = self.ensure_jit_stencil(cx, reached, index, (item, footprint), prelowered, pass)?;
        let fanout = pass.stencils.jit_templates.get(&(item, footprint)).map_or(1, |jt| jt.fanout);
        if stencil.bytes().len() > cfg.min_bytes && fanout >= cfg.min_fanout {
            return self.instantiate_jit_dict(cx, reached, index, (item, footprint), symbols, cfg, pass);
        }
        let exec = self.write_jit_code(&mut pass.memory, &stencil, &symbols)?;
        pass.stencils.aliases.insert(alias_key, (exec, false));
        if cfg.verify {
            let rec = VerifyRecord::Patched { exec, len: stencil.bytes().len(), sites: stencil.sites().to_vec(), unwind_drops: stencil.unwind_drops().to_vec() };
            self.verify_single(cx, reached, index, &rec)?;
        }
        Ok(exec)
    }

    fn ensure_jit_stencil(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, key: (ItemRef, FootprintId), prelowered: Option<(Arc<Code>, Arc<RelocPlan>)>, pass: &mut Pass) -> Result<Arc<Code>> {
        if let Some(entry) = pass.stencils.jit_templates.get_mut(&key) {
            entry.fanout += 1;
            return Ok(entry.stencil.clone());
        }
        let (stencil, reloc_plan) = match prelowered {
            Some(pair) => pair,
            None => {
                let ((code, _, plan), _) = self.lowered(cx, reached, index, usize::MAX, pass)?;
                (code, plan)
            }
        };
        pass.stencils.jit_templates.insert(key, JitTemplate { stencil: stencil.clone(), dict: None, fanout: 1, reloc_plan });
        Ok(stencil)
    }

    fn instantiate_jit_dict(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, key: (ItemRef, FootprintId), symbols: Vec<u64>, cfg: DictConfig, pass: &mut Pass) -> Result<usize> {
        let (body_exec, dict_code, dict_plan) = match pass.stencils.jit_templates.get(&key).and_then(|jt| jt.dict.clone()) {
            Some(triple) => triple,
            None => {
                let (func, _, dict_plan) = lower::lower_dict(cx, reached, self, index)?;
                let code = Arc::new(haste_codegen::compile(func, &self.compile_target())?);
                let exec = self.write_dict_body(&code, pass)?;
                if let Some(jt) = pass.stencils.jit_templates.get_mut(&key) {
                    jt.dict = Some((exec, code.clone(), Arc::clone(&dict_plan)));
                }
                (exec, code, dict_plan)
            }
        };
        let dict_relocs = lower::eval_relocs(cx, reached, self, &dict_plan, index)?;
        let dict_symbols = self.symbols(&dict_relocs);
        let tier = self.make_tier_up(index, key, &symbols, cfg.tier_up, pass)?;
        let (stub_exec, stub_bytes) = self.emit_dict_stub_with_tier(body_exec, &dict_symbols, tier, pass)?;
        if tier.is_none() {
            pass.stencils.aliases.insert((key.0, key.1, symbols), (stub_exec, true));
        }
        if cfg.verify {
            let rec = VerifyRecord::Dict { body_exec, stub_exec, stub_bytes, code: dict_code };
            self.verify_single(cx, reached, index, &rec)?;
        }
        Ok(stub_exec)
    }

    fn make_tier_up(&self, index: usize, key: (ItemRef, FootprintId), symbols: &[u64], threshold: u32, pass: &mut Pass) -> Result<Option<DictTierUp>> {
        if !self.backend.is_lazy_jit() || threshold == 0 || index >= (stub::TIER_UP_BIT as usize) {
            return Ok(None);
        }
        let counter_addr = lock(&self.shared).alloc(8)?;
        pass.stencils.tier_ups.insert(index, TierUpEntry { item: key.0, footprint: key.1, symbols: symbols.to_vec() });
        Ok(Some(DictTierUp { counter_addr, threshold, stub_entry: stub::entry_addr(), encoded_index: (index as u32) | stub::TIER_UP_BIT }))
    }

    pub(super) fn tier_up(&mut self, index: usize) -> Result<usize> {
        let Some(entry) = self.pass.stencils.tier_ups.remove(&index) else {
            let existing = self.slot(index).load(Ordering::Acquire);
            if existing != 0 {
                return Ok(existing);
            }
            bail!("missing tier-up entry for slot {index}");
        };
        let Some(jt) = self.pass.stencils.jit_templates.get(&(entry.item, entry.footprint)) else {
            bail!("missing stencil template for slot {index}");
        };
        let exec = self.pass.memory.write_patched(&jt.stencil, &entry.symbols)?;
        self.register_jit_drops(exec, jt.stencil.bytes().len(), jt.stencil.unwind_drops(), &entry.symbols);
        self.pass.stencils.aliases.insert((entry.item, entry.footprint, entry.symbols), (exec, false));
        self.publish(index, exec);
        Ok(exec)
    }

    pub(super) fn finish_verify(&self, cx: &mut Context<'_>, reached: &Reached, records: &[(usize, VerifyRecord)]) -> Result<(), String> {
        let errs = self.verify_records(cx, reached, records);
        if let Some((_, msg)) = errs.into_iter().next() {
            return Err(msg);
        }
        Ok(())
    }

    pub(super) fn verify_records(&self, cx: &mut Context<'_>, reached: &Reached, records: &[(usize, VerifyRecord)]) -> Vec<(usize, String)> {
        let mut errs = Vec::new();
        let fresh = FreshSlots(self, lower::Bodies::default());
        for (index, record) in records {
            fresh.1.clear_transient();
            if let Err(err) = Self::verify_with_fresh(cx, reached, &fresh, *index, record) {
                errs.push((*index, format!("{}: {err:#}", cx.name(reached.list[*index].0.0))));
            }
        }
        errs
    }

    fn verify_single(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, record: &VerifyRecord) -> Result<()> {
        let fresh = FreshSlots(self, lower::Bodies::default());
        Self::verify_with_fresh(cx, reached, &fresh, index, record)
    }

    fn verify_with_fresh(cx: &mut Context<'_>, reached: &Reached, fresh: &FreshSlots<'_>, index: usize, record: &VerifyRecord) -> Result<()> {
        match record {
            VerifyRecord::Patched { exec, len, sites, unwind_drops } => Self::verify_patched(cx, reached, fresh, index, (*exec, *len), sites, unwind_drops),
            VerifyRecord::Compiled { code, symbols } => Self::verify_compiled(cx, reached, fresh, index, code, symbols),
            VerifyRecord::Dict { body_exec, stub_exec, stub_bytes, code } => Self::verify_dict(cx, reached, fresh, index, (*body_exec, *stub_exec), stub_bytes, code),
        }
    }

    fn verify_compiled(cx: &mut Context<'_>, reached: &Reached, fresh: &FreshSlots<'_>, index: usize, code: &Code, symbols: &[u64]) -> Result<()> {
        let out = lower::lower_stencil(cx, reached, fresh, index)?;
        let slots: &dyn Slots = fresh;
        let fresh_syms: Vec<u64> = out.relocs.iter().map(|sym| slots.resolve_symbol(sym)).collect();
        let fresh_code = haste_codegen::compile(out.func, &fresh.0.compile_target_for(cx, reached.list[index].0.0))?;
        if code.bytes() != fresh_code.bytes() {
            bail!("stencil verification byte mismatch ({} vs {} bytes)", code.bytes().len(), fresh_code.bytes().len());
        }
        if code.sites() != fresh_code.sites() {
            bail!("stencil verification Site table mismatch");
        }
        if code.unwind_drops() != fresh_code.unwind_drops() {
            bail!("stencil verification unwind drop table mismatch");
        }
        if code.relocs() != fresh_code.relocs() {
            bail!("stencil verification code.relocs mismatch ({} vs {})", code.relocs().len(), fresh_code.relocs().len());
        }
        if symbols != fresh_syms.as_slice() {
            let first = symbols.iter().zip(&fresh_syms).position(|(a, b)| a != b);
            bail!("stencil verification symbol mismatch (len {} vs {}, first diff {first:?})", symbols.len(), fresh_syms.len());
        }
        Ok(())
    }

    fn verify_patched(cx: &mut Context<'_>, reached: &Reached, fresh: &FreshSlots<'_>, index: usize, (exec, len): (usize, usize), sites: &[Site], unwind_drops: &[CodeUnwindDrop]) -> Result<()> {
        let out = lower::lower_stencil(cx, reached, fresh, index)?;
        let slots: &dyn Slots = fresh;
        let symbols: Vec<u64> = out.relocs.iter().map(|sym| slots.resolve_symbol(sym)).collect();
        let fresh_code = haste_codegen::compile(out.func, &fresh.0.compile_target_for(cx, reached.list[index].0.0))?;
        let mut expected = vec![0u8; fresh_code.bytes().len()];
        fresh_code.patch(&symbols, &mut expected);
        // SAFETY: exec..exec + len is mapped readable executable code in this image.
        let actual = unsafe { std::slice::from_raw_parts(exec as *const u8, len) };
        if actual != expected.as_slice() {
            bail!("stencil verification byte mismatch ({} vs {} bytes)", actual.len(), expected.len());
        }
        if sites != fresh_code.sites() {
            bail!("stencil verification Site table mismatch");
        }
        if unwind_drops != fresh_code.unwind_drops() {
            bail!("stencil verification unwind drop table mismatch");
        }
        Ok(())
    }

    fn verify_dict(cx: &mut Context<'_>, reached: &Reached, fresh: &FreshSlots<'_>, index: usize, (body_exec, stub_exec): (usize, usize), stub_bytes: &[u8], code: &Code) -> Result<()> {
        let (func, _, _) = lower::lower_dict(cx, reached, fresh, index)?;
        let fresh_code = haste_codegen::compile(func, &super::stencil::target())?;
        if !fresh_code.relocs().iter().all(|reloc| reloc.as_runtime().is_some()) {
            bail!("dict body verification found non-runtime relocations");
        }
        if body_exec == 0 {
            if code.bytes() != fresh_code.bytes() || code.relocs() != fresh_code.relocs() {
                bail!("dict body verification mismatch ({} vs {} bytes)", code.bytes().len(), fresh_code.bytes().len());
            }
        } else {
            let mut expected = vec![0u8; fresh_code.bytes().len()];
            fresh_code.patch(&[], &mut expected);
            // SAFETY: body_exec..body_exec + code.bytes().len() is mapped readable executable code in this image.
            let actual_body = unsafe { std::slice::from_raw_parts(body_exec as *const u8, code.bytes().len()) };
            if actual_body != expected.as_slice() {
                bail!("dict body verification byte mismatch ({} vs {} bytes)", actual_body.len(), expected.len());
            }
        }
        if code.sites() != fresh_code.sites() {
            bail!("dict body verification Site table mismatch");
        }
        if stub_exec != 0 && !stub_bytes.is_empty() {
            // SAFETY: stub_exec..stub_exec + stub_bytes.len() is mapped readable executable code in this image.
            let actual_stub = unsafe { std::slice::from_raw_parts(stub_exec as *const u8, stub_bytes.len()) };
            if actual_stub != stub_bytes {
                bail!("dict entry stub byte mismatch");
            }
        }
        Ok(())
    }
}
