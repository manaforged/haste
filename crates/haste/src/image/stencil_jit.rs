use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use haste_codegen::{Code, DictTierUp, Site};

use crate::lower::{self, FootprintId, Slots};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sync::lock;

use super::stencil::{DictConfig, FreshSlots, JitTemplate, TierUpEntry, VerifyRecord, target};
use super::{Image, Pass, stub};

impl Image {
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
                let out = lower::lower_stencil(cx, reached, self, index)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                let symbols: Vec<u64> = out.relocs.iter().map(|sym| self.symbol_value(sym)).collect();
                let code = Arc::new(haste_codegen::compile(out.func, &target())?);
                let exec = pass.memory.write_patched(&code, &symbols)?;
                pass.stencils.first_jit.insert(item, Some((index, code, symbols, exec, out.reloc_plan)));
                Ok(Some(exec))
            }
            Some(Some((idx1, code1, syms1, exec1, plan1))) => {
                let fp1 = lower::eval_footprint(cx, reached, self, idx1)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                pass.stencils.jit_templates.insert((item, fp1), JitTemplate { stencil: code1, dict: None, fanout: 1, reloc_plan: plan1 });
                pass.stencils.aliases.insert((item, fp1, syms1), (exec1, false));
                pass.stencils.first_jit.insert(item, None);
                Ok(None)
            }
            Some(None) => Ok(None),
        }
    }

    fn compile_jit_cached(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, item: ItemRef, wanted0: usize, cfg: DictConfig, pass: &mut Pass) -> Result<usize> {
        let footprint = lower::eval_footprint(cx, reached, self, index)?;
        if self.wanted_len() > wanted0 {
            bail!("unreached inline callee");
        }
        let existing_plan = pass.stencils.jit_templates.get(&(item, footprint)).map(|jt| Arc::clone(&jt.reloc_plan));
        let (symbols, prelowered) = match existing_plan {
            Some(plan) => {
                let relocs = lower::eval_relocs(cx, reached, self, &plan, index)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                (relocs.iter().map(|sym| self.symbol_value(sym)).collect::<Vec<u64>>(), None)
            }
            None => {
                let out = lower::lower_stencil(cx, reached, self, index)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                let symbols: Vec<u64> = out.relocs.iter().map(|sym| self.symbol_value(sym)).collect();
                (symbols, Some(out))
            }
        };
        let alias_key = (item, footprint, symbols.clone());
        if let Some(&(addr, is_dict)) = pass.stencils.aliases.get(&alias_key) {
            if cfg.verify
                && let Some(jt) = pass.stencils.jit_templates.get(&(item, footprint))
            {
                let rec = match (is_dict, &jt.dict) {
                    (true, Some((body_exec, dict_code))) => VerifyRecord::Dict { body_exec: *body_exec, stub_exec: addr, stub_bytes: Vec::new(), code: dict_code.clone() },
                    _ => VerifyRecord::Patched { exec: addr, len: jt.stencil.bytes().len(), sites: jt.stencil.sites().to_vec() },
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
        let exec = pass.memory.write_patched(&stencil, &symbols)?;
        pass.stencils.aliases.insert(alias_key, (exec, false));
        if cfg.verify {
            let rec = VerifyRecord::Patched { exec, len: stencil.bytes().len(), sites: stencil.sites().to_vec() };
            self.verify_single(cx, reached, index, &rec)?;
        }
        Ok(exec)
    }

    fn ensure_jit_stencil(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, key: (ItemRef, FootprintId), prelowered: Option<lower::StencilOutput>, pass: &mut Pass) -> Result<Arc<Code>> {
        if let Some(entry) = pass.stencils.jit_templates.get_mut(&key) {
            entry.fanout += 1;
            return Ok(entry.stencil.clone());
        }
        let out = match prelowered {
            Some(out) => out,
            None => lower::lower_stencil(cx, reached, self, index)?,
        };
        let stencil = Arc::new(haste_codegen::compile(out.func, &target())?);
        pass.stencils.jit_templates.insert(key, JitTemplate { stencil: stencil.clone(), dict: None, fanout: 1, reloc_plan: out.reloc_plan });
        Ok(stencil)
    }

    fn instantiate_jit_dict(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, key: (ItemRef, FootprintId), symbols: Vec<u64>, cfg: DictConfig, pass: &mut Pass) -> Result<usize> {
        let (body_exec, dict_code) = match pass.stencils.jit_templates.get(&key).and_then(|jt| jt.dict.clone()) {
            Some(pair) => pair,
            None => {
                let (func, _, _) = lower::lower_dict(cx, reached, self, index)?;
                let code = Arc::new(haste_codegen::compile(func, &target())?);
                let exec = self.write_dict_body(&code, pass)?;
                if let Some(jt) = pass.stencils.jit_templates.get_mut(&key) {
                    jt.dict = Some((exec, code.clone()));
                }
                (exec, code)
            }
        };
        let tier = self.make_tier_up(index, key, &symbols, cfg.tier_up, pass)?;
        let (stub_exec, stub_bytes) = self.emit_dict_stub_with_tier(body_exec, &symbols, tier, pass)?;
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
        if !self.jit_mode || threshold == 0 || index >= (stub::TIER_UP_BIT as usize) {
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
            VerifyRecord::Patched { exec, len, sites } => Self::verify_patched(cx, reached, fresh, index, (*exec, *len), sites),
            VerifyRecord::Dict { body_exec, stub_exec, stub_bytes, code } => Self::verify_dict(cx, reached, fresh, index, (*body_exec, *stub_exec), stub_bytes, code),
        }
    }

    fn verify_patched(cx: &mut Context<'_>, reached: &Reached, fresh: &FreshSlots<'_>, index: usize, (exec, len): (usize, usize), sites: &[Site]) -> Result<()> {
        let out = lower::lower_stencil(cx, reached, fresh, index)?;
        let slots: &dyn Slots = fresh;
        let symbols: Vec<u64> = out.relocs.iter().map(|sym| slots.resolve_symbol(sym)).collect();
        let fresh_code = haste_codegen::compile(out.func, &target())?;
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
        Ok(())
    }

    fn verify_dict(cx: &mut Context<'_>, reached: &Reached, fresh: &FreshSlots<'_>, index: usize, (body_exec, stub_exec): (usize, usize), stub_bytes: &[u8], code: &Code) -> Result<()> {
        let (func, _, _) = lower::lower_dict(cx, reached, fresh, index)?;
        let fresh_code = haste_codegen::compile(func, &target())?;
        if !fresh_code.relocs().iter().all(|reloc| reloc.as_runtime().is_some()) {
            bail!("dict body verification found non-runtime relocations");
        }
        let mut expected = vec![0u8; fresh_code.bytes().len()];
        fresh_code.patch(&[], &mut expected);
        // SAFETY: body_exec..body_exec + code.bytes().len() is mapped readable executable code in this image.
        let actual_body = unsafe { std::slice::from_raw_parts(body_exec as *const u8, code.bytes().len()) };
        if actual_body != expected.as_slice() {
            bail!("dict body verification byte mismatch ({} vs {} bytes)", actual_body.len(), expected.len());
        }
        if code.sites() != fresh_code.sites() {
            bail!("dict body verification Site table mismatch");
        }
        if !stub_bytes.is_empty() {
            // SAFETY: stub_exec..stub_exec + stub_bytes.len() is mapped readable executable code in this image.
            let actual_stub = unsafe { std::slice::from_raw_parts(stub_exec as *const u8, stub_bytes.len()) };
            if actual_stub != stub_bytes {
                bail!("dict entry stub byte mismatch");
            }
        }
        Ok(())
    }
}
