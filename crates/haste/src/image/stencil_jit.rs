mod batch;
mod verify;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use haste_codegen::{Code, CodeUnwindDrop, DictTierUp, Site};

use crate::lower::{self, FootprintId, RelocPlan, RelocVec, Slots, SymbolSource};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sync::lock;

use super::stencil::{DictConfig, FirstJit, FirstJitState, JitTemplate, TierUpEntry, VerifyRecord};
use super::{Image, Pass, stub};

pub(super) struct Ready {
    footprint: Option<FootprintId>,
    code: Option<Arc<Code>>,
    relocs: Option<(Arc<RelocPlan>, RelocVec)>,
    thawed: bool,
}

type Lowered = (Arc<Code>, RelocVec, Arc<RelocPlan>);

impl Image {
    pub(super) fn register_jit_drops(
        &self,
        (exec, len): (usize, usize),
        (drops, sites): (&[CodeUnwindDrop], &[Site]),
        symbols: &[u64],
    ) {
        crate::host::unwind::register_shared_drops(
            self.unwind_head,
            |bytes| lock(&self.shared).alloc(bytes).ok(),
            (exec, len),
            (drops, sites),
            symbols,
        );
    }

    pub(super) fn write_jit_code(
        &self,
        memory: &mut haste_codegen::Memory,
        code: &Code,
        symbols: &[u64],
    ) -> Result<usize, haste_codegen::Error> {
        let exec = memory.write_patched(code, symbols)?;
        self.register_jit_drops(
            (exec, code.bytes().len()),
            (code.unwind_drops(), code.sites()),
            symbols,
        );
        Ok(exec)
    }

    pub(super) fn symbols(&self, relocs: &[SymbolSource]) -> Vec<u64> {
        relocs.iter().map(|sym| self.symbol_value(sym)).collect()
    }

    pub(super) fn compile_jit_one(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        pass: &mut Pass,
    ) -> Result<(usize, bool)> {
        let cfg = pass.stencils.cfg;
        let item = reached.list[index].0.0;
        let wanted0 = self.wanted_len();
        if !cfg.verify
            && cfg.min_fanout > 1
            && let Some(hit) = self.try_first_jit(cx, reached, index, item, wanted0, pass)?
        {
            return Ok(hit);
        }
        self.compile_jit_cached(cx, reached, index, item, wanted0, cfg, pass)
    }

    fn try_first_jit(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        item: ItemRef,
        wanted0: usize,
        pass: &mut Pass,
    ) -> Result<Option<(usize, bool)>> {
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
                pass.stencils.first_jit.insert(
                    item,
                    FirstJitState::First(FirstJit {
                        index,
                        code,
                        symbols,
                        exec,
                        reloc_plan,
                    }),
                );
                Ok(Some((exec, thawed)))
            }
            Some(FirstJitState::First(first)) => {
                let fp1 = self.footprint(cx, reached, first.index, pass)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                pass.stencils.jit_templates.insert(
                    (item, fp1),
                    JitTemplate {
                        stencil: first.code,
                        dict: None,
                        fanout: 1,
                        reloc_plan: first.reloc_plan,
                    },
                );
                pass.stencils
                    .aliases
                    .insert((item, fp1, first.symbols), (first.exec, false));
                pass.stencils
                    .first_jit
                    .insert(item, FirstJitState::Promoted);
                Ok(None)
            }
            Some(FirstJitState::Promoted) => Ok(None),
        }
    }

    fn compile_jit_cached(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        item: ItemRef,
        wanted0: usize,
        cfg: DictConfig,
        pass: &mut Pass,
    ) -> Result<(usize, bool)> {
        let footprint = self.footprint(cx, reached, index, pass)?;
        if self.wanted_len() > wanted0 {
            bail!("unreached inline callee");
        }
        let existing_plan = pass
            .stencils
            .jit_templates
            .get(&(item, footprint))
            .map(|jt| Arc::clone(&jt.reloc_plan));
        let (symbols, prelowered, reused) = match existing_plan {
            Some(plan) => {
                let relocs = self.relocs(cx, reached, &plan, index, pass)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                (self.symbols(&relocs), None, true)
            }
            None => {
                let ((code, relocs, plan), thawed) = if !cfg.verify
                    && pass.ready.get(&index).is_none_or(|r| r.code.is_none())
                    && let Some(hit) =
                        self.thaw_lazy_stencil(cx, reached, index, footprint, wanted0)?
                {
                    (hit, true)
                } else {
                    self.lowered(cx, reached, index, wanted0, pass)?
                };
                if !thawed {
                    self.freeze_lazy(cx, reached, index, footprint, (&code, &plan), pass);
                }
                (self.symbols(&relocs), Some((code, plan)), thawed)
            }
        };
        let alias_key = (item, footprint, symbols.clone());
        if let Some(&(addr, is_dict)) = pass.stencils.aliases.get(&alias_key) {
            if cfg.verify
                && let Some(jt) = pass.stencils.jit_templates.get(&(item, footprint))
            {
                let rec = match (is_dict, &jt.dict) {
                    (true, Some((body_exec, dict_code, _))) => VerifyRecord::Dict {
                        body_exec: *body_exec,
                        stub_exec: addr,
                        stub_bytes: Vec::new(),
                        code: dict_code.clone(),
                    },
                    _ => VerifyRecord::Patched {
                        exec: addr,
                        len: jt.stencil.bytes().len(),
                        sites: jt.stencil.sites().to_vec(),
                        unwind_drops: jt.stencil.unwind_drops().to_vec(),
                    },
                };
                self.verify_single(cx, reached, index, &rec)?;
            }
            return Ok((addr, true));
        }
        let stencil =
            self.ensure_jit_stencil(cx, reached, index, (item, footprint), prelowered, pass)?;
        let fanout = pass
            .stencils
            .jit_templates
            .get(&(item, footprint))
            .map_or(1, |jt| jt.fanout);
        if stencil.bytes().len() > cfg.min_bytes && fanout >= cfg.min_fanout {
            return Ok((
                self.instantiate_jit_dict(
                    cx,
                    reached,
                    index,
                    (item, footprint),
                    symbols,
                    cfg,
                    pass,
                )?,
                reused,
            ));
        }
        let exec = self.write_jit_code(&mut pass.memory, &stencil, &symbols)?;
        pass.stencils.aliases.insert(alias_key, (exec, false));
        if cfg.verify {
            let rec = VerifyRecord::Patched {
                exec,
                len: stencil.bytes().len(),
                sites: stencil.sites().to_vec(),
                unwind_drops: stencil.unwind_drops().to_vec(),
            };
            self.verify_single(cx, reached, index, &rec)?;
        }
        Ok((exec, reused))
    }

    fn ensure_jit_stencil(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        key: (ItemRef, FootprintId),
        prelowered: Option<(Arc<Code>, Arc<RelocPlan>)>,
        pass: &mut Pass,
    ) -> Result<Arc<Code>> {
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
        pass.stencils.jit_templates.insert(
            key,
            JitTemplate {
                stencil: stencil.clone(),
                dict: None,
                fanout: 1,
                reloc_plan,
            },
        );
        Ok(stencil)
    }

    fn instantiate_jit_dict(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        key: (ItemRef, FootprintId),
        symbols: Vec<u64>,
        cfg: DictConfig,
        pass: &mut Pass,
    ) -> Result<usize> {
        let (body_exec, dict_code, dict_plan) = match pass
            .stencils
            .jit_templates
            .get(&key)
            .and_then(|jt| jt.dict.clone())
        {
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
        let (stub_exec, stub_bytes) =
            self.emit_dict_stub_with_tier(body_exec, &dict_symbols, tier, pass)?;
        if tier.is_none() {
            pass.stencils
                .aliases
                .insert((key.0, key.1, symbols), (stub_exec, true));
        }
        if cfg.verify {
            let rec = VerifyRecord::Dict {
                body_exec,
                stub_exec,
                stub_bytes,
                code: dict_code,
            };
            self.verify_single(cx, reached, index, &rec)?;
        }
        Ok(stub_exec)
    }

    fn make_tier_up(
        &self,
        index: usize,
        key: (ItemRef, FootprintId),
        symbols: &[u64],
        threshold: u32,
        pass: &mut Pass,
    ) -> Result<Option<DictTierUp>> {
        if !self.backend.is_lazy_jit() || threshold == 0 || index >= (stub::TIER_UP_BIT as usize) {
            return Ok(None);
        }
        let counter_addr = lock(&self.shared).alloc(8)?;
        pass.stencils.tier_ups.insert(
            index,
            TierUpEntry {
                item: key.0,
                footprint: key.1,
                symbols: symbols.to_vec(),
            },
        );
        Ok(Some(DictTierUp {
            counter_addr,
            threshold,
            stub_entry: stub::entry_addr(),
            encoded_index: (index as u32) | stub::TIER_UP_BIT,
        }))
    }

    pub(super) fn tier_up(&mut self, index: usize) -> Result<usize> {
        let Some(entry) = self.pass.stencils.tier_ups.remove(&index) else {
            let existing = self.slot(index).load(Ordering::Acquire);
            if existing != 0 {
                return Ok(existing);
            }
            bail!("missing tier-up entry for slot {index}");
        };
        let Some(jt) = self
            .pass
            .stencils
            .jit_templates
            .get(&(entry.item, entry.footprint))
        else {
            bail!("missing stencil template for slot {index}");
        };
        let exec = self
            .pass
            .memory
            .write_patched(&jt.stencil, &entry.symbols)?;
        self.register_jit_drops(
            (exec, jt.stencil.bytes().len()),
            (jt.stencil.unwind_drops(), jt.stencil.sites()),
            &entry.symbols,
        );
        self.pass
            .stencils
            .aliases
            .insert((entry.item, entry.footprint, entry.symbols), (exec, false));
        self.publish(index, exec);
        Ok(exec)
    }
}
