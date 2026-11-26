use std::sync::Arc;

use haste_codegen::Code;

use super::{CachedTemplate, DictConfig, VerifyRecord, aot_relocs_valid, compile_traced, item_err};
use crate::image::aot::{AotCode, AotFn};
use crate::image::{Image, Pass};
use crate::lower::{self, FootprintId, SymbolSource};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;

impl Image {
    pub(in crate::image) fn compile_bucket(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        indices: &[usize],
        cfg: DictConfig,
        (record_cache, allow_singleton): (bool, bool),
        pass: &mut Pass,
    ) -> Vec<(usize, String)> {
        if self.backend.is_check() && !cfg.verify {
            return self.check_bucket(cx, reached, indices, pass);
        }
        if allow_singleton && indices.len() == 1 && (!self.backend.is_aot() || cfg.verify) {
            let index = indices[0];
            return self
                .compile_singleton(cx, reached, index, pass)
                .err()
                .map(|err| (index, err))
                .into_iter()
                .collect();
        }
        self.compile_partitioned(
            cx,
            reached,
            indices,
            cfg,
            record_cache && !self.backend.is_aot(),
            pass,
        )
    }

    fn check_bucket(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        indices: &[usize],
        pass: &mut Pass,
    ) -> Vec<(usize, String)> {
        let mut errors = Vec::new();
        let check_layout = !reached.unlaid.is_empty();
        for &index in indices {
            if check_layout
                && let Ok(typed) = &reached.list[index].1
                && let Err(err) = crate::sema::reach::lay_out(cx, typed)
            {
                errors.push((index, item_err(cx, reached.list[index].0.0, &err)));
                continue;
            }
            pass.record_size(index, 0, 0);
        }
        errors
    }

    fn try_aot_singleton_hit(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        slice: &crate::substrate::StencilSlice<'_>,
        pass: &mut Pass,
    ) -> Result<bool, String> {
        let Ok(code) = slice.to_code().map(Arc::new) else {
            return Ok(false);
        };
        let item = reached.list[index].0.0;
        let relocs = if code.relocs().iter().any(|r| r.symbol_index().is_some())
            || !code.unwind_drops().is_empty()
        {
            if let Some(plan) =
                self.decode_aot_reloc_plan(cx.outline, &cx.types, reached, false, slice.plan())
                && let Ok(r) = lower::eval_relocs(cx, reached, self, &plan, index)
                && cx.deferred_evals.is_empty()
            {
                self.register_aot_relocs(&r);
                r
            } else if !cx.deferred_evals.is_empty() {
                return Ok(true);
            } else {
                let out = lower::lower_stencil(cx, reached, self, index)
                    .map_err(|e| item_err(cx, item, e))?;
                self.register_aot_relocs(&out.relocs);
                out.relocs
            }
        } else {
            Vec::new()
        };
        if !aot_relocs_valid(&code, relocs.len()) {
            return Ok(false);
        }
        pass.record_size(index, code.bytes().len() as u32, 0);
        self.record_aot(
            index,
            AotFn::Direct {
                code: AotCode::from_code(&code),
                relocs,
            },
            &[],
        );
        Ok(true)
    }

    fn try_singleton_substrate(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        fp: FootprintId,
        pass: &mut Pass,
    ) -> Result<bool, String> {
        let Some((_, stencil_key, fp_bits)) = self.frozen_stencil_key(cx, reached, index, fp)
        else {
            return Ok(false);
        };
        let Some(substrate) = cx.store().substrate().cloned() else {
            return Ok(false);
        };
        let Some(slice) = substrate.stencil(stencil_key, fp_bits) else {
            return Ok(false);
        };
        if self.backend.is_check() {
            pass.record_size(index, slice.code().len() as u32, 0);
            return Ok(true);
        }
        if self.backend.is_aot() && self.try_aot_singleton_hit(cx, reached, index, &slice, pass)? {
            return Ok(true);
        }
        let (Ok(unwind_drops), Ok(sites)) = (slice.decode_unwind_drops(), slice.decode_sites())
        else {
            return Ok(false);
        };
        let item = reached.list[index].0.0;
        let symbols: Vec<u64> =
            if slice.decode_relocs().is_ok_and(|r| r.is_empty()) && unwind_drops.is_empty() {
                Vec::new()
            } else {
                let out = lower::lower_stencil(cx, reached, self, index)
                    .map_err(|e| item_err(cx, item, e))?;
                out.relocs
                    .iter()
                    .map(|sym| self.symbol_value(sym))
                    .collect()
            };
        let Ok(patched) = crate::substrate::patch_stencil(&slice, &symbols) else {
            return Ok(false);
        };
        let exec = pass
            .memory
            .write(&patched)
            .map_err(|e| item_err(cx, item, e))?;
        self.register_jit_drops(
            (exec, patched.len()),
            (&unwind_drops, &sites, slice.saved_regs()),
            &symbols,
        );
        self.publish(index, exec);
        pass.record_size(index, patched.len() as u32, 0);
        Ok(true)
    }

    fn emit_singleton_code(
        &self,
        cx: &Context<'_>,
        item: ItemRef,
        index: usize,
        code: Arc<Code>,
        relocs: Vec<SymbolSource>,
        pass: &mut Pass,
    ) -> Result<(), String> {
        if self.backend.is_jit() {
            let symbols: Vec<u64> = relocs.iter().map(|sym| self.symbol_value(sym)).collect();
            let exec = self
                .write_jit_code(&mut pass.memory, &code, &symbols)
                .map_err(|e| item_err(cx, item, e))?;
            self.publish(index, exec);
        } else {
            self.register_aot_relocs(&relocs);
            self.record_aot(
                index,
                AotFn::Direct {
                    code: AotCode::from_code(&code),
                    relocs,
                },
                &[],
            );
        }
        Ok(())
    }

    fn eval_singleton_fp(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        item: ItemRef,
        index: usize,
    ) -> Result<Option<FootprintId>, String> {
        if self.backend.is_lazy_jit() || !cx.outline.is_fixed(item.krate) {
            return Ok(None);
        }
        lower::eval_footprint(cx, reached, self, index)
            .map(Some)
            .map_err(|e| item_err(cx, item, e))
    }

    fn compile_singleton(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        pass: &mut Pass,
    ) -> Result<(), String> {
        let started = std::time::Instant::now();
        let item = reached.list[index].0.0;
        let footprint = self.eval_singleton_fp(cx, reached, item, index)?;
        if !pass.stencils.cfg.verify
            && match footprint {
                Some(fp) => self.try_singleton_substrate(cx, reached, index, fp, pass)?,
                None => false,
            }
        {
            return Ok(());
        }
        let out =
            lower::lower_stencil(cx, reached, self, index).map_err(|e| item_err(cx, item, e))?;
        if self.backend.is_check() && !pass.stencils.cfg.verify {
            pass.record_size(index, 0, started.elapsed().as_nanos() as u64);
            return Ok(());
        }
        let target = self.compile_target_for(cx, item);
        let code = Arc::new(
            compile_traced(cx, reached, index, out.func, &target)
                .map_err(|e| item_err(cx, item, e))?,
        );
        if let Some(fp) = footprint.filter(|&fp| {
            aot_relocs_valid(&code, out.relocs.len())
                && lower::eval_footprint(cx, reached, self, index).ok() == Some(fp)
        }) {
            self.freeze_stencil(
                cx,
                reached,
                index,
                fp,
                false,
                &code,
                Some(&out.reloc_plan),
                pass,
            );
        }
        let code_len = code.bytes().len() as u32;
        self.emit_singleton_code(cx, item, index, code, out.relocs, pass)?;
        pass.record_size(index, code_len, started.elapsed().as_nanos() as u64);
        Ok(())
    }

    pub(in crate::image) fn reuse_cached(
        &self,
        (item, footprint): (ItemRef, FootprintId),
        index: usize,
        symbols: &[u64],
        aot_relocs: Option<&Vec<SymbolSource>>,
        pass: &mut Pass,
    ) -> Option<VerifyRecord> {
        let alias_key = (item, footprint, symbols.to_vec());
        if let Some(&(addr_or_leader, is_dict)) = pass.stencils.aliases.get(&alias_key) {
            if self.backend.is_jit() {
                self.publish(index, addr_or_leader);
            }
            if aot_relocs.is_some() {
                self.record_aot(index, AotFn::Alias(addr_or_leader), &[]);
            }
            return self.cached_verify_record(
                (item, footprint),
                addr_or_leader,
                is_dict,
                symbols,
                pass,
            );
        }
        let template = pass.stencils.templates.get(&(item, footprint)).cloned()?;
        match template {
            CachedTemplate::Stencil { code, .. } => {
                let exec = if !self.backend.is_jit() {
                    0
                } else {
                    let e = self.write_jit_code(&mut pass.memory, &code, symbols).ok()?;
                    self.publish(index, e);
                    e
                };
                if let Some(relocs) = aot_relocs {
                    self.record_aot(
                        index,
                        AotFn::Direct {
                            code: AotCode::from_code(&code),
                            relocs: relocs.clone(),
                        },
                        &[],
                    );
                }
                pass.record_size(index, code.bytes().len() as u32, 0);
                if !self.backend.is_check() {
                    let stored = if aot_relocs.is_some() { index } else { exec };
                    pass.stencils.aliases.insert(alias_key, (stored, false));
                }
                if self.backend.is_jit() {
                    Some(VerifyRecord::Patched {
                        exec,
                        len: code.bytes().len(),
                        sites: code.sites().to_vec(),
                        unwind_drops: code.unwind_drops().to_vec(),
                    })
                } else {
                    Some(VerifyRecord::Compiled {
                        code,
                        symbols: symbols.to_vec(),
                    })
                }
            }
            CachedTemplate::Dict {
                body_exec, code, ..
            } => {
                let (stub_exec, stub_bytes) = if !self.backend.is_jit() {
                    (0, Vec::new())
                } else {
                    let (e, b) = self.emit_dict_stub(body_exec, symbols, pass).ok()?;
                    self.publish(index, e);
                    (e, b)
                };
                if let Some(relocs) = aot_relocs {
                    self.record_aot(
                        index,
                        AotFn::Dict {
                            body: AotCode::from_code(&code),
                            relocs: relocs.clone(),
                        },
                        &[],
                    );
                }
                pass.record_size(index, stub_bytes.len() as u32, 0);
                if !self.backend.is_check() {
                    let stored = if aot_relocs.is_some() {
                        index
                    } else {
                        stub_exec
                    };
                    pass.stencils.aliases.insert(alias_key, (stored, true));
                }
                Some(VerifyRecord::Dict {
                    body_exec,
                    stub_exec,
                    stub_bytes,
                    code,
                })
            }
        }
    }

    fn cached_verify_record(
        &self,
        key: (ItemRef, FootprintId),
        addr: usize,
        is_dict: bool,
        symbols: &[u64],
        pass: &Pass,
    ) -> Option<VerifyRecord> {
        match (is_dict, pass.stencils.templates.get(&key)?) {
            (false, CachedTemplate::Stencil { code, .. }) if self.backend.is_jit() => {
                Some(VerifyRecord::Patched {
                    exec: addr,
                    len: code.bytes().len(),
                    sites: code.sites().to_vec(),
                    unwind_drops: code.unwind_drops().to_vec(),
                })
            }
            (false, CachedTemplate::Stencil { code, .. }) => Some(VerifyRecord::Compiled {
                code: code.clone(),
                symbols: symbols.to_vec(),
            }),
            (
                true,
                CachedTemplate::Dict {
                    body_exec, code, ..
                },
            ) => Some(VerifyRecord::Dict {
                body_exec: *body_exec,
                stub_exec: addr,
                stub_bytes: Vec::new(),
                code: code.clone(),
            }),
            _ => None,
        }
    }
}
