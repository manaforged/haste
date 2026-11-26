use anyhow::{Result, bail};
use haste_codegen::{Code, CodeUnwindDrop, Site};

use crate::image::Image;
use crate::image::stencil::{FreshSlots, VerifyRecord};
use crate::lower::{self, Slots};
use crate::sema::Context;
use crate::sema::reach::Reached;

impl Image {
    pub(in crate::image) fn finish_verify(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        records: &[(usize, VerifyRecord)],
    ) -> Result<(), String> {
        let errs = self.verify_records(cx, reached, records);
        if let Some((_, msg)) = errs.into_iter().next() {
            return Err(msg);
        }
        Ok(())
    }

    pub(in crate::image) fn verify_records(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        records: &[(usize, VerifyRecord)],
    ) -> Vec<(usize, String)> {
        let mut errs = Vec::new();
        let fresh = FreshSlots(self, lower::Bodies::default());
        for (index, record) in records {
            fresh.1.clear_transient();
            if let Err(err) = Self::verify_with_fresh(cx, reached, &fresh, *index, record) {
                errs.push((
                    *index,
                    format!("{}: {err:#}", cx.name(reached.list[*index].0.0)),
                ));
            }
        }
        errs
    }

    pub(super) fn verify_single(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        record: &VerifyRecord,
    ) -> Result<()> {
        let fresh = FreshSlots(self, lower::Bodies::default());
        Self::verify_with_fresh(cx, reached, &fresh, index, record)
    }

    fn verify_with_fresh(
        cx: &mut Context<'_>,
        reached: &Reached,
        fresh: &FreshSlots<'_>,
        index: usize,
        record: &VerifyRecord,
    ) -> Result<()> {
        match record {
            VerifyRecord::Patched {
                exec,
                len,
                sites,
                unwind_drops,
            } => Self::verify_patched(
                cx,
                reached,
                fresh,
                index,
                (*exec, *len),
                sites,
                unwind_drops,
            ),
            VerifyRecord::Compiled { code, symbols } => {
                Self::verify_compiled(cx, reached, fresh, index, code, symbols)
            }
            VerifyRecord::Dict {
                body_exec,
                stub_exec,
                stub_bytes,
                code,
            } => Self::verify_dict(
                cx,
                reached,
                fresh,
                index,
                (*body_exec, *stub_exec),
                stub_bytes,
                code,
            ),
        }
    }

    fn verify_compiled(
        cx: &mut Context<'_>,
        reached: &Reached,
        fresh: &FreshSlots<'_>,
        index: usize,
        code: &Code,
        symbols: &[u64],
    ) -> Result<()> {
        let out = lower::lower_stencil(cx, reached, fresh, index)?;
        let slots: &dyn Slots = fresh;
        let fresh_syms: Vec<u64> = out
            .relocs
            .iter()
            .map(|sym| slots.resolve_symbol(sym))
            .collect();
        let fresh_code = haste_codegen::compile(
            out.func,
            &fresh.0.compile_target_for(cx, reached.list[index].0.0),
        )?;
        if code.bytes() != fresh_code.bytes() {
            bail!(
                "stencil verification byte mismatch ({} vs {} bytes)",
                code.bytes().len(),
                fresh_code.bytes().len()
            );
        }
        if code.sites() != fresh_code.sites() {
            bail!("stencil verification Site table mismatch");
        }
        if code.unwind_drops() != fresh_code.unwind_drops() {
            bail!("stencil verification unwind drop table mismatch");
        }
        if code.relocs() != fresh_code.relocs() {
            bail!(
                "stencil verification code.relocs mismatch ({} vs {})",
                code.relocs().len(),
                fresh_code.relocs().len()
            );
        }
        if symbols != fresh_syms.as_slice() {
            let first = symbols.iter().zip(&fresh_syms).position(|(a, b)| a != b);
            bail!(
                "stencil verification symbol mismatch (len {} vs {}, first diff {first:?})",
                symbols.len(),
                fresh_syms.len()
            );
        }
        Ok(())
    }

    fn verify_patched(
        cx: &mut Context<'_>,
        reached: &Reached,
        fresh: &FreshSlots<'_>,
        index: usize,
        (exec, len): (usize, usize),
        sites: &[Site],
        unwind_drops: &[CodeUnwindDrop],
    ) -> Result<()> {
        let out = lower::lower_stencil(cx, reached, fresh, index)?;
        let slots: &dyn Slots = fresh;
        let symbols: Vec<u64> = out
            .relocs
            .iter()
            .map(|sym| slots.resolve_symbol(sym))
            .collect();
        let fresh_code = haste_codegen::compile(
            out.func,
            &fresh.0.compile_target_for(cx, reached.list[index].0.0),
        )?;
        let mut expected = vec![0u8; fresh_code.bytes().len()];
        fresh_code.patch(&symbols, &mut expected)?;
        // SAFETY: exec..exec + len is mapped readable executable code in this image.
        let actual = unsafe { std::slice::from_raw_parts(exec as *const u8, len) };
        if actual != expected.as_slice() {
            bail!(
                "stencil verification byte mismatch ({} vs {} bytes)",
                actual.len(),
                expected.len()
            );
        }
        if sites != fresh_code.sites() {
            bail!("stencil verification Site table mismatch");
        }
        if unwind_drops != fresh_code.unwind_drops() {
            bail!("stencil verification unwind drop table mismatch");
        }
        Ok(())
    }

    fn verify_dict(
        cx: &mut Context<'_>,
        reached: &Reached,
        fresh: &FreshSlots<'_>,
        index: usize,
        (body_exec, stub_exec): (usize, usize),
        stub_bytes: &[u8],
        code: &Code,
    ) -> Result<()> {
        let (func, _, _) = lower::lower_dict(cx, reached, fresh, index)?;
        let fresh_code = haste_codegen::compile(func, &crate::image::stencil::target())?;
        if !fresh_code
            .relocs()
            .iter()
            .all(|reloc| reloc.as_runtime().is_some())
        {
            bail!("dict body verification found non-runtime relocations");
        }
        if body_exec == 0 {
            if code.bytes() != fresh_code.bytes() || code.relocs() != fresh_code.relocs() {
                bail!(
                    "dict body verification mismatch ({} vs {} bytes)",
                    code.bytes().len(),
                    fresh_code.bytes().len()
                );
            }
        } else {
            let mut expected = vec![0u8; fresh_code.bytes().len()];
            fresh_code.patch(&[], &mut expected)?;
            // SAFETY: body_exec..body_exec + code.bytes().len() is mapped readable executable code in this image.
            let actual_body =
                unsafe { std::slice::from_raw_parts(body_exec as *const u8, code.bytes().len()) };
            if actual_body != expected.as_slice() {
                bail!(
                    "dict body verification byte mismatch ({} vs {} bytes)",
                    actual_body.len(),
                    expected.len()
                );
            }
        }
        if code.sites() != fresh_code.sites() {
            bail!("dict body verification Site table mismatch");
        }
        if stub_exec != 0 && !stub_bytes.is_empty() {
            // SAFETY: stub_exec..stub_exec + stub_bytes.len() is mapped readable executable code in this image.
            let actual_stub =
                unsafe { std::slice::from_raw_parts(stub_exec as *const u8, stub_bytes.len()) };
            if actual_stub != stub_bytes {
                bail!("dict entry stub byte mismatch");
            }
        }
        Ok(())
    }
}
