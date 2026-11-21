//! Binds a foreign function to code assembled from the `global_asm!` block that defines its symbol, else to the host symbol of its name.

use haste_codegen::{Code, Memory, Target};

use crate::host::HostSymbol;
use crate::lower::SymbolSource;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::global_asm::Block;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;

use super::aot::{AotCode, AotFn};
use super::{Image, resolve_host_symbol};

/// What a foreign function binds to: code assembled from a `global_asm!` block, at its address under a JIT and zero otherwise, or a host symbol and its address.
pub(super) enum Foreign {
    Asm(usize),
    Host(HostSymbol, usize),
}

impl Image {
    /// Binds the foreign function `index`: assembles the `global_asm!` block that defines its symbol into `memory` under a JIT, or records it for the executable, else resolves the host symbol of its name.
    pub(super) fn bind_foreign(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, memory: Option<&mut Memory>) -> Result<Foreign, String> {
        let item = reached.list[index].0.0;
        let Some(block) = cx.global_asm(item) else {
            return resolve_host_symbol(cx, item, !self.backend.is_jit()).map(|(sym, address)| Foreign::Host(sym, address)).map_err(|error| unread(cx, error));
        };
        let (code, relocs) = assemble(cx, reached, &self.compile_target(), item, &block)?;
        if self.backend.is_aot() {
            self.register_aot_relocs(&relocs);
            self.record_aot(index, AotFn::Direct { code: AotCode::from_code(&code), relocs }, &[]);
            return Ok(Foreign::Asm(0));
        }
        let Some(memory) = memory.filter(|_| self.backend.is_jit()) else { return Ok(Foreign::Asm(0)) };
        let symbols: Vec<u64> = relocs.iter().map(|sym| self.symbol_value(sym)).collect();
        let exec = memory.write_patched(&code, &symbols).map_err(|error| format!("{}: {error}", cx.name(item)))?;
        self.publish(index, exec);
        Ok(Foreign::Asm(exec))
    }
}

fn unread(cx: &mut Context<'_>, error: String) -> String {
    std::iter::once(error).chain(cx.unread_global_asm()).collect::<Vec<_>>().join("; ")
}

fn assemble(cx: &Context<'_>, reached: &Reached, target: &Target, item: ItemRef, block: &Block) -> Result<(Code, Vec<SymbolSource>), String> {
    let failed = |error: String| format!("{}: {error}", cx.name(item));
    let names: Vec<&str> = block.symbols.iter().map(|(name, _)| name.as_str()).collect();
    let assembly = haste_codegen::assemble(&block.text, target, &names).map_err(failed)?;
    let label = format!("{}{}", target.symbol_prefix, cx.symbol_name(item).unwrap_or_default());
    if !assembly.labels.iter().any(|(name, offset)| *name == label && *offset == 0) {
        return Err(failed(format!("its global_asm block does not start at label `{label}`")));
    }
    let relocs = block.symbols.iter().map(|&(_, callee)| slot(cx, reached, callee)).collect::<Result<Vec<_>, _>>()?;
    Ok((assembly.code, relocs))
}

fn slot(cx: &Context<'_>, reached: &Reached, callee: ItemRef) -> Result<SymbolSource, String> {
    match reached.index.get(&(callee, Substs::EMPTY)) {
        Some(&index) if !reached.is_untyped(index) => Ok(SymbolSource::Slot(index)),
        _ => Err(format!("global_asm `sym {}` was not reached", cx.name(callee))),
    }
}
