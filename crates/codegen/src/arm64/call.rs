use super::asm::{Asm, Mem, Reg, Size};
use crate::reloc::{RelocKind, RuntimeSymbol, StencilReloc};

impl Asm {
    /// `bl` to relocation symbol `sym` when `direct`, else a load of its call table slot into `scratch` and a `blr` through it.
    pub(in crate::arm64) fn call_symbol(&mut self, sym: u32, direct: bool, scratch: Reg) -> StencilReloc {
        if direct {
            let at = self.bl_reloc();
            return StencilReloc::symbol(at, sym, RelocKind::Arm64Call26);
        }
        let at = self.mov_symbol(scratch);
        self.load(Size::X, false, scratch, Mem::new(scratch, 0));
        self.blr(scratch);
        StencilReloc::symbol(at, sym, RelocKind::Arm64Movw64)
    }

    fn call_direct(&mut self, direct: bool, scratch: Reg, build: impl FnOnce(u32, RelocKind) -> StencilReloc) -> StencilReloc {
        if direct {
            let at = self.bl_reloc();
            return build(at, RelocKind::Arm64Call26);
        }
        let at = self.mov_symbol(scratch);
        self.blr(scratch);
        build(at, RelocKind::Arm64Movw64)
    }

    /// `bl` to direct symbol `sym` when `direct`, else `movz/movk` into `scratch` and `blr`.
    pub(in crate::arm64) fn call_direct_symbol(&mut self, sym: u32, direct: bool, scratch: Reg) -> StencilReloc {
        self.call_direct(direct, scratch, |at, kind| StencilReloc::symbol(at, sym, kind))
    }

    /// `bl` to runtime helper `rt` when `direct`, else `movz/movk` into `scratch` and `blr`.
    pub(in crate::arm64) fn call_runtime(&mut self, rt: RuntimeSymbol, direct: bool, scratch: Reg) -> StencilReloc {
        self.call_direct(direct, scratch, |at, kind| StencilReloc::runtime(at, rt, kind))
    }
}
