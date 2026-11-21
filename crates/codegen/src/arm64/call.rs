use super::asm::{Asm, Mem, Reg, Size};
use crate::reloc::{RelocKind, StencilReloc};

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
}
