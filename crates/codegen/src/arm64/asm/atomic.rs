use crate::ir::{AtomicOrd, AtomicRmwOp};

use super::{Asm, Reg, Size};

impl Asm {
    /// `dmb ish`: full inner-shareable memory barrier.
    pub fn dmb(&mut self) {
        self.word(0xD503_3BBF);
    }

    /// `dmb ishld`: inner-shareable load-acquire barrier.
    pub fn dmb_ld(&mut self) {
        self.word(0xD503_39BF);
    }

    /// `ldar` load-acquire.
    pub fn load_acquire(&mut self, size: Size, dst: Reg, addr: Reg) {
        self.emit_gp(dst, 0x08DF_FC00 | (size as u32) << 30 | u32::from(addr.0) << 5 | u32::from(dst.0));
    }

    /// ARMv8.3 RCpc `ldapr` load-acquire.
    pub fn load_acquire_rcpc(&mut self, size: Size, dst: Reg, addr: Reg) {
        self.emit_gp(dst, 0x38BF_C000 | (size as u32) << 30 | u32::from(addr.0) << 5 | u32::from(dst.0));
    }

    /// ARMv8.4 RCpc2 `ldapur` load-acquire with signed 9-bit byte offset.
    pub fn load_acquire_rcpc_off(&mut self, size: Size, dst: Reg, addr: Reg, disp: i32) {
        let imm9 = (disp as u32) & 0x1FF;
        self.emit_gp(dst, 0x1940_0000 | (size as u32) << 30 | (imm9 << 12) | u32::from(addr.0) << 5 | u32::from(dst.0));
    }

    /// `ldxr` or `ldaxr` when `acquire`.
    pub fn load_exclusive(&mut self, acquire: bool, size: Size, dst: Reg, addr: Reg) {
        let base = if acquire { 0x085F_FC00 } else { 0x085F_7C00 };
        self.emit_gp(dst, base | (size as u32) << 30 | u32::from(addr.0) << 5 | u32::from(dst.0));
    }

    /// `stlr` store-release.
    pub fn store_release(&mut self, size: Size, addr: Reg, src: Reg) {
        self.word(0x089F_FC00 | (size as u32) << 30 | u32::from(addr.0) << 5 | u32::from(src.0));
    }

    /// ARMv8.4 RCpc2 `stlur` store-release with signed 9-bit byte offset.
    pub fn store_release_off(&mut self, size: Size, addr: Reg, disp: i32, src: Reg) {
        let imm9 = (disp as u32) & 0x1FF;
        self.word(0x1900_0000 | (size as u32) << 30 | (imm9 << 12) | u32::from(addr.0) << 5 | u32::from(src.0));
    }

    /// `stxr` or `stlxr` when `release`: `status` is zero when the store took.
    pub fn store_exclusive(&mut self, release: bool, size: Size, status: Reg, addr: Reg, src: Reg) {
        let base = if release { 0x0800_FC00 } else { 0x0800_7C00 };
        self.emit_gp(status, base | (size as u32) << 30 | u32::from(status.0) << 16 | u32::from(addr.0) << 5 | u32::from(src.0));
    }

    /// ARMv8.1 LSE atomic RMW (`ldadd`/`ldclr`/`ldeor`/`ldset`/`ldsmax`/`ldsmin`/`ldumax`/`ldumin`/`swp`): reads old value into `dst`.
    pub fn lse_rmw(&mut self, op: AtomicRmwOp, ord: AtomicOrd, size: Size, (addr, val, dst): (Reg, Reg, Reg)) {
        let opc: u32 = match op {
            AtomicRmwOp::Add | AtomicRmwOp::Sub => 0x0000,
            AtomicRmwOp::And => 0x1000,
            AtomicRmwOp::Xor => 0x2000,
            AtomicRmwOp::Or => 0x3000,
            AtomicRmwOp::Smax => 0x4000,
            AtomicRmwOp::Smin => 0x5000,
            AtomicRmwOp::Umax => 0x6000,
            AtomicRmwOp::Umin => 0x7000,
            AtomicRmwOp::Xchg => 0x8000,
            AtomicRmwOp::Nand => unreachable!("nand has no LSE instruction"),
        };
        let ar = (u32::from(ord.has_acquire()) << 23) | (u32::from(ord.has_release()) << 22);
        let word = 0x3820_0000 | (size as u32) << 30 | ar | u32::from(val.0) << 16 | opc | u32::from(addr.0) << 5 | u32::from(dst.0);
        self.emit_gp(dst, word);
    }

    /// ARMv8.1 LSE `cas{a,l}`: compares `[addr]` with `expected_and_old`, writes `replacement` on match, and always writes the old value into `expected_and_old`.
    pub fn lse_cas(&mut self, ord: AtomicOrd, size: Size, (addr, expected_and_old, replacement): (Reg, Reg, Reg)) {
        let ar = (u32::from(ord.has_acquire()) << 22) | (u32::from(ord.has_release()) << 15);
        let word = 0x08A0_7C00 | (size as u32) << 30 | ar | u32::from(expected_and_old.0) << 16 | u32::from(addr.0) << 5 | u32::from(replacement.0);
        self.emit_gp(expected_and_old, word);
    }
}
