use anyhow::Result;
use haste_codegen::{FRAME_CACHE_DEPTH, FRAME_CACHE_LIMIT};

use super::shims::{Emit, Shim, arm64_fwd, arm64_here, emit_arm64_adrp_add};
use super::{Buf, Plan};

pub(super) const UNWIND_STATE_BYTES: u64 = 16;
pub(super) const FRAME_CACHE_BYTES: u64 = 1024;

#[derive(Clone, Copy, Default)]
pub(super) struct TlsLayout {
    pub(super) unwind_off: u64,
    pub(super) frame_off: u64,
    pub(super) cells_off: u64,
    pub(super) total: u64,
}

impl TlsLayout {
    pub(super) fn compute(plan: &Plan<'_>, thread_slots: u32) -> Self {
        let mut cursor = 0_u64;
        let unwind_off = cursor;
        if plan.has_unwind() {
            cursor += UNWIND_STATE_BYTES;
        }
        let frame_off = cursor;
        if plan.has_frame_cache() {
            cursor += FRAME_CACHE_BYTES;
        }
        let cells_off = cursor;
        if plan.has_shim(Shim::ThreadCell) {
            cursor += u64::from(thread_slots.max(1)) * 8;
        }
        Self { unwind_off, frame_off, cells_off, total: cursor.next_multiple_of(16) }
    }
}

fn arm64_add_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    0x9100_0000 | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)
}

pub(super) fn emit_x64_tls_unwind_state(unwind_tls: u64, out: &mut Buf) {
    let disp = (unwind_tls as i64).wrapping_neg() as i32 as u32;
    out.bytes(&[0x64, 0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0, 0x48, 0x8D, 0x80]);
    out.u32(disp);
}

pub(super) fn emit_arm64_tlv_base(c: &Emit<'_>, desc_va: u64, out: &mut Buf) {
    emit_arm64_adrp_add(c.at(out), desc_va, 0, out);
    out.words(&[0xF940_0010, 0xD63F_0200]);
}

pub(super) fn emit_arm64_tlv_unwind_state(c: &Emit<'_>, desc_va: u64, unwind_off: u64, out: &mut Buf) {
    emit_arm64_tlv_base(c, desc_va, out);
    if unwind_off != 0 {
        out.u32(arm64_add_imm(0, 0, unwind_off as u32));
    }
}

pub(super) fn emit_arm64_tlv_thread_cell(c: &Emit<'_>, desc_va: u64, cells_off: u64, out: &mut Buf) -> Result<()> {
    out.words(&[0xAA00_03E9, 0xAA1E_03EA]);
    emit_arm64_tlv_base(c, desc_va, out);
    if cells_off != 0 {
        out.u32(arm64_add_imm(0, 0, cells_off as u32));
    }
    out.words(&[0x8B09_0C0B, 0xF940_0160]);
    let miss = arm64_fwd(0xB400_0000, out);
    out.words(&[0xAA0A_03FE, 0xD65F_03C0]);
    arm64_here(&[miss], out);
    out.words(&[0xD100_C3FF, 0xA900_2BFD, 0xA901_53F3, 0xAA0B_03F3, 0xAA01_03F4]);
    c.words_call(&[0xF100_029F, 0xD280_0029, 0x9A89_1294, 0xF900_13FF, 0x9100_83E0, 0xD280_0801, 0xAA14_03E2], "posix_memalign", out)?;
    let fail = arm64_fwd(0x3500_0000, out);
    c.words_call(&[0xF940_13E0, 0x5280_0001, 0xAA14_03E2], "memset", out)?;
    out.u32(0xF900_0260);
    let done = arm64_fwd(0x1400_0000, out);
    arm64_here(&[fail], out);
    out.u32(0xD280_0000);
    arm64_here(&[done], out);
    out.words(&[0xA941_53F3, 0xA940_7BFD, 0x9100_C3FF, 0xD65F_03C0]);
    Ok(())
}

pub(super) fn emit_arm64_tlv_frame_cache(c: &Emit<'_>, alloc: bool, desc_va: u64, frame_off: u64, out: &mut Buf) -> Result<()> {
    if alloc {
        out.u32(0xF100_001F | (FRAME_CACHE_LIMIT << 10));
        let big = arm64_fwd(0x5400_0002, out);
        out.words(&[0xAA00_03E9, 0xAA1E_03EA]);
        emit_arm64_tlv_base(c, desc_va, out);
        if frame_off != 0 {
            out.u32(arm64_add_imm(0, 0, frame_off as u32));
        }
        out.words(&[0x8B09_0002, 0xF940_0040]);
        let miss = arm64_fwd(0xB400_0000, out);
        out.words(&[0xF940_0008, 0xF900_0048, 0xF940_0449, 0xD100_0529, 0xF900_0449, 0xAA0A_03FE, 0xD65F_03C0]);
        arm64_here(&[miss], out);
        out.words(&[0xAA09_03E0, 0xAA0A_03FE]);
        arm64_here(&[big], out);
        c.tail("malloc", out)
    } else {
        out.u32(0xF100_003F | (FRAME_CACHE_LIMIT << 10));
        let big = arm64_fwd(0x5400_0002, out);
        out.words(&[0xAA00_03E9, 0xAA1E_03EA]);
        emit_arm64_tlv_base(c, desc_va, out);
        if frame_off != 0 {
            out.u32(arm64_add_imm(0, 0, frame_off as u32));
        }
        out.words(&[0x8B01_0002, 0xF940_0448, 0xF100_011F | (u32::from(FRAME_CACHE_DEPTH) << 10)]);
        let full = arm64_fwd(0x5400_0002, out);
        out.words(&[0xF940_004B, 0xF900_012B, 0xF900_0049, 0x9100_0508, 0xF900_0448, 0xAA0A_03FE, 0xD65F_03C0]);
        arm64_here(&[full], out);
        out.words(&[0xAA09_03E0, 0xAA0A_03FE]);
        arm64_here(&[big], out);
        c.tail("free", out)
    }
}
