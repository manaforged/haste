use anyhow::Result;
use haste_codegen::Arch;

use super::super::shims::{
    Emit, arm64_fwd, arm64_here, arm64_jcc, emit_arm64_adrp_add, emit_arm64_unwind_state,
    emit_rel_branch,
};
use super::super::{Buf, SectionAddrs, tls};
use super::{REGS_PTR_OFFSET, STOP_FP_OFFSET};

const _: () = assert!(STOP_FP_OFFSET + 8 == REGS_PTR_OFFSET);

fn arm64_patch_adr(at: usize, target: usize, out: &mut Buf) {
    let delta = (target as isize - at as isize) as u32;
    let word = u32::from_le_bytes([out.0[at], out.0[at + 1], out.0[at + 2], out.0[at + 3]]);
    out.patch_u32(
        at,
        (word & 0x9F00_001F) | ((delta & 3) << 29) | (((delta >> 2) & 0x7_FFFF) << 5),
    );
}

/// Emits the ARM64 `catch_unwind` runtime shim.
pub(in crate::aot) fn emit_arm64_catch_unwind(
    c: &Emit<'_>,
    addrs: SectionAddrs,
    out: &mut Buf,
) -> Result<()> {
    if !c.win && addrs.tlv_desc != 0 {
        out.words(&[
            0xA9BC_7BFD,
            0x9100_03FD,
            0xA901_53F3,
            0xA902_5BF5,
            0xA903_63F7,
            0xD102_C3FF,
            0xAA00_03E9,
            0xA903_0BE1,
        ]);
        tls::emit_arm64_tlv_unwind_state(c, addrs.tlv_desc, addrs.unwind_tls, out);
        out.words(&[
            0xAA00_03F3,
            0xA940_2E6A,
            0xA900_2FEA,
            0xF900_13FD,
            0xA905_6BF9,
            0xA906_73FB,
            0x6D07_27E8,
            0x6D08_2FEA,
            0x6D09_37EC,
            0x6D0A_3FEE,
        ]);
        let caught_adr = arm64_fwd(0x1000_000A, out);
        out.words(&[
            0xF900_23EA,
            0x9100_03EA,
            0xF900_026A,
            0xAA01_03E0,
            0xD63F_0120,
            0xF940_03E8,
            0xF900_0268,
            0x5280_0000,
            0xF940_0BB3,
            0x9100_03BF,
            0xA8C4_7BFD,
            0xD65F_03C0,
        ]);
        arm64_patch_adr(caught_adr, out.pos(), out);
        out.words(&[
            0xAA16_03E0,
            0x9100_43E1,
            0xD63F_02E0,
            0x5280_0020,
            0x9100_03BF,
            0xA943_63F7,
            0xA942_5BF5,
            0xA941_53F3,
            0xA8C4_7BFD,
            0xD65F_03C0,
        ]);
        return Ok(());
    }
    let sub_sp = if c.win { 0xD109_03FF } else { 0xD102_C3FF };
    let add_sp = if c.win { 0x9109_03FF } else { 0x9102_C3FF };
    out.words(&[
        0xA9BC_7BFD,
        0x9100_03FD,
        0xA901_53F3,
        0xA902_5BF5,
        0xA903_63F7,
        sub_sp,
        0xAA00_03F4,
        0xAA01_03F6,
        0xAA02_03F7,
    ]);
    emit_arm64_unwind_state(c, addrs, out)?;
    let no_state = arm64_fwd(0xB400_0000, out);
    if c.win {
        out.words(&[
            0xAA00_03F3,
            0xA940_2668,
            0xA900_27E8,
            0xA901_7FFF,
            0xF900_03FD | ((STOP_FP_OFFSET / 8) << 10),
            0x9100_03E8,
            0xF900_0268,
            0x9100_C3E0,
            0xAA1F_03E1,
        ]);
        c.call("_setjmp", out)?;
        let caught = arm64_fwd(0x3500_0000, out);
        let ok_zero = arm64_jcc(
            &[0xAA16_03E0, 0xD63F_0280, 0xF940_03E8, 0xF900_0268],
            0x1400_0000,
            out,
        );
        arm64_here(&[no_state], out);
        out.words(&[0xAA16_03E0, 0xD63F_0280]);
        arm64_here(&[ok_zero], out);
        let done = arm64_jcc(&[0x5280_0000], 0x1400_0000, out);
        arm64_here(&[caught], out);
        out.words(&[0xAA16_03E0, 0x9100_43E1, 0xD63F_02E0, 0x5280_0020]);
        arm64_here(&[done], out);
    } else {
        out.words(&[
            0xAA00_03F3,
            0xA940_2668,
            0xA900_27E8,
            0x9100_03E8,
            0xF900_13FD,
            0xA903_5FF6,
            0xA905_6BF9,
            0xA906_73FB,
            0x6D07_27E8,
            0x6D08_2FEA,
            0x6D09_37EC,
            0x6D0A_3FEE,
        ]);
        let caught_adr = arm64_fwd(0x1000_0009, out);
        out.words(&[
            0xF900_23E9,
            0xF900_0268,
            0xAA16_03E0,
            0xD63F_0280,
            0xF940_03E8,
            0xF900_0268,
            0x5280_0000,
        ]);
        let epilogue = out.pos();
        out.words(&[
            add_sp,
            0xA943_63F7,
            0xA942_5BF5,
            0xA941_53F3,
            0xA8C4_7BFD,
            0xD65F_03C0,
        ]);
        arm64_here(&[no_state], out);
        out.words(&[0xAA16_03E0, 0xD63F_0280, 0x5280_0000]);
        let b_no = ((epilogue as isize - out.pos() as isize) >> 2) as u32 & 0x03FF_FFFF;
        out.u32(0x1400_0000 | b_no);
        arm64_patch_adr(caught_adr, out.pos(), out);
        out.words(&[0xAA16_03E0, 0x9100_43E1, 0xD63F_02E0, 0x5280_0020]);
        let b_caught = ((epilogue as isize - out.pos() as isize) >> 2) as u32 & 0x03FF_FFFF;
        out.u32(0x1400_0000 | b_caught);
        return Ok(());
    }
    out.words(&[
        add_sp,
        0xA943_63F7,
        0xA942_5BF5,
        0xA941_53F3,
        0xA8C4_7BFD,
        0xD65F_03C0,
    ]);
    Ok(())
}

/// Emits the ARM64 `start_panic` runtime shim.
pub(in crate::aot) fn emit_arm64_start_panic(
    c: &Emit<'_>,
    addrs: SectionAddrs,
    out: &mut Buf,
) -> Result<()> {
    if !c.win {
        out.words(&[
            0xA9BF_7BFD,
            0x9100_03FD,
            0xD103_C3FF,
            0xA903_53F3,
            0xA904_5BF5,
            0xA905_63F7,
            0xA906_6BF9,
            0xA907_73FB,
            0x6D08_27E8,
            0x6D09_2FEA,
            0x6D0A_37EC,
            0x6D0B_3FEE,
            0xAA00_03F4,
            0xAA01_03F6,
        ]);
        emit_arm64_unwind_state(c, addrs, out)?;
        let no_state = arm64_fwd(0xB400_0000, out);
        out.words(&[
            0xAA00_03F3,
            0xF940_0EC8,
            0xAA14_03E0,
            0xD63F_0100,
            0xAA00_03F4,
            0xAA01_03F5,
            0xA900_7FF3,
            0x9100_C3E8,
            0xA900_23FF | ((STOP_FP_OFFSET / 8) << 15),
        ]);
        emit_arm64_walk_call(c, addrs, true, out);
        let uncaught = arm64_fwd(0xB400_0000, out);
        out.words(&[
            0xA901_5674,
            0x9100_4260,
            0xA941_77E9,
            0xF940_13EA,
            0xA943_53F3,
            0xA944_5BF5,
            0xA945_63F7,
            0xA946_6BF9,
            0xA947_73FB,
            0x6D48_27E8,
            0x6D49_2FEA,
            0x6D4A_37EC,
            0x6D4B_3FEE,
            0x9100_015F,
            0xD61F_0120,
        ]);
        arm64_here(&[no_state, uncaught], out);
        c.call("abort", out)?;
        out.u32(0xD420_0020);
        return Ok(());
    }
    out.words(&[
        0xA9BC_7BFD,
        0x9100_03FD,
        0xA901_53F3,
        0xA902_5BF5,
        0xA903_63F7,
        0xAA00_03F4,
        0xAA01_03F6,
    ]);
    emit_arm64_unwind_state(c, addrs, out)?;
    let no_state = arm64_fwd(0xB400_0000, out);
    let no_frame = arm64_jcc(&[0xAA00_03F3, 0xF940_0277], 0xB400_0017, out);
    out.words(&[
        0xF940_02E8,
        0xF900_0268,
        0xF940_0EC8,
        0xAA14_03E0,
        0xD63F_0100,
        0xA901_06E0,
        0xF900_02F3,
    ]);
    emit_arm64_walk_call(c, addrs, false, out);
    c.words_call(&[0x9100_C2E0, 0x5280_0021], "longjmp", out)?;
    arm64_here(&[no_state, no_frame], out);
    c.call("abort", out)?;
    out.u32(0xD420_0020);
    Ok(())
}

fn emit_arm64_walk_call(c: &Emit<'_>, addrs: SectionAddrs, ctx_sp: bool, out: &mut Buf) {
    out.words(&[0xAA1D_03E0, if ctx_sp { 0x9100_03E1 } else { 0xAA17_03E1 }]);
    emit_arm64_adrp_add(c.at(out), addrs.text, 2, out);
    if addrs.unwind_table == 0 {
        out.u32(0xD280_0003);
    } else {
        emit_arm64_adrp_add(c.at(out), addrs.unwind_table, 3, out);
    }
    emit_rel_branch(Arch::Arm64, true, c.at(out), addrs.unwind_walk, out);
}
