use anyhow::{Context as _, Result, anyhow};
use haste_codegen::{AbiParam, Arch, CallConv, Effort, Function, FunctionBuilder, IntCC, Os, Signature, TARGETS, Type};

use super::shims::{
    Emit, arm64_fwd, arm64_here, arm64_jcc, emit_arm64_adrp_add, emit_arm64_unwind_state, emit_rel_branch, emit_x64_unwind_state,
    x64_fwd32, x64_jcc, x64_patch8, x64_patch32,
};
use super::{Buf, SectionAddrs};

/// Byte offset of `stop_fp` inside the AOT `catch_unwind` frame record.
pub(super) const STOP_FP_OFFSET: u32 = 0x20;

fn arm64_patch_adr(at: usize, target: usize, out: &mut Buf) {
    let delta = (target as isize - at as isize) as u32;
    let word = u32::from_le_bytes([out.0[at], out.0[at + 1], out.0[at + 2], out.0[at + 3]]);
    out.patch_u32(at, (word & 0x9F00_001F) | ((delta & 3) << 29) | (((delta >> 2) & 0x7_FFFF) << 5));
}

pub(super) fn emit_x64_catch_unwind(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    if !c.win && addrs.unwind_tls != 0 {
        let disp0 = (addrs.unwind_tls as i64).wrapping_neg() as i32 as u32;
        let disp8 = ((addrs.unwind_tls as i64).wrapping_neg() + 8) as i32 as u32;
        out.bytes(&[0x55, 0x48, 0x89, 0xE5, 0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x48, 0x83, 0xEC, 0x58]);
        out.bytes(&[0x64, 0x48, 0x8B, 0x04, 0x25]);
        out.u32(disp0);
        out.bytes(&[0x48, 0x89, 0x04, 0x24, 0x64, 0x48, 0x8B, 0x04, 0x25]);
        out.u32(disp8);
        out.bytes(&[0x48, 0x89, 0x44, 0x24, 0x08, 0x48, 0x89, 0x6C, 0x24, STOP_FP_OFFSET as u8, 0x48, 0x89, 0x74, 0x24, 0x30, 0x48, 0x89, 0x54, 0x24, 0x38]);
        let caught_lea = x64_fwd32(&[0x48, 0x8D, 0x05], out);
        out.bytes(&[0x48, 0x89, 0x44, 0x24, 0x40, 0x64, 0x48, 0x89, 0x24, 0x25]);
        out.u32(disp0);
        out.bytes(&[0x48, 0x89, 0xF8, 0x48, 0x89, 0xF7, 0xFF, 0xD0, 0x48, 0x8B, 0x04, 0x24, 0x64, 0x48, 0x89, 0x04, 0x25]);
        out.u32(disp0);
        out.bytes(&[0x31, 0xC0, 0xC9, 0xC3]);
        x64_patch32(caught_lea, out);
        out.bytes(&[0x4C, 0x89, 0xF7, 0x48, 0x8D, 0x74, 0x24, 0x10, 0x41, 0xFF, 0xD7, 0xB8, 1, 0, 0, 0, 0x48, 0x83, 0xC4, 0x58, 0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0x5D, 0xC3]);
        return Ok(());
    }
    let d: u8 = if c.win { 0x20 } else { 0 };
    let frame: u32 = if c.win { 0x248 + u32::from(d) } else { 0x58 };
    out.bytes(&[0x55, 0x48, 0x89, 0xE5, 0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57]);
    if c.win {
        out.bytes(&[0x48, 0x81, 0xEC]);
        out.u32(frame);
        out.bytes(&[0x49, 0x89, 0xCC, 0x49, 0x89, 0xD6, 0x4D, 0x89, 0xC7]);
    } else {
        out.bytes(&[0x48, 0x83, 0xEC, frame as u8, 0x49, 0x89, 0xFC, 0x49, 0x89, 0xF6, 0x49, 0x89, 0xD7]);
    }
    emit_x64_unwind_state(c, addrs, out)?;
    let no_state = x64_jcc(&[0x48, 0x85, 0xC0], 0x74, out);
    let call_try: &[u8] = if c.win { &[0x4C, 0x89, 0xF1, 0x41, 0xFF, 0xD4] } else { &[0x4C, 0x89, 0xF7, 0x41, 0xFF, 0xD4] };
    if c.win {
        out.bytes(&[
            0x48, 0x89, 0xC3, 0x48, 0x8B, 0x03, 0x48, 0x89, 0x44, 0x24, d, 0x48, 0x8B, 0x43, 0x08, 0x48, 0x89, 0x44, 0x24, d + 8,
            0x48, 0xC7, 0x44, 0x24, d + 0x10, 0, 0, 0, 0, 0x48, 0xC7, 0x44, 0x24, d + 0x18, 0, 0, 0, 0,
            0x48, 0x89, 0x6C, 0x24, d + STOP_FP_OFFSET as u8, 0x48, 0x8D, 0x44, 0x24, d, 0x48, 0x89, 0x03,
        ]);
        c.bytes_call(&[0x48, 0x8D, 0x4C, 0x24, d + 0x30, 0x31, 0xD2], "_setjmp", out)?;
        let caught = x64_jcc(&[0x85, 0xC0], 0x75, out);
        out.bytes(call_try);
        let ok_zero = x64_jcc(&[0x48, 0x8B, 0x44, 0x24, d, 0x48, 0x89, 0x03], 0xEB, out);
        x64_patch8(no_state, out);
        out.bytes(call_try);
        x64_patch8(ok_zero, out);
        let done = x64_jcc(&[0x31, 0xC0], 0xEB, out);
        x64_patch8(caught, out);
        out.bytes(&[0x4C, 0x89, 0xF1, 0x48, 0x8D, 0x54, 0x24, d + 0x10, 0x41, 0xFF, 0xD7, 0xB8, 1, 0, 0, 0]);
        x64_patch8(done, out);
        out.bytes(&[0x48, 0x81, 0xC4]);
        out.u32(frame);
    } else {
        out.bytes(&[
            0x48, 0x89, 0xC3, 0x48, 0x8B, 0x03, 0x48, 0x89, 0x04, 0x24, 0x48, 0x8B, 0x43, 0x08, 0x48, 0x89, 0x44, 0x24, 0x08,
            0x48, 0x89, 0x6C, 0x24, STOP_FP_OFFSET as u8, 0x4C, 0x89, 0x74, 0x24, 0x30, 0x4C, 0x89, 0x7C, 0x24, 0x38,
        ]);
        let caught_lea = x64_fwd32(&[0x48, 0x8D, 0x05], out);
        out.bytes(&[0x48, 0x89, 0x44, 0x24, 0x40, 0x48, 0x89, 0x23]);
        out.bytes(call_try);
        let ok_zero = x64_jcc(&[0x48, 0x8B, 0x04, 0x24, 0x48, 0x89, 0x03], 0xEB, out);
        x64_patch8(no_state, out);
        out.bytes(call_try);
        x64_patch8(ok_zero, out);
        let done = x64_jcc(&[0x31, 0xC0], 0xEB, out);
        x64_patch32(caught_lea, out);
        out.bytes(&[0x4C, 0x89, 0xF7, 0x48, 0x8D, 0x74, 0x24, 0x10, 0x41, 0xFF, 0xD7, 0xB8, 1, 0, 0, 0]);
        x64_patch8(done, out);
        out.bytes(&[0x48, 0x83, 0xC4, frame as u8]);
    }
    out.bytes(&[0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0x5D, 0xC3]);
    Ok(())
}

pub(super) fn emit_x64_start_panic(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    out.bytes(&[0x55, 0x48, 0x89, 0xE5, 0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57]);
    out.bytes(if c.win { &[0x48, 0x83, 0xEC, 0x28, 0x49, 0x89, 0xCC, 0x49, 0x89, 0xD6] } else { &[0x48, 0x83, 0xEC, 0x08, 0x49, 0x89, 0xFC, 0x49, 0x89, 0xF6] });
    emit_x64_unwind_state(c, addrs, out)?;
    out.bytes(&[0x48, 0x85, 0xC0]);
    let no_state = x64_fwd32(&[0x0F, 0x84], out);
    out.bytes(&[0x48, 0x89, 0xC3, 0x4C, 0x8B, 0x3B, 0x4D, 0x85, 0xFF]);
    let no_frame = x64_fwd32(&[0x0F, 0x84], out);
    out.bytes(&[0x49, 0x8B, 0x07, 0x48, 0x89, 0x03, 0x49, 0x8B, 0x46, 0x18]);
    out.bytes(if c.win { &[0x4C, 0x89, 0xE1] } else { &[0x4C, 0x89, 0xE7] });
    out.bytes(&[0xFF, 0xD0, 0x49, 0x89, 0x47, 0x10, 0x49, 0x89, 0x57, 0x18, 0x49, 0x89, 0x1F]);
    emit_x64_walk_call(c, addrs, out);
    if c.win {
        c.bytes_call(&[0x49, 0x8D, 0x4F, 0x30, 0xBA, 1, 0, 0, 0], "longjmp", out)?;
    } else {
        out.bytes(&[
            0x49, 0x8B, 0x6F, STOP_FP_OFFSET as u8, 0x4C, 0x89, 0xFC, 0x4D, 0x8B, 0x77, 0x30,
            0x49, 0x8B, 0x47, 0x40, 0x4D, 0x8B, 0x7F, 0x38, 0xFF, 0xE0,
        ]);
    }
    [no_state, no_frame].into_iter().for_each(|l| x64_patch32(l, out));
    c.call("abort", out)?;
    out.bytes(&[0x0F, 0x0B]);
    Ok(())
}

pub(super) fn emit_arm64_catch_unwind(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    if !c.win && addrs.tlv_desc != 0 {
        out.words(&[
            0xA9BC_7BFD, 0x9100_03FD, 0xA901_53F3, 0xA902_5BF5, 0xA903_63F7, 0xD102_C3FF,
            0xAA00_03E9, 0xA903_0BE1,
        ]);
        super::tls::emit_arm64_tlv_unwind_state(c, addrs.tlv_desc, addrs.unwind_tls, out);
        out.words(&[
            0xAA00_03F3, 0xA940_2E6A, 0xA900_2FEA, 0xF900_13FD,
            0xA905_6BF9, 0xA906_73FB, 0x6D07_27E8, 0x6D08_2FEA, 0x6D09_37EC, 0x6D0A_3FEE,
        ]);
        let caught_adr = arm64_fwd(0x1000_000A, out);
        out.words(&[
            0xF900_23EA, 0x9100_03EA, 0xF900_026A, 0xAA01_03E0, 0xD63F_0120,
            0xF940_03E8, 0xF900_0268, 0x5280_0000, 0xF940_0BB3, 0x9100_03BF, 0xA8C4_7BFD, 0xD65F_03C0,
        ]);
        arm64_patch_adr(caught_adr, out.pos(), out);
        out.words(&[
            0xAA16_03E0, 0x9100_43E1, 0xD63F_02E0, 0x5280_0020, 0x9100_03BF,
            0xA943_63F7, 0xA942_5BF5, 0xA941_53F3, 0xA8C4_7BFD, 0xD65F_03C0,
        ]);
        return Ok(());
    }
    let sub_sp = if c.win { 0xD109_03FF } else { 0xD102_C3FF };
    let add_sp = if c.win { 0x9109_03FF } else { 0x9102_C3FF };
    out.words(&[0xA9BC_7BFD, 0x9100_03FD, 0xA901_53F3, 0xA902_5BF5, 0xA903_63F7, sub_sp, 0xAA00_03F4, 0xAA01_03F6, 0xAA02_03F7]);
    emit_arm64_unwind_state(c, addrs, out)?;
    let no_state = arm64_fwd(0xB400_0000, out);
    if c.win {
        out.words(&[0xAA00_03F3, 0xA940_2668, 0xA900_27E8, 0xA901_7FFF, 0xF900_03FD | ((STOP_FP_OFFSET / 8) << 10), 0x9100_03E8, 0xF900_0268, 0x9100_C3E0, 0xAA1F_03E1]);
        c.call("_setjmp", out)?;
        let caught = arm64_fwd(0x3500_0000, out);
        let ok_zero = arm64_jcc(&[0xAA16_03E0, 0xD63F_0280, 0xF940_03E8, 0xF900_0268], 0x1400_0000, out);
        arm64_here(&[no_state], out);
        out.words(&[0xAA16_03E0, 0xD63F_0280]);
        arm64_here(&[ok_zero], out);
        let done = arm64_jcc(&[0x5280_0000], 0x1400_0000, out);
        arm64_here(&[caught], out);
        out.words(&[0xAA16_03E0, 0x9100_43E1, 0xD63F_02E0, 0x5280_0020]);
        arm64_here(&[done], out);
    } else {
        out.words(&[
            0xAA00_03F3, 0xA940_2668, 0xA900_27E8, 0x9100_03E8,
            0xF900_13FD, 0xA903_5FF6, 0xA905_6BF9, 0xA906_73FB,
            0x6D07_27E8, 0x6D08_2FEA, 0x6D09_37EC, 0x6D0A_3FEE,
        ]);
        let caught_adr = arm64_fwd(0x1000_0009, out);
        out.words(&[0xF900_23E9, 0xF900_0268, 0xAA16_03E0, 0xD63F_0280, 0xF940_03E8, 0xF900_0268, 0x5280_0000]);
        let epilogue = out.pos();
        out.words(&[add_sp, 0xA943_63F7, 0xA942_5BF5, 0xA941_53F3, 0xA8C4_7BFD, 0xD65F_03C0]);
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
    out.words(&[add_sp, 0xA943_63F7, 0xA942_5BF5, 0xA941_53F3, 0xA8C4_7BFD, 0xD65F_03C0]);
    Ok(())
}

pub(super) fn emit_arm64_start_panic(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    out.words(&[0xA9BC_7BFD, 0x9100_03FD, 0xA901_53F3, 0xA902_5BF5, 0xA903_63F7, 0xAA00_03F4, 0xAA01_03F6]);
    emit_arm64_unwind_state(c, addrs, out)?;
    let no_state = arm64_fwd(0xB400_0000, out);
    let no_frame = arm64_jcc(&[0xAA00_03F3, 0xF940_0277], 0xB400_0017, out);
    out.words(&[0xF940_02E8, 0xF900_0268, 0xF940_0EC8, 0xAA14_03E0, 0xD63F_0100, 0xA901_06E0, 0xF900_02F3]);
    emit_arm64_walk_call(c, addrs, out);
    if c.win {
        c.words_call(&[0x9100_C2E0, 0x5280_0021], "longjmp", out)?;
    } else {
        out.words(&[
            0xA945_6AF9, 0xA946_72FB, 0x6D47_26E8, 0x6D48_2EEA, 0x6D49_36EC, 0x6D4A_3EEE,
            0xF940_12FD, 0x9100_02FF, 0xF940_22E9, 0xA943_5EF6, 0xD61F_0120,
        ]);
    }
    arm64_here(&[no_state, no_frame], out);
    c.call("abort", out)?;
    out.u32(0xD420_0020);
    Ok(())
}

/// Compiles and appends the native frame-table unwind walker `fn(cur_fp, stop_fp, text_base, table_ptr)`.
pub(super) fn emit_walk(arch: Arch, os: Os, out: &mut Buf) -> Result<()> {
    let target = TARGETS.iter().find(|t| t.arch == arch && t.os == os).copied().context("missing codegen target for unwind walker")?.effort(Effort::Quick);
    let code = haste_codegen::compile(build_unwind_walk(), &target).map_err(|e| anyhow!("{e}"))?;
    if !code.relocs().is_empty() {
        return Err(anyhow!("AOT unwind walker emitted nested relocations"));
    }
    out.bytes(code.bytes());
    Ok(())
}

/// Emits the fixed-size ARM64 call from `StartPanic` to `UnwindWalk(x29, x23, text, table)`.
pub(super) fn emit_arm64_walk_call(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) {
    out.words(&[0xAA1D_03E0, 0xAA17_03E1]);
    emit_arm64_adrp_add(c.at(out), addrs.text, 2, out);
    if addrs.unwind_table == 0 {
        out.u32(0xD280_0003);
    } else {
        emit_arm64_adrp_add(c.at(out), addrs.unwind_table, 3, out);
    }
    emit_rel_branch(Arch::Arm64, true, c.at(out), addrs.unwind_walk, out);
}

/// Emits the fixed-size x86-64 call from `StartPanic` to `UnwindWalk(rbp, r15, text, table)`.
pub(super) fn emit_x64_walk_call(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) {
    if c.win {
        out.bytes(&[0x48, 0x89, 0xE9, 0x4C, 0x89, 0xFA, 0x4C, 0x8D, 0x05]);
        out.u32(((addrs.text as i64 - (c.at(out) as i64 + 4)) as i32) as u32);
        if addrs.unwind_table == 0 {
            out.bytes(&[0x45, 0x31, 0xC9]);
        } else {
            out.bytes(&[0x4C, 0x8D, 0x0D]);
            out.u32(((addrs.unwind_table as i64 - (c.at(out) as i64 + 4)) as i32) as u32);
        }
    } else {
        out.bytes(&[0x48, 0x89, 0xEF, 0x4C, 0x89, 0xFE, 0x48, 0x8D, 0x15]);
        out.u32(((addrs.text as i64 - (c.at(out) as i64 + 4)) as i32) as u32);
        if addrs.unwind_table == 0 {
            out.bytes(&[0x31, 0xC9]);
        } else {
            out.bytes(&[0x48, 0x8D, 0x0D]);
            out.u32(((addrs.unwind_table as i64 - (c.at(out) as i64 + 4)) as i32) as u32);
        }
    }
    emit_rel_branch(Arch::X64, true, c.at(out), addrs.unwind_walk, out);
}

fn build_unwind_walk() -> Function {
    let mut sig = Signature::new(CallConv::Native);
    sig.params = vec![AbiParam::new(Type::I64); 4];
    let mut func = Function::new(sig);
    let mut b = FunctionBuilder::new(&mut func);
    let mut drop_sig = Signature::new(CallConv::Native);
    drop_sig.params = vec![AbiParam::new(Type::I64)];
    let drop_sig = b.import_signature(drop_sig);

    let entry = b.create_block();
    let load_table = b.create_block();
    let init_done = b.create_block();
    let frame_loop = b.create_block();
    let load_caller = b.create_block();
    let check_stop = b.create_block();
    let drain_loop = b.create_block();
    let drain_check_mark = b.create_block();
    let drain_check_limit = b.create_block();
    let drain_unlink = b.create_block();
    let drain_set_prev = b.create_block();
    let drain_check_next = b.create_block();
    let drain_set_next = b.create_block();
    let drain_set_head = b.create_block();
    let drain_after_unlink = b.create_block();
    let drain_invoke = b.create_block();
    let drain_done = b.create_block();
    let check_pc = b.create_block();
    let search_init = b.create_block();
    let bsearch = b.create_block();
    let bsearch_step = b.create_block();
    let bsearch_hi = b.create_block();
    let drops_init = b.create_block();
    let drop_loop = b.create_block();
    let drop_step = b.create_block();
    let drop_run = b.create_block();
    let load_ind = b.create_block();
    let do_call = b.create_block();
    let done = b.create_block();

    b.append_block_params_for_function_params(entry);
    let count = b.append_block_param(init_done, Type::I64);
    let fn_recs = b.append_block_param(init_done, Type::I64);
    let drops_base = b.append_block_param(init_done, Type::I64);
    let fp = b.append_block_param(frame_loop, Type::I64);
    let limit_addr = b.append_block_param(drain_loop, Type::I64);
    let next_fp = b.append_block_param(drain_loop, Type::I64);
    let next_ret = b.append_block_param(drain_loop, Type::I64);
    let is_final = b.append_block_param(drain_loop, Type::I8);
    let lo = b.append_block_param(bsearch, Type::I64);
    let hi = b.append_block_param(bsearch, Type::I64);
    let idx = b.append_block_param(drop_loop, Type::I64);
    let arg = b.append_block_param(do_call, Type::I64);

    b.switch_to_block(entry);
    let cur_fp = b.block_params(entry)[0];
    let frame_ptr = b.block_params(entry)[1];
    let text_base = b.block_params(entry)[2];
    let table_ptr = b.block_params(entry)[3];
    let state_ptr = b.ins().load(Type::I64, frame_ptr, 0);
    let mark = b.ins().load(Type::I64, frame_ptr, 8);
    let stop_fp = b.ins().load(Type::I64, frame_ptr, STOP_FP_OFFSET as i32);
    let zero = b.ins().iconst(Type::I64, 0);
    let c4 = b.ins().iconst(Type::I64, 4);
    let z8 = b.ins().iconst(Type::I8, 0);
    let one8 = b.ins().iconst(Type::I8, 1);
    let no_table = b.ins().icmp_imm(IntCC::Equal, table_ptr, 0);
    b.ins().brif(no_table, init_done, &[zero, zero, zero], load_table, &[]);

    b.switch_to_block(load_table);
    let t_count = b.ins().load(Type::I64, table_ptr, 0);
    let t_recs = b.ins().iadd_imm(table_ptr, 8);
    let fn_bytes = b.ins().ishl(t_count, c4);
    let t_drops = b.ins().iadd(t_recs, fn_bytes);
    b.ins().jump(init_done, &[t_count, t_recs, t_drops]);

    b.switch_to_block(init_done);
    let empty = b.ins().icmp_imm(IntCC::Equal, count, 0);
    b.ins().brif(empty, drain_loop, &[zero, zero, zero, one8], frame_loop, &[cur_fp]);

    b.switch_to_block(frame_loop);
    let fp_zero = b.ins().icmp_imm(IntCC::Equal, fp, 0);
    b.ins().brif(fp_zero, drain_loop, &[zero, zero, zero, one8], load_caller, &[]);

    b.switch_to_block(load_caller);
    let caller_fp = b.ins().load(Type::I64, fp, 0);
    let ret_pc = b.ins().load(Type::I64, fp, 8);
    let bad_fp = b.ins().icmp(IntCC::UnsignedLessThanOrEqual, caller_fp, fp);
    b.ins().brif(bad_fp, drain_loop, &[zero, zero, zero, one8], check_stop, &[]);

    b.switch_to_block(check_stop);
    let at_stop = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, caller_fp, stop_fp);
    b.ins().brif(at_stop, drain_loop, &[zero, zero, zero, one8], drain_loop, &[caller_fp, caller_fp, ret_pc, z8]);

    b.switch_to_block(drain_loop);
    let node = b.ins().load(Type::I64, state_ptr, 8);
    let node_null = b.ins().icmp_imm(IntCC::Equal, node, 0);
    b.ins().brif(node_null, drain_done, &[], drain_check_mark, &[]);

    b.switch_to_block(drain_check_mark);
    let at_mark = b.ins().icmp(IntCC::Equal, node, mark);
    b.ins().brif(at_mark, drain_done, &[], drain_check_limit, &[]);

    b.switch_to_block(drain_check_limit);
    let has_limit = b.ins().icmp_imm(IntCC::NotEqual, limit_addr, 0);
    let past_limit = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, node, limit_addr);
    let stop_limit = b.ins().band(has_limit, past_limit);
    b.ins().brif(stop_limit, drain_done, &[], drain_unlink, &[]);

    b.switch_to_block(drain_unlink);
    let prev = b.ins().load(Type::I64, node, 8);
    let next = b.ins().load(Type::I64, node, 16);
    let has_prev = b.ins().icmp_imm(IntCC::NotEqual, prev, 0);
    b.ins().brif(has_prev, drain_set_prev, &[], drain_check_next, &[]);

    b.switch_to_block(drain_set_prev);
    b.ins().store(next, prev, 16);
    b.ins().jump(drain_check_next, &[]);

    b.switch_to_block(drain_check_next);
    let has_next = b.ins().icmp_imm(IntCC::NotEqual, next, 0);
    b.ins().brif(has_next, drain_set_next, &[], drain_set_head, &[]);

    b.switch_to_block(drain_set_next);
    b.ins().store(prev, next, 8);
    b.ins().jump(drain_after_unlink, &[]);

    b.switch_to_block(drain_set_head);
    b.ins().store(prev, state_ptr, 8);
    b.ins().jump(drain_after_unlink, &[]);

    b.switch_to_block(drain_after_unlink);
    b.ins().store(zero, node, 8);
    b.ins().store(zero, node, 16);
    let st = b.ins().load(Type::I8, node, 0);
    let mask_clear = b.ins().iconst(Type::I8, !3_i64);
    let st_cleared = b.ins().band(st, mask_clear);
    b.ins().store(st_cleared, node, 0);
    let is_active = b.ins().band(st, one8);
    b.ins().brif(is_active, drain_invoke, &[], drain_loop, &[limit_addr, next_fp, next_ret, is_final]);

    b.switch_to_block(drain_invoke);
    let node_fn = b.ins().load(Type::I64, node, 24);
    let node_arg = b.ins().load(Type::I64, node, 32);
    b.ins().call_indirect(drop_sig, node_fn, &[node_arg]);
    b.ins().jump(drain_loop, &[limit_addr, next_fp, next_ret, is_final]);

    b.switch_to_block(drain_done);
    b.ins().brif(is_final, done, &[], check_pc, &[]);

    b.switch_to_block(check_pc);
    let in_text = b.ins().icmp(IntCC::UnsignedGreaterThan, next_ret, text_base);
    b.ins().brif(in_text, search_init, &[], frame_loop, &[next_fp]);

    b.switch_to_block(search_init);
    let pc_m1 = b.ins().iadd_imm(next_ret, -1);
    let rel_pc = b.ins().isub(pc_m1, text_base);
    b.ins().jump(bsearch, &[zero, count]);

    b.switch_to_block(bsearch);
    let more = b.ins().icmp(IntCC::UnsignedLessThan, lo, hi);
    b.ins().brif(more, bsearch_step, &[], frame_loop, &[next_fp]);

    b.switch_to_block(bsearch_step);
    let diff = b.ins().isub(hi, lo);
    let c1 = b.ins().iconst(Type::I64, 1);
    let half = b.ins().ushr(diff, c1);
    let mid = b.ins().iadd(lo, half);
    let rec_off = b.ins().ishl(mid, c4);
    let rec_ptr = b.ins().iadd(fn_recs, rec_off);
    let start32 = b.ins().load(Type::I32, rec_ptr, 0);
    let fn_start = b.ins().uextend(Type::I64, start32);
    let below = b.ins().icmp(IntCC::UnsignedLessThan, rel_pc, fn_start);
    b.ins().brif(below, bsearch, &[lo, mid], bsearch_hi, &[]);

    b.switch_to_block(bsearch_hi);
    let end32 = b.ins().load(Type::I32, rec_ptr, 4);
    let fn_end = b.ins().uextend(Type::I64, end32);
    let above = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, rel_pc, fn_end);
    let mid_p1 = b.ins().iadd_imm(mid, 1);
    b.ins().brif(above, bsearch, &[mid_p1, hi], drops_init, &[]);

    b.switch_to_block(drops_init);
    let dstart32 = b.ins().load(Type::I32, rec_ptr, 8);
    let dcount32 = b.ins().load(Type::I32, rec_ptr, 12);
    let dstart = b.ins().uextend(Type::I64, dstart32);
    let dcount = b.ins().uextend(Type::I64, dcount32);
    let dend = b.ins().iadd(dstart, dcount);
    b.ins().jump(drop_loop, &[dend]);

    b.switch_to_block(drop_loop);
    let has_more = b.ins().icmp(IntCC::UnsignedGreaterThan, idx, dstart);
    b.ins().brif(has_more, drop_step, &[], frame_loop, &[next_fp]);

    b.switch_to_block(drop_step);
    let next_idx = b.ins().iadd_imm(idx, -1);
    let doff = b.ins().ishl(next_idx, c4);
    let dptr = b.ins().iadd(drops_base, doff);
    let flag_off32 = b.ins().load(Type::I32, dptr, 4);
    let flag_off = b.ins().sextend(Type::I64, flag_off32);
    let flag_ptr = b.ins().iadd(next_fp, flag_off);
    let flag = b.ins().load(Type::I8, flag_ptr, 0);
    let active = b.ins().icmp_imm(IntCC::NotEqual, flag, 0);
    b.ins().brif(active, drop_run, &[], drop_loop, &[next_idx]);

    b.switch_to_block(drop_run);
    b.ins().store(z8, flag_ptr, 0);
    let local_off32 = b.ins().load(Type::I32, dptr, 0);
    let local_off = b.ins().sextend(Type::I64, local_off32);
    let slot_ptr = b.ins().iadd(next_fp, local_off);
    let ind32 = b.ins().load(Type::I32, dptr, 12);
    let is_ind = b.ins().icmp_imm(IntCC::NotEqual, ind32, 0);
    b.ins().brif(is_ind, load_ind, &[], do_call, &[slot_ptr]);

    b.switch_to_block(load_ind);
    let deref_ptr = b.ins().load(Type::I64, slot_ptr, 0);
    b.ins().jump(do_call, &[deref_ptr]);

    b.switch_to_block(do_call);
    let fn_off32 = b.ins().load(Type::I32, dptr, 8);
    let fn_off = b.ins().uextend(Type::I64, fn_off32);
    let drop_fn = b.ins().iadd(text_base, fn_off);
    b.ins().call_indirect(drop_sig, drop_fn, &[arg]);
    b.ins().jump(drop_loop, &[next_idx]);

    b.switch_to_block(done);
    b.ins().return_(&[]);
    func
}
