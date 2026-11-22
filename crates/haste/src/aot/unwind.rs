use anyhow::{Context as _, Result, anyhow};
use haste_codegen::{AbiParam, Arch, CallConv, Effort, Function, FunctionBuilder, IntCC, Os, Signature, TARGETS, Type};

use super::shims::{Emit, emit_arm64_adrp_add, emit_rel_branch};
use super::{Buf, SectionAddrs};

/// Byte offset of `stop_fp` inside the AOT `catch_unwind` frame record.
pub(super) const STOP_FP_OFFSET: u32 = 0x220;

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

/// Emits the fixed-size ARM64 call from `StartPanic` to `UnwindWalk(x29, [x23+STOP_FP_OFFSET], text, table)`.
pub(super) fn emit_arm64_walk_call(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) {
    out.words(&[0xAA1D_03E0, 0xF940_02E1 | ((STOP_FP_OFFSET / 8) << 10)]);
    emit_arm64_adrp_add(c.at(out), addrs.text, 2, out);
    emit_arm64_adrp_add(c.at(out), addrs.unwind_table, 3, out);
    emit_rel_branch(Arch::Arm64, true, c.at(out), addrs.unwind_walk, out);
}

/// Emits the fixed-size x86-64 call from `StartPanic` to `UnwindWalk(rbp, [r15+STOP_FP_OFFSET], text, table)`.
pub(super) fn emit_x64_walk_call(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) {
    let off = STOP_FP_OFFSET.to_le_bytes();
    if c.win {
        out.bytes(&[0x48, 0x89, 0xE9, 0x49, 0x8B, 0x97, off[0], off[1], off[2], off[3], 0x4C, 0x8D, 0x05]);
        out.u32(((addrs.text as i64 - (c.at(out) as i64 + 4)) as i32) as u32);
        out.bytes(&[0x4C, 0x8D, 0x0D]);
        out.u32(((addrs.unwind_table as i64 - (c.at(out) as i64 + 4)) as i32) as u32);
    } else {
        out.bytes(&[0x48, 0x89, 0xEF, 0x49, 0x8B, 0xB7, off[0], off[1], off[2], off[3], 0x48, 0x8D, 0x15]);
        out.u32(((addrs.text as i64 - (c.at(out) as i64 + 4)) as i32) as u32);
        out.bytes(&[0x48, 0x8D, 0x0D]);
        out.u32(((addrs.unwind_table as i64 - (c.at(out) as i64 + 4)) as i32) as u32);
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
    let frame_loop = b.create_block();
    let load_caller = b.create_block();
    let check_stop = b.create_block();
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
    let fp = b.append_block_param(frame_loop, Type::I64);
    let lo = b.append_block_param(bsearch, Type::I64);
    let hi = b.append_block_param(bsearch, Type::I64);
    let idx = b.append_block_param(drop_loop, Type::I64);
    let arg = b.append_block_param(do_call, Type::I64);

    b.switch_to_block(entry);
    let cur_fp = b.block_params(entry)[0];
    let stop_fp = b.block_params(entry)[1];
    let text_base = b.block_params(entry)[2];
    let table_ptr = b.block_params(entry)[3];
    let count = b.ins().load(Type::I64, table_ptr, 0);
    let empty = b.ins().icmp_imm(IntCC::Equal, count, 0);
    let fn_recs = b.ins().iadd_imm(table_ptr, 8);
    let c4 = b.ins().iconst(Type::I64, 4);
    let fn_bytes = b.ins().ishl(count, c4);
    let drops_base = b.ins().iadd(fn_recs, fn_bytes);
    b.ins().brif(empty, done, &[], frame_loop, &[cur_fp]);

    b.switch_to_block(frame_loop);
    let fp_zero = b.ins().icmp_imm(IntCC::Equal, fp, 0);
    b.ins().brif(fp_zero, done, &[], load_caller, &[]);

    b.switch_to_block(load_caller);
    let caller_fp = b.ins().load(Type::I64, fp, 0);
    let ret_pc = b.ins().load(Type::I64, fp, 8);
    let bad_fp = b.ins().icmp(IntCC::UnsignedLessThanOrEqual, caller_fp, fp);
    b.ins().brif(bad_fp, done, &[], check_stop, &[]);

    b.switch_to_block(check_stop);
    let at_stop = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, caller_fp, stop_fp);
    b.ins().brif(at_stop, done, &[], check_pc, &[]);

    b.switch_to_block(check_pc);
    let in_text = b.ins().icmp(IntCC::UnsignedGreaterThan, ret_pc, text_base);
    b.ins().brif(in_text, search_init, &[], frame_loop, &[caller_fp]);

    b.switch_to_block(search_init);
    let pc_m1 = b.ins().iadd_imm(ret_pc, -1);
    let rel_pc = b.ins().isub(pc_m1, text_base);
    let zero = b.ins().iconst(Type::I64, 0);
    b.ins().jump(bsearch, &[zero, count]);

    b.switch_to_block(bsearch);
    let more = b.ins().icmp(IntCC::UnsignedLessThan, lo, hi);
    b.ins().brif(more, bsearch_step, &[], frame_loop, &[caller_fp]);

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
    b.ins().brif(has_more, drop_step, &[], frame_loop, &[caller_fp]);

    b.switch_to_block(drop_step);
    let next_idx = b.ins().iadd_imm(idx, -1);
    let doff = b.ins().ishl(next_idx, c4);
    let dptr = b.ins().iadd(drops_base, doff);
    let flag_off32 = b.ins().load(Type::I32, dptr, 4);
    let flag_off = b.ins().sextend(Type::I64, flag_off32);
    let flag_ptr = b.ins().iadd(caller_fp, flag_off);
    let flag = b.ins().load(Type::I8, flag_ptr, 0);
    let active = b.ins().icmp_imm(IntCC::NotEqual, flag, 0);
    b.ins().brif(active, drop_run, &[], drop_loop, &[next_idx]);

    b.switch_to_block(drop_run);
    let z8 = b.ins().iconst(Type::I8, 0);
    b.ins().store(z8, flag_ptr, 0);
    let local_off32 = b.ins().load(Type::I32, dptr, 0);
    let local_off = b.ins().sextend(Type::I64, local_off32);
    let slot_ptr = b.ins().iadd(caller_fp, local_off);
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
