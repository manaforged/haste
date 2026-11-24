mod arm64;
mod x64;

use anyhow::{Context as _, Result, anyhow};
use haste_codegen::{
    AbiParam, Arch, CallConv, Effort, Function, FunctionBuilder, IntCC, Os, Signature, TARGETS,
    Type,
};

use super::Buf;

pub(super) use arm64::{emit_arm64_catch_unwind, emit_arm64_start_panic};
pub(super) use x64::{emit_x64_catch_unwind, emit_x64_start_panic};

/// Byte offset of `stop_fp` inside the AOT `catch_unwind` frame record.
pub(super) const STOP_FP_OFFSET: u32 = 0x20;

/// Compiles and appends the native frame-table unwind walker `fn(cur_fp, stop_fp, text_base, table_ptr)`.
pub(super) fn emit_walk(arch: Arch, os: Os, out: &mut Buf) -> Result<()> {
    let target = TARGETS
        .iter()
        .find(|t| t.arch == arch && t.os == os)
        .copied()
        .context("missing codegen target for unwind walker")?
        .effort(Effort::Quick);
    let code = haste_codegen::compile(build_unwind_walk(), &target).map_err(|e| anyhow!("{e}"))?;
    if !code.relocs().is_empty() {
        return Err(anyhow!("AOT unwind walker emitted nested relocations"));
    }
    out.bytes(code.bytes());
    Ok(())
}

fn build_unwind_walk() -> Function {
    let mut sig = Signature::new(CallConv::Native);
    sig.params = vec![AbiParam::new(Type::I64); 4];
    sig.returns = vec![AbiParam::new(Type::I64)];
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
    let site_check = b.create_block();
    let site_search = b.create_block();
    let site_step = b.create_block();
    let site_hi = b.create_block();
    let site_hit = b.create_block();
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
    let restore_init = b.create_block();
    let restore_loop = b.create_block();
    let restore_step = b.create_block();
    let restore_copy = b.create_block();
    let done = b.create_block();

    b.append_block_params_for_function_params(entry);
    let count = b.append_block_param(init_done, Type::I64);
    let fn_recs = b.append_block_param(init_done, Type::I64);
    let drops_base = b.append_block_param(init_done, Type::I64);
    let site_count = b.append_block_param(init_done, Type::I64);
    let sites_base = b.append_block_param(init_done, Type::I64);
    let fp = b.append_block_param(frame_loop, Type::I64);
    let slo = b.append_block_param(site_search, Type::I64);
    let shi = b.append_block_param(site_search, Type::I64);
    let limit_addr = b.append_block_param(drain_loop, Type::I64);
    let next_fp = b.append_block_param(drain_loop, Type::I64);
    let next_ret = b.append_block_param(drain_loop, Type::I64);
    let is_final = b.append_block_param(drain_loop, Type::I8);
    let lo = b.append_block_param(bsearch, Type::I64);
    let hi = b.append_block_param(bsearch, Type::I64);
    let idx = b.append_block_param(drop_loop, Type::I64);
    let arg = b.append_block_param(do_call, Type::I64);
    let rmask = b.append_block_param(restore_loop, Type::I64);
    let rbit = b.append_block_param(restore_loop, Type::I64);
    let rk = b.append_block_param(restore_loop, Type::I64);

    b.switch_to_block(entry);
    let cur_fp = b.block_params(entry)[0];
    let frame_ptr = b.block_params(entry)[1];
    let text_base = b.block_params(entry)[2];
    let table_ptr = b.block_params(entry)[3];
    let state_ptr = b.ins().load(Type::I64, frame_ptr, 0);
    let mark = b.ins().load(Type::I64, frame_ptr, 8);
    let stop_fp = b.ins().load(Type::I64, frame_ptr, STOP_FP_OFFSET as i32);
    let raw_regs_ptr = b.ins().load(Type::I64, frame_ptr, 40);
    let zero = b.ins().iconst(Type::I64, 0);
    let c1 = b.ins().iconst(Type::I64, 1);
    let c2 = b.ins().iconst(Type::I64, 2);
    let c3 = b.ins().iconst(Type::I64, 3);
    let c4 = b.ins().iconst(Type::I64, 4);
    let z8 = b.ins().iconst(Type::I8, 0);
    let one8 = b.ins().iconst(Type::I8, 1);
    let two8 = b.ins().iconst(Type::I8, 2);
    let entry_has_stop = b.ins().icmp_imm(IntCC::NotEqual, stop_fp, 0);
    let regs_ptr = b.ins().select(entry_has_stop, zero, raw_regs_ptr);
    let no_table = b.ins().icmp_imm(IntCC::Equal, table_ptr, 0);
    b.ins().brif(
        no_table,
        init_done,
        &[zero, zero, zero, zero, zero],
        load_table,
        &[],
    );

    b.switch_to_block(load_table);
    let t_count = b.ins().load(Type::I64, table_ptr, 0);
    let t_sites = b.ins().load(Type::I64, table_ptr, 8);
    let t_sites_off = b.ins().load(Type::I64, table_ptr, 16);
    let t_recs = b.ins().iadd_imm(table_ptr, 24);
    let fn_b16 = b.ins().ishl(t_count, c4);
    let fn_b4 = b.ins().ishl(t_count, c2);
    let fn_bytes = b.ins().iadd(fn_b16, fn_b4);
    let t_drops = b.ins().iadd(t_recs, fn_bytes);
    let t_sites_base = b.ins().iadd(table_ptr, t_sites_off);
    b.ins().jump(
        init_done,
        &[t_count, t_recs, t_drops, t_sites, t_sites_base],
    );

    b.switch_to_block(init_done);
    let any_entries = b.ins().bor(count, site_count);
    let empty = b.ins().icmp_imm(IntCC::Equal, any_entries, 0);
    b.ins().brif(
        empty,
        drain_loop,
        &[zero, zero, zero, one8],
        frame_loop,
        &[cur_fp],
    );

    b.switch_to_block(frame_loop);
    let fp_zero = b.ins().icmp_imm(IntCC::Equal, fp, 0);
    b.ins().brif(
        fp_zero,
        drain_loop,
        &[zero, zero, zero, one8],
        load_caller,
        &[],
    );

    b.switch_to_block(load_caller);
    let caller_fp = b.ins().load(Type::I64, fp, 0);
    let ret_pc = b.ins().load(Type::I64, fp, 8);
    let bad_fp = b.ins().icmp(IntCC::UnsignedLessThanOrEqual, caller_fp, fp);
    b.ins().brif(
        bad_fp,
        drain_loop,
        &[zero, zero, zero, one8],
        site_check,
        &[],
    );

    b.switch_to_block(site_check);
    let has_sites = b.ins().icmp_imm(IntCC::NotEqual, site_count, 0);
    let ret_in_text = b.ins().icmp(IntCC::UnsignedGreaterThan, ret_pc, text_base);
    let try_sites = b.ins().band(has_sites, ret_in_text);
    let ret_rel = b.ins().isub(ret_pc, text_base);
    b.ins()
        .brif(try_sites, site_search, &[zero, site_count], check_stop, &[]);

    b.switch_to_block(site_search);
    let s_more = b.ins().icmp(IntCC::UnsignedLessThan, slo, shi);
    b.ins().brif(s_more, site_step, &[], check_stop, &[]);

    b.switch_to_block(site_step);
    let s_diff = b.ins().isub(shi, slo);
    let s_half = b.ins().ushr(s_diff, c1);
    let s_mid = b.ins().iadd(slo, s_half);
    let s_off = b.ins().ishl(s_mid, c4);
    let s_ptr = b.ins().iadd(sites_base, s_off);
    let s_ret32 = b.ins().load(Type::I32, s_ptr, 0);
    let s_ret = b.ins().uextend(Type::I64, s_ret32);
    let s_below = b.ins().icmp(IntCC::UnsignedLessThan, ret_rel, s_ret);
    b.ins()
        .brif(s_below, site_search, &[slo, s_mid], site_hi, &[]);

    b.switch_to_block(site_hi);
    let s_above = b.ins().icmp(IntCC::UnsignedGreaterThan, ret_rel, s_ret);
    let s_mid_p1 = b.ins().iadd_imm(s_mid, 1);
    b.ins()
        .brif(s_above, site_search, &[s_mid_p1, shi], site_hit, &[]);

    b.switch_to_block(site_hit);
    let h_rel32 = b.ins().load(Type::I32, s_ptr, 4);
    let h_rel = b.ins().uextend(Type::I64, h_rel32);
    let f_off32 = b.ins().load(Type::I32, s_ptr, 8);
    let f_off = b.ins().uextend(Type::I64, f_off32);
    let handler_pc = b.ins().iadd(text_base, h_rel);
    let target_sp = b.ins().isub(caller_fp, f_off);
    b.ins().store(handler_pc, frame_ptr, 16);
    b.ins().store(caller_fp, frame_ptr, 24);
    b.ins().store(target_sp, frame_ptr, STOP_FP_OFFSET as i32);
    b.ins().jump(drain_loop, &[target_sp, zero, zero, two8]);

    b.switch_to_block(check_stop);
    let has_stop = b.ins().icmp_imm(IntCC::NotEqual, stop_fp, 0);
    let past_stop = b
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, caller_fp, stop_fp);
    let at_stop = b.ins().band(has_stop, past_stop);
    b.ins().brif(
        at_stop,
        drain_loop,
        &[zero, zero, zero, one8],
        drain_loop,
        &[caller_fp, caller_fp, ret_pc, z8],
    );

    b.switch_to_block(drain_loop);
    let node = b.ins().load(Type::I64, state_ptr, 8);
    let node_null = b.ins().icmp_imm(IntCC::Equal, node, 0);
    b.ins()
        .brif(node_null, drain_done, &[], drain_check_mark, &[]);

    b.switch_to_block(drain_check_mark);
    let at_mark = b.ins().icmp(IntCC::Equal, node, mark);
    b.ins()
        .brif(at_mark, drain_done, &[], drain_check_limit, &[]);

    b.switch_to_block(drain_check_limit);
    let has_limit = b.ins().icmp_imm(IntCC::NotEqual, limit_addr, 0);
    let past_limit = b
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, node, limit_addr);
    let stop_limit = b.ins().band(has_limit, past_limit);
    b.ins().brif(stop_limit, drain_done, &[], drain_unlink, &[]);

    b.switch_to_block(drain_unlink);
    let prev = b.ins().load(Type::I64, node, 8);
    let next = b.ins().load(Type::I64, node, 16);
    let has_prev = b.ins().icmp_imm(IntCC::NotEqual, prev, 0);
    b.ins()
        .brif(has_prev, drain_set_prev, &[], drain_check_next, &[]);

    b.switch_to_block(drain_set_prev);
    b.ins().store(next, prev, 16);
    b.ins().jump(drain_check_next, &[]);

    b.switch_to_block(drain_check_next);
    let has_next = b.ins().icmp_imm(IntCC::NotEqual, next, 0);
    b.ins()
        .brif(has_next, drain_set_next, &[], drain_set_head, &[]);

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
    b.ins().brif(
        is_active,
        drain_invoke,
        &[],
        drain_loop,
        &[limit_addr, next_fp, next_ret, is_final],
    );

    b.switch_to_block(drain_invoke);
    let node_fn = b.ins().load(Type::I64, node, 24);
    let node_arg = b.ins().load(Type::I64, node, 32);
    b.ins().call_indirect(drop_sig, node_fn, &[node_arg]);
    b.ins()
        .jump(drain_loop, &[limit_addr, next_fp, next_ret, is_final]);

    b.switch_to_block(drain_done);
    b.ins().brif(is_final, done, &[], check_pc, &[]);

    b.switch_to_block(check_pc);
    let in_text = b
        .ins()
        .icmp(IntCC::UnsignedGreaterThan, next_ret, text_base);
    b.ins()
        .brif(in_text, search_init, &[], frame_loop, &[next_fp]);

    b.switch_to_block(search_init);
    let pc_m1 = b.ins().iadd_imm(next_ret, -1);
    let rel_pc = b.ins().isub(pc_m1, text_base);
    b.ins().jump(bsearch, &[zero, count]);

    b.switch_to_block(bsearch);
    let more = b.ins().icmp(IntCC::UnsignedLessThan, lo, hi);
    b.ins()
        .brif(more, bsearch_step, &[], frame_loop, &[next_fp]);

    b.switch_to_block(bsearch_step);
    let diff = b.ins().isub(hi, lo);
    let half = b.ins().ushr(diff, c1);
    let mid = b.ins().iadd(lo, half);
    let rec_off16 = b.ins().ishl(mid, c4);
    let rec_off4 = b.ins().ishl(mid, c2);
    let rec_off = b.ins().iadd(rec_off16, rec_off4);
    let rec_ptr = b.ins().iadd(fn_recs, rec_off);
    let start32 = b.ins().load(Type::I32, rec_ptr, 0);
    let fn_start = b.ins().uextend(Type::I64, start32);
    let below = b.ins().icmp(IntCC::UnsignedLessThan, rel_pc, fn_start);
    b.ins().brif(below, bsearch, &[lo, mid], bsearch_hi, &[]);

    b.switch_to_block(bsearch_hi);
    let end32 = b.ins().load(Type::I32, rec_ptr, 4);
    let fn_end = b.ins().uextend(Type::I64, end32);
    let above = b
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, rel_pc, fn_end);
    let mid_p1 = b.ins().iadd_imm(mid, 1);
    b.ins().brif(above, bsearch, &[mid_p1, hi], drops_init, &[]);

    b.switch_to_block(drops_init);
    let dstart32 = b.ins().load(Type::I32, rec_ptr, 8);
    let dcount32 = b.ins().load(Type::I32, rec_ptr, 12);
    let mask32 = b.ins().load(Type::I32, rec_ptr, 16);
    let dstart = b.ins().uextend(Type::I64, dstart32);
    let dcount = b.ins().uextend(Type::I64, dcount32);
    let fn_mask = b.ins().uextend(Type::I64, mask32);
    let dend = b.ins().iadd(dstart, dcount);
    b.ins().jump(drop_loop, &[dend]);

    b.switch_to_block(drop_loop);
    let has_more = b.ins().icmp(IntCC::UnsignedGreaterThan, idx, dstart);
    b.ins().brif(has_more, drop_step, &[], restore_init, &[]);

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

    b.switch_to_block(restore_init);
    let has_regs = b.ins().icmp_imm(IntCC::NotEqual, regs_ptr, 0);
    let has_mask = b.ins().icmp_imm(IntCC::NotEqual, fn_mask, 0);
    let do_restore = b.ins().band(has_regs, has_mask);
    b.ins().brif(
        do_restore,
        restore_loop,
        &[fn_mask, zero, zero],
        frame_loop,
        &[next_fp],
    );

    b.switch_to_block(restore_loop);
    let rmore = b.ins().icmp_imm(IntCC::NotEqual, rmask, 0);
    b.ins()
        .brif(rmore, restore_step, &[], frame_loop, &[next_fp]);

    b.switch_to_block(restore_step);
    let bit0 = b.ins().band(rmask, c1);
    let next_rmask = b.ins().ushr(rmask, c1);
    let next_rbit = b.ins().iadd_imm(rbit, 1);
    let is_set = b.ins().icmp_imm(IntCC::NotEqual, bit0, 0);
    b.ins().brif(
        is_set,
        restore_copy,
        &[],
        restore_loop,
        &[next_rmask, next_rbit, rk],
    );

    b.switch_to_block(restore_copy);
    let next_rk = b.ins().iadd_imm(rk, 1);
    let neg_off = b.ins().ishl(next_rk, c3);
    let src_ptr = b.ins().isub(next_fp, neg_off);
    let reg_val = b.ins().load(Type::I64, src_ptr, 0);
    let dst_off = b.ins().ishl(rbit, c3);
    let dst_ptr = b.ins().iadd(regs_ptr, dst_off);
    b.ins().store(reg_val, dst_ptr, 0);
    b.ins()
        .jump(restore_loop, &[next_rmask, next_rbit, next_rk]);

    b.switch_to_block(done);
    let caught = b.ins().icmp(IntCC::Equal, is_final, two8);
    let ret = b.ins().uextend(Type::I64, caught);
    b.ins().return_(&[ret]);
    func
}
