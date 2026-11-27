mod regressions;

use crate::ir::{Binary, Op, StackSlot, Type};
use crate::promote::Moves;
use crate::{
    AbiParam, Arch, AtomicOrd, CallConv, Effort, ErrorKind, Function, FunctionBuilder, IntCC, Lane,
    Memory, ReduceOp, RelocKind, Signature, TARGETS, Target, UnwindDrop, compile, schedule,
    simplify, verify,
};

fn full_x64() -> Target {
    let mut target = *Target::find("x86_64-unknown-linux-gnu").expect("x64 linux target");
    target.effort = Effort::Full;
    target
}

fn full_arm64() -> Target {
    let mut target = *Target::find("aarch64-apple-darwin").expect("arm64 darwin target");
    target.effort = Effort::Full;
    target
}

fn host_full() -> Target {
    let mut target = *Target::host();
    target.effort = Effort::Full;
    target
}

fn sig_i64(params: usize, returns: usize) -> Signature {
    let mut sig = Signature::new(CallConv::Internal);
    for _ in 0..params {
        sig.params.push(AbiParam::new(Type::I64));
    }
    for _ in 0..returns {
        sig.returns.push(AbiParam::new(Type::I64));
    }
    sig
}

fn build_fn(
    sig: Signature,
    build: impl FnOnce(&mut FunctionBuilder<'_>, &[crate::Value]),
) -> Function {
    let mut func = Function::new(sig);
    {
        let mut b = FunctionBuilder::new(&mut func);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        let params = b.block_params(entry).to_vec();
        b.switch_to_block(entry);
        build(&mut b, &params);
    }
    func
}

fn run_full(func: &mut Function) {
    let target = full_x64();
    assert!(verify::verify(func, &target).is_ok(), "pre-opt IR invalid");
    schedule::run(func, Moves::Split, &target);
}

fn count_ops(func: &Function, pred: impl Fn(&Op) -> bool) -> usize {
    func.blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|inst| pred(&func.insts[inst.index()].op))
        .count()
}

fn verify_and_compile_both(func: &Function) {
    for target in [full_x64(), full_arm64()] {
        assert!(
            verify::verify(func, &target).is_ok(),
            "IR failed verify for {}",
            target.triple
        );
        let code = compile(func.clone(), &target).expect("compile succeeds");
        assert!(!code.bytes().is_empty());
    }
}

fn jit_run_1(func: &Function, arg0: i64) -> i64 {
    let code = compile(func.clone(), &host_full()).expect("host compile");
    let mut mem = Memory::default();
    let addr = mem.write_patched(&code, &[]).expect("map executable");
    // SAFETY: `code` was compiled for the host target with `CallConv::Native` signature `fn(i64) -> i64`.
    let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(addr) };
    f(arg0)
}

fn jit_run_2(func: &Function, arg0: i64, arg1: i64) -> i64 {
    let code = compile(func.clone(), &host_full()).expect("host compile");
    let mut mem = Memory::default();
    let addr = mem.write_patched(&code, &[]).expect("map executable");
    // SAFETY: `code` was compiled for the host target with `CallConv::Native` signature `fn(i64, i64) -> i64`.
    let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(addr) };
    f(arg0, arg1)
}

fn jit_run_3(func: &Function, arg0: i64, arg1: i64, arg2: i64) -> i64 {
    let code = compile(func.clone(), &host_full()).expect("host compile");
    let mut mem = Memory::default();
    let addr = mem.write_patched(&code, &[]).expect("map executable");
    // SAFETY: `code` was compiled for the host target with `CallConv::Native` signature `fn(i64, i64, i64) -> i64`.
    let f: extern "C" fn(i64, i64, i64) -> i64 = unsafe { std::mem::transmute(addr) };
    f(arg0, arg1, arg2)
}

fn native_sig(params: usize, returns: usize) -> Signature {
    let mut sig = sig_i64(params, returns);
    sig.call_conv = CallConv::Native;
    sig
}

extern "C" fn host_add_100(x: i64) -> i64 {
    x.wrapping_add(100)
}

#[test]
fn simplify_respects_explicit_target_arch() {
    let mut cond_val = None;
    let mut func_x64 = build_fn(sig_i64(1, 1), |b, p| {
        let zero = b.ins().iconst(Type::I64, 0);
        let cond = b.ins().icmp(IntCC::Equal, p[0], zero);
        b.func.mark_unpredictable(cond);
        cond_val = Some(cond);
        let one = b.ins().iconst(Type::I64, 1);
        let res = b.ins().select(cond, one, zero);
        b.ins().return_(&[res]);
    });
    let cond = cond_val.expect("cond recorded");
    let mut func_arm64 = func_x64.clone();
    simplify(&mut func_x64, &full_x64());
    simplify(&mut func_arm64, &full_arm64());
    assert!(!func_x64.is_unpredictable(cond));
    assert!(func_arm64.is_unpredictable(cond));
}

#[test]
fn pass_promote_eliminates_private_stack_slot_and_preserves_result() {
    let mut func = build_fn(native_sig(1, 1), |b, p| {
        let slot = b.create_stack_slot(8, 3);
        let inc = b.ins().iadd_imm(p[0], 7);
        b.ins().stack_store(inc, slot, 0);
        let loaded = b.ins().stack_load(Type::I64, slot, 0);
        b.ins().return_(&[loaded]);
    });
    assert_eq!(
        count_ops(&func, |op| matches!(
            op,
            Op::StackLoad(..) | Op::StackStore(..)
        )),
        2
    );
    run_full(&mut func);
    assert_eq!(
        count_ops(&func, |op| matches!(
            op,
            Op::StackLoad(..) | Op::StackStore(..)
        )),
        0
    );
    verify_and_compile_both(&func);
    assert_eq!(jit_run_1(&func, 35), 42);
    assert_eq!(jit_run_1(&func, -7), 0);
}

#[test]
fn pass_fold_range_merges_interval_compare_and_preserves_semantics() {
    let mut func = build_fn(native_sig(1, 1), |b, p| {
        let lo = b.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, p[0], 10);
        let hi = b.ins().icmp_imm(IntCC::SignedLessThanOrEqual, p[0], 20);
        let both = b.ins().band(lo, hi);
        let res = b.ins().uextend(Type::I64, both);
        b.ins().return_(&[res]);
    });
    assert_eq!(count_ops(&func, |op| matches!(op, Op::Icmp(..))), 2);
    run_full(&mut func);
    assert_eq!(count_ops(&func, |op| matches!(op, Op::Icmp(..))), 1);
    verify_and_compile_both(&func);
    assert_eq!(jit_run_1(&func, 9), 0);
    assert_eq!(jit_run_1(&func, 10), 1);
    assert_eq!(jit_run_1(&func, 15), 1);
    assert_eq!(jit_run_1(&func, 20), 1);
    assert_eq!(jit_run_1(&func, 21), 0);
}

#[test]
fn pass_fold_and_facts_simplify_bounded_compare_to_constant() {
    let mut func = build_fn(native_sig(1, 1), |b, p| {
        let mask = b.ins().iconst(Type::I64, 7);
        let masked = b.ins().band(p[0], mask);
        let cmp = b.ins().icmp_imm(IntCC::UnsignedLessThan, masked, 8);
        let ten = b.ins().iconst(Type::I64, 10);
        let twenty = b.ins().iconst(Type::I64, 20);
        let res = b.ins().select(cmp, ten, twenty);
        b.ins().return_(&[res]);
    });
    assert_eq!(count_ops(&func, |op| matches!(op, Op::Icmp(..))), 1);
    assert_eq!(count_ops(&func, |op| matches!(op, Op::Select(..))), 1);
    run_full(&mut func);
    assert_eq!(count_ops(&func, |op| matches!(op, Op::Icmp(..))), 0);
    assert_eq!(count_ops(&func, |op| matches!(op, Op::Select(..))), 0);
    verify_and_compile_both(&func);
    assert_eq!(jit_run_1(&func, -999), 10);
    assert_eq!(jit_run_1(&func, 123_456), 10);
}

#[test]
fn pass_licm_hoists_loop_invariant_multiply_outside_body() {
    let mut body_block = None;
    let mut func = build_fn(native_sig(3, 1), |b, p| {
        let (n, a, c) = (p[0], p[1], p[2]);
        let header = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();
        body_block = Some(body);
        let i = b.append_block_param(header, Type::I64);
        let acc = b.append_block_param(header, Type::I64);

        let zero = b.ins().iconst(Type::I64, 0);
        b.ins().jump(header, &[zero, zero]);

        b.switch_to_block(header);
        let cond = b.ins().icmp(IntCC::SignedLessThan, i, n);
        b.ins().brif(cond, body, &[], exit, &[]);

        b.switch_to_block(body);
        let inv = b.ins().imul(a, c);
        let next_acc = b.ins().iadd(acc, inv);
        let next_i = b.ins().iadd_imm(i, 1);
        b.ins().jump(header, &[next_i, next_acc]);

        b.switch_to_block(exit);
        b.ins().return_(&[acc]);
    });
    let body = body_block.expect("body recorded");
    let body_muls = |f: &Function| {
        f.blocks[body.index()]
            .insts
            .iter()
            .filter(|inst| matches!(f.insts[inst.index()].op, Op::Binary(Binary::Imul, _, _)))
            .count()
    };
    assert_eq!(body_muls(&func), 1);
    run_full(&mut func);
    assert_eq!(body_muls(&func), 0);
    assert_eq!(
        count_ops(&func, |op| matches!(op, Op::Binary(Binary::Imul, _, _))),
        1
    );
    verify_and_compile_both(&func);
    assert_eq!(jit_run_3(&func, 5, 6, 7), 210);
    assert_eq!(jit_run_3(&func, 0, 6, 7), 0);
}

#[test]
fn pass_lsr_strength_reduces_non_sib_stride_address() {
    let mut header_block = None;
    let mut func = build_fn(native_sig(2, 1), |b, p| {
        let (base, n) = (p[0], p[1]);
        let header = b.create_block();
        let exit = b.create_block();
        header_block = Some(header);
        let i = b.append_block_param(header, Type::I64);
        let acc = b.append_block_param(header, Type::I64);

        let zero = b.ins().iconst(Type::I64, 0);
        let ok = b.ins().icmp(IntCC::SignedLessThan, zero, n);
        b.ins().brif(ok, header, &[zero, zero], exit, &[zero]);

        b.switch_to_block(header);
        let off = b.ins().imul_imm(i, 24);
        let ptr = b.ins().iadd(base, off);
        let val = b.ins().load(Type::I64, ptr, 0);
        let next_acc = b.ins().iadd(acc, val);
        let next_i = b.ins().iadd_imm(i, 1);
        let more = b.ins().icmp(IntCC::SignedLessThan, next_i, n);
        b.ins()
            .brif(more, header, &[next_i, next_acc], exit, &[next_acc]);

        b.switch_to_block(exit);
        let out = b.append_block_param(exit, Type::I64);
        b.ins().return_(&[out]);
    });
    let header = header_block.expect("header recorded");
    assert_eq!(
        count_ops(&func, |op| matches!(op, Op::Binary(Binary::Imul, _, _))),
        1
    );
    let before_params = func.blocks[header.index()].params.len();
    run_full(&mut func);
    assert_eq!(
        count_ops(&func, |op| matches!(op, Op::Binary(Binary::Imul, _, _))),
        0
    );
    assert!(
        func.blocks[header.index()].params.len() >= before_params,
        "loop header should retain or add derived pointer parameter"
    );
    verify_and_compile_both(&func);
    let rows: [[i64; 3]; 4] = [[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]];
    assert_eq!(jit_run_2(&func, rows.as_ptr() as i64, 4), 100);
}

#[test]
fn pass_unroll_eliminates_small_constant_trip_loop_and_preserves_sum() {
    let mut func = build_fn(native_sig(1, 1), |b, p| {
        let header = b.create_block();
        let exit = b.create_block();
        let i = b.append_block_param(header, Type::I64);
        let acc = b.append_block_param(header, Type::I64);

        let zero = b.ins().iconst(Type::I64, 0);
        b.ins().jump(header, &[zero, p[0]]);

        b.switch_to_block(header);
        let next_acc = b.ins().iadd(acc, i);
        let next_i = b.ins().iadd_imm(i, 1);
        let cont = b.ins().icmp_imm(IntCC::SignedLessThan, next_i, 4);
        b.ins()
            .brif(cont, header, &[next_i, next_acc], exit, &[next_acc]);

        b.switch_to_block(exit);
        let out = b.append_block_param(exit, Type::I64);
        b.ins().return_(&[out]);
    });
    assert_eq!(count_ops(&func, |op| matches!(op, Op::Brif(..))), 1);
    run_full(&mut func);
    assert_eq!(count_ops(&func, |op| matches!(op, Op::Brif(..))), 0);
    verify_and_compile_both(&func);
    assert_eq!(jit_run_1(&func, 10), 16);
    assert_eq!(jit_run_1(&func, -6), 0);
}

#[test]
fn pass_slp_vectorizes_noalias_four_lane_i32_loop() {
    let mut sig = native_sig(3, 0);
    sig.params[0].noalias = true;
    sig.params[1].noalias = true;
    let mut func = build_fn(sig, |b, p| {
        let (dst, src, n) = (p[0], p[1], p[2]);
        let loop_blk = b.create_block();
        let exit = b.create_block();
        let i = b.append_block_param(loop_blk, Type::I64);

        let zero = b.ins().iconst(Type::I64, 0);
        let enter = b.ins().icmp(IntCC::SignedLessThan, zero, n);
        b.ins().brif(enter, loop_blk, &[zero], exit, &[]);

        b.switch_to_block(loop_blk);
        let step_off = b.ins().imul_imm(i, 16);
        let dst_p = b.ins().iadd(dst, step_off);
        let src_p = b.ins().iadd(src, step_off);
        let one = b.ins().iconst(Type::I32, 1);
        for lane in 0..4 {
            let off = lane * 4;
            let v = b.ins().load(Type::I32, src_p, off);
            let sum = b.ins().iadd(v, one);
            b.ins().store(sum, dst_p, off);
        }
        let next_i = b.ins().iadd_imm(i, 1);
        let cont = b.ins().icmp(IntCC::SignedLessThan, next_i, n);
        b.ins().brif(cont, loop_blk, &[next_i], exit, &[]);

        b.switch_to_block(exit);
        b.ins().return_(&[]);
    });
    assert!(!func.types.iter().any(|t| t.is_vector()));
    run_full(&mut func);
    assert!(
        func.types.iter().any(|t| t.is_vector()),
        "SLP should produce vector values:\n{func}"
    );
    verify_and_compile_both(&func);

    let code = compile(func, &host_full()).expect("host compile slp");
    let mut mem = Memory::default();
    let addr = mem.write_patched(&code, &[]).expect("map slp");
    // SAFETY: `code` was compiled for the host target with `CallConv::Native` signature `fn(*mut i32, *const i32, i64)`.
    let f: extern "C" fn(*mut i32, *const i32, i64) = unsafe { std::mem::transmute(addr) };
    let input = [1i32, 2, 3, 4, 10, 20, 30, 40];
    let mut output = [0i32; 8];
    f(output.as_mut_ptr(), input.as_ptr(), 2);
    assert_eq!(output, [2, 3, 4, 5, 11, 21, 31, 41]);
}

#[test]
fn pass_tailrec_converts_self_recursive_tail_call_to_loop() {
    let mut func = build_fn(native_sig(2, 1), |b, p| {
        b.func.self_slots.push(0);
        let sig = b.import_signature(native_sig(2, 1));
        let done = b.create_block();
        let rec = b.create_block();
        let (n, acc) = (p[0], p[1]);

        let is_zero = b.ins().icmp_imm(IntCC::SignedLessThanOrEqual, n, 0);
        b.ins().brif(is_zero, done, &[], rec, &[]);

        b.switch_to_block(done);
        b.ins().return_(&[acc]);

        b.switch_to_block(rec);
        let next_n = b.ins().iadd_imm(n, -1);
        let next_acc = b.ins().iadd(acc, n);
        let slot = b.ins().slot_addr(0);
        let callee = b.ins().load(Type::I64, slot, 0);
        let call = b.ins().call_indirect(sig, callee, &[next_n, next_acc]);
        let ret = b.inst_results(call)[0];
        b.ins().return_(&[ret]);
    });
    assert_eq!(count_ops(&func, |op| matches!(op, Op::Call { .. })), 1);
    run_full(&mut func);
    assert_eq!(count_ops(&func, |op| matches!(op, Op::Call { .. })), 0);
    verify_and_compile_both(&func);
    assert_eq!(jit_run_2(&func, 10, 0), 55);
    assert_eq!(jit_run_2(&func, 0, 7), 7);
}

#[test]
fn pass_park_and_alloc_preserve_high_pressure_values_across_call() {
    assert!(crate::park::crosses_span(&[10, 20], (5, 15)));
    assert!(!crate::park::crosses_span(&[10, 20], (1, 9)));

    let func = build_fn(native_sig(2, 1), |b, p| {
        let call_sig = b.import_signature(native_sig(1, 1));
        let mut live = Vec::new();
        for idx in 0..14i32 {
            let v = b.ins().load(Type::I64, p[0], idx * 8);
            live.push(v);
        }
        let sym = b.ins().symbol(0);
        let call1 = b.ins().call_indirect(call_sig, sym, &[p[1]]);
        let mid = b.inst_results(call1)[0];
        let call2 = b.ins().call_indirect(call_sig, sym, &[mid]);
        let mut sum = b.inst_results(call2)[0];
        let one = b.ins().iconst(Type::I64, 1);
        for v in live {
            let shifted = b.ins().ishl(sum, one);
            sum = b.ins().bxor(shifted, v);
        }
        b.ins().return_(&[sum]);
    });
    verify_and_compile_both(&func);

    let code = compile(func, &host_full()).expect("host compile park");
    let mut mem = Memory::default();
    let addr = mem
        .write_patched(&code, &[host_add_100 as *const () as usize as u64])
        .expect("map park");
    // SAFETY: `code` was compiled for the host target with `CallConv::Native` signature `fn(*const i64, i64) -> i64`.
    let f: extern "C" fn(*const i64, i64) -> i64 = unsafe { std::mem::transmute(addr) };
    let data: [i64; 14] = std::array::from_fn(|i| (i as i64 + 1) * 11);
    let expected = data
        .iter()
        .fold(host_add_100(host_add_100(5)), |acc, &v| (acc << 1) ^ v);
    assert_eq!(f(data.as_ptr(), 5), expected);
}

#[test]
fn verify_rejects_type_mismatch() {
    let func = build_fn(sig_i64(1, 1), |b, p| {
        let narrow = b.ins().iconst(Type::I32, 1);
        let res = b.ins().iadd(p[0], narrow);
        b.ins().return_(&[res]);
    });
    let err = compile(func, &full_x64()).expect_err("type mismatch must fail");
    assert_eq!(err.kind(), ErrorKind::Verify);
    assert!(err.to_string().contains("Iadd"), "got: {err}");
}

#[test]
fn verify_rejects_dominance_violation() {
    let func = build_fn(sig_i64(1, 1), |b, p| {
        let left = b.create_block();
        let right = b.create_block();
        let cond = b.ins().icmp_imm(IntCC::Equal, p[0], 0);
        b.ins().brif(cond, left, &[], right, &[]);
        b.switch_to_block(left);
        let only_left = b.ins().iadd_imm(p[0], 1);
        b.ins().return_(&[only_left]);
        b.switch_to_block(right);
        b.ins().return_(&[only_left]);
    });
    let err = compile(func, &full_x64()).expect_err("dominance error must fail");
    assert_eq!(err.kind(), ErrorKind::Verify);
    assert!(err.to_string().contains("dominat"), "got: {err}");
}

#[test]
fn verify_rejects_block_arg_type_and_count_mismatch() {
    let bad_type = build_fn(sig_i64(1, 1), |b, p| {
        let next = b.create_block();
        let np = b.append_block_param(next, Type::I32);
        b.ins().jump(next, &[p[0]]);
        b.switch_to_block(next);
        let ext = b.ins().sextend(Type::I64, np);
        b.ins().return_(&[ext]);
    });
    let err = compile(bad_type, &full_x64()).expect_err("edge type mismatch must fail");
    assert_eq!(err.kind(), ErrorKind::Verify);
    assert!(err.to_string().contains("takes"), "got: {err}");

    let bad_count = build_fn(sig_i64(1, 1), |b, p| {
        let next = b.create_block();
        let np = b.append_block_param(next, Type::I64);
        b.ins().jump(next, &[p[0], p[0]]);
        b.switch_to_block(next);
        b.ins().return_(&[np]);
    });
    let err = compile(bad_count, &full_x64()).expect_err("edge arg count mismatch must fail");
    assert_eq!(err.kind(), ErrorKind::Verify);
    assert!(err.to_string().contains("takes"), "got: {err}");
}

#[test]
fn verify_rejects_invalid_call_signature_and_missing_terminator() {
    let bad_call = build_fn(sig_i64(1, 1), |b, _| {
        let sig = b.import_signature(sig_i64(1, 1));
        let sym = b.ins().symbol(0);
        let narrow = b.ins().iconst(Type::I32, 3);
        let call = b.ins().call_indirect(sig, sym, &[narrow]);
        let r = b.inst_results(call)[0];
        b.ins().return_(&[r]);
    });
    let err = compile(bad_call, &full_x64()).expect_err("bad call arg must fail");
    assert_eq!(err.kind(), ErrorKind::Verify);
    assert!(err.to_string().contains("call takes"), "got: {err}");

    let no_term = build_fn(sig_i64(1, 1), |b, p| {
        let _ = b.ins().iadd_imm(p[0], 1);
    });
    let err = compile(no_term, &full_x64()).expect_err("missing terminator must fail");
    assert_eq!(err.kind(), ErrorKind::Verify);
}

#[test]
fn wire_round_trip_preserves_rich_function_and_rejects_corrupt_bytes() {
    let func = build_fn(sig_i64(2, 1), |b, p| {
        b.func.self_slots.push(0);
        let try_sig = b.import_signature(sig_i64(1, 0));
        let ok = b.create_block();
        let unwind = b.create_block();
        let _exc = b.append_block_param(unwind, Type::I64);
        let slot = b.create_stack_slot(16, 4);
        b.func.slots[slot.index()].poll_local = true;
        let flag = b.create_stack_slot(1, 0);
        b.func.unwind_drops.push(UnwindDrop {
            local: slot,
            local_off: 4,
            flag,
            drop_sym: 3,
            arg: crate::UnwindArg::Indirect,
        });

        b.ins().stack_store(p[0], slot, 0);
        let loaded = b.ins().stack_load(Type::I64, slot, 0);
        let loaded32 = b.ins().ireduce(Type::I32, loaded);
        let vec = b.ins().splat(
            Type::Vec {
                lane: Lane::I32,
                count: 4,
            },
            loaded32,
        );
        let red = b.ins().reduce(ReduceOp::Add, vec);
        let red64 = b.ins().sextend(Type::I64, red);
        let _atomic = b.ins().atomic_load(Type::I64, p[1], AtomicOrd::Acquire);
        let sym = b.ins().symbol(1);
        b.ins()
            .try_call_indirect(try_sig, sym, &[red64], ok, unwind);

        b.switch_to_block(ok);
        b.ins().return_(&[red64]);

        b.switch_to_block(unwind);
        b.ins().trap();
    });
    let target = full_x64();
    assert!(verify::verify(&func, &target).is_ok());

    let encoded = func.encode().expect("function encodes");
    let decoded = Function::decode(&encoded).expect("function decodes");
    assert_eq!(format!("{decoded}"), format!("{func}"));
    assert_eq!(decoded.self_slots, func.self_slots);
    assert_eq!(decoded.slots.len(), func.slots.len());
    assert!(decoded.slots[StackSlot(0).index()].poll_local);
    assert_eq!(decoded.unwind_drops, func.unwind_drops);
    assert!(verify::verify(&decoded, &target).is_ok());

    assert!(Function::decode(&[]).is_none());
    assert!(Function::decode(&encoded[..encoded.len() / 2]).is_none());
    let mut bad_ver = encoded.clone();
    bad_ver[0] ^= 0xFF;
    assert!(Function::decode(&bad_ver).is_none());
    let mut trailing = encoded.clone();
    trailing.push(0xAA);
    let _ = Function::decode(&trailing);
    for flip in [1, encoded.len() / 3, encoded.len() - 1] {
        let mut mutated = encoded.clone();
        mutated[flip] ^= 0xFF;
        let _ = Function::decode(&mutated);
    }
}

#[test]
fn target_encoders_compile_scalar_vector_and_relocs_across_all_targets() {
    let func = build_fn(sig_i64(2, 1), |b, p| {
        let call_sig = b.import_signature(sig_i64(1, 1));
        let pc = b.ins().popcnt(p[0]);
        let lz = b.ins().clz(p[1]);
        let comb = b.ins().iadd(pc, lz);
        let a32 = b.ins().ireduce(Type::I32, p[0]);
        let v = b.ins().splat(
            Type::Vec {
                lane: Lane::I32,
                count: 4,
            },
            a32,
        );
        let v2 = b.ins().iadd(v, v);
        let lane0 = b.ins().extract(v2, 0);
        let lane64 = b.ins().sextend(Type::I64, lane0);
        let arg = b.ins().iadd(comb, lane64);
        let sym = b.ins().symbol(0);
        let call = b.ins().call_indirect(call_sig, sym, &[arg]);
        let ret = b.inst_results(call)[0];
        b.ins().return_(&[ret]);
    });
    for target in TARGETS {
        let code = compile(func.clone(), target).expect("compiles for target");
        assert!(
            !code.bytes().is_empty(),
            "{} emitted empty code",
            target.triple
        );
        assert!(
            !code.relocs().is_empty(),
            "{} should record call symbol relocation",
            target.triple
        );
        for reloc in code.relocs() {
            assert!(
                (reloc.offset as usize) < code.bytes().len(),
                "{} reloc offset out of bounds",
                target.triple
            );
            match target.arch {
                Arch::Arm64 => assert!(matches!(
                    reloc.kind,
                    RelocKind::Arm64Call26 | RelocKind::Arm64AdrpAdd | RelocKind::Arm64Movw64
                )),
                Arch::X64 => assert!(matches!(
                    reloc.kind,
                    RelocKind::PcRel32 | RelocKind::Abs64 | RelocKind::X64RipRel32
                )),
            }
        }
        if target.arch == Arch::Arm64 {
            assert_eq!(code.bytes().len() % 4, 0);
        }
    }
}

#[test]
fn thread_loop_spec_preserves_dominance_when_dirty_block_reenters_latch() {
    let mut func = build_fn(native_sig(1, 1), |b, p| {
        let slot = b.create_stack_slot(8, 3);
        let head = b.create_block();
        let body = b.create_block();
        let dirty = b.create_block();
        let latch = b.create_block();
        let exit = b.create_block();
        let i = b.append_block_param(head, Type::I64);
        let acc = b.append_block_param(head, Type::I64);

        let zero = b.ins().iconst(Type::I64, 0);
        b.ins().stack_store(zero, slot, 0);
        b.ins().jump(head, &[zero, zero]);

        b.switch_to_block(head);
        let cont = b.ins().icmp(IntCC::SignedLessThan, i, p[0]);
        b.ins().brif(cont, body, &[], exit, &[]);

        b.switch_to_block(body);
        let step = b.ins().iadd_imm(i, 1);
        let flag = b.ins().stack_load(Type::I64, slot, 0);
        let is_zero = b.ins().icmp_imm(IntCC::Equal, flag, 0);
        b.ins().brif(is_zero, dirty, &[], latch, &[]);

        b.switch_to_block(dirty);
        let one = b.ins().iconst(Type::I64, 1);
        b.ins().stack_store(one, slot, 0);
        b.ins().jump(latch, &[]);

        b.switch_to_block(latch);
        let next_acc = b.ins().iadd(acc, step);
        b.ins().jump(head, &[step, next_acc]);

        b.switch_to_block(exit);
        b.ins().return_(&[acc]);
    });
    run_full(&mut func);
    verify_and_compile_both(&func);
    assert_eq!(jit_run_1(&func, 5), 15);
}

#[test]
fn pass_tailrec_allows_cold_stack_addr_and_rejects_live_stack_addr() {
    let make = |in_tail: bool| {
        build_fn(native_sig(1, 1), |b, p| {
            b.func.self_slots.push(0);
            let sig = b.import_signature(native_sig(1, 1));
            let (panic_blk, check, base, rec) = (
                b.create_block(),
                b.create_block(),
                b.create_block(),
                b.create_block(),
            );
            let slot = b.create_stack_slot(8, 3);
            let bad = b.ins().icmp_imm(IntCC::SignedLessThan, p[0], 0);
            b.ins().brif(bad, panic_blk, &[], check, &[]);

            b.switch_to_block(panic_blk);
            let cold_addr = b.ins().stack_addr(Type::I64, slot, 0);
            let _ = b.ins().load(Type::I64, cold_addr, 0);
            b.ins().trap();

            b.switch_to_block(check);
            let zero = b.ins().icmp_imm(IntCC::Equal, p[0], 0);
            b.ins().brif(zero, base, &[], rec, &[]);

            b.switch_to_block(base);
            let c0 = b.ins().iconst(Type::I64, 0);
            b.ins().return_(&[c0]);

            b.switch_to_block(rec);
            if in_tail {
                let tail_addr = b.ins().stack_addr(Type::I64, slot, 0);
                let _ = b.ins().load(Type::I64, tail_addr, 0);
            }
            let next = b.ins().iadd_imm(p[0], -1);
            let sym = b.ins().slot_addr(0);
            let callee = b.ins().load(Type::I64, sym, 0);
            let call = b.ins().call_indirect(sig, callee, &[next]);
            let res = b.inst_results(call)[0];
            b.ins().return_(&[res]);
        })
    };
    let mut cold_ok = make(false);
    run_full(&mut cold_ok);
    assert_eq!(count_ops(&cold_ok, |op| matches!(op, Op::Call { .. })), 0);
    verify_and_compile_both(&cold_ok);
    assert_eq!(jit_run_1(&cold_ok, 10), 0);

    let mut live_bad = make(true);
    let changed = crate::tailrec::run(&mut live_bad, &host_full());
    assert!(changed.is_empty());
    assert_eq!(count_ops(&live_bad, |op| matches!(op, Op::Call { .. })), 1);
}
