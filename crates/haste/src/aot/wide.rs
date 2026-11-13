use anyhow::{Context as _, Result, anyhow};
use haste_codegen::{AbiParam, Arch, CallConv, Effort, FloatCC, Function, FunctionBuilder, IntCC, Os, RuntimeSymbol, Signature, TARGETS, Type, Value};

use super::Buf;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::aot) enum WideShim {
    Runtime(RuntimeSymbol),
    Powi { f64: bool },
}

pub(super) fn emit(arch: Arch, os: Os, kind: WideShim, out: &mut Buf) -> Result<()> {
    let target = TARGETS.iter().find(|t| t.arch == arch && t.os == os).copied().context("missing codegen target for AOT shim")?.effort(Effort::Full);
    let func = match kind {
        WideShim::Runtime(RuntimeSymbol::I128Udiv) => build_divrem(false, false),
        WideShim::Runtime(RuntimeSymbol::I128Urem) => build_divrem(false, true),
        WideShim::Runtime(RuntimeSymbol::I128Sdiv) => build_divrem(true, false),
        WideShim::Runtime(RuntimeSymbol::I128Srem) => build_divrem(true, true),
        WideShim::Runtime(RuntimeSymbol::U128ToF32) => build_int_to_float(false, Type::F32),
        WideShim::Runtime(RuntimeSymbol::U128ToF64) => build_int_to_float(false, Type::F64),
        WideShim::Runtime(RuntimeSymbol::I128ToF32) => build_int_to_float(true, Type::F32),
        WideShim::Runtime(RuntimeSymbol::I128ToF64) => build_int_to_float(true, Type::F64),
        WideShim::Runtime(RuntimeSymbol::F32ToU128) => build_float_to_int(Type::F32, false),
        WideShim::Runtime(RuntimeSymbol::F64ToU128) => build_float_to_int(Type::F64, false),
        WideShim::Runtime(RuntimeSymbol::F32ToI128) => build_float_to_int(Type::F32, true),
        WideShim::Runtime(RuntimeSymbol::F64ToI128) => build_float_to_int(Type::F64, true),
        WideShim::Runtime(other) => return Err(anyhow!("unexpected runtime shim {other:?}")),
        WideShim::Powi { f64 } => build_powi(if f64 { Type::F64 } else { Type::F32 }),
    };
    let code = haste_codegen::compile(func, &target).map_err(|e| anyhow!("{e}"))?;
    if !code.relocs().is_empty() {
        return Err(anyhow!("AOT wide shim emitted nested relocations"));
    }
    out.bytes(code.bytes());
    Ok(())
}

fn new_fn(params: &[Type], ret: Type) -> Function {
    let mut sig = Signature::new(CallConv::Native);
    sig.params = params.iter().map(|&ty| AbiParam::new(ty)).collect();
    sig.returns = vec![AbiParam::new(ret)];
    Function::new(sig)
}

fn const128(b: &mut FunctionBuilder<'_>, lo: u64, hi: u64) -> Value {
    let l = b.ins().iconst(Type::I64, lo as i64);
    let h = b.ins().iconst(Type::I64, hi as i64);
    b.ins().iconcat(l, h)
}

fn abs128(b: &mut FunctionBuilder<'_>, x: Value) -> (Value, Value) {
    let c127 = b.ins().iconst(Type::I64, 127);
    let sign = b.ins().sshr(x, c127);
    let xored = b.ins().bxor(x, sign);
    (b.ins().isub(xored, sign), sign)
}

fn apply_sign128(b: &mut FunctionBuilder<'_>, u: Value, sign: Value) -> Value {
    let xored = b.ins().bxor(u, sign);
    b.ins().isub(xored, sign)
}

fn udivrem(b: &mut FunctionBuilder<'_>, x: Value, y: Value, want_rem: bool) -> Value {
    let zero = const128(b, 0, 0);
    let is_zero = b.ins().icmp(IntCC::Equal, y, zero);
    let z_blk = b.create_block();
    let pre = b.create_block();
    let hdr = b.create_block();
    let done = b.create_block();
    let q = b.append_block_param(hdr, Type::I128);
    let r = b.append_block_param(hdr, Type::I128);
    let sh = b.append_block_param(hdr, Type::I64);
    let out = b.append_block_param(done, Type::I128);
    b.ins().brif(is_zero, z_blk, &[], pre, &[]);

    b.switch_to_block(z_blk);
    let z_res = if want_rem { x } else { zero };
    b.ins().jump(done, &[z_res]);

    b.switch_to_block(pre);
    let sh0 = b.ins().iconst(Type::I64, 127);
    b.ins().jump(hdr, &[zero, zero, sh0]);

    b.switch_to_block(hdr);
    let one128 = const128(b, 1, 0);
    let c1 = b.ins().iconst(Type::I64, 1);
    let c127 = b.ins().iconst(Type::I64, 127);
    let shifted_x = b.ins().ushr(x, sh);
    let bit = b.ins().band(shifted_x, one128);
    let r_hi = b.ins().ushr(r, c127);
    let r_hi8 = b.ins().ireduce(Type::I8, r_hi);
    let r_shl = b.ins().ishl(r, c1);
    let r1 = b.ins().bor(r_shl, bit);
    let ge_cmp = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, r1, y);
    let ge = b.ins().bor(r_hi8, ge_cmp);
    let r_sub = b.ins().isub(r1, y);
    let r2 = b.ins().select(ge, r_sub, r1);
    let ge128 = b.ins().uextend(Type::I128, ge);
    let q_shl = b.ins().ishl(q, c1);
    let q1 = b.ins().bor(q_shl, ge128);
    let at_end = b.ins().icmp_imm(IntCC::Equal, sh, 0);
    let sh_next = b.ins().iadd_imm(sh, -1);
    let res = if want_rem { r2 } else { q1 };
    b.ins().brif(at_end, done, &[res], hdr, &[q1, r2, sh_next]);

    b.switch_to_block(done);
    out
}

fn build_divrem(signed: bool, want_rem: bool) -> Function {
    let mut func = new_fn(&[Type::I128, Type::I128], Type::I128);
    let mut b = FunctionBuilder::new(&mut func);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let x = b.block_params(entry)[0];
    let y = b.block_params(entry)[1];
    let res = if signed {
        let (ax, sx) = abs128(&mut b, x);
        let (ay, sy) = abs128(&mut b, y);
        let u = udivrem(&mut b, ax, ay, want_rem);
        let sign = if want_rem { sx } else { b.ins().bxor(sx, sy) };
        apply_sign128(&mut b, u, sign)
    } else {
        udivrem(&mut b, x, y, want_rem)
    };
    b.ins().return_(&[res]);
    func
}

fn bitcast_to_float(b: &mut FunctionBuilder<'_>, bits: Value, to: Type) -> Value {
    let slot = b.create_stack_slot(to.bytes(), if to == Type::F64 { 3 } else { 2 });
    b.ins().stack_store(bits, slot, 0);
    b.ins().stack_load(to, slot, 0)
}

fn u128_to_float(b: &mut FunctionBuilder<'_>, x: Value, to: Type) -> Value {
    let lz128 = b.ins().clz(x);
    let lz = b.ins().ireduce(Type::I64, lz128);
    let small = b.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, lz, 64);
    let small_blk = b.create_block();
    let big_blk = b.create_block();
    let done = b.create_block();
    let out = b.append_block_param(done, to);
    b.ins().brif(small, small_blk, &[], big_blk, &[]);

    b.switch_to_block(small_blk);
    let lo = b.ins().ireduce(Type::I64, x);
    let f_small = b.ins().fcvt_from_uint(to, lo);
    b.ins().jump(done, &[f_small]);

    b.switch_to_block(big_blk);
    let c64 = b.ins().iconst(Type::I64, 64);
    let k = b.ins().isub(c64, lz);
    let rem_sh = b.ins().iadd_imm(lz, 64);
    let dropped = b.ins().ishl(x, rem_sh);
    let zero128 = const128(b, 0, 0);
    let has_drop = b.ins().icmp(IntCC::NotEqual, dropped, zero128);
    let sticky = b.ins().uextend(Type::I64, has_drop);
    let shifted = b.ins().ushr(x, k);
    let top64 = b.ins().ireduce(Type::I64, shifted);
    let folded = b.ins().bor(top64, sticky);
    let base = b.ins().fcvt_from_uint(to, folded);
    let scale = if to == Type::F64 {
        let exp = b.ins().iadd_imm(k, 1023);
        let c52 = b.ins().iconst(Type::I64, 52);
        let bits = b.ins().ishl(exp, c52);
        bitcast_to_float(b, bits, Type::F64)
    } else {
        let exp64 = b.ins().iadd_imm(k, 127);
        let exp32 = b.ins().ireduce(Type::I32, exp64);
        let c23 = b.ins().iconst(Type::I32, 23);
        let bits = b.ins().ishl(exp32, c23);
        bitcast_to_float(b, bits, Type::F32)
    };
    let scaled = b.ins().fmul(base, scale);
    b.ins().jump(done, &[scaled]);

    b.switch_to_block(done);
    out
}

fn build_int_to_float(signed: bool, to: Type) -> Function {
    let mut func = new_fn(&[Type::I128], to);
    let mut b = FunctionBuilder::new(&mut func);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let x = b.block_params(entry)[0];
    let res = if signed {
        let (ax, sx) = abs128(&mut b, x);
        let mag = u128_to_float(&mut b, ax, to);
        let zero128 = const128(&mut b, 0, 0);
        let neg_cond = b.ins().icmp(IntCC::NotEqual, sx, zero128);
        let neg_mag = b.ins().fneg(mag);
        b.ins().select(neg_cond, neg_mag, mag)
    } else {
        u128_to_float(&mut b, x, to)
    };
    b.ins().return_(&[res]);
    func
}

fn f64_const(b: &mut FunctionBuilder<'_>, val: f64) -> Value {
    b.ins().iconst(Type::F64, val.to_bits() as i64)
}

fn f64_to_u128(b: &mut FunctionBuilder<'_>, x: Value) -> Value {
    let f0 = f64_const(b, 0.0);
    let pos = b.ins().fcmp(FloatCC::GreaterThan, x, f0);
    let z_blk = b.create_block();
    let chk_ovf = b.create_block();
    let max_blk = b.create_block();
    let chk_64 = b.create_block();
    let lo_blk = b.create_block();
    let hi_blk = b.create_block();
    let done = b.create_block();
    let out = b.append_block_param(done, Type::I128);
    b.ins().brif(pos, chk_ovf, &[], z_blk, &[]);

    b.switch_to_block(z_blk);
    let zero = const128(b, 0, 0);
    b.ins().jump(done, &[zero]);

    b.switch_to_block(chk_ovf);
    let two128 = b.ins().iconst(Type::F64, 0x47f0_0000_0000_0000_u64 as i64);
    let ovf = b.ins().fcmp(FloatCC::GreaterThanOrEqual, x, two128);
    b.ins().brif(ovf, max_blk, &[], chk_64, &[]);

    b.switch_to_block(max_blk);
    let umax = const128(b, u64::MAX, u64::MAX);
    b.ins().jump(done, &[umax]);

    b.switch_to_block(chk_64);
    let two64 = b.ins().iconst(Type::F64, 0x43f0_0000_0000_0000_u64 as i64);
    let small = b.ins().fcmp(FloatCC::LessThan, x, two64);
    b.ins().brif(small, lo_blk, &[], hi_blk, &[]);

    b.switch_to_block(lo_blk);
    let lo_only = b.ins().fcvt_to_uint_sat(Type::I64, x);
    let ext = b.ins().uextend(Type::I128, lo_only);
    b.ins().jump(done, &[ext]);

    b.switch_to_block(hi_blk);
    let inv_two64 = b.ins().iconst(Type::F64, 0x3bf0_0000_0000_0000_u64 as i64);
    let scaled_down = b.ins().fmul(x, inv_two64);
    let trunc_hi = b.ins().trunc(scaled_down);
    let hi = b.ins().fcvt_to_uint_sat(Type::I64, trunc_hi);
    let hi_f = b.ins().fcvt_from_uint(Type::F64, hi);
    let two64_b = b.ins().iconst(Type::F64, 0x43f0_0000_0000_0000_u64 as i64);
    let hi_back = b.ins().fmul(hi_f, two64_b);
    let rem_f = b.ins().fsub(x, hi_back);
    let lo = b.ins().fcvt_to_uint_sat(Type::I64, rem_f);
    let pair = b.ins().iconcat(lo, hi);
    b.ins().jump(done, &[pair]);

    b.switch_to_block(done);
    out
}

fn build_float_to_int(from: Type, signed: bool) -> Function {
    let mut func = new_fn(&[from], Type::I128);
    let mut b = FunctionBuilder::new(&mut func);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let raw = b.block_params(entry)[0];
    let x = if from == Type::F32 { b.ins().fpromote(Type::F64, raw) } else { raw };
    if !signed {
        let res = f64_to_u128(&mut b, x);
        b.ins().return_(&[res]);
        return func;
    }
    let is_nan = b.ins().fcmp(FloatCC::NotEqual, x, x);
    let nan_blk = b.create_block();
    let chk_min = b.create_block();
    let min_blk = b.create_block();
    let chk_max = b.create_block();
    let max_blk = b.create_block();
    let mid_blk = b.create_block();
    let done = b.create_block();
    let out = b.append_block_param(done, Type::I128);
    b.ins().brif(is_nan, nan_blk, &[], chk_min, &[]);

    b.switch_to_block(nan_blk);
    let zero = const128(&mut b, 0, 0);
    b.ins().jump(done, &[zero]);

    b.switch_to_block(chk_min);
    let neg_two127 = b.ins().iconst(Type::F64, 0xc7e0_0000_0000_0000_u64 as i64);
    let too_low = b.ins().fcmp(FloatCC::LessThanOrEqual, x, neg_two127);
    b.ins().brif(too_low, min_blk, &[], chk_max, &[]);

    b.switch_to_block(min_blk);
    let imin = const128(&mut b, 0, 1_u64 << 63);
    b.ins().jump(done, &[imin]);

    b.switch_to_block(chk_max);
    let pos_two127 = b.ins().iconst(Type::F64, 0x47e0_0000_0000_0000_u64 as i64);
    let too_high = b.ins().fcmp(FloatCC::GreaterThanOrEqual, x, pos_two127);
    b.ins().brif(too_high, max_blk, &[], mid_blk, &[]);

    b.switch_to_block(max_blk);
    let imax = const128(&mut b, u64::MAX, i64::MAX as u64);
    b.ins().jump(done, &[imax]);

    b.switch_to_block(mid_blk);
    let f0 = f64_const(&mut b, 0.0);
    let neg = b.ins().fcmp(FloatCC::LessThan, x, f0);
    let ax = b.ins().fabs(x);
    let u = f64_to_u128(&mut b, ax);
    let nu = b.ins().ineg(u);
    let signed_res = b.ins().select(neg, nu, u);
    b.ins().jump(done, &[signed_res]);

    b.switch_to_block(done);
    b.ins().return_(&[out]);
    func
}

fn build_powi(ty: Type) -> Function {
    let mut func = new_fn(&[ty, Type::I32], ty);
    let mut b = FunctionBuilder::new(&mut func);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let x = b.block_params(entry)[0];
    let n = b.block_params(entry)[1];

    let one_bits = if ty == Type::F64 { 1.0_f64.to_bits() as i64 } else { i64::from(1.0_f32.to_bits()) };
    let one = b.ins().iconst(ty, one_bits);
    let is_neg = b.ins().icmp_imm(IntCC::SignedLessThan, n, 0);
    let neg_n = b.ins().ineg(n);
    let exp0 = b.ins().select(is_neg, neg_n, n);

    let hdr = b.create_block();
    let step = b.create_block();
    let sq_blk = b.create_block();
    let finish = b.create_block();
    let acc = b.append_block_param(hdr, ty);
    let base = b.append_block_param(hdr, ty);
    let exp = b.append_block_param(hdr, Type::I32);
    let final_acc = b.append_block_param(finish, ty);
    b.ins().jump(hdr, &[one, x, exp0]);

    b.switch_to_block(hdr);
    let done_cond = b.ins().icmp_imm(IntCC::Equal, exp, 0);
    b.ins().brif(done_cond, finish, &[acc], step, &[]);

    b.switch_to_block(step);
    let c1 = b.ins().iconst(Type::I32, 1);
    let low_bit = b.ins().band(exp, c1);
    let odd = b.ins().icmp_imm(IntCC::NotEqual, low_bit, 0);
    let prod = b.ins().fmul(acc, base);
    let acc1 = b.ins().select(odd, prod, acc);
    let exp1 = b.ins().ushr(exp, c1);
    let more = b.ins().icmp_imm(IntCC::NotEqual, exp1, 0);
    b.ins().brif(more, sq_blk, &[], finish, &[acc1]);

    b.switch_to_block(sq_blk);
    let base1 = b.ins().fmul(base, base);
    b.ins().jump(hdr, &[acc1, base1, exp1]);

    b.switch_to_block(finish);
    let one_end = b.ins().iconst(ty, one_bits);
    let inv = b.ins().fdiv(one_end, final_acc);
    let res = b.ins().select(is_neg, inv, final_acc);
    b.ins().return_(&[res]);
    func
}
