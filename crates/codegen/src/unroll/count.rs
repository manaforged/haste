use crate::ir::{Binary, Block, Function, Inst, IntCC, Op, Type, Value};

/// The exit test of a one-block loop that compares a stepped induction value against a limit set outside the loop.
pub(super) struct Counted {
    pub(super) cmp_cc: IntCC,
    pub(super) is_ne: bool,
    pub(super) iv_ty: Type,
    pub(super) limit: Value,
    pub(super) iv_idx: usize,
    pub(super) stride: Value,
    pub(super) bias: i64,
}

/// The constant step of an induction value when it is positive, small, and one for a `!=` exit test.
pub(super) fn stride_const(global_consts: &[Option<i64>], stride: Value, is_ne: bool) -> Option<i64> {
    global_consts[stride.index()].filter(|&c| c > 0 && c <= (1_i64 << 24) && (!is_ne || c == 1))
}

/// For each value, whether block `b` defines it as a parameter or by one of `body_insts`.
pub(super) fn head_defs(func: &Function, b: usize, body_insts: &[Inst]) -> Vec<bool> {
    let mut defs = vec![false; func.types.len()];
    for &p in &func.blocks[b].params {
        defs[p.index()] = true;
    }
    for &i in body_insts {
        if let Some(r) = func.insts[i.index()].result {
            defs[r.index()] = true;
        }
    }
    defs
}

/// The blocks other than `b` that branch to block `b`.
pub(super) fn outside_preds(func: &Function, b: usize) -> Vec<usize> {
    let head = Block(b as u32);
    (0..func.blocks.len()).filter(|&p| p != b && func.successors(Block(p as u32)).any(|t| t == head)).collect()
}

fn match_iv_step(func: &Function, b: usize, body_insts: &[Inst], back_args: &[Value], iv_next: Value) -> Option<(usize, Value)> {
    let iv_inst = body_insts.iter().copied().find(|&i| func.insts[i.index()].result == Some(iv_next))?;
    let Op::Binary(Binary::Iadd, p, q) = func.insts[iv_inst.index()].op else { return None };
    for (cand_iv, cand_stride) in [(p, q), (q, p)] {
        if let Some(idx) = func.blocks[b].params.iter().position(|&param| param == cand_iv)
            && back_args[idx] == iv_next
        {
            return Some((idx, cand_stride));
        }
    }
    None
}

/// The counted exit test of the one-block loop `b` whose terminator branches back on `cond` equal to `back_on_true`.
pub(super) fn counted(
    func: &Function,
    b: usize,
    body_insts: &[Inst],
    (cond, back_on_true): (Value, bool),
    back_args: &[Value],
    head_defs: &[bool],
    global_consts: &[Option<i64>],
) -> Option<Counted> {
    let cond_inst = body_insts.iter().copied().find(|&i| func.insts[i.index()].result == Some(cond))?;
    let Op::Icmp(cc, lhs, rhs) = func.insts[cond_inst.index()].op else { return None };
    let eff_cc = if back_on_true { cc } else { cc.inverse() };
    let (cmp_cc, is_ne, iv_cmp, limit) = match eff_cc {
        IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual => (eff_cc, false, lhs, rhs),
        IntCC::UnsignedGreaterThan => (IntCC::UnsignedLessThan, false, rhs, lhs),
        IntCC::UnsignedGreaterThanOrEqual => (IntCC::UnsignedLessThanOrEqual, false, rhs, lhs),
        IntCC::NotEqual if !head_defs[rhs.index()] => (IntCC::UnsignedLessThan, true, lhs, rhs),
        IntCC::NotEqual if !head_defs[lhs.index()] => (IntCC::UnsignedLessThan, true, rhs, lhs),
        _ => return None,
    };
    let iv_ty = func.ty(iv_cmp);
    if !matches!(iv_ty, Type::I32 | Type::I64) || head_defs[limit.index()] {
        return None;
    }
    let bias_cap = if iv_ty == Type::I32 { 1_i64 << 28 } else { 1_i64 << 30 };
    let stride_ok = |s: Value| !head_defs[s.index()] || stride_const(global_consts, s, is_ne).is_some();
    let mut matched = match_iv_step(func, b, body_insts, back_args, iv_cmp).filter(|&(_, stride)| stride_ok(stride)).map(|(idx, stride)| (idx, stride, 0_i64));
    if matched.is_none()
        && let Some(cmp_add) = body_insts.iter().copied().find(|&i| func.insts[i.index()].result == Some(iv_cmp))
        && let Op::Binary(Binary::Iadd, p, q) = func.insts[cmp_add.index()].op
    {
        for (cand_next, cand_bias) in [(p, q), (q, p)] {
            if let Some(bias) = global_consts[cand_bias.index()]
                && (0..bias_cap).contains(&bias)
                && let Some((idx, stride)) = match_iv_step(func, b, body_insts, back_args, cand_next)
                && stride_ok(stride)
            {
                matched = Some((idx, stride, bias));
                break;
            }
        }
    }
    let (iv_idx, stride, bias) = matched?;
    Some(Counted { cmp_cc, is_ne, iv_ty, limit, iv_idx, stride, bias })
}

/// Validates that every `Op::Trapz` in `body_insts` is a bounds check `iv_param < inv_len` with loop-invariant `inv_len` of type `iv_ty`.
pub(super) fn guarded_trapz(
    func: &Function,
    body_insts: &[Inst],
    iv_param: Value,
    iv_ty: Type,
    head_defs: &[bool],
) -> Option<(Vec<Inst>, Vec<Value>)> {
    let mut trap_insts = Vec::new();
    let mut inv_lens = Vec::new();
    for &inst in body_insts {
        let Op::Trapz(cond) = func.insts[inst.index()].op else { continue };
        let cond_inst = body_insts.iter().copied().find(|&i| func.insts[i.index()].result == Some(cond))?;
        let Op::Icmp(cc, a, b) = func.insts[cond_inst.index()].op else { return None };
        let inv_len = match cc {
            IntCC::UnsignedLessThan if a == iv_param => b,
            IntCC::UnsignedGreaterThan if b == iv_param => a,
            _ => return None,
        };
        if head_defs[inv_len.index()] || func.ty(inv_len) != iv_ty {
            return None;
        }
        trap_insts.push(inst);
        if !inv_lens.contains(&inv_len) {
            inv_lens.push(inv_len);
        }
    }
    Some((trap_insts, inv_lens))
}

/// Emits the runtime safety guard for dynamic unrolling, returning the adjusted fast-path limit.
pub(super) fn emit_dynamic_guard(
    func: &mut Function,
    guard: Block,
    counted_info: &Counted,
    (const_stride, factor, pre_sf): (Option<i64>, i64, Option<Value>),
    (head, unrolled, guard_params): (Block, Block, Vec<Value>),
    trap_lens: &[Value],
) -> Value {
    let &Counted { cmp_cc, is_ne, iv_ty, limit, iv_idx, stride, bias } = counted_info;
    let mut emit_guard = |op: Op, ty: Type| -> Value {
        let v = func.value(ty);
        let i = func.add(op, Some(v));
        func.blocks[guard.index()].insts.push(i);
        v
    };
    let (s_need, g1) = if let Some(c) = const_stride {
        (emit_guard(Op::Const(c * factor + bias), iv_ty), None)
    } else {
        let zero = emit_guard(Op::Const(0), iv_ty);
        let sh = emit_guard(Op::Const(i64::from(iv_ty.bits() - 3)), iv_ty);
        let stride_ok = if is_ne {
            let one = emit_guard(Op::Const(1), iv_ty);
            emit_guard(Op::Icmp(IntCC::Equal, stride, one), Type::I8)
        } else {
            emit_guard(Op::Icmp(IntCC::UnsignedGreaterThan, stride, zero), Type::I8)
        };
        let or_lim = emit_guard(Op::Binary(Binary::Bor, limit, stride), iv_ty);
        let hi_bits = emit_guard(Op::Binary(Binary::Ushr, or_lim, sh), iv_ty);
        let no_wrap = emit_guard(Op::Icmp(IntCC::Equal, hi_bits, zero), Type::I8);
        let sf = pre_sf.unwrap_or_else(|| {
            let cf = emit_guard(Op::Const(factor), iv_ty);
            emit_guard(Op::Binary(Binary::Imul, stride, cf), iv_ty)
        });
        let need = if bias > 0 {
            let b_val = emit_guard(Op::Const(bias), iv_ty);
            emit_guard(Op::Binary(Binary::Iadd, sf, b_val), iv_ty)
        } else {
            sf
        };
        (need, Some(emit_guard(Op::Binary(Binary::Band, stride_ok, no_wrap), Type::I8)))
    };
    let lim_ok = emit_guard(Op::Icmp(IntCC::UnsignedLessThanOrEqual, s_need, limit), Type::I8);
    let safe_limit = emit_guard(Op::Binary(Binary::Isub, limit, s_need), iv_ty);
    let init_ok = emit_guard(Op::Icmp(cmp_cc, guard_params[iv_idx], safe_limit), Type::I8);
    let g2 = emit_guard(Op::Binary(Binary::Band, lim_ok, init_ok), Type::I8);
    let mut guard_cond = match g1 {
        Some(g1) => emit_guard(Op::Binary(Binary::Band, g1, g2), Type::I8),
        None => g2,
    };
    let len_cc = if cmp_cc == IntCC::UnsignedLessThanOrEqual {
        IntCC::UnsignedLessThan
    } else {
        IntCC::UnsignedLessThanOrEqual
    };
    for &inv_len in trap_lens {
        let in_bounds = emit_guard(Op::Icmp(len_cc, limit, inv_len), Type::I8);
        guard_cond = emit_guard(Op::Binary(Binary::Band, guard_cond, in_bounds), Type::I8);
    }
    let guard_br = func.add(Op::Brif(guard_cond, unrolled, guard_params.clone(), head, guard_params), None);
    func.blocks[guard.index()].insts.push(guard_br);
    safe_limit
}
