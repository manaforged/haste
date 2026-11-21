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
