use super::{defs, extend, known_bits};
use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

struct RangeState {
    defs: Vec<Option<Inst>>,
    uses: Vec<u32>,
    consts: Vec<Option<i64>>,
}

fn def_inst(state: &RangeState, v: Value) -> Option<Inst> {
    state.defs.get(v.index()).copied().flatten()
}

fn const_val(state: &RangeState, v: Value) -> Option<i64> {
    state.consts.get(v.index()).copied().flatten()
}

fn cmp_const(func: &Function, state: &RangeState, v: Value) -> Option<(IntCC, Value, i64)> {
    let Op::Icmp(cc, l, r) = func.insts[def_inst(state, v)?.index()].op else {
        return None;
    };
    if !func.ty(l).is_small_int() {
        return None;
    }
    const_val(state, r)
        .map(|c| (cc, l, c))
        .or_else(|| const_val(state, l).map(|c| (crate::cse::flip_int_cc(cc), r, c)))
}

fn band_args(func: &Function, state: &RangeState, v: Value, eq_cc: IntCC) -> Option<(Value, Value)> {
    let Op::Icmp(cc, l, r) = func.insts[def_inst(state, v)?.index()].op else {
        return None;
    };
    if cc != eq_cc {
        return None;
    }
    let arg = match (const_val(state, l), const_val(state, r)) {
        (_, Some(0)) => l,
        (Some(0), _) => r,
        _ => return None,
    };
    if state.uses.get(arg.index()).copied().unwrap_or(0) != 1 {
        return None;
    }
    let Op::Binary(Binary::Band, x, m) = func.insts[def_inst(state, arg)?.index()].op else {
        return None;
    };
    func.ty(x).is_small_int().then_some((x, m))
}

fn shared_mask_pair(a: (Value, Value), b: (Value, Value)) -> Option<(Value, Value, Value)> {
    if a.0 == b.0 {
        Some((a.0, a.1, b.1))
    } else if a.0 == b.1 {
        Some((a.0, a.1, b.0))
    } else if a.1 == b.0 {
        Some((a.1, a.0, b.1))
    } else if a.1 == b.1 {
        Some((a.1, a.0, b.0))
    } else {
        None
    }
}

fn try_fold_mask_zero(func: &mut Function, inst: Inst, state: &mut RangeState, next: &mut Vec<Inst>) -> bool {
    let Op::Binary(op @ (Binary::Band | Binary::Bor), a, b) = func.insts[inst.index()].op else {
        return false;
    };
    if state.uses[a.index()] != 1 || state.uses[b.index()] != 1 {
        return false;
    }
    let eq_cc = if op == Binary::Band { IntCC::Equal } else { IntCC::NotEqual };
    let (Some(p1), Some(p2)) = (band_args(func, state, a, eq_cc), band_args(func, state, b, eq_cc)) else {
        return false;
    };
    let Some((x, m1, m2)) = shared_mask_pair(p1, p2) else {
        return false;
    };
    let ty = func.ty(x);
    let mask = match (const_val(state, m1), const_val(state, m2)) {
        (Some(c1), Some(c2)) => {
            let mv = func.value(ty);
            let imm = c1 | c2;
            state.consts.push(Some(extend(imm, ty, false)));
            next.push(func.add(Op::Const(imm), Some(mv)));
            mv
        }
        _ => {
            let mv = func.value(ty);
            state.consts.push(None);
            next.push(func.add(Op::Binary(Binary::Bor, m1, m2), Some(mv)));
            mv
        }
    };
    let masked = func.value(ty);
    let zero = func.value(ty);
    state.consts.extend([None, Some(0)]);
    next.push(func.add(Op::Binary(Binary::Band, x, mask), Some(masked)));
    next.push(func.add(Op::Const(0), Some(zero)));
    func.insts[inst.index()].op = Op::Icmp(eq_cc, masked, zero);
    true
}

type FusedCmp = (Option<(Binary, i64)>, IntCC, i64);

fn fuse_eq_pair(is_and: bool, ty: Type, u1: u64, u2: u64) -> Option<FusedCmp> {
    let eq_cc = if is_and { IntCC::NotEqual } else { IntCC::Equal };
    let out_cc = if is_and { IntCC::UnsignedGreaterThan } else { IntCC::UnsignedLessThanOrEqual };
    let (lo, hi) = if u1 <= u2 { (u1, u2) } else { (u2, u1) };
    if lo == hi {
        return Some((None, eq_cc, lo as i64));
    }
    let umax = ty.unsigned_max();
    if umax >= 2 && lo == 0 && hi == 1 {
        return Some((None, out_cc, 1));
    }
    if umax >= 2 && hi - lo == 1 {
        return Some((Some((Binary::Isub, lo as i64)), out_cc, 1));
    }
    if umax >= 2 && lo == 0 && hi == umax {
        return Some((Some((Binary::Iadd, 1)), out_cc, 1));
    }
    let diff = hi - lo;
    (diff.is_power_of_two() && (lo & diff) == 0).then_some((Some((Binary::Bor, diff as i64)), eq_cc, hi as i64))
}

fn unsigned_lo(is_and: bool, ty: Type, side: (IntCC, i64)) -> Option<u64> {
    let umax = ty.unsigned_max();
    match (is_and, side.0, extend(side.1, ty, false) as u64) {
        (true, IntCC::UnsignedGreaterThanOrEqual, u) | (false, IntCC::UnsignedLessThan, u) => Some(u),
        (true, IntCC::UnsignedGreaterThan, u) | (false, IntCC::UnsignedLessThanOrEqual, u) if u < umax => Some(u + 1),
        _ => None,
    }
}

fn unsigned_hi(is_and: bool, ty: Type, side: (IntCC, i64)) -> Option<u64> {
    match (is_and, side.0, extend(side.1, ty, false) as u64) {
        (true, IntCC::UnsignedLessThanOrEqual, u) | (false, IntCC::UnsignedGreaterThan, u) => Some(u),
        (true, IntCC::UnsignedLessThan, u) | (false, IntCC::UnsignedGreaterThanOrEqual, u) if u > 0 => Some(u - 1),
        _ => None,
    }
}

fn fuse_unsigned_pair(is_and: bool, ty: Type, l: (IntCC, i64), h: (IntCC, i64)) -> Option<FusedCmp> {
    let u_lo = unsigned_lo(is_and, ty, l)?;
    let u_hi = unsigned_hi(is_and, ty, h)?;
    if u_lo > u_hi || u_hi - u_lo >= ty.unsigned_max() {
        return None;
    }
    if u_lo == u_hi {
        let eq_cc = if is_and { IntCC::Equal } else { IntCC::NotEqual };
        return Some((None, eq_cc, u_lo as i64));
    }
    let out_cc = if is_and { IntCC::UnsignedLessThanOrEqual } else { IntCC::UnsignedGreaterThan };
    Some(((u_lo != 0).then_some((Binary::Isub, u_lo as i64)), out_cc, (u_hi - u_lo) as i64))
}

fn fuse_signed_pair(is_and: bool, ty: Type, l: (IntCC, i64), h: (IntCC, i64)) -> Option<FusedCmp> {
    let umax = ty.unsigned_max();
    let (smin, smax) = if ty.bits() == 64 {
        (i64::MIN, i64::MAX)
    } else {
        (-(1_i64 << (ty.bits() - 1)), (1_i64 << (ty.bits() - 1)) - 1)
    };
    let s_lo = match (is_and, l.0, extend(l.1, ty, true)) {
        (true, IntCC::SignedGreaterThanOrEqual, s) | (false, IntCC::SignedLessThan, s) => Some(s),
        (true, IntCC::SignedGreaterThan, s) | (false, IntCC::SignedLessThanOrEqual, s) if s < smax => Some(s + 1),
        _ => None,
    }?;
    let s_hi = match (is_and, h.0, extend(h.1, ty, true)) {
        (true, IntCC::SignedLessThanOrEqual, s) | (false, IntCC::SignedGreaterThan, s) => Some(s),
        (true, IntCC::SignedLessThan, s) | (false, IntCC::SignedGreaterThanOrEqual, s) if s > smin => Some(s - 1),
        _ => None,
    }?;
    if s_lo > s_hi {
        return None;
    }
    let (span, lo_u) = ((s_hi as u64).wrapping_sub(s_lo as u64) & umax, (s_lo as u64) & umax);
    if span >= umax {
        return None;
    }
    if span == 0 {
        let eq_cc = if is_and { IntCC::Equal } else { IntCC::NotEqual };
        return Some((None, eq_cc, lo_u as i64));
    }
    let out_cc = if is_and { IntCC::UnsignedLessThanOrEqual } else { IntCC::UnsignedGreaterThan };
    Some(((lo_u != 0).then_some((Binary::Isub, lo_u as i64)), out_cc, span as i64))
}

fn signed_zero_half(is_and: bool, ty: Type, s_side: (IntCC, i64)) -> Option<bool> {
    match (is_and, s_side.0, extend(s_side.1, ty, true)) {
        (true, IntCC::SignedGreaterThanOrEqual, 0)
        | (true, IntCC::SignedGreaterThan, -1)
        | (false, IntCC::SignedLessThan, 0)
        | (false, IntCC::SignedLessThanOrEqual, -1) => Some(true),
        (true, IntCC::SignedLessThan, 0)
        | (true, IntCC::SignedLessThanOrEqual, -1)
        | (false, IntCC::SignedGreaterThanOrEqual, 0)
        | (false, IntCC::SignedGreaterThan, -1) => Some(false),
        _ => None,
    }
}

fn fuse_mixed_sign_pair(is_and: bool, ty: Type, s_side: (IntCC, i64), u_side: (IntCC, i64)) -> Option<FusedCmp> {
    let umax = ty.unsigned_max();
    if umax < 2 {
        return None;
    }
    let smax = umax >> 1;
    if signed_zero_half(is_and, ty, s_side)? {
        let u_hi = unsigned_hi(is_and, ty, u_side)?;
        let out_cc = if is_and { IntCC::UnsignedLessThanOrEqual } else { IntCC::UnsignedGreaterThan };
        Some((None, out_cc, u_hi.min(smax) as i64))
    } else {
        let u_lo = unsigned_lo(is_and, ty, u_side)?;
        let out_cc = if is_and { IntCC::UnsignedGreaterThanOrEqual } else { IntCC::UnsignedLessThan };
        Some((None, out_cc, u_lo.max(smax + 1) as i64))
    }
}

fn fuse_cmp_pair(is_and: bool, ty: Type, p1: (IntCC, i64), p2: (IntCC, i64)) -> Option<FusedCmp> {
    let eq_cc = if is_and { IntCC::NotEqual } else { IntCC::Equal };
    if p1.0 == eq_cc && p2.0 == eq_cc {
        return fuse_eq_pair(is_and, ty, extend(p1.1, ty, false) as u64, extend(p2.1, ty, false) as u64);
    }
    for (l, h) in [(p1, p2), (p2, p1)] {
        if let Some(res) = fuse_unsigned_pair(is_and, ty, l, h)
            .or_else(|| fuse_signed_pair(is_and, ty, l, h))
            .or_else(|| fuse_mixed_sign_pair(is_and, ty, l, h))
        {
            return Some(res);
        }
    }
    None
}

fn signed_max(ty: Type) -> u64 {
    (1_u64 << (ty.bits() - 1)) - 1
}

fn shift_count(state: &RangeState, ty: Type, sh: Value) -> Option<u32> {
    const_val(state, sh).map(|c| (c as u32) & (ty.bits() - 1))
}

fn pos_const(state: &RangeState, v: Value) -> Option<u64> {
    const_val(state, v).map(|d| d as u64).filter(|&d| d > 0)
}

fn smax_u_bounds(sm: u64, w: u64, (l1, h1): (u64, u64), (l2, h2): (u64, u64)) -> (u64, u64) {
    match (h1 <= sm, h2 <= sm) {
        (true, true) => (l1.max(l2), h1.max(h2)),
        (true, false) => (l1, sm),
        (false, true) => (l2, sm),
        (false, false) => (0, w),
    }
}

fn shift_div_u_bounds(func: &Function, state: &RangeState, ty: Type, bin: Binary, (a, b): (Value, Value), depth: u8) -> (u64, u64) {
    let w = ty.unsigned_max();
    let (l1, h1) = u_bounds(func, state, a, depth + 1);
    match bin {
        Binary::Ushr => shift_count(state, ty, b).map_or((0, h1), |k| (l1 >> k, h1 >> k)),
        Binary::Sshr => {
            if h1 <= signed_max(ty) { shift_count(state, ty, b).map_or((0, h1), |k| (l1 >> k, h1 >> k)) } else { (0, w) }
        }
        Binary::Ishl => shift_count(state, ty, b)
            .filter(|&k| h1 <= (w >> k))
            .map_or((0, w), |k| (l1 << k, h1 << k)),
        Binary::Urem => (0, h1.min(pos_const(state, b).map_or(w, |d| d - 1))),
        Binary::Udiv => pos_const(state, b).map_or((0, h1), |d| (l1 / d, h1 / d)),
        Binary::Srem if h1 <= signed_max(ty) => {
            let rem = const_val(state, b).map(|c| extend(c, ty, true).unsigned_abs()).filter(|&d| d > 0).map_or(w, |d| d - 1);
            (0, h1.min(rem))
        }
        Binary::Sdiv if h1 <= signed_max(ty) => {
            let (l2, h2) = u_bounds(func, state, b, depth + 1);
            if h2 <= signed_max(ty) && l2 >= 1 { (l1 / h2, h1 / l2) } else { (0, w) }
        }
        _ => (0, w),
    }
}

fn bin_u_bounds(func: &Function, state: &RangeState, ty: Type, bin: Binary, (a, b): (Value, Value), depth: u8) -> (u64, u64) {
    if matches!(bin, Binary::Ushr | Binary::Sshr | Binary::Ishl | Binary::Urem | Binary::Udiv | Binary::Srem | Binary::Sdiv) {
        return shift_div_u_bounds(func, state, ty, bin, (a, b), depth);
    }
    let w = ty.unsigned_max();
    let ((l1, h1), (l2, h2)) = (u_bounds(func, state, a, depth + 1), u_bounds(func, state, b, depth + 1));
    let sm = signed_max(ty);
    match bin {
        Binary::Iadd => h1.checked_add(h2).filter(|&h| h <= w).map_or((0, w), |h| (l1 + l2, h)),
        Binary::Isub if l1 >= h2 => (l1 - h2, h1 - l2),
        Binary::Imul => h1.checked_mul(h2).filter(|&h| h <= w).map_or((0, w), |h| (l1 * l2, h)),
        Binary::Band => (0, h1.min(h2)),
        Binary::Umin => (l1.min(l2), h1.min(h2)),
        Binary::Umax => (l1.max(l2), h1.max(h2)),
        Binary::Smin if h1 <= sm && h2 <= sm => (l1.min(l2), h1.min(h2)),
        Binary::Smax => smax_u_bounds(sm, w, (l1, h1), (l2, h2)),
        _ => (0, w),
    }
}

fn op_u_bounds(func: &Function, state: &RangeState, v: Value, depth: u8) -> (u64, u64) {
    let ty = func.ty(v);
    let w = ty.unsigned_max();
    let Some(inst) = def_inst(state, v) else { return (0, w) };
    let sub = |u: Value| u_bounds(func, state, u, depth + 1);
    match func.insts[inst.index()].op {
        Op::Unary(Unary::Bnot, a) if func.ty(a).is_small_int() => {
            let (l, h) = sub(a);
            (w - h.min(w), w - l.min(w))
        }
        Op::Unary(Unary::Clz | Unary::Ctz | Unary::Popcnt, _) => (0, u64::from(ty.bits())),
        Op::Select(_, a, b) => {
            let ((l1, h1), (l2, h2)) = (sub(a), sub(b));
            (l1.min(l2), h1.max(h2))
        }
        Op::Convert(Convert::Uextend, a) if func.ty(a).is_small_int() => {
            let (l, h) = sub(a);
            let aw = func.ty(a).unsigned_max();
            (l.min(aw), h.min(aw))
        }
        Op::Convert(Convert::Sextend, a) if func.ty(a).is_small_int() => {
            let (l, h) = sub(a);
            if h <= signed_max(func.ty(a)) { (l, h) } else { (0, w) }
        }
        Op::Convert(Convert::Ireduce, a) if func.ty(a).is_small_int() => {
            let (l, h) = sub(a);
            if h <= w { (l, h) } else { (0, w) }
        }
        Op::Binary(bin, a, b) => bin_u_bounds(func, state, ty, bin, (a, b), depth),
        _ => (0, w),
    }
}

fn u_bounds(func: &Function, state: &RangeState, v: Value, depth: u8) -> (u64, u64) {
    let ty = func.ty(v);
    let w = ty.unsigned_max();
    if !ty.is_small_int() {
        return (0, w);
    }
    let (z, o) = known_bits(func, &|u| def_inst(state, u), &|u| const_val(state, u), v, depth);
    let (bmin, bmax) = (o & w, !z & w);
    if depth >= 6 || bmin == bmax {
        return (bmin, bmax);
    }
    let (omin, omax) = op_u_bounds(func, state, v, depth);
    (bmin.max(omin), bmax.min(omax))
}

fn sext_candidate(func: &Function, inst: Inst) -> Option<(Value, Value, u64)> {
    let data = &func.insts[inst.index()];
    let (Op::Convert(Convert::Sextend, inner), Some(res)) = (&data.op, data.result) else {
        return None;
    };
    let (in_ty, out_ty) = (func.ty(*inner), func.ty(res));
    if !in_ty.is_small_int() || in_ty == Type::I64 || !out_ty.is_small_int() || out_ty.bits() <= in_ty.bits() {
        return None;
    }
    Some((res, *inner, 1_u64 << (in_ty.bits() - 1)))
}

fn try_fold_sext_to_uext(func: &mut Function, inst: Inst, state: &RangeState) -> bool {
    let Some((_, inner, sign_bit)) = sext_candidate(func, inst) else {
        return false;
    };
    if u_bounds(func, state, inner, 0).1 >= sign_bit {
        return false;
    }
    func.insts[inst.index()].op = Op::Convert(Convert::Uextend, inner);
    true
}

#[derive(Clone, Copy)]
struct UseSite {
    block: Block,
    pos: usize,
    inst: Inst,
    branch_target: Option<Block>,
}

struct GuardCtx {
    cfg: Cfg,
    def_blocks: Vec<Option<Block>>,
    uses: Vec<Vec<UseSite>>,
    has_traps: Vec<bool>,
}

impl GuardCtx {
    fn new(func: &Function) -> Self {
        let cfg = Cfg::new(func);
        let def_blocks = func.def_blocks();
        let mut uses = vec![Vec::new(); func.types.len()];
        let mut has_traps = vec![false; func.blocks.len()];
        for (b_idx, block_data) in func.blocks.iter().enumerate() {
            if !cfg.reachable[b_idx] {
                continue;
            }
            let block = Block(b_idx as u32);
            for (pos, &inst) in block_data.insts.iter().enumerate() {
                let op = &func.insts[inst.index()].op;
                if matches!(op, Op::Trapz(_) | Op::Assume(_)) {
                    has_traps[b_idx] = true;
                }
                record_uses(op, block, pos, inst, &mut uses);
            }
        }
        Self { cfg, def_blocks, uses, has_traps }
    }
}

fn record_uses(op: &Op, block: Block, pos: usize, inst: Inst, uses: &mut [Vec<UseSite>]) {
    if let Op::Brif(cond, tb, ta, eb, ea) = op {
        uses[cond.index()].push(UseSite { block, pos, inst, branch_target: None });
        for v in ta {
            uses[v.index()].push(UseSite { block, pos, inst, branch_target: Some(*tb) });
        }
        for v in ea {
            uses[v.index()].push(UseSite { block, pos, inst, branch_target: Some(*eb) });
        }
        return;
    }
    op.for_operands(|v| uses[v.index()].push(UseSite { block, pos, inst, branch_target: None }));
}

fn guard_proves_cmp(cc: IntCC, c_u: u64, sign_bit: u64) -> bool {
    match cc {
        IntCC::UnsignedLessThan => c_u <= sign_bit,
        IntCC::UnsignedLessThanOrEqual | IntCC::Equal => c_u < sign_bit,
        _ => false,
    }
}

fn cmp_operand_const(func: &Function, state: &RangeState, inst: Inst, res: Value) -> Option<(IntCC, u64)> {
    let Op::Icmp(cc, l, r) = func.insts[inst.index()].op else {
        return None;
    };
    let (cc, other) = if l == res {
        (cc, r)
    } else if r == res {
        (crate::cse::flip_int_cc(cc), l)
    } else {
        return None;
    };
    if other == res {
        return None;
    }
    let c = const_val(state, other)?;
    Some((cc, extend(c, func.ty(res), false) as u64))
}

fn is_sign_inv_cmp(func: &Function, state: &RangeState, inst: Inst, res: Value, sign_bit: u64) -> bool {
    cmp_operand_const(func, state, inst, res)
        .is_some_and(|(cc, c_u)| guard_proves_cmp(cc, c_u, sign_bit) || guard_proves_cmp(cc.inverse(), c_u, sign_bit))
}

fn bool_xor_inner(func: &Function, state: &RangeState, a: Value, b: Value) -> Option<Value> {
    [(a, b), (b, a)]
        .into_iter()
        .find_map(|(inner, one)| (const_val(state, one) == Some(1) && u_bounds(func, state, inner, 0).1 <= 1).then_some(inner))
}

fn guard_proves(func: &Function, state: &RangeState, (cond, taken, depth): (Value, bool, u8), res: Value, sign_bit: u64) -> bool {
    if cond == res {
        return !taken;
    }
    if depth >= 4 {
        return false;
    }
    let Some(inst) = def_inst(state, cond) else {
        return false;
    };
    match func.insts[inst.index()].op {
        Op::Icmp(..) => cmp_operand_const(func, state, inst, res).is_some_and(|(cc, c_u)| {
            let eff = if taken { cc } else { cc.inverse() };
            guard_proves_cmp(eff, c_u, sign_bit)
        }),
        Op::Convert(Convert::Uextend, inner) if func.ty(inner).is_small_int() => {
            guard_proves(func, state, (inner, taken, depth + 1), res, sign_bit)
        }
        Op::Binary(Binary::Bxor, a, b) => {
            bool_xor_inner(func, state, a, b).is_some_and(|inner| guard_proves(func, state, (inner, !taken, depth + 1), res, sign_bit))
        }
        Op::Binary(Binary::Band, a, b) if taken => {
            guard_proves(func, state, (a, true, depth + 1), res, sign_bit) || guard_proves(func, state, (b, true, depth + 1), res, sign_bit)
        }
        Op::Binary(Binary::Bor, a, b) if !taken => {
            guard_proves(func, state, (a, false, depth + 1), res, sign_bit) || guard_proves(func, state, (b, false, depth + 1), res, sign_bit)
        }
        _ => false,
    }
}

fn single_pred_guard(func: &Function, state: &RangeState, cfg: &Cfg, block: Block, (res, sign_bit): (Value, u64)) -> bool {
    let Some(pred) = cfg.single_pred(block) else {
        return false;
    };
    if pred == block || !cfg.reachable[pred.index()] {
        return false;
    }
    let Some(&term) = func.blocks[pred.index()].insts.last() else {
        return false;
    };
    let Op::Brif(cond, tb, _, eb, _) = func.insts[term.index()].op else {
        return false;
    };
    tb != eb && guard_proves(func, state, (cond, block == tb, 0), res, sign_bit)
}

fn block_has_trap_guard(func: &Function, state: &RangeState, ctx: &GuardCtx, (block, limit): (Block, usize), (res, sign_bit): (Value, u64)) -> bool {
    if !ctx.has_traps[block.index()] {
        return false;
    }
    func.blocks[block.index()].insts[..limit].iter().any(|&inst| {
        matches!(func.insts[inst.index()].op, Op::Trapz(cond) | Op::Assume(cond) if guard_proves(func, state, (cond, true, 0), res, sign_bit))
    })
}

fn brif_edge_guarded(func: &Function, state: &RangeState, site: UseSite, (res, sign_bit): (Value, u64)) -> bool {
    let Some(edge_dest) = site.branch_target else {
        return false;
    };
    let Op::Brif(cond, tb, _, eb, _) = func.insts[site.inst.index()].op else {
        return false;
    };
    tb != eb && guard_proves(func, state, (cond, edge_dest == tb, 0), res, sign_bit)
}

fn is_guarded_use(func: &Function, state: &RangeState, ctx: &GuardCtx, site: UseSite, (def_block, target): (Block, (Value, u64))) -> bool {
    if brif_edge_guarded(func, state, site, target)
        || block_has_trap_guard(func, state, ctx, (site.block, site.pos), target)
        || (site.block != def_block && single_pred_guard(func, state, &ctx.cfg, site.block, target))
    {
        return true;
    }
    let mut anc = site.block;
    while anc != def_block {
        let Some(parent) = ctx.cfg.idom[anc.index()] else {
            return false;
        };
        if parent == anc {
            return false;
        }
        anc = parent;
        if anc != def_block && single_pred_guard(func, state, &ctx.cfg, anc, target) {
            return true;
        }
        let len = func.blocks[anc.index()].insts.len();
        if block_has_trap_guard(func, state, ctx, (anc, len), target) {
            return true;
        }
    }
    false
}

fn can_narrow_guarded(func: &Function, state: &RangeState, ctx: &GuardCtx, (res, sign_bit): (Value, u64)) -> bool {
    let Some(def_block) = ctx.def_blocks[res.index()] else {
        return false;
    };
    ctx.uses[res.index()]
        .iter()
        .all(|&site| is_sign_inv_cmp(func, state, site.inst, res, sign_bit) || is_guarded_use(func, state, ctx, site, (def_block, (res, sign_bit))))
}

fn fold_guarded_sext(func: &mut Function, state: &mut RangeState) -> bool {
    let has_sext = func.blocks.iter().flat_map(|b| &b.insts).any(|&inst| sext_candidate(func, inst).is_some());
    if !has_sext {
        return false;
    }
    state.defs = defs(func);
    let ctx = GuardCtx::new(func);
    let mut changed = false;
    for b_idx in 0..func.blocks.len() {
        if !ctx.cfg.reachable[b_idx] {
            continue;
        }
        for pos in 0..func.blocks[b_idx].insts.len() {
            let inst = func.blocks[b_idx].insts[pos];
            let Some((res, inner, sign_bit)) = sext_candidate(func, inst) else {
                continue;
            };
            if can_narrow_guarded(func, state, &ctx, (res, sign_bit)) {
                func.insts[inst.index()].op = Op::Convert(Convert::Uextend, inner);
                changed = true;
            }
        }
    }
    changed
}

fn has_candidate(func: &Function) -> bool {
    let mut is_cmp = vec![false; func.types.len()];
    for data in func.blocks.iter().flat_map(|b| &b.insts).map(|i| &func.insts[i.index()]) {
        if let Op::Convert(Convert::Sextend, inner) = data.op
            && func.ty(inner).is_small_int()
            && func.ty(inner) != Type::I64
        {
            return true;
        }
        if let (&Op::Icmp(_, l, _), Some(res)) = (&data.op, data.result) && func.ty(l).is_small_int() {
            is_cmp[res.index()] = true;
        }
    }
    func.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|i| matches!(func.insts[i.index()].op, Op::Binary(Binary::Band | Binary::Bor, a, b) if is_cmp[a.index()] && is_cmp[b.index()]))
}

fn try_fold_cmp_pair(func: &mut Function, inst: Inst, state: &mut RangeState, next: &mut Vec<Inst>) -> bool {
    let Op::Binary(op @ (Binary::Band | Binary::Bor), a, b) = func.insts[inst.index()].op else {
        return false;
    };
    if state.uses[a.index()] != 1 || state.uses[b.index()] != 1 {
        return false;
    }
    let (Some((cc1, x1, c1)), Some((cc2, x2, c2))) = (cmp_const(func, state, a), cmp_const(func, state, b)) else {
        return false;
    };
    if x1 != x2 {
        return false;
    }
    let x_ty = func.ty(x1);
    let Some((pre, out_cc, out_imm)) = fuse_cmp_pair(op == Binary::Band, x_ty, (cc1, c1), (cc2, c2)) else {
        return false;
    };
    let lhs = if let Some((pre_op, pre_imm)) = pre {
        let (c_pre, r_pre) = (func.value(x_ty), func.value(x_ty));
        state.consts.extend([Some(extend(pre_imm, x_ty, false)), None]);
        next.push(func.add(Op::Const(pre_imm), Some(c_pre)));
        next.push(func.add(Op::Binary(pre_op, x1, c_pre), Some(r_pre)));
        r_pre
    } else {
        x1
    };
    let c_out = func.value(x_ty);
    state.consts.push(Some(extend(out_imm, x_ty, false)));
    next.push(func.add(Op::Const(out_imm), Some(c_out)));
    func.insts[inst.index()].op = Op::Icmp(out_cc, lhs, c_out);
    true
}

/// Fuses each `and` or `or` of two single-use compares of one value against constants or masks into one compare, and narrows provably non-negative `Sextend` to `Uextend`.
pub(super) fn fold_range_checks(func: &mut Function) -> bool {
    if !has_candidate(func) {
        return false;
    }
    let mut consts = vec![None; func.types.len()];
    for data in &func.insts {
        if let (&Op::Const(imm), Some(res)) = (&data.op, data.result) && func.ty(res).is_small_int() {
            consts[res.index()] = Some(extend(imm, func.ty(res), false));
        }
    }
    let mut state = RangeState { defs: defs(func), uses: func.use_counts(), consts };
    let mut changed = false;
    for b_idx in 0..func.blocks.len() {
        let orig = std::mem::take(&mut func.blocks[b_idx].insts);
        let mut next = Vec::with_capacity(orig.len());
        for inst in orig {
            if try_fold_cmp_pair(func, inst, &mut state, &mut next) || try_fold_mask_zero(func, inst, &mut state, &mut next) || try_fold_sext_to_uext(func, inst, &state) {
                changed = true;
            }
            next.push(inst);
        }
        func.blocks[b_idx].insts = next;
    }
    changed |= fold_guarded_sext(func, &mut state);
    changed
}
