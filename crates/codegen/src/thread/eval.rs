use crate::facts::{Fact, Ranges, decide, deref_ptr, truth};
use crate::fold::{evaluate_wrapped, extend};
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Value};

#[derive(Clone, Copy)]
pub(super) enum EdgeSeed {
    None,
    Brif(Value, bool),
    Equal(Value, bool),
    Const(Value, i64),
}

pub(super) struct ThreadCx<'a> {
    pub(super) defs: &'a [Option<Inst>],
    pub(super) consts: &'a mut Vec<Option<i64>>,
    pub(super) uses: &'a mut Vec<u64>,
    pub(super) ranges: &'a Ranges<'a>,
    pub(super) ranges_len: usize,
    pub(super) pinned: &'a [bool],
    pub(super) idom: &'a [Option<Block>],
}

#[derive(Default)]
pub(super) struct EvalState {
    pub(super) map: Vec<(Value, Value)>,
    pub(super) local_ops: Vec<(Value, Op)>,
    pub(super) local_consts: Vec<(Value, i64)>,
    pub(super) local_facts: Vec<(Value, Fact)>,
    pub(super) edge_cmps: Vec<(Value, Type, IntCC, i64)>,
}

impl EvalState {
    pub(super) fn seeded(&self) -> Vec<Value> {
        self.local_consts
            .iter()
            .map(|&(value, _)| value)
            .chain(self.edge_cmps.iter().map(|&(value, ..)| value))
            .chain(self.local_facts.iter().map(|&(value, _)| value))
            .collect()
    }

    /// Records non-null facts for `I64` pointers dereferenced in `pred` and not redefined in `disp`.
    pub(super) fn seed_pred(
        &mut self,
        func: &Function,
        cx: &ThreadCx<'_>,
        pred: Block,
        disp: Block,
    ) {
        for &inst in &func.blocks[pred.index()].insts {
            let Some(addr) = deref_ptr(&func.insts[inst.index()].op) else {
                continue;
            };
            if func.ty(addr) != Type::I64
                || super::redefines(func, disp, &[addr])
                || fact_of(&self.local_facts, cx.ranges, cx.ranges_len, addr)
                    .is_some_and(|f| f.lo >= 1)
            {
                continue;
            }
            self.local_facts.push((addr, Fact::range(1, u64::MAX)));
            if self.local_facts.len() >= 16 {
                break;
            }
        }
    }

    /// Records what the edge `seed` into `disp` proves: the branch condition, the values an equal compare makes one value, or a constant.
    pub(super) fn seed(&mut self, func: &Function, cx: &ThreadCx<'_>, disp: Block, seed: EdgeSeed) {
        match seed {
            EdgeSeed::None => {}
            EdgeSeed::Brif(cond, taken) => seed_edge(
                func,
                cx,
                cond,
                taken,
                0,
                (&mut self.local_consts, &mut self.edge_cmps),
            ),
            EdgeSeed::Equal(cond, taken) => {
                seed_edge(
                    func,
                    cx,
                    cond,
                    taken,
                    0,
                    (&mut self.local_consts, &mut self.edge_cmps),
                );
                if let Some((x, y)) = edge_equal(func, cx, disp, cond, taken) {
                    self.rename(cx, y, x);
                }
            }
            EdgeSeed::Const(val, imm) => self.local_consts.push((val, imm)),
        }
    }

    fn rename(&mut self, cx: &ThreadCx<'_>, from: Value, to: Value) {
        self.map.retain(|&(k, _)| k != from);
        self.map.push((from, to));
        let fact = fact_of(&self.local_facts, cx.ranges, cx.ranges_len, to);
        self.local_facts.retain(|&(k, _)| k != from);
        if let Some(f) = fact {
            self.local_facts.push((from, f));
        }
    }

    fn alias(&mut self, cx: &ThreadCx<'_>, res: Value, to: Value) {
        self.rename(cx, res, to);
        if let Some(imm) = lookup_const(&self.local_consts, cx.consts, to) {
            self.local_consts.push((res, imm));
        }
    }
}

pub(super) fn bind(cx: &ThreadCx<'_>, state: &mut EvalState, params: &[Value], args: &[Value]) {
    for (&param, &arg) in params.iter().zip(args) {
        state.rename(cx, param, arg);
    }
}

pub(super) fn step_inst(
    func: &Function,
    cx: &ThreadCx<'_>,
    pred: Option<Block>,
    inst: Inst,
    state: &mut EvalState,
) -> Option<()> {
    let data = &func.insts[inst.index()];
    let res = data.result?;
    if !data.op.is_speculable() {
        let p = pred?;
        let mut op = data.op.clone();
        op.map_operands(|v| lookup(&state.map, v));
        let imm = super::stored::pred_load_const(func, cx, state, p, &op, func.ty(res))?;
        state.local_consts.push((res, imm));
        state.local_ops.push((res, Op::Const(imm)));
        state.map.retain(|&(k, _)| k != res);
        return Some(());
    }
    let mut op = data.op.clone();
    op.map_operands(|v| lookup(&state.map, v));
    if let Some(imm) = eval_op(func, &op, func.ty(res), &|v| {
        lookup_const(&state.local_consts, cx.consts, v)
    })
    .or_else(|| eval_cmp_facts(func, cx, &op, state))
    {
        state.local_consts.push((res, imm));
    } else if let Op::Select(c, x, y) = op
        && let Some(taken) = eval_cond_truth(func, cx, state, c)
    {
        state.alias(cx, res, if taken { x } else { y });
        return Some(());
    } else if let Some(to) = propagate_pure_op(func, cx, res, &op, state) {
        state.alias(cx, res, to);
        return Some(());
    }
    state.local_ops.push((res, op));
    state.map.retain(|&(k, _)| k != res);
    Some(())
}

pub(super) fn lookup(map: &[(Value, Value)], value: Value) -> Value {
    map.iter()
        .find(|&&(k, _)| k == value)
        .map_or(value, |&(_, v)| v)
}

pub(super) fn lookup_const(
    local: &[(Value, i64)],
    global: &[Option<i64>],
    value: Value,
) -> Option<i64> {
    local
        .iter()
        .find(|&&(k, _)| k == value)
        .map(|&(_, imm)| imm)
        .or_else(|| global.get(value.index()).copied().flatten())
}

pub(super) fn fact_of(
    local_facts: &[(Value, Fact)],
    ranges: &Ranges<'_>,
    ranges_len: usize,
    value: Value,
) -> Option<Fact> {
    local_facts
        .iter()
        .rev()
        .find(|&&(k, _)| k == value)
        .map(|(_, f)| f.clone())
        .or_else(|| (value.index() < ranges_len).then(|| ranges.of(value)))
}

fn bounds_of_cmp(ty: Type, cc: IntCC, imm: i64) -> Option<(bool, i128, i128)> {
    if !cc.is_signed() {
        let (u_max, u) = (
            i128::from(ty.unsigned_max()),
            i128::from(extend(imm, ty, false) as u64),
        );
        let (lo, hi) = match cc {
            IntCC::Equal => (u, u),
            IntCC::NotEqual if u == 0 && u_max > 0 => (1, u_max),
            IntCC::NotEqual if u == u_max && u_max > 0 => (0, u_max - 1),
            IntCC::UnsignedLessThan if u > 0 => (0, u - 1),
            IntCC::UnsignedLessThanOrEqual => (0, u),
            IntCC::UnsignedGreaterThan if u < u_max => (u + 1, u_max),
            IntCC::UnsignedGreaterThanOrEqual => (u, u_max),
            _ => return None,
        };
        return Some((false, lo, hi));
    }
    let (s_min, s_max) = if ty == Type::I64 {
        (i128::from(i64::MIN), i128::from(i64::MAX))
    } else {
        let h = 1_i128 << (ty.bits() - 1);
        (-h, h - 1)
    };
    let s = i128::from(extend(imm, ty, true));
    let (lo, hi) = match cc {
        IntCC::SignedLessThan if s > s_min => (s_min, s - 1),
        IntCC::SignedLessThanOrEqual => (s_min, s),
        IntCC::SignedGreaterThan if s < s_max => (s + 1, s_max),
        IntCC::SignedGreaterThanOrEqual => (s, s_max),
        _ => return None,
    };
    Some((true, lo, hi))
}

pub(super) fn cmp_implies(ty: Type, cc1: IntCC, imm1: i64, cc2: IntCC, imm2: i64) -> Option<bool> {
    if cc1 == IntCC::NotEqual
        && matches!(cc2, IntCC::Equal | IntCC::NotEqual)
        && extend(imm1, ty, false) == extend(imm2, ty, false)
    {
        return Some(cc2 == IntCC::NotEqual);
    }
    let (signed, lo, hi) = bounds_of_cmp(ty, cc1, imm1)?;
    let c2 = if signed {
        if !cc2.is_signed() && !matches!(cc2, IntCC::Equal | IntCC::NotEqual) {
            return None;
        }
        i128::from(extend(imm2, ty, true))
    } else {
        let half = i128::from(ty.unsigned_max() >> 1);
        let c = i128::from(extend(imm2, ty, false) as u64);
        if cc2.is_signed() && (hi > half || c > half) {
            return None;
        }
        c
    };
    match cc2 {
        IntCC::Equal => (lo == hi)
            .then_some(lo == c2)
            .or_else(|| (c2 < lo || c2 > hi).then_some(false)),
        IntCC::NotEqual => (lo == hi)
            .then_some(lo != c2)
            .or_else(|| (c2 < lo || c2 > hi).then_some(true)),
        IntCC::UnsignedLessThan | IntCC::SignedLessThan => {
            if hi < c2 {
                Some(true)
            } else if lo >= c2 {
                Some(false)
            } else {
                None
            }
        }
        IntCC::UnsignedLessThanOrEqual | IntCC::SignedLessThanOrEqual => {
            if hi <= c2 {
                Some(true)
            } else if lo > c2 {
                Some(false)
            } else {
                None
            }
        }
        IntCC::UnsignedGreaterThan | IntCC::SignedGreaterThan => {
            if lo > c2 {
                Some(true)
            } else if hi <= c2 {
                Some(false)
            } else {
                None
            }
        }
        IntCC::UnsignedGreaterThanOrEqual | IntCC::SignedGreaterThanOrEqual => {
            if lo >= c2 {
                Some(true)
            } else if hi < c2 {
                Some(false)
            } else {
                None
            }
        }
    }
}

pub(super) fn record_edge_cmp(
    lhs: Value,
    ty: Type,
    ord: IntCC,
    imm: i64,
    local_consts: &mut Vec<(Value, i64)>,
    edge_cmps: &mut Vec<(Value, Type, IntCC, i64)>,
) {
    if ord == IntCC::Equal {
        local_consts.push((lhs, imm));
        return;
    }
    edge_cmps.push((lhs, ty, ord, imm));
    if !ord.is_signed()
        && let Some((_, mut lo, mut hi)) = bounds_of_cmp(ty, ord, imm)
    {
        for &(v, v_ty, c1, i1) in edge_cmps.iter() {
            if v == lhs
                && v_ty == ty
                && !c1.is_signed()
                && let Some((_, l2, h2)) = bounds_of_cmp(ty, c1, i1)
            {
                lo = lo.max(l2);
                hi = hi.min(h2);
            }
        }
        if lo == hi {
            local_consts.push((lhs, extend(lo as i64, ty, true)));
        }
    }
}

/// The two sides of the integer compare `cond` that the edge taken as `taken` proves equal, when `disp` neither defines nor binds either side.
fn edge_equal(
    func: &Function,
    cx: &ThreadCx<'_>,
    disp: Block,
    cond: Value,
    taken: bool,
) -> Option<(Value, Value)> {
    let inst = cx.defs.get(cond.index()).copied().flatten()?;
    let (cc, x, y) = crate::algebra::branch_cmp(&func.insts[inst.index()].op)?;
    let equal = matches!((cc, taken), (IntCC::Equal, true) | (IntCC::NotEqual, false));
    let data = &func.blocks[disp.index()];
    let outside = |v: Value| {
        !data.params.contains(&v)
            && cx
                .defs
                .get(v.index())
                .copied()
                .flatten()
                .is_none_or(|def| !data.insts.contains(&def))
    };
    (equal && x != y && func.ty(x).is_small_int() && outside(x) && outside(y)).then_some((x, y))
}

pub(super) fn seed_edge(
    func: &Function,
    cx: &ThreadCx<'_>,
    cond: Value,
    taken: bool,
    depth: u8,
    (local_consts, edge_cmps): (&mut Vec<(Value, i64)>, &mut Vec<(Value, Type, IntCC, i64)>),
) {
    if depth > 4 {
        return;
    }
    if !taken {
        local_consts.push((cond, 0));
    } else if is_bool_val(func, cx.defs, cx.consts, cond) {
        local_consts.push((cond, 1));
    } else if func.ty(cond).is_small_int() {
        edge_cmps.push((cond, func.ty(cond), IntCC::NotEqual, 0));
    }
    let Some(inst) = cx.defs.get(cond.index()).copied().flatten() else {
        return;
    };
    let op = &func.insts[inst.index()].op;
    match *op {
        Op::Binary(Binary::Band, a, b) if taken => {
            seed_edge(func, cx, a, true, depth + 1, (local_consts, edge_cmps));
            seed_edge(func, cx, b, true, depth + 1, (local_consts, edge_cmps));
        }
        Op::Binary(Binary::Bor, a, b) if !taken => {
            seed_edge(func, cx, a, false, depth + 1, (local_consts, edge_cmps));
            seed_edge(func, cx, b, false, depth + 1, (local_consts, edge_cmps));
        }
        _ => {}
    }
    if let Some((cc, x, y)) = crate::algebra::branch_cmp(op)
        && func.ty(x).is_small_int()
    {
        let eff = if taken { cc } else { cc.inverse() };
        let (lhs, ord, imm) = match (
            lookup_const(local_consts, cx.consts, x),
            lookup_const(local_consts, cx.consts, y),
        ) {
            (None, Some(k)) => (x, eff, k),
            (Some(k), None) => (y, eff.swapped(), k),
            _ => return,
        };
        let ty = func.ty(lhs);
        record_edge_cmp(lhs, ty, ord, imm, local_consts, edge_cmps);
        if let Some(lhs_inst) = cx.defs.get(lhs.index()).copied().flatten()
            && let Op::Convert(kind @ (Convert::Uextend | Convert::Sextend), inner) =
                func.insts[lhs_inst.index()].op
        {
            let sub_ty = func.ty(inner);
            if sub_ty.is_small_int() && sub_ty.bits() < ty.bits() {
                if kind == Convert::Uextend && !ord.is_signed() {
                    let u = extend(imm, ty, false) as u64;
                    if u <= sub_ty.unsigned_max() {
                        record_edge_cmp(
                            inner,
                            sub_ty,
                            ord,
                            extend(u as i64, sub_ty, true),
                            local_consts,
                            edge_cmps,
                        );
                    }
                } else if kind == Convert::Sextend
                    && (ord.is_signed() || matches!(ord, IntCC::Equal | IntCC::NotEqual))
                {
                    let (sc, half) = (extend(imm, ty, true), 1_i64 << (sub_ty.bits() - 1));
                    if (-half..half).contains(&sc) {
                        record_edge_cmp(
                            inner,
                            sub_ty,
                            ord,
                            extend(sc, sub_ty, true),
                            local_consts,
                            edge_cmps,
                        );
                    }
                }
            }
        }
    }
}

pub(super) fn eval_op(
    func: &Function,
    op: &Op,
    ty: Type,
    known: &dyn Fn(Value) -> Option<i64>,
) -> Option<i64> {
    if let Some(imm) = evaluate_wrapped(func, op, ty, known) {
        return Some(imm);
    }
    if !ty.is_small_int() {
        return None;
    }
    let is_zero = |v: Value| known(v).is_some_and(|imm| extend(imm, ty, false) == 0);
    match *op {
        Op::Convert(Convert::Bitcast, x)
            if func.ty(x).is_small_int() && func.ty(x).bits() == ty.bits() =>
        {
            known(x).map(|imm| extend(imm, ty, true))
        }
        Op::Select(c, x, y) => {
            if x == y {
                return known(x).map(|imm| extend(imm, ty, true));
            }
            let chosen = if extend(known(c)?, func.ty(c), false) != 0 {
                x
            } else {
                y
            };
            known(chosen).map(|imm| extend(imm, ty, true))
        }
        Op::Binary(Binary::Imul | Binary::Umulhi | Binary::Smulhi | Binary::Band, x, y)
            if is_zero(x) || is_zero(y) =>
        {
            Some(0)
        }
        Op::Binary(Binary::Isub | Binary::Bxor, x, y) if x == y => Some(0),
        Op::Icmp(cc, x, y) if x == y && func.ty(x).is_small_int() => Some(i64::from(matches!(
            cc,
            IntCC::Equal
                | IntCC::SignedGreaterThanOrEqual
                | IntCC::SignedLessThanOrEqual
                | IntCC::UnsignedGreaterThanOrEqual
                | IntCC::UnsignedLessThanOrEqual
        ))),
        _ => None,
    }
}

pub(super) fn eval_cond_truth(
    func: &Function,
    cx: &ThreadCx<'_>,
    state: &EvalState,
    cond_val: Value,
) -> Option<bool> {
    if let Some(c) = lookup_const(&state.local_consts, cx.consts, cond_val) {
        return Some(extend(c, func.ty(cond_val), false) != 0);
    }
    if let Some(t) =
        fact_of(&state.local_facts, cx.ranges, cx.ranges_len, cond_val).and_then(|f| truth(&f))
    {
        return Some(t);
    }
    state.edge_cmps.iter().find_map(|&(v, v_ty, cc1, imm1)| {
        (v == cond_val)
            .then(|| cmp_implies(v_ty, cc1, imm1, IntCC::NotEqual, 0))
            .flatten()
    })
}

/// Whether `v` is 0 or 1 because a compare, a widen or narrow of one, or an and with 1 defines it.
pub(crate) fn is_bool_val(
    func: &Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    v: Value,
) -> bool {
    let Some(inst) = defs.get(v.index()).copied().flatten() else {
        return false;
    };
    let is_one = |u: Value| consts.get(u.index()).copied().flatten() == Some(1);
    match func.insts[inst.index()].op {
        Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => true,
        Op::Convert(Convert::Uextend | Convert::Ireduce, inner) => {
            is_bool_val(func, defs, consts, inner)
        }
        Op::Binary(Binary::Band, a, b) => {
            is_one(a)
                || is_one(b)
                || is_bool_val(func, defs, consts, a)
                || is_bool_val(func, defs, consts, b)
        }
        Op::Binary(Binary::Bxor, a, b) => {
            (is_one(b) && is_bool_val(func, defs, consts, a))
                || (is_one(a) && is_bool_val(func, defs, consts, b))
        }
        _ => false,
    }
}

fn eval_cmp_facts(func: &Function, cx: &ThreadCx<'_>, op: &Op, state: &EvalState) -> Option<i64> {
    let &Op::Icmp(cc, x, y) = op else { return None };
    let ty = func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    if let (Some(fx), Some(fy)) = (
        fact_of(&state.local_facts, cx.ranges, cx.ranges_len, x),
        fact_of(&state.local_facts, cx.ranges, cx.ranges_len, y),
    ) && let Some(res) = decide(cc, ty, &fx, &fy)
    {
        return Some(i64::from(res));
    }
    let (lhs, ord, imm2) = match (
        lookup_const(&state.local_consts, cx.consts, x),
        lookup_const(&state.local_consts, cx.consts, y),
    ) {
        (None, Some(k)) => (x, cc, k),
        (Some(k), None) => (y, cc.swapped(), k),
        _ => return None,
    };
    for &(v, v_ty, cc1, imm1) in &state.edge_cmps {
        if v == lhs
            && v_ty == ty
            && let Some(res) = cmp_implies(ty, cc1, imm1, ord, imm2)
        {
            return Some(i64::from(res));
        }
    }
    None
}

fn propagate_pure_op(
    func: &Function,
    cx: &ThreadCx<'_>,
    res: Value,
    op: &Op,
    state: &mut EvalState,
) -> Option<Value> {
    let res_ty = func.ty(res);
    if !res_ty.is_small_int() {
        return None;
    }
    match *op {
        Op::Convert(Convert::Uextend, x)
            if func.ty(x).is_small_int() && res_ty.bits() > func.ty(x).bits() =>
        {
            let x_ty = func.ty(x);
            let derived: Vec<_> = state
                .edge_cmps
                .iter()
                .filter_map(|&(v, v_ty, cc1, imm1)| {
                    (v == x && v_ty == x_ty && !cc1.is_signed())
                        .then(|| (cc1, extend(imm1, x_ty, false)))
                })
                .collect();
            for (cc1, imm1) in derived {
                record_edge_cmp(
                    res,
                    res_ty,
                    cc1,
                    imm1,
                    &mut state.local_consts,
                    &mut state.edge_cmps,
                );
            }
            None
        }
        Op::Convert(Convert::Ireduce, x)
            if func.ty(x).is_small_int() && func.ty(x).bits() > res_ty.bits() =>
        {
            let x_ty = func.ty(x);
            let max_u = res_ty.unsigned_max();
            let fits = state.edge_cmps.iter().any(|&(v, v_ty, cc1, imm1)| {
                v == x
                    && v_ty == x_ty
                    && cmp_implies(
                        x_ty,
                        cc1,
                        imm1,
                        IntCC::UnsignedLessThanOrEqual,
                        max_u as i64,
                    ) == Some(true)
            });
            if fits {
                let derived: Vec<_> = state
                    .edge_cmps
                    .iter()
                    .filter_map(|&(v, v_ty, cc1, imm1)| {
                        let u = extend(imm1, x_ty, false) as u64;
                        (v == x && v_ty == x_ty && !cc1.is_signed() && u <= max_u)
                            .then(|| (cc1, extend(u as i64, res_ty, true)))
                    })
                    .collect();
                for (cc1, imm1) in derived {
                    record_edge_cmp(
                        res,
                        res_ty,
                        cc1,
                        imm1,
                        &mut state.local_consts,
                        &mut state.edge_cmps,
                    );
                }
            }
            None
        }
        Op::Binary(Binary::Band, a, b) => {
            for (x, mv) in [(a, b), (b, a)] {
                let Some(raw) = lookup_const(&state.local_consts, cx.consts, mv) else {
                    continue;
                };
                let u = extend(raw, res_ty, false) as u64;
                if u == res_ty.unsigned_max() {
                    return Some(x);
                }
                if u > 0
                    && (u & (u + 1)) == 0
                    && state.edge_cmps.iter().any(|&(v, v_ty, cc1, imm1)| {
                        v == x
                            && v_ty == res_ty
                            && cmp_implies(res_ty, cc1, imm1, IntCC::UnsignedLessThanOrEqual, raw)
                                == Some(true)
                    })
                {
                    return Some(x);
                }
            }
            None
        }
        _ => None,
    }
}
