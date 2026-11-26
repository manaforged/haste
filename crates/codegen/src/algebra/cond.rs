use crate::fold::{const_offset, extend, known_bits};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::icmp::complementary_eq_pair;
use super::select::{build_ord_select, match_ord_select};
use super::{op_of, peel_uextend};

fn fact<T: Copy + Default>(table: &[T], v: Value) -> T {
    table.get(v.index()).copied().unwrap_or_default()
}

struct BoolFacts {
    compares: Vec<Option<(IntCC, Value, Value)>>,
    consts: Vec<Option<i64>>,
    xors: Vec<Option<(Value, Value)>>,
    uextends: Vec<Option<Value>>,
    bools: Vec<bool>,
}

impl BoolFacts {
    fn collect(func: &Function) -> Self {
        let n = func.types.len();
        let mut f = Self {
            compares: vec![None; n],
            consts: vec![None; n],
            xors: vec![None; n],
            uextends: vec![None; n],
            bools: vec![false; n],
        };
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            f.record_inst(func, *inst);
        }
        f.propagate(func);
        f
    }

    fn record_inst(&mut self, func: &Function, inst: Inst) {
        let d = &func.insts[inst.index()];
        let Some(r) = d.result else { return };
        match d.op {
            Op::Icmp(cc, x, y) if func.ty(r) == Type::I8 => {
                self.compares[r.index()] = Some((cc, x, y));
                self.bools[r.index()] = true;
            }
            Op::Fcmp(..) | Op::Fits(..) if func.ty(r) == Type::I8 => self.bools[r.index()] = true,
            Op::Const(imm) if func.ty(r).is_small_int() => self.consts[r.index()] = Some(imm),
            Op::Binary(Binary::Bxor, x, y) => self.xors[r.index()] = Some((x, y)),
            Op::Convert(Convert::Uextend, inner) => self.uextends[r.index()] = Some(inner),
            _ => {}
        }
    }

    fn propagate(&mut self, func: &Function) {
        for _ in 0..2 {
            for inst in func.blocks.iter().flat_map(|block| &block.insts) {
                let d = &func.insts[inst.index()];
                if let Some(r) = d.result {
                    self.bools[r.index()] |= self.op_is_bool(&d.op);
                }
            }
        }
    }

    fn op_is_bool(&self, op: &Op) -> bool {
        match *op {
            Op::Binary(Binary::Band, x, y) => {
                matches!(self.konst(x), Some(0 | 1))
                    || matches!(self.konst(y), Some(0 | 1))
                    || (self.is_01(x) && self.is_01(y))
            }
            Op::Binary(Binary::Bor | Binary::Bxor, x, y) | Op::Select(_, x, y) => self.is_01(x) && self.is_01(y),
            Op::Convert(Convert::Uextend, inner) => self.is_bool(inner),
            _ => false,
        }
    }

    fn record_compare(&mut self, r: Value, cmp: (IntCC, Value, Value)) {
        let n = self.bools.len().max(r.index() + 1);
        self.compares.resize(n, None);
        self.bools.resize(n, false);
        self.compares[r.index()] = Some(cmp);
        self.bools[r.index()] = true;
    }

    fn compare(&self, v: Value) -> Option<(IntCC, Value, Value)> {
        fact(&self.compares, v)
    }

    fn konst(&self, v: Value) -> Option<i64> {
        fact(&self.consts, v)
    }

    fn xor(&self, v: Value) -> Option<(Value, Value)> {
        fact(&self.xors, v)
    }

    fn is_bool(&self, v: Value) -> bool {
        fact(&self.bools, v)
    }

    fn is_01(&self, v: Value) -> bool {
        self.is_bool(v) || matches!(self.konst(v), Some(0 | 1))
    }

    fn unwrap_uext(&self, v: Value) -> Value {
        peel_uextend(v, |u| fact(&self.uextends, u))
    }

    fn xor_one_src(&self, x: Value, y: Value) -> Option<Value> {
        [(x, y), (y, x)].into_iter().find_map(|(v, k)| (self.konst(k) == Some(1) && self.is_bool(v)).then_some(v))
    }
}

fn negated(op: &Op, facts: &BoolFacts) -> Option<(Value, Option<Value>)> {
    let is = |v: Value, want: i64| facts.konst(v) == Some(want);
    match *op {
        Op::Binary(Binary::Bxor, x, y) | Op::Icmp(IntCC::NotEqual, x, y) if is(y, 1) => Some((x, Some(y))),
        Op::Binary(Binary::Bxor, y, x) | Op::Icmp(IntCC::NotEqual, y, x) if is(y, 1) => Some((x, Some(y))),
        Op::Icmp(IntCC::Equal, x, y) if is(y, 0) => Some((x, None)),
        Op::Icmp(IntCC::Equal, y, x) if is(y, 0) => Some((x, None)),
        Op::Select(c, x, y) if is(x, 0) && is(y, 1) && facts.is_bool(c) => Some((c, Some(y))),
        _ => None,
    }
}

fn zero_cmp_operand(facts: &BoolFacts, types: &[Type], x: Value, y: Value) -> Option<(Value, bool)> {
    if facts.konst(y) == Some(0) && types[x.index()].is_small_int() {
        Some((x, false))
    } else if facts.konst(x) == Some(0) && types[y.index()].is_small_int() {
        Some((y, true))
    } else {
        None
    }
}

fn invert_brif(op: &mut Op, facts: &BoolFacts, types: &[Type]) -> bool {
    let Op::Brif(cond, then, then_args, otherwise, else_args) = op else { return false };
    if let Some((x, y)) = facts.xor(*cond)
        && let Some(inner) = facts.xor_one_src(x, y)
    {
        *cond = inner;
        std::mem::swap(then, otherwise);
        std::mem::swap(then_args, else_args);
        return true;
    }
    let Some((cc, x, y)) = facts.compare(*cond) else { return false };
    let Some((val, flipped)) = zero_cmp_operand(facts, types, x, y) else { return false };
    let invert_arms = matches!(
        (flipped, cc),
        (_, IntCC::Equal) | (false, IntCC::UnsignedLessThanOrEqual) | (true, IntCC::UnsignedGreaterThanOrEqual)
    );
    let keep_arms = matches!(
        (flipped, cc),
        (_, IntCC::NotEqual) | (false, IntCC::UnsignedGreaterThan) | (true, IntCC::UnsignedLessThan)
    );
    if !invert_arms && !keep_arms {
        return false;
    }
    *cond = val;
    if invert_arms {
        std::mem::swap(then, otherwise);
        std::mem::swap(then_args, else_args);
    }
    true
}

fn invert_inst(op: &mut Op, res: Option<Value>, facts: &BoolFacts, types: &[Type]) -> bool {
    if let (Some(res), Some((inner, opt_one))) = (res, negated(op, facts)) {
        let res_ty = types[res.index()];
        if res_ty == Type::I8
            && let Some((cc, a, b)) = facts.compare(inner)
        {
            *op = Op::Icmp(cc.inverse(), a, b);
            return true;
        }
        if facts.is_bool(inner)
            && let Some(one) = opt_one
            && types[inner.index()] == res_ty
            && types[one.index()] == res_ty
            && !matches!(*op, Op::Binary(Binary::Bxor, ..))
        {
            *op = Op::Binary(Binary::Bxor, inner, one);
            return true;
        }
    }
    invert_brif(op, facts, types)
}

/// Rewrites negated compares and branches on `x == 0` / `x != 0`, returning whether any instruction changed.
pub(crate) fn invert(func: &mut Function) -> bool {
    let facts = BoolFacts::collect(func);
    let mut changed = false;
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let res = func.insts[inst.index()].result;
        changed |= invert_inst(&mut func.insts[inst.index()].op, res, &facts, &func.types);
    }
    changed | crate::schedule::sink_compares(func)
}

fn xor_ne_pair(func: &Function, facts: &BoolFacts, x: Value, y: Value, allow_const: bool) -> Option<(Value, Value)> {
    if allow_const
        && let Some(pair) = [(x, y), (y, x)].into_iter().find(|&(_, c)| facts.konst(c).is_some())
    {
        return Some(pair);
    }
    if facts.is_bool(x) && facts.is_bool(y) {
        let (a, b) = (facts.unwrap_uext(x), facts.unwrap_uext(y));
        return (func.ty(a) == func.ty(b) && func.ty(a).is_small_int()).then_some((a, b));
    }
    None
}

fn flip_select_xor_one(op: &mut Op, facts: &BoolFacts, x: Value, y: Value) -> Option<Value> {
    let Op::Select(c, t, f) = op else { return None };
    let (lhs, _) = [(x, y), (y, x)].into_iter().find(|&(_, k)| facts.konst(k) == Some(1))?;
    let bool_src = facts.unwrap_uext(lhs);
    if !facts.is_bool(bool_src) {
        return None;
    }
    (*c, *t, *f) = (bool_src, *f, *t);
    Some(bool_src)
}

fn fold_bxor_cond(func: &mut Function, facts: &mut BoolFacts, out: &mut Vec<Inst>, inst: Inst) -> bool {
    let (cond, is_sel) = match func.insts[inst.index()].op {
        Op::Select(c, ..) => (c, true),
        Op::Brif(c, ..) => (c, false),
        _ => return false,
    };
    let inner = facts.unwrap_uext(cond);
    let next = facts.xor(inner).and_then(|(x, y)| {
        if let Some(flipped) = flip_select_xor_one(&mut func.insts[inst.index()].op, facts, x, y) {
            return Some(flipped);
        }
        let (a, b) = xor_ne_pair(func, facts, x, y, !is_sel)
            .filter(|&(a, b)| is_sel || (func.ty(a) == func.ty(b) && func.ty(a).is_small_int()))?;
        let cmp_val = func.value(Type::I8);
        out.push(func.add(Op::Icmp(IntCC::NotEqual, a, b), Some(cmp_val)));
        facts.record_compare(cmp_val, (IntCC::NotEqual, a, b));
        if let Op::Select(c, ..) | Op::Brif(c, ..) = &mut func.insts[inst.index()].op {
            *c = cmp_val;
        }
        Some(cmp_val)
    });
    if next.is_some() || inner == cond {
        return next.is_some();
    }
    if let Op::Select(c, ..) | Op::Brif(c, ..) = &mut func.insts[inst.index()].op {
        *c = inner;
    }
    true
}

fn tag_cond_info(facts: &BoolFacts, func: &Function, param_of: &[Option<(usize, usize)>], b_idx: usize, cond: Value) -> Option<(usize, Value, u64, bool)> {
    let inner = facts.unwrap_uext(cond);
    if let Some((cc @ (IntCC::Equal | IntCC::NotEqual), x, y)) = facts.compare(inner) {
        return [(x, y), (y, x)].into_iter().find_map(|(p, c)| {
            let ((pb, ti), imm) = (param_of.get(p.index()).copied().flatten()?, facts.konst(c)?);
            (pb == b_idx && func.ty(p).is_small_int()).then(|| (ti, p, extend(imm, func.ty(p), false) as u64, cc == IntCC::Equal))
        });
    }
    let (pb, ti) = param_of.get(inner.index()).copied().flatten()?;
    (pb == b_idx && func.ty(inner).is_small_int()).then_some((ti, inner, 0, false))
}

fn push_edge_pair(edges: &mut Vec<(Value, Value)>, b_idx: usize, op: &Op, (ti, vi): (usize, usize)) -> bool {
    let mut push = |args: &[Value]| args.get(ti).zip(args.get(vi)).is_some_and(|(&ta, &va)| {
        edges.push((ta, va));
        true
    });
    match op {
        Op::Jump(t, args) if t.index() == b_idx => push(args),
        Op::Brif(_, tb, ta, eb, ea) => (tb.index() != b_idx || push(ta)) && (eb.index() != b_idx || push(ea)),
        Op::TryCall { normal, handler, .. } => normal.index() != b_idx && handler.index() != b_idx,
        Op::BrTable(_, d, ts) => d.index() != b_idx && ts.iter().all(|t| t.index() != b_idx),
        _ => true,
    }
}

fn incoming_pair_edges(func: &Function, b_idx: usize, self_last: Option<Inst>, tag_idx: usize, val_idx: usize) -> Option<Vec<(Value, Value)>> {
    let mut edges = Vec::new();
    for (pb, block) in func.blocks.iter().enumerate() {
        let Some(last) = (if pb == b_idx { self_last } else { block.insts.last().copied() }) else { continue };
        if !push_edge_pair(&mut edges, b_idx, &func.insts[last.index()].op, (tag_idx, val_idx)) {
            return None;
        }
    }
    (edges.len() >= 2).then_some(edges)
}

fn tag_proven_not_equal(func: &Function, facts: &BoolFacts, defs: &[Option<Inst>], tag_ty: Type, match_u: u64, tag_arg: Value) -> bool {
    let (kz, ko) = known_bits(func, &|v| defs.get(v.index()).copied().flatten(), &|v| facts.konst(v), tag_arg, 0);
    let mask = tag_ty.unsigned_max() as u64;
    ((ko & mask) & !match_u) != 0 || ((kz & mask) & match_u) != 0
}

fn edge_matches_correlated(
    func: &Function,
    facts: &BoolFacts,
    defs: &[Option<Inst>],
    (tag_ty, val_ty, match_u, k_u, val_takes_eq): (Type, Type, u64, u64, bool),
    (tag_arg, val_arg): (Value, Value),
) -> bool {
    let tag_const = facts.konst(tag_arg).map(|c| extend(c, tag_ty, false) as u64);
    let proven = if val_takes_eq {
        tag_const == Some(match_u)
    } else {
        tag_const.is_some_and(|u| u != match_u) || tag_proven_not_equal(func, facts, defs, tag_ty, match_u, tag_arg)
    };
    if proven {
        return true;
    }
    tag_const.is_some_and(|u| (u == match_u) == !val_takes_eq)
        && facts.konst(val_arg).is_some_and(|vc| extend(vc, val_ty, false) as u64 == k_u)
}

fn correlated_select_target(
    func: &Function,
    facts: &BoolFacts,
    param_of: &[Option<(usize, usize)>],
    defs: &mut Option<Vec<Option<Inst>>>,
    (b_idx, self_last, cond, t, f): (usize, Option<Inst>, Value, Value, Value),
) -> Option<Value> {
    if Some(b_idx) == func.entry.map(|e| e.index()) {
        return None;
    }
    let (tag_idx, tag_phi, match_u, eq_means_true) = tag_cond_info(facts, func, param_of, b_idx, cond)?;
    let (val_phi, val_idx, k_val, val_on_true) = [(t, f, true), (f, t, false)].into_iter().find_map(|(v, k, on_t)| {
        let (pb, vi) = param_of.get(v.index()).copied().flatten()?;
        facts.konst(k)?;
        (pb == b_idx && vi != tag_idx).then_some((v, vi, k, on_t))
    })?;
    let val_ty = func.ty(val_phi);
    if !val_ty.is_small_int() {
        return None;
    }
    let k_u = extend(facts.konst(k_val)?, val_ty, false) as u64;
    let edges = incoming_pair_edges(func, b_idx, self_last, tag_idx, val_idx)?;
    let defs_ref = defs.get_or_insert_with(|| crate::fold::defs(func));
    let spec = (func.ty(tag_phi), val_ty, match_u, k_u, val_on_true == eq_means_true);
    edges.iter().all(|&edge| edge_matches_correlated(func, facts, defs_ref, spec, edge)).then_some(val_phi)
}

fn fold_correlated_select(
    func: &Function,
    facts: &BoolFacts,
    param_of: &[Option<(usize, usize)>],
    (defs, subst): (&mut Option<Vec<Option<Inst>>>, &mut Vec<Value>),
    (b_idx, self_last, inst): (usize, Option<Inst>, Inst),
) -> bool {
    let Op::Select(c, t, f) = func.insts[inst.index()].op else { return false };
    let Some(res) = func.insts[inst.index()].result else { return false };
    let (c, t, f) = if subst.is_empty() {
        (c, t, f)
    } else {
        (crate::merge::find(subst, c), crate::merge::find(subst, t), crate::merge::find(subst, f))
    };
    let Some(target) = correlated_select_target(func, facts, param_of, defs, (b_idx, self_last, c, t, f)) else { return false };
    if subst.is_empty() {
        *subst = (0..func.types.len()).map(|i| Value(i as u32)).collect();
    }
    subst[res.index()] = crate::merge::find(subst, target);
    true
}

fn fold_shifted_select(func: &mut Function, facts: &BoolFacts, seen: &mut Vec<(Value, Type, Value, u64, u64)>, out: &mut Vec<Inst>, inst: Inst) -> bool {
    let Op::Select(cond, t_val, f_val) = func.insts[inst.index()].op else { return false };
    let Some(res) = func.insts[inst.index()].result else { return false };
    let ty = func.ty(res);
    if !ty.is_small_int() {
        return false;
    }
    let (Some(ca), Some(cb)) = (facts.konst(t_val), facts.konst(f_val)) else { return false };
    let (a_u, b_u) = (extend(ca, ty, false) as u64, extend(cb, ty, false) as u64);
    if a_u == b_u {
        return false;
    }
    let w = ty.unsigned_max();
    if let Some((base_sel, shift)) = seen.iter().find_map(|&(c, st, r1, a1, b1)| {
        (c == cond && st == ty)
            .then(|| (1_u32..=4).find(|&k| ((a1 << k) & w) == a_u && ((b1 << k) & w) == b_u).map(|k| (r1, k)))
            .flatten()
    }) {
        let shift_val = func.value(ty);
        out.push(func.add(Op::Const(i64::from(shift)), Some(shift_val)));
        func.insts[inst.index()].op = Op::Binary(Binary::Ishl, base_sel, shift_val);
        return true;
    }
    seen.push((cond, ty, res, a_u, b_u));
    false
}

/// Folds `x ^ 1` / `x ^ y` in `select` and `brif` conditions, eliminates correlated join-param selects, and sinks single-use compares toward their branch.
pub(crate) fn fold_bxor_conds(func: &mut Function) -> bool {
    let mut facts = BoolFacts::collect(func);
    let mut param_of = vec![None; func.types.len()];
    let entry_idx = func.entry.map(|e| e.index());
    for (b_idx, block) in func.blocks.iter().enumerate() {
        if Some(b_idx) != entry_idx {
            for (p_idx, &p) in block.params.iter().enumerate() {
                param_of[p.index()] = Some((b_idx, p_idx));
            }
        }
    }
    let (mut changed, mut defs, mut subst) = (false, None, Vec::new());
    for b_idx in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[b_idx].insts);
        let self_last = insts.last().copied();
        let (mut out, mut seen_sels) = (Vec::with_capacity(insts.len()), Vec::new());
        for inst in insts {
            changed |= fold_bxor_cond(func, &mut facts, &mut out, inst);
            if !fold_correlated_select(func, &facts, &param_of, (&mut defs, &mut subst), (b_idx, self_last, inst)) {
                changed |= fold_shifted_select(func, &facts, &mut seen_sels, &mut out, inst);
                out.push(inst);
            }
        }
        func.blocks[b_idx].insts = out;
    }
    if !subst.is_empty() {
        for inst in func.blocks.iter().flat_map(|b| &b.insts) {
            func.insts[inst.index()].op.map_operands(|v| crate::merge::find(&subst, v));
        }
        changed = true;
    }
    if changed {
        crate::schedule::sink_compares(func);
    }
    changed
}

fn difference(func: &Function, lcx: &LateCx, v: Value) -> Option<Inst> {
    let inst = lcx.def(v)?;
    let Op::Binary(Binary::Isub | Binary::Bxor, a, b) = func.insts[inst.index()].op else { return None };
    let uses = lcx.uses.get(v.index()).copied()?;
    let all_tests = func.ty(a) == Type::I8
        && lcx.konst(func, a).is_some() != lcx.konst(func, b).is_some()
        && lcx.tests.get(v.index()) == Some(&uses);
    (func.ty(a).is_small_int() && (uses == 1 || all_tests)).then_some(inst)
}

struct LateCx {
    defs: Vec<Option<Inst>>,
    uses: Vec<u32>,
    tests: Vec<u32>,
    out: Vec<Inst>,
    dead: Vec<Inst>,
    work: Vec<Value>,
}

impl LateCx {
    fn def(&self, v: Value) -> Option<Inst> {
        self.defs.get(v.index()).copied().flatten()
    }

    fn konst(&self, func: &Function, v: Value) -> Option<i64> {
        match op_of(func, |u| self.def(u), v) {
            Some(&Op::Const(imm)) => Some(imm),
            _ => None,
        }
    }

    fn bump(&mut self, v: Value, delta: i32) {
        let Some(c) = self.uses.get_mut(v.index()) else { return };
        if delta > 0 {
            *c += 1;
        } else if *c > 0 {
            *c -= 1;
            if *c == 0 {
                self.work.push(v);
            }
        }
    }

    fn replace_op(&mut self, func: &mut Function, inst: Inst, new_op: Op, start: usize) {
        func.insts[inst.index()].op.clone().for_operands(|u| self.bump(u, -1));
        for idx in start..self.out.len() {
            let ni = self.out[idx];
            if let Some(r) = func.insts[ni.index()].result {
                if self.defs.len() <= r.index() {
                    self.defs.resize(r.index() + 1, None);
                    self.uses.resize(r.index() + 1, 0);
                }
                self.defs[r.index()] = Some(ni);
            }
            func.insts[ni.index()].op.for_operands(|u| self.bump(u, 1));
        }
        new_op.for_operands(|u| self.bump(u, 1));
        func.insts[inst.index()].op = new_op;
    }

    fn sweep_orphans(&mut self, func: &mut Function) {
        while let Some(v) = self.work.pop() {
            let Some(inst) = self.def(v) else { continue };
            if self.uses.get(v.index()).copied().unwrap_or(1) != 0 || self.dead.contains(&inst) {
                continue;
            }
            let op = &func.insts[inst.index()].op;
            if !op.is_speculable() {
                continue;
            }
            self.dead.push(inst);
            op.for_operands(|u| self.bump(u, -1));
        }
    }
}

fn late_fold_ord(func: &mut Function, lcx: &mut LateCx, inst: Inst) -> bool {
    let start = lcx.out.len();
    let def_of = |v: Value| lcx.defs.get(v.index()).copied().flatten();
    match func.insts[inst.index()].op {
        Op::Select(cond, t, f) => {
            let Some(m) = match_ord_select(func, &def_of, (cond, t, f)) else { return false };
            let new_op = build_ord_select(func.ty(t), m, &mut |uty, op| {
                let val = func.value(uty);
                lcx.out.push(func.add(op, Some(val)));
                val
            });
            lcx.replace_op(func, inst, new_op, start);
            true
        }
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), x, y) => {
            let Some((a, b)) = complementary_eq_pair(func, &def_of, x, y) else { return false };
            lcx.replace_op(func, inst, Op::Icmp(cc, a, b), start);
            true
        }
        _ => false,
    }
}

fn offset_target(func: &Function, lcx: &LateCx, op: &Op) -> Option<(Option<IntCC>, Value, i64)> {
    match *op {
        Op::Brif(cond, ..) | Op::Select(cond, ..) => Some((None, cond, 0)),
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), x, y) => match (lcx.konst(func, x), lcx.konst(func, y)) {
            (None, Some(k)) => Some((Some(cc), x, k)),
            (Some(k), None) => Some((Some(cc), y, k)),
            _ => None,
        },
        _ => None,
    }
}

fn shift_offset(func: &mut Function, lcx: &mut LateCx, gone: Inst, rhs: i64) -> Option<(Value, Value)> {
    let op = func.insts[gone.index()].op.clone();
    let Op::Binary(bin, a, b) = op else { return None };
    let ty = func.ty(a);
    if extend(rhs, ty, false) == 0 {
        return Some((a, b));
    }
    let konst = |v: Value| lcx.konst(func, v);
    let (base, imm) = match (bin, konst(a), konst(b)) {
        (Binary::Isub, Some(c), None) => (b, c.wrapping_sub(rhs)),
        (Binary::Isub, ..) => const_offset(&op, konst).map(|(base, delta)| (base, rhs.wrapping_sub(delta)))?,
        (_, None, Some(c)) | (_, Some(c), None) => (if konst(a).is_some() { b } else { a }, c ^ rhs),
        _ => return None,
    };
    let val = func.value(ty);
    lcx.out.push(func.add(Op::Const(extend(imm, ty, true)), Some(val)));
    Some((base, val))
}

fn offset_compare(func: &mut Function, lcx: &mut LateCx, inst: Inst) -> bool {
    let Some((cc, diff, rhs)) = offset_target(func, lcx, &func.insts[inst.index()].op) else { return false };
    let Some(gone) = difference(func, lcx, diff) else { return false };
    let start = lcx.out.len();
    let Some((lhs, bound)) = shift_offset(func, lcx, gone, rhs) else { return false };
    let (lhs, bound) = complementary_eq_pair(func, &|v| lcx.defs.get(v.index()).copied().flatten(), lhs, bound).unwrap_or((lhs, bound));
    let new_op = match cc {
        Some(cc) => Op::Icmp(cc, lhs, bound),
        None => {
            let test = func.value(Type::I8);
            lcx.out.push(func.add(Op::Icmp(IntCC::NotEqual, lhs, bound), Some(test)));
            let mut op = func.insts[inst.index()].op.clone();
            if let Op::Brif(cond, ..) | Op::Select(cond, ..) = &mut op {
                *cond = test;
            }
            op
        }
    };
    lcx.replace_op(func, inst, new_op, start);
    if lcx.uses.get(diff.index()) == Some(&0) && !lcx.dead.contains(&gone) {
        lcx.dead.push(gone);
        func.insts[gone.index()].op.for_operands(|u| lcx.bump(u, -1));
    }
    true
}

/// The integer compare equivalent to using `op` directly as a non-zero branch condition.
pub(crate) fn branch_cmp(op: &Op) -> Option<(IntCC, Value, Value)> {
    match *op {
        Op::Icmp(cc, a, b) => Some((cc, a, b)),
        Op::Binary(Binary::Isub | Binary::Bxor, a, b) => Some((IntCC::NotEqual, a, b)),
        _ => None,
    }
}

/// Rewrites a branch, a select, or an equality with a constant that reads `a - b` or `a ^ b` into a direct compare of `a`.
pub(crate) fn compare_offsets(func: &mut Function) -> bool {
    let mut lcx = LateCx {
        defs: crate::fold::defs(func),
        uses: func.use_counts(),
        tests: Vec::new(),
        out: Vec::new(),
        dead: Vec::new(),
        work: Vec::new(),
    };
    let mut tests = vec![0_u32; lcx.uses.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        if let Some((_, v, _)) = offset_target(func, &lcx, &func.insts[inst.index()].op)
            && let Some(n) = tests.get_mut(v.index())
        {
            *n += 1;
        }
    }
    lcx.tests = tests;
    let mut changed = false;
    for b in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[b].insts);
        lcx.out = Vec::with_capacity(insts.len());
        for inst in insts {
            changed |= late_fold_ord(func, &mut lcx, inst) || offset_compare(func, &mut lcx, inst);
            lcx.out.push(inst);
        }
        func.blocks[b].insts = std::mem::take(&mut lcx.out);
    }
    lcx.sweep_orphans(func);
    for block in &mut func.blocks {
        block.insts.retain(|inst| !lcx.dead.contains(inst));
    }
    for &inst in &lcx.dead {
        func.insts[inst.index()].op = Op::Trap;
    }
    changed || !lcx.dead.is_empty()
}
