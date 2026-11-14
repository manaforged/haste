use crate::cse::flip_int_cc;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::icmp::{ADDR_SPAN, three_way_non_zero};
use super::{RewriteCx, is_non_null_addr};

pub(super) fn simplify_select(cx: &mut RewriteCx, res: Option<Value>, cond: Value, t: Value, f: Value) -> Option<Op> {
    let given = (t, f);
    if t == f {
        cx.alias(res, t);
        return None;
    }
    if is_non_zero(cx, cond, 0) {
        cx.alias(res, t);
        return None;
    }
    let inner = cx.unwrap_uextend(cond);
    if let Some(arm) = tested_arm(cx, inner, t, f) {
        cx.alias(res, arm);
        return None;
    }
    let (t, f) = (refuted_arm(cx, inner, t, true).unwrap_or(t), refuted_arm(cx, inner, f, false).unwrap_or(f));
    if t == f {
        cx.alias(res, t);
        return None;
    }
    if res.is_some_and(|r| cx.func.ty(r) == Type::I8) && cx.func.ty(inner) == Type::I8 && cx.is_unit_bool(inner) {
        match (cx.uconst(t, Type::I8), cx.uconst(f, Type::I8)) {
            (Some(1), Some(0)) => {
                cx.alias(res, inner);
                return None;
            }
            (Some(1), _) if cx.is_unit_bool(f) => return Some(Op::Binary(Binary::Bor, inner, f)),
            (_, Some(0)) if cx.is_unit_bool(t) => return Some(Op::Binary(Binary::Band, inner, t)),
            (Some(0), Some(1)) => return Some(Op::Binary(Binary::Bxor, inner, f)),
            _ => {}
        }
    }
    (inner != cond || (t, f) != given).then_some(Op::Select(inner, t, f))
}

fn opposite(cx: &RewriteCx, a: Value, b: Value) -> bool {
    let (Some(&Op::Icmp(ca, xa, ya)), Some(&Op::Icmp(cb, xb, yb))) = (cx.op(a), cx.op(b)) else { return false };
    cx.func.ty(xa).is_small_int() && ((xa == xb && ya == yb && cb == ca.inverse()) || (xa == yb && ya == xb && cb == flip_int_cc(ca.inverse())))
}

fn refuted_arm(cx: &RewriteCx, cond: Value, arm: Value, taken: bool) -> Option<Value> {
    let &Op::Select(test, x, y) = cx.op(arm)? else { return None };
    opposite(cx, cond, cx.unwrap_uextend(test)).then_some(if taken { y } else { x })
}

pub(super) fn tested_select(cx: &RewriteCx, cond: Value) -> Option<(Value, bool)> {
    let &Op::Select(test, t, f) = cx.op(cx.unwrap_uextend(cond))? else { return None };
    let ty = cx.func.ty(t);
    if !ty.is_small_int() || !cx.func.ty(test).is_small_int() {
        return None;
    }
    if cx.uconst(f, ty) == Some(0) && is_non_zero(cx, t, 0) {
        Some((test, true))
    } else if cx.uconst(t, ty) == Some(0) && is_non_zero(cx, f, 0) {
        Some((test, false))
    } else {
        None
    }
}

fn reach(cx: &RewriteCx, v: Value) -> u64 {
    let ty = cx.func.ty(v);
    let bound = |x: Value| ty.unsigned_max() & !cx.known(x).0;
    match cx.op(v) {
        Some(&Op::Binary(Binary::Imul, x, y)) => [(x, y), (y, x)].into_iter().find_map(|(x, k)| bound(x).checked_mul(cx.uconst(k, ty)?)).unwrap_or(u64::MAX),
        _ => bound(v),
    }
}

fn tested_arm(cx: &RewriteCx, cond: Value, t: Value, f: Value) -> Option<Value> {
    let ty = cx.func.ty(t);
    if !ty.is_small_int() {
        return None;
    }
    let key = |v: Value| cx.uconst(v, ty).map_or(Ok(v), Err);
    let (pair, differ) = match *cx.op(cond)? {
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), a, b) if cx.func.ty(a) == ty => {
            ([key(a), key(b)], cc == IntCC::NotEqual)
        }
        Op::Binary(Binary::Isub | Binary::Bxor, a, b) if cx.func.ty(cond) == ty => ([key(a), key(b)], true),
        Op::Binary(Binary::Iadd, a, b) if cx.func.ty(cond) == ty => {
            let (v, k) = [(a, b), (b, a)].into_iter().find_map(|(v, k)| Some((v, cx.uconst(k, ty)?)))?;
            ([key(v), Err(k.wrapping_neg() & ty.unsigned_max())], true)
        }
        _ => return None,
    };
    let arms = [key(t), key(f)];
    (pair == arms || pair == [arms[1], arms[0]]).then_some(if differ { t } else { f })
}

fn entry_deref_bytes(cx: &RewriteCx, v: Value) -> u32 {
    let Some(entry) = cx.func.entry else { return 0 };
    cx.func.blocks[entry.index()]
        .params
        .iter()
        .position(|&p| p == v)
        .and_then(|pos| cx.func.signature.params.get(pos))
        .map_or(0, |abi| abi.deref_bytes)
}

pub(super) fn is_non_zero(cx: &RewriteCx, v: Value, depth: u8) -> bool {
    let ty = cx.func.ty(v);
    if depth >= 4 || !ty.is_small_int() {
        return false;
    }
    if is_non_null_addr(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), v, 0) || cx.known(v).1 != 0 {
        return true;
    }
    if entry_deref_bytes(cx, v) > 0 || three_way_non_zero(cx, v) {
        return true;
    }
    let Some(inst) = cx.def(v) else { return false };
    match cx.func.insts[inst.index()].op {
        Op::Binary(Binary::Iadd, a, b) => [(a, b), (b, a)].into_iter().any(|(base, off_v)| {
            let fixed = matches!(cx.op(base), Some(Op::Symbol(_) | Op::SlotAddr(_)));
            (fixed && reach(cx, off_v) <= ADDR_SPAN as u64)
                || cx.sconst(off_v).is_some_and(|c| c >= 0 && (c as u64) < u64::from(entry_deref_bytes(cx, base)))
        }),
        Op::Convert(Convert::Bitcast, a) if cx.func.ty(a).bytes() == ty.bytes() => is_non_zero(cx, a, depth + 1),
        Op::Convert(Convert::Uextend | Convert::Sextend, a) if cx.func.ty(a).is_small_int() => {
            is_non_zero(cx, a, depth + 1)
        }
        Op::Select(_, a, b) => is_non_zero(cx, a, depth + 1) && is_non_zero(cx, b, depth + 1),
        _ => false,
    }
}

fn bool_cond(cx: &RewriteCx, raw: Value) -> Option<Value> {
    let c = cx.unwrap_uextend(raw);
    (cx.func.ty(c) == Type::I8 && cx.is_unit_bool(c)).then_some(c)
}

fn simplify_1_bool(cx: &mut RewriteCx, res: Option<Value>, c: Value, b1: bool, b2: bool) -> Option<Op> {
    match (b1, b2) {
        (true, true) => Some(Op::Const(1)),
        (false, false) => Some(Op::Const(0)),
        (true, false) => {
            cx.alias(res, c);
            None
        }
        (false, true) => {
            let one = cx.emit(Type::I8, Op::Const(1));
            Some(Op::Binary(Binary::Bxor, c, one))
        }
    }
}

fn eval_const_select_cmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    for (norm, sel_val, c_val) in [(cc, x, y), (flip_int_cc(cc), y, x)] {
        let Some(c) = cx.sconst(c_val) else { continue };
        let Some(sel_inst) = cx.def(sel_val) else { continue };
        let Op::Select(raw_c1, t1, f1) = cx.func.insts[sel_inst.index()].op else { continue };
        let Some(c1) = bool_cond(cx, raw_c1) else { continue };
        let eval = |k: i64| crate::fold::holds(norm, ty, k, c);
        if let (Some(k1), Some(k2)) = (cx.sconst(t1), cx.sconst(f1)) {
            return Some(simplify_1_bool(cx, res, c1, eval(k1), eval(k2)));
        }
        if let (Some(k1), Some(f_inst)) = (cx.sconst(t1), cx.def(f1))
            && let Op::Select(raw_c2, t2, f2) = cx.func.insts[f_inst.index()].op
            && let (Some(k2), Some(k3)) = (cx.sconst(t2), cx.sconst(f2))
        {
            let (b1, b2, b3) = (eval(k1), eval(k2), eval(k3));
            if b2 == b3 {
                return Some(simplify_1_bool(cx, res, c1, b1, b2));
            }
            let Some(c2) = bool_cond(cx, raw_c2) else { continue };
            return Some(Some(match (b1, b2, b3) {
                (false, false, true) => {
                    let or = cx.emit(Type::I8, Op::Binary(Binary::Bor, c1, c2));
                    let one = cx.emit(Type::I8, Op::Const(1));
                    Op::Binary(Binary::Bxor, or, one)
                }
                (true, true, false) => Op::Binary(Binary::Bor, c1, c2),
                (false, true, false) => {
                    let one = cx.emit(Type::I8, Op::Const(1));
                    let nc1 = cx.emit(Type::I8, Op::Binary(Binary::Bxor, c1, one));
                    Op::Binary(Binary::Band, nc1, c2)
                }
                _ => {
                    let one = cx.emit(Type::I8, Op::Const(1));
                    let nc2 = cx.emit(Type::I8, Op::Binary(Binary::Bxor, c2, one));
                    Op::Binary(Binary::Bor, c1, nc2)
                }
            }));
        }
        if let (Some(t_inst), Some(k3)) = (cx.def(t1), cx.sconst(f1))
            && let Op::Select(raw_c2, t2, f2) = cx.func.insts[t_inst.index()].op
            && let (Some(k1), Some(k2)) = (cx.sconst(t2), cx.sconst(f2))
        {
            let (b1, b2, b3) = (eval(k1), eval(k2), eval(k3));
            if b1 == b2 {
                return Some(simplify_1_bool(cx, res, c1, b1, b3));
            }
            let Some(c2) = bool_cond(cx, raw_c2) else { continue };
            return Some(Some(match (b1, b2, b3) {
                (true, false, false) => Op::Binary(Binary::Band, c1, c2),
                (false, true, false) => {
                    let one = cx.emit(Type::I8, Op::Const(1));
                    let nc2 = cx.emit(Type::I8, Op::Binary(Binary::Bxor, c2, one));
                    Op::Binary(Binary::Band, c1, nc2)
                }
                (true, false, true) => {
                    let one = cx.emit(Type::I8, Op::Const(1));
                    let nc1 = cx.emit(Type::I8, Op::Binary(Binary::Bxor, c1, one));
                    Op::Binary(Binary::Bor, nc1, c2)
                }
                _ => {
                    let and = cx.emit(Type::I8, Op::Binary(Binary::Band, c1, c2));
                    let one = cx.emit(Type::I8, Op::Const(1));
                    Op::Binary(Binary::Bxor, and, one)
                }
            }));
        }
    }
    None
}

pub(super) fn eval_select_zero_cmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() || !res.is_some_and(|r| cx.func.ty(r) == Type::I8) {
        return None;
    }
    if let Some(folded) = eval_const_select_cmp(cx, res, cc, x, y) {
        return Some(folded);
    }
    for (norm, sel_val, c_val) in [(cc, x, y), (flip_int_cc(cc), y, x)] {
        let Some(k) = cx.uconst(c_val, ty) else { continue };
        let want_nonzero = match (norm, k) {
            (IntCC::NotEqual | IntCC::UnsignedGreaterThan, 0) | (IntCC::UnsignedGreaterThanOrEqual, 1) => true,
            (IntCC::Equal | IntCC::UnsignedLessThanOrEqual, 0) | (IntCC::UnsignedLessThan, 1) => false,
            _ => continue,
        };
        let Some(sel_inst) = cx.def(sel_val) else { continue };
        let Op::Select(raw_cond, t_val, f_val) = cx.func.insts[sel_inst.index()].op else { continue };
        let non_zero_arm = if cx.uconst(f_val, ty) == Some(0) && is_non_zero(cx, t_val, 0) {
            Some(true)
        } else if cx.uconst(t_val, ty) == Some(0) && is_non_zero(cx, f_val, 0) {
            Some(false)
        } else {
            None
        };
        let Some(true_is_nonzero) = non_zero_arm else { continue };
        let cond = cx.unwrap_uextend(raw_cond);
        let truth = want_nonzero == true_is_nonzero;
        if cx.func.ty(cond) == Type::I8 && cx.is_unit_bool(cond) {
            if truth {
                cx.alias(res, cond);
                return Some(None);
            }
            let one = cx.emit(Type::I8, Op::Const(1));
            return Some(Some(Op::Binary(Binary::Bxor, cond, one)));
        }
        let cond_ty = cx.func.ty(cond);
        if !cond_ty.is_small_int() {
            continue;
        }
        let zero = cx.emit(cond_ty, Op::Const(0));
        return Some(Some(Op::Icmp(if truth { IntCC::NotEqual } else { IntCC::Equal }, cond, zero)));
    }
    None
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
        let mut f = Self { compares: vec![None; n], consts: vec![None; n], xors: vec![None; n], uextends: vec![None; n], bools: vec![false; n] };
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            let d = &func.insts[inst.index()];
            match (&d.op, d.result) {
                (&Op::Icmp(cc, x, y), Some(r)) if func.ty(r) == Type::I8 => { f.compares[r.index()] = Some((cc, x, y)); f.bools[r.index()] = true; }
                (&Op::Fcmp(..) | &Op::Fits(..), Some(r)) if func.ty(r) == Type::I8 => f.bools[r.index()] = true,
                (&Op::Const(imm), Some(r)) if func.ty(r).is_small_int() => f.consts[r.index()] = Some(imm),
                (&Op::Binary(Binary::Bxor, x, y), Some(r)) => f.xors[r.index()] = Some((x, y)),
                (&Op::Convert(Convert::Uextend, inner), Some(r)) => f.uextends[r.index()] = Some(inner),
                _ => {}
            }
        }
        f.propagate(func);
        f
    }

    fn propagate(&mut self, func: &Function) {
        let is_01 = |v: Value, bools: &[bool], consts: &[Option<i64>]| bools[v.index()] || matches!(consts[v.index()], Some(0 | 1));
        for _ in 0..2 {
            for inst in func.blocks.iter().flat_map(|block| &block.insts) {
                let d = &func.insts[inst.index()];
                let Some(r) = d.result else { continue };
                self.bools[r.index()] |= match d.op {
                    Op::Binary(Binary::Band, x, y) => matches!(self.consts[x.index()], Some(0 | 1)) || matches!(self.consts[y.index()], Some(0 | 1)) || (is_01(x, &self.bools, &self.consts) && is_01(y, &self.bools, &self.consts)),
                    Op::Binary(Binary::Bor | Binary::Bxor, x, y) | Op::Select(_, x, y) => is_01(x, &self.bools, &self.consts) && is_01(y, &self.bools, &self.consts),
                    Op::Convert(Convert::Uextend, inner) => self.bools[inner.index()],
                    _ => false,
                };
            }
        }
    }

    fn unwrap_uext(&self, mut v: Value) -> Value {
        while let Some(inner) = self.uextends.get(v.index()).copied().flatten() { v = inner; }
        v
    }

    fn xor_one_src(&self, x: Value, y: Value) -> Option<Value> {
        if self.consts[y.index()] == Some(1) && self.bools[x.index()] { Some(x) }
        else if self.consts[x.index()] == Some(1) && self.bools[y.index()] { Some(y) }
        else { None }
    }
}

fn negated(op: &Op, facts: &BoolFacts) -> Option<(Value, Option<Value>)> {
    let is = |v: Value, want: i64| facts.consts[v.index()] == Some(want);
    match *op {
        Op::Binary(Binary::Bxor, x, y) | Op::Icmp(IntCC::NotEqual, x, y) if is(y, 1) => Some((x, Some(y))),
        Op::Binary(Binary::Bxor, y, x) | Op::Icmp(IntCC::NotEqual, y, x) if is(y, 1) => Some((x, Some(y))),
        Op::Icmp(IntCC::Equal, x, y) if is(y, 0) => Some((x, None)),
        Op::Icmp(IntCC::Equal, y, x) if is(y, 0) => Some((x, None)),
        Op::Select(c, x, y) if is(x, 0) && is(y, 1) && facts.bools[c.index()] => Some((c, Some(y))),
        _ => None,
    }
}

fn invert_brif(op: &mut Op, facts: &BoolFacts, types: &[Type]) {
    let Op::Brif(cond, then, then_args, otherwise, else_args) = op else { return };
    if let Some((x, y)) = facts.xors[cond.index()] && let Some(inner) = facts.xor_one_src(x, y) {
        *cond = inner; std::mem::swap(then, otherwise); std::mem::swap(then_args, else_args); return;
    }
    let Some((cc, x, y)) = facts.compares[cond.index()] else { return };
    let (val, flipped) = if facts.consts[y.index()] == Some(0) && types[x.index()].is_small_int() { (x, false) }
        else if facts.consts[x.index()] == Some(0) && types[y.index()].is_small_int() { (y, true) } else { return };
    let invert_arms = matches!((flipped, cc), (_, IntCC::Equal) | (false, IntCC::UnsignedLessThanOrEqual) | (true, IntCC::UnsignedGreaterThanOrEqual));
    if invert_arms || matches!((flipped, cc), (_, IntCC::NotEqual) | (false, IntCC::UnsignedGreaterThan) | (true, IntCC::UnsignedLessThan)) {
        *cond = val;
        if invert_arms { std::mem::swap(then, otherwise); std::mem::swap(then_args, else_args); }
    }
}

fn invert_inst(op: &mut Op, res: Option<Value>, facts: &BoolFacts, types: &[Type]) {
    if let (Some(res), Some((inner, opt_one))) = (res, negated(op, facts)) {
        let res_ty = types[res.index()];
        if res_ty == Type::I8 && let Some((cc, a, b)) = facts.compares[inner.index()] { *op = Op::Icmp(cc.inverse(), a, b); return; }
        if facts.bools[inner.index()] && let Some(one) = opt_one && types[inner.index()] == res_ty && types[one.index()] == res_ty && !matches!(*op, Op::Binary(Binary::Bxor, ..)) {
            *op = Op::Binary(Binary::Bxor, inner, one); return;
        }
    }
    invert_brif(op, facts, types);
}

/// Rewrites a negation of a compare, `x ^ 1`, `x == 0`, `x != 1`, or `select(x, 0, 1)`, as the inverse compare or `x ^ 1`, and simplifies branches on `x == 0` / `x != 0`.
pub(crate) fn invert(func: &mut Function) {
    let facts = BoolFacts::collect(func);
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let res = func.insts[inst.index()].result;
        invert_inst(&mut func.insts[inst.index()].op, res, &facts, &func.types);
    }
    crate::schedule::sink_compares(func);
}

fn xor_ne_pair(func: &Function, facts: &BoolFacts, x: Value, y: Value, allow_const: bool) -> Option<(Value, Value)> {
    if allow_const && let Some(pair) = [(x, y), (y, x)].into_iter().find(|&(_, c)| facts.consts[c.index()].is_some()) {
        return Some(pair);
    }
    if facts.bools[x.index()] && facts.bools[y.index()] {
        let (a, b) = (facts.unwrap_uext(x), facts.unwrap_uext(y));
        return (func.ty(a) == func.ty(b) && func.ty(a).is_small_int()).then_some((a, b));
    }
    None
}

fn fold_bxor_select(func: &mut Function, facts: &BoolFacts, out: &mut Vec<Inst>, inst: Inst) -> bool {
    let Op::Select(cond, t_val, f_val) = func.insts[inst.index()].op else { return false };
    let inner = facts.unwrap_uext(cond);
    if let Some((x, y)) = facts.xors[inner.index()] {
        if let Some((lhs, _)) = [(x, y), (y, x)].into_iter().find(|&(_, c)| facts.consts[c.index()] == Some(1))
            && let bool_src = facts.unwrap_uext(lhs) && facts.bools[bool_src.index()]
        {
            func.insts[inst.index()].op = Op::Select(bool_src, f_val, t_val);
            return true;
        }
        if let Some((a, b)) = xor_ne_pair(func, facts, x, y, false) {
            let cmp_val = func.value(Type::I8);
            out.push(func.add(Op::Icmp(IntCC::NotEqual, a, b), Some(cmp_val)));
            func.insts[inst.index()].op = Op::Select(cmp_val, t_val, f_val);
            return true;
        }
    }
    if inner != cond { func.insts[inst.index()].op = Op::Select(inner, t_val, f_val); return true; }
    false
}

fn fold_bxor_brif(func: &mut Function, facts: &BoolFacts, out: &mut Vec<Inst>, inst: Inst) -> bool {
    let Op::Brif(cond, ..) = func.insts[inst.index()].op else { return false };
    let inner = facts.unwrap_uext(cond);
    if let Some((x, y)) = facts.xors[inner.index()]
        && let Some((a, b)) = xor_ne_pair(func, facts, x, y, true) && func.ty(a) == func.ty(b) && func.ty(a).is_small_int()
    {
        let cmp_val = func.value(Type::I8);
        out.push(func.add(Op::Icmp(IntCC::NotEqual, a, b), Some(cmp_val)));
        if let Op::Brif(c_mut, ..) = &mut func.insts[inst.index()].op { *c_mut = cmp_val; }
        return true;
    }
    if inner != cond && let Op::Brif(c_mut, ..) = &mut func.insts[inst.index()].op { *c_mut = inner; return true; }
    false
}

/// Folds `x ^ 1` / `x ^ y` in `select` and `brif` conditions and sinks single-use compares toward their branch.
pub(crate) fn fold_bxor_conds(func: &mut Function) -> bool {
    let facts = BoolFacts::collect(func);
    let mut changed = false;
    for b_idx in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[b_idx].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts { changed |= fold_bxor_select(func, &facts, &mut out, inst) || fold_bxor_brif(func, &facts, &mut out, inst); out.push(inst); }
        func.blocks[b_idx].insts = out;
    }
    if changed { crate::schedule::sink_compares(func); }
    changed
}

fn difference(func: &Function, defs: &[Option<Inst>], uses: &[u32], v: Value) -> Option<(Inst, Value, Value)> {
    let inst = defs.get(v.index()).copied().flatten().filter(|_| uses.get(v.index()) == Some(&1))?;
    match func.insts[inst.index()].op {
        Op::Binary(Binary::Isub | Binary::Bxor, a, b) if func.ty(a).is_small_int() => Some((inst, a, b)),
        _ => None,
    }
}

fn offset_compare(func: &mut Function, defs: &[Option<Inst>], uses: &[u32], out: &mut Vec<Inst>, inst: Inst) -> Option<Inst> {
    let konst = |func: &Function, v: Value| match defs.get(v.index()).copied().flatten().map(|def| &func.insts[def.index()].op) {
        Some(&Op::Const(imm)) => Some(imm),
        _ => None,
    };
    let (cc, diff, rhs) = match func.insts[inst.index()].op {
        Op::Brif(cond, ..) | Op::Select(cond, ..) => (None, cond, 0),
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), x, y) => match (konst(func, x), konst(func, y)) {
            (None, Some(k)) => (Some(cc), x, k),
            (Some(k), None) => (Some(cc), y, k),
            _ => return None,
        },
        _ => return None,
    };
    let (gone, a, b) = difference(func, defs, uses, diff)?;
    let ty = func.ty(a);
    let (lhs, bound) = if crate::fold::extend(rhs, ty, false) == 0 {
        (a, b)
    } else {
        let sub = matches!(func.insts[gone.index()].op, Op::Binary(Binary::Isub, ..));
        let (base, imm) = match (konst(func, a), konst(func, b), sub) {
            (None, Some(c), true) => (a, c.wrapping_add(rhs)),
            (Some(c), None, true) => (b, c.wrapping_sub(rhs)),
            (None, Some(c), false) => (a, c ^ rhs),
            (Some(c), None, false) => (b, c ^ rhs),
            _ => return None,
        };
        let val = func.value(ty);
        out.push(func.add(Op::Const(crate::fold::extend(imm, ty, true)), Some(val)));
        (base, val)
    };
    match cc {
        Some(cc) => func.insts[inst.index()].op = Op::Icmp(cc, lhs, bound),
        None => {
            let test = func.value(Type::I8);
            out.push(func.add(Op::Icmp(IntCC::NotEqual, lhs, bound), Some(test)));
            if let Op::Brif(cond, ..) | Op::Select(cond, ..) = &mut func.insts[inst.index()].op {
                *cond = test;
            }
        }
    }
    Some(gone)
}

/// Rewrites a branch, a select, or an equality with a constant that is the only reader of `a - b` or `a ^ b` as a compare of `a`, and drops the difference; it runs after the passes that read ranges from the difference.
pub(crate) fn compare_offsets(func: &mut Function) -> bool {
    let (defs, uses) = (crate::fold::defs(func), func.use_counts());
    let mut dead = Vec::new();
    for b in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[b].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            dead.extend(offset_compare(func, &defs, &uses, &mut out, inst));
            out.push(inst);
        }
        func.blocks[b].insts = out;
    }
    for block in &mut func.blocks {
        block.insts.retain(|inst| !dead.contains(inst));
    }
    for &inst in &dead {
        func.insts[inst.index()].op = Op::Trap;
    }
    !dead.is_empty()
}
