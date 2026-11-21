use crate::cfg::{Cfg, Loop};
use crate::ir::{Binary, Block, Function, IntCC, Op, Type, Value};

use super::Ranges;
use super::rel::{Path, RelKind, cond_truth, rel_of};

#[derive(Clone, Copy)]
struct SumInv {
    head: Block,
    p: Value,
    q: Value,
    bound: Value,
    ty: Type,
}

#[derive(Clone, Copy)]
struct OrderInv {
    head: Block,
    lo: Value,
    hi: Value,
    bound: Value,
    ty: Type,
}

#[derive(Clone, Copy)]
struct PtrInv {
    head: Block,
    ptr: Value,
    base: Value,
    end: Value,
    len: Value,
    shift: u32,
    stride: u64,
}

#[derive(Clone, Copy)]
struct SubBound {
    idx: Value,
    end: Value,
    min_diff: u64,
    ty: Type,
}

#[derive(Clone, Copy)]
pub(super) struct RelInv {
    pub(super) head: Block,
    pub(super) param: Value,
    pub(super) bound: Value,
    pub(super) ty: Type,
    pub(super) kind: RelKind,
}

#[derive(Default)]
pub(super) struct Env {
    sums: Vec<SumInv>,
    orders: Vec<OrderInv>,
    ptrs: Vec<PtrInv>,
    pub(super) rel_invs: Vec<RelInv>,
    param_lo: Vec<(Block, Value, u64)>,
}

#[derive(Clone, Default)]
pub(super) struct ExtraPath {
    subs: Vec<SubBound>,
    diffs: Vec<(Value, Value, u64)>,
}

fn op_is_const_val(ranges: &Ranges, path: &Path, v: Value, exact: u64) -> bool {
    path.fact(ranges, v).exact() == Some(exact)
}

fn edge_ule(func: &Function, ranges: &Ranges, path: &Path, ty: Type, a: Value, b: Value) -> bool {
    a == b
        || path.fact(ranges, a).hi <= path.fact(ranges, b).lo
        || path.decide_rel(func, ranges, IntCC::UnsignedLessThanOrEqual, ty, a, b) == Some(true)
}

fn edge_min_lo(ranges: &Ranges, env: &Env, cfg: &Cfg, path: &Path, block: Block, v: Value) -> u64 {
    let mut lo = path.fact(ranges, v).lo;
    for &(head, p, min_lo) in &env.param_lo {
        if p == v && cfg.dominates(head, block) {
            lo = lo.max(min_lo);
        }
    }
    lo
}

fn outside_sum_le(func: &Function, ranges: &Ranges, path: &Path, ty: Type, a: Value, b: Value, bound: Value) -> bool {
    if let (Some(c1), Some(c2)) = (path.fact(ranges, a).exact(), path.fact(ranges, b).exact()) {
        return c1.checked_add(c2).is_some_and(|s| s <= path.fact(ranges, bound).lo);
    }
    if op_is_const_val(ranges, path, a, 0) && edge_ule(func, ranges, path, ty, b, bound) {
        return true;
    }
    if op_is_const_val(ranges, path, b, 0) && edge_ule(func, ranges, path, ty, a, bound) {
        return true;
    }
    for (sub, other) in [(a, b), (b, a)] {
        if let Some(&Op::Binary(Binary::Isub, outer, d)) = ranges.op(func, sub)
            && d == other
            && edge_ule(func, ranges, path, ty, outer, bound)
            && edge_ule(func, ranges, path, ty, d, outer)
        {
            return true;
        }
    }
    false
}

fn expand_arms(func: &Function, ranges: &Ranges, a: Value, b: Value, depth: u8, out: &mut Vec<(Value, Value)>) {
    if depth < 2 {
        match (ranges.op(func, a), ranges.op(func, b)) {
            (Some(&Op::Select(c1, at, af)), Some(&Op::Select(c2, bt, bf))) if c1 == c2 => {
                expand_arms(func, ranges, at, bt, depth + 1, out);
                expand_arms(func, ranges, af, bf, depth + 1, out);
                return;
            }
            (Some(&Op::Select(_, at, af)), _) => {
                expand_arms(func, ranges, at, b, depth + 1, out);
                expand_arms(func, ranges, af, b, depth + 1, out);
                return;
            }
            (_, Some(&Op::Select(_, bt, bf))) => {
                expand_arms(func, ranges, a, bt, depth + 1, out);
                expand_arms(func, ranges, a, bf, depth + 1, out);
                return;
            }
            _ => {}
        }
    }
    out.push((a, b));
}

fn is_ushr_of(func: &Function, ranges: &Ranges, val: Value, base: Value) -> bool {
    matches!(ranges.op(func, val), Some(&Op::Binary(Binary::Ushr, x, _)) if x == base)
}

fn is_nonzero_ushr_of(func: &Function, ranges: &Ranges, ty: Type, val: Value, base: Value) -> bool {
    matches!(
        ranges.op(func, val),
        Some(&Op::Binary(Binary::Ushr, x, sh))
            if x == base && ranges.constant(sh).is_some_and(|k| (k & u64::from(ty.bits() - 1)) >= 1)
    )
}

fn is_shrunk(func: &Function, ranges: &Ranges, path: &Path, ty: Type, expr: Value, base: Value) -> bool {
    if expr == base || is_ushr_of(func, ranges, expr, base) {
        return true;
    }
    match ranges.op(func, expr) {
        Some(&Op::Binary(Binary::Band, a, b)) if a == base || b == base => true,
        Some(&Op::Binary(Binary::Isub, x, d)) if x == base => is_ushr_of(func, ranges, d, base) || edge_ule(func, ranges, path, ty, d, base),
        _ => false,
    }
}

fn add_delta_of_depth(func: &Function, ranges: &Ranges, expr: Value, base: Value, depth: u8) -> Option<(Value, u64)> {
    let &Op::Binary(Binary::Iadd, p, q) = ranges.op(func, expr)? else { return None };
    if p == base {
        return Some((q, 0));
    }
    if q == base {
        return Some((p, 0));
    }
    if depth >= 3 {
        return None;
    }
    let max = func.ty(expr).unsigned_max();
    for (inner, k_val) in [(p, q), (q, p)] {
        if let Some(k) = ranges.constant(k_val)
            && let Some((d, c)) = add_delta_of_depth(func, ranges, inner, base, depth + 1)
            && let Some(sum) = c.checked_add(k).filter(|&s| s <= max)
        {
            return Some((d, sum));
        }
    }
    None
}

fn add_delta_of(func: &Function, ranges: &Ranges, expr: Value, base: Value) -> Option<(Value, u64)> {
    add_delta_of_depth(func, ranges, expr, base, 0)
}

fn sub_delta_of(func: &Function, ranges: &Ranges, expr: Value, base: Value) -> Option<(Value, u64)> {
    match ranges.op(func, expr) {
        Some(&Op::Binary(Binary::Isub, x, d)) if x == base => Some((d, 0)),
        Some(&Op::Binary(Binary::Isub, inner, k)) if ranges.constant(k) == Some(1) => {
            sub_delta_of(func, ranges, inner, base).filter(|&(_, c)| c == 0).map(|(d, _)| (d, 1))
        }
        _ => None,
    }
}

fn sum_pair_preserves(func: &Function, ranges: &Ranges, env: &Env, cfg: &Cfg, path: &Path, latch: Block, ty: Type, (pa, qa): (Value, Value), (p, q): (Value, Value)) -> bool {
    for (ea, eb, bp, bq) in [(pa, qa, p, q), (qa, pa, q, p)] {
        if is_shrunk(func, ranges, path, ty, ea, bp) && is_shrunk(func, ranges, path, ty, eb, bq) {
            return true;
        }
        if let (Some((d1, c1)), Some((d2, c2))) = (add_delta_of(func, ranges, ea, bp), sub_delta_of(func, ranges, eb, bq)) {
            let delta_ok = (d1 == d2 && c1 == c2) || (c1 == 0 && c2 == 0 && edge_ule(func, ranges, path, ty, d1, d2));
            let base_ok = if c2 == 0 {
                is_ushr_of(func, ranges, d2, bq) || edge_ule(func, ranges, path, ty, d2, bq)
            } else {
                c2 == 1 && is_nonzero_ushr_of(func, ranges, ty, d2, bq) && edge_min_lo(ranges, env, cfg, path, latch, bq) >= 1
            };
            if delta_ok && base_ok {
                return true;
            }
        }
    }
    false
}

fn is_halved_mid(func: &Function, ranges: &Ranges, ty: Type, expr: Value, lo: Value, hi: Value) -> bool {
    let Some((d, 0)) = add_delta_of(func, ranges, expr, lo) else { return false };
    let Some(&Op::Binary(Binary::Ushr, diff, _)) = ranges.op(func, d) else { return false };
    is_nonzero_ushr_of(func, ranges, ty, d, diff) && matches!(ranges.op(func, diff), Some(&Op::Binary(Binary::Isub, a, b)) if a == hi && b == lo)
}

fn order_pair_preserves(func: &Function, ranges: &Ranges, path: &Path, ty: Type, (la, ha): (Value, Value), (lo, hi): (Value, Value)) -> bool {
    let lt_lh = path.decide_rel(func, ranges, IntCC::UnsignedLessThan, ty, lo, hi) == Some(true);
    let hi_ok = is_shrunk(func, ranges, path, ty, ha, hi) || is_halved_mid(func, ranges, ty, ha, lo, hi);
    if !hi_ok {
        return false;
    }
    if la == lo && (ha == hi || is_halved_mid(func, ranges, ty, ha, lo, hi)) {
        return true;
    }
    if ha == hi {
        if let Some((k_val, 0)) = add_delta_of(func, ranges, la, lo)
            && path.fact(ranges, k_val).exact() == Some(1)
            && lt_lh
        {
            return true;
        }
        if let Some(&Op::Binary(Binary::Iadd, mid, one)) = ranges.op(func, la)
            && op_is_const_val(ranges, path, one, 1)
            && is_halved_mid(func, ranges, ty, mid, lo, hi)
            && lt_lh
        {
            return true;
        }
    }
    false
}

fn latch_guards(func: &Function, cfg: &Cfg, ranges: &Ranges, env: &Env, each: &Loop, latch: Block, dom_exit: Option<&Path>) -> (Path, ExtraPath) {
    let mut chain = Vec::new();
    let mut cur = latch;
    while cur != each.head && each.body.get(cur.index()).copied().unwrap_or(false) {
        chain.push(cur);
        let Some(parent) = cfg.idom.get(cur.index()).copied().flatten().filter(|&p| p != cur) else { break };
        cur = parent;
    }
    let mut path = dom_exit.cloned().unwrap_or_default();
    let mut extra = ExtraPath::default();
    let mut prev: Option<Block> = None;
    for blk in std::iter::once(each.head).chain(chain.into_iter().rev()) {
        if let Some(p) = prev && cfg.single_pred(blk) == Some(p) && !func.blocks[blk.index()].params.is_empty() {
            let edge_args = match func.last(p) {
                Some(Op::Jump(t, a)) if *t == blk => Some(a.as_slice()),
                Some(Op::Brif(_, t, ta, f, fa)) if *t != *f && ((*t == blk) ^ (*f == blk)) => Some(if *t == blk { ta.as_slice() } else { fa.as_slice() }),
                _ => None,
            };
            if let Some(args) = edge_args {
                for (&param, &arg) in func.blocks[blk.index()].params.iter().zip(args) {
                    if func.ty(param).is_small_int() && param != arg { path.push_rel_kind(RelKind::Eq, param, arg, func.ty(param)); }
                }
            }
        }
        prev = Some(blk);
        for &inst in &func.blocks[blk.index()].insts {
            if let Op::Trapz(cond) | Op::Assume(cond) = func.insts[inst.index()].op {
                path.assume(func, ranges, cond, true);
                extra.assume(func, (cfg, ranges, env, &path, blk), cond, true, 0);
            }
        }
        if blk != latch && let Some(&Op::Brif(cond, t, _, e, _)) = func.last(blk) && t != e {
            let t_taken = cfg.single_pred(t) == Some(blk) && cfg.dominates(t, latch);
            let e_taken = cfg.single_pred(e) == Some(blk) && cfg.dominates(e, latch);
            if t_taken ^ e_taken {
                path.assume(func, ranges, cond, t_taken);
                extra.assume(func, (cfg, ranges, env, &path, blk), cond, t_taken, 0);
            }
        }
    }
    if let Some(&Op::Brif(cond, t, _, e, _)) = func.last(latch) && t != e && (t == each.head) ^ (e == each.head) {
        path.assume(func, ranges, cond, t == each.head);
        extra.assume(func, (cfg, ranges, env, &path, latch), cond, t == each.head, 0);
    }
    (path, extra)
}

fn cond_proves_rel(func: &Function, ranges: &Ranges, cond: Value, taken: bool, cc: IntCC, lhs: Value, rhs: Value, depth: u8) -> bool {
    if depth > 4 { return false; }
    match ranges.op(func, cond) {
        Some(&Op::Binary(Binary::Band, a, b)) if taken => cond_proves_rel(func, ranges, a, true, cc, lhs, rhs, depth + 1) || cond_proves_rel(func, ranges, b, true, cc, lhs, rhs, depth + 1),
        Some(&Op::Binary(Binary::Bor, a, b)) if !taken => cond_proves_rel(func, ranges, a, false, cc, lhs, rhs, depth + 1) || cond_proves_rel(func, ranges, b, false, cc, lhs, rhs, depth + 1),
        Some(&Op::Icmp(raw, x, y)) if func.ty(x) == func.ty(lhs) => {
            let eff = if taken { raw } else { raw.inverse() };
            if eff.is_signed() { return false; }
            let (rk, rl, rr) = rel_of(eff, x, y);
            match cc {
                IntCC::UnsignedLessThan => rk == RelKind::Ult && rl == lhs && rr == rhs,
                IntCC::UnsignedLessThanOrEqual => (matches!(rk, RelKind::Ult | RelKind::Ule) && rl == lhs && rr == rhs) || (rk == RelKind::Eq && ((rl == lhs && rr == rhs) || (rl == rhs && rr == lhs))),
                _ => false,
            }
        }
        _ => false,
    }
}

fn jump_latch_proves(func: &Function, cfg: &Cfg, ranges: &Ranges, each: &Loop, (latch, arg, cc, bound): (Block, Value, IntCC, Value)) -> bool {
    if !matches!(func.last(latch), Some(Op::Jump(t, _)) if *t == each.head) { return false; }
    let Some(p_idx) = func.blocks[latch.index()].params.iter().position(|&v| v == arg) else { return false };
    let preds = cfg.preds_of(latch);
    !preds.is_empty() && preds.iter().all(|&p| {
        each.body.get(p.index()).copied().unwrap_or(false)
            && matches!(func.last(p), Some(&Op::Brif(cond, t, ref ta, f, ref fa)) if t != f && ((t == latch) ^ (f == latch)) && {
                let (taken, a) = if t == latch { (true, ta.get(p_idx).copied()) } else { (false, fa.get(p_idx).copied()) };
                a.is_some_and(|v| cond_proves_rel(func, ranges, cond, taken, cc, v, bound, 0))
            })
    })
}

fn ptr_end_parts(func: &Function, ranges: &Ranges, end: Value, base: Value) -> Option<(Value, u32, u64)> {
    let &Op::Binary(Binary::Iadd, a, b) = ranges.op(func, end)? else { return None };
    for (cand_base, byte_len) in [(a, b), (b, a)] {
        if cand_base != base { continue; }
        match ranges.op(func, byte_len) {
            Some(&Op::Binary(Binary::Ishl, len, sh)) => {
                if let Some(k) = ranges.constant(sh) {
                    let s = (k & 63) as u32;
                    if s < 64 { return Some((len, s, 1_u64 << s)); }
                }
            }
            Some(&Op::Binary(Binary::Imul, x, y)) => {
                for (len, sc) in [(x, y), (y, x)] {
                    if let Some(stride) = ranges.constant(sc).filter(|&s| s > 0) { return Some((len, 0, stride)); }
                }
            }
            _ => {}
        }
        return Some((byte_len, 0, 1));
    }
    None
}

impl Env {
    pub(super) fn collect_loop(&mut self, func: &Function, (cfg, ranges, incoming, complete, exits, def_blocks): (&Cfg, &Ranges, &[Vec<(Block, &[Value])>], &[bool], &[Option<Path>], &std::cell::OnceCell<Vec<Option<Block>>>), each: &Loop) {
        if !complete[each.head.index()] || func.blocks[each.head.index()].params.is_empty() { return; }
        let dom_exit = cfg.idom.get(each.head.index()).copied().flatten().filter(|&d| d != each.head).and_then(|d| exits.get(d.index()).and_then(Option::as_ref));
        let (mut outside, mut inside): (Vec<(Path, &[Value])>, Vec<(Block, Path, ExtraPath, &[Value])>) = (Vec::new(), Vec::new());
        for &(pred, args) in &incoming[each.head.index()] {
            if !cfg.reachable.get(pred.index()).copied().unwrap_or(false) { continue; }
            if each.body.get(pred.index()).copied().unwrap_or(false) {
                let (gp, ge) = latch_guards(func, cfg, ranges, self, each, pred, dom_exit);
                inside.push((pred, gp, ge, args));
            } else if let Some(mut p) = exits[pred.index()].clone() {
                p.follow(func, ranges, pred, each.head);
                outside.push((p, args));
            } else {
                return;
            }
        }
        if outside.is_empty() || inside.is_empty() { return; }
        let params = &func.blocks[each.head.index()].params;
        for (k, &p) in params.iter().enumerate() {
            if !func.ty(p).is_small_int() { continue; }
            let min_lo = outside.iter().map(|(path, args)| args.get(k).map_or(0, |&a| path.fact(ranges, a).lo))
                .chain(inside.iter().map(|(_, gp, _, args)| args.get(k).map_or(0, |&a| gp.fact(ranges, a).lo))).min().unwrap_or(0);
            if min_lo > ranges.of(p).lo { self.param_lo.push((each.head, p, min_lo)); }
        }
        let def_blocks = def_blocks.get_or_init(|| func.def_blocks());
        let dom_inv = |v: Value| def_blocks.get(v.index()).copied().flatten().is_some_and(|db| !each.body[db.index()] && cfg.dominates(db, each.head));
        let mut inv_params: Vec<(Value, Value)> = Vec::new();
        for (k, &q) in params.iter().enumerate() {
            let ty = func.ty(q);
            let Some(&init_q) = outside[0].1.get(k) else { continue };
            if ty.is_small_int()
                && dom_inv(init_q)
                && outside.iter().all(|(_, args)| args.get(k) == Some(&init_q))
                && inside.iter().all(|&(_, _, _, args)| args.get(k) == Some(&q))
            {
                inv_params.push((q, init_q));
                self.rel_invs.push(RelInv { head: each.head, param: q, bound: init_q, ty, kind: RelKind::Eq });
            }
        }
        let push_inv = |v: Value, list: &mut Vec<Value>| {
            if (dom_inv(v) || inv_params.iter().any(|&(q, _)| q == v)) && !list.contains(&v) { list.push(v); }
        };
        let mut bounds: Vec<Value> = Vec::new();
        for (_, args) in &outside {
            for &a in *args {
                push_inv(a, &mut bounds);
                if let Some(&Op::Binary(Binary::Isub, outer, _)) = ranges.op(func, a) { push_inv(outer, &mut bounds); }
            }
        }
        let mut rel_bounds: Vec<Value> = Vec::new();
        for &(latch, _, _, _) in &inside {
            if let Some(&Op::Brif(cond, ..)) = func.last(latch) && let Some(&Op::Icmp(_, x, y)) = ranges.op(func, cond) {
                push_inv(x, &mut rel_bounds);
                push_inv(y, &mut rel_bounds);
            }
        }
        for path in inside.iter().map(|(_, gp, _, _)| gp).chain(outside.iter().map(|(p, _)| p)) {
            for r in path.rels().iter().rev() { push_inv(r.lhs, &mut rel_bounds); push_inv(r.rhs, &mut rel_bounds); }
        }
        for &b in &bounds { if !rel_bounds.contains(&b) { rel_bounds.push(b); } }
        rel_bounds.truncate(16);
        let mut arms = Vec::new();
        for (k1, &p) in params.iter().enumerate() {
            let ty = func.ty(p);
            if !ty.is_small_int() { continue; }
            for (k2, &q) in params.iter().enumerate().skip(k1 + 1) {
                if func.ty(q) != ty { continue; }
                let mut ordered = [None; 2];
                for &bound in &bounds {
                    if func.ty(bound) != ty { continue; }
                    let sum_base = outside.iter().all(|(path, args)| outside_sum_le(func, ranges, path, ty, args[k1], args[k2], bound));
                    if sum_base && inside.iter().all(|&(latch, ref gp, _, args)| {
                        arms.clear();
                        expand_arms(func, ranges, args[k1], args[k2], 0, &mut arms);
                        arms.iter().all(|&pair| sum_pair_preserves(func, ranges, self, cfg, gp, latch, ty, pair, (p, q)))
                    }) {
                        self.sums.push(SumInv { head: each.head, p, q, bound, ty });
                    }
                    for (dir, (kl, kh, lo, hi)) in [(k1, k2, p, q), (k2, k1, q, p)].into_iter().enumerate() {
                        let ord_base = *ordered[dir].get_or_insert_with(|| outside.iter().all(|(path, args)| edge_ule(func, ranges, path, ty, args[kl], args[kh])))
                            && outside.iter().all(|(path, args)| edge_ule(func, ranges, path, ty, args[kh], bound));
                        if ord_base && inside.iter().all(|&(_, ref gp, _, args)| {
                            arms.clear();
                            expand_arms(func, ranges, args[kl], args[kh], 0, &mut arms);
                            arms.iter().all(|&pair| order_pair_preserves(func, ranges, gp, ty, pair, (lo, hi)))
                        }) {
                            self.orders.push(OrderInv { head: each.head, lo, hi, bound, ty });
                        }
                    }
                }
            }
        }
        for (k, &p) in params.iter().enumerate() {
            let ty = func.ty(p);
            if !ty.is_small_int() { continue; }
            for &bound in &rel_bounds {
                if func.ty(bound) != ty || bound == p { continue; }
                let out_b = inv_params.iter().find(|&&(q, _)| q == bound).map_or(bound, |&(_, init_q)| init_q);
                let in_alt = inv_params.iter().find(|&&(_, init_q)| init_q == bound).map(|&(q, _)| q);
                for (kind, cc) in [(RelKind::Ult, IntCC::UnsignedLessThan), (RelKind::Ule, IntCC::UnsignedLessThanOrEqual)] {
                    let base_ok = outside.iter().all(|(path, args)| {
                        path.decide_rel(func, ranges, cc, ty, args[k], out_b) == Some(true)
                            || super::decide(cc, ty, &path.fact(ranges, args[k]), &path.fact(ranges, out_b)) == Some(true)
                    });
                    if base_ok && inside.iter().all(|&(latch, ref gp, _, args)| {
                        let check_bnd = |b| gp.decide_rel(func, ranges, cc, ty, args[k], b) == Some(true)
                            || super::decide(cc, ty, &gp.fact(ranges, args[k]), &gp.fact(ranges, b)) == Some(true)
                            || jump_latch_proves(func, cfg, ranges, each, (latch, args[k], cc, b));
                        check_bnd(bound) || in_alt.is_some_and(check_bnd) || {
                            arms.clear();
                            expand_arms(func, ranges, args[k], bound, 0, &mut arms);
                            arms.iter().all(|&(arm, bnd)| gp.decide_rel(func, ranges, cc, ty, arm, bnd) == Some(true)
                                || super::decide(cc, ty, &gp.fact(ranges, arm), &gp.fact(ranges, bnd)) == Some(true)
                                || is_shrunk(func, ranges, gp, ty, arm, p))
                        }
                    }) {
                        self.rel_invs.push(RelInv { head: each.head, param: p, bound, ty, kind });
                        break;
                    }
                }
            }
        }
        for (k, &ptr) in params.iter().enumerate() {
            if func.ty(ptr) != Type::I64 { continue; }
            let base = outside[0].1[k];
            if !outside.iter().all(|(_, args)| args[k] == base) || !dom_inv(base) { continue; }
            let mut step: Option<u64> = None;
            let mut step_ok = true;
            for &(_, _, _, args) in &inside {
                let s = match ranges.op(func, args[k]) {
                    Some(&Op::Binary(Binary::Iadd, a, b)) if a == ptr => ranges.constant(b),
                    Some(&Op::Binary(Binary::Iadd, a, b)) if b == ptr => ranges.constant(a),
                    _ => None,
                };
                match (step, s) {
                    (None, Some(v)) if v > 0 => step = Some(v),
                    (Some(prev), Some(v)) if prev == v => {}
                    _ => { step_ok = false; break; }
                }
            }
            let Some(step) = step.filter(|_| step_ok) else { continue };
            for (end_idx, db) in def_blocks.iter().enumerate() {
                let end = Value(end_idx as u32);
                let Some((len, shift, stride)) = ptr_end_parts(func, ranges, end, base) else { continue };
                if db.is_none_or(|b| each.body[b.index()] || !cfg.dominates(b, each.head)) { continue; }
                if inside.iter().all(|&(_, ref gp, ref ge, _)| {
                    ge.diffs.iter().any(|&(de, dp, md)| de == end && dp == ptr && md >= step)
                        || (step == 1 && gp.decide_rel(func, ranges, IntCC::NotEqual, Type::I64, end, ptr) == Some(true))
                }) {
                    self.ptrs.push(PtrInv { head: each.head, ptr, base, end, len, shift, stride });
                }
            }
        }
    }
}

impl ExtraPath {
    pub(super) fn follow(&mut self, func: &Function, ctx: (&Cfg, &Ranges, &Env, &Path), from: Block, into: Block) {
        if let Some(&Op::Brif(cond, then, _, otherwise, _)) = func.last(from)
            && then != otherwise
        {
            self.assume(func, (ctx.0, ctx.1, ctx.2, ctx.3, into), cond, then == into, 0);
        }
    }

    pub(super) fn assume(&mut self, func: &Function, (cfg, ranges, env, path, block): (&Cfg, &Ranges, &Env, &Path, Block), cond: Value, taken: bool, depth: u8) {
        if depth > 4 {
            return;
        }
        match ranges.op(func, cond) {
            Some(&Op::Binary(Binary::Band, a, b)) if taken => {
                self.assume(func, (cfg, ranges, env, path, block), a, true, depth + 1);
                self.assume(func, (cfg, ranges, env, path, block), b, true, depth + 1);
            }
            Some(&Op::Binary(Binary::Bor, a, b)) if !taken => {
                self.assume(func, (cfg, ranges, env, path, block), a, false, depth + 1);
                self.assume(func, (cfg, ranges, env, path, block), b, false, depth + 1);
            }
            Some(&Op::Icmp(cc, a, b)) if func.ty(a).is_small_int() => {
                let eff = if taken { cc } else { cc.inverse() };
                if eff.is_signed() {
                    return;
                }
                let (kind, x, y) = rel_of(eff, a, b);
                let ty = func.ty(a);
                let record = |this: &mut Self, diff: Value, min_diff: u64| {
                    if min_diff == 0 {
                        return;
                    }
                    if let Some(&Op::Binary(Binary::Isub, end, cur)) = ranges.op(func, diff) {
                        if !this.diffs.iter().any(|&(e, c, m)| e == end && c == cur && m >= min_diff) {
                            this.diffs.push((end, cur, min_diff));
                        }
                        if this.proves_ule(func, (cfg, ranges, env, path, block), ty, cur, end, 0)
                            && !this.subs.iter().any(|sb| sb.idx == cur && sb.end == end && sb.min_diff >= min_diff)
                        {
                            this.subs.push(SubBound { idx: cur, end, min_diff, ty });
                        }
                    }
                };
                match kind {
                    RelKind::Ule => {
                        let lo = path.fact(ranges, x).lo;
                        record(self, y, lo);
                    }
                    RelKind::Ult => {
                        if let Some(lo) = path.fact(ranges, x).lo.checked_add(1) {
                            record(self, y, lo);
                        }
                    }
                    RelKind::Eq => {
                        record(self, x, path.fact(ranges, y).lo);
                        record(self, y, path.fact(ranges, x).lo);
                    }
                    RelKind::Ne => {
                        if path.fact(ranges, y).exact() == Some(0) {
                            record(self, x, 1);
                        } else if path.fact(ranges, x).exact() == Some(0) {
                            record(self, y, 1);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn proves_ule(&self, func: &Function, (cfg, ranges, env, path, block): (&Cfg, &Ranges, &Env, &Path, Block), ty: Type, a: Value, b: Value, depth: u8) -> bool {
        if edge_ule(func, ranges, path, ty, a, b) {
            return true;
        }
        if depth > 2 {
            return false;
        }
        self.proves_ult(func, (cfg, ranges, env, path, block), ty, a, b, depth + 1)
            || env.sums.iter().any(|s| s.ty == ty && (a == s.p || a == s.q) && cfg.dominates(s.head, block) && edge_ule(func, ranges, path, ty, s.bound, b))
            || env.orders.iter().any(|o| {
                o.ty == ty
                    && (a == o.lo || a == o.hi)
                    && cfg.dominates(o.head, block)
                    && ((a == o.lo && edge_ule(func, ranges, path, ty, o.hi, b)) || (a == o.hi && edge_ule(func, ranges, path, ty, o.bound, b)))
            })
            || env.rel_invs.iter().any(|r| r.ty == ty && a == r.param && cfg.dominates(r.head, block) && edge_ule(func, ranges, path, ty, r.bound, b))
            || self.subs.iter().any(|sb| sb.ty == ty && a == sb.idx && edge_ule(func, ranges, path, ty, sb.end, b))
    }

    fn proves_ult(&self, func: &Function, (cfg, ranges, env, path, block): (&Cfg, &Ranges, &Env, &Path, Block), ty: Type, a: Value, b: Value, depth: u8) -> bool {
        if path.fact(ranges, a).hi < path.fact(ranges, b).lo
            || path.decide_rel(func, ranges, IntCC::UnsignedLessThan, ty, a, b) == Some(true)
        {
            return true;
        }
        if depth > 2 {
            return false;
        }
        for r in &env.rel_invs {
            if r.ty == ty
                && a == r.param
                && cfg.dominates(r.head, block)
                && if r.kind == RelKind::Ult {
                    self.proves_ule(func, (cfg, ranges, env, path, block), ty, r.bound, b, depth + 1)
                } else {
                    self.proves_ult(func, (cfg, ranges, env, path, block), ty, r.bound, b, depth + 1)
                }
            {
                return true;
            }
        }
        for s in &env.sums {
            if s.ty != ty || !cfg.dominates(s.head, block) || !self.proves_ule(func, (cfg, ranges, env, path, block), ty, s.bound, b, depth + 1) {
                continue;
            }
            for (p, other) in [(s.p, s.q), (s.q, s.p)] {
                let other_lo = edge_min_lo(ranges, env, cfg, path, block, other);
                if a == p && other_lo >= 1 {
                    return true;
                }
                if let Some((delta, 0)) = add_delta_of(func, ranges, a, p) {
                    let lt_other = path.fact(ranges, delta).hi < other_lo
                        || path.decide_rel(func, ranges, IntCC::UnsignedLessThan, ty, delta, other) == Some(true)
                        || (other_lo >= 1 && is_nonzero_ushr_of(func, ranges, ty, delta, other));
                    if lt_other {
                        return true;
                    }
                }
            }
        }
        for o in &env.orders {
            if o.ty != ty || !cfg.dominates(o.head, block) {
                continue;
            }
            if !self.proves_ule(func, (cfg, ranges, env, path, block), ty, o.hi, b, depth + 1) {
                continue;
            }
            let lt_lh = path.decide_rel(func, ranges, IntCC::UnsignedLessThan, ty, o.lo, o.hi) == Some(true);
            if lt_lh && (a == o.lo || is_halved_mid(func, ranges, ty, a, o.lo, o.hi)) {
                return true;
            }
        }
        for sb in &self.subs {
            if sb.ty != ty || sb.min_diff == 0 || !self.proves_ule(func, (cfg, ranges, env, path, block), ty, sb.end, b, depth + 1) {
                continue;
            }
            if a == sb.idx {
                return true;
            }
            if let Some((c_val, extra)) = add_delta_of(func, ranges, a, sb.idx)
                && path.fact(ranges, c_val).hi.checked_add(extra).is_some_and(|total| total < sb.min_diff)
            {
                return true;
            }
        }
        if ty == Type::I64 {
            for p_inv in &env.ptrs {
                if !cfg.dominates(p_inv.head, block) || !self.proves_ule(func, (cfg, ranges, env, path, block), Type::I64, p_inv.len, b, depth + 1) {
                    continue;
                }
                let min_bytes = self
                    .diffs
                    .iter()
                    .filter(|&&(de, dp, _)| de == p_inv.end && dp == p_inv.ptr)
                    .map(|d| d.2)
                    .max()
                    .or_else(|| {
                        (p_inv.stride == 1 && path.decide_rel(func, ranges, IntCC::NotEqual, Type::I64, p_inv.end, p_inv.ptr) == Some(true))
                            .then_some(1)
                    });
                let Some(min_bytes) = min_bytes else { continue };
                let match_elem_off = |expr: Value| -> bool {
                    let Some(&Op::Binary(Binary::Isub, px, pb)) = ranges.op(func, expr) else { return false };
                    px == p_inv.ptr && pb == p_inv.base
                };
                let match_scaled_off = |expr: Value| -> bool {
                    if p_inv.stride == 1 && match_elem_off(expr) {
                        return true;
                    }
                    match ranges.op(func, expr) {
                        Some(&Op::Binary(Binary::Ushr, inner, sh)) if p_inv.shift > 0 => {
                            ranges.constant(sh).is_some_and(|k| (k & 63) as u32 == p_inv.shift) && match_elem_off(inner)
                        }
                        Some(&Op::Binary(Binary::Udiv, inner, sc)) => ranges.constant(sc) == Some(p_inv.stride) && match_elem_off(inner),
                        _ => false,
                    }
                };
                if match_scaled_off(a) && min_bytes >= p_inv.stride {
                    return true;
                }
                if let Some(&Op::Binary(Binary::Iadd, x, y)) = ranges.op(func, a) {
                    for (off_expr, c_val) in [(x, y), (y, x)] {
                        if match_scaled_off(off_expr)
                            && let Some(need) = path.fact(ranges, c_val).hi.checked_add(1).and_then(|n| n.checked_mul(p_inv.stride))
                            && min_bytes >= need
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    pub(super) fn decide(&self, func: &Function, ctx: (&Cfg, &Ranges, &Env, &Path, Block), cc: IntCC, x: Value, y: Value) -> Option<bool> {
        let ty = func.ty(x);
        if cc.is_signed() {
            return None;
        }
        let (kind, a, b) = rel_of(cc, x, y);
        match kind {
            RelKind::Ult => {
                if self.proves_ult(func, ctx, ty, a, b, 0) {
                    Some(true)
                } else if self.proves_ule(func, ctx, ty, b, a, 0) {
                    Some(false)
                } else {
                    None
                }
            }
            RelKind::Ule => {
                if self.proves_ule(func, ctx, ty, a, b, 0) {
                    Some(true)
                } else if self.proves_ult(func, ctx, ty, b, a, 0) {
                    Some(false)
                } else {
                    None
                }
            }
            RelKind::Eq if self.proves_ult(func, ctx, ty, a, b, 0) || self.proves_ult(func, ctx, ty, b, a, 0) => Some(false),
            RelKind::Ne if self.proves_ult(func, ctx, ty, a, b, 0) || self.proves_ult(func, ctx, ty, b, a, 0) => Some(true),
            _ => None,
        }
    }

    pub(super) fn cond_truth(&self, func: &Function, ctx: (&Cfg, &Ranges, &Env, &Path, Block), cond: Value, depth: u8) -> Option<bool> {
        if depth > 4 {
            return None;
        }
        match ctx.1.op(func, cond) {
            Some(&Op::Icmp(cc, x, y)) if func.ty(x).is_small_int() => self.decide(func, ctx, cc, x, y),
            Some(&Op::Binary(Binary::Bxor, a, b)) => [(a, b), (b, a)].into_iter().find_map(|(inner, mask)| {
                (ctx.3.fact(ctx.1, mask).exact() == Some(1) && ctx.3.fact(ctx.1, inner).hi <= 1)
                    .then(|| self.cond_truth(func, ctx, inner, depth + 1).map(|t| !t))
                    .flatten()
            }),
            Some(&Op::Binary(Binary::Band, a, b)) if ctx.3.fact(ctx.1, a).hi <= 1 && ctx.3.fact(ctx.1, b).hi <= 1 => {
                let ta = cond_truth(func, ctx.1, ctx.3, a).or_else(|| self.cond_truth(func, ctx, a, depth + 1));
                let tb = cond_truth(func, ctx.1, ctx.3, b).or_else(|| self.cond_truth(func, ctx, b, depth + 1));
                match (ta, tb) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                }
            }
            Some(&Op::Binary(Binary::Bor, a, b)) => {
                let ta = cond_truth(func, ctx.1, ctx.3, a).or_else(|| self.cond_truth(func, ctx, a, depth + 1));
                let tb = cond_truth(func, ctx.1, ctx.3, b).or_else(|| self.cond_truth(func, ctx, b, depth + 1));
                match (ta, tb) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
