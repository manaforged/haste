use crate::cfg::{Cfg, Loop};
use crate::ir::{Binary, Block, Function, IntCC, Op, Type, Value};

use super::super::rel::{Path, RelKind};
use super::super::{Ranges, decide};
use super::guards::{jump_latch_proves, latch_guards};
use super::{
    Env, ExtraPath, OrderInv, PtrInv, RelInv, SumInv, add_delta_of, edge_min_lo, edge_ule,
    is_halved_mid, is_nonzero_ushr_of, is_shrunk, is_ushr_of, op_is_const_val, sub_delta_of,
};

type OutsideEdge<'a> = (Path, &'a [Value]);
type InsideEdge<'a> = (Block, Path, ExtraPath, &'a [Value]);

struct LoopCx<'a, 'r, 'e> {
    func: &'a Function,
    cfg: &'a Cfg,
    ranges: &'a Ranges<'r>,
    each: &'a Loop,
    outside: &'a [OutsideEdge<'e>],
    inside: &'a [InsideEdge<'e>],
    def_blocks: &'a [Option<Block>],
}

impl LoopCx<'_, '_, '_> {
    fn is_dom_inv(&self, v: Value) -> bool {
        self.def_blocks
            .get(v.index())
            .copied()
            .flatten()
            .is_some_and(|db| !self.each.body[db.index()] && self.cfg.dominates(db, self.each.head))
    }

    fn push_inv(&self, inv_params: &[(Value, Value)], v: Value, list: &mut Vec<Value>) {
        if (self.is_dom_inv(v) || inv_params.iter().any(|&(q, _)| q == v)) && !list.contains(&v) {
            list.push(v);
        }
    }

    fn params(&self) -> &[Value] {
        &self.func.blocks[self.each.head.index()].params
    }
}

fn outside_sum_le(
    func: &Function,
    ranges: &Ranges,
    path: &Path,
    ty: Type,
    a: Value,
    b: Value,
    bound: Value,
) -> bool {
    if let (Some(c1), Some(c2)) = (path.fact(ranges, a).exact(), path.fact(ranges, b).exact()) {
        return c1
            .checked_add(c2)
            .is_some_and(|s| s <= path.fact(ranges, bound).lo);
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

fn expand_arms(
    func: &Function,
    ranges: &Ranges,
    a: Value,
    b: Value,
    depth: u8,
    out: &mut Vec<(Value, Value)>,
) {
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

fn sum_pair_preserves(
    func: &Function,
    ranges: &Ranges,
    env: &Env,
    cfg: &Cfg,
    path: &Path,
    latch: Block,
    ty: Type,
    (pa, qa): (Value, Value),
    (p, q): (Value, Value),
) -> bool {
    for (ea, eb, bp, bq) in [(pa, qa, p, q), (qa, pa, q, p)] {
        if is_shrunk(func, ranges, path, ty, ea, bp) && is_shrunk(func, ranges, path, ty, eb, bq) {
            return true;
        }
        if let (Some((d1, c1)), Some((d2, c2))) = (
            add_delta_of(func, ranges, ea, bp),
            sub_delta_of(func, ranges, eb, bq),
        ) {
            let delta_ok = (d1 == d2 && c1 == c2)
                || (c1 == 0 && c2 == 0 && edge_ule(func, ranges, path, ty, d1, d2));
            let base_ok = if c2 == 0 {
                is_ushr_of(func, ranges, d2, bq) || edge_ule(func, ranges, path, ty, d2, bq)
            } else {
                c2 == 1
                    && is_nonzero_ushr_of(func, ranges, ty, d2, bq)
                    && edge_min_lo(ranges, env, cfg, path, latch, bq) >= 1
            };
            if delta_ok && base_ok {
                return true;
            }
        }
    }
    false
}

fn order_pair_preserves(
    func: &Function,
    ranges: &Ranges,
    path: &Path,
    ty: Type,
    (la, ha): (Value, Value),
    (lo, hi): (Value, Value),
) -> bool {
    let lt_lh = path.decide_rel(func, ranges, IntCC::UnsignedLessThan, ty, lo, hi) == Some(true);
    let hi_ok =
        is_shrunk(func, ranges, path, ty, ha, hi) || is_halved_mid(func, ranges, ty, ha, lo, hi);
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

fn ptr_end_parts(
    func: &Function,
    ranges: &Ranges,
    end: Value,
    base: Value,
) -> Option<(Value, u32, u64)> {
    let &Op::Binary(Binary::Iadd, a, b) = ranges.op(func, end)? else {
        return None;
    };
    for (cand_base, byte_len) in [(a, b), (b, a)] {
        if cand_base != base {
            continue;
        }
        match ranges.op(func, byte_len) {
            Some(&Op::Binary(Binary::Ishl, len, sh)) => {
                if let Some(k) = ranges.constant(sh) {
                    let s = (k & 63) as u32;
                    if s < 64 {
                        return Some((len, s, 1_u64 << s));
                    }
                }
            }
            Some(&Op::Binary(Binary::Imul, x, y)) => {
                for (len, sc) in [(x, y), (y, x)] {
                    if let Some(stride) = ranges.constant(sc).filter(|&s| s > 0) {
                        return Some((len, 0, stride));
                    }
                }
            }
            _ => {}
        }
        return Some((byte_len, 0, 1));
    }
    None
}

fn uniform_ptr_step(cx: &LoopCx<'_, '_, '_>, k: usize, ptr: Value) -> Option<u64> {
    let mut step: Option<u64> = None;
    for &(_, _, _, args) in cx.inside {
        let s = match cx.ranges.op(cx.func, args[k]) {
            Some(&Op::Binary(Binary::Iadd, a, b)) if a == ptr => cx.ranges.constant(b),
            Some(&Op::Binary(Binary::Iadd, a, b)) if b == ptr => cx.ranges.constant(a),
            _ => None,
        };
        match (step, s) {
            (None, Some(v)) if v > 0 => step = Some(v),
            (Some(prev), Some(v)) if prev == v => {}
            _ => return None,
        }
    }
    step
}

fn collect_bounds(
    cx: &LoopCx<'_, '_, '_>,
    inv_params: &[(Value, Value)],
) -> (Vec<Value>, Vec<Value>) {
    let mut bounds: Vec<Value> = Vec::new();
    for (_, args) in cx.outside {
        for &a in *args {
            cx.push_inv(inv_params, a, &mut bounds);
            if let Some(&Op::Binary(Binary::Isub, outer, _)) = cx.ranges.op(cx.func, a) {
                cx.push_inv(inv_params, outer, &mut bounds);
            }
        }
    }
    let mut rel_bounds: Vec<Value> = Vec::new();
    for &(latch, _, _, _) in cx.inside {
        if let Some(&Op::Brif(cond, ..)) = cx.func.last(latch)
            && let Some(&Op::Icmp(_, x, y)) = cx.ranges.op(cx.func, cond)
        {
            cx.push_inv(inv_params, x, &mut rel_bounds);
            cx.push_inv(inv_params, y, &mut rel_bounds);
        }
    }
    for &inst in &cx.func.blocks[cx.each.head.index()].insts {
        if let Op::Trapz(cond) | Op::Assume(cond) = cx.func.insts[inst.index()].op
            && let Some(&Op::Icmp(_, x, y)) = cx.ranges.op(cx.func, cond)
        {
            cx.push_inv(inv_params, x, &mut rel_bounds);
            cx.push_inv(inv_params, y, &mut rel_bounds);
        }
    }
    for path in cx
        .inside
        .iter()
        .map(|(_, gp, _, _)| gp)
        .chain(cx.outside.iter().map(|(p, _)| p))
    {
        for r in path.rels().iter().rev() {
            cx.push_inv(inv_params, r.lhs, &mut rel_bounds);
            cx.push_inv(inv_params, r.rhs, &mut rel_bounds);
        }
    }
    for &b in &bounds {
        if !rel_bounds.contains(&b) {
            rel_bounds.push(b);
        }
    }
    rel_bounds.truncate(16);
    (bounds, rel_bounds)
}

impl Env {
    pub(in super::super) fn collect_loop(
        &mut self,
        func: &Function,
        (cfg, ranges, incoming, complete, exits, def_blocks): (
            &Cfg,
            &Ranges,
            &[Vec<(Block, &[Value])>],
            &[bool],
            &[Option<Path>],
            &std::cell::OnceCell<Vec<Option<Block>>>,
        ),
        each: &Loop,
    ) {
        if incoming.is_empty()
            || !complete[each.head.index()]
            || !func.blocks[each.head.index()]
                .params
                .iter()
                .any(|&p| func.ty(p).is_small_int())
        {
            return;
        }
        let Some((outside, inside)) =
            self.partition_edges(func, cfg, ranges, incoming, exits, each)
        else {
            return;
        };
        let def_blocks = def_blocks.get_or_init(|| func.def_blocks());
        let cx = LoopCx {
            func,
            cfg,
            ranges,
            each,
            outside: &outside,
            inside: &inside,
            def_blocks,
        };
        self.collect_param_lo(&cx);
        let inv_params = self.collect_inv_params(&cx);
        let (bounds, rel_bounds) = collect_bounds(&cx, &inv_params);
        let mut arms = Vec::new();
        if !bounds.is_empty() {
            self.collect_sum_and_order_invs(&cx, &bounds, &mut arms);
        }
        if !rel_bounds.is_empty() {
            self.collect_rel_invs(&cx, &inv_params, &rel_bounds, &mut arms);
        }
        self.collect_ptr_invs(&cx);
    }

    fn partition_edges<'e>(
        &self,
        func: &Function,
        cfg: &Cfg,
        ranges: &Ranges,
        incoming: &[Vec<(Block, &'e [Value])>],
        exits: &[Option<Path>],
        each: &Loop,
    ) -> Option<(Vec<OutsideEdge<'e>>, Vec<InsideEdge<'e>>)> {
        let dom_exit = cfg
            .idom
            .get(each.head.index())
            .copied()
            .flatten()
            .filter(|&d| d != each.head)
            .and_then(|d| exits.get(d.index()).and_then(Option::as_ref));
        let (mut outside, mut inside) = (Vec::new(), Vec::new());
        for &(pred, args) in &incoming[each.head.index()] {
            if !cfg.reachable.get(pred.index()).copied().unwrap_or(false) {
                continue;
            }
            if each.body.get(pred.index()).copied().unwrap_or(false) {
                let (gp, ge) = latch_guards(func, cfg, ranges, self, each, pred, dom_exit);
                inside.push((pred, gp, ge, args));
            } else if let Some(mut p) = exits[pred.index()].clone() {
                p.follow(func, ranges, pred, each.head);
                outside.push((p, args));
            } else {
                return None;
            }
        }
        (!outside.is_empty() && !inside.is_empty()).then_some((outside, inside))
    }

    fn collect_param_lo(&mut self, cx: &LoopCx<'_, '_, '_>) {
        for (k, &p) in cx.params().iter().enumerate() {
            if !cx.func.ty(p).is_small_int() {
                continue;
            }
            let min_lo =
                cx.outside
                    .iter()
                    .map(|(path, args)| args.get(k).map_or(0, |&a| path.fact(cx.ranges, a).lo))
                    .chain(cx.inside.iter().map(|(_, gp, _, args)| {
                        args.get(k).map_or(0, |&a| gp.fact(cx.ranges, a).lo)
                    }))
                    .min()
                    .unwrap_or(0);
            if min_lo > cx.ranges.of(p).lo {
                self.param_lo.push((cx.each.head, p, min_lo));
            }
        }
    }

    fn collect_inv_params(&mut self, cx: &LoopCx<'_, '_, '_>) -> Vec<(Value, Value)> {
        let mut inv_params = Vec::new();
        for (k, &q) in cx.params().iter().enumerate() {
            let ty = cx.func.ty(q);
            let Some(&init_q) = cx.outside[0].1.get(k) else {
                continue;
            };
            if ty.is_small_int()
                && cx.is_dom_inv(init_q)
                && cx
                    .outside
                    .iter()
                    .all(|(_, args)| args.get(k) == Some(&init_q))
                && cx
                    .inside
                    .iter()
                    .all(|&(_, _, _, args)| args.get(k) == Some(&q))
            {
                inv_params.push((q, init_q));
                self.rel_invs.push(RelInv {
                    head: cx.each.head,
                    param: q,
                    bound: init_q,
                    ty,
                    kind: RelKind::Eq,
                });
            }
        }
        inv_params
    }

    fn collect_sum_and_order_invs(
        &mut self,
        cx: &LoopCx<'_, '_, '_>,
        bounds: &[Value],
        arms: &mut Vec<(Value, Value)>,
    ) {
        let params = cx.params();
        for (k1, &p) in params.iter().enumerate() {
            let ty = cx.func.ty(p);
            if !ty.is_small_int() {
                continue;
            }
            for (k2, &q) in params.iter().enumerate().skip(k1 + 1) {
                if cx.func.ty(q) != ty {
                    continue;
                }
                let mut ordered = [None; 2];
                for &bound in bounds {
                    if cx.func.ty(bound) != ty {
                        continue;
                    }
                    self.try_sum_inv(cx, ty, (k1, k2), (p, q), bound, arms);
                    for (dir, (kl, kh, lo, hi)) in
                        [(k1, k2, p, q), (k2, k1, q, p)].into_iter().enumerate()
                    {
                        let ord_base = *ordered[dir].get_or_insert_with(|| {
                            cx.outside.iter().all(|(path, args)| {
                                edge_ule(cx.func, cx.ranges, path, ty, args[kl], args[kh])
                            })
                        }) && cx.outside.iter().all(|(path, args)| {
                            edge_ule(cx.func, cx.ranges, path, ty, args[kh], bound)
                        });
                        if ord_base
                            && cx.inside.iter().all(|&(_, ref gp, _, args)| {
                                arms.clear();
                                expand_arms(cx.func, cx.ranges, args[kl], args[kh], 0, arms);
                                arms.iter().all(|&pair| {
                                    order_pair_preserves(cx.func, cx.ranges, gp, ty, pair, (lo, hi))
                                })
                            })
                        {
                            self.orders.push(OrderInv {
                                head: cx.each.head,
                                lo,
                                hi,
                                bound,
                                ty,
                            });
                        }
                    }
                }
            }
        }
    }

    fn try_sum_inv(
        &mut self,
        cx: &LoopCx<'_, '_, '_>,
        ty: Type,
        (k1, k2): (usize, usize),
        (p, q): (Value, Value),
        bound: Value,
        arms: &mut Vec<(Value, Value)>,
    ) {
        let sum_base = cx.outside.iter().all(|(path, args)| {
            outside_sum_le(cx.func, cx.ranges, path, ty, args[k1], args[k2], bound)
        });
        if sum_base
            && cx.inside.iter().all(|&(latch, ref gp, _, args)| {
                arms.clear();
                expand_arms(cx.func, cx.ranges, args[k1], args[k2], 0, arms);
                arms.iter().all(|&pair| {
                    sum_pair_preserves(
                        cx.func,
                        cx.ranges,
                        self,
                        cx.cfg,
                        gp,
                        latch,
                        ty,
                        pair,
                        (p, q),
                    )
                })
            })
        {
            self.sums.push(SumInv {
                head: cx.each.head,
                p,
                q,
                bound,
                ty,
            });
        }
    }

    fn collect_rel_invs(
        &mut self,
        cx: &LoopCx<'_, '_, '_>,
        inv_params: &[(Value, Value)],
        rel_bounds: &[Value],
        arms: &mut Vec<(Value, Value)>,
    ) {
        for (k, &p) in cx.params().iter().enumerate() {
            let ty = cx.func.ty(p);
            if !ty.is_small_int() {
                continue;
            }
            for &bound in rel_bounds {
                if cx.func.ty(bound) != ty || bound == p {
                    continue;
                }
                let out_b = inv_params
                    .iter()
                    .find(|&&(q, _)| q == bound)
                    .map_or(bound, |&(_, init_q)| init_q);
                let in_alt = inv_params
                    .iter()
                    .find(|&&(_, init_q)| init_q == bound)
                    .map(|&(q, _)| q);
                for (kind, cc) in [
                    (RelKind::Ult, IntCC::UnsignedLessThan),
                    (RelKind::Ule, IntCC::UnsignedLessThanOrEqual),
                ] {
                    if rel_inv_holds(cx, ty, (k, p, bound, out_b, in_alt), cc, arms) {
                        self.rel_invs.push(RelInv {
                            head: cx.each.head,
                            param: p,
                            bound,
                            ty,
                            kind,
                        });
                        break;
                    }
                }
            }
        }
    }

    fn collect_ptr_invs(&mut self, cx: &LoopCx<'_, '_, '_>) {
        for (k, &ptr) in cx.params().iter().enumerate() {
            if cx.func.ty(ptr) != Type::I64 {
                continue;
            }
            let base = cx.outside[0].1[k];
            if !cx.outside.iter().all(|(_, args)| args[k] == base) || !cx.is_dom_inv(base) {
                continue;
            }
            let Some(step) = uniform_ptr_step(cx, k, ptr) else {
                continue;
            };
            if step > 1 {
                let mut seen_ends = Vec::new();
                for &(de, dp, md) in &cx.inside[0].2.diffs {
                    if dp == ptr && md >= step && !seen_ends.contains(&de) {
                        seen_ends.push(de);
                        self.try_ptr_end(cx, ptr, base, step, de);
                    }
                }
            } else if !cx.inside[0].2.diffs.is_empty() || cx.inside[0].1.has_rels() {
                for end_idx in 0..cx.def_blocks.len() {
                    self.try_ptr_end(cx, ptr, base, step, Value(end_idx as u32));
                }
            }
        }
    }

    fn try_ptr_end(
        &mut self,
        cx: &LoopCx<'_, '_, '_>,
        ptr: Value,
        base: Value,
        step: u64,
        end: Value,
    ) {
        let Some((len, shift, stride)) = ptr_end_parts(cx.func, cx.ranges, end, base) else {
            return;
        };
        if !cx.is_dom_inv(end) {
            return;
        }
        if cx.inside.iter().all(|&(_, ref gp, ref ge, _)| {
            ge.diffs
                .iter()
                .any(|&(de, dp, md)| de == end && dp == ptr && md >= step)
                || (step == 1
                    && gp.decide_rel(cx.func, cx.ranges, IntCC::NotEqual, Type::I64, end, ptr)
                        == Some(true))
        }) {
            self.ptrs.push(PtrInv {
                head: cx.each.head,
                ptr,
                base,
                end,
                len,
                shift,
                stride,
            });
        }
    }
}

fn rel_inv_holds(
    cx: &LoopCx<'_, '_, '_>,
    ty: Type,
    (k, p, bound, out_b, in_alt): (usize, Value, Value, Value, Option<Value>),
    cc: IntCC,
    arms: &mut Vec<(Value, Value)>,
) -> bool {
    let base_ok = cx.outside.iter().all(|(path, args)| {
        path.decide_rel(cx.func, cx.ranges, cc, ty, args[k], out_b) == Some(true)
            || decide(
                cc,
                ty,
                &path.fact(cx.ranges, args[k]),
                &path.fact(cx.ranges, out_b),
            ) == Some(true)
    });
    base_ok
        && cx.inside.iter().all(|&(latch, ref gp, _, args)| {
            let check_bnd = |b| {
                gp.decide_rel(cx.func, cx.ranges, cc, ty, args[k], b) == Some(true)
                    || decide(cc, ty, &gp.fact(cx.ranges, args[k]), &gp.fact(cx.ranges, b))
                        == Some(true)
                    || jump_latch_proves(
                        cx.func,
                        cx.cfg,
                        cx.ranges,
                        cx.each,
                        (latch, args[k], cc, b, 0),
                    )
            };
            check_bnd(bound) || in_alt.is_some_and(check_bnd) || {
                arms.clear();
                expand_arms(cx.func, cx.ranges, args[k], bound, 0, arms);
                arms.iter().all(|&(arm, bnd)| {
                    gp.decide_rel(cx.func, cx.ranges, cc, ty, arm, bnd) == Some(true)
                        || decide(cc, ty, &gp.fact(cx.ranges, arm), &gp.fact(cx.ranges, bnd))
                            == Some(true)
                        || is_shrunk(cx.func, cx.ranges, gp, ty, arm, p)
                })
            }
        })
}
