use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Function, IntCC, Op, Type, Value};

use super::super::Ranges;
use super::super::rel::{Path, RelKind, cond_truth, rel_of};
use super::{
    Env, ExtraPath, PtrInv, SubBound, add_delta_of, edge_min_lo, edge_ule, is_halved_mid,
    is_nonzero_ushr_of,
};

#[derive(Clone, Copy)]
struct QueryCx<'a, 'r> {
    cfg: &'a Cfg,
    ranges: &'a Ranges<'r>,
    env: &'a Env,
    path: &'a Path,
    block: Block,
}

fn match_ptr_elem_off(func: &Function, ranges: &Ranges, p_inv: &PtrInv, expr: Value) -> bool {
    let Some(&Op::Binary(Binary::Isub, px, pb)) = ranges.op(func, expr) else {
        return false;
    };
    px == p_inv.ptr && pb == p_inv.base
}

fn match_ptr_scaled_off(func: &Function, ranges: &Ranges, p_inv: &PtrInv, expr: Value) -> bool {
    if p_inv.stride == 1 && match_ptr_elem_off(func, ranges, p_inv, expr) {
        return true;
    }
    match ranges.op(func, expr) {
        Some(&Op::Binary(Binary::Ushr, inner, sh)) if p_inv.shift > 0 => {
            ranges
                .constant(sh)
                .is_some_and(|k| (k & 63) as u32 == p_inv.shift)
                && match_ptr_elem_off(func, ranges, p_inv, inner)
        }
        Some(&Op::Binary(Binary::Udiv, inner, sc)) => {
            ranges.constant(sc) == Some(p_inv.stride)
                && match_ptr_elem_off(func, ranges, p_inv, inner)
        }
        _ => false,
    }
}

impl ExtraPath {
    pub(in super::super) fn follow(
        &mut self,
        func: &Function,
        ctx: (&Cfg, &Ranges, &Env, &Path),
        from: Block,
        into: Block,
    ) {
        if let Some(&Op::Brif(cond, then, _, otherwise, _)) = func.last(from)
            && then != otherwise
        {
            self.assume(
                func,
                (ctx.0, ctx.1, ctx.2, ctx.3, into),
                cond,
                then == into,
                0,
            );
        }
    }

    pub(in super::super) fn assume(
        &mut self,
        func: &Function,
        ctx: (&Cfg, &Ranges, &Env, &Path, Block),
        cond: Value,
        taken: bool,
        depth: u8,
    ) {
        if depth > 4 {
            return;
        }
        let cx = QueryCx {
            cfg: ctx.0,
            ranges: ctx.1,
            env: ctx.2,
            path: ctx.3,
            block: ctx.4,
        };
        match cx.ranges.op(func, cond) {
            Some(&Op::Binary(Binary::Band, a, b)) if taken => {
                self.assume(func, ctx, a, true, depth + 1);
                self.assume(func, ctx, b, true, depth + 1);
            }
            Some(&Op::Binary(Binary::Bor, a, b)) if !taken => {
                self.assume(func, ctx, a, false, depth + 1);
                self.assume(func, ctx, b, false, depth + 1);
            }
            Some(&Op::Icmp(cc, a, b)) if func.ty(a).is_small_int() => {
                let eff = if taken { cc } else { cc.inverse() };
                if eff.is_signed() {
                    return;
                }
                let (kind, x, y) = rel_of(eff, a, b);
                let ty = func.ty(a);
                match kind {
                    RelKind::Ule => {
                        self.record_diff(func, cx, ty, y, cx.path.fact(cx.ranges, x).lo)
                    }
                    RelKind::Ult => {
                        if let Some(lo) = cx.path.fact(cx.ranges, x).lo.checked_add(1) {
                            self.record_diff(func, cx, ty, y, lo);
                        }
                    }
                    RelKind::Eq => {
                        self.record_diff(func, cx, ty, x, cx.path.fact(cx.ranges, y).lo);
                        self.record_diff(func, cx, ty, y, cx.path.fact(cx.ranges, x).lo);
                    }
                    RelKind::Ne => {
                        if cx.path.fact(cx.ranges, y).exact() == Some(0) {
                            self.record_diff(func, cx, ty, x, 1);
                        } else if cx.path.fact(cx.ranges, x).exact() == Some(0) {
                            self.record_diff(func, cx, ty, y, 1);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn record_diff(
        &mut self,
        func: &Function,
        cx: QueryCx<'_, '_>,
        ty: Type,
        diff: Value,
        min_diff: u64,
    ) {
        if min_diff == 0 {
            return;
        }
        if let Some(&Op::Binary(Binary::Isub, end, cur)) = cx.ranges.op(func, diff) {
            if !self
                .diffs
                .iter()
                .any(|&(e, c, m)| e == end && c == cur && m >= min_diff)
            {
                self.diffs.push((end, cur, min_diff));
            }
            if self.proves_ule(func, cx, ty, cur, end, 0)
                && !self
                    .subs
                    .iter()
                    .any(|sb| sb.idx == cur && sb.end == end && sb.min_diff >= min_diff)
            {
                self.subs.push(SubBound {
                    idx: cur,
                    end,
                    min_diff,
                    ty,
                });
            }
        }
    }

    fn proves_ule(
        &self,
        func: &Function,
        cx: QueryCx<'_, '_>,
        ty: Type,
        a: Value,
        b: Value,
        depth: u8,
    ) -> bool {
        if edge_ule(func, cx.ranges, cx.path, ty, a, b) {
            return true;
        }
        if depth > 2 {
            return false;
        }
        self.proves_ult(func, cx, ty, a, b, depth + 1)
            || cx.env.sums.iter().any(|s| {
                s.ty == ty
                    && (a == s.p || a == s.q)
                    && cx.cfg.dominates(s.head, cx.block)
                    && edge_ule(func, cx.ranges, cx.path, ty, s.bound, b)
            })
            || cx.env.orders.iter().any(|o| {
                o.ty == ty
                    && (a == o.lo || a == o.hi)
                    && cx.cfg.dominates(o.head, cx.block)
                    && ((a == o.lo && edge_ule(func, cx.ranges, cx.path, ty, o.hi, b))
                        || (a == o.hi && edge_ule(func, cx.ranges, cx.path, ty, o.bound, b)))
            })
            || cx.env.rel_invs.iter().any(|r| {
                r.ty == ty
                    && a == r.param
                    && cx.cfg.dominates(r.head, cx.block)
                    && edge_ule(func, cx.ranges, cx.path, ty, r.bound, b)
            })
            || self.subs.iter().any(|sb| {
                sb.ty == ty && a == sb.idx && edge_ule(func, cx.ranges, cx.path, ty, sb.end, b)
            })
    }

    fn proves_ult(
        &self,
        func: &Function,
        cx: QueryCx<'_, '_>,
        ty: Type,
        a: Value,
        b: Value,
        depth: u8,
    ) -> bool {
        if cx.path.fact(cx.ranges, a).hi < cx.path.fact(cx.ranges, b).lo
            || cx
                .path
                .decide_rel(func, cx.ranges, IntCC::UnsignedLessThan, ty, a, b)
                == Some(true)
        {
            return true;
        }
        if depth > 2 {
            return false;
        }
        let next = depth + 1;
        self.proves_ult_rel(func, cx, ty, a, b, next)
            || self.proves_ult_sum(func, cx, ty, a, b, next)
            || self.proves_ult_order(func, cx, ty, a, b, next)
            || self.proves_ult_sub(func, cx, ty, a, b, next)
            || (ty == Type::I64 && self.proves_ult_ptr(func, cx, a, b, next))
    }

    fn proves_ult_rel(
        &self,
        func: &Function,
        cx: QueryCx<'_, '_>,
        ty: Type,
        a: Value,
        b: Value,
        next: u8,
    ) -> bool {
        cx.env.rel_invs.iter().any(|r| {
            r.ty == ty
                && a == r.param
                && cx.cfg.dominates(r.head, cx.block)
                && if r.kind == RelKind::Ult {
                    self.proves_ule(func, cx, ty, r.bound, b, next)
                } else {
                    self.proves_ult(func, cx, ty, r.bound, b, next)
                }
        })
    }

    fn proves_ult_sum(
        &self,
        func: &Function,
        cx: QueryCx<'_, '_>,
        ty: Type,
        a: Value,
        b: Value,
        next: u8,
    ) -> bool {
        for s in &cx.env.sums {
            if s.ty != ty
                || !cx.cfg.dominates(s.head, cx.block)
                || !self.proves_ule(func, cx, ty, s.bound, b, next)
            {
                continue;
            }
            for (p, other) in [(s.p, s.q), (s.q, s.p)] {
                let other_lo = edge_min_lo(cx.ranges, cx.env, cx.cfg, cx.path, cx.block, other);
                if a == p && other_lo >= 1 {
                    return true;
                }
                if let Some((delta, 0)) = add_delta_of(func, cx.ranges, a, p)
                    && (cx.path.fact(cx.ranges, delta).hi < other_lo
                        || cx.path.decide_rel(
                            func,
                            cx.ranges,
                            IntCC::UnsignedLessThan,
                            ty,
                            delta,
                            other,
                        ) == Some(true)
                        || (other_lo >= 1 && is_nonzero_ushr_of(func, cx.ranges, ty, delta, other)))
                {
                    return true;
                }
            }
        }
        false
    }

    fn proves_ult_order(
        &self,
        func: &Function,
        cx: QueryCx<'_, '_>,
        ty: Type,
        a: Value,
        b: Value,
        next: u8,
    ) -> bool {
        cx.env.orders.iter().any(|o| {
            o.ty == ty
                && cx.cfg.dominates(o.head, cx.block)
                && self.proves_ule(func, cx, ty, o.hi, b, next)
                && cx
                    .path
                    .decide_rel(func, cx.ranges, IntCC::UnsignedLessThan, ty, o.lo, o.hi)
                    == Some(true)
                && (a == o.lo || is_halved_mid(func, cx.ranges, ty, a, o.lo, o.hi))
        })
    }

    fn proves_ult_sub(
        &self,
        func: &Function,
        cx: QueryCx<'_, '_>,
        ty: Type,
        a: Value,
        b: Value,
        next: u8,
    ) -> bool {
        for sb in &self.subs {
            if sb.ty != ty || sb.min_diff == 0 || !self.proves_ule(func, cx, ty, sb.end, b, next) {
                continue;
            }
            if a == sb.idx {
                return true;
            }
            if let Some((c_val, extra)) = add_delta_of(func, cx.ranges, a, sb.idx)
                && cx
                    .path
                    .fact(cx.ranges, c_val)
                    .hi
                    .checked_add(extra)
                    .is_some_and(|total| total < sb.min_diff)
            {
                return true;
            }
        }
        false
    }

    fn proves_ult_ptr(
        &self,
        func: &Function,
        cx: QueryCx<'_, '_>,
        a: Value,
        b: Value,
        next: u8,
    ) -> bool {
        for p_inv in &cx.env.ptrs {
            if !cx.cfg.dominates(p_inv.head, cx.block)
                || !self.proves_ule(func, cx, Type::I64, p_inv.len, b, next)
            {
                continue;
            }
            let min_bytes = self
                .diffs
                .iter()
                .filter(|&&(de, dp, _)| de == p_inv.end && dp == p_inv.ptr)
                .map(|d| d.2)
                .max()
                .or_else(|| {
                    (p_inv.stride == 1
                        && cx.path.decide_rel(
                            func,
                            cx.ranges,
                            IntCC::NotEqual,
                            Type::I64,
                            p_inv.end,
                            p_inv.ptr,
                        ) == Some(true))
                    .then_some(1)
                });
            let Some(min_bytes) = min_bytes else { continue };
            if match_ptr_scaled_off(func, cx.ranges, p_inv, a) && min_bytes >= p_inv.stride {
                return true;
            }
            if let Some(&Op::Binary(Binary::Iadd, x, y)) = cx.ranges.op(func, a) {
                for (off_expr, c_val) in [(x, y), (y, x)] {
                    if match_ptr_scaled_off(func, cx.ranges, p_inv, off_expr)
                        && let Some(need) = cx
                            .path
                            .fact(cx.ranges, c_val)
                            .hi
                            .checked_add(1)
                            .and_then(|n| n.checked_mul(p_inv.stride))
                        && min_bytes >= need
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub(in super::super) fn decide(
        &self,
        func: &Function,
        ctx: (&Cfg, &Ranges, &Env, &Path, Block),
        cc: IntCC,
        x: Value,
        y: Value,
    ) -> Option<bool> {
        let ty = func.ty(x);
        if cc.is_signed() {
            return None;
        }
        let cx = QueryCx {
            cfg: ctx.0,
            ranges: ctx.1,
            env: ctx.2,
            path: ctx.3,
            block: ctx.4,
        };
        let (kind, a, b) = rel_of(cc, x, y);
        match kind {
            RelKind::Ult => {
                if self.proves_ult(func, cx, ty, a, b, 0) {
                    Some(true)
                } else if self.proves_ule(func, cx, ty, b, a, 0) {
                    Some(false)
                } else {
                    None
                }
            }
            RelKind::Ule => {
                if self.proves_ule(func, cx, ty, a, b, 0) {
                    Some(true)
                } else if self.proves_ult(func, cx, ty, b, a, 0) {
                    Some(false)
                } else {
                    None
                }
            }
            RelKind::Eq
                if self.proves_ult(func, cx, ty, a, b, 0)
                    || self.proves_ult(func, cx, ty, b, a, 0) =>
            {
                Some(false)
            }
            RelKind::Ne
                if self.proves_ult(func, cx, ty, a, b, 0)
                    || self.proves_ult(func, cx, ty, b, a, 0) =>
            {
                Some(true)
            }
            _ => None,
        }
    }

    pub(in super::super) fn cond_truth(
        &self,
        func: &Function,
        ctx: (&Cfg, &Ranges, &Env, &Path, Block),
        cond: Value,
        depth: u8,
    ) -> Option<bool> {
        if depth > 4 {
            return None;
        }
        let cx = QueryCx {
            cfg: ctx.0,
            ranges: ctx.1,
            env: ctx.2,
            path: ctx.3,
            block: ctx.4,
        };
        match cx.ranges.op(func, cond) {
            Some(&Op::Icmp(cc, x, y)) if func.ty(x).is_small_int() => {
                self.decide(func, ctx, cc, x, y)
            }
            Some(&Op::Binary(Binary::Bxor, a, b)) => {
                [(a, b), (b, a)].into_iter().find_map(|(inner, mask)| {
                    (cx.path.fact(cx.ranges, mask).exact() == Some(1)
                        && cx.path.fact(cx.ranges, inner).hi <= 1)
                        .then(|| self.cond_truth(func, ctx, inner, depth + 1).map(|t| !t))
                        .flatten()
                })
            }
            Some(&Op::Binary(Binary::Band, a, b))
                if cx.path.fact(cx.ranges, a).hi <= 1 && cx.path.fact(cx.ranges, b).hi <= 1 =>
            {
                let ta = cond_truth(func, cx.ranges, cx.path, a)
                    .or_else(|| self.cond_truth(func, ctx, a, depth + 1));
                let tb = cond_truth(func, cx.ranges, cx.path, b)
                    .or_else(|| self.cond_truth(func, ctx, b, depth + 1));
                match (ta, tb) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                }
            }
            Some(&Op::Binary(Binary::Bor, a, b)) => {
                let ta = cond_truth(func, cx.ranges, cx.path, a)
                    .or_else(|| self.cond_truth(func, ctx, a, depth + 1));
                let tb = cond_truth(func, cx.ranges, cx.path, b)
                    .or_else(|| self.cond_truth(func, ctx, b, depth + 1));
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
