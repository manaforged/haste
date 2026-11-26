use haste_codegen::{Lane, Type, Value};

use crate::body::{
    BinOp, BlockKind, Body, Expr, ExprId, List, Lit, Pat, PatId, PathId, Stmt, Ty as BodyTy, TyId,
    UnOp,
};
use crate::names::{Primitive, plain};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::infer::{Res, SiteKey, Typed, parse_int};
use crate::sema::ty::{Kind, int_info, mask as int_mask};

use super::super::Lower;
use super::super::expr::unwrap_expr;
use super::super::ops::parse_float;
use super::super::reloc::SiteOrigin;

impl Lower<'_, '_> {
    fn call_arg_expr(&self, origin: &SiteOrigin, index: usize) -> Option<ExprId> {
        let SiteOrigin::Table(SiteKey::ExprRes(key)) = *origin else {
            return None;
        };
        let list = match *self.body.expr(key) {
            Expr::Call(_, list) => list,
            _ => self.body.exprs.iter().find_map(|expr| match *expr {
                Expr::Call(callee, list) if callee == key => Some(list),
                _ => None,
            })?,
        };
        self.body
            .list(list)
            .iter()
            .copied()
            .map(ExprId)
            .filter(|id| !self.typed.const_args.contains(id))
            .nth(index)
    }

    pub(in super::super) fn const_arg_u8(
        &mut self,
        origin: &SiteOrigin,
        index: usize,
        limit: u8,
    ) -> Option<u8> {
        let expr = self.call_arg_expr(origin, index)?;
        let mut ctx = ConstEval {
            cx: self.cx,
            origin_body: self.body,
            typed: self.typed,
        };
        let val = ctx.int(self.body, self.src, expr, &mut Vec::new(), 0)?;
        u8::try_from(val.val)
            .ok()
            .filter(|&n| (!val.signed || val.to_signed() >= 0) && n < limit)
    }

    pub(super) fn shuffle_const_mask(
        &mut self,
        origin: &SiteOrigin,
        count: usize,
    ) -> Option<[u8; 16]> {
        if count > 16 {
            return None;
        }
        let wide = self.shuffle_const_mask_wide(origin, count, count)?;
        let mut mask = [0u8; 16];
        mask[..count].copy_from_slice(&wide[..count]);
        Some(mask)
    }

    pub(in super::super) fn shuffle_const_mask_wide(
        &mut self,
        origin: &SiteOrigin,
        in_count: usize,
        out_count: usize,
    ) -> Option<[u8; 64]> {
        if out_count == 0 || out_count > 64 || in_count == 0 {
            return None;
        }
        let expr = self.call_arg_expr(origin, 2)?;
        let mut ctx = ConstEval {
            cx: self.cx,
            origin_body: self.body,
            typed: self.typed,
        };
        ctx.shuffle_array(
            self.body,
            self.src,
            expr,
            (in_count, out_count),
            &mut Vec::new(),
            0,
        )
    }

    pub(in super::super) fn const_splat_vec(
        &mut self,
        origin: &SiteOrigin,
        vec_ty: Type,
    ) -> Option<Value> {
        let expr = self.call_arg_expr(origin, 0)?;
        let lane = vec_ty.lane()?;
        let bits = if lane.is_int() {
            let mut ctx = ConstEval {
                cx: self.cx,
                origin_body: self.body,
                typed: self.typed,
            };
            ctx.int(self.body, self.src, expr, &mut Vec::new(), 0)?.val & int_mask(lane.bytes() * 8)
        } else {
            let lane_ty = match lane {
                Lane::F32 => Type::F32,
                Lane::F64 => Type::F64,
                _ => return None,
            };
            let inner = ConstEval {
                cx: self.cx,
                origin_body: self.body,
                typed: self.typed,
            }
            .unwrap_expr(self.body, expr);
            let Expr::Lit(Lit::Float, tok) = *self.body.expr(inner) else {
                return None;
            };
            parse_float(lane_ty, self.src.text(tok)).ok()? as u64 as u128
        };
        if bits == 0 {
            return Some(self.ins().iconst(vec_ty, 0));
        }
        let lane_bytes = lane.bytes() as usize;
        let mut bytes = vec![0u8; vec_ty.bytes() as usize];
        for chunk in bytes.chunks_exact_mut(lane_bytes) {
            chunk.copy_from_slice(&bits.to_le_bytes()[..lane_bytes]);
        }
        let addr = self.emit_const_rodata(&bytes);
        Some(self.ins().load(vec_ty, addr, 0))
    }
}

#[derive(Clone, Copy)]
struct ConstVal {
    val: u128,
    bits: u32,
    signed: bool,
    typed: bool,
}

impl ConstVal {
    fn to_signed(self) -> i128 {
        let shift = 128 - self.bits;
        ((self.val << shift) as i128) >> shift
    }

    fn cast(self, (bits, signed): (u32, bool)) -> Self {
        let extended = if self.signed {
            self.to_signed() as u128
        } else {
            self.val
        };
        Self {
            val: extended & int_mask(bits),
            bits,
            signed,
            typed: true,
        }
    }
}

fn is_shuffle_idx_path(body: &Body, src: &Source, path: PathId) -> bool {
    body.list(body.path(path).segments)
        .last()
        .is_some_and(|&s| plain(src.text(body.segments[s as usize].name)) == "SimdShuffleIdx")
}

fn pat_bind_name(body: &Body, src: &Source, mut pat: PatId) -> Option<String> {
    while let Pat::Paren(inner) = *body.pat(pat) {
        pat = inner;
    }
    match *body.pat(pat) {
        Pat::Bind {
            name, sub: None, ..
        } => Some(plain(src.text(name)).to_owned()),
        _ => None,
    }
}

struct ConstEval<'a, 'b> {
    cx: &'b mut Context<'a>,
    origin_body: &'b Body,
    typed: &'b Typed,
}

impl<'a, 'b> ConstEval<'a, 'b> {
    fn unwrap_expr(&self, body: &Body, expr: ExprId) -> ExprId {
        unwrap_expr(body, expr, |kind, _| {
            matches!(
                kind,
                BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const | BlockKind::Expansion
            )
        })
    }

    fn unwrap_shuffle(&self, body: &Body, src: &Source, mut expr: ExprId) -> ExprId {
        loop {
            expr = self.unwrap_expr(body, expr);
            match *body.expr(expr) {
                Expr::Call(callee, list) => {
                    let &[arg] = body.list(list) else {
                        return expr;
                    };
                    let is_ctor = std::ptr::eq(body, self.origin_body)
                        && matches!(
                            self.typed
                                .res
                                .get(&callee)
                                .or_else(|| self.typed.res.get(&expr)),
                            Some(Res::Ctor(..))
                        );
                    let is_idx = matches!(*body.expr(callee), Expr::Path(path) if is_shuffle_idx_path(body, src, path));
                    if is_ctor || is_idx {
                        expr = ExprId(arg);
                    } else {
                        return expr;
                    }
                }
                Expr::Struct {
                    path,
                    fields,
                    base: None,
                    ..
                } => {
                    let &[field] = body.list(fields) else {
                        return expr;
                    };
                    if is_shuffle_idx_path(body, src, path) {
                        expr = body.fields[field as usize].value;
                    } else {
                        return expr;
                    }
                }
                _ => return expr,
            }
        }
    }

    fn body_ty_int(&self, body: &Body, src: &Source, mut ty: TyId) -> Option<(u32, bool)> {
        loop {
            match *body.ty(ty) {
                BodyTy::Paren(inner) => ty = inner,
                BodyTy::Path(path) => {
                    let entry = body.path(path);
                    if entry.global || entry.qself.is_some() {
                        return None;
                    }
                    let &[seg] = body.list(entry.segments) else {
                        return None;
                    };
                    return Primitive::named(plain(src.text(body.segments[seg as usize].name)))
                        .and_then(int_info);
                }
                _ => return None,
            }
        }
    }

    fn origin_int(&self, expr: ExprId) -> Option<(u32, bool)> {
        match *self.cx.types.kind(self.typed.exprs[expr.0 as usize]) {
            Kind::Prim(prim) => int_info(prim),
            _ => None,
        }
    }

    fn with_const_item<T>(
        &mut self,
        expr: ExprId,
        f: impl FnOnce(&mut Self, &Body, &Source, ExprId) -> Option<T>,
    ) -> Option<T> {
        let Res::Fn(item, substs) = self.typed.res.get(&expr).copied()? else {
            return None;
        };
        if self.cx.item(item).1.kind != ItemKind::Const || !self.cx.types.substs(substs).is_empty()
        {
            return None;
        }
        let item_body = self.cx.body(item).ok()?;
        let root = item_body.root?;
        let item_src = self.cx.body_src(item);
        f(self, &item_body, item_src, root)
    }

    fn eval_call(
        &mut self,
        body: &Body,
        src: &Source,
        (expr, callee, list): (ExprId, ExprId, List),
        env: &mut Vec<(String, ConstVal)>,
        depth: u8,
    ) -> Option<ConstVal> {
        let Res::Fn(item, substs) = self
            .typed
            .res
            .get(&callee)
            .or_else(|| self.typed.res.get(&expr))
            .copied()?
        else {
            return None;
        };
        if self.cx.item(item).1.kind != ItemKind::Fn
            || !self.cx.types.substs(substs).is_empty()
            || self.cx.intrinsic(item).is_some()
        {
            return None;
        }
        let sig = self.cx.signature(item).ok()?;
        let arg_ids = body.list(list);
        if sig.params.len() != arg_ids.len() {
            return None;
        }
        let item_body = self.cx.body(item).ok()?;
        let root = item_body.root?;
        let item_src = self.cx.body_src(item);
        let mut sub_env = Vec::with_capacity(arg_ids.len());
        for (&(pat, pty), &arg_id) in sig.params.iter().zip(arg_ids) {
            let name = pat_bind_name(&sig.body, item_src, pat)?;
            let mut val = self.int(body, src, ExprId(arg_id), env, depth + 1)?;
            if let Some(target) = self.body_ty_int(&sig.body, item_src, pty) {
                val = val.cast(target);
            }
            sub_env.push((name, val));
        }
        let mut res = self.int(&item_body, item_src, root, &mut sub_env, depth + 1)?;
        if let Some(target) = sig
            .ret
            .and_then(|rt| self.body_ty_int(&sig.body, item_src, rt))
        {
            res = res.cast(target);
        }
        Some(res)
    }

    fn eval_path(
        &mut self,
        body: &Body,
        src: &Source,
        (expr, path): (ExprId, PathId),
        env: &[(String, ConstVal)],
        depth: u8,
    ) -> Option<ConstVal> {
        let entry = body.path(path);
        if !entry.global
            && entry.qself.is_none()
            && let &[seg] = body.list(entry.segments)
        {
            let name = plain(src.text(body.segments[seg as usize].name));
            if let Some((_, val)) = env.iter().rev().find(|(bound, _)| bound == name) {
                return Some(*val);
            }
        }
        if !std::ptr::eq(body, self.origin_body) {
            return None;
        }
        match self.typed.res.get(&expr).copied()? {
            Res::Const(val) => match *self.cx.types.kind(val) {
                Kind::Value(n) => {
                    let (bits, signed) = self.origin_int(expr).unwrap_or((64, false));
                    Some(ConstVal {
                        val: n & int_mask(bits),
                        bits,
                        signed,
                        typed: true,
                    })
                }
                _ => None,
            },
            Res::Fn(..) => self.with_const_item(expr, |this, b, s, root| {
                this.int(b, s, root, &mut Vec::new(), depth + 1)
            }),
            _ => None,
        }
    }

    fn int(
        &mut self,
        body: &Body,
        src: &Source,
        expr: ExprId,
        env: &mut Vec<(String, ConstVal)>,
        depth: u8,
    ) -> Option<ConstVal> {
        if depth > 8 {
            return None;
        }
        let expr = self.unwrap_expr(body, expr);
        match *body.expr(expr) {
            Expr::Lit(Lit::Int, tok) => {
                let (val, suffix) = parse_int(src.text(tok)).ok()?;
                let info = suffix.and_then(int_info).or_else(|| {
                    std::ptr::eq(body, self.origin_body)
                        .then(|| self.origin_int(expr))
                        .flatten()
                });
                let (bits, signed, typed) = match info {
                    Some((b, s)) => (b, s, true),
                    None => (64, false, false),
                };
                Some(ConstVal {
                    val: val & int_mask(bits),
                    bits,
                    signed,
                    typed,
                })
            }
            Expr::Lit(Lit::Bool(flag), _) => Some(ConstVal {
                val: u128::from(flag),
                bits: 1,
                signed: false,
                typed: true,
            }),
            Expr::Unary(UnOp::Neg, inner) => {
                let v = self.int(body, src, inner, env, depth + 1)?;
                Some(ConstVal {
                    val: v.val.wrapping_neg() & int_mask(v.bits),
                    ..v
                })
            }
            Expr::Unary(UnOp::Not, inner) => {
                let v = self.int(body, src, inner, env, depth + 1)?;
                Some(ConstVal {
                    val: (!v.val) & int_mask(v.bits),
                    ..v
                })
            }
            Expr::Cast(inner, ty) => {
                let v = self.int(body, src, inner, env, depth + 1)?;
                let target = self.body_ty_int(body, src, ty)?;
                Some(v.cast(target))
            }
            Expr::Binary(op, lhs, rhs) => self.binary(body, src, (op, lhs, rhs), env, depth + 1),
            Expr::If(cond, then, Some(otherwise)) => {
                let flag = self.int(body, src, cond, env, depth + 1)?;
                self.int(
                    body,
                    src,
                    if flag.val != 0 { then } else { otherwise },
                    env,
                    depth + 1,
                )
            }
            Expr::Block {
                stmts,
                tail: Some(tail),
                kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const | BlockKind::Expansion,
                ..
            } => {
                let mark = self.eval_stmts(body, src, stmts, env, depth + 1)?;
                let out = self.int(body, src, tail, env, depth + 1);
                env.truncate(mark);
                out
            }
            Expr::Index(arr, idx_expr) => {
                let idx_val = self.int(body, src, idx_expr, env, depth + 1)?;
                let idx = usize::try_from(idx_val.val).ok()?;
                self.index_array(body, src, arr, idx, env, depth + 1)
            }
            Expr::Call(callee, list) if std::ptr::eq(body, self.origin_body) => {
                self.eval_call(body, src, (expr, callee, list), env, depth)
            }
            Expr::Path(path) => self.eval_path(body, src, (expr, path), env, depth),
            _ => None,
        }
    }

    fn binary(
        &mut self,
        body: &Body,
        src: &Source,
        (op, lhs, rhs): (BinOp, ExprId, ExprId),
        env: &mut Vec<(String, ConstVal)>,
        depth: u8,
    ) -> Option<ConstVal> {
        let mut l = self.int(body, src, lhs, env, depth)?;
        if matches!(op, BinOp::And | BinOp::Or) {
            let short = (l.val != 0) == (op == BinOp::Or);
            let out = if short {
                l.val != 0
            } else {
                self.int(body, src, rhs, env, depth)?.val != 0
            };
            return Some(ConstVal {
                val: u128::from(out),
                bits: 1,
                signed: false,
                typed: true,
            });
        }
        let mut r = self.int(body, src, rhs, env, depth)?;
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            let sh = u32::try_from(r.val).ok()?;
            let val = match op {
                BinOp::Shl => l.val.checked_shl(sh).unwrap_or(0) & int_mask(l.bits),
                _ if l.signed => ((l.to_signed() >> sh.min(127)) as u128) & int_mask(l.bits),
                _ => l.val.checked_shr(sh).unwrap_or(0),
            };
            return Some(ConstVal { val, ..l });
        }
        let (bits, signed, typed) = match (l.typed, r.typed) {
            (true, _) => (l.bits, l.signed, true),
            (false, true) => (r.bits, r.signed, true),
            (false, false) => (64, false, false),
        };
        l = l.cast((bits, signed));
        r = r.cast((bits, signed));
        if matches!(
            op,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        ) {
            let ord = if signed {
                l.to_signed().cmp(&r.to_signed())
            } else {
                l.val.cmp(&r.val)
            };
            let holds = match op {
                BinOp::Eq => ord.is_eq(),
                BinOp::Ne => ord.is_ne(),
                BinOp::Lt => ord.is_lt(),
                BinOp::Le => ord.is_le(),
                BinOp::Gt => ord.is_gt(),
                _ => ord.is_ge(),
            };
            return Some(ConstVal {
                val: u128::from(holds),
                bits: 1,
                signed: false,
                typed: true,
            });
        }
        if matches!(op, BinOp::Div | BinOp::Rem) && r.val == 0 {
            return None;
        }
        let val = match op {
            BinOp::Add => l.val.wrapping_add(r.val),
            BinOp::Sub => l.val.wrapping_sub(r.val),
            BinOp::Mul => l.val.wrapping_mul(r.val),
            BinOp::Div if signed => l.to_signed().wrapping_div(r.to_signed()) as u128,
            BinOp::Rem if signed => l.to_signed().wrapping_rem(r.to_signed()) as u128,
            BinOp::Div => l.val / r.val,
            BinOp::Rem => l.val % r.val,
            BinOp::BitAnd => l.val & r.val,
            BinOp::BitOr => l.val | r.val,
            BinOp::BitXor => l.val ^ r.val,
            _ => return None,
        };
        Some(ConstVal {
            val: val & int_mask(bits),
            bits,
            signed,
            typed,
        })
    }

    fn index_array(
        &mut self,
        body: &Body,
        src: &Source,
        arr: ExprId,
        idx: usize,
        env: &mut Vec<(String, ConstVal)>,
        depth: u8,
    ) -> Option<ConstVal> {
        let arr = self.unwrap_expr(body, arr);
        match *body.expr(arr) {
            Expr::Array(list) => {
                let &elem = body.list(list).get(idx)?;
                self.int(body, src, ExprId(elem), env, depth)
            }
            Expr::Repeat(elem, count_expr) => {
                let count = self.int(body, src, count_expr, env, depth)?;
                (u128::from(idx as u64) < count.val)
                    .then(|| self.int(body, src, elem, env, depth))
                    .flatten()
            }
            Expr::Path(_) if std::ptr::eq(body, self.origin_body) => self
                .with_const_item(arr, |this, b, s, root| {
                    this.index_array(b, s, root, idx, &mut Vec::new(), depth)
                }),
            _ => None,
        }
    }

    fn eval_stmts(
        &mut self,
        body: &Body,
        src: &Source,
        stmts: List,
        env: &mut Vec<(String, ConstVal)>,
        depth: u8,
    ) -> Option<usize> {
        let mark = env.len();
        for &stmt_id in body.list(stmts) {
            match body.stmts[stmt_id as usize] {
                Stmt::Empty | Stmt::Item(..) => {}
                Stmt::Let {
                    pat,
                    ty,
                    init: Some(init),
                    otherwise: None,
                    ..
                } => {
                    let Some(name) = pat_bind_name(body, src, pat) else {
                        env.truncate(mark);
                        return None;
                    };
                    let Some(mut val) = self.int(body, src, init, env, depth) else {
                        env.truncate(mark);
                        return None;
                    };
                    if let Some(target) = ty.and_then(|t| self.body_ty_int(body, src, t)) {
                        val = val.cast(target);
                    }
                    env.push((name, val));
                }
                _ => {
                    env.truncate(mark);
                    return None;
                }
            }
        }
        Some(mark)
    }

    fn shuffle_array(
        &mut self,
        body: &Body,
        src: &Source,
        expr: ExprId,
        (in_count, out_count): (usize, usize),
        env: &mut Vec<(String, ConstVal)>,
        depth: u8,
    ) -> Option<[u8; 64]> {
        if depth > 4 {
            return None;
        }
        let expr = self.unwrap_shuffle(body, src, expr);
        let limit = u128::try_from(2 * in_count).ok()?;
        let mut mask = [0u8; 64];
        match *body.expr(expr) {
            Expr::Array(list) => {
                let ids = body.list(list);
                if ids.len() != out_count {
                    return None;
                }
                mask[..out_count]
                    .iter_mut()
                    .zip(ids)
                    .try_for_each(|(slot, &id)| {
                        let v = self.int(body, src, ExprId(id), env, depth + 1)?;
                        ((!v.signed || v.to_signed() >= 0) && v.val < limit)
                            .then(|| *slot = v.val as u8)
                    })?;
                Some(mask)
            }
            Expr::Repeat(elem, _) => {
                let v = self.int(body, src, elem, env, depth + 1)?;
                if (v.signed && v.to_signed() < 0) || v.val >= limit {
                    return None;
                }
                mask[..out_count].fill(v.val as u8);
                Some(mask)
            }
            Expr::Block {
                stmts,
                tail: Some(tail),
                kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const | BlockKind::Expansion,
                ..
            } => {
                let mark = self.eval_stmts(body, src, stmts, env, depth + 1)?;
                let out =
                    self.shuffle_array(body, src, tail, (in_count, out_count), env, depth + 1);
                env.truncate(mark);
                out
            }
            Expr::Path(_) if std::ptr::eq(body, self.origin_body) => {
                self.with_const_item(expr, |this, b, s, root| {
                    this.shuffle_array(
                        b,
                        s,
                        root,
                        (in_count, out_count),
                        &mut Vec::new(),
                        depth + 1,
                    )
                })
            }
            _ => None,
        }
    }
}
