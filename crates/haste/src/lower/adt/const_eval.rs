use rustc_abi::FieldsShape;

use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, Lit, UnOp};
use crate::names::{ItemRef, Primitive as PrimName};
use crate::outline::item::ItemKind;
use crate::sema::infer::{Res, field_types, parse_int};
use crate::sema::layout::field_offset;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, int_info, mask};

use super::super::Lower;
use super::super::promote::{eval_lit_bits, neg_float_bits};
use super::{ConstScope, tag_bits};

const CONST_RODATA_MAX_DEPTH: u32 = 16;

impl<'l, 'a> Lower<'l, 'a> {
    pub(super) fn write_const_bytes(
        &mut self,
        scope: ConstScope<'_>,
        (expr, ty): (ExprId, Ty),
        at: usize,
        out: &mut [u8],
        depth: u32,
        in_generic: bool,
    ) -> Option<()> {
        if scope.typed.coercions.contains_key(&expr) {
            return None;
        }
        self.write_const_uncoerced(scope, (expr, ty), at, out, depth, in_generic)
    }

    pub(super) fn write_const_uncoerced(
        &mut self,
        scope: ConstScope<'_>,
        (expr, ty): (ExprId, Ty),
        at: usize,
        out: &mut [u8],
        depth: u32,
        in_generic: bool,
    ) -> Option<()> {
        if depth > CONST_RODATA_MAX_DEPTH {
            return None;
        }
        match *scope.body.expr(expr) {
            Expr::Paren(inner) => {
                return self.write_const_bytes(scope, (inner, ty), at, out, depth, in_generic);
            }
            Expr::Block {
                stmts,
                tail: Some(tail),
                label: None,
                kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const,
            } if scope.body.list(stmts).is_empty() => {
                return self.write_const_bytes(scope, (tail, ty), at, out, depth, in_generic);
            }
            Expr::Path(_) => {
                if let Some(&Res::Fn(item, substs)) = scope.typed.res.get(&expr)
                    && self.cx.item(item).1.kind == ItemKind::Const
                {
                    let (body, src, item_typed, root) = self
                        .load_const_item(item, substs, ty, in_generic, false)
                        .ok()??;
                    let scoped = item_typed.scoped(&self.cx.types);
                    return self.write_const_bytes(
                        ConstScope {
                            body: &body,
                            src,
                            typed: &scoped,
                            in_const: true,
                        },
                        (root, ty),
                        at,
                        out,
                        depth + 1,
                        in_generic,
                    );
                }
            }
            _ => {}
        }
        let kind = self.cx.types.kind(ty).clone();
        match (*scope.body.expr(expr), kind) {
            (_, Kind::Pat(inner, ..)) => {
                self.write_const_uncoerced(scope, (expr, inner), at, out, depth, in_generic)
            }
            (Expr::Array(list), Kind::Array(elem, _)) => {
                let stride = usize::try_from(self.cx.layout(elem).ok()?.size.bytes()).ok()?;
                let items: Vec<ExprId> = scope
                    .body
                    .list(list)
                    .iter()
                    .map(|&item| ExprId(item))
                    .collect();
                items.into_iter().enumerate().try_for_each(|(index, item)| {
                    self.write_const_bytes(
                        scope,
                        (item, elem),
                        at + index * stride,
                        out,
                        depth,
                        in_generic,
                    )
                })
            }
            (Expr::Repeat(item, count_expr), Kind::Array(elem, count)) => {
                if in_generic && !matches!(*scope.body.expr(count_expr), Expr::Lit(Lit::Int, _)) {
                    return None;
                }
                let stride = usize::try_from(self.cx.layout(elem).ok()?.size.bytes()).ok()?;
                let count = usize::try_from(self.cx.types.len(count)?).ok()?;
                if count == 0 {
                    let mut scratch = vec![0u8; stride];
                    self.write_const_bytes(
                        scope,
                        (item, elem),
                        0,
                        &mut scratch,
                        depth,
                        in_generic,
                    )?;
                    return Some(());
                }
                if at.checked_add(count.checked_mul(stride)?)? > out.len() {
                    return None;
                }
                self.write_const_bytes(scope, (item, elem), at, out, depth, in_generic)?;
                for index in 1..count {
                    out.copy_within(at..at + stride, at + index * stride);
                }
                Some(())
            }
            (Expr::Tuple(list), Kind::Tuple(items)) => {
                let layout = self.cx.layout(ty).ok()?;
                let items = self.cx.types.substs(items).to_vec();
                let elems: Vec<ExprId> = scope
                    .body
                    .list(list)
                    .iter()
                    .map(|&item| ExprId(item))
                    .collect();
                if elems.len() != items.len() {
                    return None;
                }
                elems.into_iter().enumerate().try_for_each(|(index, item)| {
                    let offset = usize::try_from(field_offset(&layout, 0, index as u32)).ok()?;
                    self.write_const_bytes(
                        scope,
                        (item, items[index]),
                        at + offset,
                        out,
                        depth,
                        in_generic,
                    )
                })
            }
            (_, Kind::Adt(item, args)) => self.construct_const_bytes(
                scope,
                (expr, ty),
                (item, args),
                at,
                out,
                depth,
                in_generic,
            ),
            (_, Kind::Prim(primitive)) => {
                let (size, bits) =
                    self.eval_const_scalar(scope, expr, primitive, depth, in_generic)?;
                out.get_mut(at..at + size)?
                    .copy_from_slice(&bits.to_le_bytes()[..size]);
                Some(())
            }
            _ => None,
        }
    }

    fn construct_const_bytes(
        &mut self,
        scope: ConstScope<'_>,
        (expr, ty): (ExprId, Ty),
        (item, args): (ItemRef, Substs),
        at: usize,
        out: &mut [u8],
        depth: u32,
        in_generic: bool,
    ) -> Option<()> {
        if in_generic && !self.cx.types.substs(args).is_empty() {
            return None;
        }
        let (variant, fields) = const_ctor(scope, expr)?;
        let layout = self.cx.layout(ty).ok()?;
        if matches!(layout.fields, FieldsShape::Union(_)) {
            return None;
        }
        let args = self.cx.types.substs(args).to_vec();
        let types = field_types(self.cx, item, &args)
            .ok()?
            .get(variant as usize)?
            .clone();
        for (index, value) in fields {
            let &field_ty = types.get(index as usize)?;
            let offset = usize::try_from(field_offset(&layout, variant, index)).ok()?;
            self.write_const_bytes(
                scope,
                (value, field_ty),
                at + offset,
                out,
                depth,
                in_generic,
            )?;
        }
        if let Some((offset, tag, bits)) = tag_bits(self.cx, item, &layout, variant).ok()? {
            let size = tag.bytes() as usize;
            let start = at + usize::try_from(offset).ok()?;
            out.get_mut(start..start + size)?
                .copy_from_slice(&bits.to_le_bytes()[..size]);
        }
        Some(())
    }

    fn eval_const_path_scalar(
        &mut self,
        scope: ConstScope<'_>,
        expr: ExprId,
        primitive: PrimName,
        depth: u32,
        in_generic: bool,
    ) -> Option<(usize, u128)> {
        match scope.typed.res.get(&expr) {
            Some(&Res::Const(val_ty)) if !in_generic => {
                let Kind::Value(bits) = *self.cx.types.kind(val_ty) else {
                    return None;
                };
                mask_const_scalar(primitive, bits)
            }
            Some(&Res::Fn(item, substs))
                if depth < CONST_RODATA_MAX_DEPTH
                    && self.cx.item(item).1.kind == ItemKind::Const =>
            {
                let prim_ty = self.cx.types.prim(primitive);
                let (body, src, item_typed, root) = self
                    .load_const_item(item, substs, prim_ty, in_generic, false)
                    .ok()??;
                let scoped = item_typed.scoped(&self.cx.types);
                self.eval_const_scalar(
                    ConstScope {
                        body: &body,
                        src,
                        typed: &scoped,
                        in_const: true,
                    },
                    root,
                    primitive,
                    depth + 1,
                    in_generic,
                )
            }
            _ => None,
        }
    }

    fn eval_const_scalar(
        &mut self,
        scope: ConstScope<'_>,
        expr: ExprId,
        primitive: PrimName,
        depth: u32,
        in_generic: bool,
    ) -> Option<(usize, u128)> {
        if depth > CONST_RODATA_MAX_DEPTH {
            return None;
        }
        let expr = unwrap_const_uncoerced(scope, expr)?;
        match *scope.body.expr(expr) {
            Expr::Path(_) => self.eval_const_path_scalar(scope, expr, primitive, depth, in_generic),
            Expr::Lit(lit, at) => eval_lit_bits(scope.src, lit, at, primitive),
            Expr::Unary(op, inner) if !scope.typed.operators.contains_key(&expr) => {
                self.eval_const_unary(scope, op, inner, primitive, depth, in_generic)
            }
            Expr::Binary(op, lhs, rhs) if !scope.typed.operators.contains_key(&expr) => {
                self.eval_const_binary(scope, op, lhs, rhs, primitive, depth, in_generic)
            }
            Expr::Index(arr, idx_expr) if !scope.typed.operators.contains_key(&expr) => {
                self.eval_const_index(scope, arr, idx_expr, primitive, depth, in_generic)
            }
            Expr::Cast(inner, _) => {
                let Kind::Prim(src_prim) = *self
                    .cx
                    .types
                    .kind(scope.typed.expr_ty(&self.cx.types, inner))
                else {
                    return None;
                };
                cast_const_scalar(
                    self.eval_const_scalar(scope, inner, src_prim, depth, in_generic)?
                        .1,
                    src_prim,
                    primitive,
                )
            }
            _ => None,
        }
    }

    fn eval_const_index(
        &mut self,
        scope: ConstScope<'_>,
        arr: ExprId,
        idx_expr: ExprId,
        primitive: PrimName,
        depth: u32,
        in_generic: bool,
    ) -> Option<(usize, u128)> {
        if depth >= CONST_RODATA_MAX_DEPTH {
            return None;
        }
        let idx = usize::try_from(
            self.eval_const_scalar(scope, idx_expr, PrimName::Usize, depth + 1, in_generic)?
                .1,
        )
        .ok()?;
        let arr_ty = scope.typed.expr_ty(&self.cx.types, arr);
        let Kind::Array(elem_ty, count) = *self.cx.types.kind(arr_ty) else {
            return None;
        };
        if !matches!(*self.cx.types.kind(elem_ty), Kind::Prim(p) if p == primitive)
            || idx >= usize::try_from(self.cx.types.len(count)?).ok()?
        {
            return None;
        }
        self.eval_const_array_elem(scope, arr, arr_ty, primitive, idx, depth, in_generic)
    }

    fn eval_const_array_elem(
        &mut self,
        scope: ConstScope<'_>,
        arr: ExprId,
        arr_ty: Ty,
        primitive: PrimName,
        idx: usize,
        depth: u32,
        in_generic: bool,
    ) -> Option<(usize, u128)> {
        let arr = unwrap_const_uncoerced(scope, arr)?;
        match *scope.body.expr(arr) {
            Expr::Path(_) => {
                let Some(&Res::Fn(item, substs)) = scope.typed.res.get(&arr) else {
                    return None;
                };
                if depth >= CONST_RODATA_MAX_DEPTH || self.cx.item(item).1.kind != ItemKind::Const {
                    return None;
                }
                let (body, src, item_typed, root) = self
                    .load_const_item(item, substs, arr_ty, in_generic, false)
                    .ok()??;
                let scoped = item_typed.scoped(&self.cx.types);
                self.eval_const_array_elem(
                    ConstScope {
                        body: &body,
                        src,
                        typed: &scoped,
                        in_const: true,
                    },
                    root,
                    arr_ty,
                    primitive,
                    idx,
                    depth + 1,
                    in_generic,
                )
            }
            Expr::Repeat(elem, count_expr)
                if !in_generic
                    || matches!(*scope.body.expr(count_expr), Expr::Lit(Lit::Int, _)) =>
            {
                self.eval_const_scalar(scope, elem, primitive, depth + 1, in_generic)
            }
            Expr::Array(list) => {
                let items = scope.body.list(list);
                if scope.in_const {
                    return self.eval_const_scalar(
                        scope,
                        ExprId(*items.get(idx)?),
                        primitive,
                        depth + 1,
                        in_generic,
                    );
                }
                if items.len() > 16 {
                    return None;
                }
                items
                    .to_vec()
                    .into_iter()
                    .enumerate()
                    .try_fold(None, |cur, (i, elem)| {
                        let val = self.eval_const_scalar(
                            scope,
                            ExprId(elem),
                            primitive,
                            depth + 1,
                            in_generic,
                        )?;
                        Some(if i == idx { Some(val) } else { cur })
                    })?
            }
            _ => None,
        }
    }

    fn eval_const_binary(
        &mut self,
        scope: ConstScope<'_>,
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
        primitive: PrimName,
        depth: u32,
        in_generic: bool,
    ) -> Option<(usize, u128)> {
        if depth >= CONST_RODATA_MAX_DEPTH {
            return None;
        }
        if matches!(
            op,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        ) && primitive == PrimName::Bool
        {
            let Kind::Prim(cmp_prim) =
                *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, lhs))
            else {
                return None;
            };
            let (bits, signed) = const_prim_info(cmp_prim)?;
            let l = self
                .eval_const_scalar(scope, lhs, cmp_prim, depth + 1, in_generic)?
                .1;
            let r = self
                .eval_const_scalar(scope, rhs, cmp_prim, depth + 1, in_generic)?
                .1;
            return Some((1, u128::from(eval_const_cmp(op, bits, signed, l, r))));
        }
        if matches!(op, BinOp::And | BinOp::Or) && primitive == PrimName::Bool {
            let (_, l) =
                self.eval_const_scalar(scope, lhs, PrimName::Bool, depth + 1, in_generic)?;
            if (op == BinOp::And) == (l == 0) {
                return Some((1, l));
            }
            return self.eval_const_scalar(scope, rhs, PrimName::Bool, depth + 1, in_generic);
        }
        if matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor) && primitive == PrimName::Bool
        {
            let l = self
                .eval_const_scalar(scope, lhs, PrimName::Bool, depth + 1, in_generic)?
                .1;
            let r = self
                .eval_const_scalar(scope, rhs, PrimName::Bool, depth + 1, in_generic)?
                .1;
            return Some((
                1,
                (match op {
                    BinOp::BitAnd => l & r,
                    BinOp::BitOr => l | r,
                    _ => l ^ r,
                }) & 1,
            ));
        }
        let (bits, signed) = int_info(primitive)?;
        let l = self
            .eval_const_scalar(scope, lhs, primitive, depth + 1, in_generic)?
            .1;
        let rhs_prim = if matches!(op, BinOp::Shl | BinOp::Shr) {
            let Kind::Prim(p) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, rhs))
            else {
                return None;
            };
            p
        } else {
            primitive
        };
        let (rhs_bits, rhs_signed) = int_info(rhs_prim)?;
        let r = self
            .eval_const_scalar(scope, rhs, rhs_prim, depth + 1, in_generic)?
            .1;
        Some((
            bits as usize / 8,
            eval_const_int_bin(
                op,
                (bits, signed),
                l,
                (rhs_bits, rhs_signed),
                r,
                self.overflow_checks,
            )?,
        ))
    }

    fn eval_const_unary(
        &mut self,
        scope: ConstScope<'_>,
        op: UnOp,
        inner: ExprId,
        primitive: PrimName,
        depth: u32,
        in_generic: bool,
    ) -> Option<(usize, u128)> {
        match (op, primitive) {
            (UnOp::Not, PrimName::Bool) => Some((
                1,
                u128::from(
                    self.eval_const_scalar(scope, inner, PrimName::Bool, depth, in_generic)?
                        .1
                        == 0,
                ),
            )),
            (UnOp::Not, _) => {
                let (bits, _) = int_info(primitive)?;
                let (size, val) =
                    self.eval_const_scalar(scope, inner, primitive, depth, in_generic)?;
                Some((size, (!val) & mask(bits)))
            }
            (UnOp::Neg, PrimName::F32 | PrimName::F64) => {
                let (_, bits) =
                    self.eval_const_scalar(scope, inner, primitive, depth, in_generic)?;
                neg_float_bits(primitive, bits)
            }
            (UnOp::Neg, _) => {
                let (bits, true) = int_info(primitive)? else {
                    return None;
                };
                if let Some(at) = bare_int_lit(scope.body, inner) {
                    let val = parse_int(scope.src.text(at)).ok()?.0;
                    return Some((bits as usize / 8, val.wrapping_neg() & mask(bits)));
                }
                let (size, val) =
                    self.eval_const_scalar(scope, inner, primitive, depth, in_generic)?;
                if self.overflow_checks && val == (1u128 << (bits - 1)) {
                    return None;
                }
                Some((size, val.wrapping_neg() & mask(bits)))
            }
            _ => None,
        }
    }
}

fn const_ctor(scope: ConstScope<'_>, expr: ExprId) -> Option<(u32, Vec<(u32, ExprId)>)> {
    let variant = |at: ExprId| match scope.typed.res.get(&at) {
        Some(&Res::Ctor(_, variant, _)) => Some(variant),
        _ => None,
    };
    match *scope.body.expr(expr) {
        Expr::Path(_) => Some((variant(expr)?, Vec::new())),
        Expr::Call(callee, args) => Some((
            variant(expr).or_else(|| variant(callee))?,
            scope
                .body
                .list(args)
                .iter()
                .enumerate()
                .map(|(index, &at)| (index as u32, ExprId(at)))
                .collect(),
        )),
        Expr::Struct {
            fields, base: None, ..
        } => {
            let fields = scope
                .body
                .list(fields)
                .iter()
                .map(|&at| scope.body.fields[at as usize].value)
                .map(|value| Some((*scope.typed.inits.get(&value)?, value)))
                .collect::<Option<Vec<_>>>()?;
            Some((variant(expr)?, fields))
        }
        _ => None,
    }
}

fn bare_int_lit(body: &Body, mut expr: ExprId) -> Option<u32> {
    loop {
        match *body.expr(expr) {
            Expr::Paren(inner) => expr = inner,
            Expr::Lit(Lit::Int, at) => return Some(at),
            _ => return None,
        }
    }
}

fn const_prim_info(prim: PrimName) -> Option<(u32, bool)> {
    match prim {
        PrimName::Bool => Some((1, false)),
        PrimName::Char => Some((32, false)),
        _ => int_info(prim),
    }
}

fn mask_const_scalar(primitive: PrimName, bits: u128) -> Option<(usize, u128)> {
    match primitive {
        PrimName::F32 => Some((4, bits & 0xFFFF_FFFF)),
        PrimName::F64 => Some((8, bits & 0xFFFF_FFFF_FFFF_FFFF)),
        _ => {
            let (width, _) = const_prim_info(primitive)?;
            let size = if primitive == PrimName::Bool {
                1
            } else {
                width as usize / 8
            };
            Some((size, bits & mask(width)))
        }
    }
}

fn unwrap_const_uncoerced(scope: ConstScope<'_>, mut expr: ExprId) -> Option<ExprId> {
    loop {
        if scope.typed.coercions.contains_key(&expr) {
            return None;
        }
        match *scope.body.expr(expr) {
            Expr::Paren(inner) => expr = inner,
            Expr::Block {
                stmts,
                tail: Some(tail),
                label: None,
                kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const,
            } if scope.body.list(stmts).is_empty() => expr = tail,
            _ => return Some(expr),
        }
    }
}

fn sign_extend_u128(val: u128, bits: u32) -> i128 {
    let shift = 128 - bits;
    ((val << shift) as i128) >> shift
}

fn eval_const_cmp(op: BinOp, bits: u32, signed: bool, l: u128, r: u128) -> bool {
    let ord = if signed {
        sign_extend_u128(l, bits).cmp(&sign_extend_u128(r, bits))
    } else {
        l.cmp(&r)
    };
    matches!(
        (op, ord),
        (BinOp::Eq, std::cmp::Ordering::Equal)
            | (
                BinOp::Ne,
                std::cmp::Ordering::Less | std::cmp::Ordering::Greater
            )
            | (BinOp::Lt | BinOp::Le, std::cmp::Ordering::Less)
            | (BinOp::Gt | BinOp::Ge, std::cmp::Ordering::Greater)
            | (BinOp::Le | BinOp::Ge, std::cmp::Ordering::Equal)
    )
}

fn eval_const_int_bin(
    op: BinOp,
    (bits, signed): (u32, bool),
    l: u128,
    (rhs_bits, rhs_signed): (u32, bool),
    r: u128,
    overflow_checks: bool,
) -> Option<u128> {
    let m = mask(bits);
    match op {
        BinOp::BitAnd => Some((l & r) & m),
        BinOp::BitOr => Some((l | r) & m),
        BinOp::BitXor => Some((l ^ r) & m),
        BinOp::Shl | BinOp::Shr => {
            let bad = if rhs_signed {
                let s = sign_extend_u128(r, rhs_bits);
                s < 0 || s >= i128::from(bits)
            } else {
                r >= u128::from(bits)
            };
            if overflow_checks && bad {
                return None;
            }
            let sh = (r as u32) & (bits - 1);
            let out = match (op, signed) {
                (BinOp::Shl, _) => l.wrapping_shl(sh),
                (BinOp::Shr, true) => sign_extend_u128(l, bits).wrapping_shr(sh) as u128,
                _ => l >> sh,
            };
            Some(out & m)
        }
        BinOp::Div | BinOp::Rem if r == 0 => None,
        BinOp::Div | BinOp::Rem if !signed => Some(if op == BinOp::Div { l / r } else { l % r }),
        BinOp::Div | BinOp::Rem => {
            let (ls, rs) = (sign_extend_u128(l, bits), sign_extend_u128(r, bits));
            if ls == (-1_i128 << (bits - 1)) && rs == -1 {
                return None;
            }
            Some(
                (if op == BinOp::Div {
                    ls.wrapping_div(rs)
                } else {
                    ls.wrapping_rem(rs)
                } as u128)
                    & m,
            )
        }
        BinOp::Add | BinOp::Sub | BinOp::Mul if signed => {
            let (ls, rs) = (sign_extend_u128(l, bits), sign_extend_u128(r, bits));
            if overflow_checks {
                let (min, max) = (-1_i128 << (bits - 1), (m >> 1) as i128);
                return match op {
                    BinOp::Add => ls.checked_add(rs),
                    BinOp::Sub => ls.checked_sub(rs),
                    _ => ls.checked_mul(rs),
                }
                .filter(|&v| v >= min && v <= max)
                .map(|v| (v as u128) & m);
            }
            Some(
                (match op {
                    BinOp::Add => ls.wrapping_add(rs),
                    BinOp::Sub => ls.wrapping_sub(rs),
                    _ => ls.wrapping_mul(rs),
                } as u128)
                    & m,
            )
        }
        BinOp::Add | BinOp::Sub | BinOp::Mul => {
            if overflow_checks {
                return match op {
                    BinOp::Add => l.checked_add(r),
                    BinOp::Sub => l.checked_sub(r),
                    _ => l.checked_mul(r),
                }
                .filter(|&v| v <= m);
            }
            Some(
                match op {
                    BinOp::Add => l.wrapping_add(r),
                    BinOp::Sub => l.wrapping_sub(r),
                    _ => l.wrapping_mul(r),
                } & m,
            )
        }
        _ => None,
    }
}

fn cast_const_scalar(
    src_val: u128,
    src_prim: PrimName,
    dst_prim: PrimName,
) -> Option<(usize, u128)> {
    if dst_prim == PrimName::Char {
        return matches!(src_prim, PrimName::U8 | PrimName::Char)
            .then_some((4, src_val & 0xFFFF_FFFF));
    }
    let (dst_bits, _) = int_info(dst_prim)?;
    let (src_bits, src_signed) = const_prim_info(src_prim)?;
    let bits = if src_signed {
        sign_extend_u128(src_val, src_bits) as u128
    } else {
        src_val
    };
    Some((dst_bits as usize / 8, bits & mask(dst_bits)))
}
