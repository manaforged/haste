use crate::body::literal::{byte_value, char_value, string_bytes};
use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, Lit, UnOp};
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::infer::{Coercion, Res, Typed, field_types, parse_int};
use crate::sema::layout::field_offset;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty, int_info, mask};

use super::super::Lower;
use super::super::stencil::{RodataReloc, RodataTarget};

/// The most bytes of a constant that lowering lays out from its literals.
pub(super) const LITERAL_BYTES: usize = 1 << 21;

pub(super) const MAX_CONST_DEPTH: u32 = 16;

#[derive(Clone, Copy)]
pub(super) struct Scope<'s> {
    pub(super) body: &'s Body,
    pub(super) src: &'s Source,
    pub(super) typed: &'s Typed,
    pub(super) in_const: bool,
}

impl Lower<'_, '_> {
    /// Returns the bytes and intra-rodata relocations of `expr` when it evaluates from literals and pure constants within `LITERAL_BYTES`.
    pub(super) fn literal_blob(&mut self, scope: Scope<'_>, (expr, ty): (ExprId, Ty), depth: u32) -> Option<(Vec<u8>, Vec<RodataReloc>)> {
        if depth > MAX_CONST_DEPTH || ty == Ty::NONE || self.cx.types.has_params(ty) {
            return None;
        }
        let layout = self.cx.layout(ty).ok()?;
        if layout.align.bytes() > 16 {
            return None;
        }
        let size = usize::try_from(layout.size.bytes()).ok().filter(|size| *size <= LITERAL_BYTES)?;
        let mut out = vec![0; size];
        let mut relocs = Vec::new();
        self.write_blob(scope, (expr, ty), 0, (&mut out, &mut relocs), depth)?;
        Some((out, relocs))
    }

    fn write_blob(&mut self, scope: Scope<'_>, (expr, ty): (ExprId, Ty), at: usize, (out, relocs): (&mut [u8], &mut Vec<RodataReloc>), depth: u32) -> Option<()> {
        match scope.typed.coercions.get(&expr) {
            None => self.write_uncoerced(scope, (expr, ty), at, (out, relocs), depth),
            Some(&Coercion::Len(len_ty)) => self
                .write_slice_coercion(scope, (expr, ty), len_ty, at, (out, relocs), depth)
                .or_else(|| self.eval_const_into(scope, (expr, ty), at, (out, relocs), depth)),
            Some(Coercion::Deref(..)) => self.eval_const_into(scope, (expr, ty), at, (out, relocs), depth),
            Some(_) => None,
        }
    }

    fn write_slice_coercion(&mut self, scope: Scope<'_>, (expr, ty): (ExprId, Ty), len_ty: Ty, at: usize, (out, relocs): (&mut [u8], &mut Vec<RodataReloc>), depth: u32) -> Option<()> {
        let Kind::Ref(Mut::Not, pointee) = *self.cx.types.kind(ty) else { return None };
        if !matches!(*self.cx.types.kind(pointee), Kind::Slice(_)) {
            return None;
        }
        let orig_ty = scope.typed.expr_ty(&self.cx.types, expr);
        let Kind::Ref(Mut::Not, orig_pointee) = *self.cx.types.kind(orig_ty) else { return None };
        if !matches!(*self.cx.types.kind(orig_pointee), Kind::Array(..)) {
            return None;
        }
        let mut ptr_buf = [0u8; 8];
        let mut ptr_relocs = Vec::new();
        self.write_uncoerced(scope, (expr, orig_ty), 0, (&mut ptr_buf, &mut ptr_relocs), depth)?;
        let &[reloc] = ptr_relocs.as_slice() else { return None };
        if reloc.offset != 0 {
            return None;
        }
        let len = self.cx.types.len(len_ty)?;
        out.get_mut(at..at + 16)?[8..16].copy_from_slice(&len.to_le_bytes());
        relocs.push(RodataReloc { offset: at as u32, target: reloc.target });
        Some(())
    }

    fn write_uncoerced(&mut self, scope: Scope<'_>, (expr, ty): (ExprId, Ty), at: usize, (out, relocs): (&mut [u8], &mut Vec<RodataReloc>), depth: u32) -> Option<()> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        match *scope.body.expr(expr) {
            Expr::Paren(inner) => return self.write_blob(scope, (inner, ty), at, (out, relocs), depth),
            Expr::Block { stmts, tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const } if scope.body.list(stmts).is_empty() => {
                return self.write_blob(scope, (tail, ty), at, (out, relocs), depth);
            }
            Expr::Path(_) => {
                if let Some(&Res::Fn(item, substs)) = scope.typed.res.get(&expr)
                    && self.cx.item(item).1.kind == ItemKind::Const
                {
                    return self.write_const_item(item, substs, ty, at, (out, relocs), depth);
                }
            }
            _ => {}
        }
        let kind = self.cx.types.kind(ty).clone();
        match (*scope.body.expr(expr), kind) {
            (Expr::Lit(Lit::Str, span), Kind::Ref(Mut::Not, pointee)) if *self.cx.types.kind(pointee) == Kind::Prim(Primitive::Str) => {
                let bytes = string_bytes(scope.src.text(span)).ok()?;
                let child_id = self.slots.rodata(&bytes, &[]);
                out.get_mut(at..at + 16)?[8..16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
                relocs.push(RodataReloc { offset: at as u32, target: RodataTarget::Rodata(child_id) });
                Some(())
            }
            (Expr::Lit(Lit::ByteStr, span), Kind::Ref(Mut::Not, pointee)) => {
                let Kind::Array(elem, len_ty) = *self.cx.types.kind(pointee) else { return None };
                if *self.cx.types.kind(elem) != Kind::Prim(Primitive::U8) {
                    return None;
                }
                let bytes = string_bytes(scope.src.text(span)).ok()?;
                if self.cx.types.len(len_ty)? != bytes.len() as u64 {
                    return None;
                }
                let child_id = self.slots.rodata(&bytes, &[]);
                out.get_mut(at..at + 8)?;
                relocs.push(RodataReloc { offset: at as u32, target: RodataTarget::Rodata(child_id) });
                Some(())
            }
            (Expr::Ref { mutable: false, raw: false, expr: inner }, Kind::Ref(Mut::Not, pointee)) => {
                let inner_ty = scope.typed.expr_ty(&self.cx.types, inner);
                if !super::super::is_freeze_ty(self.cx, inner_ty, 0) {
                    return None;
                }
                let (child_bytes, child_relocs) = self.literal_blob(scope, (inner, inner_ty), depth + 1)?;
                let child_id = self.slots.rodata(&child_bytes, &child_relocs);
                match (self.cx.types.kind(pointee).clone(), self.cx.types.kind(inner_ty).clone()) {
                    _ if pointee == inner_ty => {
                        out.get_mut(at..at + 8)?;
                    }
                    (Kind::Slice(_), Kind::Array(_, len_ty)) => {
                        let len = self.cx.types.len(len_ty)?;
                        out.get_mut(at..at + 16)?[8..16].copy_from_slice(&len.to_le_bytes());
                    }
                    _ => return None,
                }
                relocs.push(RodataReloc { offset: at as u32, target: RodataTarget::Rodata(child_id) });
                Some(())
            }
            (Expr::Array(list), Kind::Array(elem, _)) => {
                let stride = self.cx.layout(elem).ok()?.size.bytes() as usize;
                let items: Vec<ExprId> = scope.body.list(list).iter().map(|&item| ExprId(item)).collect();
                items.into_iter().enumerate().try_for_each(|(index, item)| self.write_blob(scope, (item, elem), at + index * stride, (out, relocs), depth))
            }
            (Expr::Repeat(item, _), Kind::Array(elem, count)) => {
                let stride = self.cx.layout(elem).ok()?.size.bytes() as usize;
                let count = usize::try_from(self.cx.types.len(count)?).ok()?;
                if count == 0 {
                    return Some(());
                }
                let reloc_start = relocs.len();
                self.write_blob(scope, (item, elem), at, (out, relocs), depth)?;
                let item_relocs: Vec<RodataReloc> = relocs[reloc_start..].to_vec();
                for index in 1..count {
                    let dst = at + index * stride;
                    out.copy_within(at..at + stride, dst);
                    for reloc in &item_relocs {
                        relocs.push(RodataReloc { offset: (reloc.offset as usize - at + dst) as u32, target: reloc.target });
                    }
                }
                Some(())
            }
            (Expr::Tuple(list), Kind::Tuple(items)) => {
                let layout = self.cx.layout(ty).ok()?;
                let items = self.cx.types.substs(items).to_vec();
                let fields: Vec<(u32, ExprId)> = scope.body.list(list).iter().enumerate().map(|(index, &item)| (index as u32, ExprId(item))).collect();
                fields.into_iter().try_for_each(|(index, item)| {
                    self.write_blob(scope, (item, *items.get(index as usize)?), at + field_offset(&layout, 0, index) as usize, (out, relocs), depth)
                })
            }
            (_, Kind::Adt(item, args)) => self
                .construct_blob(scope, (expr, ty), (item, args), at, (out, relocs), depth)
                .or_else(|| self.eval_const_into(scope, (expr, ty), at, (out, relocs), depth)),
            (_, Kind::Prim(primitive)) => {
                let size = match primitive {
                    Primitive::Bool => 1,
                    Primitive::Char => 4,
                    _ => int_info(primitive)?.0 as usize / 8,
                };
                let bits = self.eval_scalar(scope, expr, primitive, depth)?;
                out.get_mut(at..at + size)?.copy_from_slice(&bits.to_le_bytes()[..size]);
                Some(())
            }
            _ => self.eval_const_into(scope, (expr, ty), at, (out, relocs), depth),
        }
    }

    fn write_const_item(&mut self, item: ItemRef, substs: Substs, ty: Ty, at: usize, (out, relocs): (&mut [u8], &mut Vec<RodataReloc>), depth: u32) -> Option<()> {
        if depth >= MAX_CONST_DEPTH {
            return None;
        }
        let (source, typed) = self.literal_source(item, substs).ok()??;
        let const_typed = typed.scoped(&self.cx.types);
        if const_typed.ret_ty() != ty {
            return None;
        }
        let body = self.cx.body(source).ok()?;
        let root = body.root?;
        let src = self.cx.body_src(source);
        self.write_blob(Scope { body: &body, src, typed: &const_typed, in_const: true }, (root, ty), at, (out, relocs), depth + 1)
    }

    fn construct_blob(&mut self, scope: Scope<'_>, (expr, ty): (ExprId, Ty), (item, args): (ItemRef, Substs), at: usize, (out, relocs): (&mut [u8], &mut Vec<RodataReloc>), depth: u32) -> Option<()> {
        let (variant, fields) = constructor(scope, expr)?;
        let layout = self.cx.layout(ty).ok()?;
        let args = self.cx.types.substs(args).to_vec();
        let types = field_types(self.cx, item, &args).ok()?.get(variant as usize)?.clone();
        for (index, value) in fields {
            self.write_blob(scope, (value, *types.get(index as usize)?), at + field_offset(&layout, variant, index) as usize, (out, relocs), depth)?;
        }
        if let Some((offset, tag, bits)) = super::super::adt::tag_bits(self.cx, item, &layout, variant).ok()? {
            let size = tag.bytes() as usize;
            let start = at + offset as usize;
            out.get_mut(start..start + size)?.copy_from_slice(&bits.to_le_bytes()[..size]);
        }
        Some(())
    }

    fn eval_scalar(&mut self, scope: Scope<'_>, expr: ExprId, primitive: Primitive, depth: u32) -> Option<u128> {
        if depth > MAX_CONST_DEPTH || scope.typed.coercions.contains_key(&expr) {
            return None;
        }
        let value = match *scope.body.expr(expr) {
            Expr::Paren(inner) => return self.eval_scalar(scope, inner, primitive, depth),
            Expr::Block { stmts, tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const } if scope.body.list(stmts).is_empty() => {
                return self.eval_scalar(scope, tail, primitive, depth);
            }
            Expr::Lit(Lit::Bool(value), _) if primitive == Primitive::Bool => u128::from(value),
            Expr::Lit(Lit::Int, at) if int_info(primitive).is_some() => parse_int(scope.src.text(at)).ok()?.0,
            Expr::Lit(Lit::Byte, at) if primitive == Primitive::U8 => u128::from(byte_value(scope.src.text(at)).ok()?),
            Expr::Lit(Lit::Char, at) if primitive == Primitive::Char => u128::from(u32::from(char_value(scope.src.text(at)).ok()?)),
            Expr::Path(_) => match scope.typed.res.get(&expr) {
                Some(&Res::Const(val_ty)) => match *self.cx.types.kind(val_ty) {
                    Kind::Value(bits) => bits,
                    _ => return None,
                },
                Some(&Res::Fn(item, substs)) if depth < MAX_CONST_DEPTH && self.cx.item(item).1.kind == ItemKind::Const => {
                    let (source, typed) = self.literal_source(item, substs).ok()??;
                    let const_typed = typed.scoped(&self.cx.types);
                    let Kind::Prim(ret_prim) = *self.cx.types.kind(const_typed.ret_ty()) else { return None };
                    if ret_prim != primitive {
                        return None;
                    }
                    let body = self.cx.body(source).ok()?;
                    let root = body.root?;
                    let src = self.cx.body_src(source);
                    return self.eval_scalar(Scope { body: &body, src, typed: &const_typed, in_const: true }, root, primitive, depth + 1);
                }
                _ => return None,
            },
            Expr::Unary(op, inner) if !scope.typed.operators.contains_key(&expr) => match op {
                UnOp::Neg if int_info(primitive).is_some_and(|(_, signed)| signed) => self.eval_scalar(scope, inner, primitive, depth)?.wrapping_neg(),
                UnOp::Not if primitive == Primitive::Bool => u128::from(self.eval_scalar(scope, inner, Primitive::Bool, depth)? == 0),
                UnOp::Not if int_info(primitive).is_some() => !self.eval_scalar(scope, inner, primitive, depth)?,
                _ => return None,
            },
            Expr::Cast(inner, _) => {
                let Kind::Prim(src_prim) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, inner)) else { return None };
                let src_val = self.eval_scalar(scope, inner, src_prim, depth)?;
                return cast_scalar(src_val, src_prim, primitive);
            }
            Expr::Binary(op, lhs, rhs) if !scope.typed.operators.contains_key(&expr) => return self.eval_binary(scope, op, lhs, rhs, primitive, depth),
            _ => return self.eval_const_scalar_fn(scope, expr, primitive, depth),
        };
        Some(int_info(primitive).map_or(value, |(bits, _)| value & mask(bits)))
    }

    fn eval_binary(&mut self, scope: Scope<'_>, op: BinOp, lhs: ExprId, rhs: ExprId, primitive: Primitive, depth: u32) -> Option<u128> {
        match op {
            BinOp::And | BinOp::Or if primitive == Primitive::Bool => {
                let l = self.eval_scalar(scope, lhs, Primitive::Bool, depth)? != 0;
                if (op == BinOp::And && !l) || (op == BinOp::Or && l) {
                    return Some(u128::from(l));
                }
                let r = self.eval_scalar(scope, rhs, Primitive::Bool, depth)? != 0;
                Some(u128::from(r))
            }
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge if primitive == Primitive::Bool => {
                let Kind::Prim(lp) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, lhs)) else { return None };
                let Kind::Prim(rp) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, rhs)) else { return None };
                if lp != rp {
                    return None;
                }
                let (bits, signed) = prim_info(lp)?;
                let l = self.eval_scalar(scope, lhs, lp, depth)?;
                let r = self.eval_scalar(scope, rhs, rp, depth)?;
                Some(u128::from(cmp_scalar(op, l, r, (bits, signed))))
            }
            BinOp::Shl | BinOp::Shr => {
                let Kind::Prim(rp) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, rhs)) else { return None };
                let rinfo = int_info(rp)?;
                let l = self.eval_scalar(scope, lhs, primitive, depth)?;
                let r = self.eval_scalar(scope, rhs, rp, depth)?;
                apply_shift(op, primitive, (l, r), rinfo)
            }
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                let l = self.eval_scalar(scope, lhs, primitive, depth)?;
                let r = self.eval_scalar(scope, rhs, primitive, depth)?;
                apply_int_binary(op, primitive, l, r)
            }
            _ => None,
        }
    }
}

pub(super) fn apply_shift(op: BinOp, primitive: Primitive, (l, r): (u128, u128), (rbits, rsigned): (u32, bool)) -> Option<u128> {
    let (bits, signed) = int_info(primitive)?;
    let shift = if rsigned { u32::try_from(sign_extend(r, rbits)).ok() } else { u32::try_from(r).ok() }?;
    if shift >= bits {
        return None;
    }
    match op {
        BinOp::Shl => Some((l << shift) & mask(bits)),
        _ if signed => Some(((sign_extend(l, bits) >> shift) as u128) & mask(bits)),
        _ => Some((l & mask(bits)) >> shift),
    }
}

pub(super) fn apply_int_binary(op: BinOp, primitive: Primitive, l: u128, r: u128) -> Option<u128> {
    if matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor) {
        let (bits, _) = if primitive == Primitive::Bool { (1, false) } else { int_info(primitive)? };
        let out = match op {
            BinOp::BitAnd => l & r,
            BinOp::BitOr => l | r,
            _ => l ^ r,
        };
        return Some(out & mask(bits));
    }
    let (bits, signed) = int_info(primitive)?;
    if signed {
        apply_checked_signed(op, bits, l, r)
    } else {
        apply_checked_unsigned(op, bits, l, r)
    }
}

fn apply_checked_signed(op: BinOp, bits: u32, l: u128, r: u128) -> Option<u128> {
    let (sa, sb) = (sign_extend(l, bits), sign_extend(r, bits));
    let res = match op {
        BinOp::Add => sa.checked_add(sb)?,
        BinOp::Sub => sa.checked_sub(sb)?,
        BinOp::Mul => sa.checked_mul(sb)?,
        BinOp::Div => sa.checked_div(sb)?,
        BinOp::Rem => sa.checked_rem(sb)?,
        _ => return None,
    };
    check_signed(res, bits)
}

fn apply_checked_unsigned(op: BinOp, bits: u32, l: u128, r: u128) -> Option<u128> {
    let res = match op {
        BinOp::Add => l.checked_add(r)?,
        BinOp::Sub => l.checked_sub(r)?,
        BinOp::Mul => l.checked_mul(r)?,
        BinOp::Div => l.checked_div(r)?,
        BinOp::Rem => l.checked_rem(r)?,
        _ => return None,
    };
    (res <= mask(bits)).then_some(res)
}

/// The variant `expr` constructs, with each field it gives by index, when it is a unit, tuple or struct constructor with no base.
fn constructor(scope: Scope<'_>, expr: ExprId) -> Option<(u32, Vec<(u32, ExprId)>)> {
    let variant = |at: ExprId| match scope.typed.res.get(&at) {
        Some(&Res::Ctor(_, variant, _)) => Some(variant),
        _ => None,
    };
    match *scope.body.expr(expr) {
        Expr::Path(_) if !scope.typed.coercions.contains_key(&expr) => Some((variant(expr)?, Vec::new())),
        Expr::Call(callee, args) => Some((variant(callee)?, scope.body.list(args).iter().enumerate().map(|(index, &at)| (index as u32, ExprId(at))).collect())),
        Expr::Struct { fields, base: None, .. } => {
            let fields = scope.body.list(fields).iter().map(|&at| scope.body.fields[at as usize].value).map(|value| Some((*scope.typed.inits.get(&value)?, value))).collect::<Option<Vec<_>>>()?;
            Some((variant(expr)?, fields))
        }
        _ => None,
    }
}

pub(super) fn prim_info(prim: Primitive) -> Option<(u32, bool)> {
    match prim {
        Primitive::Bool => Some((1, false)),
        Primitive::Char => Some((32, false)),
        _ => int_info(prim),
    }
}

pub(super) fn sign_extend(val: u128, bits: u32) -> i128 {
    let shift = 128 - bits;
    ((val << shift) as i128) >> shift
}

fn check_signed(res: i128, bits: u32) -> Option<u128> {
    if bits == 128 {
        return Some(res as u128);
    }
    let min = -(1i128 << (bits - 1));
    let max = (1i128 << (bits - 1)) - 1;
    (res >= min && res <= max).then_some((res as u128) & mask(bits))
}

pub(super) fn cast_scalar(src_val: u128, src_prim: Primitive, dst_prim: Primitive) -> Option<u128> {
    if dst_prim == Primitive::Char {
        return matches!(src_prim, Primitive::U8 | Primitive::Char).then_some(src_val & 0xFFFF_FFFF);
    }
    let (dst_bits, _) = int_info(dst_prim)?;
    let (src_bits, src_signed) = prim_info(src_prim)?;
    let bits = if src_signed { sign_extend(src_val, src_bits) as u128 } else { src_val };
    Some(bits & mask(dst_bits))
}

pub(super) fn cmp_scalar(op: BinOp, l: u128, r: u128, (bits, signed): (u32, bool)) -> bool {
    if signed {
        let (sa, sb) = (sign_extend(l, bits), sign_extend(r, bits));
        match op {
            BinOp::Eq => sa == sb,
            BinOp::Ne => sa != sb,
            BinOp::Lt => sa < sb,
            BinOp::Le => sa <= sb,
            BinOp::Gt => sa > sb,
            _ => sa >= sb,
        }
    } else {
        match op {
            BinOp::Eq => l == r,
            BinOp::Ne => l != r,
            BinOp::Lt => l < r,
            BinOp::Le => l <= r,
            BinOp::Gt => l > r,
            _ => l >= r,
        }
    }
}
