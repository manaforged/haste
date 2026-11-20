use anyhow::{Result, bail};
use haste_codegen::{IntCC, StackSlot, Type, Value};
use rustc_abi::{FieldsShape, Primitive, TagEncoding, Variants};
use rustc_index::Idx;

use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, List, Lit, UnOp};
use crate::names::{ItemRef, Primitive as PrimName};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::infer::{ALIGN_SLOT, Res, Typed, field_types, parse_int};
use crate::sema::layout::{Layout, Variant, field_offset, target};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, int_info, mask};

use super::ops::Num;
use super::place::{Base, Place};
use super::promote::{eval_lit_bits, neg_float_bits};
use super::{Lower, Parts, Repr, repr};

mod rodata;

const CONST_RODATA_MAX_BYTES: usize = 1 << 21;
const CONST_RODATA_MAX_DEPTH: u32 = 16;

#[derive(Clone, Copy)]
struct ConstScope<'s> {
    body: &'s Body,
    src: &'s Source,
    typed: &'s Typed,
    in_const: bool,
}

pub(super) fn tag_type(bytes: u64) -> Result<Type> {
    Ok(match bytes {
        1 => Type::I8,
        2 => Type::I16,
        4 => Type::I32,
        8 => Type::I64,
        16 => Type::I128,
        other => bail!("no tag of {other} bytes"),
    })
}

/// Where an enum keeps its tag and how it encodes its variants in it.
struct Tag {
    offset: i32,
    ty: Type,
    mask: u128,
    encoding: TagEncoding<Variant>,
}

fn tag_of(layout: &Layout) -> Result<Option<Tag>> {
    let Variants::Multiple { tag, tag_encoding, tag_field, .. } = &layout.variants else {
        return Ok(None);
    };
    let size = tag.size(target()?);
    let mask = crate::sema::ty::mask(size.bits() as u32);
    Ok(Some(Tag { offset: layout.fields.offset(tag_field.index()).bytes() as i32, ty: tag_type(size.bytes())?, mask, encoding: tag_encoding.clone() }))
}

/// The offset, type and bits of the tag that a value of variant `variant` of the enum `item` with `layout` holds, or none when the layout keeps no tag for it.
pub(super) fn tag_bits(cx: &mut Context<'_>, item: ItemRef, layout: &Layout, variant: u32) -> Result<Option<(i32, Type, u128)>> {
    let Some(tag) = tag_of(layout)? else {
        return Ok(None);
    };
    let value = match &tag.encoding {
        TagEncoding::Direct => cx.discriminants(item)?[variant as usize] as u128,
        TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
            if untagged_variant.index() == variant as usize {
                return Ok(None);
            }
            ((variant as u128).wrapping_sub(niche_variants.start.index() as u128)).wrapping_add(*niche_start)
        }
    };
    Ok(Some((tag.offset, tag.ty, value & tag.mask)))
}

impl Lower<'_, '_> {
    /// Writes variant `variant` of `ty`, an ADT `item` at `args`, with `fields` given by index, over a copy of `base`; returns the value.
    pub fn construct(&mut self, ty: Ty, variant: u32, fields: Vec<(u32, Option<Value>)>, base: Option<Value>) -> Result<Option<Value>> {
        let Kind::Adt(item, args) = *self.cx.types.kind(ty) else { bail!("constructing {:?}, a type that is not an ADT", self.cx.types.kind(ty)) };
        let whole = repr(self.cx, ty)?;
        let Some(slot) = self.slot(whole) else {
            return Ok(None);
        };
        let place = Place::slot(slot, whole);
        if let Some(base) = base {
            self.write(place, base);
        }
        let layout = self.cx.layout(ty)?;
        let slice = self.cx.types.substs(args).to_vec();
        let all = field_types(self.cx, item, &slice)?;
        let types = &all[variant as usize];
        if base.is_none() && self.try_construct_simd(slot, whole, &layout, types, &fields)? {
            return Ok(self.read(place));
        }
        for (index, value) in fields {
            let Some(&field_ty) = types.get(index as usize) else { bail!("variant {variant} of {} has no field {index}", self.cx.name(item)) };
            let field = repr(self.cx, field_ty)?;
            if let Some(value) = value {
                self.write(place.at(field_offset(&layout, variant, index), field), value);
            }
        }
        self.set_tag(place, item, &layout, variant)?;
        Ok(self.read(place))
    }

    fn try_construct_simd(&mut self, slot: StackSlot, whole: Repr, layout: &Layout, types: &[Ty], fields: &[(u32, Option<Value>)]) -> Result<bool> {
        let Repr::Mem { size, parts: Parts::One(vec_ty @ Type::Vec { lane, count }), .. } = whole else {
            return Ok(false);
        };
        if !matches!(layout.variants, Variants::Single { .. }) {
            return Ok(false);
        }
        let n = usize::from(count);
        if types.len() == 1
            && let [(0, Some(addr))] = fields
            && matches!(*self.cx.types.kind(types[0]), Kind::Array(..))
            && matches!(repr(self.cx, types[0])?, Repr::Mem { size: field_size, .. } if field_size == size)
        {
            let vec = self.ins().load(vec_ty, *addr, 0);
            self.ins().stack_store(vec, slot, 0);
            return Ok(true);
        }
        if types.len() == n && fields.len() == n {
            let mut lanes = vec![None; n];
            for &(index, value) in fields {
                let Some(&field_ty) = types.get(index as usize) else { return Ok(false) };
                if !matches!(repr(self.cx, field_ty)?, Repr::Scalar(ty) if ty == lane.to_type()) {
                    return Ok(false);
                }
                lanes[index as usize] = value;
            }
            if let Some(first) = lanes.first().copied().flatten()
                && lanes[1..].iter().all(Option::is_some)
            {
                let mut vec = self.ins().splat(vec_ty, first);
                for (i, lane_val) in lanes.into_iter().enumerate().skip(1) {
                    if let Some(val) = lane_val {
                        vec = self.ins().insert(vec, val, i as u8);
                    }
                }
                self.ins().stack_store(vec, slot, 0);
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn set_tag(&mut self, place: Place, item: ItemRef, layout: &Layout, variant: u32) -> Result<()> {
        let Some((offset, ty, bits)) = tag_bits(self.cx, item, layout, variant)? else {
            return Ok(());
        };
        let value = self.int_const(ty, bits);
        self.write(place.at(offset, Repr::Scalar(ty)), value);
        Ok(())
    }

    /// Whether the value of ADT `ty` at `place`, which a zero-sized value lacks, holds variant `variant`, or none when it is the only variant the value can hold.
    pub fn is_variant(&mut self, place: Option<Place>, ty: Ty, variant: u32) -> Result<Option<Value>> {
        let Kind::Adt(item, _) = self.cx.types.kind(ty).clone() else { bail!("variant test on a type that is not an ADT") };
        let layout = self.cx.layout(ty)?;
        let Some(tag) = tag_of(&layout)? else {
            return Ok(match layout.variants {
                Variants::Single { index } if index.index() != variant as usize => Some(self.ins().iconst(Type::I8, 0)),
                _ => None,
            });
        };
        let Some(value) = place.and_then(|place| self.read(place.at(tag.offset, Repr::Scalar(tag.ty)))) else { bail!("tag with no value") };
        Ok(Some(match &tag.encoding {
            TagEncoding::Direct => {
                let expected = self.cx.discriminants(item)?[variant as usize] as u128 & tag.mask;
                let expected = self.int_const(tag.ty, expected);
                self.ins().icmp(IntCC::Equal, value, expected)
            }
            TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                let start = self.int_const(tag.ty, *niche_start & tag.mask);
                let relative = self.ins().isub(value, start);
                let first = niche_variants.start.index() as u128;
                let span = self.int_const(tag.ty, niche_variants.last.index() as u128 - first);
                if untagged_variant.index() == variant as usize {
                    self.ins().icmp(IntCC::UnsignedGreaterThan, relative, span)
                } else {
                    let expected = self.int_const(tag.ty, (variant as u128).wrapping_sub(first));
                    self.ins().icmp(IntCC::Equal, relative, expected)
                }
            }
        }))
    }

    /// The discriminant of the fieldless enum `ty` at `place`, in the integer type of its tag.
    pub(super) fn discriminant(&mut self, place: Option<Place>, ty: Ty) -> Result<(Value, Num)> {
        let Kind::Adt(item, _) = self.cx.types.kind(ty).clone() else { bail!("discriminant of a type that is not an enum") };
        let layout = self.cx.layout(ty)?;
        let Variants::Multiple { tag: scalar, .. } = &layout.variants else {
            let value = self.cx.discriminants(item)?.first().copied().unwrap_or(0);
            return Ok((self.int_const(Type::I64, value as u128), Num::Int { ty: Type::I64, signed: true }));
        };
        let signed = matches!(scalar.primitive(), Primitive::Int(_, true));
        let Some(tag) = tag_of(&layout)? else { bail!("enum with variants has no tag") };
        if !matches!(tag.encoding, TagEncoding::Direct) {
            bail!("casting an enum with fields");
        }
        let Some(value) = place.and_then(|place| self.read(place.at(tag.offset, Repr::Scalar(tag.ty)))) else { bail!("enum with no value") };
        Ok((value, Num::Int { ty: tag.ty, signed }))
    }

    /// The declared discriminant of the enum `ty` at `place`, in its discriminant integer type, whatever its tag encoding.
    pub(super) fn discriminant_value(&mut self, place: Place, ty: Ty) -> Result<Value> {
        let Kind::Adt(item, _) = self.cx.types.kind(ty).clone() else { bail!("discriminant_value of a type that is not an enum") };
        let int = self.cx.types.prim(self.cx.discriminant_primitive(item));
        let Repr::Scalar(to) = repr(self.cx, int)? else { bail!("a discriminant that is not a scalar") };
        let values = self.cx.discriminants(item)?;
        let Some((&last, rest)) = values.split_last() else { bail!("discriminant_value of an enum with no variants") };
        let mut found = self.int_const(to, last as u128);
        for (variant, &value) in rest.iter().enumerate() {
            let Some(test) = self.is_variant(Some(place), ty, variant as u32)? else {
                continue;
            };
            let value = self.int_const(to, value as u128);
            found = self.ins().select(test, value, found);
        }
        Ok(found)
    }

    /// The place of field `index` of variant `variant` inside the ADT `ty` at `place`.
    pub fn variant_field(&mut self, place: Place, ty: Ty, (variant, index): (u32, u32)) -> Result<Place> {
        let Kind::Adt(item, args) = *self.cx.types.kind(ty) else { bail!("field of a type that is not an ADT") };
        let layout = self.cx.layout(ty)?;
        let slice = self.cx.types.substs(args).to_vec();
        let fields = field_types(self.cx, item, &slice)?;
        let Some(&field) = fields.get(variant as usize).and_then(|fields| fields.get(index as usize)) else { bail!("{} has no field {index} in variant {variant}", self.cx.name(item)) };
        if layout.is_unsized() {
            return self.unsized_field(place, field_offset(&layout, variant, index), field);
        }
        Ok(place.at(field_offset(&layout, variant, index), repr(self.cx, field)?))
    }

    fn unsized_field(&mut self, wide: Place, offset: i32, field: Ty) -> Result<Place> {
        let Some(pair) = self.read(wide) else { bail!("an unsized value with no pointer") };
        let data = self.ins().load(Type::I64, pair, 0);
        if !self.cx.is_unsized(field)? {
            let at = self.ins().iadd_imm(data, i64::from(offset));
            return Ok(Place { base: Base::Addr(at), offset: 0, repr: repr(self.cx, field)? });
        }
        let metadata = self.ins().load(Type::I64, pair, 8);
        let tail = self.cx.tail(field)?;
        let offset = if matches!(self.cx.types.kind(tail), Kind::Dyn(..)) {
            let align = self.ins().load(Type::I64, metadata, ALIGN_SLOT as i32 * 8);
            let start = self.ins().iconst(Type::I64, i64::from(offset));
            self.align_up(start, align)
        } else {
            self.ins().iconst(Type::I64, i64::from(offset))
        };
        let at = self.ins().iadd(data, offset);
        Ok(Place { base: Base::Addr(self.wide(at, metadata)), offset: 0, repr: super::ops::WIDE })
    }

    /// Lowers the struct literal `expr` with field initializers `fields`.
    pub fn struct_literal(&mut self, expr: ExprId, fields: List, base: Option<ExprId>) -> Result<Option<Value>> {
        if base.is_none()
            && let Some(addr) = self.materialize_const_expr(expr, (true, false))?
        {
            return Ok(Some(addr));
        }
        let Some(Res::Ctor(_, variant, _)) = self.typed.res.get(&expr).cloned() else { bail!("struct literal has no constructor") };
        let mut values = Vec::new();
        for at in self.body.list(fields).to_vec() {
            let init = self.body.fields[at as usize];
            let Some(&index) = self.typed.inits.get(&init.value) else { bail!("field initializer has no field") };
            values.push((index, self.eval_operand(init.value)?));
        }
        let base = match base {
            Some(base) => self.eval_operand(base)?,
            None => None,
        };
        self.construct(self.ty_of(expr), variant, values, base)
    }

    /// Lowers the body of a constructor function: variant `variant` of the return type, built from the parameters.
    pub fn ctor_body(&mut self, variant: u32) -> Result<Option<Value>> {
        let Some(ret) = self.typed.ret else { bail!("a constructor function with no return type") };
        let mut values = Vec::new();
        for (index, local) in self.typed.params.clone().into_iter().enumerate() {
            values.push((index as u32, self.locals[local as usize].and_then(|place| self.read(place))));
            self.moved_out(super::moves::Root::Local(local), &[])?;
        }
        self.construct(ret, variant, values, None)
    }

    /// Lowers a call of the tuple constructor of variant `variant` with `args`.
    pub fn ctor_call(&mut self, expr: ExprId, variant: u32, args: List) -> Result<Option<Value>> {
        if let Some(addr) = self.materialize_const_expr(expr, (true, false))? {
            return Ok(Some(addr));
        }
        let mut values = Vec::new();
        for (index, at) in self.body.list(args).to_vec().into_iter().enumerate() {
            values.push((index as u32, self.eval_operand(ExprId(at))?));
        }
        self.construct(self.ty_of(expr), variant, values, None)
    }
}

impl<'l, 'a> Lower<'l, 'a> {
    fn write_const_bytes(&mut self, scope: ConstScope<'_>, (expr, ty): (ExprId, Ty), at: usize, out: &mut [u8], depth: u32, in_generic: bool) -> Option<()> {
        if scope.typed.coercions.contains_key(&expr) {
            return None;
        }
        self.write_const_uncoerced(scope, (expr, ty), at, out, depth, in_generic)
    }

    fn write_const_uncoerced(&mut self, scope: ConstScope<'_>, (expr, ty): (ExprId, Ty), at: usize, out: &mut [u8], depth: u32, in_generic: bool) -> Option<()> {
        if depth > CONST_RODATA_MAX_DEPTH {
            return None;
        }
        match *scope.body.expr(expr) {
            Expr::Paren(inner) => return self.write_const_bytes(scope, (inner, ty), at, out, depth, in_generic),
            Expr::Block { stmts, tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const } if scope.body.list(stmts).is_empty() => {
                return self.write_const_bytes(scope, (tail, ty), at, out, depth, in_generic);
            }
            Expr::Path(_) => {
                if let Some(&Res::Fn(item, substs)) = scope.typed.res.get(&expr)
                    && self.cx.item(item).1.kind == ItemKind::Const
                {
                    let (body, src, item_typed, root) = self.load_const_item(item, substs, ty, in_generic, false).ok()??;
                    let scoped = item_typed.scoped(&self.cx.types);
                    return self.write_const_bytes(ConstScope { body: &body, src, typed: &scoped, in_const: true }, (root, ty), at, out, depth + 1, in_generic);
                }
            }
            _ => {}
        }
        let kind = self.cx.types.kind(ty).clone();
        match (*scope.body.expr(expr), kind) {
            (_, Kind::Pat(inner, ..)) => self.write_const_uncoerced(scope, (expr, inner), at, out, depth, in_generic),
            (Expr::Array(list), Kind::Array(elem, _)) => {
                let stride = usize::try_from(self.cx.layout(elem).ok()?.size.bytes()).ok()?;
                let items: Vec<ExprId> = scope.body.list(list).iter().map(|&item| ExprId(item)).collect();
                items.into_iter().enumerate().try_for_each(|(index, item)| self.write_const_bytes(scope, (item, elem), at + index * stride, out, depth, in_generic))
            }
            (Expr::Repeat(item, count_expr), Kind::Array(elem, count)) => {
                if in_generic && !matches!(*scope.body.expr(count_expr), Expr::Lit(Lit::Int, _)) {
                    return None;
                }
                let stride = usize::try_from(self.cx.layout(elem).ok()?.size.bytes()).ok()?;
                let count = usize::try_from(self.cx.types.len(count)?).ok()?;
                if count == 0 {
                    let mut scratch = vec![0u8; stride];
                    self.write_const_bytes(scope, (item, elem), 0, &mut scratch, depth, in_generic)?;
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
                let elems: Vec<ExprId> = scope.body.list(list).iter().map(|&item| ExprId(item)).collect();
                if elems.len() != items.len() {
                    return None;
                }
                elems.into_iter().enumerate().try_for_each(|(index, item)| {
                    let offset = usize::try_from(field_offset(&layout, 0, index as u32)).ok()?;
                    self.write_const_bytes(scope, (item, items[index]), at + offset, out, depth, in_generic)
                })
            }
            (_, Kind::Adt(item, args)) => self.construct_const_bytes(scope, (expr, ty), (item, args), at, out, depth, in_generic),
            (_, Kind::Prim(primitive)) => {
                let (size, bits) = self.eval_const_scalar(scope, expr, primitive, depth, in_generic)?;
                out.get_mut(at..at + size)?.copy_from_slice(&bits.to_le_bytes()[..size]);
                Some(())
            }
            _ => None,
        }
    }

    fn construct_const_bytes(&mut self, scope: ConstScope<'_>, (expr, ty): (ExprId, Ty), (item, args): (ItemRef, Substs), at: usize, out: &mut [u8], depth: u32, in_generic: bool) -> Option<()> {
        if in_generic && !self.cx.types.substs(args).is_empty() {
            return None;
        }
        let (variant, fields) = const_ctor(scope, expr)?;
        let layout = self.cx.layout(ty).ok()?;
        if matches!(layout.fields, FieldsShape::Union(_)) {
            return None;
        }
        let args = self.cx.types.substs(args).to_vec();
        let types = field_types(self.cx, item, &args).ok()?.get(variant as usize)?.clone();
        for (index, value) in fields {
            let &field_ty = types.get(index as usize)?;
            let offset = usize::try_from(field_offset(&layout, variant, index)).ok()?;
            self.write_const_bytes(scope, (value, field_ty), at + offset, out, depth, in_generic)?;
        }
        if let Some((offset, tag, bits)) = tag_bits(self.cx, item, &layout, variant).ok()? {
            let size = tag.bytes() as usize;
            let start = at + usize::try_from(offset).ok()?;
            out.get_mut(start..start + size)?.copy_from_slice(&bits.to_le_bytes()[..size]);
        }
        Some(())
    }

    fn eval_const_path_scalar(&mut self, scope: ConstScope<'_>, expr: ExprId, primitive: PrimName, depth: u32, in_generic: bool) -> Option<(usize, u128)> {
        match scope.typed.res.get(&expr) {
            Some(&Res::Const(val_ty)) if !in_generic => {
                let Kind::Value(bits) = *self.cx.types.kind(val_ty) else { return None };
                mask_const_scalar(primitive, bits)
            }
            Some(&Res::Fn(item, substs)) if depth < CONST_RODATA_MAX_DEPTH && self.cx.item(item).1.kind == ItemKind::Const => {
                let prim_ty = self.cx.types.prim(primitive);
                let (body, src, item_typed, root) = self.load_const_item(item, substs, prim_ty, in_generic, false).ok()??;
                let scoped = item_typed.scoped(&self.cx.types);
                self.eval_const_scalar(ConstScope { body: &body, src, typed: &scoped, in_const: true }, root, primitive, depth + 1, in_generic)
            }
            _ => None,
        }
    }

    fn eval_const_scalar(&mut self, scope: ConstScope<'_>, expr: ExprId, primitive: PrimName, depth: u32, in_generic: bool) -> Option<(usize, u128)> {
        if depth > CONST_RODATA_MAX_DEPTH {
            return None;
        }
        let expr = unwrap_const_uncoerced(scope, expr)?;
        match *scope.body.expr(expr) {
            Expr::Path(_) => self.eval_const_path_scalar(scope, expr, primitive, depth, in_generic),
            Expr::Lit(lit, at) => eval_lit_bits(scope.src, lit, at, primitive),
            Expr::Unary(op, inner) if !scope.typed.operators.contains_key(&expr) => self.eval_const_unary(scope, op, inner, primitive, depth, in_generic),
            Expr::Binary(op, lhs, rhs) if !scope.typed.operators.contains_key(&expr) => self.eval_const_binary(scope, op, lhs, rhs, primitive, depth, in_generic),
            Expr::Index(arr, idx_expr) if !scope.typed.operators.contains_key(&expr) => self.eval_const_index(scope, arr, idx_expr, primitive, depth, in_generic),
            Expr::Cast(inner, _) => {
                let Kind::Prim(src_prim) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, inner)) else { return None };
                cast_const_scalar(self.eval_const_scalar(scope, inner, src_prim, depth, in_generic)?.1, src_prim, primitive)
            }
            _ => None,
        }
    }

    fn eval_const_index(&mut self, scope: ConstScope<'_>, arr: ExprId, idx_expr: ExprId, primitive: PrimName, depth: u32, in_generic: bool) -> Option<(usize, u128)> {
        if depth >= CONST_RODATA_MAX_DEPTH {
            return None;
        }
        let idx = usize::try_from(self.eval_const_scalar(scope, idx_expr, PrimName::Usize, depth + 1, in_generic)?.1).ok()?;
        let arr_ty = scope.typed.expr_ty(&self.cx.types, arr);
        let Kind::Array(elem_ty, count) = *self.cx.types.kind(arr_ty) else { return None };
        if !matches!(*self.cx.types.kind(elem_ty), Kind::Prim(p) if p == primitive) || idx >= usize::try_from(self.cx.types.len(count)?).ok()? {
            return None;
        }
        self.eval_const_array_elem(scope, arr, arr_ty, primitive, idx, depth, in_generic)
    }

    fn eval_const_array_elem(&mut self, scope: ConstScope<'_>, arr: ExprId, arr_ty: Ty, primitive: PrimName, idx: usize, depth: u32, in_generic: bool) -> Option<(usize, u128)> {
        let arr = unwrap_const_uncoerced(scope, arr)?;
        match *scope.body.expr(arr) {
            Expr::Path(_) => {
                let Some(&Res::Fn(item, substs)) = scope.typed.res.get(&arr) else { return None };
                if depth >= CONST_RODATA_MAX_DEPTH || self.cx.item(item).1.kind != ItemKind::Const {
                    return None;
                }
                let (body, src, item_typed, root) = self.load_const_item(item, substs, arr_ty, in_generic, false).ok()??;
                let scoped = item_typed.scoped(&self.cx.types);
                self.eval_const_array_elem(ConstScope { body: &body, src, typed: &scoped, in_const: true }, root, arr_ty, primitive, idx, depth + 1, in_generic)
            }
            Expr::Repeat(elem, count_expr) if !in_generic || matches!(*scope.body.expr(count_expr), Expr::Lit(Lit::Int, _)) => {
                self.eval_const_scalar(scope, elem, primitive, depth + 1, in_generic)
            }
            Expr::Array(list) => {
                let items = scope.body.list(list);
                if scope.in_const {
                    return self.eval_const_scalar(scope, ExprId(*items.get(idx)?), primitive, depth + 1, in_generic);
                }
                if items.len() > 16 {
                    return None;
                }
                items.to_vec().into_iter().enumerate().try_fold(None, |cur, (i, elem)| {
                    let val = self.eval_const_scalar(scope, ExprId(elem), primitive, depth + 1, in_generic)?;
                    Some(if i == idx { Some(val) } else { cur })
                })?
            }
            _ => None,
        }
    }

    fn eval_const_binary(&mut self, scope: ConstScope<'_>, op: BinOp, lhs: ExprId, rhs: ExprId, primitive: PrimName, depth: u32, in_generic: bool) -> Option<(usize, u128)> {
        if depth >= CONST_RODATA_MAX_DEPTH {
            return None;
        }
        if matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) && primitive == PrimName::Bool {
            let Kind::Prim(cmp_prim) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, lhs)) else { return None };
            let (bits, signed) = const_prim_info(cmp_prim)?;
            let l = self.eval_const_scalar(scope, lhs, cmp_prim, depth + 1, in_generic)?.1;
            let r = self.eval_const_scalar(scope, rhs, cmp_prim, depth + 1, in_generic)?.1;
            return Some((1, u128::from(eval_const_cmp(op, bits, signed, l, r))));
        }
        if matches!(op, BinOp::And | BinOp::Or) && primitive == PrimName::Bool {
            let (_, l) = self.eval_const_scalar(scope, lhs, PrimName::Bool, depth + 1, in_generic)?;
            if (op == BinOp::And) == (l == 0) {
                return Some((1, l));
            }
            return self.eval_const_scalar(scope, rhs, PrimName::Bool, depth + 1, in_generic);
        }
        if matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor) && primitive == PrimName::Bool {
            let l = self.eval_const_scalar(scope, lhs, PrimName::Bool, depth + 1, in_generic)?.1;
            let r = self.eval_const_scalar(scope, rhs, PrimName::Bool, depth + 1, in_generic)?.1;
            return Some((1, (match op { BinOp::BitAnd => l & r, BinOp::BitOr => l | r, _ => l ^ r }) & 1));
        }
        let (bits, signed) = int_info(primitive)?;
        let l = self.eval_const_scalar(scope, lhs, primitive, depth + 1, in_generic)?.1;
        let rhs_prim = if matches!(op, BinOp::Shl | BinOp::Shr) {
            let Kind::Prim(p) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, rhs)) else { return None };
            p
        } else {
            primitive
        };
        let (rhs_bits, rhs_signed) = int_info(rhs_prim)?;
        let r = self.eval_const_scalar(scope, rhs, rhs_prim, depth + 1, in_generic)?.1;
        Some((bits as usize / 8, eval_const_int_bin(op, (bits, signed), l, (rhs_bits, rhs_signed), r, self.overflow_checks)?))
    }

    fn eval_const_unary(&mut self, scope: ConstScope<'_>, op: UnOp, inner: ExprId, primitive: PrimName, depth: u32, in_generic: bool) -> Option<(usize, u128)> {
        match (op, primitive) {
            (UnOp::Not, PrimName::Bool) => Some((1, u128::from(self.eval_const_scalar(scope, inner, PrimName::Bool, depth, in_generic)?.1 == 0))),
            (UnOp::Not, _) => {
                let (bits, _) = int_info(primitive)?;
                let (size, val) = self.eval_const_scalar(scope, inner, primitive, depth, in_generic)?;
                Some((size, (!val) & mask(bits)))
            }
            (UnOp::Neg, PrimName::F32 | PrimName::F64) => {
                let (_, bits) = self.eval_const_scalar(scope, inner, primitive, depth, in_generic)?;
                neg_float_bits(primitive, bits)
            }
            (UnOp::Neg, _) => {
                let (bits, true) = int_info(primitive)? else { return None };
                if let Some(at) = bare_int_lit(scope.body, inner) {
                    let val = parse_int(scope.src.text(at)).ok()?.0;
                    return Some((bits as usize / 8, val.wrapping_neg() & mask(bits)));
                }
                let (size, val) = self.eval_const_scalar(scope, inner, primitive, depth, in_generic)?;
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
        Expr::Call(callee, args) => Some((variant(expr).or_else(|| variant(callee))?, scope.body.list(args).iter().enumerate().map(|(index, &at)| (index as u32, ExprId(at))).collect())),
        Expr::Struct { fields, base: None, .. } => {
            let fields = scope.body.list(fields).iter().map(|&at| scope.body.fields[at as usize].value).map(|value| Some((*scope.typed.inits.get(&value)?, value))).collect::<Option<Vec<_>>>()?;
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
            let size = if primitive == PrimName::Bool { 1 } else { width as usize / 8 };
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
            Expr::Block { stmts, tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const } if scope.body.list(stmts).is_empty() => expr = tail,
            _ => return Some(expr),
        }
    }
}

fn sign_extend_u128(val: u128, bits: u32) -> i128 {
    let shift = 128 - bits;
    ((val << shift) as i128) >> shift
}

fn eval_const_cmp(op: BinOp, bits: u32, signed: bool, l: u128, r: u128) -> bool {
    let ord = if signed { sign_extend_u128(l, bits).cmp(&sign_extend_u128(r, bits)) } else { l.cmp(&r) };
    matches!((op, ord), (BinOp::Eq, std::cmp::Ordering::Equal) | (BinOp::Ne, std::cmp::Ordering::Less | std::cmp::Ordering::Greater) | (BinOp::Lt | BinOp::Le, std::cmp::Ordering::Less) | (BinOp::Gt | BinOp::Ge, std::cmp::Ordering::Greater) | (BinOp::Le | BinOp::Ge, std::cmp::Ordering::Equal))
}

fn eval_const_int_bin(op: BinOp, (bits, signed): (u32, bool), l: u128, (rhs_bits, rhs_signed): (u32, bool), r: u128, overflow_checks: bool) -> Option<u128> {
    let m = mask(bits);
    match op {
        BinOp::BitAnd => Some((l & r) & m),
        BinOp::BitOr => Some((l | r) & m),
        BinOp::BitXor => Some((l ^ r) & m),
        BinOp::Shl | BinOp::Shr => {
            let bad = if rhs_signed { let s = sign_extend_u128(r, rhs_bits); s < 0 || s >= i128::from(bits) } else { r >= u128::from(bits) };
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
            Some((if op == BinOp::Div { ls.wrapping_div(rs) } else { ls.wrapping_rem(rs) } as u128) & m)
        }
        BinOp::Add | BinOp::Sub | BinOp::Mul if signed => {
            let (ls, rs) = (sign_extend_u128(l, bits), sign_extend_u128(r, bits));
            if overflow_checks {
                let (min, max) = (-1_i128 << (bits - 1), (m >> 1) as i128);
                return match op { BinOp::Add => ls.checked_add(rs), BinOp::Sub => ls.checked_sub(rs), _ => ls.checked_mul(rs) }.filter(|&v| v >= min && v <= max).map(|v| (v as u128) & m);
            }
            Some((match op { BinOp::Add => ls.wrapping_add(rs), BinOp::Sub => ls.wrapping_sub(rs), _ => ls.wrapping_mul(rs) } as u128) & m)
        }
        BinOp::Add | BinOp::Sub | BinOp::Mul => {
            if overflow_checks {
                return match op { BinOp::Add => l.checked_add(r), BinOp::Sub => l.checked_sub(r), _ => l.checked_mul(r) }.filter(|&v| v <= m);
            }
            Some(match op { BinOp::Add => l.wrapping_add(r), BinOp::Sub => l.wrapping_sub(r), _ => l.wrapping_mul(r) } & m)
        }
        _ => None,
    }
}

fn cast_const_scalar(src_val: u128, src_prim: PrimName, dst_prim: PrimName) -> Option<(usize, u128)> {
    if dst_prim == PrimName::Char {
        return matches!(src_prim, PrimName::U8 | PrimName::Char).then_some((4, src_val & 0xFFFF_FFFF));
    }
    let (dst_bits, _) = int_info(dst_prim)?;
    let (src_bits, src_signed) = const_prim_info(src_prim)?;
    let bits = if src_signed { sign_extend_u128(src_val, src_bits) as u128 } else { src_val };
    Some((dst_bits as usize / 8, bits & mask(dst_bits)))
}
