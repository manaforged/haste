mod scalar;

pub(super) use scalar::{
    apply_int_binary, apply_shift, cast_scalar, cmp_scalar, prim_info, sign_extend,
};
pub(in crate::lower) use scalar::{eval_lit_bits, neg_float_bits};

use haste_codegen::Value;

use crate::body::literal::string_bytes;
use crate::body::{BlockKind, Body, Expr, ExprId, Lit};
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::infer::{Coercion, Res, Typed, field_types};
use crate::sema::layout::field_offset;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::Lower;
use super::super::stencil::{RodataId, RodataReloc, RodataTarget, SymbolSource};
use scalar::prim_scalar_bytes;

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
    fn child_rodata(&mut self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
        let id = self.slots.rodata(bytes, relocs);
        let known = relocs.is_empty().then(|| bytes.to_vec());
        match self.child_rodata.iter_mut().find(|(seen, _)| *seen == id) {
            Some((_, kept)) if *kept != known => *kept = None,
            Some(_) => {}
            None => self.child_rodata.push((id, known)),
        }
        id
    }

    /// The address of interned literal blob `id`, with its bytes visible to constant folding when `child_rodata` kept them.
    pub(super) fn emit_child_rodata(&mut self, id: RodataId) -> Value {
        match self.child_rodata.iter().find(|(seen, _)| *seen == id) {
            Some((_, Some(bytes))) => {
                let bytes = bytes.clone();
                self.emit_rodata_id(id, &bytes)
            }
            _ => self.emit_symbol(SymbolSource::Rodata(id)),
        }
    }

    /// Returns the bytes and intra-rodata relocations of `expr` when it evaluates from literals and pure constants within `LITERAL_BYTES`.
    pub(super) fn literal_blob(
        &mut self,
        scope: Scope<'_>,
        (expr, ty): (ExprId, Ty),
        depth: u32,
    ) -> Option<(Vec<u8>, Vec<RodataReloc>)> {
        if depth > MAX_CONST_DEPTH || ty == Ty::NONE || self.cx.types.has_params(ty) {
            return None;
        }
        let layout = self.cx.layout(ty).ok()?;
        if layout.align.bytes() > 16 {
            return None;
        }
        let size = usize::try_from(layout.size.bytes())
            .ok()
            .filter(|size| *size <= LITERAL_BYTES)?;
        let mut out = vec![0; size];
        let mut relocs = Vec::new();
        self.write_blob(scope, (expr, ty), 0, (&mut out, &mut relocs), depth)?;
        Some((out, relocs))
    }

    fn write_blob(
        &mut self,
        scope: Scope<'_>,
        (expr, ty): (ExprId, Ty),
        at: usize,
        (out, relocs): (&mut [u8], &mut Vec<RodataReloc>),
        depth: u32,
    ) -> Option<()> {
        match scope.typed.coercions.get(&expr) {
            None => self.write_uncoerced(scope, (expr, ty), at, (out, relocs), depth),
            Some(&Coercion::Len(len_ty)) => self
                .write_slice_coercion(scope, (expr, ty), len_ty, at, (out, relocs), depth)
                .or_else(|| self.eval_const_into(scope, (expr, ty), at, (out, relocs), depth)),
            Some(Coercion::Deref(..)) => {
                self.eval_const_into(scope, (expr, ty), at, (out, relocs), depth)
            }
            Some(&Coercion::Reify(Res::Fn(item, substs))) => {
                let idx = self.rodata_fn_target(item, substs)?;
                out.get_mut(at..at + 8)?.fill(0);
                relocs.push(RodataReloc {
                    offset: u32::try_from(at).ok()?,
                    target: RodataTarget::Fn(idx),
                });
                Some(())
            }
            Some(_) => None,
        }
    }

    fn write_slice_coercion(
        &mut self,
        scope: Scope<'_>,
        (expr, ty): (ExprId, Ty),
        len_ty: Ty,
        at: usize,
        (out, relocs): (&mut [u8], &mut Vec<RodataReloc>),
        depth: u32,
    ) -> Option<()> {
        let Kind::Ref(Mut::Not, pointee) = *self.cx.types.kind(ty) else {
            return None;
        };
        if !matches!(*self.cx.types.kind(pointee), Kind::Slice(_)) {
            return None;
        }
        let orig_ty = scope.typed.expr_ty(&self.cx.types, expr);
        let Kind::Ref(Mut::Not, orig_pointee) = *self.cx.types.kind(orig_ty) else {
            return None;
        };
        if !matches!(*self.cx.types.kind(orig_pointee), Kind::Array(..)) {
            return None;
        }
        let mut ptr_buf = [0u8; 8];
        let mut ptr_relocs = Vec::new();
        self.write_uncoerced(
            scope,
            (expr, orig_ty),
            0,
            (&mut ptr_buf, &mut ptr_relocs),
            depth,
        )?;
        let &[reloc] = ptr_relocs.as_slice() else {
            return None;
        };
        if reloc.offset != 0 {
            return None;
        }
        let len = self.cx.types.len(len_ty)?;
        out.get_mut(at..at + 16)?[8..16].copy_from_slice(&len.to_le_bytes());
        relocs.push(RodataReloc {
            offset: at as u32,
            target: reloc.target,
        });
        Some(())
    }

    fn peel_step(body: &Body, expr: ExprId) -> Option<ExprId> {
        match *body.expr(expr) {
            Expr::Paren(inner) => Some(inner),
            Expr::Block {
                stmts,
                tail: Some(tail),
                label: None,
                kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const,
            } if body.list(stmts).is_empty() => Some(tail),
            _ => None,
        }
    }

    fn peel_uncoerced<'b>(scope: Scope<'b>, mut expr: ExprId) -> ExprId {
        while let Some(inner) =
            Self::peel_step(scope.body, expr).filter(|i| !scope.typed.coercions.contains_key(i))
        {
            expr = inner;
        }
        expr
    }

    fn write_str_or_bytes(
        &mut self,
        scope: Scope<'_>,
        (lit, span): (Lit, u32),
        pointee: Ty,
        at: usize,
        (out, relocs): (&mut [u8], &mut Vec<RodataReloc>),
    ) -> Option<()> {
        let bytes = string_bytes(scope.src.text(span)).ok()?;
        if lit == Lit::Str && *self.cx.types.kind(pointee) == Kind::Prim(Primitive::Str) {
            let child_id = self.child_rodata(&bytes, &[]);
            out.get_mut(at..at + 16)?[8..16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
            relocs.push(RodataReloc {
                offset: at as u32,
                target: RodataTarget::Rodata(child_id),
            });
            return Some(());
        }
        let Kind::Array(elem, len_ty) = *self.cx.types.kind(pointee) else {
            return None;
        };
        if lit != Lit::ByteStr
            || *self.cx.types.kind(elem) != Kind::Prim(Primitive::U8)
            || self.cx.types.len(len_ty)? != bytes.len() as u64
        {
            return None;
        }
        let child_id = self.child_rodata(&bytes, &[]);
        out.get_mut(at..at + 8)?;
        relocs.push(RodataReloc {
            offset: at as u32,
            target: RodataTarget::Rodata(child_id),
        });
        Some(())
    }

    fn write_ref_blob(
        &mut self,
        scope: Scope<'_>,
        (inner, pointee): (ExprId, Ty),
        at: usize,
        (out, relocs): (&mut [u8], &mut Vec<RodataReloc>),
        depth: u32,
    ) -> Option<()> {
        let inner_ty = scope.typed.expr_ty(&self.cx.types, inner);
        if !super::super::is_freeze_ty(self.cx, inner_ty, 0) {
            return None;
        }
        let (child_bytes, child_relocs) = self.literal_blob(scope, (inner, inner_ty), depth + 1)?;
        let child_id = self.child_rodata(&child_bytes, &child_relocs);
        match (
            self.cx.types.kind(pointee).clone(),
            self.cx.types.kind(inner_ty).clone(),
        ) {
            _ if pointee == inner_ty => {
                out.get_mut(at..at + 8)?;
            }
            (Kind::Slice(_), Kind::Array(_, len_ty)) => {
                let len = self.cx.types.len(len_ty)?;
                out.get_mut(at..at + 16)?[8..16].copy_from_slice(&len.to_le_bytes());
            }
            _ => return None,
        }
        relocs.push(RodataReloc {
            offset: at as u32,
            target: RodataTarget::Rodata(child_id),
        });
        Some(())
    }

    fn write_repeat_blob(
        &mut self,
        scope: Scope<'_>,
        (item, elem, count): (ExprId, Ty, Ty),
        at: usize,
        (out, relocs): (&mut [u8], &mut Vec<RodataReloc>),
        depth: u32,
    ) -> Option<()> {
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
                relocs.push(RodataReloc {
                    offset: (reloc.offset as usize - at + dst) as u32,
                    target: reloc.target,
                });
            }
        }
        Some(())
    }

    fn write_seq_blob(
        &mut self,
        scope: Scope<'_>,
        (expr, ty): (ExprId, Ty),
        at: usize,
        (out, relocs): (&mut [u8], &mut Vec<RodataReloc>),
        depth: u32,
    ) -> Option<()> {
        match (*scope.body.expr(expr), self.cx.types.kind(ty).clone()) {
            (Expr::Array(list), Kind::Array(elem, _)) => {
                let stride = self.cx.layout(elem).ok()?.size.bytes() as usize;
                let items: Vec<ExprId> = scope
                    .body
                    .list(list)
                    .iter()
                    .map(|&item| ExprId(item))
                    .collect();
                items.into_iter().enumerate().try_for_each(|(i, item)| {
                    self.write_blob(scope, (item, elem), at + i * stride, (out, relocs), depth)
                })
            }
            (Expr::Repeat(item, _), Kind::Array(elem, count)) => {
                self.write_repeat_blob(scope, (item, elem, count), at, (out, relocs), depth)
            }
            (Expr::Tuple(list), Kind::Tuple(items)) => {
                let layout = self.cx.layout(ty).ok()?;
                let items = self.cx.types.substs(items).to_vec();
                let fields: Vec<(u32, ExprId)> = scope
                    .body
                    .list(list)
                    .iter()
                    .enumerate()
                    .map(|(i, &item)| (i as u32, ExprId(item)))
                    .collect();
                fields.into_iter().try_for_each(|(i, item)| {
                    self.write_blob(
                        scope,
                        (item, *items.get(i as usize)?),
                        at + field_offset(&layout, 0, i) as usize,
                        (out, relocs),
                        depth,
                    )
                })
            }
            _ => None,
        }
    }

    fn write_uncoerced(
        &mut self,
        scope: Scope<'_>,
        (expr, ty): (ExprId, Ty),
        at: usize,
        (out, relocs): (&mut [u8], &mut Vec<RodataReloc>),
        depth: u32,
    ) -> Option<()> {
        let expr = Self::peel_uncoerced(scope, expr);
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        if let Some(inner) = Self::peel_step(scope.body, expr) {
            return self.write_blob(scope, (inner, ty), at, (out, relocs), depth);
        }
        if let Expr::Path(_) = *scope.body.expr(expr)
            && let Some(&Res::Fn(item, substs)) = scope.typed.res.get(&expr)
            && self.cx.item(item).1.kind == ItemKind::Const
        {
            return self.write_const_item(item, substs, ty, at, (out, relocs), depth);
        }
        match (*scope.body.expr(expr), self.cx.types.kind(ty).clone()) {
            (Expr::Lit(lit @ (Lit::Str | Lit::ByteStr), span), Kind::Ref(Mut::Not, pointee)) => {
                self.write_str_or_bytes(scope, (lit, span), pointee, at, (out, relocs))
            }
            (
                Expr::Ref {
                    mutable: false,
                    raw: false,
                    expr: inner,
                },
                Kind::Ref(Mut::Not, pointee),
            ) => self.write_ref_blob(scope, (inner, pointee), at, (out, relocs), depth),
            (Expr::Array(_) | Expr::Repeat(..) | Expr::Tuple(_), _) => {
                self.write_seq_blob(scope, (expr, ty), at, (out, relocs), depth)
            }
            (_, Kind::Adt(item, args)) => self
                .construct_blob(scope, (expr, ty), (item, args), at, (out, relocs), depth)
                .or_else(|| self.eval_const_into(scope, (expr, ty), at, (out, relocs), depth)),
            (_, Kind::Prim(primitive)) => {
                let (size, bits) = (
                    prim_scalar_bytes(primitive)?,
                    self.eval_scalar(scope, expr, primitive, depth)?,
                );
                out.get_mut(at..at + size)?
                    .copy_from_slice(&bits.to_le_bytes()[..size]);
                Some(())
            }
            _ => self.eval_const_into(scope, (expr, ty), at, (out, relocs), depth),
        }
    }

    fn write_const_item(
        &mut self,
        item: ItemRef,
        substs: Substs,
        ty: Ty,
        at: usize,
        (out, relocs): (&mut [u8], &mut Vec<RodataReloc>),
        depth: u32,
    ) -> Option<()> {
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
        self.write_blob(
            Scope {
                body: &body,
                src,
                typed: &const_typed,
                in_const: true,
            },
            (root, ty),
            at,
            (out, relocs),
            depth + 1,
        )
    }

    fn construct_blob(
        &mut self,
        scope: Scope<'_>,
        (expr, ty): (ExprId, Ty),
        (item, args): (ItemRef, Substs),
        at: usize,
        (out, relocs): (&mut [u8], &mut Vec<RodataReloc>),
        depth: u32,
    ) -> Option<()> {
        let (variant, fields) = constructor(scope, expr)?;
        let layout = self.cx.layout(ty).ok()?;
        let args = self.cx.types.substs(args).to_vec();
        let types = field_types(self.cx, item, &args)
            .ok()?
            .get(variant as usize)?
            .clone();
        for (index, value) in fields {
            self.write_blob(
                scope,
                (value, *types.get(index as usize)?),
                at + field_offset(&layout, variant, index) as usize,
                (out, relocs),
                depth,
            )?;
        }
        if let Some((offset, tag, bits)) =
            super::super::adt::tag_bits(self.cx, item, &layout, variant).ok()?
        {
            let size = tag.bytes() as usize;
            let start = at + offset as usize;
            out.get_mut(start..start + size)?
                .copy_from_slice(&bits.to_le_bytes()[..size]);
        }
        Some(())
    }
}

fn constructor(scope: Scope<'_>, expr: ExprId) -> Option<(u32, Vec<(u32, ExprId)>)> {
    let variant = |at: ExprId| match scope.typed.res.get(&at) {
        Some(&Res::Ctor(_, variant, _)) => Some(variant),
        _ => None,
    };
    match *scope.body.expr(expr) {
        Expr::Path(_) if !scope.typed.coercions.contains_key(&expr) => {
            Some((variant(expr)?, Vec::new()))
        }
        Expr::Call(callee, args) => Some((
            variant(callee)?,
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
