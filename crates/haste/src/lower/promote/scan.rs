use anyhow::Result;
use haste_codegen::{Type, Value};

use crate::body::{BlockKind, Expr, ExprId, List, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::infer::{Res, Shape, Typed};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::stencil::{RodataReloc, RodataTarget, SymbolSource};
use super::super::{Lower, Repr, repr};
use super::literal::Scope;

impl Lower<'_, '_> {
    /// Whether `&mut inner` can promote into a program-lifetime writable cell: outside loops in a `static mut` initializer (or any ZST static initializer).
    pub(in super::super) fn can_promote_mut_static(&mut self, inner: ExprId) -> Result<bool> {
        if !self.loops.is_empty() {
            return Ok(false);
        }
        let item = self.reached.list[self.index].0.0;
        if self.cx.item(item).1.kind != ItemKind::Static {
            return Ok(false);
        }
        Ok(self.is_mutable_static(item) || self.cx.layout(self.ty_of(inner))?.size.bytes() == 0)
    }

    /// Whether the function being lowered is a `const` or `static` initializer.
    pub(in super::super) fn in_const_or_static(&self) -> bool {
        let item = self.reached.list[self.index].0.0;
        matches!(
            self.cx.item(item).1.kind,
            ItemKind::Const | ItemKind::Static
        )
    }

    /// Whether `&expr` names a value that lives for the whole program: a literal, a constant, a function, a closure that captures nothing, a constructor, a shared reference to one, an aggregate or constructor call of them, or a field of one; in a constant or static initializer, which the compiler evaluates once, also a call, operator or block of them.
    pub fn promotable(&mut self, expr: ExprId) -> Result<bool> {
        self.promotable_in(expr, self.in_const_or_static())
    }

    fn promotable_in(&mut self, expr: ExprId, constant: bool) -> Result<bool> {
        if let Some(inner) = self.transparent(expr) {
            return self.promotable_in(inner, constant);
        }
        if !constant && crate::sema::infer::needs_drop(self.cx, self.ty_of(expr))? {
            return Ok(false);
        }
        Ok(match *self.body.expr(expr) {
            Expr::Lit(..) => true,
            Expr::Closure { .. } => !self.cx.closure_takes_env(self.ty_of(expr)),
            Expr::Unary(UnOp::Neg, inner)
            | Expr::Field(inner, _)
            | Expr::TupleField(inner, _)
            | Expr::Ref {
                mutable: false,
                raw: false,
                expr: inner,
            }
            | Expr::Repeat(inner, _) => self.promotable_in(inner, constant)?,
            Expr::Path(_) => self.promotable_path(expr),
            Expr::Call(callee, list) => self.promotable_call(callee, list, constant)?,
            Expr::Array(list) | Expr::Tuple(list) => self.promotable_list(list, constant)?,
            Expr::Range { from, to, .. } => {
                self.promotable_opt(from, constant)? && self.promotable_opt(to, constant)?
            }
            Expr::Struct { fields, base, .. } => self.promotable_struct(fields, base, constant)?,
            _ => constant && self.promotable_const_expr(expr)?,
        })
    }

    fn promotable_const_expr(&mut self, expr: ExprId) -> Result<bool> {
        Ok(match *self.body.expr(expr) {
            Expr::Method {
                receiver, params, ..
            } => self.promotable_in(receiver, true)? && self.promotable_list(params, true)?,
            Expr::Unary(_, inner) | Expr::Cast(inner, _) => self.promotable_in(inner, true)?,
            Expr::Binary(_, left, right) | Expr::Index(left, right) => {
                self.promotable_in(left, true)? && self.promotable_in(right, true)?
            }
            Expr::Block {
                stmts,
                tail: Some(tail),
                label: None,
                kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const,
            } => self.body.list(stmts).is_empty() && self.promotable_in(tail, true)?,
            Expr::If(cond, then, alt) => {
                self.promotable_in(cond, true)?
                    && self.promotable_in(then, true)?
                    && self.promotable_opt(alt, true)?
            }
            _ => false,
        })
    }

    fn promotable_path(&self, expr: ExprId) -> bool {
        match self.typed.res.get(&expr) {
            Some(Res::Fn(item, _)) => {
                matches!(self.cx.item(*item).1.kind, ItemKind::Const | ItemKind::Fn)
            }
            Some(Res::Ctor(..) | Res::Const(_)) => true,
            _ => false,
        }
    }

    fn promotable_call(&mut self, callee: ExprId, list: List, constant: bool) -> Result<bool> {
        let callable = match self.typed.res.get(&callee) {
            Some(Res::Ctor(..)) => true,
            Some(Res::Fn(item, _)) => constant || self.cx.rustc_promotable(*item),
            _ => false,
        };
        Ok(callable && self.promotable_list(list, constant)?)
    }

    fn promotable_list(&mut self, list: List, constant: bool) -> Result<bool> {
        for &at in self.body.list(list).to_vec().iter() {
            if !self.promotable_in(ExprId(at), constant)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn promotable_opt(&mut self, expr: Option<ExprId>, constant: bool) -> Result<bool> {
        expr.map_or(Ok(true), |e| self.promotable_in(e, constant))
    }

    fn promotable_struct(
        &mut self,
        fields: List,
        base: Option<ExprId>,
        constant: bool,
    ) -> Result<bool> {
        for &at in self.body.list(fields).to_vec().iter() {
            if !self.promotable_in(self.body.fields[at as usize].value, constant)? {
                return Ok(false);
            }
        }
        self.promotable_opt(base, constant)
    }

    /// Whether `expr` evaluates purely to all-zero literal bytes within `LITERAL_BYTES`.
    pub(in super::super) fn literal_is_zero(&mut self, expr: ExprId) -> Result<bool> {
        if !self
            .cx
            .types
            .substs(self.reached.list[self.index].0.1)
            .is_empty()
        {
            return Ok(false);
        }
        let ty = self.ty_of(expr);
        let in_const = matches!(
            self.cx.item(self.reached.list[self.index].0.0).1.kind,
            ItemKind::Const | ItemKind::Static
        );
        let (body, src, typed) = (self.body, self.src, self.typed);
        Ok(self
            .literal_blob(
                Scope {
                    body,
                    src,
                    typed,
                    in_const,
                },
                (expr, ty),
                0,
            )
            .is_some_and(|(bytes, relocs)| relocs.is_empty() && bytes.iter().all(|&b| b == 0)))
    }

    /// The value of the constant `item` at `substs` that `expr` reads: an immediate, or bytes laid out once, when its body is literals of a crate that cannot change while haste runs; else a call of its body.
    pub(in super::super) fn constant(
        &mut self,
        expr: ExprId,
        item: ItemRef,
        substs: Substs,
    ) -> Result<Option<Value>> {
        if let Some(value) = self.literal_constant(expr, item, substs)? {
            return Ok(Some(value));
        }
        match self.literal_reference(item, substs)? {
            Some(value) => Ok(Some(value)),
            None => self.invoke(
                super::super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::ExprRes(expr)),
                item,
                substs,
                Vec::new(),
            ),
        }
    }

    /// The definition and inference of the constant `item` at `substs` when lowering may read its value from its body: a body that takes no type arguments, of a crate whose sources cannot change while haste runs.
    pub(in super::super) fn literal_source(
        &mut self,
        item: ItemRef,
        substs: Substs,
    ) -> Result<Option<(ItemRef, std::sync::Arc<Typed>)>> {
        let index = self.reached_index(item, substs)?;
        let ((source, _), Ok(typed)) = &self.reached.list[index] else {
            return Ok(None);
        };
        let env = &self.cx.outline.crates[source.krate as usize].env;
        let readable = matches!(typed.shape, Shape::Body)
            && (env.release || env.fixed)
            && self.cx.types.substs(substs).is_empty();
        Ok(readable.then(|| (*source, typed.clone())))
    }

    fn literal_constant(
        &mut self,
        expr: ExprId,
        item: ItemRef,
        substs: Substs,
    ) -> Result<Option<Value>> {
        let Some((source, typed)) = self.literal_source(item, substs)? else {
            return Ok(None);
        };
        let typed = typed.scoped(&self.cx.types);
        let ty = typed.ret_ty();
        let body = self.cx.body(source)?;
        let (Some(root), src) = (body.root, self.cx.body_src(source)) else {
            return Ok(None);
        };
        let Some(blob) = self.literal_blob(
            Scope {
                body: &body,
                src,
                typed: &typed,
                in_const: true,
            },
            (root, ty),
            0,
        ) else {
            return Ok(None);
        };
        self.materialize_blob(expr, ty, blob)
    }

    /// Resolves a function item to a reached index when safe to embed as a constant rodata relocation.
    pub(in super::super) fn rodata_fn_target(
        &mut self,
        item: ItemRef,
        substs: Substs,
    ) -> Option<usize> {
        let enclosing_generic = !self
            .cx
            .types
            .substs(self.reached.list[self.index].0.1)
            .is_empty();
        if (enclosing_generic && !self.cx.types.substs(substs).is_empty())
            || self.cx.types.substs_have_params(substs)
            || self.cx.intrinsic(item).is_some()
        {
            return None;
        }
        self.reached_index(item, substs).ok()
    }

    fn emit_rodata_blob(&mut self, bytes: &[u8], relocs: &[RodataReloc]) -> Value {
        if relocs.is_empty() {
            self.emit_const_rodata(bytes)
        } else {
            let id = self.slots.rodata(bytes, relocs);
            self.emit_symbol(SymbolSource::Rodata(id))
        }
    }

    fn materialize_mem_blob(
        &mut self,
        expr: ExprId,
        ty: Ty,
        bytes: &[u8],
        relocs: &[RodataReloc],
    ) -> Result<Option<Value>> {
        if bytes.len() == 16
            && let [
                RodataReloc {
                    offset: 0,
                    target: RodataTarget::Rodata(child_id),
                },
            ] = relocs
            && matches!(*self.cx.types.kind(ty), Kind::Ref(Mut::Not, _))
        {
            let data = self.emit_child_rodata(*child_id);
            let len = u64::from_le_bytes(bytes[8..].try_into()?);
            return Ok(Some(self.wide_pointer(data, len as i64)));
        }
        if self.lent_mutably(expr) {
            return Ok(None);
        }
        if relocs.is_empty() {
            return self.materialize_value_rodata(ty, bytes);
        }
        Ok(Some(self.emit_rodata_blob(bytes, relocs)))
    }

    pub(super) fn materialize_blob(
        &mut self,
        expr: ExprId,
        ty: Ty,
        (bytes, relocs): (Vec<u8>, Vec<RodataReloc>),
    ) -> Result<Option<Value>> {
        match repr(self.cx, ty)? {
            Repr::Scalar(machine) if relocs.is_empty() => {
                let mut word = [0; 16];
                word[..bytes.len().min(16)].copy_from_slice(&bytes[..bytes.len().min(16)]);
                Ok(Some(self.int_const(machine, u128::from_le_bytes(word))))
            }
            Repr::Scalar(Type::I64) if bytes.len() == 8 => match relocs.as_slice() {
                [
                    RodataReloc {
                        offset: 0,
                        target: RodataTarget::Rodata(child_id),
                    },
                ] => Ok(Some(self.emit_child_rodata(*child_id))),
                _ => Ok(None),
            },
            Repr::Mem { .. } => self.materialize_mem_blob(expr, ty, &bytes, &relocs),
            _ => Ok(None),
        }
    }

    fn literal_reference(&mut self, item: ItemRef, substs: Substs) -> Result<Option<Value>> {
        let Some((source, typed)) = self.literal_source(item, substs)? else {
            return Ok(None);
        };
        let typed = typed.scoped(&self.cx.types);
        let ty = typed.ret_ty();
        let Kind::Ref(Mut::Not, pointee) = *self.cx.types.kind(ty) else {
            return Ok(None);
        };
        let body = self.cx.body(source)?;
        let (Some(root), src) = (body.root, self.cx.body_src(source)) else {
            return Ok(None);
        };
        let mut root = root;
        while let Expr::Paren(inner) = *body.expr(root) {
            root = inner;
        }
        let Expr::Ref {
            mutable: false,
            raw: false,
            expr: inner,
        } = *body.expr(root)
        else {
            return Ok(None);
        };
        let inner_ty = typed.expr_ty(&self.cx.types, inner);
        let Some((bytes, relocs)) = self.literal_blob(
            Scope {
                body: &body,
                src,
                typed: &typed,
                in_const: true,
            },
            (inner, inner_ty),
            0,
        ) else {
            return Ok(None);
        };
        let data = self.emit_rodata_blob(&bytes, &relocs);
        Ok(
            match (
                self.cx.types.kind(pointee).clone(),
                self.cx.types.kind(inner_ty).clone(),
            ) {
                (Kind::Slice(_), Kind::Array(_, len)) => self
                    .cx
                    .types
                    .len(len)
                    .map(|len| self.wide_pointer(data, len as i64)),
                (Kind::Slice(_), _) => None,
                _ if pointee == inner_ty => Some(data),
                _ => None,
            },
        )
    }

    pub(super) fn literal_place(&mut self, expr: ExprId) -> Result<Option<Value>> {
        let inner = self.transparent(expr).unwrap_or(expr);
        if let Some(&Res::Fn(item, substs)) = self
            .typed
            .res
            .get(&inner)
            .filter(|_| matches!(self.body.expr(inner), Expr::Path(_)))
            && self.cx.item(item).1.kind == ItemKind::Const
            && !self.lent_mutably(expr)
            && matches!(
                repr(self.cx, self.ty_of(inner))?,
                Repr::Scalar(_) | Repr::Mem { .. }
            )
            && let Some((source, typed)) = self.literal_source(item, substs)?
        {
            let typed = typed.scoped(&self.cx.types);
            let ty = typed.ret_ty();
            let body = self.cx.body(source)?;
            if let (Some(root), src) = (body.root, self.cx.body_src(source))
                && let Some((bytes, relocs)) = self.literal_blob(
                    Scope {
                        body: &body,
                        src,
                        typed: &typed,
                        in_const: true,
                    },
                    (root, ty),
                    0,
                )
            {
                return Ok(Some(self.emit_rodata_blob(&bytes, &relocs)));
            }
        }
        if !self.lent_mutably(expr)
            && self
                .cx
                .types
                .substs(self.reached.list[self.index].0.1)
                .is_empty()
            && !self.holds_uninit(expr)?
        {
            let ty = self.ty_of(expr);
            let (body, src, typed) = (self.body, self.src, self.typed);
            if let Some((bytes, relocs)) = self.literal_blob(
                Scope {
                    body,
                    src,
                    typed,
                    in_const: true,
                },
                (expr, ty),
                0,
            ) {
                return Ok(Some(self.emit_rodata_blob(&bytes, &relocs)));
            }
        }
        Ok(None)
    }

    fn lent_mutably(&self, expr: ExprId) -> bool {
        let mutable = |at: ExprId| match *self.body.expr(at) {
            Expr::Assign(lhs, _)
            | Expr::AssignOp(_, lhs, _)
            | Expr::Ref {
                mutable: true,
                expr: lhs,
                ..
            } => self.place_root(lhs) == expr,
            Expr::Method { receiver, .. } => {
                self.typed
                    .adjusts
                    .get(&at)
                    .is_some_and(|adjust| adjust.autoref == Some(Mut::Mut))
                    && self.place_root(receiver) == expr
            }
            _ => false,
        };
        self.typed
            .ref_binds
            .values()
            .any(|binding| *binding == Mut::Mut)
            || (0..self.body.exprs.len() as u32).map(ExprId).any(mutable)
    }

    fn place_root(&self, expr: ExprId) -> ExprId {
        match *self.body.expr(expr) {
            Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Index(base, _) => {
                self.place_root(base)
            }
            _ => self
                .transparent(expr)
                .map_or(expr, |inner| self.place_root(inner)),
        }
    }
}
