use anyhow::{Result, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::body::{
    Args, Body, Bound, Expr, ExprId, GenericArg, List, PathId, Stmt, Ty as SyntaxTy, TyId,
};
use crate::diag::DiagnosticKind;
use crate::names::{ItemRef, ModRef, Primitive};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;

use super::super::Context;
use super::super::evaluate;
use super::super::substs::Substs;
use super::super::ty::{Kind, Ty, int_info};
use super::{Infer, detached, scope};

impl Infer<'_, '_> {
    fn union_field_allowed(&mut self, ty: Ty) -> Result<bool> {
        let ty = self.resolve(ty);
        if let Kind::Adt(adt, _) = *self.cx.types.kind(ty)
            && self.cx.lang(adt) == Some("manually_drop")
        {
            return Ok(true);
        }
        self.is_copy(ty)
    }
}

/// Validates that every field of the union `item` is `Copy` or `ManuallyDrop<T>` (E0740).
pub(in crate::sema) fn check_union_fields(cx: &mut Context<'_>, item: ItemRef) -> Result<()> {
    if cx.outline.is_sysroot(item.krate) || cx.foreign(item) {
        return Ok(());
    }
    let def = cx.adt_def(item)?;
    let Some(variant) = def.variants.first() else {
        return Ok(());
    };
    if variant.fields.is_empty() {
        return Ok(());
    }
    let param_count = scope::type_params(&def.generics.params).count();
    if param_count > u16::MAX as usize {
        return Ok(());
    }
    detached(cx, item, |infer| {
        let before = infer.cx.quiet_mark();
        let args: Vec<Ty> = (0..param_count as u16)
            .map(|idx| infer.cx.types.intern(Kind::Param(idx)))
            .collect();
        if param_count > 0 {
            infer.poly = Some(Box::new(super::poly::PolyState::default()));
        }
        let frame = infer.adt_frame(item, &def.generics.params, &args)?;
        infer.swap_frame(frame);
        if param_count > 0 {
            infer.init_poly_env(item)?;
        }
        for field in &variant.fields {
            let ty = infer.lower_ty(&def.body, field.ty)?;
            if infer.cx.quiet_mark() != before {
                return Ok(());
            }
            let ty = infer.resolve(ty);
            if !infer.union_field_allowed(ty)? {
                if infer.cx.quiet_mark() != before {
                    return Ok(());
                }
                let field_name = field.name.map_or("_", |tok| infer.src.text(tok));
                bail!(
                    "field `{field_name}` of union `{}` must be `Copy` or `ManuallyDrop<T>`: type `{}` does not implement `Copy`",
                    infer.cx.name(item),
                    infer.show(ty),
                );
            }
        }
        Ok(())
    })
}

/// Validates concrete field types of union `item` during layout computation (E0740).
pub(in crate::sema) fn check_union_concrete_fields(
    cx: &mut Context<'_>,
    item: ItemRef,
    fields: &[Ty],
) -> Result<()> {
    if cx.outline.is_sysroot(item.krate) || cx.foreign(item) || fields.is_empty() {
        return Ok(());
    }
    detached(cx, item, |infer| {
        let before = infer.cx.quiet_mark();
        for &ty in fields {
            let ty = infer.resolve(ty);
            if infer.has_vars(ty) || infer.has_params(ty) {
                continue;
            }
            if !infer.union_field_allowed(ty)? {
                if infer.cx.quiet_mark() != before {
                    return Ok(());
                }
                bail!(
                    "union `{}` field of type `{}` must be `Copy` or `ManuallyDrop<T>`",
                    infer.cx.name(item),
                    infer.show(ty),
                );
            }
        }
        Ok(())
    })
}

impl Infer<'_, '_> {
    /// Checks that a non-`mut`, non-`thread_local` static item's declared type implements `Sync`.
    pub(in crate::sema) fn check_static_item_sync(
        cx: &mut Context<'_>,
        item: ItemRef,
    ) -> Result<()> {
        if cx.outline.is_sysroot(item.krate)
            || cx.foreign(item)
            || cx.thread_local(item)
            || cx.is_static_mut(item)
        {
            return Ok(());
        }
        detached(cx, item, |infer| {
            let sig = infer.cx.signature(item)?;
            let Some(ret) = sig.ret else {
                return Ok(());
            };
            let before = infer.cx.quiet_mark();
            let ty = infer.lower_ty(&sig.body, ret)?;
            let ty = infer.resolve(ty);
            if infer.cx.quiet_mark() != before || infer.has_vars(ty) || infer.has_params(ty) {
                return Ok(());
            }
            let Some(sync) = infer.cx.lang_item("sync") else {
                return Ok(());
            };
            if infer.thread_auto_holds(sync, ty) || infer.cx.quiet_mark() != before {
                return Ok(());
            }
            bail!(
                "`{}` cannot be shared between threads safely",
                infer.show(ty)
            );
        })
    }

    /// Validates generic bounds on a local ADT instantiation `item<args>` (E0277).
    pub(super) fn check_adt_bounds(
        &mut self,
        expr: Option<ExprId>,
        item: ItemRef,
        args: Substs,
    ) -> Result<()> {
        if !self.root_is_user() || self.cx.outline.is_fixed(item.krate) || self.cx.foreign(item) {
            return Ok(());
        }
        let slice = self.cx.types.substs(args).to_vec();
        if slice.is_empty() {
            return Ok(());
        }
        let def = self.cx.adt_def(item)?;
        if def.generics.predicates.is_empty()
            && def.generics.params.iter().all(|g| g.bounds.len == 0)
        {
            return Ok(());
        }
        let param_count = scope::type_params(&def.generics.params).count();
        if slice.len() != param_count {
            return Ok(());
        }
        let frame = self.adt_frame(item, &def.generics.params, &slice)?;
        let home = frame.home;
        let mut wanted = Vec::new();
        for (generic, &arg) in scope::type_params(&def.generics.params).zip(&slice) {
            wanted.extend(def.body.list(generic.bounds).iter().map(|&b| (arg, b)));
        }
        self.with_frame(frame, |infer| {
            for (arg, bound) in wanted {
                infer.require_adt_bound(&def.body, home, expr, arg, bound)?;
            }
            for predicate in &def.generics.predicates {
                let Some(ty_id) = predicate.ty else { continue };
                let ty = infer.lower_ty(&def.body, ty_id)?;
                for &bound in def.body.list(predicate.bounds) {
                    infer.require_adt_bound(&def.body, home, expr, ty, bound)?;
                }
            }
            Ok(())
        })
    }

    fn require_adt_bound(
        &mut self,
        arena: &Body,
        home: ModRef,
        expr: Option<ExprId>,
        ty: Ty,
        bound: u32,
    ) -> Result<()> {
        let Some((of, path)) = self.bound_trait_at(arena, bound, home) else {
            return Ok(());
        };
        let ty = self.resolve(ty);
        if self.has_params(ty) && self.poly.is_none() {
            return Ok(());
        }
        if self.has_vars(ty) {
            let args = self.bound_args(of, arena, path, ty)?;
            let mut list = Vec::with_capacity(1 + args.len());
            list.push(ty);
            list.extend(args);
            self.note_obligation_site(of, ty);
            self.obligations.push((of, list));
            return Ok(());
        }
        let before = self.cx.quiet_mark();
        if self.bound_holds(arena, home, (ty, bound), 0)? || self.cx.quiet_mark() != before {
            return Ok(());
        }
        let shown = self.show(ty);
        let trait_name = self.cx.name(of);
        Err(self.diag_expr(
            DiagnosticKind::Trait,
            expr,
            format!("the trait bound `{shown}: {trait_name}` is not satisfied"),
            format!("the trait `{trait_name}` is not implemented for `{shown}`"),
        ))
    }

    /// Evaluates an inline `const { ... }` block at compile time to catch arithmetic/division errors.
    pub(super) fn check_inline_const(&mut self, expr: ExprId, ty: Ty) -> Result<()> {
        if !self.root_is_user() {
            return Ok(());
        }
        let resolved = self.resolve(ty);
        if self.has_params(resolved) {
            return Ok(());
        }
        let body = self.body;
        self.check_inline_const_expr(body, expr)
    }

    fn check_inline_const_expr(&mut self, arena: &Body, expr: ExprId) -> Result<()> {
        let want = self
            .out
            .exprs
            .get(expr.0 as usize)
            .copied()
            .filter(|&ty| ty != Ty::NONE)
            .and_then(|ty| match *self.cx.types.kind(self.resolve(ty)) {
                Kind::Prim(prim) => {
                    int_info(prim).or((prim == Primitive::Bool).then_some((1, false)))
                }
                Kind::IntVar(_) => Some((32, true)),
                _ => None,
            });
        match self.const_eval(arena, expr, want) {
            Ok(_) => return Ok(()),
            Err(err) if evaluate::is_const_arithmetic_error(&err) => {
                return Err(self.diag_expr(
                    DiagnosticKind::Type,
                    Some(expr),
                    err.to_string(),
                    "evaluation of constant value failed",
                ));
            }
            Err(_) => {}
        }
        match *arena.expr(expr) {
            Expr::Paren(inner) | Expr::Unary(_, inner) | Expr::Cast(inner, _) => {
                self.check_inline_const_expr(arena, inner)
            }
            Expr::Ref { expr: inner, .. } => self.check_inline_const_expr(arena, inner),
            Expr::Binary(_, lhs, rhs) => {
                self.check_inline_const_expr(arena, lhs)?;
                self.check_inline_const_expr(arena, rhs)
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                for &id in arena.list(list) {
                    self.check_inline_const_expr(arena, ExprId(id))?;
                }
                Ok(())
            }
            Expr::Repeat(val, _) => self.check_inline_const_expr(arena, val),
            Expr::Block { stmts, tail, .. } => {
                for &id in arena.list(stmts) {
                    match arena.stmts[id as usize] {
                        Stmt::Let {
                            init, otherwise, ..
                        } => {
                            if let Some(e) = init {
                                self.check_inline_const_expr(arena, e)?;
                            }
                            if let Some(e) = otherwise {
                                self.check_inline_const_expr(arena, e)?;
                            }
                        }
                        Stmt::Expr(e, _) => self.check_inline_const_expr(arena, e)?,
                        Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
                    }
                }
                if let Some(tail) = tail {
                    self.check_inline_const_expr(arena, tail)?;
                }
                Ok(())
            }
            Expr::If(cond, then, otherwise) => {
                self.check_inline_const_expr(arena, cond)?;
                self.check_inline_const_expr(arena, then)?;
                if let Some(alt) = otherwise {
                    self.check_inline_const_expr(arena, alt)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Checks all types in a user body for `dyn Trait` object safety (E0038) and local ADT generic bounds (E0277).
    pub(super) fn check_body_user_types(&mut self) -> Result<()> {
        let mut seen = HashSet::default();
        let mut checked_traits = HashSet::default();
        let locals: Vec<Ty> = self.out.locals.iter().map(|l| l.ty).collect();
        let exprs: Vec<(ExprId, Ty)> = self
            .out
            .exprs
            .iter()
            .enumerate()
            .filter_map(|(i, &ty)| (ty != Ty::NONE).then_some((ExprId(i as u32), ty)))
            .collect();
        if let Some(ret) = self.out.ret {
            self.check_dyn_in_ty(None, ret, &mut seen, &mut checked_traits)?;
        }
        for ty in locals {
            if ty == Ty::NONE {
                continue;
            }
            self.check_dyn_in_ty(None, ty, &mut seen, &mut checked_traits)?;
            if let Kind::Adt(item, args) = *self.cx.types.kind(self.resolve(ty)) {
                self.check_adt_bounds(None, item, args)?;
            }
        }
        for (expr, ty) in exprs {
            self.check_dyn_in_ty(Some(expr), ty, &mut seen, &mut checked_traits)?;
        }
        Ok(())
    }

    /// Validates `dyn Trait` object safety on a local concrete ADT's field types.
    pub(in crate::sema) fn check_item_dyn_object_safety(
        cx: &mut Context<'_>,
        item: ItemRef,
    ) -> Result<()> {
        if cx.outline.is_fixed(item.krate) || cx.foreign(item) {
            return Ok(());
        }
        let def = cx.adt_def(item)?;
        if scope::type_params(&def.generics.params).next().is_some() {
            return Ok(());
        }
        detached(cx, item, |infer| {
            let fields = infer.adt_fields(item, &[])?;
            let mut seen = HashSet::default();
            let mut checked_traits = HashSet::default();
            for &ty in fields.iter().flatten() {
                infer.check_dyn_in_ty(None, ty, &mut seen, &mut checked_traits)?;
            }
            Ok(())
        })
    }

    fn check_dyn_in_ty(
        &mut self,
        expr: Option<ExprId>,
        ty: Ty,
        seen: &mut HashSet<Ty>,
        checked_traits: &mut HashSet<ItemRef>,
    ) -> Result<()> {
        let ty = self.resolve(ty);
        if ty == Ty::NONE || !seen.insert(ty) {
            return Ok(());
        }
        let kind = self.cx.types.kind(ty).clone();
        if let Kind::Dyn(of, _, _, _) = kind
            && checked_traits.insert(of)
        {
            self.check_trait_object_safe(expr, of)?;
        }
        let children: Vec<_> = kind.children(&self.cx.types).collect();
        for child in children {
            self.check_dyn_in_ty(expr, child, seen, checked_traits)?;
        }
        Ok(())
    }

    fn check_trait_object_safe(&mut self, expr: Option<ExprId>, of: ItemRef) -> Result<()> {
        let implied = self.implied_traits(vec![of])?;
        let trait_name = self.cx.name(of);
        if let Some(sized) = self.cx.lang_item("sized")
            && implied.contains(&sized)
        {
            return Err(self.diag_expr(
                DiagnosticKind::Trait,
                expr,
                format!("the trait `{trait_name}` cannot be made into an object"),
                format!("`{trait_name}` requires `Self: Sized`"),
            ));
        }
        for owner in implied {
            let tree = &self.cx.outline.crates[owner.krate as usize];
            for member in self.cx.children(owner) {
                let (_, entry) = self.cx.item(member);
                if entry.kind != ItemKind::Fn || !tree.env.enabled(&entry.attrs) {
                    continue;
                }
                if self.needs_sized_self(member)? {
                    continue;
                }
                let Ok(sig) = self.cx.signature(member) else {
                    continue;
                };
                let method_name = self.cx.name(member);
                if scope::type_params(&sig.generics).next().is_some() {
                    return Err(self.diag_expr(
                        DiagnosticKind::Trait,
                        expr,
                        format!("the trait `{trait_name}` cannot be made into an object"),
                        format!("method `{method_name}` has generic type parameters"),
                    ));
                }
                if sig.receiver.is_none() {
                    return Err(self.diag_expr(
                        DiagnosticKind::Trait,
                        expr,
                        format!("the trait `{trait_name}` cannot be made into an object"),
                        format!("associated function `{method_name}` has no `self` receiver"),
                    ));
                }
                let msrc = &self.cx.file(member).source;
                let returns_self = sig
                    .ret
                    .is_some_and(|r| ty_mentions_bare_self(&sig.body, msrc, r));
                let param_self = sig
                    .params
                    .iter()
                    .any(|&(_, p)| ty_mentions_bare_self(&sig.body, msrc, p));
                if returns_self || param_self {
                    return Err(self.diag_expr(
                        DiagnosticKind::Trait,
                        expr,
                        format!("the trait `{trait_name}` cannot be made into an object"),
                        format!("method `{method_name}` references the `Self` type"),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn ty_mentions_bare_self(arena: &Body, src: &Source, ty: TyId) -> bool {
    match *arena.ty(ty) {
        SyntaxTy::Path(path) => path_mentions_bare_self(arena, src, path),
        SyntaxTy::Ref { ty, .. }
        | SyntaxTy::Ptr { ty, .. }
        | SyntaxTy::Slice(ty)
        | SyntaxTy::Array(ty, _)
        | SyntaxTy::Paren(ty)
        | SyntaxTy::Pat(ty, _) => ty_mentions_bare_self(arena, src, ty),
        SyntaxTy::Tuple(list) => arena
            .list(list)
            .iter()
            .any(|&id| ty_mentions_bare_self(arena, src, TyId(id))),
        SyntaxTy::Fn { params, ret, .. } => {
            arena
                .list(params)
                .iter()
                .any(|&id| ty_mentions_bare_self(arena, src, TyId(id)))
                || ret.is_some_and(|r| ty_mentions_bare_self(arena, src, r))
        }
        SyntaxTy::ImplTrait(_, list) | SyntaxTy::DynTrait(list) => {
            bounds_mention_bare_self(arena, src, list)
        }
        _ => false,
    }
}

fn bounds_mention_bare_self(arena: &Body, src: &Source, list: List) -> bool {
    arena
        .list(list)
        .iter()
        .any(|&b| match arena.bounds[b as usize] {
            Bound::Trait { path, .. } => path_mentions_bare_self(arena, src, path),
            _ => false,
        })
}

fn path_mentions_bare_self(arena: &Body, src: &Source, path_id: PathId) -> bool {
    let path = arena.path(path_id);
    if let Some((qself, trait_path)) = path.qself
        && (ty_mentions_bare_self(arena, src, qself)
            || trait_path.is_some_and(|tp| path_mentions_bare_self(arena, src, tp)))
    {
        return true;
    }
    let segs = arena.list(path.segments);
    if let [only] = segs {
        let seg = arena.segments[*only as usize];
        if path.qself.is_none()
            && !path.global
            && seg.args.is_none()
            && src.text(seg.name) == "Self"
        {
            return true;
        }
    }
    segs.iter().any(|&s| {
        arena.segments[s as usize]
            .args
            .is_some_and(|args| args_mention_bare_self(arena, src, args))
    })
}

fn args_mention_bare_self(arena: &Body, src: &Source, args: Args) -> bool {
    match args {
        Args::Angle(list) => arena
            .list(list)
            .iter()
            .any(|&id| match arena.args[id as usize] {
                GenericArg::Ty(ty) | GenericArg::Binding { ty, .. } => {
                    ty_mentions_bare_self(arena, src, ty)
                }
                GenericArg::Constraint { bounds, .. } => {
                    bounds_mention_bare_self(arena, src, bounds)
                }
                _ => false,
            }),
        Args::Paren { inputs, output } => {
            arena
                .list(inputs)
                .iter()
                .any(|&id| ty_mentions_bare_self(arena, src, TyId(id)))
                || output.is_some_and(|r| ty_mentions_bare_self(arena, src, r))
        }
        Args::ReturnType => false,
    }
}
