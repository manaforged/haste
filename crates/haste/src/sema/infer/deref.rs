use anyhow::{Result, bail};

use crate::body::{Expr, ExprId, UnOp};

use crate::outline::item::ItemKind;

use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Res};

impl Infer<'_, '_> {
    /// Infers the dereference `expr` of a value of type `found`: through a pointer, a `Box`, or `Deref::deref`.
    pub(super) fn deref(&mut self, expr: ExprId, found: Ty) -> Result<Ty> {
        let ty = self.known_ty(found)?;
        if let Some(inner) = self.builtin_deref(ty) {
            return Ok(inner);
        }
        if matches!(self.cx.types.kind(ty), Kind::Proj(_)) {
            self.poly_fallback();
        }
        match self.overloaded_deref(ty, Mut::Not)? {
            Some((res, target)) => {
                self.out.derefs.insert((expr, 0), res);
                Ok(target)
            }
            None => bail!("cannot dereference {}", self.show(ty)),
        }
    }

    /// The type that the compiler's own dereference of `ty` reaches: the pointee of a reference or pointer, or what a `Box` owns.
    pub(in crate::sema) fn builtin_deref(&self, ty: Ty) -> Option<Ty> {
        match self.cx.types.kind(ty) {
            Kind::Ref(_, inner) | Kind::Ptr(_, inner) => Some(*inner),
            _ => self.cx.boxed(ty),
        }
    }

    /// The type whose field `field` the access `expr` on a value of type `found` reads, after it dereferences references, every `Box` whose own fields are private here, and `Deref` impls; records the steps on `expr`.
    pub(super) fn field_base(&mut self, expr: ExprId, found: Ty, field: &str) -> Result<Ty> {
        let (mut ty, mut steps) = (self.known_ty(found)?, 0);
        loop {
            let Some(next) = self.next_field_step(expr, steps, ty, field)? else {
                break;
            };
            (ty, steps) = (self.resolve(next), steps + 1);
            if steps as usize == crate::RECURSION_LIMIT {
                bail!(
                    "reached the recursion limit while dereferencing {}",
                    self.show(found)
                );
            }
        }
        if steps > 0 {
            self.out.autoderefs.insert(expr, steps);
        }
        Ok(ty)
    }

    fn next_field_step(
        &mut self,
        expr: ExprId,
        steps: u32,
        ty: Ty,
        field: &str,
    ) -> Result<Option<Ty>> {
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(_, inner) => self.known_ty(inner).map(Some),
            Kind::Adt(item, _) if self.cx.boxed(ty).is_some() && !self.sees_private(item)? => {
                Ok(self.cx.boxed(ty))
            }
            Kind::Adt(..) if !self.has_field(ty, field)? => self.deref_step(expr, steps, ty),
            Kind::Param(_) | Kind::Proj(_) => {
                self.poly_fallback();
                self.deref_step(expr, steps, ty)
            }
            _ => Ok(None),
        }
    }

    fn has_field(&mut self, ty: Ty, field: &str) -> Result<bool> {
        let Kind::Adt(item, _) = *self.cx.types.kind(ty) else {
            return Ok(false);
        };
        if self.cx.item(item).1.kind != ItemKind::Struct || self.cx.foreign(item) {
            return Ok(false);
        }
        let Some(index) = self.field_index(item, 0, field)? else {
            return Ok(false);
        };
        self.field_visible(item, index, None)
    }

    /// The type that step `step` of the autoderef of `expr` reaches from `ty` through `Deref::deref`, recorded on `(expr, step)`; `None` when `ty` has no `Deref` impl.
    pub(super) fn deref_step(&mut self, expr: ExprId, step: u32, ty: Ty) -> Result<Option<Ty>> {
        if matches!(self.cx.types.kind(ty), Kind::Proj(_)) {
            self.poly_fallback();
        }
        self.deref_step_raw(expr, step, ty)
    }

    pub(super) fn deref_step_raw(&mut self, expr: ExprId, step: u32, ty: Ty) -> Result<Option<Ty>> {
        let Some((res, target)) = self.overloaded_deref(ty, Mut::Not)? else {
            return Ok(None);
        };
        self.out.derefs.insert((expr, step), res);
        Ok(Some(target))
    }

    /// Makes each `Deref::deref` step among the first `count` autoderef steps of `expr` from `found` call `DerefMut::deref_mut`.
    pub(super) fn steps_mutably(&mut self, expr: ExprId, found: Ty, count: u32) -> Result<()> {
        let mut ty = self.resolve(found);
        for step in 0..count {
            let next = match self.out.derefs.contains_key(&(expr, step)) {
                true => {
                    let Some((res, target)) = self.overloaded_deref(ty, Mut::Mut)? else {
                        bail!("{} cannot be dereferenced mutably", self.show(ty))
                    };
                    self.out.derefs.insert((expr, step), res);
                    Some(target)
                }
                false => self.builtin_deref(ty),
            };
            let Some(next) = next else { break };
            ty = self.resolve(next);
        }
        Ok(())
    }

    /// The callable type that the call `expr` of a callee of type `found` invokes, after it dereferences references and `Box`es; records the count on `expr`.
    pub(super) fn callee_base(&mut self, expr: ExprId, found: Ty) -> Ty {
        let (mut ty, mut steps) = (self.resolve(found), 0);
        while !matches!(
            self.cx.types.kind(ty),
            Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..) | Kind::Ptr(..)
        ) && let Some(inner) = self.builtin_deref(ty)
        {
            (ty, steps) = (self.resolve(inner), steps + 1);
        }
        if steps > 0 {
            self.out.autoderefs.insert(expr, steps);
        }
        ty
    }

    fn index_mutably(&mut self, expr: ExprId) -> Result<()> {
        let Some(Res::Fn(_, substs)) = self.out.operators.get(&expr).cloned() else {
            return Ok(());
        };
        let Some(method) = self
            .cx
            .lang_item("index_mut")
            .and_then(|of| self.cx.trait_member(of, "index_mut"))
        else {
            bail!("core has no `IndexMut::index_mut`")
        };
        self.out.operators.insert(expr, Res::Fn(method, substs));
        Ok(())
    }

    fn sees_private(&mut self, item: crate::names::ItemRef) -> Result<bool> {
        self.sees_private_from(self.home, item)
    }

    fn sees_private_from(
        &mut self,
        from: crate::names::ModRef,
        item: crate::names::ItemRef,
    ) -> Result<bool> {
        let module = self.cx.module_of(item)?;
        Ok(from.krate == module.krate && self.cx.paths(from.krate).within(from, module))
    }

    pub(super) fn macro_home_of(&mut self, tok: Option<u32>) -> Option<crate::names::ModRef> {
        let tok = tok?;
        let crate::outline::source::Mark::Home { krate, file, item } = self.src.mark(tok)? else {
            return None;
        };
        self.cx
            .module_of(crate::names::ItemRef { krate, file, item })
            .ok()
    }

    fn field_visible_from(
        &mut self,
        from: crate::names::ModRef,
        item: crate::names::ItemRef,
        vis: crate::body::adt::FieldVis,
    ) -> Result<bool> {
        match vis {
            crate::body::adt::FieldVis::Public => Ok(true),
            crate::body::adt::FieldVis::Crate => Ok(from.krate == item.krate),
            crate::body::adt::FieldVis::Super => {
                let module = self.cx.module_of(item)?;
                if from.krate != module.krate {
                    return Ok(false);
                }
                let paths = self.cx.paths(from.krate);
                let parent = paths.parent(module).unwrap_or(module);
                Ok(paths.within(from, parent))
            }
            crate::body::adt::FieldVis::Private => self.sees_private_from(from, item),
        }
    }

    pub(super) fn field_visible(
        &mut self,
        item: crate::names::ItemRef,
        index: u32,
        tok: Option<u32>,
    ) -> Result<bool> {
        if self.cx.item(item).1.kind == ItemKind::Enum || self.cx.foreign(item) {
            return Ok(true);
        }
        let Some(vis) = self
            .cx
            .adt_def(item)?
            .variants
            .first()
            .and_then(|v| v.fields.get(index as usize))
            .map(|f| f.vis)
        else {
            return Ok(false);
        };
        if self.field_visible_from(self.home, item, vis)? {
            return Ok(true);
        }
        if let Some(macro_home) = self.macro_home_of(tok) {
            return self.field_visible_from(macro_home, item, vis);
        }
        Ok(false)
    }

    pub(super) fn check_field_visible(
        &mut self,
        item: crate::names::ItemRef,
        index: u32,
        site: Option<ExprId>,
        tok: Option<u32>,
        field_name: &str,
    ) -> Result<()> {
        if self.cx.outline.crates[self.home.krate as usize].env.fixed {
            return Ok(());
        }
        let check_tok = tok.or_else(|| site.and_then(|e| self.body.expr_span(e).map(|(s, _)| s)));
        if self.field_visible(item, index, check_tok)? {
            return Ok(());
        }
        let msg = format!(
            "field `{field_name}` of struct `{}` is private",
            self.cx.name(item)
        );
        if let Some(t) = tok {
            return Err(self.diag_tok(
                crate::diag::DiagnosticKind::Privacy,
                t,
                msg,
                "private field",
            ));
        }
        Err(self.diag_expr(
            crate::diag::DiagnosticKind::Privacy,
            site,
            msg,
            "private field",
        ))
    }

    pub(super) fn overloaded_deref(
        &mut self,
        ty: Ty,
        mutability: Mut,
    ) -> Result<Option<(Res, Ty)>> {
        let (lang, name) = match mutability {
            Mut::Not => ("deref", "deref"),
            Mut::Mut => ("deref_mut", "deref_mut"),
        };
        let Some(method) = self
            .cx
            .lang_item(lang)
            .and_then(|of| self.cx.trait_member(of, name))
        else {
            bail!("core has no `{lang}` trait")
        };
        let receiver = self.cx.types.intern(Kind::Ref(mutability, ty));
        let Some((item, substs)) = self.trait_method(method, receiver)? else {
            return Ok(None);
        };
        let (_, ret) = self.fn_types(item, &substs)?;
        let Kind::Ref(_, target) = *self.cx.types.kind(self.resolve(ret)) else {
            bail!(
                "`{}` returns a value that is not a reference",
                self.cx.name(item)
            )
        };
        Ok(Some((Res::fn_of(&self.cx.types, item, &substs), target)))
    }

    /// Makes every overloaded dereference in the place expression `expr` call `DerefMut::deref_mut`, as a write or a unique borrow of the place needs.
    pub(super) fn deref_mutably(&mut self, expr: ExprId) -> Result<()> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.deref_mutably(inner),
            Expr::Index(inner, _) => {
                self.index_mutably(expr)?;
                let count = self.out.autoderefs.get(&expr).copied().unwrap_or(0);
                self.steps_mutably(expr, self.out.exprs[inner.0 as usize], count)?;
                self.deref_mutably(inner)
            }
            Expr::Field(inner, _) | Expr::TupleField(inner, _) => {
                let count = self.out.autoderefs.get(&expr).copied().unwrap_or(0);
                self.steps_mutably(expr, self.out.exprs[inner.0 as usize], count)?;
                self.deref_mutably(inner)
            }
            Expr::Unary(UnOp::Deref, inner) => {
                self.steps_mutably(expr, self.out.exprs[inner.0 as usize], 1)?;
                self.deref_mutably(inner)
            }
            _ => Ok(()),
        }
    }
}
