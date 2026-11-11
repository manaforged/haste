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
        match self.overloaded_deref(ty, Mut::Not)? {
            Some((res, target)) => {
                self.out.derefs.insert((expr, 0), res);
                Ok(target)
            }
            None => bail!("cannot dereference {}", self.show(ty)),
        }
    }

    /// The type that the compiler's own dereference of `ty` reaches: the pointee of a reference or pointer, or what a `Box` owns.
    pub(super) fn builtin_deref(&self, ty: Ty) -> Option<Ty> {
        match self.cx.types.kind(ty) {
            Kind::Ref(_, inner) | Kind::Ptr(_, inner) => Some(*inner),
            _ => self.cx.boxed(ty),
        }
    }

    /// The type whose field `field` the access `expr` on a value of type `found` reads, after it dereferences references, every `Box` whose own fields are private here, and `Deref` impls; records the steps on `expr`.
    pub(super) fn field_base(&mut self, expr: ExprId, found: Ty, field: &str) -> Result<Ty> {
        let (mut ty, mut steps) = (self.known_ty(found)?, 0);
        loop {
            let next = match self.cx.types.kind(ty).clone() {
                Kind::Ref(_, inner) => Some(self.known_ty(inner)?),
                Kind::Adt(item, _) if self.cx.boxed(ty).is_some() && !self.sees_private(item)? => self.cx.boxed(ty),
                Kind::Adt(..) if !self.has_field(ty, field)? => self.deref_step(expr, steps, ty)?,
                _ => None,
            };
            let Some(next) = next else { break };
            (ty, steps) = (self.resolve(next), steps + 1);
            if steps as usize == crate::RECURSION_LIMIT {
                bail!("reached the recursion limit while dereferencing {}", self.show(found));
            }
        }
        if steps > 0 {
            self.out.autoderefs.insert(expr, steps);
        }
        Ok(ty)
    }

    fn has_field(&mut self, ty: Ty, field: &str) -> Result<bool> {
        let Kind::Adt(item, _) = *self.cx.types.kind(ty) else { return Ok(false) };
        Ok(self.cx.item(item).1.kind == ItemKind::Struct && self.field_index(item, 0, field)?.is_some())
    }

    /// The type that step `step` of the autoderef of `expr` reaches from `ty` through `Deref::deref`, recorded on `(expr, step)`; `None` when `ty` has no `Deref` impl.
    pub(super) fn deref_step(&mut self, expr: ExprId, step: u32, ty: Ty) -> Result<Option<Ty>> {
        let Some((res, target)) = self.overloaded_deref(ty, Mut::Not)? else { return Ok(None) };
        self.out.derefs.insert((expr, step), res);
        Ok(Some(target))
    }

    /// Makes each `Deref::deref` step among the first `count` autoderef steps of `expr` from `found` call `DerefMut::deref_mut`.
    pub(super) fn steps_mutably(&mut self, expr: ExprId, found: Ty, count: u32) -> Result<()> {
        let mut ty = self.resolve(found);
        for step in 0..count {
            let next = match self.out.derefs.contains_key(&(expr, step)) {
                true => {
                    let Some((res, target)) = self.overloaded_deref(ty, Mut::Mut)? else { bail!("{} cannot be dereferenced mutably", self.show(ty)) };
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
        while !matches!(self.cx.types.kind(ty), Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..) | Kind::Ptr(..))
            && let Some(inner) = self.builtin_deref(ty)
        {
            (ty, steps) = (self.resolve(inner), steps + 1);
        }
        if steps > 0 {
            self.out.autoderefs.insert(expr, steps);
        }
        ty
    }

    fn index_mutably(&mut self, expr: ExprId) -> Result<()> {
        let Some(Res::Fn(_, substs)) = self.out.operators.get(&expr).cloned() else { return Ok(()) };
        let Some(method) = self.cx.lang_item("index_mut").and_then(|of| self.cx.trait_member(of, "index_mut")) else { bail!("core has no `IndexMut::index_mut`") };
        self.out.operators.insert(expr, Res::Fn(method, substs));
        Ok(())
    }

    fn sees_private(&mut self, item: crate::names::ItemRef) -> Result<bool> {
        let module = self.cx.module_of(item)?;
        Ok(self.cx.paths(self.home.krate).within(self.home, module))
    }

    pub(super) fn overloaded_deref(&mut self, ty: Ty, mutability: Mut) -> Result<Option<(Res, Ty)>> {
        let (lang, name) = match mutability {
            Mut::Not => ("deref", "deref"),
            Mut::Mut => ("deref_mut", "deref_mut"),
        };
        let Some(method) = self.cx.lang_item(lang).and_then(|of| self.cx.trait_member(of, name)) else { bail!("core has no `{lang}` trait") };
        let receiver = self.cx.types.intern(Kind::Ref(mutability, ty));
        let Some((item, substs)) = self.trait_method(method, receiver)? else { return Ok(None) };
        let (_, ret) = self.fn_types(item, &substs)?;
        let Kind::Ref(_, target) = *self.cx.types.kind(self.resolve(ret)) else { bail!("`{}` returns a value that is not a reference", self.cx.name(item)) };
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
