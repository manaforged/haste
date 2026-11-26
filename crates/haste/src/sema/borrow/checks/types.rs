use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::GenericKind;
use crate::body::{BlockKind, Expr, ExprId, Stmt, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::impls::ImplRef;
use crate::sema::infer::{By, Res};
use crate::sema::ty::{Kind, Ty};

use super::super::Checker;

impl Checker<'_, '_> {
    pub(in super::super) fn ty_holds_borrow(&mut self, ty: Ty) -> bool {
        let mut seen = HashSet::default();
        self.ty_holds_borrow_inner(ty, &mut seen)
    }

    fn ty_holds_borrow_inner(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        if ty == Ty::NONE || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(..) | Kind::Dyn(..) => true,
            Kind::Slice(elem) | Kind::Array(elem, _) | Kind::Pat(elem, ..) => {
                self.ty_holds_borrow_inner(elem, seen)
            }
            Kind::Tuple(items) => {
                let items = self.cx.types.substs(items).to_vec();
                items
                    .into_iter()
                    .any(|item| self.ty_holds_borrow_inner(item, seen))
            }
            Kind::Closure(..) => self.closure_ty_holds_borrow(ty, seen),
            Kind::Adt(item, args) => {
                let args = self.cx.types.substs(args).to_vec();
                if args
                    .into_iter()
                    .any(|arg| self.ty_holds_borrow_inner(arg, seen))
                {
                    return true;
                }
                self.cx.adt_def(item).is_ok_and(|def| {
                    def.generics
                        .params
                        .iter()
                        .any(|g| g.kind == GenericKind::Lifetime)
                })
            }
            _ => false,
        }
    }

    fn closure_ty_holds_borrow(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        let Kind::Closure(parent, _, at) = *self.cx.types.kind(ty) else {
            return false;
        };
        if let Some(info) = self.cx.closure(ty) {
            return info.closure.captures.iter().any(|c| {
                c.by != By::Value
                    || self.ty_holds_borrow_inner(
                        info.typed
                            .locals
                            .get(c.local as usize)
                            .map_or(Ty::NONE, |l| l.ty),
                        seen,
                    )
            });
        }
        if parent == self.item
            && let Some(closure) = self.typed.closures.get(&ExprId(at))
        {
            return closure.captures.iter().any(|c| {
                c.by != By::Value || self.ty_holds_borrow_inner(self.local_ty(c.local), seen)
            });
        }
        true
    }

    pub(in super::super) fn ty_holds_closure(&mut self, ty: Ty) -> bool {
        let mut seen = HashSet::default();
        self.ty_holds_closure_inner(ty, &mut seen)
    }

    fn ty_holds_closure_inner(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        if ty == Ty::NONE || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty).clone() {
            Kind::Closure(..) | Kind::Dyn(..) | Kind::FnDef(..) | Kind::FnPtr(..) => true,
            Kind::Ref(_, inner)
            | Kind::Slice(inner)
            | Kind::Array(inner, _)
            | Kind::Pat(inner, ..) => self.ty_holds_closure_inner(inner, seen),
            Kind::Tuple(items) => {
                let items = self.cx.types.substs(items).to_vec();
                items
                    .into_iter()
                    .any(|item| self.ty_holds_closure_inner(item, seen))
            }
            Kind::Adt(item, items) => {
                if self.is_iter_mod(item) {
                    return true;
                }
                let items = self.cx.types.substs(items).to_vec();
                items
                    .into_iter()
                    .any(|item| self.ty_holds_closure_inner(item, seen))
            }
            _ => false,
        }
    }

    pub(in super::super) fn item_in_std_mod(&mut self, item: ItemRef, submod: &str) -> bool {
        let mod_path = self.cx.module_text(item);
        let Some(rest) = mod_path
            .strip_prefix("core::")
            .or_else(|| mod_path.strip_prefix("std::"))
        else {
            return false;
        };
        rest == submod
            || (rest.starts_with(submod) && rest.as_bytes().get(submod.len()) == Some(&b':'))
    }

    pub(in super::super) fn is_iter_mod(&mut self, item: ItemRef) -> bool {
        self.item_in_std_mod(item, "iter")
    }

    pub(in super::super) fn ty_has_borrow_drop(&mut self, ty: Ty) -> bool {
        if ty == Ty::NONE {
            return false;
        }
        if let Some(&cached) = self.drop_cache.get(&ty) {
            return cached;
        }
        let mut seen = HashSet::default();
        let res = self.ty_has_borrow_drop_inner(ty, &mut seen);
        self.drop_cache.insert(ty, res);
        res
    }

    fn ty_has_borrow_drop_inner(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        if ty == Ty::NONE || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty).clone() {
            Kind::Dyn(..) => true,
            Kind::Tuple(items) => self
                .cx
                .types
                .substs(items)
                .to_vec()
                .into_iter()
                .any(|t| self.ty_has_borrow_drop_inner(t, seen)),
            Kind::Array(elem, len) => {
                self.cx.types.len(len) != Some(0) && self.ty_has_borrow_drop_inner(elem, seen)
            }
            Kind::Adt(item, _) => {
                if self.cx.lang(item) == Some("manually_drop")
                    || self.cx.item(item).1.kind == ItemKind::Union
                {
                    return false;
                }
                let (drop_blk, parts) = crate::sema::infer::detached(self.cx, self.item, |infer| {
                    let drop_blk = infer.drop_impl(ty)?.map(|(blk, _)| blk);
                    let parts = infer.drop_parts(ty)?;
                    Ok((drop_blk, parts))
                })
                .unwrap_or((None, Vec::new()));
                if let Some(block) = drop_blk
                    && self.cx.boxed(ty).is_none()
                    && !self.impl_has_may_dangle(block)
                {
                    return true;
                }
                parts
                    .into_iter()
                    .any(|part| self.ty_has_borrow_drop_inner(part, seen))
            }
            _ => false,
        }
    }

    fn impl_has_may_dangle(&mut self, block: ImplRef) -> bool {
        let Ok(hdr) = self.cx.impl_header(block) else {
            return false;
        };
        let start = match block {
            ImplRef::Body(item) => self.cx.item(item).1.start,
            ImplRef::Module { krate, index } => {
                let entry = &self.cx.outline.crates[krate as usize].impls[index as usize];
                self.cx.outline.crates[krate as usize].files[entry.file]
                    .items
                    .items[entry.at as usize]
                    .start
            }
        };
        let end = hdr.generics.last().map_or(start, |g| g.span.1);
        let src = &self.cx.impl_file(block).source;
        (start..end).any(|t| src.text(t) == "may_dangle")
    }

    pub(in super::super) fn is_non_promotable_temp(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Paren(inner)
            | Expr::Cast(inner, _)
            | Expr::Field(inner, _)
            | Expr::TupleField(inner, _)
            | Expr::Unary(UnOp::Deref, inner)
            | Expr::Ref { expr: inner, .. } => self.is_non_promotable_temp(inner),
            Expr::Call(callee, args) => {
                let is_ctor = matches!(self.typed.res.get(&callee), Some(Res::Ctor(..)))
                    || matches!(self.typed.res.get(&expr), Some(Res::Ctor(..)));
                if !is_ctor {
                    let is_const = matches!(
                        self.typed.res.get(&callee).or_else(|| self.typed.res.get(&expr)),
                        Some(Res::Fn(item, _)) if self.is_const_fn(*item)
                    );
                    if !is_const {
                        return true;
                    }
                }
                self.body
                    .list(args)
                    .iter()
                    .any(|&a| self.is_non_promotable_temp(ExprId(a)))
            }
            Expr::Method { .. } | Expr::Await(_) | Expr::Try(_) => true,
            Expr::Block {
                kind, stmts, tail, ..
            } => {
                !matches!(kind, BlockKind::Const)
                    && (self
                        .body
                        .list(stmts)
                        .iter()
                        .any(|&s| match self.body.stmts[s as usize] {
                            Stmt::Let { init, .. } => {
                                init.is_some_and(|e| self.is_non_promotable_temp(e))
                            }
                            Stmt::Expr(e, _) => self.is_non_promotable_temp(e),
                            _ => false,
                        })
                        || tail.is_some_and(|t| self.is_non_promotable_temp(t)))
            }
            Expr::Tuple(list) | Expr::Array(list) => self
                .body
                .list(list)
                .iter()
                .any(|&a| self.is_non_promotable_temp(ExprId(a))),
            Expr::Struct { fields, base, .. } => {
                self.body
                    .list(fields)
                    .iter()
                    .any(|&f| self.is_non_promotable_temp(self.body.fields[f as usize].value))
                    || base.is_some_and(|b| self.is_non_promotable_temp(b))
            }
            Expr::Repeat(a, _) | Expr::Index(a, _) => self.is_non_promotable_temp(a),
            Expr::If(_, t, e) => {
                self.is_non_promotable_temp(t)
                    || e.is_some_and(|alt| self.is_non_promotable_temp(alt))
            }
            Expr::Match(_, arms) => self
                .body
                .list(arms)
                .iter()
                .any(|&a| self.is_non_promotable_temp(self.body.arms[a as usize].body)),
            _ => false,
        }
    }

    fn is_const_fn(&self, item: ItemRef) -> bool {
        let (src, entry) = self.cx.item(item);
        (entry.start..entry.name.unwrap_or(entry.start)).any(|t| src.text(t) == "const")
    }
}
