use anyhow::{Result, bail};

use crate::body::{BinOp, Expr, ExprId, UnOp};
use crate::diag::DiagnosticKind;
use crate::names::{ItemRef, Primitive};

use super::super::super::summary::OpCheck;
use super::super::super::ty::{Kind, Ty, float_bits, int_info};
use super::super::{Infer, Res};

impl Infer<'_, '_> {
    /// Whether `ty` is an integer or an integer variable.
    pub fn is_int(&self, ty: Ty) -> bool {
        match self.cx.types.kind(self.resolve(ty)) {
            Kind::IntVar(_) => true,
            Kind::Prim(primitive) => int_info(*primitive).is_some(),
            _ => false,
        }
    }

    pub(in super::super) fn is_float(&self, ty: Ty) -> bool {
        match self.cx.types.kind(self.resolve(ty)) {
            Kind::FloatVar(_) => true,
            Kind::Prim(primitive) => float_bits(*primitive).is_some(),
            _ => false,
        }
    }

    pub(super) fn unary(
        &mut self,
        expr: ExprId,
        (op, operand): (UnOp, ExprId),
        expected: Option<Ty>,
    ) -> Result<Ty> {
        if op == UnOp::Deref {
            let found = self.expr(operand, None)?;
            let ty = self.known_ty(found)?;
            if matches!(self.cx.types.kind(ty), Kind::Ptr(..))
                && !self.cx.outline.is_sysroot(self.home.krate)
                && !self.in_unsafe_ctx()
            {
                return Err(self.diag_expr(
                    DiagnosticKind::Unsafe,
                    Some(expr),
                    "dereference of raw pointer is unsafe and requires unsafe function or block",
                    "dereference of raw pointer",
                ));
            }
            return self.deref(expr, ty);
        }
        let ty = self.expr(operand, expected)?;
        if self.is_int(ty) || self.is_float(ty) || self.resolve(ty) == self.cx.types.bool() {
            return Ok(ty);
        }
        let name = if op == UnOp::Not { "not" } else { "neg" };
        let Some((of, method)) = self.operator_impl((name, name), ty, &[])? else {
            return Err(self.diag_expr(
                DiagnosticKind::Type,
                Some(expr),
                format!("unsupported unary {op:?} on {}", self.show(ty)),
                "cannot apply unary operator",
            ));
        };
        self.out
            .operators
            .insert(expr, Res::fn_of(&self.cx.types, method, &[ty]));
        self.project(of, ty, Vec::new(), "Output")
    }

    pub(super) fn binary(
        &mut self,
        expr: ExprId,
        op: BinOp,
        (lhs, rhs): (ExprId, ExprId),
        expected: Option<Ty>,
    ) -> Result<Ty> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.logical(lhs, rhs);
        }
        let left = self.expr(lhs, None)?;
        self.right_operand(expr, op, (left, rhs), expected)
    }

    fn right_operand(
        &mut self,
        expr: ExprId,
        op: BinOp,
        (left, rhs): (Ty, ExprId),
        expected: Option<Ty>,
    ) -> Result<Ty> {
        let right = self.right_value(op, left, rhs)?;
        if is_compare(op) && matches!(self.cx.types.kind(self.resolve(left)), Kind::Ptr(..)) {
            self.coerce_expr(rhs, right, left)?;
            return Ok(self.cx.types.bool());
        }
        if self.pending(op, (left, right))? {
            let out = if is_compare(op) {
                self.cx.types.bool()
            } else {
                self.fresh(Kind::Var)
            };
            self.operands.push((expr, op, (left, right), out));
            return Ok(out);
        }
        self.operator_type(expr, op, (left, right), expected)
    }

    fn right_value(&mut self, op: BinOp, left: Ty, rhs: ExprId) -> Result<Ty> {
        let shift = matches!(op, BinOp::Shl | BinOp::Shr);
        let builtin = self.builtin_left(op, left);
        let hint = match (builtin && !shift).then_some(left) {
            Some(hint) => Some(hint),
            None => self.unique_right(op, left),
        };
        let right = self.expr(rhs, hint)?;
        match hint.filter(|_| !is_compare(op) && !builtin) {
            Some(want) => self.coerce_expr(rhs, right, want).map(|()| want),
            None => Ok(right),
        }
    }

    fn unique_right(&mut self, op: BinOp, left: Ty) -> Option<Ty> {
        let of = operator_trait(op).and_then(|(lang, _)| self.cx.lang_item(lang))?;
        self.only_right(of, left)
    }

    pub(in super::super) fn only_right(&mut self, of: ItemRef, left: Ty) -> Option<Ty> {
        let left = self.resolve(left);
        if matches!(self.cx.types.kind(left), Kind::Param(_) | Kind::Proj(_)) {
            return self.poly_only_right(of, left);
        }
        if !matches!(self.cx.types.kind(left), Kind::Adt(..)) {
            return None;
        }
        self.probe(|infer| {
            let right = infer.fresh(Kind::Var);
            infer.select(of, left, &[right]).ok().flatten()?;
            let found = infer.substitute(right);
            (!infer.has_vars(found)).then_some(found)
        })
    }

    fn builtin_left(&self, op: BinOp, left: Ty) -> bool {
        let kind = self.cx.types.kind(self.resolve(left));
        self.is_scalar(left)
            || is_compare(op) && matches!(kind, Kind::Ptr(..))
            || matches!(op, BinOp::Eq | BinOp::Ne)
                && matches!(kind, Kind::FnPtr(..) | Kind::UnsafeFnPtr(..))
    }

    pub(in crate::sema) fn is_builtin_op(&mut self, check: OpCheck) -> bool {
        match check {
            OpCheck::Unary(ty) => {
                self.is_int(ty) || self.is_float(ty) || self.resolve(ty) == self.cx.types.bool()
            }
            OpCheck::Binary(op, left, right) => self.is_builtin_binary(op, left, right),
            OpCheck::AssignOp(op, place, right) => {
                self.builtin_compound(op, place)
                    && !matches!(
                        self.cx.types.kind(self.resolve(right)),
                        Kind::Ref(..) | Kind::Adt(..)
                    )
            }
            OpCheck::Index(step, idx) => {
                step.is_some_and(|s| {
                    matches!(self.cx.types.kind(s), Kind::Array(..) | Kind::Slice(_))
                }) && matches!(self.cx.types.kind(idx), Kind::Prim(Primitive::Usize))
            }
            OpCheck::Never => false,
        }
    }

    fn is_builtin_binary(&self, op: BinOp, left: Ty, right: Ty) -> bool {
        let ptr_cmp =
            is_compare(op) && matches!(self.cx.types.kind(self.resolve(left)), Kind::Ptr(..));
        ptr_cmp
            || (self.builtin_left(op, left)
                && !matches!(
                    self.cx.types.kind(self.resolve(right)),
                    Kind::Ref(..) | Kind::Adt(..)
                ))
    }

    fn operator_type(
        &mut self,
        expr: ExprId,
        op: BinOp,
        (left, right): (Ty, Ty),
        expected: Option<Ty>,
    ) -> Result<Ty> {
        if !self.builtin_left(op, left)
            || matches!(
                self.cx.types.kind(self.resolve(right)),
                Kind::Ref(..) | Kind::Adt(..)
            )
        {
            return self.overloaded(expr, op, (left, right));
        }
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            self.hint_scalar_var(left, expected)?;
            return self.shift(expr, left, right);
        }
        self.unify(left, right)?;
        if !is_compare(op) {
            self.hint_scalar_var(left, expected)?;
        }
        Ok(if is_compare(op) {
            self.cx.types.bool()
        } else {
            left
        })
    }

    fn hint_scalar_var(&mut self, left: Ty, expected: Option<Ty>) -> Result<()> {
        let Some(exp) = self.known(expected).map(|ty| self.resolve(ty)) else {
            return Ok(());
        };
        let unify_scalar = match self.cx.types.kind(self.resolve(left)) {
            Kind::IntVar(_) => self.is_int(exp),
            Kind::FloatVar(_) => self.is_float(exp),
            _ => false,
        };
        if unify_scalar {
            self.unify(left, exp)?;
        }
        Ok(())
    }

    pub(in super::super) fn settle_operators(&mut self, force: bool) -> Result<()> {
        loop {
            let before = self.operands.len();
            let mut at = 0;
            while at < self.operands.len() {
                let (expr, op, operands, out) = self.operands[at];
                if matches!(op, BinOp::Shl | BinOp::Shr) {
                    self.hint_scalar_var(operands.0, Some(out))?;
                }
                let compound = matches!(self.body.expr(expr), Expr::AssignOp(..));
                if if compound {
                    self.unknown_operand(operands.0)
                } else {
                    self.pending(op, operands)?
                } {
                    at += 1;
                    continue;
                }
                self.operands.swap_remove(at);
                let found = match *self.body.expr(expr) {
                    Expr::AssignOp(_, _, rhs) => self
                        .compound(expr, op, (operands.0, rhs, operands.1))
                        .map(|()| out)?,
                    _ => self.operator_type(expr, op, operands, Some(out))?,
                };
                self.unify(found, out)?;
            }
            if self.operands.is_empty() || self.operands.len() == before {
                break;
            }
        }
        match self.operands.first() {
            Some((_, op, ..)) if force => {
                bail!("the type of the left operand of {op:?} is not known")
            }
            _ => Ok(()),
        }
    }

    fn pending(&mut self, op: BinOp, (left, right): (Ty, Ty)) -> Result<bool> {
        if self.unknown_operand(left) {
            return Ok(true);
        }
        let left_num_var = self.is_num_var(left);
        let right_num_var = self.is_num_var(right);
        let right_unknown = !self.builtin_left(op, left) && self.unknown_operand(right);
        let right_adt_or_ref = matches!(
            self.cx.types.kind(self.resolve(right)),
            Kind::Ref(..) | Kind::Adt(..)
        );
        if (self.builtin_left(op, left) && !right_adt_or_ref)
            || (!left_num_var && !right_num_var && !right_unknown)
        {
            return Ok(false);
        }
        let Some((lang, _)) = operator_trait(op) else {
            return Ok(false);
        };
        let Some(of) = self.cx.lang_item(lang) else {
            return Ok(false);
        };
        Ok(!self.probe(|infer| infer.select_or_assume(of, left, &[right]).unwrap_or(false)))
    }

    pub(in super::super) fn fall_back_operands(&mut self) -> Result<()> {
        loop {
            let rights: Vec<Ty> = self
                .operands
                .iter()
                .filter(|(_, _, (left, right), _)| {
                    !self.unknown_operand(*left) && self.is_num_var(*right)
                })
                .map(|(_, _, (_, right), _)| *right)
                .collect();
            if rights.is_empty() {
                return Ok(());
            }
            self.fall_back_in(rights)?;
            self.settle_operators(false)?;
            self.settle_projection_loop()?;
        }
    }

    fn unknown_operand(&self, ty: Ty) -> bool {
        match self.cx.types.kind(self.resolve(ty)) {
            Kind::Var(_) => true,
            Kind::Ref(_, inner) => self.unknown_operand(*inner) || self.is_num_var(*inner),
            _ => false,
        }
    }

    fn is_num_var(&self, ty: Ty) -> bool {
        match self.cx.types.kind(self.resolve(ty)) {
            Kind::IntVar(_) | Kind::FloatVar(_) => true,
            Kind::Ref(_, inner) => self.is_num_var(*inner),
            _ => false,
        }
    }

    fn overloaded(&mut self, expr: ExprId, op: BinOp, (left, right): (Ty, Ty)) -> Result<Ty> {
        let Some((lang, name)) = operator_trait(op) else {
            bail!("{op:?} has no operator trait")
        };
        let Some((of, method)) = self.operator_impl((lang, name), left, &[right])? else {
            return Err(self.diag_expr(
                DiagnosticKind::Type,
                Some(expr),
                format!("unsupported operator {op:?} on {}", self.show(left)),
                "cannot apply binary operator",
            ));
        };
        self.out
            .operators
            .insert(expr, Res::fn_of(&self.cx.types, method, &[left, right]));
        match is_compare(op) {
            true => Ok(self.cx.types.bool()),
            false => self.project(of, left, vec![right], "Output"),
        }
    }

    pub(in super::super) fn operator_impl(
        &mut self,
        (lang, name): (&str, &str),
        left: Ty,
        args: &[Ty],
    ) -> Result<Option<(ItemRef, ItemRef)>> {
        let Some(of) = self.cx.lang_item(lang) else {
            bail!("core has no `{lang}` trait")
        };
        let Some(method) = self.cx.trait_member(of, name) else {
            bail!("`{lang}` has no `{name}`")
        };
        let open = self.has_vars(left) || args.iter().any(|&arg| self.has_vars(arg));
        if !(self.select_or_prove(of, left, args)? || open && self.may_implement(of, left, args)?) {
            return Ok(None);
        }
        if open {
            self.obligations.push((
                of,
                std::iter::once(left).chain(args.iter().copied()).collect(),
            ));
        }
        Ok(Some((of, method)))
    }

    fn logical(&mut self, lhs: ExprId, rhs: ExprId) -> Result<Ty> {
        let bool_ty = self.cx.types.bool();
        let found = self.expr(lhs, Some(bool_ty))?;
        self.coerce(found, bool_ty)?;
        let before = self.diverges;
        let found = self.expr(rhs, Some(bool_ty));
        self.diverges = before;
        self.coerce(found?, bool_ty)?;
        Ok(bool_ty)
    }

    fn shift(&mut self, expr: ExprId, left: Ty, right: Ty) -> Result<Ty> {
        if !self.is_int(left) || !self.is_int(right) {
            return Err(self.diag_expr(
                DiagnosticKind::Type,
                Some(expr),
                format!("unsupported shift on {}", self.show(left)),
                "shift operands must be integers",
            ));
        }
        Ok(left)
    }

    fn is_scalar(&self, ty: Ty) -> bool {
        let scalar = matches!(
            self.cx.types.kind(self.resolve(ty)),
            Kind::Prim(Primitive::Bool | Primitive::Char)
        );
        scalar || self.is_int(ty) || self.is_float(ty)
    }
}

pub(in super::super) fn is_compare(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
    )
}

const OPERATORS: [(BinOp, &str, &str); 16] = [
    (BinOp::Add, "add", "add"),
    (BinOp::Sub, "sub", "sub"),
    (BinOp::Mul, "mul", "mul"),
    (BinOp::Div, "div", "div"),
    (BinOp::Rem, "rem", "rem"),
    (BinOp::BitAnd, "bitand", "bitand"),
    (BinOp::BitOr, "bitor", "bitor"),
    (BinOp::BitXor, "bitxor", "bitxor"),
    (BinOp::Shl, "shl", "shl"),
    (BinOp::Shr, "shr", "shr"),
    (BinOp::Eq, "eq", "eq"),
    (BinOp::Ne, "eq", "ne"),
    (BinOp::Lt, "partial_ord", "lt"),
    (BinOp::Le, "partial_ord", "le"),
    (BinOp::Gt, "partial_ord", "gt"),
    (BinOp::Ge, "partial_ord", "ge"),
];

pub(in super::super) fn operator_trait(op: BinOp) -> Option<(&'static str, &'static str)> {
    OPERATORS
        .iter()
        .find(|(each, ..)| *each == op)
        .map(|&(_, lang, name)| (lang, name))
}

impl Infer<'_, '_> {
    pub(super) fn check_static_mut_ref(&self, mut cur: ExprId, mutable: bool) -> Result<()> {
        if self.cx.outline.is_sysroot(self.home.krate)
            || !self
                .cx
                .names
                .crates
                .get(self.home.krate as usize)
                .is_some_and(|c| c.edition == "2024")
        {
            return Ok(());
        }
        loop {
            match *self.body.expr(cur) {
                Expr::Paren(inner) => cur = inner,
                Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Index(base, _)
                    if self.out.autoderefs.get(&cur).copied().unwrap_or(0) == 0 =>
                {
                    cur = base;
                }
                Expr::Path(_) => {
                    if let Some(&Res::Fn(item, _)) = self.out.res.get(&cur)
                        && self.cx.is_static_mut(item)
                    {
                        let kind = if mutable { "mutable" } else { "shared" };
                        let msg = format!("creating a {kind} reference to mutable static");
                        return Err(self.diag_expr(
                            DiagnosticKind::Unsafe,
                            Some(cur),
                            msg,
                            "reference to mutable static",
                        ));
                    }
                    break;
                }
                _ => break,
            }
        }
        Ok(())
    }

    pub(in super::super) fn check_packed_ref(
        &mut self,
        mut cur: ExprId,
        field_ty: Ty,
    ) -> Result<()> {
        let field_ty = self.resolve(field_ty);
        let (has_vars, has_params) = (self.has_vars(field_ty), self.has_params(field_ty));
        let field_align = (!has_vars && !has_params)
            .then(|| self.substitute(field_ty))
            .and_then(|sub| self.cx.layout(sub).ok().map(|l| l.align.bytes()));
        loop {
            match *self.body.expr(cur) {
                Expr::Paren(inner) => cur = inner,
                Expr::Index(base, _)
                    if self.out.autoderefs.get(&cur).copied().unwrap_or(0) == 0
                        && matches!(
                            self.cx
                                .types
                                .kind(self.resolve(self.out.exprs[base.0 as usize])),
                            Kind::Array(..)
                        ) =>
                {
                    cur = base
                }
                Expr::Field(base, _) | Expr::TupleField(base, _) => {
                    let steps = self.out.autoderefs.get(&cur).copied().unwrap_or(0);
                    if let Some(cont_ty) = self.field_container_for_ref(cur, base, steps)
                        && let Kind::Adt(item, _) = *self.cx.types.kind(cont_ty)
                        && let Some(pack) = self.cx.repr(item).pack
                    {
                        if has_params && self.poly.is_some() {
                            self.poly_fallback();
                        }
                        if field_align.is_some_and(|align| align > pack.bytes()) {
                            let msg = "reference to packed field is unaligned";
                            return Err(self.diag_expr(
                                crate::diag::DiagnosticKind::Type,
                                None,
                                msg,
                                msg,
                            ));
                        }
                    }
                    if steps > 0 {
                        break;
                    }
                    cur = base;
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn field_container_for_ref(&mut self, expr: ExprId, base: ExprId, steps: u32) -> Option<Ty> {
        let mut ty = self.resolve(*self.out.exprs.get(base.0 as usize)?);
        for step in 0..steps {
            if let Some(&Res::Fn(method, substs)) = self.out.derefs.get(&(expr, step)) {
                let slice = self.cx.types.substs(substs).to_vec();
                let ret = self.fn_types(method, &slice).ok()?.1;
                let Kind::Ref(_, target) = *self.cx.types.kind(self.resolve(ret)) else {
                    return None;
                };
                ty = self.resolve(target);
            } else {
                ty = self.resolve(self.builtin_deref(ty)?);
            }
        }
        Some(ty)
    }
}
