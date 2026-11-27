mod ops;

pub(super) use ops::{is_compare, operator_trait};

use anyhow::{Result, bail};

use crate::body::{Body, Expr, ExprId, List, Lit, TyId, UnOp};
use crate::names::{ItemRef, Primitive};

use super::super::ty::{Kind, Mut, Ty, float_bits, int_info, mask};
use super::path::parse_int;
use super::{Breaks, Infer, Res};

impl Infer<'_, '_> {
    /// Infers `expr`, with the type the context expects when it knows one.
    pub fn expr(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        let is_root = self.body.root == Some(expr);
        let root_static = is_root
            .then_some(self.instance)
            .flatten()
            .filter(|&(it, _)| self.cx.item(it).1.kind == crate::outline::item::ItemKind::Static)
            .map(|(it, _)| it);
        let cycle_frame = self.enter_const_root_cycle(expr, is_root)?;
        let _unsafe = self.push_unsafe_ctx(expr, is_root);
        let prev = self.current_expr;
        self.current_expr = Some(expr);
        let res = self.expr_kind(expr, expected);
        self.current_expr = prev;
        let ty = res?;
        if self.is_never(ty) {
            self.diverges = true;
        }
        self.out.exprs[expr.0 as usize] = ty;
        if self.root_is_user()
            && let Kind::Adt(item, args) = *self.cx.types.kind(self.resolve(ty))
        {
            self.check_adt_bounds(Some(expr), item, args)?;
        }
        if let Some(item) = root_static {
            let static_ty = expected.map(|t| self.resolve(t)).unwrap_or(ty);
            self.check_static_sync(expr, item, static_ty)?;
        }
        if cycle_frame.is_some() {
            self.check_const_deps(expr)?;
        }
        if is_root && self.root_is_user() {
            self.check_body_user_types()?;
        }
        Ok(ty)
    }

    fn expr_kind(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        match *self.body.expr(expr) {
            Expr::Lit(lit, at) => self.lit(lit, at, expected),
            Expr::Path(path) => {
                let ty = self.path_expr(expr, path)?;
                if let Some(&Res::Fn(item, _)) = self.out.res.get(&expr)
                    && self.cx.item(item).1.kind == crate::outline::item::ItemKind::Static
                {
                    self.check_static_sync(expr, item, ty)?;
                }
                Ok(ty)
            }
            Expr::Paren(inner) => self.expr(inner, expected),
            Expr::Unary(op, operand) => self.unary(expr, (op, operand), expected),
            Expr::Binary(op, lhs, rhs) => self.binary(expr, op, (lhs, rhs), expected),
            Expr::Assign(lhs, rhs) => self.assign(expr, (lhs, rhs), None),
            Expr::AssignOp(op, lhs, rhs) => self.assign(expr, (lhs, rhs), Some(op)),
            Expr::Cast(inner, ty) => self.cast(inner, ty),
            Expr::Call(callee, args) => self.call(expr, (callee, args), expected),
            Expr::Closure {
                params,
                ret,
                body,
                capture,
            } => {
                let deduced = expected
                    .map(|ty| self.dyn_callable_super_sig(ty))
                    .transpose()?
                    .flatten();
                self.closure_expr(expr, (params, ret, body, capture), deduced.or(expected))
            }
            Expr::Tuple(list) => self.tuple(list, expected),
            Expr::Ref {
                mutable,
                raw,
                expr: inner,
            } => self.reference(inner, (mutable, raw), expected),
            Expr::TupleField(base, index) => self.tuple_field(expr, base, index),
            Expr::Repeat(value, len) => self.checked_repeat_expr(value, len, expected),
            _ => self.flow(expr, expected),
        }
    }

    fn cast(&mut self, inner: ExprId, ty: TyId) -> Result<Ty> {
        let target = self.lower_ty(self.body, ty)?;
        let hint = self.cast_hint(inner, target)?;
        let found = self.expr(inner, hint)?;
        self.note_cast_pointee(inner, found, target);
        let address = matches!(
            self.cx.types.kind(self.resolve(target)),
            Kind::Ptr(..) | Kind::Prim(Primitive::Usize | Primitive::Isize)
        );
        let coercion = match self.cx.types.kind(self.resolve(found)).clone() {
            Kind::FnDef(item, substs) if address => {
                Some(super::coerce::Coercion::Reify(Res::Fn(item, substs)))
            }
            _ => self.coercion(found, target)?,
        };
        if let Some(coercion) = coercion {
            self.out.coercions.insert(inner, coercion);
        } else if !address {
            self.finish_identity_cast(found, target)?;
        }
        Ok(target)
    }

    fn cast_hint(&self, inner: ExprId, target: Ty) -> Result<Option<Ty>> {
        let target = self.resolve(target);
        let kind = self.cx.types.kind(target);
        if matches!(kind, Kind::FnPtr(..) | Kind::UnsafeFnPtr(..)) {
            return Ok(Some(target));
        }
        let (prim, hint_ty) = match *kind {
            Kind::Prim(prim) => (prim, target),
            Kind::Ptr(..) => (Primitive::Usize, self.cx.types.prim(Primitive::Usize)),
            _ => return Ok(None),
        };
        let Some((lit, _, _)) = cast_lit_kind(self.body, inner) else {
            return Ok(None);
        };
        match lit {
            Lit::Float => Ok(float_bits(prim).is_some().then_some(hint_ty)),
            Lit::Int => {
                let Some(int) = int_info(prim) else {
                    return Ok(None);
                };
                Ok(cast_int_lit_fits(self.body, self.src, inner, int)?.then_some(hint_ty))
            }
            _ => Ok(None),
        }
    }

    fn note_cast_pointee(&mut self, inner: ExprId, found: Ty, target: Ty) {
        let Kind::Ptr(_, open) = *self.cx.types.kind(self.resolve(target)) else {
            return;
        };
        if self.has_vars(open) {
            self.casts.push((inner, found, target));
        }
    }

    fn tuple(&mut self, list: List, expected: Option<Ty>) -> Result<Ty> {
        let wanted = match expected.map(|ty| self.cx.types.kind(self.resolve(ty)).clone()) {
            Some(Kind::Tuple(w)) if self.cx.types.substs(w).len() == list.len as usize => {
                self.cx.types.substs(w).to_vec()
            }
            _ => Vec::new(),
        };
        let mut items = Vec::new();
        for (index, &item) in self.body.list(list).iter().enumerate() {
            let want = wanted.get(index).copied();
            let found = self.expr(ExprId(item), want)?;
            items.push(match want {
                Some(want) if self.coerces_to(found, want) => {
                    self.coerce_expr(ExprId(item), found, want)?;
                    want
                }
                _ => found,
            });
        }
        Ok(self.cx.types.tuple(&items))
    }

    fn reference(
        &mut self,
        inner: ExprId,
        (mutable, raw): (bool, bool),
        expected: Option<Ty>,
    ) -> Result<Ty> {
        let pointee =
            expected
                .map(|ty| self.resolve(ty))
                .and_then(|ty| match self.cx.types.kind(ty) {
                    Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) => Some(*pointee),
                    _ => None,
                });
        let pointee = pointee.filter(|&ty| match self.cx.types.kind(self.resolve(ty)) {
            Kind::Dyn(..) => matches!(self.body.expr(inner), Expr::Closure { .. }),
            Kind::Slice(_) | Kind::Prim(Primitive::Str) => matches!(
                self.body.expr(inner),
                Expr::Array(_) | Expr::Repeat(..) | Expr::Lit(..)
            ),
            _ => true,
        });
        let ty = self.expr(inner, pointee)?;
        if mutable {
            self.note_mut(inner)?;
        }
        if !raw {
            self.check_packed_ref(inner, ty)?;
            self.check_static_mut_ref(inner, mutable)?;
        }
        let mutability = if mutable { Mut::Mut } else { Mut::Not };
        let kind = if raw {
            Kind::Ptr(mutability, ty)
        } else {
            Kind::Ref(mutability, ty)
        };
        Ok(self.cx.types.intern(kind))
    }

    fn checked_repeat_expr(
        &mut self,
        value: ExprId,
        len: ExprId,
        expected: Option<Ty>,
    ) -> Result<Ty> {
        let ty = self.repeat_expr(value, len, expected)?;
        if self.root_is_user()
            && !self.repeat_value_is_const(value)
            && let Kind::Array(element, _) = *self.cx.types.kind(ty)
            && let Some(copy) = self.cx.lang_item("copy")
        {
            let element = self.resolve(element);
            self.note_obligation_site(copy, element);
            self.obligations.push((copy, vec![element]));
        }
        Ok(ty)
    }

    fn repeat_value_is_const(&self, mut expr: ExprId) -> bool {
        loop {
            match *self.body.expr(expr) {
                Expr::Paren(inner) => expr = inner,
                Expr::Block {
                    stmts,
                    tail: Some(tail),
                    label: None,
                    kind,
                } if kind.is_plain_scope() && self.body.list(stmts).is_empty() => expr = tail,
                Expr::Block {
                    kind: crate::body::BlockKind::Const,
                    ..
                } => return true,
                _ => {
                    return matches!(self.out.res.get(&expr), Some(Res::Const(_)))
                        || matches!(self.out.res.get(&expr), Some(Res::Fn(it, _))
                            if self.cx.item(*it).1.kind == crate::outline::item::ItemKind::Const);
                }
            }
        }
    }

    fn tuple_field(&mut self, expr: ExprId, base: ExprId, index: u32) -> Result<Ty> {
        let found = self.expr(base, None)?;
        let ty = self.field_base(expr, found, &index.to_string())?;
        match self.cx.types.kind(ty).clone() {
            Kind::Tuple(items) if (index as usize) < self.cx.types.substs(items).len() => {
                Ok(self.cx.types.substs(items)[index as usize])
            }
            Kind::Adt(item, args)
                if self.cx.item(item).1.kind == crate::outline::item::ItemKind::Struct =>
            {
                let slice = self.cx.types.substs(args).to_vec();
                let fields = self.adt_fields(item, &slice)?;
                let field_ty = fields[0].get(index as usize).copied().ok_or_else(|| {
                    self.diag_expr(
                        crate::diag::DiagnosticKind::Name,
                        Some(expr),
                        format!("no field {index} on {}", self.cx.name(item)),
                        "unknown field",
                    )
                })?;
                self.check_field_visible(item, index, Some(expr), None, &index.to_string())?;
                Ok(field_ty)
            }
            _ => Err(self.diag_expr(
                crate::diag::DiagnosticKind::Name,
                Some(expr),
                format!("no field {index} on {}", self.show(ty)),
                "unknown field",
            )),
        }
    }

    fn flow(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        use crate::body::BlockKind;
        match *self.body.expr(expr) {
            Expr::Method {
                receiver,
                name,
                args,
                params,
            } => self.method_call(expr, receiver, name, (args, params), expected),
            Expr::Block {
                stmts,
                tail,
                label: None,
                kind: BlockKind::Try,
            } => self.try_block(expr, (stmts, tail), expected),
            Expr::Block {
                kind: kind @ (BlockKind::Async | BlockKind::AsyncMove),
                ..
            } => self.coroutine_expr(expr, (None, kind == BlockKind::AsyncMove), &[]),
            Expr::Await(inner) => self.await_expr(expr, inner),
            Expr::Block {
                stmts,
                tail,
                label: None,
                kind: BlockKind::Const,
            } => {
                let ty = self.block(stmts, tail, expected)?;
                self.check_inline_const(expr, ty)?;
                Ok(ty)
            }
            Expr::Block {
                stmts,
                tail,
                label: None,
                ..
            } => self.block(stmts, tail, expected),
            Expr::Block {
                stmts,
                tail,
                label: Some(label),
                ..
            } => self.labeled_block((stmts, tail), label, expected),
            Expr::If(cond, then, otherwise) => self.if_expr(cond, then, otherwise, expected),
            Expr::While { cond, body, label } => self.while_expr(cond, body, label),
            Expr::Loop { body, label } => self.loop_expr(body, label, expected),
            Expr::Break(label, value) => self.break_expr(label, value),
            Expr::Continue(_) => Ok(self.cx.types.intern(Kind::Never)),
            Expr::Return(value) => self.return_expr(value),
            Expr::Try(inner) => self.try_expr(expr, inner),
            _ => self.data(expr, expected),
        }
    }

    fn lit(&mut self, lit: Lit, at: u32, expected: Option<Ty>) -> Result<Ty> {
        let text = self.src.text(at);
        let wanted = self.known(expected).map(|ty| self.resolve(ty));
        let numeric = |float: bool| {
            wanted.filter(|&ty| match *self.cx.types.kind(ty) {
                Kind::Prim(prim) if float => float_bits(prim).is_some(),
                Kind::Prim(prim) => int_info(prim).is_some(),
                _ => false,
            })
        };
        Ok(match lit {
            Lit::Bool(_) => self.cx.types.bool(),
            Lit::Char => self.cx.types.prim(Primitive::Char),
            Lit::Byte => self.cx.types.prim(Primitive::U8),
            Lit::Int => match parse_int(text)?.1 {
                Some(primitive) => self.cx.types.prim(primitive),
                None => numeric(false).unwrap_or_else(|| self.fresh(Kind::IntVar)),
            },
            Lit::Float => match crate::body::FLOAT_SUFFIXES
                .iter()
                .find(|suffix| text.ends_with(*suffix))
            {
                Some(suffix) => self
                    .cx
                    .types
                    .prim(Primitive::named(suffix).unwrap_or(Primitive::F64)),
                None => numeric(true).unwrap_or_else(|| self.fresh(Kind::FloatVar)),
            },
            Lit::Str => {
                let str_ty = self.cx.types.prim(Primitive::Str);
                self.cx.types.intern(Kind::Ref(Mut::Not, str_ty))
            }
            Lit::ByteStr => {
                let len = crate::body::literal::string_bytes(text)?.len() as u64;
                let byte = self.cx.types.prim(Primitive::U8);
                let array = self.cx.types.array(byte, len);
                self.cx.types.intern(Kind::Ref(Mut::Not, array))
            }
            Lit::CStr => {
                let Some(cstr) = self.cx.lang_item("CStr") else {
                    bail!("core has no `CStr`")
                };
                let cstr = self.cx.types.adt(cstr, &[]);
                self.cx.types.intern(Kind::Ref(Mut::Not, cstr))
            }
        })
    }

    fn call(
        &mut self,
        expr: ExprId,
        (callee, args): (ExprId, List),
        expected: Option<Ty>,
    ) -> Result<Ty> {
        let found = self.expr(callee, None)?;
        let callee_ty = self.callee_base(expr, found);
        let opaque_mark = self.opaques.len();
        let (params, ret, def) = match self.cx.types.kind(callee_ty).clone() {
            Kind::Closure(..) => return self.closure_call(expr, callee_ty, args),
            Kind::Dyn(of, ..) if self.is_callable_trait(of) => {
                return self.object_call((expr, found), callee_ty, args);
            }
            Kind::FnDef(item, substs) => {
                if !self.body.names_callee(callee)
                    || !matches!(self.out.res.get(&callee), Some(Res::Fn(..) | Res::Ctor(..)))
                {
                    self.out.res.insert(expr, Res::Fn(item, substs));
                }
                let slice = self.cx.types.substs(substs).to_vec();
                let (params, ret) = self.fn_types(item, &slice)?;
                (
                    params,
                    ret,
                    (!super::adt::is_ctor(self.cx, item)).then_some((item, slice)),
                )
            }
            Kind::FnPtr(params, ret) => (self.cx.types.substs(params).to_vec(), ret, None),
            Kind::UnsafeFnPtr(params, ret) => {
                self.check_unsafe_fn_ptr_call(expr)?;
                (self.cx.types.substs(params).to_vec(), ret, None)
            }
            Kind::Adt(..) => return self.overloaded_call(expr, found, args),
            _ => return self.other_call(expr, found, callee_ty, args),
        };
        let held_opaques = self.opaques.split_off(opaque_mark);
        let args = self.passed_args(def.as_ref(), args, params.len())?;
        let held = self.hold_open_obligation(def.as_ref())?;
        let def = def.or_else(|| self.ctor_callee(callee));
        let res = self.call_args(args, (params, ret, def), (expected, held));
        let had_new = !held_opaques.is_empty();
        self.opaques.extend(held_opaques);
        let ret = res?;
        if had_new {
            self.settle_opaques(false)?;
        }
        Ok(ret)
    }

    fn other_call(&mut self, expr: ExprId, found: Ty, callee_ty: Ty, args: List) -> Result<Ty> {
        if matches!(self.cx.types.kind(callee_ty), Kind::Dyn(..))
            && let Some(ret) = self.dyn_super_call((expr, found, 0), callee_ty, args)?
        {
            return Ok(ret);
        }
        if self.poly.is_some()
            && matches!(
                self.cx.types.kind(callee_ty),
                Kind::Param(_) | Kind::Proj(_)
            )
        {
            return self.poly_call(expr, found, callee_ty, args);
        }
        if matches!(self.cx.types.kind(callee_ty), Kind::Var(_))
            && let Some((params, ret)) = self.callable_sig(callee_ty)?
        {
            return self.call_once_of(expr, callee_ty, (params, ret), args);
        }
        Err(self.diag_not_callable(expr, callee_ty))
    }

    fn passed_args(
        &mut self,
        def: Option<&(ItemRef, Vec<Ty>)>,
        args: List,
        params: usize,
    ) -> Result<Vec<u32>> {
        match def {
            Some((item, substs)) => self.legacy_const_args((*item, substs), args, params),
            None => Ok(self.body.list(args).to_vec()),
        }
    }

    fn ctor_callee(&self, callee: ExprId) -> Option<(ItemRef, Vec<Ty>)> {
        match self.out.res.get(&callee) {
            Some(&Res::Ctor(item, _, args)) => Some((item, self.cx.types.substs(args).to_vec())),
            _ => None,
        }
    }

    fn legacy_const_args(
        &mut self,
        (item, substs): (ItemRef, &[Ty]),
        args: List,
        params: usize,
    ) -> Result<Vec<u32>> {
        use crate::body::sig::GenericKind;
        let all = self.body.list(args).to_vec();
        let positions = self.cx.legacy_const_args(item);
        if positions.is_empty() || all.len() != params + positions.len() {
            return Ok(all);
        }
        let sig = self.cx.signature(item)?;
        let own: Vec<_> = sig
            .generics
            .iter()
            .filter(|generic| generic.kind != GenericKind::Lifetime)
            .collect();
        let outer = substs.len().saturating_sub(own.len());
        let consts: Vec<usize> = own
            .iter()
            .enumerate()
            .filter(|(_, generic)| matches!(generic.kind, GenericKind::Const(_)))
            .map(|(at, _)| outer + at)
            .collect();
        for (&position, &slot) in positions.iter().zip(&consts) {
            let (Some(&arg), Some(&param)) = (all.get(position), substs.get(slot)) else {
                bail!(
                    "`{}` has no argument {position} for a const generic",
                    self.cx.name(item)
                )
            };
            let value = self.const_arg(self.body, ExprId(arg))?;
            self.unify(param, value)?;
            self.out.const_args.insert(ExprId(arg));
        }
        Ok(all
            .into_iter()
            .enumerate()
            .filter(|(at, _)| !positions.contains(at))
            .map(|(_, arg)| arg)
            .collect())
    }

    fn if_expr(
        &mut self,
        cond: ExprId,
        then: ExprId,
        otherwise: Option<ExprId>,
        expected: Option<Ty>,
    ) -> Result<Ty> {
        let bool_ty = self.cx.types.bool();
        let depth = self.scopes.len();
        let found = self.expr(cond, Some(bool_ty))?;
        self.coerce(found, bool_ty)?;
        let before = std::mem::replace(&mut self.diverges, false);
        let Some(otherwise) = otherwise else {
            return self.if_without_else(then, depth, before);
        };
        let then_ty = self.expr(then, expected)?;
        self.scopes.truncate(depth);
        let then_diverges = std::mem::replace(&mut self.diverges, false);
        let hint = Some(then_ty).filter(|&ty| {
            !self.is_never(ty) && !matches!(self.cx.types.kind(self.resolve(ty)), Kind::FnDef(..))
        });
        let else_ty = self.expr(otherwise, expected.or(hint))?;
        self.diverges = before || (then_diverges && self.diverges);
        let target = match self.known(expected) {
            Some(target) => target,
            None => match self.branch_target(then_ty, else_ty)? {
                Some(target) => target,
                None => return self.join(then_ty, else_ty),
            },
        };
        self.coerce_expr(then, then_ty, target)?;
        self.coerce_expr(otherwise, else_ty, target)?;
        Ok(target)
    }

    fn if_without_else(&mut self, then: ExprId, depth: usize, before: bool) -> Result<Ty> {
        let then_ty = self.expr(then, None);
        self.scopes.truncate(depth);
        self.diverges = before;
        let unit = self.cx.types.unit();
        self.coerce(then_ty?, unit)?;
        Ok(unit)
    }

    pub(super) fn is_never(&self, ty: Ty) -> bool {
        matches!(self.cx.types.kind(self.resolve(ty)), Kind::Never)
    }

    /// The common type of two branches, where a diverging branch takes the other's type.
    pub fn join(&mut self, a: Ty, b: Ty) -> Result<Ty> {
        if self.is_never(a) {
            return Ok(b);
        }
        if self.is_never(b) {
            return Ok(a);
        }
        if let Some(shared) = self.shared_pointer(a, b)? {
            return Ok(shared);
        }
        self.unify(a, b)?;
        Ok(a)
    }

    fn while_expr(&mut self, cond: ExprId, body: ExprId, label: Option<u32>) -> Result<Ty> {
        let bool_ty = self.cx.types.bool();
        let depth = self.scopes.len();
        let found = self.expr(cond, Some(bool_ty))?;
        self.coerce(found, bool_ty)?;
        self.unit_loop_body(body, label, depth)
    }

    pub(super) fn unit_loop_body(
        &mut self,
        body: ExprId,
        label: Option<u32>,
        depth: usize,
    ) -> Result<Ty> {
        let unit = self.cx.types.unit();
        self.loops.push(Breaks {
            ty: Some(unit),
            broke: false,
            label,
            block: false,
        });
        let diverges = self.diverges;
        let body_ty = self.expr(body, Some(unit));
        self.scopes.truncate(depth);
        self.diverges = diverges;
        self.loops.pop();
        self.coerce(body_ty?, unit)?;
        Ok(unit)
    }

    fn labeled_block(
        &mut self,
        (stmts, tail): (List, Option<ExprId>),
        label: u32,
        expected: Option<Ty>,
    ) -> Result<Ty> {
        let before = self.diverges;
        self.loops.push(Breaks {
            ty: self.known(expected),
            broke: false,
            label: Some(label),
            block: true,
        });
        let found = self.block(stmts, tail, expected);
        let breaks = self.loops.pop().expect("the block pushed its breaks");
        let found = found?;
        match (breaks.broke, breaks.ty) {
            (true, Some(ty)) => {
                if !self.is_never(found) {
                    self.coerce(found, ty)?;
                }
                self.diverges = before;
                Ok(ty)
            }
            _ => Ok(found),
        }
    }

    fn loop_expr(&mut self, body: ExprId, label: Option<u32>, expected: Option<Ty>) -> Result<Ty> {
        let before = self.diverges;
        self.loops.push(Breaks {
            ty: self.known(expected),
            broke: false,
            label,
            block: false,
        });
        let unit = self.cx.types.unit();
        let body_ty = self.expr(body, Some(unit));
        let breaks = self.loops.pop().expect("the loop pushed its breaks");
        self.coerce(body_ty?, unit)?;
        match (breaks.broke, breaks.ty) {
            (true, Some(ty)) => {
                self.diverges = before;
                Ok(ty)
            }
            _ => Ok(self.cx.types.intern(Kind::Never)),
        }
    }

    fn break_expr(&mut self, label: Option<u32>, value: Option<ExprId>) -> Result<Ty> {
        let src = self.src;
        let Some(at) = self
            .loops
            .iter()
            .rposition(|frame| crate::body::jumps_to(src, (frame.label, frame.block), label))
        else {
            bail!("`break` outside a loop")
        };
        let slot = self.loops[at];
        let found = match value {
            Some(value) => self.expr(value, slot.ty)?,
            None => self.cx.types.unit(),
        };
        let ty = match (slot.ty, value) {
            (Some(known), Some(value)) => self.coerce_expr(value, found, known).map(|_| known)?,
            (Some(known), None) => self.coerce(found, known).map(|_| known)?,
            (None, _) => found,
        };
        self.loops[at] = Breaks {
            ty: Some(ty),
            broke: true,
            ..slot
        };
        Ok(self.cx.types.intern(Kind::Never))
    }

    fn return_expr(&mut self, value: Option<ExprId>) -> Result<Ty> {
        let ret = self.out.ret.unwrap_or_else(|| self.cx.types.unit());
        match value {
            Some(value) => {
                let ty = self.expr(value, Some(ret))?;
                self.coerce_expr(value, ty, ret)?;
            }
            None => {
                let unit = self.cx.types.unit();
                self.coerce(unit, ret)?;
            }
        }
        Ok(self.cx.types.intern(Kind::Never))
    }

    fn check_static_sync(&mut self, expr: ExprId, item: ItemRef, ty: Ty) -> Result<()> {
        if self.cx.outline.is_sysroot(item.krate)
            || self.cx.item(item).1.kind != crate::outline::item::ItemKind::Static
            || self.cx.foreign(item)
            || self.cx.thread_local(item)
            || self.cx.is_static_mut(item)
        {
            return Ok(());
        }
        let ty = self.resolve(ty);
        if self.has_params(ty) {
            return Ok(());
        }
        let Some(sync) = self.cx.lang_item("sync") else {
            return Ok(());
        };
        if self.has_vars(ty) {
            self.obligations.push((sync, vec![ty]));
            self.note_obligation_site(sync, ty);
            return Ok(());
        }
        let before = self.cx.quiet_mark();
        if self.thread_auto_holds(sync, ty) {
            return Ok(());
        }
        if self.cx.quiet_mark() != before {
            self.obligations.push((sync, vec![ty]));
            self.note_obligation_site(sync, ty);
            return Ok(());
        }
        let shown = self.show(ty);
        let msg = format!("`{shown}` cannot be shared between threads safely");
        Err(self.diag_expr(
            crate::diag::DiagnosticKind::Trait,
            Some(expr),
            msg.clone(),
            msg,
        ))
    }
}

pub(super) fn cast_lit_kind(body: &Body, mut expr: ExprId) -> Option<(Lit, u32, bool)> {
    let mut negated = false;
    loop {
        match *body.expr(expr) {
            Expr::Paren(inner) => expr = inner,
            Expr::Block {
                stmts,
                tail: Some(tail),
                label: None,
                kind,
            } if kind.is_plain_scope() && body.list(stmts).is_empty() => expr = tail,
            Expr::Unary(UnOp::Neg, inner) => {
                negated = true;
                expr = inner;
            }
            Expr::Lit(lit @ (Lit::Int | Lit::Float), at) => return Some((lit, at, negated)),
            _ => return None,
        }
    }
}

pub(super) fn cast_int_lit_fits(
    body: &Body,
    src: &crate::outline::source::Source,
    inner: ExprId,
    int: (u32, bool),
) -> Result<bool> {
    let Some((Lit::Int, at, negated)) = cast_lit_kind(body, inner) else {
        return Ok(false);
    };
    let (value, suffix) = parse_int(src.text(at))?;
    Ok(suffix.is_none() && int_lit_fits(value, int, negated))
}

pub(super) fn int_lit_fits(value: u128, (bits, signed): (u32, bool), negated: bool) -> bool {
    match (signed, negated) {
        (false, true) => false,
        (false, false) => value <= mask(bits),
        (true, false) => value <= (mask(bits) >> 1),
        (true, true) => value <= (mask(bits) >> 1) + 1,
    }
}
