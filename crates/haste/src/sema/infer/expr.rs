use anyhow::{Result, bail};

use crate::body::{BinOp, Expr, ExprId, List, Lit, TyId, UnOp};
use crate::names::{ItemRef, Primitive};

use super::super::ty::{Kind, Mut, Ty, float_bits, int_info};
use super::path::parse_int;
use super::{Breaks, Infer, Res};

impl Infer<'_, '_> {
    /// Infers `expr`, with the type the context expects when it knows one.
    pub fn expr(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        let ty = self.expr_kind(expr, expected)?;
        if self.is_never(ty) {
            self.diverges = true;
        }
        self.out.exprs[expr.0 as usize] = ty;
        Ok(ty)
    }

    fn expr_kind(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        match *self.body.expr(expr) {
            Expr::Lit(lit, at) => self.lit(lit, at, expected),
            Expr::Path(path) => self.path_expr(expr, path),
            Expr::Paren(inner) => self.expr(inner, expected),
            Expr::Unary(op, operand) => self.unary(expr, (op, operand), expected),
            Expr::Binary(op, lhs, rhs) => self.binary(expr, op, (lhs, rhs), expected),
            Expr::Assign(lhs, rhs) => self.assign(expr, (lhs, rhs), None),
            Expr::AssignOp(op, lhs, rhs) => self.assign(expr, (lhs, rhs), Some(op)),
            Expr::Cast(inner, ty) => self.cast(inner, ty),
            Expr::Call(callee, args) => self.call(expr, (callee, args), expected),
            Expr::Closure { params, ret, body, capture } => self.closure_expr(expr, (params, ret, body, capture), expected),
            Expr::Tuple(list) => self.tuple(list, expected),
            Expr::Ref { mutable, raw, expr: inner } => self.reference(inner, (mutable, raw), expected),
            Expr::TupleField(base, index) => self.tuple_field(expr, base, index),
            _ => self.flow(expr, expected),
        }
    }

    fn cast(&mut self, inner: ExprId, ty: TyId) -> Result<Ty> {
        let target = self.lower_ty(self.body, ty)?;
        let hint = matches!(self.cx.types.kind(self.resolve(target)), Kind::FnPtr(..)).then_some(target);
        let found = self.expr(inner, hint)?;
        self.note_cast_pointee(found, target);
        let address = matches!(self.cx.types.kind(self.resolve(target)), Kind::Ptr(..) | Kind::Prim(Primitive::Usize | Primitive::Isize));
        let coercion = match self.cx.types.kind(self.resolve(found)).clone() {
            Kind::FnDef(item, substs) if address => Some(super::coerce::Coercion::Reify(Res::Fn(item, substs))),
            _ => self.coercion(found, target)?,
        };
        if let Some(coercion) = coercion {
            self.out.coercions.insert(inner, coercion);
        }
        Ok(target)
    }

    /// Notes a cast to `*const _` or `*mut _` from the reference or pointer `found`, whose open pointee takes the pointee of `found` when nothing else fixes it, as the coercion to a raw pointer gives it.
    fn note_cast_pointee(&mut self, found: Ty, target: Ty) {
        let Kind::Ptr(_, open) = *self.cx.types.kind(self.resolve(target)) else { return };
        if let Kind::Var(_) = self.cx.types.kind(self.resolve(open))
            && let Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) = *self.cx.types.kind(self.resolve(found))
        {
            self.casts.push((open, pointee));
        }
    }

    /// Gives each cast pointee that is still open the pointee of the value cast.
    pub(super) fn settle_casts(&mut self) -> Result<()> {
        for (open, pointee) in std::mem::take(&mut self.casts) {
            if matches!(self.cx.types.kind(self.resolve(open)), Kind::Var(_)) {
                self.unify(open, pointee)?;
            }
        }
        Ok(())
    }

    fn tuple(&mut self, list: List, expected: Option<Ty>) -> Result<Ty> {
        let wanted = match expected.map(|ty| self.cx.types.kind(self.resolve(ty)).clone()) {
            Some(Kind::Tuple(wanted)) if self.cx.types.substs(wanted).len() == list.len as usize => self.cx.types.substs(wanted).to_vec(),
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

    fn reference(&mut self, inner: ExprId, (mutable, raw): (bool, bool), expected: Option<Ty>) -> Result<Ty> {
        let pointee = expected.map(|ty| self.resolve(ty)).and_then(|ty| match self.cx.types.kind(ty) {
            Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) => Some(*pointee),
            _ => None,
        });
        let pointee = pointee.filter(|&ty| match self.cx.types.kind(self.resolve(ty)) {
            Kind::Dyn(..) => matches!(self.body.expr(inner), Expr::Closure { .. }),
            Kind::Slice(_) | Kind::Prim(Primitive::Str) => matches!(self.body.expr(inner), Expr::Array(_) | Expr::Repeat(..) | Expr::Lit(..)),
            _ => true,
        });
        let ty = self.expr(inner, pointee)?;
        if mutable {
            self.note_mut(inner)?;
        }
        let mutability = if mutable { Mut::Mut } else { Mut::Not };
        Ok(self.cx.types.intern(if raw { Kind::Ptr(mutability, ty) } else { Kind::Ref(mutability, ty) }))
    }

    fn tuple_field(&mut self, expr: ExprId, base: ExprId, index: u32) -> Result<Ty> {
        let found = self.expr(base, None)?;
        let ty = self.field_base(expr, found, &index.to_string())?;
        match self.cx.types.kind(ty).clone() {
            Kind::Tuple(items) if (index as usize) < self.cx.types.substs(items).len() => Ok(self.cx.types.substs(items)[index as usize]),
            Kind::Adt(item, args) if self.cx.item(item).1.kind == crate::outline::item::ItemKind::Struct => {
                let slice = self.cx.types.substs(args).to_vec();
                let fields = self.adt_fields(item, &slice)?;
                fields[0].get(index as usize).copied().ok_or_else(|| anyhow::anyhow!("no field {index} on {}", self.cx.name(item)))
            }
            _ => bail!("no field {index} on {}", self.show(ty)),
        }
    }

    fn flow(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        match *self.body.expr(expr) {
            Expr::Method { receiver, name, args, params } => self.method_call(expr, receiver, name, (args, params), expected),
            Expr::Block { stmts, tail, label: None, kind: crate::body::BlockKind::Try } => self.try_block(expr, (stmts, tail), expected),
            Expr::Block { kind: kind @ (crate::body::BlockKind::Async | crate::body::BlockKind::AsyncMove), .. } => self.coroutine_expr(expr, (None, kind == crate::body::BlockKind::AsyncMove), &[]),
            Expr::Await(inner) => self.await_expr(expr, inner),
            Expr::Block { stmts, tail, label: None, .. } => self.block(stmts, tail, expected),
            Expr::Block { stmts, tail, label: Some(label), .. } => self.labeled_block((stmts, tail), label, expected),
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

    /// The type of the literal `lit` at token `at`: its suffix's, else the integer or float type `expected` names, as rustc types an unsuffixed literal, else a fresh numeric variable.
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
            Lit::Float => match ["f16", "f32", "f64", "f128"].iter().find(|suffix| text.ends_with(*suffix)) {
                Some(suffix) => self.cx.types.prim(Primitive::named(suffix).unwrap_or(Primitive::F64)),
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
                let Some(cstr) = self.cx.lang_item("CStr") else { bail!("core has no `CStr`") };
                let cstr = self.cx.types.adt(cstr, &[]);
                self.cx.types.intern(Kind::Ref(Mut::Not, cstr))
            }
        })
    }

    /// Whether `ty` is an integer or an integer variable.
    pub fn is_int(&self, ty: Ty) -> bool {
        match self.cx.types.kind(self.resolve(ty)) {
            Kind::IntVar(_) => true,
            Kind::Prim(primitive) => int_info(*primitive).is_some(),
            _ => false,
        }
    }

    pub(super) fn is_float(&self, ty: Ty) -> bool {
        match self.cx.types.kind(self.resolve(ty)) {
            Kind::FloatVar(_) => true,
            Kind::Prim(primitive) => float_bits(*primitive).is_some(),
            _ => false,
        }
    }

    fn unary(&mut self, expr: ExprId, (op, operand): (UnOp, ExprId), expected: Option<Ty>) -> Result<Ty> {
        if op == UnOp::Deref {
            let found = self.expr(operand, None)?;
            return self.deref(expr, found);
        }
        let ty = self.expr(operand, expected)?;
        if self.is_int(ty) || self.is_float(ty) || self.resolve(ty) == self.cx.types.bool() {
            return Ok(ty);
        }
        let name = if op == UnOp::Not { "not" } else { "neg" };
        let Some((of, method)) = self.operator_impl((name, name), ty, &[])? else { bail!("unsupported unary {op:?} on {}", self.show(ty)) };
        self.out.operators.insert(expr, Res::fn_of(&self.cx.types, method, &[ty]));
        self.project(of, ty, Vec::new(), "Output")
    }

    fn binary(&mut self, expr: ExprId, op: BinOp, (lhs, rhs): (ExprId, ExprId), expected: Option<Ty>) -> Result<Ty> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.logical(lhs, rhs);
        }
        let left = self.expr(lhs, if is_compare(op) { None } else { expected })?;
        self.right_operand(expr, op, (left, rhs))
    }

    fn right_operand(&mut self, expr: ExprId, op: BinOp, (left, rhs): (Ty, ExprId)) -> Result<Ty> {
        let right = self.right_value(op, left, rhs)?;
        if is_compare(op) && matches!(self.cx.types.kind(self.resolve(left)), Kind::Ptr(..)) {
            self.coerce_expr(rhs, right, left)?;
            return Ok(self.cx.types.bool());
        }
        if self.pending(op, (left, right))? {
            let out = if is_compare(op) { self.cx.types.bool() } else { self.fresh(Kind::Var) };
            self.operands.push((expr, op, (left, right), out));
            return Ok(out);
        }
        self.operator_type(expr, op, (left, right))
    }

    /// The type of the right operand `rhs` of `op`, coerced to the one type an overloaded operator on `left` takes there.
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

    /// The right operand type of `op` when one impl of its trait applies to the ADT `left`, as for `Option<T>` or `String + &str`, which rustc learns before it reads the right operand.
    fn unique_right(&mut self, op: BinOp, left: Ty) -> Option<Ty> {
        let of = operator_trait(op).and_then(|(lang, _)| self.cx.lang_item(lang))?;
        self.only_right(of, left)
    }

    /// The right operand type of the operator trait `of` when one impl of it applies to the ADT `left`.
    pub(super) fn only_right(&mut self, of: ItemRef, left: Ty) -> Option<Ty> {
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
        self.is_scalar(left) || is_compare(op) && matches!(kind, Kind::Ptr(..)) || matches!(op, BinOp::Eq | BinOp::Ne) && matches!(kind, Kind::FnPtr(..))
    }

    pub(in crate::sema) fn is_builtin_op(&mut self, check: super::super::summary::OpCheck) -> bool {
        use super::super::summary::OpCheck;
        match check {
            OpCheck::Unary(ty) => self.is_int(ty) || self.is_float(ty) || self.resolve(ty) == self.cx.types.bool(),
            OpCheck::Binary(op, left, right) => self.is_builtin_binary(op, left, right),
            OpCheck::AssignOp(op, place, right) => self.builtin_compound(op, place) && !matches!(self.cx.types.kind(self.resolve(right)), Kind::Ref(..) | Kind::Adt(..)),
            OpCheck::Index(step, idx) => step.is_some_and(|s| matches!(self.cx.types.kind(s), Kind::Array(..) | Kind::Slice(_))) && matches!(self.cx.types.kind(idx), Kind::Prim(Primitive::Usize)),
            OpCheck::Never => false,
        }
    }

    fn is_builtin_binary(&self, op: BinOp, left: Ty, right: Ty) -> bool {
        let ptr_cmp = is_compare(op) && matches!(self.cx.types.kind(self.resolve(left)), Kind::Ptr(..));
        ptr_cmp || (self.builtin_left(op, left) && !matches!(self.cx.types.kind(self.resolve(right)), Kind::Ref(..) | Kind::Adt(..)))
    }

    fn operator_type(&mut self, expr: ExprId, op: BinOp, (left, right): (Ty, Ty)) -> Result<Ty> {
        if !self.builtin_left(op, left) || matches!(self.cx.types.kind(self.resolve(right)), Kind::Ref(..) | Kind::Adt(..)) {
            return self.overloaded(expr, op, (left, right));
        }
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            return self.shift(left, right);
        }
        self.unify(left, right)?;
        Ok(if is_compare(op) { self.cx.types.bool() } else { left })
    }

    /// Types each operator whose left operand was not known when it was read, once it is.
    pub(super) fn settle_operators(&mut self, force: bool) -> Result<()> {
        loop {
            let before = self.operands.len();
            let mut at = 0;
            while at < self.operands.len() {
                let (expr, op, operands, out) = self.operands[at];
                let compound = matches!(self.body.expr(expr), Expr::AssignOp(..));
                if if compound { self.unknown_left(operands.0) } else { self.pending(op, operands)? } {
                    at += 1;
                    continue;
                }
                self.operands.swap_remove(at);
                let found = match *self.body.expr(expr) {
                    Expr::AssignOp(_, _, rhs) => self.compound(expr, op, (operands.0, rhs, operands.1)).map(|()| out)?,
                    _ => self.operator_type(expr, op, operands)?,
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

    /// Whether operator `op` on `left` and `right` must wait to pick its impl: the left operand is a variable or a reference to one, or several impls fit only because the right operand is an untyped number.
    fn pending(&mut self, op: BinOp, (left, right): (Ty, Ty)) -> Result<bool> {
        if self.unknown_left(left) {
            return Ok(true);
        }
        if self.builtin_left(op, left) || !matches!(self.cx.types.kind(self.resolve(right)), Kind::IntVar(_) | Kind::FloatVar(_)) {
            return Ok(false);
        }
        let Some((lang, _)) = operator_trait(op) else { return Ok(false) };
        let Some(of) = self.cx.lang_item(lang) else { return Ok(false) };
        Ok(!self.probe(|infer| infer.select_or_assume(of, left, &[right]).unwrap_or(false)))
    }

    /// Whether the left operand `left` is a variable or a reference to one, so no operator impl can be picked for it yet.
    fn unknown_left(&self, left: Ty) -> bool {
        match self.cx.types.kind(self.resolve(left)) {
            Kind::Var(_) => true,
            Kind::Ref(_, inner) => matches!(self.cx.types.kind(self.resolve(*inner)), Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)),
            _ => false,
        }
    }

    fn overloaded(&mut self, expr: ExprId, op: BinOp, (left, right): (Ty, Ty)) -> Result<Ty> {
        let Some((lang, name)) = operator_trait(op) else { bail!("{op:?} has no operator trait") };
        let Some((of, method)) = self.operator_impl((lang, name), left, &[right])? else { bail!("unsupported operator {op:?} on {}", self.show(left)) };
        self.out.operators.insert(expr, Res::fn_of(&self.cx.types, method, &[left, right]));
        match is_compare(op) {
            true => Ok(self.cx.types.bool()),
            false => self.project(of, left, vec![right], "Output"),
        }
    }

    /// The trait with lang item `lang` and its method `name`, when an impl of that trait applies to `left` with the trait arguments `args`.
    pub(super) fn operator_impl(&mut self, (lang, name): (&str, &str), left: Ty, args: &[Ty]) -> Result<Option<(ItemRef, ItemRef)>> {
        let Some(of) = self.cx.lang_item(lang) else { bail!("core has no `{lang}` trait") };
        let Some(method) = self.cx.trait_member(of, name) else { bail!("`{lang}` has no `{name}`") };
        Ok(self.select_or_prove(of, left, args)?.then_some((of, method)))
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

    fn shift(&mut self, left: Ty, right: Ty) -> Result<Ty> {
        if !self.is_int(left) || !self.is_int(right) {
            bail!("unsupported shift on {}", self.show(left));
        }
        Ok(left)
    }

    fn is_scalar(&self, ty: Ty) -> bool {
        let scalar = matches!(self.cx.types.kind(self.resolve(ty)), Kind::Prim(Primitive::Bool | Primitive::Char));
        scalar || self.is_int(ty) || self.is_float(ty)
    }

    fn call(&mut self, expr: ExprId, (callee, args): (ExprId, List), expected: Option<Ty>) -> Result<Ty> {
        let found = self.callee(callee)?;
        let callee_ty = self.callee_base(expr, found);
        let (params, ret, def) = match self.cx.types.kind(callee_ty).clone() {
            Kind::Closure(..) => return self.closure_call(expr, callee_ty, args),
            Kind::Dyn(of, ..) if self.is_callable_trait(of) => {
                return self.object_call((expr, found), callee_ty, args);
            }
            Kind::FnDef(item, substs) => {
                if !self.body.names_callee(callee) || !matches!(self.out.res.get(&callee), Some(Res::Fn(..))) {
                    self.out.res.insert(expr, Res::Fn(item, substs));
                }
                let slice = self.cx.types.substs(substs).to_vec();
                let (params, ret) = self.fn_types(item, &slice)?;
                (params, ret, Some((item, slice)))
            }
            Kind::FnPtr(params, ret) => (self.cx.types.substs(params).to_vec(), ret, None),
            Kind::Adt(..) => return self.overloaded_call(expr, found, args),
            _ => return self.other_call(expr, found, callee_ty, args),
        };
        let args = self.passed_args(def.as_ref(), args, params.len())?;
        let held = self.hold_open_obligation(def.as_ref())?;
        let def = def.or_else(|| self.ctor_callee(callee));
        self.call_args(args, (params, ret, def), (expected, held))
    }

    fn other_call(&mut self, expr: ExprId, found: Ty, callee_ty: Ty, args: List) -> Result<Ty> {
        if self.poly.is_some() && matches!(self.cx.types.kind(callee_ty), Kind::Param(_) | Kind::Proj(_)) {
            return self.poly_call(expr, found, callee_ty, args);
        }
        if matches!(self.cx.types.kind(callee_ty), Kind::Var(_))
            && let Some((params, ret)) = self.callable_sig(callee_ty)?
        {
            return self.call_once_of(expr, callee_ty, (params, ret), args);
        }
        bail!("calling {} is not supported yet", self.show(callee_ty))
    }

    /// The arguments `args` that pass values in a call of the function `def`, or all of them when the callee is not a function item.
    fn passed_args(&mut self, def: Option<&(ItemRef, Vec<Ty>)>, args: List, params: usize) -> Result<Vec<u32>> {
        match def {
            Some((item, substs)) => self.legacy_const_args((*item, substs), args, params),
            None => Ok(self.body.list(args).to_vec()),
        }
    }

    /// The struct, enum or union and its type arguments when `callee` names a tuple constructor, whose declared bounds give closure arguments their signatures.
    fn ctor_callee(&self, callee: ExprId) -> Option<(ItemRef, Vec<Ty>)> {
        match self.out.res.get(&callee) {
            Some(&Res::Ctor(item, _, args)) => Some((item, self.cx.types.substs(args).to_vec())),
            _ => None,
        }
    }

    /// The arguments `args` of a call of `item` at `substs`, which takes `params` values, that pass values, after the ones that `#[rustc_legacy_const_generics]` names are tied to its const generics.
    fn legacy_const_args(&mut self, (item, substs): (ItemRef, &[Ty]), args: List, params: usize) -> Result<Vec<u32>> {
        use crate::body::sig::GenericKind;
        let all = self.body.list(args).to_vec();
        let positions = self.cx.legacy_const_args(item);
        if positions.is_empty() || all.len() != params + positions.len() {
            return Ok(all);
        }
        let sig = self.cx.signature(item)?;
        let own: Vec<_> = sig.generics.iter().filter(|generic| generic.kind != GenericKind::Lifetime).collect();
        let outer = substs.len().saturating_sub(own.len());
        let consts: Vec<usize> = own.iter().enumerate().filter(|(_, generic)| matches!(generic.kind, GenericKind::Const(_))).map(|(at, _)| outer + at).collect();
        for (&position, &slot) in positions.iter().zip(&consts) {
            let (Some(&arg), Some(&param)) = (all.get(position), substs.get(slot)) else { bail!("`{}` has no argument {position} for a const generic", self.cx.name(item)) };
            let value = self.const_arg(self.body, ExprId(arg))?;
            self.unify(param, value)?;
            self.out.const_args.insert(ExprId(arg));
        }
        Ok(all.into_iter().enumerate().filter(|(at, _)| !positions.contains(at)).map(|(_, arg)| arg).collect())
    }

    /// Infers the arguments `args` of a call of a function with `params` that returns `ret`, the function item `def` when it is one, and returns `ret`.
    fn call_args(&mut self, args: Vec<u32>, (params, ret, def): (Vec<Ty>, Ty, Option<(ItemRef, Vec<Ty>)>), (expected, mut held): (Option<Ty>, Option<(ItemRef, Vec<Ty>)>)) -> Result<Ty> {
        let extra = self.variadic_extra(def.as_ref().map(|(item, _)| *item), args.len(), params.len())?;
        let hints = self.expected_inputs(&params, ret, expected);
        for ((&arg, param), hint) in args.iter().zip(params).zip(hints) {
            if self.is_closure(ExprId(arg)) {
                self.release_obligation(&mut held)?;
            }
            self.call_arg(def.as_ref().map(|(item, substs)| (*item, &substs[..])), ExprId(arg), (param, hint), &mut held)?;
        }
        for &arg in &args[args.len() - extra..] {
            self.expr(ExprId(arg), None)?;
        }
        self.release_obligation(&mut held)?;
        Ok(ret)
    }

    fn callee(&mut self, callee: ExprId) -> Result<Ty> {
        let found = self.expr(callee, None)?;
        if matches!(self.out.res.get(&callee), Some(Res::Ctor(..))) {
            self.out.coercions.remove(&callee);
        }
        Ok(found)
    }

    /// How many arguments past its `params` a call of `callee` with `given` arguments passes, which only a C-variadic foreign function takes.
    fn variadic_extra(&mut self, callee: Option<ItemRef>, given: usize, params: usize) -> Result<usize> {
        let variadic = match callee {
            Some(item) => self.cx.foreign(item) && self.cx.signature(item)?.variadic,
            None => false,
        };
        match given.checked_sub(params) {
            Some(0) => Ok(0),
            Some(extra) if variadic => Ok(extra),
            _ => bail!("call takes {params} arguments, {given} given"),
        }
    }

    pub(super) fn expected_inputs(&mut self, params: &[Ty], ret: Ty, expected: Option<Ty>) -> Vec<Option<Ty>> {
        let Some(expected) = expected else {
            return vec![None; params.len()];
        };
        self.probe(|infer| match infer.unifies(ret, expected) {
            true => params.iter().map(|&param| Some(infer.substitute(param))).collect(),
            false => vec![None; params.len()],
        })
    }

    fn if_expr(&mut self, cond: ExprId, then: ExprId, otherwise: Option<ExprId>, expected: Option<Ty>) -> Result<Ty> {
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
        let hint = Some(then_ty).filter(|&ty| !self.is_never(ty) && !matches!(self.cx.types.kind(self.resolve(ty)), Kind::FnDef(..)));
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

    /// Whether `ty` is the never type `!`.
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

    /// The type both branches coerce to when they differ: an unsized slice, a function pointer, or one branch's reference or pointer of the same type, such as a `Box`, that the other coerces to.
    pub(super) fn branch_target(&mut self, a: Ty, b: Ty) -> Result<Option<Ty>> {
        if let Some(target) = self.slice_branch(a, b) {
            return Ok(Some(target));
        }
        if let Some(target) = self.fn_branch(a, b)? {
            return Ok(Some(target));
        }
        let (ka, kb) = (self.cx.types.kind(self.resolve(a)), self.cx.types.kind(self.resolve(b)));
        let pointers = match (ka, kb) {
            (Kind::Ref(..), Kind::Ref(..)) => true,
            (Kind::Adt(x, _), Kind::Adt(y, _)) => x == y,
            _ => false,
        };
        if !pointers {
            return Ok(None);
        }
        if self.trial(|infer| Ok(infer.coercion(b, a).ok().flatten()))?.is_some() {
            return Ok(Some(a));
        }
        Ok(self.trial(|infer| Ok(infer.coercion(a, b).ok().flatten()))?.map(|_| b))
    }

    /// The branch type that the other branch unsizes to, when one branch points to an array and the other to a slice, as rustc's common type of two branches does.
    fn slice_branch(&self, a: Ty, b: Ty) -> Option<Ty> {
        let pointee = |ty: Ty| match self.cx.types.kind(self.resolve(ty)) {
            Kind::Ref(_, inner) | Kind::Ptr(_, inner) => Some(self.cx.types.kind(self.resolve(*inner)).clone()),
            _ => None,
        };
        match (pointee(a)?, pointee(b)?) {
            (Kind::Array(..), Kind::Slice(_)) => Some(b),
            (Kind::Slice(_), Kind::Array(..)) => Some(a),
            _ => None,
        }
    }

    /// The function pointer type both branches reify to, when they are two different function items or a function item and a function pointer, as rustc's common type of two branches does.
    fn fn_branch(&mut self, a: Ty, b: Ty) -> Result<Option<Ty>> {
        let (a_kind, b_kind) = (self.cx.types.kind(self.resolve(a)).clone(), self.cx.types.kind(self.resolve(b)).clone());
        let (item, substs) = match (&a_kind, &b_kind) {
            (Kind::FnDef(..), Kind::FnPtr(..)) => return Ok(Some(b)),
            (Kind::FnPtr(..), Kind::FnDef(..)) => return Ok(Some(a)),
            (Kind::FnDef(item, substs), Kind::FnDef(..)) if a_kind != b_kind => (*item, *substs),
            _ => return Ok(None),
        };
        let slice = self.cx.types.substs(substs).to_vec();
        let (params, ret) = self.fn_types(item, &slice)?;
        Ok(Some(self.cx.types.fn_ptr(&params, ret)))
    }

    fn shared_pointer(&mut self, a: Ty, b: Ty) -> Result<Option<Ty>> {
        let (a, b) = (self.cx.types.kind(self.resolve(a)).clone(), self.cx.types.kind(self.resolve(b)).clone());
        let (shared, inner, other) = match (a, b) {
            (Kind::Ptr(x, inner), Kind::Ptr(y, other)) if x != y => (Kind::Ptr(Mut::Not, inner), inner, other),
            (Kind::Ref(x, inner), Kind::Ref(y, other)) if x != y => (Kind::Ref(Mut::Not, inner), inner, other),
            _ => return Ok(None),
        };
        self.unify(inner, other)?;
        Ok(Some(self.cx.types.intern(shared)))
    }

    fn while_expr(&mut self, cond: ExprId, body: ExprId, label: Option<u32>) -> Result<Ty> {
        let bool_ty = self.cx.types.bool();
        let depth = self.scopes.len();
        let found = self.expr(cond, Some(bool_ty))?;
        self.coerce(found, bool_ty)?;
        self.unit_loop_body(body, label, depth)
    }

    /// Infers the body of a `while` or `for` loop labeled `label`, whose `break`s give unit, and drops the scopes above `depth` after it.
    pub(super) fn unit_loop_body(&mut self, body: ExprId, label: Option<u32>, depth: usize) -> Result<Ty> {
        let unit = self.cx.types.unit();
        self.loops.push(Breaks { ty: Some(unit), broke: false, label, block: false });
        let diverges = self.diverges;
        let body_ty = self.expr(body, Some(unit));
        self.scopes.truncate(depth);
        self.diverges = diverges;
        self.loops.pop();
        self.coerce(body_ty?, unit)?;
        Ok(unit)
    }

    fn labeled_block(&mut self, (stmts, tail): (List, Option<ExprId>), label: u32, expected: Option<Ty>) -> Result<Ty> {
        let before = self.diverges;
        self.loops.push(Breaks { ty: self.known(expected), broke: false, label: Some(label), block: true });
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
        self.loops.push(Breaks { ty: self.known(expected), broke: false, label, block: false });
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
        let Some(at) = self.loops.iter().rposition(|frame| crate::body::jumps_to(src, (frame.label, frame.block), label)) else { bail!("`break` outside a loop") };
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
        self.loops[at] = Breaks { ty: Some(ty), broke: true, ..slot };
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
}

/// Whether `op` compares its operands and gives a `bool`.
pub(super) fn is_compare(op: BinOp) -> bool {
    matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)
}

/// Each binary operator that a trait overloads, with the trait's lang item and its method.
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

/// The lang item of the trait that overloads `op`, and its method.
pub(super) fn operator_trait(op: BinOp) -> Option<(&'static str, &'static str)> {
    OPERATORS.iter().find(|(each, ..)| *each == op).map(|&(_, lang, name)| (lang, name))
}
