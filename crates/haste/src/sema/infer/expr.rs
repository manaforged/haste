use anyhow::{Result, bail};

use crate::body::{BinOp, Body, Expr, ExprId, List, Lit, TyId, UnOp};
use crate::names::{ItemRef, Primitive};

use super::super::ty::{Kind, Mut, Ty, float_bits, int_info, mask};
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
            Expr::Closure { params, ret, body, capture } => {
                let deduced = match expected {
                    Some(ty) => self.dyn_callable_super_sig(ty)?,
                    None => None,
                };
                self.closure_expr(expr, (params, ret, body, capture), deduced.or(expected))
            }
            Expr::Tuple(list) => self.tuple(list, expected),
            Expr::Ref { mutable, raw, expr: inner } => self.reference(inner, (mutable, raw), expected),
            Expr::TupleField(base, index) => self.tuple_field(expr, base, index),
            _ => self.flow(expr, expected),
        }
    }

    fn cast(&mut self, inner: ExprId, ty: TyId) -> Result<Ty> {
        let target = self.lower_ty(self.body, ty)?;
        let hint = self.cast_hint(inner, target)?;
        let found = self.expr(inner, hint)?;
        self.note_cast_pointee(inner, found, target);
        let address = matches!(self.cx.types.kind(self.resolve(target)), Kind::Ptr(..) | Kind::Prim(Primitive::Usize | Primitive::Isize));
        let coercion = match self.cx.types.kind(self.resolve(found)).clone() {
            Kind::FnDef(item, substs) if address => Some(super::coerce::Coercion::Reify(Res::Fn(item, substs))),
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
        if matches!(kind, Kind::FnPtr(..)) {
            return Ok(Some(target));
        }
        let Kind::Prim(prim) = *kind else {
            return Ok(None);
        };
        let Some((lit, at, negated)) = cast_lit_kind(self.body, inner) else {
            return Ok(None);
        };
        match lit {
            Lit::Float => Ok(float_bits(prim).is_some().then_some(target)),
            Lit::Int => {
                let Some(int) = int_info(prim) else { return Ok(None) };
                let (value, suffix) = parse_int(self.src.text(at))?;
                Ok((suffix.is_none() && int_lit_fits(value, int, negated)).then_some(target))
            }
            _ => Ok(None),
        }
    }

    /// Notes a cast to a raw pointer whose pointee has unresolved inference variables.
    fn note_cast_pointee(&mut self, inner: ExprId, found: Ty, target: Ty) {
        let Kind::Ptr(_, open) = *self.cx.types.kind(self.resolve(target)) else { return };
        if self.has_vars(open) {
            self.casts.push((inner, found, target));
        }
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
        let left = self.expr(lhs, None)?;
        self.right_operand(expr, op, (left, rhs), expected)
    }

    fn right_operand(&mut self, expr: ExprId, op: BinOp, (left, rhs): (Ty, ExprId), expected: Option<Ty>) -> Result<Ty> {
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
        self.operator_type(expr, op, (left, right), expected)
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

    fn operator_type(&mut self, expr: ExprId, op: BinOp, (left, right): (Ty, Ty), expected: Option<Ty>) -> Result<Ty> {
        if !self.builtin_left(op, left) || matches!(self.cx.types.kind(self.resolve(right)), Kind::Ref(..) | Kind::Adt(..)) {
            return self.overloaded(expr, op, (left, right));
        }
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            self.hint_scalar_var(left, expected)?;
            return self.shift(left, right);
        }
        self.unify(left, right)?;
        if !is_compare(op) {
            self.hint_scalar_var(left, expected)?;
        }
        Ok(if is_compare(op) { self.cx.types.bool() } else { left })
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

    /// Types each operator whose left operand was not known when it was read, once it is.
    pub(super) fn settle_operators(&mut self, force: bool) -> Result<()> {
        loop {
            let before = self.operands.len();
            let mut at = 0;
            while at < self.operands.len() {
                let (expr, op, operands, out) = self.operands[at];
                if matches!(op, BinOp::Shl | BinOp::Shr) {
                    self.hint_scalar_var(operands.0, Some(out))?;
                }
                let compound = matches!(self.body.expr(expr), Expr::AssignOp(..));
                if if compound { self.unknown_operand(operands.0) } else { self.pending(op, operands)? } {
                    at += 1;
                    continue;
                }
                self.operands.swap_remove(at);
                let found = match *self.body.expr(expr) {
                    Expr::AssignOp(_, _, rhs) => self.compound(expr, op, (operands.0, rhs, operands.1)).map(|()| out)?,
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
        let right_adt_or_ref = matches!(self.cx.types.kind(self.resolve(right)), Kind::Ref(..) | Kind::Adt(..));
        if (self.builtin_left(op, left) && !right_adt_or_ref) || (!left_num_var && !right_num_var && !right_unknown) {
            return Ok(false);
        }
        let Some((lang, _)) = operator_trait(op) else { return Ok(false) };
        let Some(of) = self.cx.lang_item(lang) else { return Ok(false) };
        Ok(!self.probe(|infer| infer.select_or_assume(of, left, &[right]).unwrap_or(false)))
    }

    /// Falls back the number right operand of each pending operator whose left operand is known and types those operators, so their results reach the numbers that use them before those fall back, as with `b >> bit & 1` for `b: u64` and `bit: &{integer}`.
    pub(super) fn fall_back_operands(&mut self) -> Result<()> {
        loop {
            let rights: Vec<Ty> = self.operands.iter().filter(|(_, _, (left, right), _)| !self.unknown_operand(*left) && self.is_num_var(*right)).map(|(_, _, (_, right), _)| *right).collect();
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
        let Some((lang, name)) = operator_trait(op) else { bail!("{op:?} has no operator trait") };
        let Some((of, method)) = self.operator_impl((lang, name), left, &[right])? else { bail!("unsupported operator {op:?} on {}", self.show(left)) };
        self.out.operators.insert(expr, Res::fn_of(&self.cx.types, method, &[left, right]));
        match is_compare(op) {
            true => Ok(self.cx.types.bool()),
            false => self.project(of, left, vec![right], "Output"),
        }
    }

    /// The trait with lang item `lang` and its method `name`, when an impl of that trait applies to `left` with the trait arguments `args`, or may apply once their inference variables are known, as rustc defers it.
    pub(super) fn operator_impl(&mut self, (lang, name): (&str, &str), left: Ty, args: &[Ty]) -> Result<Option<(ItemRef, ItemRef)>> {
        let Some(of) = self.cx.lang_item(lang) else { bail!("core has no `{lang}` trait") };
        let Some(method) = self.cx.trait_member(of, name) else { bail!("`{lang}` has no `{name}`") };
        let open = self.has_vars(left) || args.iter().any(|&arg| self.has_vars(arg));
        if !(self.select_or_prove(of, left, args)? || open && self.may_implement(of, left, args)?) {
            return Ok(None);
        }
        if open {
            self.obligations.push((of, std::iter::once(left).chain(args.iter().copied()).collect()));
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
        let found = self.expr(callee, None)?;
        let callee_ty = self.callee_base(expr, found);
        let opaque_mark = self.opaques.len();
        let (params, ret, def) = match self.cx.types.kind(callee_ty).clone() {
            Kind::Closure(..) => return self.closure_call(expr, callee_ty, args),
            Kind::Dyn(of, ..) if self.is_callable_trait(of) => {
                return self.object_call((expr, found), callee_ty, args);
            }
            Kind::FnDef(item, substs) => {
                if !self.body.names_callee(callee) || !matches!(self.out.res.get(&callee), Some(Res::Fn(..) | Res::Ctor(..))) {
                    self.out.res.insert(expr, Res::Fn(item, substs));
                }
                let slice = self.cx.types.substs(substs).to_vec();
                let (params, ret) = self.fn_types(item, &slice)?;
                (params, ret, (!super::adt::is_ctor(self.cx, item)).then_some((item, slice)))
            }
            Kind::FnPtr(params, ret) => (self.cx.types.substs(params).to_vec(), ret, None),
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

pub(super) fn cast_lit_kind(body: &Body, mut expr: ExprId) -> Option<(Lit, u32, bool)> {
    let mut negated = false;
    loop {
        match *body.expr(expr) {
            Expr::Paren(inner) => expr = inner,
            Expr::Block { stmts, tail: Some(tail), label: None, kind } if kind.is_plain_scope() && body.list(stmts).is_empty() => expr = tail,
            Expr::Unary(UnOp::Neg, inner) => {
                negated = true;
                expr = inner;
            }
            Expr::Lit(lit @ (Lit::Int | Lit::Float), at) => return Some((lit, at, negated)),
            _ => return None,
        }
    }
}

pub(super) fn int_lit_fits(value: u128, (bits, signed): (u32, bool), negated: bool) -> bool {
    match (signed, negated) {
        (false, true) => false,
        (false, false) => value <= mask(bits),
        (true, false) => value <= (mask(bits) >> 1),
        (true, true) => value <= (mask(bits) >> 1) + 1,
    }
}
