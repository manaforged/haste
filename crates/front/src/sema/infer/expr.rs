use anyhow::{Result, bail};

use crate::body::{BinOp, Expr, ExprId, List, Lit, TyId, UnOp};
use crate::names::Primitive;

use super::super::ty::{Kind, Mut, Ty, float_bits, int_info};
use super::{Breaks, Infer, Res};
use super::path::parse_int;

impl Infer<'_, '_> {
    /// Infers `expr`, with the type the context expects when it knows one.
    pub fn expr(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        let ty = self.expr_kind(expr, expected)?;
        if matches!(self.cx.types.kind(self.resolve(ty)), Kind::Never) {
            self.diverges = true;
        }
        self.out.exprs[expr.0 as usize] = ty;
        Ok(ty)
    }

    fn expr_kind(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        match *self.body.expr(expr) {
            Expr::Lit(lit, at) => self.lit(lit, at),
            Expr::Path(path) => self.path_expr(expr, path),
            Expr::Paren(inner) => self.expr(inner, expected),
            Expr::Unary(op, operand) => self.unary(expr, (op, operand), expected),
            Expr::Binary(op, lhs, rhs) => self.binary(expr, op, (lhs, rhs), expected),
            Expr::Assign(lhs, rhs) => self.assign(lhs, rhs, None),
            Expr::AssignOp(op, lhs, rhs) => self.assign(lhs, rhs, Some(op)),
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
        let found = self.expr(inner, None)?;
        let target = self.lower_ty(self.body, ty)?;
        if let Some(coercion) = self.coercion(found, target)? {
            self.out.coercions.insert(inner, coercion);
        }
        Ok(target)
    }

    fn tuple(&mut self, list: List, expected: Option<Ty>) -> Result<Ty> {
        let wanted = match expected.map(|ty| self.cx.types.kind(self.resolve(ty)).clone()) {
            Some(Kind::Tuple(wanted)) if wanted.len() == list.len as usize => wanted,
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
        Ok(self.cx.types.intern(Kind::Tuple(items)))
    }

    fn reference(&mut self, inner: ExprId, (mutable, raw): (bool, bool), expected: Option<Ty>) -> Result<Ty> {
        let pointee = expected.map(|ty| self.resolve(ty)).and_then(|ty| match self.cx.types.kind(ty) {
            Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) => Some(*pointee),
            _ => None,
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
            Kind::Tuple(items) if (index as usize) < items.len() => Ok(items[index as usize]),
            Kind::Adt(item, args) if self.cx.item(item).1.kind == crate::outline::item::ItemKind::Struct => {
                let fields = self.adt_fields(item, &args)?.swap_remove(0);
                fields.get(index as usize).copied().ok_or_else(|| anyhow::anyhow!("no field {index} on {}", self.cx.name(item)))
            }
            _ => bail!("no field {index} on {}", self.show(ty)),
        }
    }

    fn flow(&mut self, expr: ExprId, expected: Option<Ty>) -> Result<Ty> {
        match *self.body.expr(expr) {
            Expr::Method { receiver, name, args, params } => self.method_call(expr, receiver, name, (args, params)),
            Expr::Block { stmts, tail, label: None, kind: crate::body::BlockKind::Try } => self.try_block(expr, (stmts, tail), expected),
            Expr::Block { stmts, tail, label: None, .. } => self.block(stmts, tail, expected),
            Expr::If(cond, then, otherwise) => self.if_expr(cond, then, otherwise, expected),
            Expr::While { cond, body, label: None } => self.while_expr(cond, body),
            Expr::Loop { body, label: None } => self.loop_expr(body, expected),
            Expr::Break(None, value) => self.break_expr(value),
            Expr::Continue(None) => Ok(self.cx.types.intern(Kind::Never)),
            Expr::Return(value) => self.return_expr(value),
            Expr::Try(inner) => self.try_expr(expr, inner),
            _ => self.data(expr, expected),
        }
    }

    fn lit(&mut self, lit: Lit, at: u32) -> Result<Ty> {
        let text = self.src.text(at);
        Ok(match lit {
            Lit::Bool(_) => self.cx.types.bool(),
            Lit::Char => self.cx.types.prim(Primitive::Char),
            Lit::Byte => self.cx.types.prim(Primitive::U8),
            Lit::Int => match parse_int(text)?.1 {
                Some(primitive) => self.cx.types.prim(primitive),
                None => self.fresh(Kind::IntVar),
            },
            Lit::Float => match ["f16", "f32", "f64", "f128"].iter().find(|suffix| text.ends_with(*suffix)) {
                Some(suffix) => self.cx.types.prim(Primitive::named(suffix).unwrap_or(Primitive::F64)),
                None => self.fresh(Kind::FloatVar),
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
            other => bail!("unsupported literal {other:?}"),
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

    fn is_float(&self, ty: Ty) -> bool {
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
        match op {
            UnOp::Neg | UnOp::Not if self.is_int(ty) || self.is_float(ty) || self.resolve(ty) == self.cx.types.bool() => Ok(ty),
            _ => bail!("unsupported unary {op:?} on {}", self.show(ty)),
        }
    }

    fn binary(&mut self, expr: ExprId, op: BinOp, (lhs, rhs): (ExprId, ExprId), expected: Option<Ty>) -> Result<Ty> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.logical(lhs, rhs);
        }
        let left = self.expr(lhs, if is_compare(op) { None } else { expected })?;
        self.right_operand(expr, op, (left, rhs))
    }

    fn right_operand(&mut self, expr: ExprId, op: BinOp, (left, rhs): (Ty, ExprId)) -> Result<Ty> {
        let shift = matches!(op, BinOp::Shl | BinOp::Shr);
        let hint = (self.builtin_left(op, left) && !shift).then_some(left);
        let right = self.expr(rhs, hint)?;
        if self.pending(op, (left, right))? {
            let out = if is_compare(op) { self.cx.types.bool() } else { self.fresh(Kind::Var) };
            self.operands.push((expr, op, (left, right), out));
            return Ok(out);
        }
        self.operator_type(expr, op, (left, right))
    }

    fn builtin_left(&self, op: BinOp, left: Ty) -> bool {
        self.is_scalar(left) || is_compare(op) && matches!(self.cx.types.kind(self.resolve(left)), Kind::Ptr(..))
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
                if self.pending(op, operands)? {
                    at += 1;
                    continue;
                }
                self.operands.swap_remove(at);
                let found = self.operator_type(expr, op, operands)?;
                self.unify(found, out)?;
            }
            if self.operands.is_empty() || self.operands.len() == before {
                break;
            }
        }
        match self.operands.first() {
            Some((_, op, ..)) if force => bail!("the type of the left operand of {op:?} is not known"),
            _ => Ok(()),
        }
    }

    /// Whether operator `op` on `left` and `right` must wait to pick its impl: the left operand is a variable or a reference to one, or several impls fit only because the right operand is an untyped number.
    fn pending(&mut self, op: BinOp, (left, right): (Ty, Ty)) -> Result<bool> {
        let unknown_left = match self.cx.types.kind(self.resolve(left)) {
            Kind::Var(_) => true,
            Kind::Ref(_, inner) => matches!(self.cx.types.kind(self.resolve(*inner)), Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)),
            _ => false,
        };
        if unknown_left {
            return Ok(true);
        }
        if self.builtin_left(op, left) || !matches!(self.cx.types.kind(self.resolve(right)), Kind::IntVar(_) | Kind::FloatVar(_)) {
            return Ok(false);
        }
        let Some((lang, _)) = operator_trait(op) else { return Ok(false) };
        let Some(of) = self.cx.lang_item(lang) else { return Ok(false) };
        Ok(self.probe(|infer| infer.select(of, left, &[right]))?.is_none())
    }

    fn overloaded(&mut self, expr: ExprId, op: BinOp, (left, right): (Ty, Ty)) -> Result<Ty> {
        let Some((lang, name)) = operator_trait(op) else { bail!("{op:?} has no operator trait") };
        let Some(of) = self.cx.lang_item(lang) else { bail!("core has no `{lang}` trait") };
        let Some(method) = self.cx.trait_member(of, name) else { bail!("`{lang}` has no `{name}`") };
        if self.trial(|infer| infer.select(of, left, &[right]))?.is_none() {
            bail!("unsupported operator {op:?} on {}", self.show(left));
        }
        self.out.operators.insert(expr, Res::Fn(method, vec![left, right]));
        match is_compare(op) {
            true => Ok(self.cx.types.bool()),
            false => self.project(of, left, vec![right], "Output"),
        }
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

    fn assign(&mut self, lhs: ExprId, rhs: ExprId, op: Option<BinOp>) -> Result<Ty> {
        let place = self.expr(lhs, None)?;
        self.note_mut(lhs)?;
        let shift = matches!(op, Some(BinOp::Shl | BinOp::Shr));
        let value = self.expr(rhs, (!shift).then_some(place))?;
        match op {
            Some(BinOp::Shl | BinOp::Shr) => {}
            Some(bin) if !self.builtin_compound(bin, place) => bail!("unsupported compound assignment on {}", self.show(place)),
            _ => self.coerce_expr(rhs, value, place)?,
        }
        Ok(self.cx.types.unit())
    }

    fn builtin_compound(&mut self, op: BinOp, place: Ty) -> bool {
        let logical = matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor);
        self.is_int(place) || self.is_float(place) || logical && self.resolve(place) == self.cx.types.bool()
    }

    fn call(&mut self, expr: ExprId, (callee, args): (ExprId, List), expected: Option<Ty>) -> Result<Ty> {
        let found = self.expr(callee, None)?;
        let callee_ty = self.callee_base(expr, found);
        let (params, ret, def) = match self.cx.types.kind(callee_ty).clone() {
            Kind::Closure(..) => return self.closure_call(expr, callee_ty, args),
            Kind::Dyn(of, ..) if self.is_callable_trait(of) => return self.object_call((expr, found), callee_ty, args),
            Kind::FnDef(item, substs) => {
                if !matches!(self.out.res.get(&callee), Some(Res::Fn(..))) {
                    self.out.res.insert(expr, Res::Fn(item, substs.clone()));
                }
                let (params, ret) = self.fn_types(item, &substs)?;
                (params, ret, Some((item, substs)))
            }
            Kind::FnPtr(params, ret) => (params, ret, None),
            _ => bail!("calling {} is not supported yet", self.show(callee_ty)),
        };
        let args = self.body.list(args).to_vec();
        if args.len() != params.len() {
            bail!("call takes {} arguments, {} given", params.len(), args.len());
        }
        let hints = self.expected_inputs(&params, ret, expected);
        for ((arg, param), hint) in args.into_iter().zip(params).zip(hints) {
            self.call_arg(def.as_ref().map(|(item, substs)| (*item, &substs[..])), ExprId(arg), (param, hint))?;
        }
        Ok(ret)
    }

    fn expected_inputs(&mut self, params: &[Ty], ret: Ty, expected: Option<Ty>) -> Vec<Option<Ty>> {
        let Some(expected) = expected else { return vec![None; params.len()] };
        self.probe(|infer| match infer.unify(ret, expected) {
            Ok(()) => params.iter().map(|&param| Some(infer.zonk(param))).collect(),
            Err(_) => vec![None; params.len()],
        })
    }

    fn if_expr(&mut self, cond: ExprId, then: ExprId, otherwise: Option<ExprId>, expected: Option<Ty>) -> Result<Ty> {
        let bool_ty = self.cx.types.bool();
        let depth = self.scopes.len();
        let found = self.expr(cond, Some(bool_ty))?;
        self.coerce(found, bool_ty)?;
        let before = std::mem::replace(&mut self.diverges, false);
        let Some(otherwise) = otherwise else {
            let then_ty = self.expr(then, None);
            self.scopes.truncate(depth);
            self.diverges = before;
            let unit = self.cx.types.unit();
            self.coerce(then_ty?, unit)?;
            return Ok(unit);
        };
        let then_ty = self.expr(then, expected)?;
        self.scopes.truncate(depth);
        let then_diverges = std::mem::replace(&mut self.diverges, false);
        let else_ty = self.expr(otherwise, expected.or(Some(then_ty)))?;
        self.diverges = before || (then_diverges && self.diverges);
        let Some(target) = self.known(expected) else { return self.join(then_ty, else_ty) };
        self.coerce_expr(then, then_ty, target)?;
        self.coerce_expr(otherwise, else_ty, target)?;
        Ok(target)
    }

    /// The common type of two branches, where a diverging branch takes the other's type.
    pub fn join(&mut self, a: Ty, b: Ty) -> Result<Ty> {
        let never = |infer: &Self, ty: Ty| matches!(infer.cx.types.kind(infer.resolve(ty)), Kind::Never);
        if never(self, a) {
            return Ok(b);
        }
        if never(self, b) {
            return Ok(a);
        }
        if let Some(shared) = self.shared_pointer(a, b)? {
            return Ok(shared);
        }
        self.unify(a, b)?;
        Ok(a)
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

    fn while_expr(&mut self, cond: ExprId, body: ExprId) -> Result<Ty> {
        let bool_ty = self.cx.types.bool();
        let depth = self.scopes.len();
        let found = self.expr(cond, Some(bool_ty))?;
        self.coerce(found, bool_ty)?;
        self.unit_loop_body(body, depth)
    }

    /// Infers the body of a `while` or `for` loop, whose `break`s give unit, and drops the scopes above `depth` after it.
    pub(super) fn unit_loop_body(&mut self, body: ExprId, depth: usize) -> Result<Ty> {
        let unit = self.cx.types.unit();
        self.loops.push(Breaks { ty: Some(unit), broke: false });
        let diverges = self.diverges;
        let body_ty = self.expr(body, Some(unit));
        self.scopes.truncate(depth);
        self.diverges = diverges;
        self.loops.pop();
        self.coerce(body_ty?, unit)?;
        Ok(unit)
    }

    fn loop_expr(&mut self, body: ExprId, expected: Option<Ty>) -> Result<Ty> {
        let before = self.diverges;
        self.loops.push(Breaks { ty: self.known(expected), broke: false });
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

    fn break_expr(&mut self, value: Option<ExprId>) -> Result<Ty> {
        let Some(&slot) = self.loops.last() else { bail!("`break` outside a loop") };
        let found = match value {
            Some(value) => self.expr(value, slot.ty)?,
            None => self.cx.types.unit(),
        };
        let ty = match (slot.ty, value) {
            (Some(known), Some(value)) => self.coerce_expr(value, found, known).map(|_| known)?,
            (Some(known), None) => self.coerce(found, known).map(|_| known)?,
            (None, _) => found,
        };
        if let Some(last) = self.loops.last_mut() {
            *last = Breaks { ty: Some(ty), broke: true };
        }
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
fn operator_trait(op: BinOp) -> Option<(&'static str, &'static str)> {
    OPERATORS.iter().find(|(each, ..)| *each == op).map(|&(_, lang, name)| (lang, name))
}
