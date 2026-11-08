use anyhow::{Result, bail};

use crate::body::{BinOp, Expr, ExprId, Lit, UnOp};
use crate::names::Primitive;

use super::super::ty::{Kind, Mut, Ty, float_bits, int_info};
use super::{Breaks, Infer};
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
            Expr::Unary(op, operand) => self.unary(op, operand, expected),
            Expr::Binary(op, lhs, rhs) => self.binary(op, lhs, rhs, expected),
            Expr::Assign(lhs, rhs) => self.assign(lhs, rhs, None),
            Expr::AssignOp(op, lhs, rhs) => self.assign(lhs, rhs, Some(op)),
            Expr::Cast(inner, ty) => {
                self.expr(inner, None)?;
                self.lower_ty(self.body, ty)
            }
            Expr::Call(callee, args) => self.call(callee, args),
            Expr::Tuple(list) => {
                let items = self.body.list(list).iter().map(|&item| self.expr(ExprId(item), None)).collect::<Result<Vec<_>>>()?;
                Ok(self.cx.types.intern(Kind::Tuple(items)))
            }
            Expr::Ref { mutable, raw, expr: inner } => self.reference(inner, (mutable, raw), expected),
            Expr::TupleField(base, index) => self.tuple_field(base, index),
            _ => self.flow(expr, expected),
        }
    }

    fn reference(&mut self, inner: ExprId, (mutable, raw): (bool, bool), expected: Option<Ty>) -> Result<Ty> {
        let pointee = expected.map(|ty| self.resolve(ty)).and_then(|ty| match self.cx.types.kind(ty) {
            Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) => Some(*pointee),
            _ => None,
        });
        let ty = self.expr(inner, pointee)?;
        let mutability = if mutable { Mut::Mut } else { Mut::Not };
        Ok(self.cx.types.intern(if raw { Kind::Ptr(mutability, ty) } else { Kind::Ref(mutability, ty) }))
    }

    /// `ty` with every outer reference followed, as field access sees it.
    pub fn autoderef(&self, mut ty: Ty) -> Ty {
        loop {
            ty = self.resolve(ty);
            match self.cx.types.kind(ty) {
                Kind::Ref(_, inner) => ty = *inner,
                _ => return ty,
            }
        }
    }

    fn tuple_field(&mut self, base: ExprId, index: u32) -> Result<Ty> {
        let found = self.expr(base, None)?;
        let ty = self.autoderef(found);
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
                let array = self.cx.types.intern(Kind::Array(byte, len));
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

    fn unary(&mut self, op: UnOp, operand: ExprId, expected: Option<Ty>) -> Result<Ty> {
        if op == UnOp::Deref {
            let found = self.expr(operand, None)?;
            let ty = self.resolve(found);
            return match self.cx.types.kind(ty) {
                Kind::Ref(_, inner) | Kind::Ptr(_, inner) => Ok(*inner),
                _ => bail!("cannot dereference {}", self.show(ty)),
            };
        }
        let ty = self.expr(operand, expected)?;
        match op {
            UnOp::Neg | UnOp::Not if self.is_int(ty) || self.is_float(ty) || self.resolve(ty) == self.cx.types.bool() => Ok(ty),
            _ => bail!("unsupported unary {op:?} on {}", self.show(ty)),
        }
    }

    fn binary(&mut self, op: BinOp, lhs: ExprId, rhs: ExprId, expected: Option<Ty>) -> Result<Ty> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.logical(lhs, rhs);
        }
        let compare = matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge);
        let left = self.expr(lhs, if compare { None } else { expected })?;
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            return self.shift(left, rhs);
        }
        let right = self.expr(rhs, Some(left))?;
        let pointers = compare && matches!(self.cx.types.kind(self.resolve(left)), Kind::Ptr(..));
        if !self.is_scalar(left) && !pointers {
            bail!("unsupported operator {op:?} on {}", self.show(left));
        }
        self.unify(left, right)?;
        Ok(if compare { self.cx.types.bool() } else { left })
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

    fn shift(&mut self, left: Ty, rhs: ExprId) -> Result<Ty> {
        let right = self.expr(rhs, None)?;
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
        let value = self.expr(rhs, Some(place))?;
        match op {
            Some(BinOp::Shl | BinOp::Shr) => {}
            Some(_) if !(self.is_int(place) || self.is_float(place) || self.resolve(place) == self.cx.types.bool()) => bail!("unsupported compound assignment on {}", self.show(place)),
            _ => self.coerce_expr(rhs, value, place)?,
        }
        Ok(self.cx.types.unit())
    }

    fn call(&mut self, callee: ExprId, args: crate::body::List) -> Result<Ty> {
        let callee_ty = self.expr(callee, None)?;
        let (params, ret) = match self.cx.types.kind(self.resolve(callee_ty)).clone() {
            Kind::FnDef(item, substs) => self.fn_types(item, &substs)?,
            Kind::FnPtr(params, ret) => (params, ret),
            _ => bail!("calling {} is not supported yet", self.show(callee_ty)),
        };
        let args = self.body.list(args).to_vec();
        if args.len() != params.len() {
            bail!("call takes {} arguments, {} given", params.len(), args.len());
        }
        for (arg, param) in args.into_iter().zip(params) {
            let found = self.expr(ExprId(arg), Some(param))?;
            self.coerce_expr(ExprId(arg), found, param)?;
        }
        Ok(ret)
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
        if !never(self, b) {
            self.unify(a, b)?;
        }
        Ok(a)
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
