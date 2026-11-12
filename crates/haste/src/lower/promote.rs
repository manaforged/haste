use anyhow::Result;
use haste_codegen::{IntCC, Type, Value};

use crate::body::literal::{byte_value, char_value};
use crate::body::{BlockKind, Body, Expr, ExprId, Lit, UnOp};
use crate::host::HostSymbol;
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::infer::{Res, Shape, parse_int};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty, int_info, mask};

use super::place::{Base, Place};
use super::stencil::SymbolSource;
use super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Whether `&expr` names a value that lives for the whole program: a literal, a constant, a function, a closure that captures nothing, a constructor, a shared reference to one, an aggregate or constructor call of them, or a field of one; in a constant or static initializer, which the compiler evaluates once, also a call, operator or block of them.
    pub fn promotable(&self, expr: ExprId) -> bool {
        let item = self.reached.list[self.index].0.0;
        self.promotable_in(expr, matches!(self.cx.item(item).1.kind, ItemKind::Const | ItemKind::Static))
    }

    fn promotable_in(&self, expr: ExprId, constant: bool) -> bool {
        if let Some(inner) = self.transparent(expr) {
            return self.promotable_in(inner, constant);
        }
        let all = |list: crate::body::List| self.body.list(list).iter().all(|&at| self.promotable_in(ExprId(at), constant));
        match *self.body.expr(expr) {
            Expr::Lit(..) => true,
            Expr::Closure { .. } => !self.cx.closure_takes_env(self.ty_of(expr)),
            Expr::Unary(UnOp::Neg, inner) | Expr::Field(inner, _) | Expr::TupleField(inner, _) | Expr::Ref { mutable: false, raw: false, expr: inner } => self.promotable_in(inner, constant),
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Fn(item, _)) => {
                    matches!(self.cx.item(*item).1.kind, ItemKind::Const | ItemKind::Fn)
                }
                Some(Res::Ctor(..) | Res::Const(_)) => true,
                _ => false,
            },
            Expr::Call(callee, list) => {
                let callable = match self.typed.res.get(&callee) {
                    Some(Res::Ctor(..)) => true,
                    Some(Res::Fn(item, _)) => constant || self.cx.rustc_promotable(*item),
                    _ => false,
                };
                callable && all(list)
            }
            Expr::Method { receiver, params, .. } => constant && self.promotable_in(receiver, constant) && all(params),
            Expr::Unary(_, inner) | Expr::Cast(inner, _) => constant && self.promotable_in(inner, constant),
            Expr::Binary(_, left, right) => constant && self.promotable_in(left, constant) && self.promotable_in(right, constant),
            Expr::Block { stmts, tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const } => constant && self.body.list(stmts).is_empty() && self.promotable_in(tail, constant),
            Expr::Array(list) | Expr::Tuple(list) => all(list),
            Expr::Struct { fields, base: None, .. } => self.body.list(fields).iter().all(|&at| self.promotable_in(self.body.fields[at as usize].value, constant)),
            _ => false,
        }
    }

    /// The address of a program-lifetime cell that holds `expr`, which the first use fills.
    pub fn promoted(&mut self, expr: ExprId) -> Result<Option<Value>> {
        if let Some(address) = self.literal_place(expr)? {
            return Ok(Some(address));
        }
        let ty = self.ty_of(expr);
        let size = self.cx.layout(ty)?.size.bytes() as usize;
        let seq = self.promoted_seq;
        self.promoted_seq += 1;
        let address = self.emit_symbol(SymbolSource::PromotedCell { owner: self.index, seq, size: (size + 1) as u32 });
        let filled = self.ins().load(Type::I8, address, size as i32);
        let (fill, done) = (self.b.create_block(), self.b.create_block());
        self.brif(filled, (done, &[]), (fill, &[]));
        self.enter(fill);
        let whole = repr(self.cx, ty)?;
        if let Some(value) = self.expr(expr)? {
            self.write(Place { base: Base::Addr(address), offset: 0, repr: whole }, value);
        }
        let one = self.ins().iconst(Type::I8, 1);
        self.ins().store(one, address, size as i32);
        self.jump(done, &[]);
        self.enter(done);
        Ok(Some(address))
    }

    /// The place of the static `item` at `substs`, of type `ty`: the prepped library's copy when Haste shares it, whose value the library's loading set, else its cell, which the static's initializer fills on the first use that finds it empty.
    pub fn static_place(&mut self, (item, substs): (ItemRef, Substs), ty: Ty) -> Result<Place> {
        if let Some(address) = crate::prep::static_address(self.cx, item) {
            let base = self.emit_symbol(SymbolSource::Word(address as u64));
            return Ok(Place { base: Base::Addr(base), offset: 0, repr: repr(self.cx, ty)? });
        }
        let index = self.reached_index(item, substs)?;
        let (address, size) = self.static_cell(index, ty)?;
        if !self.cx.foreign(item) && index != self.index {
            let state = self.ins().load(Type::I8, address, size);
            let filled = self.ins().icmp_imm(IntCC::Equal, state, crate::image::FILLED as i64);
            let (fill, done) = (self.b.create_block(), self.b.create_block());
            self.brif(filled, (done, &[]), (fill, &[]));
            self.enter(fill);
            let usize_ty = self.cx.types.prim(Primitive::Usize);
            let flag = self.ins().iadd_imm(address, i64::from(size));
            let code = self.code_of(index);
            let once = self.emit_symbol(SymbolSource::Host(HostSymbol::FillOnce));
            self.call_code(once, (vec![usize_ty, usize_ty], None), vec![flag, code])?;
            self.jump(done, &[]);
            self.enter(done);
        }
        Ok(Place { base: Base::Addr(address), offset: 0, repr: repr(self.cx, ty)? })
    }

    fn static_cell(&mut self, index: usize, ty: Ty) -> Result<(Value, i32)> {
        let size = self.cx.layout(ty)?.size.bytes() as u32;
        let item = self.reached.list[index].0.0;
        if !self.cx.foreign(item) && self.cx.thread_local(item) {
            let usize_ty = self.cx.types.prim(Primitive::Usize);
            let code = self.emit_symbol(SymbolSource::Host(HostSymbol::ThreadCell));
            let idx_val = self.emit_symbol(SymbolSource::Word(index as u64));
            let args = vec![idx_val, self.ins().iconst(Type::I64, i64::from(size) + 1)];
            let cell = self.call_code(code, (vec![usize_ty, usize_ty], Some(usize_ty)), args)?;
            return Ok((cell.ok_or_else(|| anyhow::anyhow!("no thread cell"))?, size as i32));
        }
        let address = self.emit_symbol(SymbolSource::StaticCell { index, size });
        Ok((address, size as i32))
    }

    /// Marks the static that this function initializes as filled once its value is written, so that later uses read it without running this again.
    fn mark_static(&mut self) -> Result<()> {
        let (address, size) = self.static_cell(self.index, self.typed.ret_ty())?;
        let one = self.ins().iconst(Type::I8, i64::from(crate::image::FILLED));
        self.ins().store(one, address, size);
        Ok(())
    }

    /// Writes `value`, the value of the static that this function initializes, into the static's cell, and returns.
    pub fn fill_static(&mut self, value: Option<Value>) -> Result<()> {
        let ((item, substs), ty) = (self.reached.list[self.index].0, self.typed.ret_ty());
        if self.open && crate::prep::static_address(self.cx, item).is_none() {
            let place = self.static_place((item, substs), ty)?;
            if let Some(value) = value {
                self.write(place, value);
            }
            self.mark_static()?;
        }
        self.finish(None)
    }
}

/// The most bytes of a constant that lowering lays out from its literals.
const LITERAL_BYTES: usize = 1 << 16;

/// The bits of the scalar literal `expr` of `body` as a value of `primitive`, when it is one the compiler reads without running anything.
fn literal_scalar((body, src): (&Body, &Source), expr: ExprId, primitive: Primitive) -> Option<u128> {
    let int = int_info(primitive);
    let value = match *body.expr(expr) {
        Expr::Paren(inner) => literal_scalar((body, src), inner, primitive)?,
        Expr::Lit(Lit::Bool(value), _) if primitive == Primitive::Bool => u128::from(value),
        Expr::Lit(Lit::Int, at) if int.is_some() => parse_int(src.text(at)).ok()?.0,
        Expr::Lit(Lit::Byte, at) if primitive == Primitive::U8 => u128::from(byte_value(src.text(at)).ok()?),
        Expr::Lit(Lit::Char, at) if primitive == Primitive::Char => u128::from(u32::from(char_value(src.text(at)).ok()?)),
        Expr::Unary(UnOp::Neg, inner) if int.is_some_and(|(_, signed)| signed) => literal_scalar((body, src), inner, primitive)?.wrapping_neg(),
        _ => return None,
    };
    Some(int.map_or(value, |(bits, _)| value & mask(bits)))
}

/// Appends to `out` the bytes of `expr` of `body`, a value of `ty`, when it is made of literals only: arrays and repeats of integer, `bool` and `char` literals.
fn literal_bytes(cx: &Context<'_>, arena: (&Body, &Source), (expr, ty): (ExprId, Ty), out: &mut Vec<u8>) -> Option<()> {
    let (body, _) = arena;
    match (*body.expr(expr), cx.types.kind(ty)) {
        (Expr::Paren(inner), _) => literal_bytes(cx, arena, (inner, ty), out),
        (Expr::Array(list), &Kind::Array(elem, _)) => body.list(list).iter().try_for_each(|&at| literal_bytes(cx, arena, (ExprId(at), elem), out)),
        (Expr::Repeat(item, _), &Kind::Array(elem, count)) => {
            let mut one = Vec::new();
            literal_bytes(cx, arena, (item, elem), &mut one)?;
            let count = usize::try_from(cx.types.len(count)?).ok()?;
            (one.len().checked_mul(count)? + out.len() <= LITERAL_BYTES).then(|| out.extend(one.repeat(count)))
        }
        (_, &Kind::Prim(primitive)) => {
            let size = match primitive {
                Primitive::Bool => 1,
                Primitive::Char => 4,
                _ => int_info(primitive)?.0 as usize / 8,
            };
            let bits = literal_scalar(arena, expr, primitive)?;
            (out.len() + size <= LITERAL_BYTES).then(|| out.extend_from_slice(&bits.to_le_bytes()[..size]))
        }
        _ => None,
    }
}

impl Lower<'_, '_> {
    /// The value of the constant `item` at `substs` that `expr` reads: an immediate, or bytes laid out once, when its body is literals of a crate that cannot change while haste runs; else a call of its body.
    pub(super) fn constant(&mut self, expr: ExprId, item: ItemRef, substs: Substs) -> Result<Option<Value>> {
        match self.literal_constant(expr, item, substs)? {
            Some(value) => Ok(Some(value)),
            None => self.invoke(item, substs, Vec::new()),
        }
    }

    fn literal_constant(&mut self, expr: ExprId, item: ItemRef, substs: Substs) -> Result<Option<Value>> {
        let index = self.reached_index(item, substs)?;
        let ((source, _), Ok(typed)) = &self.reached.list[index] else { return Ok(None) };
        let env = &self.cx.outline.crates[source.krate as usize].env;
        if !matches!(typed.shape, Shape::Body) || !(env.release || env.fixed) {
            return Ok(None);
        }
        let (source, ty) = (*source, typed.ret_ty());
        let body = self.cx.body(source)?;
        let (Some(root), src) = (body.root, self.cx.body_src(source)) else { return Ok(None) };
        match (repr(self.cx, ty)?, self.cx.types.kind(ty)) {
            (Repr::Scalar(machine), &Kind::Prim(primitive)) => Ok(literal_scalar((&body, src), root, primitive).map(|bits| self.int_const(machine, bits))),
            (Repr::Mem { .. }, Kind::Array(..)) if !self.lent_mutably(expr) => {
                let mut bytes = Vec::new();
                if literal_bytes(self.cx, (&body, src), (root, ty), &mut bytes).is_none() {
                    return Ok(None);
                }
                let id = self.slots.rodata(&bytes, &[]);
                Ok(Some(self.emit_symbol(SymbolSource::Rodata(id))))
            }
            _ => Ok(None),
        }
    }

    /// The address of the bytes the compiler laid out for `expr` when it names a constant made of literals that lives in memory.
    fn literal_place(&mut self, expr: ExprId) -> Result<Option<Value>> {
        let inner = self.transparent(expr).unwrap_or(expr);
        let Some(&Res::Fn(item, substs)) = self.typed.res.get(&inner).filter(|_| matches!(self.body.expr(inner), Expr::Path(_))) else { return Ok(None) };
        if self.cx.item(item).1.kind != ItemKind::Const || !matches!(repr(self.cx, self.ty_of(inner))?, Repr::Mem { .. }) {
            return Ok(None);
        }
        self.literal_constant(inner, item, substs)
    }

    /// Whether the body writes through or lends mutably a place whose root is the value `expr`.
    fn lent_mutably(&self, expr: ExprId) -> bool {
        let mutable = |at: ExprId| match *self.body.expr(at) {
            Expr::Assign(lhs, _) | Expr::AssignOp(_, lhs, _) | Expr::Ref { mutable: true, expr: lhs, .. } => self.place_root(lhs) == expr,
            Expr::Method { receiver, .. } => self.typed.adjusts.get(&at).is_some_and(|adjust| adjust.autoref == Some(Mut::Mut)) && self.place_root(receiver) == expr,
            _ => false,
        };
        self.typed.ref_binds.values().any(|binding| *binding == Mut::Mut) || (0..self.body.exprs.len() as u32).map(ExprId).any(mutable)
    }

    /// The expression a place expression `expr` projects from through fields, indexes and parentheses.
    fn place_root(&self, expr: ExprId) -> ExprId {
        match *self.body.expr(expr) {
            Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Index(base, _) => self.place_root(base),
            _ => self.transparent(expr).map_or(expr, |inner| self.place_root(inner)),
        }
    }
}
