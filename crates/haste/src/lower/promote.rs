use anyhow::Result;
use haste_codegen::{IntCC, Type, Value};

use crate::body::{Expr, ExprId, UnOp};
use crate::host::HostSymbol;
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::sema::infer::Res;
use crate::sema::substs::Substs;
use crate::sema::ty::Ty;

use super::place::{Base, Place};
use super::stencil::SymbolSource;
use super::{Lower, repr};

impl Lower<'_, '_> {
    /// Whether `&expr` names a value that lives for the whole program: a literal, a constant, a function, a closure that captures nothing, a constructor, a shared reference to one, an aggregate or constructor call of them, or a field of one.
    pub fn promotable(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Lit(..) => true,
            Expr::Closure { .. } => !self.cx.closure_takes_env(self.ty_of(expr)),
            Expr::Paren(inner) | Expr::Unary(UnOp::Neg, inner) | Expr::Field(inner, _) | Expr::TupleField(inner, _) | Expr::Ref { mutable: false, raw: false, expr: inner } => self.promotable(inner),
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Fn(item, _)) => {
                    matches!(self.cx.item(*item).1.kind, ItemKind::Const | ItemKind::Fn)
                }
                Some(Res::Ctor(..) | Res::Const(_)) => true,
                _ => false,
            },
            Expr::Call(callee, list) => matches!(self.typed.res.get(&callee), Some(Res::Ctor(..))) && self.body.list(list).iter().all(|&at| self.promotable(ExprId(at))),
            Expr::Array(list) | Expr::Tuple(list) => self.body.list(list).iter().all(|&at| self.promotable(ExprId(at))),
            Expr::Struct { fields, base: None, .. } => self.body.list(fields).iter().all(|&at| self.promotable(self.body.fields[at as usize].value)),
            _ => false,
        }
    }

    /// The address of a program-lifetime cell that holds `expr`, which the first use fills.
    pub fn promoted(&mut self, expr: ExprId) -> Result<Option<Value>> {
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

    /// The place of the static `item` at `substs`, of type `ty`: its cell, which the static's initializer fills on the first use that finds it empty.
    pub fn static_place(&mut self, (item, substs): (ItemRef, Substs), ty: Ty) -> Result<Place> {
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
        if self.open {
            let place = self.static_place((item, substs), ty)?;
            if let Some(value) = value {
                self.write(place, value);
            }
            self.mark_static()?;
        }
        self.finish(None)
    }
}
