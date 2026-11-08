use anyhow::Result;
use haste_codegen::{Type, Value};

use crate::body::{Expr, ExprId, UnOp};
use crate::outline::item::ItemKind;
use crate::sema::infer::Res;

use super::place::{Base, Place};
use super::{Lower, repr};

impl Lower<'_, '_> {
    /// Whether `&expr` names a value that lives for the whole program: a literal, a constant, or an aggregate of them.
    pub fn promotable(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Lit(..) => true,
            Expr::Paren(inner) | Expr::Unary(UnOp::Neg, inner) => self.promotable(inner),
            Expr::Path(_) => matches!(self.typed.res.get(&expr), Some(Res::Fn(item, _)) if self.cx.item(*item).1.kind == ItemKind::Const),
            Expr::Array(list) | Expr::Tuple(list) => self.body.list(list).iter().all(|&at| self.promotable(ExprId(at))),
            Expr::Struct { fields, base: None, .. } => self.body.list(fields).iter().all(|&at| self.promotable(self.body.fields[at as usize].value)),
            _ => false,
        }
    }

    /// The address of a program-lifetime cell that holds `expr`, which the first use fills.
    pub fn promoted(&mut self, expr: ExprId) -> Result<Option<Value>> {
        let ty = self.ty_of(expr);
        let size = self.cx.layout(ty)?.size.bytes() as usize;
        let cell = self.slots.data(&vec![0; size + 1]);
        let address = self.ins().iconst(Type::I64, cell);
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
}
