use anyhow::Result;
use haste_codegen::{IntCC, Type, Value};

use crate::body::{BlockKind, Expr, ExprId, UnOp};
use crate::host::HostSymbol;
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::sema::infer::{Res, Shape, Typed};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty};

use super::place::{Base, Place};
use super::stencil::{RodataReloc, RodataTarget, SymbolSource};
use super::{Lower, Repr, repr};

mod literal;
use literal::Scope;

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
        let cell_size = (size + 1) as u32;
        let address = self.emit_symbol_with(SymbolSource::PromotedCell { owner: self.index, seq, size: cell_size }, super::reloc::RelocStep::Promoted { seq, size: cell_size });
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

    fn is_mutable_static(&self, item: ItemRef) -> bool {
        let (src, entry) = self.cx.item(item);
        entry.name.is_none_or(|name| name == 0 || src.text(name - 1) == "mut")
    }

    fn static_rodata_blob(&mut self, item: ItemRef, substs: Substs, ty: Ty) -> Result<Option<(Vec<u8>, Vec<RodataReloc>)>> {
        if self.cx.foreign(item) || self.cx.thread_local(item) || self.is_mutable_static(item) || !super::is_freeze_ty(self.cx, ty, 0) {
            return Ok(None);
        }
        if self.cx.layout(ty)?.align.bytes() > 16 {
            return Ok(None);
        }
        let Some((source, typed)) = self.literal_source(item, substs)? else { return Ok(None) };
        let typed = typed.scoped(&self.cx.types);
        let static_ty = typed.ret_ty();
        if static_ty != ty {
            return Ok(None);
        }
        let body = self.cx.body(source)?;
        let (Some(root), src) = (body.root, self.cx.body_src(source)) else { return Ok(None) };
        Ok(self.literal_blob(Scope { body: &body, src, typed: &typed }, (root, static_ty), 0))
    }

    fn emit_rodata_blob(&mut self, bytes: &[u8], relocs: &[RodataReloc]) -> Value {
        if relocs.is_empty() {
            self.emit_const_rodata(bytes)
        } else {
            let id = self.slots.rodata(bytes, relocs);
            self.emit_symbol(SymbolSource::Rodata(id))
        }
    }

    fn emit_static_rodata_blob(&mut self, index: usize, bytes: &[u8], relocs: &[RodataReloc]) -> Value {
        if relocs.is_empty() {
            self.emit_static_rodata(index, bytes)
        } else {
            let id = self.slots.rodata(bytes, relocs);
            self.emit_symbol(SymbolSource::Rodata(id))
        }
    }

    /// The place of the static `item` at `substs`, of type `ty`: the prepped library's copy when Haste shares it, immutable literal bytes in read-only data when eligible, else its cell, which the static's initializer fills on the first use that finds it empty.
    pub fn static_place(&mut self, (item, substs): (ItemRef, Substs), ty: Ty) -> Result<Place> {
        if let Some(address) = crate::prep::static_address(self.cx, item) {
            let base = self.emit_symbol(SymbolSource::Word(address as u64));
            return Ok(Place { base: Base::Addr(base), offset: 0, repr: repr(self.cx, ty)? });
        }
        let index = self.reached_index(item, substs)?;
        if let Some((bytes, relocs)) = self.static_rodata_blob(item, substs, ty)? {
            let base = self.emit_static_rodata_blob(index, &bytes, &relocs);
            return Ok(Place { base: Base::Addr(base), offset: 0, repr: repr(self.cx, ty)? });
        }
        let (address, size) = self.static_cell(index, ty)?;
        if !self.cx.foreign(item) && index != self.index {
            let flag = self.ins().iadd_imm(address, i64::from(size));
            let state = self.ins().atomic_load(Type::I8, flag);
            let filled = self.ins().icmp_imm(IntCC::Equal, state, i64::from(crate::image::StaticInit::Filled as u8));
            let (fill, done) = (self.b.create_block(), self.b.create_block());
            self.brif(filled, (done, &[]), (fill, &[]));
            self.enter(fill);
            let usize_ty = self.cx.types.prim(Primitive::Usize);
            let table = self.emit_slot_with(SymbolSource::Slot(index), super::reloc::RelocStep::Const(SymbolSource::Slot(index)));
            let code = self.ins().load(Type::I64, table, 0);
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
            let idx_val = self.emit_symbol(SymbolSource::ThreadSlot(index));
            let args = vec![idx_val, self.ins().iconst(Type::I64, i64::from(size) + 1)];
            let cell = self.call_code(code, (vec![usize_ty, usize_ty], Some(usize_ty)), args)?;
            return Ok((cell.ok_or_else(|| anyhow::anyhow!("no thread cell"))?, size as i32));
        }
        let address = self.emit_symbol(SymbolSource::StaticCell { index, size });
        Ok((address, size as i32))
    }

    /// Writes `value`, the value of the static that this function initializes, into the static's cell, and returns.
    pub fn fill_static(&mut self, value: Option<Value>) -> Result<()> {
        let ((item, substs), ty) = (self.reached.list[self.index].0, self.typed.ret_ty());
        if self.open && crate::prep::static_address(self.cx, item).is_none() && self.static_rodata_blob(item, substs, ty)?.is_none() {
            let place = self.static_place((item, substs), ty)?;
            if let Some(value) = value {
                self.write(place, value);
            }
        }
        self.finish(None)
    }
}

impl Lower<'_, '_> {
    /// Whether `expr` evaluates purely to all-zero literal bytes within `LITERAL_BYTES`.
    pub(super) fn literal_is_zero(&mut self, expr: ExprId) -> Result<bool> {
        if !self.cx.types.substs(self.reached.list[self.index].0.1).is_empty() {
            return Ok(false);
        }
        let ty = self.ty_of(expr);
        let (body, src, typed) = (self.body, self.src, self.typed);
        Ok(self.literal_blob(Scope { body, src, typed }, (expr, ty), 0).is_some_and(|(bytes, relocs)| relocs.is_empty() && bytes.iter().all(|&b| b == 0)))
    }

    /// The value of the constant `item` at `substs` that `expr` reads: an immediate, or bytes laid out once, when its body is literals of a crate that cannot change while haste runs; else a call of its body.
    pub(super) fn constant(&mut self, expr: ExprId, item: ItemRef, substs: Substs) -> Result<Option<Value>> {
        if let Some(value) = self.literal_constant(expr, item, substs)? {
            return Ok(Some(value));
        }
        match self.literal_reference(item, substs)? {
            Some(value) => Ok(Some(value)),
            None => self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::ExprRes(expr)), item, substs, Vec::new()),
        }
    }

    /// The definition and inference of the constant `item` at `substs` when lowering may read its value from its body: a body that takes no type arguments, of a crate whose sources cannot change while haste runs.
    pub(super) fn literal_source(&mut self, item: ItemRef, substs: Substs) -> Result<Option<(ItemRef, std::sync::Arc<Typed>)>> {
        let index = self.reached_index(item, substs)?;
        let ((source, _), Ok(typed)) = &self.reached.list[index] else { return Ok(None) };
        let env = &self.cx.outline.crates[source.krate as usize].env;
        let readable = matches!(typed.shape, Shape::Body) && (env.release || env.fixed) && self.cx.types.substs(substs).is_empty();
        Ok(readable.then(|| (*source, typed.clone())))
    }

    fn literal_constant(&mut self, expr: ExprId, item: ItemRef, substs: Substs) -> Result<Option<Value>> {
        let Some((source, typed)) = self.literal_source(item, substs)? else { return Ok(None) };
        let typed = typed.scoped(&self.cx.types);
        let ty = typed.ret_ty();
        let body = self.cx.body(source)?;
        let (Some(root), src) = (body.root, self.cx.body_src(source)) else { return Ok(None) };
        let Some((bytes, relocs)) = self.literal_blob(Scope { body: &body, src, typed: &typed }, (root, ty), 0) else { return Ok(None) };
        match repr(self.cx, ty)? {
            Repr::Scalar(machine) if relocs.is_empty() => {
                let mut word = [0; 16];
                word[..bytes.len().min(16)].copy_from_slice(&bytes[..bytes.len().min(16)]);
                Ok(Some(self.int_const(machine, u128::from_le_bytes(word))))
            }
            Repr::Scalar(Type::I64) if bytes.len() == 8 => {
                if let [RodataReloc { offset: 0, target: RodataTarget::Rodata(child_id) }] = relocs.as_slice() {
                    Ok(Some(self.emit_symbol(SymbolSource::Rodata(*child_id))))
                } else {
                    Ok(None)
                }
            }
            Repr::Mem { .. } if !self.lent_mutably(expr) => Ok(Some(self.emit_rodata_blob(&bytes, &relocs))),
            _ => Ok(None),
        }
    }

    /// The value of the constant `item` at `substs`, a shared reference whose body is `&` of a value made of literals: the address of those bytes laid out once, with the length of the array it points to when the reference is to a slice.
    fn literal_reference(&mut self, item: ItemRef, substs: Substs) -> Result<Option<Value>> {
        let Some((source, typed)) = self.literal_source(item, substs)? else { return Ok(None) };
        let typed = typed.scoped(&self.cx.types);
        let ty = typed.ret_ty();
        let Kind::Ref(Mut::Not, pointee) = *self.cx.types.kind(ty) else { return Ok(None) };
        let body = self.cx.body(source)?;
        let (Some(root), src) = (body.root, self.cx.body_src(source)) else { return Ok(None) };
        let mut root = root;
        while let Expr::Paren(inner) = *body.expr(root) {
            root = inner;
        }
        let Expr::Ref { mutable: false, raw: false, expr: inner } = *body.expr(root) else { return Ok(None) };
        let inner_ty = typed.expr_ty(&self.cx.types, inner);
        let Some((bytes, relocs)) = self.literal_blob(Scope { body: &body, src, typed: &typed }, (inner, inner_ty), 0) else { return Ok(None) };
        let data = self.emit_rodata_blob(&bytes, &relocs);
        Ok(match (self.cx.types.kind(pointee).clone(), self.cx.types.kind(inner_ty).clone()) {
            (Kind::Slice(_), Kind::Array(_, len)) => self.cx.types.len(len).map(|len| self.wide_pointer(data, len as i64)),
            (Kind::Slice(_), _) => None,
            _ if pointee == inner_ty => Some(data),
            _ => None,
        })
    }

    /// The address of the bytes the compiler laid out for `expr` when it names a constant made of literals that lives in memory.
    fn literal_place(&mut self, expr: ExprId) -> Result<Option<Value>> {
        let inner = self.transparent(expr).unwrap_or(expr);
        if let Some(&Res::Fn(item, substs)) = self.typed.res.get(&inner).filter(|_| matches!(self.body.expr(inner), Expr::Path(_)))
            && self.cx.item(item).1.kind == ItemKind::Const
            && !self.lent_mutably(expr)
            && matches!(repr(self.cx, self.ty_of(inner))?, Repr::Scalar(_) | Repr::Mem { .. })
            && let Some((source, typed)) = self.literal_source(item, substs)?
        {
            let typed = typed.scoped(&self.cx.types);
            let ty = typed.ret_ty();
            let body = self.cx.body(source)?;
            if let (Some(root), src) = (body.root, self.cx.body_src(source))
                && let Some((bytes, relocs)) = self.literal_blob(Scope { body: &body, src, typed: &typed }, (root, ty), 0)
            {
                return Ok(Some(self.emit_rodata_blob(&bytes, &relocs)));
            }
        }
        if !self.lent_mutably(expr) && self.cx.types.substs(self.reached.list[self.index].0.1).is_empty() {
            let ty = self.ty_of(expr);
            let (body, src, typed) = (self.body, self.src, self.typed);
            if let Some((bytes, relocs)) = self.literal_blob(Scope { body, src, typed }, (expr, ty), 0) {
                return Ok(Some(self.emit_rodata_blob(&bytes, &relocs)));
            }
        }
        Ok(None)
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
