use anyhow::Result;
use haste_codegen::{IntCC, Type, Value};

use crate::body::{Expr, ExprId, List, Stmt};
use crate::host::HostSymbol;
use crate::names::{ItemRef, Primitive};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty};

use super::place::{Base, Place};
use super::stencil::{RodataReloc, SymbolSource};
use super::{Lower, Repr, repr};

mod const_fn;
mod const_ops;
mod const_pat;
mod const_val;
mod literal;
mod scan;
use literal::Scope;
pub(super) use literal::{eval_lit_bits, neg_float_bits};

impl Lower<'_, '_> {
    /// The address of a program-lifetime cell that holds `expr`, which the first use fills.
    pub fn promoted(&mut self, expr: ExprId) -> Result<Option<Value>> {
        if let Some(address) = self.literal_place(expr)? {
            return Ok(Some(address));
        }
        self.once(self.ty_of(expr), |lower| lower.expr(expr))
            .map(Some)
    }

    pub(super) fn once(
        &mut self,
        ty: Ty,
        fill: impl FnOnce(&mut Self) -> Result<Option<Value>>,
    ) -> Result<Value> {
        let size = self.cx.layout(ty)?.size.bytes() as usize;
        let seq = self.promoted_seq;
        self.promoted_seq += 1;
        let cell_size = (size + 1) as u32;
        let address = self.emit_symbol_with(
            SymbolSource::PromotedCell {
                owner: self.index,
                seq,
                size: cell_size,
            },
            super::reloc::RelocStep::Promoted {
                seq,
                size: cell_size,
            },
        );
        let filled = self.ins().load(Type::I8, address, size as i32);
        self.unless(filled, |lower| {
            let whole = repr(lower.cx, ty)?;
            if let Some(value) = fill(lower)? {
                lower.write(
                    Place {
                        base: Base::Addr(address),
                        offset: 0,
                        repr: whole,
                    },
                    value,
                );
            }
            let one = lower.ins().iconst(Type::I8, 1);
            lower.ins().store(one, address, size as i32);
            Ok(())
        })?;
        Ok(address)
    }

    fn unless(&mut self, filled: Value, fill: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let (empty, done) = (self.b.create_block(), self.b.create_block());
        self.brif(filled, (done, &[]), (empty, &[]));
        self.enter(empty);
        fill(self)?;
        self.jump(done, &[]);
        self.enter(done);
        Ok(())
    }

    /// The value of the `const` block `expr`: evaluated at compile time when possible, computed on the first use into a cell of this function when it loops and its value holds no address, else computed in place.
    pub(super) fn const_block(
        &mut self,
        expr: ExprId,
        stmts: List,
        tail: Option<ExprId>,
    ) -> Result<Option<Value>> {
        let ty = self.ty_of(expr);
        let whole = repr(self.cx, ty)?;
        let plain = match whole {
            Repr::Zst => true,
            Repr::Scalar(_) => matches!(*self.cx.types.kind(ty), Kind::Prim(_)),
            Repr::Mem { .. } => false,
        };
        if self
            .cx
            .types
            .substs(self.reached.list[self.index].0.1)
            .is_empty()
            && let Some(blob) = self.eval_const_block(expr, ty)
        {
            if whole == Repr::Zst {
                return Ok(None);
            }
            if let Some(val) = self.materialize_blob(expr, ty, blob)? {
                return Ok(Some(val));
            }
        }
        if !plain || !self.loops(stmts, tail) {
            return self.block(stmts, tail);
        }
        let address = self.once(ty, |lower| lower.block(stmts, tail))?;
        Ok(self.read(Place {
            base: Base::Addr(address),
            offset: 0,
            repr: whole,
        }))
    }

    fn loops(&self, stmts: List, tail: Option<ExprId>) -> bool {
        let looping = |expr: ExprId| {
            matches!(
                *self.body.expr(expr),
                Expr::While { .. } | Expr::Loop { .. } | Expr::For { .. }
            )
        };
        tail.is_some_and(looping)
            || self
                .body
                .list(stmts)
                .iter()
                .any(|&stmt| matches!(self.body.stmts[stmt as usize], Stmt::Expr(expr, _) if looping(expr)))
    }

    pub(super) fn is_mutable_static(&self, item: ItemRef) -> bool {
        let (src, entry) = self.cx.item(item);
        entry
            .name
            .is_none_or(|name| name == 0 || src.text(name - 1) == "mut")
    }

    pub(super) fn static_rodata_blob(
        &mut self,
        item: ItemRef,
        substs: Substs,
        ty: Ty,
    ) -> Result<Option<(Vec<u8>, Vec<RodataReloc>)>> {
        if self.is_mutable_static(item)
            || self.cx.thread_local(item)
            || !super::is_freeze_ty(self.cx, ty, 0)
        {
            return Ok(None);
        }
        self.static_literal(item, substs, ty)
    }

    fn zero_static(&mut self, item: ItemRef, substs: Substs, ty: Ty) -> Result<bool> {
        Ok(self
            .static_literal(item, substs, ty)?
            .is_some_and(|(bytes, relocs)| relocs.is_empty() && bytes.iter().all(|&b| b == 0)))
    }

    fn static_literal(
        &mut self,
        item: ItemRef,
        substs: Substs,
        ty: Ty,
    ) -> Result<Option<(Vec<u8>, Vec<RodataReloc>)>> {
        if self.cx.foreign(item) {
            return Ok(None);
        }
        if self.cx.layout(ty)?.align.bytes() > 16 {
            return Ok(None);
        }
        if !self.blob_statics.insert((item, substs)) {
            return Ok(None);
        }
        let found = self.static_blob(item, substs, ty);
        self.blob_statics.remove(&(item, substs));
        found
    }

    fn static_blob(
        &mut self,
        item: ItemRef,
        substs: Substs,
        ty: Ty,
    ) -> Result<Option<(Vec<u8>, Vec<RodataReloc>)>> {
        let Some((source, typed)) = self.literal_source(item, substs)? else {
            return Ok(None);
        };
        let typed = typed.scoped(&self.cx.types);
        let static_ty = typed.ret_ty();
        if static_ty != ty {
            return Ok(None);
        }
        let body = self.cx.body(source)?;
        let (Some(root), src) = (body.root, self.cx.body_src(source)) else {
            return Ok(None);
        };
        Ok(self.literal_blob(
            Scope {
                body: &body,
                src,
                typed: &typed,
                in_const: true,
            },
            (root, static_ty),
            0,
        ))
    }

    fn emit_static_rodata_blob(
        &mut self,
        index: usize,
        bytes: &[u8],
        relocs: &[RodataReloc],
    ) -> Value {
        if relocs.is_empty() {
            self.emit_static_rodata(index, bytes)
        } else {
            let id = self.slots.rodata(bytes, relocs);
            self.emit_symbol(SymbolSource::Rodata(id))
        }
    }

    fn non_zero_static_rodata(
        &mut self,
        index: usize,
        item: ItemRef,
        substs: Substs,
        ty: Ty,
    ) -> Result<Option<Value>> {
        let Some((bytes, relocs)) = self.static_rodata_blob(item, substs, ty)? else {
            return Ok(None);
        };
        if relocs.is_empty() && bytes.len() >= 64 && bytes.iter().all(|&b| b == 0) {
            return Ok(None);
        }
        Ok(Some(self.emit_static_rodata_blob(index, &bytes, &relocs)))
    }

    fn ensure_static_filled(&mut self, index: usize, address: Value, size: i32) -> Result<()> {
        let flag = self.ins().iadd_imm(address, i64::from(size));
        let state = self
            .ins()
            .atomic_load(Type::I8, flag, haste_codegen::AtomicOrd::Acquire);
        let filled = self.ins().icmp_imm(
            IntCC::Equal,
            state,
            i64::from(crate::image::StaticInit::Filled as u8),
        );
        self.unless(filled, |lower| {
            let usize_ty = lower.cx.types.prim(Primitive::Usize);
            let table = lower.emit_slot_with(
                SymbolSource::Slot(index),
                super::reloc::RelocStep::Const(SymbolSource::Slot(index)),
            );
            let code = lower.ins().load(Type::I64, table, 0);
            let once = lower.emit_symbol(SymbolSource::Host(HostSymbol::FillOnce));
            lower.call_code(once, (vec![usize_ty, usize_ty], None), vec![flag, code])?;
            Ok(())
        })
    }

    /// The place of the static `item` at `substs`, of type `ty`: the prepped library's copy when Haste shares it, immutable literal bytes in read-only data when eligible, else its cell, which the static's initializer fills on the first use that finds it empty unless it starts as zero bytes.
    pub fn static_place(&mut self, (item, substs): (ItemRef, Substs), ty: Ty) -> Result<Place> {
        if let Some(address) = crate::prep::static_address(self.cx, item) {
            let base = self.emit_symbol(SymbolSource::Word(address as u64));
            return Ok(Place {
                base: Base::Addr(base),
                offset: 0,
                repr: repr(self.cx, ty)?,
            });
        }
        let index = self.reached_index(item, substs)?;
        if let Some(base) = self.non_zero_static_rodata(index, item, substs, ty)? {
            return Ok(Place {
                base: Base::Addr(base),
                offset: 0,
                repr: repr(self.cx, ty)?,
            });
        }
        let is_tls = !self.cx.foreign(item) && self.cx.thread_local(item);
        let needs_fill =
            !self.cx.foreign(item) && index != self.index && !self.zero_static(item, substs, ty)?;
        let (address, size) = self.static_cell(index, ty, is_tls && needs_fill)?;
        if !is_tls && needs_fill {
            self.ensure_static_filled(index, address, size)?;
        }
        Ok(Place {
            base: Base::Addr(address),
            offset: 0,
            repr: repr(self.cx, ty)?,
        })
    }

    /// Calls `HostSymbol::ThreadCell(slot, size, init_slot)` with the shared 3-argument ABI.
    pub(super) fn call_thread_cell(
        &mut self,
        slot: Value,
        size: i64,
        init_slot: Option<Value>,
    ) -> Result<Value> {
        let usize_ty = self.cx.types.prim(Primitive::Usize);
        let code = self.emit_symbol(SymbolSource::Host(HostSymbol::ThreadCell));
        self.b.func.mark_thread_cell(code);
        let init = init_slot.unwrap_or_else(|| self.ins().iconst(Type::I64, 0));
        let args = vec![slot, self.ins().iconst(Type::I64, size), init];
        let cell = self.call_code(
            code,
            (vec![usize_ty, usize_ty, usize_ty], Some(usize_ty)),
            args,
        )?;
        cell.ok_or_else(|| anyhow::anyhow!("no thread cell"))
    }

    fn static_cell(&mut self, index: usize, ty: Ty, init_tls: bool) -> Result<(Value, i32)> {
        let size = self.cx.layout(ty)?.size.bytes() as u32;
        let item = self.reached.list[index].0.0;
        if !self.cx.foreign(item) && self.cx.thread_local(item) {
            let idx_val = self.emit_symbol(SymbolSource::ThreadSlot(index));
            let init = init_tls.then(|| {
                self.emit_slot_with(
                    SymbolSource::Slot(index),
                    super::reloc::RelocStep::Const(SymbolSource::Slot(index)),
                )
            });
            return Ok((
                self.call_thread_cell(idx_val, i64::from(size), init)?,
                size as i32,
            ));
        }
        let address = self.emit_symbol(SymbolSource::StaticCell { index, size });
        Ok((address, size as i32))
    }

    /// Writes `value`, the value of the static that this function initializes, into the static's cell, and returns.
    pub fn fill_static(&mut self, value: Option<Value>) -> Result<()> {
        let (item, ty) = (self.reached.list[self.index].0.0, self.typed.ret_ty());
        if self.open && crate::prep::static_address(self.cx, item).is_none() {
            let (address, _) = self.static_cell(self.index, ty, false)?;
            let place = Place {
                base: Base::Addr(address),
                offset: 0,
                repr: repr(self.cx, ty)?,
            };
            if let Some(value) = value {
                self.write(place, value);
            }
        }
        self.finish(None)
    }
}
