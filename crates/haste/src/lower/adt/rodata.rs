use anyhow::Result;
use haste_codegen::{RuntimeSymbol, StackSlot, Type, Value};

use crate::body::{Body, ExprId};
use crate::names::ItemRef;
use crate::outline::source::Source;
use crate::sema::infer::{Shape, Typed, needs_drop};
use crate::sema::substs::Substs;
use crate::sema::ty::Ty;

use super::super::place::Place;
use super::super::{Lower, Parts, Repr, repr};
use super::{CONST_RODATA_MAX_BYTES, ConstScope};

const SPARSE_MIN_BYTES: usize = 256;
const SPARSE_INLINE_SPAN: usize = 32;

fn is_zero_chunk(bytes: &[u8], at: usize) -> bool {
    bytes[at..(at + 8).min(bytes.len())].iter().all(|&b| b == 0)
}

fn next_nonzero_span(bytes: &[u8], mut at: usize) -> Option<(usize, usize)> {
    while at < bytes.len() && is_zero_chunk(bytes, at) {
        at += 8;
    }
    if at >= bytes.len() {
        return None;
    }
    let start = at;
    while at < bytes.len() && !is_zero_chunk(bytes, at) {
        at = (at + 8).min(bytes.len());
    }
    Some((start, at))
}

impl<'l, 'a> Lower<'l, 'a> {
    pub(in crate::lower) fn const_rodata_expr(&mut self, expr: ExprId) -> Result<Option<Value>> {
        self.materialize_const_expr(expr, (false, false))
    }

    pub(in crate::lower) fn materialize_const_expr(
        &mut self,
        expr: ExprId,
        (allow_slot, borrowed): (bool, bool),
    ) -> Result<Option<Value>> {
        let lent_mut = self.expr_lent_mutably(expr);
        if (!allow_slot && lent_mut) || !self.open {
            return Ok(None);
        }
        let ty = self.ty_of(expr);
        let Some(size) = self.rodata_size(ty, borrowed)? else {
            return Ok(None);
        };
        let in_generic = !self
            .cx
            .types
            .substs(self.reached.list[self.index].0.1)
            .is_empty();
        let mut bytes = vec![0u8; size];
        let scope = ConstScope {
            body: self.body,
            src: self.src,
            typed: self.typed,
            in_const: false,
        };
        if self
            .write_const_uncoerced(scope, (expr, ty), 0, &mut bytes, 0, in_generic)
            .is_none()
        {
            return Ok(None);
        }
        if allow_slot
            && let whole @ Repr::Mem {
                parts: Parts::One(vec_ty @ Type::Vec { .. }),
                ..
            } = repr(self.cx, ty)?
        {
            let all_zero = bytes.iter().all(|&b| b == 0);
            if all_zero || lent_mut {
                return self.store_vec_const(whole, vec_ty, &bytes, all_zero);
            }
        }
        if lent_mut {
            return Ok(None);
        }
        if borrowed {
            return Ok(Some(self.emit_const_rodata(&bytes)));
        }
        self.materialize_value_rodata(ty, &bytes)
    }

    fn store_vec_const(
        &mut self,
        whole: Repr,
        vec_ty: Type,
        bytes: &[u8],
        all_zero: bool,
    ) -> Result<Option<Value>> {
        let Some(slot) = self.slot(whole) else {
            return Ok(None);
        };
        let vec = if all_zero {
            self.ins().iconst(vec_ty, 0)
        } else {
            let addr = self.emit_const_rodata(bytes);
            self.ins().load(vec_ty, addr, 0)
        };
        self.ins().stack_store(vec, slot, 0);
        Ok(self.read(Place::slot(slot, whole)))
    }

    pub(in crate::lower) fn materialize_value_rodata(
        &mut self,
        ty: Ty,
        bytes: &[u8],
    ) -> Result<Option<Value>> {
        if bytes.len() >= SPARSE_MIN_BYTES
            && bytes.iter().filter(|&&b| b != 0).count() * 4 <= bytes.len()
        {
            return self.emit_sparse_slot(ty, bytes);
        }
        Ok(Some(self.emit_const_rodata(bytes)))
    }

    fn emit_sparse_slot(&mut self, ty: Ty, bytes: &[u8]) -> Result<Option<Value>> {
        let whole = repr(self.cx, ty)?;
        let Some(slot) = self.slot(whole) else {
            return Ok(None);
        };
        let base = Place::slot(slot, whole);
        if self.open {
            let dst = self.address(base);
            let zero = self.ins().iconst(Type::I64, 0);
            let len = self.ins().iconst(Type::I64, bytes.len() as i64);
            self.call_libc(RuntimeSymbol::Memset, vec![dst, zero, len], None)?;
            self.write_nonzero_spans(slot, bytes);
        }
        Ok(self.read(base))
    }

    fn write_nonzero_spans(&mut self, slot: StackSlot, bytes: &[u8]) {
        let mut cursor = 0;
        while let Some((lo, hi)) = next_nonzero_span(bytes, cursor) {
            cursor = hi;
            let span = &bytes[lo..hi];
            if span.len() <= SPARSE_INLINE_SPAN {
                self.write_inline_span(slot, lo, span);
            } else {
                let sub = self.emit_const_rodata(span);
                let to = self.ins().stack_addr(Type::I64, slot, lo as i32);
                self.ins().move_value(to, sub, span.len() as u32);
            }
        }
    }

    fn write_inline_span(&mut self, slot: StackSlot, base_off: usize, span: &[u8]) {
        let mut off = 0;
        while off + 8 <= span.len() {
            let mut word = [0u8; 8];
            word.copy_from_slice(&span[off..off + 8]);
            let val = self.ins().iconst(Type::I64, i64::from_le_bytes(word));
            self.ins().stack_store(val, slot, (base_off + off) as i32);
            off += 8;
        }
        while off < span.len() {
            let val = self.ins().iconst(Type::I8, i64::from(span[off]));
            self.ins().stack_store(val, slot, (base_off + off) as i32);
            off += 1;
        }
    }

    pub(in crate::lower) fn const_rodata_item(
        &mut self,
        expr: ExprId,
        item: ItemRef,
        substs: Substs,
    ) -> Result<Option<Value>> {
        if !self.open || self.expr_lent_mutably(expr) {
            return Ok(None);
        }
        let ty = self.ty_of(expr);
        let Some(size) = self.rodata_size(ty, false)? else {
            return Ok(None);
        };
        let in_generic = !self
            .cx
            .types
            .substs(self.reached.list[self.index].0.1)
            .is_empty();
        let Some((body, src, item_typed, root)) =
            self.load_const_item(item, substs, ty, in_generic, true)?
        else {
            return Ok(None);
        };
        let mut bytes = vec![0u8; size];
        let scoped = item_typed.scoped(&self.cx.types);
        if self
            .write_const_bytes(
                ConstScope {
                    body: &body,
                    src,
                    typed: &scoped,
                    in_const: true,
                },
                (root, ty),
                0,
                &mut bytes,
                1,
                in_generic,
            )
            .is_none()
        {
            return Ok(None);
        }
        self.materialize_value_rodata(ty, &bytes)
    }

    pub(super) fn rodata_size(&mut self, ty: Ty, scalar: bool) -> Result<Option<usize>> {
        if ty == Ty::NONE || self.cx.types.has_params(ty) {
            return Ok(None);
        }
        let (size, align) = match repr(self.cx, ty)? {
            Repr::Mem { size, align, .. } if size >= 8 => (size as usize, align),
            Repr::Scalar(machine) if scalar => (machine.bytes() as usize, 0),
            _ => return Ok(None),
        };
        if align > 4
            || size > CONST_RODATA_MAX_BYTES
            || !super::super::is_freeze_ty(self.cx, ty, 0)
            || needs_drop(self.cx, ty)?
        {
            return Ok(None);
        }
        Ok(Some(size))
    }

    fn const_item_readable(&self, item: ItemRef) -> bool {
        let env = &self.cx.outline.crates[item.krate as usize].env;
        let owner = self.reached.list[self.index].0.0;
        env.release || env.fixed || item.krate == owner.krate
    }

    pub(super) fn load_const_item(
        &mut self,
        item: ItemRef,
        substs: Substs,
        ty: Ty,
        in_generic: bool,
        require_reached: bool,
    ) -> Result<
        Option<(
            std::sync::Arc<Body>,
            &'a Source,
            std::sync::Arc<Typed>,
            ExprId,
        )>,
    > {
        if !self.const_item_readable(item)
            || (in_generic && !self.cx.types.substs(substs).is_empty())
        {
            return Ok(None);
        }
        let index = match require_reached {
            true => self.reached_index(item, substs)?,
            false => match self.reached.index.get(&(item, substs)) {
                Some(&idx) => idx,
                None => return Ok(None),
            },
        };
        let ((source, _), Ok(item_typed)) = &self.reached.list[index] else {
            return Ok(None);
        };
        if !matches!(item_typed.shape, Shape::Body) {
            return Ok(None);
        }
        let item_typed = item_typed.clone();
        if item_typed.scoped(&self.cx.types).ret_ty() != ty {
            return Ok(None);
        }
        let source = *source;
        let body = self.cx.body(source)?;
        let (Some(root), src) = (body.root, self.cx.body_src(source)) else {
            return Ok(None);
        };
        Ok(Some((body, src, item_typed, root)))
    }
}
