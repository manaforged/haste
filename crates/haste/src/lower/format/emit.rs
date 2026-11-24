use anyhow::{Result, bail};
use haste_codegen::{Block, FloatCC, IntCC, Type, Value};

use crate::names::{ItemRef, Primitive};
use crate::sema::format::{DEFAULT_FLAGS, PRECISION_SET};
use crate::sema::infer::Vtable;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, int_info};

use super::super::place::{Base, Place};
use super::super::reloc::SiteOrigin;
use super::super::stencil::SymbolSource;
use super::super::{Lower, Repr, machine, repr};
use super::{HolePlan, HoleTrait};

impl Lower<'_, '_> {
    pub(super) fn emit_write_lit(
        &mut self,
        dst: Value,
        write_str: (ItemRef, Substs),
        bytes: &[u8],
    ) -> Result<Option<Value>> {
        let id = self.slots.rodata(bytes, &[]);
        let data = self.emit_symbol(SymbolSource::Rodata(id));
        let str_val = self.wide_pointer(data, bytes.len() as i64);
        self.invoke(
            SiteOrigin::Known(write_str.0, write_str.1),
            write_str.0,
            write_str.1,
            vec![dst, str_val],
        )
    }

    pub(super) fn branch_on_fmt_err(
        &mut self,
        res: Option<Value>,
        result_ty: Ty,
        err_block: Block,
    ) -> Result<()> {
        let Some(res) = res else { return Ok(()) };
        let whole = repr(self.cx, result_ty)?;
        let place = match whole {
            Repr::Mem { .. } => Place {
                base: Base::Addr(res),
                offset: 0,
                repr: whole,
            },
            _ => {
                let Some(slot) = self.slot(whole) else {
                    return Ok(());
                };
                let place = Place::slot(slot, whole);
                self.write(place, res);
                place
            }
        };
        let Some(is_err) = self.is_variant(Some(place), result_ty, 1)? else {
            return Ok(());
        };
        let cont = self.b.create_block();
        self.brif(is_err, (err_block, &[]), (cont, &[]));
        self.enter(cont);
        Ok(())
    }

    pub(super) fn emit_hole(
        &mut self,
        dst: Value,
        write_str: (ItemRef, Substs),
        vtable: &Vtable,
        hole: &HolePlan,
        arg_val: Value,
        result_ty: Ty,
        err_block: Block,
    ) -> Result<()> {
        let default_spec =
            hole.flags == DEFAULT_FLAGS && hole.width.is_none() && hole.precision.is_none();
        if default_spec && matches!(hole.kind, HoleTrait::Display | HoleTrait::Debug) {
            if let Some((mty, signed)) = self.int_spec(hole.inner_ty)? {
                let raw = self.ins().load(mty, arg_val, 0);
                return self.emit_write_int(dst, write_str, raw, mty, signed, result_ty, err_block);
            }
            if hole.kind == HoleTrait::Debug
                && let Kind::Adt(adt, args) = *self.cx.types.kind(hole.inner_ty)
                && self.cx.lang(adt) == Some("option")
                && let [elem_ty] = self.cx.types.substs(args)
                && let Some((mty, signed)) = self.int_spec(*elem_ty)?
            {
                return self.emit_write_option_int(
                    dst,
                    write_str,
                    arg_val,
                    hole.inner_ty,
                    mty,
                    signed,
                    result_ty,
                    err_block,
                );
            }
        }
        if hole.kind == HoleTrait::Display
            && hole.flags == (DEFAULT_FLAGS | PRECISION_SET)
            && hole.width.is_none()
            && let Some(prec) = hole.precision
            && (1..=6).contains(&prec)
            && matches!(
                *self.cx.types.kind(hole.inner_ty),
                Kind::Prim(Primitive::F64)
            )
        {
            return self.emit_write_f64_prec(
                dst, write_str, vtable, hole, arg_val, prec, result_ty, err_block,
            );
        }
        self.emit_formatter_call(dst, vtable, hole, arg_val, result_ty, err_block)
    }

    fn int_spec(&mut self, ty: Ty) -> Result<Option<(Type, bool)>> {
        let Kind::Prim(prim) = *self.cx.types.kind(ty) else {
            return Ok(None);
        };
        let Some((_, signed)) = int_info(prim) else {
            return Ok(None);
        };
        let Some(mty) = machine(self.cx, ty)? else {
            return Ok(None);
        };
        if matches!(mty, Type::I8 | Type::I16 | Type::I32 | Type::I64) {
            Ok(Some((mty, signed)))
        } else {
            Ok(None)
        }
    }

    fn emit_digits_loop(&mut self, mag: Value, ptr_end: Value) -> Value {
        let zero = self.ins().iconst(Type::I64, 0);
        let loop_bb = self.b.create_block();
        let done_bb = self.b.create_block();
        let rem = self.b.append_block_param(loop_bb, Type::I64);
        let ptr = self.b.append_block_param(loop_bb, Type::I64);
        let start_ptr = self.b.append_block_param(done_bb, Type::I64);
        self.jump(loop_bb, &[mag, ptr_end]);
        self.enter(loop_bb);
        let ptr_next = self.ins().iadd_imm(ptr, -1);
        let ten = self.ins().iconst(Type::I64, 10);
        let q = self.ins().udiv(rem, ten);
        let q10 = self.ins().imul(q, ten);
        let r = self.ins().isub(rem, q10);
        let digit = self.ins().iadd_imm(r, i64::from(b'0'));
        let digit8 = self.ins().ireduce(Type::I8, digit);
        self.ins().store(digit8, ptr_next, 0);
        let nz = self.ins().icmp(IntCC::NotEqual, q, zero);
        self.brif(nz, (loop_bb, &[q, ptr_next]), (done_bb, &[ptr_next]));
        self.enter(done_bb);
        start_ptr
    }

    fn emit_write_int(
        &mut self,
        dst: Value,
        write_str: (ItemRef, Substs),
        raw: Value,
        mty: Type,
        signed: bool,
        result_ty: Ty,
        err_block: Block,
    ) -> Result<()> {
        let val64 = match (mty, signed) {
            (Type::I64, _) => raw,
            (_, true) => self.ins().sextend(Type::I64, raw),
            (_, false) => self.ins().uextend(Type::I64, raw),
        };
        let zero = self.ins().iconst(Type::I64, 0);
        let (neg, mag) = if signed {
            let neg = self.ins().icmp(IntCC::SignedLessThan, val64, zero);
            let neg_val = self.ins().isub(zero, val64);
            (Some(neg), self.ins().select(neg, neg_val, val64))
        } else {
            (None, val64)
        };
        let buf = self.b.create_stack_slot(24, 0);
        let end = self.ins().stack_addr(Type::I64, buf, 24);
        let start_ptr = self.emit_digits_loop(mag, end);
        let final_ptr = if let Some(neg) = neg {
            let minus_ptr = self.ins().iadd_imm(start_ptr, -1);
            let minus = self.ins().iconst(Type::I8, i64::from(b'-'));
            self.ins().store(minus, minus_ptr, 0);
            self.ins().select(neg, minus_ptr, start_ptr)
        } else {
            start_ptr
        };
        let len = self.ins().isub(end, final_ptr);
        let slice = self.wide(final_ptr, len);
        let res = self.invoke(
            SiteOrigin::Known(write_str.0, write_str.1),
            write_str.0,
            write_str.1,
            vec![dst, slice],
        )?;
        self.branch_on_fmt_err(res, result_ty, err_block)
    }

    fn emit_write_f64_prec(
        &mut self,
        dst: Value,
        write_str: (ItemRef, Substs),
        vtable: &Vtable,
        hole: &HolePlan,
        arg_val: Value,
        prec: u16,
        result_ty: Ty,
        err_block: Block,
    ) -> Result<()> {
        let val = self.ins().load(Type::F64, arg_val, 0);
        let bits = self.ins().load(Type::I64, arg_val, 0);
        let limit = self
            .ins()
            .iconst(Type::I64, 0x4170_0000_0000_0000_u64 as i64);
        let in_range = self.ins().icmp(IntCC::UnsignedLessThan, bits, limit);
        let check_tie_bb = self.b.create_block();
        let check_hi_bb = self.b.create_block();
        let fast_bb = self.b.create_block();
        let slow_bb = self.b.create_block();
        let join_bb = self.b.create_block();
        let scaled = self.b.append_block_param(fast_bb, Type::I64);
        self.brif(in_range, (check_tie_bb, &[]), (slow_bb, &[]));
        self.enter(check_tie_bb);
        let scale = 10_u64.pow(u32::from(prec));
        let scale_f = self.float_const(Type::F64, scale as f64)?;
        let y = self.ins().fmul(val, scale_f);
        let n = self.ins().fcvt_to_uint_sat(Type::I64, y);
        let n_f = self.ins().fcvt_from_uint(Type::F64, n);
        let frac = self.ins().fsub(y, n_f);
        let lo = self.float_const(Type::F64, 0.484375_f64)?;
        let hi = self.float_const(Type::F64, 0.515625_f64)?;
        let is_lo = self.ins().fcmp(FloatCC::LessThan, frac, lo);
        self.brif(is_lo, (fast_bb, &[n]), (check_hi_bb, &[]));
        self.enter(check_hi_bb);
        let is_hi = self.ins().fcmp(FloatCC::GreaterThan, frac, hi);
        let n_plus_1 = self.ins().iadd_imm(n, 1);
        self.brif(is_hi, (fast_bb, &[n_plus_1]), (slow_bb, &[]));
        self.enter(fast_bb);
        let buf = self.b.create_stack_slot(32, 0);
        let end = self.ins().stack_addr(Type::I64, buf, 32);
        let ten = self.ins().iconst(Type::I64, 10);
        let mut rem = scaled;
        let mut cursor = 32_i32;
        for _ in 0..prec {
            cursor -= 1;
            let q = self.ins().udiv(rem, ten);
            let q10 = self.ins().imul(q, ten);
            let r = self.ins().isub(rem, q10);
            let digit = self.ins().iadd_imm(r, i64::from(b'0'));
            let digit8 = self.ins().ireduce(Type::I8, digit);
            self.ins().stack_store(digit8, buf, cursor);
            rem = q;
        }
        cursor -= 1;
        let dot = self.ins().iconst(Type::I8, i64::from(b'.'));
        self.ins().stack_store(dot, buf, cursor);
        let frac_ptr = self.ins().stack_addr(Type::I64, buf, cursor);
        let start_ptr = self.emit_digits_loop(rem, frac_ptr);
        let len = self.ins().isub(end, start_ptr);
        let slice = self.wide(start_ptr, len);
        let res = self.invoke(
            SiteOrigin::Known(write_str.0, write_str.1),
            write_str.0,
            write_str.1,
            vec![dst, slice],
        )?;
        self.branch_on_fmt_err(res, result_ty, err_block)?;
        self.jump(join_bb, &[]);
        self.enter(slow_bb);
        self.emit_formatter_call(dst, vtable, hole, arg_val, result_ty, err_block)?;
        self.jump(join_bb, &[]);
        self.enter(join_bb);
        Ok(())
    }

    fn emit_write_option_int(
        &mut self,
        dst: Value,
        write_str: (ItemRef, Substs),
        arg_val: Value,
        opt_ty: Ty,
        mty: Type,
        signed: bool,
        result_ty: Ty,
        err_block: Block,
    ) -> Result<()> {
        let opt_repr = repr(self.cx, opt_ty)?;
        let place = Place {
            base: Base::Addr(arg_val),
            offset: 0,
            repr: opt_repr,
        };
        let Some(is_some) = self.is_variant(Some(place), opt_ty, 1)? else {
            let res = self.emit_write_lit(dst, write_str, b"None")?;
            return self.branch_on_fmt_err(res, result_ty, err_block);
        };
        let some_bb = self.b.create_block();
        let none_bb = self.b.create_block();
        let join_bb = self.b.create_block();
        self.brif(is_some, (some_bb, &[]), (none_bb, &[]));
        self.enter(none_bb);
        let res_none = self.emit_write_lit(dst, write_str, b"None")?;
        self.branch_on_fmt_err(res_none, result_ty, err_block)?;
        self.jump(join_bb, &[]);
        self.enter(some_bb);
        let res_open = self.emit_write_lit(dst, write_str, b"Some(")?;
        self.branch_on_fmt_err(res_open, result_ty, err_block)?;
        let field_place = self.variant_field(place, opt_ty, (1, 0))?;
        let Some(raw) = self.read(field_place) else {
            bail!("Option<Int> payload has no value");
        };
        self.emit_write_int(dst, write_str, raw, mty, signed, result_ty, err_block)?;
        let res_close = self.emit_write_lit(dst, write_str, b")")?;
        self.branch_on_fmt_err(res_close, result_ty, err_block)?;
        self.jump(join_bb, &[]);
        self.enter(join_bb);
        Ok(())
    }

    fn emit_formatter_call(
        &mut self,
        dst: Value,
        vtable: &Vtable,
        hole: &HolePlan,
        arg_val: Value,
        result_ty: Ty,
        err_block: Block,
    ) -> Result<()> {
        let &fmt_idx = self
            .reached
            .index
            .get(&(hole.fmt_item, hole.fmt_substs))
            .ok_or_else(|| anyhow::anyhow!("missing fmt fn"))?;
        let Ok(fmt_typed) = &self.reached.list[fmt_idx].1 else {
            bail!("fmt fn did not type-check");
        };
        let &fmt_param = fmt_typed
            .params
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("fmt fn missing Formatter param"))?;
        let fmt_ref_ty = fmt_typed.locals[fmt_param as usize].ty;
        let Kind::Ref(_, fmt_ty) = *self.cx.types.kind(fmt_ref_ty) else {
            bail!("expected &mut Formatter");
        };
        let Kind::Adt(fmt_adt, fmt_args) = *self.cx.types.kind(fmt_ty) else {
            bail!("expected Formatter ADT");
        };
        let fmt_layout = self.cx.layout(fmt_ty)?;
        let fmt_slice = self.cx.types.substs(fmt_args).to_vec();
        let fields = crate::sema::infer::field_types(self.cx, fmt_adt, &fmt_slice)?;
        let opts_ty = *fields
            .first()
            .and_then(|v| v.first())
            .ok_or_else(|| anyhow::anyhow!("Formatter missing Options field"))?;
        let opts_off = self.field_offset(fmt_ty, 0)?;
        let buf_off = self.field_offset(fmt_ty, 1)?;
        let flags_off = opts_off + self.field_offset(opts_ty, 0)?;
        let width_off = opts_off + self.field_offset(opts_ty, 1)?;
        let prec_off = opts_off + self.field_offset(opts_ty, 2)?;
        let slot = self.b.create_stack_slot(
            fmt_layout.size.bytes() as u32,
            fmt_layout.align.bytes().trailing_zeros() as u8,
        );
        let fmt_ptr = self.ins().stack_addr(Type::I64, slot, 0);
        let flags_val = self.ins().iconst(Type::I32, i64::from(hole.flags));
        let width_val = self
            .ins()
            .iconst(Type::I16, i64::from(hole.width.unwrap_or(0)));
        let prec_val = self
            .ins()
            .iconst(Type::I16, i64::from(hole.precision.unwrap_or(0)));
        self.ins().stack_store(flags_val, slot, flags_off);
        self.ins().stack_store(width_val, slot, width_off);
        self.ins().stack_store(prec_val, slot, prec_off);
        let vt_id =
            super::super::object::build_vtable_id(self.cx, self.reached, self.slots, vtable)?;
        let vt_ptr = self.emit_symbol(SymbolSource::Vtable(vt_id));
        self.ins().stack_store(dst, slot, buf_off);
        self.ins().stack_store(vt_ptr, slot, buf_off + 8);
        let res = self.invoke(
            SiteOrigin::Known(hole.fmt_item, hole.fmt_substs),
            hole.fmt_item,
            hole.fmt_substs,
            vec![arg_val, fmt_ptr],
        )?;
        self.branch_on_fmt_err(res, result_ty, err_block)
    }
}
