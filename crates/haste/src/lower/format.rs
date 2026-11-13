use anyhow::{Result, bail};
use haste_codegen::{Block, IntCC, Type, Value};

use crate::body::literal::string_bytes;
use crate::body::{Expr, ExprId, List, Lit};
use crate::names::ItemRef;
use crate::sema::format::DEFAULT_FLAGS;
use crate::sema::infer::{Coercion, Res, Vtable};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, int_info};

use super::place::{Base, Place};
use super::reloc::SiteOrigin;
use super::stencil::{LowerMode, SymbolSource};
use super::{Lower, Repr, machine, repr};

enum Segment {
    Lit(Vec<u8>),
    Hole { flags: u32, width: Option<u16>, precision: Option<u16> },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HoleTrait {
    Display,
    Debug,
    Other,
}

struct HolePlan {
    arg_idx: usize,
    kind: HoleTrait,
    fmt_item: ItemRef,
    fmt_substs: Substs,
    inner_ty: Ty,
    flags: u32,
    width: Option<u16>,
    precision: Option<u16>,
}

enum EmitPlan {
    Literal(Vec<u8>),
    Staged {
        arg_refs: Vec<ExprId>,
        segments: Vec<Segment>,
        holes: Vec<HolePlan>,
    },
}

impl Lower<'_, '_> {
    pub(super) fn try_write_fmt(&mut self, expr: ExprId, item: ItemRef, substs: Substs, first: Option<Value>, params: List) -> Result<Option<Value>> {
        if !self.open || self.mode != LowerMode::Stencil || !self.cx.types.substs(self.reached.list[self.index].0.1).is_empty() {
            return Ok(None);
        }
        if !self.is_core_write_fmt(item) {
            return Ok(None);
        }
        let Some(dst) = first else { return Ok(None) };
        let [arg_raw] = self.body.list(params) else { return Ok(None) };
        let arg_expr = ExprId(*arg_raw);
        let Some((write_str_item, write_str_substs, vtable)) = self.resolve_write_str(item, substs) else {
            return Ok(None);
        };
        let Some(plan) = self.plan_format_args(arg_expr)? else {
            return Ok(None);
        };
        let result_ty = self.ty_of(expr);
        match plan {
            EmitPlan::Literal(bytes) => {
                if bytes.is_empty() {
                    return self.construct(result_ty, 0, Vec::new(), None);
                }
                self.emit_write_lit(dst, (write_str_item, write_str_substs), &bytes)
            }
            EmitPlan::Staged { arg_refs, segments, holes } => {
                let mut arg_vals = Vec::with_capacity(arg_refs.len());
                for arg_ref in arg_refs {
                    let Some(val) = self.expr(arg_ref)? else { return Ok(None) };
                    arg_vals.push(val);
                }
                let err_block = self.b.create_block();
                let (done_block, joined, join_slot) = self.join_block(expr)?;
                let mut hole_cursor = 0;
                for segment in segments {
                    match segment {
                        Segment::Lit(bytes) if bytes.is_empty() => {}
                        Segment::Lit(bytes) => {
                            let res = self.emit_write_lit(dst, (write_str_item, write_str_substs), &bytes)?;
                            self.branch_on_fmt_err(res, result_ty, err_block)?;
                        }
                        Segment::Hole { .. } => {
                            let hole = &holes[hole_cursor];
                            hole_cursor += 1;
                            let arg_val = arg_vals[hole.arg_idx];
                            self.emit_hole(dst, (write_str_item, write_str_substs), &vtable, hole, arg_val, result_ty, err_block)?;
                        }
                    }
                }
                let ok_val = self.construct(result_ty, 0, Vec::new(), None)?;
                self.branch_into(ok_val, (done_block, join_slot));
                self.enter(err_block);
                let err_val = self.construct(result_ty, 1, Vec::new(), None)?;
                self.branch_into(err_val, (done_block, join_slot));
                self.enter(done_block);
                Ok(self.joined(joined, join_slot))
            }
        }
    }

    fn is_core(&self, item: ItemRef) -> bool {
        self.cx.outline.roots[item.krate as usize].name == "core"
    }

    fn is_core_write_fmt(&mut self, item: ItemRef) -> bool {
        self.cx.name(item) == "write_fmt"
            && self.is_core(item)
            && self.cx.module_of(item).map(|at| self.cx.outline.module_path(at)).is_ok_and(|p| p == "core::fmt")
    }

    fn vtable_in(&self, idx: usize) -> Option<Vtable> {
        if self.reached.is_untyped(idx) {
            return None;
        }
        let Ok(typed) = &self.reached.list[idx].1 else { return None };
        let scoped = typed.scoped(&self.cx.types);
        scoped.coercions.values().find_map(|c| match c {
            Coercion::Vtable(vt) => Some(vt.clone()),
            _ => None,
        })
    }

    fn resolve_write_str(&self, write_fmt_item: ItemRef, write_fmt_substs: Substs) -> Option<(ItemRef, Substs, Vtable)> {
        let &wf_idx = self.reached.index.get(&(write_fmt_item, write_fmt_substs))?;
        if self.reached.is_untyped(wf_idx) {
            return None;
        }
        let Ok(wf_typed) = &self.reached.list[wf_idx].1 else { return None };
        let vtable = self.vtable_in(wf_idx).or_else(|| {
            wf_typed
                .calls()
                .find(|&(it, _)| self.cx.name(it) == "spec_write_fmt")
                .and_then(|key| self.reached.index.get(&key).copied())
                .and_then(|idx| self.vtable_in(idx))
        })?;
        let Some(Res::Fn(ws_item, ws_substs)) = vtable
            .methods
            .iter()
            .flatten()
            .find(|r| matches!(r, Res::Fn(it, _) if self.cx.name(*it) == "write_str"))
            .copied()
        else {
            return None;
        };
        let &ws_idx = self.reached.index.get(&(ws_item, ws_substs))?;
        if self.reached.is_untyped(ws_idx) || self.reached.list[ws_idx].1.is_err() {
            return None;
        }
        Some((ws_item, ws_substs, vtable))
    }

    fn unwrap_fmt_expr(&self, expr: ExprId) -> ExprId {
        self.unwrap_expr(expr, true)
    }

    fn plan_format_args(&self, arg_expr: ExprId) -> Result<Option<EmitPlan>> {
        let inner = self.unwrap_fmt_expr(arg_expr);
        match self.body.expr(inner) {
            Expr::Call(callee, args) => {
                let Some(&Res::Fn(fn_item, _)) = self.typed.res.get(callee) else { return Ok(None) };
                if self.cx.name(fn_item) != "from_str" || !self.is_core(fn_item) {
                    return Ok(None);
                }
                let [lit_raw] = self.body.list(*args) else { return Ok(None) };
                let lit_expr = self.unwrap_fmt_expr(ExprId(*lit_raw));
                let Expr::Lit(Lit::Str, at) = self.body.expr(lit_expr) else { return Ok(None) };
                let bytes = string_bytes(self.src.text(*at))?;
                Ok(Some(EmitPlan::Literal(bytes)))
            }
            Expr::Match(scrut, arms) => {
                let [arm_raw] = self.body.list(*arms) else { return Ok(None) };
                let arm = self.body.arms[*arm_raw as usize];
                if arm.guard.is_some() {
                    return Ok(None);
                }
                let call_expr = self.unwrap_fmt_expr(arm.body);
                let Expr::Call(new_callee, new_args) = self.body.expr(call_expr) else { return Ok(None) };
                let Some(&Res::Fn(new_item, _)) = self.typed.res.get(new_callee) else { return Ok(None) };
                if self.cx.name(new_item) != "new" || !self.is_core(new_item) {
                    return Ok(None);
                }
                let [tpl_raw, rt_raw] = self.body.list(*new_args) else { return Ok(None) };
                let tpl_expr = self.unwrap_fmt_expr(ExprId(*tpl_raw));
                let Expr::Lit(Lit::ByteStr, tpl_at) = self.body.expr(tpl_expr) else { return Ok(None) };
                let bytecode = string_bytes(self.src.text(*tpl_at))?;
                let Some(segments) = decode_bytecode(&bytecode) else { return Ok(None) };
                let Expr::Tuple(arg_raws) = self.body.expr(self.unwrap_fmt_expr(*scrut)) else { return Ok(None) };
                let arg_refs: Vec<ExprId> = self.body.list(*arg_raws).iter().map(|&id| ExprId(id)).collect();
                let Expr::Ref { expr: arr_expr, .. } = self.body.expr(self.unwrap_fmt_expr(ExprId(*rt_raw))) else {
                    return Ok(None);
                };
                let Expr::Array(ctor_raws) = self.body.expr(self.unwrap_fmt_expr(*arr_expr)) else {
                    return Ok(None);
                };
                let ctor_list = self.body.list(*ctor_raws);
                let hole_segments: Vec<(u32, Option<u16>, Option<u16>)> = segments
                    .iter()
                    .filter_map(|seg| match seg {
                        Segment::Hole { flags, width, precision } => Some((*flags, *width, *precision)),
                        Segment::Lit(_) => None,
                    })
                    .collect();
                if hole_segments.len() != ctor_list.len() {
                    return Ok(None);
                }
                let mut holes = Vec::with_capacity(ctor_list.len());
                for (i, &ctor_raw) in ctor_list.iter().enumerate() {
                    let Some(plan) = self.plan_hole(ExprId(ctor_raw), &arg_refs, hole_segments[i])? else {
                        return Ok(None);
                    };
                    holes.push(plan);
                }
                Ok(Some(EmitPlan::Staged { arg_refs, segments, holes }))
            }
            _ => Ok(None),
        }
    }

    fn plan_hole(&self, ctor_expr: ExprId, arg_refs: &[ExprId], (flags, width, precision): (u32, Option<u16>, Option<u16>)) -> Result<Option<HolePlan>> {
        let Expr::Call(ctor_callee, ctor_args) = self.body.expr(self.unwrap_fmt_expr(ctor_expr)) else {
            return Ok(None);
        };
        let [arg_field_raw] = self.body.list(*ctor_args) else { return Ok(None) };
        let Expr::TupleField(_, arg_idx_u32) = self.body.expr(self.unwrap_fmt_expr(ExprId(*arg_field_raw))) else {
            return Ok(None);
        };
        let arg_idx = *arg_idx_u32 as usize;
        let Some(&arg_ref_expr) = arg_refs.get(arg_idx) else { return Ok(None) };
        let Kind::Ref(_, inner_ty) = *self.cx.types.kind(self.ty_of(arg_ref_expr)) else {
            return Ok(None);
        };
        let Some(&Res::Fn(ctor_item, ctor_substs)) = self.typed.res.get(ctor_callee) else {
            return Ok(None);
        };
        let kind = match self.cx.name(ctor_item) {
            "new_display" => HoleTrait::Display,
            "new_debug" => HoleTrait::Debug,
            _ => HoleTrait::Other,
        };
        let Some(&ctor_idx) = self.reached.index.get(&(ctor_item, ctor_substs)) else { return Ok(None) };
        if self.reached.is_untyped(ctor_idx) {
            return Ok(None);
        }
        let Ok(ctor_typed) = &self.reached.list[ctor_idx].1 else { return Ok(None) };
        let Some((fmt_item, fmt_substs)) = ctor_typed.calls().find(|&(item, _)| self.cx.name(item) == "fmt") else {
            return Ok(None);
        };
        let Some(&fmt_idx) = self.reached.index.get(&(fmt_item, fmt_substs)) else { return Ok(None) };
        if self.reached.is_untyped(fmt_idx) || self.reached.list[fmt_idx].1.is_err() {
            return Ok(None);
        }
        Ok(Some(HolePlan { arg_idx, kind, fmt_item, fmt_substs, inner_ty, flags, width, precision }))
    }

    fn emit_write_lit(&mut self, dst: Value, write_str: (ItemRef, Substs), bytes: &[u8]) -> Result<Option<Value>> {
        let id = self.slots.rodata(bytes, &[]);
        let data = self.emit_symbol(SymbolSource::Rodata(id));
        let str_val = self.wide_pointer(data, bytes.len() as i64);
        self.invoke(SiteOrigin::Known(write_str.0, write_str.1), write_str.0, write_str.1, vec![dst, str_val])
    }

    fn branch_on_fmt_err(&mut self, res: Option<Value>, result_ty: Ty, err_block: Block) -> Result<()> {
        let Some(res) = res else { return Ok(()) };
        let whole = repr(self.cx, result_ty)?;
        let place = match whole {
            Repr::Mem { .. } => Place { base: Base::Addr(res), offset: 0, repr: whole },
            _ => {
                let Some(slot) = self.slot(whole) else { return Ok(()) };
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

    fn emit_hole(
        &mut self,
        dst: Value,
        write_str: (ItemRef, Substs),
        vtable: &Vtable,
        hole: &HolePlan,
        arg_val: Value,
        result_ty: Ty,
        err_block: Block,
    ) -> Result<()> {
        let default_spec = hole.flags == DEFAULT_FLAGS && hole.width.is_none() && hole.precision.is_none();
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
                return self.emit_write_option_int(dst, write_str, arg_val, hole.inner_ty, mty, signed, result_ty, err_block);
            }
        }
        self.emit_formatter_call(dst, vtable, hole, arg_val, result_ty, err_block)
    }

    fn int_spec(&mut self, ty: Ty) -> Result<Option<(Type, bool)>> {
        let Kind::Prim(prim) = *self.cx.types.kind(ty) else { return Ok(None) };
        let Some((_, signed)) = int_info(prim) else { return Ok(None) };
        let Some(mty) = machine(self.cx, ty)? else { return Ok(None) };
        if matches!(mty, Type::I8 | Type::I16 | Type::I32 | Type::I64) {
            Ok(Some((mty, signed)))
        } else {
            Ok(None)
        }
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
        let loop_bb = self.b.create_block();
        let done_bb = self.b.create_block();
        let rem = self.b.append_block_param(loop_bb, Type::I64);
        let ptr = self.b.append_block_param(loop_bb, Type::I64);
        let start_ptr = self.b.append_block_param(done_bb, Type::I64);
        self.jump(loop_bb, &[mag, end]);
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
        let res = self.invoke(SiteOrigin::Known(write_str.0, write_str.1), write_str.0, write_str.1, vec![dst, slice])?;
        self.branch_on_fmt_err(res, result_ty, err_block)
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
        let place = Place { base: Base::Addr(arg_val), offset: 0, repr: opt_repr };
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
        let Some(raw) = self.read(field_place) else { bail!("Option<Int> payload has no value") };
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
        let &fmt_idx = self.reached.index.get(&(hole.fmt_item, hole.fmt_substs)).ok_or_else(|| anyhow::anyhow!("missing fmt fn"))?;
        let Ok(fmt_typed) = &self.reached.list[fmt_idx].1 else { bail!("fmt fn did not type-check") };
        let &fmt_param = fmt_typed.params.get(1).ok_or_else(|| anyhow::anyhow!("fmt fn missing Formatter param"))?;
        let fmt_ref_ty = fmt_typed.locals[fmt_param as usize].ty;
        let Kind::Ref(_, fmt_ty) = *self.cx.types.kind(fmt_ref_ty) else { bail!("expected &mut Formatter") };
        let Kind::Adt(fmt_adt, fmt_args) = *self.cx.types.kind(fmt_ty) else { bail!("expected Formatter ADT") };
        let fmt_layout = self.cx.layout(fmt_ty)?;
        let fmt_slice = self.cx.types.substs(fmt_args).to_vec();
        let fields = crate::sema::infer::field_types(self.cx, fmt_adt, &fmt_slice)?;
        let opts_ty = *fields.first().and_then(|v| v.first()).ok_or_else(|| anyhow::anyhow!("Formatter missing Options field"))?;
        let opts_off = self.field_offset(fmt_ty, 0)?;
        let buf_off = self.field_offset(fmt_ty, 1)?;
        let flags_off = opts_off + self.field_offset(opts_ty, 0)?;
        let width_off = opts_off + self.field_offset(opts_ty, 1)?;
        let prec_off = opts_off + self.field_offset(opts_ty, 2)?;
        let slot = self.b.create_stack_slot(fmt_layout.size.bytes() as u32, fmt_layout.align.bytes().trailing_zeros() as u8);
        let fmt_ptr = self.ins().stack_addr(Type::I64, slot, 0);
        let flags_val = self.ins().iconst(Type::I32, i64::from(hole.flags));
        let width_val = self.ins().iconst(Type::I16, i64::from(hole.width.unwrap_or(0)));
        let prec_val = self.ins().iconst(Type::I16, i64::from(hole.precision.unwrap_or(0)));
        self.ins().stack_store(flags_val, slot, flags_off);
        self.ins().stack_store(width_val, slot, width_off);
        self.ins().stack_store(prec_val, slot, prec_off);
        let vt_id = super::object::build_vtable_id(self.cx, self.reached, self.slots, vtable)?;
        let vt_ptr = self.emit_symbol(SymbolSource::Vtable(vt_id));
        self.ins().stack_store(dst, slot, buf_off);
        self.ins().stack_store(vt_ptr, slot, buf_off + 8);
        let res = self.invoke(SiteOrigin::Known(hole.fmt_item, hole.fmt_substs), hole.fmt_item, hole.fmt_substs, vec![arg_val, fmt_ptr])?;
        self.branch_on_fmt_err(res, result_ty, err_block)
    }
}

fn decode_bytecode(bytes: &[u8]) -> Option<Vec<Segment>> {
    let mut segments = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let lead = bytes[i];
        i += 1;
        match lead {
            0 => break,
            1..0x80 => {
                let len = lead as usize;
                let end = i.checked_add(len)?;
                let text = bytes.get(i..end)?.to_vec();
                i = end;
                segments.push(Segment::Lit(text));
            }
            0x80 => {
                let len_bytes: [u8; 2] = bytes.get(i..i + 2)?.try_into().ok()?;
                i += 2;
                let len = u16::from_le_bytes(len_bytes) as usize;
                let end = i.checked_add(len)?;
                let text = bytes.get(i..end)?.to_vec();
                i = end;
                segments.push(Segment::Lit(text));
            }
            _ if lead >= 0xC0 => {
                if lead & (8 | 16 | 32) != 0 {
                    return None;
                }
                let flags = if lead & 1 != 0 {
                    let b: [u8; 4] = bytes.get(i..i + 4)?.try_into().ok()?;
                    i += 4;
                    u32::from_le_bytes(b)
                } else {
                    DEFAULT_FLAGS
                };
                let width = if lead & 2 != 0 {
                    let b: [u8; 2] = bytes.get(i..i + 2)?.try_into().ok()?;
                    i += 2;
                    Some(u16::from_le_bytes(b))
                } else {
                    None
                };
                let precision = if lead & 4 != 0 {
                    let b: [u8; 2] = bytes.get(i..i + 2)?.try_into().ok()?;
                    i += 2;
                    Some(u16::from_le_bytes(b))
                } else {
                    None
                };
                segments.push(Segment::Hole { flags, width, precision });
            }
            _ => return None,
        }
    }
    Some(segments)
}
