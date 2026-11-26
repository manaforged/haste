use std::sync::Arc;

use anyhow::{Result, bail};
use haste_codegen::{Signature, Type, Value};

use crate::body::{Expr, ExprId, List};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::infer::{Res, SiteKey};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty};

use super::super::place::Place;
use super::super::reloc::{RelocStep, SiteOrigin};
use super::super::stencil::{Form, InlinedCallee, LowerMode, SymbolSource, resolve_inline};
use super::super::{Lower, Parts, Repr, repr};

const BOX_ZEROED_MIN: u32 = 64;

enum Callee {
    Code(Value),
    Reached(usize, SiteOrigin),
}

impl Lower<'_, '_> {
    pub(super) fn call(
        &mut self,
        expr: ExprId,
        callee: ExprId,
        args: List,
    ) -> Result<Option<Value>> {
        if self.body.list(args).is_empty()
            && super::super::array::is_maybe_uninit_fn(self.cx, self.typed, expr, callee)
        {
            if !self.open {
                return Ok(None);
            }
            let whole = repr(self.cx, self.ty_of(expr))?;
            return Ok(self
                .slot(whole)
                .and_then(|slot| self.read(Place::slot(slot, whole))));
        }
        if let Some(&Res::Fn(item, substs)) = self.typed.res.get(&expr) {
            if let Some(res) = self.try_box_new_zeroed(expr, item, args)? {
                return Ok(Some(res));
            }
            return self.call_value((expr, callee), (item, substs), args);
        }
        let named = self
            .typed
            .res
            .get(&callee)
            .filter(|_| self.body.names_callee(callee));
        let (item, substs) = match named.copied() {
            Some(Res::Fn(item, _))
                if matches!(
                    self.cx.item(item).1.kind,
                    ItemKind::Const | ItemKind::Static
                ) =>
            {
                return self.pointer_call(expr, callee, args);
            }
            Some(Res::Fn(item, substs)) => (item, substs),
            Some(Res::Ctor(_, variant, _)) => return self.ctor_call(expr, variant, args),
            _ => return self.pointer_call(expr, callee, args),
        };
        if let Some(res) = self.named_fn_call((expr, callee), (item, substs), args)? {
            return Ok(res);
        }
        let values = self.args(args, Vec::new())?;
        self.invoke(
            SiteOrigin::Table(SiteKey::ExprRes(callee)),
            item,
            substs,
            values,
        )
    }

    fn named_fn_call(
        &mut self,
        (expr, callee): (ExprId, ExprId),
        (item, substs): (ItemRef, Substs),
        args: List,
    ) -> Result<Option<Option<Value>>> {
        if let Some(res) = self.try_box_new_zeroed(expr, item, args)? {
            return Ok(Some(Some(res)));
        }
        if !self.tupled_closure_call(item, substs) {
            return Ok(None);
        }
        let arg_list = self.body.list(args).to_vec();
        let [f_expr, tuple_expr] = arg_list.as_slice() else {
            bail!("tupled closure call expects 2 arguments")
        };
        let f_expr = ExprId(*f_expr);
        let values = self.callee_env(f_expr, true)?;
        let guards = self.guard_consumed_callee(f_expr, &values, true)?;
        let mut values = values;
        self.untuple_arg_guarded(ExprId(*tuple_expr), &mut values, guards)?;
        self.invoke(
            SiteOrigin::Table(SiteKey::ExprRes(callee)),
            item,
            substs,
            values,
        )
        .map(Some)
    }

    fn try_box_new_zeroed(
        &mut self,
        expr: ExprId,
        item: ItemRef,
        args: List,
    ) -> Result<Option<Value>> {
        if !self.open
            || self.cx.global_allocator().is_some()
            || self.cx.name(item) != "new"
            || self.cx.outline.roots[item.krate as usize].name != "alloc"
        {
            return Ok(None);
        }
        let &[arg] = self.body.list(args) else {
            return Ok(None);
        };
        let arg = ExprId(arg);
        let box_ty = self.ty_of(expr);
        let Some(inner_ty) = self.cx.boxed(box_ty) else {
            return Ok(None);
        };
        if inner_ty != self.ty_of(arg) || self.typed.coercions.contains_key(&arg) {
            return Ok(None);
        }
        let Repr::Mem { size, align, .. } = repr(self.cx, inner_ty)? else {
            return Ok(None);
        };
        if size <= BOX_ZEROED_MIN
            || align >= 63
            || !self.box_repr_ptr(box_ty)?
            || !self.is_zero_init_expr(arg)?
        {
            return Ok(None);
        }
        let size_val = self.ins().iconst(Type::I64, i64::from(size));
        let align_val = self.ins().iconst(Type::I64, 1_i64 << align);
        let Some(ptr) = self.call_host(
            crate::host::HostSymbol::AllocZeroed,
            &[size_val, align_val],
            Some(Type::I64),
        ) else {
            return Ok(None);
        };
        self.ins().trapz(ptr);
        self.received(box_ty, &[(Type::I64, 0)], &[ptr])
    }

    fn box_repr_ptr(&mut self, box_ty: Ty) -> Result<bool> {
        Ok(matches!(
            repr(self.cx, box_ty)?,
            Repr::Scalar(Type::I64)
                | Repr::Mem {
                    size: 8,
                    parts: Parts::One(Type::I64),
                    ..
                }
        ))
    }

    fn is_zero_init_expr(&mut self, expr: ExprId) -> Result<bool> {
        if self.typed.coercions.contains_key(&expr) {
            return Ok(false);
        }
        let e = self.unwrap_expr(expr, true);
        if self.typed.coercions.contains_key(&e) {
            return Ok(false);
        }
        match *self.body.expr(e) {
            Expr::Lit(..) => self.literal_is_zero(e),
            Expr::Repeat(val, _) => self.is_zero_init_expr(val),
            Expr::Array(items) | Expr::Tuple(items) => self.all_zero_init(items),
            Expr::Struct {
                fields,
                base: None,
                rest: false,
                ..
            } => self.struct_zero_init(e, fields),
            _ => Ok(false),
        }
    }

    fn all_zero_init(&mut self, items: List) -> Result<bool> {
        for item in self.body.list(items).to_vec() {
            if !self.is_zero_init_expr(ExprId(item))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn struct_zero_init(&mut self, expr: ExprId, fields: List) -> Result<bool> {
        let Kind::Adt(adt, _) = *self.cx.types.kind(self.ty_of(expr)) else {
            return Ok(false);
        };
        if self.cx.item(adt).1.kind != ItemKind::Struct {
            return Ok(false);
        }
        for at in self.body.list(fields).to_vec() {
            let val = self.body.fields[at as usize].value;
            if !self.is_zero_init_expr(val)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn method(
        &mut self,
        expr: ExprId,
        receiver: ExprId,
        params: List,
    ) -> Result<Option<Value>> {
        let Some(&Res::Fn(item, substs)) = self.typed.res.get(&expr) else {
            bail!("method call has no resolution")
        };
        let Some(&adjust) = self.typed.adjusts.get(&expr) else {
            bail!("method call has no receiver adjustment")
        };
        let first = self.receiver((expr, receiver), adjust)?;
        if let Some(result) = self.try_write_fmt(expr, item, substs, first, params)? {
            return Ok(Some(result));
        }
        if !self.open {
            return Ok(None);
        }
        let mut guards = Vec::new();
        if adjust
            == (crate::sema::infer::Adjust {
                derefs: 0,
                autoref: None,
            })
            && !self.body.list(params).is_empty()
            && self.wants_field_unwind()?
        {
            self.push_arg_guard(receiver, first, &mut guards)?;
        }
        if self.tupled_closure_call(item, substs) {
            let param_list = self.body.list(params).to_vec();
            let [tuple_expr] = param_list.as_slice() else {
                bail!("tupled closure method expects 1 argument")
            };
            let closure_ty = self.cx.types.substs(substs)[0];
            let mut values = if self.cx.closure_takes_env(closure_ty) {
                first.into_iter().collect()
            } else {
                Vec::new()
            };
            self.untuple_arg_guarded(ExprId(*tuple_expr), &mut values, guards)?;
            return self.invoke(
                SiteOrigin::Table(SiteKey::ExprRes(expr)),
                item,
                substs,
                values,
            );
        }
        let values = self.eval_guarded_args(params, first.into_iter().collect(), guards, true)?;
        self.invoke(
            SiteOrigin::Table(SiteKey::ExprRes(expr)),
            item,
            substs,
            values,
        )
    }

    pub(in crate::lower) fn args(&mut self, args: List, values: Vec<Value>) -> Result<Vec<Value>> {
        self.eval_guarded_args(args, values, Vec::new(), true)
    }

    pub(in crate::lower) fn invoke(
        &mut self,
        origin: SiteOrigin,
        item: ItemRef,
        substs: Substs,
        values: Vec<Value>,
    ) -> Result<Option<Value>> {
        if !self.open {
            return Ok(None);
        }
        if let Some(which) = self.cx.intrinsic(item) {
            return self.intrinsic(
                &origin,
                which,
                &self.cx.types.substs(substs).to_vec(),
                &values,
            );
        }
        let index = self.reached_index(item, substs)?;
        let callee = self.reached.typed(index, self.cx.name(item))?;
        let params = callee
            .params
            .iter()
            .map(|&local| callee.locals[local as usize].ty)
            .collect();
        if self.cx.foreign(item) {
            if callee
                .ret
                .map_or(Ok(true), |ret| repr(self.cx, ret).map(|r| r == Repr::Zst))?
            {
                match self.cx.symbol_name(item).as_deref() {
                    Some(n)
                        if crate::host::HostSymbol::resolve(n)
                            == Some(crate::host::HostSymbol::Noop) =>
                    {
                        return Ok(None);
                    }
                    Some("llvm.aarch64.isb") => {
                        self.ins().fence(haste_codegen::AtomicOrd::Isb);
                        return Ok(None);
                    }
                    Some("llvm.aarch64.dmb") => {
                        self.ins().fence(haste_codegen::AtomicOrd::SeqCst);
                        return Ok(None);
                    }
                    _ => {}
                }
            }
            let code = self.code_of(index, origin);
            if let Some(host) = self
                .cx
                .symbol_name(item)
                .as_deref()
                .and_then(crate::host::HostSymbol::resolve)
            {
                self.mark_if_fresh_alloc(&host, code);
            }
            return self.native_call(code, (params, callee.ret), values);
        }
        self.call_with(Callee::Reached(index, origin), (params, callee.ret), values)
    }

    fn inline_callee(&mut self, index: usize) -> Result<Option<Arc<InlinedCallee>>> {
        let in_loop = self.loops.iter().any(|frame| !frame.block);
        let Some(limit) = super::super::budget::caller_inline_limit(
            self.cx,
            self.reached,
            (self.index, index),
            in_loop,
        ) else {
            return Ok(None);
        };
        let (callee, cycle) = resolve_inline(
            self.cx,
            (self.reached, self.slots),
            self.inline,
            index,
            (limit, false),
        )?;
        self.cycle_root = self.cycle_root.min(cycle);
        let Some(found) = callee.filter(|found| {
            self.inline.host.takes(self.cx, self.reached, index, found)
                && (self.mode != LowerMode::Dict || found.body.unwind_drops().is_empty())
        }) else {
            return Ok(None);
        };
        if !self.inline.host.compact || found.form != Form::Dense {
            return Ok(Some(found));
        }
        let (compact, cycle) = resolve_inline(
            self.cx,
            (self.reached, self.slots),
            self.inline,
            index,
            (limit, true),
        )?;
        self.cycle_root = self.cycle_root.min(cycle);
        Ok(compact
            .filter(|found| self.mode != LowerMode::Dict || found.body.unwind_drops().is_empty()))
    }

    fn emit_inlined(
        &mut self,
        inlined: &InlinedCallee,
        origin: SiteOrigin,
        values: &[Value],
    ) -> Result<Vec<Value>> {
        if !self.open {
            let dead = *self.dead.get_or_insert_with(|| self.b.create_block());
            self.b.switch_to_block(dead);
        }
        let sym_base = self.relocs.len() as u32;
        self.relocs.extend(inlined.relocs.iter().cloned());
        self.reloc_plan.push(RelocStep::SiteInline {
            origin,
            plan: Arc::clone(&inlined.reloc_plan),
        });
        Ok(match self.mode {
            LowerMode::Stencil => self.b.inline_shifted(&inlined.body, values, sym_base),
            LowerMode::Dict => {
                let Some(dict) = self.dict_param else {
                    bail!("dictionary parameter is not set")
                };
                self.b.inline_dict(&inlined.body, values, dict, sym_base)
            }
        })
    }

    fn emit_call(
        &mut self,
        callee: Callee,
        signature: Signature,
        values: &[Value],
    ) -> Result<Vec<Value>> {
        let caller_tracked = self.cx.track_caller(self.reached.list[self.index].0.0);
        let mut track_site: Option<bool> = None;
        let code = match callee {
            Callee::Code(code) => code,
            Callee::Reached(index, origin) => {
                let callee_tracked = self.cx.track_caller(self.reached.list[index].0.0);
                let wants_track = !caller_tracked && callee_tracked;
                if let Some(inlined) = self
                    .slots
                    .allow_inline()
                    .then(|| self.inline_callee(index))
                    .transpose()?
                    .flatten()
                {
                    if caller_tracked && !callee_tracked && inlined.body.has_calls() {
                        return self.with_caller_loc(Some(false), |this| {
                            this.emit_inlined(&inlined, origin, values)
                        });
                    }
                    let (was_open, caller_blk, from_blk) =
                        (self.open, self.b.current_block(), self.b.block_count());
                    let res = self.emit_inlined(&inlined, origin, values)?;
                    if wants_track && was_open {
                        self.wrap_inlined_caller_loc(
                            caller_blk,
                            from_blk,
                            self.b.block_count().saturating_sub(1),
                        )?;
                    }
                    return Ok(res);
                }
                track_site = (caller_tracked != callee_tracked).then_some(wants_track);
                self.code_of(index, origin)
            }
        };
        self.with_caller_loc(track_site, |this| {
            let sig = this.b.import_signature(signature);
            let call = this.ins().call_indirect(sig, code, values);
            Ok(this.b.inst_results(call).to_vec())
        })
    }

    fn pointer_call(&mut self, expr: ExprId, callee: ExprId, args: List) -> Result<Option<Value>> {
        let (code, ty) = match self.typed.autoderefs.contains_key(&expr) {
            true => match self.autoderef_place(expr, callee)? {
                Some((place, ty)) => (self.read(place), ty),
                None => return Ok(None),
            },
            false => (self.expr(callee)?, self.ty_of(callee)),
        };
        if let Kind::FnDef(item, substs) = *self.cx.types.kind(ty) {
            let values = self.args(args, Vec::new())?;
            return self.invoke(
                SiteOrigin::PointerFnDef { expr, callee },
                item,
                substs,
                values,
            );
        }
        let Kind::FnPtr(params, ret) = *self.cx.types.kind(ty) else {
            bail!("cannot lower calls through values yet")
        };
        let Some(code) = code else { return Ok(None) };
        let params = self.cx.types.substs(params).to_vec();
        let values = self.args(args, Vec::new())?;
        if !self.open {
            return Ok(None);
        }
        let clear = self
            .cx
            .track_caller(self.reached.list[self.index].0.0)
            .then_some(false);
        self.with_caller_loc(clear, |this| {
            this.call_code(code, (params, Some(ret)), values)
        })
    }

    pub(in crate::lower) fn reached_index(&self, item: ItemRef, substs: Substs) -> Result<usize> {
        match self.try_reached_index(item, substs)? {
            Some(index) => Ok(index),
            None => bail!("a use of `{}` that was not reached", self.cx.name(item)),
        }
    }

    pub(in crate::lower) fn code_of(&mut self, index: usize, origin: SiteOrigin) -> Value {
        let table = self.emit_slot(SymbolSource::Slot(index), origin);
        self.ins().load(Type::I64, table, 0)
    }

    pub(in crate::lower) fn call_code(
        &mut self,
        code: Value,
        types: (Vec<Ty>, Option<Ty>),
        values: Vec<Value>,
    ) -> Result<Option<Value>> {
        self.call_with(Callee::Code(code), types, values)
    }

    fn call_with(
        &mut self,
        callee: Callee,
        (params, ret): (Vec<Ty>, Option<Ty>),
        values: Vec<Value>,
    ) -> Result<Option<Value>> {
        let signature = super::super::signature_of(self.cx, ret, params.iter().copied())?;
        let mut values = self.pass_all(&params, values)?;
        let result = ret
            .map(|ret| repr(self.cx, ret))
            .transpose()?
            .unwrap_or(Repr::Zst);
        let out = if super::super::indirect(&signature) {
            self.slot(result)
        } else {
            None
        };
        if let Some(out) = out {
            let address = self.ins().stack_addr(Type::I64, out, 0);
            values.insert(0, address);
        }
        let results = self.emit_call(callee, signature, &values)?;
        let Some(ret) = ret else { return Ok(None) };
        if matches!(self.cx.types.kind(ret), Kind::Never) {
            self.ins().trap();
            self.close();
            return Ok(None);
        }
        if let Some(out) = out {
            return Ok(Some(self.ins().stack_addr(Type::I64, out, 0)));
        }
        let parts: Vec<(Type, i32)> = super::super::passed_parts(self.cx, ret)?
            .into_iter()
            .flat_map(Parts::scalars)
            .collect();
        self.received(ret, &parts, &results)
    }

    pub(in crate::lower) fn pass_all(
        &mut self,
        params: &[Ty],
        values: Vec<Value>,
    ) -> Result<Vec<Value>> {
        let mut values = values.into_iter();
        let mut passed = Vec::new();
        for &param in params {
            if repr(self.cx, param)? == Repr::Zst {
                continue;
            }
            let Some(value) = values.next() else { break };
            self.push_passed(param, value, &mut passed)?;
        }
        passed.extend(values);
        Ok(passed)
    }
}
