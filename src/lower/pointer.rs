use haste_codegen::{IntCC, Signature, Type, Value};
use hir_def::{AdtId, FunctionId, ItemContainerId, TraitId, VariantId};
use hir_ty::mir::PlaceTy;
use hir_ty::next_solver::{GenericArgs, Ty, TyKind};
use hir_ty::utils::all_super_traits;
use rustc_type_ir::inherent::{BoundExistentialPredicates as _, GenericArgs as _, IntoKind, SliceLike as _};
use rustc_type_ir::Binder;

use super::{Lower, Place, pointee};
use crate::backend::{Key, Shape, Sig, Target, key_of, pointer_signature, resolve_def};

const VTABLE_DROP: i32 = 0;
const VTABLE_SIZE: i32 = 8;
const VTABLE_ALIGN: i32 = 16;
const VTABLE_HEADER: usize = 3;

fn methods_of<'db>(db: &'db dyn hir_ty::db::HirDatabase, principal: TraitId) -> Vec<(TraitId, FunctionId)> {
    let mut methods = Vec::new();
    for owner in all_super_traits(db, principal) {
        for (_, item) in &owner.trait_items(db).items {
            if let hir_def::AssocItemId::FunctionId(func) = item {
                methods.push((*owner, *func));
            }
        }
    }
    methods
}

fn peel<'db>(ty: Ty<'db>) -> Ty<'db> {
    match ty.kind() {
        TyKind::Pat(inner, _) => inner,
        _ => ty,
    }
}

fn is_pointer(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), TyKind::Ref(..) | TyKind::RawPtr(..))
}

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn is_fn_trait(&self, func: FunctionId) -> bool {
        let ItemContainerId::TraitId(owner) = func.loc(self.db).container else {
            return false;
        };
        let items = self.infcx.interner.lang_items();
        [items.Fn, items.FnMut, items.FnOnce].contains(&Some(owner))
    }

    pub(super) fn fields_of(&self, variant: VariantId, args: GenericArgs<'db>) -> Vec<Ty<'db>> {
        let types = self.db.field_types(variant);
        variant
            .fields(self.db)
            .fields()
            .iter()
            .map(|(field, _)| types[field].ty().instantiate(self.infcx.interner, args).skip_norm_wip())
            .map(|ty| self.normalized(ty))
            .collect()
    }

    pub(super) fn coerce(&mut self, from: Place<'db>, to: Place<'db>) -> Result<(), String> {
        let (source, target) = (peel(from.ty.ty), peel(to.ty.ty));
        if is_pointer(source) && is_pointer(target) {
            let (old, new) = (pointee(source)?, pointee(target)?);
            let previous =
                if self.layout(old)?.is_unsized() { Some(self.load(Type::I64, from.addr, 8)) } else { None };
            let data = self.load(Type::I64, from.addr, 0);
            let meta = self.unsize_meta(old, new, previous)?;
            self.store(data, to.addr, 0);
            self.store(meta, to.addr, 8);
            return Ok(());
        }
        let (TyKind::Adt(left, left_args), TyKind::Adt(right, right_args)) = (source.kind(), target.kind()) else {
            return Err(format!("unsize of {source:?} to {target:?}"));
        };
        let AdtId::StructId(id) = left.def_id() else {
            return Err("unsize of a non-struct".to_owned());
        };
        if left.def_id() != right.def_id() {
            return Err("unsize across different structs".to_owned());
        }
        let left_fields = self.fields_of(id.into(), left_args);
        let right_fields = self.fields_of(id.into(), right_args);
        let (left_layout, right_layout) = (self.layout(source)?, self.layout(target)?);
        for (index, (old, new)) in left_fields.into_iter().zip(right_fields).enumerate() {
            let at = left_layout.fields.offset(index).bytes() as i64;
            let source = self.builder.ins().iadd_imm(from.addr, at);
            let at = right_layout.fields.offset(index).bytes() as i64;
            let dest = self.builder.ins().iadd_imm(to.addr, at);
            if old == new {
                let size = self.layout(old)?.size.bytes();
                self.copy(dest, source, size);
            } else {
                self.coerce(
                    Place { addr: source, ty: PlaceTy::from_ty(old), meta: None },
                    Place { addr: dest, ty: PlaceTy::from_ty(new), meta: None },
                )?;
            }
        }
        Ok(())
    }

    fn unsize_meta(&mut self, old: Ty<'db>, new: Ty<'db>, previous: Option<Value>) -> Result<Value, String> {
        let (old, new) = (peel(old), peel(new));
        match (old.kind(), new.kind()) {
            (TyKind::Array(..), TyKind::Slice(_)) => {
                let count = self.layout(old)?.fields.count();
                Ok(self.builder.ins().iconst(Type::I64, count as i64))
            }
            (TyKind::Dynamic(left, _), TyKind::Dynamic(right, _)) => {
                if left.principal_def_id() != right.principal_def_id() {
                    return Err("trait upcast".to_owned());
                }
                previous.ok_or_else(|| "dyn without a vtable".to_owned())
            }
            (_, TyKind::Dynamic(..)) => {
                let table = self.vtable(old, new)?;
                Ok(self.builder.ins().iconst(Type::I64, table))
            }
            (TyKind::Adt(left, left_args), TyKind::Adt(right, right_args)) if left.def_id() == right.def_id() => {
                let AdtId::StructId(id) = left.def_id() else {
                    return Err("unsized enum".to_owned());
                };
                let last_left = *self.fields_of(id.into(), left_args).last().ok_or("empty unsized struct")?;
                let last_right = *self.fields_of(id.into(), right_args).last().ok_or("empty unsized struct")?;
                self.unsize_meta(last_left, last_right, previous)
            }
            _ => Err(format!("unsize of {old:?} to {new:?}")),
        }
    }

    pub(super) fn vtable(&mut self, concrete: Ty<'db>, dynamic: Ty<'db>) -> Result<i64, String> {
        let key = (concrete.store(), dynamic.store());
        if let Some(table) = self.slots.vtables.get(&key) {
            return Ok(*table);
        }
        let TyKind::Dynamic(predicates, _) = dynamic.kind() else {
            return Err(format!("vtable for {dynamic:?}"));
        };
        let layout = self.layout(concrete)?;
        let mut entries = vec![0_i64, layout.size.bytes() as i64, layout.align.abi.bytes() as i64];
        if self.needs_drop(concrete) {
            let slot = self.glue_slot(concrete)?;
            entries[0] = self.slots.address(slot);
        }
        if let Some(principal) = predicates.principal() {
            let principal = Binder::skip_binder(principal);
            for (_, func) in methods_of(self.db, principal.def_id.0) {
                let entry = self.vtable_entry(concrete, func, principal.args).unwrap_or(0);
                entries.push(entry);
            }
        }
        let table = Box::leak(entries.into_boxed_slice()).as_ptr() as i64;
        self.slots.vtables.insert(key, table);
        Ok(table)
    }

    fn vtable_entry(&mut self, concrete: Ty<'db>, func: FunctionId, existential: GenericArgs<'db>) -> Result<i64, String> {
        let params = hir_def::GenericDefId::from(func);
        let own = hir_def::hir::generics::GenericParams::of(self.db, params);
        if own.len_type_or_consts() != 0 {
            return Err("generic method".to_owned());
        }
        let erased = hir_ty::next_solver::Region::new_erased(self.infcx.interner);
        let lifetimes = std::iter::repeat_n(erased.into(), own.len_early_bound_lifetimes());
        let args = GenericArgs::new_from_iter(std::iter::once(concrete.into()).chain(existential.iter()).chain(lifetimes));
        let ItemContainerId::TraitId(owner) = func.loc(self.db).container else {
            return Err("vtable method outside a trait".to_owned());
        };
        if hir_def::hir::generics::GenericParams::of(self.db, owner.into()).len() > args.as_slice().len() {
            return Err("supertrait with its own arguments".to_owned());
        }
        let resolved = resolve_def(self.db, self.env, hir_def::CallableDefId::FunctionId(func), args)?;
        let key = key_of(&resolved.target).ok_or("vtable entry without code")?;
        let slot = if resolved.shape == Shape::FnTrait {
            let sig = self.thunk_sig(&resolved.target, true)?;
            self.slots.get(Key::Thunk(Box::new(key), true), sig, self.env.store())?
        } else {
            let sig = self.target_sig(&resolved.target)?;
            self.slots.get(key, sig, self.env.store())?
        };
        Ok(self.slots.address(slot))
    }

    pub(super) fn thunk_sig(&self, target: &Target<'db>, data: bool) -> Result<Sig, String> {
        let (params, ret) = self.target_shape(target)?;
        let mut count = usize::from(data);
        for param in params {
            count += usize::from(self.layout(param)?.size.bytes() != 0);
        }
        Ok(Sig { params: count, ret: self.layout(ret)?.size.bytes() != 0 })
    }

    fn target_sig(&self, target: &Target<'db>) -> Result<Sig, String> {
        if let Target::Derive(method, _, args) = target {
            let self_ty = args.as_slice().first().and_then(|arg| arg.ty()).ok_or("derive without a self type")?;
            return super::derive::derive_sig(|ty| Ok(self.layout(ty)?.size.bytes()), *method, self_ty);
        }
        let (params, ret) = self.target_shape(target)?;
        let mut count = 0;
        for param in params {
            count += usize::from(self.layout(param)?.size.bytes() != 0);
        }
        Ok(Sig { params: count, ret: self.layout(ret)?.size.bytes() != 0 })
    }

    pub(super) fn target_shape(&self, target: &Target<'db>) -> Result<(Vec<Ty<'db>>, Ty<'db>), String> {
        let body = crate::backend::body_of(self.db, self.env, target)?;
        let skip = usize::from(matches!(target, Target::Closure(..)));
        let params = body.param_locals[skip..].iter().map(|local| body.locals[*local].ty.as_ref()).collect();
        let (_, ret) = body.locals.iter().next().ok_or("body has no return local")?;
        Ok((params, ret.ty.as_ref()))
    }

    pub(super) fn reify(&mut self, ty: Ty<'db>) -> Result<i64, String> {
        if self.is_native_fn(ty) {
            return self.native_symbol(ty)?.map(|code| code as i64).ok_or_else(|| "C callback in constant data".to_owned());
        }
        let target = match ty.kind() {
            TyKind::FnDef(def, args) => resolve_def(self.db, self.env, def.0, args)?.target,
            TyKind::Closure(closure, args) => {
                let parent = GenericArgs::new_from_slice(args.as_closure().parent_args());
                Target::Closure(closure.0, parent)
            }
            other => return Err(format!("fn pointer to {other:?}")),
        };
        let Some(key) = key_of(&target) else {
            let sig = self.shim_sig(ty)?;
            let slot = self.slots.get(Key::Shim(ty.store()), sig, self.env.store())?;
            return Ok(self.slots.address(slot));
        };
        let slot = if matches!(target, Target::Closure(..)) {
            let sig = self.thunk_sig(&target, false)?;
            self.slots.get(Key::Thunk(Box::new(key), false), sig, self.env.store())?
        } else {
            let sig = self.target_sig(&target)?;
            self.slots.get(key, sig, self.env.store())?
        };
        Ok(self.slots.address(slot))
    }

    pub(super) fn call_pointer(
        &mut self,
        code_cell: Value,
        dest: Place<'db>,
        places: &[Place<'db>],
    ) -> Result<(), String> {
        let mut values = Vec::with_capacity(places.len() + 1);
        let ret = self.layout(dest.ty.ty)?.size.bytes() != 0;
        if ret {
            values.push(dest.addr);
        }
        for place in places {
            if self.layout(place.ty.ty)?.size.bytes() != 0 {
                values.push(place.addr);
            }
        }
        self.call_cell(code_cell, Sig { params: values.len() - usize::from(ret), ret }, &values);
        Ok(())
    }

    pub(super) fn call_cell(&mut self, cell: Value, sig: Sig, values: &[Value]) {
        let mut signature = Signature::new(crate::backend::INTERNAL_CONV);
        pointer_signature(&mut signature, sig);
        let signature = self.builder.import_signature(signature);
        let code = self.load(Type::I64, cell, 0);
        self.invoke(signature, code, values);
    }

    pub(super) fn dyn_call(
        &mut self,
        func: FunctionId,
        self_ty: Ty<'db>,
        dest: Place<'db>,
        args: &[hir_ty::mir::Operand],
    ) -> Result<(), String> {
        let TyKind::Dynamic(predicates, _) = self_ty.kind() else {
            return Err(format!("dyn call on {self_ty:?}"));
        };
        let index = self.vtable_index(func, self_ty)?;
        let (receiver, rest) = args.split_first().ok_or("dyn call without a receiver")?;
        let receiver = self.operand(receiver)?;
        let fn_family = self.is_fn_trait(func);
        let (first, table) = if fn_family {
            if matches!(receiver.ty.ty.kind(), TyKind::Ref(..)) {
                (self.load(Type::I64, receiver.addr, 0), self.load(Type::I64, receiver.addr, 8))
            } else {
                (receiver.addr, receiver.meta.ok_or("unsized receiver without a vtable")?)
            }
        } else {
            (receiver.addr, self.load(Type::I64, receiver.addr, 8))
        };
        let tupled = predicates.principal().and_then(|principal| {
            Binder::skip_binder(principal).args.as_slice().first().and_then(|arg| arg.ty())
        });
        let places = self.operands(rest)?;
        let items: Option<Vec<Ty<'db>>> = tupled.and_then(|tuple| match tuple.kind() {
            TyKind::Tuple(items) => Some(items.as_slice().to_vec()),
            _ => None,
        });
        let places = match places[..] {
            [only] if fn_family && super::is_tupled(only.ty.ty, items.as_deref()) => self.spread(&rest[0])?,
            _ => places,
        };
        self.invoke_vtable(index, table, first, dest, &places)
    }

    fn vtable_index(&self, func: FunctionId, self_ty: Ty<'db>) -> Result<usize, String> {
        let TyKind::Dynamic(predicates, _) = self_ty.kind() else {
            return Err(format!("dyn call on {self_ty:?}"));
        };
        let principal = predicates.principal_def_id().ok_or("dyn call without a principal")?.0;
        methods_of(self.db, principal)
            .iter()
            .position(|(_, method)| *method == func)
            .ok_or_else(|| "method is not in the vtable".to_owned())
    }

    pub(super) fn dyn_item_call(
        &mut self,
        func: FunctionId,
        self_ty: Ty<'db>,
        dest: Place<'db>,
        places: &[Place<'db>],
    ) -> Result<(), String> {
        let index = self.vtable_index(func, self_ty)?;
        let (receiver, rest) = places.split_first().ok_or("dyn call without a receiver")?;
        let table = self.load(Type::I64, receiver.addr, 8);
        self.invoke_vtable(index, table, receiver.addr, dest, rest)
    }

    fn invoke_vtable(
        &mut self,
        index: usize,
        table: Value,
        first: Value,
        dest: Place<'db>,
        places: &[Place<'db>],
    ) -> Result<(), String> {
        let mut values = Vec::with_capacity(places.len() + 2);
        let ret = self.layout(dest.ty.ty)?.size.bytes() != 0;
        if ret {
            values.push(dest.addr);
        }
        values.push(first);
        for place in places {
            if self.layout(place.ty.ty)?.size.bytes() != 0 {
                values.push(place.addr);
            }
        }
        let cell = self.load(Type::I64, table, ((VTABLE_HEADER + index) * 8) as i32);
        self.builder.ins().trapz(cell);
        self.call_cell(cell, Sig { params: values.len() - usize::from(ret), ret }, &values);
        Ok(())
    }

    pub(super) fn drop_dyn(&mut self, data: Value, table: Value) {
        let cell = self.load(Type::I64, table, VTABLE_DROP);
        let call = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.ins().brif(cell, call, &[], done, &[]);
        self.builder.switch_to_block(call);
        let holder = self.thin_holder(data);
        self.call_cell(cell, Sig { params: 1, ret: false }, &[holder]);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
    }

    pub(super) fn size_align(&mut self, ty: Ty<'db>, meta: Option<Value>) -> Result<(Value, Value), String> {
        let layout = self.layout(ty)?;
        if !layout.is_unsized() {
            let size = self.builder.ins().iconst(Type::I64, layout.size.bytes() as i64);
            let align = self.builder.ins().iconst(Type::I64, layout.align.abi.bytes() as i64);
            return Ok((size, align));
        }
        let meta = meta.ok_or_else(|| format!("{ty:?} without metadata"))?;
        match ty.kind() {
            TyKind::Str => Ok((meta, self.builder.ins().iconst(Type::I64, 1))),
            TyKind::Slice(item) => {
                let item = self.layout(item)?;
                let size = self.builder.ins().imul_imm(meta, item.size.bytes() as i64);
                Ok((size, self.builder.ins().iconst(Type::I64, item.align.abi.bytes() as i64)))
            }
            TyKind::Dynamic(..) => Ok((self.load(Type::I64, meta, VTABLE_SIZE), self.load(Type::I64, meta, VTABLE_ALIGN))),
            TyKind::Adt(adt, args) => {
                let AdtId::StructId(id) = adt.def_id() else {
                    return Err("unsized enum".to_owned());
                };
                let fields = self.fields_of(id.into(), args);
                let last = *fields.last().ok_or("empty unsized struct")?;
                let offset = layout.fields.offset(fields.len() - 1).bytes() as i64;
                let (size, align) = self.size_align(last, Some(meta))?;
                let start = self.builder.ins().iconst(Type::I64, offset);
                let start = self.round_up(start, align);
                let end = self.builder.ins().iadd(start, size);
                let base = self.builder.ins().iconst(Type::I64, layout.align.abi.bytes() as i64);
                let wider = self.builder.ins().icmp(IntCC::UnsignedGreaterThan, base, align);
                let align = self.builder.ins().select(wider, base, align);
                Ok((self.round_up(end, align), align))
            }
            other => Err(format!("size of {other:?}")),
        }
    }

    pub(super) fn round_up(&mut self, value: Value, align: Value) -> Value {
        let mask = self.builder.ins().iadd_imm(align, -1);
        let sum = self.builder.ins().iadd(value, mask);
        let inverted = self.builder.ins().bnot(mask);
        self.builder.ins().band(sum, inverted)
    }

    pub(super) fn field_address(
        &mut self,
        base: Value,
        offset: u64,
        field: Ty<'db>,
        meta: Option<Value>,
    ) -> Result<Value, String> {
        if let Some(meta) = meta
            && self.dyn_tail(field)
        {
            let (_, align) = self.size_align(field, Some(meta))?;
            let start = self.builder.ins().iconst(Type::I64, offset as i64);
            let start = self.round_up(start, align);
            return Ok(self.builder.ins().iadd(base, start));
        }
        Ok(self.builder.ins().iadd_imm(base, offset as i64))
    }

    fn dyn_tail(&self, ty: Ty<'db>) -> bool {
        match ty.kind() {
            TyKind::Dynamic(..) => true,
            TyKind::Adt(adt, args) => {
                let AdtId::StructId(id) = adt.def_id() else {
                    return false;
                };
                self.fields_of(id.into(), args).last().is_some_and(|last| self.dyn_tail(*last))
            }
            _ => false,
        }
    }

    pub(super) fn indirect_callee(&mut self, func: &hir_ty::mir::Operand) -> Result<Value, String> {
        let place = self.operand(func)?;
        Ok(self.load(Type::I64, place.addr, 0))
    }

    fn callable(&self, ty: Ty<'db>) -> Result<(Vec<Ty<'db>>, Ty<'db>), String> {
        let sig = ty.callable_sig(self.infcx.interner).ok_or_else(|| format!("{ty:?} is not callable"))?;
        let sig = Binder::skip_binder(sig);
        let params = sig.inputs().iter().map(|param| self.normalized(*param)).collect();
        Ok((params, self.normalized(sig.output())))
    }

    fn shim_sig(&self, ty: Ty<'db>) -> Result<Sig, String> {
        let (params, ret) = self.callable(ty)?;
        let mut count = 0;
        for param in params {
            count += usize::from(self.layout(param)?.size.bytes() != 0);
        }
        Ok(Sig { params: count, ret: self.layout(ret)?.size.bytes() != 0 })
    }

    pub fn shim(mut self, ty: Ty<'db>) -> Result<(), String> {
        let entry = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry);
        self.builder.switch_to_block(entry);
        let mut values = self.builder.block_params(entry).to_vec().into_iter();
        let (params, ret) = self.callable(ty)?;
        let dest = if self.layout(ret)?.size.bytes() != 0 { values.next().ok_or("shim return pointer")? } else { self.builder.ins().iconst(Type::I64, 16) };
        let mut places = Vec::with_capacity(params.len());
        for param in params {
            let addr = if self.layout(param)?.size.bytes() != 0 { values.next().ok_or("shim argument")? } else { self.builder.ins().iconst(Type::I64, 16) };
            places.push(Place { addr, ty: PlaceTy::from_ty(param), meta: None });
        }
        let dest = Place { addr: dest, ty: PlaceTy::from_ty(ret), meta: None };
        let TyKind::FnDef(def, args) = ty.kind() else {
            return Err(format!("shim of {ty:?}"));
        };
        match resolve_def(self.db, self.env, def.0, args)?.target {
            Target::Ctor(variant) => self.write_variant(dest, variant, &places)?,
            Target::Foreign(name) => self.foreign(&name, dest, &places)?,
            Target::Intrinsic(name, generics) => self.intrinsic(&name, generics, dest, &places)?,
            Target::Dyn(method, self_ty) => self.dyn_item_call(method, self_ty, dest, &places)?,
            _ => return Err("shim of a target with code".to_owned()),
        }
        self.builder.ins().return_(&[]);
        Ok(())
    }

    pub fn thunk(mut self, target: Target<'db>, data: bool) -> Result<(), String> {
        let entry = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry);
        self.builder.switch_to_block(entry);
        let mut values = self.builder.block_params(entry).to_vec().into_iter();
        let (params, ret) = self.target_shape(&target)?;
        let dest = if self.layout(ret)?.size.bytes() != 0 {
            values.next().ok_or("thunk return pointer")?
        } else {
            self.builder.ins().iconst(Type::I64, 16)
        };
        let receiver = if data {
            values.next().ok_or("thunk data pointer")?
        } else {
            self.builder.ins().iconst(Type::I64, 16)
        };
        let mut places = Vec::with_capacity(params.len());
        for param in params {
            let addr = if self.layout(param)?.size.bytes() != 0 {
                values.next().ok_or("thunk argument")?
            } else {
                self.builder.ins().iconst(Type::I64, 16)
            };
            places.push(Place { addr, ty: PlaceTy::from_ty(param), meta: None });
        }
        let dest = Place { addr: dest, ty: PlaceTy::from_ty(ret), meta: None };
        self.direct(&target, dest, Some(receiver), &places)?;
        self.builder.ins().return_(&[]);
        Ok(())
    }
}
