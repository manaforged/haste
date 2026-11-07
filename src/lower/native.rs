use cranelift_codegen::ir::{AbiParam, InstBuilder, Signature, StackSlotData, StackSlotKind, Type, Value, types};
use hir_def::StaticId;
use hir_def::attrs::AttrFlags;
use hir_ty::mir::PlaceTy;
use hir_ty::next_solver::{Ty, TyKind};
use rustc_type_ir::Binder;
use rustc_type_ir::inherent::IntoKind;

use super::{Lower, Place, intrinsic};
use crate::backend::{Key, Sig, Target, key_of, resolve_def};

const THREAD_LOCAL_ATTR: &str = "thread_local";

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct NativeSig {
    pub params: Vec<Type>,
    pub ret: Option<Type>,
}

impl NativeSig {
    pub fn signature(&self, signature: &mut Signature) {
        signature.params.extend(self.params.iter().map(|ty| AbiParam::new(*ty)));
        signature.returns.extend(self.ret.iter().map(|ty| AbiParam::new(*ty)));
    }
}

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn is_native_fn(&self, ty: Ty<'db>) -> bool {
        let Some(sig) = ty.callable_sig(self.infcx.interner) else {
            return false;
        };
        matches!(Binder::skip_binder(sig).abi(), rustc_abi::ExternAbi::C { .. } | rustc_abi::ExternAbi::System { .. })
    }

    pub(super) fn native_symbol(&self, ty: Ty<'db>) -> Result<Option<usize>, String> {
        let TyKind::FnDef(def, args) = ty.kind() else {
            return Ok(None);
        };
        match resolve_def(self.db, self.env, def.0, args)?.target {
            Target::Foreign(name) => intrinsic::symbol(&name).map(Some).ok_or_else(|| format!("no symbol: {name}")),
            _ => Ok(None),
        }
    }

    pub(super) fn native_pointer(&mut self, ty: Ty<'db>) -> Result<Value, String> {
        if let Some(code) = self.native_symbol(ty)? {
            return Ok(self.builder.ins().iconst(types::I64, code as i64));
        }
        let slot = self.native_slot(ty)?;
        let cell = self.builder.ins().iconst(types::I64, self.slots.address(slot));
        Ok(self.load(types::I64, cell, 0))
    }

    pub(super) fn native_slot(&mut self, ty: Ty<'db>) -> Result<usize, String> {
        let TyKind::FnDef(def, args) = ty.kind() else {
            return Err(format!("native pointer to {ty:?}"));
        };
        let target = resolve_def(self.db, self.env, def.0, args)?.target;
        let key = key_of(&target).ok_or("native pointer to a target without code")?;
        let (params, ret) = self.target_shape(&target)?;
        let native = self.native_sig(&params, ret)?;
        let sig = Sig { params: native.params.len(), ret: native.ret.is_some() };
        self.slots.get(Key::Native(Box::new(key), native), sig, self.env.store())
    }

    fn native_sig(&self, params: &[Ty<'db>], ret: Ty<'db>) -> Result<NativeSig, String> {
        let mut types = Vec::with_capacity(params.len());
        for param in params {
            if self.layout(*param)?.size.bytes() != 0 {
                types.push(self.scalar(*param)?.ty);
            }
        }
        let layout = self.layout(ret)?;
        let ret = if layout.size.bytes() == 0 || layout.is_uninhabited() { None } else { Some(self.scalar(ret)?.ty) };
        Ok(NativeSig { params: types, ret })
    }

    pub fn native(mut self, target: Target<'db>, native: &NativeSig) -> Result<(), String> {
        let entry = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry);
        self.builder.switch_to_block(entry);
        let mut values = self.builder.block_params(entry).to_vec().into_iter();
        let (params, ret) = self.target_shape(&target)?;
        let mut places = Vec::with_capacity(params.len());
        for param in params {
            let size = self.layout(param)?.size.bytes();
            let addr = self.frame(size);
            if size != 0 {
                let value = values.next().ok_or("native argument")?;
                self.store(value, addr, 0);
            }
            places.push(Place { addr, ty: PlaceTy::from_ty(param), meta: None });
        }
        let addr = self.frame(self.layout(ret)?.size.bytes());
        let dest = Place { addr, ty: PlaceTy::from_ty(ret), meta: None };
        self.direct(&target, dest, None, &places)?;
        match native.ret {
            Some(ty) => {
                let value = self.load(ty, addr, 0);
                self.builder.ins().return_(&[value]);
            }
            None => {
                self.builder.ins().return_(&[]);
            }
        }
        self.builder.seal_all_blocks();
        Ok(())
    }

    fn frame(&mut self, size: u64) -> Value {
        let data = StackSlotData::new(StackSlotKind::ExplicitSlot, size.max(8) as u32, 4);
        let slot = self.builder.create_sized_stack_slot(data);
        self.builder.ins().stack_addr(types::I64, slot, 0)
    }

    pub(super) fn native_call(&mut self, code: Value, dest: Place<'db>, places: &[Place<'db>]) -> Result<(), String> {
        let mut signature = Signature::new(self.call_conv);
        let mut values = Vec::with_capacity(places.len());
        for place in places {
            if self.layout(place.ty.ty)?.size.bytes() == 0 {
                continue;
            }
            let scalar = self.scalar(place.ty.ty)?;
            signature.params.push(AbiParam::new(scalar.ty));
            values.push(self.load(scalar.ty, place.addr, 0));
        }
        let layout = self.layout(dest.ty.ty)?;
        let ret = if layout.size.bytes() == 0 || layout.is_uninhabited() { None } else { Some(self.scalar(dest.ty.ty)?) };
        if let Some(ret) = ret {
            signature.returns.push(AbiParam::new(ret.ty));
        }
        let signature = self.builder.import_signature(signature);
        let call = self.builder.ins().call_indirect(signature, code, &values);
        if ret.is_some() {
            let result = self.builder.inst_results(call)[0];
            self.store(result, dest.addr, 0);
        }
        Ok(())
    }

    pub(super) fn is_thread_local(&self, id: StaticId) -> bool {
        AttrFlags::has_attr(self.db, id.into(), THREAD_LOCAL_ATTR)
    }

    pub(super) fn thread_local_address(&mut self, id: StaticId, template: usize, size: u64) -> Value {
        let next = self.slots.thread_locals.len();
        let index = *self.slots.thread_locals.entry(id).or_insert(next);
        let mut signature = Signature::new(self.call_conv);
        signature.params.extend([AbiParam::new(types::I64); 3]);
        signature.returns.push(AbiParam::new(types::I64));
        let signature = self.builder.import_signature(signature);
        let code = self.builder.ins().iconst(types::I64, intrinsic::thread_local_code() as i64);
        let index = self.builder.ins().iconst(types::I64, index as i64);
        let template = self.builder.ins().iconst(types::I64, template as i64);
        let size = self.builder.ins().iconst(types::I64, size as i64);
        let call = self.builder.ins().call_indirect(signature, code, &[index, template, size]);
        self.builder.inst_results(call)[0]
    }
}
