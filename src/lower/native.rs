use haste_codegen::{AbiParam, Signature, Type, Value};
use hir_def::StaticId;
use hir_def::attrs::AttrFlags;
use hir_ty::mir::PlaceTy;
use hir_ty::next_solver::{Ty, TyKind};
use rustc_type_ir::Binder;
use rustc_type_ir::inherent::IntoKind;

use super::c_abi::Pass;
use super::{Lower, Place, intrinsic};
use crate::backend::{Key, Sig, Target, key_of, resolve_def};

const THREAD_LOCAL_ATTR: &str = "thread_local";

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct NativeSig {
    pub params: Vec<Pass>,
    pub ret: Option<Pass>,
}

impl NativeSig {
    pub fn signature(&self, signature: &mut Signature) {
        for param in &self.params {
            param.add_param(signature);
        }
        if let Some(ret) = &self.ret {
            ret.add_return(signature);
        }
    }

    pub fn arity(&self) -> usize {
        self.params.len()
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
            return Ok(self.builder.ins().iconst(Type::I64, code as i64));
        }
        let slot = self.native_slot(ty)?;
        let cell = self.builder.ins().iconst(Type::I64, self.slots.address(slot));
        Ok(self.load(Type::I64, cell, 0))
    }

    pub(super) fn native_slot(&mut self, ty: Ty<'db>) -> Result<usize, String> {
        let TyKind::FnDef(def, args) = ty.kind() else {
            return Err(format!("native pointer to {ty:?}"));
        };
        let target = resolve_def(self.db, self.env, def.0, args)?.target;
        let key = key_of(&target).ok_or("native pointer to a target without code")?;
        let (params, ret) = self.target_shape(&target)?;
        let native = self.native_sig(&params, ret)?;
        let sig = Sig::new(native.arity(), native.ret.is_some());
        self.slots.get(Key::Native(Box::new(key), native), sig, self.env.store())
    }

    fn native_sig(&self, params: &[Ty<'db>], ret: Ty<'db>) -> Result<NativeSig, String> {
        let mut passes = Vec::with_capacity(params.len());
        for param in params {
            passes.extend(self.pass_of(*param)?);
        }
        let ret = if self.layout(ret)?.is_uninhabited() { None } else { self.pass_of(ret)? };
        Ok(NativeSig { params: passes, ret })
    }

    fn take_param(&mut self, pass: &Pass, values: &mut impl Iterator<Item = Value>, size: u64) -> Result<Value, String> {
        match pass {
            Pass::Words(words) => {
                let addr = self.frame(size);
                for (_, offset) in words {
                    let value = values.next().ok_or("native argument")?;
                    self.store(value, addr, *offset as i32);
                }
                Ok(addr)
            }
            Pass::Stack(_) | Pass::Pointer(_) => values.next().ok_or_else(|| "native argument".to_owned()),
        }
    }

    pub fn native(mut self, target: Target<'db>, native: &NativeSig) -> Result<(), String> {
        let entry = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry);
        self.builder.switch_to_block(entry);
        let mut values = self.builder.block_params(entry).to_vec().into_iter();
        let sret = match &native.ret {
            Some(ret) if ret.in_memory() => Some(values.next().ok_or("struct return pointer")?),
            _ => None,
        };
        let (params, ret) = self.target_shape(&target)?;
        let mut passes = native.params.iter();
        let mut places = Vec::with_capacity(params.len());
        for param in params {
            let size = self.layout(param)?.size.bytes();
            let addr = match size {
                0 => self.frame(0),
                _ => {
                    let pass = passes.next().ok_or("native parameter shape")?;
                    self.take_param(pass, &mut values, size)?
                }
            };
            places.push(Place { addr, ty: PlaceTy::from_ty(param), meta: None });
        }
        let addr = match sret {
            Some(pointer) => pointer,
            None => self.frame(self.layout(ret)?.size.bytes()),
        };
        let dest = Place { addr, ty: PlaceTy::from_ty(ret), meta: None };
        self.direct(&target, dest, None, &places)?;
        match &native.ret {
            Some(Pass::Words(words)) => {
                let results: Vec<Value> = words.iter().map(|(ty, offset)| self.load(*ty, addr, *offset as i32)).collect();
                self.builder.ins().return_(&results);
            }
            _ => {
                self.builder.ins().return_(&[]);
            }
        }
        Ok(())
    }

    fn frame(&mut self, size: u64) -> Value {
                let slot = self.builder.create_stack_slot(size.max(8) as u32, 4);
        self.builder.ins().stack_addr(Type::I64, slot, 0)
    }

    fn push_arg(&mut self, pass: &Pass, place: &Place<'db>, values: &mut Vec<Value>) -> Result<(), String> {
        match pass {
            Pass::Words(words) => {
                let size = self.layout(place.ty.ty)?.size.bytes();
                let from = match padded(words, size) {
                    Some(end) => {
                        let staged = self.frame(end);
                        self.copy(staged, place.addr, size);
                        staged
                    }
                    None => place.addr,
                };
                values.extend(words.iter().map(|(ty, offset)| self.load(*ty, from, *offset as i32)));
            }
            Pass::Stack(_) => values.push(place.addr),
            Pass::Pointer(size) => {
                let copy = self.frame(u64::from(*size));
                self.copy(copy, place.addr, u64::from(*size));
                values.push(copy);
            }
        }
        Ok(())
    }

    pub(super) fn native_call(&mut self, code: Value, dest: Place<'db>, places: &[Place<'db>]) -> Result<(), String> {
        let mut signature = Signature::new(self.call_conv);
        let mut values = Vec::with_capacity(places.len() + 1);
        let layout = self.layout(dest.ty.ty)?;
        let ret = if layout.is_uninhabited() { None } else { self.pass_of(dest.ty.ty)? };
        if let Some(ret) = &ret {
            ret.add_return(&mut signature);
            if ret.in_memory() {
                values.push(dest.addr);
            }
        }
        for place in places {
            let Some(pass) = self.pass_of(place.ty.ty)? else { continue };
            pass.add_param(&mut signature);
            self.push_arg(&pass, place, &mut values)?;
        }
        let signature = self.builder.import_signature(signature);
        let call = self.builder.ins().call_indirect(signature, code, &values);
        if let Some(Pass::Words(words)) = &ret {
            let results = self.builder.inst_results(call).to_vec();
            let size = layout.size.bytes();
            let staged = padded(words, size).map(|end| self.frame(end));
            for ((_, offset), result) in words.iter().zip(results) {
                self.store(result, staged.unwrap_or(dest.addr), *offset as i32);
            }
            if let Some(staged) = staged {
                self.copy(dest.addr, staged, size);
            }
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
        signature.params.extend([AbiParam::new(Type::I64); 3]);
        signature.returns.push(AbiParam::new(Type::I64));
        let signature = self.builder.import_signature(signature);
        let code = self.builder.ins().iconst(Type::I64, intrinsic::thread_local_code() as i64);
        let index = self.builder.ins().iconst(Type::I64, index as i64);
        let template = self.builder.ins().iconst(Type::I64, template as i64);
        let size = self.builder.ins().iconst(Type::I64, size as i64);
        let call = self.builder.ins().call_indirect(signature, code, &[index, template, size]);
        self.builder.inst_results(call)[0]
    }
}

fn padded(words: &[(Type, u32)], size: u64) -> Option<u64> {
    let end = words.iter().map(|(ty, at)| u64::from(*at) + u64::from(ty.bytes())).max()?;
    (end > size).then_some(end)
}
