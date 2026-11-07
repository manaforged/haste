use hir_def::attrs::AttrFlags;
use std::collections::HashMap;

use haste_codegen::Type;
use hir_def::expr_store::Body;
use hir_def::signatures::{StaticFlags, StaticSignature};
use hir_def::{AdtId, DefWithBodyId, StaticId};
use hir_ty::InferenceResult;
use hir_ty::MemoryMap;
use hir_ty::db::GeneralConstId;
use hir_ty::layout::RustcEnumVariantIdx;
use hir_ty::method_resolution::lookup_impl_const;
use hir_ty::mir::PlaceTy;
use hir_ty::next_solver::{Allocation, Const, ConstKind, GenericArgs, Region, Ty, TyKind, ValTree, ValueConst};
use rustc_abi::Size;
use rustc_type_ir::inherent::IntoKind;
use rustc_abi::{TagEncoding, Variants};

use super::{Lower, Place, intrinsic};
use crate::backend::{Key, Sig};

const EVALUATOR_ADDRESSES: usize = 1 << 59;

fn leak(bytes: &[u8]) -> usize {
    let words = bytes.len().div_ceil(8).max(1);
    let memory = Box::leak(vec![0_u64; words].into_boxed_slice());
    // SAFETY: the u64 buffer covers at least bytes.len() bytes and u8 has no alignment or validity needs.
    let target = unsafe { std::slice::from_raw_parts_mut(memory.as_mut_ptr().cast::<u8>(), words * 8) };
    target[..bytes.len()].copy_from_slice(bytes);
    memory.as_ptr() as usize
}

fn read_word(at: usize) -> usize {
    // SAFETY: at points into leaked constant memory that covers the value being walked.
    unsafe { (at as *const usize).read_unaligned() }
}

fn write_word(at: usize, value: usize) {
    // SAFETY: at points into leaked constant memory that covers the value being walked.
    unsafe { (at as *mut usize).write_unaligned(value) };
}

fn whole_allocation<'m>(map: &'m MemoryMap<'_>, addr: usize, known: &'m [u8]) -> &'m [u8] {
    let mut best = known;
    let mut step = known.len().max(1);
    while let Some(longer) = map.get(addr, best.len() + step).filter(|bytes| bytes.len() == best.len() + step) {
        best = longer;
        step *= 2;
    }
    while step > 0 {
        if let Some(longer) = map.get(addr, best.len() + step).filter(|bytes| bytes.len() == best.len() + step) {
            best = longer;
        }
        step /= 2;
    }
    best
}

struct Reloc<'m, 'db> {
    map: &'m MemoryMap<'db>,
    seen: HashMap<usize, usize>,
}


impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn allocation(&mut self, allocation: Allocation<'db>) -> Result<Place<'db>, String> {
        let data = &*allocation;
        let at = self.load_constant(data.ty, &data.memory, &data.memory_map)?;
        let addr = self.builder.ins().iconst(Type::I64, at as i64);
        Ok(Place { addr, ty: PlaceTy::from_ty(data.ty), meta: None })
    }

    fn load_constant(&mut self, ty: Ty<'db>, bytes: &[u8], map: &MemoryMap<'db>) -> Result<usize, String> {
        let at = leak(bytes);
        let mut reloc = Reloc { map, seen: HashMap::new() };
        self.relocate(&mut reloc, ty, at)?;
        Ok(at)
    }

    pub(super) fn constant(&mut self, konst: Const<'db>, ty: Ty<'db>) -> Result<Place<'db>, String> {
        if self.layout(ty)?.size.bytes() == 0 {
            let addr = self.builder.ins().iconst(Type::I64, 16);
            return Ok(Place { addr, ty: PlaceTy::from_ty(ty), meta: None });
        }
        let unevaluated = match konst.kind() {
            ConstKind::Unevaluated(unevaluated) => unevaluated,
            ConstKind::Value(value) => return self.value_constant(value),
            other => return Err(format!("constant {other:?}")),
        };
        let failed = |error| format!("const eval: {error:?}");
        let allocation = match unevaluated.def.0 {
            GeneralConstId::ConstId(id) => {
                let (id, args) = lookup_impl_const(&self.infcx, self.env.param_env, id, unevaluated.args);
                let evaluated = self.db.const_eval(id, args, Some(self.env)).map_err(failed);
                let loaded = evaluated.and_then(|allocation| self.allocation(allocation));
                return loaded.or_else(|_| self.run_once(id, args, ty));
            }
            GeneralConstId::StaticId(id) => {
                let at = self.static_address(id)?;
                let addr = self.builder.ins().iconst(Type::I64, at as i64);
                return Ok(Place { addr, ty: PlaceTy::from_ty(ty), meta: None });
            }
            GeneralConstId::AnonConstId(id) => {
                self.db.anon_const_eval(id, unevaluated.args, Some(self.env)).map_err(failed)?
            }
        };
        self.allocation(allocation)
    }

    fn run_once(&mut self, id: hir_def::ConstId, args: GenericArgs<'db>, ty: Ty<'db>) -> Result<Place<'db>, String> {
        let layout = self.layout(ty)?;
        let buffer = leak(&vec![0_u8; layout.size.bytes() as usize + 8]);
        let cell = leak(&[0_u8; 8]);
        let slot = self.slots.get(Key::Const(id, args.store()), Sig { params: 0, ret: true }, self.env.store())?;
        let addr = self.builder.ins().iconst(Type::I64, buffer as i64);
        let done_at = self.builder.ins().iconst(Type::I64, cell as i64);
        let done = self.load(Type::I64, done_at, 0);
        let (run, next) = (self.builder.create_block(), self.builder.create_block());
        self.builder.ins().brif(done, next, &[], run, &[]);
        self.builder.switch_to_block(run);
        let code = self.builder.ins().iconst(Type::I64, self.slots.address(slot));
        self.call_cell(code, Sig { params: 0, ret: true }, &[addr]);
        let one = self.builder.ins().iconst(Type::I64, 1);
        self.store(one, done_at, 0);
        self.builder.ins().jump(next, &[]);
        self.builder.switch_to_block(next);
        Ok(Place { addr, ty: PlaceTy::from_ty(ty), meta: None })
    }

    fn value_constant(&mut self, value: ValueConst<'db>) -> Result<Place<'db>, String> {
        let bytes = self.value_bytes(value.ty, value.value)?;
        let addr = self.builder.ins().iconst(Type::I64, leak(&bytes) as i64);
        Ok(Place { addr, ty: PlaceTy::from_ty(value.ty), meta: None })
    }

    fn value_bytes(&self, ty: Ty<'db>, value: ValTree<'db>) -> Result<Vec<u8>, String> {
        match ty.kind() {
            TyKind::Bool | TyKind::Char | TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_) | TyKind::RawPtr(..) => {
                let size = self.layout(ty)?.size;
                let raw = value.inner().to_leaf().to_uint(Size::from_bytes(size.bytes()));
                Ok(raw.to_le_bytes()[..size.bytes() as usize].to_vec())
            }
            TyKind::Ref(_, inner, _) if matches!(inner.kind(), TyKind::Str) => {
                let mut text = Vec::new();
                for item in value.inner().to_branch().iter() {
                    let ConstKind::Value(item) = item.kind() else {
                        return Err("string constant item".to_owned());
                    };
                    text.push(item.value.inner().to_leaf().to_u8());
                }
                let mut bytes = leak(&text).to_le_bytes().to_vec();
                bytes.extend(text.len().to_le_bytes());
                Ok(bytes)
            }
            other => Err(format!("value constant of {other:?}")),
        }
    }

    pub(super) fn static_cell(&mut self, id: StaticId) -> Result<Place<'db>, String> {
        let at = self.static_address(id)?;
        if let Some((slot, done)) = self.slots.lazy_statics.get(&id).copied() {
            self.init_once(slot, at, done);
        }
        let region = Region::new_erased(self.infcx.interner);
        let ty = Ty::new_imm_ref(self.infcx.interner, region, self.static_type(id));
        if self.is_thread_local(id) {
            let size = self.layout(self.static_type(id))?.size.bytes();
            let address = self.thread_local_address(id, at, size);
                        let slot = self.builder.create_stack_slot(8, 3);
            let addr = self.builder.ins().stack_addr(Type::I64, slot, 0);
            self.store(address, addr, 0);
            return Ok(Place { addr, ty: PlaceTy::from_ty(ty), meta: None });
        }
        let cell = leak(&at.to_le_bytes());
        let addr = self.builder.ins().iconst(Type::I64, cell as i64);
        Ok(Place { addr, ty: PlaceTy::from_ty(ty), meta: None })
    }

    fn static_type(&self, id: StaticId) -> Ty<'db> {
        InferenceResult::of(self.db, DefWithBodyId::from(id)).expr_ty(Body::of(self.db, id.into()).root_expr())
    }

    fn static_address(&mut self, id: StaticId) -> Result<usize, String> {
        if let Some(at) = self.slots.statics.get(&id) {
            return Ok(*at);
        }
        let signature = StaticSignature::of(self.db, id);
        let at = if signature.flags.contains(StaticFlags::EXTERN) {
            let named = AttrFlags::link_name(self.db, id.into());
            let symbol = intrinsic::symbol(named.as_ref().map_or(signature.name.as_str(), |symbol| symbol.as_str()));
            if AttrFlags::has_attr(self.db, id.into(), "linkage") {
                leak(&symbol.unwrap_or(0).to_le_bytes())
            } else {
                symbol.ok_or_else(|| format!("no extern static {}", signature.name.as_str()))?
            }
        } else {
            match self.db.const_eval_static(id) {
                Ok(allocation) => {
                    let data = &*allocation;
                    self.load_constant(data.ty, &data.memory, &data.memory_map)?
                }
                Err(_) => self.lazy_static(id)?,
            }
        };
        self.slots.statics.insert(id, at);
        Ok(at)
    }

    fn lazy_static(&mut self, id: StaticId) -> Result<usize, String> {
        let layout = self.layout(self.static_type(id))?;
        let buffer = leak(&vec![0_u8; layout.size.bytes() as usize + 8]);
        let sig = Sig { params: 0, ret: layout.size.bytes() != 0 };
        let slot = self.slots.get(Key::Static(id), sig, self.env.store())?;
        let done = leak(&[0_u8; 8]);
        self.slots.lazy_statics.insert(id, (slot, done as usize));
        Ok(buffer as usize)
    }

    fn init_once(&mut self, slot: usize, at: usize, done: usize) {
        let sig = self.slots.sig_of(slot);
        let flag = self.builder.ins().iconst(Type::I64, done as i64);
        let seen = self.load(Type::I64, flag, 0);
        let (run, next) = (self.builder.create_block(), self.builder.create_block());
        self.builder.ins().brif(seen, next, &[], run, &[]);
        self.builder.switch_to_block(run);
        let one = self.builder.ins().iconst(Type::I64, 1);
        self.store(one, flag, 0);
        let code = self.builder.ins().iconst(Type::I64, self.slots.address(slot));
        let addr = self.builder.ins().iconst(Type::I64, at as i64);
        if sig.ret {
            self.call_cell(code, sig, &[addr]);
        } else {
            self.call_cell(code, sig, &[]);
        }
        self.builder.ins().jump(next, &[]);
        self.builder.switch_to_block(next);
    }

    fn relocate(&mut self, reloc: &mut Reloc<'_, 'db>, ty: Ty<'db>, at: usize) -> Result<(), String> {
        match ty.kind() {
            TyKind::Ref(_, inner, _) => {
                self.relocate_pointer(reloc, inner, at)?;
                if read_word(at) == 0 {
                    write_word(at, self.layout(inner)?.align.abi.bytes() as usize);
                }
                Ok(())
            }
            TyKind::RawPtr(inner, _) => self.relocate_pointer(reloc, inner, at),
            TyKind::FnPtr(..) => {
                let id = read_word(at);
                let target = reloc.map.vtable_ty(id).map_err(|error| format!("fn pointer constant: {error:?}"))?;
                if self.is_native_fn(target) && self.native_symbol(target)?.is_none() {
                    let slot = self.native_slot(target)?;
                    self.slots.native_words.push((at, slot));
                } else {
                    write_word(at, self.reify(target)? as usize);
                }
                Ok(())
            }
            TyKind::Tuple(items) => {
                let layout = self.layout(ty)?;
                for (index, item) in items.as_slice().iter().enumerate() {
                    self.relocate(reloc, *item, at + layout.fields.offset(index).bytes() as usize)?;
                }
                Ok(())
            }
            TyKind::Array(item, _) => {
                let (count, size) = (self.layout(ty)?.fields.count(), self.layout(item)?.size.bytes() as usize);
                for index in 0..count {
                    self.relocate(reloc, item, at + index * size)?;
                }
                Ok(())
            }
            TyKind::Adt(adt, args) => self.relocate_adt(reloc, ty, (adt.def_id(), args), at),
            TyKind::Pat(inner, _) => self.relocate(reloc, inner, at),
            _ => Ok(()),
        }
    }

    fn unsized_bytes(&self, ty: Ty<'db>, meta: usize) -> Result<usize, String> {
        match ty.kind() {
            TyKind::Str => Ok(meta),
            TyKind::Slice(item) => Ok(meta * self.layout(item)?.size.bytes() as usize),
            TyKind::Adt(adt, args) => {
                let AdtId::StructId(id) = adt.def_id() else {
                    return Err(format!("constant pointer to {ty:?}"));
                };
                let fields = self.fields_of(id.into(), args);
                let last = *fields.last().ok_or("empty unsized struct")?;
                let offset = self.layout(ty)?.fields.offset(fields.len() - 1).bytes() as usize;
                Ok(offset + self.unsized_bytes(last, meta)?)
            }
            _ => Err(format!("constant pointer to {ty:?}")),
        }
    }

    fn relocate_pointer(&mut self, reloc: &mut Reloc<'_, 'db>, inner: Ty<'db>, at: usize) -> Result<(), String> {
        let old = read_word(at);
        let meta = read_word(at + 8);
        let unsized_inner = self.layout(inner)?.is_unsized();
        let (size, item) = match inner.kind() {
            TyKind::Str if unsized_inner => (meta, None),
            TyKind::Slice(item) => (meta * self.layout(item)?.size.bytes() as usize, Some((item, meta))),
            TyKind::Dynamic(..) => {
                let concrete = reloc.map.vtable_ty(meta).map_err(|error| format!("dyn constant: {error:?}"))?;
                write_word(at + 8, self.vtable(concrete, inner)? as usize);
                (self.layout(concrete)?.size.bytes() as usize, None)
            }
            _ if unsized_inner => (self.unsized_bytes(inner, meta)?, None),
            _ => (self.layout(inner)?.size.bytes() as usize, None),
        };
        if size == 0 {
            return Ok(());
        }
        if let Some(moved) = reloc.seen.get(&old) {
            write_word(at, *moved);
            return Ok(());
        }
        let Some(bytes) = reloc.map.get(old, size).filter(|bytes| bytes.len() == size) else {
            if old >= EVALUATOR_ADDRESSES {
                return Err(format!("constant holds a pointer the evaluator kept no bytes for: {inner:?}"));
            }
            return Ok(());
        };
        let bytes = whole_allocation(reloc.map, old, bytes);
        let moved = leak(bytes);
        reloc.seen.insert(old, moved);
        write_word(at, moved);
        match item {
            Some((item, count)) => {
                let width = self.layout(item)?.size.bytes() as usize;
                for index in 0..count {
                    self.relocate(reloc, item, moved + index * width)?;
                }
                Ok(())
            }
            None if !unsized_inner => self.relocate(reloc, inner, moved),
            None => Ok(()),
        }
    }

    fn relocate_adt(
        &mut self,
        reloc: &mut Reloc<'_, 'db>,
        ty: Ty<'db>,
        (adt, args): (AdtId, hir_ty::next_solver::GenericArgs<'db>),
        at: usize,
    ) -> Result<(), String> {
        let layout = self.layout(ty)?;
        let (variant, shape): (hir_def::VariantId, _) = match adt {
            AdtId::StructId(id) => (id.into(), &*layout),
            AdtId::UnionId(id) => {
                for field in self.fields_of(id.into(), args) {
                    let _unknown_active_field = self.relocate(reloc, field, at);
                }
                return Ok(());
            }
            AdtId::EnumId(id) => {
                let variants = &id.enum_variants(self.db).variants;
                let index = match &layout.variants {
                    Variants::Empty => return Ok(()),
                    Variants::Single { index } => index.0,
                    Variants::Multiple { .. } => self.variant_at(&layout, id, at)?,
                };
                let shape = match &layout.variants {
                    Variants::Multiple { variants, .. } => &variants[RustcEnumVariantIdx(index)],
                    _ => &*layout,
                };
                (variants[index].0.into(), shape)
            }
        };
        for (index, field) in self.fields_of(variant, args).into_iter().enumerate() {
            self.relocate(reloc, field, at + shape.fields.offset(index).bytes() as usize)?;
        }
        Ok(())
    }

    fn variant_at(&self, layout: &hir_ty::layout::Layout, id: hir_def::EnumId, at: usize) -> Result<usize, String> {
        let Variants::Multiple { tag, tag_encoding, variants, .. } = &layout.variants else {
            return Err("variant of a single-variant enum".to_owned());
        };
        let bytes = super::variant::tag_bytes(tag)? as usize;
        let offset = layout.fields.offset(0).bytes() as usize;
        let mut raw = [0_u8; 16];
        // SAFETY: the enum value occupies this tag range inside the leaked constant memory.
        unsafe { std::ptr::copy_nonoverlapping((at + offset) as *const u8, raw.as_mut_ptr(), bytes) };
        let raw = u128::from_le_bytes(raw);
        let mask = if bytes == 16 { u128::MAX } else { (1 << (bytes * 8)) - 1 };
        match tag_encoding {
            TagEncoding::Direct => {
                let found = id.enum_variants(self.db).variants.values().position(|variant| {
                    self.db.const_eval_discriminant(variant.0).is_ok_and(|value| value as u128 & mask == raw)
                });
                found.ok_or_else(|| "constant enum tag".to_owned())
            }
            TagEncoding::Niche { untagged_variant, niche_start, .. } => {
                let relative = raw.wrapping_sub(*niche_start) & mask;
                let others: Vec<usize> = variants
                    .iter_enumerated()
                    .map(|(it, _)| it)
                    .filter(|it| it != untagged_variant)
                    .map(|it| it.0)
                    .collect();
                Ok(others.get(relative as usize).copied().unwrap_or(untagged_variant.0))
            }
        }
    }
}
