use haste_codegen::{Block, IntCC, StackSlot, Type, Value};
use either::Either;
use hir_def::{AdtId, ItemContainerId, VariantId};
use hir_def::signatures::{StructFlags, StructSignature};
use hir_ty::drop::{DropGlue, has_drop_glue};
use hir_ty::mir::{LocalId, PlaceTy, ProjectionElem, StoredPlace, TerminatorKind};
use hir_ty::next_solver::{GenericArgs, Ty, TyKind};
use rustc_type_ir::IntTy;
use rustc_type_ir::inherent::{GenericArgs as _, IntoKind};

use super::moves::{Cut, Step};
use super::{Lower, Place};
use crate::backend::{Key, Sig};

const GLUE: Sig = Sig::new(1, false);

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn needs_drop(&self, ty: Ty<'db>) -> bool {
        !matches!(has_drop_glue(&self.infcx, ty, self.env.param_env), DropGlue::None)
    }

    pub(super) fn normalized(&self, ty: Ty<'db>) -> Ty<'db> {
        hir_ty::next_solver::normalize::deeply_normalize(
            self.infcx.at(&hir_ty::next_solver::infer::traits::ObligationCause::dummy(), self.env.param_env),
            ty,
        )
        .unwrap_or(ty)
    }

    pub(super) fn glue_slot(&mut self, ty: Ty<'db>) -> Result<usize, String> {
        self.slots.get(Key::Glue(ty.store()), GLUE, self.env.store())
    }

    pub(super) fn make_flags(&mut self) -> Result<(), String> {
        let body = self.body;
        for (_, block) in body.basic_blocks.iter() {
            let Some(terminator) = &block.terminator else {
                continue;
            };
            let TerminatorKind::Drop { place, .. } = &terminator.kind else {
                continue;
            };
            let local = place.as_ref().local;
            if self.flags.contains_key(&local) || !self.needs_drop(body.locals[local].ty.as_ref()) {
                continue;
            }
                        let slot = self.builder.create_stack_slot(1, 0);
            let zero = self.builder.ins().iconst(Type::I8, 0);
            self.builder.ins().stack_store(zero, slot, 0);
            self.flags.insert(local, slot);
        }
        self.make_move_paths();
        Ok(())
    }

    pub(super) fn set_flag(&mut self, local: LocalId, on: bool) {
        if let Some(slot) = self.flags.get(&local).copied() {
            let value = self.builder.ins().iconst(Type::I8, i64::from(on));
            self.builder.ins().stack_store(value, slot, 0);
        }
    }

    pub(super) fn initialized(&mut self, target: &StoredPlace) {
        let local = target.as_ref().local;
        let Some(steps) = self.owned_paths(target).and_then(|paths| paths.into_iter().next()) else {
            return;
        };
        if steps.is_empty() {
            self.set_flag(local, true);
        }
        self.clear_moved(local, &steps);
    }

    pub(super) fn is_box(&self, ty: Ty<'db>) -> bool {
        let TyKind::Adt(adt, _) = ty.kind() else {
            return false;
        };
        AdtId::StructId(self.infcx.interner.lang_items().OwnedBox.unwrap_or_else(|| unreachable!())) == adt.def_id()
    }

    fn if_flag(&mut self, slot: StackSlot) -> Block {
        let value = self.builder.ins().stack_load(Type::I8, slot, 0);
        let run = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.ins().brif(value, run, &[], done, &[]);
        self.builder.switch_to_block(run);
        done
    }

    pub(super) fn drop_local(&mut self, place: &StoredPlace) -> Result<(), String> {
        let found = self.place(place)?;
        self.drop_flagged(place.as_ref().local, found)
    }

    pub(super) fn drop_flagged(&mut self, local: LocalId, found: Place<'db>) -> Result<(), String> {
        if !self.needs_drop(found.ty.ty) {
            return Ok(());
        }
        let Some(slot) = self.flags.get(&local).copied() else {
            return Ok(());
        };
        let done = self.if_flag(slot);
        self.set_flag(local, false);
        let paths = self.move_paths.get(&local).cloned().unwrap_or_default();
        self.drop_open(found.ty.ty, found.addr, found.meta, Cut::whole(&paths))?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(())
    }

    pub(super) fn replace_old(&mut self, target: &StoredPlace, dest: Place<'db>) -> Result<(), String> {
        if !self.needs_drop(dest.ty.ty) {
            return Ok(());
        }
        let stored = target.as_ref();
        let Some(steps) = self.owned_paths(target).and_then(|paths| paths.into_iter().next()) else {
            return self.call_glue(dest.ty.ty, dest.addr, dest.meta);
        };
        let Some(slot) = self.flags.get(&stored.local).copied() else {
            if steps.contains(&Step::Box) {
                return self.call_glue(dest.ty.ty, dest.addr, dest.meta);
            }
            return Ok(());
        };
        let paths = self.paths_under(stored.local, &steps);
        let done = self.if_flag(slot);
        if steps.is_empty() {
            self.set_flag(stored.local, false);
        }
        self.drop_open(dest.ty.ty, dest.addr, dest.meta, Cut::at(&paths, steps.len()))?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(())
    }

    pub(super) fn thin_holder(&mut self, data: Value) -> Value {
        let slot = self.builder.create_stack_slot(16, 3);
        self.builder.ins().stack_store(data, slot, 0);
        self.builder.ins().stack_addr(Type::I64, slot, 0)
    }

    pub(super) fn call_glue(&mut self, ty: Ty<'db>, addr: Value, meta: Option<Value>) -> Result<(), String> {
        if !self.needs_drop(ty) {
            return Ok(());
        }
        let holder = self.thin_holder(addr);
        if let Some(meta) = meta {
            self.store(meta, holder, 8);
        }
        self.call_glue_holder(ty, holder)
    }

    fn call_glue_holder(&mut self, ty: Ty<'db>, holder: Value) -> Result<(), String> {
        let slot = self.glue_slot(ty)?;
        let cell = self.builder.ins().iconst(Type::I64, self.slots.address(slot));
        self.call_cell(cell, GLUE, &[holder]);
        Ok(())
    }

    pub fn glue(mut self, ty: Ty<'db>) -> Result<(), String> {
        let entry = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry);
        self.builder.switch_to_block(entry);
        let holder = self.builder.block_params(entry)[0];
        let ty = self.normalized(ty);
        let addr = self.load(Type::I64, holder, 0);
        let meta = if self.layout(ty)?.is_unsized() { Some(self.load(Type::I64, holder, 8)) } else { None };
        if self.is_box(ty)
            && let TyKind::Adt(_, args) = ty.kind()
            && let Some(inner) = args.as_slice().first().and_then(|arg| arg.ty())
            && self.needs_drop(inner)
        {
            self.call_glue_holder(inner, addr)?;
        }
        self.drop_impl(ty, holder)?;
        self.drop_parts(ty, addr, meta, Cut::none())?;
        self.builder.ins().return_(&[]);
        Ok(())
    }

    pub(super) fn drop_impl(&mut self, ty: Ty<'db>, holder: Value) -> Result<(), String> {
        let Some(drop_fn) = self.infcx.interner.lang_items().Drop_drop else {
            return Ok(());
        };
        if !matches!(ty.kind(), TyKind::Adt(..)) {
            return Ok(());
        }
        let args = GenericArgs::new_from_slice(&[ty.into()]);
        let (imp, args) = self.db.lookup_impl_method(self.env, drop_fn, args);
        let Either::Left(imp) = imp else {
            return Ok(());
        };
        if !matches!(imp.loc(self.db).container, ItemContainerId::ImplId(_)) {
            return Ok(());
        }
        let slot = self.slots.get(Key::Func(imp, args.store()), GLUE, self.env.store())?;
        let cell = self.builder.ins().iconst(Type::I64, self.slots.address(slot));
        self.call_cell(cell, GLUE, &[holder]);
        Ok(())
    }

    pub(super) fn drop_parts(
        &mut self,
        ty: Ty<'db>,
        addr: Value,
        meta: Option<Value>,
        cut: Cut<'_>,
    ) -> Result<(), String> {
        match ty.kind() {
            TyKind::Adt(adt, args) => match adt.def_id() {
                AdtId::StructId(id) => {
                    if StructSignature::of(self.db, id).flags.contains(StructFlags::IS_MANUALLY_DROP) {
                        return Ok(());
                    }
                    let layout = self.layout(ty)?;
                    let offsets = |index: usize| layout.fields.offset(index).bytes();
                    let fields = self.fields_of(id.into(), args);
                    self.drop_fields(&fields, &offsets, addr, meta, cut)
                }
                AdtId::UnionId(_) => Ok(()),
                AdtId::EnumId(id) => self.drop_enum(ty, id, args, addr, cut),
            },
            TyKind::Tuple(items) => {
                let layout = self.layout(ty)?;
                let offsets = |index: usize| layout.fields.offset(index).bytes();
                self.drop_fields(items.as_slice(), &offsets, addr, meta, cut)
            }
            TyKind::Closure(_, args) => self.drop_parts(args.as_closure().tupled_upvars_ty(), addr, meta, cut),
            TyKind::Array(item, _) => {
                let count = self.layout(ty)?.fields.count();
                self.drop_open_items(item, addr, count as u64, cut)
            }
            TyKind::Slice(item) => self.drop_items(item, addr, meta.ok_or("slice without a length")?),
            TyKind::Dynamic(..) => {
                self.drop_dyn(addr, meta.ok_or("dyn without a vtable")?);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn drop_fields(
        &mut self,
        fields: &[Ty<'db>],
        offsets: &dyn Fn(usize) -> u64,
        addr: Value,
        meta: Option<Value>,
        cut: Cut<'_>,
    ) -> Result<(), String> {
        for (index, field) in fields.iter().enumerate() {
            if !self.needs_drop(*field) {
                continue;
            }
            let unsized_field = self.layout(*field)?.is_unsized();
            let at = self.field_address(addr, offsets(index), *field, meta.filter(|_| unsized_field))?;
            let paths = cut.field(index as u32);
            self.drop_open(*field, at, meta.filter(|_| unsized_field), cut.into(&paths))?;
        }
        Ok(())
    }

    pub(super) fn drop_items(&mut self, item: Ty<'db>, addr: Value, count: Value) -> Result<(), String> {
        if !self.needs_drop(item) {
            return Ok(());
        }
        let stride = self.layout(item)?.size.bytes() as i64;
        self.for_each_item(addr, count, stride, |lower, at| lower.call_glue(item, at, None))
    }

    pub(super) fn for_each_item(
        &mut self,
        addr: Value,
        count: Value,
        stride: i64,
        mut work: impl FnMut(&mut Self, Value) -> Result<(), String>,
    ) -> Result<(), String> {
        let head = self.builder.create_block();
        let index = self.builder.append_block_param(head, Type::I64);
        let run = self.builder.create_block();
        let done = self.builder.create_block();
        let zero = self.builder.ins().iconst(Type::I64, 0);
        self.builder.ins().jump(head, &[zero.into()]);
        self.builder.switch_to_block(head);
        let more = self.builder.ins().icmp(IntCC::UnsignedLessThan, index, count);
        self.builder.ins().brif(more, run, &[], done, &[]);
        self.builder.switch_to_block(run);
        let delta = self.builder.ins().imul_imm(index, stride);
        let at = self.builder.ins().iadd(addr, delta);
        work(self, at)?;
        let next = self.builder.ins().iadd_imm(index, 1);
        self.builder.ins().jump(head, &[next.into()]);
        self.builder.switch_to_block(done);
        Ok(())
    }

    fn drop_enum(
        &mut self,
        ty: Ty<'db>,
        id: hir_def::EnumId,
        args: GenericArgs<'db>,
        addr: Value,
        cut: Cut<'_>,
    ) -> Result<(), String> {
        let layout = self.layout(ty)?;
        let wide = Ty::new_int(self.infcx.interner, IntTy::Isize);
        let slot = self.builder.create_stack_slot(8, 3);
        let tag_at = self.builder.ins().stack_addr(Type::I64, slot, 0);
        self.discriminant(
            Place { addr: tag_at, ty: PlaceTy::from_ty(wide), meta: None },
            Place { addr, ty: PlaceTy::from_ty(ty), meta: None },
        )?;
        let tag = self.load(Type::I64, tag_at, 0);
        let done = self.builder.create_block();
        let variants: Vec<_> = id.enum_variants(self.db).variants.values().map(|variant| variant.0).collect();
        for variant in variants {
            let fields = self.fields_of(VariantId::EnumVariantId(variant), args);
            if !fields.iter().any(|field| self.needs_drop(*field)) {
                continue;
            }
            let value = self.db.const_eval_discriminant(variant).map_err(|error| format!("discriminant: {error:?}"))?;
            let hit = self.builder.ins().icmp_imm_u(IntCC::Equal, tag, value as i64);
            let run = self.builder.create_block();
            let next = self.builder.create_block();
            self.builder.ins().brif(hit, run, &[], next, &[]);
            self.builder.switch_to_block(run);
            let shape = self.variant_fields(&layout, Some(VariantId::EnumVariantId(variant)))?;
            let offsets = |index: usize| shape.fields.offset(index).bytes();
            self.drop_fields(&fields, &offsets, addr, None, cut.variant(variant))?;
            self.builder.ins().jump(done, &[]);
            self.builder.switch_to_block(next);
        }
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(())
    }
}
