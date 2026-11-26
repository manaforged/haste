use anyhow::{Result, bail};
use haste_codegen::{StackSlot, Type, UnwindArg, UnwindDrop, Value};

use crate::body::ExprId;
use crate::host::HostSymbol;
use crate::sema::infer::{has_drop_impl, needs_drop};
use crate::sema::reach::unwind_enabled;
use crate::sema::ty::Ty;

use super::super::moves::{BOX_STEP, MovePath, Root};
use super::super::place::{Base, Place};
use super::super::reloc::{RelocStep, SiteOrigin, TyOrigin};
use super::super::stencil::{LowerMode, SymbolSource};
use super::super::{Lower, Repr};
use super::{DROP_LIST_ACTIVE_BIT, DROP_LIST_CLEAR_MASK, Life, Owned, Part};

impl Lower<'_, '_> {
    /// Whether runtime checks and drops should emit real panic/unwind paths rather than hardware traps.
    pub(in crate::lower) fn should_unwind(&self) -> bool {
        self.overflow_checks || unwind_enabled(self.cx, self.reached)
    }

    /// Whether constructing fields should register per-field unwind drop guards while evaluating later fields.
    pub(in crate::lower) fn wants_field_unwind(&mut self) -> Result<bool> {
        Ok(self.open && self.resume.is_none() && self.should_unwind() && self.body_may_unwind()?)
    }

    pub(in crate::lower) fn push_owned_guard(
        &mut self,
        ty: Ty,
        place: Option<Place>,
        expr: ExprId,
    ) -> Result<Option<usize>> {
        if !self.open || !needs_drop(self.cx, ty)? {
            return Ok(None);
        }
        let want_unwind =
            self.resume.is_none() && self.should_unwind() && self.body_may_unwind()?;
        let flag = if want_unwind {
            self.push_unwind_origin(TyOrigin::Expr(expr), ty, place, true)?
        } else {
            self.flag(true)
        };
        let idx = self.owned.len();
        self.owned.push(Owned {
            root: Root::Held(expr),
            ty,
            place,
            flag,
            unwind: want_unwind,
            parts: Vec::new(),
            life: Life::Statement,
        });
        Ok(Some(idx))
    }

    pub(in crate::lower) fn disarm_owned_guards(&mut self, guards: Vec<usize>) {
        if !self.open {
            return;
        }
        for idx in guards.into_iter().rev() {
            let owned = self.owned.remove(idx);
            self.disarm_unwind(owned.flag, owned.unwind);
        }
    }

    pub(in crate::lower) fn eval_and_write_part(
        &mut self,
        part: Option<Place>,
        elem_ty: Ty,
        expr: ExprId,
        unwind: bool,
        guards: &mut Vec<usize>,
    ) -> Result<bool> {
        let value = self.eval_operand(expr)?;
        if !self.open {
            return Ok(false);
        }
        if let (Some(place), Some(value)) = (part, value) {
            self.write(place, value);
        }
        if unwind && let Some(guard) = self.push_owned_guard(elem_ty, part, expr)? {
            guards.push(guard);
        }
        Ok(true)
    }

    pub(super) fn expand_unwind_parts(
        &mut self,
        root: Root,
        ty: Ty,
        place: Option<Place>,
        paths: &[MovePath],
        init: bool,
    ) -> Result<Vec<Part>> {
        let Some(place) = place else {
            return Ok(self.plain_parts(paths, init));
        };
        let mut parts = Vec::new();
        self.push_unwind_parts(root, &[], ty, place, paths, init, &mut parts)?;
        Ok(parts)
    }

    fn push_unwind_parts(
        &mut self,
        root: Root,
        path: &[(u32, u32)],
        ty: Ty,
        place: Place,
        paths: &[MovePath],
        init: bool,
        out: &mut Vec<Part>,
    ) -> Result<()> {
        if !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        let below = paths
            .iter()
            .any(|part| part.len() > path.len() && part.starts_with(path));
        let inside: MovePath = path.iter().copied().chain([BOX_STEP]).collect();
        let into_box = paths.iter().any(|part| part.starts_with(&inside));
        if !into_box && (!below || has_drop_impl(self.cx, ty)?) {
            let origin = TyOrigin::from_root(root, self.typed).stepped(path);
            let slot = self.push_unwind_origin(origin, ty, Some(place), init)?;
            out.push(Part {
                path: path.to_vec(),
                flag: slot,
                unwind: true,
            });
            return Ok(());
        }
        if !into_box && let Some(fields) = self.single_variant_fields(ty)? {
            for (index, field_ty) in fields.into_iter().enumerate().rev() {
                let mut sub = path.to_vec();
                sub.push((0, index as u32));
                let part = self.part_place(place, ty, (0, index as u32))?;
                self.push_unwind_parts(root, &sub, field_ty, part, paths, init, out)?;
            }
            return Ok(());
        }
        for part in paths.iter().filter(|part| part.starts_with(path)) {
            out.push(Part {
                path: part.clone(),
                flag: self.flag(init),
                unwind: false,
            });
        }
        Ok(())
    }

    pub(in crate::lower) fn body_may_unwind(&mut self) -> Result<bool> {
        Ok(self.open && self.reached.may_unwind(self.cx, self.index))
    }

    pub(super) fn push_unwind(
        &mut self,
        root: Root,
        ty: Ty,
        place: Option<Place>,
        init: bool,
    ) -> Result<StackSlot> {
        let origin = TyOrigin::from_root(root, self.typed);
        self.push_unwind_origin(origin, ty, place, init)
    }

    pub(super) fn uses_drop_list(&self) -> bool {
        self.mode == LowerMode::Dict
    }

    fn copy_wide_to_slot(&mut self, place: Option<Place>, ty: Ty) -> Result<StackSlot> {
        let addr = self.address_or_dangling(place, ty)?;
        let wide_slot = self.b.create_stack_slot(16, 3);
        let data = self.ins().load(Type::I64, addr, 0);
        let meta = self.ins().load(Type::I64, addr, 8);
        self.ins().stack_store(data, wide_slot, 0);
        self.ins().stack_store(meta, wide_slot, 8);
        Ok(wide_slot)
    }

    pub(super) fn push_unwind_origin(
        &mut self,
        origin: TyOrigin,
        ty: Ty,
        place: Option<Place>,
        init: bool,
    ) -> Result<StackSlot> {
        let Some(glue) = self.cx.lang_item("drop_glue") else {
            bail!("core has no drop glue")
        };
        let substs = self.cx.types.intern_substs(&[ty]);
        let glue_index = self.reached_index(glue, substs)?;
        let is_wide = self.cx.is_unsized(ty)?;
        if !self.uses_drop_list() {
            let drop_sym = self.push_reloc(
                SymbolSource::Slot(glue_index),
                RelocStep::SiteSlot(SiteOrigin::DropGlue(origin)),
            );
            let (local, local_off, arg) = match place {
                Some(Place {
                    base: Base::Slot(slot),
                    offset,
                    repr,
                }) if !matches!(repr, Repr::Zst) && self.b.func.slot_align_shift(slot) <= 4 => (
                    slot,
                    offset,
                    if is_wide {
                        UnwindArg::Wide
                    } else {
                        UnwindArg::Direct
                    },
                ),
                _ if is_wide => (self.copy_wide_to_slot(place, ty)?, 0, UnwindArg::Wide),
                _ => {
                    let addr = self.address_or_dangling(place, ty)?;
                    let ptr_slot = self.b.create_stack_slot(8, 3);
                    self.ins().stack_store(addr, ptr_slot, 0);
                    (ptr_slot, 0, UnwindArg::Indirect)
                }
            };
            let flag = self.flag(init);
            self.b.func.add_unwind_drop(UnwindDrop {
                local,
                local_off,
                flag,
                drop_sym,
                arg,
            });
            return Ok(flag);
        }
        let drop_fn = self.code_of(glue_index, SiteOrigin::DropGlue(origin));
        let addr = if is_wide {
            match place {
                Some(Place {
                    base: Base::Slot(slot),
                    offset,
                    ..
                }) => self.ins().stack_addr(Type::I64, slot, offset),
                _ => {
                    let wide_slot = self.copy_wide_to_slot(place, ty)?;
                    self.ins().stack_addr(Type::I64, wide_slot, 0)
                }
            }
        } else {
            self.address_or_dangling(place, ty)?
        };
        let slot = self
            .b
            .create_stack_slot(crate::host::unwind::DROP_NODE_BYTES, 3);
        self.flags.push(slot);
        let init_bits = u8::from(init)
            | if is_wide {
                crate::host::unwind::DROP_WIDE
            } else {
                0
            };
        let state_val = self.ins().iconst(Type::I8, i64::from(init_bits));
        self.ins().stack_store(state_val, slot, 0);
        self.ins()
            .stack_store(drop_fn, slot, crate::host::unwind::DROP_NODE_FN_OFF);
        self.ins()
            .stack_store(addr, slot, crate::host::unwind::DROP_NODE_ARG_OFF);
        let node = self.ins().stack_addr(Type::I64, slot, 0);
        self.call_host(HostSymbol::DropPush, &[node], None);
        Ok(slot)
    }

    pub(super) fn pop_unwind(&mut self, slot: StackSlot) {
        if !self.uses_drop_list() {
            return;
        }
        let node = self.ins().stack_addr(Type::I64, slot, 0);
        self.call_host(HostSymbol::DropPop, &[node], None);
    }

    pub(super) fn flag(&mut self, init: bool) -> StackSlot {
        let slot = self.b.create_stack_slot(1, 0);
        self.flags.push(slot);
        self.set(slot, init);
        slot
    }

    fn set(&mut self, slot: StackSlot, value: bool) {
        let value = self.ins().iconst(Type::I8, i64::from(value));
        self.ins().stack_store(value, slot, 0);
    }

    pub(in crate::lower) fn set_dynamic_active(
        &mut self,
        slot: StackSlot,
        unwind: bool,
        active: Value,
    ) {
        let next = if unwind && self.uses_drop_list() {
            let cur = self.ins().stack_load(Type::I8, slot, 0);
            let mask = self.ins().iconst(Type::I8, DROP_LIST_CLEAR_MASK);
            let cleared = self.ins().band(cur, mask);
            self.ins().bor(cleared, active)
        } else {
            active
        };
        self.ins().stack_store(next, slot, 0);
    }

    pub(super) fn set_active(&mut self, slot: StackSlot, unwind: bool, value: bool) {
        if !(unwind && self.uses_drop_list()) {
            return self.set(slot, value);
        }
        let cur = self.ins().stack_load(Type::I8, slot, 0);
        let next = if value {
            let bit = self.ins().iconst(Type::I8, DROP_LIST_ACTIVE_BIT);
            self.ins().bor(cur, bit)
        } else {
            let mask = self.ins().iconst(Type::I8, DROP_LIST_CLEAR_MASK);
            self.ins().band(cur, mask)
        };
        self.ins().stack_store(next, slot, 0);
    }

    pub(super) fn disarm_unwind(&mut self, slot: StackSlot, unwind: bool) {
        self.set_active(slot, unwind, false);
        if unwind {
            self.pop_unwind(slot);
        }
    }
}
