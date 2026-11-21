use anyhow::{Result, bail};
use haste_codegen::{IntCC, StackSlot, Type, Value};

use crate::body::{Expr, ExprId};
use crate::host::HostSymbol;
use crate::sema::infer::{Fields, Res, Shape, drop_fn, field_types, has_drop_impl, needs_drop};
use crate::sema::intrinsic::Intrinsic;
use crate::sema::ty::{Kind, Ty};

use super::moves::{BOX_STEP, MovePath, Root};
use super::place::{Base, Place};
use super::{Lower, Repr, repr};

fn skips_fields(cx: &crate::sema::Context<'_>, item: crate::names::ItemRef) -> bool {
    cx.lang(item) == Some("manually_drop")
        || cx.item(item).1.kind == crate::outline::item::ItemKind::Union
}

/// One tracked sub-part of an owned value and its presence/unwind guard slot.
#[derive(Clone, Debug)]
struct Part {
    path: MovePath,
    flag: StackSlot,
    unwind: bool,
}

/// A value that the body owns and drops when its scope ends: where it lives, and the flags that say which of it is still there.
#[derive(Clone, Debug)]
pub struct Owned {
    root: Root,
    ty: Ty,
    place: Option<Place>,
    flag: StackSlot,
    unwind: bool,
    parts: Vec<Part>,
    life: Life,
}

/// A move that a matched pattern's binding makes, applied once the arm is taken: only when each `(which, alternative)` holds, for a move inside or-pattern alternatives.
#[derive(Clone, Debug)]
pub struct Pending {
    pub root: Root,
    pub path: MovePath,
    pub when: Vec<(Value, i64)>,
}

/// How long an owned value lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Life {
    /// A temporary dropped at the end of its statement.
    Statement,
    /// A binding, or an extended temporary of a finished `let`, dropped at the end of its block.
    Scope,
    /// A temporary that a `let` still being lowered extends, which outlives the blocks inside its initializer.
    Extended,
}

impl Lower<'_, '_> {
    /// Takes ownership of the value of `ty` at `place` for `root`, when dropping it runs code, and marks it present when `init` holds.
    pub fn own(&mut self, root: Root, ty: Ty, place: Option<Place>, init: bool) -> Result<()> {
        if !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        let paths = self.moved.get(&root).cloned().unwrap_or_default();
        let want_unwind = self.open && self.resume.is_none() && self.reached.has_catch_unwind(self.cx) && self.body_may_unwind()?;
        let into_box = paths.iter().any(|path| path.contains(&BOX_STEP));
        let (flag, unwind, parts) = if want_unwind && (paths.is_empty() || (!into_box && has_drop_impl(self.cx, ty)?)) {
            (self.push_unwind(root, ty, place, init)?, true, Vec::new())
        } else if want_unwind {
            let parts = self.expand_unwind_parts(root, ty, place, &paths, init)?;
            (self.flag(init), false, parts)
        } else {
            (self.flag(init), false, self.plain_parts(&paths, init))
        };
        let life = match root {
            Root::Local(_) => Life::Scope,
            Root::Temp(expr) if self.extended.contains(&expr) => Life::Extended,
            Root::Temp(_) | Root::Held(_) => Life::Statement,
        };
        self.owned.push(Owned { root, ty, place, flag, unwind, parts, life });
        Ok(())
    }

    fn plain_parts(&mut self, paths: &[MovePath], init: bool) -> Vec<Part> {
        paths.iter().map(|path| Part { path: path.clone(), flag: self.flag(init), unwind: false }).collect()
    }

    fn expand_unwind_parts(&mut self, root: Root, ty: Ty, place: Option<Place>, paths: &[MovePath], init: bool) -> Result<Vec<Part>> {
        let Some(place) = place else {
            return Ok(self.plain_parts(paths, init));
        };
        let mut parts = Vec::new();
        self.push_unwind_parts(root, &[], ty, place, paths, init, &mut parts)?;
        Ok(parts)
    }

    fn single_variant_fields(&mut self, ty: Ty) -> Result<Option<Vec<Ty>>> {
        Ok(match *self.cx.types.kind(ty) {
            Kind::Tuple(items) => Some(self.cx.types.substs(items).to_vec()),
            Kind::Array(element, len) => Some(vec![element; self.len(len)? as usize]),
            Kind::Adt(item, _) if skips_fields(self.cx, item) => Some(Vec::new()),
            Kind::Adt(item, args) => {
                let slice = self.cx.types.substs(args).to_vec();
                let variants = field_types(self.cx, item, &slice)?;
                (variants.len() == 1).then(|| variants[0].clone())
            }
            _ => None,
        })
    }

    fn push_unwind_parts(&mut self, root: Root, path: &[(u32, u32)], ty: Ty, place: Place, paths: &[MovePath], init: bool, out: &mut Vec<Part>) -> Result<()> {
        if !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        let below = paths.iter().any(|part| part.len() > path.len() && part.starts_with(path));
        let inside: MovePath = path.iter().copied().chain([BOX_STEP]).collect();
        let into_box = paths.iter().any(|part| part.starts_with(&inside));
        if !into_box && (!below || has_drop_impl(self.cx, ty)?) {
            let slot = self.push_unwind(root, ty, Some(place), init)?;
            out.push(Part { path: path.to_vec(), flag: slot, unwind: true });
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
            out.push(Part { path: part.clone(), flag: self.flag(init), unwind: false });
        }
        Ok(())
    }

    fn body_may_unwind(&self) -> Result<bool> {
        if !self.open {
            return Ok(false);
        }
        for (item, _) in self.typed.calls() {
            if let Some(intr) = self.cx.intrinsic(item) {
                if matches!(intr, Intrinsic::ConstEvalSelect | Intrinsic::CatchUnwind) {
                    return Ok(true);
                }
                continue;
            }
            if self.cx.foreign(item) {
                let host = self.cx.symbol_name(item).as_deref().and_then(|name| HostSymbol::resolve_for(name, true));
                if matches!(host, None | Some(HostSymbol::StartPanic | HostSymbol::CatchUnwind | HostSymbol::Dynamic(_))) {
                    return Ok(true);
                }
                continue;
            }
            return Ok(true);
        }
        for (idx, expr) in self.body.exprs.iter().enumerate() {
            match expr {
                Expr::FormatArgs(_) => return Ok(true),
                Expr::Call(callee, _)
                    if !matches!(self.typed.res.get(&ExprId(idx as u32)), Some(Res::Fn(..) | Res::Ctor(..)))
                        && !matches!(self.typed.res.get(callee), Some(Res::Fn(..) | Res::Ctor(..))) =>
                {
                    return Ok(true);
                }
                _ => {}
            }
        }
        Ok(false)
    }

    fn push_unwind(&mut self, root: Root, ty: Ty, place: Option<Place>, init: bool) -> Result<StackSlot> {
        let Some(glue) = self.cx.lang_item("drop_glue") else { bail!("core has no drop glue") };
        let substs = self.cx.types.intern_substs(&[ty]);
        let glue_index = self.reached_index(glue, substs)?;
        let origin = super::reloc::TyOrigin::from_root(root, self.typed);
        let drop_fn = self.code_of(glue_index, super::reloc::SiteOrigin::DropGlue(origin));
        let addr = self.address_or_dangling(place, ty)?;
        let slot = self.b.create_stack_slot(40, 3);
        self.flags.push(slot);
        self.set(slot, init);
        self.ins().stack_store(drop_fn, slot, 24);
        self.ins().stack_store(addr, slot, 32);
        let node = self.ins().stack_addr(Type::I64, slot, 0);
        self.call_host(HostSymbol::DropPush, &[node], None);
        Ok(slot)
    }

    fn pop_unwind(&mut self, slot: StackSlot) {
        let node = self.ins().stack_addr(Type::I64, slot, 0);
        self.call_host(HostSymbol::DropPop, &[node], None);
    }

    fn flag(&mut self, init: bool) -> StackSlot {
        let slot = self.b.create_stack_slot(1, 0);
        self.flags.push(slot);
        self.set(slot, init);
        slot
    }

    fn set(&mut self, slot: StackSlot, value: bool) {
        let value = self.ins().iconst(Type::I8, i64::from(value));
        self.ins().stack_store(value, slot, 0);
    }

    fn set_active(&mut self, slot: StackSlot, unwind: bool, value: bool) {
        if !unwind {
            return self.set(slot, value);
        }
        let cur = self.ins().stack_load(Type::I8, slot, 0);
        let next = if value {
            let bit = self.ins().iconst(Type::I8, 1);
            self.ins().bor(cur, bit)
        } else {
            let mask = self.ins().iconst(Type::I8, 0xFE);
            self.ins().band(cur, mask)
        };
        self.ins().stack_store(next, slot, 0);
    }

    fn owner(&self, root: Root) -> Option<usize> {
        self.owned.iter().rposition(|owned| owned.root == root)
    }

    /// Marks the part `path` of `root` as moved out, so that its scope does not drop it.
    pub fn moved_out(&mut self, root: Root, path: &[(u32, u32)]) -> Result<()> {
        for (slot, unwind) in self.move_flags(root, path)? {
            self.set_active(slot, unwind, false);
        }
        Ok(())
    }

    fn collect_targets(&self, at: usize, path: &[(u32, u32)]) -> Vec<(StackSlot, bool)> {
        let mut slots: Vec<(StackSlot, bool)> = self.owned[at]
            .parts
            .iter()
            .filter(|part| part.path.starts_with(path))
            .map(|part| (part.flag, part.unwind))
            .collect();
        if path.is_empty() {
            slots.push((self.owned[at].flag, self.owned[at].unwind));
        }
        slots
    }

    fn move_flags(&self, root: Root, path: &[(u32, u32)]) -> Result<Vec<(StackSlot, bool)>> {
        let Some(at) = self.owner(root) else {
            return Ok(Vec::new());
        };
        let slots = self.collect_targets(at, path);
        if !path.is_empty() && slots.is_empty() {
            bail!("a move out of {root:?} at {path:?} that the move analysis did not find");
        }
        Ok(slots)
    }

    /// Marks the part `path` of `root` as moved out when every `(which, alternative)` of `when` holds.
    fn moved_out_when(&mut self, root: Root, path: &[(u32, u32)], when: &[(Value, i64)]) -> Result<()> {
        let slots = self.move_flags(root, path)?;
        if slots.is_empty() {
            return Ok(());
        }
        let mut taken = self.ins().iconst(Type::I8, 1);
        for &(which, alternative) in when {
            let holds = self.ins().icmp_imm(IntCC::Equal, which, alternative);
            taken = self.ins().band(taken, holds);
        }
        for (slot, unwind) in slots {
            let present = self.ins().stack_load(Type::I8, slot, 0);
            let cleared = if unwind {
                let mask = self.ins().iconst(Type::I8, 0xFE);
                self.ins().band(present, mask)
            } else {
                self.ins().iconst(Type::I8, 0)
            };
            let kept = self.ins().select(taken, cleared, present);
            self.ins().stack_store(kept, slot, 0);
        }
        Ok(())
    }

    /// Marks the part `path` of `root` and everything below it as present again, after an assignment.
    pub fn reinit(&mut self, root: Root, path: &[(u32, u32)]) {
        let Some(at) = self.owner(root) else { return };
        for (slot, unwind) in self.collect_targets(at, path) {
            self.set_active(slot, unwind, true);
        }
    }

    /// Marks every owned value from index `mark` on as absent, on a path where their bindings did not happen.
    pub fn forget_from(&mut self, mark: usize) {
        let slots: Vec<(StackSlot, bool)> = (mark..self.owned.len()).flat_map(|at| self.collect_targets(at, &[])).collect();
        for (slot, unwind) in slots {
            self.set_active(slot, unwind, false);
            if unwind {
                self.pop_unwind(slot);
            }
        }
    }

    /// Writes `value` over the place `lhs` names after dropping the value there, when its type needs drop; says whether it did.
    pub fn replace(&mut self, lhs: ExprId, place: Place, value: Value) -> Result<bool> {
        let ty = self.ty_of(lhs);
        if !needs_drop(self.cx, ty)? {
            return Ok(false);
        }
        let tracked = self.move_path(lhs);
        let replaced = match &tracked {
            Some((root, path)) => self.drop_replaced(*root, path, ty, place)?,
            None => false,
        };
        if !replaced {
            self.glue(ty, Some(place), super::reloc::TyOrigin::Expr(lhs))?;
        }
        self.write(place, value);
        if let Some((root, path)) = tracked {
            self.reinit(root, &path);
        }
        Ok(true)
    }

    fn drop_replaced(&mut self, root: Root, path: &[(u32, u32)], ty: Ty, place: Place) -> Result<bool> {
        let Some(at) = self.owner(root) else {
            return Ok(false);
        };
        let (flag, unwind) = (self.owned[at].flag, self.owned[at].unwind);
        self.guarded_slot(flag, unwind, |lower| {
            if path.is_empty() {
                for (slot, slot_unwind) in lower.collect_targets(at, &[]) {
                    lower.set_active(slot, slot_unwind, false);
                }
            }
            lower.drop_path(at, path, ty, Some(place))?;
            if !path.is_empty() {
                for (slot, slot_unwind) in lower.collect_targets(at, path) {
                    lower.set_active(slot, slot_unwind, false);
                }
            }
            Ok(())
        })?;
        Ok(true)
    }

    /// Applies the moves that the bindings of a matched pattern made.
    pub fn commit_moves(&mut self) -> Result<()> {
        for Pending { root, path, when } in std::mem::take(&mut self.pending) {
            match when.is_empty() {
                true => self.moved_out(root, &path)?,
                false => self.moved_out_when(root, &path, &when)?,
            }
        }
        Ok(())
    }

    /// Drops every owned value from index `mark` on, innermost first, each only when its flag says it is present.
    pub fn drop_to(&mut self, mark: usize) -> Result<()> {
        for at in (mark..self.owned.len()).rev() {
            self.drop_owned(at)?;
        }
        Ok(())
    }

    /// Ends the scope that began at `mark`: drops what it owns, and hands the temporaries a `let` extends to the scope of that `let`.
    pub fn end_scope(&mut self, mark: usize) -> Result<()> {
        self.end(mark, |life| life == Life::Extended)
    }

    /// Hands every value owned from `mark` on to the scope around the enclosing block, as a `super let` binding lives.
    pub fn extend_owned(&mut self, mark: usize) {
        for owned in &mut self.owned[mark..] {
            owned.life = Life::Extended;
        }
    }

    /// Ends a statement that began at `mark`: drops its temporaries and keeps its bindings and the temporaries it extends.
    pub fn end_statement(&mut self, mark: usize) -> Result<()> {
        self.end(mark, |life| life != Life::Statement)?;
        for owned in &mut self.owned[mark..] {
            owned.life = Life::Scope;
        }
        Ok(())
    }

    fn end(&mut self, mark: usize, keeps: impl Fn(Life) -> bool) -> Result<()> {
        let rest = self.owned.split_off(mark);
        let (keep, ended): (Vec<Owned>, Vec<Owned>) = rest.into_iter().partition(|owned| keeps(owned.life));
        self.owned.extend(ended);
        if self.open {
            self.drop_to(mark)?;
        }
        self.owned.truncate(mark);
        self.owned.extend(keep);
        Ok(())
    }

    fn drop_owned(&mut self, at: usize) -> Result<()> {
        if !self.open {
            return Ok(());
        }
        let Owned { ty, place, flag, unwind, parts, .. } = self.owned[at].clone();
        if unwind {
            self.pop_unwind(flag);
        }
        for part in &parts {
            if part.unwind {
                self.pop_unwind(part.flag);
            }
        }
        self.guarded_slot(flag, unwind, |lower| {
            lower.set_active(flag, unwind, false);
            lower.drop_path(at, &[], ty, place)
        })
    }

    fn guarded_slot(&mut self, flag: StackSlot, unwind: bool, run: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let raw = self.ins().stack_load(Type::I8, flag, 0);
        let present = if unwind {
            let mask = self.ins().iconst(Type::I8, 1);
            self.ins().band(raw, mask)
        } else {
            raw
        };
        self.when(present, run)
    }

    fn when(&mut self, cond: Value, run: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let (yes, after) = (self.b.create_block(), self.b.create_block());
        self.brif(cond, (yes, &[]), (after, &[]));
        self.enter(yes);
        run(self)?;
        self.jump(after, &[]);
        self.enter(after);
        Ok(())
    }

    fn drop_path(&mut self, at: usize, path: &[(u32, u32)], ty: Ty, place: Option<Place>) -> Result<()> {
        let parts = &self.owned[at].parts;
        let below = parts.iter().any(|part| part.path.len() > path.len() && part.path.starts_with(path));
        let part_flag = parts.iter().find(|part| part.path == path).map(|part| (part.flag, part.unwind)).filter(|_| !path.is_empty());
        let inside: MovePath = path.iter().copied().chain([BOX_STEP]).collect();
        let into_box = parts.iter().any(|part| part.path.starts_with(&inside));
        let opened = below && !has_drop_impl(self.cx, ty)?;
        let origin = super::reloc::TyOrigin::from_root(self.owned[at].root, self.typed).stepped(path);
        let run = |lower: &mut Self| match (into_box, opened) {
            (true, _) => lower.open_box(at, &inside, ty, place, origin.clone()),
            (false, true) => lower.open_drop(at, path, ty, place),
            (false, false) => lower.glue(ty, place, origin.clone()),
        };
        match part_flag {
            Some((flag, unwind)) => self.guarded_slot(flag, unwind, |lower| {
                lower.set_active(flag, unwind, false);
                run(lower)
            }),
            None => run(self),
        }
    }

    /// Drops what is left of the value the `Box` of type `ty` at `place` owns, part by part, then frees the box with its own `Drop::drop`.
    fn open_box(&mut self, at: usize, inside: &[(u32, u32)], ty: Ty, place: Option<Place>, origin: super::reloc::TyOrigin) -> Result<()> {
        let (Some(place), Some(owned)) = (place, self.cx.boxed(ty)) else { bail!("a partly moved box with no bytes") };
        let target = self.address(place);
        let contents = self.ins().load(Type::I64, target, 0);
        let part = Place { base: Base::Addr(contents), offset: 0, repr: repr(self.cx, owned)? };
        self.drop_path(at, inside, owned, Some(part))?;
        let Some(Res::Fn(item, substs)) = drop_fn(self.cx, ty)? else { bail!("`Box` has no `Drop::drop`") };
        self.invoke(super::reloc::SiteOrigin::DropFn(origin), item, substs, vec![target]).map(drop)
    }

    fn open_drop(&mut self, at: usize, path: &[(u32, u32)], ty: Ty, place: Option<Place>) -> Result<()> {
        let Some(place) = place else { bail!("a partly moved value with no bytes") };
        let variants: Fields = match *self.cx.types.kind(ty) {
            Kind::Tuple(items) => vec![self.cx.types.substs(items).to_vec()].into(),
            Kind::Array(element, len) => vec![vec![element; self.len(len)? as usize]].into(),
            Kind::Adt(item, _) if skips_fields(self.cx, item) => return Ok(()),
            Kind::Adt(item, args) => {
                let slice = self.cx.types.substs(args).to_vec();
                field_types(self.cx, item, &slice)?
            }
            ref other => bail!("a partly moved {other:?}"),
        };
        self.each_field((ty, place), &variants, &mut |lower, field, field_ty, part| {
            let mut sub = path.to_vec();
            sub.push(field);
            lower.drop_path(at, &sub, field_ty, Some(part))
        })
    }

    fn each_field(&mut self, (ty, place): (Ty, Place), variants: &[Vec<Ty>], visit: &mut dyn FnMut(&mut Self, (u32, u32), Ty, Place) -> Result<()>) -> Result<()> {
        let enumerated = variants.len() > 1;
        for (variant, fields) in variants.iter().enumerate() {
            let variant = variant as u32;
            let mut each = |lower: &mut Self| -> Result<()> {
                for (index, &field) in fields.iter().enumerate() {
                    let part = lower.part_place(place, ty, (variant, index as u32))?;
                    visit(lower, (variant, index as u32), field, part)?;
                }
                Ok(())
            };
            match enumerated {
                true => self.when_variant(place, ty, variant, &mut each)?,
                false => each(self)?,
            }
        }
        Ok(())
    }

    fn when_variant(&mut self, place: Place, ty: Ty, variant: u32, run: &mut dyn FnMut(&mut Self) -> Result<()>) -> Result<()> {
        match self.is_variant(Some(place), ty, variant)? {
            Some(test) => self.when(test, run),
            None => run(self),
        }
    }

    /// Runs the drop glue of `ty` on the value at `place`.
    pub(super) fn glue(&mut self, ty: Ty, place: Option<Place>, origin: super::reloc::TyOrigin) -> Result<()> {
        if !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        let Some(glue) = self.cx.lang_item("drop_glue") else { bail!("core has no drop glue") };
        let target = self.address_or_dangling(place, ty)?;
        let substs = self.cx.types.intern_substs(&[ty]);
        self.invoke(super::reloc::SiteOrigin::DropGlue(origin), glue, substs, vec![target]).map(drop)
    }

    /// Lowers the body of the drop glue for `ty` whose target is `target`: `Drop::drop`, then the parts.
    pub fn glue_body(&mut self, ty: Ty, target: Option<Value>) -> Result<()> {
        let Some(target) = target else { return Ok(()) };
        let glue = match &self.typed.shape {
            Shape::Glue(glue) => glue.clone(),
            _ => Default::default(),
        };
        if let Some(resume) = glue.resume {
            return self.coroutine_glue(ty, target, resume);
        }
        let drop_fn = match glue.drop {
            Some(Res::Fn(item, substs)) => Some((item, substs)),
            _ => None,
        };
        let origin = super::reloc::TyOrigin::Subst(0);
        if let Some(owned) = self.cx.boxed(ty) {
            self.box_contents(owned, target, origin.step(BOX_STEP.0, 0))?;
            if let Some((item, substs)) = drop_fn {
                self.invoke(super::reloc::SiteOrigin::ShapeGlueDrop, item, substs, vec![target])?;
            }
            return Ok(());
        }
        if let Some((item, substs)) = drop_fn {
            self.invoke(super::reloc::SiteOrigin::ShapeGlueDrop, item, substs, vec![target])?;
        }
        self.drop_parts(ty, target, origin)
    }

    fn box_contents(&mut self, inner: Ty, target: Value, origin: super::reloc::TyOrigin) -> Result<()> {
        let reference = self.cx.types.intern(Kind::Ref(crate::sema::ty::Mut::Mut, inner));
        let contents = match repr(self.cx, reference)? {
            Repr::Scalar(_) => self.ins().load(Type::I64, target, 0),
            _ => target,
        };
        let place = Place { base: Base::Addr(contents), offset: 0, repr: repr(self.cx, inner).unwrap_or(Repr::Zst) };
        match self.cx.types.kind(inner) {
            Kind::Slice(_) | Kind::Prim(crate::names::Primitive::Str) | Kind::Dyn(..) => self.glue_wide(inner, contents, origin),
            _ => self.glue(inner, Some(place), origin),
        }
    }

    fn glue_wide(&mut self, ty: Ty, wide: Value, origin: super::reloc::TyOrigin) -> Result<()> {
        if !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        let Some(glue) = self.cx.lang_item("drop_glue") else { bail!("core has no drop glue") };
        let substs = self.cx.types.intern_substs(&[ty]);
        self.invoke(super::reloc::SiteOrigin::DropGlue(origin), glue, substs, vec![wide]).map(drop)
    }

    fn drop_parts(&mut self, ty: Ty, target: Value, origin: super::reloc::TyOrigin) -> Result<()> {
        match self.cx.types.kind(ty).clone() {
            Kind::Array(inner, count) => {
                let count = self.len(count)?;
                let count = self.ins().iconst(Type::I64, count);
                self.drop_elements(inner, target, count, origin.step(0, 0))
            }
            Kind::Slice(inner) => {
                let (data, len) = (self.ins().load(Type::I64, target, 0), self.ins().load(Type::I64, target, 8));
                self.drop_elements(inner, data, len, origin.step(0, 0))
            }
            Kind::Tuple(_) | Kind::Adt(..) => {
                let repr = if self.cx.is_unsized(ty)? { super::ops::WIDE } else { repr(self.cx, ty)? };
                let place = Place { base: Base::Addr(target), offset: 0, repr };
                self.drop_fields(ty, place, origin)
            }
            Kind::Closure(..) => match self.cx.closure_env(ty) {
                Some(env) => self.drop_parts(env, target, origin.step(super::reloc::CLOSURE_ENV_STEP, 0)),
                None => bail!("drop of a closure whose defining body was not inferred"),
            },
            Kind::Dyn(..) => self.drop_object(target),
            _ => Ok(()),
        }
    }

    fn drop_object(&mut self, wide: Value) -> Result<()> {
        let data = self.ins().load(Type::I64, wide, 0);
        let vtable = self.ins().load(Type::I64, wide, 8);
        let code = self.ins().load(Type::I64, vtable, 0);
        let unit = self.cx.types.unit();
        let pointer = self.cx.types.intern(Kind::Ptr(crate::sema::ty::Mut::Mut, unit));
        self.when(code, |lower| lower.call_code(code, (vec![pointer], None), vec![data]).map(drop))
    }

    pub(super) fn drop_fields(&mut self, ty: Ty, place: Place, origin: super::reloc::TyOrigin) -> Result<()> {
        let variants: Fields = match *self.cx.types.kind(ty) {
            Kind::Tuple(items) => vec![self.cx.types.substs(items).to_vec()].into(),
            Kind::Adt(item, _) if skips_fields(self.cx, item) => return Ok(()),
            Kind::Adt(item, args) => {
                let slice = self.cx.types.substs(args).to_vec();
                field_types(self.cx, item, &slice)?
            }
            _ => return Ok(()),
        };
        self.each_field((ty, place), &variants, &mut |lower, (variant, field), field_ty, part| lower.glue(field_ty, Some(part), origin.clone().step(variant, field)))
    }

    fn drop_elements(&mut self, inner: Ty, data: Value, count: Value, origin: super::reloc::TyOrigin) -> Result<()> {
        if !needs_drop(self.cx, inner)? {
            return Ok(());
        }
        let size = self.cx.layout(inner)?.size.bytes() as i64;
        let element = repr(self.cx, inner)?;
        let mut failed = None;
        self.each(count, |lower, index| {
            let offset = lower.ins().imul_imm(index, size);
            let at = lower.ins().iadd(data, offset);
            if let Err(error) = lower.glue(inner, Some(Place { base: Base::Addr(at), offset: 0, repr: element }), origin.clone()) {
                failed = Some(error);
            }
        });
        failed.map_or(Ok(()), Err)
    }
}
