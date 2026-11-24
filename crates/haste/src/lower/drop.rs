mod glue;
mod unwind;

use anyhow::{Result, bail};
use haste_codegen::{IntCC, StackSlot, Type, Value};

use crate::body::ExprId;
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{Fields, Res, drop_fn, field_types, has_drop_impl, needs_drop};
use crate::sema::ty::{Kind, Ty};

use super::moves::{BOX_STEP, MovePath, Root};
use super::place::{Base, Place};
use super::reloc::{SiteOrigin, TyOrigin};
use super::{Lower, repr};

const DROP_LIST_ACTIVE_BIT: i64 = 1;
const DROP_LIST_CLEAR_MASK: i64 = 0xFE;

fn skips_fields(cx: &Context<'_>, item: ItemRef) -> bool {
    cx.lang(item) == Some("manually_drop") || cx.item(item).1.kind == ItemKind::Union
}

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
    pub(super) flag: StackSlot,
    pub(super) unwind: bool,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Life {
    Statement,
    Scope,
    Extended,
}

impl Lower<'_, '_> {
    /// Takes ownership of the value of `ty` at `place` for `root`, when dropping it runs code, and marks it present when `init` holds.
    pub fn own(&mut self, root: Root, ty: Ty, place: Option<Place>, init: bool) -> Result<()> {
        if !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        let paths = self.moved.get(&root).cloned().unwrap_or_default();
        let want_unwind =
            self.open && self.resume.is_none() && self.should_unwind() && self.body_may_unwind()?;
        let into_box = paths.iter().any(|path| path.contains(&BOX_STEP));
        let (flag, unwind, parts) =
            if want_unwind && (paths.is_empty() || (!into_box && has_drop_impl(self.cx, ty)?)) {
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
        self.owned.push(Owned {
            root,
            ty,
            place,
            flag,
            unwind,
            parts,
            life,
        });
        Ok(())
    }

    fn plain_parts(&mut self, paths: &[MovePath], init: bool) -> Vec<Part> {
        paths
            .iter()
            .map(|path| Part {
                path: path.clone(),
                flag: self.flag(init),
                unwind: false,
            })
            .collect()
    }

    fn adt_variants(&mut self, ty: Ty) -> Result<Option<Fields>> {
        match *self.cx.types.kind(ty) {
            Kind::Adt(item, _) if skips_fields(self.cx, item) => Ok(None),
            Kind::Adt(item, args) => {
                let slice = self.cx.types.substs(args).to_vec();
                Ok(Some(field_types(self.cx, item, &slice)?))
            }
            _ => Ok(None),
        }
    }

    fn single_variant_fields(&mut self, ty: Ty) -> Result<Option<Vec<Ty>>> {
        Ok(match *self.cx.types.kind(ty) {
            Kind::Tuple(items) => Some(self.cx.types.substs(items).to_vec()),
            Kind::Array(element, len) => Some(vec![element; self.len(len)? as usize]),
            Kind::Adt(..) => match self.adt_variants(ty)? {
                Some(variants) => (variants.len() == 1).then(|| variants[0].clone()),
                None => Some(Vec::new()),
            },
            _ => None,
        })
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

    fn moved_out_when(
        &mut self,
        root: Root,
        path: &[(u32, u32)],
        when: &[(Value, i64)],
    ) -> Result<()> {
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
            let cleared = if unwind && self.uses_drop_list() {
                let mask = self.ins().iconst(Type::I8, DROP_LIST_CLEAR_MASK);
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
        let slots: Vec<(StackSlot, bool)> = (mark..self.owned.len())
            .flat_map(|at| self.collect_targets(at, &[]))
            .collect();
        for (slot, unwind) in slots {
            self.disarm_unwind(slot, unwind);
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
            self.glue(ty, Some(place), TyOrigin::Expr(lhs))?;
        }
        self.write(place, value);
        if let Some((root, path)) = tracked {
            self.reinit(root, &path);
        }
        Ok(true)
    }

    fn drop_replaced(
        &mut self,
        root: Root,
        path: &[(u32, u32)],
        ty: Ty,
        place: Place,
    ) -> Result<bool> {
        let Some(at) = self.owner(root) else {
            return Ok(false);
        };
        let (flag, unwind) = (self.owned[at].flag, self.owned[at].unwind);
        self.guarded_slot(flag, unwind, |lower| {
            if path.is_empty() {
                lower.set_active(flag, unwind, false);
            }
            lower.drop_path(at, path, ty, Some(place))?;
            for (slot, slot_unwind) in lower.collect_targets(at, path) {
                lower.set_active(slot, slot_unwind, false);
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
        let (keep, ended): (Vec<Owned>, Vec<Owned>) =
            rest.into_iter().partition(|owned| keeps(owned.life));
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
        let Owned {
            ty,
            place,
            flag,
            unwind,
            parts,
            ..
        } = self.owned[at].clone();
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

    fn guarded_slot(
        &mut self,
        flag: StackSlot,
        unwind: bool,
        run: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        let raw = self.ins().stack_load(Type::I8, flag, 0);
        let present = if unwind && self.uses_drop_list() {
            let mask = self.ins().iconst(Type::I8, DROP_LIST_ACTIVE_BIT);
            self.ins().band(raw, mask)
        } else {
            raw
        };
        self.when(present, run)
    }

    pub(super) fn when(
        &mut self,
        cond: Value,
        run: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        let (yes, after) = (self.b.create_block(), self.b.create_block());
        self.brif(cond, (yes, &[]), (after, &[]));
        self.enter(yes);
        run(self)?;
        self.jump(after, &[]);
        self.enter(after);
        Ok(())
    }

    fn drop_path(
        &mut self,
        at: usize,
        path: &[(u32, u32)],
        ty: Ty,
        place: Option<Place>,
    ) -> Result<()> {
        let parts = &self.owned[at].parts;
        let below = parts
            .iter()
            .any(|part| part.path.len() > path.len() && part.path.starts_with(path));
        let part_flag = parts
            .iter()
            .find(|part| part.path == path)
            .map(|part| (part.flag, part.unwind))
            .filter(|_| !path.is_empty());
        let inside: MovePath = path.iter().copied().chain([BOX_STEP]).collect();
        let into_box = parts.iter().any(|part| part.path.starts_with(&inside));
        let opened = below && !has_drop_impl(self.cx, ty)?;
        let origin = TyOrigin::from_root(self.owned[at].root, self.typed).stepped(path);
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

    fn open_box(
        &mut self,
        at: usize,
        inside: &[(u32, u32)],
        ty: Ty,
        place: Option<Place>,
        origin: TyOrigin,
    ) -> Result<()> {
        let (Some(place), Some(owned)) = (place, self.cx.boxed(ty)) else {
            bail!("a partly moved box with no bytes")
        };
        let target = self.address(place);
        let contents = self.ins().load(Type::I64, target, 0);
        let part = Place {
            base: Base::Addr(contents),
            offset: 0,
            repr: repr(self.cx, owned)?,
        };
        self.drop_path(at, inside, owned, Some(part))?;
        let Some(Res::Fn(item, substs)) = drop_fn(self.cx, ty)? else {
            bail!("`Box` has no `Drop::drop`")
        };
        self.invoke(SiteOrigin::DropFn(origin), item, substs, vec![target])
            .map(drop)
    }

    fn open_drop(
        &mut self,
        at: usize,
        path: &[(u32, u32)],
        ty: Ty,
        place: Option<Place>,
    ) -> Result<()> {
        let variants: Fields = match *self.cx.types.kind(ty) {
            Kind::Tuple(items) => vec![self.cx.types.substs(items).to_vec()].into(),
            Kind::Array(element, len) => vec![vec![element; self.len(len)? as usize]].into(),
            Kind::Adt(..) => match self.adt_variants(ty)? {
                Some(variants) => variants,
                None => return Ok(()),
            },
            ref other => bail!("a partly moved {other:?}"),
        };
        self.each_field(
            (ty, place),
            &variants,
            &mut |lower, field, field_ty, part| {
                let mut sub = path.to_vec();
                sub.push(field);
                lower.drop_path(at, &sub, field_ty, part)
            },
        )
    }

    fn each_field(
        &mut self,
        (ty, place): (Ty, Option<Place>),
        variants: &[Vec<Ty>],
        visit: &mut dyn FnMut(&mut Self, (u32, u32), Ty, Option<Place>) -> Result<()>,
    ) -> Result<()> {
        let enumerated = variants.len() > 1;
        for (variant, fields) in variants.iter().enumerate() {
            let variant = variant as u32;
            let mut each = |lower: &mut Self| -> Result<()> {
                for (index, &field) in fields.iter().enumerate() {
                    let part = place
                        .map(|place| lower.part_place(place, ty, (variant, index as u32)))
                        .transpose()?;
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

    fn when_variant(
        &mut self,
        place: Option<Place>,
        ty: Ty,
        variant: u32,
        run: &mut dyn FnMut(&mut Self) -> Result<()>,
    ) -> Result<()> {
        match self.is_variant(place, ty, variant)? {
            Some(test) => self.when(test, run),
            None => run(self),
        }
    }
}
