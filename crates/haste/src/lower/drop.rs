use anyhow::{Result, bail};
use haste_codegen::{IntCC, StackSlot, Type, Value};

use crate::body::ExprId;
use crate::sema::infer::{Fields, Res, Shape, drop_fn, field_types, has_drop_impl, needs_drop};
use crate::sema::ty::{Kind, Ty};

use super::moves::{BOX_STEP, MovePath, Root};
use super::place::{Base, Place};
use super::{Lower, Repr, repr};

/// A value that the body owns and drops when its scope ends: where it lives, and the flags that say which of it is still there.
#[derive(Clone, Debug)]
pub struct Owned {
    root: Root,
    ty: Ty,
    place: Option<Place>,
    flag: StackSlot,
    parts: Vec<(MovePath, StackSlot)>,
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
        let flag = self.flag(init);
        let paths = self.moved.get(&root).cloned().unwrap_or_default();
        let parts = paths.into_iter().map(|path| (path, self.flag(init))).collect();
        let life = match root {
            Root::Local(_) => Life::Scope,
            Root::Temp(expr) if self.extended.contains(&expr) => Life::Extended,
            Root::Temp(_) | Root::Held(_) => Life::Statement,
        };
        self.owned.push(Owned { root, ty, place, flag, parts, life });
        Ok(())
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

    fn owner(&self, root: Root) -> Option<usize> {
        self.owned.iter().rposition(|owned| owned.root == root)
    }

    /// Marks the part `path` of `root` as moved out, so that its scope does not drop it.
    pub fn moved_out(&mut self, root: Root, path: &[(u32, u32)]) -> Result<()> {
        if let Some(slot) = self.move_flag(root, path)? {
            self.set(slot, false);
        }
        Ok(())
    }

    fn move_flag(&self, root: Root, path: &[(u32, u32)]) -> Result<Option<StackSlot>> {
        let Some(at) = self.owner(root) else {
            return Ok(None);
        };
        Ok(Some(match path {
            [] => self.owned[at].flag,
            _ => match self.owned[at].parts.iter().find(|(part, _)| part == path) {
                Some(&(_, slot)) => slot,
                None => {
                    bail!("a move out of {root:?} at {path:?} that the move analysis did not find")
                }
            },
        }))
    }

    /// Marks the part `path` of `root` as moved out when every `(which, alternative)` of `when` holds.
    fn moved_out_when(&mut self, root: Root, path: &[(u32, u32)], when: &[(Value, i64)]) -> Result<()> {
        let Some(slot) = self.move_flag(root, path)? else {
            return Ok(());
        };
        let mut taken = self.ins().iconst(Type::I8, 1);
        for &(which, alternative) in when {
            let holds = self.ins().icmp_imm(IntCC::Equal, which, alternative);
            taken = self.ins().band(taken, holds);
        }
        let present = self.ins().stack_load(Type::I8, slot, 0);
        let zero = self.ins().iconst(Type::I8, 0);
        let kept = self.ins().select(taken, zero, present);
        self.ins().stack_store(kept, slot, 0);
        Ok(())
    }

    /// Marks the part `path` of `root` and everything below it as present again, after an assignment.
    pub fn reinit(&mut self, root: Root, path: &[(u32, u32)]) {
        let Some(at) = self.owner(root) else { return };
        let mut slots: Vec<StackSlot> = self.owned[at].parts.iter().filter(|(part, _)| part.starts_with(path)).map(|&(_, slot)| slot).collect();
        if path.is_empty() {
            slots.push(self.owned[at].flag);
        }
        for slot in slots {
            self.set(slot, true);
        }
    }

    /// Marks every owned value from index `mark` on as absent, on a path where their bindings did not happen.
    pub fn forget_from(&mut self, mark: usize) {
        let slots: Vec<StackSlot> = self.owned[mark..].iter().map(|owned| owned.flag).collect();
        for slot in slots {
            self.set(slot, false);
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
        let flag = self.owned[at].flag;
        self.guarded(flag, |lower| lower.drop_path(at, path, ty, Some(place)))?;
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
        let Owned { ty, place, flag, .. } = self.owned[at].clone();
        self.guarded(flag, |lower| {
            lower.drop_path(at, &[], ty, place)?;
            lower.set(flag, false);
            Ok(())
        })
    }

    fn guarded(&mut self, flag: StackSlot, run: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let present = self.ins().stack_load(Type::I8, flag, 0);
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
        let below = parts.iter().any(|(part, _)| part.len() > path.len() && part.starts_with(path));
        let flag = parts.iter().find(|(part, _)| part == path).map(|&(_, slot)| slot).filter(|_| !path.is_empty());
        let inside: MovePath = path.iter().copied().chain([BOX_STEP]).collect();
        let into_box = parts.iter().any(|(part, _)| part.starts_with(&inside));
        let opened = below && !has_drop_impl(self.cx, ty)?;
        let origin = super::reloc::TyOrigin::from_root(self.owned[at].root, self.typed).stepped(path);
        let run = |lower: &mut Self| match (into_box, opened) {
            (true, _) => lower.open_box(at, &inside, ty, place, origin.clone()),
            (false, true) => lower.open_drop(at, path, ty, place),
            (false, false) => lower.glue(ty, place, origin.clone()),
        };
        match flag {
            Some(flag) => self.guarded(flag, run),
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
            Kind::Adt(item, _) if self.cx.lang(item) == Some("manually_drop") || self.cx.item(item).1.kind == crate::outline::item::ItemKind::Union => return Ok(()),
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
