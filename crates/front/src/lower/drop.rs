use anyhow::{Result, bail};
use haste_codegen::{StackSlot, Type, Value};

use crate::body::ExprId;
use crate::sema::infer::{Res, Shape, field_types, has_drop_impl, needs_drop};
use crate::sema::ty::{Kind, Ty};

use super::moves::{MovePath, Root};
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
            Root::Temp(_) => Life::Statement,
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
        let Some(at) = self.owner(root) else { return Ok(()) };
        let slot = match path {
            [] => self.owned[at].flag,
            _ => match self.owned[at].parts.iter().find(|(part, _)| part == path) {
                Some(&(_, slot)) => slot,
                None => bail!("a move out of {root:?} at {path:?} that the move analysis did not find"),
            },
        };
        self.set(slot, false);
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
            self.glue(ty, Some(place))?;
        }
        self.write(place, value);
        if let Some((root, path)) = tracked {
            self.reinit(root, &path);
        }
        Ok(true)
    }

    fn drop_replaced(&mut self, root: Root, path: &[(u32, u32)], ty: Ty, place: Place) -> Result<bool> {
        let Some(at) = self.owner(root) else { return Ok(false) };
        let flag = self.owned[at].flag;
        self.guarded(flag, |lower| lower.drop_path(at, path, ty, Some(place)))?;
        Ok(true)
    }

    /// Applies the moves that the bindings of a matched pattern made.
    pub fn commit_moves(&mut self) -> Result<()> {
        for (root, path) in std::mem::take(&mut self.pending) {
            self.moved_out(root, &path)?;
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
        let opened = below && !has_drop_impl(self.cx, ty)?;
        let run = |lower: &mut Self| if opened { lower.open_drop(at, path, ty, place) } else { lower.glue(ty, place) };
        match flag {
            Some(flag) => self.guarded(flag, run),
            None => run(self),
        }
    }

    fn open_drop(&mut self, at: usize, path: &[(u32, u32)], ty: Ty, place: Option<Place>) -> Result<()> {
        let Some(place) = place else { bail!("a partly moved value with no bytes") };
        let variants = match self.cx.types.kind(ty).clone() {
            Kind::Tuple(items) => vec![items],
            Kind::Adt(item, args) => field_types(self.cx, item, &args)?,
            other => bail!("a partly moved {other:?}"),
        };
        self.each_field((ty, place), variants, &mut |lower, field, field_ty, part| {
            let mut sub = path.to_vec();
            sub.push(field);
            lower.drop_path(at, &sub, field_ty, Some(part))
        })
    }

    fn each_field(&mut self, (ty, place): (Ty, Place), variants: Vec<Vec<Ty>>, visit: &mut dyn FnMut(&mut Self, (u32, u32), Ty, Place) -> Result<()>) -> Result<()> {
        let enumerated = variants.len() > 1;
        for (variant, fields) in variants.into_iter().enumerate() {
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
    pub fn glue(&mut self, ty: Ty, place: Option<Place>) -> Result<()> {
        if !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        let Some(glue) = self.cx.lang_item("drop_glue") else { bail!("core has no drop glue") };
        let target = self.address_or_dangling(place, ty)?;
        self.invoke(glue, &[ty], vec![target]).map(drop)
    }

    /// Lowers the body of the drop glue for `ty` whose target is `target`: `Drop::drop`, then the parts.
    pub fn glue_body(&mut self, ty: Ty, target: Option<Value>) -> Result<()> {
        let Some(target) = target else { return Ok(()) };
        let glue = match &self.typed.shape {
            Shape::Glue(glue) => glue.clone(),
            _ => Default::default(),
        };
        let drop_fn = match glue.drop {
            Some(Res::Fn(item, substs)) => Some((item, substs)),
            _ => None,
        };
        if let Some(owned) = self.cx.boxed(ty) {
            self.box_contents(owned, target)?;
            if let Some((item, substs)) = drop_fn {
                self.invoke(item, &substs, vec![target])?;
            }
            return Ok(());
        }
        if let Some((item, substs)) = drop_fn {
            self.invoke(item, &substs, vec![target])?;
        }
        self.drop_parts(ty, target)
    }

    fn box_contents(&mut self, inner: Ty, target: Value) -> Result<()> {
        let reference = self.cx.types.intern(Kind::Ref(crate::sema::ty::Mut::Mut, inner));
        let contents = match repr(self.cx, reference)? {
            Repr::Scalar(_) => self.ins().load(Type::I64, target, 0),
            _ => target,
        };
        let place = Place { base: Base::Addr(contents), offset: 0, repr: repr(self.cx, inner).unwrap_or(Repr::Zst) };
        match self.cx.types.kind(inner) {
            Kind::Slice(_) | Kind::Prim(crate::names::Primitive::Str) | Kind::Dyn(..) => self.glue_wide(inner, contents),
            _ => self.glue(inner, Some(place)),
        }
    }

    fn glue_wide(&mut self, ty: Ty, wide: Value) -> Result<()> {
        if !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        let Some(glue) = self.cx.lang_item("drop_glue") else { bail!("core has no drop glue") };
        self.invoke(glue, &[ty], vec![wide]).map(drop)
    }

    fn drop_parts(&mut self, ty: Ty, target: Value) -> Result<()> {
        match self.cx.types.kind(ty).clone() {
            Kind::Array(inner, count) => {
                let count = self.len(count)?;
                let count = self.ins().iconst(Type::I64, count);
                self.drop_elements(inner, target, count)
            }
            Kind::Slice(inner) => {
                let (data, len) = (self.ins().load(Type::I64, target, 0), self.ins().load(Type::I64, target, 8));
                self.drop_elements(inner, data, len)
            }
            Kind::Tuple(_) | Kind::Adt(..) => {
                let place = Place { base: Base::Addr(target), offset: 0, repr: repr(self.cx, ty)? };
                self.drop_fields(ty, place)
            }
            Kind::Closure(..) => match self.cx.closure_env(ty) {
                Some(env) => self.drop_parts(env, target),
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
        let signature = super::signature_of(self.cx, None, [pointer])?;
        self.when(code, |lower| lower.call_code(code, (signature, None), vec![data]).map(drop))
    }

    fn drop_fields(&mut self, ty: Ty, place: Place) -> Result<()> {
        let variants = match self.cx.types.kind(ty).clone() {
            Kind::Tuple(items) => vec![items],
            Kind::Adt(item, _) if self.cx.lang(item) == Some("manually_drop") || self.cx.item(item).1.kind == crate::outline::item::ItemKind::Union => return Ok(()),
            Kind::Adt(item, args) => field_types(self.cx, item, &args)?,
            _ => return Ok(()),
        };
        self.each_field((ty, place), variants, &mut |lower, _, field_ty, part| lower.glue(field_ty, Some(part)))
    }

    fn drop_elements(&mut self, inner: Ty, data: Value, count: Value) -> Result<()> {
        if !needs_drop(self.cx, inner)? {
            return Ok(());
        }
        let size = self.cx.layout(inner)?.size.bytes() as i64;
        let element = repr(self.cx, inner)?;
        let mut failed = None;
        self.each(count, |lower, index| {
            let offset = lower.ins().imul_imm(index, size);
            let at = lower.ins().iadd(data, offset);
            if let Err(error) = lower.glue(inner, Some(Place { base: Base::Addr(at), offset: 0, repr: element })) {
                failed = Some(error);
            }
        });
        failed.map_or(Ok(()), Err)
    }
}
