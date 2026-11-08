use anyhow::{Result, bail};

use crate::body::{Body, Bound, PathId, Ty as SyntaxTy};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, ModRef};
use crate::outline::item::ItemKind;

use super::super::impls::{ImplRef, head};
use super::super::ty::{Kind, Ty};
use super::Infer;
use super::scope::{Frame, type_params};

/// An impl that a trait obligation picked, with its type arguments.
pub type Picked = (ImplRef, Vec<Ty>);

const DEPTH: u32 = 8;

const BUILTIN: [&str; 12] = ["sized", "meta_sized", "pointee_sized", "copy", "clone", "destruct", "tuple_trait", "freeze", "pointee_trait", "unsize", "sync", "unpin"];

impl<'a> Infer<'_, 'a> {
    /// The one impl of trait `of` at trait arguments `args` for `self_ty`, or none when no impl or several apply.
    pub fn select(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<Option<Picked>> {
        self.select_at(of, self_ty, args, 0)
    }

    fn select_at(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty], depth: u32) -> Result<Option<Picked>> {
        let self_ty = self.resolve(self_ty);
        if depth > DEPTH || matches!(self.cx.types.kind(self_ty), Kind::Var(_)) {
            return Ok(None);
        }
        let holders = self.holders(of, self_ty, args);
        let blocks = self.cx.trait_impls(of, head(self.cx.types.kind(self_ty)), &holders);
        let mut found = Vec::new();
        for block in blocks {
            let saved = self.vars.clone();
            let applies = self.impl_applies(block, self_ty, args, depth);
            let vars = std::mem::replace(&mut self.vars, saved);
            if let Some(impl_args) = applies? {
                found.push(((block, impl_args), vars));
            }
        }
        if found.len() > 1 {
            let mut kept = Vec::new();
            for entry in found {
                if !self.blanket(entry.0.0)? {
                    kept.push(entry);
                }
            }
            found = kept;
        }
        if found.len() > 1 {
            found.retain(|((block, _), _)| !self.cx.specializable(*block));
        }
        let Ok([(picked, vars)]) = <[_; 1]>::try_from(found) else { return Ok(None) };
        self.vars = vars;
        Ok(Some(picked))
    }

    fn blanket(&mut self, block: ImplRef) -> Result<bool> {
        let count = self.impl_params(block)?;
        let args: Vec<Ty> = (0..count).map(|_| self.fresh(Kind::Var)).collect();
        let (_, self_ty) = self.impl_frame(block, &args)?;
        Ok(args.contains(&self_ty))
    }

    fn holders(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Vec<u32> {
        let mut crates = vec![of.krate];
        let mut stack: Vec<Ty> = std::iter::once(self_ty).chain(args.iter().copied()).collect();
        while let Some(ty) = stack.pop() {
            match self.cx.types.kind(self.resolve(ty)).clone() {
                Kind::Adt(item, inner) => {
                    if !crates.contains(&item.krate) {
                        crates.push(item.krate);
                    }
                    stack.extend(inner);
                }
                Kind::Ref(_, inner) | Kind::Ptr(_, inner) | Kind::Slice(inner) | Kind::Array(inner, _) | Kind::Pat(inner, ..) => stack.push(inner),
                Kind::Tuple(items) => stack.extend(items),
                _ => {}
            }
        }
        crates
    }

    fn impl_applies(&mut self, block: ImplRef, self_ty: Ty, args: &[Ty], depth: u32) -> Result<Option<Vec<Ty>>> {
        let Some(impl_args) = self.match_trait_impl(block, self_ty, args)? else { return Ok(None) };
        Ok(self.bounds_hold(block, &impl_args, depth)?.then_some(impl_args))
    }

    fn match_trait_impl(&mut self, block: ImplRef, self_ty: Ty, args: &[Ty]) -> Result<Option<Vec<Ty>>> {
        let Some(impl_args) = self.match_impl(block, self_ty) else { return Ok(None) };
        let given = self.impl_trait_args(block, &impl_args, self_ty)?;
        if given.len() != args.len() {
            bail!("an impl gives {} trait arguments where {} are wanted", given.len(), args.len());
        }
        for (found, wanted) in given.iter().zip(args) {
            if self.unify(*found, *wanted).is_err() {
                return Ok(None);
            }
        }
        Ok(Some(impl_args))
    }

    fn impl_frame_of(&mut self, block: ImplRef, impl_args: &[Ty], self_ty: Option<Ty>) -> Result<Frame<'a>> {
        let header = self.cx.impl_header(block)?;
        let file = self.cx.impl_file(block);
        let generics = type_params(&header.generics).map(|generic| file.source.text(generic.name)).zip(impl_args.iter().copied()).collect();
        Ok(Frame { src: &file.source, home: self.cx.impl_module(block), generics, self_ty, block: Some(block) })
    }

    /// The trait arguments of impl `block` at `impl_args`, with the trait's defaults filled in.
    pub fn impl_trait_args(&mut self, block: ImplRef, impl_args: &[Ty], self_ty: Ty) -> Result<Vec<Ty>> {
        let header = self.cx.impl_header(block)?;
        let Some(SyntaxTy::Path(path)) = header.trait_ref.map(|at| *header.body.ty(at)) else { bail!("impl has no trait") };
        let Some(of) = self.cx.impl_trait(block) else { bail!("impl of something that is not a trait") };
        let frame = self.impl_frame_of(block, impl_args, Some(self_ty))?;
        let count = header.body.list(header.body.path(path).segments).len();
        let given = self.with_frame(frame, |infer| infer.segment_args(&header.body, path, count))?;
        self.trait_args(of, given, self_ty)
    }

    /// The arguments `given` to trait `of` for `self_ty`, completed with the trait's defaults.
    pub fn trait_args(&mut self, of: ItemRef, mut given: Vec<Ty>, self_ty: Ty) -> Result<Vec<Ty>> {
        let def = self.cx.trait_generics(of)?;
        let file = self.cx.file(of);
        let home = self.cx.module_of(of)?;
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        for param in params.iter().skip(given.len()) {
            let Some(default) = param.default else { bail!("`{}` needs {} type arguments", self.cx.name(of), params.len()) };
            let generics = params.iter().map(|generic| file.source.text(generic.name)).zip(given.iter().copied()).collect();
            let frame = Frame { src: &file.source, home, generics, self_ty: Some(self_ty), block: None };
            given.push(self.with_frame(frame, |infer| infer.lower_ty(&def.0, default))?);
        }
        Ok(given)
    }

    fn bounds_hold(&mut self, block: ImplRef, impl_args: &[Ty], depth: u32) -> Result<bool> {
        let header = self.cx.impl_header(block)?;
        let frame = self.impl_frame_of(block, impl_args, None)?;
        let home = frame.home;
        let mut wanted = Vec::new();
        for (generic, &arg) in type_params(&header.generics).zip(impl_args) {
            wanted.extend(header.body.list(generic.bounds).iter().map(|&bound| (arg, bound)));
        }
        let mut held = true;
        self.with_frame(frame, |infer| {
            for predicate in &header.predicates {
                let Some(ty) = predicate.ty else { continue };
                let ty = infer.lower_ty(&header.body, ty)?;
                wanted.extend(header.body.list(predicate.bounds).iter().map(|&bound| (ty, bound)));
            }
            for (ty, bound) in wanted {
                held &= infer.bound_holds(&header.body, home, (ty, bound), depth)?;
            }
            Ok(())
        })?;
        Ok(held)
    }

    fn bound_holds(&mut self, arena: &Body, home: ModRef, (ty, bound): (Ty, u32), depth: u32) -> Result<bool> {
        let Bound::Trait { path, maybe: false, .. } = arena.bounds[bound as usize] else { return Ok(true) };
        let Some(of) = self.trait_named(arena, path, home) else { return Ok(true) };
        if self.cx.lang(of).is_some_and(|lang| BUILTIN.contains(&lang)) || self.cx.is_auto_trait(of) {
            return Ok(true);
        }
        if matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)) {
            return Ok(true);
        }
        let count = arena.list(arena.path(path).segments).len();
        let given = self.segment_args(arena, path, count)?;
        let args = self.trait_args(of, given, ty)?;
        Ok(self.select_at(of, ty, &args, depth + 1)?.is_some())
    }

    /// The trait that `path` in `arena` names from module `home`.
    pub fn trait_named(&mut self, arena: &Body, path: PathId, home: ModRef) -> Option<ItemRef> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        match self.cx.paths(home.krate).resolve(&NamePath { segments: &segments, absolute, dollar }, home)? {
            Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Trait => Some(item),
            _ => None,
        }
    }
}
