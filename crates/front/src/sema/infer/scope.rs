use anyhow::{Result, bail};

use crate::body::Body;
use crate::body::sig::{Generic, GenericKind, SelfKind, Signature};
use crate::names::{ItemRef, ModRef};
use crate::outline::source::Source;

use super::super::impls::{ImplRef, head};
use super::super::ty::{Kind, Mut, Ty};
use super::Infer;

/// Where inference reads names from: the file and module of an item, its type parameters and `Self`.
pub struct Frame<'a> {
    pub src: &'a Source,
    pub home: ModRef,
    pub generics: Vec<(&'a str, Ty)>,
    pub self_ty: Option<Ty>,
    pub block: Option<ImplRef>,
}

/// The parameters of `params` that take a type argument.
pub fn type_params(params: &[Generic]) -> impl Iterator<Item = &Generic> {
    params.iter().filter(|generic| generic.kind != GenericKind::Lifetime)
}

impl<'a> Infer<'_, 'a> {
    /// The frame of function `item` at `substs`: the type parameters of its impl, then its own.
    pub fn frame(&mut self, item: ItemRef, substs: &[Ty]) -> Result<Frame<'a>> {
        let sig = self.cx.signature(item)?;
        let file = self.cx.file(item);
        let home = self.cx.module_of(item)?;
        let (mut generics, self_ty, rest) = self.parent_frame(item, substs)?;
        generics.extend(type_params(&sig.generics).map(|generic| file.source.text(generic.name)).zip(rest.iter().copied()));
        let block = self.cx.owner_impl(item);
        Ok(Frame { src: &file.source, home, generics, self_ty, block })
    }

    fn parent_frame<'s>(&mut self, item: ItemRef, substs: &'s [Ty]) -> Result<(Vec<(&'a str, Ty)>, Option<Ty>, &'s [Ty])> {
        if let Some(block) = self.cx.owner_impl(item) {
            let count = self.impl_params(block)?;
            if substs.len() < count {
                bail!("{} needs {count} impl type arguments, {} given", self.cx.name(item), substs.len());
            }
            let (outer, rest) = substs.split_at(count);
            let (generics, self_ty) = self.impl_frame(block, outer)?;
            return Ok((generics, Some(self_ty), rest));
        }
        let Some(of) = self.cx.owner_trait(item) else { return Ok((Vec::new(), None, substs)) };
        let def = self.cx.trait_generics(of)?;
        let count = 1 + type_params(&def.1.params).count();
        if substs.len() < count {
            bail!("{} needs `Self` and the trait's type arguments", self.cx.name(item));
        }
        let (outer, rest) = substs.split_at(count);
        let src = &self.cx.file(of).source;
        let generics = type_params(&def.1.params).map(|generic| src.text(generic.name)).zip(outer[1..].iter().copied()).collect();
        Ok((generics, Some(outer[0]), rest))
    }

    /// How many type arguments impl `block` takes.
    pub fn impl_params(&mut self, block: ImplRef) -> Result<usize> {
        Ok(type_params(&self.cx.impl_header(block)?.generics).count())
    }

    /// The type parameters of impl `block` set to `args`, and its self type at them.
    pub fn impl_frame(&mut self, block: ImplRef, args: &[Ty]) -> Result<(Vec<(&'a str, Ty)>, Ty)> {
        let header = self.cx.impl_header(block)?;
        let file = self.cx.impl_file(block);
        let generics: Vec<(&'a str, Ty)> = type_params(&header.generics).map(|generic| file.source.text(generic.name)).zip(args.iter().copied()).collect();
        let frame = Frame { src: &file.source, home: self.cx.impl_module(block), generics: generics.clone(), self_ty: None, block: Some(block) };
        let self_ty = self.with_frame(frame, |infer| infer.lower_ty(&header.body, header.self_ty))?;
        Ok((generics, self_ty))
    }

    /// Runs `run` with names read through `frame`.
    pub fn with_frame<T>(&mut self, frame: Frame<'a>, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let saved = self.swap_frame(frame);
        let result = run(self);
        self.swap_frame(saved);
        result
    }

    /// Reads names through `frame` from now on, and returns the frame it replaces.
    pub fn swap_frame(&mut self, frame: Frame<'a>) -> Frame<'a> {
        let old = Frame { src: self.src, home: self.home, generics: std::mem::take(&mut self.generics), self_ty: self.self_ty, block: self.block };
        (self.src, self.home, self.generics, self.self_ty, self.block) = (frame.src, frame.home, frame.generics, frame.self_ty, frame.block);
        old
    }

    /// The parameter and return types of the function `item` at `substs`, its receiver first.
    pub fn fn_types(&mut self, item: ItemRef, substs: &[Ty]) -> Result<(Vec<Ty>, Ty)> {
        let sig = self.cx.signature(item)?;
        let frame = self.frame(item, substs)?;
        self.with_frame(frame, |infer| infer.sig_types(&sig))
    }

    /// The parameter types of `sig`, its receiver first, and its return type.
    pub fn sig_types(&mut self, sig: &Signature) -> Result<(Vec<Ty>, Ty)> {
        let mut params = Vec::new();
        if let Some((kind, _)) = sig.receiver {
            params.push(self.receiver_ty(&sig.body, kind)?);
        }
        for &(_, ty) in &sig.params {
            params.push(self.lower_ty(&sig.body, ty)?);
        }
        let ret = match sig.ret {
            Some(ty) => self.lower_ty(&sig.body, ty)?,
            None => self.cx.types.unit(),
        };
        Ok((params, ret))
    }

    /// The type of a `self` parameter of `kind`.
    pub fn receiver_ty(&mut self, arena: &Body, kind: SelfKind) -> Result<Ty> {
        let Some(self_ty) = self.self_ty else { bail!("`self` outside an impl") };
        Ok(match kind {
            SelfKind::Value => self_ty,
            SelfKind::Ref => self.cx.types.intern(Kind::Ref(Mut::Not, self_ty)),
            SelfKind::RefMut => self.cx.types.intern(Kind::Ref(Mut::Mut, self_ty)),
            SelfKind::Typed(ty) => return self.lower_ty(arena, ty),
        })
    }

    /// The inherent member `name` of type `ty`, with the type arguments of the impl that holds it.
    pub fn inherent_member(&mut self, ty: Ty, name: &str) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let ty = self.resolve(ty);
        let Some(head) = head(self.cx.types.kind(ty)) else { return Ok(None) };
        for block in self.cx.inherent(head) {
            let Some(member) = self.cx.member(block, name) else { continue };
            let saved = self.vars.clone();
            if let Some(args) = self.match_impl(block, ty) {
                return Ok(Some((member, args)));
            }
            self.vars = saved;
        }
        Ok(None)
    }

    /// Fresh type arguments for impl `block` that make its self type `ty`, or none when no arguments do.
    pub fn match_impl(&mut self, block: ImplRef, ty: Ty) -> Option<Vec<Ty>> {
        let count = self.impl_params(block).ok()?;
        let args: Vec<Ty> = (0..count).map(|_| self.fresh(Kind::Var)).collect();
        let (_, self_ty) = self.impl_frame(block, &args).ok()?;
        self.unify(self_ty, ty).ok().map(|_| args)
    }
}
