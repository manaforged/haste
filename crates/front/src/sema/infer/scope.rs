use std::rc::Rc;

use anyhow::{Result, bail};

use crate::body::adt::AdtDef;
use crate::body::sig::{Generic, GenericKind, ImplHeader, Predicate, SelfKind, Signature};
use crate::body::{Body, Ty as SyntaxTy, TyId};
use crate::names::{ItemRef, ModRef};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;

use super::super::impls::{ImplRef, head};
use super::super::traits::TraitGenerics;
use super::super::ty::{Kind, Mut, Ty};
use super::Infer;

/// Where inference reads names from: the file and module of an item, its type parameters and `Self`.
pub struct Frame<'a> {
    pub src: &'a Source,
    pub home: ModRef,
    pub generics: Vec<(&'a str, Ty)>,
    pub self_ty: Option<Ty>,
    pub block: Option<ImplRef>,
    /// The function or type whose header the frame reads, whose bounds name the traits of its type parameters.
    pub item: Option<ItemRef>,
    /// The function whose body declares the items that the frame's names may refer to.
    pub root: Option<ItemRef>,
}

/// The names of the type parameters of `params` in `src`, each paired with its argument in `args`.
pub fn named_args<'a>(src: &'a Source, params: &[Generic], args: &[Ty]) -> Vec<(&'a str, Ty)> {
    type_params(params).map(|generic| src.text(generic.name)).zip(args.iter().copied()).collect()
}

/// The parameters of `params` that take a type argument.
pub fn type_params(params: &[Generic]) -> impl Iterator<Item = &Generic> {
    params.iter().filter(|generic| generic.kind != GenericKind::Lifetime)
}

/// A header whose generic parameters and where clauses bound the type parameters of the frame.
pub(super) enum Header {
    Fn(Rc<Signature>),
    Adt(Rc<AdtDef>),
    Impl(Rc<ImplHeader>),
    Generics(Rc<TraitGenerics>),
}

impl Header {
    /// The arena, generic parameters and where predicates of the header.
    pub(super) fn parts(&self) -> (&Body, &[Generic], &[Predicate]) {
        match self {
            Header::Fn(sig) => (&sig.body, &sig.generics, &sig.predicates),
            Header::Adt(def) => (&def.body, &def.generics.params, &def.generics.predicates),
            Header::Impl(header) => (&header.body, &header.generics, &header.predicates),
            Header::Generics(def) => (&def.0, &def.1.params, &def.1.predicates),
        }
    }
}

/// The bounds that a header puts on its type parameter `name`: those on the parameter, then those of the where predicates on it.
pub(super) fn named_bounds(arena: &Body, src: &Source, (generics, predicates): (&[Generic], &[Predicate]), name: &str) -> Vec<u32> {
    let declared = generics.iter().filter(|generic| same_param(src.text(generic.name), name)).flat_map(|generic| arena.list(generic.bounds).iter().copied());
    declared.chain(predicate_bounds(arena, src, predicates, name)).collect()
}

/// Whether two parameter names read from source are the same parameter: equal text, and for an anonymous `impl Trait` parameter the same token.
pub(super) fn same_param(a: &str, b: &str) -> bool {
    a == b && (a != "impl" || std::ptr::eq(a, b))
}

/// The bounds on the one generic parameter `generic`: its own, then the where-clause bounds on its name.
pub(super) fn generic_bounds(arena: &Body, src: &Source, predicates: &[Predicate], generic: &Generic) -> Vec<u32> {
    let declared = arena.list(generic.bounds).iter().copied();
    declared.chain(predicate_bounds(arena, src, predicates, src.text(generic.name))).collect()
}

fn predicate_bounds<'b>(arena: &'b Body, src: &'b Source, predicates: &'b [Predicate], name: &'b str) -> impl Iterator<Item = u32> + 'b {
    let named = predicates.iter().filter(move |predicate| predicate.ty.is_some_and(|ty| names_param(arena, src, ty, name)));
    named.flat_map(|predicate| arena.list(predicate.bounds).iter().copied())
}

fn names_param(arena: &Body, src: &Source, ty: TyId, name: &str) -> bool {
    let SyntaxTy::Path(path) = *arena.ty(ty) else { return false };
    let path = arena.path(path);
    match arena.list(path.segments) {
        [segment] => path.qself.is_none() && src.text(arena.segments[*segment as usize].name) == name,
        _ => false,
    }
}

impl<'a> Infer<'_, 'a> {
    /// Whether trait method `member` says `where Self: Sized`, which keeps it out of the trait's vtable.
    pub(crate) fn needs_sized_self(&mut self, member: ItemRef) -> Result<bool> {
        let (sig, src, home) = (self.cx.signature(member)?, &self.cx.file(member).source, self.cx.module_of(member)?);
        let Some(sized) = self.cx.lang_item("sized") else { return Ok(false) };
        let bounds: Vec<u32> = predicate_bounds(&sig.body, src, &sig.predicates, "Self").collect();
        self.in_source(src, |infer| Ok(bounds.into_iter().any(|bound| infer.bound_trait_at(&sig.body, bound, home).is_some_and(|(of, _)| of == sized))))
    }

    /// The frame of the type item or trait `item` with its type parameters `params` set to `args`, `Self` as `self_ty`, and the bounds of `bounds`.
    pub(super) fn item_frame(&mut self, item: ItemRef, (params, args): (&[Generic], &[Ty]), (self_ty, bounds): (Option<Ty>, Option<ItemRef>)) -> Result<Frame<'a>> {
        let file = self.cx.file(item);
        let generics = named_args(&file.source, params, args);
        Ok(Frame { src: &file.source, home: self.cx.module_of(item)?, generics, self_ty, block: None, item: bounds, root: item.declaring_body() })
    }

    /// The frame of function `item` at `substs`: the type parameters of its impl, then its own.
    pub fn frame(&mut self, item: ItemRef, substs: &[Ty]) -> Result<Frame<'a>> {
        let sig = self.cx.signature(item)?;
        let file = self.cx.file(item);
        let home = self.cx.module_of(item)?;
        let (mut generics, self_ty, rest) = self.parent_frame(item, substs)?;
        generics.extend(type_params(&sig.generics).map(|generic| file.source.text(generic.name)).zip(rest.iter().copied()));
        let block = self.cx.owner_impl(item);
        Ok(Frame { src: &file.source, home, generics, self_ty, block, item: Some(item), root: item.declaring_body() })
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
        let frame = Frame { src: &file.source, home: self.cx.impl_module(block)?, generics: generics.clone(), self_ty: None, block: Some(block), item: None, root: block.root() };
        let self_ty = self.with_frame(frame, |infer| infer.lower_ty(&header.body, header.self_ty))?;
        Ok((generics, self_ty))
    }

    /// The headers whose generic parameters and where clauses the frame's type parameters come from: the item's own, its impl's, and its trait's.
    pub(super) fn headers(&mut self) -> Result<Vec<(Header, &'a Source)>> {
        let mut headers = Vec::new();
        if let Some(item) = self.frame_item {
            let header = match self.cx.item(item).1.kind {
                ItemKind::Struct | ItemKind::Enum | ItemKind::Union => Header::Adt(self.cx.adt_def(item)?),
                ItemKind::Type => Header::Generics(Rc::new(self.cx.alias_generics(item)?)),
                _ => Header::Fn(self.cx.signature(item)?),
            };
            headers.push((header, &self.cx.file(item).source));
        }
        if let Some(block) = self.block {
            headers.push((Header::Impl(self.cx.impl_header(block)?), &self.cx.impl_file(block).source));
        } else if let Some(of) = self.frame_item.and_then(|item| self.cx.owner_trait(item)) {
            headers.push((Header::Generics(self.cx.trait_generics(of)?), &self.cx.file(of).source));
        }
        Ok(headers)
    }

    /// The declared type and the value of the const parameter `name` of the frame, when it has one.
    pub(super) fn const_param(&mut self, name: &str) -> Result<Option<(Ty, Ty)>> {
        let Some(&(_, value)) = self.generics.iter().find(|(generic, _)| *generic == name) else { return Ok(None) };
        for (header, src) in self.headers()? {
            let (arena, generics, _) = header.parts();
            let declared = generics.iter().find(|generic| src.text(generic.name) == name).map(|generic| generic.kind);
            if let Some(GenericKind::Const(ty)) = declared {
                return Ok(Some((self.in_source(src, |infer| infer.lower_ty(arena, ty))?, value)));
            }
        }
        Ok(None)
    }

    /// Runs `run` with names read through `frame`.
    pub fn with_frame<T>(&mut self, frame: Frame<'a>, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let saved = self.swap_frame(frame);
        let result = run(self);
        self.swap_frame(saved);
        result
    }

    /// Runs `run` in the current frame with token text read from `src`, the source of the syntax it lowers.
    pub(super) fn in_source<T>(&mut self, src: &'a Source, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if std::ptr::eq(self.src, src) {
            return run(self);
        }
        let frame = Frame { src, home: self.home, generics: self.generics.clone(), self_ty: self.self_ty, block: self.block, item: self.frame_item, root: self.root };
        self.with_frame(frame, run)
    }

    /// Reads names through `frame` from now on, and returns the frame it replaces.
    pub fn swap_frame(&mut self, frame: Frame<'a>) -> Frame<'a> {
        let old = Frame { src: self.src, home: self.home, generics: std::mem::take(&mut self.generics), self_ty: self.self_ty, block: self.block, item: self.frame_item, root: self.root };
        (self.src, self.home, self.generics, self.self_ty, self.block, self.frame_item, self.root) = (frame.src, frame.home, frame.generics, frame.self_ty, frame.block, frame.item, frame.root);
        old
    }

    /// The parameter and return types of the function `item` at `substs`, its receiver first.
    pub fn fn_types(&mut self, item: ItemRef, substs: &[Ty]) -> Result<(Vec<Ty>, Ty)> {
        let sig = self.cx.signature(item)?;
        let frame = self.frame(item, substs)?;
        let foreign = sig.opaque && self.instance.as_ref().is_none_or(|(own, _)| *own != item);
        let (types, opaque) = self.with_frame(frame, |infer| {
            let types = infer.sig_types(&sig)?;
            let opaque = match foreign {
                true => Some((infer.ret_fn_sig(&sig)?, infer.ret_bounds(&sig, types.1)?)),
                false => None,
            };
            Ok((types, opaque))
        })?;
        if let Some((sig, bounds)) = opaque {
            self.opaque_ret(super::opaque::Opaque { item, substs: substs.to_vec(), ret: types.1, sig, bounds })?;
        }
        Ok(types)
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
            _ if sig.is_async => self.fresh(Kind::Var),
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
            if let Some(args) = self.trial(|infer| Ok(infer.match_impl(block, ty)))? {
                return Ok(Some((member, args)));
            }
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
