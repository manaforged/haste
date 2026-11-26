use anyhow::{Result, bail};

use crate::body::sig::{Generic, GenericKind};
use crate::body::{self as syntax, Args, Body, GenericArg, PathId, TyId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, Sym, plain};
use crate::outline::item::ItemKind;

use super::super::super::impls::ImplRef;
use super::super::super::substs::{ItemList, Substs};
use super::super::super::ty::{Kind, Mut, Ty};
use super::super::Infer;
use super::PathSyms;

impl<'a> Infer<'_, 'a> {
    fn within<T>(
        &mut self,
        item: ItemRef,
        params: &[Generic],
        args: &[Ty],
        run: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let frame = self.item_frame(item, (params, args), (None, None))?;
        self.with_frame(frame, run)
    }

    pub(in super::super) fn segment_args(
        &mut self,
        arena: &Body,
        path: PathId,
        count: usize,
    ) -> Result<Vec<Ty>> {
        let Some(&last) = arena.list(arena.path(path).segments)[..count].last() else {
            return Ok(Vec::new());
        };
        let list = match arena.segments[last as usize].args {
            None => return Ok(Vec::new()),
            Some(Args::Angle(list)) => list,
            Some(Args::Paren { inputs, output }) => {
                let (inputs, _) = self.paren_sig(arena, inputs, output)?;
                return Ok(vec![self.cx.types.tuple(&inputs)]);
            }
            Some(Args::ReturnType) => bail!("return type notation is not supported yet"),
        };
        self.arg_list(arena, list)
    }

    pub(in super::super) fn paren_sig(
        &mut self,
        arena: &Body,
        inputs: syntax::List,
        output: Option<TyId>,
    ) -> Result<(Vec<Ty>, Ty)> {
        let inputs = arena
            .list(inputs)
            .iter()
            .map(|&at| self.lower_ty(arena, TyId(at)))
            .collect::<Result<Vec<_>>>()?;
        let output = match output {
            Some(ty) => self.lower_ty(arena, ty)?,
            None => self.cx.types.unit(),
        };
        Ok((inputs, output))
    }

    pub(in super::super) fn path_bindings(
        &mut self,
        arena: &Body,
        path: PathId,
    ) -> Result<Vec<(String, Ty)>> {
        let Some(&last) = arena.list(arena.path(path).segments).last() else {
            return Ok(Vec::new());
        };
        match arena.segments[last as usize].args {
            Some(Args::Paren { inputs, output }) => Ok(vec![(
                "Output".to_string(),
                self.paren_sig(arena, inputs, output)?.1,
            )]),
            Some(Args::Angle(list)) => {
                let mut found = Vec::new();
                for &at in arena.list(list) {
                    let GenericArg::Binding {
                        name,
                        args: None,
                        ty,
                    } = arena.args[at as usize]
                    else {
                        continue;
                    };
                    found.push((self.src.text(name).to_string(), self.lower_ty(arena, ty)?));
                }
                Ok(found)
            }
            Some(Args::ReturnType) => bail!("return type notation is not supported yet"),
            None => Ok(Vec::new()),
        }
    }

    pub(in super::super) fn own_args(
        &mut self,
        item: ItemRef,
        mut given: Vec<Ty>,
    ) -> Result<Vec<Ty>> {
        let count = super::super::scope::type_params(&self.cx.signature(item)?.generics).count();
        if given.len() > count {
            bail!(
                "`{}` takes {count} type arguments, {} given",
                self.cx.name(item),
                given.len()
            );
        }
        given.extend((given.len()..count).map(|_| self.fresh(Kind::Var)));
        Ok(given)
    }

    pub(in super::super) fn arg_list(
        &mut self,
        arena: &Body,
        list: syntax::List,
    ) -> Result<Vec<Ty>> {
        let mut tys = Vec::new();
        for &at in arena.list(list) {
            match arena.args[at as usize] {
                GenericArg::Ty(ty) => tys.push(self.ty_or_const_arg(arena, ty)?),
                GenericArg::Lifetime(_) => {}
                GenericArg::Const(expr) => tys.push(self.const_arg(arena, expr)?),
                GenericArg::Binding { .. } | GenericArg::Constraint { .. } => {}
            }
        }
        Ok(tys)
    }

    pub(in super::super) fn adt_args(
        &mut self,
        item: ItemRef,
        given: Vec<Ty>,
        fresh: bool,
    ) -> Result<Vec<Ty>> {
        let def = self.cx.adt_def(item)?;
        let (arena, parsed) = (&def.body, &def.generics);
        let params: Vec<&Generic> = parsed
            .params
            .iter()
            .filter(|param| param.kind != GenericKind::Lifetime)
            .collect();
        if given.len() > params.len() {
            bail!(
                "`{}` takes {} type arguments, {} given",
                self.cx.name(item),
                params.len(),
                given.len()
            );
        }
        let mut args = given;
        if fresh && args.is_empty() {
            return Ok(params.iter().map(|_| self.fresh(Kind::Var)).collect());
        }
        for param in &params[args.len()..] {
            let Some(default) = param.default else {
                bail!(
                    "`{}` needs {} type arguments, {} given",
                    self.cx.name(item),
                    params.len(),
                    args.len()
                );
            };
            let known = args.clone();
            args.push(self.within(item, &parsed.params, &known, |infer| {
                infer.ty_or_const_arg(arena, default)
            })?);
        }
        Ok(args)
    }

    /// The type that the syntax type `ty` in `arena` names.
    pub fn lower_ty(&mut self, arena: &Body, ty: TyId) -> Result<Ty> {
        match *arena.ty(ty) {
            syntax::Ty::Path(path) => self.path_ty(arena, path),
            syntax::Ty::Paren(inner) => self.lower_ty(arena, inner),
            syntax::Ty::Infer => Ok(self.fresh(Kind::Var)),
            syntax::Ty::Macro { path, open } => self.macro_ty(arena, path, open),
            syntax::Ty::ImplTrait(at, _) => Ok(self.impl_param(at)),
            syntax::Ty::Const(expr) => self.const_arg(arena, expr),
            _ => {
                let kind = self.lower_kind(arena, ty)?;
                Ok(self.cx.types.intern(kind))
            }
        }
    }

    fn lower_kind(&mut self, arena: &Body, ty: TyId) -> Result<Kind> {
        match *arena.ty(ty) {
            syntax::Ty::Ref { mutable, ty, .. } => self
                .lower_ty(arena, ty)
                .map(|inner| Kind::Ref(mutability(mutable), inner)),
            syntax::Ty::Ptr { mutable, ty } => self
                .lower_ty(arena, ty)
                .map(|inner| Kind::Ptr(mutability(mutable), inner)),
            syntax::Ty::Slice(ty) => self.lower_ty(arena, ty).map(Kind::Slice),
            syntax::Ty::Array(ty, len) => self.array_ty(arena, ty, len),
            syntax::Ty::Tuple(list) => {
                let items = self.lower_list(arena, list)?;
                Ok(Kind::Tuple(self.cx.types.intern_substs(&items)))
            }
            syntax::Ty::Never => Ok(Kind::Never),
            syntax::Ty::DynTrait(bounds) => self.object_ty(arena, bounds),
            syntax::Ty::Fn {
                params,
                ret,
                variadic,
                is_unsafe,
            } => self.fn_ptr_ty(arena, (params, ret), variadic, is_unsafe),
            other => bail!("unsupported type {other:?}"),
        }
    }

    fn fn_ptr_ty(
        &mut self,
        arena: &Body,
        (params, ret): (syntax::List, Option<TyId>),
        variadic: bool,
        is_unsafe: bool,
    ) -> Result<Kind> {
        if variadic {
            bail!("variadic function pointer types are not supported yet");
        }
        let params = self.lower_list(arena, params)?;
        let ret = match ret {
            Some(ret) => self.lower_ty(arena, ret)?,
            None => self.cx.types.unit(),
        };
        let params = self.cx.types.intern_substs(&params);
        Ok(if is_unsafe {
            Kind::UnsafeFnPtr(params, ret)
        } else {
            Kind::FnPtr(params, ret)
        })
    }

    fn object_ty(&mut self, arena: &Body, bounds: syntax::List) -> Result<Kind> {
        let (mut principal, mut autos) = (None, Vec::new());
        for &bound in arena.list(bounds) {
            let syntax::Bound::Trait {
                path, maybe: false, ..
            } = arena.bounds[bound as usize]
            else {
                continue;
            };
            let Some(of) = self.trait_named(arena, path, self.home) else {
                bail!("a trait object bound that names no trait");
            };
            if self.cx.is_auto_trait(of) {
                autos.push(of);
            } else if principal.replace((of, path)).is_some() {
                bail!("a trait object with two traits that are not auto traits");
            }
        }
        autos.sort_by_key(|item| (item.krate, item.file, item.item));
        autos.dedup();
        let Some((of, path)) = principal else {
            if autos.is_empty() {
                bail!("a trait object with no trait");
            }
            let first = autos.remove(0);
            let autos = self.cx.types.intern_items(&autos);
            return Ok(Kind::Dyn(first, Substs::EMPTY, ItemList::EMPTY, autos));
        };
        let count = arena.list(arena.path(path).segments).len();
        let given = self.segment_args(arena, path, count)?;
        let wanted =
            super::super::scope::type_params(&self.cx.trait_generics(of)?.1.params).count();
        if given.len() != wanted {
            bail!(
                "a trait object of `{}` that leaves trait arguments to their defaults",
                self.cx.name(of)
            );
        }
        let object = self.fresh(Kind::Var);
        let mut bound = Vec::new();
        for (name, ty) in self.path_bindings(arena, path)? {
            let Some((owner, _)) = self.assoc_owner((of, given.clone()), object, &name, 0)? else {
                bail!("`{}` has no associated type `{name}`", self.cx.name(of));
            };
            let Some(member) = self.cx.trait_member(owner, &name) else {
                bail!("`{}` has no member `{name}`", self.cx.name(owner));
            };
            bound.push((member, ty));
        }
        bound.sort_by_key(|(member, _)| (member.krate, member.file, member.item));
        let (members, tys): (Vec<ItemRef>, Vec<Ty>) = bound.into_iter().unzip();
        let parts: Vec<Ty> = given.into_iter().chain(tys).collect();
        let (parts, members, autos) = (
            self.cx.types.intern_substs(&parts),
            self.cx.types.intern_items(&members),
            self.cx.types.intern_items(&autos),
        );
        Ok(Kind::Dyn(of, parts, members, autos))
    }

    pub(in super::super) fn impl_alias(
        &mut self,
        block: ImplRef,
        (name, own): (&str, &[Ty]),
    ) -> Result<Option<Ty>> {
        let Some(member) = self.cx.member(block, name) else {
            return Ok(None);
        };
        if self.cx.item(member).1.kind != ItemKind::Type {
            return Ok(None);
        }
        let alias = self.cx.alias_def(member)?;
        let home = self.cx.module_of(member)?;
        let src = &self.cx.file(member).source;
        let outer = self.generics.len();
        self.generics.extend(
            alias
                .params
                .iter()
                .map(|param| src.text(param.name))
                .zip(own.iter().copied()),
        );
        let found = self.in_source_home(src, home, |infer| infer.lower_ty(&alias.arena, alias.ty));
        self.generics.truncate(outer);
        found.map(Some)
    }

    pub(in super::super) fn alias(
        &mut self,
        item: ItemRef,
        given: Vec<Ty>,
        fresh: bool,
    ) -> Result<Ty> {
        let alias = self.cx.alias_def(item)?;
        let (params, arena, ty, declared) =
            (&alias.params, &alias.arena, alias.ty, &alias.generics);
        if let Some(definer) = self.opaque_alias(item, params, arena) {
            return super::super::opaque::revealed(self.cx, definer, Substs::EMPTY);
        }
        let file = self.cx.file(item);
        let mut args = given;
        let infer_all = fresh && args.is_empty();
        for param in &params[args.len().min(params.len())..] {
            let arg = match param.default.filter(|_| !infer_all) {
                Some(default) => {
                    let known = args.clone();
                    self.within(item, &params[..known.len()], &known, |infer| {
                        infer.ty_or_const_arg(&declared.0, default)
                    })?
                }
                None => self.fresh(Kind::Var),
            };
            args.push(arg);
        }
        let names: Vec<&str> = params
            .iter()
            .map(|generic| file.source.text(generic.name))
            .collect();
        let generics = names.into_iter().zip(args).collect();
        let frame = super::super::scope::Frame {
            src: &file.source,
            home: self.cx.module_of(item)?,
            generics,
            self_ty: None,
            block: None,
            item: Some(item),
            root: item.declaring_body(),
        };
        self.with_frame(frame, |infer| infer.lower_ty(arena, ty))
    }

    fn opaque_alias(&self, item: ItemRef, params: &[Generic], arena: &Body) -> Option<ItemRef> {
        if !params.is_empty()
            || item.outer().is_some()
            || !arena
                .tys
                .iter()
                .any(|ty| matches!(ty, syntax::Ty::ImplTrait(..)))
        {
            return None;
        }
        let definer = super::super::opaque::opaque_definer(self.cx, item)?;
        self.instance
            .as_ref()
            .is_none_or(|(at, _)| *at != definer)
            .then_some(definer)
    }

    fn impl_param(&mut self, at: u32) -> Ty {
        let key = self.src.text(at);
        match self
            .generics
            .iter()
            .find(|(name, _)| super::super::scope::same_param(name, key))
        {
            Some(&(_, ty)) => ty,
            None => {
                let hidden = self.fresh(Kind::Var);
                self.hidden.push((at, hidden));
                hidden
            }
        }
    }

    fn array_ty(&mut self, arena: &Body, ty: TyId, len: syntax::ExprId) -> Result<Kind> {
        Ok(Kind::Array(
            self.lower_ty(arena, ty)?,
            self.const_arg(arena, len)?,
        ))
    }

    fn lower_list(&mut self, arena: &Body, list: syntax::List) -> Result<Vec<Ty>> {
        arena
            .list(list)
            .iter()
            .map(|&item| self.lower_ty(arena, TyId(item)))
            .collect()
    }

    fn path_ty(&mut self, arena: &Body, path: PathId) -> Result<Ty> {
        match arena.path(path).qself {
            Some((qself, Some(trait_path))) => {
                let Some(&last) = arena.list(arena.path(path).segments).last() else {
                    bail!("a projection with no name");
                };
                self.projection(arena, (qself, trait_path), last)
            }
            Some((qself, None)) => self.qself_ty(arena, (qself, path)),
            None => {
                let count = arena.path(path).segments.len as usize;
                self.named_ty(arena, path, count, false)
            }
        }
    }

    fn qself_ty(&mut self, arena: &Body, (qself, path): (TyId, PathId)) -> Result<Ty> {
        let mut inner = qself;
        while let syntax::Ty::Paren(next) = *arena.ty(inner) {
            inner = next;
        }
        let owner = match *arena.ty(inner) {
            syntax::Ty::Path(named) if arena.path(named).qself.is_none() => {
                self.segments(arena, named).0
            }
            _ => PathSyms::EMPTY,
        };
        let name = self.segments(arena, path).0;
        if let ([owner], [name]) = (&owner[..], &name[..])
            && let Some(ty) = self.scoped_ty(&[*owner, *name])?
        {
            return Ok(ty);
        }
        bail!(
            "ambiguous associated type `<{}>::{}`",
            self.spell(&owner),
            self.spell(&name)
        )
    }

    pub(in super::super) fn named_ty(
        &mut self,
        arena: &Body,
        path: PathId,
        count: usize,
        fresh: bool,
    ) -> Result<Ty> {
        let (mut segments, absolute, dollar) = self.segments(arena, path);
        segments.truncate(count);
        if let Some(ty) = self.scoped_ty(&segments)? {
            return Ok(ty);
        }
        let first_tok = Self::path_first_tok(arena, path);
        let def = self.type_def_checked(
            &NamePath {
                segments: &segments,
                absolute,
                dollar,
            },
            first_tok,
        )?;
        let def = match (def, &segments[..]) {
            (Some(Def::Module(_)), [name]) => crate::names::Symbols::primitive(*name)
                .map(Def::Primitive)
                .or(def),
            (None, [name]) => self
                .def_site(arena, path, *name)
                .and_then(|found| found.types)
                .map(|binding| binding.def),
            _ => def,
        };
        match def {
            Some(Def::Primitive(primitive)) => Ok(self.cx.types.prim(primitive)),
            Some(Def::Item(item))
                if matches!(
                    self.cx.item(item).1.kind,
                    ItemKind::Type | ItemKind::Struct | ItemKind::Enum | ItemKind::Union
                ) =>
            {
                let given = self.segment_args(arena, path, count)?;
                self.item_ty(item, given, fresh)
            }
            other => bail!("cannot lower type `{}` ({other:?})", self.spell(&segments)),
        }
    }

    fn item_ty(&mut self, item: ItemRef, given: Vec<Ty>, fresh: bool) -> Result<Ty> {
        if self.cx.item(item).1.kind == ItemKind::Type {
            if self.cx.foreign(item) {
                return Ok(self.cx.types.adt(item, &[]));
            }
            return self.alias(item, given, fresh);
        }
        let args = self.adt_args(item, given, fresh)?;
        Ok(self.cx.types.adt(item, &args))
    }

    fn scoped_ty(&mut self, segments: &[Sym]) -> Result<Option<Ty>> {
        if self.self_ty.is_none() && self.generics.is_empty() {
            return Ok(None);
        }
        let symbols = &self.cx.names.symbols;
        match *segments {
            [one] => {
                let text = symbols.text(one);
                if text == "Self" && self.self_ty.is_some() {
                    return Ok(self.self_ty);
                }
                Ok(self
                    .generics
                    .iter()
                    .find(|(generic, _)| *generic == text)
                    .map(|&(_, ty)| ty))
            }
            [param, name] => {
                let param_text = symbols.text(param);
                if param_text == "Self"
                    || self
                        .generics
                        .iter()
                        .any(|(generic, _)| *generic == param_text)
                {
                    let (param, name) = (param_text.to_string(), symbols.text(name).to_string());
                    self.param_projection(&param, &name)
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    pub(in super::super) fn projection(
        &mut self,
        arena: &Body,
        (qself, trait_path): (TyId, PathId),
        segment: u32,
    ) -> Result<Ty> {
        let ty = self.lower_ty(arena, qself)?;
        let (segments, absolute, dollar) = self.segments(arena, trait_path);
        let found = self.trait_at(
            &NamePath {
                segments: &segments,
                absolute,
                dollar,
            },
            self.home,
        );
        let name = plain(self.src.text(arena.segments[segment as usize].name));
        let Some(item) = found else {
            bail!(
                "projection through `{}`, which is not a trait",
                self.spell(&segments)
            );
        };
        let own = match arena.segments[segment as usize].args {
            Some(Args::Angle(list)) => self.arg_list(arena, list)?,
            _ => Vec::new(),
        };
        if self.cx.lang(item) == Some("pointee_trait") && name == "Metadata" {
            return self.metadata_of(ty);
        }
        let args = self.bound_args(item, arena, trait_path, ty)?;
        self.project_own(item, ty, args, (name, &own))
    }

    /// The pointer metadata of `ty`: a length for slices and `str`, nothing for sized types, or a variable until `ty` is known.
    pub fn metadata_of(&mut self, ty: Ty) -> Result<Ty> {
        if let Some(found) = self.pointee_metadata(ty)? {
            return Ok(found);
        }
        let tailed = self.tail_ty(ty)?;
        let tail = self.resolve(tailed);
        let result = self.fresh(Kind::Var);
        self.metadata.push((tail, result));
        Ok(result)
    }

    pub(in super::super) fn param_metadata(&mut self, tail: Ty) -> Result<Ty> {
        if self.poly.is_none() {
            bail!("metadata of open parameter requires concrete type");
        }
        if !self.poly_sized(tail) {
            self.poly_fallback();
        }
        Ok(self.cx.types.unit())
    }
}

fn mutability(mutable: bool) -> Mut {
    if mutable { Mut::Mut } else { Mut::Not }
}
