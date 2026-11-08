use anyhow::{Result, bail};
use rustc_lexer::TokenKind;

use crate::body::sig::{Generic, GenericKind};
use crate::body::{self as syntax, Args, Body, GenericArg, PathId, TyId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, Primitive, Sym, Symbols, plain};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;

use super::super::impls::ImplRef;
use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Res};

/// The interned segments of `path` in `arena`, whether it starts with `::`, and the crate a leading `$crate` names.
pub fn segments(symbols: &Symbols, src: &Source, arena: &Body, path: PathId) -> (Vec<Sym>, bool, Option<u32>) {
    let path = arena.path(path);
    let segments: Vec<Sym> = arena.list(path.segments).iter().map(|&at| symbols.intern(plain(src.text(arena.segments[at as usize].name)))).collect();
    let first = arena.list(path.segments).first().map(|&at| arena.segments[at as usize].name).unwrap_or(0);
    let dollar = first.checked_sub(1).and_then(|at| src.dollar(at));
    (segments, path.global, dollar)
}

impl<'a> Infer<'_, 'a> {
    pub(super) fn segments(&self, arena: &Body, path: PathId) -> (Vec<Sym>, bool, Option<u32>) {
        segments(&self.cx.names.symbols, self.src, arena, path)
    }

    /// The item a value path names, resolved from the body's module.
    pub fn value_item(&mut self, arena: &Body, path: PathId) -> Result<Def> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        self.value_def(&segments, absolute, dollar)
    }

    /// The item in the value namespace that the path `segments` names.
    pub(super) fn value_def(&mut self, segments: &[Sym], absolute: bool, dollar: Option<u32>) -> Result<Def> {
        let paths = self.cx.paths(self.home.krate);
        let Some((last, prefix)) = segments.split_last() else { bail!("empty path") };
        if let (true, Some(found)) = (prefix.is_empty(), self.declared_item(self.cx.names.symbols.text(*last))) {
            return Ok(Def::Item(found));
        }
        let found = if prefix.is_empty() {
            paths.first(*last, self.home, &NamePath { segments, absolute, dollar }).values
        } else {
            let head = NamePath { segments: prefix, absolute, dollar };
            paths.resolve(&head, self.home).map(|def| paths.member(def, *last)).and_then(|entry| entry.values)
        };
        match found {
            Some(binding) => Ok(binding.def),
            None => bail!("cannot find value `{}`", self.spell(segments)),
        }
    }

    /// The item named `name` that a block around the use declares, or else some block of the body.
    pub(super) fn declared_item(&self, name: &str) -> Option<ItemRef> {
        let scoped = self.items.iter().rev().find(|(bound, _)| *bound == name).map(|(_, item)| *item);
        scoped.or_else(|| self.root.and_then(|root| self.cx.body_item(root, name)))
    }

    /// Resolves the value path expression `expr` to a local or an item, and returns its type.
    pub fn path_expr(&mut self, expr: syntax::ExprId, path: PathId) -> Result<Ty> {
        let (ty, res) = self.value_path(path)?;
        if let Res::Local(local) = res {
            self.note_read(expr, local);
        }
        self.out.res.insert(expr, res);
        Ok(ty)
    }

    /// The type and resolution of the value path `path`: a local, a constructor, a function or a constant.
    pub(super) fn value_path(&mut self, path: PathId) -> Result<(Ty, Res)> {
        let body = self.body;
        let entry = body.path(path);
        let single = entry.qself.is_none() && !entry.global && entry.segments.len == 1;
        if single && let Some(found) = self.bound_value(crate::names::plain(self.src.text(body.segments[body.list(entry.segments)[0] as usize].name)))? {
            return Ok(found);
        }
        if let Some(ctor) = self.ctor(body, path, true)? {
            return self.ctor_value(ctor);
        }
        match self.value_item(body, path) {
            Ok(def) => self.item_value(path, def),
            Err(error) => self.assoc_value(path)?.ok_or(error),
        }
    }

    fn item_value(&mut self, path: PathId, def: Def) -> Result<(Ty, Res)> {
        match def {
            Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Fn => {
                let given = self.segment_args(self.body, path, self.body.path(path).segments.len as usize)?;
                self.fn_item(item, Vec::new(), given)
            }
            Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Const => self.const_item(item, Vec::new()),
            other => bail!("unsupported value {other:?}"),
        }
    }

    fn bound_value(&mut self, name: &str) -> Result<Option<(Ty, Res)>> {
        if let Some(&(_, local)) = self.scopes.iter().rev().find(|(bound, _)| *bound == name) {
            self.note_use(local);
            return Ok(Some((self.out.locals[local as usize].ty, Res::Local(local))));
        }
        Ok(self.const_param(name)?.map(|(ty, value)| (ty, Res::Const(value))))
    }

    pub(super) fn const_item(&mut self, item: ItemRef, outer: Vec<Ty>) -> Result<(Ty, Res)> {
        let ty = self.fn_types(item, &outer)?.1;
        Ok((ty, Res::Fn(item, outer)))
    }

    /// The function item `item` inside the type arguments `outer`, with its own type parameters set to `given`.
    pub(super) fn fn_item(&mut self, item: ItemRef, outer: Vec<Ty>, given: Vec<Ty>) -> Result<(Ty, Res)> {
        let mut substs = outer;
        substs.extend(self.own_args(item, given)?);
        Ok((self.cx.types.intern(Kind::FnDef(item, substs.clone())), Res::Fn(item, substs)))
    }

    /// Runs `run` as if inside the type item `item`, with its type parameters set to `args`.
    fn within<T>(&mut self, item: ItemRef, params: &[Generic], args: &[Ty], run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let frame = self.item_frame(item, (params, args), (None, None))?;
        self.with_frame(frame, run)
    }

    /// The type and const arguments of segment `count` of `path`, skipping lifetimes.
    pub(super) fn segment_args(&mut self, arena: &Body, path: PathId, count: usize) -> Result<Vec<Ty>> {
        let Some(&last) = arena.list(arena.path(path).segments)[..count].last() else { return Ok(Vec::new()) };
        let list = match arena.segments[last as usize].args {
            None => return Ok(Vec::new()),
            Some(Args::Angle(list)) => list,
            Some(other) => bail!("unsupported generic arguments {other:?}"),
        };
        self.arg_list(arena, list)
    }

    /// The own type arguments of function `item`: the turbofish arguments `given`, then fresh variables for the parameters they leave out.
    pub(super) fn own_args(&mut self, item: ItemRef, mut given: Vec<Ty>) -> Result<Vec<Ty>> {
        let count = super::scope::type_params(&self.cx.signature(item)?.generics).count();
        if given.len() > count {
            bail!("`{}` takes {count} type arguments, {} given", self.cx.name(item), given.len());
        }
        given.extend((given.len()..count).map(|_| self.fresh(Kind::Var)));
        Ok(given)
    }

    /// The types and const values that the generic argument list `list` gives, skipping lifetimes.
    pub(super) fn arg_list(&mut self, arena: &Body, list: syntax::List) -> Result<Vec<Ty>> {
        let mut tys = Vec::new();
        for &at in arena.list(list) {
            match arena.args[at as usize] {
                GenericArg::Ty(ty) => tys.push(self.lower_ty(arena, ty)?),
                GenericArg::Lifetime(_) => {}
                GenericArg::Const(expr) => tys.push(self.const_arg(arena, expr)?),
                other => bail!("unsupported generic argument {other:?}"),
            }
        }
        Ok(tys)
    }

    /// The type arguments of the struct, enum or union `item`: `given`, then the defaults of the rest.
    pub(super) fn adt_args(&mut self, item: ItemRef, given: Vec<Ty>, fresh: bool) -> Result<Vec<Ty>> {
        let def = self.cx.adt_def(item)?;
        let (arena, parsed) = (&def.body, &def.generics);
        let params: Vec<&Generic> = parsed.params.iter().filter(|param| param.kind != GenericKind::Lifetime).collect();
        if given.len() > params.len() {
            bail!("`{}` takes {} type arguments, {} given", self.cx.name(item), params.len(), given.len());
        }
        let mut args = given;
        if fresh && args.is_empty() {
            return Ok(params.iter().map(|_| self.fresh(Kind::Var)).collect());
        }
        for param in &params[args.len()..] {
            let (GenericKind::Type, Some(default)) = (param.kind, param.default) else { bail!("`{}` needs {} type arguments, {} given", self.cx.name(item), params.len(), args.len()) };
            let known = args.clone();
            args.push(self.within(item, &parsed.params, &known, |infer| infer.lower_ty(arena, default))?);
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
            syntax::Ty::ImplTrait(at, _) => self.impl_param(at),
            _ => {
                let kind = self.lower_kind(arena, ty)?;
                Ok(self.cx.types.intern(kind))
            }
        }
    }

    fn lower_kind(&mut self, arena: &Body, ty: TyId) -> Result<Kind> {
        match *arena.ty(ty) {
            syntax::Ty::Ref { mutable, ty, .. } => self.lower_ty(arena, ty).map(|inner| Kind::Ref(mutability(mutable), inner)),
            syntax::Ty::Ptr { mutable, ty } => self.lower_ty(arena, ty).map(|inner| Kind::Ptr(mutability(mutable), inner)),
            syntax::Ty::Slice(ty) => self.lower_ty(arena, ty).map(Kind::Slice),
            syntax::Ty::Array(ty, len) => self.array_ty(arena, ty, len),
            syntax::Ty::Tuple(list) => self.lower_list(arena, list).map(Kind::Tuple),
            syntax::Ty::Never => Ok(Kind::Never),
            syntax::Ty::DynTrait(bounds) => self.object_ty(arena, bounds),
            syntax::Ty::Fn { params, ret, variadic } => self.fn_ptr_ty(arena, (params, ret), variadic),
            other => bail!("unsupported type {other:?}"),
        }
    }

    fn fn_ptr_ty(&mut self, arena: &Body, (params, ret): (syntax::List, Option<TyId>), variadic: bool) -> Result<Kind> {
        if variadic {
            bail!("variadic function pointer types are not supported yet");
        }
        let params = self.lower_list(arena, params)?;
        let ret = match ret {
            Some(ret) => self.lower_ty(arena, ret)?,
            None => self.cx.types.unit(),
        };
        Ok(Kind::FnPtr(params, ret))
    }

    fn object_ty(&mut self, arena: &Body, bounds: syntax::List) -> Result<Kind> {
        let mut principal = None;
        for &bound in arena.list(bounds) {
            let syntax::Bound::Trait { path, maybe: false, .. } = arena.bounds[bound as usize] else { continue };
            let Some(of) = self.trait_named(arena, path, self.home) else { bail!("a trait object bound that names no trait") };
            if !self.cx.is_auto_trait(of) && principal.replace((of, path)).is_some() {
                bail!("a trait object with two traits that are not auto traits");
            }
        }
        let Some((of, path)) = principal else { bail!("a trait object of auto traits only") };
        let count = arena.list(arena.path(path).segments).len();
        let given = self.segment_args(arena, path, count)?;
        let wanted = super::scope::type_params(&self.cx.trait_generics(of)?.1.params).count();
        if given.len() != wanted {
            bail!("a trait object of `{}` that leaves trait arguments to their defaults", self.cx.name(of));
        }
        Ok(Kind::Dyn(of, given))
    }

    /// The type that the associated type `name` of impl `block` is, read in the current frame, when the impl has one.
    pub(super) fn impl_alias(&mut self, block: ImplRef, name: &str) -> Result<Option<Ty>> {
        let Some(member) = self.cx.member(block, name) else { return Ok(None) };
        if self.cx.item(member).1.kind != ItemKind::Type {
            return Ok(None);
        }
        let (_, arena, ty) = self.alias_parts(member)?;
        self.lower_ty(&arena, ty).map(Some)
    }

    /// The type that the type alias `item` names at the type arguments `given`, with fresh ones for the rest.
    fn alias(&mut self, item: ItemRef, given: Vec<Ty>) -> Result<Ty> {
        let (params, arena, ty) = self.alias_parts(item)?;
        let file = self.cx.file(item);
        let mut args = given.into_iter();
        let names: Vec<&str> = params.iter().map(|generic| file.source.text(generic.name)).collect();
        let generics = names.into_iter().map(|name| (name, args.next().unwrap_or_else(|| self.fresh(Kind::Var)))).collect();
        let frame = super::scope::Frame { src: &file.source, home: self.cx.module_of(item)?, generics, self_ty: None, block: None, item: None };
        self.with_frame(frame, |infer| infer.lower_ty(&arena, ty))
    }

    fn alias_parts(&mut self, item: ItemRef) -> Result<(Vec<Generic>, Body, TyId)> {
        let (src, entry) = self.cx.item(item);
        let Some(at) = entry.name else { bail!("type alias with no name") };
        let Some(eq) = (at + 1..entry.end).find(|&token| src.kind(token) == TokenKind::Eq) else { bail!("type `{}` has no `=`", src.text(at)) };
        let params = match src.text(at + 1) {
            "<" => crate::body::sig::generics(src, at + 1, eq)?.1.params.into_iter().filter(|generic| generic.kind != GenericKind::Lifetime).collect(),
            _ => Vec::new(),
        };
        let (arena, ty) = syntax::parse_ty(src, eq + 1, entry.end - 1)?;
        Ok((params, arena, ty))
    }

    fn impl_param(&self, at: u32) -> Result<Ty> {
        let key = self.src.text(at).as_ptr();
        match self.generics.iter().find(|(name, _)| name.as_ptr() == key) {
            Some(&(_, ty)) => Ok(ty),
            None => bail!("`impl Trait` is only supported in argument position"),
        }
    }

    fn array_ty(&mut self, arena: &Body, ty: TyId, len: syntax::ExprId) -> Result<Kind> {
        Ok(Kind::Array(self.lower_ty(arena, ty)?, self.const_arg(arena, len)?))
    }

    fn lower_list(&mut self, arena: &Body, list: syntax::List) -> Result<Vec<Ty>> {
        arena.list(list).iter().map(|&item| self.lower_ty(arena, TyId(item))).collect()
    }


    fn path_ty(&mut self, arena: &Body, path: PathId) -> Result<Ty> {
        if let Some((qself, Some(trait_path))) = arena.path(path).qself {
            return self.projection(arena, path, qself, trait_path);
        }
        let count = arena.path(path).segments.len as usize;
        self.named_ty(arena, path, count, false)
    }

    /// The type that the first `count` segments of `path` name, with fresh arguments for an ADT given none when `fresh` holds.
    pub(super) fn named_ty(&mut self, arena: &Body, path: PathId, count: usize, fresh: bool) -> Result<Ty> {
        let (mut segments, absolute, dollar) = self.segments(arena, path);
        segments.truncate(count);
        if let Some(ty) = self.scoped_ty(&segments)? {
            return Ok(ty);
        }
        let paths = self.cx.paths(self.home.krate);
        let inner = if let [name] = segments[..] { self.declared_item(self.cx.names.symbols.text(name)) } else { None };
        let def = inner.map(Def::Item).or_else(|| paths.resolve(&NamePath { segments: &segments, absolute, dollar }, self.home));
        let def = match (def, &segments[..]) {
            (Some(Def::Module(_)), [name]) => crate::names::Symbols::primitive(*name).map(Def::Primitive).or(def),
            _ => def,
        };
        match def {
            Some(Def::Primitive(primitive)) => Ok(self.cx.types.prim(primitive)),
            Some(Def::Item(item)) if matches!(self.cx.item(item).1.kind, ItemKind::Type | ItemKind::Struct | ItemKind::Enum | ItemKind::Union) => {
                let given = self.segment_args(arena, path, count)?;
                self.item_ty(item, given, fresh)
            }
            other => bail!("cannot lower type `{}` ({other:?})", self.spell(&segments)),
        }
    }

    fn item_ty(&mut self, item: ItemRef, given: Vec<Ty>, fresh: bool) -> Result<Ty> {
        if self.cx.item(item).1.kind == ItemKind::Type {
            return self.alias(item, given);
        }
        let args = self.adt_args(item, given, fresh)?;
        Ok(self.cx.types.intern(Kind::Adt(item, args)))
    }

    /// The type that `segments` names through the frame: `Self`, a type parameter, or an associated type `Param::Name` of one.
    fn scoped_ty(&mut self, segments: &[Sym]) -> Result<Option<Ty>> {
        let symbols = &self.cx.names.symbols;
        match *segments {
            [one] if symbols.text(one) == "Self" && self.self_ty.is_some() => Ok(self.self_ty),
            [one] => Ok(self.generics.iter().find(|(generic, _)| *generic == symbols.text(one)).map(|&(_, ty)| ty)),
            [param, name] if symbols.text(param) == "Self" || self.generics.iter().any(|(generic, _)| *generic == symbols.text(param)) => {
                let (param, name) = (symbols.text(param).to_string(), symbols.text(name).to_string());
                self.param_projection(&param, &name)
            }
            _ => Ok(None),
        }
    }

    /// The type `<qself as Trait>::Name` names: the impl's associated type, or the pointer metadata the compiler provides.
    fn projection(&mut self, arena: &Body, path: PathId, qself: TyId, trait_path: PathId) -> Result<Ty> {
        let ty = self.lower_ty(arena, qself)?;
        let (segments, absolute, dollar) = self.segments(arena, trait_path);
        let paths = self.cx.paths(self.home.krate);
        let def = paths.resolve(&NamePath { segments: &segments, absolute, dollar }, self.home);
        let name = arena.list(arena.path(path).segments).last().map(|&at| plain(self.src.text(arena.segments[at as usize].name)));
        let Some(Def::Item(item)) = def else { bail!("projection through `{}`, which is not a trait", self.spell(&segments)) };
        let Some(name) = name else { bail!("a projection with no name") };
        if self.cx.lang(item) == Some("pointee_trait") && name == "Metadata" {
            return self.metadata_of(ty);
        }
        let count = arena.list(arena.path(trait_path).segments).len();
        let given = self.segment_args(arena, trait_path, count)?;
        let args = self.trait_args(item, given, ty)?;
        self.project(item, ty, args, name)
    }

    /// The pointer metadata of `ty`: a length for slices and `str`, nothing for sized types, or a variable until `ty` is known.
    pub fn metadata_of(&mut self, ty: Ty) -> Result<Ty> {
        match self.cx.types.kind(self.resolve(ty)) {
            Kind::Slice(_) | Kind::Prim(Primitive::Str) => Ok(self.cx.types.prim(Primitive::Usize)),
            Kind::Dyn(..) => {
                let Some(metadata) = self.cx.lang_item("dyn_metadata") else { bail!("core has no `DynMetadata`") };
                Ok(self.cx.types.intern(Kind::Adt(metadata, vec![ty])))
            }
            Kind::Var(_) => {
                let result = self.fresh(Kind::Var);
                self.metadata.push((ty, result));
                Ok(result)
            }
            _ => Ok(self.cx.types.unit()),
        }
    }

    pub(super) fn spell(&self, segments: &[Sym]) -> String {
        segments.iter().map(|sym| self.cx.names.symbols.text(*sym)).collect::<Vec<_>>().join("::")
    }
}

fn mutability(mutable: bool) -> Mut {
    if mutable { Mut::Mut } else { Mut::Not }
}

/// The value of an integer literal and its suffix type, if it has one.
pub fn parse_int(text: &str) -> Result<(u128, Option<Primitive>)> {
    let digits: String = text.chars().filter(|ch| *ch != '_').collect();
    let (radix, rest) = match digits.get(..2) {
        Some("0x") => (16, &digits[2..]),
        Some("0o") => (8, &digits[2..]),
        Some("0b") => (2, &digits[2..]),
        _ => (10, &digits[..]),
    };
    let split = rest.find(|ch: char| !ch.is_digit(radix)).unwrap_or(rest.len());
    let (number, suffix) = rest.split_at(split);
    let value = u128::from_str_radix(number, radix)?;
    let suffix = if suffix.is_empty() { None } else { Some(Primitive::named(suffix).ok_or_else(|| anyhow::anyhow!("bad suffix {suffix}"))?) };
    Ok((value, suffix))
}
