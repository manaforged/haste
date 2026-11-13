use anyhow::{Result, bail};

use crate::body::adt::Shape;
use crate::body::{Body, ExprId, List, PathId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, Sym, plain};
use crate::outline::item::ItemKind;

use super::super::substs::Substs;
use super::super::ty::{Kind, Ty};
use super::scope::Form;
use super::{Infer, Res};

/// A constructor that a path names: the ADT, the index of its variant, and its type arguments.
pub type Ctor = (ItemRef, u32, Substs);

/// The field types of each variant of an ADT at some type arguments, shared with the cache that holds them.
pub type Fields = std::sync::Arc<[Vec<Ty>]>;

impl<'a> Infer<'_, 'a> {
    /// The field types of each variant of the struct, enum or union `item` at `args`, lowered once per ADT type when no body state can change them.
    pub fn adt_fields(&mut self, item: ItemRef, args: &[Ty]) -> Result<Fields> {
        let (key, cached) = self.cached_fields(item, args);
        if let Some(found) = cached {
            return Ok(found);
        }
        if let Some(form) = self.generic((Form::Adt(item), args.len()), |infer, params| infer.lower_fields(item, params)) {
            let args: Vec<Ty> = args.iter().map(|&arg| self.substitute(arg)).collect();
            let fields: Fields = form.iter().map(|variant| variant.iter().map(|&ty| self.instantiate(ty, &args)).collect()).collect::<Vec<Vec<Ty>>>().into();
            if let Some(key) = key {
                self.cx.store.lookups.fields.insert(key, fields.clone());
            }
            return Ok(fields);
        }
        let before = self.unsettled();
        let fields = self.lower_fields(item, args)?;
        let closed = self.unsettled() == before && !fields.iter().flatten().any(|&ty| self.has_vars(ty));
        if let Some(key) = key.filter(|_| closed) {
            self.cx.store.lookups.fields.insert(key, fields.clone());
        }
        Ok(fields)
    }

    /// The field types of each variant of `item` at `args`, lowered from its definition.
    fn lower_fields(&mut self, item: ItemRef, args: &[Ty]) -> Result<Fields> {
        if self.cx.foreign(item) {
            return Ok(vec![Vec::new()].into());
        }
        let def = self.cx.adt_def(item)?;
        let frame = self.adt_frame(item, &def.generics.params, args)?;
        Ok(self.with_frame(frame, |infer| def.variants.iter().map(|variant| variant.fields.iter().map(|field| infer.lower_ty(&def.body, field.ty).map(|ty| infer.substitute(ty))).collect()).collect::<Result<Vec<Vec<Ty>>>>())?.into())
    }

    /// The key that caches the field types of `item` at `args` across bodies, the ADT type with its variables filled in, and what it holds; no key when body state or an unbound inference variable could change the answer.
    fn cached_fields(&mut self, item: ItemRef, args: &[Ty]) -> (Option<Ty>, Option<Fields>) {
        if !self.shared_here(item.declaring_body()) || args.iter().any(|&arg| self.has_vars(arg) || self.has_params(arg)) {
            return (None, None);
        }
        let args: Vec<Ty> = args.iter().map(|&arg| self.substitute(arg)).collect();
        let key = self.cx.types.adt(item, &args);
        (Some(key), self.cx.store.lookups.fields.get(&key))
    }

    /// The type of the last field of the first variant of `item` at `args`, the only field that can make a struct unsized, read from the cached field types when they are there.
    pub fn adt_tail_field(&mut self, item: ItemRef, args: &[Ty]) -> Result<Option<Ty>> {
        if self.cx.foreign(item) {
            return Ok(None);
        }
        if let (_, Some(found)) = self.cached_fields(item, args) {
            return Ok(found.first().and_then(|variant| variant.last()).copied());
        }
        let def = self.cx.adt_def(item)?;
        let Some(last) = def.variants.first().and_then(|variant| variant.fields.last()) else {
            return Ok(None);
        };
        let frame = self.adt_frame(item, &def.generics.params, args)?;
        self.with_frame(frame, |infer| infer.lower_ty(&def.body, last.ty).map(|ty| infer.substitute(ty))).map(Some)
    }

    /// Whether the struct `item` at `args` ends in an unsized field once each `sized` stand-in is an open variable, so a projection on it stays open and counts as sized.
    pub fn open_tail_unsized(&mut self, item: ItemRef, args: &[Ty], sized: Ty) -> Result<bool> {
        let open: Vec<Ty> = args.iter().map(|&arg| if arg == sized { self.fresh(Kind::Var) } else { arg }).collect();
        let tail = self.cx.types.adt(item, &open);
        self.is_unsized(tail)
    }

    /// The unsized tail of `ty`, walking struct tail fields in this inference context, cached in the store when `ty` and its tail hold no variable or parameter and no quieted error arose.
    pub(in crate::sema) fn tail_ty(&mut self, ty: Ty) -> Result<Ty> {
        let ty = self.substitute(ty);
        let Kind::Adt(item, args) = *self.cx.types.kind(ty) else { return Ok(ty) };
        let cacheable = !self.cx.types.has_params(ty) && !self.cx.types.has_var_flag(ty);
        if cacheable && let Some(found) = self.cx.store.tails.get(&ty) {
            return Ok(found);
        }
        let before = self.cx.quiet_mark();
        let slice = self.cx.types.substs(args).to_vec();
        let last = if self.cx.item(item).1.kind == ItemKind::Struct { self.adt_tail_field(item, &slice)? } else { None };
        let found = match last {
            Some(last) => self.tail_ty(last)?,
            None => ty,
        };
        if cacheable && !self.cx.types.has_params(found) && !self.cx.types.has_var_flag(found) && self.cx.quiet_mark() == before {
            self.cx.store.tails.insert(ty, found);
        }
        Ok(found)
    }

    /// Whether a pointer to `ty` carries metadata after its address, decided in this inference context.
    pub(in crate::sema) fn is_unsized(&mut self, ty: Ty) -> Result<bool> {
        let tail = self.tail_ty(ty)?;
        Ok(self.cx.types.kind(tail).is_unsized())
    }

    /// The frame of the struct, enum or union `item` with its type parameters `params` set to `args`, whose header supplies their bounds.
    pub(super) fn adt_frame(&mut self, item: ItemRef, params: &[crate::body::sig::Generic], args: &[Ty]) -> Result<super::scope::Frame<'a>> {
        let self_ty = Some(self.cx.types.adt(item, args));
        self.item_frame(item, (params, args), (self_ty, Some(item)))
    }

    /// The constructor that `path` names in the value namespace when `values` holds, else in the type namespace, where a type that names a struct or union also does.
    pub fn ctor(&mut self, arena: &Body, path: PathId, values: bool) -> Result<Option<Ctor>> {
        if values {
            return self.named_ctor(arena, path, values);
        }
        if arena.path(path).qself.is_some() {
            return self.type_ctor(arena, path);
        }
        match self.named_ctor(arena, path, values)? {
            None => self.type_ctor(arena, path),
            found => Ok(found),
        }
    }

    /// The struct or union that `path` names as a type, through a type alias, `Self` or an associated type, at the type's arguments.
    fn type_ctor(&mut self, arena: &Body, path: PathId) -> Result<Option<Ctor>> {
        let entry = arena.path(path);
        let ty = match (entry.qself, arena.list(entry.segments).last()) {
            (Some((qself, Some(trait_path))), Some(&last)) => self.projection(arena, (qself, trait_path), last)?,
            _ => self.named_ty(arena, path, entry.segments.len as usize, true)?,
        };
        let Kind::Adt(item, args) = self.cx.types.kind(self.resolve(ty)).clone() else {
            return Ok(None);
        };
        Ok(matches!(self.cx.item(item).1.kind, ItemKind::Struct | ItemKind::Union).then_some((item, 0, args)))
    }

    /// The struct, union or variant that `path` names directly, or an enum variant through a type alias.
    fn named_ctor(&mut self, arena: &Body, path: PathId, values: bool) -> Result<Option<Ctor>> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        let texts: Vec<&str> = segments.iter().map(|sym| self.cx.names.symbols.text(*sym)).collect();
        if texts.first() == Some(&"Self") {
            return self.self_ctor(&texts[1..]);
        }
        let name_path = NamePath { segments: &segments, absolute, dollar };
        let def = if values { self.value_item(arena, path).ok() } else { self.type_def(&name_path) };
        let count = segments.len();
        match def {
            Some(Def::Item(item)) if matches!(self.cx.item(item).1.kind, ItemKind::Struct | ItemKind::Union) => {
                let given = self.segment_args(arena, path, count)?;
                let args = self.adt_args(item, given, true)?;
                Ok(Some((item, 0, self.cx.types.intern_substs(&args))))
            }
            Some(Def::Variant(item, index)) => {
                let on_enum = if count >= 2 { self.segment_args(arena, path, count - 1)? } else { Vec::new() };
                let given = if on_enum.is_empty() { self.segment_args(arena, path, count)? } else { on_enum };
                let args = self.adt_args(item, given, true)?;
                let variant = self.cx.variant_position(item, index)?;
                Ok(Some((item, variant as u32, self.cx.types.intern_substs(&args))))
            }
            _ => self.alias_variant(arena, path, &segments),
        }
    }

    /// The variant that `Alias::Variant` names, when the type alias names an enum.
    fn alias_variant(&mut self, arena: &Body, path: PathId, segments: &[Sym]) -> Result<Option<Ctor>> {
        let Some((name, prefix)) = segments.split_last().filter(|(_, prefix)| !prefix.is_empty()) else {
            return Ok(None);
        };
        let (_, absolute, dollar) = self.segments(arena, path);
        let Some(Def::Item(alias)) = self.type_def(&NamePath { segments: prefix, absolute, dollar }) else {
            return Ok(None);
        };
        if self.cx.item(alias).1.kind != ItemKind::Type {
            return Ok(None);
        }
        let given = self.segment_args(arena, path, prefix.len())?;
        let named = self.alias(alias, given, true)?;
        let Kind::Adt(item, args) = self.cx.types.kind(named).clone() else {
            return Ok(None);
        };
        let name = self.cx.names.symbols.text(*name).to_string();
        Ok(self.variant_named(item, &name)?.map(|variant| (item, variant, args)))
    }

    fn self_ctor(&mut self, rest: &[&str]) -> Result<Option<Ctor>> {
        let Some(Kind::Adt(item, args)) = self.self_ty.map(|ty| self.cx.types.kind(ty).clone()) else {
            return Ok(None);
        };
        match rest {
            [] => Ok(Some((item, 0, args))),
            [name] => Ok(self.variant_named(item, name)?.map(|variant| (item, variant, args))),
            _ => Ok(None),
        }
    }

    pub(super) fn variant_named(&mut self, item: ItemRef, name: &str) -> Result<Option<u32>> {
        let def = self.cx.adt_def(item)?;
        let src = &self.cx.file(item).source;
        Ok(def.variants.iter().position(|variant| src.text(variant.name) == name).map(|at| at as u32))
    }

    /// The position of the field named `name` in variant `variant` of `item`: a field name, or an index for a tuple variant.
    pub fn field_index(&mut self, item: ItemRef, variant: u32, name: &str) -> Result<Option<u32>> {
        let def = self.cx.adt_def(item)?;
        let src = &self.cx.file(item).source;
        let fields = &def.variants[variant as usize].fields;
        Ok(match name.parse::<usize>() {
            Ok(index) if index < fields.len() && def.variants[variant as usize].shape == Shape::Tuple => Some(index as u32),
            _ => fields.iter().position(|field| field.name.is_some_and(|at| plain(src.text(at)) == name)).map(|at| at as u32),
        })
    }

    /// The type of a path that names the constructor `ctor`: the value of a unit variant, or a function from the fields of a tuple variant.
    pub fn ctor_value(&mut self, (item, variant, args): Ctor) -> Result<(Ty, Res)> {
        let shape = self.cx.adt_def(item)?.variants[variant as usize].shape;
        let adt = self.cx.types.intern(Kind::Adt(item, args));
        let ty = match shape {
            Shape::Unit => adt,
            Shape::Tuple => {
                let slice = self.cx.types.substs(args).to_vec();
                let fields = self.adt_fields(item, &slice)?[variant as usize].clone();
                self.cx.types.fn_ptr(&fields, adt)
            }
            Shape::Named => bail!("`{}` needs its fields named", self.cx.name(item)),
        };
        Ok((ty, Res::Ctor(item, variant, args)))
    }

    /// Infers the struct literal `expr`.
    pub fn struct_expr(&mut self, expr: ExprId, path: PathId, (fields, base): (List, Option<ExprId>), expected: Option<Ty>) -> Result<Ty> {
        let body = self.body;
        let Some((item, variant, args)) = self.ctor(body, path, false)? else { bail!("struct literal of something that is not a struct") };
        let adt = self.cx.types.intern(Kind::Adt(item, args));
        if let Some(expected) = expected {
            self.trial(|infer| Ok(infer.unifies(adt, expected).then_some(())))?;
        }
        let slice = self.cx.types.substs(args).to_vec();
        let tys = self.adt_fields(item, &slice)?[variant as usize].clone();
        for &init in body.list(fields) {
            let field = body.fields[init as usize];
            let name = plain(self.src.text(field.name));
            let Some(index) = self.field_index(item, variant, name)? else { bail!("no field `{name}` on {}", self.cx.name(item)) };
            let ty = tys[index as usize];
            self.out.inits.insert(field.value, index);
            self.call_arg(Some((item, &slice[..])), field.value, (ty, None), &mut None)?;
        }
        if let Some(base) = base {
            let found = self.expr(base, Some(adt))?;
            self.coerce(found, adt)?;
        }
        self.out.res.insert(expr, Res::Ctor(item, variant, args));
        Ok(adt)
    }

    /// Infers `offset_of!(ty, fields)`: a `usize` whose value, the byte offset of the field path in `ty`, lowering reads.
    pub fn offset_of(&mut self, expr: ExprId, ty: crate::body::TyId, fields: crate::body::List) -> Result<Ty> {
        let mut current = self.lower_ty(self.body, ty)?;
        if self.poly.is_some() && self.has_params(current) {
            return self.poly_offset_of(expr, current, fields);
        }
        let mut offset = 0;
        for &at in self.body.list(fields) {
            for name in self.src.text(at).split('.').map(plain) {
                let (index, field) = self.field_at(current, name)?;
                offset += crate::sema::layout::field_offset(&*self.cx.layout(current)?, 0, index) as u64;
                current = field;
            }
        }
        self.out.offsets.insert(expr, offset);
        Ok(self.cx.types.prim(crate::names::Primitive::Usize))
    }

    pub(super) fn field_at(&mut self, ty: Ty, name: &str) -> Result<(u32, Ty)> {
        match self.cx.types.kind(self.resolve(ty)).clone() {
            Kind::Tuple(items) => {
                let index: u32 = name.parse()?;
                let slice = self.cx.types.substs(items);
                slice.get(index as usize).map(|&item| (index, item)).ok_or_else(|| anyhow::anyhow!("no field {index} on a tuple of {}", slice.len()))
            }
            Kind::Adt(item, args) => {
                let Some(index) = self.field_index(item, 0, name)? else { bail!("no field `{name}` on {}", self.cx.name(item)) };
                let slice = self.cx.types.substs(args).to_vec();
                Ok((index, self.adt_fields(item, &slice)?[0][index as usize]))
            }
            _ => bail!("offset_of on {}", self.show(ty)),
        }
    }

    /// Infers the field access `expr` on `base`, through references.
    pub fn field_expr(&mut self, expr: ExprId, base: ExprId, name: &str) -> Result<Ty> {
        let found = self.expr(base, None)?;
        let ty = self.field_base(expr, found, name)?;
        let Kind::Adt(item, args) = self.cx.types.kind(ty).clone() else { bail!("no field `{name}` on {}", self.show(ty)) };
        if self.cx.item(item).1.kind == ItemKind::Enum {
            bail!("no field `{name}` on enum {}", self.cx.name(item));
        }
        let Some(index) = self.field_index(item, 0, name)? else { bail!("no field `{name}` on {}", self.cx.name(item)) };
        self.out.fields.insert(expr, index);
        let slice = self.cx.types.substs(args).to_vec();
        Ok(self.adt_fields(item, &slice)?[0][index as usize])
    }
}

/// The function that the tuple constructor of variant `variant` of `item` at `args` is when a path uses it as a value: `item` at `args` and then the variant.
pub fn ctor_fn(cx: &mut super::super::Context<'_>, (item, variant, args): Ctor) -> Res {
    let marker = cx.types.intern(Kind::Value(u128::from(variant)));
    let mut full = cx.types.substs(args).to_vec();
    full.push(marker);
    Res::fn_of(&cx.types, item, &full)
}

/// The typed form of the constructor function that `ctor_fn` names, which builds the variant from its parameters.
pub fn ctor_typed(cx: &mut super::super::Context<'_>, item: ItemRef, substs: Substs) -> Result<super::Typed> {
    let slice = cx.types.substs(substs).to_vec();
    let Some((&marker, args)) = slice.split_last() else { bail!("a constructor function with no variant") };
    let Kind::Value(variant) = *cx.types.kind(marker) else { bail!("a constructor function with no variant") };
    let variant = u32::try_from(variant)?;
    let all = super::field_types(cx, item, args)?;
    let fields = &all[variant as usize];
    let locals = fields.iter().map(|&ty| super::Local { ty }).collect();
    let ret = cx.types.adt(item, args);
    Ok(super::Typed { locals, params: (0..fields.len() as u32).collect(), ret: Some(ret), shape: super::Shape::Ctor(variant), ..super::Typed::default() })
}

/// The field types of each variant of the struct, enum or union `item` at `args`.
pub fn field_types(cx: &mut super::super::Context<'_>, item: ItemRef, args: &[Ty]) -> Result<Fields> {
    if args.iter().any(|&a| cx.types.has_var_flag(a)) {
        bail!("cannot compute fields of open inference variable outside Infer");
    }
    super::detached(cx, item, |infer| infer.adt_fields(item, args))
}

/// Whether the struct `item` may end in an unsized field, which its definition decides for every instantiation as rustc's layout does; with no `?Sized` parameter, a tail the stand-ins cannot lower, such as a projection on one, is sized.
pub fn maybe_unsized(cx: &mut super::super::Context<'_>, item: ItemRef) -> Result<bool> {
    if cx.item(item).1.kind != ItemKind::Struct {
        return Ok(false);
    }
    let def = cx.adt_def(item)?;
    let src = &cx.file(item).source;
    let byte = cx.types.intern(Kind::Prim(crate::names::Primitive::U8));
    let stand_ins = [cx.types.intern(Kind::Value(1)), cx.types.intern(Kind::Slice(byte)), byte];
    let args: Vec<Ty> = super::scope::type_params(&def.generics.params)
        .map(|generic| match generic.kind {
            crate::body::sig::GenericKind::Const(_) => stand_ins[0],
            _ if param_relaxes_sized(&def.body, src, &def.generics.predicates, generic) => stand_ins[1],
            _ => stand_ins[2],
        })
        .collect();
    let any_relaxed = args.contains(&stand_ins[1]);
    let ty = cx.types.adt(item, &args);
    match cx.is_unsized(ty) {
        Err(_) if !any_relaxed => Ok(false),
        Err(_) => super::detached(cx, item, |infer| infer.open_tail_unsized(item, &args, stand_ins[2])),
        found => found,
    }
}

fn bound_relaxes_sized(arena: &Body, src: &crate::outline::source::Source, bound: &crate::body::Bound) -> bool {
    match *bound {
        crate::body::Bound::Trait { maybe: true, .. } => true,
        crate::body::Bound::Trait { path, .. } => arena
            .list(arena.path(path).segments)
            .last()
            .is_some_and(|&seg| matches!(src.text(arena.segments[seg as usize].name), "PointeeSized" | "MetaSized")),
        _ => false,
    }
}

pub(in crate::sema) fn body_relaxes_sized(arena: &Body, src: &crate::outline::source::Source) -> bool {
    arena.bounds.iter().any(|b| bound_relaxes_sized(arena, src, b))
}

pub(in crate::sema) fn param_relaxes_sized(
    arena: &Body,
    src: &crate::outline::source::Source,
    predicates: &[crate::body::sig::Predicate],
    generic: &crate::body::sig::Generic,
) -> bool {
    super::scope::generic_bounds(arena, src, predicates, generic)
        .into_iter()
        .any(|bound| bound_relaxes_sized(arena, src, &arena.bounds[bound as usize]))
}
