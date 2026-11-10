use anyhow::{Result, bail};

use crate::body::adt::Shape;
use crate::body::{Body, ExprId, List, PathId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, Sym, plain};
use crate::outline::item::ItemKind;

use super::super::ty::{Kind, Ty};
use super::{Infer, Res};

/// A constructor that a path names: the ADT, the index of its variant, and its type arguments.
pub type Ctor = (ItemRef, u32, Vec<Ty>);

impl<'a> Infer<'_, 'a> {
    /// The field types of each variant of the struct, enum or union `item` at `args`.
    pub fn adt_fields(&mut self, item: ItemRef, args: &[Ty]) -> Result<Vec<Vec<Ty>>> {
        let def = self.cx.adt_def(item)?;
        let frame = self.adt_frame(item, &def.generics.params, args)?;
        self.with_frame(frame, |infer| def.variants.iter().map(|variant| variant.fields.iter().map(|field| infer.lower_ty(&def.body, field.ty).map(|ty| infer.substitute(ty))).collect()).collect())
    }

    /// The type of the last field of the first variant of `item` at `args`, the only field that can make a struct unsized.
    pub fn adt_tail_field(&mut self, item: ItemRef, args: &[Ty]) -> Result<Option<Ty>> {
        let def = self.cx.adt_def(item)?;
        let Some(last) = def.variants.first().and_then(|variant| variant.fields.last()) else { return Ok(None) };
        let frame = self.adt_frame(item, &def.generics.params, args)?;
        self.with_frame(frame, |infer| infer.lower_ty(&def.body, last.ty).map(|ty| infer.substitute(ty))).map(Some)
    }

    fn adt_frame(&mut self, item: ItemRef, params: &[crate::body::sig::Generic], args: &[Ty]) -> Result<super::scope::Frame<'a>> {
        let self_ty = Some(self.cx.types.intern(Kind::Adt(item, args.to_vec())));
        self.item_frame(item, (params, args), (self_ty, Some(item)))
    }

    /// The constructor that `path` names in the value namespace when `values` holds, else in the type namespace.
    pub fn ctor(&mut self, arena: &Body, path: PathId, values: bool) -> Result<Option<Ctor>> {
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
                Ok(Some((item, 0, self.adt_args(item, given, true)?)))
            }
            Some(Def::Variant(item, index)) => {
                let on_enum = if count >= 2 { self.segment_args(arena, path, count - 1)? } else { Vec::new() };
                let given = if on_enum.is_empty() { self.segment_args(arena, path, count)? } else { on_enum };
                let args = self.adt_args(item, given, true)?;
                let variant = self.cx.variant_position(item, index)?;
                Ok(Some((item, variant as u32, args)))
            }
            _ => self.alias_variant(arena, path, &segments),
        }
    }

    /// The variant that `Alias::Variant` names, when the type alias names an enum.
    fn alias_variant(&mut self, arena: &Body, path: PathId, segments: &[Sym]) -> Result<Option<Ctor>> {
        let Some((name, prefix)) = segments.split_last().filter(|(_, prefix)| !prefix.is_empty()) else { return Ok(None) };
        let (_, absolute, dollar) = self.segments(arena, path);
        let Some(Def::Item(alias)) = self.type_def(&NamePath { segments: prefix, absolute, dollar }) else { return Ok(None) };
        if self.cx.item(alias).1.kind != ItemKind::Type {
            return Ok(None);
        }
        let given = self.segment_args(arena, path, prefix.len())?;
        let named = self.alias(alias, given, true)?;
        let Kind::Adt(item, args) = self.cx.types.kind(named).clone() else { return Ok(None) };
        let name = self.cx.names.symbols.text(*name).to_string();
        Ok(self.variant_named(item, &name)?.map(|variant| (item, variant, args)))
    }

    fn self_ctor(&mut self, rest: &[&str]) -> Result<Option<Ctor>> {
        let Some(Kind::Adt(item, args)) = self.self_ty.map(|ty| self.cx.types.kind(ty).clone()) else { return Ok(None) };
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
        let adt = self.cx.types.intern(Kind::Adt(item, args.clone()));
        let ty = match shape {
            Shape::Unit => adt,
            Shape::Tuple => {
                let fields = self.adt_fields(item, &args)?.swap_remove(variant as usize);
                self.cx.types.intern(Kind::FnPtr(fields, adt))
            }
            Shape::Named => bail!("`{}` needs its fields named", self.cx.name(item)),
        };
        Ok((ty, Res::Ctor(item, variant, args)))
    }

    /// Infers the struct literal `expr`.
    pub fn struct_expr(&mut self, expr: ExprId, path: PathId, (fields, base): (List, Option<ExprId>), expected: Option<Ty>) -> Result<Ty> {
        let body = self.body;
        let Some((item, variant, args)) = self.ctor(body, path, false)? else { bail!("struct literal of something that is not a struct") };
        if let Some(expected) = expected {
            let adt = self.cx.types.intern(Kind::Adt(item, args.clone()));
            self.trial(|infer| Ok(infer.unifies(adt, expected).then_some(())))?;
        }
        let tys = self.adt_fields(item, &args)?.swap_remove(variant as usize);
        for &init in body.list(fields) {
            let field = body.fields[init as usize];
            let name = plain(self.src.text(field.name));
            let Some(index) = self.field_index(item, variant, name)? else { bail!("no field `{name}` on {}", self.cx.name(item)) };
            let ty = tys[index as usize];
            self.out.inits.insert(field.value, index);
            let found = self.expr(field.value, Some(ty))?;
            self.coerce_expr(field.value, found, ty)?;
        }
        let adt = self.cx.types.intern(Kind::Adt(item, args.clone()));
        if let Some(base) = base {
            let found = self.expr(base, Some(adt))?;
            self.coerce(found, adt)?;
        }
        self.out.res.insert(expr, Res::Ctor(item, variant, args));
        Ok(adt)
    }

    /// Infers the field access `expr` on `base`, through references.
    /// Infers `offset_of!(ty, fields)`: a `usize` whose value, the byte offset of the field path in `ty`, lowering reads.
    pub fn offset_of(&mut self, expr: ExprId, ty: crate::body::TyId, fields: crate::body::List) -> Result<Ty> {
        let mut current = self.lower_ty(self.body, ty)?;
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

    fn field_at(&mut self, ty: Ty, name: &str) -> Result<(u32, Ty)> {
        match self.cx.types.kind(self.resolve(ty)).clone() {
            Kind::Tuple(items) => {
                let index: u32 = name.parse()?;
                items.get(index as usize).map(|&item| (index, item)).ok_or_else(|| anyhow::anyhow!("no field {index} on a tuple of {}", items.len()))
            }
            Kind::Adt(item, args) => {
                let Some(index) = self.field_index(item, 0, name)? else { bail!("no field `{name}` on {}", self.cx.name(item)) };
                Ok((index, self.adt_fields(item, &args)?[0][index as usize]))
            }
            _ => bail!("offset_of on {}", self.show(ty)),
        }
    }

    pub fn field_expr(&mut self, expr: ExprId, base: ExprId, name: &str) -> Result<Ty> {
        let found = self.expr(base, None)?;
        let ty = self.field_base(expr, found, name)?;
        let Kind::Adt(item, args) = self.cx.types.kind(ty).clone() else { bail!("no field `{name}` on {}", self.show(ty)) };
        if self.cx.item(item).1.kind == ItemKind::Enum {
            bail!("no field `{name}` on enum {}", self.cx.name(item));
        }
        let Some(index) = self.field_index(item, 0, name)? else { bail!("no field `{name}` on {}", self.cx.name(item)) };
        self.out.fields.insert(expr, index);
        Ok(self.adt_fields(item, &args)?[0][index as usize])
    }
}

/// The function that the tuple constructor of variant `variant` of `item` at `args` is when a path uses it as a value: `item` at `args` and then the variant.
pub fn ctor_fn(cx: &mut super::super::Context<'_>, (item, variant, args): Ctor) -> Res {
    let marker = cx.types.intern(Kind::Value(u128::from(variant)));
    Res::Fn(item, args.into_iter().chain(std::iter::once(marker)).collect())
}

/// The typed form of the constructor function that `ctor_fn` names, which builds the variant from its parameters.
pub fn ctor_typed(cx: &mut super::super::Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<super::Typed> {
    let Some((&marker, args)) = substs.split_last() else { bail!("a constructor function with no variant") };
    let Kind::Value(variant) = *cx.types.kind(marker) else { bail!("a constructor function with no variant") };
    let variant = u32::try_from(variant)?;
    let fields = super::detached(cx, item, |infer| infer.adt_fields(item, args))?.swap_remove(variant as usize);
    let locals = fields.iter().map(|&ty| super::Local { ty }).collect();
    let ret = cx.types.intern(Kind::Adt(item, args.to_vec()));
    Ok(super::Typed { locals, params: (0..fields.len() as u32).collect(), ret: Some(ret), shape: super::Shape::Ctor(variant), ..super::Typed::default() })
}
