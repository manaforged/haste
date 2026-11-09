use anyhow::{Result, bail};

use crate::body::adt::Shape;
use crate::body::{Body, ExprId, List, PathId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, plain};
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
                let given = if count >= 2 { self.segment_args(arena, path, count - 1)? } else { Vec::new() };
                let args = self.adt_args(item, given, true)?;
                let variant = self.cx.variant_position(item, index)?;
                Ok(Some((item, variant as u32, args)))
            }
            _ => Ok(None),
        }
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
    pub fn struct_expr(&mut self, expr: ExprId, path: PathId, fields: List, base: Option<ExprId>) -> Result<Ty> {
        let body = self.body;
        let Some((item, variant, args)) = self.ctor(body, path, false)? else { bail!("struct literal of something that is not a struct") };
        let tys = self.adt_fields(item, &args)?.swap_remove(variant as usize);
        for &init in body.list(fields) {
            let field = body.fields[init as usize];
            let name = plain(self.src.text(field.name));
            let Some(index) = self.field_index(item, variant, name)? else { bail!("no field `{name}` on {}", self.cx.name(item)) };
            let ty = tys[index as usize];
            self.out.fields.insert(field.value, index);
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
