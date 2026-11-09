use anyhow::{Result, bail};

use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::super::Context;
use super::super::ty::{Kind, Ty};
use super::coerce::Coercion;
use super::{Infer, Res};

/// The vtable of the type `ty` as the trait object `object`: its drop glue and the function that each method slot calls.
#[derive(Clone, Debug)]
pub struct Vtable {
    pub ty: Ty,
    pub object: Ty,
    pub drop: Option<Res>,
    pub methods: Vec<Option<Res>>,
}

/// The vtable slot that holds the size of the value.
pub const SIZE_SLOT: u32 = 1;

/// The vtable slot that holds the alignment of the value.
pub const ALIGN_SLOT: u32 = 2;

/// The first vtable slot that holds a method, after the drop glue, the size and the alignment.
pub const FIRST_METHOD: u32 = 3;

impl Infer<'_, '_> {
    /// The unsizing of the pointee `from` to the pointee `to`: an array to a slice, a struct through its last field, or a sized type to a trait object that it implements.
    pub(super) fn unsize_pointee(&mut self, from: Ty, to: Ty) -> Result<Option<Coercion>> {
        let (from, to) = (self.resolve(from), self.resolve(to));
        match (self.cx.types.kind(from).clone(), self.cx.types.kind(to).clone()) {
            (Kind::Array(element, len), Kind::Slice(target)) => {
                self.unify(element, target)?;
                Ok(Some(Coercion::Len(len)))
            }
            (Kind::Adt(item, from), Kind::Adt(other, to)) if item == other && from != to && self.cx.item(item).1.kind == ItemKind::Struct => self.unsize_tail(item, &from, &to),
            (Kind::Dyn(..) | Kind::Var(_), _) => Ok(None),
            (_, Kind::Dyn(of, args, members)) => {
                let (given, bound) = super::super::ty::object_parts(&args, &members);
                if !self.object_of(of, from, given)? {
                    bail!("{} does not implement `{}`", self.show(from), self.cx.name(of));
                }
                let given = given.to_vec();
                for (member, ty) in bound.collect::<Vec<_>>() {
                    let name = self.cx.name(member);
                    if !self.binding_holds((of, &given), from, (&name, ty))? {
                        bail!("{} does not give `{name}` as the trait object wants", self.show(from));
                    }
                }
                Ok(Some(Coercion::Vtable(Vtable { ty: from, object: to, drop: None, methods: Vec::new() })))
            }
            _ => Ok(None),
        }
    }

    fn closure_slot(&mut self, method: ItemRef, ty: Ty) -> Option<ItemRef> {
        if !matches!(self.cx.types.kind(ty), Kind::Closure(..)) || self.cx.call_once() != Some(method) {
            return None;
        }
        let of = self.cx.lang_item("fn_mut")?;
        self.cx.trait_member(of, "call_mut")
    }

    fn object_of(&mut self, of: ItemRef, from: Ty, given: &[Ty]) -> Result<bool> {
        if !self.cx.lang(of).is_some_and(|lang| super::select::CALLABLE.contains(&lang)) {
            return Ok(self.select(of, from, given)?.is_some());
        }
        let (Some((params, _)), Some(&wanted)) = (self.callable_sig(from)?, given.first()) else { return Ok(false) };
        let inputs = self.cx.types.intern(Kind::Tuple(params));
        Ok(self.unify(inputs, wanted).is_ok())
    }

    fn unsize_tail(&mut self, item: ItemRef, from: &[Ty], to: &[Ty]) -> Result<Option<Coercion>> {
        let (have, want) = (self.adt_fields(item, from)?, self.adt_fields(item, to)?);
        let (Some(&a), Some(&b)) = (have.first().and_then(|fields| fields.last()), want.first().and_then(|fields| fields.last())) else { return Ok(None) };
        let Some(unsize) = self.unsize_pointee(a, b)? else { return Ok(None) };
        for (&found, &wanted) in from.iter().zip(to) {
            let saved = self.vars.clone();
            if self.unify(found, wanted).is_err() {
                self.vars = saved;
            }
        }
        Ok(Some(unsize))
    }

    /// The vtable `vtable` once its types are known, with each slot resolved to the function that it calls.
    pub(super) fn zonk_vtable(&mut self, vtable: Vtable) -> Result<Vtable> {
        let (ty, object) = (self.zonk(vtable.ty), self.zonk(vtable.object));
        let Kind::Dyn(of, args, members) = self.cx.types.kind(object).clone() else { bail!("a vtable for a type that is not a trait object") };
        let args = super::super::ty::object_parts(&args, &members).0.to_vec();
        let mut methods = Vec::new();
        for method in self.cx.vtable_methods(of)? {
            let substs = std::iter::once(ty).chain(args.iter().copied()).collect();
            let method = method.map(|method| self.closure_slot(method, ty).unwrap_or(method));
            methods.push(method.map(|method| self.trait_call(method, substs)).transpose()?);
        }
        Ok(Vtable { ty, object, drop: None, methods })
    }
}

/// Sets the drop glue of each vtable of the finished body `typed`, once the closures that it defines are registered, and says whether it set any.
pub fn fill_drops(cx: &mut Context<'_>, typed: &mut super::Typed) -> Result<bool> {
    let mut filled = false;
    for coercion in typed.coercions.values_mut() {
        if let Coercion::Vtable(vtable) = coercion {
            vtable.drop = super::drop::glue_call(cx, vtable.ty)?;
            filled = true;
        }
    }
    Ok(filled)
}

/// The vtable slot that a call of trait method `item` at `substs` reads its code from, when `Self` is a trait object of the method's trait.
pub fn virtual_slot(cx: &mut Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<Option<u32>> {
    let Some(of) = cx.owner_trait(item) else { return Ok(None) };
    let Some(&Kind::Dyn(object, ..)) = substs.first().map(|ty| cx.types.kind(*ty)) else { return Ok(None) };
    if object != of {
        bail!("a supertrait method through a trait object is not supported yet");
    }
    let Some(index) = cx.vtable_methods(of)?.iter().position(|method| *method == Some(item)) else { bail!("`{}` cannot be called through a trait object", cx.name(item)) };
    Ok(Some(FIRST_METHOD + index as u32))
}
