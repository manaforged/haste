use std::sync::{Arc, Mutex};

use rustc_hash::FxHashMap as HashMap;

use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::ty::{Kind, Ty};

use super::bridge::Carry;
use super::key::Keys;
use super::record::{Abi, Object, Pointer, Slot};
use super::conv;

/// How code pointers of one trait object or function pointer type travel between Haste's code and rustc's.
#[derive(Clone)]
pub(super) enum Mode {
    /// rustc's code never makes, calls or hands on one, so each side calls only its own.
    Free,
    /// Every vtable of the type has rustc's layout and every call through it uses rustc's convention, on both sides.
    Rustc(Arc<Vtable>),
    /// The two sides would call one otherwise, so no value of the type may cross.
    Opaque,
}

/// A vtable as rustc lays it out, for Haste's code to build and call: each method slot, and the slot of each supertrait's vtable.
pub struct Vtable {
    pub slots: Vec<Method>,
    pub supers: Vec<(ItemRef, Option<usize>)>,
}

/// One method slot: the trait method it holds, its index, the types a call through it passes, as rustc counts them, and returns, and how each argument reaches rustc's convention.
pub struct Method {
    pub method: ItemRef,
    pub at: usize,
    pub params: Vec<Ty>,
    pub ret: Option<Ty>,
    pub carries: Vec<Carry>,
}

/// The trait object and function pointer types that rustc's code of one library touches, by key, and the mode each type was given.
pub(super) struct Indirect {
    objects: HashMap<String, Object>,
    pointers: HashMap<String, Pointer>,
    unkeyed: bool,
    modes: Mutex<HashMap<Ty, Mode>>,
}

impl Indirect {
    /// The facts of a library: its trait object and function pointer types, and whether rustc touches one with no key.
    pub(super) fn new(objects: Vec<Object>, pointers: Vec<Pointer>, unkeyed: bool) -> Indirect {
        let objects = objects.into_iter().map(|object| (object.object.clone(), object)).collect();
        let pointers = pointers.into_iter().map(|pointer| (pointer.pointer.clone(), pointer)).collect();
        Indirect { objects, pointers, unkeyed, modes: Mutex::default() }
    }

    /// How code pointers of trait object or function pointer type `ty` travel, worked out once per type.
    pub(super) fn mode(&self, cx: &mut Context<'_>, keys: &Keys, ty: Ty) -> Mode {
        if let Some(found) = self.modes.lock().ok().and_then(|modes| modes.get(&ty).cloned()) {
            return found;
        }
        let mut visiting = vec![ty];
        let found = self.work_out(cx, keys, ty, &mut visiting);
        if let Ok(mut modes) = self.modes.lock() {
            modes.entry(ty).or_insert(found).clone()
        } else {
            Mode::Opaque
        }
    }

    /// Whether every trait object and function pointer type in `found` may cross.
    pub(super) fn cross(&self, cx: &mut Context<'_>, keys: &Keys, found: &[Ty]) -> bool {
        found.iter().all(|&ty| !matches!(self.mode(cx, keys, ty), Mode::Opaque))
    }

    fn work_out(&self, cx: &mut Context<'_>, keys: &Keys, ty: Ty, visiting: &mut Vec<Ty>) -> Mode {
        let Ok(key) = keys.ty(cx, ty) else { return Mode::Opaque };
        let kind = cx.types.kind(ty).clone();
        match kind {
            Kind::FnPtr(params, ret) => match self.pointers.get(&key) {
                None if !self.unkeyed => Mode::Free,
                None => Mode::Opaque,
                Some(pointer) => {
                    let params = cx.types.substs(params).to_vec();
                    let neutral = self.method(cx, keys, (&params, Some(ret), 0), (&pointer.abi, &pointer.layouts), visiting).is_some_and(|(_, carries)| carries.iter().all(|&carry| carry == Carry::Same));
                    if neutral { Mode::Free } else { Mode::Opaque }
                }
            },
            Kind::Dyn(of, ..) => match self.objects.get(&key) {
                None if !self.unkeyed => Mode::Free,
                None => Mode::Opaque,
                Some(object) => self.vtable(cx, keys, (ty, of), object, visiting).map_or(Mode::Opaque, |vtable| Mode::Rustc(Arc::new(vtable))),
            },
            _ => Mode::Opaque,
        }
    }

    /// rustc's vtable of trait object type `ty` of trait `of` as `object` describes it, when Haste's vtable holds the same methods and supertraits and every call through a slot can carry its arguments.
    fn vtable(&self, cx: &mut Context<'_>, keys: &Keys, (ty, of): (Ty, ItemRef), object: &Object, visiting: &mut Vec<Ty>) -> Option<Vtable> {
        let methods: Vec<ItemRef> = cx.vtable_methods(of).ok()?.into_iter().flatten().collect();
        if methods.len() != object.slots.len() {
            return None;
        }
        let mut slots = Vec::new();
        for slot in &object.slots {
            let method = *methods.iter().find(|&&method| slot.method.as_deref().is_some_and(|theirs| keys.def(cx, method).is_ok_and(|ours| ours == theirs)))?;
            slots.push(self.slot(cx, keys, (ty, of, method), slot, visiting)?);
        }
        let mut supers = Vec::new();
        for inner in cx.vtable_supers(of).ok()? {
            let ours = keys.def(cx, inner).ok()?;
            match object.supers.iter().find(|theirs| theirs.of.as_deref() == Some(ours.as_str())) {
                Some(found) => supers.push((inner, found.vptr)),
                None if cx.vtable_methods(inner).ok()?.is_empty() => {}
                None => return None,
            }
        }
        Some(Vtable { slots, supers })
    }

    /// Method slot `slot` of the vtable of `ty`, which holds trait method `method` of a trait that `of` implies.
    fn slot(&self, cx: &mut Context<'_>, keys: &Keys, (ty, of, method): (Ty, ItemRef, ItemRef), slot: &Slot, visiting: &mut Vec<Ty>) -> Option<Method> {
        let (params, ret) = virtual_signature(cx, (ty, of), method)?;
        let (params, carries) = self.method(cx, keys, (&params, ret, 1), (&slot.abi, &slot.layouts), visiting)?;
        Some(Method { method, at: slot.at, params, ret, carries })
    }

    /// How each argument of a call through a code pointer that takes `params`, the first `thin` of them receivers, and returns `ret` reaches rustc's convention `abi`, when the layouts it crosses besides its receivers are `layouts` and every trait object and function pointer type among them may cross.
    fn method(&self, cx: &mut Context<'_>, keys: &Keys, (params, ret, thin): (&[Ty], Option<Ty>, usize), (abi, layouts): (&Abi, &Option<Vec<String>>), visiting: &mut Vec<Ty>) -> Option<(Vec<Ty>, Vec<Carry>)> {
        let (spread, carries) = conv::spread_carries(cx, keys, (params, ret), abi, thin)?;
        let mut types = spread.get(thin..)?.to_vec();
        types.push(ret.unwrap_or_else(|| cx.types.tuple(&[])));
        let (printed, found) = conv::layouts(cx, &types).ok()?;
        (layouts.as_ref() == Some(&printed)).then_some(())?;
        for inner in found {
            if visiting.contains(&inner) {
                continue;
            }
            visiting.push(inner);
            let mode = self.work_out(cx, keys, inner, visiting);
            visiting.pop();
            if matches!(mode, Mode::Opaque) {
                return None;
            }
        }
        Some((spread, carries))
    }
}

/// The parameter and return types of the call that a trait object of type `ty` and trait `of` makes through its slot for trait method `method`, whose trait is `of` itself or one without type parameters.
fn virtual_signature(cx: &mut Context<'_>, (ty, of): (Ty, ItemRef), method: ItemRef) -> Option<(Vec<Ty>, Option<Ty>)> {
    let owner = cx.owner_trait(method)?;
    let Kind::Dyn(_, args, members, _) = cx.types.kind(ty).clone() else { return None };
    let substs = if owner == of {
        let all = cx.types.substs(args).to_vec();
        let given = crate::sema::ty::object_parts(&all, &members).0.to_vec();
        std::iter::once(ty).chain(given).collect::<Vec<_>>()
    } else {
        let generics = cx.trait_generics(owner).ok()?;
        crate::sema::infer::type_params(&generics.1.params).next().is_none().then_some(vec![ty])?
    };
    let substs = cx.types.intern_substs(&substs);
    let typed = crate::sema::infer::infer(cx, method, substs).ok()?;
    Some(super::signature(&typed))
}
