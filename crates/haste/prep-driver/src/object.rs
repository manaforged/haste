use std::ops::ControlFlow;

use rustc_middle::ty::{self, GenericArgs, GenericParamDefKind, Instance, InstanceKind, Ty, TyCtxt};
use rustc_trait_selection::traits::vtable::{VtblSegment, prepare_vtable_segments};
use serde_json::{Value, json};

use crate::error::Result;
use crate::{abi, key, layout};

/// The key of a line of a unit's manifest that describes a trait object type rustc's code touches.
pub const OBJECT: &str = "object";

/// The key of a line of a unit's manifest that describes a function pointer type rustc's code touches.
pub const POINTER: &str = "pointer";

/// The key of the library fact that says rustc's code touches a trait object or function pointer type that has no key.
pub const UNKEYED: &str = "unkeyed";

/// The line that describes trait object or function pointer type `ty`: for a trait object its vtable as rustc lays it out, each method slot with the convention a call through it uses and the layouts that call crosses, and where each supertrait's vtable lies; for a function pointer the convention of a call through it; a library fact when `ty` has no key.
pub fn facts<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<Value> {
    let Some(name) = key::ty(tcx, ty) else { return Ok(json!({ crate::record::LIBRARY: { UNKEYED: true } })) };
    match *ty.kind() {
        ty::Dynamic(predicates, _) => match predicates.principal() {
            Some(principal) => object(tcx, name, tcx.instantiate_bound_regions_with_erased(principal.with_self_ty(tcx, ty))),
            None => Ok(json!({ OBJECT: name, "slots": [], "supers": [] })),
        },
        _ => {
            let (abi, crossing) = abi::of_pointer(tcx, ty)?;
            Ok(json!({ POINTER: name, "abi": abi, "layouts": layout::reached(tcx, &crossing) }))
        }
    }
}

fn object<'tcx>(tcx: TyCtxt<'tcx>, name: String, trait_ref: ty::TraitRef<'tcx>) -> Result<Value> {
    let (mut slots, mut supers, mut at, mut failed) = (Vec::new(), Vec::new(), TyCtxt::COMMON_VTABLE_ENTRIES.len(), None);
    prepare_vtable_segments(tcx, trait_ref, |segment| {
        let VtblSegment::TraitOwnEntries { trait_ref, emit_vptr } = segment else { return ControlFlow::<()>::Continue(()) };
        for &method in tcx.own_existential_vtable_entries(trait_ref.def_id) {
            match slot(tcx, trait_ref, method, at) {
                Ok(found) => slots.push(found),
                Err(error) => failed = Some(error),
            }
            at += 1;
        }
        supers.push(json!({ "trait": key::def(tcx, trait_ref.def_id), "vptr": emit_vptr.then_some(at) }));
        at += usize::from(emit_vptr);
        ControlFlow::Continue(())
    });
    if let Some(error) = failed {
        return Err(error);
    }
    Ok(json!({ OBJECT: name, "slots": slots, "supers": supers }))
}

/// Method slot `at` of a vtable of `trait_ref`, which holds trait method `method`: the method's key, the convention of a call through the slot, and the layouts it crosses besides the receiver, which travels as a thin pointer to the value.
fn slot<'tcx>(tcx: TyCtxt<'tcx>, trait_ref: ty::TraitRef<'tcx>, method: rustc_hir::def_id::DefId, at: usize) -> Result<Value> {
    let args = GenericArgs::for_item(tcx, method, |param, _| match param.kind {
        GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
        GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => trait_ref.args[param.index as usize],
    });
    let (abi, crossing) = abi::of(tcx, Instance { def: InstanceKind::Virtual(method, at), args })?;
    Ok(json!({ "at": at, "method": key::def(tcx, method), "abi": abi, "layouts": layout::reached(tcx, crossing.get(1..).unwrap_or_default()) }))
}
