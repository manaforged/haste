use rustc_abi::{FieldIdx, LayoutData, VariantIdx};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::ty::{self, Ty, TyCtxt, TypingEnv};

use crate::print;

/// The most types one instance's arguments and return may reach before the driver stops listing their layouts.
const REACHED_TYPES: usize = 256;

/// The layout of every type that the values of `roots` hold or point to, in the order a depth-first walk first meets them, each printed in the form Haste prints its own layouts in, a trait object or function pointer as `print::INDIRECT`; none when one cannot be printed or there are too many.
pub fn reached<'tcx>(tcx: TyCtxt<'tcx>, roots: &[Ty<'tcx>]) -> Option<Vec<String>> {
    let env = TypingEnv::fully_monomorphized();
    let (mut seen, mut stack, mut printed) = (FxHashSet::default(), roots.to_vec(), Vec::new());
    while let Some(ty) = stack.pop() {
        if !seen.insert(ty) {
            continue;
        }
        if seen.len() > REACHED_TYPES {
            return None;
        }
        if indirect_type(ty) {
            printed.push(print::INDIRECT.to_owned());
            continue;
        }
        let layout = tcx.layout_of(env.as_query_input(ty)).ok()?;
        printed.push(print(&layout.layout.0.0));
        stack.extend(inner(tcx, env, ty)?);
    }
    Some(printed)
}

/// The trait object and function pointer types that the values of `roots` hold or point to; every one when a walk cannot see past a type.
pub fn indirect<'tcx>(tcx: TyCtxt<'tcx>, roots: &[Ty<'tcx>]) -> Vec<Ty<'tcx>> {
    let env = TypingEnv::fully_monomorphized();
    let (mut seen, mut stack, mut found) = (FxHashSet::default(), roots.to_vec(), Vec::new());
    while let Some(ty) = stack.pop() {
        if !seen.insert(ty) {
            continue;
        }
        if indirect_type(ty) {
            found.push(ty);
            continue;
        }
        stack.extend(inner(tcx, env, ty).unwrap_or_default());
    }
    found
}

fn indirect_type(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Dynamic(..) | ty::FnPtr(..))
}

fn inner<'tcx>(tcx: TyCtxt<'tcx>, env: TypingEnv<'tcx>, ty: Ty<'tcx>) -> Option<Vec<Ty<'tcx>>> {
    Some(match *ty.kind() {
        ty::Bool | ty::Char | ty::Str | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Never | ty::FnDef(..) => Vec::new(),
        ty::Closure(_, args) => args.as_closure().upvar_tys().to_vec(),
        ty::Ref(_, inner, _) | ty::RawPtr(inner, _) | ty::Slice(inner) | ty::Array(inner, _) | ty::Pat(inner, _) => vec![inner],
        ty::Tuple(items) => items.to_vec(),
        ty::Adt(def, args) => def.all_fields().map(|field| tcx.normalize_erasing_regions(env, field.ty(tcx, args))).collect(),
        _ => return None,
    })
}

/// One layout as text: size, alignment, passing form, field placement and variants with their tag encoding.
pub fn print(layout: &LayoutData<FieldIdx, VariantIdx>) -> String {
    print::layout((layout.size.bytes(), layout.align.abi.bytes()), &layout.backend_repr, &print::fields(&layout.fields), &layout.variants)
}
