use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, Instance, TyCtxt, TypingEnv};
use serde_json::{Value, json};

/// The key of a line of a unit's manifest that describes a static holding state.
pub const STATIC: &str = "static";

/// The line that describes static `def_id` when it holds state and has a key: its key, symbol, the layouts its value holds or points to, and whether each thread has its own.
pub fn facts(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Value> {
    if !crate::uses::stateful(tcx, def_id) {
        return None;
    }
    let env = TypingEnv::fully_monomorphized();
    let ty = tcx.normalize_erasing_regions(env, ty::Unnormalized::new_wip(tcx.type_of(def_id).instantiate_identity().skip_normalization()));
    Some(json!({
        STATIC: key(tcx, def_id)?,
        "symbol": tcx.symbol_name(Instance::mono(tcx, def_id)).name,
        "layouts": crate::layout::reached(tcx, &[ty]),
        "thread_local": tcx.is_thread_local_static(def_id),
    }))
}

/// The key of static `def_id`: its definition's, or, for the first static of its name that a function's body declares, the function's key and its name.
fn key(tcx: TyCtxt<'_>, def_id: DefId) -> Option<String> {
    if let Some(key) = crate::key::def(tcx, def_id) {
        return Some(key);
    }
    let parent = tcx.parent(def_id);
    let first = tcx.def_key(def_id).disambiguated_data.disambiguator == 0;
    (first && matches!(tcx.def_kind(parent), DefKind::Fn | DefKind::AssocFn)).then_some(())?;
    Some(crate::source::nested(&crate::key::def(tcx, parent)?, tcx.item_name(def_id).as_str()))
}
