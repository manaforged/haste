use std::collections::BTreeSet;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret::{GlobalAlloc, Scalar};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Body, CastKind, Const, ConstOperand, ConstValue, Location, Rvalue, Terminator, TerminatorKind};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::print::{with_no_trimmed_paths, with_no_visible_paths};
use rustc_middle::ty::{self, EarlyBinder, Instance, InstanceKind, Ty, TyCtxt, TypeFoldable, TypingEnv, VtblEntry};
use serde_json::Value;

/// Where the functions are whose MIR rustc's inliner copied into the body of `instance`.
pub fn inlined<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Vec<Value> {
    let body = tcx.instance_mir(instance.def);
    let mut seen = FxHashSet::default();
    let callees = body.source_scopes.iter().filter_map(|scope| scope.inlined).map(|(callee, _)| callee.def_id());
    callees.filter(|def_id| seen.insert(*def_id)).map(|def_id| crate::key::at(tcx, def_id)).collect()
}

/// What the optimized body of `instance` reaches outside itself: the symbols of the instances it calls, drops, reifies or puts in a vtable, and whether it reads or writes a static outside the sysroot that is mutable, thread-local or holds interior mutability, or reaches standard library code whose state each copy of std keeps apart.
pub struct Used {
    pub symbols: Vec<String>,
    pub state: bool,
}

/// What the optimized body of `instance` reaches outside itself.
pub fn used<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Used {
    let body = tcx.instance_mir(instance.def);
    let mut uses = Uses { tcx, instance, body, found: BTreeSet::new(), state: false };
    uses.visit_body(body);
    Used { symbols: uses.found.into_iter().collect(), state: uses.state }
}

/// The standard library code whose state each copy of std keeps for itself: the standard streams, and the exit that flushes them.
const HELD_APART: [&str; 2] = ["std::io::stdio::", "std::process::exit"];

/// Whether `def_id` is standard library code that reads or writes state the shared standard library keeps apart from the std Haste compiles.
fn held_apart(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    if !crate::link::sysroot(tcx, def_id.krate) {
        return false;
    }
    let path = with_no_visible_paths!(with_no_trimmed_paths!(tcx.def_path_str(def_id)));
    HELD_APART.iter().any(|held| path.contains(held))
}

struct Uses<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'a Body<'tcx>,
    found: BTreeSet<String>,
    state: bool,
}

impl<'tcx> Uses<'_, 'tcx> {
    fn mono<T: TypeFoldable<TyCtxt<'tcx>>>(&self, value: T) -> T {
        self.instance.instantiate_mir_and_normalize_erasing_regions(self.tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(self.tcx, value))
    }

    fn add(&mut self, instance: Instance<'tcx>) {
        if matches!(instance.def, InstanceKind::Intrinsic(_) | InstanceKind::Virtual(..)) {
            return;
        }
        self.state |= held_apart(self.tcx, instance.def_id());
        self.found.insert(self.tcx.symbol_name(instance).name.to_string());
    }

    fn touch(&mut self, def_id: DefId) {
        let env = TypingEnv::fully_monomorphized();
        let mutable = self.tcx.static_mutability(def_id) == Some(rustc_hir::Mutability::Mut);
        let cell = !self.tcx.type_of(def_id).instantiate_identity().skip_normalization().is_freeze(self.tcx, env);
        let shared = mutable || cell || self.tcx.is_thread_local_static(def_id);
        self.state |= shared && !crate::link::sysroot(self.tcx, def_id.krate);
    }

    fn constant(&mut self, constant: &ConstOperand<'tcx>) {
        let Const::Val(ConstValue::Scalar(Scalar::Ptr(pointer, _)), _) = constant.const_ else {
            return;
        };
        if let Some(GlobalAlloc::Static(def_id)) = self.tcx.try_get_global_alloc(pointer.provenance.alloc_id()) {
            self.touch(def_id);
        }
    }

    fn function(&mut self, ty: Ty<'tcx>) {
        if let ty::FnDef(def_id, args) = *ty.kind()
            && let Some(args) = args.no_bound_vars()
            && let Ok(Some(instance)) = Instance::try_resolve(self.tcx, TypingEnv::fully_monomorphized(), def_id, args)
        {
            self.add(instance);
        }
    }

    fn closure(&mut self, ty: Ty<'tcx>) {
        if let ty::Closure(def_id, args) = *ty.kind() {
            self.add(Instance::resolve_closure(self.tcx, def_id, args, ty::ClosureKind::FnOnce));
        }
    }

    fn vtable(&mut self, source: Ty<'tcx>, target: Ty<'tcx>) {
        let (Some(source), Some(target)) = (source.builtin_deref(true), target.builtin_deref(true)) else {
            return;
        };
        let ty::Dynamic(predicates, ..) = *target.kind() else {
            return;
        };
        let Some(principal) = predicates.principal() else {
            return;
        };
        let trait_ref = self.tcx.instantiate_bound_regions_with_erased(principal.with_self_ty(self.tcx, source));
        for entry in self.tcx.vtable_entries(trait_ref) {
            if let VtblEntry::Method(method) = entry {
                self.add(*method);
            }
        }
        self.add(Instance::resolve_drop_glue(self.tcx, source));
    }
}

impl<'tcx> Visitor<'tcx> for Uses<'_, 'tcx> {
    fn visit_const_operand(&mut self, constant: &ConstOperand<'tcx>, location: Location) {
        let ty = self.mono(constant.ty());
        self.function(ty);
        self.constant(constant);
        self.super_const_operand(constant, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let TerminatorKind::Drop { place, .. } = &terminator.kind {
            let ty = self.mono(place.ty(self.body, self.tcx).ty);
            self.add(Instance::resolve_drop_glue(self.tcx, ty));
        }
        self.super_terminator(terminator, location);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Rvalue::ThreadLocalRef(def_id) = rvalue {
            self.touch(*def_id);
        }
        if let Rvalue::Cast(CastKind::PointerCoercion(coercion, _), operand, target) = rvalue {
            let source = self.mono(operand.ty(self.body, self.tcx));
            match coercion {
                PointerCoercion::Unsize => self.vtable(source, self.mono(*target)),
                PointerCoercion::ClosureFnPointer(_) => self.closure(source),
                _ => {}
            }
        }
        self.super_rvalue(rvalue, location);
    }
}
