use std::collections::BTreeSet;

use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Body, CastKind, ConstOperand, Location, Rvalue, Terminator, TerminatorKind};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::print::{with_resolve_crate_name, with_no_trimmed_paths};
use rustc_middle::ty::{self, EarlyBinder, Instance, InstanceKind, Ty, TyCtxt, TypeFoldable, TypingEnv, VtblEntry};

/// The definition paths of the functions whose MIR rustc's inliner copied into the body of `instance`.
pub fn inlined<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Vec<String> {
    let body = tcx.instance_mir(instance.def);
    let callees = body.source_scopes.iter().filter_map(|scope| scope.inlined).map(|(callee, _)| callee.def_id());
    let paths: BTreeSet<String> = callees.map(|def_id| with_no_trimmed_paths!(with_resolve_crate_name!(tcx.def_path_str(def_id)))).collect();
    paths.into_iter().collect()
}

/// The symbols of the instances that the optimized body of `instance` calls, drops, reifies or puts in a vtable.
pub fn used<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Vec<String> {
    let body = tcx.instance_mir(instance.def);
    let mut uses = Uses { tcx, instance, body, found: BTreeSet::new() };
    uses.visit_body(body);
    uses.found.into_iter().collect()
}

struct Uses<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'a Body<'tcx>,
    found: BTreeSet<String>,
}

impl<'tcx> Uses<'_, 'tcx> {
    fn mono<T: TypeFoldable<TyCtxt<'tcx>>>(&self, value: T) -> T {
        self.instance.instantiate_mir_and_normalize_erasing_regions(self.tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(self.tcx, value))
    }

    fn add(&mut self, instance: Instance<'tcx>) {
        if matches!(instance.def, InstanceKind::Intrinsic(_) | InstanceKind::Virtual(..)) {
            return;
        }
        self.found.insert(self.tcx.symbol_name(instance).name.to_string());
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
