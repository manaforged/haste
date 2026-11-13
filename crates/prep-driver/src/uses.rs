use std::collections::BTreeSet;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret::{AllocId, GlobalAlloc, Scalar};
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

/// The most statements a leaf's body holds.
const LEAF_STATEMENTS: usize = 16;

/// Whether the optimized body of `instance` is a small leaf: at most `LEAF_STATEMENTS` statements, and no call, drop, assembly or tail call besides calls of intrinsics.
pub fn leaf<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    let body = tcx.instance_mir(instance.def);
    let statements: usize = body.basic_blocks.iter().map(|block| block.statements.len()).sum();
    let plain = |terminator: &Terminator<'tcx>| match &terminator.kind {
        TerminatorKind::Call { func, .. } => matches!(*func.ty(body, tcx).kind(), ty::FnDef(def_id, _) if tcx.intrinsic(def_id).is_some()),
        TerminatorKind::Drop { .. } | TerminatorKind::InlineAsm { .. } | TerminatorKind::TailCall { .. } => false,
        _ => true,
    };
    statements <= LEAF_STATEMENTS && body.basic_blocks.iter().all(|block| plain(block.terminator()))
}

/// What the optimized body of `instance` reaches outside itself: the trait object and function pointer types whose code pointers it makes, calls or hands to the shared standard library, the functions of the shared standard library that take no type arguments it calls or reifies, the symbols of the instances it calls, drops, reifies or puts in a vtable, the symbols of the statics outside the sysroot it reads or writes that are mutable, thread-local or hold interior mutability, the keys of the function pointer types it calls through (none when one has no key), and whether it reaches standard library code whose state each copy of std keeps apart.
pub struct Used<'tcx> {
    pub symbols: Vec<String>,
    pub statics: Vec<String>,
    pub calls: Option<Vec<String>>,
    pub held: bool,
    pub indirect: FxHashSet<Ty<'tcx>>,
    pub upstream: FxHashSet<Instance<'tcx>>,
}

/// What the optimized body of `instance` reaches outside itself.
pub fn used<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Used<'tcx> {
    let body = tcx.instance_mir(instance.def);
    let mut uses = Uses { tcx, instance, body, found: BTreeSet::new(), statics: BTreeSet::new(), calls: Some(BTreeSet::new()), held: false, indirect: FxHashSet::default(), upstream: FxHashSet::default(), seen: FxHashSet::default() };
    uses.visit_body(body);
    let calls = uses.calls.map(|calls| calls.into_iter().collect());
    Used { symbols: uses.found.into_iter().collect(), statics: uses.statics.into_iter().collect(), calls, held: uses.held, indirect: uses.indirect, upstream: uses.upstream }
}

/// Whether static `def_id` holds state that two copies would split: it is outside the sysroot and mutable, thread-local or of a type with interior mutability.
pub fn stateful(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let env = TypingEnv::fully_monomorphized();
    let mutable = tcx.static_mutability(def_id) == Some(rustc_hir::Mutability::Mut);
    let cell = !tcx.type_of(def_id).instantiate_identity().skip_normalization().is_freeze(tcx, env);
    (mutable || cell || tcx.is_thread_local_static(def_id)) && !crate::link::sysroot(tcx, def_id.krate)
}

/// How many allocations deep a body's constant pointers are followed to the functions and statics they reach.
const ALLOCATION_DEPTH: usize = 4;

/// The standard library code whose state each copy of std keeps for itself: the standard streams, and the exit that flushes them.
const HELD_APART: [&str; 2] = ["std::io::stdio::", "std::process::exit"];

/// Whether `def_id` is standard library code that reads or writes state the shared standard library keeps apart from the std Haste compiles.
pub fn held_apart(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
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
    statics: BTreeSet<String>,
    calls: Option<BTreeSet<String>>,
    held: bool,
    indirect: FxHashSet<Ty<'tcx>>,
    upstream: FxHashSet<Instance<'tcx>>,
    seen: FxHashSet<AllocId>,
}

impl<'tcx> Uses<'_, 'tcx> {
    fn mono<T: TypeFoldable<TyCtxt<'tcx>>>(&self, value: T) -> T {
        self.instance.instantiate_mir_and_normalize_erasing_regions(self.tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(self.tcx, value))
    }

    fn add(&mut self, instance: Instance<'tcx>) {
        if let InstanceKind::Virtual(..) = instance.def {
            self.indirect.insert(instance.args.type_at(0));
        }
        if matches!(instance.def, InstanceKind::Intrinsic(_) | InstanceKind::Virtual(..)) {
            return;
        }
        if crate::link::sysroot(self.tcx, instance.def_id().krate) && !instance.args.iter().any(|arg| arg.as_type().is_some()) {
            self.handed(instance);
            if let InstanceKind::Item(_) = instance.def {
                self.upstream.insert(instance);
            }
        }
        self.held |= held_apart(self.tcx, instance.def_id());
        self.found.insert(self.tcx.symbol_name(instance).name.to_string());
    }

    fn touch(&mut self, def_id: DefId) {
        if stateful(self.tcx, def_id) {
            self.statics.insert(self.tcx.symbol_name(Instance::mono(self.tcx, def_id)).name.to_string());
        }
    }

    fn call_through(&mut self, pointer: Ty<'tcx>) {
        let key = crate::key::ty(self.tcx, pointer);
        self.calls = self.calls.take().zip(key).map(|(mut calls, key)| {
            calls.insert(key);
            calls
        });
    }

    /// Notes what constant `constant` points to, evaluated at this instance's type arguments.
    fn constant(&mut self, constant: &ConstOperand<'tcx>) {
        let value = match self.mono(constant.const_) {
            Const::Val(value, _) => Some(value),
            unevaluated => unevaluated.eval(self.tcx, TypingEnv::fully_monomorphized(), constant.span).ok(),
        };
        let id = match value {
            Some(ConstValue::Scalar(Scalar::Ptr(pointer, _))) => pointer.provenance.alloc_id(),
            Some(ConstValue::Indirect { alloc_id, .. }) => alloc_id,
            _ => return,
        };
        self.points_to(id, ALLOCATION_DEPTH);
    }

    /// Notes what a pointer to allocation `id` reaches: a static it touches, a function it may call, and, `depth` allocations deep, what the static's value or the memory points to, so that a function a static holds a pointer to, such as a thread-local's accessor, counts as one the body calls.
    fn points_to(&mut self, id: AllocId, depth: usize) {
        let allocation = match self.tcx.try_get_global_alloc(id) {
            Some(GlobalAlloc::Static(def_id)) => {
                self.touch(def_id);
                self.tcx.eval_static_initializer(def_id).ok()
            }
            Some(GlobalAlloc::Function { instance }) => {
                self.add(instance);
                None
            }
            Some(GlobalAlloc::Memory(allocation)) => Some(allocation),
            _ => None,
        };
        let Some(allocation) = allocation.filter(|_| depth > 0 && self.seen.insert(id)) else {
            return;
        };
        let inner: Vec<AllocId> = allocation.inner().provenance().ptrs().values().map(|pointer| pointer.alloc_id()).collect();
        for inner in inner {
            self.points_to(inner, depth - 1);
        }
    }

    /// Notes the trait objects and function pointers that the arguments and return of `instance`, code of the shared standard library that no record shows, can reach.
    fn handed(&mut self, instance: Instance<'tcx>) {
        let sig = self.tcx.instantiate_bound_regions_with_erased(self.tcx.fn_sig(instance.def_id()).instantiate(self.tcx, instance.args).skip_normalization());
        let types: Vec<Ty<'tcx>> = sig.inputs_and_output.iter().map(|ty| self.tcx.normalize_erasing_regions(TypingEnv::fully_monomorphized(), ty::Unnormalized::new_wip(ty))).collect();
        self.indirect.extend(crate::layout::indirect(self.tcx, &types));
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
        self.indirect.insert(target);
        if let ty::Dynamic(..) = source.kind() {
            self.indirect.insert(source);
            return;
        }
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
        if let TerminatorKind::Call { func, .. } = &terminator.kind
            && let callee = self.mono(func.ty(self.body, self.tcx))
            && let ty::FnPtr(..) = callee.kind()
        {
            self.indirect.insert(callee);
            self.call_through(callee);
        }
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
                PointerCoercion::ClosureFnPointer(_) => {
                    self.indirect.insert(self.mono(*target));
                    self.closure(source);
                }
                PointerCoercion::ReifyFnPointer(_) => _ = self.indirect.insert(self.mono(*target)),
                _ => {}
            }
        }
        self.super_rvalue(rvalue, location);
    }
}
