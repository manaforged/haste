use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::time::Instant;

use either::Either;
use hir_def::hir::generics::GenericParams;
use hir_def::signatures::FunctionSignature;
use hir_def::{AssocItemId, CallableDefId, FunctionId, HasModule, ItemContainerId};
use hir_ty::db::{HirDatabase, InternedClosureId};
use hir_ty::method_resolution::is_dyn_method;
use hir_ty::mir::{CastKind, MirBody, Operand, OperandKind, Rvalue, StatementKind, TerminatorKind, interpret_mir};
use hir_ty::next_solver::{DbInterner, GenericArgs, Ty, TyKind};
use hir_ty::utils::all_super_traits;
use hir_ty::{ParamEnvAndCrate, PointerCast, attach_db};
use rustc_type_ir::inherent::{BoundExistentialPredicates as _, GenericArgs as _, IntoKind};

#[derive(Default)]
struct Walk {
    bodies: usize,
    blocks: usize,
    closures: usize,
    failed: BTreeMap<String, Vec<String>>,
    intrinsics: BTreeSet<String>,
    externs: BTreeSet<String>,
    dyn_calls: usize,
    pointer_calls: usize,
    vtables: usize,
    vtable_unknown: usize,
    unresolved: BTreeSet<String>,
}

pub fn run(db: &dyn HirDatabase, main: FunctionId) {
    let started = Instant::now();
    let walk = attach_db(db, || collect(db, main));
    println!("collect {:.2}s", started.elapsed().as_secs_f64());
    println!(
        "instances_ok {} blocks {} closures {} dyn_calls {} pointer_calls {} vtables {} vtable_unknown {}",
        walk.bodies, walk.blocks, walk.closures, walk.dyn_calls, walk.pointer_calls, walk.vtables, walk.vtable_unknown
    );
    println!("intrinsics {}: {}", walk.intrinsics.len(), join(&walk.intrinsics));
    println!("externs {}: {}", walk.externs.len(), join(&walk.externs));
    println!("unresolved {}: {}", walk.unresolved.len(), join(&walk.unresolved));
    interpret(db, main);
    let failed: usize = walk.failed.values().map(Vec::len).sum();
    println!("instances_failed {failed}");
    for (reason, names) in &walk.failed {
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        println!("  {} {reason}: {}", names.len(), sorted.join(" "));
    }
}

fn join(names: &BTreeSet<String>) -> String {
    names.iter().cloned().collect::<Vec<_>>().join(" ")
}

fn interpret(db: &dyn HirDatabase, main: FunctionId) {
    let started = Instant::now();
    attach_db(db, || {
        let env = ParamEnvAndCrate {
            param_env: db.trait_environment(main.into()),
            krate: main.module(db).krate(db),
        };
        let body = db.monomorphized_mir_body(main.into(), GenericArgs::empty().store(), env.store());
        let Ok(body) = body else {
            println!("interpret: main did not lower");
            return;
        };
        match interpret_mir(db, body, false, None) {
            Ok((result, output)) => println!(
                "interpret ok={} stdout={:?} {:.2}s",
                result.is_ok(),
                output.stdout(),
                started.elapsed().as_secs_f64()
            ),
            Err(error) => println!("interpret setup failed: {error:?}"),
        }
    });
}

enum Item<'db> {
    Func(FunctionId, GenericArgs<'db>),
    Closure(InternedClosureId<'db>, GenericArgs<'db>),
}

struct Collector<'a, 'db> {
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    walk: &'a mut Walk,
    stack: Vec<Item<'db>>,
    seen: HashSet<(FunctionId, GenericArgs<'db>)>,
    seen_closures: HashSet<(InternedClosureId<'db>, GenericArgs<'db>)>,
    seen_types: HashSet<Ty<'db>>,
}

fn collect(db: &dyn HirDatabase, main: FunctionId) -> Walk {
    let env = ParamEnvAndCrate {
        param_env: db.trait_environment(main.into()),
        krate: main.module(db).krate(db),
    };
    let mut walk = Walk::default();
    let mut collector = Collector {
        db,
        env,
        walk: &mut walk,
        stack: vec![Item::Func(main, GenericArgs::empty())],
        seen: HashSet::new(),
        seen_closures: HashSet::new(),
        seen_types: HashSet::new(),
    };
    while let Some(item) = collector.stack.pop() {
        match item {
            Item::Func(func, args) => collector.func(func, args),
            Item::Closure(closure, args) => collector.closure(closure, args),
        }
    }
    walk
}

impl<'db> Collector<'_, 'db> {
    fn name(&self, func: FunctionId) -> String {
        FunctionSignature::of(self.db, func).name.as_str().to_owned()
    }

    fn func(&mut self, func: FunctionId, args: GenericArgs<'db>) {
        if !self.seen.insert((func, args)) {
            return;
        }
        let interner = DbInterner::new_with(self.db, self.env.krate);
        if is_dyn_method(interner, self.env.param_env, func, args).is_some() {
            self.walk.dyn_calls += 1;
            return;
        }
        let (imp, args) = self.db.lookup_impl_method(self.env, func, args);
        let Either::Left(imp) = imp else {
            self.walk.unresolved.insert(format!("derive:{}", self.name(func)));
            return;
        };
        if FunctionSignature::is_intrinsic(self.db, imp) {
            self.walk.intrinsics.insert(self.name(imp));
            return;
        }
        if !FunctionSignature::of(self.db, imp).has_body() {
            if matches!(imp.loc(self.db).container, ItemContainerId::TraitId(_)) {
                self.fn_trait(imp, args);
            } else {
                self.walk.externs.insert(self.name(imp));
            }
            return;
        }
        match self.db.monomorphized_mir_body(imp.into(), args.store(), self.env.store()) {
            Ok(body) => self.body(&body),
            Err(error) => {
                let reason = format!("{error:?}").chars().take(60).collect::<String>();
                let name = self.name(imp);
                self.walk.failed.entry(reason).or_default().push(name);
            }
        }
    }

    fn fn_trait(&mut self, func: FunctionId, args: GenericArgs<'db>) {
        let mut self_ty = args.as_slice().first().and_then(|arg| arg.ty());
        while let Some(TyKind::Ref(_, inner, _)) = self_ty.map(|ty| ty.kind()) {
            self_ty = Some(inner);
        }
        match self_ty.map(|ty| ty.kind()) {
            Some(TyKind::FnDef(def, args)) => {
                if let CallableDefId::FunctionId(id) = def.0 {
                    self.stack.push(Item::Func(id, args));
                }
            }
            Some(TyKind::Closure(closure, args)) => {
                let parent = GenericArgs::new_from_slice(args.as_closure().parent_args());
                self.stack.push(Item::Closure(closure.0, parent));
            }
            Some(TyKind::FnPtr(..)) => self.walk.pointer_calls += 1,
            _ => {
                self.walk.unresolved.insert(self.name(func));
            }
        }
    }

    fn closure(&mut self, closure: InternedClosureId<'db>, args: GenericArgs<'db>) {
        if !self.seen_closures.insert((closure, args)) {
            return;
        }
        self.walk.closures += 1;
        let result = self.db.monomorphized_mir_body_for_closure(closure, args.store(), self.env.store());
        match result {
            Ok(body) => self.body(&body),
            Err(error) => {
                let reason = format!("{error:?}").chars().take(60).collect::<String>();
                self.walk.failed.entry(reason).or_default().push("closure".to_owned());
            }
        }
    }

    fn body(&mut self, body: &MirBody<'db>) {
        self.walk.bodies += 1;
        self.walk.blocks += body.basic_blocks.len();
        for (_, local) in body.locals.iter() {
            self.drops(local.ty.as_ref());
        }
        for (_, block) in body.basic_blocks.iter() {
            for statement in &block.statements {
                if let StatementKind::Assign(_, rvalue) = &statement.kind {
                    self.rvalue(body, rvalue);
                }
            }
            if let Some(terminator) = &block.terminator
                && let TerminatorKind::Call { func, args, .. } = &terminator.kind
            {
                if matches!(func.kind, OperandKind::Copy(_) | OperandKind::Move(_)) {
                    self.walk.pointer_calls += 1;
                }
                self.operand(func);
                args.iter().for_each(|op| self.operand(op));
            }
        }
    }

    fn rvalue(&mut self, body: &MirBody<'db>, rvalue: &Rvalue) {
        match rvalue {
            Rvalue::Use(op) => self.operand(op),
            Rvalue::Cast(kind, op, target) => {
                self.operand(op);
                if matches!(kind, CastKind::PointerCoercion(PointerCast::Unsize)) {
                    self.unsize(body, op, target.as_ref());
                }
            }
            Rvalue::Aggregate(_, ops) => ops.iter().for_each(|op| self.operand(op)),
            _ => {}
        }
    }

    fn unsize(&mut self, body: &MirBody<'db>, op: &Operand, target: Ty<'db>) {
        let Some(TyKind::Dynamic(bounds, _)) = pointee(target).map(|ty| ty.kind()) else {
            return;
        };
        let source = match &op.kind {
            OperandKind::Copy(place) | OperandKind::Move(place) => {
                let place = place.as_ref();
                if place.projection.as_slice().is_empty() {
                    pointee(body.locals[place.local].ty.as_ref())
                } else {
                    None
                }
            }
            OperandKind::Constant { ty, .. } => pointee(ty.as_ref()),
            OperandKind::Allocation { allocation } => pointee(allocation.as_ref().ty),
            OperandKind::Static(_) => None,
        };
        let (Some(source), Some(principal)) = (source, bounds.principal_def_id()) else {
            self.walk.vtable_unknown += 1;
            return;
        };
        self.walk.vtables += 1;
        self.drops(source);
        for trait_id in all_super_traits(self.db, principal.0) {
            for (_, item) in trait_id.trait_items(self.db).items.iter() {
                let AssocItemId::FunctionId(method) = item else {
                    continue;
                };
                if GenericParams::of(self.db, (*method).into()).len() != 0 {
                    continue;
                }
                let args = GenericArgs::new_from_slice(&[source.into()]);
                self.stack.push(Item::Func(*method, args));
            }
        }
    }

    fn drops(&mut self, ty: Ty<'db>) {
        if !self.seen_types.insert(ty) {
            return;
        }
        let Some(drop_fn) = hir_def::lang_item::lang_items(self.db, self.env.krate).Drop_drop else {
            return;
        };
        let args = GenericArgs::new_from_slice(&[ty.into()]);
        let (imp, imp_args) = self.db.lookup_impl_method(self.env, drop_fn, args);
        if let Either::Left(imp) = imp
            && matches!(imp.loc(self.db).container, ItemContainerId::ImplId(_))
        {
            self.stack.push(Item::Func(imp, imp_args));
        }
        match ty.kind() {
            TyKind::Adt(_, args) => {
                for inner in args.as_slice().iter().filter_map(|arg| arg.ty()) {
                    self.drops(inner);
                }
            }
            TyKind::Tuple(items) => items.as_slice().iter().for_each(|item| self.drops(*item)),
            TyKind::Array(inner, _) | TyKind::Slice(inner) => self.drops(inner),
            _ => {}
        }
    }

    fn operand(&mut self, op: &Operand) {
        let ty = match &op.kind {
            OperandKind::Constant { ty, .. } => ty.as_ref(),
            OperandKind::Allocation { allocation } => allocation.as_ref().ty,
            _ => return,
        };
        match ty.kind() {
            TyKind::FnDef(def, args) => {
                if let CallableDefId::FunctionId(id) = def.0 {
                    self.stack.push(Item::Func(id, args));
                }
            }
            TyKind::Closure(closure, args) => {
                let parent = GenericArgs::new_from_slice(args.as_closure().parent_args());
                self.stack.push(Item::Closure(closure.0, parent));
            }
            _ => {}
        }
    }
}

fn pointee(ty: Ty<'_>) -> Option<Ty<'_>> {
    match ty.kind() {
        TyKind::Ref(_, inner, _) | TyKind::RawPtr(inner, _) => Some(inner),
        TyKind::Adt(_, args) => args.as_slice().first().and_then(|arg| arg.ty()),
        _ => None,
    }
}
