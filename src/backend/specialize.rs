use either::Either;
use hir_def::hir::generics::GenericParams;
use hir_def::signatures::FunctionSignature;
use hir_def::{AssocItemId, FunctionId, ImplId, ItemContainerId};
use hir_ty::ParamEnvAndCrate;
use hir_ty::db::HirDatabase;
use hir_ty::method_resolution::TraitImpls;
use hir_ty::next_solver::infer::DbInternerInferExt;
use hir_ty::next_solver::infer::resolve::ReplaceInferWithError;
use hir_ty::next_solver::infer::traits::{Obligation, ObligationCause};
use hir_ty::next_solver::obligation_ctxt::ObligationCtxt;
use hir_ty::next_solver::{AnyImplId, DbInterner, GenericArgs, TraitRef, TypingMode};
use rustc_type_ir::TypeFoldable;

fn item_of(db: &dyn HirDatabase, imp: ImplId, name: &hir_expand::name::Name) -> Option<FunctionId> {
    imp.impl_items(db).items.iter().find_map(|(found, item)| match item {
        AssocItemId::FunctionId(func) if found == name => Some(*func),
        _ => None,
    })
}

fn matches_ref<'db>(
    interner: DbInterner<'db>,
    env: ParamEnvAndCrate<'db>,
    imp: ImplId,
    trait_ref: TraitRef<'db>,
) -> Option<GenericArgs<'db>> {
    let infcx = interner.infer_ctxt().build(TypingMode::PostAnalysis);
    let fresh = infcx.fresh_args_for_item(hir_ty::Span::Dummy, imp.into());
    let declared = interner.db.impl_trait(imp)?.instantiate(interner, fresh).skip_norm_wip();
    let mut ocx = ObligationCtxt::new(&infcx);
    let cause = ObligationCause::dummy();
    ocx.eq(&cause, env.param_env, declared, trait_ref).ok()?;
    for clause in rustc_type_ir::Interner::predicates_of(interner, imp.into()).iter_instantiated(interner, fresh) {
        ocx.register_obligation(Obligation::new(interner, cause.clone(), env.param_env, clause.skip_norm_wip()));
    }
    if !ocx.evaluate_obligations_error_on_ambiguity().is_empty() {
        return None;
    }
    Some(infcx.resolve_vars_if_possible(fresh).fold_with(&mut ReplaceInferWithError::new(interner)))
}

fn matching<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    trait_ref: TraitRef<'db>,
    name: &hir_expand::name::Name,
) -> Vec<(ImplId, FunctionId, GenericArgs<'db>)> {
    let interner = DbInterner::new_with(db, env.krate);
    let owner = trait_ref.def_id.0;
    let mut candidates: Vec<ImplId> = Vec::new();
    TraitImpls::for_each_crate_and_block(db, env.krate, owner.loc(db).container.block(db), &mut |impls| {
        impls.for_trait(owner, |found| {
            if let Either::Left(found) = found {
                candidates.extend_from_slice(found);
            }
        });
    });
    candidates
        .into_iter()
        .filter_map(|imp| {
            let item = item_of(db, imp, name)?;
            Some((imp, item, matches_ref(interner, env, imp, trait_ref)?))
        })
        .collect()
}

pub fn specialized<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    func: FunctionId,
    args: GenericArgs<'db>,
) -> Option<(FunctionId, GenericArgs<'db>)> {
    let ItemContainerId::TraitId(owner) = func.loc(db).container else {
        return None;
    };
    let count = GenericParams::of(db, owner.into()).len();
    let interner = DbInterner::new_with(db, env.krate);
    let trait_ref = TraitRef::new_from_args(interner, owner.into(), GenericArgs::new_from_slice(&args.as_slice()[..count]));
    let found = matching(db, env, trait_ref, &FunctionSignature::of(db, func).name);
    let best = found.iter().find(|(imp, ..)| {
        !found.iter().any(|(other, ..)| {
            other != imp && rustc_type_ir::Interner::impl_specializes(interner, AnyImplId::ImplId(*other), AnyImplId::ImplId(*imp))
        })
    })?;
    let rest = args.as_slice().iter().skip(count).copied();
    Some((best.1, GenericArgs::new_from_iter(best.2.iter().chain(rest))))
}

pub fn parent<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    func: FunctionId,
    args: GenericArgs<'db>,
) -> Option<(FunctionId, GenericArgs<'db>)> {
    let ItemContainerId::ImplId(imp) = func.loc(db).container else {
        return None;
    };
    let interner = DbInterner::new_with(db, env.krate);
    let count = GenericParams::of(db, imp.into()).len();
    let trait_ref = db.impl_trait(imp)?.instantiate(interner, GenericArgs::new_from_slice(&args.as_slice()[..count])).skip_norm_wip();
    let found = matching(db, env, trait_ref, &FunctionSignature::of(db, func).name);
    let best = found.iter().find(|(other, item, _)| {
        rustc_type_ir::Interner::impl_specializes(interner, AnyImplId::ImplId(imp), AnyImplId::ImplId(*other))
            && db.mir_body((*item).into()).is_ok()
    })?;
    let rest = args.as_slice().iter().skip(count).copied();
    Some((best.1, GenericArgs::new_from_iter(best.2.iter().chain(rest))))
}
