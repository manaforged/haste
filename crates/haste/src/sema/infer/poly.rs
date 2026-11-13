use std::sync::Arc;

use anyhow::{Result, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::body::{Body, Bound, ExprId, List, PathId};
use crate::names::{ItemRef, ModRef, Primitive, plain};
use crate::outline::item::ItemKind;

use super::super::Context;
use super::super::intrinsic::Intrinsic;
use super::super::substs::{ProjDef, ProjId, Substs};
use super::super::summary::{ConstHole, PolyObligation, PolySummary, SummaryKind};
use super::super::ty::{Kind, Ty, Types};
use super::project::path_constraints;
use super::scope::{named_bounds, predicate_bounds, type_params};
use super::{Infer, Res, Typed};

const MAX_ENV: usize = 256;
const MAX_CONSTRAINT_DEPTH: u32 = 6;

#[derive(Default)]
pub(in crate::sema) struct PolyState {
    pub env_ready: bool,
    pub fallback: bool,
    pub unsized_params: HashSet<u16>,
    pub assumptions: Vec<(ItemRef, Vec<Ty>)>,
    pub assoc_eqs: Vec<(ItemRef, Ty, Substs, String, Ty)>,
    pub fn_bounds: Vec<(Ty, ItemRef, Ty)>,
    pub obligations: Vec<PolyObligation>,
    pub projections: Vec<ProjId>,
    pub proj_set: HashSet<ProjId>,
    pub offset_holes: Vec<ConstHole>,
}

pub(in crate::sema) fn build_poly_summary(cx: &mut Context<'_>, item: ItemRef, param_count: u16) -> SummaryKind {
    let before = cx.quiet_mark();
    if item.declaring_body().is_some() || cx.attr_at(item, "define_opaque").is_some() {
        return SummaryKind::MonomorphicFallback;
    }
    let (Ok(sig), Ok(body), Ok(home)) = (cx.signature(item), cx.body(item), cx.module_of(item)) else {
        return SummaryKind::MonomorphicFallback;
    };
    if sig.is_async {
        return SummaryKind::MonomorphicFallback;
    }
    let args: Vec<Ty> = (0..param_count).map(|idx| cx.types.intern(Kind::Param(idx))).collect();
    let substs = cx.types.intern_substs(&args);
    let src = cx.body_src(item);
    let root = Some(item.body_owner());
    let mut infer = Infer::start(cx, (&body, &sig, src), home);
    (infer.root, infer.instance) = (root, Some((item, substs)));
    infer.poly = Some(Box::new(PolyState::default()));
    let Ok(mut frame) = infer.frame(item, &args) else { return SummaryKind::MonomorphicFallback };
    (frame.src, frame.root) = (src, root);
    infer.swap_frame(frame);
    if infer.init_poly_env(item).is_err() {
        return SummaryKind::MonomorphicFallback;
    }
    infer.out.exprs.resize(body.exprs.len(), Ty::NONE);
    if infer.run(false).is_err() {
        return SummaryKind::MonomorphicFallback;
    }
    let Ok((mut typed, poly, clean)) = infer.finish_poly() else { return SummaryKind::MonomorphicFallback };
    if cx.quiet_mark() != before || (!typed.coercions.is_empty() && super::object::fill_drops(cx, &mut typed).is_err()) {
        return SummaryKind::MonomorphicFallback;
    }
    finalize_summary(cx, typed, poly, clean, param_count)
}

pub(super) fn zonk_poly_state(infer: &mut Infer<'_, '_>, poly: &mut PolyState) {
    poly.obligations = zonk_obligations(infer, &mut poly.obligations);
    let raw_offsets = std::mem::take(&mut poly.offset_holes);
    poly.offset_holes = zonk_offset_holes(infer, raw_offsets);
}

fn finalize_summary(cx: &Context<'_>, typed: Typed, poly: Option<Box<PolyState>>, clean: bool, param_count: u16) -> SummaryKind {
    let Some(mut poly) = poly else { return SummaryKind::MonomorphicFallback };
    if poly.fallback || !clean || has_param_closure(cx, &typed) {
        return SummaryKind::MonomorphicFallback;
    }
    collect_remaining_projs(&cx.types, &mut poly, &typed);
    let mut const_holes = poly.offset_holes;
    let mut layout_tys = Vec::new();
    let mut layout_seen = HashSet::default();
    collect_layouts_and_consts(cx, &typed, &mut layout_tys, &mut layout_seen, &mut const_holes);
    let super::LazyTables::Eager(template) = typed.tables else { return SummaryKind::MonomorphicFallback };
    SummaryKind::Poly(Arc::new(PolySummary {
        param_count,
        template,
        locals: typed.locals.into_boxed_slice(),
        params: typed.params.into_boxed_slice(),
        ret: typed.ret,
        obligations: poly.obligations.into_boxed_slice(),
        projections: poly.projections.into_boxed_slice(),
        parametric_tys: Box::new([]),
        layout_tys: layout_tys.into_boxed_slice(),
        const_holes: const_holes.into_boxed_slice(),
        fixed_calls: Box::default(),
        call_holes: Box::default(),
        op_holes: Box::default(),
        fixed_drops: Box::default(),
        drop_tys: Box::default(),
    }))
}

fn has_param_closure(cx: &Context<'_>, typed: &Typed) -> bool {
    typed
        .closures
        .as_slice()
        .iter()
        .any(|(_, c)| c.captures.iter().any(|cap| cx.types.has_params(typed.locals[cap.local as usize].ty)))
}

fn zonk_obligations(infer: &mut Infer<'_, '_>, raw: &mut Vec<PolyObligation>) -> Vec<PolyObligation> {
    let mut out = Vec::with_capacity(raw.len());
    for entry in raw.drain(..) {
        let mapped = match entry {
            PolyObligation::Prove { of, self_ty, args } => {
                let self_ty = infer.zonk(self_ty);
                let mapped_args = infer.map_substs(args, Infer::zonk);
                PolyObligation::Prove { of, self_ty, args: mapped_args }
            }
            PolyObligation::Call { expr, found, tuple } => {
                PolyObligation::Call { expr, found: infer.zonk(found), tuple: infer.zonk(tuple) }
            }
        };
        if !out.contains(&mapped) {
            out.push(mapped);
        }
    }
    out
}

fn zonk_offset_holes(infer: &mut Infer<'_, '_>, raw: Vec<ConstHole>) -> Vec<ConstHole> {
    raw.into_iter()
        .map(|hole| match hole {
            ConstHole::OffsetOf { expr, steps } => {
                let mapped: Vec<(Ty, u32)> = steps.iter().map(|&(ty, idx)| (infer.zonk(ty), idx)).collect();
                ConstHole::OffsetOf { expr, steps: mapped.into_boxed_slice() }
            }
            other => other,
        })
        .collect()
}

fn collect_layouts_and_consts(
    cx: &Context<'_>,
    typed: &Typed,
    layout_tys: &mut Vec<Ty>,
    layout_seen: &mut HashSet<Ty>,
    const_holes: &mut Vec<ConstHole>,
) {
    let mut push_layout = |ty: Ty| {
        if ty != Ty::NONE && cx.types.has_params(ty) && layout_seen.insert(ty) {
            layout_tys.push(ty);
        }
    };
    for hole in const_holes.iter() {
        if let ConstHole::OffsetOf { steps, .. } = hole {
            steps.iter().for_each(|&(ty, _)| push_layout(ty));
        }
    }
    typed.locals.iter().for_each(|local| push_layout(local.ty));
    typed.exprs.iter().copied().for_each(&mut push_layout);
    if let Some(ret) = typed.ret {
        push_layout(ret);
    }
    collect_res_holes(cx, typed, &mut push_layout, const_holes);
}

fn collect_res_holes(cx: &Context<'_>, typed: &Typed, push_layout: &mut impl FnMut(Ty), const_holes: &mut Vec<ConstHole>) {
    for res in typed.res.values().chain(typed.pat_res.values()) {
        if let &Res::Const(ty) = res
            && cx.types.has_params(ty)
        {
            let hole = ConstHole::Value(ty);
            if !const_holes.contains(&hole) {
                const_holes.push(hole);
            }
        }
    }
    for (item, substs) in typed.calls() {
        let Some(intrinsic) = cx.intrinsic(item) else { continue };
        for &arg in cx.types.substs(substs) {
            push_layout(arg);
        }
        if cx.types.substs_have_params(substs) && matches!(intrinsic, Intrinsic::SizeOf | Intrinsic::AlignOf | Intrinsic::NeedsDrop | Intrinsic::TypeId | Intrinsic::TypeName) {
            let hole = ConstHole::Intrinsic { item, substs };
            if !const_holes.contains(&hole) {
                const_holes.push(hole);
            }
        }
    }
}

fn push_proj_topo(types: &Types, state: &mut PolyState, seen: &mut HashSet<Ty>, id: ProjId) {
    if state.proj_set.contains(&id) {
        return;
    }
    let def = types.proj(id);
    visit_ty_projs(types, state, seen, def.self_ty);
    for &arg in types.substs(def.args) {
        visit_ty_projs(types, state, seen, arg);
    }
    for &arg in types.substs(def.own) {
        visit_ty_projs(types, state, seen, arg);
    }
    if state.proj_set.insert(id) {
        state.projections.push(id);
    }
}

fn visit_ty_projs(types: &Types, state: &mut PolyState, seen: &mut HashSet<Ty>, ty: Ty) {
    if !types.has_projs(ty) || !seen.insert(ty) {
        return;
    }
    let kind = types.kind(ty).clone();
    match kind {
        Kind::Proj(inner) => push_proj_topo(types, state, seen, inner),
        Kind::Closure(_, args, _) => types.substs(args).iter().for_each(|&arg| visit_ty_projs(types, state, seen, arg)),
        _ => kind.children(types).for_each(|child| visit_ty_projs(types, state, seen, child)),
    }
}

fn collect_remaining_projs(types: &Types, state: &mut PolyState, typed: &Typed) {
    let mut seen = HashSet::default();
    let res_ty = |r: &Res, seen: &mut HashSet<Ty>, st: &mut PolyState| match *r {
        Res::Const(t) => visit_ty_projs(types, st, seen, t),
        Res::Fn(_, s) | Res::Ctor(_, _, s) => types.substs(s).iter().for_each(|&t| visit_ty_projs(types, st, seen, t)),
        Res::Local(_) => {}
    };
    let tys = typed.locals.iter().map(|l| l.ty).chain(typed.ret).chain(typed.exprs.iter().copied()).chain(typed.pats.values().copied()).chain(typed.closures.values().map(|c| c.ret));
    let flow_tys = typed.fors.values().flat_map(|f| [f.iter, f.option]).chain(typed.tries.values().map(|t| t.flow)).chain(typed.awaits.values().flat_map(|a| [a.future, a.polled]));
    let hole_tys: Vec<Ty> = state.obligations.iter().flat_map(|o| match *o { PolyObligation::Prove { self_ty, args, .. } => { let mut v = types.substs(args).to_vec(); v.push(self_ty); v } PolyObligation::Call { found, tuple, .. } => vec![found, tuple] }).chain(state.offset_holes.iter().flat_map(|h| match h { ConstHole::OffsetOf { steps, .. } => steps.iter().map(|&(t, _)| t).collect::<Vec<_>>(), _ => Vec::new() })).collect();
    tys.chain(flow_tys).chain(hole_tys).for_each(|ty| visit_ty_projs(types, state, &mut seen, ty));
    let direct = typed.res.values().chain(typed.pat_res.values()).chain(typed.derefs.values()).chain(typed.operators.values()).chain(typed.try_blocks.values()).chain(typed.box_frees.values());
    let flow_res = typed.fors.values().flat_map(|f| [&f.into_iter, &f.next]).chain(typed.tries.values().flat_map(|t| [&t.branch, &t.from_residual])).chain(typed.awaits.values().flat_map(|a| [&a.into_future, &a.poll]));
    direct.chain(flow_res).for_each(|r| res_ty(r, &mut seen, state));
    for c in typed.coercions.values() {
        match c {
            super::Coercion::Len(t) => visit_ty_projs(types, state, &mut seen, *t),
            super::Coercion::Reify(r) => res_ty(r, &mut seen, state),
            super::Coercion::Deref(steps, to) => {
                steps.iter().for_each(|(t, r)| { visit_ty_projs(types, state, &mut seen, *t); r.iter().for_each(|r| res_ty(r, &mut seen, state)); });
                visit_ty_projs(types, state, &mut seen, *to);
            }
            _ => {}
        }
    }
}

impl Infer<'_, '_> {
    pub(super) fn poly_fallback(&mut self) {
        if let Some(poly) = &mut self.poly {
            poly.fallback = true;
        }
    }

    pub(super) fn init_poly_env(&mut self, item: ItemRef) -> Result<()> {
        if self.cx.owner_trait(item).is_some() && self.cx.owner_impl(item).is_none()
            && let Some(poly) = &mut self.poly
        {
            poly.unsized_params.insert(0);
        }
        if let Some(own) = self.own_trait()? {
            self.push_assumption(own);
        }
        self.collect_header_env()?;
        if let Some(poly) = &mut self.poly {
            poly.env_ready = true;
        }
        self.close_poly_env()?;
        self.prune_sized_params();
        Ok(())
    }

    fn collect_header_env(&mut self) -> Result<()> {
        let params = self.generics.clone();
        let self_ty = self.self_ty;
        for (header, src) in self.headers()? {
            let (arena, generics, predicates) = header.parts();
            self.in_source(src, |infer| {
                let home = infer.home;
                for &(name, ty) in &params {
                    for bound in named_bounds(arena, src, (generics, predicates), name) {
                        infer.note_maybe_unsized(arena, home, bound, ty);
                        infer.collect_env_bound(arena, home, (ty, bound), 0)?;
                    }
                }
                if let Some(self_ty) = self_ty {
                    for bound in predicate_bounds(arena, src, predicates, "Self") {
                        infer.note_maybe_unsized(arena, home, bound, self_ty);
                        infer.collect_env_bound(arena, home, (self_ty, bound), 0)?;
                    }
                }
                for (ty, bound) in infer.typed_predicates(arena, src, predicates) {
                    infer.note_maybe_unsized(arena, home, bound, ty);
                    infer.collect_env_bound(arena, home, (ty, bound), 0)?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    fn bound_relaxes_sized(&mut self, arena: &Body, home: ModRef, bound: u32) -> bool {
        match arena.bounds[bound as usize] {
            Bound::Trait { maybe: true, .. } => true,
            Bound::Trait { path, maybe: false, .. } => self
                .trait_named(arena, path, home)
                .and_then(|of| self.cx.lang(of))
                .is_some_and(|lang| matches!(lang, "pointee_sized" | "meta_sized")),
            _ => false,
        }
    }

    fn note_maybe_unsized(&mut self, arena: &Body, home: ModRef, bound: u32, ty: Ty) {
        if self.bound_relaxes_sized(arena, home, bound)
            && let Kind::Param(idx) = *self.cx.types.kind(self.resolve(ty))
            && let Some(poly) = &mut self.poly
        {
            poly.unsized_params.insert(idx);
        }
    }

    fn collect_env_bound(&mut self, arena: &Body, home: ModRef, (ty, bound): (Ty, u32), depth: u32) -> Result<()> {
        if depth > MAX_CONSTRAINT_DEPTH {
            return Ok(());
        }
        let Some((of, path)) = self.bound_trait_at(arena, bound, home) else { return Ok(()) };
        let args = self.bound_args(of, arena, path, ty)?;
        let full: Vec<Ty> = std::iter::once(ty).chain(args.iter().copied()).collect();
        self.push_assumption((of, full));
        if self.is_callable_trait(of)
            && let Some(sig) = self.fn_bound(arena, home, bound)?
        {
            self.push_fn_bound(ty, of, sig);
        }
        for (assoc_name, assoc_ty) in self.path_bindings(arena, path)? {
            self.record_assoc_eq((of, args.clone()), ty, assoc_name, assoc_ty)?;
        }
        for (name_tok, sub_bounds) in path_constraints(arena, path) {
            let assoc_name = self.src.text(name_tok).to_string();
            let Some((owner, owner_args)) = self.assoc_owner((of, args.clone()), ty, &assoc_name, 0)? else { continue };
            let assoc = self.project(owner, ty, owner_args, &assoc_name)?;
            for &sub in arena.list(sub_bounds) {
                self.collect_env_bound(arena, home, (assoc, sub), depth + 1)?;
            }
        }
        Ok(())
    }

    fn close_poly_env(&mut self) -> Result<()> {
        let mut at = 0;
        while let Some(pair) = self.poly.as_ref().and_then(|poly| (at < poly.assumptions.len() && poly.assumptions.len() < MAX_ENV).then(|| poly.assumptions[at].clone())) {
            let (of, substs) = pair;
            for ((parent, parent_substs), bindings) in self.super_bindings(of, &substs)? {
                self.push_assumption((parent, parent_substs.clone()));
                if self.is_callable_trait(of) && self.is_callable_trait(parent) {
                    self.propagate_fn_bound(substs[0], of, parent);
                }
                for (assoc_name, assoc_ty) in bindings {
                    self.record_assoc_eq((parent, parent_substs[1..].to_vec()), substs[0], assoc_name, assoc_ty)?;
                }
            }
            self.collect_alias_decl_bounds(of, &substs)?;
            at += 1;
        }
        Ok(())
    }

    fn collect_alias_decl_bounds(&mut self, of: ItemRef, substs: &[Ty]) -> Result<()> {
        let def = self.cx.trait_generics(of)?;
        let params: Vec<_> = type_params(&def.1.params).copied().collect();
        if substs.len() != params.len() + 1 {
            return Ok(());
        }
        let members: Vec<ItemRef> = self.cx.children(of).into_iter().filter(|&m| self.cx.item(m).1.kind == ItemKind::Type).collect();
        for member in members {
            let bounds = self.cx.trait_generics(member)?;
            if bounds.1.supers.len == 0 {
                continue;
            }
            let member_name = self.cx.name(member).to_string();
            let Some(value) = self.picked_alias(of, substs[0], &substs[1..], (&member_name, &[]))? else { continue };
            let frame = self.item_frame(of, (&params, &substs[1..]), (Some(substs[0]), None))?;
            let (home, src) = (frame.home, frame.src);
            self.with_frame(frame, |infer| {
                infer.in_source(src, |infer| {
                    for &bound in bounds.0.list(bounds.1.supers) {
                        infer.collect_env_bound(&bounds.0, home, (value, bound), 1)?;
                    }
                    Ok(())
                })
            })?;
        }
        Ok(())
    }

    fn prune_sized_params(&mut self) {
        let Some(sized) = self.cx.lang_item("sized") else { return };
        let Some(poly) = &mut self.poly else { return };
        let relaxed_params: Vec<u16> = poly.unsized_params.iter().copied().collect();
        for idx in relaxed_params {
            let param_ty = self.cx.types.intern(Kind::Param(idx));
            if poly.assumptions.iter().any(|(of, tys)| *of == sized && tys.as_slice() == [param_ty]) {
                poly.unsized_params.remove(&idx);
            }
        }
    }

    fn push_assumption(&mut self, entry: (ItemRef, Vec<Ty>)) {
        if let Some(poly) = &mut self.poly
            && poly.assumptions.len() < MAX_ENV
            && !poly.assumptions.contains(&entry)
        {
            poly.assumptions.push(entry);
        }
    }

    fn push_fn_bound(&mut self, ty: Ty, of: ItemRef, sig: Ty) {
        if let Kind::FnPtr(inputs, output) = *self.cx.types.kind(sig)
            && let Some(once) = self.cx.lang_item("fn_once")
        {
            let tuple = self.cx.types.intern(Kind::Tuple(inputs));
            let trait_args = self.cx.types.intern_substs(&[tuple]);
            self.push_assoc_eq_raw(once, ty, trait_args, "Output".to_string(), output);
        }
        if let Some(poly) = &mut self.poly
            && !poly.fn_bounds.contains(&(ty, of, sig))
        {
            poly.fn_bounds.push((ty, of, sig));
        }
    }

    fn propagate_fn_bound(&mut self, ty: Ty, from: ItemRef, to: ItemRef) {
        let sigs: Vec<Ty> = self
            .poly
            .as_ref()
            .map(|p| p.fn_bounds.iter().filter(|&&(t, o, _)| t == ty && o == from).map(|&(_, _, s)| s).collect())
            .unwrap_or_default();
        for sig in sigs {
            self.push_fn_bound(ty, to, sig);
        }
    }

    fn record_assoc_eq(&mut self, (of, args): (ItemRef, Vec<Ty>), ty: Ty, name: String, eq_ty: Ty) -> Result<()> {
        let Some((owner, owner_args)) = self.assoc_owner((of, args), ty, &name, 0)? else { return Ok(()) };
        let interned = self.cx.types.intern_substs(&owner_args);
        self.push_assoc_eq_raw(owner, ty, interned, name, eq_ty);
        Ok(())
    }

    fn push_assoc_eq_raw(&mut self, owner: ItemRef, ty: Ty, args: Substs, name: String, eq_ty: Ty) {
        if let Some(poly) = &mut self.poly
            && !poly.assoc_eqs.iter().any(|(o, t, a, n, _)| *o == owner && *t == ty && *a == args && *n == name)
        {
            poly.assoc_eqs.push((owner, ty, args, name, eq_ty));
        }
    }

    pub(super) fn poly_sized(&mut self, ty: Ty) -> bool {
        let tail = self.tail_ty(ty).unwrap_or_else(|_| self.resolve(ty));
        let tail = self.resolve(tail);
        match *self.cx.types.kind(tail) {
            Kind::Slice(_) | Kind::Prim(Primitive::Str) | Kind::Dyn(..) => false,
            Kind::Param(idx) => self.poly.as_ref().is_none_or(|poly| !poly.unsized_params.contains(&idx)),
            Kind::Proj(id) => self.proj_is_sized(tail, id),
            _ => true,
        }
    }

    fn proj_is_sized(&mut self, proj_ty: Ty, id: ProjId) -> bool {
        if let Some(sized) = self.cx.lang_item("sized")
            && self.poly.as_ref().is_some_and(|p| p.assumptions.iter().any(|(of, tys)| *of == sized && tys.as_slice() == [proj_ty]))
        {
            return true;
        }
        let def = self.cx.types.proj(id);
        let (Ok(bounds), Ok(assoc_home)) = (self.cx.trait_generics(def.assoc), self.cx.module_of(def.assoc)) else {
            return false;
        };
        let assoc_src = &self.cx.file(def.assoc).source;
        let relaxed = self.in_source(assoc_src, |infer| {
            Ok(bounds.0.list(bounds.1.supers).iter().any(|&b| infer.bound_relaxes_sized(&bounds.0, assoc_home, b)))
        });
        !relaxed.unwrap_or(true)
    }

    pub(super) fn poly_is_copy(&mut self, ty: Ty) -> Result<bool> {
        let Some(copy) = self.cx.lang_item("copy") else { return Ok(false) };
        self.poly_holds_in_env(copy, ty, &[])
    }

    pub(super) fn poly_fn_sig(&self, ty: Ty) -> Option<(Vec<Ty>, Ty)> {
        let ty = self.resolve(ty);
        let poly = self.poly.as_ref()?;
        let &(_, _, sig) = poly.fn_bounds.iter().find(|&&(bound_ty, _, _)| self.resolve(bound_ty) == ty)?;
        let Kind::FnPtr(params, ret) = *self.cx.types.kind(sig) else { return None };
        Some((self.cx.types.substs(params).to_vec(), ret))
    }

    pub(super) fn poly_may_implement(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        let Some(assumptions) = self.poly.as_ref().map(|poly| poly.assumptions.clone()) else { return Ok(false) };
        let wanted: Vec<Ty> = std::iter::once(self_ty).chain(args.iter().copied()).collect();
        for (held, types) in assumptions {
            if held == of && types.len() == wanted.len() && self.probe(|infer| wanted.iter().zip(&types).all(|(&a, &b)| infer.unifies(a, b))) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) fn poly_holds_in_env(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        let Some(assumptions) = self.poly.as_ref().map(|poly| poly.assumptions.clone()) else { return Ok(false) };
        let wanted: Vec<Ty> = std::iter::once(self_ty).chain(args.iter().copied()).collect();
        if !self.obligation_open(self_ty, args) {
            let resolved: Vec<Ty> = wanted.into_iter().map(|t| self.substitute(t)).collect();
            return Ok(assumptions.iter().any(|(held, types)| {
                *held == of && types.len() == resolved.len() && types.iter().zip(&resolved).all(|(&a, &b)| self.substitute(a) == b)
            }));
        }
        let mut matched = Vec::new();
        for (held, types) in assumptions {
            if held == of && types.len() == wanted.len() && !matched.contains(&types) && self.probe(|infer| wanted.iter().zip(&types).all(|(&a, &b)| infer.unifies(a, b))) {
                matched.push(types);
            }
        }
        if let [unique] = &matched[..] {
            wanted.iter().zip(unique).try_for_each(|(&a, &b)| self.unify(a, b))?;
        }
        Ok(!matched.is_empty())
    }

    pub(super) fn poly_bound_check(
        &mut self,
        (arena, home, bound): (&Body, ModRef, u32),
        (of, path, ty): (ItemRef, PathId, Ty),
        depth: u32,
    ) -> Result<Option<bool>> {
        let kind = self.cx.types.kind(self.resolve(ty));
        if self.is_callable_trait(of) && matches!(kind, Kind::Param(_) | Kind::Proj(_)) {
            return self.poly_callable_bound_holds((arena, home, bound), (of, path, ty), depth).map(Some);
        }
        let args = self.bound_args(of, arena, path, ty)?;
        if self.has_params(ty) || self.any_has_params(&args) {
            let ty = self.substitute(ty);
            let args: Vec<Ty> = args.into_iter().map(|a| self.substitute(a)).collect();
            if self.poly_env_holds((arena, home, path), (of, ty, &args), depth)? {
                return Ok(Some(true));
            }
        }
        Ok(None)
    }

    pub(super) fn poly_env_holds(&mut self, (arena, home, path): (&Body, ModRef, PathId), (of, ty, args): (ItemRef, Ty, &[Ty]), depth: u32) -> Result<bool> {
        if !self.poly_holds_in_env(of, ty, args)? {
            return Ok(false);
        }
        self.assoc_bounds_hold((arena, home, path), (of, ty, args), depth)
    }

    pub(super) fn poly_callable_bound_holds(&mut self, (arena, home, bound): (&Body, ModRef, u32), (of, path, ty): (ItemRef, PathId, Ty), depth: u32) -> Result<bool> {
        let ty = self.resolve(ty);
        let found = self
            .poly
            .as_ref()
            .and_then(|p| p.fn_bounds.iter().find(|&&(t, o, _)| self.resolve(t) == ty && o == of).map(|&(_, _, s)| s));
        if let Some(sig) = found {
            let Some(wanted) = self.fn_bound(arena, home, bound)? else { return Ok(true) };
            return Ok(self.unifies(sig, wanted));
        }
        let args = self.bound_args(of, arena, path, ty)?;
        self.poly_env_holds((arena, home, path), (of, ty, &args), depth)
    }

    pub(super) fn poly_picked_alias(&mut self, of: ItemRef, ty: Ty, args: &[Ty], (name, own): (&str, &[Ty])) -> Result<Option<Ty>> {
        if self.has_vars(ty) || args.iter().any(|&a| self.has_vars(a)) || own.iter().any(|&o| self.has_vars(o)) {
            return Ok(None);
        }
        let ty = self.zonk(ty);
        let args: Vec<Ty> = args.iter().map(|&a| self.zonk(a)).collect();
        let own: Vec<Ty> = own.iter().map(|&o| self.zonk(o)).collect();
        if self.cx.lang(of) == Some("discriminant_kind") {
            return self.discriminant_ty(ty);
        }
        let interned_args = self.cx.types.intern_substs(&args);
        if let Some(fast) = self.poly_fast_alias(of, ty, interned_args, (name, &own))? {
            return Ok(Some(fast));
        }
        self.poly_selected_or_proj(of, ty, (&args, interned_args), (name, &own))
    }

    fn poly_fast_alias(&mut self, of: ItemRef, ty: Ty, interned_args: Substs, (name, own): (&str, &[Ty])) -> Result<Option<Ty>> {
        if own.is_empty()
            && let Some(eq_ty) = self.lookup_assoc_eq(of, ty, interned_args, name)
        {
            return Ok(Some(eq_ty));
        }
        self.builtin_alias(of, ty, name)
    }

    fn poly_selected_or_proj(&mut self, of: ItemRef, ty: Ty, (args, interned_args): (&[Ty], Substs), (name, own): (&str, &[Ty])) -> Result<Option<Ty>> {
        let selected = self.select(of, ty, args)?;
        if let Some(picked) = &selected
            && let Some(concrete) = self.non_default_impl_alias(of, ty, args, picked.clone(), (name, own))?
        {
            return Ok(Some(concrete));
        }
        if selected.is_none() && !self.poly_holds_in_env(of, ty, args)? {
            return Ok(None);
        }
        let Some(assoc) = self.cx.trait_member(of, name) else { return Ok(None) };
        self.poly_prove(of, ty, args);
        let own_args = self.cx.types.intern_substs(own);
        let proj_id = self.cx.types.intern_proj(ProjDef { of, self_ty: ty, args: interned_args, assoc, own: own_args });
        if let Some(poly) = &mut self.poly {
            let mut seen = HashSet::default();
            push_proj_topo(&self.cx.types, poly, &mut seen, proj_id);
        }
        Ok(Some(self.cx.types.intern(Kind::Proj(proj_id))))
    }

    fn non_default_impl_alias(&mut self, of: ItemRef, ty: Ty, args: &[Ty], picked: super::select::Picked, (name, own): (&str, &[Ty])) -> Result<Option<Ty>> {
        let (block, impl_args) = self.impl_member(of, (ty, args), picked.clone(), name)?.map_or(picked, |(_, found)| found);
        if self.cx.specializable(block) {
            return Ok(None);
        }
        let frame = self.impl_frame_of(block, &impl_args, Some(ty))?;
        match self.with_frame(frame, |infer| infer.impl_alias(block, (name, own)))? {
            Some(found) => Ok(Some(found)),
            None => bail!("the impl of `{}` for {} has no type `{name}`", self.cx.name(of), self.show(ty)),
        }
    }

    fn lookup_assoc_eq(&mut self, of: ItemRef, ty: Ty, args: Substs, name: &str) -> Option<Ty> {
        let len = self.poly.as_ref()?.assoc_eqs.len();
        for idx in 0..len {
            let (o, t, a, eq, same_name) = {
                let (o, t, a, n, eq) = &self.poly.as_ref()?.assoc_eqs[idx];
                (*o, *t, *a, *eq, n == name)
            };
            if o == of && a == args && same_name && self.substitute(t) == ty {
                return Some(eq);
            }
        }
        None
    }

    pub(super) fn poly_prove(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) {
        let self_ty = self.resolve(self_ty);
        let resolved: Vec<Ty> = args.iter().map(|&a| self.resolve(a)).collect();
        let interned = self.cx.types.intern_substs(&resolved);
        let entry = PolyObligation::Prove { of, self_ty, args: interned };
        if let Some(poly) = &mut self.poly
            && !poly.obligations.contains(&entry)
        {
            poly.obligations.push(entry);
        }
    }

    pub(super) fn poly_settle_obligation(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        if self.compiler_decides(of, self_ty) {
            if !self.compiler_holds(of, self_ty) {
                self.poly_fallback();
            }
            return Ok(true);
        }
        if self.compiler_impl(of, self_ty) || self.object_holds(of, self_ty, args) == Some(true) {
            return Ok(true);
        }
        if self.poly_holds_in_env(of, self_ty, args)? || self.select(of, self_ty, args)?.is_some() {
            self.poly_prove(of, self_ty, args);
            return Ok(true);
        }
        self.poly_fallback();
        Ok(true)
    }

    pub(super) fn select_or_assume(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        if self.select(of, self_ty, args)?.is_some() {
            return Ok(true);
        }
        if self.poly.is_some() {
            let full: Vec<Ty> = std::iter::once(self_ty).chain(args.iter().copied()).collect();
            return self.assume(of, &full);
        }
        Ok(false)
    }

    pub(super) fn select_or_prove(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        let has_params = self.poly.is_some() && (self.has_params(self_ty) || self.any_has_params(args));
        if self.trial(|infer| infer.select(of, self_ty, args))?.is_some() {
            if has_params {
                self.poly_prove(of, self_ty, args);
            }
            return Ok(true);
        }
        if has_params {
            let full: Vec<Ty> = std::iter::once(self_ty).chain(args.iter().copied()).collect();
            let _ = self.assume(of, &full)?;
            if self.poly_holds_in_env(of, self_ty, args)? {
                self.poly_prove(of, self_ty, args);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) fn poly_only_right(&mut self, of: ItemRef, left: Ty) -> Option<Ty> {
        let candidates: Vec<(Ty, Ty)> = self
            .poly
            .as_ref()?
            .assumptions
            .iter()
            .filter(|(held, types)| *held == of && types.len() == 2)
            .map(|(_, types)| (types[0], types[1]))
            .collect();
        let mut found = None;
        for (lhs, rhs) in candidates {
            if self.substitute(lhs) == left {
                let mapped = self.substitute(rhs);
                if found.replace(mapped).is_some_and(|prev| prev != mapped) {
                    return None;
                }
            }
        }
        found
    }

    pub(super) fn poly_call(&mut self, expr: ExprId, found: Ty, callee_ty: Ty, args: List) -> Result<Ty> {
        let Some((method, params, ret)) = self.poly_callable_target(callee_ty) else {
            self.poly_fallback();
            bail!("calling {} requires concrete callee type", self.show(callee_ty));
        };
        let arg_ids = self.body.list(args).to_vec();
        if arg_ids.len() != params.len() {
            bail!("call takes {} arguments, {} given", params.len(), arg_ids.len());
        }
        for (&arg, &param) in arg_ids.iter().zip(&params) {
            self.call_arg(None, ExprId(arg), (param, None), &mut None)?;
        }
        let tuple = self.cx.types.tuple(&params);
        self.out.res.insert(expr, Res::fn_of(&self.cx.types, method, &[callee_ty, tuple]));
        if let Some(poly) = &mut self.poly {
            poly.obligations.push(PolyObligation::Call { expr, found, tuple });
        }
        Ok(ret)
    }

    fn poly_callable_target(&mut self, callee_ty: Ty) -> Option<(ItemRef, Vec<Ty>, Ty)> {
        let callee_ty = self.resolve(callee_ty);
        let bounds = self.poly.as_ref()?.fn_bounds.clone();
        for (lang, name) in [("fn", "call"), ("fn_mut", "call_mut"), ("fn_once", "call_once")] {
            let Some(of) = self.cx.lang_item(lang) else { continue };
            let Some(&(_, _, sig)) = bounds.iter().find(|&&(t, o, _)| o == of && self.resolve(t) == callee_ty) else { continue };
            let Kind::FnPtr(params, ret) = *self.cx.types.kind(sig) else { continue };
            let method = self.cx.trait_member(of, name)?;
            return Some((method, self.cx.types.substs(params).to_vec(), ret));
        }
        None
    }

    pub(super) fn poly_offset_of(&mut self, expr: ExprId, mut current: Ty, fields: List) -> Result<Ty> {
        let mut steps = Vec::new();
        for &at in self.body.list(fields) {
            for name in self.src.text(at).split('.').map(plain) {
                if matches!(self.cx.types.kind(self.resolve(current)), Kind::Param(_) | Kind::Proj(_)) {
                    self.poly_fallback();
                    bail!("offset_of on open parameter");
                }
                let (index, field) = self.field_at(current, name)?;
                steps.push((current, index));
                current = field;
            }
        }
        if let Some(poly) = &mut self.poly {
            poly.offset_holes.push(ConstHole::OffsetOf { expr, steps: steps.into_boxed_slice() });
        }
        Ok(self.cx.types.prim(Primitive::Usize))
    }
}
