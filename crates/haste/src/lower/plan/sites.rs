use anyhow::Result;
use rustc_hash::FxHashSet;

use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{BodyTables, ClosureBody, Coercion, Res, Shape, SiteKey, Typed, glue_call};
use crate::sema::layout::field_offset;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::reloc::{eval_autoderef_ty, eval_deref_step};
use super::build::build_plan;
use super::ty::{HashMix, hash_num_tag, hash_repr, hash_tuple_offsets, hash_ty_facts};
use super::{BodyFootprintPlan, FieldSite, PatSite, PatSiteKind};

pub(super) fn closure_locals(plan: &BodyFootprintPlan, typed: &Typed, tables: &BodyTables, cb: &ClosureBody) -> Vec<u32> {
    let mut out: Vec<u32> = typed.params.iter().copied().chain(cb.closure.captures.iter().map(|c| c.local)).collect();
    out.extend(plan.pat_sites.iter().filter_map(|ps| tables.syntax.binds.get(&ps.pat).copied()));
    out.sort_unstable();
    out.dedup();
    out
}

pub(super) fn closure_own_calls(cx: &mut Context<'_>, plan: &BodyFootprintPlan, typed: &Typed, tables: &BodyTables, cb: &ClosureBody) -> Result<Vec<(ItemRef, Substs)>> {
    let mut calls: Vec<Res> = Vec::new();
    for &expr in &plan.expr_ids {
        calls.extend([&tables.res, &tables.operators, &tables.try_blocks, &tables.box_frees].into_iter().filter_map(|m| m.get(&expr).copied()));
        calls.extend(tables.fors.get(&expr).into_iter().flat_map(|f| [f.into_iter, f.next]));
        calls.extend(tables.tries.get(&expr).into_iter().flat_map(|t| [t.branch, t.from_residual]));
        calls.extend(tables.awaits.get(&expr).into_iter().flat_map(|a| [a.into_future, a.poll]));
        if let Some(c) = tables.coercions.get(&expr) {
            push_coercion_calls(c, &mut calls);
        }
    }
    for &((expr, _), res) in tables.derefs.as_slice() {
        if plan.contains_expr(expr) {
            calls.push(res);
        }
    }
    for ps in &plan.pat_sites {
        if let Some(&res) = tables.pat_res.get(&ps.pat) {
            calls.push(res);
        }
    }
    let mut out: Vec<(ItemRef, Substs)> = calls.into_iter().filter_map(|r| match r { Res::Fn(i, s) => Some((i, s)), _ => None }).collect();
    if !typed.drops.is_empty() {
        let mut seen = FxHashSet::default();
        let tys = closure_locals(plan, typed, tables, cb).into_iter().map(|l| typed.locals[l as usize].ty).chain(plan.expr_ids.iter().map(|&e| tables.exprs[e.0 as usize]));
        for ty in tys {
            if ty != Ty::NONE && seen.insert(ty) && let Some(res @ Res::Fn(di, ds)) = glue_call(cx, ty)? && typed.drops.contains(&res) {
                out.push((di, ds));
            }
        }
    }
    Ok(out)
}

pub(in crate::lower) fn closure_calls_uncached(cx: &mut Context<'_>, typed: &Typed) -> Option<Vec<(ItemRef, Substs)>> {
    let Shape::Closure(cb) = &typed.shape else { return None };
    let body = cx.body(cb.parent).ok()?;
    let plan = build_plan(&body, Some(cb.closure.body));
    let tables = match &typed.tables {
        crate::sema::infer::LazyTables::Eager(t) => t.as_ref(),
        crate::sema::infer::LazyTables::Poly(p) => p.ensure_tables(&cx.types),
    };
    closure_own_calls(cx, &plan, typed, tables, cb).ok()
}

fn push_coercion_calls(coercion: &Coercion, out: &mut Vec<Res>) {
    match coercion {
        Coercion::Vtable(vt) => out.extend(vt.calls().into_iter().copied()),
        Coercion::Reify(res) => out.push(*res),
        Coercion::Deref(steps, _) => out.extend(steps.iter().filter_map(|(_, r)| *r)),
        Coercion::Len(_) | Coercion::Autos | Coercion::Upcast(_) => {}
    }
}

pub(super) fn hash_plan_sites(cx: &mut Context<'_>, reached: &Reached, mix: &mut HashMix, plan: &BodyFootprintPlan, typed: &Typed) -> Result<()> {
    for site in &plan.field_sites {
        hash_field_site(cx, reached, mix, typed, site)?;
    }
    for ps in &plan.pat_sites {
        hash_pat_site(cx, mix, typed, ps)?;
    }
    for &(call_expr, callee) in &plan.repeat_sites {
        hash_uninit_site(cx, mix, typed, call_expr, super::super::array::is_maybe_uninit_fn(cx, typed, call_expr, callee))?;
    }
    for &clos_expr in &plan.closure_exprs {
        let clos_ty = typed.expr_ty(&cx.types, clos_expr);
        if clos_ty != Ty::NONE
            && let Some(env_ty) = cx.closure_env(clos_ty)
        {
            hash_ty_facts(cx, mix, env_ty)?;
            hash_tuple_offsets(cx, mix, env_ty)?;
        }
    }
    Ok(())
}

fn hash_uninit_site(cx: &mut Context<'_>, mix: &mut HashMix, typed: &Typed, expr: crate::body::ExprId, is_uninit: bool) -> Result<()> {
    mix.bool(is_uninit);
    if is_uninit {
        let ty = typed.expr_ty(&cx.types, expr);
        if ty == Ty::NONE {
            mix.byte(0xff);
        } else if !cx.is_unsized(ty)? {
            hash_repr(cx, mix, ty)?;
        }
    }
    Ok(())
}

fn hash_access_site(cx: &mut Context<'_>, reached: &Reached, mix: &mut HashMix, typed: &Typed, expr: crate::body::ExprId, base: crate::body::ExprId, idx: u32) -> Result<()> {
    let owner = eval_autoderef_ty(cx, reached, typed, expr, base)?;
    if owner == Ty::NONE {
        mix.byte(0xff);
    } else if cx.is_unsized(owner)? {
        mix.byte(1);
    } else {
        mix.byte(0);
        let l = cx.layout(owner)?;
        mix.i32(field_offset(&l, 0, idx));
    }
    let out_ty = typed.expr_ty(&cx.types, expr);
    if out_ty == Ty::NONE {
        mix.byte(0xff);
    } else if !cx.is_unsized(out_ty)? {
        hash_repr(cx, mix, out_ty)?;
    }
    Ok(())
}

fn hash_field_site(cx: &mut Context<'_>, reached: &Reached, mix: &mut HashMix, typed: &Typed, site: &FieldSite) -> Result<()> {
    match site {
        FieldSite::TupleAccess { expr, base, index: idx } => hash_access_site(cx, reached, mix, typed, *expr, *base, *idx)?,
        FieldSite::NamedAccess { expr, base } => match typed.syntax().fields.get(expr) {
            None => mix.byte(0),
            Some(&idx) => {
                mix.byte(1);
                mix.u32(idx);
                hash_access_site(cx, reached, mix, typed, *expr, *base, idx)?;
            }
        },
        FieldSite::StructInit { expr, values } => {
            let ty = typed.expr_ty(&cx.types, *expr);
            if ty == Ty::NONE {
                mix.byte(0xff);
                return Ok(());
            }
            let variant = match typed.site_res(&cx.types, SiteKey::ExprRes(*expr)) {
                Some(Res::Ctor(_, v, _)) => v,
                _ => 0,
            };
            mix.u32(variant);
            let l = cx.layout(ty)?;
            for val in values {
                if let Some(&idx) = typed.syntax().inits.get(val) {
                    mix.u32(idx);
                    mix.i32(field_offset(&l, variant, idx));
                }
            }
        }
        FieldSite::RangeInit { expr } => {
            let ty = typed.expr_ty(&cx.types, *expr);
            if ty == Ty::NONE {
                mix.byte(0xff);
                return Ok(());
            }
            if let Some(indices) = typed.syntax().ranges.get(expr) {
                mix.usize(indices.len());
                let l = cx.layout(ty)?;
                for &idx in indices.iter() {
                    mix.u32(idx);
                    mix.i32(field_offset(&l, 0, idx));
                }
            } else {
                mix.byte(0);
            }
        }
        FieldSite::CallSite { expr, callee, args } => {
            if let Some(Res::Ctor(_, variant, _)) = typed.site_res(&cx.types, SiteKey::ExprRes(*callee)) {
                mix.byte(1);
                mix.u32(variant);
                let ty = typed.expr_ty(&cx.types, *expr);
                if ty == Ty::NONE {
                    mix.byte(0xff);
                } else {
                    let l = cx.layout(ty)?;
                    for idx in 0..args.len() as u32 {
                        mix.i32(field_offset(&l, variant, idx));
                    }
                }
            } else {
                mix.byte(0);
                if typed.is_overloaded_call(*expr) {
                    let arg_tys: Vec<Ty> = args.iter().map(|&a| typed.expr_ty(&cx.types, a)).collect();
                    if arg_tys.iter().any(|&t| t == Ty::NONE) {
                        mix.byte(0xff);
                    } else {
                        let tup = cx.types.tuple(&arg_tys);
                        hash_tuple_offsets(cx, mix, tup)?;
                    }
                }
            }
            for arg in args {
                mix.bool(typed.syntax().const_args.contains(arg));
            }
            hash_uninit_site(cx, mix, typed, *expr, args.is_empty() && super::super::array::is_maybe_uninit_fn(cx, typed, *expr, *callee))?;
        }
        FieldSite::MethodSite { expr, receiver, args } => {
            let mut ty = typed.expr_ty(&cx.types, *receiver);
            if let Some(adj) = typed.syntax().adjusts.get(expr) {
                mix.byte(1);
                mix.u32(adj.derefs);
                mix.byte(match adj.autoref {
                    None => 0,
                    Some(Mut::Not) => 1,
                    Some(Mut::Mut) => 2,
                });
                for step in 0..adj.derefs {
                    if ty == Ty::NONE {
                        mix.byte(0xff);
                        break;
                    }
                    if matches!(cx.types.kind(ty), Kind::Array(..)) {
                        hash_ty_facts(cx, mix, ty)?;
                        mix.u64(cx.layout(ty)?.align.bytes());
                        ty = Ty::NONE;
                        break;
                    }
                    ty = eval_deref_step(cx, reached, typed, *expr, step, ty)?;
                }
                if ty != Ty::NONE {
                    hash_ty_facts(cx, mix, ty)?;
                    if adj.autoref.is_some() && !cx.is_unsized(ty)? {
                        mix.u64(cx.layout(ty)?.align.bytes());
                    }
                }
            } else {
                mix.byte(0);
                if ty != Ty::NONE {
                    hash_ty_facts(cx, mix, ty)?;
                }
            }
            for arg in args {
                mix.bool(typed.syntax().const_args.contains(arg));
            }
        }
        FieldSite::IndexSite { expr, base } => {
            mix.bool(typed.site_fn(SiteKey::Operator(*expr)).is_some());
            let base_ty = eval_autoderef_ty(cx, reached, typed, *expr, *base)?;
            if base_ty == Ty::NONE {
                mix.byte(0xff);
            } else {
                hash_ty_facts(cx, mix, base_ty)?;
            }
            let raw_base = typed.expr_ty(&cx.types, *base);
            if raw_base != Ty::NONE && !cx.is_unsized(raw_base)? {
                mix.u64(cx.layout(raw_base)?.align.bytes());
            }
            let elem_ty = typed.expr_ty(&cx.types, *expr);
            if elem_ty != Ty::NONE && !cx.is_unsized(elem_ty)? {
                mix.u64(cx.layout(elem_ty)?.size.bytes());
                hash_repr(cx, mix, elem_ty)?;
            }
        }
        FieldSite::RefSite { inner, shared } => {
            let inner_ty = typed.expr_ty(&cx.types, *inner);
            if inner_ty == Ty::NONE {
                mix.byte(0xff);
            } else if !cx.is_unsized(inner_ty)? {
                hash_ty_facts(cx, mix, inner_ty)?;
                mix.u64(cx.layout(inner_ty)?.align.bytes());
            }
            if *shared {
                let mono_const = typed.site_fn(SiteKey::ExprRes(*inner)).is_some_and(|(i, s)| s.is_empty() && matches!(cx.item(i).1.kind, ItemKind::Const));
                mix.bool(mono_const);
                mix.bool(typed.syntax().ref_binds.values().any(|&b| b == Mut::Mut));
            }
        }
    }
    Ok(())
}

fn hash_pat_site(cx: &mut Context<'_>, mix: &mut HashMix, typed: &Typed, ps: &PatSite) -> Result<()> {
    let Some(mut ty) = typed.pat_ty(&cx.types, ps.pat) else { return Ok(()) };
    if ty == Ty::NONE {
        mix.byte(0xff);
        return Ok(());
    }
    hash_ty_facts(cx, mix, ty)?;
    let derefs = typed.syntax().pat_derefs.get(&ps.pat).copied().unwrap_or(0);
    mix.u32(derefs);
    for _ in 0..derefs {
        if let Some(inner) = cx.boxed(ty) {
            ty = inner;
        } else if let Kind::Ref(_, inner) | Kind::Ptr(_, inner) = *cx.types.kind(ty) {
            ty = inner;
        }
        if ty == Ty::NONE {
            mix.byte(0xff);
            return Ok(());
        }
        hash_ty_facts(cx, mix, ty)?;
    }
    mix.byte(match typed.syntax().ref_binds.get(&ps.pat) {
        None => 0,
        Some(Mut::Not) => 1,
        Some(Mut::Mut) => 2,
    });
    mix.bool(typed.syntax().or_binds.contains(&ps.pat));
    let res = typed.site_res(&cx.types, SiteKey::PatRes(ps.pat));
    if matches!(res, Some(Res::Const(_))) {
        let mut cur = ty;
        for _ in 0..3 {
            let Kind::Ref(_, inner) = *cx.types.kind(cur) else { break };
            if inner == Ty::NONE {
                mix.byte(0xff);
                break;
            }
            hash_ty_facts(cx, mix, inner)?;
            cur = inner;
        }
    }
    let variant = match res {
        Some(Res::Ctor(_, v, _)) => v,
        _ => 0,
    };
    mix.u32(variant);
    match &ps.kind {
        PatSiteKind::Plain => {}
        PatSiteKind::Range => hash_num_tag(cx, mix, ty),
        PatSiteKind::Tuple(count) => {
            if !cx.is_unsized(ty)? {
                let l = cx.layout(ty)?;
                for idx in 0..*count {
                    mix.i32(field_offset(&l, variant, idx));
                }
            }
        }
        PatSiteKind::Slice => {
            if let Kind::Array(elem, _) | Kind::Slice(elem) = *cx.types.kind(ty) {
                if elem == Ty::NONE {
                    mix.byte(0xff);
                } else {
                    mix.u64(cx.layout(elem)?.size.bytes());
                }
            }
        }
        PatSiteKind::Struct(sub_pats) => {
            if !cx.is_unsized(ty)? {
                let l = cx.layout(ty)?;
                for sub in sub_pats {
                    if let Some(&idx) = typed.syntax().field_pats.get(sub) {
                        mix.u32(idx);
                        mix.i32(field_offset(&l, variant, idx));
                    }
                }
            }
        }
    }
    Ok(())
}
