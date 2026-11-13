use anyhow::Result;

use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{LazyTables, Res, SiteKey, Typed};
use crate::sema::layout::field_offset;
use crate::sema::reach::Reached;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::reloc::{eval_autoderef_ty, eval_deref_step, eval_expr_ty, eval_table_site};
use super::ty::{HashMix, hash_num_tag, hash_repr, hash_tuple_offsets, hash_ty_facts};
use super::{BodyFootprintPlan, FieldSite, PatSite, PatSiteKind, pat_ty_any, site_res_any};

pub(super) fn hash_plan_sites(cx: &mut Context<'_>, reached: &Reached, mix: &mut HashMix, plan: &BodyFootprintPlan, typed: &Typed) -> Result<()> {
    for site in &plan.field_sites {
        hash_field_site(cx, reached, mix, typed, site)?;
    }
    for ps in &plan.pat_sites {
        hash_pat_site(cx, mix, typed, ps)?;
    }
    for &(_, val_expr) in &plan.repeat_sites {
        let val_ty = eval_expr_ty(cx, typed, val_expr);
        let is_uninit = val_ty != Ty::NONE
            && eval_table_site(typed, SiteKey::ExprRes(val_expr)).is_some_and(|(i, _)| {
                let ret_adt = match *cx.types.kind(val_ty) {
                    Kind::Adt(adt, _) => cx.lang(adt) == Some("maybe_uninit"),
                    _ => false,
                };
                (cx.name(i) == "uninit" && ret_adt) || cx.lang(i) == Some("maybe_uninit_uninit")
            });
        mix.bool(is_uninit);
    }
    for &clos_expr in &plan.closure_exprs {
        let clos_ty = eval_expr_ty(cx, typed, clos_expr);
        if clos_ty != Ty::NONE
            && let Some(env_ty) = cx.closure_env(clos_ty)
        {
            hash_ty_facts(cx, mix, env_ty)?;
            hash_tuple_offsets(cx, mix, env_ty)?;
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
    let out_ty = eval_expr_ty(cx, typed, expr);
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
            let ty = eval_expr_ty(cx, typed, *expr);
            if ty == Ty::NONE {
                mix.byte(0xff);
                return Ok(());
            }
            let variant = match site_res_any(cx, typed, SiteKey::ExprRes(*expr)) {
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
            let ty = eval_expr_ty(cx, typed, *expr);
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
            if let Some(Res::Ctor(_, variant, _)) = site_res_any(cx, typed, SiteKey::ExprRes(*callee)) {
                mix.byte(1);
                mix.u32(variant);
                let ty = eval_expr_ty(cx, typed, *expr);
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
                let overloaded = match &typed.tables {
                    LazyTables::Poly(p) => p.is_overloaded_call(*expr),
                    LazyTables::Eager(t) => t.overloaded.contains(expr),
                };
                if overloaded {
                    let arg_tys: Vec<Ty> = args.iter().map(|&a| eval_expr_ty(cx, typed, a)).collect();
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
        }
        FieldSite::MethodSite { expr, receiver, args } => {
            let mut ty = eval_expr_ty(cx, typed, *receiver);
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
            mix.bool(eval_table_site(typed, SiteKey::Operator(*expr)).is_some());
            let base_ty = eval_autoderef_ty(cx, reached, typed, *expr, *base)?;
            if base_ty == Ty::NONE {
                mix.byte(0xff);
            } else {
                hash_ty_facts(cx, mix, base_ty)?;
            }
            let raw_base = eval_expr_ty(cx, typed, *base);
            if raw_base != Ty::NONE && !cx.is_unsized(raw_base)? {
                mix.u64(cx.layout(raw_base)?.align.bytes());
            }
            let elem_ty = eval_expr_ty(cx, typed, *expr);
            if elem_ty != Ty::NONE && !cx.is_unsized(elem_ty)? {
                mix.u64(cx.layout(elem_ty)?.size.bytes());
                hash_repr(cx, mix, elem_ty)?;
            }
        }
        FieldSite::RefSite { inner, shared } => {
            let inner_ty = eval_expr_ty(cx, typed, *inner);
            if inner_ty == Ty::NONE {
                mix.byte(0xff);
            } else if !cx.is_unsized(inner_ty)? {
                hash_ty_facts(cx, mix, inner_ty)?;
                mix.u64(cx.layout(inner_ty)?.align.bytes());
            }
            if *shared {
                let mono_const = eval_table_site(typed, SiteKey::ExprRes(*inner)).is_some_and(|(i, s)| s.is_empty() && matches!(cx.item(i).1.kind, ItemKind::Const));
                mix.bool(mono_const);
                mix.bool(typed.syntax().ref_binds.values().any(|&b| b == Mut::Mut));
            }
        }
    }
    Ok(())
}

fn hash_pat_site(cx: &mut Context<'_>, mix: &mut HashMix, typed: &Typed, ps: &PatSite) -> Result<()> {
    let Some(mut ty) = pat_ty_any(cx, typed, ps.pat) else { return Ok(()) };
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
    let res = site_res_any(cx, typed, SiteKey::PatRes(ps.pat));
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
