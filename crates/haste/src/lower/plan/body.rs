use std::sync::Arc;

use anyhow::{Result, bail};

use crate::body::ExprId;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{BodyTables, Coercion, LazyTables, Res, Shape, Typed};
use crate::sema::layout::field_offset;
use crate::sema::reach::Reached;
use crate::sema::summary::{ConstHole, PolyInst};
use crate::sema::ty::{Kind, Ty};

use super::super::{Inline, Slots};
use super::BodyFootprintPlan;
use super::build::build_plan;
use super::sites::{closure_locals, closure_own_calls, hash_plan_sites};
use super::ty::{
    HashMix, hash_call_edge, hash_drop_tree, hash_dyn_facts, hash_num_tag, hash_ty_facts,
};

pub(super) fn eval_body_footprint(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    index: usize,
    (source_item, closure_root): (ItemRef, Option<ExprId>),
    typed: &Typed,
    cycle_root: &mut usize,
) -> Result<()> {
    let plan = get_or_build_plan(cx, slots, source_item, closure_root)?;
    match &typed.tables {
        LazyTables::Poly(poly) => eval_poly_body(
            cx,
            (reached, slots),
            inline,
            mix,
            index,
            &plan,
            typed,
            poly,
            cycle_root,
        ),
        LazyTables::Eager(tables) => eval_eager_body(
            cx,
            (reached, slots),
            inline,
            mix,
            index,
            &plan,
            typed,
            tables,
            cycle_root,
        ),
        LazyTables::CheckOnly(_) => bail!("cannot lower check-only sysroot instance"),
    }
}

fn hash_const_value(cx: &Context<'_>, mix: &mut HashMix, vty: Ty) {
    if let Kind::Value(bits) = *cx.types.kind(vty) {
        mix.u128(bits);
    }
}

fn hash_const_res(cx: &Context<'_>, mix: &mut HashMix, r: &Res) {
    if let Res::Const(vty) = *r {
        hash_const_value(cx, mix, vty);
    }
}

fn eval_poly_body(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    index: usize,
    plan: &BodyFootprintPlan,
    typed: &Typed,
    poly: &PolyInst,
    cycle_root: &mut usize,
) -> Result<()> {
    mix.byte(0xa1);
    for &param_ty in &poly.summary.parametric_tys {
        let concrete = poly.sub_ty(&cx.types, param_ty);
        hash_ty_facts(cx, mix, concrete)?;
        hash_num_tag(cx, mix, concrete);
    }
    let has_partial_moves = plan.has_partial_moves || !typed.syntax().unpacks.is_empty();
    if has_partial_moves {
        for local in &typed.locals {
            hash_drop_tree(
                cx,
                (reached, slots),
                inline,
                mix,
                index,
                local.ty,
                0,
                cycle_root,
            )?;
        }
        for &param_ty in &poly.summary.parametric_tys {
            let concrete = poly.sub_ty(&cx.types, param_ty);
            hash_drop_tree(
                cx,
                (reached, slots),
                inline,
                mix,
                index,
                concrete,
                0,
                cycle_root,
            )?;
        }
    }
    mix.usize(poly.builtin_ops.len());
    for &op_expr in &poly.builtin_ops {
        mix.u32(op_expr.0);
    }
    mix.usize(poly.offset_overrides.len());
    for &(expr, off) in &poly.offset_overrides {
        mix.u32(expr.0);
        mix.u64(off);
    }
    mix.usize(poly.call_overrides.len());
    for ov in &poly.call_overrides {
        mix.u32(ov.expr.0);
        mix.u32(ov.autoderefs.unwrap_or(u32::MAX));
        mix.bool(ov.overloaded);
        hash_override_res(cx, (reached, slots), inline, mix, index, ov.res, cycle_root)?;
        hash_override_res(
            cx,
            (reached, slots),
            inline,
            mix,
            index,
            ov.box_free,
            cycle_root,
        )?;
    }
    for hole in &poly.summary.const_holes {
        match *hole {
            ConstHole::Value(vty) => {
                let concrete = poly.sub_ty(&cx.types, vty);
                hash_const_value(cx, mix, concrete);
            }
            ConstHole::Intrinsic { item, substs } => {
                let concrete = poly.sub_substs(&cx.types, substs);
                hash_call_edge(
                    cx,
                    (reached, slots),
                    inline,
                    mix,
                    index,
                    item,
                    concrete,
                    cycle_root,
                )?;
            }
            ConstHole::OffsetOf { .. } => {}
        }
    }
    hash_plan_sites(cx, reached, mix, plan, typed)?;
    hash_poly_flow_and_coercions(cx, mix, typed, poly)?;
    for &(ci, cs) in &poly.summary.fixed_calls {
        hash_call_edge(cx, (reached, slots), inline, mix, index, ci, cs, cycle_root)?;
    }
    for &(_, ci, cs) in &poly.resolved_calls {
        hash_call_edge(cx, (reached, slots), inline, mix, index, ci, cs, cycle_root)?;
    }
    for drop_res in &typed.drops {
        if let Res::Fn(di, ds) = *drop_res {
            hash_call_edge(cx, (reached, slots), inline, mix, index, di, ds, cycle_root)?;
        }
    }
    Ok(())
}

fn hash_override_res(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    index: usize,
    res: Option<Res>,
    cycle_root: &mut usize,
) -> Result<()> {
    match res {
        None => mix.byte(0),
        Some(Res::Fn(ci, cs)) => {
            mix.byte(1);
            hash_call_edge(cx, (reached, slots), inline, mix, index, ci, cs, cycle_root)?;
        }
        Some(Res::Ctor(_, v, _)) => {
            mix.byte(2);
            mix.u32(v);
        }
        Some(Res::Local(l)) => {
            mix.byte(3);
            mix.u32(l);
        }
        Some(Res::Const(vty)) => {
            mix.byte(4);
            hash_const_value(cx, mix, vty);
        }
    }
    Ok(())
}

fn hash_eager_tables(
    cx: &Context<'_>,
    mix: &mut HashMix,
    plan: &BodyFootprintPlan,
    tables: &BodyTables,
    is_closure: bool,
) {
    for &(e, s) in tables.autoderefs.as_slice() {
        if !is_closure || plan.contains_expr(e) {
            mix.u32(e.0);
            mix.u32(s);
        }
    }
    for &e in tables.overloaded.as_slice() {
        if !is_closure || plan.contains_expr(e) {
            mix.u32(e.0);
        }
    }
    for &(e, _) in tables.operators.as_slice() {
        if !is_closure || plan.contains_expr(e) {
            mix.u32(e.0);
        }
    }
    for &(e, off) in tables.offsets.as_slice() {
        if !is_closure || plan.contains_expr(e) {
            mix.u32(e.0);
            mix.u64(off);
        }
    }
    for &(e, ref r) in tables.res.as_slice() {
        if !is_closure || plan.contains_expr(e) {
            hash_const_res(cx, mix, r);
        }
    }
    for &(p, ref r) in tables.pat_res.as_slice() {
        if !is_closure || plan.contains_pat(p) {
            hash_const_res(cx, mix, r);
        }
    }
}

fn eval_eager_body(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    mix: &mut HashMix,
    index: usize,
    plan: &BodyFootprintPlan,
    typed: &Typed,
    tables: &BodyTables,
    cycle_root: &mut usize,
) -> Result<()> {
    mix.byte(0xa2);
    let closure = match &typed.shape {
        Shape::Closure(cb) => Some(cb),
        _ => None,
    };
    let is_closure = closure.is_some();
    let has_partial_moves = plan.has_partial_moves
        || if is_closure {
            typed
                .params
                .iter()
                .any(|l| typed.syntax().unpacks.contains_key(l))
        } else {
            !typed.syntax().unpacks.is_empty()
        };
    let own_locals: Vec<u32> = match closure {
        Some(cb) => closure_locals(plan, typed, tables, cb),
        None => (0..typed.locals.len() as u32).collect(),
    };
    for l in own_locals {
        let ty = typed.locals[l as usize].ty;
        hash_ty_facts(cx, mix, ty)?;
        if has_partial_moves {
            hash_drop_tree(cx, (reached, slots), inline, mix, index, ty, 0, cycle_root)?;
        }
    }
    for &expr in &plan.expr_ids {
        let ty = tables.exprs[expr.0 as usize];
        hash_ty_facts(cx, mix, ty)?;
        if has_partial_moves {
            hash_drop_tree(cx, (reached, slots), inline, mix, index, ty, 0, cycle_root)?;
        }
    }
    for &expr in &plan.num_sites {
        hash_num_tag(cx, mix, tables.exprs[expr.0 as usize]);
    }
    hash_eager_tables(cx, mix, plan, tables, is_closure);
    hash_plan_sites(cx, reached, mix, plan, typed)?;
    hash_eager_flow_and_coercions(cx, mix, plan, typed, tables, is_closure)?;
    match closure {
        Some(cb) => {
            for (ci, cs) in closure_own_calls(cx, plan, typed, tables, cb)? {
                hash_call_edge(cx, (reached, slots), inline, mix, index, ci, cs, cycle_root)?;
            }
        }
        None => {
            for (ci, cs) in typed.calls() {
                hash_call_edge(cx, (reached, slots), inline, mix, index, ci, cs, cycle_root)?;
            }
        }
    }
    Ok(())
}

fn hash_for_loop(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    iter_ty: Ty,
    opt_ty: Ty,
    some: u32,
) -> Result<()> {
    hash_ty_facts(cx, mix, iter_ty)?;
    hash_ty_facts(cx, mix, opt_ty)?;
    mix.i32(field_offset(&*cx.layout(opt_ty)?, some, 0));
    Ok(())
}

fn hash_try_expr(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    flow_ty: Ty,
    stop: u32,
    proceed: u32,
) -> Result<()> {
    hash_ty_facts(cx, mix, flow_ty)?;
    let fl = cx.layout(flow_ty)?;
    mix.i32(field_offset(&fl, stop, 0));
    mix.i32(field_offset(&fl, proceed, 0));
    Ok(())
}

fn hash_poly_flow_and_coercions(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    typed: &Typed,
    poly: &PolyInst,
) -> Result<()> {
    let tpl = &poly.summary.template;
    for (_, f) in &tpl.fors {
        let (iter_ty, opt_ty) = (
            poly.sub_ty(&cx.types, f.iter),
            poly.sub_ty(&cx.types, f.option),
        );
        hash_for_loop(cx, mix, iter_ty, opt_ty, f.some)?;
    }
    for (_, t) in &tpl.tries {
        let flow_ty = poly.sub_ty(&cx.types, t.flow);
        hash_try_expr(cx, mix, flow_ty, t.stop, t.proceed)?;
    }
    for (&id, c) in &tpl.coercions {
        hash_one_coercion(cx, mix, typed, id, c, Some(poly))?;
    }
    Ok(())
}

fn hash_eager_flow_and_coercions(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    plan: &BodyFootprintPlan,
    typed: &Typed,
    tables: &BodyTables,
    is_closure: bool,
) -> Result<()> {
    for (&id, f) in &tables.fors {
        if !is_closure || plan.contains_expr(id) {
            hash_for_loop(cx, mix, f.iter, f.option, f.some)?;
        }
    }
    for (&id, t) in &tables.tries {
        if !is_closure || plan.contains_expr(id) {
            hash_try_expr(cx, mix, t.flow, t.stop, t.proceed)?;
        }
    }
    for (&id, c) in &tables.coercions {
        if !is_closure || plan.contains_expr(id) {
            hash_one_coercion(cx, mix, typed, id, c, None)?;
        }
    }
    Ok(())
}

fn hash_one_coercion(
    cx: &mut Context<'_>,
    mix: &mut HashMix,
    typed: &Typed,
    id: ExprId,
    c: &Coercion,
    poly: Option<&PolyInst>,
) -> Result<()> {
    let sub = |cx: &Context<'_>, t: Ty| poly.map_or(t, |p| p.sub_ty(&cx.types, t));
    match c {
        Coercion::Len(len_ty) => {
            mix.byte(1);
            mix.u64(cx.types.len(sub(cx, *len_ty)).unwrap_or(u64::MAX));
        }
        Coercion::Autos => mix.byte(2),
        Coercion::Upcast(slot) => {
            mix.byte(3);
            mix.u32(*slot);
            let expr_ty = typed.expr_ty(&cx.types, id);
            if expr_ty == Ty::NONE {
                mix.byte(0xff);
            } else if let Kind::Ref(_, inner) | Kind::Ptr(_, inner) = *cx.types.kind(expr_ty)
                && let Kind::Dyn(of, ..) = *cx.types.kind(inner)
            {
                hash_dyn_facts(cx, mix, inner, of)?;
            }
        }
        Coercion::Vtable(vt) => {
            mix.byte(4);
            let ty = sub(cx, vt.ty);
            hash_ty_facts(cx, mix, ty)?;
            let obj = sub(cx, vt.object);
            hash_ty_facts(cx, mix, obj)?;
        }
        Coercion::Reify(_) => mix.byte(5),
        Coercion::Deref(steps, to) => {
            mix.byte(6);
            mix.usize(steps.len());
            for (sty, r) in steps {
                let ty = sub(cx, *sty);
                hash_ty_facts(cx, mix, ty)?;
                mix.bool(r.is_some());
            }
            let to_ty = sub(cx, *to);
            hash_ty_facts(cx, mix, to_ty)?;
        }
        Coercion::Field { target, inner } => {
            mix.byte(7);
            let target_ty = sub(cx, *target);
            hash_ty_facts(cx, mix, target_ty)?;
            hash_one_coercion(cx, mix, typed, id, inner, poly)?;
        }
    }
    Ok(())
}

fn get_or_build_plan(
    cx: &mut Context<'_>,
    slots: &dyn Slots,
    source_item: ItemRef,
    closure_root: Option<ExprId>,
) -> Result<Arc<BodyFootprintPlan>> {
    let key = (source_item, closure_root);
    if let Some(found) = slots.bodies().plans.get(&key) {
        return Ok(found);
    }
    let body = cx.body(source_item)?;
    let built = Arc::new(build_plan(&body, closure_root));
    slots.bodies().plans.insert(key, Arc::clone(&built));
    Ok(built)
}
