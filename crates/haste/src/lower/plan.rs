mod build;
mod sites;
mod ty;

use std::sync::Arc;

use anyhow::{Result, bail};

use crate::body::{ExprId, PatId};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{ClosureBody, Coercion, Fields, Glue, LazyTables, Res, Shape, Typed, cloned_tuple, field_types, needs_drop};
use crate::sema::layout::field_offset;
use crate::sema::reach::Reached;
use crate::sema::summary::{ConstHole, PolyInst};
use crate::sema::ty::{Kind, Mut, Ty};

use super::stencil::FootprintId;
use super::{Inline, Slots, passed_parts, signature, signature_of};

use build::build_plan;
pub(super) use sites::closure_calls_uncached;
use sites::{closure_locals, closure_own_calls, hash_plan_sites};
use ty::{HashMix, hash_call_edge, hash_drop_tree, hash_dyn_facts, hash_num_tag, hash_parts, hash_repr, hash_sig, hash_tuple_offsets, hash_ty_facts};

#[derive(Clone, Debug)]
enum FieldSite {
    TupleAccess { expr: ExprId, base: ExprId, index: u32 },
    NamedAccess { expr: ExprId, base: ExprId },
    StructInit { expr: ExprId, values: Box<[ExprId]> },
    RangeInit { expr: ExprId },
    CallSite { expr: ExprId, callee: ExprId, args: Box<[ExprId]> },
    MethodSite { expr: ExprId, receiver: ExprId, args: Box<[ExprId]> },
    IndexSite { expr: ExprId, base: ExprId },
    RefSite { inner: ExprId, shared: bool },
}

#[derive(Clone, Debug)]
enum PatSiteKind {
    Plain,
    Range,
    Tuple(u32),
    Slice,
    Struct(Box<[PatId]>),
}

#[derive(Clone, Debug)]
struct PatSite {
    pat: PatId,
    kind: PatSiteKind,
}

/// Reusable per-body structural hole plan cached once per `(source_item, closure_root)`.
#[derive(Debug)]
pub(super) struct BodyFootprintPlan {
    expr_ids: Box<[ExprId]>,
    pat_sites: Box<[PatSite]>,
    field_sites: Box<[FieldSite]>,
    num_sites: Box<[ExprId]>,
    repeat_sites: Box<[(ExprId, ExprId)]>,
    closure_exprs: Box<[ExprId]>,
    has_partial_moves: bool,
}

impl BodyFootprintPlan {
    pub(super) fn contains_expr(&self, expr: ExprId) -> bool {
        self.expr_ids.binary_search(&expr).is_ok()
    }

    pub(super) fn contains_pat(&self, pat: PatId) -> bool {
        self.pat_sites.binary_search_by_key(&pat, |ps| ps.pat).is_ok()
    }
}

/// Evaluates the exact stencil `FootprintId` of reached instance `index` without running AST lowering.
pub fn eval_footprint(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize) -> Result<FootprintId> {
    let mut inline = Inline::default();
    let (fp, _) = eval_footprint_with_cycle(cx, (reached, slots), &mut inline, index)?;
    Ok(fp)
}

pub(super) fn eval_footprint_with_cycle(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize) -> Result<(FootprintId, usize)> {
    let ((item, substs), result) = &reached.list[index];
    let typed = match result {
        Ok(t) => t.as_ref(),
        Err(err) => bail!("{}: {err}", cx.name(*item)),
    };
    let (item, substs) = (*item, *substs);
    if let Some(unique) = try_unique_footprint(cx, slots, item, substs, typed, index) {
        return Ok((unique, usize::MAX));
    }
    let mut mix = HashMix::new(0x51a7_0000);
    mix.item(cx.outline, item);
    let mut cycle_root = usize::MAX;
    hash_instance_header(cx, &mut mix, (slots.low_entries(), slots.allow_inline()), item, typed)?;
    match &typed.shape {
        Shape::Glue(glue) => hash_glue_shape(cx, (reached, slots), inline, &mut mix, index, substs, glue, &mut cycle_root)?,
        Shape::Untuple(target) => hash_untuple_shape(cx, (reached, slots), inline, &mut mix, index, typed, *target, &mut cycle_root)?,
        Shape::Clone(calls) => hash_clone_shape(cx, (reached, slots), inline, &mut mix, index, typed, calls, &mut cycle_root)?,
        Shape::Ctor(variant) => hash_ctor_shape(cx, &mut mix, typed, *variant)?,
        Shape::Virtual(slot) => hash_virtual_shape(cx, &mut mix, typed, *slot)?,
        Shape::Body => eval_body_footprint(cx, (reached, slots), inline, &mut mix, index, (item, None), typed, &mut cycle_root)?,
        Shape::Closure(cb) => {
            hash_closure_header(cx, &mut mix, typed, cb)?;
            eval_body_footprint(cx, (reached, slots), inline, &mut mix, index, (cb.parent, Some(cb.closure.body)), typed, &mut cycle_root)?;
        }
        Shape::Anon(_) => mix.local_index(index),
    }
    Ok((mix.finish(), cycle_root))
}

fn try_unique_footprint(cx: &Context<'_>, slots: &dyn Slots, item: ItemRef, substs: crate::sema::substs::Substs, typed: &Typed, index: usize) -> Option<FootprintId> {
    if cx.foreign(item) {
        let mut mix = HashMix::new(0xff);
        mix.item(cx.outline, item);
        return Some(mix.finish());
    }
    if matches!(typed.shape, Shape::Anon(_)) || typed.has_awaits() {
        let mut mix = HashMix::new(0x51a7_1c02);
        mix.item(cx.outline, item);
        mix.local_index(index);
        return Some(mix.finish());
    }
    if substs.is_empty() && matches!(typed.shape, Shape::Body) {
        let mut mix = HashMix::new(0x51a7_1c01);
        mix.item(cx.outline, item);
        mix.bool(slots.low_entries());
        mix.bool(slots.allow_inline());
        mix.bool(cx.outline.crates[item.krate as usize].env.overflow_checks());
        return Some(mix.finish());
    }
    if substs.is_empty() && !matches!(typed.shape, Shape::Closure(_)) {
        let mut mix = HashMix::new(0x51a7_1c01);
        mix.item(cx.outline, item);
        mix.local_index(index);
        return Some(mix.finish());
    }
    if let Shape::Closure(cb) = &typed.shape {
        if cb.closure.coroutine {
            let mut mix = HashMix::new(0x51a7_1c03);
            mix.local_index(index);
            return Some(mix.finish());
        }
        if let Some(&first) = cx.types.substs(substs).first()
            && let Kind::Closure(_, p_substs, _) = *cx.types.kind(first)
            && p_substs.is_empty()
        {
            let mut mix = HashMix::new(0x51a7_1c04);
            mix.local_index(index);
            return Some(mix.finish());
        }
    }
    None
}

fn hash_instance_header(cx: &mut Context<'_>, mix: &mut HashMix, (low, allow_inline): (bool, bool), item: ItemRef, typed: &Typed) -> Result<()> {
    mix.bool(low);
    mix.bool(allow_inline);
    mix.bool(cx.outline.crates[item.krate as usize].env.overflow_checks());
    let shape_tag = match &typed.shape {
        Shape::Body => 0,
        Shape::Glue(_) => 1,
        Shape::Closure(_) => 2,
        Shape::Virtual(_) => 3,
        Shape::Untuple(_) => 4,
        Shape::Ctor(_) => 5,
        Shape::Clone(_) => 6,
        Shape::Anon(_) => 7,
    };
    mix.byte(shape_tag);
    let sig = signature(cx, typed)?;
    hash_sig(mix, &sig);
    mix.usize(typed.params.len());
    for &p in &typed.params {
        let pty = typed.locals[p as usize].ty;
        hash_repr(cx, mix, pty)?;
        if let Some(parts) = passed_parts(cx, pty)? {
            mix.byte(1);
            hash_parts(mix, parts);
        } else {
            mix.byte(0);
        }
    }
    match typed.ret {
        Some(ret) => {
            mix.byte(1);
            hash_ty_facts(cx, mix, ret)?;
        }
        None => mix.byte(0),
    }
    Ok(())
}

fn hash_glue_shape(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, index: usize, substs: crate::sema::substs::Substs, glue: &Glue, cycle_root: &mut usize) -> Result<()> {
    if glue.resume.is_some() {
        mix.local_index(index);
        return Ok(());
    }
    let Some(&target_ty) = cx.types.substs(substs).first() else { bail!("glue missing target type") };
    hash_ty_facts(cx, mix, target_ty)?;
    if let Some(Res::Fn(di, ds)) = glue.drop {
        mix.byte(1);
        hash_call_edge(cx, (reached, slots), inline, mix, index, di, ds, cycle_root)?;
    } else {
        mix.byte(0);
    }
    if let Some(inner) = cx.boxed(target_ty) {
        mix.byte(0xb1);
        let rm = cx.types.intern(Kind::Ref(Mut::Mut, inner));
        hash_repr(cx, mix, rm)?;
        hash_ty_facts(cx, mix, inner)?;
        if needs_drop(cx, inner)?
            && let Some(g) = cx.lang_item("drop_glue")
        {
            hash_call_edge(cx, (reached, slots), inline, mix, index, g, cx.types.intern_substs(&[inner]), cycle_root)?;
        }
        return Ok(());
    }
    let mut cur = target_ty;
    while let Kind::Closure(..) = *cx.types.kind(cur) {
        mix.byte(0xc1);
        let Some(env) = cx.closure_env(cur) else { bail!("closure without env") };
        cur = env;
    }
    match cx.types.kind(cur).clone() {
        Kind::Array(elem, len_ty) => {
            mix.byte(1);
            mix.u64(cx.types.len(len_ty).unwrap_or(u64::MAX));
            hash_elem_glue(cx, (reached, slots), inline, mix, index, elem, cycle_root)?;
        }
        Kind::Slice(elem) => {
            mix.byte(2);
            hash_elem_glue(cx, (reached, slots), inline, mix, index, elem, cycle_root)?;
        }
        Kind::Dyn(..) => mix.byte(3),
        Kind::Tuple(items) => {
            mix.byte(4);
            let variants: Fields = vec![cx.types.substs(items).to_vec()].into();
            hash_glue_variants(cx, (reached, slots), inline, mix, index, cur, &variants, cycle_root)?;
        }
        Kind::Adt(adt_item, adt_args) => {
            if cx.lang(adt_item) == Some("manually_drop") || cx.item(adt_item).1.kind == ItemKind::Union {
                mix.byte(5);
            } else {
                mix.byte(6);
                let slice = cx.types.substs(adt_args).to_vec();
                let variants = field_types(cx, adt_item, &slice)?;
                hash_glue_variants(cx, (reached, slots), inline, mix, index, cur, &variants, cycle_root)?;
            }
        }
        _ => mix.byte(7),
    }
    Ok(())
}

fn hash_elem_glue(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, index: usize, elem: Ty, cycle_root: &mut usize) -> Result<()> {
    let nd = needs_drop(cx, elem)?;
    mix.bool(nd);
    if nd {
        mix.u64(cx.layout(elem)?.size.bytes());
        hash_repr(cx, mix, elem)?;
        if let Some(g) = cx.lang_item("drop_glue") {
            hash_call_edge(cx, (reached, slots), inline, mix, index, g, cx.types.intern_substs(&[elem]), cycle_root)?;
        }
    }
    Ok(())
}

fn hash_glue_variants(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, index: usize, cur: Ty, variants: &[Vec<Ty>], cycle_root: &mut usize) -> Result<()> {
    let unsized_cur = cx.is_unsized(cur)?;
    mix.bool(unsized_cur);
    hash_ty_facts(cx, mix, cur)?;
    let layout = (!unsized_cur).then(|| cx.layout(cur)).transpose()?;
    mix.usize(variants.len());
    for (v_idx, fields) in variants.iter().enumerate() {
        mix.usize(fields.len());
        for (f_idx, &fty) in fields.iter().enumerate() {
            if let Some(ref l) = layout {
                mix.i32(field_offset(l, v_idx as u32, f_idx as u32));
            }
            let f_unsized = cx.is_unsized(fty)?;
            mix.bool(f_unsized);
            if !f_unsized {
                hash_repr(cx, mix, fty)?;
            }
            let nd = needs_drop(cx, fty)?;
            mix.bool(nd);
            if nd && let Some(g) = cx.lang_item("drop_glue") {
                hash_call_edge(cx, (reached, slots), inline, mix, index, g, cx.types.intern_substs(&[fty]), cycle_root)?;
            }
        }
    }
    Ok(())
}

fn hash_untuple_shape(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, index: usize, typed: &Typed, target: Res, cycle_root: &mut usize) -> Result<()> {
    let (recv_ty, tuple_ty) = (typed.locals[0].ty, typed.locals[1].ty);
    hash_ty_facts(cx, mix, recv_ty)?;
    hash_ty_facts(cx, mix, tuple_ty)?;
    hash_tuple_offsets(cx, mix, tuple_ty)?;
    match target {
        Res::Fn(ti, ts) => {
            mix.byte(1);
            hash_call_edge(cx, (reached, slots), inline, mix, index, ti, ts, cycle_root)?;
        }
        Res::Local(l) => {
            mix.byte(2);
            mix.u32(l);
            let mut inner = recv_ty;
            while let Kind::Ref(_, sub) | Kind::Ptr(_, sub) = *cx.types.kind(inner) {
                inner = sub;
            }
            if let Some(boxed) = cx.boxed(inner) {
                inner = boxed;
            }
            if let Kind::FnPtr(params, ret) = *cx.types.kind(inner) {
                let params = cx.types.substs(params).to_vec();
                let sig = signature_of(cx, Some(ret), params)?;
                hash_sig(mix, &sig);
            }
        }
        _ => mix.byte(3),
    }
    Ok(())
}

fn hash_clone_shape(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, index: usize, typed: &Typed, calls: &[Res], cycle_root: &mut usize) -> Result<()> {
    let ret = typed.ret_ty();
    mix.bool(cloned_tuple(cx, ret).is_some());
    hash_ty_facts(cx, mix, ret)?;
    hash_tuple_offsets(cx, mix, ret)?;
    mix.usize(calls.len());
    for call in calls {
        if let Res::Fn(ci, cs) = *call {
            hash_call_edge(cx, (reached, slots), inline, mix, index, ci, cs, cycle_root)?;
        }
    }
    Ok(())
}

fn hash_ctor_shape(cx: &mut Context<'_>, mix: &mut HashMix, typed: &Typed, variant: u32) -> Result<()> {
    let ret = typed.ret_ty();
    mix.u32(variant);
    hash_ty_facts(cx, mix, ret)?;
    let layout = cx.layout(ret)?;
    for (idx, &param) in typed.params.iter().enumerate() {
        mix.i32(field_offset(&layout, variant, idx as u32));
        hash_repr(cx, mix, typed.locals[param as usize].ty)?;
    }
    Ok(())
}

fn hash_virtual_shape(cx: &mut Context<'_>, mix: &mut HashMix, typed: &Typed, slot: u32) -> Result<()> {
    mix.u32(slot);
    let recv_ty = typed.locals[0].ty;
    hash_ty_facts(cx, mix, recv_ty)?;
    let mut obj = recv_ty;
    if let Some(inner) = cx.boxed(obj) {
        obj = inner;
    } else if let Kind::Ref(_, inner) | Kind::Ptr(_, inner) = *cx.types.kind(obj) {
        obj = inner;
    }
    if let Kind::Dyn(of, ..) = *cx.types.kind(obj) {
        hash_dyn_facts(cx, mix, obj, of)?;
    }
    Ok(())
}

fn hash_closure_header(cx: &mut Context<'_>, mix: &mut HashMix, typed: &Typed, cb: &ClosureBody) -> Result<()> {
    mix.item(cx.outline, cb.parent);
    mix.u32(cb.closure.body.0);
    mix.bool(cb.closure.consumes);
    mix.bool(cb.env.is_some());
    if let Some(env_local) = cb.env {
        let env_ref = typed.locals[env_local as usize].ty;
        hash_ty_facts(cx, mix, env_ref)?;
        if let Kind::Ref(_, pointee) = *cx.types.kind(env_ref) {
            let env_ty = match *cx.types.kind(pointee) {
                Kind::Ref(_, inner) => inner,
                _ => pointee,
            };
            hash_tuple_offsets(cx, mix, env_ty)?;
        }
    }
    mix.usize(cb.closure.captures.len());
    for cap in &cb.closure.captures {
        mix.byte(cap.by as u8);
        hash_ty_facts(cx, mix, typed.locals[cap.local as usize].ty)?;
    }
    Ok(())
}

fn eval_body_footprint(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, index: usize, (source_item, closure_root): (ItemRef, Option<ExprId>), typed: &Typed, cycle_root: &mut usize) -> Result<()> {
    let plan = get_or_build_plan(cx, slots, source_item, closure_root)?;
    match &typed.tables {
        LazyTables::Poly(poly) => eval_poly_body(cx, (reached, slots), inline, mix, index, &plan, typed, poly, cycle_root),
        LazyTables::Eager(tables) => eval_eager_body(cx, (reached, slots), inline, mix, index, &plan, typed, tables, cycle_root),
    }
}

fn eval_poly_body(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, index: usize, plan: &BodyFootprintPlan, typed: &Typed, poly: &PolyInst, cycle_root: &mut usize) -> Result<()> {
    mix.byte(0xa1);
    for &param_ty in &poly.summary.parametric_tys {
        let concrete = poly.sub_ty(&cx.types, param_ty);
        hash_ty_facts(cx, mix, concrete)?;
        hash_num_tag(cx, mix, concrete);
    }
    let has_partial_moves = plan.has_partial_moves || !typed.syntax().unpacks.is_empty();
    if has_partial_moves {
        for local in &typed.locals {
            hash_drop_tree(cx, (reached, slots), inline, mix, index, local.ty, 0, cycle_root)?;
        }
        for &param_ty in &poly.summary.parametric_tys {
            let concrete = poly.sub_ty(&cx.types, param_ty);
            hash_drop_tree(cx, (reached, slots), inline, mix, index, concrete, 0, cycle_root)?;
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
        hash_override_res(cx, (reached, slots), inline, mix, index, ov.box_free, cycle_root)?;
    }
    for hole in &poly.summary.const_holes {
        match *hole {
            ConstHole::Value(vty) => {
                if let Kind::Value(bits) = *cx.types.kind(poly.sub_ty(&cx.types, vty)) {
                    mix.u128(bits);
                }
            }
            ConstHole::Intrinsic { item, substs } => {
                let concrete = poly.sub_substs(&cx.types, substs);
                hash_call_edge(cx, (reached, slots), inline, mix, index, item, concrete, cycle_root)?;
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

fn hash_override_res(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, index: usize, res: Option<Res>, cycle_root: &mut usize) -> Result<()> {
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
            if let Kind::Value(bits) = *cx.types.kind(vty) {
                mix.u128(bits);
            }
        }
    }
    Ok(())
}

fn eval_eager_body(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, mix: &mut HashMix, index: usize, plan: &BodyFootprintPlan, typed: &Typed, tables: &crate::sema::infer::BodyTables, cycle_root: &mut usize) -> Result<()> {
    mix.byte(0xa2);
    let closure = match &typed.shape {
        Shape::Closure(cb) => Some(cb),
        _ => None,
    };
    let is_closure = closure.is_some();
    let has_partial_moves = plan.has_partial_moves
        || if is_closure { typed.params.iter().any(|l| typed.syntax().unpacks.contains_key(l)) } else { !typed.syntax().unpacks.is_empty() };
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
        if (!is_closure || plan.contains_expr(e))
            && let Res::Const(vty) = *r
            && let Kind::Value(bits) = *cx.types.kind(vty)
        {
            mix.u128(bits);
        }
    }
    for &(p, ref r) in tables.pat_res.as_slice() {
        if (!is_closure || plan.contains_pat(p))
            && let Res::Const(vty) = *r
            && let Kind::Value(bits) = *cx.types.kind(vty)
        {
            mix.u128(bits);
        }
    }
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

fn hash_poly_flow_and_coercions(cx: &mut Context<'_>, mix: &mut HashMix, typed: &Typed, poly: &PolyInst) -> Result<()> {
    let tpl = &poly.summary.template;
    for (_, f) in &tpl.fors {
        let (iter_ty, opt_ty) = (poly.sub_ty(&cx.types, f.iter), poly.sub_ty(&cx.types, f.option));
        hash_ty_facts(cx, mix, iter_ty)?;
        hash_ty_facts(cx, mix, opt_ty)?;
        mix.i32(field_offset(&*cx.layout(opt_ty)?, f.some, 0));
    }
    for (_, t) in &tpl.tries {
        let flow_ty = poly.sub_ty(&cx.types, t.flow);
        hash_ty_facts(cx, mix, flow_ty)?;
        let fl = cx.layout(flow_ty)?;
        mix.i32(field_offset(&fl, t.stop, 0));
        mix.i32(field_offset(&fl, t.proceed, 0));
    }
    for (&id, c) in &tpl.coercions {
        hash_one_coercion(cx, mix, typed, id, c, Some(poly))?;
    }
    Ok(())
}

fn hash_eager_flow_and_coercions(cx: &mut Context<'_>, mix: &mut HashMix, plan: &BodyFootprintPlan, typed: &Typed, tables: &crate::sema::infer::BodyTables, is_closure: bool) -> Result<()> {
    for (&id, f) in &tables.fors {
        if !is_closure || plan.contains_expr(id) {
            hash_ty_facts(cx, mix, f.iter)?;
            hash_ty_facts(cx, mix, f.option)?;
            mix.i32(field_offset(&*cx.layout(f.option)?, f.some, 0));
        }
    }
    for (&id, t) in &tables.tries {
        if !is_closure || plan.contains_expr(id) {
            hash_ty_facts(cx, mix, t.flow)?;
            let fl = cx.layout(t.flow)?;
            mix.i32(field_offset(&fl, t.stop, 0));
            mix.i32(field_offset(&fl, t.proceed, 0));
        }
    }
    for (&id, c) in &tables.coercions {
        if !is_closure || plan.contains_expr(id) {
            hash_one_coercion(cx, mix, typed, id, c, None)?;
        }
    }
    Ok(())
}

fn hash_one_coercion(cx: &mut Context<'_>, mix: &mut HashMix, typed: &Typed, id: ExprId, c: &Coercion, poly: Option<&PolyInst>) -> Result<()> {
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
    }
    Ok(())
}

fn get_or_build_plan(cx: &mut Context<'_>, slots: &dyn Slots, source_item: ItemRef, closure_root: Option<ExprId>) -> Result<Arc<BodyFootprintPlan>> {
    let key = (source_item, closure_root);
    if let Some(found) = slots.bodies().plans.get(&key) {
        return Ok(found);
    }
    let body = cx.body(source_item)?;
    let built = Arc::new(build_plan(&body, closure_root));
    slots.bodies().plans.insert(key, Arc::clone(&built));
    Ok(built)
}
