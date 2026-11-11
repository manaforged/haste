use crate::body::ExprId;
use crate::lower;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{Coercion, Res, Shape, Typed, Vtable, field_types, needs_drop};
use crate::sema::intrinsic::{Intrinsic, Simd};
use crate::sema::reach::Reached;
use crate::sema::ty::{Kind, Ty};

use super::digest::{Mix, TraitFacts, hash_num_tag, hash_signature};
use super::footprint::{Edge, HashBins, hash_closure_env_offsets, hash_field_at, hash_type_use};

pub(super) fn hash_control_and_coercions(cx: &mut Context<'_>, reached: &Reached, typed: &Typed, bins: &mut HashBins, edges: &mut Vec<Edge>) {
    let mut offsets: Vec<(u32, u64)> = typed.offsets.iter().map(|(&e, &off)| (e.0, off)).collect();
    offsets.sort_unstable_by_key(|&(e, _)| e);
    for (e, off) in offsets {
        bins.k.u32(e);
        bins.k.u64(off);
        bins.offset.u64(off);
    }
    let mut ranges: Vec<(u32, &Vec<u32>)> = typed.ranges.iter().map(|(&e, v)| (e.0, v)).collect();
    ranges.sort_unstable_by_key(|&(e, _)| e);
    for (e, fields) in ranges {
        if let Some(&ty) = typed.exprs.get(e as usize) && ty != Ty::NONE {
            for &idx in fields {
                hash_field_at(cx, ty, 0, idx, bins);
            }
        }
    }
    hash_loop_and_try_layouts(cx, typed, bins);
    hash_coercion_table(cx, reached, typed, bins, edges);
}

fn hash_loop_and_try_layouts(cx: &mut Context<'_>, typed: &Typed, bins: &mut HashBins) {
    let mut fors: Vec<u32> = typed.fors.keys().map(|e| e.0).collect();
    fors.sort_unstable();
    for key in fors {
        let each = &typed.fors[&ExprId(key)];
        hash_type_use(cx, each.iter, bins);
        hash_type_use(cx, each.option, bins);
        hash_field_at(cx, each.option, each.some, 0, bins);
    }
    let mut tries: Vec<u32> = typed.tries.keys().map(|e| e.0).collect();
    tries.sort_unstable();
    for key in tries {
        let tried = &typed.tries[&ExprId(key)];
        hash_type_use(cx, tried.flow, bins);
        hash_field_at(cx, tried.flow, tried.stop, 0, bins);
        hash_field_at(cx, tried.flow, tried.proceed, 0, bins);
    }
    let mut awaits: Vec<u32> = typed.awaits.keys().map(|e| e.0).collect();
    awaits.sort_unstable();
    for key in awaits {
        let awaited = &typed.awaits[&ExprId(key)];
        hash_type_use(cx, awaited.future, bins);
        hash_type_use(cx, awaited.polled, bins);
        hash_field_at(cx, awaited.polled, 0, 0, bins);
    }
    let mut closures: Vec<u32> = typed.closures.keys().map(|e| e.0).collect();
    closures.sort_unstable();
    for key in closures {
        if let Some(&ty) = typed.exprs.get(key as usize) && ty != Ty::NONE {
            hash_closure_env_offsets(cx, ty, bins);
        }
    }
}

fn hash_coercion_table(cx: &mut Context<'_>, reached: &Reached, typed: &Typed, bins: &mut HashBins, edges: &mut Vec<Edge>) {
    let mut keys: Vec<u32> = typed.coercions.keys().map(|e| e.0).collect();
    keys.sort_unstable();
    for key in keys {
        bins.k.u32(key);
        hash_one_coercion(cx, reached, (key, &typed.coercions[&ExprId(key)]), bins, edges);
    }
}

fn hash_one_coercion(cx: &mut Context<'_>, reached: &Reached, (key, coercion): (u32, &Coercion), bins: &mut HashBins, edges: &mut Vec<Edge>) {
    match coercion {
        Coercion::Len(len) => {
            bins.k.byte(1);
            bins.k.u64(cx.types.len(*len).unwrap_or(0));
        }
        Coercion::Vtable(vtable) => {
            bins.k.byte(2);
            bins.d.byte(2);
            bins.d.u32(key);
            hash_vtable(cx, reached, vtable, &mut bins.d);
            bins.d_slots = bins.d_slots.saturating_add(1);
        }
        Coercion::Deref(steps, to) => {
            bins.k.byte(3);
            bins.k.usize(steps.len());
            bins.k.flag(cx.is_unsized(*to).unwrap_or(false));
            for (step_idx, (step_ty, step_res)) in steps.iter().enumerate() {
                hash_type_use(cx, *step_ty, bins);
                if let Some(Res::Fn(item, substs)) = step_res
                    && let Some(&callee) = reached.index.get(&(*item, substs.clone()))
                {
                    edges.push(Edge { tag: 9, key: (u64::from(key) << 16) | (step_idx as u64), callee, item: *item, foreign: cx.foreign(*item) });
                }
            }
        }
        Coercion::Upcast(slot) => {
            bins.k.byte(4);
            bins.k.u32(*slot);
        }
        Coercion::Reify(Res::Fn(item, substs)) => {
            bins.k.byte(5);
            if let Some(&callee) = reached.index.get(&(*item, substs.clone())) {
                bins.d.byte(5);
                bins.d.u32(key);
                bins.d.usize(callee);
                bins.d_slots = bins.d_slots.saturating_add(1);
            }
        }
        Coercion::Reify(_) | Coercion::Autos => bins.k.byte(6),
    }
}

fn hash_vtable(cx: &mut Context<'_>, reached: &Reached, vtable: &Vtable, mix: &mut Mix) {
    if let Ok(layout) = cx.layout(vtable.ty) {
        mix.u64(layout.size.bytes());
        mix.u64(layout.align.abi.bytes());
    }
    hash_vtable_fn(reached, vtable.drop.as_ref(), mix);
    mix.usize(vtable.methods.len());
    for method in &vtable.methods {
        hash_vtable_fn(reached, method.as_ref(), mix);
    }
    mix.usize(vtable.supers.len());
    for inner in &vtable.supers {
        hash_vtable(cx, reached, inner, mix);
    }
}

fn hash_vtable_fn(reached: &Reached, res: Option<&Res>, mix: &mut Mix) {
    match res {
        Some(Res::Fn(item, substs)) => {
            mix.byte(1);
            mix.usize(reached.index.get(&(*item, substs.clone())).copied().unwrap_or(usize::MAX));
        }
        _ => mix.byte(0),
    }
}

pub(super) fn collect_calls(cx: &mut Context<'_>, reached: &Reached, (owner, typed): (ItemRef, &Typed), (bins, edges): (&mut HashBins, &mut Vec<Edge>), facts: &mut TraitFacts) {
    let mut sites = gather_call_sites(typed);
    sites.sort_unstable_by_key(|&(tag, key, _)| (tag, key));
    for (tag, key, res) in sites {
        record_res_site(cx, reached, (owner, tag, key, res), (bins, edges), facts);
    }
    edges.sort_unstable_by_key(|e| (e.tag, e.key));
}

fn gather_call_sites(typed: &Typed) -> Vec<(u8, u64, &Res)> {
    let mut sites: Vec<(u8, u64, &Res)> = Vec::new();
    sites.extend(typed.res.iter().map(|(&e, r)| (1, u64::from(e.0), r)));
    sites.extend(typed.pat_res.iter().map(|(&p, r)| (2, u64::from(p.0), r)));
    sites.extend(typed.derefs.iter().map(|(&(e, step), r)| (3, (u64::from(e.0) << 16) | u64::from(step), r)));
    sites.extend(typed.operators.iter().map(|(&e, r)| (4, u64::from(e.0), r)));
    sites.extend(typed.try_blocks.iter().map(|(&e, r)| (5, u64::from(e.0), r)));
    sites.extend(typed.box_frees.iter().map(|(&e, r)| (6, u64::from(e.0), r)));
    for (&expr, each) in &typed.fors {
        sites.push((7, u64::from(expr.0) * 2, &each.into_iter));
        sites.push((7, u64::from(expr.0) * 2 + 1, &each.next));
    }
    for (&expr, tried) in &typed.tries {
        sites.push((8, u64::from(expr.0) * 2, &tried.branch));
        sites.push((8, u64::from(expr.0) * 2 + 1, &tried.from_residual));
    }
    for (&expr, awaited) in &typed.awaits {
        sites.push((10, u64::from(expr.0) * 2, &awaited.into_future));
        sites.push((10, u64::from(expr.0) * 2 + 1, &awaited.poll));
    }
    sites.extend(typed.drops.iter().enumerate().map(|(idx, r)| (11, idx as u64, r)));
    append_shape_calls(&typed.shape, &mut sites);
    sites
}

fn append_shape_calls<'a>(shape: &'a Shape, sites: &mut Vec<(u8, u64, &'a Res)>) {
    match shape {
        Shape::Glue(glue) => {
            if let Some(drop) = &glue.drop {
                sites.push((12, 0, drop));
            }
            sites.extend(glue.parts.iter().enumerate().map(|(idx, r)| (12, 1 + idx as u64, r)));
            if let Some(resume) = &glue.resume {
                sites.push((12, 0xffff, resume));
            }
        }
        Shape::Untuple(res) => sites.push((13, 0, res)),
        Shape::Clone(calls) => sites.extend(calls.iter().enumerate().map(|(idx, r)| (14, idx as u64, r))),
        _ => {}
    }
}

fn record_res_site(cx: &mut Context<'_>, reached: &Reached, (owner, tag, key, res): (ItemRef, u8, u64, &Res), (bins, edges): (&mut HashBins, &mut Vec<Edge>), facts: &mut TraitFacts) {
    match res {
        Res::Const(val) => {
            bins.k.byte(tag);
            bins.k.u64(key);
            if let Kind::Value(bits) = *cx.types.kind(*val) {
                bins.k.u128(bits);
            }
        }
        Res::Ctor(_, variant, _) => {
            bins.k.byte(tag);
            bins.k.u64(key);
            bins.k.u32(*variant);
        }
        Res::Fn(item, substs) => record_fn_site(cx, reached, (owner, tag, key), (*item, substs), (bins, edges, facts)),
        Res::Local(_) => {}
    }
}

fn record_fn_site(cx: &mut Context<'_>, reached: &Reached, (owner, tag, key): (ItemRef, u8, u64), (item, substs): (ItemRef, &[Ty]), (bins, edges, facts): (&mut HashBins, &mut Vec<Edge>, &mut TraitFacts)) {
    if let Some(which) = cx.intrinsic(item) {
        hash_intrinsic_site(cx, reached, (owner, tag, key), (which, substs), (bins, edges));
        return;
    }
    let Some(&callee) = reached.index.get(&(item, substs.to_vec())) else { return };
    if lower::is_static(cx, item) {
        bins.k.byte(tag);
        bins.k.u64(key);
        bins.d.byte(tag);
        bins.d.u64(key);
        bins.d.usize(callee);
        bins.d_slots = bins.d_slots.saturating_add(1);
        return;
    }
    if facts.is_unbounded_spec_callee(cx, owner, item) {
        bins.spec_callee.byte(tag);
        bins.spec_callee.u64(key);
        bins.spec_callee.item(item);
    }
    if let Ok(callee_typed) = &reached.list[callee].1
        && let Ok(sig) = lower::signature(cx, callee_typed)
    {
        bins.k.byte(tag);
        bins.k.u64(key);
        hash_signature(&sig, &mut bins.k);
        if facts.trait_of(cx, item).is_some() {
            bins.assoc_abi.byte(tag);
            bins.assoc_abi.u64(key);
            hash_signature(&sig, &mut bins.assoc_abi);
        }
    }
    edges.push(Edge { tag, key, callee, item, foreign: cx.foreign(item) });
}

fn hash_intrinsic_site(cx: &mut Context<'_>, reached: &Reached, (owner, tag, key): (ItemRef, u8, u64), (which, substs): (Intrinsic, &[Ty]), (bins, edges): (&mut HashBins, &mut Vec<Edge>)) {
    bins.k.byte(tag);
    bins.k.u64(key);
    match which {
        Intrinsic::TypeId => {
            bins.k.byte(1);
            if let Some(&ty) = substs.first() {
                bins.d.byte(20);
                bins.d.u64(key);
                bins.d.u32(ty.0);
                bins.d_slots = bins.d_slots.saturating_add(1);
                bins.reflection.byte(20);
                bins.reflection.u64(key);
                bins.reflection.u32(ty.0);
            }
        }
        Intrinsic::TypeName => {
            bins.k.byte(2);
            if let Some(&ty) = substs.first() {
                let text = cx.type_text(ty, &|t| t);
                bins.d.byte(21);
                bins.d.u64(key);
                bins.d.bytes(text.as_bytes());
                bins.d_slots = bins.d_slots.saturating_add(2);
                bins.reflection.byte(21);
                bins.reflection.u64(key);
                bins.reflection.bytes(text.as_bytes());
            }
        }
        Intrinsic::CallerLocation => {
            bins.k.byte(3);
            bins.d.byte(22);
            bins.d.u64(key);
            bins.d.item(owner);
            bins.d_slots = bins.d_slots.saturating_add(1);
        }
        Intrinsic::SizeOf | Intrinsic::AlignOf | Intrinsic::NeedsDrop => hash_layout_intrinsic(cx, which, substs.first().copied(), bins),
        Intrinsic::ConstEvalSelect => hash_const_eval_select(cx, reached, (tag, key), substs, (bins, edges)),
        Intrinsic::Simd(op) => hash_simd_intrinsic(cx, op, substs, bins),
        _ => {
            bins.k.byte(8);
            for &ty in substs {
                hash_type_use(cx, ty, bins);
                hash_num_tag(cx, ty, &mut bins.k);
            }
        }
    }
}

fn hash_layout_intrinsic(cx: &mut Context<'_>, which: Intrinsic, ty: Option<Ty>, bins: &mut HashBins) {
    match (which, ty) {
        (Intrinsic::SizeOf, Some(ty)) => {
            bins.k.byte(4);
            bins.k.u64(cx.layout(ty).map_or(0, |l| l.size.bytes()));
        }
        (Intrinsic::AlignOf, Some(ty)) => {
            bins.k.byte(5);
            bins.k.u64(cx.layout(ty).map_or(1, |l| l.align.abi.bytes()));
        }
        (Intrinsic::NeedsDrop, Some(ty)) => {
            bins.k.byte(6);
            bins.k.flag(needs_drop(cx, ty).unwrap_or(false));
        }
        _ => bins.k.byte(0),
    }
}

fn hash_const_eval_select(cx: &mut Context<'_>, reached: &Reached, (tag, key): (u8, u64), substs: &[Ty], (bins, edges): (&mut HashBins, &mut Vec<Edge>)) {
    bins.k.byte(7);
    if let Some(&tuple) = substs.first() {
        hash_type_use(cx, tuple, bins);
    }
    if let Some(&runtime) = substs.get(2)
        && let Kind::FnDef(fn_item, fn_substs) = cx.types.kind(runtime).clone()
        && let Some(&callee) = reached.index.get(&(fn_item, fn_substs))
    {
        edges.push(Edge { tag, key, callee, item: fn_item, foreign: cx.foreign(fn_item) });
    }
}

fn simd_op_code(op: Simd) -> (u8, u8) {
    match op {
        Simd::Splat => (1, 0),
        Simd::Bitmask => (2, 0),
        Simd::Lanewise(bin) => (3, bin as u8),
        Simd::Cast => (4, 0),
        Simd::Shuffle => (5, 0),
        Simd::Extract => (6, 0),
        Simd::Insert => (7, 0),
        Simd::Reduce { all } => (8 + u8::from(all), 0),
        Simd::Extreme { max } => (10 + u8::from(max), 0),
    }
}

fn hash_simd_intrinsic(cx: &mut Context<'_>, op: Simd, substs: &[Ty], bins: &mut HashBins) {
    let (code, sub) = simd_op_code(op);
    bins.k.byte(9);
    bins.k.byte(code);
    bins.k.byte(sub);
    for &ty in substs {
        hash_simd_type(cx, ty, bins);
    }
}

fn hash_simd_type(cx: &mut Context<'_>, ty: Ty, bins: &mut HashBins) {
    hash_type_use(cx, ty, bins);
    hash_num_tag(cx, ty, &mut bins.k);
    if let Kind::Adt(item, args) = cx.types.kind(ty).clone()
        && let Ok(fields) = field_types(cx, item, &args)
        && let Some(&array) = fields.first().and_then(|v| v.first())
    {
        hash_type_use(cx, array, bins);
        if let Kind::Array(elem, _) = *cx.types.kind(array) {
            hash_num_tag(cx, elem, &mut bins.k);
        }
    }
}
