use std::sync::{Arc, OnceLock};

use anyhow::Result;

use crate::names::ItemRef;
use crate::outline::Outline;

use super::super::Context;
use super::super::infer::{LazyTables, Local, Res, Typed};
use super::super::reach::Instance;
use super::super::substs::{ProjId, Substs};
use super::super::table::Table;
use super::super::ty::{Kind, Ty, Types};
use super::{PolyInst, PolyObligation, PolySummary, codec};

pub(super) fn frozen_summary(cx: &mut Context<'_>, item: ItemRef) -> Option<Arc<PolySummary>> {
    let tree = cx.outline.crates.get(item.krate as usize)?;
    if !tree.env.release || !tree.env.fixed {
        return None;
    }
    let substrate = cx.store.lookups.substrate.clone()?;
    let key = crate::substrate::sysroot_item_key(cx.outline, item)?;
    let bytes = substrate.summary(key)?;
    let index = cx.store.lookups.sysroot_index.get_or_init(|| Arc::new(codec::SysrootIndex::build(cx.outline, crate::substrate::sysroot_item_key)));
    codec::decode_summary(index, &cx.types, bytes).ok().flatten().map(Arc::new)
}

pub(super) fn frozen_instance(
    cx: &mut Context<'_>,
    instance: Instance,
    summary: &Arc<PolySummary>,
) -> Option<(PolyInst, Vec<Local>, Option<Ty>, Vec<Ty>, Vec<Res>)> {
    if !summary.template.closures.is_empty() {
        return None;
    }
    let tree = cx.outline.crates.get(instance.0.krate as usize)?;
    if !tree.env.release || !tree.env.fixed {
        return None;
    }
    let substrate = cx.store.lookups.substrate.clone()?;
    let item_key = crate::substrate::sysroot_item_key(cx.outline, instance.0)?;
    let substs_key = codec::substs_key(cx.outline, &cx.types, instance.1, crate::substrate::sysroot_item_key)?;
    let index = cx.store.lookups.sysroot_index.get_or_init(|| Arc::new(codec::SysrootIndex::build(cx.outline, crate::substrate::sysroot_item_key)));
    for bytes in substrate.instance_candidates(item_key, substs_key) {
        if let Ok(Some(decoded)) = codec::decode_instance(index, &cx.types, instance.1, bytes) {
            let inst = PolyInst {
                args: instance.1,
                summary: Arc::clone(summary),
                projs: decoded.projs,
                resolved_calls: decoded.resolved_calls,
                builtin_ops: decoded.builtin_ops,
                call_overrides: decoded.call_overrides,
                offset_overrides: decoded.offset_overrides,
                materialized: OnceLock::new(),
            };
            let locals = summary.locals.iter().map(|local| Local { ty: inst.sub_ty(&cx.types, local.ty) }).collect();
            let ret = summary.ret.map(|ty| inst.sub_ty(&cx.types, ty));
            let layouts = summary.layout_tys.iter().map(|&ty| inst.sub_ty(&cx.types, ty)).collect();
            return Some((inst, locals, ret, layouts, decoded.drops));
        }
    }
    None
}

pub(in crate::sema) fn export_sysroot_instances(
    outline: &Outline,
    types: &Types,
    finals: &Table<Instance, Arc<Typed>>,
    builder: &mut crate::substrate::SubstrateBuilder,
) -> Result<()> {
    let mut err = None;
    finals.retain(|&(item, substs), typed| {
        if err.is_some() {
            return true;
        }
        let tree = &outline.crates[item.krate as usize];
        if !tree.env.release || !tree.env.fixed || substs.is_empty() {
            return true;
        }
        if let LazyTables::Poly(inst) = &typed.tables
            && inst.summary.template.closures.is_empty()
            && let Some(item_key) = crate::substrate::sysroot_item_key(outline, item)
            && let Some(substs_key) = codec::substs_key(outline, types, substs, crate::substrate::sysroot_item_key)
            && let Ok(Some(bytes)) = codec::encode_instance(outline, types, substs, inst, &typed.drops, crate::substrate::sysroot_item_key)
            && let Err(e) = builder.add_instance(item_key, substs_key, &bytes)
        {
            err = Some(e);
        }
        true
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn resolve_proj_ty(cx: &Context<'_>, args: &[Ty], projs: &mut Vec<(ProjId, Ty)>, ty: Ty, depth: u8) -> Option<Ty> {
    if ty == Ty::NONE || !cx.types.has_params(ty) {
        return Some(ty);
    }
    if depth > 8 {
        return None;
    }
    match cx.types.kind(ty).clone() {
        Kind::Param(idx) => args.get(idx as usize).copied(),
        Kind::Proj(proj_id) => {
            if let Ok(idx) = projs.binary_search_by_key(&proj_id.0, |&(p, _)| p.0) {
                return Some(projs[idx].1);
            }
            let def = cx.types.proj(proj_id);
            let self_ty = resolve_proj_ty(cx, args, projs, def.self_ty, depth + 1)?;
            let trait_substs = resolve_proj_substs(cx, args, projs, def.args, depth + 1)?;
            let own_substs = resolve_proj_substs(cx, args, projs, def.own, depth + 1)?;
            if cx.types.has_params(self_ty) || cx.types.substs_have_params(trait_substs) || cx.types.substs_have_params(own_substs) {
                return None;
            }
            let val = cx.store.lookups.assoc_types.get(&(def.of, self_ty, trait_substs, def.assoc, own_substs))?;
            let at = projs.partition_point(|&(p, _)| p.0 < proj_id.0);
            projs.insert(at, (proj_id, val));
            Some(val)
        }
        kind => {
            let mut failed = false;
            let pools = cx.types.pools();
            let mapped = cx.types.intern(kind.map_pool(&pools, |child| match resolve_proj_ty(cx, args, projs, child, depth + 1) {
                Some(t) => t,
                None => {
                    failed = true;
                    Ty::NONE
                }
            }));
            (!failed).then_some(mapped)
        }
    }
}

fn resolve_proj_substs(cx: &Context<'_>, args: &[Ty], projs: &mut Vec<(ProjId, Ty)>, substs: Substs, depth: u8) -> Option<Substs> {
    if !cx.types.substs_have_params(substs) {
        return Some(substs);
    }
    let slice = cx.types.substs(substs).to_vec();
    let mut out = Vec::with_capacity(slice.len());
    for ty in slice {
        out.push(resolve_proj_ty(cx, args, projs, ty, depth)?);
    }
    Some(cx.types.intern_substs(&out))
}

pub(super) fn try_cached_instantiate(
    cx: &Context<'_>,
    instance: Instance,
    summary: &Arc<PolySummary>,
) -> Option<(PolyInst, Vec<Local>, Option<Ty>, Vec<Ty>, Vec<Ty>)> {
    if !summary.op_holes.is_empty()
        || !summary.const_holes.is_empty()
        || !summary.template.closures.is_empty()
        || summary.obligations.iter().any(|ob| matches!(ob, PolyObligation::Call { .. }))
    {
        return None;
    }
    let args = cx.types.substs(instance.1).to_vec();
    let mut projs: Vec<(ProjId, Ty)> = Vec::with_capacity(summary.projections.len());
    for &proj_id in &summary.projections {
        resolve_proj_ty(cx, &args, &mut projs, cx.types.intern(Kind::Proj(proj_id)), 0)?;
    }
    for &ty in &summary.parametric_tys {
        resolve_proj_ty(cx, &args, &mut projs, ty, 0)?;
    }
    let mut inst = PolyInst {
        args: instance.1,
        summary: Arc::clone(summary),
        projs: projs.into_boxed_slice(),
        resolved_calls: Box::new([]),
        builtin_ops: Box::new([]),
        call_overrides: Box::new([]),
        offset_overrides: Box::new([]),
        materialized: OnceLock::new(),
    };
    for ob in &summary.obligations {
        let PolyObligation::Prove { of, self_ty, args: ob_args } = *ob else { return None };
        let concrete_self = inst.sub_ty(&cx.types, self_ty);
        let concrete_args = inst.sub_substs(&cx.types, ob_args);
        let Some(Some(_)) = cx.store.lookups.selections.get(&(of, concrete_self, concrete_args)) else {
            return None;
        };
    }
    let mut resolved = Vec::with_capacity(summary.call_holes.len());
    for &(key, item, substs) in &summary.call_holes {
        let mapped = inst.sub_substs(&cx.types, substs);
        let cached = cx.store.lookups.trait_calls.get(&(item, mapped))?;
        if let Res::Fn(ri, rs) = cached {
            resolved.push((key, ri, rs));
        }
    }
    resolved.sort_unstable_by_key(|&(key, ..)| key);
    inst.resolved_calls = resolved.into_boxed_slice();
    let locals = summary.locals.iter().map(|local| Local { ty: inst.sub_ty(&cx.types, local.ty) }).collect();
    let ret = summary.ret.map(|ty| inst.sub_ty(&cx.types, ty));
    let layouts = summary.layout_tys.iter().map(|&ty| inst.sub_ty(&cx.types, ty)).collect();
    let drop_tys = summary.drop_tys.iter().map(|&ty| inst.sub_ty(&cx.types, ty)).collect();
    Some((inst, locals, ret, layouts, drop_tys))
}
