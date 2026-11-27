mod layout;

use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::sig::GenericKind;
use crate::names::ItemRef;

use super::super::Context;
use super::super::impls::head;
use super::super::infer::{LazyTables, Shape, Typed};
use super::super::intrinsic::Intrinsic;
use super::super::substs::Substs;
use super::super::summary::{PolySummary, poly_summary};
use super::super::ty::{Kind, Ty};
use super::Instance;
use layout::{concrete_ty_is_sized, local_sealed, propagate_callee_sized};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct SealInfo {
    sealed: bool,
    params: u16,
    sized_mask: u64,
}

impl SealInfo {
    const UNSEALED: Self = Self {
        sealed: false,
        params: 0,
        sized_mask: 0,
    };
}

#[derive(Default)]
pub(super) struct ReachFilter {
    pub(super) seen_fixed_poly: HashSet<ItemRef>,
    pub(super) seen_sealed_sysroot: HashSet<ItemRef>,
    sealed_cache: HashMap<ItemRef, SealInfo>,
    sealed_log: Vec<(ItemRef, SealInfo)>,
    seen_sealed_log: Vec<ItemRef>,
    dirty: Vec<(ItemRef, SealInfo)>,
    ty_non_fixed: Vec<u8>,
    substs_non_fixed: Vec<u8>,
}

impl ReachFilter {
    pub(super) fn forget_crates(&mut self, only_crates: Option<&[u32]>) {
        match only_crates {
            Some(crates) => self
                .seen_fixed_poly
                .retain(|item| !crates.contains(&item.krate)),
            None => self.seen_fixed_poly.clear(),
        }
    }

    pub(super) fn seed_clone(&self) -> Self {
        Self {
            seen_fixed_poly: self.seen_fixed_poly.clone(),
            seen_sealed_sysroot: self.seen_sealed_sysroot.clone(),
            sealed_cache: self.sealed_cache.clone(),
            ..Self::default()
        }
    }

    pub(super) fn log_lens(&self) -> (usize, usize) {
        (self.sealed_log.len(), self.seen_sealed_log.len())
    }

    fn record_sealed(&mut self, item: ItemRef, info: SealInfo) {
        if self.sealed_cache.insert(item, info).is_none() {
            self.dirty.push((item, info));
        }
    }

    pub(super) fn mark_seen_sealed(&mut self, item: ItemRef) -> bool {
        let fresh = self.seen_sealed_sysroot.insert(item);
        if fresh {
            self.seen_sealed_log.push(item);
        }
        fresh
    }

    pub(super) fn take_delta(
        &self,
        synced: &mut (usize, usize),
    ) -> (Vec<(ItemRef, SealInfo)>, Vec<ItemRef>) {
        let sealed = if self.sealed_log.len() > synced.0 {
            let slice = self.sealed_log[synced.0..].to_vec();
            synced.0 = self.sealed_log.len();
            slice
        } else {
            Vec::new()
        };
        let seen = if self.seen_sealed_log.len() > synced.1 {
            let slice = self.seen_sealed_log[synced.1..].to_vec();
            synced.1 = self.seen_sealed_log.len();
            slice
        } else {
            Vec::new()
        };
        (sealed, seen)
    }

    pub(super) fn apply_delta(&mut self, sealed: &[(ItemRef, SealInfo)], seen: &[ItemRef]) {
        for &(k, v) in sealed {
            self.sealed_cache.entry(k).or_insert(v);
        }
        if !seen.is_empty() {
            self.seen_sealed_sysroot.extend(seen.iter().copied());
        }
    }

    pub(super) fn drain_to(&mut self, global: &mut Self) {
        for (k, v) in self.dirty.drain(..) {
            if global.sealed_cache.insert(k, v).is_none() {
                global.sealed_log.push((k, v));
            }
        }
    }

    fn seal_info(&mut self, cx: &mut Context<'_>, item: ItemRef) -> SealInfo {
        if let Some(&info) = self.sealed_cache.get(&item) {
            return info;
        }
        let mut stack = Vec::new();
        let Some(_) = self.compute_seal(cx, item, &mut stack) else {
            return SealInfo::UNSEALED;
        };
        self.sealed_cache
            .get(&item)
            .copied()
            .unwrap_or(SealInfo::UNSEALED)
    }

    fn compute_seal(
        &mut self,
        cx: &mut Context<'_>,
        item: ItemRef,
        stack: &mut Vec<ItemRef>,
    ) -> Option<usize> {
        if let Some(&info) = self.sealed_cache.get(&item) {
            return info.sealed.then_some(usize::MAX);
        }
        if let Some(intrinsic) = cx.intrinsic(item) {
            let sealed = !matches!(intrinsic, Intrinsic::Transmute);
            self.record_sealed(
                item,
                SealInfo {
                    sealed,
                    params: 0,
                    sized_mask: 0,
                },
            );
            return sealed.then_some(usize::MAX);
        }
        if !cx.outline.is_fixed(item.krate)
            || matches!(cx.lang(item), Some("drop_glue" | "clone_fn" | "poll"))
        {
            self.record_sealed(item, SealInfo::UNSEALED);
            return None;
        }
        if let Some(pos) = stack.iter().position(|&x| x == item) {
            return Some(pos);
        }
        let Some(summary) = poly_summary(cx, item) else {
            self.record_sealed(item, SealInfo::UNSEALED);
            return None;
        };
        let Some(mut req_sized) = local_sealed(cx, item, &summary) else {
            self.record_sealed(item, SealInfo::UNSEALED);
            return None;
        };
        let my_depth = stack.len();
        stack.push(item);
        let dep = self.seal_callees(cx, item, &summary, stack, &mut req_sized);
        stack.pop();
        let Some(dep) = dep else {
            self.record_sealed(item, SealInfo::UNSEALED);
            return None;
        };
        if dep < my_depth {
            if req_sized != 0 {
                self.record_sealed(item, SealInfo::UNSEALED);
                return None;
            }
            return Some(dep);
        }
        self.record_sealed(
            item,
            SealInfo {
                sealed: true,
                params: summary.param_count,
                sized_mask: req_sized,
            },
        );
        Some(usize::MAX)
    }

    fn seal_callees(
        &mut self,
        cx: &mut Context<'_>,
        caller: ItemRef,
        summary: &PolySummary,
        stack: &mut Vec<ItemRef>,
        req_sized: &mut u64,
    ) -> Option<usize> {
        let mut dep = usize::MAX;
        for &(_, callee, callee_substs) in &summary.direct_calls {
            if cx.intrinsic(callee).is_some() {
                continue;
            }
            let d = self.compute_seal(cx, callee, stack)?;
            dep = dep.min(d);
            if d == usize::MAX
                && let Some(info) = self.sealed_cache.get(&callee).copied()
                && !propagate_callee_sized(cx, caller, callee_substs, info.sized_mask, req_sized)
            {
                return None;
            }
        }
        for &(_, hole_item, hole_substs) in &summary.call_holes {
            dep = dep.min(self.call_hole_seal(
                cx,
                caller,
                hole_item,
                hole_substs,
                stack,
                req_sized,
            )?);
        }
        Some(dep)
    }

    fn call_hole_seal(
        &mut self,
        cx: &mut Context<'_>,
        caller: ItemRef,
        item: ItemRef,
        substs: Substs,
        stack: &mut Vec<ItemRef>,
        req_sized: &mut u64,
    ) -> Option<usize> {
        let tr = cx.owner_trait(item)?;
        if !cx.outline.is_fixed(tr.krate) {
            return None;
        }
        let tg = cx.trait_generics(tr).ok()?;
        let count = 1 + tg
            .1
            .params
            .iter()
            .filter(|g| g.kind != GenericKind::Lifetime)
            .count();
        let slice = cx.types.substs(substs).to_vec();
        if slice.len() < count
            || slice[1..count]
                .iter()
                .any(|&ty| self.ty_has_non_fixed(cx, ty))
        {
            return None;
        }
        if !ok_call_hole_head(cx, self, slice[0]) {
            return None;
        }
        let head_key = head(cx.types.kind(slice[0]))?;
        let holders = fixed_holders(cx, &slice[..count], tr.krate);
        let blocks = cx.trait_impls(tr, Some(head_key), &holders);
        if blocks.is_empty() {
            return None;
        }
        let name = cx.name(item);
        let mut dep = usize::MAX;
        for block in blocks {
            let member = cx.member(block, name).unwrap_or(item);
            let d = self.compute_seal(cx, member, stack)?;
            dep = dep.min(d);
            if d == usize::MAX
                && let Some(info) = self.sealed_cache.get(&member).copied()
                && !propagate_callee_sized(cx, caller, substs, info.sized_mask, req_sized)
            {
                return None;
            }
        }
        Some(dep)
    }

    fn ty_has_non_fixed(&mut self, cx: &Context<'_>, ty: Ty) -> bool {
        ty_has_non_fixed_memo(
            cx.outline,
            &cx.types,
            ty,
            Some((&mut self.ty_non_fixed, &mut self.substs_non_fixed)),
        )
    }

    fn substs_has_non_fixed(&mut self, cx: &Context<'_>, substs: Substs) -> bool {
        substs_has_non_fixed_memo(
            cx.outline,
            &cx.types,
            substs,
            Some((&mut self.ty_non_fixed, &mut self.substs_non_fixed)),
        )
    }
}

fn ok_call_hole_head(cx: &Context<'_>, filter: &mut ReachFilter, ty: Ty) -> bool {
    match *cx.types.kind(ty) {
        Kind::Prim(_)
        | Kind::Never
        | Kind::Ptr(..)
        | Kind::Slice(..)
        | Kind::Array(..)
        | Kind::FnPtr(..)
        | Kind::UnsafeFnPtr(..) => true,
        Kind::Ref(..) | Kind::Tuple(..) => !filter.ty_has_non_fixed(cx, ty),
        Kind::Adt(adt, _) => {
            cx.outline.is_fixed(adt.krate)
                && ((cx.lang(adt) != Some("owned_box") && !cx.has_attr(adt, "fundamental"))
                    || !filter.ty_has_non_fixed(cx, ty))
        }
        _ => false,
    }
}

fn fixed_holders(cx: &Context<'_>, root_tys: &[Ty], tr_krate: u32) -> Vec<u32> {
    let mut holders = vec![tr_krate];
    let mut work: Vec<Ty> = root_tys.to_vec();
    while let Some(ty) = work.pop() {
        let kind = cx.types.kind(ty).clone();
        if let Kind::Adt(def, _) | Kind::Dyn(def, ..) = kind
            && cx.outline.is_fixed(def.krate)
            && !holders.contains(&def.krate)
        {
            holders.push(def.krate);
        }
        work.extend(kind.children(&cx.types));
    }
    holders
}

fn ty_has_non_fixed_memo(
    outline: &crate::outline::Outline,
    types: &super::super::ty::Types,
    ty: Ty,
    mut memo: Option<(&mut Vec<u8>, &mut Vec<u8>)>,
) -> bool {
    if ty == Ty::NONE {
        return true;
    }
    let idx = ty.0 as usize;
    if let Some((ty_memo, _)) = memo.as_ref()
        && let Some(&v) = ty_memo.get(idx)
        && v != 0
    {
        return v == 2;
    }
    let res = match types.kind(ty) {
        Kind::Prim(_) | Kind::Never | Kind::Value(_) => false,
        Kind::Ref(_, inner) | Kind::Ptr(_, inner) | Kind::Slice(inner) | Kind::Pat(inner, ..) => {
            ty_has_non_fixed_memo(
                outline,
                types,
                *inner,
                memo.as_mut().map(|(a, b)| (&mut **a, &mut **b)),
            )
        }
        Kind::Array(elem, count) => {
            ty_has_non_fixed_memo(
                outline,
                types,
                *elem,
                memo.as_mut().map(|(a, b)| (&mut **a, &mut **b)),
            ) || ty_has_non_fixed_memo(
                outline,
                types,
                *count,
                memo.as_mut().map(|(a, b)| (&mut **a, &mut **b)),
            )
        }
        Kind::Tuple(args) => substs_has_non_fixed_memo(
            outline,
            types,
            *args,
            memo.as_mut().map(|(a, b)| (&mut **a, &mut **b)),
        ),
        Kind::FnPtr(args, ret) | Kind::UnsafeFnPtr(args, ret) => {
            substs_has_non_fixed_memo(
                outline,
                types,
                *args,
                memo.as_mut().map(|(a, b)| (&mut **a, &mut **b)),
            ) || ty_has_non_fixed_memo(
                outline,
                types,
                *ret,
                memo.as_mut().map(|(a, b)| (&mut **a, &mut **b)),
            )
        }
        Kind::Adt(def, args) | Kind::FnDef(def, args) | Kind::Closure(def, args, _) => {
            !outline.is_fixed(def.krate)
                || substs_has_non_fixed_memo(
                    outline,
                    types,
                    *args,
                    memo.as_mut().map(|(a, b)| (&mut **a, &mut **b)),
                )
        }
        Kind::Dyn(tr, args, assoc, auto) => {
            !outline.is_fixed(tr.krate)
                || types
                    .items(*assoc)
                    .iter()
                    .chain(types.items(*auto))
                    .any(|d| !outline.is_fixed(d.krate))
                || substs_has_non_fixed_memo(
                    outline,
                    types,
                    *args,
                    memo.as_mut().map(|(a, b)| (&mut **a, &mut **b)),
                )
        }
        Kind::Proj(_) | Kind::Param(_) | Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => true,
    };
    if let Some((ty_memo, _)) = memo {
        if idx >= ty_memo.len() {
            ty_memo.resize(idx + 1, 0);
        }
        ty_memo[idx] = if res { 2 } else { 1 };
    }
    res
}

fn substs_has_non_fixed_memo(
    outline: &crate::outline::Outline,
    types: &super::super::ty::Types,
    substs: Substs,
    mut memo: Option<(&mut Vec<u8>, &mut Vec<u8>)>,
) -> bool {
    if substs.is_empty() {
        return false;
    }
    let idx = substs.0 as usize;
    if let Some((_, sub_memo)) = memo.as_ref()
        && let Some(&v) = sub_memo.get(idx)
        && v != 0
    {
        return v == 2;
    }
    let mut res = false;
    for &ty in types.substs(substs) {
        if ty_has_non_fixed_memo(
            outline,
            types,
            ty,
            memo.as_mut().map(|(a, b)| (&mut **a, &mut **b)),
        ) {
            res = true;
            break;
        }
    }
    if let Some((_, sub_memo)) = memo {
        if idx >= sub_memo.len() {
            sub_memo.resize(idx + 1, 0);
        }
        sub_memo[idx] = if res { 2 } else { 1 };
    }
    res
}

pub(in crate::sema) fn substs_has_non_fixed_raw(
    outline: &crate::outline::Outline,
    types: &super::super::ty::Types,
    substs: Substs,
) -> bool {
    substs_has_non_fixed_memo(outline, types, substs, None)
}

pub(super) fn pre_collect_callees(
    cx: &mut Context<'_>,
    filter: &mut ReachFilter,
    (caller, typed, include_fixed): (Instance, &Result<Arc<Typed>, String>, bool),
    out: &mut Vec<(Instance, bool)>,
) {
    let Ok(typed) = typed else { return };
    let check_only = cx.store.lookups.skip_warm_sysroot_bodies;
    let caller_is_fixed = cx.outline.is_fixed(caller.0.krate);
    let prune_fixed = check_only && caller_is_fixed && !matches!(typed.shape, Shape::Glue(_));
    let mut push = |item: ItemRef, substs: Substs| {
        if check_only && cx.outline.is_fixed(item.krate) {
            return;
        }
        push_filtered_callee(cx, filter, prune_fixed, (item, substs), out)
    };
    match &typed.tables {
        LazyTables::Poly(poly) => push_poly_callees(
            typed,
            poly.calls(),
            poly.dynamic_calls(),
            include_fixed,
            &mut push,
        ),
        LazyTables::CheckOnly(check) => push_poly_callees(
            typed,
            check.calls(),
            check.dynamic_calls(),
            include_fixed,
            &mut push,
        ),
        LazyTables::Eager(_) => {
            for (item, substs) in typed.calls() {
                push(item, substs);
            }
        }
    }
}

fn push_filtered_callee(
    cx: &mut Context<'_>,
    filter: &mut ReachFilter,
    prune_fixed: bool,
    (item, substs): Instance,
    out: &mut Vec<(Instance, bool)>,
) {
    if cx.intrinsic(item).is_some() {
        return;
    }
    let mut sealed = false;
    if prune_fixed && cx.outline.is_fixed(item.krate) {
        if !filter.substs_has_non_fixed(cx, substs) {
            return;
        }
        sealed = can_prune_fixed(cx, filter, (item, substs));
        if sealed && filter.seen_sealed_sysroot.contains(&item) {
            return;
        }
    }
    out.push(((item, substs), sealed));
}

fn can_prune_fixed(
    cx: &mut Context<'_>,
    filter: &mut ReachFilter,
    (item, substs): Instance,
) -> bool {
    let info = filter.seal_info(cx, item);
    if !info.sealed {
        return false;
    }
    let slice = cx.types.substs(substs);
    if slice.len() != usize::from(info.params)
        || slice
            .iter()
            .any(|&ty| matches!(cx.types.kind(ty), Kind::Never | Kind::Dyn(..)))
    {
        return false;
    }
    if info.sized_mask == 0 {
        return true;
    }
    let args = slice.to_vec();
    let mut mask = info.sized_mask;
    while mask != 0 {
        let idx = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        if !concrete_ty_is_sized(cx, args[idx]) {
            return false;
        }
    }
    true
}

fn push_poly_callees(
    typed: &Typed,
    all: impl Iterator<Item = (ItemRef, Substs)>,
    dynamic: impl Iterator<Item = (ItemRef, Substs)>,
    include_fixed: bool,
    push: &mut impl FnMut(ItemRef, Substs),
) {
    if include_fixed {
        for (item, substs) in all {
            push(item, substs);
        }
    } else {
        for (item, substs) in dynamic {
            push(item, substs);
        }
    }
    super::push_tail_calls(typed, &mut |(item, substs)| push(item, substs));
}

pub(super) fn commit_filter_caller(
    cx: &Context<'_>,
    filter: &mut ReachFilter,
    caller: Instance,
    typed: &Result<Arc<Typed>, String>,
) {
    if let Ok(typed) = typed
        && matches!(typed.tables, LazyTables::Poly(_) | LazyTables::CheckOnly(_))
    {
        filter.seen_fixed_poly.insert(caller.0);
        if cx.outline.is_fixed(caller.0.krate) {
            filter.mark_seen_sealed(caller.0);
        }
    }
}

pub(super) fn collect_filtered_callees(
    cx: &mut Context<'_>,
    filter: &mut ReachFilter,
    caller: Instance,
    typed: &Result<Arc<Typed>, String>,
    out: &mut Vec<Instance>,
) {
    let include_fixed = !filter.seen_fixed_poly.contains(&caller.0);
    let mut pre = Vec::new();
    pre_collect_callees(cx, filter, (caller, typed, include_fixed), &mut pre);
    commit_filter_caller(cx, filter, caller, typed);
    for (callee, sealed) in pre {
        if sealed && !filter.mark_seen_sealed(callee.0) {
            continue;
        }
        out.push(callee);
    }
}
