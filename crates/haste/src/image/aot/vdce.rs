use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::lower::{VTABLE_HEADER, VtableEntry, Word};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{Coercion, Res, Shape, Typed, Vtable, is_callable_lang};
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, Types};

const OBJECT_DEPTH: usize = 6;

/// Per-vtable slot liveness mask built from closed-world virtual call and upcast demand.
pub(super) struct VtableMask {
    all_live: bool,
    masks: Vec<Option<Vec<bool>>>,
}

impl VtableMask {
    /// Returns a conservative mask that keeps every vtable slot live.
    pub(super) fn all_live() -> Self {
        Self { all_live: true, masks: Vec::new() }
    }

    /// Iterates closed-world `InlineRTA` and `step` until live function and vtable counts stabilize.
    pub(super) fn converge<T>(
        cx: &mut Context<'_>,
        reached: &Reached,
        vtables: &[Vec<VtableEntry>],
        mut step: impl FnMut(&Self) -> (T, Vec<bool>, (usize, usize)),
    ) -> (T, Self) {
        let mut vmask = Self::all_live();
        let (mut live, mut fns, mut counts) = step(&vmask);
        if vtables.is_empty() {
            return (live, vmask);
        }
        for _ in 0..4 {
            let next_mask = Self::build_from_live(cx, reached, vtables, &fns);
            let (next_live, next_fns, next_counts) = step(&next_mask);
            let converged = next_counts == counts;
            live = next_live;
            fns = next_fns;
            counts = next_counts;
            vmask = next_mask;
            if converged {
                break;
            }
        }
        (live, vmask)
    }

    /// Builds a vtable slot liveness mask from `InlineRTA` seeded at `live_fns`.
    pub(super) fn build_from_live(cx: &mut Context<'_>, reached: &Reached, vtables: &[Vec<VtableEntry>], live_fns: &[bool]) -> Self {
        let mut rta = RtaCollector::new(cx, reached, live_fns);
        rta.run();
        rta.scan_vtable_impls(vtables);
        rta.close_layouts();
        let mut matcher = Matcher {
            cx: rta.cx,
            reached,
            vtables,
            demand: &rta.demand,
            layouts: &rta.layouts,
            cand_list: &rta.cand_list,
            fn_methods: &mut rta.fn_methods,
            memo: HashMap::default(),
        };
        let masks = (0..vtables.len()).map(|id| matcher.vtable_mask(id)).collect();
        Self { all_live: false, masks }
    }

    /// Reports whether `slot` of vtable `id` must remain live in AOT emission.
    pub(super) fn slot_live(&self, id: usize, slot: usize) -> bool {
        if self.all_live || slot < VTABLE_HEADER {
            return true;
        }
        self.masks.get(id).and_then(Option::as_deref).and_then(|bits| bits.get(slot).copied()).unwrap_or(true)
    }
}

#[derive(Default)]
struct Demand {
    vcalls: HashSet<(ItemRef, u32)>,
    upcasts: HashSet<(ItemRef, u32)>,
    wild_methods: HashSet<ItemRef>,
    wild_slots: HashSet<u32>,
    wild_upcasts: HashSet<u32>,
    cands: HashSet<ItemRef>,
    seen_tys: HashSet<Ty>,
}

#[derive(Clone, PartialEq, Eq)]
enum MethodExpect {
    Item(ItemRef),
    Name(String),
}

struct TraitLayout {
    methods: Vec<Option<MethodExpect>>,
    supers: Vec<ItemRef>,
    callable: bool,
}

struct RtaCollector<'a, 'b> {
    cx: &'a mut Context<'b>,
    reached: &'a Reached,
    demand: Demand,
    fn_methods: HashMap<ItemRef, Option<ItemRef>>,
    layouts: HashMap<ItemRef, TraitLayout>,
    cand_list: Vec<ItemRef>,
    visited: Vec<bool>,
    work: Vec<usize>,
    waiting_vcalls: HashMap<(ItemRef, u32), Vec<usize>>,
    waiting_supers: HashMap<(ItemRef, u32), Vec<&'a Vtable>>,
}

impl<'a, 'b> RtaCollector<'a, 'b> {
    fn new(cx: &'a mut Context<'b>, reached: &'a Reached, live_fns: &[bool]) -> Self {
        let mut visited = vec![false; reached.list.len()];
        let mut work = Vec::new();
        for (idx, &is_live) in live_fns.iter().enumerate() {
            if is_live && idx < visited.len() {
                visited[idx] = true;
                work.push(idx);
            }
        }
        Self {
            cx,
            reached,
            demand: Demand::default(),
            fn_methods: HashMap::default(),
            layouts: HashMap::default(),
            cand_list: Vec::new(),
            visited,
            work,
            waiting_vcalls: HashMap::default(),
            waiting_supers: HashMap::default(),
        }
    }

    fn run(&mut self) {
        while let Some(idx) = self.work.pop() {
            let Some(((item, substs), res)) = self.reached.list.get(idx) else { continue };
            for &ty in self.cx.types.substs(*substs) {
                collect_dyn_tys(&self.cx.types, ty, &mut self.demand.seen_tys, &mut self.demand.cands);
            }
            if let Ok(typed) = res {
                self.visit_typed(*item, *substs, typed);
            }
        }
    }

    fn visit_typed(&mut self, item: ItemRef, substs: Substs, typed: &'a Typed) {
        if let Shape::Virtual(slot) = typed.shape {
            self.record_virtual(item, substs, slot);
        }
        if self.cx.foreign(item)
            && self.cx.symbol_name(item).as_deref() == Some("__rust_start_panic")
            && let Some(first_ty) = typed.params.first().map(|&l| typed.locals[l as usize].ty)
            && let Some(of) = dyn_trait_in(&self.cx.types, first_ty, OBJECT_DEPTH)
            && self.demand.vcalls.insert((of, VTABLE_HEADER as u32))
            && let Some(pending) = self.waiting_vcalls.remove(&(of, VTABLE_HEADER as u32))
        {
            pending.into_iter().for_each(|idx| enqueue_idx(&mut self.visited, &mut self.work, idx));
        }
        for local in &typed.locals {
            collect_dyn_tys(&self.cx.types, local.ty, &mut self.demand.seen_tys, &mut self.demand.cands);
        }
        if let Some(ret) = typed.ret {
            collect_dyn_tys(&self.cx.types, ret, &mut self.demand.seen_tys, &mut self.demand.cands);
        }
        let (coercions, exprs) = typed.raw_coercions_and_exprs(&self.cx.types);
        for (&expr, coercion) in coercions.iter() {
            self.record_coercion(coercion, exprs.get(expr.0 as usize).copied());
        }
        for &ty in exprs {
            collect_dyn_tys(&self.cx.types, ty, &mut self.demand.seen_tys, &mut self.demand.cands);
        }
        let (types, reached, visited, work) = (&self.cx.types, self.reached, &mut self.visited, &mut self.work);
        typed.non_vtable_callees(types, |callee, c_substs| enqueue_instance(reached, visited, work, callee, c_substs));
    }

    fn record_virtual(&mut self, item: ItemRef, substs: Substs, slot: u32) {
        let first = self.cx.types.substs(substs).first().copied();
        if let Some(of) = first.and_then(|ty| dyn_trait_in(&self.cx.types, ty, OBJECT_DEPTH)) {
            if self.demand.vcalls.insert((of, slot))
                && let Some(pending) = self.waiting_vcalls.remove(&(of, slot))
            {
                pending.into_iter().for_each(|idx| enqueue_idx(&mut self.visited, &mut self.work, idx));
            }
        } else if let Some(method) = trait_method_of(self.cx, &mut self.fn_methods, item) {
            if self.demand.wild_methods.insert(method) {
                self.wake_wild_method(method);
            }
        } else if self.demand.wild_slots.insert(slot) {
            self.wake_wild_slot(slot);
        }
    }

    fn record_coercion(&mut self, coercion: &'a Coercion, expr_ty: Option<Ty>) {
        match coercion {
            Coercion::Upcast(slot) => self.record_upcast(*slot, expr_ty),
            Coercion::Vtable(vt) => self.visit_vtable(vt),
            _ => {}
        }
    }

    fn record_upcast(&mut self, slot: u32, expr_ty: Option<Ty>) {
        if let Some(of) = expr_ty.and_then(|ty| dyn_trait_in(&self.cx.types, ty, OBJECT_DEPTH)) {
            if self.demand.upcasts.insert((of, slot))
                && let Some(pending) = self.waiting_supers.remove(&(of, slot))
            {
                pending.into_iter().for_each(|vt| self.visit_vtable(vt));
            }
        } else if self.demand.wild_upcasts.insert(slot) {
            self.wake_wild_upcast(slot);
        }
    }

    fn visit_vtable(&mut self, vt: &'a Vtable) {
        collect_vtable_cands(&self.cx.types, vt, &mut self.demand.seen_tys, &mut self.demand.cands);
        if let Some(res) = &vt.drop {
            enqueue_res(self.reached, &mut self.visited, &mut self.work, res);
        }
        let Some(of) = dyn_trait_in(&self.cx.types, vt.object, 1) else {
            self.enqueue_all_vtable(vt);
            return;
        };
        if !self.ensure_layout_for_vtable(of, vt) {
            self.enqueue_all_vtable(vt);
            return;
        }
        let layout = &self.layouts[&of];
        if layout.callable || vt.methods.len() != layout.methods.len() || vt.supers.len() != layout.supers.len() {
            self.enqueue_all_vtable(vt);
            return;
        }
        self.visit_vtable_methods(of, &vt.methods);
        self.visit_vtable_supers(of, vt.methods.len(), &vt.supers);
    }

    fn visit_vtable_methods(&mut self, of: ItemRef, methods: &[Option<Res>]) {
        for (k, method_res) in methods.iter().enumerate() {
            let Some(Res::Fn(item, substs)) = method_res else { continue };
            let Some(&fn_idx) = self.reached.index.get(&(*item, *substs)) else { continue };
            let slot = (VTABLE_HEADER + k) as u32;
            let wild = self.demand.wild_slots.contains(&slot)
                || self.layouts[&of].methods[k].as_ref().is_some_and(|m| method_is_wild(self.cx, &self.demand.wild_methods, m));
            if wild || self.demand.vcalls.contains(&(of, slot)) {
                enqueue_idx(&mut self.visited, &mut self.work, fn_idx);
            } else {
                self.waiting_vcalls.entry((of, slot)).or_default().push(fn_idx);
            }
        }
    }

    fn visit_vtable_supers(&mut self, of: ItemRef, m_len: usize, supers: &'a [Vtable]) {
        for (s, sup) in supers.iter().enumerate() {
            let slot = (VTABLE_HEADER + m_len + s) as u32;
            if self.demand.wild_upcasts.contains(&slot) || self.demand.upcasts.contains(&(of, slot)) {
                self.visit_vtable(sup);
            } else {
                self.waiting_supers.entry((of, slot)).or_default().push(sup);
            }
        }
    }

    fn enqueue_all_vtable(&mut self, vt: &Vtable) {
        for res in vt.calls() {
            enqueue_res(self.reached, &mut self.visited, &mut self.work, res);
        }
    }

    fn wake_wild_method(&mut self, method: ItemRef) {
        let plain = self.cx.plain_name(method);
        let ready: Vec<(ItemRef, u32)> = self
            .waiting_vcalls
            .keys()
            .copied()
            .filter(|&(of, slot)| {
                slot as usize >= VTABLE_HEADER
                    && self
                        .layouts
                        .get(&of)
                        .and_then(|l| l.methods.get(slot as usize - VTABLE_HEADER))
                        .and_then(Option::as_ref)
                        .is_some_and(|m| expect_matches_wild(method, plain, m))
            })
            .collect();
        for key in ready {
            if let Some(pending) = self.waiting_vcalls.remove(&key) {
                pending.into_iter().for_each(|idx| enqueue_idx(&mut self.visited, &mut self.work, idx));
            }
        }
    }

    fn wake_wild_slot(&mut self, slot: u32) {
        let ready: Vec<(ItemRef, u32)> = self.waiting_vcalls.keys().copied().filter(|&(_, s)| s == slot).collect();
        for key in ready {
            if let Some(pending) = self.waiting_vcalls.remove(&key) {
                pending.into_iter().for_each(|idx| enqueue_idx(&mut self.visited, &mut self.work, idx));
            }
        }
    }

    fn wake_wild_upcast(&mut self, slot: u32) {
        let ready: Vec<(ItemRef, u32)> = self.waiting_supers.keys().copied().filter(|&(_, s)| s == slot).collect();
        for key in ready {
            if let Some(pending) = self.waiting_supers.remove(&key) {
                pending.into_iter().for_each(|vt| self.visit_vtable(vt));
            }
        }
    }

    fn scan_vtable_impls(&mut self, vtables: &[Vec<VtableEntry>]) {
        for entries in vtables {
            for entry in entries.iter().skip(VTABLE_HEADER) {
                let VtableEntry::Word(Word::Code(fn_idx)) = *entry else { continue };
                let Some(((item, _), _)) = self.reached.list.get(fn_idx) else { continue };
                if let Some(tr) = trait_method_of(self.cx, &mut self.fn_methods, *item).and_then(|m| self.cx.owner_trait(m)) {
                    self.demand.cands.insert(tr);
                }
            }
        }
    }

    fn ensure_layout(&mut self, tr: ItemRef) -> bool {
        if self.layouts.contains_key(&tr) {
            return true;
        }
        let before = self.cx.quiet_mark();
        let (Ok(methods), Ok(supers)) = (self.cx.vtable_methods(tr), self.cx.vtable_supers(tr)) else {
            return false;
        };
        if self.cx.quiet_mark() != before {
            return false;
        }
        let callable = self.cx.lang(tr).is_some_and(is_callable_lang);
        self.demand.cands.extend(supers.iter().copied());
        let methods = methods.into_iter().map(|m| m.map(MethodExpect::Item)).collect();
        self.layouts.insert(tr, TraitLayout { methods, supers, callable });
        self.cand_list.push(tr);
        true
    }

    fn ensure_layout_for_vtable(&mut self, of: ItemRef, vt: &'a Vtable) -> bool {
        if self.ensure_layout(of) {
            let layout = &self.layouts[&of];
            if layout.callable || (vt.methods.len() == layout.methods.len() && vt.supers.len() == layout.supers.len()) {
                return true;
            }
            self.layouts.remove(&of);
        }
        self.learn_layout_from_vtable(of, vt)
    }

    fn learn_layout_from_vtable(&mut self, of: ItemRef, vt: &'a Vtable) -> bool {
        let callable = self.cx.lang(of).is_some_and(is_callable_lang);
        let mut supers = Vec::with_capacity(vt.supers.len());
        for sup in &vt.supers {
            let Some(sup_tr) = dyn_trait_in(&self.cx.types, sup.object, 1) else { return false };
            if !self.ensure_layout_for_vtable(sup_tr, sup) {
                return false;
            }
            supers.push(sup_tr);
        }
        let mut methods = Vec::with_capacity(vt.methods.len());
        for slot in &vt.methods {
            let exp = match slot {
                None => None,
                Some(Res::Fn(item, _)) => Some(match trait_method_of(self.cx, &mut self.fn_methods, *item) {
                    Some(tm) => MethodExpect::Item(tm),
                    None => MethodExpect::Name(self.cx.plain_name(*item).to_owned()),
                }),
                Some(_) => return false,
            };
            methods.push(exp);
        }
        self.demand.cands.insert(of);
        self.demand.cands.extend(supers.iter().copied());
        if !self.cand_list.contains(&of) {
            self.cand_list.push(of);
        }
        self.layouts.insert(of, TraitLayout { methods, supers, callable });
        true
    }

    fn close_layouts(&mut self) {
        let mut work: Vec<ItemRef> = self.demand.cands.iter().copied().collect();
        while let Some(tr) = work.pop() {
            let had = self.layouts.contains_key(&tr);
            if !self.ensure_layout(tr) || had {
                continue;
            }
            if let Some(layout) = self.layouts.get(&tr) {
                work.extend(layout.supers.iter().copied());
            }
        }
    }
}

struct Matcher<'a, 'b> {
    cx: &'a mut Context<'b>,
    reached: &'a Reached,
    vtables: &'a [Vec<VtableEntry>],
    demand: &'a Demand,
    layouts: &'a HashMap<ItemRef, TraitLayout>,
    cand_list: &'a [ItemRef],
    fn_methods: &'a mut HashMap<ItemRef, Option<ItemRef>>,
    memo: HashMap<(usize, ItemRef), Option<bool>>,
}

impl Matcher<'_, '_> {
    fn vtable_mask(&mut self, id: usize) -> Option<Vec<bool>> {
        let entries = self.vtables.get(id)?;
        if entries.len() < VTABLE_HEADER || has_nonzero_int_tail(entries) {
            return None;
        }
        let matched: Vec<ItemRef> = self.cand_list.iter().copied().filter(|&of| self.matches_trait(id, of)).collect();
        if matched.is_empty() || matched.iter().any(|of| self.layouts[of].callable) {
            return None;
        }
        let mut live = vec![false; entries.len()];
        live[..VTABLE_HEADER].fill(true);
        for of in matched {
            self.mark_trait_slots(of, &mut live);
        }
        Some(live)
    }

    fn mark_trait_slots(&self, of: ItemRef, live: &mut [bool]) {
        let layout = &self.layouts[&of];
        let m_len = layout.methods.len();
        for (k, method) in layout.methods.iter().enumerate() {
            let slot = VTABLE_HEADER + k;
            let s32 = slot as u32;
            if self.demand.vcalls.contains(&(of, s32))
                || self.demand.wild_slots.contains(&s32)
                || method.as_ref().is_some_and(|m| method_is_wild(self.cx, &self.demand.wild_methods, m))
            {
                live[slot] = true;
            }
        }
        for s in 0..layout.supers.len() {
            let slot = VTABLE_HEADER + m_len + s;
            if self.demand.upcasts.contains(&(of, slot as u32)) || self.demand.wild_upcasts.contains(&(slot as u32)) {
                live[slot] = true;
            }
        }
    }

    fn matches_trait(&mut self, id: usize, of: ItemRef) -> bool {
        if let Some(cached) = self.memo.get(&(id, of)) {
            return cached.unwrap_or(false);
        }
        let (Some(entries), Some(layout)) = (self.vtables.get(id), self.layouts.get(&of)) else {
            return false;
        };
        let (m_len, s_len) = (layout.methods.len(), layout.supers.len());
        if entries.len() < VTABLE_HEADER || entries.len() != VTABLE_HEADER + m_len + s_len {
            self.memo.insert((id, of), Some(false));
            return false;
        }
        self.memo.insert((id, of), None);
        let ok = self.methods_match(&entries[VTABLE_HEADER..VTABLE_HEADER + m_len], &layout.methods)
            && self.supers_match(&entries[VTABLE_HEADER + m_len..], &layout.supers);
        self.memo.insert((id, of), Some(ok));
        ok
    }

    fn fn_matches_method(&mut self, fn_idx: usize, expected: &MethodExpect) -> bool {
        let Some(((item, _), _)) = self.reached.list.get(fn_idx) else { return false };
        match expected {
            MethodExpect::Item(exp) => match trait_method_of(self.cx, self.fn_methods, *item) {
                Some(found) => found == *exp,
                None => is_assoc_fn(self.cx, *item) && self.cx.plain_name(*item) == self.cx.plain_name(*exp),
            },
            MethodExpect::Name(name) => is_assoc_fn(self.cx, *item) && self.cx.plain_name(*item) == name.as_str(),
        }
    }

    fn methods_match(&mut self, slots: &[VtableEntry], methods: &[Option<MethodExpect>]) -> bool {
        slots.iter().zip(methods).all(|(entry, expected)| match (*entry, expected.as_ref()) {
            (VtableEntry::Word(Word::Int(0)), None) => true,
            (VtableEntry::Word(Word::Code(fn_idx)), Some(m)) => self.fn_matches_method(fn_idx, m),
            _ => false,
        })
    }

    fn supers_match(&mut self, slots: &[VtableEntry], supers: &[ItemRef]) -> bool {
        slots.iter().zip(supers).all(|(entry, &sup)| matches!(*entry, VtableEntry::Super(sub_id) if self.matches_trait(sub_id.0 as usize, sup)))
    }
}

fn is_assoc_fn(cx: &mut Context<'_>, item: ItemRef) -> bool {
    cx.owner_impl(item).is_some() || cx.owner_trait(item).is_some()
}

fn expect_matches_wild(wild: ItemRef, wild_plain: &str, exp: &MethodExpect) -> bool {
    match exp {
        MethodExpect::Item(m) => *m == wild,
        MethodExpect::Name(name) => name == wild_plain,
    }
}

fn method_is_wild(cx: &Context<'_>, wilds: &HashSet<ItemRef>, exp: &MethodExpect) -> bool {
    match exp {
        MethodExpect::Item(m) => wilds.contains(m),
        MethodExpect::Name(name) => wilds.iter().any(|&w| cx.plain_name(w) == name.as_str()),
    }
}

fn enqueue_idx(visited: &mut [bool], work: &mut Vec<usize>, idx: usize) {
    if let Some(slot) = visited.get_mut(idx)
        && !*slot
    {
        *slot = true;
        work.push(idx);
    }
}

fn enqueue_instance(reached: &Reached, visited: &mut [bool], work: &mut Vec<usize>, item: ItemRef, substs: Substs) {
    if let Some(&idx) = reached.index.get(&(item, substs)) {
        enqueue_idx(visited, work, idx);
    }
}

fn enqueue_res(reached: &Reached, visited: &mut [bool], work: &mut Vec<usize>, res: &Res) {
    if let Res::Fn(item, substs) = *res {
        enqueue_instance(reached, visited, work, item, substs);
    }
}

fn dyn_trait_in(types: &Types, ty: Ty, depth: usize) -> Option<ItemRef> {
    if depth == 0 || ty == Ty::NONE {
        return None;
    }
    match types.kind(ty) {
        Kind::Dyn(of, ..) => Some(*of),
        Kind::Ref(_, inner) | Kind::Ptr(_, inner) => dyn_trait_in(types, *inner, depth - 1),
        Kind::Adt(_, args) => types.substs(*args).iter().find_map(|&arg| dyn_trait_in(types, arg, depth - 1)),
        _ => None,
    }
}

fn collect_dyn_tys(types: &Types, ty: Ty, seen: &mut HashSet<Ty>, cands: &mut HashSet<ItemRef>) {
    if ty == Ty::NONE || !seen.insert(ty) {
        return;
    }
    if let Kind::Dyn(of, ..) = types.kind(ty) {
        cands.insert(*of);
    }
    for child in types.kind(ty).children(types) {
        collect_dyn_tys(types, child, seen, cands);
    }
}

fn collect_vtable_cands(types: &Types, vt: &Vtable, seen: &mut HashSet<Ty>, cands: &mut HashSet<ItemRef>) {
    collect_dyn_tys(types, vt.ty, seen, cands);
    collect_dyn_tys(types, vt.object, seen, cands);
    for sup in &vt.supers {
        collect_vtable_cands(types, sup, seen, cands);
    }
}

fn trait_method_of(cx: &mut Context<'_>, cache: &mut HashMap<ItemRef, Option<ItemRef>>, item: ItemRef) -> Option<ItemRef> {
    if let Some(found) = cache.get(&item) {
        return *found;
    }
    let resolved = if cx.owner_trait(item).is_some() {
        Some(item)
    } else if let Some(imp) = cx.owner_impl(item) {
        cx.impl_trait(imp).and_then(|tr| cx.trait_member(tr, cx.name(item)))
    } else {
        None
    };
    cache.insert(item, resolved);
    resolved
}

fn has_nonzero_int_tail(entries: &[VtableEntry]) -> bool {
    entries.iter().skip(VTABLE_HEADER).any(|entry| matches!(*entry, VtableEntry::Word(Word::Int(val)) if val != 0))
}
