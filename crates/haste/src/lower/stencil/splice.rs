use std::sync::Arc;

use anyhow::Result;
use haste_codegen::Function;
use rustc_hash::FxHashMap;

use crate::sema::Context;
use crate::sema::reach::Reached;

use super::super::reloc::{RelocPlan, RelocStep, SiteOrigin};
use super::super::{Inline, Slots};
use super::{InlinedCallee, LowerMode, RelocVec, SymbolSource, Tally, resolve_inline, simplify_tallied};

pub(super) fn should_splice_devirt(slots: &dyn Slots, func: &Function, mode: LowerMode, reloc_plan: &[RelocStep]) -> bool {
    mode == LowerMode::Stencil
        && slots.allow_inline()
        && !func.is_coroutine()
        && func.has_call()
        && (func.has_vtable_slots() || RelocPlan::steps_have_fn_ptrs(reloc_plan))
}

fn is_free_mono_fn(cx: &Context<'_>, reached: &Reached, idx: usize) -> bool {
    let ((item, substs), res) = &reached.list[idx];
    if item.outer().is_some() || !substs.is_empty() || !(cx.foreign(*item) || matches!(res, Ok(t) if matches!(t.shape, crate::sema::infer::Shape::Body))) {
        return false;
    }
    let Some(file) = cx.outline.crates.get(item.krate as usize).and_then(|k| k.files.get(item.file as usize)) else { return false };
    let mut cur = file.items.items.get(item.item as usize).and_then(|it| it.parent);
    while let Some(p) = cur {
        let entry = &file.items.items[p as usize];
        if !matches!(entry.kind, crate::outline::item::ItemKind::Mod | crate::outline::item::ItemKind::ExternBlock) {
            return false;
        }
        cur = entry.parent;
    }
    true
}

struct RemapWalk<'a, 'c> {
    cx: &'a Context<'c>,
    reached: &'a Reached,
    relocs: &'a [SymbolSource],
    self_slots: &'a [u32],
    top_level: bool,
    concrete: bool,
    offset: usize,
    canon: FxHashMap<SymbolSource, u32>,
    pairs: Vec<(u32, u32)>,
}

impl RemapWalk<'_, '_> {
    fn record(&mut self, idx: u32, src: &SymbolSource) {
        if self.self_slots.contains(&idx) {
            return;
        }
        if let Some(&first) = self.canon.get(src) {
            self.pairs.push((idx, first));
        } else {
            self.canon.insert(src.clone(), idx);
        }
    }

    fn visit(&mut self, steps: &[RelocStep]) {
        for step in steps {
            match step {
                RelocStep::Const(src) => {
                    let idx = self.offset as u32;
                    self.offset += 1;
                    self.record(idx, src);
                }
                RelocStep::SiteInline { plan, .. } => self.visit(&plan.steps),
                RelocStep::SiteSlot(origin) if self.top_level => {
                    let idx = self.offset as u32;
                    self.offset += 1;
                    let can_dedup = self.concrete || (
                        matches!(origin, SiteOrigin::Known(..) | SiteOrigin::Table(crate::sema::infer::SiteKey::ExprRes(_)))
                        && matches!(self.relocs.get(idx as usize), Some(SymbolSource::Slot(ci)) if is_free_mono_fn(self.cx, self.reached, *ci))
                    );
                    if can_dedup && let Some(src) = self.relocs.get(idx as usize) {
                        self.record(idx, src);
                    }
                }
                other => self.offset += other.symbols(),
            }
        }
    }
}

pub(super) fn canonicalize_const_symbols(
    (cx, reached): (&Context<'_>, &Reached),
    func: &mut Function,
    (steps, relocs): (&[RelocStep], &[SymbolSource]),
    (concrete, top_level): (bool, bool),
) {
    func.remap_symbols(|self_slots| {
        let mut walk = RemapWalk { cx, reached, relocs, self_slots, top_level, concrete, offset: 0, canon: FxHashMap::default(), pairs: Vec::new() };
        walk.visit(steps);
        if walk.pairs.is_empty() {
            return None;
        }
        let mut remap: Vec<u32> = (0..walk.offset as u32).collect();
        for (from, to) in walk.pairs {
            remap[from as usize] = to;
        }
        Some(remap)
    });
}

fn resolve_splice_site(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    (relocs, reloc_plan): (&RelocVec, &[RelocStep]),
    callee_sym: u32,
) -> Result<Option<(Arc<InlinedCallee>, SiteOrigin, usize)>> {
    let Some(&SymbolSource::Slot(callee_idx)) = relocs.get(callee_sym as usize) else {
        return Ok(None);
    };
    if cx.track_caller(reached.list[callee_idx].0.0) {
        return Ok(None);
    }
    let origin = match RelocPlan::vtable_slot_at_slice(reloc_plan, callee_sym as usize) {
        Some((expr, slot)) => SiteOrigin::VtableMethod { expr, slot },
        None if RelocPlan::step_at_slice(reloc_plan, callee_sym as usize).is_some_and(RelocStep::is_reify) => SiteOrigin::Symbol(callee_sym),
        None => return Ok(None),
    };
    let Some(limit) = super::super::budget::inline_limit(cx, reached, callee_idx) else {
        return Ok(None);
    };
    let (hit, cycle) = resolve_inline(cx, (reached, slots), inline, callee_idx, (limit, false))?;
    Ok(hit.filter(|c| c.body.can_inline()).map(|inlined| (inlined, origin, cycle)))
}

pub(super) fn splice_devirt_vtable_calls(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    (func, relocs, reloc_plan): (&mut Function, &mut RelocVec, &mut Vec<RelocStep>),
    (mut cycle_root, concrete): (usize, bool),
) -> Result<usize> {
    let mut any = false;
    for _ in 0..8 {
        simplify_tallied(func, Tally::Devirt);
        let mut spliced = false;
        for (block, pos, callee_sym) in func.symbol_call_sites() {
            let Some((inlined, origin, cycle)) = resolve_splice_site(cx, (reached, slots), inline, (relocs, reloc_plan), callee_sym)? else {
                continue;
            };
            cycle_root = cycle_root.min(cycle);
            let sym_base = relocs.len() as u32;
            relocs.extend(inlined.relocs.iter().cloned());
            reloc_plan.push(RelocStep::SiteInline { origin, plan: Arc::clone(&inlined.reloc_plan) });
            func.splice_call(block, pos, &inlined.body, sym_base);
            spliced = true;
            any = true;
            break;
        }
        if !spliced {
            break;
        }
    }
    if any {
        canonicalize_const_symbols((cx, reached), func, (reloc_plan, relocs), (concrete, inline.stack.is_empty()));
        simplify_tallied(func, Tally::Devirt);
    }
    Ok(cycle_root)
}
