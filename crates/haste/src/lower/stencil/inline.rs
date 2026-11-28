use std::sync::Arc;

use anyhow::Result;
use haste_codegen::Function;

use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;

use super::super::plan::eval_footprint_with_cycle;
use super::super::reloc::{RelocPlan, eval_relocs};
use super::super::{Inline, Slots};
use super::emit::{run_with_cycle, sym_may_unwind};
use super::inflight::TemplateClaim;
use super::persist;
use super::{
    Bodies, FootprintId, Form, InlineMeta, InlineTemplate, InlinedCallee, LowerMode, Quiet,
    RelocVec, SymbolSource, Tally, TemplateKey, simplify_tallied, tally,
};

const MAX_INLINE_HEIGHT: u8 = 12;

enum Miss {
    Shape,
    Over,
}

type Lowered = (
    std::result::Result<(Arc<Function>, bool), Miss>,
    RelocVec,
    Arc<RelocPlan>,
    usize,
);

enum Found {
    Body(Arc<InlineTemplate>, Option<RelocVec>),
    Shape(bool, u8, u16),
}

type InlineHit = (FootprintId, Found);

fn at_height_cap(inline: &Inline, height: u8) -> bool {
    !inline.stack.is_empty() && height >= MAX_INLINE_HEIGHT
}

fn record_meta_height(
    inline: &mut Inline,
    meta: Option<(ItemRef, FootprintId, bool, u8, u16)>,
) -> Option<(ItemRef, FootprintId, bool)> {
    meta.filter(|meta| !at_height_cap(inline, meta.3))
        .map(|(item, fp, returns, h, sz)| {
            inline.max_callee_height = inline.max_callee_height.max(h);
            inline.max_callee_size = inline.max_callee_size.max(sz);
            (item, fp, returns)
        })
}

fn record_callee_height(
    inline: &mut Inline,
    callee: Option<Arc<InlinedCallee>>,
) -> Option<Arc<InlinedCallee>> {
    let callee = callee.filter(|c| !at_height_cap(inline, c.height));
    if let Some(c) = &callee {
        inline.max_callee_height = inline.max_callee_height.max(c.height);
        inline.max_callee_size = inline.max_callee_size.max(c.size);
    }
    callee
}

fn run_inline_probe(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    index: usize,
    (limit, shape_only, compact): (usize, bool, bool),
) -> Result<(Option<InlineHit>, usize, usize, Quiet)> {
    let (depth, quiet) = (inline.stack.len(), Quiet::mark(cx, slots));
    if inline.aborted {
        return Ok((None, usize::MAX, depth, quiet));
    }
    let Some(_guard) = slots
        .bodies()
        .inflight
        .enter(index, inline.warming && depth == 0)
    else {
        inline.aborted = true;
        return Ok((None, usize::MAX, depth, quiet));
    };
    inline.stack.push(index);
    let saved = std::mem::take(&mut inline.host);
    inline.host.compact = compact;
    inline.host.sole = super::super::budget::stable_sole(cx, reached, index);
    let outcome = ensure_inline_template(
        cx,
        (reached, slots),
        inline,
        index,
        (limit, shape_only, &quiet),
    );
    inline.host = saved;
    inline.pop(depth);
    let (tmpl, cycle) = outcome?;
    Ok((tmpl, cycle, depth, quiet))
}

fn limit_flags(
    cx: &mut Context<'_>,
    reached: &Reached,
    (bodies, inline): (&Bodies, &Inline),
    index: usize,
    limit: usize,
) -> Option<(usize, Option<usize>, bool, bool)> {
    let cap = if !reached.is_sole_called(index) && bodies.is_cyclic(cx, reached, index) {
        let host = inline.stack.last().copied().filter(|&host| host >= index);
        if host.is_some_and(|host| bodies.is_cyclic(cx, reached, host)) {
            return None;
        }
        super::super::budget::cycle_limit(cx, reached, index)?
    } else {
        usize::MAX
    };
    let std_limit =
        super::super::budget::inline_limit(cx, reached, index).map(|standard| standard.min(cap));
    let limit = limit.min(cap);
    Some((
        limit,
        std_limit,
        std_limit.is_some_and(|sl| limit >= sl),
        std_limit.is_some_and(|sl| limit <= sl),
    ))
}

fn cached_inline(
    bodies: &Bodies,
    inline: &Inline,
    (index, limit, std_limit): (usize, usize, Option<usize>),
    (pos_valid, neg_valid, shape_only): (bool, bool, bool),
) -> Option<(Option<Arc<InlinedCallee>>, Option<InlineMeta>, usize)> {
    let whole = |callee: Option<Arc<InlinedCallee>>, cycle: usize| {
        let meta = callee.as_deref().map(InlinedCallee::meta);
        (callee, meta, cycle)
    };
    let ok_cal = |callee: &Option<Arc<InlinedCallee>>| {
        if callee.is_some() {
            pos_valid
        } else {
            neg_valid
        }
    };
    if let Some(callee) = bodies.inlined.get(&(index, limit)) {
        return Some(whole(callee, usize::MAX));
    }
    if let Some(sl) = std_limit.filter(|&sl| sl != limit)
        && let Some(callee) = bodies.inlined.get(&(index, sl)).filter(ok_cal)
    {
        return Some(whole(callee, usize::MAX));
    }
    if let Some(meta) = bodies
        .inline_meta
        .get(&(index, limit))
        .filter(|m| shape_only || m.is_none())
    {
        return Some((None, meta, usize::MAX));
    }
    if neg_valid
        && let Some(sl) = std_limit.filter(|&sl| sl != limit)
        && let Some(None) = bodies.inline_meta.get(&(index, sl))
    {
        return Some((None, None, usize::MAX));
    }
    if let Some(pos) = inline.stack.iter().position(|&at| at == index) {
        return Some((None, None, pos));
    }
    if let Some((callee, cycle)) = inline.local.get(&(index, limit)) {
        return Some(whole(callee.clone(), *cycle));
    }
    if let Some(sl) = std_limit.filter(|&sl| sl != limit)
        && let Some((callee, cycle)) = inline
            .local
            .get(&(index, sl))
            .filter(|(callee, _)| ok_cal(callee))
    {
        return Some(whole(callee.clone(), *cycle));
    }
    inline
        .local_meta
        .get(&(index, limit))
        .filter(|(m, _)| shape_only || m.is_none())
        .map(|&(meta, cycle)| (None, meta, cycle))
}

fn build_inlined(
    item: ItemRef,
    footprint: FootprintId,
    t: &InlineTemplate,
    relocs: RelocVec,
) -> Arc<InlinedCallee> {
    Arc::new(InlinedCallee {
        item,
        footprint,
        relocs,
        reloc_plan: Arc::clone(&t.reloc_plan),
        body: Arc::clone(&t.body),
        returns: t.returns,
        height: t.height,
        size: t.size,
        form: t.form,
    })
}

pub(in crate::lower) fn resolve_inline_meta(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    index: usize,
    limit: usize,
) -> Result<(Option<(ItemRef, FootprintId, bool)>, usize)> {
    let bodies = slots.bodies();
    let Some((limit, std_limit, pos_valid, neg_valid)) =
        limit_flags(cx, reached, (bodies, &*inline), index, limit)
    else {
        return Ok((None, usize::MAX));
    };
    if let Some((_, meta, cycle)) = cached_inline(
        bodies,
        inline,
        (index, limit, std_limit),
        (pos_valid, neg_valid, true),
    ) {
        return Ok((record_meta_height(inline, meta), cycle));
    }
    let item = reached.list[index].0.0;
    let (tmpl, cycle, depth, quiet) =
        run_inline_probe(cx, (reached, slots), inline, index, (limit, true, false))?;
    if inline.aborted {
        return Ok((None, usize::MAX));
    }
    let (meta, inlined) = match tmpl {
        Some((footprint, Found::Body(t, relocs))) => (
            Some((item, footprint, t.returns, t.height, t.size)),
            relocs.map(|r| Some(build_inlined(item, footprint, &t, r))),
        ),
        Some((footprint, Found::Shape(returns, height, size))) => {
            (Some((item, footprint, returns, height, size)), None)
        }
        None => (None, Some(None)),
    };
    if cycle > depth && quiet.holds(cx, slots) {
        if let Some(Some(cal)) = inlined {
            let stored = bodies.cache_inlined((index, limit), Some(cal));
            return Ok((
                record_meta_height(inline, stored.as_deref().map(InlinedCallee::meta)),
                cycle,
            ));
        }
        let stored = bodies
            .inline_meta
            .get_or_insert_with((index, limit), || meta);
        return Ok((record_meta_height(inline, stored), cycle));
    }
    let local_cyc = if cycle < depth { cycle } else { usize::MAX };
    if let Some(cal) = inlined {
        inline.local.insert((index, limit), (cal, local_cyc));
    } else {
        inline.local_meta.insert((index, limit), (meta, local_cyc));
    }
    Ok((record_meta_height(inline, meta), cycle))
}

fn finish_inline(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    (index, callee, cycle): (usize, Option<Arc<InlinedCallee>>, usize),
) -> Result<(Option<Arc<InlinedCallee>>, usize)> {
    let callee =
        super::super::budget::maybe_partial(cx, (reached, slots), inline, (index, callee))?;
    Ok((record_callee_height(inline, callee), cycle))
}

pub(in crate::lower) fn resolve_inline(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    index: usize,
    (limit, compact): (usize, bool),
) -> Result<(Option<Arc<InlinedCallee>>, usize)> {
    let bodies = slots.bodies();
    let Some((limit, std_limit, pos_valid, neg_valid)) =
        limit_flags(cx, reached, (bodies, &*inline), index, limit)
    else {
        return Ok((None, usize::MAX));
    };
    if !compact
        && let Some((callee, _, cycle)) = cached_inline(
            bodies,
            inline,
            (index, limit, std_limit),
            (pos_valid, neg_valid, false),
        )
    {
        return finish_inline(cx, (reached, slots), inline, (index, callee, cycle));
    }
    let item = reached.list[index].0.0;
    let (tmpl, cycle, depth, quiet) =
        run_inline_probe(cx, (reached, slots), inline, index, (limit, false, compact))?;
    if inline.aborted {
        return Ok((None, usize::MAX));
    }
    let result = match tmpl {
        Some((footprint, Found::Body(t, Some(relocs)))) => {
            Some(build_inlined(item, footprint, &t, relocs))
        }
        Some((footprint, Found::Body(t, None))) => {
            let evaluated =
                quiet.lowered(slots, eval_relocs(cx, reached, slots, &t.reloc_plan, index))?;
            evaluated.map(|relocs| build_inlined(item, footprint, &t, relocs))
        }
        Some((_, Found::Shape(..))) => {
            unreachable!("a body probe never answers from recorded metadata")
        }
        None => None,
    };
    if !compact {
        if cycle > depth && quiet.holds(cx, slots) {
            if let Some(cal) = result {
                let stored = bodies.cache_inlined((index, limit), Some(cal));
                return finish_inline(cx, (reached, slots), inline, (index, stored, cycle));
            }
            bodies
                .inline_meta
                .get_or_insert_with((index, limit), || None);
            return finish_inline(cx, (reached, slots), inline, (index, None, cycle));
        }
        inline.local.insert(
            (index, limit),
            (
                result.clone(),
                if cycle < depth { cycle } else { usize::MAX },
            ),
        );
    }
    finish_inline(cx, (reached, slots), inline, (index, result, cycle))
}

fn ensure_inline_template(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    index: usize,
    probe: (usize, bool, &Quiet),
) -> Result<(Option<InlineHit>, usize)> {
    let saved_h = std::mem::replace(&mut inline.max_callee_height, 0);
    let saved_sz = std::mem::replace(&mut inline.max_callee_size, 0);
    let res = ensure_inline_template_inner(cx, (reached, slots), inline, index, probe);
    inline.max_callee_height = saved_h;
    inline.max_callee_size = saved_sz;
    res
}

fn try_cached_template<'a>(
    cx: &Context<'_>,
    (reached, slots): (&Reached, &'a dyn Slots),
    key: &TemplateKey,
    (shape_only, open_ok, quiet): (bool, bool, &Quiet),
) -> Result<Option<(Option<Found>, bool)>, Option<persist::Disk<'a>>> {
    if slots.bodies().rejected.get(key).is_some() {
        tally(Tally::Hit);
        return Ok(Some((None, false)));
    }
    if let Some(template) = slots.bodies().templates.get(key) {
        tally(Tally::Hit);
        return Ok(Some((Some(Found::Body(template, None)), false)));
    }
    tally(Tally::Miss);
    let disk = if key.compact {
        None
    } else {
        slots.bodies().disk(cx, key)
    };
    if shape_only
        && open_ok
        && let Some(shape) = disk.as_ref().and_then(persist::Disk::hit)
    {
        return Ok(Some((
            shape.map(|(returns, height, size)| Found::Shape(returns, height, size)),
            false,
        )));
    }
    if !shape_only
        && open_ok
        && let Some(d) = &disk
        && let Some(mut found) = d.template(cx, (reached, slots))
    {
        if quiet.holds(cx, slots) {
            match found {
                Some(t) => {
                    found = Some(slots.bodies().cache_template(*key, t).0);
                }
                None => {
                    slots.bodies().rejected.get_or_insert_with(*key, || ());
                }
            }
        }
        return Ok(Some((found.map(|t| Found::Body(t, None)), true)));
    }
    Err(disk)
}

fn settle_built_template(
    cx: &Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    key: TemplateKey,
    mut template: Option<Arc<InlineTemplate>>,
    (disk, calm, servable, in_cycle): (Option<persist::Disk<'_>>, bool, bool, bool),
) -> Option<Arc<InlineTemplate>> {
    if in_cycle {
        tally(Tally::OpenCycle);
    } else if calm {
        match template {
            Some(t) => {
                let (stored, inserted) = slots.bodies().cache_template(key, t);
                if !inserted {
                    tally(Tally::Raced);
                }
                template = Some(stored);
            }
            None => {
                slots.bodies().rejected.get_or_insert_with(key, || ());
            }
        }
    } else {
        tally(Tally::Deferred);
    }
    if let Some(disk) = disk {
        disk.settle(
            cx,
            (reached, slots),
            template.as_deref(),
            (calm && !in_cycle, servable && calm),
        );
    }
    template
}

fn pack_template(
    fit: std::result::Result<(Arc<Function>, bool), Miss>,
    reloc_plan: Arc<RelocPlan>,
    form: Form,
    (base_h, base_size): (u8, usize),
) -> Option<Arc<InlineTemplate>> {
    fit.ok().and_then(|(body, returns)| {
        let body_size = body.size();
        let grew = (base_size == 0 && body_size > 32)
            || (base_size > 0 && body_size.saturating_sub(base_size) > 16);
        let height = if grew {
            base_h.saturating_add(1)
        } else {
            base_h
        };
        let size = body_size.min(u16::MAX as usize) as u16;
        (height <= MAX_INLINE_HEIGHT).then(|| {
            Arc::new(InlineTemplate {
                body,
                returns,
                reloc_plan,
                height,
                size,
                form,
            })
        })
    })
}

fn lower_with_compact(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    index: usize,
    (limit, compact, quiet): (usize, bool, &Quiet),
) -> Result<(Option<Lowered>, bool)> {
    let mut lowered = lower_template(cx, (reached, slots), inline, index, (limit, quiet))?;
    let over = !compact && inline.host.retry && matches!(lowered, Some((Err(Miss::Over), ..)));
    let Some(roomy) = over
        .then(|| super::super::budget::compact_limit(cx, reached, index, limit))
        .flatten()
    else {
        return Ok((lowered, false));
    };
    let Some(first) = lowered.as_ref().map(|found| found.3) else {
        return Ok((lowered, false));
    };
    tally(Tally::Retry);
    inline.host.compact = true;
    lowered = lower_template(cx, (reached, slots), inline, index, (roomy, quiet))?
        .map(|(fit, relocs, plan, cycle)| (fit, relocs, plan, cycle.min(first)));
    Ok((lowered, true))
}

fn apply_inline_cache_hit(
    bodies: &Bodies,
    key: TemplateKey,
    hit: crate::image::aot_cache::InlineCacheHit,
) -> Option<InlineHit> {
    use crate::image::aot_cache::InlineCacheHit;
    match hit {
        InlineCacheHit::Negative(footprint) => {
            bodies
                .rejected
                .get_or_insert_with(TemplateKey { footprint, ..key }, || ());
            None
        }
        InlineCacheHit::Shape(footprint, returns, height, size) => {
            Some((footprint, Found::Shape(returns, height, size)))
        }
        InlineCacheHit::Body(footprint, template) => {
            let (stored, _) = bodies.cache_template(TemplateKey { footprint, ..key }, template);
            Some((footprint, Found::Body(stored, None)))
        }
    }
}

fn ensure_inline_template_inner(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    index: usize,
    (limit, shape_only, quiet): (usize, bool, &Quiet),
) -> Result<(Option<InlineHit>, usize)> {
    let item = reached.list[index].0.0;
    let depth = inline.stack.len().saturating_sub(1);
    let cyclic = slots.bodies().is_cyclic(cx, reached, index);
    let key_low = slots.low_entries();
    let key_compact = inline.host.compact;
    let key_sole = super::super::budget::stable_sole(cx, reached, index);
    let aot_cache = slots.aot_cache();
    if !cyclic
        && quiet.holds(cx, slots)
        && let Some(cache) = aot_cache.as_ref()
        && let Some(hit) = cache.lookup_inline(
            (slots, cx, reached),
            index,
            (limit, key_low, key_compact, key_sole, shape_only),
        )
    {
        let probe_key = TemplateKey {
            item,
            footprint: FootprintId(0),
            limit,
            low: key_low,
            compact: key_compact,
            sole: key_sole,
        };
        let applied = apply_inline_cache_hit(slots.bodies(), probe_key, hit);
        return Ok((applied, usize::MAX));
    }
    let fp_res = eval_footprint_with_cycle(cx, (reached, slots), inline, index);
    if inline.aborted {
        return Ok((None, usize::MAX));
    }
    let fp_h = std::mem::replace(&mut inline.max_callee_height, 0);
    let fp_sz = std::mem::replace(&mut inline.max_callee_size, 0);
    let Some((footprint, fp_cycle)) = quiet.lowered(slots, fp_res)? else {
        return Ok((None, usize::MAX));
    };
    let key = TemplateKey {
        item,
        footprint,
        limit,
        low: key_low,
        compact: key_compact,
        sole: key_sole,
    };
    let servable = shape_only && fp_cycle > depth;
    let disk = match try_cached_template(
        cx,
        (reached, slots),
        &key,
        (shape_only, fp_cycle > depth, quiet),
    ) {
        Ok(Some((hit, from_disk))) => {
            if from_disk
                && !cyclic
                && fp_cycle > depth
                && quiet.holds(cx, slots)
                && let Some(cache) = aot_cache.as_ref()
            {
                let empty_plan = RelocPlan::default();
                let (tmpl, plan) = match &hit {
                    Some(Found::Body(t, _)) => (Some(t), t.reloc_plan.as_ref()),
                    _ => (None, &empty_plan),
                };
                cache.record_inline(
                    (cx, reached),
                    index,
                    (limit, key.low, key.compact, key.sole),
                    footprint,
                    tmpl,
                    plan,
                    &[],
                );
            }
            return Ok((hit.map(|f| (footprint, f)), fp_cycle));
        }
        Ok(None) => None,
        Err(disk) => disk,
    };
    let claim = if fp_cycle > depth && quiet.holds(cx, slots) {
        let bodies = slots.bodies();
        match bodies
            .template_flight
            .claim(key, inline.warming, |k| bodies.cached_template_state(k))
        {
            TemplateClaim::Build(guard) => Some(guard),
            TemplateClaim::Ready(hit) => {
                tally(Tally::Hit);
                return Ok((hit.map(|t| (footprint, Found::Body(t, None))), fp_cycle));
            }
            TemplateClaim::Cycle => None,
            TemplateClaim::Abort => {
                inline.aborted = true;
                return Ok((None, usize::MAX));
            }
        }
    } else {
        None
    };
    let (lowered, retried) = lower_with_compact(
        cx,
        (reached, slots),
        inline,
        index,
        (limit, key.compact, quiet),
    )?;
    if inline.aborted {
        return Ok((None, usize::MAX));
    }
    let form = if inline.host.compact {
        Form::Compact
    } else {
        inline
            .host
            .seen
            .max(super::super::budget::own_form(cx, reached, index))
    };
    let (ir_h, ir_sz) = (inline.max_callee_height, inline.max_callee_size);
    let Some((fit, relocs, reloc_plan, ir_cycle)) = lowered else {
        if let Some(disk) = disk {
            disk.settle(
                cx,
                (reached, slots),
                None,
                (false, servable && quiet.holds(cx, slots)),
            );
        }
        return Ok((None, fp_cycle));
    };
    let cycle = fp_cycle.min(ir_cycle);
    let plan_for_cache = Arc::clone(&reloc_plan);
    let template = pack_template(
        fit,
        reloc_plan,
        form,
        (fp_h.max(ir_h), fp_sz.max(ir_sz) as usize),
    );
    if retried && template.is_some() {
        tally(Tally::Compact);
    }
    let calm = quiet.holds(cx, slots);
    let template = settle_built_template(
        cx,
        (reached, slots),
        key,
        template,
        (disk, calm, servable, cycle <= depth),
    );
    if let Some(guard) = claim {
        guard.finish((cycle > depth && calm).then(|| template.clone()));
    }
    if !cyclic
        && cycle > depth
        && quiet.holds(cx, slots)
        && let Some(cache) = aot_cache.as_ref()
    {
        cache.record_inline(
            (cx, reached),
            index,
            (limit, key.low, key.compact, key.sole),
            footprint,
            template.as_ref(),
            &plan_for_cache,
            &relocs,
        );
    }
    Ok((
        template.map(|t| (footprint, Found::Body(t, Some(relocs)))),
        cycle,
    ))
}

fn lower_template(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    index: usize,
    (limit, quiet): (usize, &Quiet),
) -> Result<Option<Lowered>> {
    let lowered = run_with_cycle(
        cx,
        (reached, slots),
        inline,
        index,
        LowerMode::Stencil,
        true,
    );
    Ok(quiet
        .lowered(slots, lowered)?
        .map(|(func, relocs, plan, cycle, simplified)| {
            let limit = if plan.has_calls() {
                limit
            } else {
                super::super::budget::leaf_limit(cx, reached, index, limit)
            };
            let elide = !reached.is_incremental() || slots.allow_inline();
            let res = if haste_codegen::trace_enabled() {
                let label = cx.name(reached.list[index].0.0).to_owned();
                haste_codegen::with_trace_label(&label, || {
                    simplify_inline(cx, reached, func, &relocs, (limit, simplified, elide))
                })
            } else {
                simplify_inline(cx, reached, func, &relocs, (limit, simplified, elide))
            };
            (res, relocs, plan, cycle)
        }))
}

fn simplify_inline(
    cx: &mut Context<'_>,
    reached: &Reached,
    mut func: Function,
    relocs: &[SymbolSource],
    (limit, simplified, elide): (usize, bool, bool),
) -> std::result::Result<(Arc<Function>, bool), Miss> {
    let quick = crate::profile() != crate::outline::tree::Profile::Release;
    let pre_cap = if quick {
        limit.saturating_mul(6).max(256)
    } else {
        limit.saturating_mul(6).max(1600)
    };
    if func.size() > pre_cap {
        return Err(Miss::Shape);
    }
    if func.is_coroutine() {
        haste_codegen::split_coroutine(&mut func);
    }
    if !simplified {
        simplify_tallied(&mut func, Tally::Built);
    }
    if !func.unwind_drops().is_empty() {
        let elided = elide
            && (!quick || func.inline_size() <= limit.saturating_mul(2))
            && func.elide_dead_unwind_drops(|sym| {
                sym_may_unwind(cx, reached, relocs.get(sym as usize))
            });
        func.zero_init_unwind_flags();
        if elided {
            simplify_tallied(&mut func, Tally::Built);
        }
    }
    if !func.can_inline() {
        return Err(Miss::Shape);
    }
    if func.inline_size() > limit {
        return Err(Miss::Over);
    }
    let returns = func.has_return();
    Ok((Arc::new(func), returns))
}
