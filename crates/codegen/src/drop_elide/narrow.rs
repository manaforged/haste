use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, Op, StackSlot, Value};

use super::sink::{entry_first_zeros, flag_call_liveness, plan_flag_sink, plan_local_sink};
use super::{CalleeKind, FlagOp, UnwindDrop, analyze_flag, classify_flag_op, def_op};

pub(super) struct NarrowAnalysis<'a> {
    pub(super) defs: &'a [Option<Inst>],
    pub(super) kinds: &'a [CalleeKind],
    pub(super) addr_slot: &'a [Option<(StackSlot, i32)>],
    pub(super) slot_escaped: &'a [bool],
    pub(super) slot_loaded: &'a [bool],
    pub(super) local_write_only: &'a [bool],
}

type LocalStore = (StackSlot, Value, i32);
type CallHook = (Inst, Vec<LocalStore>, Option<StackSlot>);
type HeadSet = (Block, Vec<LocalStore>, Option<StackSlot>);

struct StoreTables {
    flag_uses: Vec<u32>,
    local_uses: Vec<u32>,
    flag_stores: Vec<Vec<(Block, Inst, Option<bool>)>>,
    local_stores: Vec<Vec<(Block, Inst, Value, i32)>>,
}

struct NarrowCtx<'a> {
    cfg: &'a Cfg,
    depths: &'a [u32],
    analysis: &'a NarrowAnalysis<'a>,
    tables: &'a StoreTables,
    keep_zero: Option<&'a [Option<Inst>]>,
    post_init: bool,
}

struct NarrowEdits {
    remove: Vec<bool>,
    before_call: Vec<CallHook>,
    block_head_sets: Vec<HeadSet>,
    entry_zero_flags: Vec<StackSlot>,
}

struct FlagSummary {
    dead_zeros: Vec<Inst>,
    fresh_sets: Vec<Inst>,
    flag_visited: Vec<u8>,
    always_active_in: Vec<bool>,
    always_active_out: Vec<bool>,
}

fn blocks_reaching(cfg: &Cfg, nblocks: usize, targets: &[(Block, Inst)]) -> Vec<bool> {
    let mut seen = vec![false; nblocks];
    let mut work: Vec<Block> = Vec::with_capacity(targets.len());
    for &(b, _) in targets {
        if !std::mem::replace(&mut seen[b.index()], true) {
            work.push(b);
        }
    }
    while let Some(b) = work.pop() {
        for &pred in cfg.preds_of(b) {
            if !std::mem::replace(&mut seen[pred.index()], true) {
                work.push(pred);
            }
        }
    }
    seen
}

fn mark_dead_zeros(
    remove: &mut [bool],
    dead_zeros: &[Inst],
    keep_zero: Option<&[Option<Inst>]>,
    fi: usize,
) {
    let kept = keep_zero.and_then(|kz| kz[fi]);
    for &zinst in dead_zeros {
        if kept != Some(zinst) {
            remove[zinst.index()] = true;
        }
    }
}

fn collect_store_tables(func: &Function, cfg: &Cfg, analysis: &NarrowAnalysis<'_>) -> StoreTables {
    let nslots = func.slots.len();
    let mut flag_uses = vec![0u32; nslots];
    let mut local_uses = vec![0u32; nslots];
    for u in &func.unwind_drops {
        if u.flag.index() < nslots {
            flag_uses[u.flag.index()] += 1;
        }
        if u.local.index() < nslots {
            local_uses[u.local.index()] += 1;
        }
    }
    let mut flag_stores: Vec<Vec<(Block, Inst, Option<bool>)>> = vec![Vec::new(); nslots];
    let mut local_stores: Vec<Vec<(Block, Inst, Value, i32)>> = vec![Vec::new(); nslots];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            match func.insts[inst.index()].op {
                Op::StackStore(v, slot, off) if slot.index() < nslots => {
                    if flag_uses[slot.index()] == 1 {
                        let bit = (off == 0)
                            .then(|| match def_op(func, analysis.defs, v) {
                                Some(&Op::Const(c)) => Some(c != 0),
                                _ => None,
                            })
                            .flatten();
                        flag_stores[slot.index()].push((b, inst, bit));
                    }
                    if local_uses[slot.index()] == 1 && !analysis.slot_escaped[slot.index()] {
                        local_stores[slot.index()].push((b, inst, v, off));
                    }
                }
                Op::Store(v, addr, off) => {
                    if let Some((slot, base)) = analysis.addr_slot[addr.index()]
                        && slot.index() < nslots
                        && local_uses[slot.index()] == 1
                        && analysis.local_write_only[slot.index()]
                        && let Some(total) = base.checked_add(off)
                        && crate::mem::in_range(
                            i64::from(total),
                            func.ty(v).bytes(),
                            func.slots[slot.index()].size,
                        )
                    {
                        local_stores[slot.index()].push((b, inst, v, total));
                    }
                }
                _ => {}
            }
        }
    }
    StoreTables {
        flag_uses,
        local_uses,
        flag_stores,
        local_stores,
    }
}

fn try_local_sinks(
    func: &Function,
    ctx: &NarrowCtx<'_>,
    u: &UnwindDrop,
    flag_visited: &[u8],
    edits: &mut NarrowEdits,
) {
    let nslots = func.slots.len();
    let li = u.local.index();
    if u.arg.is_indirect()
        || u.local == u.flag
        || li >= nslots
        || ctx.tables.local_uses[li] != 1
        || ctx.analysis.slot_escaped[li]
    {
        return;
    }
    for &(lb, linst, val, off) in &ctx.tables.local_stores[li] {
        if !edits.remove[linst.index()]
            && let Some(frontier) = plan_local_sink(
                func,
                ctx.cfg,
                ctx.analysis.defs,
                ctx.depths,
                (flag_visited, u.local, u.flag),
                (lb, linst, off, func.ty(val).bytes()),
            )
        {
            edits.remove[linst.index()] = true;
            for t in frontier {
                edits
                    .block_head_sets
                    .push((t, vec![(u.local, val, off)], None));
            }
        }
    }
}

fn prune_dead_flag_stores(
    func: &Function,
    ctx: &NarrowCtx<'_>,
    u: &UnwindDrop,
    stores: &[(Block, Inst, Option<bool>)],
    live_out: &[bool],
    remove: &mut [bool],
) {
    let kept_entry = ctx.keep_zero.and_then(|kz| kz[u.flag.index()]);
    let mut visited_blocks = Vec::new();
    for &(sb, _, _) in stores {
        if visited_blocks.contains(&sb) {
            continue;
        }
        visited_blocks.push(sb);
        let mut live = live_out[sb.index()];
        for &inst in func.blocks[sb.index()].insts.iter().rev() {
            if remove[inst.index()] {
                continue;
            }
            match classify_flag_op(func, ctx.analysis.defs, ctx.analysis.kinds, u.flag, inst) {
                Some(FlagOp::Exit) => live = false,
                Some(FlagOp::MayUnwindCall | FlagOp::TryCall | FlagOp::Load) => live = true,
                Some(op @ (FlagOp::Clear | FlagOp::Set)) => {
                    if !live && kept_entry != Some(inst) {
                        remove[inst.index()] = true;
                    }
                    if matches!(op, FlagOp::Clear) && !remove[inst.index()] {
                        live = false;
                    }
                }
                None => {}
            }
        }
    }
}

fn collect_co_sunk_locals(
    func: &Function,
    ctx: &NarrowCtx<'_>,
    u: &UnwindDrop,
    sb: Block,
    frontier_len: usize,
    remove: &mut [bool],
) -> Vec<LocalStore> {
    let nslots = func.slots.len();
    let li = u.local.index();
    let local_stores = &ctx.tables.local_stores;
    if u.local == u.flag
        || li >= nslots
        || ctx.tables.local_uses[li] != 1
        || !ctx.analysis.local_write_only[li]
        || local_stores[li].is_empty()
        || local_stores[li].len() > 2
    {
        return Vec::new();
    }
    let cheap_vals = frontier_len <= 2
        || local_stores[li].iter().all(|&(_, _, val, _)| {
            matches!(
                def_op(func, ctx.analysis.defs, val),
                Some(Op::Const(_) | Op::StackAddr(..))
            )
        });
    if !cheap_vals || !local_stores[li].iter().all(|&(lb, ..)| lb == sb) {
        return Vec::new();
    }
    let mut sink_local = Vec::new();
    for &(_, linst, val, off) in &local_stores[li] {
        remove[linst.index()] = true;
        sink_local.push((u.local, val, off));
    }
    sink_local
}

fn sink_unbounded_flag_drop(
    func: &Function,
    ctx: &NarrowCtx<'_>,
    u: &UnwindDrop,
    loaded_or_dyn: bool,
    arm_blocks: &[Block],
    summary: &FlagSummary,
    edits: &mut NarrowEdits,
) {
    let fi = u.flag.index();
    let stores = &ctx.tables.flag_stores[fi];
    mark_dead_zeros(&mut edits.remove, &summary.dead_zeros, ctx.keep_zero, fi);
    let (live_in, live_out) = flag_call_liveness(
        func,
        ctx.cfg,
        ctx.analysis.defs,
        ctx.analysis.kinds,
        &edits.remove,
        u.flag,
    );
    prune_dead_flag_stores(func, ctx, u, stores, &live_out, &mut edits.remove);
    for &(sb, sinst, is_set) in stores {
        if is_set == Some(true)
            && !edits.remove[sinst.index()]
            && summary.fresh_sets.contains(&sinst)
            && let Some(frontier) = plan_flag_sink(
                func,
                ctx.cfg,
                ctx.analysis.kinds,
                ctx.depths,
                (
                    &live_in,
                    (&summary.always_active_in, &summary.always_active_out),
                    sb,
                    sinst,
                    u.flag,
                ),
            )
        {
            edits.remove[sinst.index()] = true;
            let sink_local = if !loaded_or_dyn && arm_blocks.len() == 1 {
                collect_co_sunk_locals(func, ctx, u, sb, frontier.len(), &mut edits.remove)
            } else {
                Vec::new()
            };
            for t in frontier {
                if !edits
                    .block_head_sets
                    .iter()
                    .any(|&(bt, _, bf)| bt == t && bf == Some(u.flag))
                {
                    edits
                        .block_head_sets
                        .push((t, sink_local.clone(), Some(u.flag)));
                }
            }
        }
    }
    try_local_sinks(func, ctx, u, &summary.flag_visited, edits);
}

fn collect_call_sunk_locals(
    func: &Function,
    ctx: &NarrowCtx<'_>,
    u: &UnwindDrop,
    live_calls: &[(Block, Inst)],
    remove: &mut [bool],
) -> Vec<LocalStore> {
    let nslots = func.slots.len();
    let li = u.local.index();
    if u.local == u.flag
        || li >= nslots
        || ctx.tables.local_uses[li] != 1
        || !ctx.analysis.local_write_only[li]
        || ctx.tables.local_stores[li].is_empty()
    {
        return Vec::new();
    }
    let reaches = blocks_reaching(ctx.cfg, func.blocks.len(), live_calls);
    let mut dom_stores = Vec::new();
    for &(sb, sinst, val, off) in &ctx.tables.local_stores[li] {
        if live_calls
            .iter()
            .all(|&(cb, _)| sb != cb && ctx.cfg.dominates(sb, cb))
        {
            dom_stores.push((sb, sinst, val, off));
        } else if reaches[sb.index()] {
            return Vec::new();
        }
    }
    let max_call_depth = live_calls
        .iter()
        .map(|&(cb, _)| ctx.depths[cb.index()])
        .max()
        .unwrap_or(0);
    let min_store_depth = dom_stores
        .iter()
        .map(|&(sb, ..)| ctx.depths[sb.index()])
        .min()
        .unwrap_or(0);
    if dom_stores.is_empty()
        || dom_stores.len() > 4
        || dom_stores.len() * live_calls.len() > 8
        || max_call_depth > min_store_depth
    {
        return Vec::new();
    }
    for &(_, sinst, _, _) in &ctx.tables.local_stores[li] {
        remove[sinst.index()] = true;
    }
    dom_stores
        .into_iter()
        .map(|(_, _, val, off)| (u.local, val, off))
        .collect()
}

fn narrow_bounded_flag_drop(
    func: &Function,
    ctx: &NarrowCtx<'_>,
    u: &UnwindDrop,
    arm_blocks: &[Block],
    live_calls: &[(Block, Inst)],
    summary: &FlagSummary,
    edits: &mut NarrowEdits,
) {
    let fi = u.flag.index();
    let flag_needs_narrow = arm_blocks
        .iter()
        .any(|ab| !live_calls.iter().any(|&(cb, _)| cb == *ab));
    let sink_local = collect_call_sunk_locals(func, ctx, u, live_calls, &mut edits.remove);
    if sink_local.is_empty() {
        try_local_sinks(func, ctx, u, &summary.flag_visited, edits);
    }
    if !flag_needs_narrow && sink_local.is_empty() {
        mark_dead_zeros(&mut edits.remove, &summary.dead_zeros, ctx.keep_zero, fi);
        return;
    }
    if flag_needs_narrow {
        for &(_, inst, _) in &ctx.tables.flag_stores[fi] {
            edits.remove[inst.index()] = true;
        }
        if ctx.post_init {
            edits.entry_zero_flags.push(u.flag);
        }
    }
    let flag_opt = flag_needs_narrow.then_some(u.flag);
    for &(_, cinst) in live_calls {
        edits
            .before_call
            .push((cinst, sink_local.clone(), flag_opt));
    }
}

fn emit_hook_stores(
    func: &mut Function,
    new: &mut Vec<Inst>,
    locs: &[LocalStore],
    flag: Option<StackSlot>,
    consts: Option<(Value, Inst, Value, Inst)>,
) {
    for &(slot, pv, off) in locs {
        new.push(func.add(Op::StackStore(pv, slot, off), None));
    }
    if let (Some(flag), Some((one, ..))) = (flag, consts) {
        new.push(func.add(Op::StackStore(one, flag, 0), None));
    }
}

fn apply_narrow_edits(func: &mut Function, entry: Block, edits: NarrowEdits) -> (bool, bool) {
    let NarrowEdits {
        remove,
        before_call,
        block_head_sets,
        entry_zero_flags,
    } = edits;
    if before_call.is_empty() && block_head_sets.is_empty() && !remove.iter().any(|&r| r) {
        return (false, false);
    }
    let sunk_flag = block_head_sets.iter().any(|(_, _, f)| f.is_some());
    let needs_consts = before_call.iter().any(|(_, _, f)| f.is_some()) || sunk_flag;
    let consts = needs_consts.then(|| {
        let one = func.value(crate::Type::I8);
        let one_inst = func.add(Op::Const(1), Some(one));
        let zero = func.value(crate::Type::I8);
        let zero_inst = func.add(Op::Const(0), Some(zero));
        (one, one_inst, zero, zero_inst)
    });
    let mut per_call: Vec<Vec<(Vec<LocalStore>, Option<StackSlot>)>> =
        vec![Vec::new(); remove.len()];
    for (cinst, locs, flag) in before_call {
        per_call[cinst.index()].push((locs, flag));
    }
    let mut per_head: Vec<Vec<(Vec<LocalStore>, Option<StackSlot>)>> =
        vec![Vec::new(); func.blocks.len()];
    for (b, locs, flag) in block_head_sets {
        per_head[b.index()].push((locs, flag));
    }
    for bi in 0..func.blocks.len() {
        let old = std::mem::take(&mut func.blocks[bi].insts);
        let mut new = Vec::with_capacity(old.len() + 4);
        if bi == entry.index()
            && let Some((_, one_inst, zero, zero_inst)) = consts
        {
            new.push(one_inst);
            new.push(zero_inst);
            for &flag in &entry_zero_flags {
                new.push(func.add(Op::StackStore(zero, flag, 0), None));
            }
        }
        for (locs, flag) in &per_head[bi] {
            emit_hook_stores(func, &mut new, locs, *flag, consts);
        }
        for inst in old {
            if inst.index() < remove.len() && remove[inst.index()] {
                continue;
            }
            let hooks = if inst.index() < per_call.len() {
                &per_call[inst.index()]
            } else {
                &[][..]
            };
            for (locs, flag) in hooks {
                emit_hook_stores(func, &mut new, locs, *flag, consts);
            }
            new.push(inst);
            if let Some((_, _, zero, _)) = consts {
                for (_, flag) in hooks {
                    if let Some(flag) = flag {
                        new.push(func.add(Op::StackStore(zero, *flag, 0), None));
                    }
                }
            }
        }
        func.blocks[bi].insts = new;
    }
    (true, sunk_flag)
}

pub(super) fn narrow_unwind_drops(
    func: &mut Function,
    entry: Block,
    post_init: bool,
    analysis: &NarrowAnalysis<'_>,
) -> (bool, bool) {
    let nslots = func.slots.len();
    let keep_zero = post_init.then(|| entry_first_zeros(func, analysis.defs, entry, nslots));
    let cfg = func.cfg();
    let depths = &cfg.loops().depths;
    let tables = collect_store_tables(func, &cfg, analysis);
    let ctx = NarrowCtx {
        cfg: &cfg,
        depths,
        analysis,
        tables: &tables,
        keep_zero: keep_zero.as_deref(),
        post_init,
    };
    let mut edits = NarrowEdits {
        remove: vec![false; func.insts.len()],
        before_call: Vec::new(),
        block_head_sets: Vec::new(),
        entry_zero_flags: Vec::new(),
    };
    for u in &func.unwind_drops {
        let fi = u.flag.index();
        if fi >= nslots || tables.flag_uses[fi] != 1 || analysis.slot_escaped[fi] {
            continue;
        }
        let stores = &tables.flag_stores[fi];
        if stores.is_empty() {
            continue;
        }
        let loaded_or_dyn = analysis.slot_loaded[fi] || stores.iter().any(|s| s.2.is_none());
        let arm_blocks: Vec<Block> = stores
            .iter()
            .filter(|s| s.2 == Some(true))
            .map(|s| s.0)
            .collect();
        if arm_blocks.is_empty() {
            continue;
        }
        let (dead_zeros, fresh_sets, live_calls, flag_visited, flag_out) =
            analyze_flag(func, analysis.defs, analysis.kinds, entry, u.flag);
        let summary = FlagSummary {
            dead_zeros,
            fresh_sets,
            always_active_in: flag_visited.iter().map(|&v| v == 2).collect(),
            always_active_out: flag_out.iter().map(|&v| v == 2).collect(),
            flag_visited,
        };
        let narrow_calls = (!loaded_or_dyn)
            .then_some(live_calls)
            .flatten()
            .filter(|lc| {
                !lc.is_empty()
                    && lc.len() <= 4
                    && lc
                        .iter()
                        .map(|&(b, _)| depths[b.index()])
                        .max()
                        .unwrap_or(0)
                        <= arm_blocks
                            .iter()
                            .map(|&b| depths[b.index()])
                            .min()
                            .unwrap_or(0)
            });
        if let Some(live_calls) = narrow_calls {
            narrow_bounded_flag_drop(
                func,
                &ctx,
                u,
                &arm_blocks,
                &live_calls,
                &summary,
                &mut edits,
            );
        } else {
            sink_unbounded_flag_drop(
                func,
                &ctx,
                u,
                loaded_or_dyn,
                &arm_blocks,
                &summary,
                &mut edits,
            );
        }
    }
    apply_narrow_edits(func, entry, edits)
}
