use super::coalesce::{ScanBuf, is_pure_scan_op, slot_unescaped, try_coalesce_store};
use super::concat::{assemble_const, emit_assembled, forward_concat, is_silent_store};
use super::dom::{clobbers_loc, invalidate_preserving, record_base, write_preserving};
use super::elide::{try_fold_private_rmw, try_scalarize_slot_move};
use super::join::{BlockClobber, JoinCtx, resolve_select_load, rodata_const, try_join_phi_forward};
use super::remat::{
    FarCseGuard, SubwordAction, SubwordSlices, drop_covered_stores, drop_freed_stores,
    drop_stores_if, resolve_subword_plan,
};
use super::{State, addr_loc, emit_op};
use crate::ir::{Block, Function, Inst, Op, Type, Value};
use crate::mem::{KnownLoads, MemBase, MemLoc, MemRoot, Provenance, may_alias, may_alias_at};
use crate::merge::find;
use crate::reloc::RuntimeSymbol;

enum Effect {
    None,
    Read(MemLoc, Type, Value),
    Write(MemLoc, Type, Value),
    Unknown,
}

enum Step {
    Keep,
    SkipSync,
    Skip,
}

fn effect(func: &Function, state: &State<'_>, inst: Inst) -> Effect {
    let data = &func.insts[inst.index()];
    match (&data.op, data.result) {
        (&Op::Load(addr, offset), Some(result)) => {
            let ty = func.ty(result);
            Effect::Read(
                addr_loc(state.prov, state.subst, addr, offset, ty.bytes()),
                ty,
                result,
            )
        }
        (&Op::StackLoad(from, offset), Some(result)) => {
            let ty = func.ty(result);
            Effect::Read(
                MemLoc::slot(from, i64::from(offset), ty.bytes()),
                ty,
                result,
            )
        }
        (&Op::Store(value, addr, offset), _) => {
            let ty = func.ty(value);
            Effect::Write(
                addr_loc(state.prov, state.subst, addr, offset, ty.bytes()),
                ty,
                find(state.subst, value),
            )
        }
        (&Op::StackStore(value, to, offset), _) => {
            let ty = func.ty(value);
            Effect::Write(
                MemLoc::slot(to, i64::from(offset), ty.bytes()),
                ty,
                find(state.subst, value),
            )
        }
        (&Op::Call { callee, .. }, _)
            if state.prov.frame_runtime(callee) == Some(RuntimeSymbol::Malloc)
                || func.is_fresh_alloc(callee)
                || func
                    .pure_thread_cell_result(&data.op, |v| state.consts[v.index()] == Some(0))
                    .is_some() =>
        {
            Effect::None
        }
        (op, _) if crate::mem::is_atomic_barrier(op) => Effect::Unknown,
        (op, _)
            if matches!(
                op,
                Op::Copy(..) | Op::Move(..) | Op::Call { .. } | Op::TryCall { .. }
            ) =>
        {
            Effect::Unknown
        }
        _ => Effect::None,
    }
}

fn forward_subword(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    ctx: (&KnownLoads, &SubwordSlices, &FarCseGuard),
    (inst, loc, ty, result): (Inst, MemLoc, Type, Value),
) -> Option<Option<Value>> {
    let action = resolve_subword_plan(
        func,
        (inst, loc, ty),
        ctx,
        (state.post_inline, state.defs, state.subst, state.consts),
    )?;
    match action {
        SubwordAction::Forwarded(val) => Some(Some(val)),
        SubwordAction::Const(imm) => {
            func.insts[inst.index()].op = Op::Const(imm);
            if ty.is_small_int() {
                state.consts[result.index()] = Some(imm);
            }
            Some(None)
        }
        SubwordAction::Plan(plan) => {
            func.insts[inst.index()].op =
                plan.build_op(|op_ty, op| emit_op(func, state, kept, op_ty, op));
            Some(None)
        }
    }
}

fn copy_len(op: &Op, consts: &[Option<i64>]) -> Option<(Value, Value, u32)> {
    match *op {
        Op::Move(dst, src, bytes) => Some((dst, src, bytes)),
        Op::Copy(dst, src, len) => consts
            .get(len.index())
            .copied()
            .flatten()
            .and_then(|n| u32::try_from(n).ok())
            .map(|b| (dst, src, b)),
        _ => None,
    }
}

fn call_passes_noalias(op: &Op, prov: &Provenance, p: u16) -> bool {
    match *op {
        Op::Call { ref args, .. } | Op::TryCall { ref args, .. } => args.iter().any(|arg| {
            prov.param_roots.get(arg.index()).copied() == Some(MemRoot::NoAliasParam(p))
        }),
        _ => false,
    }
}

fn pre_scan_op(
    func: &mut Function,
    state: &State<'_>,
    bufs: &mut ScanBuf,
    guard: &mut FarCseGuard,
    inst: Inst,
) -> bool {
    if let Some((_, from, bytes)) = copy_len(&func.insts[inst.index()].op, state.consts) {
        if bytes > 0 {
            let read_loc = addr_loc(state.prov, state.subst, from, 0, bytes);
            bufs.pending_stores
                .retain(|&(ploc, _)| !may_alias(read_loc, ploc, state.prov));
            bufs.const_stores
                .retain(|cs| !may_alias(read_loc, cs.loc, state.prov));
        }
        return false;
    }
    let mut changed = false;
    match func.insts[inst.index()].op {
        Op::Copy(_, from, _) => {
            let r = addr_loc(state.prov, state.subst, from, 0, u32::MAX);
            let ok = |l: MemLoc| {
                l.base != r.base
                    && (l.root == MemRoot::Unknown || l.root != r.root)
                    && !may_alias(r, l, state.prov)
            };
            bufs.pending_stores.retain(|&(p, _)| ok(p));
            bufs.const_stores.retain(|cs| ok(cs.loc));
        }
        Op::Trap | Op::Trapz(_) | Op::Assume(_) => {
            bufs.pending_stores
                .retain(|&(ploc, _)| slot_unescaped(state.prov, ploc));
            bufs.const_stores
                .retain(|cs| slot_unescaped(state.prov, cs.loc));
        }
        Op::TryCall { .. } => {
            bufs.pending_stores.clear();
            bufs.const_stores.clear();
        }
        Op::Return(..) if state.post_inline => {
            changed |= drop_stores_if(func, bufs, guard, |ploc| slot_unescaped(state.prov, ploc));
        }
        _ => {}
    }
    changed | drop_freed_stores(func, state, (bufs, guard), inst)
}

fn try_forward_select_load(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    known: &KnownLoads,
    (inst, ty): (Inst, Type),
) -> Option<Option<Value>> {
    let Op::Load(addr, offset) = func.insts[inst.index()].op else {
        return None;
    };
    if ty == Type::I128 {
        return None;
    }
    let &(c, ptr_a, ptr_b) = state
        .selects
        .get(find(state.subst, addr).index())
        .and_then(Option::as_ref)?;
    if func.ty(c).is_vector() {
        return None;
    }
    let loc_a = addr_loc(state.prov, state.subst, ptr_a, offset, ty.bytes());
    let loc_b = addr_loc(state.prov, state.subst, ptr_b, offset, ty.bytes());
    let (val_a, val_b) = resolve_select_load(func, state, kept, known, ty, (loc_a, loc_b))?;
    if val_a == val_b {
        Some(Some(val_a))
    } else {
        func.insts[inst.index()].op = Op::Select(find(state.subst, c), val_a, val_b);
        Some(None)
    }
}

struct ScanCtx<'a, 's, 'j> {
    block: Block,
    state: &'a mut State<'s>,
    known: &'a mut KnownLoads,
    bases: &'a mut Vec<(MemBase, MemRoot)>,
    join: &'a mut JoinCtx<'j>,
    bufs: &'a mut ScanBuf,
    clobbers_seen: &'a mut Vec<BlockClobber>,
    guard: &'a mut FarCseGuard,
    slices: &'a mut SubwordSlices,
    changed: bool,
    renamed: bool,
}

fn handle_read(
    func: &mut Function,
    ctx: &mut ScanCtx<'_, '_, '_>,
    (inst, loc, ty, result): (Inst, MemLoc, Type, Value),
) -> Step {
    let (mut forwarded, skip_far) = ctx.guard.check_cse(
        &func.insts[inst.index()].op,
        ty,
        ctx.known.get(loc.base, loc.offset, ty),
    );
    if forwarded.is_none() {
        if let Some(imm) = rodata_const(func, loc, ty) {
            func.insts[inst.index()].op = Op::Const(imm);
            ctx.state.consts[result.index()] = Some(imm);
            ctx.known.remember(loc, ty, result);
            ctx.changed = true;
        } else if let Some(assembled) = assemble_const(ctx.known, loc, ty, ctx.state.consts) {
            emit_assembled(
                func,
                ctx.state,
                &mut ctx.bufs.kept,
                inst,
                ty,
                result,
                assembled,
            );
            ctx.known.remember(loc, ty, result);
            ctx.changed = true;
        } else if !skip_far
            && let Some(sub_fwd) = forward_subword(
                func,
                ctx.state,
                &mut ctx.bufs.kept,
                (&*ctx.known, &*ctx.slices, &*ctx.guard),
                (inst, loc, ty, result),
            )
        {
            if let Some(val) = sub_fwd {
                forwarded = Some(val);
            } else {
                ctx.known.remember(loc, ty, result);
            }
            ctx.changed = true;
        } else if !skip_far
            && forward_concat(
                func,
                ctx.state,
                &mut ctx.bufs.kept,
                ctx.known,
                inst,
                loc,
                ty,
            )
        {
            ctx.known.remember(loc, ty, result);
            ctx.changed = true;
        } else if !skip_far
            && let Some(sel_fwd) =
                try_forward_select_load(func, ctx.state, &mut ctx.bufs.kept, ctx.known, (inst, ty))
        {
            if let Some(val) = sel_fwd {
                forwarded = Some(val);
            } else {
                ctx.known.remember(loc, ty, result);
            }
            ctx.changed = true;
        } else if !skip_far
            && let Some(fwd) = try_join_phi_forward(
                func,
                (ctx.block, inst, loc, ty, result),
                ctx.state,
                ctx.known,
                ctx.join,
                (ctx.clobbers_seen.as_slice(), ctx.bufs.kept.as_slice()),
            )
        {
            forwarded = fwd;
            ctx.changed = true;
        } else {
            ctx.known.remember(loc, ty, result);
            ctx.bufs
                .pending_stores
                .retain(|&(ploc, _)| !may_alias_at(loc, ploc, ctx.state.prov, Some(inst)));
            let safe_read = slot_unescaped(ctx.state.prov, loc);
            ctx.bufs.const_stores.retain(|cs| {
                !may_alias_at(loc, cs.loc, ctx.state.prov, Some(inst))
                    && (slot_unescaped(ctx.state.prov, cs.loc)
                        || safe_read
                        || cs.loc.base == loc.base)
            });
        }
        record_base(ctx.bases, loc);
    }
    if let Some(value) = forwarded {
        ctx.state.subst[result.index()] = value;
        if let Some(imm) = ctx.state.consts.get(value.index()).copied().flatten() {
            ctx.state.consts[result.index()] = Some(imm);
        }
        ctx.renamed = true;
        ctx.changed = true;
        return Step::SkipSync;
    }
    Step::Keep
}

fn handle_write(
    func: &mut Function,
    ctx: &mut ScanCtx<'_, '_, '_>,
    (inst, loc, ty, value): (Inst, MemLoc, Type, Value),
) -> Step {
    if is_silent_store(ctx.known, loc, ty, value, ctx.state.consts) {
        func.insts[inst.index()].op = Op::Trap;
        ctx.changed = true;
        return Step::Skip;
    }
    ctx.clobbers_seen.push(BlockClobber::Loc(loc));
    if ctx.state.post_inline {
        ctx.slices
            .prune_on_write(func, ctx.state.prov, loc, Some(inst));
        ctx.slices
            .save_split_slices(ctx.known, ctx.state.subst, loc);
    }
    write_preserving(
        func,
        ctx.known,
        ctx.bases,
        loc,
        ty,
        value,
        ctx.state.prov,
        Some(inst),
    );
    record_base(ctx.bases, loc);
    let track = ctx.state.post_inline
        || !matches!(loc.root, MemRoot::Slot(s) if !ctx.state.prov.slot_unescaped(s));
    if track && !ctx.state.post_inline {
        ctx.changed |= drop_covered_stores(func, ctx.bufs, ctx.guard, (loc, false));
    }
    let mut push_loc = loc;
    if let Some(wide_loc) = try_coalesce_store(
        func,
        ctx.state,
        ctx.known,
        ctx.bases,
        ctx.bufs,
        (inst, loc, ty, value, track),
    ) {
        push_loc = wide_loc;
        ctx.changed = true;
    }
    if track {
        if ctx.state.post_inline {
            ctx.changed |= drop_covered_stores(func, ctx.bufs, ctx.guard, (push_loc, true));
        }
        ctx.bufs
            .pending_stores
            .push((push_loc, ctx.bufs.kept.len()));
    }
    Step::Keep
}

type CopyRemaps = (MemLoc, Vec<(i64, Type, Value)>);

fn prepare_copy_move(
    func: &Function,
    state: &State<'_>,
    known: &KnownLoads,
    clobbers_seen: &mut Vec<BlockClobber>,
    inst: Inst,
) -> Option<(u32, MemLoc, Op, Option<CopyRemaps>)> {
    let (dst, src, bytes) = copy_len(&func.insts[inst.index()].op, state.consts)?;
    let dst_loc = addr_loc(state.prov, state.subst, dst, 0, bytes);
    let mut copy_remaps = None;
    if bytes > 0 {
        clobbers_seen.push(BlockClobber::Loc(dst_loc));
        let src_loc = addr_loc(state.prov, state.subst, src, 0, bytes);
        if !may_alias(dst_loc, src_loc, state.prov)
            && let Some(src_end) = src_loc.offset.checked_add(i64::from(bytes))
        {
            let copied = known
                .entries_for(src_loc.base)
                .iter()
                .filter_map(|&(off, kty, kval)| {
                    (off >= src_loc.offset
                        && off
                            .checked_add(i64::from(kty.bytes()))
                            .is_some_and(|e| e <= src_end))
                    .then(|| {
                        dst_loc
                            .offset
                            .checked_add(off - src_loc.offset)
                            .map(|d_off| (d_off, kty, find(state.subst, kval)))
                    })
                    .flatten()
                })
                .collect::<Vec<_>>();
            if !copied.is_empty() {
                copy_remaps = Some((dst_loc, copied));
            }
        }
    }
    let mov = Op::Move(find(state.subst, dst), find(state.subst, src), bytes);
    Some((bytes, dst_loc, mov, copy_remaps))
}

fn apply_copy_remaps(
    known: &mut KnownLoads,
    bases: &mut Vec<(MemBase, MemRoot)>,
    bytes: u32,
    copy_remaps: Option<CopyRemaps>,
) {
    let Some((dst_loc, copied)) = copy_remaps else {
        return;
    };
    let Some(dst_end) = dst_loc.offset.checked_add(i64::from(bytes)) else {
        return;
    };
    let clear = known.entries_for(dst_loc.base).iter().all(|&(o, t, _)| {
        o.checked_add(i64::from(t.bytes()))
            .is_some_and(|e| e <= dst_loc.offset)
            || o >= dst_end
    });
    if clear {
        for (dst_off, kty, kval) in copied {
            known.remember(
                MemLoc::new(dst_loc.root, dst_loc.base, dst_off, kty.bytes()),
                kty,
                kval,
            );
        }
        record_base(bases, dst_loc);
    }
}

fn retain_stores_after_unknown(
    func: &Function,
    state: &State<'_>,
    bufs: &mut ScanBuf,
    inst: Inst,
    move_op: Option<&Op>,
) {
    let cop = move_op.unwrap_or(&func.insts[inst.index()].op);
    let keep_fn = |ploc: MemLoc| match ploc.root {
        MemRoot::Slot(s) => state.prov.slot_unescaped(s),
        MemRoot::CallerLoc => !matches!(cop, Op::Call { .. } | Op::TryCall { .. }),
        MemRoot::NoAliasParam(p) => {
            !call_passes_noalias(cop, state.prov, p)
                && !clobbers_loc(func, (cop, Some(inst)), ploc, state.prov)
                && state.prov.noalias_unescaped(p)
        }
        _ => false,
    };
    bufs.pending_stores.retain(|&(ploc, _)| keep_fn(ploc));
    bufs.const_stores.retain(|cs| keep_fn(cs.loc));
}

fn handle_unknown(func: &mut Function, ctx: &mut ScanCtx<'_, '_, '_>, inst: Inst) -> Step {
    if let Some((wloc, ..)) = try_fold_private_rmw(
        func,
        (ctx.join.cfg, ctx.state.prov, ctx.block, inst),
        (ctx.state.consts, ctx.state.subst),
        (ctx.known, &mut ctx.bufs.kept),
    ) {
        ctx.clobbers_seen.push(BlockClobber::Loc(wloc));
        record_base(ctx.bases, wloc);
        if ctx.state.post_inline {
            ctx.slices
                .prune_on_write(func, ctx.state.prov, wloc, Some(inst));
            drop_covered_stores(func, ctx.bufs, ctx.guard, (wloc, true));
        }
        ctx.bufs
            .pending_stores
            .retain(|&(ploc, _)| !may_alias_at(wloc, ploc, ctx.state.prov, Some(inst)));
        ctx.bufs
            .const_stores
            .retain(|cs| !may_alias_at(wloc, cs.loc, ctx.state.prov, Some(inst)));
        ctx.changed = true;
        return Step::SkipSync;
    }
    if ctx.state.post_inline
        && let Some(op) = copy_len(&func.insts[inst.index()].op, ctx.state.consts)
        && let Some(stores) = try_scalarize_slot_move(
            func,
            (ctx.join.cfg, ctx.state.prov, ctx.block, inst),
            op,
            ctx.state.subst,
            ctx.known,
        )
    {
        for (wloc, store) in stores {
            ctx.clobbers_seen.push(BlockClobber::Loc(wloc));
            record_base(ctx.bases, wloc);
            ctx.slices
                .prune_on_write(func, ctx.state.prov, wloc, Some(inst));
            ctx.bufs
                .const_stores
                .retain(|cs| !may_alias_at(wloc, cs.loc, ctx.state.prov, Some(inst)));
            ctx.bufs.pending_stores.push((wloc, ctx.bufs.kept.len()));
            ctx.bufs.kept.push(Some(store));
        }
        ctx.changed = true;
        return Step::SkipSync;
    }
    let prepared = prepare_copy_move(func, ctx.state, ctx.known, ctx.clobbers_seen, inst);
    if let Some((0, ..)) = prepared {
        return Step::Keep;
    }
    let (canon_move, copy_remaps) = match prepared {
        Some((bytes, dst_loc, mov, remaps)) => (Some((bytes, dst_loc, mov)), remaps),
        None => (None, None),
    };
    let clobber_op = match canon_move {
        Some((_, _, ref m)) => m,
        None => {
            ctx.clobbers_seen.push(match func.insts[inst.index()].op {
                Op::Copy(dst, _, _) => BlockClobber::CopyLoc(addr_loc(
                    ctx.state.prov,
                    ctx.state.subst,
                    dst,
                    0,
                    u32::MAX,
                )),
                _ => BlockClobber::Op(inst),
            });
            &func.insts[inst.index()].op
        }
    };
    if ctx.state.post_inline {
        ctx.slices.on_unknown(func, clobber_op, ctx.state.prov);
    }
    invalidate_preserving(
        func,
        ctx.known,
        ctx.bases,
        (clobber_op, Some(inst)),
        ctx.state.prov,
    );
    if let Some((bytes, dst_loc, _)) = canon_move.as_ref() {
        if ctx.state.post_inline && *bytes > 0 {
            ctx.changed |= drop_covered_stores(func, ctx.bufs, ctx.guard, (*dst_loc, true));
        }
        apply_copy_remaps(ctx.known, ctx.bases, *bytes, copy_remaps);
    }
    retain_stores_after_unknown(
        func,
        ctx.state,
        ctx.bufs,
        inst,
        canon_move.as_ref().map(|(_, _, m)| m),
    );
    Step::Keep
}

pub(super) fn scan(
    func: &mut Function,
    block: Block,
    state: &mut State<'_>,
    known: &mut KnownLoads,
    bases: &mut Vec<(MemBase, MemRoot)>,
    join: &mut JoinCtx<'_>,
    (bufs, clobbers_seen, guard): (&mut ScanBuf, &mut Vec<BlockClobber>, &mut FarCseGuard),
) -> (bool, bool) {
    let mut insts = std::mem::take(&mut func.blocks[block.index()].insts);
    bufs.kept.clear();
    bufs.pending_stores.clear();
    bufs.const_stores.clear();
    clobbers_seen.clear();
    guard.reset(insts.len(), func.types.len());
    let mut slices = SubwordSlices::new();
    let mut ctx = ScanCtx {
        block,
        state,
        known,
        bases,
        join,
        bufs,
        clobbers_seen,
        guard,
        slices: &mut slices,
        changed: false,
        renamed: false,
    };
    for &inst in &insts {
        let kept_before = ctx.bufs.kept.len();
        if is_pure_scan_op(&func.insts[inst.index()].op) {
            ctx.bufs.kept.push(Some(inst));
            ctx.guard
                .sync_kept(func, ctx.state.subst, &ctx.bufs.kept[kept_before..]);
            continue;
        }
        ctx.changed |= pre_scan_op(func, ctx.state, ctx.bufs, ctx.guard, inst);
        let step = match effect(func, ctx.state, inst) {
            Effect::Read(loc, ty, result) => handle_read(func, &mut ctx, (inst, loc, ty, result)),
            Effect::Write(loc, ty, value) => handle_write(func, &mut ctx, (inst, loc, ty, value)),
            Effect::Unknown => handle_unknown(func, &mut ctx, inst),
            Effect::None => Step::Keep,
        };
        match step {
            Step::Keep => {
                ctx.bufs.kept.push(Some(inst));
                ctx.guard
                    .sync_kept(func, ctx.state.subst, &ctx.bufs.kept[kept_before..]);
            }
            Step::SkipSync => {
                ctx.guard
                    .sync_kept(func, ctx.state.subst, &ctx.bufs.kept[kept_before..]);
            }
            Step::Skip => {}
        }
    }
    let (changed, renamed) = (ctx.changed, ctx.renamed);
    ctx.bases
        .retain(|&(base, _)| !ctx.known.entries_for(base).is_empty());
    if changed {
        insts.clear();
        insts.extend(ctx.bufs.kept.drain(..).flatten());
    }
    func.blocks[block.index()].insts = insts;
    (changed, renamed)
}
