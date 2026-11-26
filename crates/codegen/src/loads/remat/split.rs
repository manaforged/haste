use super::{FAR_MIN_DIST, MIN_BLOCK_INSTS, MIN_GAP_DEFS, is_scalar_remat_ty};
use crate::ir::{Block, Function, Inst, Op, Type, Value};
use crate::loads::dom::clobbers_loc;
use crate::mem::{Provenance, is_hard_barrier};

const PAIR_MAX_WINDOW: usize = 16;

fn is_mem_write(op: &Op) -> bool {
    matches!(
        op,
        Op::Store(..) | Op::StackStore(..) | Op::Copy(..) | Op::Move(..)
    )
}

struct RawCand {
    def_pos: usize,
    val: Value,
    ptr: Value,
    off: i32,
    ty: Type,
}

#[derive(Clone, Copy)]
struct Plan {
    def_pos: usize,
    orig_val: Value,
    ptr: Value,
    off: i32,
    ty: Type,
    last_near_use: usize,
    first_far_use: usize,
}

struct Insertion {
    pos: usize,
    plans: Vec<Plan>,
}

struct BlockScan {
    def_prefix: Vec<usize>,
    hard_barriers: Vec<usize>,
    writes: Vec<usize>,
    in_block_uses: Vec<u32>,
    last_use: Vec<Option<usize>>,
    def_in_block: Vec<bool>,
    raw_cands: Vec<RawCand>,
    cand_uses: Vec<Vec<usize>>,
}

struct ScanCtx<'a> {
    func: &'a Function,
    prov: &'a Provenance,
    insts: &'a [Inst],
    all_uses: &'a [u32],
    scan: &'a BlockScan,
}

fn scan_block(func: &Function, insts: &[Inst]) -> BlockScan {
    let n_vals = func.types.len();
    let len = insts.len();
    let mut def_prefix = Vec::with_capacity(len + 1);
    let mut hard_barriers = Vec::with_capacity(len + 1);
    def_prefix.push(0);
    hard_barriers.push(0);
    let mut writes = Vec::new();
    let mut in_block_uses = vec![0_u32; n_vals];
    let mut last_use = vec![None; n_vals];
    let mut def_in_block = vec![false; n_vals];
    let mut cand_idx = vec![usize::MAX; n_vals];
    let mut raw_cands = Vec::new();
    let mut cand_uses: Vec<Vec<usize>> = Vec::new();

    for (pos, &inst) in insts.iter().enumerate() {
        let data = &func.insts[inst.index()];
        def_prefix.push(def_prefix[pos] + usize::from(data.result.is_some()));
        hard_barriers.push(hard_barriers[pos] + usize::from(is_hard_barrier(&data.op)));
        if is_mem_write(&data.op) {
            writes.push(pos);
        }
        data.op.for_operands(|opnd| {
            let idx = opnd.index();
            in_block_uses[idx] += 1;
            last_use[idx] = Some(pos);
            let ci = cand_idx[idx];
            if ci != usize::MAX && cand_uses[ci].last() != Some(&pos) {
                cand_uses[ci].push(pos);
            }
        });
        if let Some(res) = data.result {
            def_in_block[res.index()] = true;
            if let Op::Load(ptr, off) = data.op {
                let ty = func.ty(res);
                if is_scalar_remat_ty(ty) {
                    cand_idx[res.index()] = raw_cands.len();
                    raw_cands.push(RawCand {
                        def_pos: pos,
                        val: res,
                        ptr,
                        off,
                        ty,
                    });
                    cand_uses.push(Vec::new());
                }
            }
        }
    }

    BlockScan {
        def_prefix,
        hard_barriers,
        writes,
        in_block_uses,
        last_use,
        def_in_block,
        raw_cands,
        cand_uses,
    }
}

fn has_clobber_between(ctx: &ScanCtx<'_>, raw: &RawCand, end_pos: usize) -> bool {
    let w_lo = ctx.scan.writes.partition_point(|&w| w <= raw.def_pos);
    let w_hi = ctx.scan.writes.partition_point(|&w| w < end_pos);
    if w_lo >= w_hi {
        return false;
    }
    let read_loc = ctx.prov.loc(raw.ptr, i64::from(raw.off), raw.ty.bytes());
    ctx.scan.writes[w_lo..w_hi].iter().any(|&w| {
        clobbers_loc(
            ctx.func,
            (&ctx.func.insts[ctx.insts[w].index()].op, Some(ctx.insts[w])),
            read_loc,
            ctx.prov,
        )
    })
}

fn ptr_live_at_or_past(ctx: &ScanCtx<'_>, ptr: Value, pos: usize) -> bool {
    let idx = ptr.index();
    ctx.scan.last_use[idx].is_some_and(|u| u >= pos)
        || (ctx.scan.def_in_block[idx] && ctx.scan.in_block_uses[idx] < ctx.all_uses[idx])
}

fn first_chasm(uses: &[usize], def_prefix: &[usize]) -> Option<(usize, usize)> {
    uses.windows(2).find_map(|pair| {
        let (last_near, first_far) = (pair[0], pair[1]);
        (first_far >= last_near + FAR_MIN_DIST
            && def_prefix[first_far] - def_prefix[last_near + 1] >= MIN_GAP_DEFS)
            .then_some((last_near, first_far))
    })
}

fn validate_cand(ctx: &ScanCtx<'_>, raw: &RawCand, uses: &[usize]) -> Option<Plan> {
    if ctx.scan.in_block_uses[raw.val.index()] != ctx.all_uses[raw.val.index()] {
        return None;
    }
    let (last_near_use, first_far_use) = first_chasm(uses, &ctx.scan.def_prefix)?;
    if ctx.scan.hard_barriers[first_far_use] != ctx.scan.hard_barriers[raw.def_pos + 1] {
        return None;
    }
    if has_clobber_between(ctx, raw, first_far_use) {
        return None;
    }
    Some(Plan {
        def_pos: raw.def_pos,
        orig_val: raw.val,
        ptr: raw.ptr,
        off: raw.off,
        ty: raw.ty,
        last_near_use,
        first_far_use,
    })
}

fn can_pair(a: &Plan, b: &Plan, def_prefix: &[usize]) -> bool {
    if a.ptr != b.ptr
        || a.ty.bytes() != b.ty.bytes()
        || a.ty.is_float() != b.ty.is_float()
        || !matches!(a.ty.bytes(), 4 | 8)
    {
        return false;
    }
    if i64::from(a.off).checked_add(i64::from(a.ty.bytes())) != Some(i64::from(b.off)) {
        return false;
    }
    if a.first_far_use.abs_diff(b.first_far_use) > PAIR_MAX_WINDOW {
        return false;
    }
    let pos = a.first_far_use.min(b.first_far_use);
    let later = if a.first_far_use > pos { a } else { b };
    pos >= later.last_near_use + FAR_MIN_DIST
        && def_prefix[pos] - def_prefix[later.last_near_use + 1] >= MIN_GAP_DEFS
}

fn find_pair_partner(
    plans: &[Plan],
    paired: &[bool],
    i: usize,
    def_prefix: &[usize],
) -> Option<usize> {
    let a = &plans[i];
    let max_off = i64::from(a.off).checked_add(i64::from(a.ty.bytes()))?;
    for j in (i + 1)..plans.len() {
        if plans[j].ptr != a.ptr || i64::from(plans[j].off) > max_off {
            break;
        }
        if !paired[j] && can_pair(a, &plans[j], def_prefix) {
            return Some(j);
        }
    }
    None
}

fn group_insertions(mut plans: Vec<Plan>, def_prefix: &[usize]) -> Vec<Insertion> {
    plans.sort_by_key(|p| (p.ptr.0, p.off, p.first_far_use));
    let mut paired = vec![false; plans.len()];
    let mut insertions = Vec::with_capacity(plans.len());
    for i in 0..plans.len() {
        if paired[i] {
            continue;
        }
        paired[i] = true;
        if let Some(j) = find_pair_partner(&plans, &paired, i, def_prefix) {
            paired[j] = true;
            let pos = plans[i].first_far_use.min(plans[j].first_far_use);
            insertions.push(Insertion {
                pos,
                plans: vec![plans[i], plans[j]],
            });
        } else {
            insertions.push(Insertion {
                pos: plans[i].first_far_use,
                plans: vec![plans[i]],
            });
        }
    }
    insertions.sort_by_key(|ins| (ins.pos, ins.plans[0].def_pos));
    insertions
}

fn emit_group(
    func: &mut Function,
    group: &[Plan],
    out: &mut Vec<Inst>,
    remap: &mut [Option<Value>],
) {
    for plan in group {
        let ptr = remap
            .get(plan.ptr.index())
            .copied()
            .flatten()
            .unwrap_or(plan.ptr);
        let new_val = func.value(plan.ty);
        out.push(func.add(Op::Load(ptr, plan.off), Some(new_val)));
        remap[plan.orig_val.index()] = Some(new_val);
    }
}

fn apply_insertions(func: &mut Function, block: Block, insertions: Vec<Insertion>) {
    let extra: usize = insertions.iter().map(|ins| ins.plans.len()).sum();
    let insts = std::mem::take(&mut func.blocks[block.index()].insts);
    let mut out = Vec::with_capacity(insts.len() + extra);
    let mut remap: Vec<Option<Value>> = vec![None; func.types.len()];
    let mut any_remap = false;
    let mut ins_idx = 0;

    for (pos, &inst) in insts.iter().enumerate() {
        while ins_idx < insertions.len() && insertions[ins_idx].pos == pos {
            emit_group(func, &insertions[ins_idx].plans, &mut out, &mut remap);
            any_remap = true;
            ins_idx += 1;
        }
        if any_remap {
            func.insts[inst.index()]
                .op
                .map_operands(|v| remap.get(v.index()).copied().flatten().unwrap_or(v));
        }
        out.push(inst);
    }
    func.blocks[block.index()].insts = out;
}

fn plan_survives(ctx: &ScanCtx<'_>, plans: &[Plan], i: usize) -> bool {
    let plan = &plans[i];
    if ptr_live_at_or_past(ctx, plan.ptr, plan.first_far_use) {
        return true;
    }
    plans.iter().enumerate().any(|(j, other)| {
        j != i
            && other.orig_val != plan.orig_val
            && other.ptr == plan.ptr
            && plan.last_near_use.max(other.last_near_use)
                < plan.first_far_use.min(other.first_far_use)
    })
}

fn plan_block(
    func: &Function,
    prov: &Provenance,
    all_uses: &[u32],
    insts: &[Inst],
) -> Vec<Insertion> {
    let scan = scan_block(func, insts);
    if scan.raw_cands.is_empty() {
        return Vec::new();
    }
    let ctx = ScanCtx {
        func,
        prov,
        insts,
        all_uses,
        scan: &scan,
    };
    let mut plans: Vec<Plan> = scan
        .raw_cands
        .iter()
        .zip(&scan.cand_uses)
        .filter_map(|(raw, uses)| validate_cand(&ctx, raw, uses))
        .collect();
    let keep: Vec<bool> = (0..plans.len())
        .map(|i| plan_survives(&ctx, &plans, i))
        .collect();
    let mut idx = 0_usize;
    plans.retain(|_| {
        let ok = keep[idx];
        idx += 1;
        ok
    });
    if plans.is_empty() {
        return Vec::new();
    }
    group_insertions(plans, &scan.def_prefix)
}

fn split_long_loads_round(func: &mut Function, prov: &Provenance) -> bool {
    let all_uses = func.use_counts();
    let mut changed = false;
    for b_idx in 0..func.blocks.len() {
        if func.blocks[b_idx].insts.len() < MIN_BLOCK_INSTS {
            continue;
        }
        let insertions = plan_block(func, prov, &all_uses, &func.blocks[b_idx].insts);
        if !insertions.is_empty() {
            apply_insertions(func, Block(b_idx as u32), insertions);
            changed = true;
        }
    }
    changed
}

/// Splits long same-block scalar load live ranges across barrier-free instruction chasms by rematerializing before distant uses.
pub(in crate::loads) fn split_long_loads(func: &mut Function, prov: &Provenance) -> bool {
    const MAX_SPLIT_ROUNDS: usize = 4;
    if !func.blocks.iter().any(|b| b.insts.len() >= MIN_BLOCK_INSTS) {
        return false;
    }
    let mut changed = false;
    for _ in 0..MAX_SPLIT_ROUNDS {
        if !split_long_loads_round(func, prov) {
            break;
        }
        changed = true;
    }
    changed
}
