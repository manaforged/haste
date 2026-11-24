use crate::ir::{Function, Inst, Op, Value};

use super::{count_operand_uses, first_use_in_block, uses_operand};

enum LoadGroup {
    One(LoadItem),
    Pair(LoadItem, LoadItem),
}

#[derive(Clone, Copy)]
struct LoadItem {
    pos: usize,
    inst: Inst,
    ptr: Value,
    off: i32,
    size: i32,
    first_use: usize,
}

impl LoadGroup {
    fn first_def(&self) -> usize {
        match self {
            Self::One(a) => a.pos,
            Self::Pair(a, b) => a.pos.min(b.pos),
        }
    }

    fn last_def(&self) -> usize {
        match self {
            Self::One(a) => a.pos,
            Self::Pair(a, b) => a.pos.max(b.pos),
        }
    }

    fn first_use(&self) -> usize {
        match self {
            Self::One(a) => a.first_use,
            Self::Pair(a, b) => a.first_use.min(b.first_use),
        }
    }

    fn count(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Pair(..) => 2,
        }
    }

    fn push_insts(&self, out: &mut Vec<(Inst, usize)>) {
        match *self {
            Self::One(a) => out.push((a.inst, a.first_use)),
            Self::Pair(a, b) => {
                let u = a.first_use.min(b.first_use);
                out.push((a.inst, u));
                out.push((b.inst, u));
            }
        }
    }
}

fn has_mem_or_trap(op: &Op) -> bool {
    crate::mem::is_hard_barrier(op)
        || matches!(
            op,
            Op::Store(..) | Op::StackStore(..) | Op::Copy(..) | Op::Move(..)
        )
}

fn barrier_prefix(func: &Function, insts: &[Inst]) -> Vec<u32> {
    let mut pre = Vec::with_capacity(insts.len() + 1);
    pre.push(0);
    let mut acc = 0u32;
    for &inst in insts {
        acc += u32::from(has_mem_or_trap(&func.insts[inst.index()].op));
        pre.push(acc);
    }
    pre
}

fn collect_loads(func: &Function, insts: &[Inst]) -> Vec<LoadItem> {
    let mut out = Vec::new();
    for (pos, &inst) in insts.iter().enumerate() {
        let data = &func.insts[inst.index()];
        let (Op::Load(ptr, off), Some(res)) = (data.op.clone(), data.result) else {
            continue;
        };
        let Some(first_use) = first_use_in_block(func, insts, pos, res) else {
            continue;
        };
        out.push(LoadItem {
            pos,
            inst,
            ptr,
            off,
            size: func.ty(res).bytes() as i32,
            first_use,
        });
    }
    out
}

fn group_ptr_loads(mut items: Vec<LoadItem>) -> Vec<LoadGroup> {
    items.sort_by_key(|l| (l.off, l.pos));
    let mut groups = Vec::new();
    let mut i = 0;
    while i < items.len() {
        let a = items[i];
        if i + 1 < items.len() {
            let b = items[i + 1];
            if a.size == b.size
                && matches!(a.size, 4 | 8)
                && a.off.checked_add(a.size) == Some(b.off)
            {
                groups.push(LoadGroup::Pair(a, b));
                i += 2;
                continue;
            }
        }
        groups.push(LoadGroup::One(a));
        i += 1;
    }
    groups
}

fn ptr_live_at_use(
    func: &Function,
    insts: &[Inst],
    (ptr, first_use): (Value, usize),
    uses: &[u32],
    moving: &[bool],
) -> bool {
    let block_uses: u32 = insts
        .iter()
        .map(|&i| count_operand_uses(func, i, ptr))
        .sum();
    if uses.get(ptr.index()).copied().unwrap_or(0) > block_uses {
        return true;
    }
    insts[first_use..]
        .iter()
        .any(|&i| !moving[i.index()] && uses_operand(func, i, ptr))
}

fn select_ptr_moves(
    func: &Function,
    insts: &[Inst],
    barriers: &[u32],
    uses: &[u32],
    items: Vec<LoadItem>,
) -> Vec<(Inst, usize)> {
    let ptr = items[0].ptr;
    let groups = group_ptr_loads(items);
    let eligible: Vec<LoadGroup> = groups
        .into_iter()
        .filter(|g| {
            let fu = g.first_use();
            fu > g.last_def() + 16 && barriers[fu] - barriers[g.first_def() + 1] == 0
        })
        .collect();
    let cand_count: usize = eligible.iter().map(LoadGroup::count).sum();
    let mut moving = vec![false; func.insts.len()];
    let mut tmp = Vec::new();
    for g in &eligible {
        g.push_insts(&mut tmp);
    }
    for &(inst, _) in &tmp {
        moving[inst.index()] = true;
    }
    let mut out = Vec::new();
    for g in eligible {
        if cand_count >= 3 || ptr_live_at_use(func, insts, (ptr, g.first_use()), uses, &moving) {
            g.push_insts(&mut out);
        }
    }
    out
}

fn apply_load_moves(func: &mut Function, bi: usize, moves: Vec<(Inst, usize)>) {
    if moves.is_empty() {
        return;
    }
    let orig = func.blocks[bi].insts.clone();
    let mut moving = vec![false; func.insts.len()];
    for &(inst, _) in &moves {
        moving[inst.index()] = true;
    }
    let mut emitted = vec![false; func.insts.len()];
    let mut before: Vec<Vec<Inst>> = vec![Vec::new(); func.insts.len()];
    for (inst, first_use) in moves {
        let anchor = orig[first_use];
        if !moving[anchor.index()] {
            emitted[inst.index()] = true;
            before[anchor.index()].push(inst);
        }
    }
    let mut rebuilt = Vec::with_capacity(orig.len());
    for inst in orig {
        if emitted[inst.index()] {
            continue;
        }
        rebuilt.extend(before[inst.index()].iter().copied());
        rebuilt.push(inst);
    }
    func.blocks[bi].insts = rebuilt;
}

pub(super) fn sink_loads_block(func: &mut Function, uses: &[u32], bi: usize) {
    let insts = func.blocks[bi].insts.clone();
    if insts.len() < 48 {
        return;
    }
    let loads = collect_loads(func, &insts);
    if loads.len() < 4 {
        return;
    }
    let barriers = barrier_prefix(func, &insts);
    let mut by_ptr: Vec<(Value, Vec<LoadItem>)> = Vec::new();
    for item in loads {
        if let Some((_, list)) = by_ptr.iter_mut().find(|(p, _)| *p == item.ptr) {
            list.push(item);
        } else {
            by_ptr.push((item.ptr, vec![item]));
        }
    }
    let mut moves = Vec::new();
    for (_, list) in by_ptr {
        if list.len() >= 4 {
            moves.extend(select_ptr_moves(func, &insts, &barriers, uses, list));
        }
    }
    apply_load_moves(func, bi, moves);
}
