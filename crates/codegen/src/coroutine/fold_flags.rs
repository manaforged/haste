use rustc_hash::FxHashMap as HashMap;

use crate::fold::{constant, extend, holds};
use crate::ir::{Block, Convert, Function, Op, StackSlot, Value};

use super::Coroutine;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotVal {
    Uninit,
    Const(i64),
    Varying,
}

impl SlotVal {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Uninit, x) | (x, Self::Uninit) => x,
            (Self::Const(a), Self::Const(b)) if a == b => Self::Const(a),
            _ => Self::Varying,
        }
    }
}

fn find_pin_slot(func: &Function, spec: &Coroutine, old_entry: Block) -> Option<(Option<StackSlot>, Vec<bool>)> {
    let entry_obj = *func.blocks[old_entry.index()].params.get(spec.object)?;
    let mut obj_params = vec![false; func.types.len()];
    obj_params[entry_obj.index()] = true;
    for &resume in &spec.resumes {
        let param = *func.blocks[resume.index()].params.get(spec.object)?;
        obj_params[param.index()] = true;
    }
    let mut pin_slot = None;
    for &inst in &func.blocks[old_entry.index()].insts {
        if let Op::StackStore(v, slot, 0) = func.insts[inst.index()].op
            && v == entry_obj
        {
            pin_slot = Some(slot);
            break;
        }
    }
    if let Some(slot) = pin_slot {
        for block in &func.blocks {
            for &inst in &block.insts {
                match func.insts[inst.index()].op {
                    Op::StackAddr(s, _) if s == slot => {
                        pin_slot = None;
                        break;
                    }
                    Op::StackStore(v, s, off) if s == slot && (off != 0 || !obj_params[v.index()]) => {
                        pin_slot = None;
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
    let mut is_obj = obj_params;
    if let Some(slot) = pin_slot {
        for block in &func.blocks {
            for &inst in &block.insts {
                let data = &func.insts[inst.index()];
                if let (Op::StackLoad(s, 0), Some(res)) = (&data.op, data.result)
                    && *s == slot
                {
                    is_obj[res.index()] = true;
                }
            }
        }
    }
    Some((pin_slot, is_obj))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StateFlow {
    Uninit,
    Done,
    Suspend(usize),
    Mixed,
}

impl StateFlow {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Uninit, x) | (x, Self::Uninit) => x,
            (a, b) if a == b => a,
            _ => Self::Mixed,
        }
    }
}

fn find_suspend_edges(func: &Function, spec: &Coroutine, old_entry: Block, consts: &[Option<i64>]) -> Option<Vec<Option<usize>>> {
    let (_, is_obj) = find_pin_slot(func, spec, old_entry)?;
    let n = func.blocks.len();
    let mut blk_set: Vec<Option<StateFlow>> = vec![None; n];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            if let Op::Store(val, addr, off) = func.insts[inst.index()].op
                && is_obj[addr.index()]
                && off == spec.header.0
            {
                let state = consts[val.index()]?;
                let flow = if state >= Coroutine::FIRST_SUSPEND {
                    let k = usize::try_from(state - Coroutine::FIRST_SUSPEND).ok()?;
                    if k >= spec.resumes.len() {
                        return None;
                    }
                    StateFlow::Suspend(k)
                } else {
                    StateFlow::Done
                };
                blk_set[b_idx] = Some(flow);
            }
        }
    }
    let mut in_flow = vec![StateFlow::Uninit; n];
    let mut out_flow = vec![StateFlow::Uninit; n];
    in_flow[old_entry.index()] = StateFlow::Done;
    for &resume in &spec.resumes {
        in_flow[resume.index()] = StateFlow::Done;
    }
    loop {
        let mut changed = false;
        for b_idx in 0..n {
            if in_flow[b_idx] == StateFlow::Uninit {
                continue;
            }
            let out = blk_set[b_idx].unwrap_or(in_flow[b_idx]);
            if out != out_flow[b_idx] {
                out_flow[b_idx] = out;
                changed = true;
            }
            for succ in func.successors(Block(b_idx as u32)) {
                let next = in_flow[succ.index()].join(out);
                if next != in_flow[succ.index()] {
                    in_flow[succ.index()] = next;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut ret_suspend = vec![None; n];
    for b_idx in 0..n {
        let blk = Block(b_idx as u32);
        if matches!(func.last(blk), Some(Op::Return(..))) && !spec.done.contains(&blk) {
            if out_flow[b_idx] == StateFlow::Uninit {
                continue;
            }
            match out_flow[b_idx] {
                StateFlow::Suspend(k) => ret_suspend[b_idx] = Some(k),
                _ => return None,
            }
        }
    }
    Some(ret_suspend)
}

fn dropping_blocks(func: &Function, spec: &Coroutine) -> Vec<bool> {
    let mut dropping = vec![false; func.blocks.len()];
    let mut stack = Vec::new();
    for &resume in &spec.resumes {
        if let Some(&Op::Brif(_, _, _, drop_arm, _)) = func.last(resume) {
            stack.push(drop_arm);
        }
    }
    while let Some(b) = stack.pop() {
        if !std::mem::replace(&mut dropping[b.index()], true) {
            stack.extend(func.successors(b));
        }
    }
    dropping
}

fn collect_scalar_slots(func: &Function) -> (Vec<u32>, usize) {
    let nslots = func.slots.len();
    let mut ok: Vec<bool> = (0..nslots)
        .map(|s| {
            let size = func.slots[s].size;
            size > 0 && size <= 8 && !func.is_unwind_slot(StackSlot(s as u32))
        })
        .collect();
    let mut has_load = vec![false; nslots];
    let mut has_store = vec![false; nslots];
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            match data.op {
                Op::StackAddr(s, _) => ok[s.index()] = false,
                Op::StackStore(v, s, off) => {
                    let ty = func.ty(v);
                    if off != 0 || !ty.is_small_int() || ty.bytes() != func.slots[s.index()].size {
                        ok[s.index()] = false;
                    } else {
                        has_store[s.index()] = true;
                    }
                }
                Op::StackLoad(s, off) => {
                    let good = data.result.is_some_and(|r| {
                        let ty = func.ty(r);
                        ty.is_small_int() && ty.bytes() == func.slots[s.index()].size
                    });
                    if off != 0 || !good {
                        ok[s.index()] = false;
                    } else {
                        has_load[s.index()] = true;
                    }
                }
                _ => {}
            }
        }
    }
    let mut dense = vec![u32::MAX; nslots];
    let mut count = 0usize;
    for s in 0..nslots {
        if ok[s] && has_load[s] && has_store[s] {
            dense[s] = count as u32;
            count += 1;
        }
    }
    (dense, count)
}

fn eval_op(func: &Function, op: &Op, res_ty: crate::ir::Type, lookup: impl Fn(Value) -> Option<i64>) -> Option<i64> {
    match *op {
        Op::Const(k) => Some(k),
        Op::Binary(bin, a, b) => constant(bin, lookup(a)?, lookup(b)?).map(|k| extend(k, res_ty, false)),
        Op::Icmp(cc, a, b) => Some(i64::from(holds(cc, func.ty(a), lookup(a)?, lookup(b)?))),
        Op::Convert(Convert::Uextend, a) => Some(extend(lookup(a)?, func.ty(a), false)),
        Op::Convert(Convert::Sextend, a) => Some(extend(extend(lookup(a)?, func.ty(a), true), res_ty, false)),
        Op::Convert(Convert::Ireduce, a) => Some(extend(lookup(a)?, res_ty, false)),
        _ => None,
    }
}

/// Folds constant unescaped scalar stack slots (such as drop flags) across coroutine suspend/resume edges before state-machine splitting.
pub(super) fn run(func: &mut Function, spec: &Coroutine, old_entry: Block) {
    if spec.resumes.is_empty() || func.slots.is_empty() {
        return;
    }
    let mut global_consts = vec![None; func.types.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let (Op::Const(k), Some(res)) = (&data.op, data.result) {
                global_consts[res.index()] = Some(*k);
            }
        }
    }
    let Some(ret_suspend) = find_suspend_edges(func, spec, old_entry, &global_consts) else {
        return;
    };
    let (dense, tracked) = collect_scalar_slots(func);
    if tracked == 0 {
        return;
    }
    let n = func.blocks.len();
    let mut in_vals = vec![vec![SlotVal::Uninit; tracked]; n];
    let mut reached = vec![false; n];
    reached[old_entry.index()] = true;
    in_vals[old_entry.index()].fill(SlotVal::Varying);
    let mut work = vec![old_entry];
    let mut in_work = vec![false; n];
    in_work[old_entry.index()] = true;
    let mut state = vec![SlotVal::Uninit; tracked];
    let mut local_consts: HashMap<Value, i64> = HashMap::default();
    while let Some(b) = work.pop() {
        let bi = b.index();
        in_work[bi] = false;
        state.copy_from_slice(&in_vals[bi]);
        local_consts.clear();
        for &inst in &func.blocks[bi].insts {
            let data = &func.insts[inst.index()];
            match data.op {
                Op::StackStore(v, s, 0) if dense[s.index()] != u32::MAX => {
                    let t = dense[s.index()] as usize;
                    let c = local_consts.get(&v).copied().or_else(|| global_consts[v.index()]);
                    state[t] = c.map_or(SlotVal::Varying, SlotVal::Const);
                }
                Op::StackLoad(s, 0) if dense[s.index()] != u32::MAX => {
                    if let (Some(res), SlotVal::Const(k)) = (data.result, state[dense[s.index()] as usize]) {
                        local_consts.insert(res, extend(k, func.ty(res), false));
                    }
                }
                _ => {
                    if let Some(res) = data.result
                        && let Some(k) = eval_op(func, &data.op, func.ty(res), |v| local_consts.get(&v).copied().or_else(|| global_consts[v.index()]))
                    {
                        local_consts.insert(res, k);
                    }
                }
            }
        }
        let mut targets: Vec<Block> = Vec::new();
        if let Some(&Op::Brif(cond, t, _, f, _)) = func.last(b)
            && let Some(c) = local_consts.get(&cond).copied().or_else(|| global_consts[cond.index()])
        {
            targets.push(if c != 0 { t } else { f });
        } else {
            targets.extend(func.successors(b));
        }
        if let Some(k) = ret_suspend[bi] {
            targets.push(spec.resumes[k]);
        }
        for succ in targets {
            let si = succ.index();
            let mut changed = !std::mem::replace(&mut reached[si], true);
            for slot in 0..tracked {
                let merged = in_vals[si][slot].join(state[slot]);
                if merged != in_vals[si][slot] {
                    in_vals[si][slot] = merged;
                    changed = true;
                }
            }
            if changed && !in_work[si] {
                in_work[si] = true;
                work.push(succ);
            }
        }
    }
    let dropping = dropping_blocks(func, spec);
    let mut cfg_changed = false;
    for bi in 0..n {
        if !reached[bi] || dropping[bi] {
            continue;
        }
        state.copy_from_slice(&in_vals[bi]);
        local_consts.clear();
        let insts = func.blocks[bi].insts.clone();
        for inst in insts {
            let data = &func.insts[inst.index()];
            match data.op {
                Op::StackStore(v, s, 0) if dense[s.index()] != u32::MAX => {
                    let t = dense[s.index()] as usize;
                    let c = local_consts.get(&v).copied().or_else(|| global_consts[v.index()]);
                    state[t] = c.map_or(SlotVal::Varying, SlotVal::Const);
                }
                Op::StackLoad(s, 0) if dense[s.index()] != u32::MAX => {
                    if let (Some(res), SlotVal::Const(k)) = (data.result, state[dense[s.index()] as usize]) {
                        let k = extend(k, func.ty(res), false);
                        func.insts[inst.index()].op = Op::Const(k);
                        local_consts.insert(res, k);
                    }
                }
                Op::Brif(cond, t, ref t_args, f, ref f_args) => {
                    if let Some(c) = local_consts.get(&cond).copied().or_else(|| global_consts[cond.index()]) {
                        let (dst, args) = if c != 0 { (t, t_args.clone()) } else { (f, f_args.clone()) };
                        func.insts[inst.index()].op = Op::Jump(dst, args);
                        cfg_changed = true;
                    }
                }
                _ => {
                    if let Some(res) = data.result
                        && let Some(k) = eval_op(func, &data.op, func.ty(res), |v| local_consts.get(&v).copied().or_else(|| global_consts[v.index()]))
                    {
                        local_consts.insert(res, k);
                    }
                }
            }
        }
    }
    if cfg_changed {
        let mut active = vec![false; n];
        let mut stack = vec![old_entry];
        stack.extend_from_slice(&spec.resumes);
        while let Some(b) = stack.pop() {
            if !std::mem::replace(&mut active[b.index()], true) {
                stack.extend(func.successors(b));
            }
        }
        for bi in 0..n {
            if !active[bi] {
                for inst in func.blocks[bi].insts.drain(..) {
                    func.insts[inst.index()].op = Op::Trap;
                }
            }
        }
        func.bump_cfg();
    }
}
