use std::borrow::Cow;
use std::cmp::Reverse;

use rustc_hash::FxHashMap;

use crate::cfg::{Cfg, Loop, Loops};
use crate::ir::{ArgumentPurpose, Binary, Block, BlockData, Convert, Domain, Function, Inst, Op, Type, Value};
use crate::live::{BitMatrix, results, set_bits, solve_backward_liveness};

enum TailCase {
    Pure { args: Vec<Value>, callee: Value },
    Accum { op: Binary, other: Value, args: Vec<Value>, callee: Value },
}

fn is_self_callee(func: &Function, def_inst: &[Option<Inst>], callee: Value) -> Option<(Value, Value)> {
    let load_inst = def_inst[callee.index()]?;
    let Op::Load(addr, 0) = func.insts[load_inst.index()].op else { return None };
    let sym_inst = def_inst[addr.index()]?;
    match func.insts[sym_inst.index()].op {
        Op::SlotAddr(sym) | Op::Symbol(sym) if func.self_slots.contains(&sym) => Some((callee, addr)),
        _ => None,
    }
}

fn is_accum_ty(ty: Type) -> bool {
    matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64)
}

fn is_assoc_comm_int(op: Binary) -> bool {
    matches!(op, Binary::Iadd | Binary::Imul | Binary::Band | Binary::Bor | Binary::Bxor)
}

fn identity_of(op: Binary) -> i64 {
    match op {
        Binary::Imul => 1,
        Binary::Band => -1,
        _ => 0,
    }
}

/// The returned values of `term`, unwrapping a pure jump into a pure return block when safe.
fn effective_return<'a>(func: &'a Function, block: usize, term: Inst) -> Option<Cow<'a, [Value]>> {
    match &func.insts[term.index()].op {
        Op::Return(vals) => Some(Cow::Borrowed(vals.as_slice())),
        Op::Jump(target, jump_args) if target.index() != block => {
            let target_data = &func.blocks[target.index()];
            let (&ret_inst, prefix) = target_data.insts.split_last()?;
            if !prefix.is_empty() || target_data.params.len() != jump_args.len() {
                return None;
            }
            let Op::Return(ret_vals) = &func.insts[ret_inst.index()].op else { return None };
            let resolved = ret_vals
                .iter()
                .map(|val| target_data.params.iter().position(|param| param == val).map_or(*val, |pos| jump_args[pos]))
                .collect();
            Some(Cow::Owned(resolved))
        }
        _ => None,
    }
}

fn analyze_block(
    func: &Function,
    block: usize,
    old_params_len: usize,
    def_inst: &[Option<Inst>],
    uses: &[u32],
) -> Option<TailCase> {
    let insts = &func.blocks[block].insts;
    let &term_inst = insts.last()?;
    let ret_vals = effective_return(func, block, term_inst)?;
    if insts.len() >= 2 {
        let call_inst = insts[insts.len() - 2];
        if let Op::Call { sig, callee, args, results } = &func.insts[call_inst.index()].op
            && func.sigs[sig.index()] == func.signature
            && args.len() == old_params_len
            && results.len() == func.signature.returns.len()
            && results.as_slice() == &ret_vals[..]
            && is_self_callee(func, def_inst, *callee).is_some()
        {
            return Some(TailCase::Pure { args: args.clone(), callee: *callee });
        }
    }
    if insts.len() >= 3
        && func.signature.returns.len() == 1
        && let [ret_val] = ret_vals[..]
        && is_accum_ty(func.ty(ret_val))
    {
        let bin_inst = insts[insts.len() - 2];
        let call_inst = insts[insts.len() - 3];
        if func.insts[bin_inst.index()].result == Some(ret_val)
            && let Op::Binary(op, lhs, rhs) = func.insts[bin_inst.index()].op
            && is_assoc_comm_int(op)
            && let Op::Call { sig, callee, args, results } = &func.insts[call_inst.index()].op
            && func.sigs[sig.index()] == func.signature
            && args.len() == old_params_len
            && let [call_res] = results[..]
            && uses[call_res.index()] == 1
            && is_self_callee(func, def_inst, *callee).is_some()
        {
            let other = if lhs == call_res && rhs != call_res {
                rhs
            } else if rhs == call_res && lhs != call_res {
                lhs
            } else {
                return None;
            };
            return Some(TailCase::Accum { op, other, args: args.clone(), callee: *callee });
        }
    }
    None
}

fn kill_dead_value(func: &mut Function, def_inst: &[Option<Inst>], val: Value) -> Option<Value> {
    let inst = def_inst[val.index()]?;
    let next = match func.insts[inst.index()].op {
        Op::Load(addr, 0) => Some(addr),
        Op::SlotAddr(_) | Op::Symbol(_) => None,
        _ => return None,
    };
    let used_elsewhere = func.blocks.iter().flat_map(|b| &b.insts).any(|&i| {
        i != inst && {
            let mut hit = false;
            func.insts[i.index()].op.for_operands(|v| {
                if v == val {
                    hit = true;
                }
            });
            hit
        }
    });
    if used_elsewhere {
        return None;
    }
    for block in &mut func.blocks {
        block.insts.retain(|&i| i != inst);
    }
    func.insts[inst.index()].op = Op::Trap;
    next
}

/// Turns self-recursive tail calls into loops and splits call-crossing invariants into call-free innermost loops.
pub(crate) fn run(func: &mut Function, target: &crate::Target) -> Domain {
    let mut changed = elim_tail_rec(func);
    changed |= split_innermost_loops(func, target);
    changed
}

fn elim_tail_rec(func: &mut Function) -> Domain {
    if func.self_slots.is_empty() || func.coroutine.is_some() {
        return Domain::NONE;
    }
    let Some(old_entry) = func.entry else { return Domain::NONE };
    if func.signature.params.iter().any(|p| p.purpose != ArgumentPurpose::Normal)
        || func.signature.returns.iter().any(|p| p.purpose != ArgumentPurpose::Normal)
    {
        return Domain::NONE;
    }
    for block in &func.blocks {
        if block.insts.last().is_some_and(|&i| func.insts[i.index()].op.targets().any(|t| t == old_entry)) {
            return Domain::NONE;
        }
        for &inst in &block.insts {
            if matches!(func.insts[inst.index()].op, Op::StackAddr(..) | Op::TryCall { .. }) {
                return Domain::NONE;
            }
        }
    }
    let mut def_inst = vec![None; func.types.len()];
    let mut uses = vec![0_u32; func.types.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result {
                def_inst[res.index()] = Some(inst);
            }
            if let Op::Call { results, .. } = &data.op {
                for &res in results {
                    def_inst[res.index()] = Some(inst);
                }
            }
            data.op.for_operands(|v| uses[v.index()] += 1);
        }
    }
    let old_params_len = func.blocks[old_entry.index()].params.len();
    let mut cases = Vec::new();
    let mut accum_op: Option<Binary> = None;
    for block in 0..func.blocks.len() {
        let Some(case) = analyze_block(func, block, old_params_len, &def_inst, &uses) else { continue };
        if let TailCase::Accum { op, .. } = case {
            match accum_op {
                Some(prev) if prev != op => return Domain::NONE,
                None => accum_op = Some(op),
                _ => {}
            }
        }
        cases.push((block, case));
    }
    if cases.is_empty() {
        return Domain::NONE;
    }
    let accum_ret_ty = accum_op.and_then(|_| {
        cases.iter().find_map(|(_, case)| match case {
            TailCase::Accum { other, .. } => Some(func.ty(*other)),
            TailCase::Pure { .. } => None,
        })
    });
    let old_params = std::mem::take(&mut func.blocks[old_entry.index()].params);
    let fresh_params: Vec<Value> = old_params.iter().map(|&p| func.value(func.ty(p))).collect();
    let loop_head = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: old_params, insts: Vec::new() });
    let new_entry = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: fresh_params.clone(), insts: Vec::new() });
    func.entry = Some(new_entry);

    let (acc_param, ret_ty) = if let (Some(op), Some(ret_ty)) = (accum_op, accum_ret_ty) {
        let acc = func.value(ret_ty);
        func.blocks[loop_head.index()].params.push(acc);
        let init_acc = func.value(ret_ty);
        let c_inst = func.add(Op::Const(identity_of(op)), Some(init_acc));
        let mut jump_args = fresh_params;
        jump_args.push(init_acc);
        let j_inst = func.add(Op::Jump(loop_head, jump_args), None);
        func.blocks[new_entry.index()].insts = vec![c_inst, j_inst];
        (Some(acc), Some(ret_ty))
    } else {
        let j_inst = func.add(Op::Jump(loop_head, fresh_params), None);
        func.blocks[new_entry.index()].insts = vec![j_inst];
        (None, None)
    };
    let j_old = func.add(Op::Jump(old_entry, Vec::new()), None);
    func.blocks[loop_head.index()].insts = vec![j_old];

    let orig_blocks = new_entry.index().min(loop_head.index());
    let mut rewritten = vec![false; orig_blocks];
    let mut dead_callees = Vec::new();
    for (b_idx, case) in cases {
        rewritten[b_idx] = true;
        let drop_count = match case {
            TailCase::Pure { .. } => 2,
            TailCase::Accum { .. } => 3,
        };
        let len = func.blocks[b_idx].insts.len();
        for inst in func.blocks[b_idx].insts.drain(len - drop_count..) {
            func.insts[inst.index()].op = Op::Trap;
        }
        match case {
            TailCase::Pure { mut args, callee } => {
                dead_callees.push(callee);
                if let Some(acc) = acc_param {
                    args.push(acc);
                }
                let j = func.add(Op::Jump(loop_head, args), None);
                func.blocks[b_idx].insts.push(j);
            }
            TailCase::Accum { op, other, mut args, callee } => {
                dead_callees.push(callee);
                let (Some(acc), Some(rty)) = (acc_param, ret_ty) else { continue };
                let next_acc = func.value(rty);
                let b_inst = func.add(Op::Binary(op, acc, other), Some(next_acc));
                args.push(next_acc);
                let j = func.add(Op::Jump(loop_head, args), None);
                func.blocks[b_idx].insts.extend([b_inst, j]);
            }
        }
    }
    peel_entry_guard(func, (old_entry, new_entry, loop_head), old_params_len, orig_blocks, &rewritten, accum_op.zip(ret_ty));
    func.bump_cfg();
    if let (Some(op), Some(acc), Some(rty)) = (accum_op, acc_param, ret_ty) {
        let post_cfg = func.cfg();
        for b_idx in 0..orig_blocks {
            if rewritten[b_idx] || !post_cfg.reachable[b_idx] {
                continue;
            }
            let Some(&ret_inst) = func.blocks[b_idx].insts.last() else { continue };
            let Op::Return(vals) = &func.insts[ret_inst.index()].op else { continue };
            let [val] = vals[..] else { continue };
            let combined = func.value(rty);
            let b_inst = func.add(Op::Binary(op, acc, val), Some(combined));
            func.insts[ret_inst.index()].op = Op::Return(vec![combined]);
            let pos = func.blocks[b_idx].insts.len() - 1;
            func.blocks[b_idx].insts.insert(pos, b_inst);
        }
    }
    for callee in dead_callees {
        if let Some(addr) = kill_dead_value(func, &def_inst, callee) {
            kill_dead_value(func, &def_inst, addr);
        }
    }
    func.bump_cfg();
    Domain::CFG | Domain::VALUES
}

fn clone_pure_insts(
    func: &mut Function,
    insts: &[Inst],
    map: &mut rustc_hash::FxHashMap<Value, Value>,
    out: &mut Vec<Inst>,
) -> bool {
    for &inst in insts {
        let mut data = func.insts[inst.index()].clone();
        if crate::mem::is_atomic_barrier(&data.op)
            || matches!(data.op, Op::Call { .. } | Op::TryCall { .. } | Op::Store(..) | Op::StackStore(..) | Op::Copy(..) | Op::Move(..) | Op::Trap | Op::Trapz(..))
        {
            return false;
        }
        let mut ok = true;
        data.op.map_operands(|v| match map.get(&v) {
            Some(&mapped) => mapped,
            None => {
                ok = false;
                v
            }
        });
        if !ok {
            return false;
        }
        let res = data.result.map(|old| {
            let fresh = func.value_like(old);
            map.insert(old, fresh);
            fresh
        });
        out.push(func.add(data.op, res));
    }
    true
}

fn peel_entry_guard(
    func: &mut Function,
    (old_entry, new_entry, loop_head): (Block, Block, Block),
    old_params_len: usize,
    orig_blocks: usize,
    rewritten: &[bool],
    accum_op: Option<(Binary, Type)>,
) {
    let Some(&term) = func.blocks[old_entry.index()].insts.last() else { return };
    let Op::Brif(cond, then_b, then_args, else_b, else_args) = func.insts[term.index()].op.clone() else { return };
    if then_b == else_b || func.blocks[new_entry.index()].insts.is_empty() {
        return;
    }
    func.bump_cfg();
    let cfg = func.cfg();
    let is_base = |b: Block| {
        b != old_entry
            && b.index() < orig_blocks
            && !rewritten[b.index()]
            && cfg.pred_count(b) == 1
            && func.blocks[b.index()].insts.last().is_some_and(|&t| matches!(func.insts[t.index()].op, Op::Return(_)))
    };
    let (base_b, base_args, rec_b, rec_args, then_is_base) = if is_base(then_b) {
        (then_b, then_args, else_b, else_args, true)
    } else if is_base(else_b) {
        (else_b, else_args, then_b, then_args, false)
    } else {
        return;
    };

    let mut map = rustc_hash::FxHashMap::default();
    for (&old_p, &fresh_p) in func.blocks[loop_head.index()].params[..old_params_len].iter().zip(&func.blocks[new_entry.index()].params) {
        map.insert(old_p, fresh_p);
    }
    let old_insts = func.blocks[old_entry.index()].insts.clone();
    let Some((&_, old_body)) = old_insts.split_last() else { return };
    let mut entry_insts = Vec::with_capacity(old_body.len() + 1);
    if !clone_pure_insts(func, old_body, &mut map, &mut entry_insts) {
        return;
    }
    let Some(&cloned_cond) = map.get(&cond) else { return };
    let base_insts = func.blocks[base_b.index()].insts.clone();
    let base_params = func.blocks[base_b.index()].params.clone();
    if base_params.len() != base_args.len() {
        return;
    }
    for (&param, &arg) in base_params.iter().zip(&base_args) {
        let Some(&mapped) = map.get(&arg) else { return };
        map.insert(param, mapped);
    }
    let mut fast_insts = Vec::with_capacity(base_insts.len());
    if !clone_pure_insts(func, &base_insts, &mut map, &mut fast_insts) {
        return;
    }

    let slow_insts = std::mem::take(&mut func.blocks[new_entry.index()].insts);
    let slow_preheader = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: Vec::new(), insts: slow_insts });
    let fast_ret = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: Vec::new(), insts: fast_insts });

    let brif = if then_is_base {
        Op::Brif(cloned_cond, fast_ret, Vec::new(), slow_preheader, Vec::new())
    } else {
        Op::Brif(cloned_cond, slow_preheader, Vec::new(), fast_ret, Vec::new())
    };
    entry_insts.push(func.add(brif, None));
    func.blocks[new_entry.index()].insts = entry_insts;

    if cfg.preds_of(old_entry).iter().any(|&p| p != loop_head) {
        return;
    }
    let mut old_defs = vec![false; func.types.len()];
    for &inst in old_body {
        if let Some(res) = func.insts[inst.index()].result {
            old_defs[res.index()] = true;
        }
    }
    if rec_args.iter().any(|v| old_defs[v.index()]) {
        return;
    }
    let mut used_outside = false;
    for (b, block) in func.blocks[..orig_blocks].iter().enumerate() {
        if b == old_entry.index() || b == base_b.index() {
            continue;
        }
        for &inst in &block.insts {
            func.insts[inst.index()].op.for_operands(|v| {
                if old_defs[v.index()] {
                    used_outside = true;
                }
            });
        }
    }
    if used_outside {
        return;
    }

    let loop_param_tys: Vec<Type> = func.blocks[loop_head.index()].params.iter().map(|&v| func.ty(v)).collect();
    let latch_params: Vec<Value> = loop_param_tys.iter().map(|&ty| func.value(ty)).collect();
    let mut latch_map = rustc_hash::FxHashMap::default();
    for (&old_p, &latch_p) in func.blocks[loop_head.index()].params[..old_params_len].iter().zip(&latch_params[..old_params_len]) {
        latch_map.insert(old_p, latch_p);
    }
    let mut latch_insts = Vec::with_capacity(old_body.len() + 1);
    if !clone_pure_insts(func, old_body, &mut latch_map, &mut latch_insts) {
        return;
    }
    let Some(&latch_cond) = latch_map.get(&cond) else { return };
    for (&param, &arg) in base_params.iter().zip(&base_args) {
        let Some(&mapped) = latch_map.get(&arg) else { return };
        latch_map.insert(param, mapped);
    }
    let mut latch_ret_insts = Vec::with_capacity(base_insts.len() + 1);
    if !clone_pure_insts(func, &base_insts, &mut latch_map, &mut latch_ret_insts) {
        return;
    }
    if let Some((op, rty)) = accum_op
        && latch_params.len() == old_params_len + 1
    {
        let Some(&last_ret) = latch_ret_insts.last() else { return };
        let Op::Return(vals) = &func.insts[last_ret.index()].op else { return };
        if vals.len() != 1 {
            return;
        }
        let val = vals[0];
        let latch_acc = latch_params[old_params_len];
        let combined = func.value(rty);
        let add_inst = func.add(Op::Binary(op, latch_acc, val), Some(combined));
        latch_ret_insts.pop();
        latch_ret_insts.push(add_inst);
        latch_ret_insts.push(func.add(Op::Return(vec![combined]), None));
    }

    let latch_ret = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: Vec::new(), insts: latch_ret_insts });
    let latch = Block(func.blocks.len() as u32);
    let latch_brif = if then_is_base {
        Op::Brif(latch_cond, latch_ret, Vec::new(), loop_head, latch_params.clone())
    } else {
        Op::Brif(latch_cond, loop_head, latch_params.clone(), latch_ret, Vec::new())
    };
    latch_insts.push(func.add(latch_brif, None));
    func.blocks.push(BlockData { params: latch_params, insts: latch_insts });

    for (b, block) in func.blocks[..orig_blocks].iter().enumerate() {
        if rewritten[b]
            && let Some(&last) = block.insts.last()
            && let Op::Jump(target, _) = &mut func.insts[last.index()].op
            && *target == loop_head
        {
            *target = latch;
        }
    }
    func.blocks[loop_head.index()].insts = vec![func.add(Op::Jump(rec_b, rec_args), None)];
    func.blocks[old_entry.index()].insts.clear();
    func.blocks[base_b.index()].params.clear();
    func.blocks[base_b.index()].insts.clear();
}

const MAX_SPLIT: usize = 4;
pub(crate) use crate::alloc::FOLD_LIMIT;

fn reg_budgets(arch: crate::target::Arch) -> (usize, usize) {
    match arch {
        crate::target::Arch::X64 => (5, 9),
        crate::target::Arch::Arm64 | crate::target::Arch::Riscv64 | crate::target::Arch::Wasm32 => (usize::MAX, 0),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StaticVal { Dynamic, Imm(i64), Frame }

fn is_call_site(func: &Function, insts: &[Inst], idx: usize) -> bool {
    match func.insts[insts[idx].index()].op {
        Op::TryCall { .. } => true,
        Op::Call { .. } => !insts.get(idx + 1).is_some_and(|&n| matches!(func.insts[n.index()].op, Op::Trap)),
        _ => false,
    }
}

fn static_step(func: &Function, op: &Op, ty: Type, vals: &[StaticVal]) -> StaticVal {
    match *op {
        Op::Const(imm) if ty.is_small_int() => StaticVal::Imm(imm),
        Op::StackAddr(..) | Op::FramePointer if ty == Type::I64 => StaticVal::Frame,
        Op::Icmp(cc, x, y) => match (vals[x.index()], vals[y.index()]) {
            (StaticVal::Imm(a), StaticVal::Imm(b)) if func.ty(x).is_small_int() => StaticVal::Imm(i64::from(crate::fold::holds(cc, func.ty(x), a, b))),
            _ => StaticVal::Dynamic,
        },
        Op::Binary(bin, x, y) => match (vals[x.index()], vals[y.index()]) {
            (StaticVal::Frame, StaticVal::Imm(imm)) | (StaticVal::Imm(imm), StaticVal::Frame)
                if bin == Binary::Iadd && ty == Type::I64 && (-FOLD_LIMIT..=FOLD_LIMIT).contains(&imm) => StaticVal::Frame,
            (StaticVal::Imm(a), StaticVal::Imm(b)) if func.ty(x).is_small_int() => crate::fold::constant(bin, a, b).map_or(StaticVal::Dynamic, StaticVal::Imm),
            _ => StaticVal::Dynamic,
        },
        Op::Convert(Convert::Uextend | Convert::Ireduce, x) => match vals[x.index()] {
            StaticVal::Imm(imm) if func.ty(x).is_small_int() => StaticVal::Imm(crate::fold::extend(imm, func.ty(x), false)),
            _ => StaticVal::Dynamic,
        },
        _ => StaticVal::Dynamic,
    }
}

fn static_values(func: &Function) -> Vec<bool> {
    let mut vals = vec![StaticVal::Dynamic; func.types.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for data in &func.insts {
            let Some(res) = data.result else { continue };
            if vals[res.index()] != StaticVal::Dynamic { continue; }
            let next = static_step(func, &data.op, func.ty(res), &vals);
            if next != StaticVal::Dynamic {
                vals[res.index()] = next;
                changed = true;
            }
        }
    }
    vals.into_iter().map(|v| v != StaticVal::Dynamic).collect()
}

fn fusable_cmp_flags(func: &Function, uses: &[u32]) -> Vec<bool> {
    let mut out = vec![false; func.types.len()];
    for block in &func.blocks {
        let Some(&last) = block.insts.last() else { continue };
        let (Op::Brif(cond, ..) | Op::Trapz(cond)) = func.insts[last.index()].op else { continue };
        if uses[cond.index()] == 1
            && let Some(&prev) = block.insts.iter().rev().nth(1)
            && func.insts[prev.index()].result == Some(cond)
            && matches!(func.insts[prev.index()].op, Op::Icmp(_, a, _) | Op::Fits(_, _, a, _) if !func.ty(a).is_vector())
        {
            out[cond.index()] = true;
        }
    }
    out
}

fn unique_head_preds(cfg: &Cfg, head: Block) -> Vec<Block> {
    let mut preds = Vec::new();
    for &p in cfg.preds_of(head) {
        if !preds.contains(&p) { preds.push(p); }
    }
    preds
}

fn eligible_loop(func: &Function, cfg: &Cfg, loops: &Loops, each: &Loop) -> Option<Vec<Block>> {
    if Some(each.head) == func.entry || !cfg.reachable[each.head.index()] { return None; }
    if loops.list.iter().any(|o| o.head != each.head && each.body[o.head.index()]) { return None; }
    let body_ok = (0..func.blocks.len()).filter(|&b| each.body[b]).all(|b| {
        cfg.reachable[b]
            && cfg.dominates(each.head, Block(b as u32))
            && !func.blocks[b].insts.iter().any(|&i| matches!(func.insts[i.index()].op, Op::Call { .. } | Op::TryCall { .. }))
    });
    if !body_ok { return None; }
    let preds = unique_head_preds(cfg, each.head);
    let ok_preds = preds.iter().all(|&p| cfg.reachable[p.index()] && matches!(func.last(p), Some(Op::Jump(..) | Op::Brif(..))));
    let (has_out, has_in) = (preds.iter().any(|&p| !each.body[p.index()]), preds.iter().any(|&p| each.body[p.index()]));
    (ok_preds && has_out && has_in).then_some(preds)
}

fn for_loop_operands(op: &Op, body: &[bool], mut f: impl FnMut(Value)) {
    match op {
        Op::Jump(target, args) => if body[target.index()] { args.iter().for_each(|&v| f(v)); },
        Op::Brif(cond, t, ta, fb, fa) => {
            f(*cond);
            if body[t.index()] { ta.iter().for_each(|&v| f(v)); }
            if body[fb.index()] { fa.iter().for_each(|&v| f(v)); }
        }
        _ => op.for_operands(f),
    }
}

fn local_live_sets(func: &Function, dense: &[Option<usize>], count: usize) -> (BitMatrix, BitMatrix) {
    let blocks = func.blocks.len();
    let (mut uses, mut defs) = (BitMatrix::new(blocks, count), BitMatrix::new(blocks, count));
    for (b, block) in func.blocks.iter().enumerate() {
        block.params.iter().for_each(|&p| if let Some(d) = dense[p.index()] { defs.set(b, d); });
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| if let Some(d) = dense[v.index()] && !defs.get(b, d) { uses.set(b, d); });
            results(data).for_each(|r| if let Some(d) = dense[r.index()] { defs.set(b, d); });
        }
    }
    (uses, defs)
}

fn compute_live_in(func: &Function, cfg: &Cfg, dense: &[Option<usize>], count: usize) -> BitMatrix {
    if count == 0 { return BitMatrix::new(func.blocks.len(), 0); }
    let (uses, defs) = local_live_sets(func, dense, count);
    solve_backward_liveness(func, cfg.rpo.iter().copied(), &uses, &defs)
}

fn find_crosses_call(func: &Function, cfg: &Cfg, candidates: &[usize]) -> (Vec<bool>, BitMatrix, Vec<Option<usize>>) {
    let mut dense = vec![None; func.types.len()];
    candidates.iter().enumerate().for_each(|(i, &v)| dense[v] = Some(i));
    let live_in = compute_live_in(func, cfg, &dense, candidates.len());
    let mut crosses = vec![false; func.types.len()];
    let words = live_in.words();
    if words == 0 { return (crosses, live_in, dense); }
    let mut cur = vec![0_u64; words];
    for &block in &cfg.rpo {
        live_in.fill_succ_union(func, block, &mut cur);
        let insts = &func.blocks[block.index()].insts;
        for idx in (0..insts.len()).rev() {
            let data = &func.insts[insts[idx].index()];
            results(data).for_each(|r| if let Some(d) = dense[r.index()] { cur[d >> 6] &= !(1_u64 << (d & 63)); });
            if is_call_site(func, insts, idx) {
                for i in set_bits(&cur) {
                    crosses[candidates[i]] = true;
                }
            }
            data.op.for_operands(|v| if let Some(d) = dense[v.index()] { cur[d >> 6] |= 1_u64 << (d & 63); });
        }
    }
    (crosses, live_in, dense)
}

fn collect_loop_uses(func: &Function, cfg: &Cfg, each: &Loop, def_blocks: &[Option<Block>], static_vals: &[bool]) -> Vec<(usize, u32)> {
    let mut uses_in_l = FxHashMap::default();
    for (b, block) in func.blocks.iter().enumerate() {
        if !each.body[b] { continue; }
        for &inst in &block.insts {
            for_loop_operands(&func.insts[inst.index()].op, &each.body, |v| {
                let idx = v.index();
                if func.ty(v).is_small_int() && !static_vals[idx]
                    && let Some(db) = def_blocks[idx]
                    && cfg.reachable[db.index()] && !each.body[db.index()] && cfg.dominates(db, each.head)
                {
                    *uses_in_l.entry(idx).or_insert(0_u32) += 1;
                }
            });
        }
    }
    uses_in_l.into_iter().collect()
}

fn block_peak_gprs(func: &Function, live_in: &BitMatrix, dense: &[Option<usize>], block: Block, cur: &mut [u64]) -> usize {
    live_in.fill_succ_union(func, block, cur);
    let mut peak: usize = cur.iter().map(|w| w.count_ones() as usize).sum();
    for &inst in func.blocks[block.index()].insts.iter().rev() {
        let data = &func.insts[inst.index()];
        results(data).for_each(|r| if let Some(d) = dense[r.index()] { cur[d >> 6] &= !(1_u64 << (d & 63)); });
        data.op.for_operands(|v| if let Some(d) = dense[v.index()] { cur[d >> 6] |= 1_u64 << (d & 63); });
        peak = peak.max(cur.iter().map(|w| w.count_ones() as usize).sum());
    }
    peak
}

fn loop_dense_gprs(func: &Function, each: &Loop, (def_blocks, excluded, crosses): (&[Option<Block>], &[bool], &[bool])) -> (Vec<Option<usize>>, usize) {
    let (mut dense, mut count) = (vec![None; func.types.len()], 0_usize);
    let mut add = |v: usize| if func.types[v].is_small_int() && !excluded[v] && dense[v].is_none() { dense[v] = Some(count); count += 1; };
    for block in func.blocks.iter().enumerate().filter_map(|(b, blk)| each.body[b].then_some(blk)) {
        block.params.iter().for_each(|p| add(p.index()));
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            results(data).for_each(|r| add(r.index()));
            data.op.for_operands(|v| if let Some(db) = def_blocks[v.index()] && !each.body[db.index()] && !crosses[v.index()] { add(v.index()); });
        }
    }
    (dense, count)
}

fn loop_peak_gprs(func: &Function, cfg: &Cfg, each: &Loop, loop_budget: usize, ctx: (&[Option<Block>], &[bool], &[bool])) -> usize {
    let excluded = ctx.1;
    let head_params = func.blocks[each.head.index()].params.iter().filter(|&&p| func.ty(p).is_small_int() && !excluded[p.index()]).count();
    if head_params >= loop_budget { return head_params; }
    let (dense, count) = loop_dense_gprs(func, each, ctx);
    let live_in = compute_live_in(func, cfg, &dense, count);
    let words = live_in.words();
    if words == 0 { return head_params; }
    let (mut cur, mut peak) = (vec![0_u64; words], head_params);
    for (b, &inside) in each.body.iter().enumerate() {
        if inside { peak = peak.max(block_peak_gprs(func, &live_in, &dense, Block(b as u32), &mut cur)); }
    }
    peak
}

fn apply_loop_split(func: &mut Function, each: &Loop, preds: &[Block], chosen: &[usize]) {
    let (mut remap, mut pairs) = (FxHashMap::default(), Vec::with_capacity(chosen.len()));
    for &v_idx in chosen {
        let v = Value(v_idx as u32);
        let v_loop = func.value(func.ty(v));
        func.blocks[each.head.index()].params.push(v_loop);
        remap.insert(v, v_loop);
        pairs.push((v, v_loop));
    }
    for block in func.blocks.iter().enumerate().filter_map(|(b, blk)| each.body[b].then_some(blk)) {
        block.insts.iter().for_each(|&i| func.insts[i.index()].op.map_operands(|x| remap.get(&x).copied().unwrap_or(x)));
    }
    for &pred in preds {
        let is_back = each.body[pred.index()];
        let Some(&term) = func.blocks[pred.index()].insts.last() else { continue };
        let add = |args: &mut Vec<Value>| args.extend(pairs.iter().map(|&(v, v_loop)| if is_back { v_loop } else { v }));
        match &mut func.insts[term.index()].op {
            Op::Jump(target, args) if *target == each.head => add(args),
            Op::Brif(_, t, ta, f, fa) => { if *t == each.head { add(ta); } if *f == each.head { add(fa); } }
            _ => {}
        }
    }
}

fn plan_loop_split(
    (func, cfg): (&Function, &Cfg),
    (each, preds, mut cands): (Loop, Vec<Block>, Vec<(usize, u32)>),
    (saved_floor, loop_budget): (usize, usize),
    (all_gprs, live_in, dense): (&[usize], &BitMatrix, &[Option<usize>]),
    ctx: (&[Option<Block>], &[bool], &[bool]),
) -> Option<(Loop, Vec<Block>, Vec<usize>)> {
    let crosses = ctx.2;
    let hrow = each.head.index();
    let live_crosses = all_gprs.iter().filter(|&&v| crosses[v] && dense[v].is_some_and(|d| live_in.get(hrow, d))).count();
    let overflow = live_crosses.saturating_sub(saved_floor);
    if overflow == 0 { return None; }
    let out_preds = preds.iter().filter(|&&p| !each.body[p.index()]).count();
    cands.retain(|&(v, c)| crosses[v] && (c >= 2 || (c >= 1 && each.depth >= 2 && out_preds == 1)));
    if cands.is_empty() { return None; }
    let budget = loop_budget.saturating_sub(loop_peak_gprs(func, cfg, &each, loop_budget, ctx)).min(MAX_SPLIT).min(overflow);
    if budget == 0 { return None; }
    cands.sort_unstable_by_key(|&(v, count)| (Reverse(count), v));
    Some((each, preds, cands.into_iter().take(budget).map(|(v, _)| v).collect()))
}

fn split_innermost_loops(func: &mut Function, target: &crate::Target) -> Domain {
    let (saved_floor, loop_budget) = reg_budgets(target.arch);
    if loop_budget == 0 { return Domain::NONE; }
    let has_call = func.blocks.iter().any(|b| (0..b.insts.len()).any(|i| is_call_site(func, &b.insts, i)));
    if func.coroutine.is_some() || func.entry.is_none() || func.blocks.len() < 2 || !has_call { return Domain::NONE; }
    let uses = func.use_counts();
    let (mut static_vals, def_blocks, fusable) = (static_values(func), func.def_blocks(), fusable_cmp_flags(func, &uses));
    if target.arch == crate::Arch::X64 {
        let defs = crate::fold::defs(func);
        for (s, a) in static_vals.iter_mut().zip(crate::alloc::atomic_elidable_addrs(func, &defs, &uses)) {
            *s |= a;
        }
    }
    let excluded: Vec<bool> = static_vals.iter().zip(&fusable).map(|(&s, &f)| s || f).collect();
    let plans: Vec<(Loop, Vec<Block>, Vec<usize>)> = {
        let cfg = func.cfg();
        let loops = cfg.loops();
        let eligible: Vec<_> = loops.list.iter().filter_map(|each| {
            let preds = eligible_loop(func, &cfg, loops, each)?;
            let cands = collect_loop_uses(func, &cfg, each, &def_blocks, &static_vals);
            (!cands.is_empty()).then(|| (each.clone(), preds, cands))
        }).collect();
        if eligible.is_empty() { return Domain::NONE; }
        let all_gprs: Vec<usize> = (0..func.types.len()).filter(|&v| func.types[v].is_small_int() && !excluded[v]).collect();
        let (crosses, live_in, dense) = find_crosses_call(func, &cfg, &all_gprs);
        eligible.into_iter().filter_map(|item| {
            plan_loop_split((func, &cfg), item, (saved_floor, loop_budget), (&all_gprs, &live_in, &dense), (&def_blocks, &excluded, &crosses))
        }).collect()
    };
    if plans.is_empty() { return Domain::NONE; }
    for (each, preds, chosen) in plans { apply_loop_split(func, &each, &preds, &chosen); }
    Domain::VALUES
}
