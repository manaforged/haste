use crate::ir::{ArgumentPurpose, Binary, Block, BlockData, Function, Inst, Op, Type, Value};

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

fn analyze_block(
    func: &Function,
    block: usize,
    old_params_len: usize,
    def_inst: &[Option<Inst>],
    uses: &[u32],
) -> Option<TailCase> {
    let insts = &func.blocks[block].insts;
    let &ret_inst = insts.last()?;
    let Op::Return(ret_vals) = &func.insts[ret_inst.index()].op else { return None };
    if insts.len() >= 2 {
        let call_inst = insts[insts.len() - 2];
        if let Op::Call { sig, callee, args, results } = &func.insts[call_inst.index()].op
            && func.sigs[sig.index()] == func.signature
            && args.len() == old_params_len
            && results.len() == func.signature.returns.len()
            && results == ret_vals
            && is_self_callee(func, def_inst, *callee).is_some()
        {
            return Some(TailCase::Pure { args: args.clone(), callee: *callee });
        }
    }
    if insts.len() >= 3
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

/// Turns self-recursive tail calls (and associative/commutative accumulator calls) into loops.
pub(crate) fn run(func: &mut Function) -> bool {
    if func.self_slots.is_empty() || func.coroutine.is_some() {
        return false;
    }
    let Some(old_entry) = func.entry else { return false };
    if func.signature.params.iter().any(|p| p.purpose != ArgumentPurpose::Normal)
        || func.signature.returns.len() > 1
        || func.signature.returns.iter().any(|p| p.purpose != ArgumentPurpose::Normal)
    {
        return false;
    }
    for block in &func.blocks {
        if block.insts.last().is_some_and(|&i| func.insts[i.index()].op.targets().any(|t| t == old_entry)) {
            return false;
        }
        for &inst in &block.insts {
            if matches!(func.insts[inst.index()].op, Op::StackAddr(..) | Op::TryCall { .. }) {
                return false;
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
                Some(prev) if prev != op => return false,
                None => accum_op = Some(op),
                _ => {}
            }
        }
        cases.push((block, case));
    }
    if cases.is_empty() {
        return false;
    }
    let old_params = std::mem::take(&mut func.blocks[old_entry.index()].params);
    let fresh_params: Vec<Value> = old_params.iter().map(|&p| func.value(func.ty(p))).collect();
    let loop_head = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: old_params, insts: Vec::new() });
    let new_entry = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: fresh_params.clone(), insts: Vec::new() });
    func.entry = Some(new_entry);

    let (acc_param, ret_ty) = if let Some(op) = accum_op {
        let ret_ty = match &cases[0].1 {
            TailCase::Accum { other, .. } => func.ty(*other),
            TailCase::Pure { args: _, callee: _ } => {
                let Some(last) = func.blocks[cases[0].0].insts.last() else { return false };
                let Op::Return(vals) = &func.insts[last.index()].op else { return false };
                func.ty(vals[0])
            }
        };
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
    if let (Some(op), Some(acc), Some(rty)) = (accum_op, acc_param, ret_ty) {
        for b_idx in 0..orig_blocks {
            if rewritten[b_idx] {
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
    true
}

fn clone_pure_insts(
    func: &mut Function,
    insts: &[Inst],
    map: &mut rustc_hash::FxHashMap<Value, Value>,
    out: &mut Vec<Inst>,
) -> bool {
    for &inst in insts {
        let mut data = func.insts[inst.index()].clone();
        if matches!(
            data.op,
            Op::Call { .. }
                | Op::TryCall { .. }
                | Op::Store(..)
                | Op::StackStore(..)
                | Op::Copy(..)
                | Op::Move(..)
                | Op::AtomicLoad(..)
                | Op::AtomicStore(..)
                | Op::AtomicRmw(..)
                | Op::AtomicCas(..)
                | Op::Fence
                | Op::Trap
                | Op::Trapz(..)
        ) {
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
            let fresh = func.value(func.ty(old));
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
    let cfg = crate::cfg::Cfg::new(func);
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
