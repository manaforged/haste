use crate::cfg::Cfg;
use crate::ir::{ArgumentPurpose, Binary, Block, BlockData, Convert, Domain, Function, Inst, IntCC, Op, SigRef, Signature, Type, Unary, Value};
use crate::reloc::RuntimeSymbol;
use rustc_hash::FxHashMap;

const MAX_RETURN_PREFIX: usize = 3;
const MIN_TAIL_GROUP: usize = 3;
const MAX_TAIL_PARAMS: usize = 4;
const MAX_SINK_PARAMS: usize = 2;
const MAX_SINK_DEPTH: usize = 3;

/// Turns every jump into a tiny return block into that return, then factors identical tail calls and common trailing ops across arms.
pub(crate) fn run(func: &mut Function) -> Domain {
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
    let mut changed = crate::merge::lower_binary_br_tables(func);
    changed |= inline_returns(func);
    if factor_tail_calls(func, changed) {
        changed = true;
    }
    for _ in 0..MAX_SINK_DEPTH {
        let round = factor_common_returns(func, changed);
        let sunk = sink_common_jump_tails(func, changed || round);
        if !round && !sunk {
            break;
        }
        changed = true;
    }
    if changed {
        func.bump_cfg();
        Domain::CFG | Domain::VALUES
    } else {
        Domain::NONE
    }
}

fn inline_returns(func: &mut Function) -> bool {
    let cfg = func.cfg();
    let mut threaded = vec![0_u32; func.blocks.len()];
    let mut changed = false;
    loop {
        let mut progress = false;
        for block in 0..func.blocks.len() {
            if !cfg.reachable[block] {
                continue;
            }
            let Some(&last) = func.blocks[block].insts.last() else { continue };
            let Op::Jump(target, args) = func.insts[last.index()].op.clone() else { continue };
            if target.index() == block {
                continue;
            }
            let Some((prefix, values)) = inline_return(func, target, &args) else { continue };
            threaded[target.index()] += 1;
            func.insts[last.index()].op = Op::Return(values);
            if !prefix.is_empty() {
                let len = func.blocks[block].insts.len();
                func.blocks[block].insts.splice(len - 1..len - 1, prefix);
            }
            changed = true;
            progress = true;
        }
        if !progress {
            break;
        }
    }
    for (block, count) in threaded.iter().enumerate() {
        let b = Block(block as u32);
        if *count > 0 && *count == cfg.pred_count(b) && func.entry != Some(b) {
            func.blocks[block].insts.clear();
            changed = true;
        }
    }
    changed
}

/// The duplicated prefix instructions and returned values for jumping into `target` with `args`.
fn inline_return(func: &mut Function, target: Block, args: &[Value]) -> Option<(Vec<Inst>, Vec<Value>)> {
    let data = &func.blocks[target.index()];
    let (&ret_inst, prefix) = data.insts.split_last()?;
    if prefix.len() > MAX_RETURN_PREFIX || data.params.len() != args.len() {
        return None;
    }
    let Op::Return(ret_vals) = &func.insts[ret_inst.index()].op else { return None };
    if !prefix.iter().all(|inst| func.insts[inst.index()].op.is_speculable()) {
        return None;
    }
    let params = data.params.clone();
    let ret_vals = ret_vals.clone();
    let prefix_data: Vec<(Op, Option<Value>)> = prefix
        .iter()
        .map(|inst| {
            let inst_data = &func.insts[inst.index()];
            (inst_data.op.clone(), inst_data.result)
        })
        .collect();
    let mut locals: Vec<(Value, Value)> = Vec::with_capacity(prefix_data.len());
    let subst = |locals: &[(Value, Value)], value: Value| -> Value {
        if let Some(pos) = params.iter().position(|param| *param == value) {
            return args[pos];
        }
        if let Some(&(_, fresh)) = locals.iter().find(|&&(old, _)| old == value) {
            return fresh;
        }
        value
    };
    let mut cloned = Vec::with_capacity(prefix_data.len());
    for (mut op, old_res) in prefix_data {
        op.map_operands(|v| subst(&locals, v));
        let res = old_res.map(|old| {
            let fresh = func.value(func.ty(old));
            locals.push((old, fresh));
            fresh
        });
        cloned.push(func.add(op, res));
    }
    let values = ret_vals.into_iter().map(|v| subst(&locals, v)).collect();
    Some((cloned, values))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum CalleeKey {
    Symbol(u32),
    Runtime(RuntimeSymbol),
    Value(Value),
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct TailKey {
    callee: CalleeKey,
    callee_ty: Type,
    sig: Signature,
    ret_types: Vec<Type>,
}

struct TailSite {
    block: usize,
    sig: SigRef,
    callee: Value,
    args: Vec<Value>,
    ret_types: Vec<Type>,
    callee_def: Option<(usize, Inst)>,
}

fn factor_tail_calls(func: &mut Function, cfg_dirty: bool) -> bool {
    if func.coroutine.is_some() {
        return false;
    }
    if cfg_dirty {
        func.bump_cfg();
    }
    let cfg = func.cfg();
    let defs = symbol_defs(func);
    let mut index: FxHashMap<TailKey, usize> = FxHashMap::default();
    let mut groups: Vec<(TailKey, Vec<TailSite>)> = Vec::new();
    for block in 0..func.blocks.len() {
        if !cfg.reachable[block] || func.entry == Some(Block(block as u32)) {
            continue;
        }
        let Some((key, site)) = tail_site(func, block, &defs) else { continue };
        match index.get(&key).copied() {
            Some(at) => groups[at].1.push(site),
            None => {
                index.insert(key.clone(), groups.len());
                groups.push((key, vec![site]));
            }
        }
    }
    if !groups.iter().any(|(_, g)| g.len() >= MIN_TAIL_GROUP) {
        return false;
    }
    let uses = func.use_counts();
    let mut changed = false;
    for (key, group) in &groups {
        if group.len() >= MIN_TAIL_GROUP && emit_shared_tail(func, key, group, &uses) {
            changed = true;
        }
    }
    changed
}

fn symbol_defs(func: &Function) -> Vec<Option<(CalleeKey, usize, Inst)>> {
    let mut defs = vec![None; func.types.len()];
    for (block, data) in func.blocks.iter().enumerate() {
        for &inst in &data.insts {
            let item = &func.insts[inst.index()];
            let Some(res) = item.result else { continue };
            let key = match item.op {
                Op::Symbol(sym) => CalleeKey::Symbol(sym),
                Op::Runtime(rt) => CalleeKey::Runtime(rt),
                _ => continue,
            };
            defs[res.index()] = Some((key, block, inst));
        }
    }
    defs
}

fn tail_site(func: &Function, block: usize, defs: &[Option<(CalleeKey, usize, Inst)>]) -> Option<(TailKey, TailSite)> {
    let insts = &func.blocks[block].insts;
    let len = insts.len();
    if len < 2 {
        return None;
    }
    let Op::Return(ret_vals) = &func.insts[insts[len - 1].index()].op else { return None };
    let Op::Call { sig, callee, args, results } = &func.insts[insts[len - 2].index()].op else { return None };
    let sig_data = &func.sigs[sig.index()];
    if args.len() != sig_data.params.len()
        || results.len() != sig_data.returns.len()
        || results != ret_vals
        || sig_data.params.iter().any(|p| p.purpose == ArgumentPurpose::Variadic)
    {
        return None;
    }
    let ret_types: Vec<Type> = results.iter().map(|&v| func.ty(v)).collect();
    let (key, callee_def) = match defs[callee.index()] {
        Some((key, def_block, def_inst)) => (key, Some((def_block, def_inst))),
        None => (CalleeKey::Value(*callee), None),
    };
    let tail_key = TailKey { callee: key, callee_ty: func.ty(*callee), sig: sig_data.clone(), ret_types: ret_types.clone() };
    let site = TailSite { block, sig: *sig, callee: *callee, args: args.clone(), ret_types, callee_def };
    Some((tail_key, site))
}

fn varying_arg_positions(func: &Function, group: &[TailSite]) -> Option<Vec<usize>> {
    let arity = group[0].args.len();
    let mut varying = Vec::new();
    for pos in 0..arity {
        let first = group[0].args[pos];
        if group[1..].iter().any(|site| func.ty(site.args[pos]) != func.ty(first)) {
            return None;
        }
        if group[1..].iter().any(|site| site.args[pos] != first) {
            varying.push(pos);
            if varying.len() > MAX_TAIL_PARAMS {
                return None;
            }
        }
    }
    Some(varying)
}

fn emit_callee_def(func: &mut Function, key: &TailKey, insts: &mut Vec<Inst>) -> Value {
    match key.callee {
        CalleeKey::Symbol(sym) => {
            let val = func.value(key.callee_ty);
            insts.push(func.add(Op::Symbol(sym), Some(val)));
            val
        }
        CalleeKey::Runtime(rt) => {
            let val = func.value(key.callee_ty);
            insts.push(func.add(Op::Runtime(rt), Some(val)));
            val
        }
        CalleeKey::Value(val) => val,
    }
}

fn emit_shared_tail(func: &mut Function, key: &TailKey, group: &[TailSite], uses: &[u32]) -> bool {
    let Some(varying) = varying_arg_positions(func, group) else {
        return false;
    };
    let shared = Block(func.blocks.len() as u32);
    let params: Vec<Value> = varying.iter().map(|&pos| func.value(func.ty(group[0].args[pos]))).collect();
    let mut insts = Vec::with_capacity(3);
    let same_callee = group[1..].iter().all(|site| site.callee == group[0].callee);
    let callee = if same_callee { group[0].callee } else { emit_callee_def(func, key, &mut insts) };
    let mut call_args = group[0].args.clone();
    for (param_idx, &arg_pos) in varying.iter().enumerate() {
        call_args[arg_pos] = params[param_idx];
    }
    let results: Vec<Value> = group[0].ret_types.iter().map(|&ty| func.value(ty)).collect();
    insts.push(func.add(Op::Call { sig: group[0].sig, callee, args: call_args, results: results.clone() }, None));
    insts.push(func.add(Op::Return(results), None));
    func.blocks.push(BlockData { params, insts });
    for site in group {
        rewrite_tail_site(func, site, (shared, &varying), (!same_callee).then_some(uses));
    }
    true
}

fn rewrite_tail_site(func: &mut Function, site: &TailSite, (shared, varying): (Block, &[usize]), uses: Option<&[u32]>) {
    let jump_args: Vec<Value> = varying.iter().map(|&pos| site.args[pos]).collect();
    let jump = func.add(Op::Jump(shared, jump_args), None);
    let blk_insts = &mut func.blocks[site.block].insts;
    let len = blk_insts.len();
    blk_insts.truncate(len - 2);
    if let Some(counts) = uses
        && let Some((def_block, def_inst)) = site.callee_def
        && def_block == site.block
        && counts[site.callee.index()] == 1
    {
        blk_insts.retain(|&inst| inst != def_inst);
    }
    blk_insts.push(jump);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum TailKind {
    Const(i64),
    Unary(Unary),
    Convert(Convert),
    Binary(Binary),
    Icmp(IntCC),
    Select,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct TailOpSig {
    kind: TailKind,
    res_ty: Type,
    arity: u8,
    arg_tys: [Type; 3],
    const_args: [Option<i64>; 3],
}

enum SharedInput {
    Const(Type, i64),
    Common(Value),
    Param(usize),
}

struct ReturnTailSite {
    block: usize,
    args: [Value; 3],
    ret_vals: Vec<Value>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ReturnGroupKey {
    sig: TailOpSig,
    ret_slot: usize,
    ret_types: Vec<Type>,
    const_rets: Vec<Option<i64>>,
}

fn collect_consts(func: &Function) -> Vec<Option<i64>> {
    let mut consts = vec![None; func.types.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            let item = &func.insts[inst.index()];
            if let (Op::Const(imm), Some(res)) = (&item.op, item.result) {
                consts[res.index()] = Some(*imm);
            }
        }
    }
    consts
}

fn tail_op_sig(func: &Function, consts: &[Option<i64>], inst: Inst) -> Option<(TailOpSig, Value, [Value; 3])> {
    let item = &func.insts[inst.index()];
    let res = item.result?;
    if !item.op.is_speculable() {
        return None;
    }
    let (kind, arity, args) = match item.op {
        Op::Const(imm) => (TailKind::Const(imm), 0_u8, [Value(0); 3]),
        Op::Unary(u, x) => (TailKind::Unary(u), 1, [x, Value(0), Value(0)]),
        Op::Convert(c, x) => (TailKind::Convert(c), 1, [x, Value(0), Value(0)]),
        Op::Binary(b, x, y) => (TailKind::Binary(b), 2, [x, y, Value(0)]),
        Op::Icmp(cc, x, y) => (TailKind::Icmp(cc), 2, [x, y, Value(0)]),
        Op::Select(c, x, y) => (TailKind::Select, 3, [c, x, y]),
        _ => return None,
    };
    let mut arg_tys = [Type::I8; 3];
    let mut const_args = [None; 3];
    for slot in 0..arity as usize {
        arg_tys[slot] = func.ty(args[slot]);
        const_args[slot] = consts[args[slot].index()];
    }
    Some((TailOpSig { kind, res_ty: func.ty(res), arity, arg_tys, const_args }, res, args))
}

fn plan_shared_columns(columns: Vec<(Type, Option<i64>, Vec<Value>)>, max_params: usize) -> Option<(Vec<SharedInput>, Vec<(Type, Vec<Value>)>)> {
    let mut inputs = Vec::with_capacity(columns.len());
    let mut param_cols: Vec<(Type, Vec<Value>)> = Vec::new();
    for (ty, const_val, vals) in columns {
        if let Some(imm) = const_val {
            inputs.push(SharedInput::Const(ty, imm));
        } else if vals.iter().all(|&v| v == vals[0]) {
            inputs.push(SharedInput::Common(vals[0]));
        } else if let Some(existing) = param_cols.iter().position(|(p_ty, col)| *p_ty == ty && *col == vals) {
            inputs.push(SharedInput::Param(existing));
        } else {
            if param_cols.len() >= max_params {
                return None;
            }
            inputs.push(SharedInput::Param(param_cols.len()));
            param_cols.push((ty, vals));
        }
    }
    Some((inputs, param_cols))
}

fn emit_tail_op(kind: TailKind, args: &[Value]) -> Op {
    match kind {
        TailKind::Const(imm) => Op::Const(imm),
        TailKind::Unary(u) => Op::Unary(u, args[0]),
        TailKind::Convert(c) => Op::Convert(c, args[0]),
        TailKind::Binary(b) => Op::Binary(b, args[0], args[1]),
        TailKind::Icmp(cc) => Op::Icmp(cc, args[0], args[1]),
        TailKind::Select => Op::Select(args[0], args[1], args[2]),
    }
}

fn materialize_inputs(
    func: &mut Function,
    insts: &mut Vec<Inst>,
    inputs: &[SharedInput],
    params: &[Value],
) -> Vec<Value> {
    let mut emitted_consts: Vec<(Type, i64, Value)> = Vec::new();
    inputs
        .iter()
        .map(|input| match *input {
            SharedInput::Const(ty, imm) => {
                if let Some(&(_, _, existing)) = emitted_consts.iter().find(|&&(cty, cimm, _)| cty == ty && cimm == imm) {
                    return existing;
                }
                let val = func.value(ty);
                insts.push(func.add(Op::Const(imm), Some(val)));
                emitted_consts.push((ty, imm, val));
                val
            }
            SharedInput::Common(v) => v,
            SharedInput::Param(idx) => params[idx],
        })
        .collect()
}

fn prune_dead_consts(func: &mut Function, blocks: &[usize]) {
    let uses = func.use_counts();
    for &b in blocks {
        func.blocks[b].insts.retain(|&inst| {
            let item = &func.insts[inst.index()];
            !matches!(item.op, Op::Const(_)) || item.result.is_some_and(|r| uses[r.index()] > 0)
        });
    }
}

fn return_site_entry(func: &Function, consts: &[Option<i64>], uses: &[u32], block: usize) -> Option<(ReturnGroupKey, ReturnTailSite)> {
    let insts = &func.blocks[block].insts;
    let len = insts.len();
    if len < 2 {
        return None;
    }
    let Op::Return(ret_vals) = &func.insts[insts[len - 1].index()].op else { return None };
    let (sig, res, args) = tail_op_sig(func, consts, insts[len - 2])?;
    if uses[res.index()] != 1 {
        return None;
    }
    if matches!(sig.kind, TailKind::Const(imm) if ret_vals.len() == 1 && (-1..=0xffff).contains(&imm)) {
        return None;
    }
    let ret_slot = ret_vals.iter().position(|&v| v == res)?;
    let ret_types: Vec<Type> = ret_vals.iter().map(|&v| func.ty(v)).collect();
    let const_rets: Vec<Option<i64>> = ret_vals.iter().enumerate().map(|(r, &v)| (r != ret_slot).then(|| consts[v.index()]).flatten()).collect();
    let key = ReturnGroupKey { sig, ret_slot, ret_types, const_rets };
    let site = ReturnTailSite { block, args, ret_vals: ret_vals.clone() };
    Some((key, site))
}

fn factor_return_group(func: &mut Function, key: &ReturnGroupKey, group: &[ReturnTailSite]) -> bool {
    let arity = key.sig.arity as usize;
    let mut columns = Vec::with_capacity(arity + key.ret_types.len());
    for slot in 0..arity {
        let vals = group.iter().map(|s| s.args[slot]).collect();
        columns.push((key.sig.arg_tys[slot], key.sig.const_args[slot], vals));
    }
    for (r, &ty) in key.ret_types.iter().enumerate() {
        if r != key.ret_slot {
            let vals = group.iter().map(|s| s.ret_vals[r]).collect();
            columns.push((ty, key.const_rets[r], vals));
        }
    }
    let Some((inputs, param_cols)) = plan_shared_columns(columns, MAX_SINK_PARAMS) else { return false };
    let shared = Block(func.blocks.len() as u32);
    let params: Vec<Value> = param_cols.iter().map(|(ty, _)| func.value(*ty)).collect();
    let mut insts = Vec::with_capacity(4);
    let resolved = materialize_inputs(func, &mut insts, &inputs, &params);
    let shared_res = func.value(key.sig.res_ty);
    insts.push(func.add(emit_tail_op(key.sig.kind, &resolved[..arity]), Some(shared_res)));
    let mut ret_vals = Vec::with_capacity(key.ret_types.len());
    let mut extra_at = arity;
    for r in 0..key.ret_types.len() {
        if r == key.ret_slot {
            ret_vals.push(shared_res);
        } else {
            ret_vals.push(resolved[extra_at]);
            extra_at += 1;
        }
    }
    insts.push(func.add(Op::Return(ret_vals), None));
    func.blocks.push(BlockData { params, insts });
    let mut touched = Vec::with_capacity(group.len());
    for (site_idx, site) in group.iter().enumerate() {
        let jump_args: Vec<Value> = param_cols.iter().map(|(_, col)| col[site_idx]).collect();
        let jump = func.add(Op::Jump(shared, jump_args), None);
        let blk_insts = &mut func.blocks[site.block].insts;
        let len = blk_insts.len();
        blk_insts.truncate(len - 2);
        blk_insts.push(jump);
        touched.push(site.block);
    }
    prune_dead_consts(func, &touched);
    true
}

fn factor_common_returns(func: &mut Function, cfg_dirty: bool) -> bool {
    if cfg_dirty {
        func.bump_cfg();
    }
    let cfg = func.cfg();
    let consts = collect_consts(func);
    let uses = func.use_counts();
    let mut index: FxHashMap<ReturnGroupKey, usize> = FxHashMap::default();
    let mut groups: Vec<(ReturnGroupKey, Vec<ReturnTailSite>)> = Vec::new();
    for block in 0..func.blocks.len() {
        if !cfg.reachable[block] || func.entry == Some(Block(block as u32)) {
            continue;
        }
        let Some((key, site)) = return_site_entry(func, &consts, &uses, block) else { continue };
        match index.get(&key).copied() {
            Some(at) => groups[at].1.push(site),
            None => {
                index.insert(key.clone(), groups.len());
                groups.push((key, vec![site]));
            }
        }
    }
    let mut changed = false;
    for (key, group) in &groups {
        if group.len() >= MIN_TAIL_GROUP && factor_return_group(func, key, group) {
            changed = true;
        }
    }
    changed
}

struct JumpTailPlan {
    param_idx: usize,
    sig: TailOpSig,
    inputs: Vec<SharedInput>,
    param_cols: Vec<(Type, Vec<Value>)>,
}

fn collect_jump_preds(func: &Function, cfg: &Cfg, target: Block) -> Option<Vec<usize>> {
    let preds = cfg.preds_of(target);
    if preds.len() < 2 || preds.len() != cfg.pred_count(target) as usize {
        return None;
    }
    let param_len = func.blocks[target.index()].params.len();
    let mut out = Vec::with_capacity(preds.len());
    for (idx, &pred) in preds.iter().enumerate() {
        if pred == target || cfg.dominates(target, pred) || preds[..idx].contains(&pred) {
            return None;
        }
        let insts = &func.blocks[pred.index()].insts;
        if insts.len() < 2 {
            return None;
        }
        let Op::Jump(t, ref args) = func.insts[insts[insts.len() - 1].index()].op else { return None };
        if t != target || args.len() != param_len {
            return None;
        }
        out.push(pred.index());
    }
    Some(out)
}

fn find_jump_tail_plan(func: &Function, consts: &[Option<i64>], uses: &[u32], preds: &[usize], param_len: usize) -> Option<JumpTailPlan> {
    let mut pred_tails = Vec::with_capacity(preds.len());
    for &p in preds {
        let insts = &func.blocks[p].insts;
        let tail = insts[insts.len() - 2];
        let (sig, res, args) = tail_op_sig(func, consts, tail)?;
        if uses[res.index()] != 1 {
            return None;
        }
        let Op::Jump(_, ref j_args) = func.insts[insts[insts.len() - 1].index()].op else { return None };
        pred_tails.push((sig, res, args, j_args.as_slice()));
    }
    let first_sig = &pred_tails[0].0;
    if !pred_tails[1..].iter().all(|(sig, ..)| sig == first_sig) {
        return None;
    }
    let param_idx = (0..param_len).find(|&idx| pred_tails.iter().all(|(_, res, _, j_args)| j_args[idx] == *res))?;
    let arity = first_sig.arity as usize;
    let columns = (0..arity)
        .map(|slot| {
            let vals = pred_tails.iter().map(|(_, _, args, _)| args[slot]).collect();
            (first_sig.arg_tys[slot], first_sig.const_args[slot], vals)
        })
        .collect();
    let (inputs, param_cols) = plan_shared_columns(columns, 1)?;
    Some(JumpTailPlan { param_idx, sig: first_sig.clone(), inputs, param_cols })
}

fn apply_jump_tail_plan(func: &mut Function, target_idx: usize, preds: &[usize], plan: JumpTailPlan) {
    let new_params: Vec<Value> = plan.param_cols.iter().map(|(ty, _)| func.value(*ty)).collect();
    let mut prepended = Vec::with_capacity(4);
    let resolved = materialize_inputs(func, &mut prepended, &plan.inputs, &new_params);
    let new_res = func.value(plan.sig.res_ty);
    prepended.push(func.add(emit_tail_op(plan.sig.kind, &resolved), Some(new_res)));
    prepended.append(&mut func.blocks[target_idx].insts);
    func.blocks[target_idx].insts = prepended;
    let old_param = func.blocks[target_idx].params.remove(plan.param_idx);
    func.blocks[target_idx].params.extend(new_params);
    for (pred_pos, &pred_idx) in preds.iter().enumerate() {
        let len = func.blocks[pred_idx].insts.len();
        let term = func.blocks[pred_idx].insts[len - 1];
        func.blocks[pred_idx].insts.remove(len - 2);
        if let Op::Jump(_, ref mut args) = func.insts[term.index()].op {
            args.remove(plan.param_idx);
            args.extend(plan.param_cols.iter().map(|(_, col)| col[pred_pos]));
        }
    }
    let mut subst = crate::merge::identity(func.types.len());
    subst[old_param.index()] = new_res;
    crate::merge::rename(func, &subst);
    prune_dead_consts(func, preds);
}

fn sink_common_jump_tails(func: &mut Function, cfg_dirty: bool) -> bool {
    if cfg_dirty {
        func.bump_cfg();
    }
    let cfg = func.cfg();
    let mut consts = collect_consts(func);
    let mut uses = func.use_counts();
    let mut changed = false;
    for j_idx in 0..func.blocks.len() {
        let target = Block(j_idx as u32);
        let param_len = func.blocks[j_idx].params.len();
        if !cfg.reachable[j_idx] || func.entry == Some(target) || param_len == 0 {
            continue;
        }
        let Some(preds) = collect_jump_preds(func, &cfg, target) else { continue };
        let Some(plan) = find_jump_tail_plan(func, &consts, &uses, &preds, param_len) else { continue };
        apply_jump_tail_plan(func, j_idx, &preds, plan);
        consts = collect_consts(func);
        uses = func.use_counts();
        changed = true;
    }
    changed
}
