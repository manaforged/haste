use crate::ir::{ArgumentPurpose, Block, BlockData, Domain, Function, Inst, Op, SigRef, Signature, Type, Value};
use crate::reloc::RuntimeSymbol;
use rustc_hash::FxHashMap;

const MAX_RETURN_PREFIX: usize = 3;
const MIN_TAIL_GROUP: usize = 3;
const MAX_TAIL_PARAMS: usize = 4;

/// Turns every jump into a tiny return block into that return, then factors identical tail calls across arms.
pub(crate) fn run(func: &mut Function) -> Domain {
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
    let mut changed = crate::merge::lower_binary_br_tables(func);
    let cfg = func.cfg();
    let mut threaded = vec![0_u32; func.blocks.len()];
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
    if factor_tail_calls(func, changed) {
        changed = true;
    }
    if changed {
        func.bump_cfg();
        Domain::CFG | Domain::VALUES
    } else {
        Domain::NONE
    }
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
