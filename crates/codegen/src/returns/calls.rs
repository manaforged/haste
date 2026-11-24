use rustc_hash::FxHashMap;

use crate::ir::{
    ArgumentPurpose, Block, BlockData, Function, Inst, Op, SigRef, Signature, Type, Value,
};
use crate::reloc::RuntimeSymbol;

use super::MIN_TAIL_GROUP;

const MAX_TAIL_PARAMS: usize = 4;

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

pub(super) fn factor_tail_calls(func: &mut Function, cfg_dirty: bool) -> bool {
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
        let Some((key, site)) = tail_site(func, block, &defs) else {
            continue;
        };
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

fn tail_site(
    func: &Function,
    block: usize,
    defs: &[Option<(CalleeKey, usize, Inst)>],
) -> Option<(TailKey, TailSite)> {
    let insts = &func.blocks[block].insts;
    let len = insts.len();
    if len < 2 {
        return None;
    }
    let Op::Return(ret_vals) = &func.insts[insts[len - 1].index()].op else {
        return None;
    };
    let Op::Call {
        sig,
        callee,
        args,
        results,
    } = &func.insts[insts[len - 2].index()].op
    else {
        return None;
    };
    let sig_data = &func.sigs[sig.index()];
    if args.len() != sig_data.params.len()
        || results.len() != sig_data.returns.len()
        || results != ret_vals
        || sig_data
            .params
            .iter()
            .any(|p| p.purpose == ArgumentPurpose::Variadic)
    {
        return None;
    }
    let ret_types: Vec<Type> = results.iter().map(|&v| func.ty(v)).collect();
    let (key, callee_def) = match defs[callee.index()] {
        Some((key, def_block, def_inst)) => (key, Some((def_block, def_inst))),
        None => (CalleeKey::Value(*callee), None),
    };
    let tail_key = TailKey {
        callee: key,
        callee_ty: func.ty(*callee),
        sig: sig_data.clone(),
        ret_types: ret_types.clone(),
    };
    let site = TailSite {
        block,
        sig: *sig,
        callee: *callee,
        args: args.clone(),
        ret_types,
        callee_def,
    };
    Some((tail_key, site))
}

fn varying_arg_positions(func: &Function, group: &[TailSite]) -> Option<Vec<usize>> {
    let arity = group[0].args.len();
    let mut varying = Vec::new();
    for pos in 0..arity {
        let first = group[0].args[pos];
        if group[1..]
            .iter()
            .any(|site| func.ty(site.args[pos]) != func.ty(first))
        {
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
    let params: Vec<Value> = varying
        .iter()
        .map(|&pos| func.value(func.ty(group[0].args[pos])))
        .collect();
    let mut insts = Vec::with_capacity(3);
    let same_callee = group[1..].iter().all(|site| site.callee == group[0].callee);
    let callee = if same_callee {
        group[0].callee
    } else {
        emit_callee_def(func, key, &mut insts)
    };
    let mut call_args = group[0].args.clone();
    for (param_idx, &arg_pos) in varying.iter().enumerate() {
        call_args[arg_pos] = params[param_idx];
    }
    let results: Vec<Value> = group[0]
        .ret_types
        .iter()
        .map(|&ty| func.value(ty))
        .collect();
    insts.push(func.add(
        Op::Call {
            sig: group[0].sig,
            callee,
            args: call_args,
            results: results.clone(),
        },
        None,
    ));
    insts.push(func.add(Op::Return(results), None));
    func.blocks.push(BlockData { params, insts });
    for site in group {
        rewrite_tail_site(
            func,
            site,
            (shared, &varying),
            (!same_callee).then_some(uses),
        );
    }
    true
}

fn rewrite_tail_site(
    func: &mut Function,
    site: &TailSite,
    (shared, varying): (Block, &[usize]),
    uses: Option<&[u32]>,
) {
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
