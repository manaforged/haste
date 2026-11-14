use crate::ir::{Block, Domain, Function, Inst, Op, Value};

const MAX_RETURN_PREFIX: usize = 3;

/// Turns every jump into a tiny return block into that return, so the edge moves no values and takes no branch.
pub(crate) fn run(func: &mut Function) -> Domain {
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
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
