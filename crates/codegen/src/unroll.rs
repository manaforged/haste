use crate::fold::{evaluate, extend};
use crate::ir::{Block, Function, Op, Value};

fn remap_val(map: &[(Value, Value)], v: Value) -> Value {
    map.iter().rev().find(|&&(old, _)| old == v).map_or(v, |&(_, new)| new)
}

fn eval_step(func: &Function, block_idx: usize, cond: Value, back_on_true: bool, back_args: &[Value], params: &[Option<i64>], env: &mut [Option<i64>]) -> Option<(bool, Vec<Option<i64>>)> {
    let block = &func.blocks[block_idx];
    for (p, val) in block.params.iter().zip(params) {
        env[p.index()] = *val;
    }
    for &inst in &block.insts[..block.insts.len() - 1] {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        let val = match data.op {
            Op::Select(c, x, y) => env[c.index()].and_then(|cv| if cv != 0 { env[x.index()] } else { env[y.index()] }),
            _ => evaluate(func, &data.op, &|v| env[v.index()]),
        };
        env[res.index()] = val.map(|imm| extend(imm, func.ty(res), true));
    }
    let cond_val = env[cond.index()]?;
    let takes_back = (cond_val != 0) == back_on_true;
    let next_params = back_args.iter().map(|v| env[v.index()]).collect();
    Some((takes_back, next_params))
}

fn single_outside_pred(func: &Function, head: Block) -> Option<Vec<Value>> {
    let mut found = None;
    for (p, block) in func.blocks.iter().enumerate() {
        if p == head.index() || !func.successors(Block(p as u32)).any(|t| t == head) {
            continue;
        }
        if found.is_some() {
            return None;
        }
        let Some(&last) = block.insts.last() else { return None };
        let Op::Jump(target, args) = &func.insts[last.index()].op else { return None };
        if *target != head {
            return None;
        }
        found = Some(args.clone());
    }
    found
}

/// Unrolls single-block self-loops with a statically provable constant trip count in `1..=4`.
pub(crate) fn run(func: &mut Function) -> bool {
    let Some(entry) = func.entry else { return false };
    let preds = func.pred_counts();
    let mut global_consts = vec![None; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        if let (Op::Const(imm), Some(res)) = (&data.op, data.result) {
            global_consts[res.index()] = Some(*imm);
        }
    }
    let mut changed = false;
    for b in 0..func.blocks.len() {
        let head = Block(b as u32);
        if head == entry || preds[b] != 2 || func.blocks[b].insts.is_empty() || func.blocks[b].insts.len() > 32 {
            continue;
        }
        let Some(&last) = func.blocks[b].insts.last() else { continue };
        let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = &func.insts[last.index()].op else { continue };
        let (back_on_true, back_args, exit_blk, exit_args) = if *t_blk == head && *f_blk != head {
            (true, t_args.clone(), *f_blk, f_args.clone())
        } else if *f_blk == head && *t_blk != head {
            (false, f_args.clone(), *t_blk, t_args.clone())
        } else {
            continue;
        };
        let cond = *cond;
        let bad_op = func.blocks[b].insts[..func.blocks[b].insts.len() - 1]
            .iter()
            .any(|&i| matches!(func.insts[i.index()].op, Op::Call { .. } | Op::TryCall { .. } | Op::BrTable(..) | Op::Jump(..)));
        if bad_op {
            continue;
        }
        let Some(init_args) = single_outside_pred(func, head) else { continue };
        if init_args.len() != func.blocks[b].params.len() || back_args.len() != func.blocks[b].params.len() {
            continue;
        }
        global_consts.resize(func.types.len(), None);
        let mut cur_params: Vec<Option<i64>> = init_args.iter().map(|v| global_consts[v.index()]).collect();
        let mut proven = None;
        for step in 0..4_usize {
            let mut env = global_consts.clone();
            let Some((takes_back, next_params)) = eval_step(func, b, cond, back_on_true, &back_args, &cur_params, &mut env) else {
                break;
            };
            if !takes_back {
                proven = Some(step);
                break;
            }
            cur_params = next_params;
        }
        let Some(back_steps) = proven else { continue };
        if back_steps == 0 {
            func.insts[last.index()].op = Op::Jump(exit_blk, exit_args);
            changed = true;
            continue;
        }
        let inst_count = func.blocks[b].insts.len();
        if (back_steps + 1) * inst_count > 64 {
            continue;
        }
        let body_insts = func.blocks[b].insts[..inst_count - 1].to_vec();
        func.blocks[b].insts.pop();
        let mut cur_back_args = back_args.clone();
        let mut last_map = Vec::new();
        for _ in 0..back_steps {
            let mut map: Vec<(Value, Value)> = func.blocks[b].params.iter().copied().zip(cur_back_args.iter().copied()).collect();
            for &orig_inst in &body_insts {
                let mut op = func.insts[orig_inst.index()].op.clone();
                op.map_operands(|v| remap_val(&map, v));
                let new_res = func.insts[orig_inst.index()].result.map(|r| {
                    let nr = func.value(func.ty(r));
                    map.push((r, nr));
                    nr
                });
                let ni = func.add(op, new_res);
                func.blocks[b].insts.push(ni);
            }
            cur_back_args = back_args.iter().map(|&v| remap_val(&map, v)).collect();
            last_map = map;
        }
        let final_exit_args: Vec<Value> = exit_args.iter().map(|&v| remap_val(&last_map, v)).collect();
        let jump = func.add(Op::Jump(exit_blk, final_exit_args), None);
        func.blocks[b].insts.push(jump);
        for ob in 0..func.blocks.len() {
            if ob != b {
                for idx in 0..func.blocks[ob].insts.len() {
                    let inst = func.blocks[ob].insts[idx];
                    func.insts[inst.index()].op.map_operands(|v| remap_val(&last_map, v));
                }
            }
        }
        changed = true;
    }
    changed
}
