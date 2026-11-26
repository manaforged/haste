use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, Op, Value};

fn jump_only_target(func: &Function, block: Block) -> Option<(Block, &[Value])> {
    let [inst] = func.blocks[block.index()].insts[..] else {
        return None;
    };
    let Op::Jump(target, ref args) = func.insts[inst.index()].op else {
        return None;
    };
    (target != block).then_some((target, args.as_slice()))
}

fn trampoline_target<'a>(
    func: &'a Function,
    block: Block,
    is_trampoline: &[bool],
) -> Option<(Block, &'a [Value])> {
    is_trampoline[block.index()]
        .then(|| jump_only_target(func, block))
        .flatten()
}

fn forward_edge(
    func: &Function,
    is_trampoline: &[bool],
    target: &mut Block,
    args: &mut Vec<Value>,
    require_empty: bool,
) -> bool {
    let (mut cur, mut cur_args) = (*target, args.clone());
    let mut visited = [usize::MAX; 16];
    let mut changed = false;
    for step in 0..16 {
        let Some((next, jump_args)) = trampoline_target(func, cur, is_trampoline) else {
            break;
        };
        if visited[..step].contains(&next.index()) {
            break;
        }
        let params = &func.blocks[cur.index()].params;
        if params.len() != cur_args.len() {
            break;
        }
        let composed: Vec<Value> = jump_args
            .iter()
            .map(|&v| {
                params
                    .iter()
                    .position(|&p| p == v)
                    .map_or(v, |idx| cur_args[idx])
            })
            .collect();
        if require_empty && !composed.is_empty() {
            break;
        }
        visited[step] = cur.index();
        cur = next;
        cur_args = composed;
        changed = true;
    }
    if changed {
        *target = cur;
        *args = cur_args;
    }
    changed
}

fn jumps_onward(func: &Function, entry: Option<usize>, b: usize) -> bool {
    Some(b) != entry && jump_only_target(func, Block(b as u32)).is_some()
}

fn outside_param_uses(func: &Function, cfg: &Cfg, param_owner: &[usize]) -> Vec<bool> {
    let mut used_outside = vec![false; func.types.len()];
    for (b, data) in func.blocks.iter().enumerate() {
        if !cfg.reachable[b] {
            continue;
        }
        for &inst in &data.insts {
            func.insts[inst.index()].op.for_operands(|v| {
                let owner = param_owner[v.index()];
                if owner != usize::MAX && owner != b {
                    used_outside[v.index()] = true;
                }
            });
        }
    }
    used_outside
}

fn compute_trampoline_mask(func: &Function, cfg: &Cfg, entry: Option<usize>) -> Option<Vec<bool>> {
    let n = func.blocks.len();
    if !(0..n).any(|b| cfg.reachable[b] && jumps_onward(func, entry, b)) {
        return None;
    }
    let mut param_owner = vec![usize::MAX; func.types.len()];
    let mut has_params = false;
    for (b, data) in func.blocks.iter().enumerate() {
        if cfg.reachable[b] {
            for &p in &data.params {
                param_owner[p.index()] = b;
                has_params = true;
            }
        }
    }
    let used_outside = if has_params {
        outside_param_uses(func, cfg, &param_owner)
    } else {
        Vec::new()
    };
    let mask: Vec<bool> = (0..n)
        .map(|b| {
            cfg.reachable[b]
                && jumps_onward(func, entry, b)
                && func.blocks[b]
                    .params
                    .iter()
                    .all(|p| !used_outside[p.index()])
        })
        .collect();
    mask.iter().any(|&t| t).then_some(mask)
}

fn collapse_identical_brifs(func: &mut Function, cfg: &Cfg) -> bool {
    let mut changed = false;
    for b in 0..func.blocks.len() {
        if !cfg.reachable[b] {
            continue;
        }
        let Some(&term) = func.blocks[b].insts.last() else {
            continue;
        };
        if let Op::Brif(_, then, ref then_args, otherwise, ref else_args) =
            func.insts[term.index()].op
            && then == otherwise
            && then_args == else_args
        {
            let args = then_args.clone();
            func.insts[term.index()].op = Op::Jump(then, args);
            changed = true;
        }
    }
    changed
}

fn forward_term_op(func: &Function, is_trampoline: &[bool], op: &mut Op) -> bool {
    match op {
        Op::Jump(target, args) => forward_edge(func, is_trampoline, target, args, false),
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            let mut did = forward_edge(func, is_trampoline, then, then_args, false);
            did |= forward_edge(func, is_trampoline, otherwise, else_args, false);
            if then == otherwise && then_args == else_args {
                *op = Op::Jump(*then, std::mem::take(then_args));
                did = true;
            }
            did
        }
        Op::BrTable(_, default, targets) => {
            let mut empty = Vec::new();
            let mut did = forward_edge(func, is_trampoline, default, &mut empty, true);
            for target in targets {
                let mut empty = Vec::new();
                did |= forward_edge(func, is_trampoline, target, &mut empty, true);
            }
            did
        }
        Op::TryCall { normal, .. } => {
            let mut empty = Vec::new();
            forward_edge(func, is_trampoline, normal, &mut empty, true)
        }
        _ => false,
    }
}

pub(super) fn forward_trampolines(func: &mut Function) -> bool {
    if func.coroutine.is_some() {
        return false;
    }
    let cfg = func.cfg();
    let collapsed = collapse_identical_brifs(func, &cfg);
    let Some(is_trampoline) = compute_trampoline_mask(func, &cfg, func.entry.map(Block::index))
    else {
        return collapsed;
    };
    let terms: Vec<Option<(usize, Inst)>> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(b, data)| data.insts.last().copied().map(|i| (b, i)))
        .collect();
    let mut forwarded = false;
    for &(b, term) in terms.iter().flatten() {
        if !cfg.reachable[b] {
            continue;
        }
        let mut op = func.insts[term.index()].op.clone();
        if forward_term_op(func, &is_trampoline, &mut op) {
            func.insts[term.index()].op = op;
            forwarded = true;
        }
    }
    if !forwarded {
        return collapsed;
    }
    let mut referenced = vec![false; func.blocks.len()];
    for (_, term) in terms.into_iter().flatten() {
        func.insts[term.index()]
            .op
            .targets()
            .for_each(|b| referenced[b.index()] = true);
    }
    for (b, is_t) in is_trampoline.into_iter().enumerate() {
        if is_t && !referenced[b] {
            if let Some(inst) = func.blocks[b].insts.pop() {
                func.insts[inst.index()].op = Op::Trap;
            }
            func.blocks[b].params.clear();
        }
    }
    true
}
