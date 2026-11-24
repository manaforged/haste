use crate::cfg::{Cfg, Loop, Loops};
use crate::ir::{Block, Function, Inst, Op, Value};

use super::ArmMatch;

fn edge_args_to(func: &Function, pred: Block, target: Block) -> Option<&[Value]> {
    match func.last(pred) {
        Some(Op::Jump(t, args)) if *t == target => Some(args.as_slice()),
        Some(Op::Brif(_, t, ta, f, fa)) if (*t == target) ^ (*f == target) => {
            Some(if *t == target {
                ta.as_slice()
            } else {
                fa.as_slice()
            })
        }
        _ => None,
    }
}

fn cond_reads_loop_load(
    func: &Function,
    cfg: &Cfg,
    each: &Loop,
    (defs, def_blocks): (&[Option<Inst>], &[Option<Block>]),
    cond: Value,
) -> bool {
    let mut seen = vec![cond];
    let mut work = vec![cond];
    while let Some(v) = work.pop() {
        let Some(db) = def_blocks.get(v.index()).copied().flatten() else {
            continue;
        };
        if !each.body.get(db.index()).copied().unwrap_or(false) {
            continue;
        }
        if let Some(inst) = defs.get(v.index()).copied().flatten() {
            let op = &func.insts[inst.index()].op;
            if matches!(op, Op::Load(..) | Op::StackLoad(..) | Op::AtomicLoad(..)) {
                return true;
            }
            op.for_operands(|u| {
                if !seen.contains(&u) {
                    seen.push(u);
                    work.push(u);
                }
            });
        } else if db != each.head
            && let Some(p_idx) = func.blocks[db.index()].params.iter().position(|&p| p == v)
        {
            for &pred in cfg.preds_of(db) {
                if each.body.get(pred.index()).copied().unwrap_or(false)
                    && let Some(&arg) = edge_args_to(func, pred, db).and_then(|a| a.get(p_idx))
                    && !seen.contains(&arg)
                {
                    seen.push(arg);
                    work.push(arg);
                }
            }
        }
    }
    false
}

fn propagate_loop_taint(
    func: &Function,
    cfg: &Cfg,
    each: &Loop,
    tainted: &mut [bool],
    check_load_addr: bool,
) -> (bool, Vec<Value>) {
    let mut head_phis = Vec::new();
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &cfg.rpo {
            if !each.body.get(b.index()).copied().unwrap_or(false) {
                continue;
            }
            for &inst in &func.blocks[b.index()].insts {
                let data = &func.insts[inst.index()];
                if check_load_addr
                    && let Op::Load(addr, _) | Op::AtomicLoad(addr, _) = data.op
                    && tainted[addr.index()]
                {
                    return (true, head_phis);
                }
                if !matches!(data.op, Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..))
                    && data.op.is_speculable()
                {
                    let mut hit = false;
                    match data.op {
                        Op::Select(_, t, f) => hit = tainted[t.index()] || tainted[f.index()],
                        _ => data.op.for_operands(|v| hit |= tainted[v.index()]),
                    }
                    if hit
                        && let Some(res) = data.result
                        && !std::mem::replace(&mut tainted[res.index()], true)
                    {
                        changed = true;
                    }
                }
            }
            let mut propagate_edge = |target: Block, args: &[Value]| {
                if !each.body.get(target.index()).copied().unwrap_or(false) {
                    return;
                }
                let t_params = &func.blocks[target.index()].params;
                for (&p, &arg) in t_params.iter().zip(args) {
                    if !tainted[arg.index()] {
                        continue;
                    }
                    if target == each.head {
                        if !check_load_addr && !head_phis.contains(&p) {
                            head_phis.push(p);
                        }
                    } else if !std::mem::replace(&mut tainted[p.index()], true) {
                        changed = true;
                    }
                }
            };
            match func.last(b) {
                Some(Op::Jump(target, args)) => propagate_edge(*target, args),
                Some(Op::Brif(_, t, ta, f, fa)) => {
                    propagate_edge(*t, ta);
                    propagate_edge(*f, fa);
                }
                _ => {}
            }
        }
    }
    (false, head_phis)
}

pub(super) fn feeds_load_addr_cycle(
    func: &Function,
    cfg: &Cfg,
    loop_info: Option<(&Loops, &[Option<Inst>], &[Option<Block>])>,
    head: Block,
    cond: Value,
    m: &ArmMatch,
) -> bool {
    let Some((loops, defs, def_blocks)) = loop_info else {
        return false;
    };
    let join_params = &func.blocks[m.join.index()].params;
    for each in &loops.list {
        if !each.body.get(head.index()).copied().unwrap_or(false)
            || !each.body.get(m.join.index()).copied().unwrap_or(false)
        {
            continue;
        }
        if !cond_reads_loop_load(func, cfg, each, (defs, def_blocks), cond) {
            continue;
        }
        let selected: Vec<Value> = join_params
            .iter()
            .zip(m.t_args.iter().zip(&m.e_args))
            .filter_map(|(&p, (t, e))| (t != e).then_some(p))
            .collect();
        if selected.is_empty() {
            continue;
        }
        let head_phis = if m.join == each.head {
            selected
        } else {
            let mut tainted = vec![false; func.types.len()];
            for p in selected {
                tainted[p.index()] = true;
            }
            propagate_loop_taint(func, cfg, each, &mut tainted, false).1
        };
        if head_phis.is_empty() {
            continue;
        }
        let mut tainted = vec![false; func.types.len()];
        for p in head_phis {
            tainted[p.index()] = true;
        }
        if propagate_loop_taint(func, cfg, each, &mut tainted, true).0 {
            return true;
        }
    }
    false
}
