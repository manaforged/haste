use super::{
    MAX_CHAIN_TRIPS, MAX_CHAIN_UNROLLED_INSTS, MAX_FP_FULL_INSTS, MAX_FP_FULL_TRIPS,
    MAX_FULL_INSTS, MAX_FULL_TRIPS, MAX_VEC_FULL_TRIPS, block_has_float, block_has_libcall_mem,
    block_has_vec, clone_body, eval_block, has_bad_non_term, outside_const_params, pop_terminator,
    push_params_map, remap_other_blocks, remap_val,
};
use crate::cfg::{Cfg, Loop};
use crate::fold::evaluate_wrapped;
use crate::ir::{Block, Function, Inst, Op, Value};

struct ChainSeg {
    blk: Block,
    params: Vec<Value>,
    non_term: Vec<Inst>,
    edge_args: Vec<Value>,
}

enum LoopChainKind {
    TopTested {
        cond: Value,
        body_on_true: bool,
        exit_blk: Block,
        exit_args: Vec<Value>,
    },
    BottomTested {
        cond: Value,
        back_on_true: bool,
        exit_blk: Block,
        exit_args: Vec<Value>,
    },
}

struct LoopChain {
    segs: Vec<ChainSeg>,
    kind: LoopChainKind,
}

fn retire_block(func: &mut Function, blk_idx: usize) {
    for inst in std::mem::take(&mut func.blocks[blk_idx].insts) {
        if func.insts[inst.index()].op.is_terminator() {
            func.insts[inst.index()].op = Op::Trap;
        }
    }
    func.blocks[blk_idx].params.clear();
}

fn seg_from_block(func: &Function, blk: Block, edge_args: Vec<Value>) -> ChainSeg {
    let data = &func.blocks[blk.index()];
    ChainSeg {
        blk,
        params: data.params.clone(),
        non_term: data.insts[..data.insts.len() - 1].to_vec(),
        edge_args,
    }
}

fn clean_chain_block(func: &Function, cfg: &Cfg, each: &Loop, blk: Block) -> bool {
    let idx = blk.index();
    cfg.reachable.get(idx).copied().unwrap_or(false)
        && each.body.get(idx).copied().unwrap_or(false)
        && func.entry != Some(blk)
        && (blk == each.head || cfg.pred_count(blk) == 1)
        && !func.blocks[idx].insts.is_empty()
        && !has_bad_non_term(func, idx)
}

fn collect_chain(func: &Function, cfg: &Cfg, each: &Loop) -> Option<LoopChain> {
    let head = each.head;
    let latch = each.latches[0];
    if head == latch || !clean_chain_block(func, cfg, each, head) {
        return None;
    }
    let head_last = *func.blocks[head.index()].insts.last()?;
    let (mut cur, mut cur_args, mut segs, is_top, top_meta) =
        match &func.insts[head_last.index()].op {
            Op::Jump(next_blk, next_args) if *next_blk != head => (
                *next_blk,
                next_args.clone(),
                vec![seg_from_block(func, head, next_args.clone())],
                false,
                None,
            ),
            Op::Brif(cond, t_blk, t_args, f_blk, f_args) if t_blk != f_blk => {
                let t_in = each.body.get(t_blk.index()).copied().unwrap_or(false);
                let f_in = each.body.get(f_blk.index()).copied().unwrap_or(false);
                let (body_on_true, body_blk, body_args, exit_blk, exit_args) =
                    if t_in && !f_in && *t_blk != head {
                        (true, *t_blk, t_args.clone(), *f_blk, f_args.clone())
                    } else if f_in && !t_in && *f_blk != head {
                        (false, *f_blk, f_args.clone(), *t_blk, t_args.clone())
                    } else {
                        return None;
                    };
                (
                    body_blk,
                    body_args.clone(),
                    vec![seg_from_block(func, head, body_args)],
                    true,
                    Some((*cond, body_on_true, exit_blk, exit_args)),
                )
            }
            _ => return None,
        };

    loop {
        if segs.len() >= each.size
            || segs.iter().any(|s| s.blk == cur)
            || !clean_chain_block(func, cfg, each, cur)
            || cur_args.len() != func.blocks[cur.index()].params.len()
        {
            return None;
        }
        let last = *func.blocks[cur.index()].insts.last()?;
        if cur != latch {
            let Op::Jump(nxt, nxt_args) = &func.insts[last.index()].op else {
                return None;
            };
            let nxt_args = nxt_args.clone();
            segs.push(seg_from_block(func, cur, nxt_args.clone()));
            cur = *nxt;
            cur_args = nxt_args;
            continue;
        }
        if segs.len() + 1 != each.size {
            return None;
        }
        if is_top {
            let Op::Jump(target, back_args) = &func.insts[last.index()].op else {
                return None;
            };
            if *target != head || back_args.len() != func.blocks[head.index()].params.len() {
                return None;
            }
            segs.push(seg_from_block(func, cur, back_args.clone()));
            let (cond, body_on_true, exit_blk, exit_args) = top_meta?;
            return Some(LoopChain {
                segs,
                kind: LoopChainKind::TopTested {
                    cond,
                    body_on_true,
                    exit_blk,
                    exit_args,
                },
            });
        }
        let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = &func.insts[last.index()].op else {
            return None;
        };
        if t_blk == f_blk {
            return None;
        }
        let (back_on_true, back_args, exit_blk, exit_args) = if *t_blk == head {
            (true, t_args.clone(), *f_blk, f_args.clone())
        } else if *f_blk == head {
            (false, f_args.clone(), *t_blk, t_args.clone())
        } else {
            return None;
        };
        if each.body.get(exit_blk.index()).copied().unwrap_or(false)
            || back_args.len() != func.blocks[head.index()].params.len()
        {
            return None;
        }
        segs.push(seg_from_block(func, cur, back_args));
        return Some(LoopChain {
            segs,
            kind: LoopChainKind::BottomTested {
                cond: *cond,
                back_on_true,
                exit_blk,
                exit_args,
            },
        });
    }
}

fn eval_chain(
    func: &Function,
    chain: &LoopChain,
    init_params: Vec<Option<i64>>,
    global_consts: &[Option<i64>],
    max_trips: usize,
) -> Option<usize> {
    let mut cur_head_params = init_params;
    match &chain.kind {
        LoopChainKind::TopTested {
            cond, body_on_true, ..
        } => {
            for step in 0..=max_trips {
                let mut env = global_consts.to_vec();
                eval_block(
                    func,
                    chain.segs[0].blk.index(),
                    &cur_head_params,
                    &mut env,
                    evaluate_wrapped,
                );
                let cond_val = env[cond.index()]?;
                if (cond_val != 0) != *body_on_true {
                    return Some(step);
                }
                if step == max_trips {
                    return None;
                }
                let mut cur_args: Vec<Option<i64>> = chain.segs[0]
                    .edge_args
                    .iter()
                    .map(|v| env[v.index()])
                    .collect();
                for seg in &chain.segs[1..] {
                    eval_block(func, seg.blk.index(), &cur_args, &mut env, evaluate_wrapped);
                    cur_args = seg.edge_args.iter().map(|v| env[v.index()]).collect();
                }
                cur_head_params = cur_args;
            }
            None
        }
        LoopChainKind::BottomTested {
            cond, back_on_true, ..
        } => {
            for step in 0..max_trips {
                let mut env = global_consts.to_vec();
                let mut cur_args = cur_head_params;
                for seg in &chain.segs {
                    eval_block(func, seg.blk.index(), &cur_args, &mut env, evaluate_wrapped);
                    cur_args = seg.edge_args.iter().map(|v| env[v.index()]).collect();
                }
                let cond_val = env[cond.index()]?;
                if (cond_val != 0) != *back_on_true {
                    return Some(step);
                }
                cur_head_params = cur_args;
            }
            None
        }
    }
}

fn finish_chain_exit(
    func: &mut Function,
    b: usize,
    segs: &[ChainSeg],
    (exit_blk, exit_args): (Block, &[Value]),
    skip: &[usize],
    map: &[(Value, Value)],
) -> bool {
    let final_exit_args: Vec<Value> = exit_args.iter().map(|&v| remap_val(map, v)).collect();
    let jump = func.add(Op::Jump(exit_blk, final_exit_args), None);
    func.blocks[b].insts.push(jump);
    remap_other_blocks(func, skip, map);
    for seg in &segs[1..] {
        retire_block(func, seg.blk.index());
    }
    true
}

fn emit_top_chain(
    func: &mut Function,
    segs: &[ChainSeg],
    trips: usize,
    (exit_blk, exit_args): (Block, Vec<Value>),
    skip: &[usize],
) -> bool {
    let head = segs[0].blk;
    let b = head.index();
    if trips == 0 {
        let last = *func.blocks[b].insts.last().unwrap_or(&Inst(0));
        func.insts[last.index()].op = Op::Jump(exit_blk, exit_args);
        segs[1..]
            .iter()
            .for_each(|seg| retire_block(func, seg.blk.index()));
        return true;
    }
    pop_terminator(func, b);
    let mut head_map = Vec::new();
    let mut cur_body_args = segs[0].edge_args.clone();
    for _ in 0..trips {
        let mut iter_map = head_map.clone();
        for seg in &segs[1..] {
            push_params_map(&mut iter_map, &seg.params, &cur_body_args);
            clone_body(func, &seg.non_term, head, &mut iter_map);
            cur_body_args = seg
                .edge_args
                .iter()
                .map(|&v| remap_val(&iter_map, v))
                .collect();
        }
        let mut next_head_map = Vec::new();
        push_params_map(&mut next_head_map, &segs[0].params, &cur_body_args);
        clone_body(func, &segs[0].non_term, head, &mut next_head_map);
        head_map = next_head_map;
        cur_body_args = segs[0]
            .edge_args
            .iter()
            .map(|&v| remap_val(&head_map, v))
            .collect();
    }
    finish_chain_exit(func, b, segs, (exit_blk, &exit_args), skip, &head_map)
}

fn emit_bottom_chain(
    func: &mut Function,
    segs: &[ChainSeg],
    proven: usize,
    (exit_blk, exit_args): (Block, Vec<Value>),
    skip: &[usize],
) -> bool {
    let head = segs[0].blk;
    pop_terminator(func, head.index());
    let mut last_map = Vec::new();
    let mut cur_edge_args = segs[0].edge_args.clone();
    for seg in &segs[1..] {
        push_params_map(&mut last_map, &seg.params, &cur_edge_args);
        clone_body(func, &seg.non_term, head, &mut last_map);
        cur_edge_args = seg
            .edge_args
            .iter()
            .map(|&v| remap_val(&last_map, v))
            .collect();
    }
    for _ in 0..proven {
        let mut step_map = Vec::new();
        for seg in segs {
            push_params_map(&mut step_map, &seg.params, &cur_edge_args);
            clone_body(func, &seg.non_term, head, &mut step_map);
            cur_edge_args = seg
                .edge_args
                .iter()
                .map(|&v| remap_val(&step_map, v))
                .collect();
        }
        last_map = step_map;
    }
    finish_chain_exit(
        func,
        head.index(),
        segs,
        (exit_blk, &exit_args),
        skip,
        &last_map,
    )
}

fn chain_fits(func: &Function, segs: &[ChainSeg], body_insts: usize, trips: usize) -> bool {
    if trips == 0 {
        return true;
    }
    if trips > 4
        && segs
            .iter()
            .any(|s| block_has_libcall_mem(func, s.blk.index()))
    {
        return false;
    }
    if segs.iter().any(|s| block_has_float(func, s.blk.index())) {
        return trips <= MAX_FP_FULL_TRIPS && trips * body_insts <= MAX_FP_FULL_INSTS;
    }
    if trips > MAX_VEC_FULL_TRIPS && segs.iter().any(|s| block_has_vec(func, s.blk.index())) {
        return false;
    }
    let is_two_seg = segs.len() == 2;
    let total_insts = body_insts + segs.len();
    let head_len = segs[0].non_term.len() + 1;
    let fits_legacy =
        is_two_seg && total_insts <= 64 && head_len + trips * total_insts <= MAX_FULL_INSTS;
    fits_legacy || (trips <= MAX_CHAIN_TRIPS && trips * body_insts <= MAX_CHAIN_UNROLLED_INSTS)
}

pub(super) fn unroll_chain(
    func: &mut Function,
    cfg: &Cfg,
    each: &Loop,
    global_consts: &[Option<i64>],
) -> bool {
    let Some(chain) = collect_chain(func, cfg, each) else {
        return false;
    };
    let head = chain.segs[0].blk;
    let body_insts: usize = chain.segs.iter().map(|s| s.non_term.len()).sum();
    let is_two_seg = chain.segs.len() == 2;
    if body_insts > MAX_CHAIN_UNROLLED_INSTS {
        return false;
    }
    let Some(init_params) = outside_const_params(func, cfg, head, &each.latches, global_consts)
    else {
        return false;
    };
    let max_trips = if is_two_seg {
        MAX_FULL_TRIPS
    } else {
        MAX_CHAIN_TRIPS
    };
    let Some(proven) = eval_chain(func, &chain, init_params, global_consts, max_trips) else {
        return false;
    };
    let trips = match &chain.kind {
        LoopChainKind::TopTested { .. } => proven,
        LoopChainKind::BottomTested { .. } => proven + 1,
    };
    if !chain_fits(func, &chain.segs, body_insts, trips) {
        return false;
    }
    let skip: Vec<usize> = chain.segs.iter().map(|s| s.blk.index()).collect();
    match chain.kind {
        LoopChainKind::TopTested {
            exit_blk,
            exit_args,
            ..
        } => emit_top_chain(func, &chain.segs, trips, (exit_blk, exit_args), &skip),
        LoopChainKind::BottomTested {
            exit_blk,
            exit_args,
            ..
        } => emit_bottom_chain(func, &chain.segs, proven, (exit_blk, exit_args), &skip),
    }
}
