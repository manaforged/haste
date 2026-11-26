use crate::cfg::{Cfg, Loops};
use crate::ir::{Block, BlockData, Domain, Function, Inst, Op, Value};

use super::super::body;

fn can_split_exit(
    func: &Function,
    cfg: &Cfg,
    uses: &[u32],
    block: Block,
    (target, args): (Block, &[Value]),
) -> bool {
    if args.is_empty() || cfg.loops().depth_of(target) >= cfg.loops().depth_of(block) {
        return false;
    }
    body(func, block).iter().any(|&inst| {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else {
            return false;
        };
        !matches!(data.op, Op::Const(_) | Op::StackAddr(..))
            && data.op.is_speculable()
            && uses[res.index()] > 0
            && uses[res.index()] == args.iter().filter(|&&a| a == res).count() as u32
    })
}

fn add_jump_block(func: &mut Function, target: Block, args: Vec<Value>) -> Block {
    let split = Block(func.blocks.len() as u32);
    let jump = func.add(Op::Jump(target, args), None);
    func.blocks.push(BlockData {
        params: Vec::new(),
        insts: vec![jump],
    });
    split
}

fn split_cold_loop_exits(func: &mut Function) -> bool {
    let orig_len = func.blocks.len();
    let has_cand = {
        let cfg = func.cfg();
        !cfg.loops().list.is_empty()
            && (0..orig_len).any(|bi| {
                let block = Block(bi as u32);
                let depth = cfg.loops().depth_of(block);
                cfg.reachable[bi]
                    && depth > 0
                    && matches!(
                        func.last(block),
                        Some(Op::Brif(_, t_b, t_a, e_b, e_a))
                            if t_b != e_b
                                && ((!t_a.is_empty() && cfg.loops().depth_of(*t_b) < depth)
                                    || (!e_a.is_empty() && cfg.loops().depth_of(*e_b) < depth))
                    )
            })
    };
    if !has_cand {
        return false;
    }
    let uses = func.use_counts();
    let cfg = func.cfg().clone();
    let mut split_any = false;
    for bi in 0..orig_len {
        let block = Block(bi as u32);
        if !cfg.reachable[bi] || cfg.loops().depth_of(block) == 0 {
            continue;
        }
        let Some(&term) = func.blocks[bi].insts.last() else {
            continue;
        };
        let Op::Brif(cond, t_b, t_a, e_b, e_a) = func.insts[term.index()].op.clone() else {
            continue;
        };
        if t_b == e_b {
            continue;
        }
        let (mut nt_b, mut nt_a, mut ne_b, mut ne_a) = (t_b, t_a, e_b, e_a);
        let mut changed = false;
        if can_split_exit(func, &cfg, &uses, block, (nt_b, &nt_a)) {
            nt_b = add_jump_block(func, nt_b, std::mem::take(&mut nt_a));
            changed = true;
        }
        if can_split_exit(func, &cfg, &uses, block, (ne_b, &ne_a)) {
            ne_b = add_jump_block(func, ne_b, std::mem::take(&mut ne_a));
            changed = true;
        }
        if changed {
            func.insts[term.index()].op = Op::Brif(cond, nt_b, nt_a, ne_b, ne_a);
            split_any = true;
        }
    }
    split_any
}

fn loop_has_calls(func: &Function, loops: &Loops) -> Vec<bool> {
    loops
        .list
        .iter()
        .map(|each| {
            each.body.iter().enumerate().any(|(b, &in_l)| {
                in_l && func.blocks[b].insts.iter().any(|&i| {
                    matches!(
                        func.insts[i.index()].op,
                        Op::Call { .. } | Op::TryCall { .. }
                    )
                })
            })
        })
        .collect()
}

fn vec_const_defs(func: &Function, cfg: &Cfg) -> Vec<bool> {
    let mut is_scalar_const = vec![false; func.types.len()];
    let mut is_vec_const = vec![false; func.types.len()];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else {
                continue;
            };
            let ty = func.ty(res);
            if ty.is_vector() {
                is_vec_const[res.index()] = match &data.op {
                    Op::Const(_) => true,
                    Op::Insert(base, scalar, _) => {
                        is_vec_const[base.index()] && is_scalar_const[scalar.index()]
                    }
                    Op::Shuffle(a, b, crate::ir::ShuffleMask::Const(_)) => {
                        is_vec_const[a.index()] && is_vec_const[b.index()]
                    }
                    Op::Convert(crate::ir::Convert::Bitcast, x) => is_vec_const[x.index()],
                    _ => false,
                };
            } else if matches!(data.op, Op::Const(_)) {
                is_scalar_const[res.index()] = true;
            }
        }
    }
    is_vec_const
}

fn loops_sink_ok(
    cfg: &Cfg,
    loop_calls: &[bool],
    (inner, outer): (Block, Block),
    (is_vec_const, is_const): (bool, bool),
) -> bool {
    cfg.loops().list.iter().enumerate().all(|(idx, each)| {
        !each.body[inner.index()]
            || each.body[outer.index()]
            || (is_vec_const && loop_calls[idx])
            || (is_const
                && loop_calls[idx]
                && !crate::licm::always_executes_in_loop(cfg, each, inner))
    })
}

fn non_trivial_op(op: &Op) -> bool {
    !matches!(op, Op::Const(_) | Op::Select(..))
}

fn diamond_safe(
    func: &Function,
    cfg: &Cfg,
    (block, target): (Block, Block),
    (op, queued): (&Op, &[Inst]),
) -> bool {
    if cfg.loops().depth_of(target) != cfg.loops().depth_of(block)
        || cfg.pred_count(target) != 1
        || !matches!(func.last(target), Some(Op::Jump(..)))
        || !non_trivial_op(op)
    {
        return true;
    }
    let insts = body(func, target);
    if insts.iter().any(|&i| {
        let op = &func.insts[i.index()].op;
        !op.is_speculable() && !matches!(op, Op::Load(..) | Op::StackLoad(..))
    }) {
        return true;
    }
    let base = insts
        .iter()
        .filter(|&&i| non_trivial_op(&func.insts[i.index()].op))
        .count();
    base >= 6
        || base
            + queued
                .iter()
                .filter(|&&i| non_trivial_op(&func.insts[i.index()].op))
                .count()
            < 6
}

fn cheap_call_addr(func: &Function, defs: &[Option<Inst>], op: &Op) -> bool {
    crate::licm::cheap_call_addr(op, |v| {
        defs.get(v.index())
            .copied()
            .flatten()
            .is_some_and(|i| matches!(func.insts[i.index()].op, Op::Const(_)))
    })
}

struct SinkInfo<'a> {
    defs: Option<&'a [Option<Inst>]>,
    loop_calls: &'a [bool],
    vec_consts: &'a [bool],
}

fn pick_sink_target(
    func: &Function,
    cfg: &Cfg,
    info: &SinkInfo<'_>,
    block: Block,
    sites: &[(Inst, Block)],
    (res, op, moves): (Value, &Op, &[Vec<Inst>]),
) -> Option<Block> {
    let mut lca = sites.first()?.1;
    for &(_, site) in sites {
        if !cfg.reachable[site.index()] {
            return None;
        }
        while !cfg.dominates(lca, site) {
            lca = cfg.idom[lca.index()]?;
        }
    }
    let is_vec_const = info.vec_consts.get(res.index()).copied().unwrap_or(false);
    let is_const = !is_vec_const && matches!(op, Op::Const(_));
    let mut target = lca;
    if !(target != block
        && !loops_sink_ok(cfg, info.loop_calls, (target, block), (false, false))
        && info.defs.is_some_and(|d| cheap_call_addr(func, d, op))
        && crate::live::has_regular_call(func, target))
    {
        while target != block
            && !loops_sink_ok(
                cfg,
                info.loop_calls,
                (target, block),
                (is_vec_const, is_const),
            )
        {
            target = cfg.idom[target.index()]?;
        }
    }
    if target == block || !cfg.dominates(block, target) {
        return None;
    }
    (is_vec_const || diamond_safe(func, cfg, (block, target), (op, &moves[target.index()])))
        .then_some(target)
}

fn build_users(func: &Function, spec_def: &[bool]) -> Vec<Vec<(Inst, Block)>> {
    let mut users = vec![Vec::new(); func.types.len()];
    for (bi, data) in func.blocks.iter().enumerate() {
        let blk = Block(bi as u32);
        for &inst in &data.insts {
            func.insts[inst.index()].op.for_operands(|v| {
                if spec_def[v.index()] {
                    users[v.index()].push((inst, blk));
                }
            });
        }
    }
    users
}

fn topo_visit(
    func: &Function,
    moves: &[Inst],
    defs: &[(u32, usize)],
    state: &mut [u8],
    ordered: &mut Vec<Inst>,
    i: usize,
) {
    if state[i] != 0 {
        return;
    }
    state[i] = 1;
    func.insts[moves[i].index()].op.for_operands(|v| {
        if let Ok(pos) = defs.binary_search_by_key(&v.0, |&(r, _)| r) {
            topo_visit(func, moves, defs, state, ordered, defs[pos].1);
        }
    });
    state[i] = 2;
    ordered.push(moves[i]);
}

fn topo_sort_moves(func: &Function, moves: &mut Vec<Inst>) {
    let n = moves.len();
    if n <= 1 {
        return;
    }
    let mut defs: Vec<(u32, usize)> = moves
        .iter()
        .enumerate()
        .filter_map(|(i, &inst)| func.insts[inst.index()].result.map(|r| (r.0, i)))
        .collect();
    if defs.len() <= 1 {
        return;
    }
    defs.sort_unstable_by_key(|&(r, _)| r);
    let mut ordered = Vec::with_capacity(n);
    let mut state = vec![0_u8; n];
    for i in 0..n {
        topo_visit(func, moves, &defs, &mut state, &mut ordered, i);
    }
    *moves = ordered;
}

fn flush_moves(func: &mut Function, block: Block, moves: &mut Vec<Inst>) {
    if moves.is_empty() {
        return;
    }
    topo_sort_moves(func, moves);
    let mut next: Vec<Inst> = moves.drain(..).collect();
    next.extend_from_slice(&func.blocks[block.index()].insts);
    func.blocks[block.index()].insts = next;
}

fn sink_block(
    func: &mut Function,
    cfg: &Cfg,
    info: &SinkInfo<'_>,
    block: Block,
    has_child: bool,
    users: &mut [Vec<(Inst, Block)>],
    moves: &mut [Vec<Inst>],
) -> bool {
    flush_moves(func, block, &mut moves[block.index()]);
    if !has_child {
        return false;
    }
    let bi = block.index();
    let len = func.blocks[bi].insts.len();
    let body_len = len.saturating_sub(1);
    if body_len == 0 {
        return false;
    }
    let mut kept: Vec<bool> = Vec::new();
    let mut touched: Vec<(usize, usize)> = Vec::new();
    for pos in (0..body_len).rev() {
        let inst = func.blocks[bi].insts[pos];
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else {
            continue;
        };
        if !data.op.is_speculable() {
            continue;
        }
        let Some(target) = pick_sink_target(
            func,
            cfg,
            info,
            block,
            &users[res.index()],
            (res, &data.op, moves),
        ) else {
            continue;
        };
        if kept.is_empty() {
            kept.resize(len, true);
        }
        kept[pos] = false;
        let ti = target.index();
        if !touched.iter().any(|&(t, _)| t == ti) {
            touched.push((ti, moves[ti].len()));
        }
        moves[ti].push(inst);
        data.op.for_operands(|v| {
            for entry in &mut users[v.index()] {
                if entry.0 == inst {
                    entry.1 = target;
                }
            }
        });
    }
    for (ti, start) in touched {
        moves[ti][start..].reverse();
    }
    if !kept.is_empty() {
        let mut idx = 0;
        func.blocks[bi].insts.retain(|_| {
            let keep = kept[idx];
            idx += 1;
            keep
        });
        return true;
    }
    false
}

fn sink_dominated(func: &mut Function) -> bool {
    if func.blocks.len() <= 1 {
        return false;
    }
    let cfg = func.cfg().clone();
    let mut has_child = vec![false; func.blocks.len()];
    let mut any_child = false;
    for &b in &cfg.rpo {
        if let Some(p) = cfg.idom[b.index()]
            && p != b
        {
            has_child[p.index()] = true;
            any_child = true;
        }
    }
    if !any_child {
        return false;
    }
    let mut spec_def = vec![false; func.types.len()];
    let mut any_spec = false;
    for &b in &cfg.rpo {
        if !has_child[b.index()] {
            continue;
        }
        let insts = &func.blocks[b.index()].insts;
        for &inst in &insts[..insts.len().saturating_sub(1)] {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result
                && data.op.is_speculable()
            {
                spec_def[res.index()] = true;
                any_spec = true;
            }
        }
    }
    if !any_spec {
        return false;
    }
    let has_loops = !cfg.loops().list.is_empty();
    let defs = has_loops.then(|| crate::fold::defs(func));
    let loop_calls = if has_loops {
        loop_has_calls(func, cfg.loops())
    } else {
        Vec::new()
    };
    let vec_consts = if has_loops && func.types.iter().any(|ty| ty.is_vector()) {
        vec_const_defs(func, &cfg)
    } else {
        Vec::new()
    };
    let info = SinkInfo {
        defs: defs.as_deref(),
        loop_calls: &loop_calls,
        vec_consts: &vec_consts,
    };
    let mut users = build_users(func, &spec_def);
    let mut moves = vec![Vec::new(); func.blocks.len()];
    let mut changed = false;
    for &block in &cfg.rpo {
        changed |= sink_block(
            func,
            &cfg,
            &info,
            block,
            has_child[block.index()],
            &mut users,
            &mut moves,
        );
    }
    changed
}

pub(in super::super) fn sink_cold(func: &mut Function) -> Domain {
    let mut out = Domain::NONE;
    if split_cold_loop_exits(func) {
        func.bump_cfg();
        out |= Domain::CFG | Domain::VALUES;
    }
    if sink_dominated(func) {
        out |= Domain::VALUES;
    }
    out
}
