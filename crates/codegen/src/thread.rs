mod carry;
mod decided;
mod edges;
mod eval;
mod loop_spec;
mod ret;
mod sink;
mod stored;
mod tail;
mod widen;

use crate::ir::{Block, Domain, Function, Inst, InstData, Op, Value};

use self::edges::{split_select_branches, thread_br_table, thread_brif, thread_jump};
pub(crate) use self::eval::is_bool_val;
use self::eval::{EvalState, ThreadCx, eval_op, lookup};

const MAX_DISP_INSTS: usize = 10;
const MAX_JUMP_DISP_INSTS: usize = 5;

struct Eval {
    chosen_target: Block,
    chosen_args: Vec<Value>,
    local_ops: Vec<(Value, Op)>,
    local_consts: Vec<(Value, i64)>,
    needed: Vec<bool>,
    resolved_branch: bool,
    carry: carry::Carry,
}

fn block_bit(block: Block) -> u64 {
    1 << (block.index() % 64)
}

pub(super) fn defs_and_uses(func: &Function) -> (Vec<Option<Inst>>, Vec<u32>) {
    let mut defs = vec![None; func.types.len()];
    let mut users = vec![0_u32; func.types.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result {
                defs[res.index()] = Some(inst);
            }
            data.op.for_operands(|v| users[v.index()] += 1);
        }
    }
    (defs, users)
}

pub(super) fn retarget_empty_preds(func: &mut Function, moved: &[(Block, Block, Block)]) -> bool {
    let mut changed = false;
    for &(pred, from, to) in moved {
        let Some(&term) = func.blocks[pred.index()].insts.last() else {
            continue;
        };
        changed |= func.insts[term.index()].op.retarget_empty(from, to);
    }
    if changed {
        func.bump_cfg();
    }
    changed
}

fn note_use(uses: &mut Vec<u64>, block: Block, value: Value) {
    if uses.len() <= value.index() {
        uses.resize(value.index() + 1, 0);
    }
    uses[value.index()] |= block_bit(block);
}

fn outside_uses(func: &Function) -> Vec<u64> {
    let mut uses = vec![0; func.types.len()];
    let mut home = vec![u32::MAX; func.types.len()];
    for (index, block) in func.blocks.iter().enumerate() {
        let id = index as u32;
        block
            .params
            .iter()
            .for_each(|param| home[param.index()] = id);
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| {
                if home[v.index()] != id {
                    uses[v.index()] |= block_bit(Block(id));
                }
            });
            if let Some(res) = data.result {
                home[res.index()] = id;
            }
        }
    }
    uses
}

fn redefines(func: &Function, block: Block, held: &[Value]) -> bool {
    !held.is_empty()
        && func.blocks[block.index()]
            .insts
            .iter()
            .filter_map(|inst| func.insts[inst.index()].result)
            .any(|result| held.contains(&result))
}

fn needed_ops(state: &EvalState, args: &[Value]) -> Vec<bool> {
    let mut needed = vec![false; state.local_ops.len()];
    for &arg in args {
        if let Some(idx) = state.local_ops.iter().position(|&(r, _)| r == arg) {
            needed[idx] = true;
        }
    }
    for i in (0..state.local_ops.len()).rev() {
        if !needed[i]
            || state
                .local_consts
                .iter()
                .any(|&(r, _)| r == state.local_ops[i].0)
        {
            continue;
        }
        state.local_ops[i].1.for_operands(|v| {
            if let Some(idx) = state.local_ops.iter().position(|&(r, _)| r == v) {
                needed[idx] = true;
            }
        });
    }
    needed
}

fn push_emit_inst(
    func: &mut Function,
    consts: &mut Vec<Option<i64>>,
    emit_map: &mut Vec<(Value, Value)>,
    inserted: &mut Vec<Inst>,
    orig: Value,
    op: Op,
    imm: Option<i64>,
) -> Value {
    let val = Value(func.types.len() as u32);
    func.types.push(func.ty(orig));
    consts.push(imm);
    let inst = Inst(func.insts.len() as u32);
    func.insts.push(InstData {
        op,
        result: Some(val),
        loc: 0,
    });
    inserted.push(inst);
    emit_map.push((orig, val));
    val
}

fn resolve_emit_val(
    func: &mut Function,
    consts: &mut Vec<Option<i64>>,
    local_consts: &[(Value, i64)],
    emit_map: &mut Vec<(Value, Value)>,
    inserted: &mut Vec<Inst>,
    val: Value,
) -> Value {
    let mapped = lookup(emit_map, val);
    if mapped != val {
        return mapped;
    }
    let Some(&(_, imm)) = local_consts.iter().rev().find(|&&(r, _)| r == val) else {
        return val;
    };
    push_emit_inst(
        func,
        consts,
        emit_map,
        inserted,
        val,
        Op::Const(imm),
        Some(imm),
    )
}

fn emit_needed(
    func: &mut Function,
    pred: Block,
    (consts, uses): (&mut Vec<Option<i64>>, &mut Vec<u64>),
    eval: Eval,
) -> (Block, Vec<Value>) {
    let mut emit_map: Vec<(Value, Value)> = Vec::new();
    let mut inserted: Vec<Inst> = Vec::new();
    for (idx, (res, op)) in eval.local_ops.into_iter().enumerate() {
        if !eval.needed[idx] {
            continue;
        }
        let (new_op, imm_opt) =
            if let Some(&(_, imm)) = eval.local_consts.iter().find(|&&(r, _)| r == res) {
                (Op::Const(imm), Some(imm))
            } else {
                let mut mapped = op;
                mapped.map_operands(|v| {
                    resolve_emit_val(
                        func,
                        consts,
                        &eval.local_consts,
                        &mut emit_map,
                        &mut inserted,
                        v,
                    )
                });
                (mapped, None)
            };
        push_emit_inst(
            func,
            consts,
            &mut emit_map,
            &mut inserted,
            res,
            new_op,
            imm_opt,
        );
    }
    let new_args: Vec<Value> = eval
        .chosen_args
        .into_iter()
        .map(|v| {
            resolve_emit_val(
                func,
                consts,
                &eval.local_consts,
                &mut emit_map,
                &mut inserted,
                v,
            )
        })
        .collect();
    for &inst in &inserted {
        func.insts[inst.index()]
            .op
            .for_operands(|v| note_use(uses, pred, v));
    }
    new_args.iter().for_each(|&v| note_use(uses, pred, v));
    if !inserted.is_empty() {
        let insts = &mut func.blocks[pred.index()].insts;
        let at = insts.len() - 1;
        insts.splice(at..at, inserted);
    }
    (eval.chosen_target, new_args)
}

fn emit_select(
    func: &mut Function,
    cx: &mut ThreadCx<'_>,
    pred: Block,
    cond: Value,
    then: Value,
    otherwise: Value,
) -> Value {
    let value = func.value(func.ty(then));
    cx.consts.push(None);
    let inst = func.add(Op::Select(cond, then, otherwise), Some(value));
    let insts = &mut func.blocks[pred.index()].insts;
    let at = insts.len() - 1;
    insts.insert(at, inst);
    value
}

fn select_arms(
    func: &Function,
    defs: &[Option<Inst>],
    value: Value,
) -> Option<(Value, Value, Value)> {
    if func.is_unpredictable(value) {
        return None;
    }
    match func.insts[defs.get(value.index()).copied().flatten()?.index()].op {
        Op::Select(cond, then, otherwise) => Some((cond, then, otherwise)),
        _ => None,
    }
}

fn select_split(
    func: &Function,
    defs: &[Option<Inst>],
    args: &[Value],
    at: usize,
) -> Option<(Value, Vec<Value>, Vec<Value>)> {
    let (sel_c, sel_t, sel_f) = select_arms(func, defs, args[at])?;
    if sel_t == sel_f
        || args[..at]
            .iter()
            .any(|&a| select_arms(func, defs, a).is_some_and(|(c, t, f)| c == sel_c && t != f))
    {
        return None;
    }
    let mut args_t = args.to_vec();
    let mut args_f = args.to_vec();
    for (j, &a) in args.iter().enumerate() {
        if let Some((c2, t2, f2)) = select_arms(func, defs, a)
            && c2 == sel_c
        {
            args_t[j] = t2;
            args_f[j] = f2;
        }
    }
    Some((sel_c, args_t, args_f))
}

fn single_pred_brif_seed(
    func: &Function,
    single: &[Option<Block>],
    pred: Block,
) -> Option<(Value, bool)> {
    let mut cur = pred;
    for _ in 0..2 {
        let parent = single[cur.index()]?;
        match func.last(parent) {
            Some(Op::Brif(cond, t_blk, _, e_blk, _))
                if *t_blk != *e_blk && (*t_blk == cur || *e_blk == cur) =>
            {
                return Some((*cond, *t_blk == cur));
            }
            Some(Op::Jump(target, _)) if *target == cur => cur = parent,
            _ => break,
        }
    }
    None
}

fn known_consts(func: &Function) -> Vec<Option<i64>> {
    let mut consts = vec![None; func.types.len()];
    for data in &func.insts {
        if let (Op::Const(_), Some(res)) = (&data.op, data.result) {
            consts[res.index()] = eval_op(func, &data.op, func.ty(res), &|_| None);
        }
    }
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result
                && consts[res.index()].is_none()
            {
                consts[res.index()] = eval_op(func, &data.op, func.ty(res), &|v| consts[v.index()]);
            }
        }
    }
    consts
}

fn body(func: &Function, block: Block) -> &[Inst] {
    func.blocks[block.index()]
        .insts
        .split_last()
        .map_or(&[][..], |(_, body)| body)
}

fn dispatches(func: &Function, pinned: &[bool], disp: Block) -> bool {
    if pinned[disp.index()] || !stored::steppable(func, disp, MAX_DISP_INSTS) {
        return false;
    }
    match func.last(disp) {
        Some(Op::Jump(next, _)) => {
            func.blocks[disp.index()].insts.len() <= MAX_JUMP_DISP_INSTS
                && *next != disp
                && !pinned[next.index()]
                && stored::steppable(func, *next, MAX_DISP_INSTS)
        }
        _ => true,
    }
}

fn threadable(func: &Function, pinned: &[bool]) -> bool {
    (0..func.blocks.len() as u32).map(Block).any(|pred| {
        func.successors(pred)
            .any(|disp| disp != pred && dispatches(func, pinned, disp))
    })
}

/// The blocks below the immediate dominator of each block holding an unpredictable select that reach that block, which together decide its condition.
pub(crate) fn unpredictable_region(func: &Function, cfg: &crate::cfg::Cfg) -> Vec<bool> {
    let mut region = vec![false; func.blocks.len()];
    if !func.has_unpredictable() {
        return region;
    }
    for (index, block) in func.blocks.iter().enumerate() {
        let marked = block.insts.iter().any(|inst| {
            func.insts[inst.index()]
                .result
                .is_some_and(|res| func.is_unpredictable(res))
        });
        if !marked || !cfg.reachable[index] {
            continue;
        }
        let here = Block(index as u32);
        let wall = cfg.idom[index].unwrap_or(here);
        let body = crate::cfg::natural_loop(cfg, wall, vec![here]).body;
        for (at, (inside, hit)) in region.iter_mut().zip(body).enumerate() {
            *inside |= hit && (at != wall.index() || at == index);
        }
    }
    region
}

/// Threads correlated constant- and range-condition jump chains through small pure dispatcher blocks.
pub(crate) fn run(func: &mut Function) -> Domain {
    let Some(entry) = func.entry else {
        return Domain::NONE;
    };
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
    let split = split_select_branches(func)
        | decided::run(func)
        | stored::run(func)
        | tail::run(func)
        | loop_spec::run(func);
    let mut pinned = vec![false; func.blocks.len()];
    pinned[entry.index()] = true;
    for data in &func.insts {
        if let Op::TryCall {
            normal, handler, ..
        } = data.op
        {
            pinned[normal.index()] = true;
            pinned[handler.index()] = true;
        }
    }
    if func.has_unpredictable() {
        let region = unpredictable_region(func, &func.cfg());
        pinned
            .iter_mut()
            .zip(region)
            .for_each(|(pin, inside)| *pin |= inside);
    }
    let mut consts = Vec::new();
    let opened = ret::open_returns(func, &pinned);
    let mut any = split;
    for round in 0..16 {
        let mut defs = vec![None; func.types.len()];
        for block in &func.blocks {
            for &inst in &block.insts {
                let data = &func.insts[inst.index()];
                if let Some(res) = data.result {
                    defs[res.index()] = Some(inst);
                }
                if let Op::Call { results, .. } = &data.op {
                    for &res in results {
                        defs[res.index()] = Some(inst);
                    }
                }
            }
        }
        if !threadable(func, &pinned) && !sink::wanted(func, &defs, &pinned) {
            break;
        }
        if round == 0 {
            consts = known_consts(func);
        }
        let cfg = func.cfg();
        let single: Vec<Option<Block>> = (0..func.blocks.len())
            .map(|b| cfg.single_pred(Block(b as u32)))
            .collect();
        let ranges =
            crate::facts::Ranges::new(func, (&defs, &consts), (&cfg.rpo, &single, &cfg.idom));
        let ranges_len = func.types.len();
        let mut uses = outside_uses(func);
        let mut cx = ThreadCx {
            defs: &defs,
            consts: &mut consts,
            uses: &mut uses,
            ranges: &ranges,
            ranges_len,
            pinned: &pinned,
            idom: &cfg.idom,
        };
        let mut changed = sink::run(func, &mut cx, (&*cfg, &single));
        let mut stale = false;
        let mut extra_preds: Vec<Vec<Block>> = Vec::new();
        let mut live = single.clone();
        for disp_idx in 0..func.blocks.len() {
            let disp = Block(disp_idx as u32);
            if !dispatches(func, cx.pinned, disp) {
                continue;
            }
            let mut into = cfg.preds_of(disp).to_vec();
            if let Some(extra) = extra_preds.get(disp_idx) {
                into.extend_from_slice(extra);
            }
            into.sort_unstable();
            into.dedup();
            for pred in into {
                if pred == disp {
                    continue;
                }
                let mut pred_changed = false;
                pred_changed |= thread_jump(func, &mut cx, &mut live, pred, disp);
                pred_changed |= thread_brif(func, &mut cx, &mut live, pred, disp);
                pred_changed |= thread_br_table(func, &cx, &live, pred, disp);
                if pred_changed {
                    changed = true;
                    if extra_preds.is_empty() {
                        extra_preds.resize(func.blocks.len(), Vec::new());
                    }
                    func.successors(pred)
                        .for_each(|s| extra_preds[s.index()].push(pred));
                    if func.successors(pred).any(|s| {
                        s != disp
                            && single[s.index()] == live[s.index()]
                            && (single[s.index()].is_some()
                                || cfg.idom[s.index()].is_some_and(|d| {
                                    d != s && !crate::cfg::dominates(&cfg.idom, d, pred)
                                }))
                    }) {
                        stale = true;
                        break;
                    }
                }
            }
            if stale {
                break;
            }
        }
        if !changed {
            break;
        }
        func.bump_cfg();
        any = true;
    }
    any |= ret::close_returns(func, &opened) | !opened.is_empty();
    let widened = widen::run(func);
    let mut dirty = if any || widened {
        crate::fold::prune(func, entry);
        Domain::CFG | Domain::VALUES
    } else {
        Domain::NONE
    };
    dirty |= sink::sink_cold(func);
    dirty
}
