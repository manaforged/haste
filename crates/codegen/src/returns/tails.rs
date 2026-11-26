use rustc_hash::FxHashMap;

use crate::cfg::Cfg;
use crate::ir::{Binary, Block, BlockData, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

use super::{MIN_TAIL_GROUP, collect_consts};

const MAX_SINK_PARAMS: usize = 2;

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

struct FactorSite {
    block: usize,
    tail_pos: usize,
    args: [Value; 3],
    out_vals: Vec<Value>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ReturnGroupKey {
    sig: TailOpSig,
    ret_slot: usize,
    ret_types: Vec<Type>,
    const_rets: Vec<Option<i64>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct JumpGroupKey {
    target: Block,
    sig: TailOpSig,
    positions: Vec<usize>,
    out_types: Vec<Type>,
    const_outs: Vec<Option<i64>>,
}

struct JumpTailPlan {
    positions: Vec<usize>,
    tail_positions: Vec<usize>,
    sig: TailOpSig,
    inputs: Vec<SharedInput>,
    param_cols: Vec<(Type, Vec<Value>)>,
}

fn tail_op_sig(
    func: &Function,
    consts: &[Option<i64>],
    inst: Inst,
) -> Option<(TailOpSig, Value, [Value; 3])> {
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
    Some((
        TailOpSig {
            kind,
            res_ty: func.ty(res),
            arity,
            arg_tys,
            const_args,
        },
        res,
        args,
    ))
}

fn plan_shared_columns(
    columns: Vec<(Type, Option<i64>, Vec<Value>)>,
    max_params: usize,
) -> Option<(Vec<SharedInput>, Vec<(Type, Vec<Value>)>)> {
    let mut inputs = Vec::with_capacity(columns.len());
    let mut param_cols: Vec<(Type, Vec<Value>)> = Vec::new();
    for (ty, const_val, vals) in columns {
        if let Some(imm) = const_val {
            inputs.push(SharedInput::Const(ty, imm));
        } else if vals.iter().all(|&v| v == vals[0]) {
            inputs.push(SharedInput::Common(vals[0]));
        } else if let Some(existing) = param_cols
            .iter()
            .position(|(p_ty, col)| *p_ty == ty && *col == vals)
        {
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
                if let Some(&(_, _, existing)) = emitted_consts
                    .iter()
                    .find(|&&(cty, cimm, _)| cty == ty && cimm == imm)
                {
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

fn tail_non_const_pos(func: &Function, insts: &[Inst]) -> Option<usize> {
    let len = insts.len();
    if len < 2 {
        return None;
    }
    let mut pos = len - 2;
    while pos > 0 && matches!(func.insts[insts[pos].index()].op, Op::Const(_)) {
        pos -= 1;
    }
    (!matches!(func.insts[insts[pos].index()].op, Op::Const(_))).then_some(pos)
}

fn extra_const_outs(vals: &[Value], consts: &[Option<i64>], skip: &[usize]) -> Vec<Option<i64>> {
    vals.iter()
        .enumerate()
        .filter(|(r, _)| !skip.contains(r))
        .map(|(_, &v)| consts[v.index()])
        .collect()
}

fn return_site_entry(
    func: &Function,
    consts: &[Option<i64>],
    uses: &[u32],
    block: usize,
) -> Option<(ReturnGroupKey, FactorSite)> {
    let insts = &func.blocks[block].insts;
    let len = insts.len();
    if len < 2 {
        return None;
    }
    let Op::Return(ret_vals) = &func.insts[insts[len - 1].index()].op else {
        return None;
    };
    let tail_pos = len - 2;
    let (sig, res, args) = tail_op_sig(func, consts, insts[tail_pos])?;
    if uses[res.index()] != 1 {
        return None;
    }
    if matches!(sig.kind, TailKind::Const(imm) if ret_vals.len() == 1 && (-1..=0xffff).contains(&imm))
    {
        return None;
    }
    let ret_slot = ret_vals.iter().position(|&v| v == res)?;
    let ret_types: Vec<Type> = ret_vals.iter().map(|&v| func.ty(v)).collect();
    let const_rets = extra_const_outs(ret_vals, consts, &[ret_slot]);
    let key = ReturnGroupKey {
        sig,
        ret_slot,
        ret_types,
        const_rets,
    };
    let site = FactorSite {
        block,
        tail_pos,
        args,
        out_vals: ret_vals.clone(),
    };
    Some((key, site))
}

fn build_group_columns(
    sig: &TailOpSig,
    group: &[FactorSite],
    out_types: &[Type],
    const_outs: &[Option<i64>],
    res_slots: &[usize],
) -> Vec<(Type, Option<i64>, Vec<Value>)> {
    let arity = sig.arity as usize;
    let mut columns = Vec::with_capacity(arity + out_types.len());
    for slot in 0..arity {
        let vals = group.iter().map(|s| s.args[slot]).collect();
        columns.push((sig.arg_tys[slot], sig.const_args[slot], vals));
    }
    let mut extra_at = 0;
    for (r, &ty) in out_types.iter().enumerate() {
        if !res_slots.contains(&r) {
            let vals = group.iter().map(|s| s.out_vals[r]).collect();
            columns.push((ty, const_outs[extra_at], vals));
            extra_at += 1;
        }
    }
    columns
}

fn emit_shared_tail_group(
    func: &mut Function,
    sig: &TailOpSig,
    group: &[FactorSite],
    (out_types, const_outs, res_slots, jump_target): (
        &[Type],
        &[Option<i64>],
        &[usize],
        Option<Block>,
    ),
    max_params: usize,
) -> bool {
    let columns = build_group_columns(sig, group, out_types, const_outs, res_slots);
    let Some((inputs, param_cols)) = plan_shared_columns(columns, max_params) else {
        return false;
    };
    if jump_target.is_some()
        && param_cols.len() > 1
        && group.len() < 4
        && !matches!(sig.kind, TailKind::Select | TailKind::Icmp(_))
    {
        return false;
    }
    let shared = Block(func.blocks.len() as u32);
    let params: Vec<Value> = param_cols.iter().map(|(ty, _)| func.value(*ty)).collect();
    let mut insts = Vec::with_capacity(4);
    let resolved = materialize_inputs(func, &mut insts, &inputs, &params);
    let shared_res = func.value(sig.res_ty);
    let arity = sig.arity as usize;
    insts.push(func.add(emit_tail_op(sig.kind, &resolved[..arity]), Some(shared_res)));
    let mut out_args = Vec::with_capacity(out_types.len());
    let mut extra_at = arity;
    for r in 0..out_types.len() {
        if res_slots.contains(&r) {
            out_args.push(shared_res);
        } else {
            out_args.push(resolved[extra_at]);
            extra_at += 1;
        }
    }
    let term_op = if let Some(target) = jump_target {
        Op::Jump(target, out_args)
    } else {
        Op::Return(out_args)
    };
    insts.push(func.add(term_op, None));
    func.blocks.push(BlockData { params, insts });
    let mut touched = Vec::with_capacity(group.len());
    for (site_idx, site) in group.iter().enumerate() {
        let jump_args: Vec<Value> = param_cols.iter().map(|(_, col)| col[site_idx]).collect();
        let jump = func.add(Op::Jump(shared, jump_args), None);
        let blk_insts = &mut func.blocks[site.block].insts;
        blk_insts.pop();
        blk_insts.remove(site.tail_pos);
        blk_insts.push(jump);
        touched.push(site.block);
    }
    prune_dead_consts(func, &touched);
    true
}

fn group_tail_sites<K: Eq + std::hash::Hash + Clone>(
    func: &Function,
    cfg: &Cfg,
    mut pick: impl FnMut(usize) -> Option<(K, FactorSite)>,
) -> Vec<(K, Vec<FactorSite>)> {
    let mut index: FxHashMap<K, usize> = FxHashMap::default();
    let mut groups: Vec<(K, Vec<FactorSite>)> = Vec::new();
    for block in 0..func.blocks.len() {
        if !cfg.reachable[block] || func.entry == Some(Block(block as u32)) {
            continue;
        }
        let Some((key, site)) = pick(block) else {
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
    groups
}

pub(super) fn factor_common_returns(
    func: &mut Function,
    cfg_dirty: bool,
    meta: &mut Option<(Vec<Option<i64>>, Vec<u32>)>,
) -> bool {
    if func
        .blocks
        .iter()
        .filter(|b| {
            b.insts.len() >= 2
                && matches!(
                    b.insts.last().map(|i| &func.insts[i.index()].op),
                    Some(Op::Return(_))
                )
        })
        .count()
        < MIN_TAIL_GROUP
    {
        return false;
    }
    if cfg_dirty {
        func.bump_cfg();
    }
    let cfg = func.cfg();
    let (consts, uses) = meta.get_or_insert_with(|| (collect_consts(func), func.use_counts()));
    let groups = group_tail_sites(func, &cfg, |b| return_site_entry(func, consts, uses, b));
    let mut changed = false;
    for (key, group) in &groups {
        if group.len() >= MIN_TAIL_GROUP
            && emit_shared_tail_group(
                func,
                &key.sig,
                group,
                (&key.ret_types, &key.const_rets, &[key.ret_slot], None),
                MAX_SINK_PARAMS,
            )
        {
            changed = true;
        }
    }
    if changed {
        *meta = None;
    }
    changed
}

fn jump_site_entry(
    func: &Function,
    consts: &[Option<i64>],
    uses: &[u32],
    block: usize,
) -> Option<(JumpGroupKey, FactorSite)> {
    let insts = &func.blocks[block].insts;
    let tail_pos = tail_non_const_pos(func, insts)?;
    let Op::Jump(target, j_args) = &func.insts[insts[insts.len() - 1].index()].op else {
        return None;
    };
    if target.index() == block
        || func.entry == Some(*target)
        || j_args.is_empty()
        || j_args.len() != func.blocks[target.index()].params.len()
    {
        return None;
    }
    let (sig, res, args) = tail_op_sig(func, consts, insts[tail_pos])?;
    if matches!(sig.kind, TailKind::Const(_) | TailKind::Convert(_)) {
        return None;
    }
    let positions: Vec<usize> = (0..j_args.len()).filter(|&i| j_args[i] == res).collect();
    if positions.is_empty() || uses[res.index()] != positions.len() as u32 {
        return None;
    }
    let out_types: Vec<Type> = j_args.iter().map(|&v| func.ty(v)).collect();
    let const_outs = extra_const_outs(j_args, consts, &positions);
    let key = JumpGroupKey {
        target: *target,
        sig,
        positions,
        out_types,
        const_outs,
    };
    let site = FactorSite {
        block,
        tail_pos,
        args,
        out_vals: j_args.clone(),
    };
    Some((key, site))
}

pub(super) fn factor_jump_tails(
    func: &mut Function,
    cfg_dirty: bool,
    meta: &mut Option<(Vec<Option<i64>>, Vec<u32>)>,
) -> bool {
    let mut counts = vec![0_u32; func.blocks.len()];
    let mut has_group = false;
    for (bi, b) in func.blocks.iter().enumerate() {
        if b.insts.len() >= 2
            && let Some(Op::Jump(t, a)) = b.insts.last().map(|i| &func.insts[i.index()].op)
            && !a.is_empty()
            && t.index() != bi
        {
            counts[t.index()] += 1;
            if counts[t.index()] >= MIN_TAIL_GROUP as u32 {
                has_group = true;
                break;
            }
        }
    }
    if !has_group {
        return false;
    }
    if cfg_dirty {
        func.bump_cfg();
    }
    let cfg = func.cfg();
    let (consts, uses) = meta.get_or_insert_with(|| (collect_consts(func), func.use_counts()));
    let groups = group_tail_sites(func, &cfg, |b| jump_site_entry(func, consts, uses, b));
    let mut changed = false;
    for (key, group) in &groups {
        let max_params = key.out_types.len().min(MAX_SINK_PARAMS);
        if group.len() >= MIN_TAIL_GROUP
            && emit_shared_tail_group(
                func,
                &key.sig,
                group,
                (
                    &key.out_types,
                    &key.const_outs,
                    &key.positions,
                    Some(key.target),
                ),
                max_params,
            )
        {
            changed = true;
        }
    }
    if changed {
        *meta = None;
    }
    changed
}

fn collect_jump_preds(func: &Function, cfg: &Cfg, target: Block) -> Option<Vec<usize>> {
    let preds = cfg.preds_of(target);
    if preds.len() < 2
        || preds.len() != cfg.pred_count(target) as usize
        || cfg.loops().list.iter().any(|l| l.head == target)
    {
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
        let Op::Jump(t, ref args) = func.insts[insts[insts.len() - 1].index()].op else {
            return None;
        };
        if t != target || args.len() != param_len {
            return None;
        }
        out.push(pred.index());
    }
    Some(out)
}

fn find_jump_tail_plan_at(
    func: &Function,
    consts: &[Option<i64>],
    uses: &[u32],
    preds: &[usize],
    (param_len, skip_consts): (usize, bool),
) -> Option<JumpTailPlan> {
    let mut pred_tails = Vec::with_capacity(preds.len());
    let mut tail_positions = Vec::with_capacity(preds.len());
    for &p in preds {
        let insts = &func.blocks[p].insts;
        let tail_pos = if skip_consts {
            tail_non_const_pos(func, insts)?
        } else {
            insts.len() - 2
        };
        let (sig, res, args) = tail_op_sig(func, consts, insts[tail_pos])?;
        let Op::Jump(_, ref j_args) = func.insts[insts[insts.len() - 1].index()].op else {
            return None;
        };
        let count = j_args.iter().filter(|&&v| v == res).count() as u32;
        if count == 0 || uses[res.index()] != count {
            return None;
        }
        tail_positions.push(tail_pos);
        pred_tails.push((sig, res, args, j_args.as_slice()));
    }
    let first_sig = &pred_tails[0].0;
    if !pred_tails[1..].iter().all(|(sig, ..)| sig == first_sig) {
        return None;
    }
    let first_res = pred_tails[0].1;
    let first_args = pred_tails[0].3;
    let positions: Vec<usize> = (0..param_len)
        .filter(|&idx| first_args[idx] == first_res)
        .collect();
    if !pred_tails[1..].iter().all(|(_, res, _, j_args)| {
        (0..param_len).all(|idx| (j_args[idx] == *res) == positions.contains(&idx))
    }) {
        return None;
    }
    let arity = first_sig.arity as usize;
    let columns = (0..arity)
        .map(|slot| {
            let vals = pred_tails
                .iter()
                .map(|(_, _, args, _)| args[slot])
                .collect();
            (first_sig.arg_tys[slot], first_sig.const_args[slot], vals)
        })
        .collect();
    let max_params = if preds.len() >= MIN_TAIL_GROUP
        && matches!(first_sig.kind, TailKind::Select | TailKind::Icmp(_))
    {
        MAX_SINK_PARAMS
    } else {
        positions.len().min(MAX_SINK_PARAMS)
    };
    let (inputs, param_cols) = plan_shared_columns(columns, max_params)?;
    Some(JumpTailPlan {
        positions,
        tail_positions,
        sig: first_sig.clone(),
        inputs,
        param_cols,
    })
}

fn find_jump_tail_plan(
    func: &Function,
    consts: &[Option<i64>],
    uses: &[u32],
    preds: &[usize],
    param_len: usize,
) -> Option<JumpTailPlan> {
    find_jump_tail_plan_at(func, consts, uses, preds, (param_len, false))
        .or_else(|| find_jump_tail_plan_at(func, consts, uses, preds, (param_len, true)))
}

fn apply_jump_tail_plan(
    func: &mut Function,
    target_idx: usize,
    preds: &[usize],
    plan: JumpTailPlan,
) {
    let new_params: Vec<Value> = plan
        .param_cols
        .iter()
        .map(|(ty, _)| func.value(*ty))
        .collect();
    let mut prepended = Vec::with_capacity(4);
    let resolved = materialize_inputs(func, &mut prepended, &plan.inputs, &new_params);
    let new_res = func.value(plan.sig.res_ty);
    prepended.push(func.add(emit_tail_op(plan.sig.kind, &resolved), Some(new_res)));
    prepended.append(&mut func.blocks[target_idx].insts);
    func.blocks[target_idx].insts = prepended;
    let mut old_params = Vec::with_capacity(plan.positions.len());
    for &idx in plan.positions.iter().rev() {
        old_params.push(func.blocks[target_idx].params.remove(idx));
    }
    func.blocks[target_idx].params.extend(new_params);
    for (pred_pos, &pred_idx) in preds.iter().enumerate() {
        let len = func.blocks[pred_idx].insts.len();
        let term = func.blocks[pred_idx].insts[len - 1];
        func.blocks[pred_idx]
            .insts
            .remove(plan.tail_positions[pred_pos]);
        if let Op::Jump(_, ref mut args) = func.insts[term.index()].op {
            for &idx in plan.positions.iter().rev() {
                args.remove(idx);
            }
            args.extend(plan.param_cols.iter().map(|(_, col)| col[pred_pos]));
        }
    }
    let mut subst = crate::merge::identity(func.types.len());
    for old in old_params {
        subst[old.index()] = new_res;
    }
    crate::merge::rename(func, &subst);
    prune_dead_consts(func, preds);
}

pub(super) fn sink_common_jump_tails(
    func: &mut Function,
    cfg_dirty: bool,
    meta: &mut Option<(Vec<Option<i64>>, Vec<u32>)>,
) -> bool {
    if cfg_dirty {
        func.bump_cfg();
    }
    let cfg = func.cfg();
    let mut changed = false;
    for j_idx in 0..func.blocks.len() {
        let target = Block(j_idx as u32);
        let param_len = func.blocks[j_idx].params.len();
        if !cfg.reachable[j_idx] || func.entry == Some(target) || param_len == 0 {
            continue;
        }
        let Some(preds) = collect_jump_preds(func, &cfg, target) else {
            continue;
        };
        let (consts, uses) = meta.get_or_insert_with(|| (collect_consts(func), func.use_counts()));
        let Some(plan) = find_jump_tail_plan(func, consts, uses, &preds, param_len) else {
            continue;
        };
        apply_jump_tail_plan(func, j_idx, &preds, plan);
        *meta = None;
        changed = true;
    }
    changed
}
