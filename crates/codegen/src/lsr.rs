use crate::cfg::{Cfg, Loop};
use crate::fold::{defs, evaluate_wrapped};
use crate::ir::{Binary, Block, Domain, Function, Inst, IntCC, Op, Type, Value};
use crate::merge::{find, identity, rename};

mod analyze;
mod lftr;

use analyze::{Group, IvInfo, collect_groups, collect_head_edges, collect_ivs, plan_lftr_for_iv};
use lftr::retire_counter;

/// Outgoing control-flow arm on a predecessor edge.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeArm {
    Jump,
    Then,
    Else,
}

struct State<'a> {
    func: &'a mut Function,
    cfg: &'a Cfg,
    defs: Vec<Option<Inst>>,
    def_blocks: Vec<Option<Block>>,
    consts: Vec<Option<i64>>,
    subst: Vec<Value>,
    renamed: bool,
}

impl<'a> State<'a> {
    fn new(func: &'a mut Function, cfg: &'a Cfg) -> Self {
        let defs = defs(func);
        let def_blocks = func.def_blocks();
        let mut consts = vec![None; func.types.len()];
        for &blk in &cfg.rpo {
            for &inst in &func.blocks[blk.index()].insts {
                let data = &func.insts[inst.index()];
                let Some(res) = data.result else { continue };
                if let Some(imm) =
                    evaluate_wrapped(func, &data.op, func.ty(res), &|v| consts[v.index()])
                {
                    consts[res.index()] = Some(imm);
                }
            }
        }
        let subst = identity(func.types.len());
        Self {
            func,
            cfg,
            defs,
            def_blocks,
            consts,
            subst,
            renamed: false,
        }
    }

    fn value_of(&self, v: Value) -> Value {
        find(&self.subst, v)
    }

    fn const_i64(&self, v: Value) -> Option<i64> {
        let v = self.value_of(v);
        (self.func.ty(v) == Type::I64)
            .then(|| self.consts.get(v.index()).copied().flatten())
            .flatten()
    }

    fn peel_const_offset(&self, op: &Op) -> Option<(Value, i64)> {
        match *op {
            Op::Binary(Binary::Iadd, a, b) => self
                .const_i64(b)
                .map(|c| (self.value_of(a), c))
                .or_else(|| self.const_i64(a).map(|c| (self.value_of(b), c))),
            Op::Binary(Binary::Isub, a, b) => self
                .const_i64(b)
                .map(|c| (self.value_of(a), c.wrapping_neg())),
            _ => None,
        }
    }

    fn norm_cmp(&self, cc: IntCC, a: Value, b: Value, lhs: Value) -> Option<(IntCC, Value)> {
        norm_cmp(cc, self.value_of(a), self.value_of(b), lhs)
    }

    fn grow_tables(
        &mut self,
        v: Value,
        inst: Option<Inst>,
        block: Option<Block>,
        imm: Option<i64>,
    ) {
        let len = self.func.types.len();
        self.defs.resize(len, None);
        self.def_blocks.resize(len, None);
        self.consts.resize(len, None);
        self.subst
            .extend((self.subst.len()..len).map(|i| Value(i as u32)));
        self.defs[v.index()] = inst;
        self.def_blocks[v.index()] = block;
        self.consts[v.index()] = imm;
    }

    fn add_param(&mut self, block: Block, ty: Type) -> Value {
        let v = self.func.value(ty);
        self.func.blocks[block.index()].params.push(v);
        self.grow_tables(v, None, Some(block), None);
        v
    }

    fn emit_at(&mut self, block: Block, pos: usize, op: Op, ty: Type) -> Value {
        let v = self.func.value(ty);
        let imm = match op {
            Op::Const(c) if ty == Type::I64 => Some(c),
            _ => None,
        };
        let inst = self.func.add(op, Some(v));
        self.func.blocks[block.index()].insts.insert(pos, inst);
        self.grow_tables(v, Some(inst), Some(block), imm);
        v
    }

    fn emit_before_term(&mut self, block: Block, op: Op, ty: Type) -> Value {
        let pos = self.func.blocks[block.index()]
            .insts
            .len()
            .saturating_sub(1);
        self.emit_at(block, pos, op, ty)
    }

    fn emit_before_inst(&mut self, block: Block, target: Inst, op: Op, ty: Type) -> Value {
        let pos = self.func.blocks[block.index()]
            .insts
            .iter()
            .position(|&i| i == target)
            .unwrap_or_else(|| {
                self.func.blocks[block.index()]
                    .insts
                    .len()
                    .saturating_sub(1)
            });
        self.emit_at(block, pos, op, ty)
    }
}

/// Reorients an integer comparison so `lhs` appears on the left-hand side when present.
pub(crate) fn norm_cmp(cc: IntCC, a: Value, b: Value, lhs: Value) -> Option<(IntCC, Value)> {
    if a == lhs {
        Some((cc, b))
    } else if b == lhs {
        Some((cc.swapped(), a))
    } else {
        None
    }
}

/// Returns the `idx`-th edge argument passed along `arm` of `pred`'s terminator.
pub(crate) fn edge_arg(func: &Function, pred: Block, arm: EdgeArm, idx: usize) -> Option<Value> {
    let last = *func.blocks[pred.index()].insts.last()?;
    let args = match (&func.insts[last.index()].op, arm) {
        (Op::Jump(_, args), EdgeArm::Jump) => args,
        (Op::Brif(_, _, t_args, _, _), EdgeArm::Then) => t_args,
        (Op::Brif(_, _, _, _, e_args), EdgeArm::Else) => e_args,
        _ => return None,
    };
    args.get(idx).copied()
}

fn edge_args_mut(func: &mut Function, pred: Block, arm: EdgeArm) -> Option<&mut Vec<Value>> {
    let &last = func.blocks[pred.index()].insts.last()?;
    match (&mut func.insts[last.index()].op, arm) {
        (Op::Jump(_, args), EdgeArm::Jump) => Some(args),
        (Op::Brif(_, _, t_args, _, _), EdgeArm::Then) => Some(t_args),
        (Op::Brif(_, _, _, _, e_args), EdgeArm::Else) => Some(e_args),
        _ => None,
    }
}

fn push_edge_arg(func: &mut Function, pred: Block, arm: EdgeArm, val: Value) {
    if let Some(args) = edge_args_mut(func, pred, arm) {
        args.push(val);
    }
}

/// Whether block `blk` belongs to `each`'s loop body.
pub(crate) fn is_in_loop(each: &Loop, blk: Block) -> bool {
    each.body.get(blk.index()).copied().unwrap_or(false)
}

fn emit_init_ptr(
    state: &mut State<'_>,
    pred: Block,
    base: Value,
    init_val: Value,
    stride: i64,
    anchor_offset: i64,
) -> Value {
    let base_in_pred = match state.def_blocks.get(base.index()).copied().flatten() {
        Some(def_blk) if state.cfg.dominates(def_blk, pred) => base,
        _ => {
            let Some(inst) = state.defs.get(base.index()).copied().flatten() else {
                return base;
            };
            let op = state.func.insts[inst.index()].op.clone();
            state.emit_before_term(pred, op, Type::I64)
        }
    };
    if let Some(init_c) = state.const_i64(init_val) {
        let offset = init_c.wrapping_mul(stride).wrapping_add(anchor_offset);
        if offset == 0 {
            return base_in_pred;
        }
        let off_val = state.emit_before_term(pred, Op::Const(offset), Type::I64);
        return state.emit_before_term(
            pred,
            Op::Binary(Binary::Iadd, base_in_pred, off_val),
            Type::I64,
        );
    }
    let scaled = if stride == 1 {
        init_val
    } else if stride > 0 && (stride as u64).is_power_of_two() {
        let sh = i64::from((stride as u64).trailing_zeros());
        let sh_val = state.emit_before_term(pred, Op::Const(sh), Type::I64);
        state.emit_before_term(pred, Op::Binary(Binary::Ishl, init_val, sh_val), Type::I64)
    } else {
        let st_val = state.emit_before_term(pred, Op::Const(stride), Type::I64);
        state.emit_before_term(pred, Op::Binary(Binary::Imul, init_val, st_val), Type::I64)
    };
    let ptr = state.emit_before_term(
        pred,
        Op::Binary(Binary::Iadd, base_in_pred, scaled),
        Type::I64,
    );
    if anchor_offset == 0 {
        ptr
    } else {
        let off_val = state.emit_before_term(pred, Op::Const(anchor_offset), Type::I64);
        state.emit_before_term(pred, Op::Binary(Binary::Iadd, ptr, off_val), Type::I64)
    }
}

fn cached_end(
    state: &mut State<'_>,
    ends: &mut Vec<(Value, Value)>,
    len: Value,
    make: impl FnOnce(&mut State<'_>) -> Value,
) -> Value {
    if let Some(&(_, end)) = ends.iter().find(|&&(k, _)| k == len) {
        return end;
    }
    let end = make(state);
    ends.push((len, end));
    end
}

fn emit_step_ptr(
    state: &mut State<'_>,
    pred: Block,
    before: Option<Inst>,
    ptr_phi: Value,
    step: i64,
    stride: i64,
) -> Value {
    let delta = step.wrapping_mul(stride);
    if delta == 0 {
        return ptr_phi;
    }
    let emit = |s: &mut State<'_>, op: Op| match before {
        Some(inst) => s.emit_before_inst(pred, inst, op, Type::I64),
        None => s.emit_before_term(pred, op, Type::I64),
    };
    let delta_val = emit(state, Op::Const(delta));
    emit(state, Op::Binary(Binary::Iadd, ptr_phi, delta_val))
}

fn rewrite_loop(state: &mut State<'_>, each: &Loop) -> bool {
    let Some(edges) = collect_head_edges(state, each) else {
        return false;
    };
    let mut ivs = collect_ivs(state, each, &edges);
    let retired = retire_counter(state, each, &edges, &ivs);
    if retired {
        ivs = collect_ivs(state, each, &edges);
    }
    if ivs.is_empty() {
        return retired;
    }
    let groups = collect_groups(state, each, &ivs);
    if groups.is_empty() {
        return retired;
    }
    let driver_for_iv: Vec<_> = ivs
        .iter()
        .enumerate()
        .map(|(iv_pos, iv_info)| {
            let iv_idxs: Vec<usize> = (0..groups.len())
                .filter(|&i| groups[i].key.iv_pos == iv_pos)
                .collect();
            let iv_refs: Vec<&Group> = iv_idxs.iter().map(|&i| &groups[i]).collect();
            plan_lftr_for_iv(state, each, iv_info, &iv_refs)
                .map(|(local, lftr)| (iv_idxs[local], lftr))
        })
        .collect();
    for (g_idx, group) in groups.into_iter().enumerate() {
        let anchor = if group.occs.iter().any(|o| o.byte_offset == 0) {
            0
        } else {
            group.occs.iter().map(|o| o.byte_offset).min().unwrap_or(0)
        };
        let ptr_phi = state.add_param(each.head, Type::I64);
        let iv_info: &IvInfo = &ivs[group.key.iv_pos];
        let lftr = driver_for_iv[group.key.iv_pos]
            .as_ref()
            .filter(|(d, _)| *d == g_idx)
            .map(|(_, m)| m.as_slice());
        for &(pred, arm, init_val) in &iv_info.init_edges {
            let init_ptr = emit_init_ptr(
                state,
                pred,
                group.key.base,
                init_val,
                group.key.stride,
                anchor,
            );
            push_edge_arg(state.func, pred, arm, init_ptr);
        }
        let mut latch_next_ptr = None;
        for &(pred, arm, step) in &iv_info.back_steps {
            let before = lftr.and_then(|ms| {
                state.func.blocks[pred.index()]
                    .insts
                    .iter()
                    .copied()
                    .find(|&inst| {
                        ms.iter()
                            .any(|m| m.use_next && m.blk == pred && m.inst == inst)
                    })
            });
            let next_ptr = emit_step_ptr(state, pred, before, ptr_phi, step, group.key.stride);
            push_edge_arg(state.func, pred, arm, next_ptr);
            if before.is_some() {
                latch_next_ptr = Some(next_ptr);
            }
        }
        for occ in group.occs {
            let diff = occ.byte_offset.wrapping_sub(anchor);
            if diff == 0 {
                state.subst[occ.res.index()] = ptr_phi;
                state.renamed = true;
            } else {
                let c_val = state.emit_before_inst(occ.blk, occ.inst, Op::Const(diff), Type::I64);
                state.func.insts[occ.inst.index()].op = Op::Binary(Binary::Iadd, ptr_phi, c_val);
            }
        }
        if let Some(matches) = lftr {
            let pre = iv_info.init_edges[0].0;
            let mut end_ptrs: Vec<(Value, Value)> = Vec::new();
            for m in matches {
                let len = state.value_of(m.len);
                let end_ptr = cached_end(state, &mut end_ptrs, len, |s| {
                    emit_init_ptr(s, pre, group.key.base, len, group.key.stride, anchor)
                });
                let Some(lhs) = (if m.use_next {
                    latch_next_ptr
                } else {
                    Some(ptr_phi)
                }) else {
                    continue;
                };
                state.func.insts[m.inst.index()].op = Op::Icmp(m.new_cc, lhs, end_ptr);
            }
        }
    }
    true
}

/// Replaces loop-invariant-base scaled induction-variable address calculations inside natural loops with pointer block parameters.
pub(crate) fn run(func: &mut Function) -> Domain {
    if func.entry.is_none() || func.coroutine.is_some() {
        return Domain::NONE;
    }
    let cfg = func.cfg();
    let loops = cfg.loops();
    if loops.list.is_empty() {
        return Domain::NONE;
    }
    let mut state = State::new(func, &cfg);
    let mut changed = false;
    for each in &loops.list {
        changed |= rewrite_loop(&mut state, each);
    }
    if state.renamed {
        rename(state.func, &state.subst);
    }
    if changed {
        Domain::VALUES
    } else {
        Domain::NONE
    }
}
