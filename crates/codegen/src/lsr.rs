use rustc_hash::FxHashMap as HashMap;

use crate::cfg::{Cfg, Loop};
use crate::fold::{defs, evaluate_wrapped};
use crate::ir::{Binary, Block, Domain, Function, Inst, Op, Type, Value};
use crate::merge::{find, identity, rename};

const MAX_PHIS_PER_LOOP: usize = 4;

#[derive(Clone, Copy)]
enum EdgeArm {
    Jump,
    Then,
    Else,
}

struct IvInfo {
    iv: Value,
    init_edges: Vec<(Block, EdgeArm, Value)>,
    back_steps: Vec<(Block, EdgeArm, i64)>,
}

struct Occ {
    blk: Block,
    inst: Inst,
    res: Value,
    byte_offset: i64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GroupKey {
    base: Value,
    iv_pos: usize,
    stride: i64,
}

struct Group {
    key: GroupKey,
    has_mul: bool,
    total_uses: u32,
    occs: Vec<Occ>,
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
                if let Some(imm) = evaluate_wrapped(func, &data.op, func.ty(res), &|v| consts[v.index()]) {
                    consts[res.index()] = Some(imm);
                }
            }
        }
        let subst = identity(func.types.len());
        Self { func, cfg, defs, def_blocks, consts, subst, renamed: false }
    }

    fn value_of(&self, v: Value) -> Value {
        find(&self.subst, v)
    }

    fn const_i64(&self, v: Value) -> Option<i64> {
        let v = self.value_of(v);
        (self.func.ty(v) == Type::I64).then(|| self.consts.get(v.index()).copied().flatten()).flatten()
    }

    fn grow_tables(&mut self, v: Value, inst: Option<Inst>, block: Option<Block>, imm: Option<i64>) {
        let len = self.func.types.len();
        self.defs.resize(len, None);
        self.def_blocks.resize(len, None);
        self.consts.resize(len, None);
        self.subst.extend((self.subst.len()..len).map(|i| Value(i as u32)));
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
        let pos = self.func.blocks[block.index()].insts.len().saturating_sub(1);
        self.emit_at(block, pos, op, ty)
    }

    fn emit_before_inst(&mut self, block: Block, target: Inst, op: Op, ty: Type) -> Value {
        let pos = self.func.blocks[block.index()]
            .insts
            .iter()
            .position(|&i| i == target)
            .unwrap_or_else(|| self.func.blocks[block.index()].insts.len().saturating_sub(1));
        self.emit_at(block, pos, op, ty)
    }
}

fn edge_arg(func: &Function, pred: Block, arm: EdgeArm, idx: usize) -> Option<Value> {
    let last = *func.blocks[pred.index()].insts.last()?;
    let args = match (&func.insts[last.index()].op, arm) {
        (Op::Jump(_, args), EdgeArm::Jump) => args,
        (Op::Brif(_, _, t_args, _, _), EdgeArm::Then) => t_args,
        (Op::Brif(_, _, _, _, e_args), EdgeArm::Else) => e_args,
        _ => return None,
    };
    args.get(idx).copied()
}

fn push_edge_arg(func: &mut Function, pred: Block, arm: EdgeArm, val: Value) {
    let Some(&last) = func.blocks[pred.index()].insts.last() else { return };
    match (&mut func.insts[last.index()].op, arm) {
        (Op::Jump(_, args), EdgeArm::Jump) => args.push(val),
        (Op::Brif(_, _, t_args, _, _), EdgeArm::Then) => t_args.push(val),
        (Op::Brif(_, _, _, _, e_args), EdgeArm::Else) => e_args.push(val),
        _ => {}
    }
}

fn collect_head_edges(state: &State<'_>, each: &Loop) -> Option<Vec<(Block, EdgeArm, bool)>> {
    if Some(each.head) == state.func.entry {
        return None;
    }
    let param_len = state.func.blocks[each.head.index()].params.len();
    if param_len == 0 {
        return None;
    }
    let mut seen_preds = Vec::new();
    for &pred in state.cfg.preds_of(each.head) {
        if !seen_preds.contains(&pred) {
            seen_preds.push(pred);
        }
    }
    if seen_preds.is_empty() {
        return None;
    }
    let mut edges = Vec::new();
    let mut has_outside = false;
    let mut has_inside = false;
    for pred in seen_preds {
        if !state.cfg.reachable.get(pred.index()).copied().unwrap_or(false) {
            return None;
        }
        let inside = each.body.get(pred.index()).copied().unwrap_or(false);
        match state.func.last(pred) {
            Some(Op::Jump(target, args)) if *target == each.head => {
                if args.len() != param_len {
                    return None;
                }
                edges.push((pred, EdgeArm::Jump, inside));
            }
            Some(Op::Brif(_, t_blk, t_args, e_blk, e_args)) => {
                if *t_blk == each.head {
                    if t_args.len() != param_len {
                        return None;
                    }
                    edges.push((pred, EdgeArm::Then, inside));
                }
                if *e_blk == each.head {
                    if e_args.len() != param_len {
                        return None;
                    }
                    edges.push((pred, EdgeArm::Else, inside));
                }
            }
            _ => return None,
        }
        has_outside |= !inside;
        has_inside |= inside;
    }
    (has_outside && has_inside).then_some(edges)
}

fn iv_step_from(state: &State<'_>, each: &Loop, iv: Value, val: Value, depth: u8) -> Option<i64> {
    let val = state.value_of(val);
    if val == iv {
        return Some(0);
    }
    if depth >= 4 {
        return None;
    }
    let inst = state.defs.get(val.index()).copied().flatten()?;
    let def_blk = state.def_blocks.get(val.index()).copied().flatten()?;
    if !each.body.get(def_blk.index()).copied().unwrap_or(false) {
        return None;
    }
    match state.func.insts[inst.index()].op {
        Op::Binary(Binary::Iadd, a, b) => {
            if let Some(c) = state.const_i64(b) {
                iv_step_from(state, each, iv, a, depth + 1).map(|s| s.wrapping_add(c))
            } else if let Some(c) = state.const_i64(a) {
                iv_step_from(state, each, iv, b, depth + 1).map(|s| s.wrapping_add(c))
            } else {
                None
            }
        }
        Op::Binary(Binary::Isub, a, b) => {
            let c = state.const_i64(b)?;
            iv_step_from(state, each, iv, a, depth + 1).map(|s| s.wrapping_sub(c))
        }
        _ => None,
    }
}

fn collect_ivs(state: &State<'_>, each: &Loop, edges: &[(Block, EdgeArm, bool)]) -> Vec<IvInfo> {
    let mut ivs = Vec::new();
    for (iv_idx, &iv) in state.func.blocks[each.head.index()].params.iter().enumerate() {
        if state.func.ty(iv) != Type::I64 {
            continue;
        }
        let mut init_edges = Vec::new();
        let mut back_steps = Vec::new();
        let mut any_nonzero = false;
        let mut valid = true;
        for &(pred, arm, inside) in edges {
            let Some(arg) = edge_arg(state.func, pred, arm, iv_idx) else {
                valid = false;
                break;
            };
            if inside {
                let Some(step) = iv_step_from(state, each, iv, arg, 0) else {
                    valid = false;
                    break;
                };
                any_nonzero |= step != 0;
                back_steps.push((pred, arm, step));
            } else {
                init_edges.push((pred, arm, state.value_of(arg)));
            }
        }
        if valid && any_nonzero {
            ivs.push(IvInfo { iv, init_edges, back_steps });
        }
    }
    ivs
}

fn loop_addr_uses(state: &State<'_>, each: &Loop) -> Vec<u32> {
    let mut uses = vec![0_u32; state.func.types.len()];
    let mut mark = |raw: Value| {
        let mut cur = state.value_of(raw);
        uses[cur.index()] += 1;
        for _ in 0..2 {
            let Some(inst) = state.defs.get(cur.index()).copied().flatten() else { break };
            let Some(blk) = state.def_blocks.get(cur.index()).copied().flatten() else { break };
            if !each.body.get(blk.index()).copied().unwrap_or(false) {
                break;
            }
            let next = match state.func.insts[inst.index()].op {
                Op::Binary(Binary::Iadd, a, b) => {
                    if state.const_i64(b).is_some() {
                        Some(state.value_of(a))
                    } else if state.const_i64(a).is_some() {
                        Some(state.value_of(b))
                    } else {
                        None
                    }
                }
                Op::Binary(Binary::Isub, a, b) if state.const_i64(b).is_some() => Some(state.value_of(a)),
                _ => None,
            };
            let Some(inner) = next else { break };
            uses[inner.index()] += 1;
            cur = inner;
        }
    };
    for b in 0..state.func.blocks.len() {
        if !each.body.get(b).copied().unwrap_or(false) {
            continue;
        }
        for &inst in &state.func.blocks[b].insts {
            match state.func.insts[inst.index()].op {
                Op::Load(addr, _) | Op::AtomicLoad(addr) => mark(addr),
                Op::Store(_, addr, _)
                | Op::AtomicStore(_, addr)
                | Op::AtomicRmw(_, addr, _)
                | Op::AtomicCas(addr, _, _) => mark(addr),
                Op::Copy(dst, src, _) | Op::Move(dst, src, _) => {
                    mark(dst);
                    mark(src);
                }
                _ => {}
            }
        }
    }
    uses
}

fn is_pow2_scale(stride: i64) -> bool {
    matches!(stride, 1 | 2 | 4 | 8)
}

fn match_affine(state: &State<'_>, each: &Loop, ivs: &[IvInfo], raw: Value, depth: u8) -> Option<(usize, i64, i64, bool)> {
    let v = state.value_of(raw);
    if state.func.ty(v) != Type::I64 {
        return None;
    }
    for (pos, info) in ivs.iter().enumerate() {
        if v == info.iv {
            return Some((pos, 1, 0, false));
        }
    }
    if depth >= 5 {
        return None;
    }
    let inst = state.defs.get(v.index()).copied().flatten()?;
    let def_blk = state.def_blocks.get(v.index()).copied().flatten()?;
    if !each.body.get(def_blk.index()).copied().unwrap_or(false) {
        return None;
    }
    match state.func.insts[inst.index()].op {
        Op::Binary(Binary::Iadd, a, b) => {
            if let Some(c) = state.const_i64(b) {
                let (pos, s, off, m) = match_affine(state, each, ivs, a, depth + 1)?;
                Some((pos, s, off.wrapping_add(c), m))
            } else if let Some(c) = state.const_i64(a) {
                let (pos, s, off, m) = match_affine(state, each, ivs, b, depth + 1)?;
                Some((pos, s, off.wrapping_add(c), m))
            } else {
                let (p1, s1, o1, m1) = match_affine(state, each, ivs, a, depth + 1)?;
                let (p2, s2, o2, m2) = match_affine(state, each, ivs, b, depth + 1)?;
                (p1 == p2).then_some((p1, s1.wrapping_add(s2), o1.wrapping_add(o2), m1 || m2 || (s1 != 0 && s2 != 0)))
            }
        }
        Op::Binary(Binary::Isub, a, b) => {
            if let Some(c) = state.const_i64(b) {
                let (pos, s, off, m) = match_affine(state, each, ivs, a, depth + 1)?;
                Some((pos, s, off.wrapping_sub(c), m))
            } else {
                let (p1, s1, o1, m1) = match_affine(state, each, ivs, a, depth + 1)?;
                let (p2, s2, o2, m2) = match_affine(state, each, ivs, b, depth + 1)?;
                (p1 == p2).then_some((p1, s1.wrapping_sub(s2), o1.wrapping_sub(o2), m1 || m2 || (s1 != 0 && s2 != 0)))
            }
        }
        Op::Binary(Binary::Ishl, idx, sh) => {
            let k = (state.const_i64(sh)? as u32) & 63;
            let scale = (1_u64 << k) as i64;
            let (pos, s, off, m) = match_affine(state, each, ivs, idx, depth + 1)?;
            Some((pos, s.wrapping_mul(scale), off.wrapping_mul(scale), m))
        }
        Op::Binary(Binary::Imul, a, b) => [(a, b), (b, a)].into_iter().find_map(|(idx, sc)| {
            let stride = state.const_i64(sc)?;
            if stride == 0 {
                return None;
            }
            let (pos, s, off, m) = match_affine(state, each, ivs, idx, depth + 1)?;
            let eff = s.wrapping_mul(stride);
            Some((pos, eff, off.wrapping_mul(stride), m || !is_pow2_scale(stride)))
        }),
        _ => None,
    }
}

fn is_outside_dom(state: &State<'_>, each: &Loop, val: Value) -> bool {
    let Some(def_blk) = state.def_blocks.get(val.index()).copied().flatten() else { return false };
    !each.body.get(def_blk.index()).copied().unwrap_or(false) && state.cfg.dominates(def_blk, each.head)
}

fn is_invariant_base(state: &State<'_>, each: &Loop, base: Value) -> bool {
    if is_outside_dom(state, each, base) {
        return true;
    }
    state
        .defs
        .get(base.index())
        .copied()
        .flatten()
        .is_some_and(|inst| matches!(state.func.insts[inst.index()].op, Op::Const(_) | Op::Symbol(_) | Op::SlotAddr(_) | Op::StackAddr(..)))
}

fn unwrap_base(state: &State<'_>, each: &Loop, raw: Value) -> Option<(Value, i64)> {
    let mut cur = state.value_of(raw);
    let mut extra = 0_i64;
    for _ in 0..3 {
        if is_invariant_base(state, each, cur) {
            return Some((cur, extra));
        }
        let inst = state.defs.get(cur.index()).copied().flatten()?;
        match state.func.insts[inst.index()].op {
            Op::Binary(Binary::Iadd, a, b) => {
                if let Some(c) = state.const_i64(b) {
                    cur = state.value_of(a);
                    extra = extra.wrapping_add(c);
                } else if let Some(c) = state.const_i64(a) {
                    cur = state.value_of(b);
                    extra = extra.wrapping_add(c);
                } else {
                    return None;
                }
            }
            Op::Binary(Binary::Isub, a, b) => {
                let c = state.const_i64(b)?;
                cur = state.value_of(a);
                extra = extra.wrapping_sub(c);
            }
            _ => return None,
        }
    }
    is_invariant_base(state, each, cur).then_some((cur, extra))
}

fn collect_groups(state: &State<'_>, each: &Loop, ivs: &[IvInfo]) -> Vec<Group> {
    let addr_uses = loop_addr_uses(state, each);
    let mut index_of: HashMap<GroupKey, usize> = HashMap::default();
    let mut groups: Vec<Group> = Vec::new();
    for &blk in &state.cfg.rpo {
        if !each.body.get(blk.index()).copied().unwrap_or(false) {
            continue;
        }
        for &inst in &state.func.blocks[blk.index()].insts {
            let data = &state.func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if state.func.ty(res) != Type::I64 {
                continue;
            }
            let uses = addr_uses.get(res.index()).copied().unwrap_or(0);
            if uses == 0 {
                continue;
            }
            let matched = match data.op {
                Op::Binary(Binary::Iadd, a, b) => [(a, b), (b, a)].into_iter().find_map(|(base_raw, scaled)| {
                    let (base, base_off) = unwrap_base(state, each, base_raw)?;
                    let (iv_pos, stride, scaled_off, is_mul) = match_affine(state, each, ivs, scaled, 0)?;
                    (stride != 0).then_some((base, iv_pos, stride, base_off.wrapping_add(scaled_off), is_mul))
                }),
                Op::Binary(Binary::Isub, base_raw, scaled) => unwrap_base(state, each, base_raw).and_then(|(base, base_off)| {
                    let (iv_pos, stride, scaled_off, is_mul) = match_affine(state, each, ivs, scaled, 0)?;
                    (stride != 0).then_some((base, iv_pos, stride.wrapping_neg(), base_off.wrapping_sub(scaled_off), is_mul))
                }),
                _ => None,
            };
            let Some((base, iv_pos, stride, byte_offset, is_mul)) = matched else { continue };
            let key = GroupKey { base, iv_pos, stride };
            let occ = Occ { blk, inst, res, byte_offset };
            match index_of.get(&key).copied() {
                Some(idx) => {
                    groups[idx].has_mul |= is_mul;
                    groups[idx].total_uses += uses;
                    groups[idx].occs.push(occ);
                }
                None => {
                    let idx = groups.len();
                    index_of.insert(key, idx);
                    groups.push(Group { key, has_mul: is_mul, total_uses: uses, occs: vec![occ] });
                }
            }
        }
    }
    groups.retain(|g| {
        let non_zero_off = g.occs.iter().any(|o| o.byte_offset != 0);
        let non_dom_base = !is_outside_dom(state, each, g.key.base);
        if g.key.stride == 1 {
            g.occs.len() >= 2 || non_zero_off || non_dom_base
        } else {
            g.has_mul || g.key.stride <= 0 || g.key.stride > 8 || g.occs.len() >= 2 || non_zero_off || g.total_uses >= 2 || non_dom_base
        }
    });
    if groups.len() > MAX_PHIS_PER_LOOP {
        let mut order: Vec<usize> = (0..groups.len()).collect();
        order.sort_by_key(|&i| (std::cmp::Reverse(groups[i].has_mul), std::cmp::Reverse(groups[i].total_uses), i));
        order.truncate(MAX_PHIS_PER_LOOP);
        order.sort_unstable();
        groups = order.into_iter().map(|i| std::mem::replace(&mut groups[i], Group { key: GroupKey { base: Value(0), iv_pos: 0, stride: 0 }, has_mul: false, total_uses: 0, occs: Vec::new() })).collect();
    }
    groups
}

fn emit_init_ptr(state: &mut State<'_>, pred: Block, base: Value, init_val: Value, stride: i64, anchor_offset: i64) -> Value {
    let base_in_pred = match state.def_blocks.get(base.index()).copied().flatten() {
        Some(def_blk) if state.cfg.dominates(def_blk, pred) => base,
        _ => {
            let Some(inst) = state.defs.get(base.index()).copied().flatten() else { return base };
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
        return state.emit_before_term(pred, Op::Binary(Binary::Iadd, base_in_pred, off_val), Type::I64);
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
    let ptr = state.emit_before_term(pred, Op::Binary(Binary::Iadd, base_in_pred, scaled), Type::I64);
    if anchor_offset == 0 {
        ptr
    } else {
        let off_val = state.emit_before_term(pred, Op::Const(anchor_offset), Type::I64);
        state.emit_before_term(pred, Op::Binary(Binary::Iadd, ptr, off_val), Type::I64)
    }
}

fn emit_step_ptr(state: &mut State<'_>, pred: Block, ptr_phi: Value, step: i64, stride: i64) -> Value {
    let delta = step.wrapping_mul(stride);
    if delta == 0 {
        return ptr_phi;
    }
    let delta_val = state.emit_before_term(pred, Op::Const(delta), Type::I64);
    state.emit_before_term(pred, Op::Binary(Binary::Iadd, ptr_phi, delta_val), Type::I64)
}

fn rewrite_loop(state: &mut State<'_>, each: &Loop) -> bool {
    let Some(edges) = collect_head_edges(state, each) else { return false };
    let ivs = collect_ivs(state, each, &edges);
    if ivs.is_empty() {
        return false;
    }
    let groups = collect_groups(state, each, &ivs);
    if groups.is_empty() {
        return false;
    }
    for group in groups {
        let anchor = if group.occs.iter().any(|o| o.byte_offset == 0) {
            0
        } else {
            let mut min_off = group.occs[0].byte_offset;
            for occ in &group.occs[1..] {
                if occ.byte_offset < min_off {
                    min_off = occ.byte_offset;
                }
            }
            min_off
        };
        let ptr_phi = state.add_param(each.head, Type::I64);
        let iv_info = &ivs[group.key.iv_pos];
        for &(pred, arm, init_val) in &iv_info.init_edges {
            let init_ptr = emit_init_ptr(state, pred, group.key.base, init_val, group.key.stride, anchor);
            push_edge_arg(state.func, pred, arm, init_ptr);
        }
        for &(pred, arm, step) in &iv_info.back_steps {
            let next_ptr = emit_step_ptr(state, pred, ptr_phi, step, group.key.stride);
            push_edge_arg(state.func, pred, arm, next_ptr);
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
    if changed { Domain::VALUES } else { Domain::NONE }
}
