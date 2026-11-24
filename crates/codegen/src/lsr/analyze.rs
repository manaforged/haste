use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Loop;
use crate::ir::{Binary, Block, Inst, Op, Type, Value};

use super::lftr::{LftrMatch, collect_lftr_matches, dead_iv_chain};
use super::{EdgeArm, State, edge_arg, is_in_loop};

const MAX_PHIS_PER_LOOP: usize = 4;

pub(super) struct IvInfo {
    pub(super) iv: Value,
    pub(super) iv_idx: usize,
    pub(super) init_edges: Vec<(Block, EdgeArm, Value)>,
    pub(super) back_steps: Vec<(Block, EdgeArm, i64)>,
}

pub(super) struct Occ {
    pub(super) blk: Block,
    pub(super) inst: Inst,
    pub(super) res: Value,
    pub(super) byte_offset: i64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct GroupKey {
    pub(super) base: Value,
    pub(super) iv_pos: usize,
    pub(super) stride: i64,
}

pub(super) struct Group {
    pub(super) key: GroupKey,
    has_mul: bool,
    total_uses: u32,
    pub(super) occs: Vec<Occ>,
}

pub(super) fn collect_head_edges(
    state: &State<'_>,
    each: &Loop,
) -> Option<Vec<(Block, EdgeArm, bool)>> {
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
        if !state
            .cfg
            .reachable
            .get(pred.index())
            .copied()
            .unwrap_or(false)
        {
            return None;
        }
        let inside = is_in_loop(each, pred);
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
    if !is_in_loop(each, def_blk) {
        return None;
    }
    let (inner, c) = state.peel_const_offset(&state.func.insts[inst.index()].op)?;
    iv_step_from(state, each, iv, inner, depth + 1).map(|s| s.wrapping_add(c))
}

pub(super) fn collect_ivs(
    state: &State<'_>,
    each: &Loop,
    edges: &[(Block, EdgeArm, bool)],
) -> Vec<IvInfo> {
    let mut ivs = Vec::new();
    for (iv_idx, &iv) in state.func.blocks[each.head.index()]
        .params
        .iter()
        .enumerate()
    {
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
            ivs.push(IvInfo {
                iv,
                iv_idx,
                init_edges,
                back_steps,
            });
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
            let Some(inst) = state.defs.get(cur.index()).copied().flatten() else {
                break;
            };
            let Some(blk) = state.def_blocks.get(cur.index()).copied().flatten() else {
                break;
            };
            if !is_in_loop(each, blk) {
                break;
            }
            let Some((inner, _)) = state.peel_const_offset(&state.func.insts[inst.index()].op)
            else {
                break;
            };
            uses[inner.index()] += 1;
            cur = inner;
        }
    };
    for b in 0..state.func.blocks.len() {
        if !is_in_loop(each, Block(b as u32)) {
            continue;
        }
        for &inst in &state.func.blocks[b].insts {
            match state.func.insts[inst.index()].op {
                Op::Load(addr, _) | Op::AtomicLoad(addr, _) => mark(addr),
                Op::Store(_, addr, _)
                | Op::AtomicStore(_, addr, _)
                | Op::AtomicRmw(_, _, addr, _)
                | Op::AtomicCas(_, addr, _, _) => mark(addr),
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

fn match_affine(
    state: &State<'_>,
    each: &Loop,
    ivs: &[IvInfo],
    raw: Value,
    depth: u8,
) -> Option<(usize, i64, i64, bool)> {
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
    if !is_in_loop(each, def_blk) {
        return None;
    }
    let inst_op = &state.func.insts[inst.index()].op;
    if let Some((inner, c)) = state.peel_const_offset(inst_op) {
        let (pos, s, off, m) = match_affine(state, each, ivs, inner, depth + 1)?;
        return Some((pos, s, off.wrapping_add(c), m));
    }
    match *inst_op {
        Op::Binary(bin @ (Binary::Iadd | Binary::Isub), a, b) => {
            let (p1, s1, o1, m1) = match_affine(state, each, ivs, a, depth + 1)?;
            let (p2, s2, o2, m2) = match_affine(state, each, ivs, b, depth + 1)?;
            let sub = matches!(bin, Binary::Isub);
            let s = if sub {
                s1.wrapping_sub(s2)
            } else {
                s1.wrapping_add(s2)
            };
            let o = if sub {
                o1.wrapping_sub(o2)
            } else {
                o1.wrapping_add(o2)
            };
            (p1 == p2).then_some((p1, s, o, m1 || m2 || (s1 != 0 && s2 != 0)))
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
            Some((
                pos,
                eff,
                off.wrapping_mul(stride),
                m || !is_pow2_scale(stride),
            ))
        }),
        _ => None,
    }
}

fn is_outside_dom(state: &State<'_>, each: &Loop, val: Value) -> bool {
    let Some(def_blk) = state.def_blocks.get(val.index()).copied().flatten() else {
        return false;
    };
    !is_in_loop(each, def_blk) && state.cfg.dominates(def_blk, each.head)
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
        .is_some_and(|inst| {
            matches!(
                state.func.insts[inst.index()].op,
                Op::Const(_) | Op::Symbol(_) | Op::SlotAddr(_) | Op::StackAddr(..)
            )
        })
}

fn unwrap_base(state: &State<'_>, each: &Loop, raw: Value) -> Option<(Value, i64)> {
    let mut cur = state.value_of(raw);
    let mut extra = 0_i64;
    for _ in 0..3 {
        if is_invariant_base(state, each, cur) {
            return Some((cur, extra));
        }
        let inst = state.defs.get(cur.index()).copied().flatten()?;
        let (inner, c) = state.peel_const_offset(&state.func.insts[inst.index()].op)?;
        cur = inner;
        extra = extra.wrapping_add(c);
    }
    is_invariant_base(state, each, cur).then_some((cur, extra))
}

fn standard_group_ok(state: &State<'_>, each: &Loop, g: &Group) -> bool {
    let off_or_nondom =
        g.occs.iter().any(|o| o.byte_offset != 0) || !is_outside_dom(state, each, g.key.base);
    let hw_scale = matches!(g.key.stride, 1 | 2 | 4 | 8) && !g.has_mul;
    g.occs.len() >= 2 || off_or_nondom || !hw_scale
}

pub(super) fn plan_lftr_for_iv(
    state: &State<'_>,
    each: &Loop,
    iv_info: &IvInfo,
    iv_groups: &[&Group],
) -> Option<(usize, Vec<LftrMatch>)> {
    let local_idx = (0..iv_groups.len()).min_by_key(|&i| {
        let g = iv_groups[i];
        (
            g.key.stride != 1,
            (g.key.stride & 1) == 0,
            std::cmp::Reverse(g.total_uses),
            i,
        )
    })?;
    let lftr = collect_lftr_matches(state, each, iv_info, iv_groups[local_idx].key.stride);
    (!lftr.is_empty() && dead_iv_chain(state, each, iv_info, iv_groups, &lftr).is_some())
        .then_some((local_idx, lftr))
}

pub(super) fn collect_groups(state: &State<'_>, each: &Loop, ivs: &[IvInfo]) -> Vec<Group> {
    let addr_uses = loop_addr_uses(state, each);
    let mut index_of: HashMap<GroupKey, usize> = HashMap::default();
    let mut groups: Vec<Group> = Vec::new();
    for &blk in state.cfg.rpo.iter().filter(|&&b| is_in_loop(each, b)) {
        for &inst in &state.func.blocks[blk.index()].insts {
            let data = &state.func.insts[inst.index()];
            let Some(res) = data.result.filter(|&r| state.func.ty(r) == Type::I64) else {
                continue;
            };
            let uses = addr_uses.get(res.index()).copied().unwrap_or(0);
            if uses == 0 {
                continue;
            }
            let matched = match data.op {
                Op::Binary(Binary::Iadd, a, b) => {
                    [(a, b), (b, a)].into_iter().find_map(|(base_raw, scaled)| {
                        let (base, base_off) = unwrap_base(state, each, base_raw)?;
                        let (iv_pos, stride, scaled_off, is_mul) =
                            match_affine(state, each, ivs, scaled, 0)?;
                        (stride != 0).then_some((
                            base,
                            iv_pos,
                            stride,
                            base_off.wrapping_add(scaled_off),
                            is_mul,
                        ))
                    })
                }
                Op::Binary(Binary::Isub, base_raw, scaled) => unwrap_base(state, each, base_raw)
                    .and_then(|(base, base_off)| {
                        let (iv_pos, stride, scaled_off, is_mul) =
                            match_affine(state, each, ivs, scaled, 0)?;
                        (stride != 0).then_some((
                            base,
                            iv_pos,
                            stride.wrapping_neg(),
                            base_off.wrapping_sub(scaled_off),
                            is_mul,
                        ))
                    }),
                _ => None,
            };
            let Some((base, iv_pos, stride, byte_offset, is_mul)) = matched else {
                continue;
            };
            let key = GroupKey {
                base,
                iv_pos,
                stride,
            };
            let occ = Occ {
                blk,
                inst,
                res,
                byte_offset,
            };
            match index_of.get(&key).copied() {
                Some(idx) => {
                    groups[idx].has_mul |= is_mul;
                    groups[idx].total_uses += uses;
                    groups[idx].occs.push(occ);
                }
                None => {
                    let idx = groups.len();
                    index_of.insert(key, idx);
                    groups.push(Group {
                        key,
                        has_mul: is_mul,
                        total_uses: uses,
                        occs: vec![occ],
                    });
                }
            }
        }
    }
    let mut lftr_iv = vec![false; ivs.len()];
    for (iv_pos, iv_info) in ivs.iter().enumerate() {
        let iv_groups: Vec<&Group> = groups.iter().filter(|g| g.key.iv_pos == iv_pos).collect();
        lftr_iv[iv_pos] = plan_lftr_for_iv(state, each, iv_info, &iv_groups).is_some();
    }
    groups.retain(|g| standard_group_ok(state, each, g) || lftr_iv[g.key.iv_pos]);
    if groups.len() > MAX_PHIS_PER_LOOP {
        let mut order: Vec<usize> = (0..groups.len()).collect();
        order.sort_by_key(|&i| {
            (
                std::cmp::Reverse(groups[i].has_mul),
                std::cmp::Reverse(groups[i].total_uses),
                i,
            )
        });
        order.truncate(MAX_PHIS_PER_LOOP);
        order.sort_unstable();
        groups = order
            .into_iter()
            .map(|i| {
                std::mem::replace(
                    &mut groups[i],
                    Group {
                        key: GroupKey {
                            base: Value(0),
                            iv_pos: 0,
                            stride: 0,
                        },
                        has_mul: false,
                        total_uses: 0,
                        occs: Vec::new(),
                    },
                )
            })
            .collect();
    }
    groups
}
