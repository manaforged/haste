use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Function, Inst, IntCC, Op, Type, Value};

use super::{emit_bin, emit_const, emit_icmp, prune_dead_local};

#[derive(Clone, Copy)]
enum FoldedCmp {
    Const(bool),
    Single(IntCC, u64),
    Range(u64, u64),
    MaskEq(u64, u64, bool),
}

fn def_const(func: &Function, defs: &[Option<Inst>], val: Value) -> Option<i64> {
    match func.insts[defs[val.index()]?.index()].op {
        Op::Const(imm) => Some(imm),
        _ => None,
    }
}

fn parse_const_cmp(
    func: &Function,
    defs: &[Option<Inst>],
    mut cond: Value,
) -> Option<(Value, IntCC, i64)> {
    let mut invert = false;
    for _ in 0..2 {
        match func.insts[defs[cond.index()]?.index()].op {
            Op::Binary(Binary::Bxor, a, b) if def_const(func, defs, b) == Some(1) => {
                cond = a;
                invert = !invert;
            }
            Op::Binary(Binary::Bxor, a, b) if def_const(func, defs, a) == Some(1) => {
                cond = b;
                invert = !invert;
            }
            Op::Icmp(cc, a, b) if func.ty(a).is_small_int() => {
                let (x, eff_cc, imm) = if let Some(k) = def_const(func, defs, b) {
                    (a, cc, k)
                } else {
                    (b, cc.swapped(), def_const(func, defs, a)?)
                };
                return Some((x, if invert { eff_cc.inverse() } else { eff_cc }, imm));
            }
            _ => return None,
        }
    }
    None
}

fn range_bounds(
    cc: IntCC,
    raw_k: i64,
    (u_max, sign_bit): (u64, u64),
    signed: bool,
) -> Option<(u64, u64)> {
    let key = ((raw_k as u64) & u_max) ^ sign_bit;
    if cc == IntCC::Equal {
        return Some((key, key));
    }
    if cc == IntCC::NotEqual {
        return (key == 0)
            .then_some((1, u_max))
            .or_else(|| (key == u_max).then_some((0, u_max - 1)));
    }
    if cc.is_signed() != signed {
        return None;
    }
    match cc {
        IntCC::SignedGreaterThanOrEqual | IntCC::UnsignedGreaterThanOrEqual => Some((key, u_max)),
        IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => {
            (key < u_max).then_some((key + 1, u_max))
        }
        IntCC::SignedLessThanOrEqual | IntCC::UnsignedLessThanOrEqual => Some((0, key)),
        IntCC::SignedLessThan | IntCC::UnsignedLessThan => (key > 0).then_some((0, key - 1)),
        _ => None,
    }
}

fn fold_interval_pair(
    ty: Type,
    (cc1, k1): (IntCC, i64),
    (cc2, k2): (IntCC, i64),
    signed: bool,
) -> Option<FoldedCmp> {
    let (u_max, sign_bit) = (
        ty.unsigned_max(),
        if signed { 1_u64 << (ty.bits() - 1) } else { 0 },
    );
    let (a_lo, a_hi) = range_bounds(cc1, k1, (u_max, sign_bit), signed)?;
    let (b_lo, b_hi) = range_bounds(cc2, k2, (u_max, sign_bit), signed)?;
    let (lo, hi) = (a_lo.max(b_lo), a_hi.min(b_hi));
    if lo > hi {
        return Some(FoldedCmp::Const(false));
    }
    if lo == 0 && hi == u_max {
        return Some(FoldedCmp::Const(true));
    }
    let span = hi - lo;
    if span >= u_max {
        return None;
    }
    let (lo_raw, hi_raw) = (lo ^ sign_bit, hi ^ sign_bit);
    if lo == hi {
        return Some(FoldedCmp::Single(IntCC::Equal, lo_raw));
    }
    if hi == u_max {
        return Some(FoldedCmp::Single(
            if signed {
                IntCC::SignedGreaterThanOrEqual
            } else {
                IntCC::UnsignedGreaterThanOrEqual
            },
            lo_raw,
        ));
    }
    if lo == 0 {
        return Some(FoldedCmp::Single(
            if signed {
                IntCC::SignedLessThanOrEqual
            } else {
                IntCC::UnsignedLessThanOrEqual
            },
            hi_raw,
        ));
    }
    if lo_raw == 0 {
        return Some(FoldedCmp::Single(IntCC::UnsignedLessThanOrEqual, span));
    }
    Some(FoldedCmp::Range(lo_raw, span))
}

fn fold_and_cmps(ty: Type, (cc1, k1): (IntCC, i64), (cc2, k2): (IntCC, i64)) -> Option<FoldedCmp> {
    if cc1 == IntCC::NotEqual && cc2 == IntCC::NotEqual {
        let u_max = ty.unsigned_max();
        let (u1, u2) = ((k1 as u64) & u_max, (k2 as u64) & u_max);
        if u1 == u2 {
            return Some(FoldedCmp::Single(IntCC::NotEqual, u1));
        }
        let d = u1 ^ u2;
        if d.is_power_of_two() {
            return Some(FoldedCmp::MaskEq(u_max & !d, u1 & !d, true));
        }
    }
    fold_interval_pair(ty, (cc1, k1), (cc2, k2), false)
        .or_else(|| fold_interval_pair(ty, (cc1, k1), (cc2, k2), true))
}

fn mid_self_contained(func: &Function, uses: &[u32], mid: usize) -> bool {
    let insts = &func.blocks[mid].insts;
    let Some((&term, body)) = insts.split_last() else {
        return false;
    };
    if insts.len() > 5
        || !body
            .iter()
            .all(|&i| func.insts[i.index()].op.is_speculable())
    {
        return false;
    }
    let Op::Brif(cond, _, _, _, _) = func.insts[term.index()].op else {
        return false;
    };
    body.iter().all(|&i| {
        let Some(res) = func.insts[i.index()].result else {
            return true;
        };
        let internal: u32 = body
            .iter()
            .map(|&j| {
                let mut n = 0;
                func.insts[j.index()].op.for_operands(|v| {
                    if v == res {
                        n += 1;
                    }
                });
                n
            })
            .sum::<u32>()
            + u32::from(cond == res);
        uses[res.index()] == internal
    })
}

fn emit_folded_cmp(
    func: &mut Function,
    insts: &mut Vec<Inst>,
    x: Value,
    ty: Type,
    folded: FoldedCmp,
) -> Option<Value> {
    match folded {
        FoldedCmp::Const(_) => None,
        FoldedCmp::Single(cc, k) => {
            let c = emit_const(func, insts, ty, k);
            Some(emit_icmp(func, insts, cc, (x, c)))
        }
        FoldedCmp::Range(lo_raw, span) => {
            let lo_c = emit_const(func, insts, ty, lo_raw);
            let diff = emit_bin(func, insts, ty, Binary::Isub, (x, lo_c));
            let span_c = emit_const(func, insts, ty, span);
            Some(emit_icmp(
                func,
                insts,
                IntCC::UnsignedLessThanOrEqual,
                (diff, span_c),
            ))
        }
        FoldedCmp::MaskEq(mask, expect, ne) => {
            let mask_c = emit_const(func, insts, ty, mask);
            let masked = emit_bin(func, insts, ty, Binary::Band, (x, mask_c));
            let exp_c = emit_const(func, insts, ty, expect);
            Some(emit_icmp(
                func,
                insts,
                if ne { IntCC::NotEqual } else { IntCC::Equal },
                (masked, exp_c),
            ))
        }
    }
}

type TriangleMatch = (
    usize,
    Value,
    FoldedCmp,
    (Block, Vec<Value>),
    (Block, Vec<Value>),
);

fn try_mid_arm(
    func: &Function,
    (cfg, uses, defs): (&Cfg, &[u32], &[Option<Inst>]),
    (b, x1, eff1, k1): (usize, Value, IntCC, i64),
    (mid_blk, other_blk, other_args): (Block, Block, &[Value]),
) -> Option<TriangleMatch> {
    let mid = mid_blk.index();
    let eligible = mid != b
        && Some(mid) != func.entry.map(Block::index)
        && cfg.pred_count(mid_blk) == 1
        && func.blocks[mid].params.is_empty()
        && mid_self_contained(func, uses, mid);
    if !eligible {
        return None;
    }
    let Op::Brif(c2, t2, ta2, e2, ea2) = func.last(mid_blk)?.clone() else {
        return None;
    };
    let (x2, cc2, k2) = parse_const_cmp(func, defs, c2)?;
    if x1 != x2 {
        return None;
    }
    let (true_tgt, inv2) = if e2 == other_blk && ea2.as_slice() == other_args {
        ((t2, ta2), false)
    } else if t2 == other_blk && ta2.as_slice() == other_args {
        ((e2, ea2), true)
    } else {
        return None;
    };
    let eff2 = if inv2 { cc2.inverse() } else { cc2 };
    fold_and_cmps(func.ty(x1), (eff1, k1), (eff2, k2))
        .map(|f| (mid, x1, f, true_tgt, (other_blk, other_args.to_vec())))
}

fn match_triangle_arm(
    func: &Function,
    cfg: &Cfg,
    uses: &[u32],
    defs: &[Option<Inst>],
    b: usize,
) -> Option<TriangleMatch> {
    let Op::Brif(c1, t1, ta1, e1, ea1) = func.last(Block(b as u32))?.clone() else {
        return None;
    };
    if t1 == e1 {
        return None;
    }
    let (x1, cc1, k1) = parse_const_cmp(func, defs, c1)?;
    if ta1.is_empty()
        && let Some(m) = try_mid_arm(func, (cfg, uses, defs), (b, x1, cc1, k1), (t1, e1, &ea1))
    {
        return Some(m);
    }
    if ea1.is_empty() {
        return try_mid_arm(
            func,
            (cfg, uses, defs),
            (b, x1, cc1.inverse(), k1),
            (e1, t1, &ta1),
        );
    }
    None
}

fn collect_triangle_round(func: &Function) -> Vec<(usize, TriangleMatch)> {
    if func.blocks.len() < 3 {
        return Vec::new();
    }
    let has_brif_pair = func.blocks.iter().enumerate().any(|(b, _)| {
        matches!(func.last(Block(b as u32)), Some(&Op::Brif(_, t1, _, e1, _)) if t1 != e1 && (
            matches!(func.last(t1), Some(&Op::Brif(_, t2, _, e2, _)) if t2 != e2)
            || matches!(func.last(e1), Some(&Op::Brif(_, t2, _, e2, _)) if t2 != e2)
        ))
    });
    if !has_brif_pair {
        return Vec::new();
    }
    let (cfg, uses) = (func.cfg(), func.use_counts());
    let mut defs = vec![None; func.types.len()];
    for inst in func.blocks.iter().flat_map(|blk| &blk.insts) {
        if let Some(res) = func.insts[inst.index()].result {
            defs[res.index()] = Some(*inst);
        }
    }
    let mut busy = vec![false; func.blocks.len()];
    let mut out = Vec::new();
    for b in 0..func.blocks.len() {
        if !cfg.reachable[b] || busy[b] {
            continue;
        }
        let Some(m) = match_triangle_arm(func, &cfg, &uses, &defs, b) else {
            continue;
        };
        let (mid, t_b, f_b) = (m.0, (m.3).0.index(), (m.4).0.index());
        if busy[mid] {
            continue;
        }
        for idx in [b, mid, t_b, f_b] {
            busy[idx] = true;
        }
        out.push((b, m));
    }
    out
}

/// Folds load-free single-operand interval and power-of-two bit-pair short-circuit triangles into one compare.
pub(crate) fn fold_short_circuit_compares(func: &mut Function) -> bool {
    let mut changed = false;
    for _ in 0..8 {
        let round = collect_triangle_round(func);
        if round.is_empty() {
            break;
        }
        for (b, (mid, x, folded, t_tgt, f_tgt)) in round {
            let (ty, mut extra) = (func.ty(x), Vec::new());
            let new_term = match emit_folded_cmp(func, &mut extra, x, ty, folded) {
                Some(cond) => Op::Brif(cond, t_tgt.0, t_tgt.1, f_tgt.0, f_tgt.1),
                None => Op::Jump(
                    if matches!(folded, FoldedCmp::Const(true)) {
                        t_tgt.0
                    } else {
                        f_tgt.0
                    },
                    if matches!(folded, FoldedCmp::Const(true)) {
                        t_tgt.1
                    } else {
                        f_tgt.1
                    },
                ),
            };
            let Some(term_inst) = func.blocks[b].insts.pop() else {
                continue;
            };
            func.insts[term_inst.index()].op = new_term;
            func.blocks[b].insts.extend(extra);
            func.blocks[b].insts.push(term_inst);
            if let Some(m_term) = func.blocks[mid].insts.pop() {
                func.insts[m_term.index()].op = Op::Trap;
                func.blocks[mid].insts.clear();
                func.blocks[mid].insts.push(m_term);
            }
            prune_dead_local(func, b);
        }
        func.bump_cfg();
        changed = true;
    }
    changed
}
