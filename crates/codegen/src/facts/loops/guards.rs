use crate::cfg::{Cfg, Loop};
use crate::ir::{Binary, Block, Function, IntCC, Op, Value};

use super::super::Ranges;
use super::super::rel::{Path, RelKind, rel_of};
use super::{Env, ExtraPath};

pub(super) fn latch_guards(
    func: &Function,
    cfg: &Cfg,
    ranges: &Ranges,
    env: &Env,
    each: &Loop,
    latch: Block,
    dom_exit: Option<&Path>,
) -> (Path, ExtraPath) {
    let mut chain = Vec::new();
    let mut cur = latch;
    while cur != each.head && each.body.get(cur.index()).copied().unwrap_or(false) {
        chain.push(cur);
        let Some(parent) = cfg
            .idom
            .get(cur.index())
            .copied()
            .flatten()
            .filter(|&p| p != cur)
        else {
            break;
        };
        cur = parent;
    }
    let mut path = dom_exit.cloned().unwrap_or_default();
    let mut extra = ExtraPath::default();
    let mut prev: Option<Block> = None;
    for blk in std::iter::once(each.head).chain(chain.into_iter().rev()) {
        if let Some(p) = prev {
            if cfg.single_pred(blk) == Some(p) && !func.blocks[blk.index()].params.is_empty() {
                let edge_args = match func.last(p) {
                    Some(Op::Jump(t, a)) if *t == blk => Some(a.as_slice()),
                    Some(Op::Brif(_, t, ta, f, fa)) if *t != *f && ((*t == blk) ^ (*f == blk)) => {
                        Some(if *t == blk {
                            ta.as_slice()
                        } else {
                            fa.as_slice()
                        })
                    }
                    _ => None,
                };
                if let Some(args) = edge_args {
                    for (&param, &arg) in func.blocks[blk.index()].params.iter().zip(args) {
                        if func.ty(param).is_small_int() && param != arg {
                            path.push_rel_kind(RelKind::Eq, param, arg, func.ty(param));
                        }
                    }
                }
            } else if blk != each.head && !func.blocks[blk.index()].params.is_empty() {
                assume_inner_loop_rels(func, cfg, ranges, &mut path, each, (p, blk));
            }
        }
        prev = Some(blk);
        for inv in &env.rel_invs {
            if inv.head == blk {
                path.push_rel_kind(inv.kind, inv.param, inv.bound, inv.ty);
            }
        }
        for &inst in &func.blocks[blk.index()].insts {
            if let Op::Trapz(cond) | Op::Assume(cond) = func.insts[inst.index()].op {
                path.assume(func, ranges, cond, true);
                extra.assume(func, (cfg, ranges, env, &path, blk), cond, true, 0);
            }
        }
        if blk != latch
            && let Some(&Op::Brif(cond, t, _, e, _)) = func.last(blk)
            && t != e
        {
            let t_taken = cfg.single_pred(t) == Some(blk) && cfg.dominates(t, latch);
            let e_taken = cfg.single_pred(e) == Some(blk) && cfg.dominates(e, latch);
            if t_taken ^ e_taken {
                path.assume(func, ranges, cond, t_taken);
                extra.assume(func, (cfg, ranges, env, &path, blk), cond, t_taken, 0);
            }
        }
    }
    if let Some(&Op::Brif(cond, t, _, e, _)) = func.last(latch)
        && t != e
        && (t == each.head) ^ (e == each.head)
    {
        path.assume(func, ranges, cond, t == each.head);
        extra.assume(
            func,
            (cfg, ranges, env, &path, latch),
            cond,
            t == each.head,
            0,
        );
    }
    (path, extra)
}

fn assume_inner_loop_rels(
    func: &Function,
    cfg: &Cfg,
    ranges: &Ranges,
    path: &mut Path,
    each: &Loop,
    (prev, blk): (Block, Block),
) {
    let Some(inner) = cfg
        .loops()
        .list
        .iter()
        .find(|other| other.head == blk && each.body[blk.index()])
    else {
        return;
    };
    let outside: Vec<Block> = cfg
        .preds_of(blk)
        .iter()
        .copied()
        .filter(|&p| cfg.reachable[p.index()] && !inner.body[p.index()])
        .collect();
    if outside != [prev] {
        return;
    }
    let (init_args, edge_cond) = match func.last(prev) {
        Some(Op::Jump(t, a)) if *t == blk => (a.as_slice(), None),
        Some(Op::Brif(c, t, ta, e, ea)) if *t != *e && ((*t == blk) ^ (*e == blk)) => (
            if *t == blk {
                ta.as_slice()
            } else {
                ea.as_slice()
            },
            Some((*c, *t == blk)),
        ),
        _ => return,
    };
    let mut entry = path.clone();
    if let Some((cond, taken)) = edge_cond {
        entry.assume(func, ranges, cond, taken);
    }
    let mut latches = Vec::new();
    for &p in cfg.preds_of(blk) {
        if cfg.reachable[p.index()] && inner.body[p.index()] {
            let Some(item) = (match func.last(p) {
                Some(Op::Brif(c, t, ta, e, ea)) if *t != *e && ((*t == blk) ^ (*e == blk)) => {
                    Some((
                        *c,
                        *t == blk,
                        if *t == blk {
                            ta.as_slice()
                        } else {
                            ea.as_slice()
                        },
                    ))
                }
                _ => None,
            }) else {
                return;
            };
            latches.push(item);
        }
    }
    if latches.is_empty() {
        return;
    }
    let params = &func.blocks[blk.index()].params;
    let rels: Vec<_> = entry.rels().to_vec();
    for (k, &q) in params.iter().enumerate() {
        let ty = func.ty(q);
        let Some(&init_q) = init_args.get(k).filter(|_| ty.is_small_int()) else {
            continue;
        };
        for r in &rels {
            let bound = r.rhs;
            if r.ty != ty
                || !matches!(r.kind, RelKind::Ult | RelKind::Ule)
                || bound == q
                || params.contains(&bound)
            {
                continue;
            }
            if entry.decide_rel(func, ranges, IntCC::UnsignedLessThan, ty, init_q, bound)
                == Some(true)
                && latches.iter().all(|&(c, taken, args)| {
                    args.get(k).is_some_and(|&a| {
                        let mut lp = entry.clone();
                        lp.assume(func, ranges, c, taken);
                        lp.decide_rel(func, ranges, IntCC::UnsignedLessThan, ty, a, bound)
                            == Some(true)
                            || cond_proves_rel(
                                func,
                                ranges,
                                c,
                                taken,
                                IntCC::UnsignedLessThan,
                                a,
                                bound,
                                0,
                            )
                    })
                })
            {
                path.push_rel_kind(RelKind::Ult, q, bound, ty);
            }
        }
    }
}

fn cond_proves_rel(
    func: &Function,
    ranges: &Ranges,
    cond: Value,
    taken: bool,
    cc: IntCC,
    lhs: Value,
    rhs: Value,
    depth: u8,
) -> bool {
    if depth > 4 {
        return false;
    }
    match ranges.op(func, cond) {
        Some(&Op::Binary(Binary::Band, a, b)) if taken => {
            cond_proves_rel(func, ranges, a, true, cc, lhs, rhs, depth + 1)
                || cond_proves_rel(func, ranges, b, true, cc, lhs, rhs, depth + 1)
        }
        Some(&Op::Binary(Binary::Bor, a, b)) if !taken => {
            cond_proves_rel(func, ranges, a, false, cc, lhs, rhs, depth + 1)
                || cond_proves_rel(func, ranges, b, false, cc, lhs, rhs, depth + 1)
        }
        Some(&Op::Icmp(raw, x, y)) if func.ty(x) == func.ty(lhs) => {
            let eff = if taken { raw } else { raw.inverse() };
            if eff.is_signed() {
                return false;
            }
            let (rk, rl, rr) = rel_of(eff, x, y);
            match cc {
                IntCC::UnsignedLessThan => rk == RelKind::Ult && rl == lhs && rr == rhs,
                IntCC::UnsignedLessThanOrEqual => {
                    (matches!(rk, RelKind::Ult | RelKind::Ule) && rl == lhs && rr == rhs)
                        || (rk == RelKind::Eq
                            && ((rl == lhs && rr == rhs) || (rl == rhs && rr == lhs)))
                }
                _ => false,
            }
        }
        _ => false,
    }
}

pub(super) fn jump_latch_proves(
    func: &Function,
    cfg: &Cfg,
    ranges: &Ranges,
    each: &Loop,
    (latch, arg, cc, bound, depth): (Block, Value, IntCC, Value, u8),
) -> bool {
    if depth > 3
        || !matches!(func.last(latch), Some(Op::Jump(t, _)) if depth > 0 || *t == each.head)
    {
        return false;
    }
    let Some(p_idx) = func.blocks[latch.index()]
        .params
        .iter()
        .position(|&v| v == arg)
    else {
        return false;
    };
    let preds = cfg.preds_of(latch);
    !preds.is_empty()
        && preds.iter().all(|&p| {
            each.body.get(p.index()).copied().unwrap_or(false)
                && match func.last(p) {
                    Some(&Op::Brif(cond, t, ref ta, f, ref fa))
                        if t != f && ((t == latch) ^ (f == latch)) =>
                    {
                        let (taken, a) = if t == latch {
                            (true, ta.get(p_idx).copied())
                        } else {
                            (false, fa.get(p_idx).copied())
                        };
                        a.is_some_and(|v| {
                            cond_proves_rel(func, ranges, cond, taken, cc, v, bound, 0)
                        })
                    }
                    Some(Op::Jump(t, ja)) if *t == latch => ja.get(p_idx).is_some_and(|&v| {
                        jump_latch_proves(func, cfg, ranges, each, (p, v, cc, bound, depth + 1))
                    }),
                    _ => false,
                }
        })
}
