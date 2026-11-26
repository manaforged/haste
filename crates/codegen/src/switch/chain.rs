use crate::ir::{Function, Inst, IntCC, Op, Type, Value};

use super::bitmask::{build_runs, can_bitmask_runs};
use super::emit::{emit_node, emit_sub_range, forward_cached, try_signed_br_table};
use super::lut;
use super::member;
use super::{
    Arm, BlockTest, Edge, Info, Range, TestKind, body_clean, can_follow_into,
    clean_dead_head_insts, collect_hoisted_consts, const_bool, edge_of, intersect_ranges,
    match_test, push_before_last, range_fits, range_of_test, same_edge,
};

struct ArmMatch {
    val: Value,
    arm: Arm,
    next: Edge,
    extra_blocks: Vec<usize>,
    used: [Option<Inst>; 3],
    terminates: bool,
}

pub(super) struct Chain {
    val: Value,
    arms: Vec<Arm>,
    default_edge: Edge,
    chain_blocks: Vec<usize>,
    two_block_arms: u32,
    head_used: [Option<Inst>; 3],
}

fn match_two_block_range(
    func: &Function,
    info: &Info,
    ctx: (usize, &[bool], &[usize]),
    (v, ty, kind1, want_true1): (Value, Type, TestKind, bool),
    (hit_edge, fail_edge): (&Edge, &Edge),
) -> Option<(Arm, Vec<usize>)> {
    let (mid, mut extra) = can_follow_into(func, info, ctx, hit_edge)?;
    let r1 = range_of_test(ty, kind1, want_true1)?;
    let last = *func.blocks[mid].insts.last()?;
    let arm = match &func.insts[last.index()].op {
        Op::Brif(cond2, t2, t2_args, f2, f2_args) => {
            let test2 = match_test(func, info, *cond2, Some(v), Some(mid))?;
            if !body_clean(func, info, mid, &|i| test2.used.contains(&Some(i)), None) {
                return None;
            }
            let (t2_edge, f2_edge) = (edge_of(func, *t2, t2_args), edge_of(func, *f2, f2_args));
            let (want_true2, hit) = if same_edge(func, info, &f2_edge, fail_edge) {
                (true, t2_edge)
            } else if same_edge(func, info, &t2_edge, fail_edge) {
                (false, f2_edge)
            } else {
                return None;
            };
            Arm {
                range: intersect_ranges(ty, r1, range_of_test(ty, test2.kind, want_true2)?)?,
                hit,
            }
        }
        Op::Jump(target, args)
            if args.len() == 1 && *target == fail_edge.block && fail_edge.args.len() == 1 =>
        {
            let want_true2 = !const_bool(func, info, fail_edge.args[0])?;
            let cond2 = args[0];
            let test2 = (func.ty(cond2) == Type::I8)
                .then(|| match_test(func, info, cond2, Some(v), Some(mid)))
                .flatten()?;
            if !body_clean(
                func,
                info,
                mid,
                &|i| test2.used.contains(&Some(i)),
                Some(cond2),
            ) {
                return None;
            }
            let range = intersect_ranges(ty, r1, range_of_test(ty, test2.kind, want_true2)?)?;
            Arm {
                range,
                hit: Edge {
                    raw: *target,
                    block: *target,
                    args: Vec::new(),
                    synth_bool: Some(want_true2),
                },
            }
        }
        _ => return None,
    };
    extra.push(mid);
    Some((arm, extra))
}

fn pick_arm_direction(
    func: &Function,
    info: &Info,
    test: &BlockTest,
    (t_edge, f_edge, t_cont, f_cont): (Edge, Edge, bool, bool),
    (preferred_hit, terminates): (Option<&Edge>, bool),
) -> Option<ArmMatch> {
    let ty = func.ty(test.val);
    let t_range = range_of_test(ty, test.kind, true).filter(|&r| range_fits(r));
    let f_range = range_of_test(ty, test.kind, false).filter(|&r| range_fits(r));
    let pick_true = match (t_range, f_range) {
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (Some(_), Some(_)) if f_cont != t_cont => f_cont,
        (Some(tr), Some(fr)) => {
            let p_t = preferred_hit.is_some_and(|p| same_edge(func, info, p, &t_edge));
            let p_f = preferred_hit.is_some_and(|p| same_edge(func, info, p, &f_edge));
            let score = |r: Range| {
                if r.lo <= r.hi {
                    r.hi - r.lo
                } else {
                    r.s_hi.abs_diff(r.s_lo)
                }
            };
            if p_t != p_f {
                p_t
            } else {
                score(tr) <= score(fr)
            }
        }
        (None, None) => return None,
    };
    let range = if pick_true { t_range? } else { f_range? };
    let (hit, next) = if pick_true {
        (t_edge, f_edge)
    } else {
        (f_edge, t_edge)
    };
    Some(ArmMatch {
        val: test.val,
        arm: Arm { range, hit },
        next,
        extra_blocks: Vec::new(),
        used: test.used,
        terminates,
    })
}

fn match_arm(
    func: &Function,
    info: &Info,
    ctx @ (b0, _, _): (usize, &[bool], &[usize]),
    (b, expected_val, preferred_hit): (usize, Option<Value>, Option<&Edge>),
) -> Option<ArmMatch> {
    let is_head = b == b0;
    let last = *func.blocks[b].insts.last()?;
    match &func.insts[last.index()].op {
        Op::Brif(cond, t_blk, t_args, f_blk, f_args) => {
            let test = match_test(func, info, *cond, expected_val, (!is_head).then_some(b))?;
            if !is_head && !body_clean(func, info, b, &|i| test.used.contains(&Some(i)), None) {
                return None;
            }
            let ty = func.ty(test.val);
            let (t_edge, f_edge) = (edge_of(func, *t_blk, t_args), edge_of(func, *f_blk, f_args));
            if let TestKind::Cmp(cc @ (IntCC::Equal | IntCC::NotEqual), _) = test.kind {
                let want_true = cc == IntCC::Equal;
                let range = range_of_test(ty, test.kind, want_true)?;
                let (hit, next) = if want_true {
                    (t_edge, f_edge)
                } else {
                    (f_edge, t_edge)
                };
                return Some(ArmMatch {
                    val: test.val,
                    arm: Arm { range, hit },
                    next,
                    extra_blocks: Vec::new(),
                    used: test.used,
                    terminates: false,
                });
            }
            for (want, hit_e, next_e) in [(true, &t_edge, &f_edge), (false, &f_edge, &t_edge)] {
                if let Some((arm, extra_blocks)) = match_two_block_range(
                    func,
                    info,
                    ctx,
                    (test.val, ty, test.kind, want),
                    (hit_e, next_e),
                ) {
                    return Some(ArmMatch {
                        val: test.val,
                        arm,
                        next: next_e.clone(),
                        extra_blocks,
                        used: test.used,
                        terminates: false,
                    });
                }
            }
            let t_cont = can_follow_into(func, info, ctx, &t_edge)
                .is_some_and(|(tb, _)| peek_chain(func, info, tb, test.val));
            let f_cont = can_follow_into(func, info, ctx, &f_edge)
                .is_some_and(|(fb, _)| peek_chain(func, info, fb, test.val));
            if t_cont && f_cont {
                return None;
            }
            pick_arm_direction(
                func,
                info,
                &test,
                (t_edge, f_edge, t_cont, f_cont),
                (preferred_hit, false),
            )
        }
        Op::Jump(target, args)
            if !is_head
                && args.len() == 1
                && func.blocks[target.index()].params.len() == 1
                && func.ty(args[0]) == Type::I8 =>
        {
            let cond = args[0];
            let test = match_test(func, info, cond, Some(expected_val?), Some(b))?;
            if !body_clean(func, info, b, &|i| test.used.contains(&Some(i)), Some(cond)) {
                return None;
            }
            let t_edge = Edge {
                raw: *target,
                block: *target,
                args: Vec::new(),
                synth_bool: Some(true),
            };
            let f_edge = Edge {
                raw: *target,
                block: *target,
                args: Vec::new(),
                synth_bool: Some(false),
            };
            pick_arm_direction(
                func,
                info,
                &test,
                (t_edge, f_edge, false, false),
                (preferred_hit, true),
            )
        }
        _ => None,
    }
}

fn peek_chain(func: &Function, info: &Info, b: usize, v: Value) -> bool {
    let Some(&last) = func.blocks[b].insts.last() else {
        return false;
    };
    match &func.insts[last.index()].op {
        Op::Brif(cond, ..) => match_test(func, info, *cond, Some(v), Some(b))
            .is_some_and(|t| body_clean(func, info, b, &|i| t.used.contains(&Some(i)), None)),
        Op::Jump(_, args) if args.len() == 1 => {
            func.ty(args[0]) == Type::I8
                && match_test(func, info, args[0], Some(v), Some(b)).is_some_and(|t| {
                    body_clean(func, info, b, &|i| t.used.contains(&Some(i)), Some(args[0]))
                })
        }
        _ => false,
    }
}

fn resolve_edge(
    func: &mut Function,
    b0: usize,
    bool_consts: &mut [Option<Value>; 2],
    edge: &mut Edge,
) {
    if let Some(b) = edge.synth_bool.take() {
        let val = *bool_consts[usize::from(b)].get_or_insert_with(|| {
            let v = func.value(Type::I8);
            let inst = func.add(Op::Const(i64::from(b)), Some(v));
            push_before_last(func, b0, inst);
            v
        });
        edge.args = vec![val];
    }
}

pub(super) fn collect_chain(
    func: &Function,
    info: &Info,
    b0: usize,
    cleared: &[bool],
) -> Option<Chain> {
    let head = match_arm(func, info, (b0, cleared, &[]), (b0, None, None))?;
    let v = head.val;
    let mut chain_blocks = head.extra_blocks;
    let mut two_block_arms = u32::from(!chain_blocks.is_empty());
    let (mut arms, mut default_edge) = (vec![head.arm], head.next);
    while !head.terminates
        && let Some((cur, hops)) =
            can_follow_into(func, info, (b0, cleared, &chain_blocks), &default_edge)
    {
        let Some(m) = match_arm(
            func,
            info,
            (b0, cleared, &chain_blocks),
            (cur, Some(v), Some(&arms[0].hit)),
        ) else {
            break;
        };
        chain_blocks.extend(hops);
        chain_blocks.push(cur);
        if !m.extra_blocks.is_empty() {
            chain_blocks.extend(m.extra_blocks);
            two_block_arms += 1;
        }
        arms.push(m.arm);
        default_edge = m.next;
        if m.terminates {
            break;
        }
    }
    (!chain_blocks.is_empty()).then_some(Chain {
        val: v,
        arms,
        default_edge,
        chain_blocks,
        two_block_arms,
        head_used: head.used,
    })
}

fn clear_blocks(func: &mut Function, blocks: &[usize], hoisted: &[Inst], cleared: &mut [bool]) {
    for &b in blocks {
        for inst in func.blocks[b].insts.drain(..) {
            if !hoisted.contains(&inst) {
                func.insts[inst.index()].op = Op::Trap;
            }
        }
        cleared[b] = true;
    }
}

pub(super) fn rewrite_chain(
    func: &mut Function,
    info: &Info,
    (b0, last0): (usize, Inst),
    mut ch: Chain,
    (cleared, flag_chain): (&mut [bool], bool),
) -> bool {
    let ty = func.ty(ch.val);
    let single_mod =
        ch.arms.len() == 1 && ch.two_block_arms == 1 && ch.arms[0].range.lo > ch.arms[0].range.hi;
    let mut runs = if single_mod {
        Vec::new()
    } else {
        build_runs(func, info, ty, &ch.arms)
    };
    let head_dead: Vec<Inst> = ch.head_used.into_iter().flatten().collect();
    if !single_mod
        && lut::try_emit_runs_lut(
            func,
            info,
            (b0, last0),
            ch.val,
            &runs,
            &ch.default_edge,
            &head_dead,
        )
    {
        let hoisted = collect_hoisted_consts(func, info, &ch.chain_blocks);
        hoisted
            .iter()
            .for_each(|&inst| push_before_last(func, b0, inst));
        clear_blocks(func, &ch.chain_blocks, &hoisted, cleared);
        cleared[b0] = true;
        return true;
    }
    let covered: u64 = runs
        .iter()
        .map(|r| r.hi.saturating_sub(r.lo).saturating_add(1))
        .sum();
    let members = (flag_chain && !single_mod)
        .then(|| member::member_values(func, info, ty, (&runs, covered)))
        .flatten();
    if !single_mod && members.is_none() {
        if covered < 4
            && ch.two_block_arms == 0
            && runs.len() >= ch.arms.len()
            && !can_bitmask_runs(func, info, ty, &runs)
        {
            return false;
        }
    }
    let hoisted = collect_hoisted_consts(func, info, &ch.chain_blocks);
    hoisted
        .iter()
        .for_each(|&inst| push_before_last(func, b0, inst));
    let mut bool_consts = [None; 2];
    ch.arms
        .iter_mut()
        .for_each(|a| resolve_edge(func, b0, &mut bool_consts, &mut a.hit));
    runs.iter_mut()
        .for_each(|r| resolve_edge(func, b0, &mut bool_consts, &mut r.hit));
    resolve_edge(func, b0, &mut bool_consts, &mut ch.default_edge);
    clean_dead_head_insts(func, info, b0, &head_dead);
    let mut cache = Vec::new();
    let default_blk = forward_cached(
        func,
        &mut cache,
        ch.default_edge.block,
        ch.default_edge.args,
    );
    if single_mod {
        let r = ch.arms[0].range;
        emit_sub_range(
            func,
            Some((b0, last0)),
            (ch.val, ty),
            (r.s_lo, r.s_hi.abs_diff(r.s_lo)),
            (&ch.arms[0].hit, default_blk),
        );
    } else if let Some((&head, rest)) = members.as_deref().and_then(<[u64]>::split_first) {
        member::emit_member(
            func,
            (b0, last0),
            (ch.val, ty),
            (head, rest),
            (&runs[0].hit, default_blk),
        );
    } else if !try_signed_br_table(
        func,
        (b0, last0),
        (ch.val, ty),
        &ch.arms,
        (default_blk, &mut cache),
    ) {
        emit_node(
            func,
            info,
            Some((b0, last0)),
            (ch.val, ty),
            &runs,
            (default_blk, &mut cache),
        );
    }
    clear_blocks(func, &ch.chain_blocks, &hoisted, cleared);
    cleared[b0] = true;
    true
}
