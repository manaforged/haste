use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Function, Op, Value};

use super::{emit_select, hoist_block_body, remap_val, retire_terminator, scalar_select};

const MAX_MID_BODY: usize = 3;

type BranchParts = (Value, Block, Vec<Value>, Block, Vec<Value>);

fn extract_const_leaf(
    func: &Function,
    cfg: &Cfg,
    is_const: &[bool],
    leaf: Block,
    in_args: &[Value],
    join: Block,
) -> Option<(Option<Block>, Value)> {
    if leaf == join {
        return (in_args.len() == 1 && is_const[in_args[0].index()]).then_some((None, in_args[0]));
    }
    if cfg.pred_count(leaf) != 1 || func.entry == Some(leaf) {
        return None;
    }
    let data = &func.blocks[leaf.index()];
    if data.params.len() != in_args.len() {
        return None;
    }
    let (&last, body) = data.insts.split_last()?;
    if body.len() > 2
        || !body
            .iter()
            .all(|&i| matches!(func.insts[i.index()].op, Op::Const(_)))
    {
        return None;
    }
    let Op::Jump(target, args) = &func.insts[last.index()].op else {
        return None;
    };
    if *target != join || args.len() != 1 {
        return None;
    }
    let val = remap_val(&data.params, in_args, args[0]);
    is_const[val.index()].then_some((Some(leaf), val))
}

fn branch_block(
    func: &Function,
    cfg: &Cfg,
    (block, in_args): (Block, &[Value]),
    max_body: usize,
    keep: impl Fn(&Op) -> bool,
) -> Option<BranchParts> {
    let data = &func.blocks[block.index()];
    if cfg.pred_count(block) != 1 || func.entry == Some(block) || data.params.len() != in_args.len()
    {
        return None;
    }
    let (&last, body) = data.insts.split_last()?;
    if body.len() > max_body || !body.iter().all(|&inst| keep(&func.insts[inst.index()].op)) {
        return None;
    }
    let Op::Brif(cond, then_b, then_args, else_b, else_args) = func.insts[last.index()].op.clone()
    else {
        return None;
    };
    Some((cond, then_b, then_args, else_b, else_args))
}

fn compare_only(op: &Op) -> bool {
    matches!(op, Op::Const(_) | Op::Icmp(..) | Op::Fcmp(..))
}

pub(super) fn try_nested_classifier(
    func: &mut Function,
    cfg: &Cfg,
    is_const: &[bool],
    head: Block,
    cleared: &mut [bool],
) -> bool {
    let Some(&brif_inst) = func.blocks[head.index()].insts.last() else {
        return false;
    };
    let Op::Brif(c1, then_b, then_args, else_b, else_args) =
        func.insts[brif_inst.index()].op.clone()
    else {
        return false;
    };
    if !func.ty(c1).is_small_int() || then_b == else_b || then_b == head || else_b == head {
        return false;
    }
    for (mid_is_then, mid, mid_in, other, other_in) in [
        (true, then_b, &then_args, else_b, &else_args),
        (false, else_b, &else_args, then_b, &then_args),
    ] {
        let Some((c2, mt, mt_args, me, me_args)) =
            branch_block(func, cfg, (mid, mid_in), MAX_MID_BODY, compare_only)
        else {
            continue;
        };
        if !func.ty(c2).is_small_int()
            || mt == me
            || mt == head
            || mt == mid
            || me == head
            || me == mid
        {
            continue;
        }
        let join =
            if other != mt && other != me && other_in.len() == 1 && cfg.pred_count(other) >= 2 {
                other
            } else if let Some(Op::Jump(j, _)) = func.last(other) {
                *j
            } else {
                continue;
            };
        if join == head
            || join == mid
            || func.entry == Some(join)
            || cfg.pred_count(join) < 2
            || func.blocks[join.index()].params.len() != 1
        {
            continue;
        }
        let param_ty = func.ty(func.blocks[join.index()].params[0]);
        if !scalar_select(param_ty) || !param_ty.is_small_int() {
            continue;
        }
        let Some((other_leaf, other_val)) =
            extract_const_leaf(func, cfg, is_const, other, other_in, join)
        else {
            continue;
        };
        let Some((mt_leaf, mt_val)) = extract_const_leaf(func, cfg, is_const, mt, &mt_args, join)
        else {
            continue;
        };
        let Some((me_leaf, me_val)) = extract_const_leaf(func, cfg, is_const, me, &me_args, join)
        else {
            continue;
        };
        retire_terminator(func, head);
        for (leaf_opt, leaf_in) in [
            (other_leaf, other_in.as_slice()),
            (mt_leaf, mt_args.as_slice()),
            (me_leaf, me_args.as_slice()),
        ] {
            if let Some(leaf) = leaf_opt {
                hoist_block_body(func, head, leaf, leaf_in, cleared);
            }
        }
        let (mid_params, _) = hoist_block_body(func, head, mid, mid_in, cleared);
        let c2_val = remap_val(&mid_params, mid_in, c2);
        let inner = emit_select(func, head, param_ty, c2_val, mt_val, me_val);
        let (t_val, e_val) = if mid_is_then {
            (inner, other_val)
        } else {
            (other_val, inner)
        };
        let outer = emit_select(func, head, param_ty, c1, t_val, e_val);
        let jump = func.add(Op::Jump(join, vec![outer]), None);
        func.blocks[head.index()].insts.push(jump);
        return true;
    }
    false
}

fn defines_compare(func: &Function, block: Block, value: Value) -> bool {
    func.blocks[block.index()].insts.iter().any(|&inst| {
        func.insts[inst.index()].result == Some(value)
            && matches!(func.insts[inst.index()].op, Op::Icmp(..) | Op::Fcmp(..))
    })
}

pub(super) fn try_shared_target(
    func: &mut Function,
    cfg: &Cfg,
    region: &[bool],
    head: Block,
    cleared: &mut [bool],
) -> bool {
    let Some(Op::Brif(c1, t1, ta1, e1, ea1)) = func.last(head).cloned() else {
        return false;
    };
    if t1 == e1 || !defines_compare(func, head, c1) {
        return false;
    }
    for (mid, mid_in, both) in [(t1, &ta1, true), (e1, &ea1, false)] {
        if !region[mid.index()] || !mid_in.is_empty() || mid == head {
            continue;
        }
        let Some((c2, t2, ta2, e2, ea2)) =
            branch_block(func, cfg, (mid, mid_in), MAX_MID_BODY, compare_only)
        else {
            continue;
        };
        let shared = if both {
            e2 == e1 && ea2 == ea1
        } else {
            t2 == t1 && ta2 == ta1
        };
        if !shared
            || t2 == e2
            || t2 == mid
            || e2 == mid
            || func.ty(c1) != func.ty(c2)
            || !defines_compare(func, mid, c2)
        {
            continue;
        }
        retire_terminator(func, head);
        hoist_block_body(func, head, mid, mid_in, cleared);
        let cond = func.value(func.ty(c1));
        let joined = func.add(
            Op::Binary(if both { Binary::Band } else { Binary::Bor }, c1, c2),
            Some(cond),
        );
        let term = func.add(Op::Brif(cond, t2, ta2, e2, ea2), None);
        func.blocks[head.index()].insts.extend([joined, term]);
        return true;
    }
    false
}
