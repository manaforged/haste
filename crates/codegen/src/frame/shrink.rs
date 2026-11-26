use super::{EdgeScratch, Frame, Loc, Step, edge_steps};
use crate::alloc::{Location, Plan};
use crate::ir::{ArgumentPurpose, Block, Function, Inst, Op, Type, Value};

pub(crate) struct ShrinkWrap {
    pub fast: Vec<bool>,
    pub split_ret: Vec<bool>,
}

fn pure_value<R: Copy + PartialEq>(
    func: &Function,
    defs: &[Option<Inst>],
    value: Value,
    plan: &Plan<R>,
    frame: &Frame<R>,
    clobbers: &[R],
) -> bool {
    if matches!(func.ty(value), Type::I128 | Type::Vec { .. }) {
        return false;
    }
    if plan.elided[value.index()]
        || (plan.fused[value.index()]
            && defs.get(value.index()).copied().flatten().is_some_and(|i| {
                match func.insts[i.index()].op {
                    Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Icmp(..) | Op::Fcmp(..) => true,
                    Op::Fits(..) => plan.pairs[value.index()].is_none(),
                    _ => false,
                }
            }))
    {
        return true;
    }
    match plan.location(value) {
        Location::Imm(_) | Location::Symbol(_) | Location::SlotAddr(_) => true,
        Location::Reg(r) | Location::Offset(r, _) => {
            !frame.saves.contains(&r) && !clobbers.contains(&r)
        }
        Location::Float(f) => !frame.float_saves.contains(&f),
        Location::Slot | Location::Frame(..) => false,
    }
}

fn pure_op(inst: Inst, op: &Op, fast_call: &impl Fn(Inst, &Op) -> bool) -> bool {
    match op {
        Op::Call { .. } => fast_call(inst, op),
        Op::StackAddr(..)
        | Op::StackLoad(..)
        | Op::StackStore(..)
        | Op::FramePointer
        | Op::TryCall { .. }
        | Op::Copy(..)
        | Op::Move(..)
        | Op::Trap
        | Op::Trapz(..) => false,
        _ => true,
    }
}

fn pure_block<R: Copy + PartialEq>(
    func: &Function,
    defs: &[Option<Inst>],
    block: Block,
    plan: &Plan<R>,
    frame: &Frame<R>,
    clobbers: &[R],
    fast_call: &impl Fn(Inst, &Op) -> bool,
) -> bool {
    let data = &func.blocks[block.index()];
    let Some((&term, body)) = data.insts.split_last() else {
        return false;
    };
    let ok_val = |v: Value| pure_value(func, defs, v, plan, frame, clobbers);
    if !data.params.iter().all(|&p| ok_val(p)) {
        return false;
    }
    for &inst in body {
        let id = &func.insts[inst.index()];
        let mut ok = pure_op(inst, &id.op, fast_call) && crate::live::results(id).all(ok_val);
        id.op.for_operands(|v| ok &= ok_val(v));
        if !ok {
            return false;
        }
    }
    match &func.insts[term.index()].op {
        Op::Return(vals) => vals.iter().all(|&v| ok_val(v)),
        Op::Jump(_, args) => args.iter().all(|&v| ok_val(v)),
        Op::Brif(cond, _, ta, _, ea) => {
            ok_val(*cond)
                && ta
                    .iter()
                    .chain(ea)
                    .all(|&v| !plan.fused[v.index()] && ok_val(v))
        }
        _ => false,
    }
}

fn edge_uses_stack<R: Copy + PartialEq>(
    func: &Function,
    plan: &Plan<R>,
    frame: &Frame<R>,
    to: Block,
    args: &[Value],
) -> bool {
    edge_steps(
        func,
        &plan.locations,
        &frame.values,
        args,
        &func.blocks[to.index()].params,
    )
    .iter()
    .any(|s| {
        matches!(
            s,
            Step::Save(_, EdgeScratch::Stack(_)) | Step::Restore(EdgeScratch::Stack(_), _)
        )
    })
}

fn visit_edges(func: &Function, block: Block, mut f: impl FnMut(Block, &[Value])) {
    match func.last(block) {
        Some(Op::Jump(t, args)) => f(*t, args),
        Some(Op::Brif(_, t, ta, e, ea)) => {
            f(*t, ta);
            f(*e, ea);
        }
        _ => {}
    }
}

pub(crate) fn shrink_wrap<R: Copy + PartialEq>(
    func: &Function,
    framed: bool,
    plan: &Plan<R>,
    frame: &Frame<R>,
    params: &[Loc<R>],
    clobbers: &[R],
    fast_call: impl Fn(Inst, &Op) -> bool,
) -> Option<ShrinkWrap> {
    if !framed
        || func
            .signature
            .params
            .iter()
            .any(|p| p.purpose == ArgumentPurpose::StructReturn)
    {
        return None;
    }
    let entry = func.entry?;
    let defs = crate::fold::defs(func);
    if params
        .iter()
        .any(|loc| matches!(loc, Loc::Stack(_) | Loc::Struct(..) | Loc::Pair(..)))
        || func.blocks[entry.index()]
            .params
            .iter()
            .any(|&p| !pure_value(func, &defs, p, plan, frame, clobbers))
    {
        return None;
    }
    let cfg = func.cfg();
    if cfg.pred_count(entry) != 0 {
        return None;
    }
    let n = func.blocks.len();
    let mut fast: Vec<bool> = (0..n)
        .map(|i| {
            cfg.reachable[i]
                && pure_block(
                    func,
                    &defs,
                    Block(i as u32),
                    plan,
                    frame,
                    clobbers,
                    &fast_call,
                )
        })
        .collect();
    if !fast[entry.index()] {
        return None;
    }
    loop {
        let mut seen = vec![false; n];
        let mut stack = vec![entry];
        seen[entry.index()] = true;
        while let Some(b) = stack.pop() {
            for s in func.successors(b) {
                if fast[s.index()] && !seen[s.index()] {
                    seen[s.index()] = true;
                    stack.push(s);
                }
            }
        }
        let mut changed = false;
        for i in 0..n {
            if !fast[i] {
                continue;
            }
            let b = Block(i as u32);
            let bad_pred = b != entry
                && cfg
                    .preds_of(b)
                    .iter()
                    .any(|&p| cfg.reachable[p.index()] && !fast[p.index()]);
            let jt = func.jump_target(b);
            if !seen[i] || bad_pred || (jt != b && !fast[jt.index()]) {
                fast[i] = false;
                changed = true;
                continue;
            }
            visit_edges(func, b, |to, args| {
                if fast[to.index()] && edge_uses_stack(func, plan, frame, to, args) {
                    fast[to.index()] = false;
                    changed = true;
                }
            });
        }
        if !changed {
            break;
        }
    }
    let mut split_ret: Vec<bool> = (0..n)
        .map(|i| {
            let b = Block(i as u32);
            !fast[i]
                && cfg.reachable[i]
                && func.blocks[i].insts.len() == 1
                && matches!(func.last(b), Some(Op::Return(_)))
                && pure_block(func, &defs, b, plan, frame, clobbers, &fast_call)
        })
        .collect();
    for i in 0..n {
        if !fast[i] {
            continue;
        }
        visit_edges(func, Block(i as u32), |to, args| {
            let jt = func.jump_target(to).index();
            if split_ret[jt] && edge_uses_stack(func, plan, frame, to, args) {
                split_ret[jt] = false;
            }
        });
    }
    let mut can_ret = vec![false; n];
    let mut rstack: Vec<Block> = (0..n)
        .filter(|&i| {
            let b = Block(i as u32);
            fast[i]
                && (matches!(func.last(b), Some(Op::Return(_)))
                    || func
                        .successors(b)
                        .any(|s| split_ret[func.jump_target(s).index()]))
        })
        .map(|i| {
            can_ret[i] = true;
            Block(i as u32)
        })
        .collect();
    while let Some(b) = rstack.pop() {
        for &p in cfg.preds_of(b) {
            if fast[p.index()] && !can_ret[p.index()] {
                can_ret[p.index()] = true;
                rstack.push(p);
            }
        }
    }
    for i in 0..n {
        fast[i] &= can_ret[i];
    }
    (fast[entry.index()] && (0..n).any(|i| cfg.reachable[i] && !fast[i]))
        .then_some(ShrinkWrap { fast, split_ret })
}
