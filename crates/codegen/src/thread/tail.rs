use crate::ir::{Block, Function, Inst, IntCC, Op, Value};

/// The most instructions a copied block may run before its branch.
const MAX_COPY: usize = 8;

/// The compare a block branches on: the condition, its two sides, and the two targets.
struct Branch {
    cc: IntCC,
    sides: [Value; 2],
    then: Block,
    other: Block,
}

/// Copies a short block that branches on a compare into each predecessor whose jump arguments make both sides of that compare one expression, and sends the copy straight to the side the compare takes; returns whether it copied any block.
pub(super) fn run(func: &mut Function) -> bool {
    let cfg = func.cfg();
    let defs = crate::fold::defs(func);
    let mut copies: Vec<(Block, Block, Block)> = Vec::new();
    for at in 0..func.blocks.len() {
        let block = Block(at as u32);
        if !cfg.reachable[at] {
            continue;
        }
        let Some(found) = branch(func, &defs, block) else { continue };
        for &pred in cfg.preds_of(block) {
            let Some(Op::Jump(target, args)) = func.last(pred) else { continue };
            if pred == block || *target != block || copies.iter().any(|&(p, ..)| p == pred) || !same(func, &defs, block, args, found.sides) {
                continue;
            }
            let taken = matches!(found.cc, IntCC::Equal | IntCC::UnsignedLessThanOrEqual | IntCC::UnsignedGreaterThanOrEqual | IntCC::SignedLessThanOrEqual | IntCC::SignedGreaterThanOrEqual);
            let (to, skipped) = if taken { (found.then, found.other) } else { (found.other, found.then) };
            if to != skipped && cfg.single_pred(skipped) == Some(block) && stays_under(func, &cfg, block, skipped) {
                copies.push((pred, block, to));
            }
        }
    }
    for &(pred, block, to) in &copies {
        localize_consts(func, block);
        copy_into(func, pred, block, to);
    }
    if !copies.is_empty() {
        func.bump_cfg();
    }
    !copies.is_empty()
}

/// The branch ending `block` when its body is a short run of pure instructions, loads and stores and its condition is a compare the body defines.
fn branch(func: &Function, defs: &[Option<Inst>], block: Block) -> Option<Branch> {
    let (&last, body) = func.blocks[block.index()].insts.split_last()?;
    let Op::Brif(cond, then, _, other, _) = func.insts[last.index()].op else { return None };
    if body.len() > MAX_COPY || !body.iter().all(|inst| matches!(func.insts[inst.index()].op, Op::Store(..) | Op::Load(..)) || func.insts[inst.index()].op.is_speculable()) {
        return None;
    }
    let def = defs[cond.index()].filter(|inst| body.contains(inst))?;
    let Op::Icmp(cc, a, b) = func.insts[def.index()].op else { return None };
    Some(Branch { cc, sides: [a, b], then, other })
}

/// Whether both `sides` of the compare in `block` compute one expression once its parameters take `args`.
fn same(func: &Function, defs: &[Option<Inst>], block: Block, args: &[Value], sides: [Value; 2]) -> bool {
    let params = &func.blocks[block.index()].params;
    let local = |v: Value| defs[v.index()].is_some_and(|inst| func.blocks[block.index()].insts.contains(&inst));
    let subst = |v: Value| params.iter().position(|&p| p == v).map_or(v, |i| args[i]);
    let constant = |v: Value| match defs[v.index()].map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Const(imm)) => Some(imm),
        _ => None,
    };
    let leaf = |u: Value, v: Value| u == v || constant(u).is_some_and(|c| constant(v) == Some(c));
    let shape = |v: Value| {
        let mapped = subst(v);
        let op = match defs[mapped.index()].map(|inst| &func.insts[inst.index()].op) {
            Some(&Op::Binary(bin, x, y)) if local(mapped) => Some((bin, subst(x), subst(y))),
            Some(&Op::Binary(bin, x, y)) => Some((bin, x, y)),
            _ => None,
        };
        (mapped, op)
    };
    let ((a, a_op), (b, b_op)) = (shape(sides[0]), shape(sides[1]));
    a == b || matches!((a_op, b_op), (Some((p, x1, y1)), Some((q, x2, y2))) if p == q && leaf(x1, x2) && leaf(y1, y2))
}

/// Whether every read of a parameter or computed value of `block` outside `block` sits where `skipped` dominates.
fn stays_under(func: &Function, cfg: &crate::cfg::Cfg, block: Block, skipped: Block) -> bool {
    let data = &func.blocks[block.index()];
    let mut own = vec![false; func.types.len()];
    data.params.iter().for_each(|&p| own[p.index()] = true);
    data.insts.iter().filter(|inst| !matches!(func.insts[inst.index()].op, Op::Const(_))).filter_map(|inst| func.insts[inst.index()].result).for_each(|r| own[r.index()] = true);
    func.blocks.iter().enumerate().all(|(at, other)| {
        let here = Block(at as u32);
        here == block
            || cfg.dominates(skipped, here)
            || other.insts.iter().all(|inst| {
                let mut reads = false;
                func.insts[inst.index()].op.for_operands(|v| reads |= own[v.index()]);
                !reads
            })
    })
}

/// Gives each other block that reads a constant of `block` its own copy of that constant.
fn localize_consts(func: &mut Function, block: Block) {
    let consts: Vec<(Value, i64)> = func.blocks[block.index()]
        .insts
        .iter()
        .filter_map(|inst| match (&func.insts[inst.index()].op, func.insts[inst.index()].result) {
            (&Op::Const(imm), Some(res)) => Some((res, imm)),
            _ => None,
        })
        .collect();
    for at in 0..func.blocks.len() {
        if at == block.index() {
            continue;
        }
        let mut local: Vec<(Value, Value)> = Vec::new();
        for i in 0..func.blocks[at].insts.len() {
            let inst = func.blocks[at].insts[i];
            let mut reads = Vec::new();
            func.insts[inst.index()].op.for_operands(|v| reads.extend(consts.iter().filter(|&&(c, _)| c == v).map(|&(c, _)| c)));
            for c in reads {
                if local.iter().all(|&(from, _)| from != c) {
                    let fresh = func.value(func.ty(c));
                    local.push((c, fresh));
                }
            }
            let mut op = func.insts[inst.index()].op.clone();
            op.map_operands(|v| local.iter().find(|&&(from, _)| from == v).map_or(v, |&(_, into)| into));
            func.insts[inst.index()].op = op;
        }
        let defs: Vec<Inst> = local.iter().map(|&(from, into)| {
            let imm = consts.iter().find(|&&(c, _)| c == from).map_or(0, |&(_, imm)| imm);
            func.add(Op::Const(imm), Some(into))
        }).collect();
        func.blocks[at].insts.splice(0..0, defs);
    }
}

/// Puts a copy of the body of `block` before the jump that ends `pred` and points that jump at `to` with the arguments `block` passes it.
fn copy_into(func: &mut Function, pred: Block, block: Block, to: Block) {
    let Some(&jump) = func.blocks[pred.index()].insts.last() else { return };
    let Op::Jump(_, ref args) = func.insts[jump.index()].op else { return };
    let mut map: Vec<(Value, Value)> = func.blocks[block.index()].params.iter().copied().zip(args.iter().copied()).collect();
    let (&last, body) = func.blocks[block.index()].insts.split_last().expect("invariant: branch() saw a terminator");
    let body = body.to_vec();
    let mut copied = Vec::with_capacity(body.len());
    for inst in body {
        let data = func.insts[inst.index()].clone();
        let mut op = data.op;
        op.map_operands(|v| map.iter().find(|&&(from, _)| from == v).map_or(v, |&(_, into)| into));
        let result = data.result.map(|r| {
            let fresh = func.value(func.ty(r));
            map.push((r, fresh));
            fresh
        });
        let new = func.add(op, result);
        func.insts[new.index()].loc = data.loc;
        copied.push(new);
    }
    let Op::Brif(_, then, ref then_args, _, ref other_args) = func.insts[last.index()].op else { return };
    let mut to_args = if to == then { then_args.clone() } else { other_args.clone() };
    to_args.iter_mut().for_each(|v| *v = map.iter().find(|&&(from, _)| from == *v).map_or(*v, |&(_, into)| into));
    func.insts[jump.index()].op = Op::Jump(to, to_args);
    let insts = &mut func.blocks[pred.index()].insts;
    let at = insts.len() - 1;
    insts.splice(at..at, copied);
}
