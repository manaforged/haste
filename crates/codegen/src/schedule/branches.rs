use crate::ir::{Binary, Block, BlockData, Function, Inst, Op, Value};

/// Splits each branch on the `and` or `or` of two single-use conditions of its own block into a branch on the first and, in a new block, a branch on the second, and moves there the pure instructions only the second condition reads, unless both compare one value and so fold into one range compare.
pub(super) fn split_conditions(func: &mut Function) {
    let uses = func.use_counts();
    let (defs, consts) = (crate::fold::defs(func), crate::returns::collect_consts(func));
    let mut changed = false;
    for block in 0..func.blocks.len() {
        let Some((&term, body)) = func.blocks[block].insts.split_last() else { continue };
        let Op::Brif(cond, then, ref then_args, otherwise, ref else_args) = func.insts[term.index()].op else { continue };
        let (then_args, else_args) = (then_args.clone(), else_args.clone());
        let pos = |v: Value| defs[v.index()].and_then(|inst| body.iter().position(|&at| at == inst));
        let Some(join) = pos(cond).filter(|_| uses[cond.index()] == 1) else { continue };
        let Op::Binary(op @ (Binary::Band | Binary::Bor), first, second) = func.insts[body[join].index()].op else { continue };
        let single = |c: Value| uses[c.index()] == 1 && pos(c).is_some() && crate::thread::is_bool_val(func, &defs, &consts, c);
        if then == otherwise || first == second || !single(first) || !single(second) || shares_operand(func, &defs, first, second) {
            continue;
        }
        let moved = exclusive(func, body, &uses, second);
        let next = Block(func.blocks.len() as u32);
        let mut kept: Vec<Inst> = body.iter().enumerate().filter(|&(at, inst)| at != join && !moved.contains(inst)).map(|(_, &inst)| inst).collect();
        let mut tail: Vec<Inst> = body.iter().filter(|inst| moved.contains(inst)).copied().collect();
        let head = match op {
            Binary::Band => Op::Brif(first, next, Vec::new(), otherwise, else_args.clone()),
            _ => Op::Brif(first, then, then_args.clone(), next, Vec::new()),
        };
        func.insts[term.index()].op = head;
        kept.push(term);
        tail.push(func.add(Op::Brif(second, then, then_args, otherwise, else_args), None));
        func.blocks[block].insts = kept;
        func.blocks.push(BlockData { params: Vec::new(), insts: tail });
        changed = true;
    }
    if changed {
        func.bump_cfg();
    }
}

/// The pure instructions of `body` that compute `second` and whose results nothing else reads.
fn exclusive(func: &Function, body: &[Inst], uses: &[u32], second: Value) -> Vec<Inst> {
    let mut moved: Vec<Inst> = Vec::new();
    let mut reads = vec![0_u32; uses.len()];
    for &inst in body.iter().rev() {
        let data = &func.insts[inst.index()];
        let Some(result) = data.result else { continue };
        if !data.op.is_speculable() || matches!(data.op, Op::Load(..) | Op::StackLoad(..)) || (result != second && (reads[result.index()] == 0 || reads[result.index()] != uses[result.index()])) {
            continue;
        }
        data.op.for_operands(|value| reads[value.index()] += 1);
        moved.push(inst);
    }
    moved
}

/// Whether `first` and `second` are integer compares that read a common value.
fn shares_operand(func: &Function, defs: &[Option<Inst>], first: Value, second: Value) -> bool {
    let operands = |c: Value| match defs[c.index()].map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Icmp(_, x, y)) => Some([x, y]),
        _ => None,
    };
    matches!((operands(first), operands(second)), (Some(a), Some(b)) if a.iter().any(|v| b.contains(v)))
}
