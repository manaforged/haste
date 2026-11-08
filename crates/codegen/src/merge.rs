use crate::ir::{Block, Function, Inst, Op, Value};

fn terminator(func: &Function, block: usize) -> Option<Inst> {
    func.blocks[block].insts.last().copied()
}

/// Folds every block that only one jump enters into the block that jumps, so its values stay block-local.
pub(crate) fn run(func: &mut Function) {
    let mut preds = vec![0_u32; func.blocks.len()];
    for block in 0..func.blocks.len() {
        if let Some(last) = terminator(func, block) {
            for target in func.insts[last.index()].op.targets() {
                preds[target.index()] += 1;
            }
        }
    }
    let mut subst: Vec<Value> = (0..func.types.len() as u32).map(Value).collect();
    let mut merged = false;
    for block in 0..func.blocks.len() {
        while let Some(next) = single_jump(func, &preds, block) {
            absorb(func, block, next, &mut subst);
            merged = true;
        }
    }
    if merged {
        rename(func, &subst);
    }
}

fn single_jump(func: &Function, preds: &[u32], block: usize) -> Option<usize> {
    let last = terminator(func, block)?;
    let Op::Jump(next, _) = &func.insts[last.index()].op else { return None };
    let next = next.index();
    let entry = func.entry.map(Block::index);
    (next != block && preds[next] == 1 && Some(next) != entry && !func.blocks[next].insts.is_empty()).then_some(next)
}

fn absorb(func: &mut Function, block: usize, next: usize, subst: &mut [Value]) {
    let Some(last) = func.blocks[block].insts.pop() else { return };
    if let Op::Jump(_, args) = std::mem::replace(&mut func.insts[last.index()].op, Op::Trap) {
        for (param, arg) in func.blocks[next].params.iter().zip(args) {
            subst[param.index()] = arg;
        }
    }
    let moved = std::mem::take(&mut func.blocks[next]);
    func.blocks[block].insts.extend(moved.insts);
}

fn rename(func: &mut Function, subst: &[Value]) {
    let find = |mut value: Value| {
        while subst[value.index()] != value {
            value = subst[value.index()];
        }
        value
    };
    for data in &mut func.insts {
        data.op.map_operands(find);
    }
}
