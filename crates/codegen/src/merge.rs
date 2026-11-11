use crate::ir::{Block, Function, Op, Value};

/// Folds every block that only one jump enters into the block that jumps, so its values stay block-local.
pub(crate) fn run(func: &mut Function) {
    let mut preds = vec![0_u32; func.blocks.len()];
    for block in 0..func.blocks.len() {
        for target in func.successors(Block(block as u32)) {
            preds[target.index()] += 1;
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
    let Op::Jump(next, _) = func.last(Block(block as u32))? else { return None };
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
    for data in &mut func.insts {
        data.op.map_operands(|value| find(subst, value));
    }
}

/// The value that `value` stands for once every replacement in `subst` is followed.
pub(crate) fn find(subst: &[Value], mut value: Value) -> Value {
    while subst[value.index()] != value {
        value = subst[value.index()];
    }
    value
}
