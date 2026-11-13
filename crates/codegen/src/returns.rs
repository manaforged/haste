use crate::ir::{Block, Function, Op, Value};

/// Turns every jump into a block that only returns into that return, so the edge moves no values and takes no branch.
pub(crate) fn run(func: &mut Function) -> bool {
    let mut entered = vec![0_u32; func.blocks.len()];
    for block in 0..func.blocks.len() {
        func.successors(Block(block as u32)).for_each(|target| entered[target.index()] += 1);
    }
    let mut threaded = vec![0_u32; func.blocks.len()];
    let mut changed = false;
    for block in 0..func.blocks.len() {
        let Some(&last) = func.blocks[block].insts.last() else { continue };
        let Op::Jump(target, args) = &func.insts[last.index()].op else { continue };
        let Some(values) = returned(func, *target, args) else { continue };
        threaded[target.index()] += 1;
        func.insts[last.index()].op = Op::Return(values);
        changed = true;
    }
    for (block, count) in threaded.iter().enumerate() {
        if *count > 0 && *count == entered[block] && func.entry != Some(Block(block as u32)) {
            func.blocks[block].insts.clear();
            changed = true;
        }
    }
    changed
}

/// The values a jump into `target` with `args` returns, when `target` holds only a return.
fn returned(func: &Function, target: Block, args: &[Value]) -> Option<Vec<Value>> {
    let data = &func.blocks[target.index()];
    let [only] = data.insts[..] else { return None };
    let Op::Return(values) = &func.insts[only.index()].op else { return None };
    let substitute = |value: &Value| data.params.iter().position(|param| param == value).map_or(*value, |at| args[at]);
    Some(values.iter().map(substitute).collect())
}
