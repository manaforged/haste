use crate::ir::{Block, Function};

/// The blocks of `func` reachable from `entry`, in reverse postorder.
pub(crate) fn reverse_postorder(func: &Function, entry: Block) -> Vec<Block> {
    let mut seen = vec![false; func.blocks.len()];
    let mut post = Vec::with_capacity(func.blocks.len());
    let mut stack = vec![(entry, 0_usize)];
    seen[entry.index()] = true;
    while let Some(top) = stack.last_mut() {
        let (block, next) = (top.0, top.1);
        top.1 += 1;
        match func.successors(block).nth(next) {
            Some(target) if !seen[target.index()] => {
                seen[target.index()] = true;
                stack.push((target, 0));
            }
            Some(_) => {}
            None => {
                post.push(block);
                stack.pop();
            }
        }
    }
    post.reverse();
    post
}

/// The position of each block in `order`, or `usize::MAX` for a block not in it.
pub(crate) fn ranks(order: &[Block], blocks: usize) -> Vec<usize> {
    let mut rank = vec![usize::MAX; blocks];
    order.iter().enumerate().for_each(|(at, block)| rank[block.index()] = at);
    rank
}

/// The nearest block that dominates both `a` and `b`, walking up `idom` by reverse postorder `rank`.
fn intersect(idom: &[Option<Block>], rank: &[usize], mut a: Block, mut b: Block) -> Block {
    while a != b {
        while rank[a.index()] > rank[b.index()] {
            a = idom[a.index()].unwrap_or(a);
        }
        while rank[b.index()] > rank[a.index()] {
            b = idom[b.index()].unwrap_or(b);
        }
    }
    a
}

/// The immediate dominator of every block in `order`, by the iterative method of Cooper, Harvey and Kennedy.
fn immediate(order: &[Block], preds: &[Vec<Block>], blocks: usize) -> Vec<Option<Block>> {
    let rank = ranks(order, blocks);
    let mut idom = vec![None; blocks];
    let Some(&entry) = order.first() else { return idom };
    idom[entry.index()] = Some(entry);
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &order[1..] {
            let mut done = preds[block.index()].iter().copied().filter(|pred| idom[pred.index()].is_some());
            let Some(first) = done.next() else { continue };
            let new = done.fold(first, |acc, pred| intersect(&idom, &rank, pred, acc));
            if idom[block.index()] != Some(new) {
                idom[block.index()] = Some(new);
                changed = true;
            }
        }
    }
    idom
}

/// The reachable blocks of `func` in reverse postorder, the reachable predecessors of each block, and the immediate dominator of each reachable block.
pub(crate) fn dominators(func: &Function, entry: Block) -> (Vec<Block>, Vec<Vec<Block>>, Vec<Option<Block>>) {
    let order = reverse_postorder(func, entry);
    let mut preds = vec![Vec::new(); func.blocks.len()];
    for &block in &order {
        func.successors(block).for_each(|target| preds[target.index()].push(block));
    }
    let idom = immediate(&order, &preds, func.blocks.len());
    (order, preds, idom)
}

/// The blocks each block reachable from `entry` immediately dominates.
pub(crate) fn dominated(func: &Function, entry: Block) -> Vec<Vec<Block>> {
    let (order, _, idom) = dominators(func, entry);
    let mut children = vec![Vec::new(); func.blocks.len()];
    for &block in &order[1..] {
        if let Some(parent) = idom[block.index()] {
            children[parent.index()].push(block);
        }
    }
    children
}
