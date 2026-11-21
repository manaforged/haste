use crate::ir::{Block, BlockData, Function, Op, Value};
use std::sync::OnceLock;

#[derive(Clone)]
pub(crate) struct Loop {
    pub head: Block,
    pub latches: Vec<Block>,
    pub body: Vec<bool>,
    pub size: usize,
    pub depth: u32,
}

#[derive(Clone)]
pub(crate) struct Loops {
    pub list: Vec<Loop>,
    pub depths: Vec<u32>,
}

impl Loops {
    pub(crate) fn depth_of(&self, block: Block) -> u32 {
        self.depths.get(block.index()).copied().unwrap_or(0)
    }
}

pub(crate) struct Cfg {
    pub entry: Option<Block>,
    pub reachable: Vec<bool>,
    pub rpo: Vec<Block>,
    pub preds: Vec<Vec<Block>>,
    pub idom: Vec<Option<Block>>,
    edges: Edges,
    loops: OnceLock<Loops>,
}

impl Clone for Cfg {
    fn clone(&self) -> Self {
        Self { entry: self.entry, reachable: self.reachable.clone(), rpo: self.rpo.clone(), preds: self.preds.clone(), idom: self.idom.clone(), edges: self.edges.clone(), loops: OnceLock::new() }
    }
}

#[derive(Clone)]
struct Edges {
    targets: Vec<Block>,
    ends: Vec<u32>,
}

impl Edges {
    fn new(func: &Function) -> Self {
        let blocks = func.blocks.len();
        let mut edges = Self { targets: Vec::with_capacity(blocks * 2), ends: Vec::with_capacity(blocks) };
        for b in 0..blocks {
            edges.targets.extend(func.successors(Block(b as u32)));
            edges.ends.push(edges.targets.len() as u32);
        }
        edges
    }

    fn of(&self, block: Block) -> &[Block] {
        let at = block.index();
        let start = at.checked_sub(1).map_or(0, |prev| self.ends[prev] as usize);
        &self.targets[start..self.ends[at] as usize]
    }

    fn matches(&self, func: &Function) -> bool {
        self.ends.len() == func.blocks.len()
            && (0..self.ends.len()).all(|b| {
                let block = Block(b as u32);
                func.successors(block).eq(self.of(block).iter().copied())
            })
    }

    fn preds(&self) -> Vec<Vec<Block>> {
        let mut preds = vec![Vec::new(); self.ends.len()];
        for b in 0..self.ends.len() {
            let block = Block(b as u32);
            self.of(block).iter().for_each(|target| preds[target.index()].push(block));
        }
        preds
    }
}

fn order<I: Iterator<Item = Block>>(blocks: usize, entry: Block, successors: impl Fn(Block) -> I) -> (Vec<Block>, Vec<bool>) {
    let mut seen = vec![false; blocks];
    if entry.index() >= blocks {
        return (Vec::new(), seen);
    }
    let mut post = Vec::with_capacity(blocks);
    let mut stack = vec![(entry, successors(entry))];
    seen[entry.index()] = true;
    while let Some((block, targets)) = stack.last_mut() {
        match targets.next() {
            Some(target) if !seen[target.index()] => {
                seen[target.index()] = true;
                stack.push((target, successors(target)));
            }
            Some(_) => {}
            None => {
                post.push(*block);
                stack.pop();
            }
        }
    }
    post.reverse();
    (post, seen)
}

pub(crate) fn reverse_postorder(func: &Function, entry: Block) -> (Vec<Block>, Vec<bool>) {
    order(func.blocks.len(), entry, |block| func.successors(block))
}

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

fn immediate(order: &[Block], rank: &[usize], preds: &[Vec<Block>], blocks: usize) -> Vec<Option<Block>> {
    let mut idom = vec![None; blocks];
    let Some(&entry) = order.first() else { return idom };
    idom[entry.index()] = Some(entry);
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &order[1..] {
            let mut done = preds[block.index()].iter().copied().filter(|pred| idom[pred.index()].is_some());
            let Some(first) = done.next() else { continue };
            let new = done.fold(first, |acc, pred| intersect(&idom, rank, pred, acc));
            if idom[block.index()] != Some(new) {
                idom[block.index()] = Some(new);
                changed = true;
            }
        }
    }
    idom
}

pub(crate) fn dominates(idom: &[Option<Block>], ancestor: Block, mut block: Block) -> bool {
    loop {
        if block == ancestor {
            return true;
        }
        match idom.get(block.index()).copied().flatten() {
            Some(parent) if parent != block => block = parent,
            _ => return false,
        }
    }
}

pub(crate) fn natural_loop(cfg: &Cfg, head: Block, latches: Vec<Block>) -> Loop {
    let mut body = vec![false; cfg.preds.len()];
    body[head.index()] = true;
    let mut todo = latches.clone();
    while let Some(block) = todo.pop() {
        let idx = block.index();
        if cfg.reachable[idx] && !std::mem::replace(&mut body[idx], true) {
            todo.extend(cfg.preds[idx].iter().copied().filter(|p| cfg.reachable[p.index()]));
        }
    }
    let size = body.iter().filter(|inside| **inside).count();
    Loop { head, latches, body, size, depth: 0 }
}

impl Cfg {
    pub(crate) fn new(func: &Function) -> Self {
        let blocks = func.blocks.len();
        let edges = Edges::new(func);
        let preds = edges.preds();
        let (rpo, reachable) = match func.entry {
            Some(entry) => order(blocks, entry, |block| edges.of(block).iter().copied()),
            None => (Vec::new(), vec![false; blocks]),
        };
        let rank = Self::ranks(&rpo, blocks);
        let idom = immediate(&rpo, &rank, &preds, blocks);
        Self { entry: func.entry, reachable, rpo, preds, idom, edges, loops: OnceLock::new() }
    }

    /// Whether this graph is what `Cfg::new` builds for `func` now.
    pub(crate) fn matches(&self, func: &Function) -> bool {
        self.entry == func.entry && self.edges.matches(func)
    }

    pub(crate) fn ranks(order: &[Block], blocks: usize) -> Vec<usize> {
        let mut rank = vec![usize::MAX; blocks];
        order.iter().enumerate().for_each(|(at, block)| rank[block.index()] = at);
        rank
    }

    pub(crate) fn pred_count(&self, block: Block) -> u32 {
        self.preds[block.index()].len() as u32
    }

    pub(crate) fn preds_of(&self, block: Block) -> &[Block] {
        &self.preds[block.index()]
    }

    pub(crate) fn single_pred(&self, block: Block) -> Option<Block> {
        if Some(block) == self.entry {
            return None;
        }
        match self.preds[block.index()][..] {
            [pred] => Some(pred),
            _ => None,
        }
    }

    pub(crate) fn dominates(&self, ancestor: Block, block: Block) -> bool {
        dominates(&self.idom, ancestor, block)
    }

    pub(crate) fn children(&self) -> Vec<Vec<Block>> {
        let mut children = vec![Vec::new(); self.preds.len()];
        if let Some((_, rest)) = self.rpo.split_first() {
            for &block in rest {
                if let Some(parent) = self.idom[block.index()] {
                    children[parent.index()].push(block);
                }
            }
        }
        children
    }

    pub(crate) fn loops(&self) -> &Loops {
        self.loops.get_or_init(|| self.find_loops())
    }

    fn find_loops(&self) -> Loops {
        let mut earlier = vec![false; self.preds.len()];
        let mut list: Vec<Loop> = self
            .rpo
            .iter()
            .filter_map(|&head| {
                let mut latches = Vec::new();
                for &pred in &self.preds[head.index()] {
                    if self.reachable[pred.index()] && !earlier[pred.index()] && self.dominates(head, pred) && !latches.contains(&pred) {
                        latches.push(pred);
                    }
                }
                earlier[head.index()] = true;
                (!latches.is_empty()).then(|| natural_loop(self, head, latches))
            })
            .collect();
        list.sort_by_key(|each| each.size);
        let mut depths = vec![0_u32; self.preds.len()];
        for each in &list {
            depths.iter_mut().zip(&each.body).filter(|(_, inside)| **inside).for_each(|(d, _)| *d += 1);
        }
        for each in &mut list {
            each.depth = depths[each.head.index()];
        }
        Loops { list, depths }
    }
}

fn outside_preds(cfg: &Cfg, each: &Loop) -> Vec<Block> {
    let mut outside = Vec::new();
    for &pred in cfg.preds_of(each.head) {
        if !each.body.get(pred.index()).copied().unwrap_or(false) && !outside.contains(&pred) {
            outside.push(pred);
        }
    }
    outside
}

fn sole_entry(func: &Function, each: &Loop, outside: &[Block]) -> Option<Block> {
    match outside {
        [single] if matches!(func.last(*single), Some(Op::Jump(target, _)) if *target == each.head) => Some(*single),
        _ => None,
    }
}

/// The block outside `each` whose jump is the loop's only entry, when one already exists.
pub(crate) fn existing_preheader(func: &Function, cfg: &Cfg, each: &Loop) -> Option<Block> {
    if Some(each.head) == func.entry {
        return None;
    }
    sole_entry(func, each, &outside_preds(cfg, each))
}

pub(crate) fn preheader(func: &mut Function, cfg: &Cfg, each: &Loop) -> Option<Block> {
    if Some(each.head) == func.entry {
        return None;
    }
    let outside = outside_preds(cfg, each);
    if let Some(single) = sole_entry(func, each, &outside) {
        return Some(single);
    }
    if outside.is_empty() || !outside.iter().all(|&pred| matches!(func.last(pred), Some(Op::Jump(..) | Op::Brif(..)))) {
        return None;
    }
    let param_types: Vec<_> = func.blocks[each.head.index()].params.iter().map(|&p| func.ty(p)).collect();
    let pre_params: Vec<Value> = param_types.into_iter().map(|ty| func.value(ty)).collect();
    let jump = func.add(Op::Jump(each.head, pre_params.clone()), None);
    let pre = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: pre_params, insts: vec![jump] });
    for pred in outside {
        let Some(&last) = func.blocks[pred.index()].insts.last() else { continue };
        match &mut func.insts[last.index()].op {
            Op::Jump(target, _) if *target == each.head => *target = pre,
            Op::Brif(_, then, _, otherwise, _) => {
                if *then == each.head {
                    *then = pre;
                }
                if *otherwise == each.head {
                    *otherwise = pre;
                }
            }
            _ => {}
        }
    }
    func.bump_cfg();
    Some(pre)
}

/// Undoes each `(preheader, head)` that `preheader` made and that still only jumps on to its head with its own parameters, and returns whether any made one stays.
pub(crate) fn drop_bare_preheaders(func: &mut Function, made: &[(Block, Block)]) -> bool {
    let bare: Vec<(Block, Block)> = made
        .iter()
        .copied()
        .filter(|&(pre, head)| {
            let data = &func.blocks[pre.index()];
            matches!(data.insts[..], [jump] if matches!(&func.insts[jump.index()].op, Op::Jump(target, args) if *target == head && *args == data.params))
        })
        .collect();
    if bare.is_empty() {
        return !made.is_empty();
    }
    for block in 0..func.blocks.len() {
        let Some(&last) = func.blocks[block].insts.last() else { continue };
        func.insts[last.index()].op.map_targets(|target| bare.iter().find(|&&(pre, _)| pre == target).map_or(target, |&(_, head)| head));
    }
    for &(pre, _) in bare.iter().rev() {
        func.blocks[pre.index()].insts.drain(..).for_each(|inst| func.insts[inst.index()].op = Op::Trap);
        if pre.index() + 1 == func.blocks.len() {
            func.blocks.pop();
        }
    }
    func.bump_cfg();
    bare.len() < made.len()
}
