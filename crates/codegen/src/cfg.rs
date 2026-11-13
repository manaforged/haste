use crate::ir::{Block, BlockData, Function, Op, Value};

pub(crate) struct Loop {
    pub head: Block,
    pub latches: Vec<Block>,
    pub body: Vec<bool>,
    pub size: usize,
    pub depth: u32,
}

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
}

fn all_preds(func: &Function) -> Vec<Vec<Block>> {
    let mut preds = vec![Vec::new(); func.blocks.len()];
    for b in 0..func.blocks.len() {
        let block = Block(b as u32);
        func.successors(block).for_each(|target| preds[target.index()].push(block));
    }
    preds
}

fn reverse_postorder(func: &Function, entry: Block) -> (Vec<Block>, Vec<bool>) {
    let mut seen = vec![false; func.blocks.len()];
    if entry.index() >= func.blocks.len() {
        return (Vec::new(), seen);
    }
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
    (post, seen)
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

fn natural_loop(cfg: &Cfg, head: Block, latches: Vec<Block>) -> Loop {
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
        match func.entry {
            Some(entry) => Self::from_entry(func, entry),
            None => Self {
                entry: None,
                reachable: vec![false; func.blocks.len()],
                rpo: Vec::new(),
                preds: all_preds(func),
                idom: vec![None; func.blocks.len()],
            },
        }
    }

    pub(crate) fn from_entry(func: &Function, entry: Block) -> Self {
        let preds = all_preds(func);
        let (rpo, reachable) = reverse_postorder(func, entry);
        let rank = Self::ranks(&rpo, func.blocks.len());
        let idom = immediate(&rpo, &rank, &preds, func.blocks.len());
        Self { entry: Some(entry), reachable, rpo, preds, idom }
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

    pub(crate) fn loops(&self, _func: &Function) -> Loops {
        let mut list: Vec<Loop> = self
            .rpo
            .iter()
            .filter_map(|&head| {
                let mut latches = Vec::new();
                for &pred in &self.preds[head.index()] {
                    if self.reachable[pred.index()] && self.dominates(head, pred) && !latches.contains(&pred) {
                        latches.push(pred);
                    }
                }
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

pub(crate) fn preheader(func: &mut Function, cfg: &Cfg, each: &Loop) -> Option<Block> {
    if Some(each.head) == func.entry {
        return None;
    }
    let mut outside = Vec::new();
    for &pred in cfg.preds_of(each.head) {
        if !each.body.get(pred.index()).copied().unwrap_or(false) && !outside.contains(&pred) {
            outside.push(pred);
        }
    }
    if let [single] = outside[..]
        && matches!(func.last(single), Some(Op::Jump(target, _)) if *target == each.head)
    {
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
    Some(pre)
}
