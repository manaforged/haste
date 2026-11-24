use crate::cfg::{Cfg, Loops};
use crate::ir::{Block, Function, Value};

use super::DomState;
use super::loops::ExtraPath;
use super::rel::Path;

/// What settle knows on exit from each block it has passed, handed on to the blocks that start from it.
pub(super) struct Exits {
    paths: Vec<Option<Path>>,
    extras: Vec<Option<ExtraPath>>,
    doms: Vec<Option<DomState>>,
    follow: Vec<Option<Block>>,
    last_reader: Vec<Option<Block>>,
    pinned: Vec<bool>,
}

impl Exits {
    /// Plans the walk in reverse postorder: the predecessor each block follows, the last block to start from each exit, and the exits a loop reads.
    pub(super) fn new(
        func: &Function,
        cfg: &Cfg,
        (single, entry): (&[Option<Block>], Block),
        (loops, incoming): (&Loops, &[Vec<(Block, &[Value])>]),
    ) -> Self {
        let n = func.blocks.len();
        let rank = Cfg::ranks(&cfg.rpo, n);
        let (mut follow, mut last_reader, mut pinned) =
            (vec![None; n], vec![None; n], vec![false; n]);
        for &block in &cfg.rpo {
            let b = block.index();
            follow[b] = single[b].or_else(|| {
                (block != entry)
                    .then(|| {
                        let mut outside = cfg
                            .preds_of(block)
                            .iter()
                            .copied()
                            .filter(|&p| cfg.reachable[p.index()] && !cfg.dominates(block, p));
                        let first = outside.next()?;
                        (outside.next().is_none() && cfg.idom[b] == Some(first)).then_some(first)
                    })
                    .flatten()
            });
            let earlier = |p: &Block| rank[p.index()] < rank[b];
            let source = follow[b]
                .filter(earlier)
                .or_else(|| cfg.idom[b].filter(|_| block != entry).filter(earlier));
            if let Some(source) = source {
                last_reader[source.index()] = Some(block);
            }
        }
        for each in &loops.list {
            if let Some(dom) = cfg.idom[each.head.index()] {
                pinned[dom.index()] = true;
            }
            for &(pred, _) in incoming
                .get(each.head.index())
                .map_or(&[][..], Vec::as_slice)
            {
                pinned[pred.index()] = true;
            }
        }
        Self {
            paths: vec![None; n],
            extras: vec![None; n],
            doms: vec![None; n],
            follow,
            last_reader,
            pinned,
        }
    }

    /// The path on exit from each block passed so far.
    pub(super) fn paths(&self) -> &[Option<Path>] {
        &self.paths
    }

    /// The state `block` starts from, with the predecessor it follows when that predecessor's exit is known, else its immediate dominator's exit.
    pub(super) fn start(
        &mut self,
        cfg: &Cfg,
        block: Block,
        entry: Block,
    ) -> (Option<Block>, Path, ExtraPath, DomState) {
        if let Some(pred) = self.follow[block.index()].filter(|p| self.paths[p.index()].is_some()) {
            let (path, extra, dom) = self.read(pred, block);
            return (Some(pred), path, extra, dom);
        }
        match cfg.idom[block.index()].filter(|_| block != entry) {
            Some(dom) => {
                let (path, extra, dom) = self.read(dom, block);
                (None, path, extra, dom)
            }
            None => (
                None,
                Path::default(),
                ExtraPath::default(),
                DomState::default(),
            ),
        }
    }

    fn read(&mut self, source: Block, block: Block) -> (Path, ExtraPath, DomState) {
        let s = source.index();
        if self.last_reader[s] == Some(block) && !self.pinned[s] {
            (
                self.paths[s].take().unwrap_or_default(),
                self.extras[s].take().unwrap_or_default(),
                self.doms[s].take().unwrap_or_default(),
            )
        } else {
            (
                self.paths[s].clone().unwrap_or_default(),
                self.extras[s].clone().unwrap_or_default(),
                self.doms[s].clone().unwrap_or_default(),
            )
        }
    }

    /// Records the state on exit from `block`.
    pub(super) fn store(&mut self, block: Block, (path, extra, dom): (Path, ExtraPath, DomState)) {
        self.paths[block.index()] = Some(path);
        self.extras[block.index()] = Some(extra);
        self.doms[block.index()] = Some(dom);
    }
}
