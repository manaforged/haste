use super::SelectRec;
use crate::cfg::Cfg;
use crate::ir::Block;

/// The selects of the blocks that dominate the block being settled, in the order the function lists them.
#[derive(Default)]
pub(super) struct Live {
    pub(super) recs: Vec<SelectRec>,
    near: Vec<usize>,
    nodes: Vec<(usize, usize, usize)>,
    spans: Vec<(usize, usize)>,
}

impl Live {
    /// Moves to `block`, after every block that dominates it; `selects` is sorted by block.
    pub(super) fn enter(&mut self, cfg: &Cfg, selects: &[SelectRec], block: Block) {
        if selects.is_empty() {
            return;
        }
        self.near.resize(cfg.idom.len(), usize::MAX);
        let up = cfg.idom[block.index()].filter(|&d| d != block).map_or(usize::MAX, |d| self.near[d.index()]);
        let start = selects.partition_point(|rec| rec.block.index() < block.index());
        let end = start + selects[start..].partition_point(|rec| rec.block == block);
        let mut at = up;
        if start < end {
            at = self.nodes.len();
            self.nodes.push((start, end, up));
        }
        self.near[block.index()] = at;
        self.spans.clear();
        while let Some(&(start, end, up)) = self.nodes.get(at) {
            self.spans.push((start, end));
            at = up;
        }
        self.spans.sort_unstable();
        self.recs.clear();
        for &(start, end) in &self.spans {
            self.recs.extend_from_slice(&selects[start..end]);
        }
    }
}
