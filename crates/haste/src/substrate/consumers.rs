//! Live-consumer table that bounds a substrate layer to the working sets of the builds still on disk.

use super::{read_u32, read_u64};

pub(super) const CONSUMER_REC_LEN: usize = 16;
pub(super) const KEPT_WORKSPACES: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Consumer {
    pub(super) workspace: u64,
    pub(super) kind: u32,
    pub(super) generation: u32,
}

impl Consumer {
    pub(super) fn read(data: &[u8], rec: usize) -> Self {
        Self { workspace: read_u64(data, rec), kind: read_u32(data, rec + 8), generation: read_u32(data, rec + 12) }
    }

    pub(super) fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.workspace.to_le_bytes());
        out.extend_from_slice(&self.kind.to_le_bytes());
        out.extend_from_slice(&self.generation.to_le_bytes());
    }
}

pub(super) fn retain_live(consumers: &mut Vec<Consumer>, current: Consumer) -> u32 {
    consumers.retain(|c| (c.workspace, c.kind) != (current.workspace, current.kind));
    consumers.push(current);
    consumers.sort_by_key(|c| std::cmp::Reverse(c.generation));
    let mut live = Vec::with_capacity(KEPT_WORKSPACES);
    for c in consumers.iter() {
        if !live.contains(&c.workspace) && live.len() < KEPT_WORKSPACES {
            live.push(c.workspace);
        }
    }
    consumers.retain(|c| live.contains(&c.workspace));
    consumers.sort_by_key(|c| (c.workspace, c.kind));
    consumers.iter().map(|c| c.generation).min().unwrap_or(current.generation)
}
