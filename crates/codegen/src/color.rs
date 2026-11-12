//! Frame homes shared between values that live in memory over disjoint spans.

use crate::alloc::Location;
use crate::frame::{WORD, round};
use crate::ir::Function;
use crate::live::Live;

/// For each value that lives in memory, has a span in `live` and is not `alone`, a home it shares with values whose spans it does not overlap, by linear scan; the values of one home have the same size in words.
pub(crate) fn share<R>(func: &Function, live: &Live, locations: &[Location<R>], alone: &[bool]) -> Vec<Option<u32>> {
    let memory = |value: usize| !alone[value] && matches!(locations[value], Location::Slot);
    let mut spans: Vec<(u32, u32, usize)> = (0..locations.len()).filter(|value| memory(*value)).filter_map(|value| live.span[value].map(|(start, end)| (start, end, value))).collect();
    spans.sort_unstable();
    let mut homes = vec![None; locations.len()];
    let mut sizes: Vec<u32> = Vec::new();
    let (mut active, mut free): (Vec<(u32, u32)>, Vec<u32>) = (Vec::new(), Vec::new());
    for (start, end, value) in spans {
        active.retain(|&(until, home)| {
            let ended = until < start;
            if ended {
                free.push(home);
            }
            !ended
        });
        let size = round(func.types[value].bytes(), WORD);
        let home = match free.iter().position(|home| sizes[*home as usize] == size) {
            Some(at) => free.swap_remove(at),
            None => {
                sizes.push(size);
                sizes.len() as u32 - 1
            }
        };
        homes[value] = Some(home);
        active.push((end, home));
    }
    homes
}
