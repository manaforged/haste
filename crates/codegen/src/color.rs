use crate::alloc::Location;
use crate::frame::{WORD, round};
use crate::ir::Function;
use crate::live::{Live, overlaps};

pub(crate) fn share<R>(func: &Function, live: &Live, locations: &[Location<R>], alone: &[bool]) -> Vec<Option<u32>> {
    let memory = |value: usize| !alone[value] && matches!(locations[value], Location::Slot);
    let mut values: Vec<(u32, u32, usize)> = (0..locations.len()).filter(|value| memory(*value)).filter_map(|value| live.bounds(value).map(|(start, end)| (start, end, value))).collect();
    values.sort_unstable();
    let mut homes = vec![None; locations.len()];
    let mut sizes: Vec<u32> = Vec::new();
    let mut holders: Vec<Vec<usize>> = Vec::new();
    for (start, _, value) in values {
        let ranges = live.ranges_of(value);
        let size = round(func.types[value].bytes(), WORD);
        let mut chosen = None;
        for (home, list) in holders.iter_mut().enumerate() {
            if sizes[home] != size {
                continue;
            }
            list.retain(|&other| live.last_end(other).is_some_and(|end| end >= start));
            if list.iter().all(|&other| !overlaps(live.ranges_of(other), ranges)) {
                chosen = Some(home as u32);
                break;
            }
        }
        let home = match chosen {
            Some(home) => home,
            None => {
                sizes.push(size);
                holders.push(Vec::new());
                sizes.len() as u32 - 1
            }
        };
        holders[home as usize].push(value);
        homes[value] = Some(home);
    }
    homes
}
