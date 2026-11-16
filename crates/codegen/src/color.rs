use crate::alloc::Location;
use crate::frame::{WORD, round};
use crate::ir::Function;
use crate::live::{Live, overlaps};

/// The most values one web holds, which bounds the live-range checks a merge makes.
const MAX_WEB: usize = 16;

/// Values that block edges copy into one another and that are never live at once, grouped to share one frame home.
pub(crate) struct Webs {
    root: Vec<usize>,
    members: Vec<Vec<usize>>,
}

impl Webs {
    /// One web per value of `count` values.
    pub(crate) fn new(count: usize) -> Self {
        Webs { root: (0..count).collect(), members: vec![Vec::new(); count] }
    }

    fn members(&self, root: usize) -> &[usize] {
        if self.members[root].is_empty() { std::slice::from_ref(&self.root[root]) } else { &self.members[root] }
    }

    /// Whether the webs of `a` and `b` are one web or can join: together they stay small and hold no two values live at once.
    pub(crate) fn fits(&self, live: &Live, a: usize, b: usize) -> bool {
        let (left, right) = (self.members(self.root[a]), self.members(self.root[b]));
        self.root[a] == self.root[b]
            || (left.len() + right.len() <= MAX_WEB && left.iter().all(|&x| right.iter().all(|&y| !overlaps(live.ranges_of(x), live.ranges_of(y)))))
    }

    /// Joins the webs of `a` and `b`.
    pub(crate) fn merge(&mut self, a: usize, b: usize) {
        let (keep, gone) = (self.root[a], self.root[b]);
        if keep == gone {
            return;
        }
        let moved = if self.members[gone].is_empty() { vec![gone] } else { std::mem::take(&mut self.members[gone]) };
        if self.members[keep].is_empty() {
            self.members[keep].push(keep);
        }
        moved.iter().for_each(|&value| self.root[value] = keep);
        self.members[keep].extend(moved);
    }
}

pub(crate) fn share<R>(func: &Function, live: &Live, locations: &[Location<R>], alone: &[bool], webs: &Webs) -> Vec<Option<u32>> {
    let memory = |value: usize| !alone[value] && matches!(locations[value], Location::Slot);
    let span = |root: usize| {
        webs.members(root).iter().filter(|&&value| memory(value)).filter_map(|&value| live.bounds(value)).reduce(|(start, end), (s, e)| (start.min(s), end.max(e)))
    };
    let mut units: Vec<(u32, u32, usize)> = (0..locations.len()).filter(|&value| webs.root[value] == value).filter_map(|root| span(root).map(|(start, end)| (start, end, root))).collect();
    units.sort_unstable();
    let mut homes = vec![None; locations.len()];
    let mut sizes: Vec<u32> = Vec::new();
    let mut holders: Vec<Vec<usize>> = Vec::new();
    for (start, _, root) in units {
        let members: Vec<usize> = webs.members(root).iter().copied().filter(|&value| memory(value) && live.bounds(value).is_some()).collect();
        let size = round(func.types[root].bytes(), WORD);
        let clear = |list: &[usize]| list.iter().all(|&other| members.iter().all(|&value| !overlaps(live.ranges_of(other), live.ranges_of(value))));
        let mut chosen = None;
        for (home, list) in holders.iter_mut().enumerate() {
            if sizes[home] != size {
                continue;
            }
            list.retain(|&other| live.last_end(other).is_some_and(|end| end >= start));
            if clear(list) {
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
        for &value in &members {
            homes[value] = Some(home);
        }
        holders[home as usize].extend(members);
    }
    homes
}
