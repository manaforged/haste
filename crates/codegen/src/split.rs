use crate::ir::Type;

/// One end of a move inside a slot: the slot and the offset there.
pub(crate) type End = Option<(usize, i64)>;

/// The offsets and types at which each slot is read or written, which a move that copies the slot is split along.
pub(crate) struct Layouts {
    spans: Vec<Vec<(i64, Type)>>,
}

impl Layouts {
    /// The layouts of `slots` slots from their accesses, each a slot, an offset and a type.
    pub(crate) fn new(slots: usize, accesses: impl Iterator<Item = (usize, i64, Type)>) -> Self {
        let mut spans = vec![Vec::new(); slots];
        for (slot, offset, ty) in accesses {
            let known: &mut Vec<(i64, Type)> = &mut spans[slot];
            if !known.contains(&(offset, ty)) {
                known.push((offset, ty));
            }
        }
        Layouts { spans }
    }

    /// The accesses that lie inside the `bytes` bytes at `end`, at offsets from it.
    fn window(&self, end: End, bytes: u32) -> impl Iterator<Item = (i64, Type)> + '_ {
        end.into_iter().flat_map(move |(slot, base)| {
            let limit = base + i64::from(bytes);
            self.spans[slot].iter().filter(move |(at, ty)| *at >= base && at + i64::from(ty.bytes()) <= limit).map(move |(at, ty)| (at - base, *ty))
        })
    }

    /// Gives the end of each move that has no access inside it the accesses of its other end, until no move changes one.
    pub(crate) fn spread(&mut self, moves: &[(End, End, u32)]) {
        for _ in 0..moves.len() {
            let mut changed = false;
            for &(to, from, bytes) in moves {
                changed |= self.adopt(to, from, bytes);
                changed |= self.adopt(from, to, bytes);
            }
            if !changed {
                return;
            }
        }
    }

    /// Copies into `target` the accesses inside `source` that do not overlap any existing access in `target`, and tells whether it copied any.
    fn adopt(&mut self, target: End, source: End, bytes: u32) -> bool {
        let Some((slot, base)) = target else { return false };
        let existing: Vec<(i64, i64)> = self.window(target, bytes).map(|(at, ty)| (at, at + i64::from(ty.bytes()))).collect();
        let found: Vec<(i64, Type)> = self
            .window(source, bytes)
            .filter(|&(at, ty)| {
                let end = at + i64::from(ty.bytes());
                !existing.iter().any(|&(ea, ee)| at < ee && ea < end)
            })
            .map(|(at, ty)| (at + base, ty))
            .filter(|pair| !self.spans[slot].contains(pair))
            .collect();
        let adopted = !found.is_empty();
        self.spans[slot].extend(found);
        adopted
    }

    /// The pieces that copy `bytes` bytes from `from` to `to`: the accesses of either end where they do not overlap, and words between them; none when neither end has an access there.
    pub(crate) fn pieces(&self, to: End, from: End, bytes: u32) -> Option<Vec<(Type, u32)>> {
        let mut found: Vec<(i64, Type)> = self.window(to, bytes).chain(self.window(from, bytes)).collect();
        if found.is_empty() {
            return None;
        }
        found.sort_unstable_by_key(|(at, ty)| (*at, ty.bytes()));
        let mut pieces = Vec::with_capacity(found.len());
        let mut cursor = 0;
        for (at, ty) in found {
            if at < cursor {
                continue;
            }
            gap(&mut pieces, cursor, at);
            pieces.push((ty, at as u32));
            cursor = at + i64::from(ty.bytes());
        }
        gap(&mut pieces, cursor, i64::from(bytes));
        Some(pieces)
    }
}

/// Appends the words that copy the bytes from offset `start` to offset `end`.
fn gap(pieces: &mut Vec<(Type, u32)>, start: i64, end: i64) {
    let base = start as u32;
    pieces.extend(crate::copy::words((end - start) as u32).into_iter().map(|(ty, at)| (ty, at + base)));
}
