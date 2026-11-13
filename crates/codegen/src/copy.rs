use crate::ir::{Function, Inst, Op, Type, Value};

/// The most bytes a move is emitted inline with; a larger one calls `memmove`.
pub(crate) const MOVE_INLINE: u32 = 64;

/// The most chunks a copy of a constant size is emitted inline with, one scratch register each.
const CHUNKS: usize = 4;

/// The integer type and offset of each load and store that copy `bytes` bytes word by word, widest first.
pub(crate) fn words(bytes: u32) -> Vec<(Type, u32)> {
    let mut words = Vec::new();
    let mut at = 0;
    while at < bytes {
        let ty = [Type::I64, Type::I32, Type::I16].into_iter().find(|ty| ty.bytes() <= bytes - at).unwrap_or(Type::I8);
        words.push((ty, at));
        at += ty.bytes();
    }
    words
}

/// The widths and offsets of the loads that copy `bytes` bytes inline, widest first, when there are at most `CHUNKS` of them.
pub(crate) fn chunks(bytes: i64) -> Option<Vec<(u32, u32)>> {
    let bytes = u32::try_from(bytes).ok().filter(|bytes| *bytes <= 8 * CHUNKS as u32)?;
    let words = words(bytes);
    (words.len() <= CHUNKS).then(|| words.iter().map(|&(ty, at)| (ty.bytes(), at)).collect())
}

/// A move that `expand` split into the loads and stores in `pairs`, which `rejoin` can turn back into it.
pub(crate) struct Split {
    block: usize,
    original: Inst,
    pairs: Vec<Inst>,
}

/// Replaces each move for which `split` gives pieces, from its target, source and size, with a load and a store per piece, in order, and returns what it split.
pub(crate) fn expand(func: &mut Function, split: impl Fn(Value, Value, u32) -> Option<Vec<(Type, u32)>>) -> Vec<Split> {
    let mut splits = Vec::new();
    for block in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            let pieces = match func.insts[inst.index()].op {
                Op::Move(to, from, bytes) => split(to, from, bytes).map(|pieces| (to, from, pieces)),
                _ => None,
            };
            let Some((to, from, pieces)) = pieces else {
                out.push(inst);
                continue;
            };
            let start = out.len();
            pairs(func, (to, from, pieces), &mut out);
            splits.push(Split { block, original: inst, pairs: out[start..].to_vec() });
        }
        func.blocks[block].insts = out;
    }
    splits
}

/// Appends to `out` a load from `from` and a store to `to` for each piece of a move.
fn pairs(func: &mut Function, (to, from, pieces): (Value, Value, Vec<(Type, u32)>), out: &mut Vec<Inst>) {
    for (ty, at) in pieces {
        let word = func.value(ty);
        out.push(func.add(Op::Load(from, at as i32), Some(word)));
        out.push(func.add(Op::Store(word, to, at as i32), None));
    }
}

/// Turns back into its move each split whose loads and stores all still stand together, each store writing the word its load read.
pub(crate) fn rejoin(func: &mut Function, splits: &[Split]) {
    for split in splits {
        let insts = &func.blocks[split.block].insts;
        let Some(at) = insts.iter().position(|inst| Some(inst) == split.pairs.first()) else { continue };
        let Some(ends) = insts.get(at..at + split.pairs.len()).filter(|window| *window == split.pairs).and_then(|_| bases(func, &split.pairs)) else { continue };
        let Op::Move(_, _, bytes) = func.insts[split.original.index()].op else { continue };
        func.insts[split.original.index()].op = Op::Move(ends.0, ends.1, bytes);
        func.blocks[split.block].insts.splice(at..at + split.pairs.len(), [split.original]);
    }
}

/// The target and source address of loads and stores that copy word by word, when every store writes the word the load before it read.
fn bases(func: &Function, pairs: &[Inst]) -> Option<(Value, Value)> {
    let mut ends = None;
    for pair in pairs.chunks(2) {
        let [load, store] = pair else { return None };
        let (Op::Load(from, _), Some(word)) = (&func.insts[load.index()].op, func.insts[load.index()].result) else { return None };
        let Op::Store(value, to, _) = func.insts[store.index()].op else { return None };
        if value != word || ends.is_some_and(|ends| ends != (to, *from)) {
            return None;
        }
        ends = Some((to, *from));
    }
    ends
}
