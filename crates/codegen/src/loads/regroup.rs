use super::dom::clobbers_loc;
use super::pack::pred_chain;
use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Function, Inst, Op, Type, Value};
use crate::mem::{MemLoc, Provenance, may_alias};

const MIN_PIECES: usize = 4;
const MIN_BYTES: u32 = 16;

#[derive(Clone, Copy)]
struct Piece {
    at: usize,
    off: i64,
    bytes: u32,
    load: Inst,
}

struct Run {
    dst: Value,
    src: Value,
    delta: i64,
    pieces: Vec<Piece>,
}

/// Regroups field copies into moves when some block stores enough loaded values to be worth it.
pub(super) fn run(func: &mut Function) -> bool {
    let defs = defs(func);
    if !worth(func, &defs) {
        return false;
    }
    let cfg = func.cfg();
    let prov = Provenance::new(func, &cfg);
    regroup(func, &prov, &cfg, &defs)
}

fn worth(func: &Function, defs: &[Option<Inst>]) -> bool {
    func.blocks.iter().any(|block| block.insts.iter().filter(|inst| loaded_store(func, defs, **inst).is_some()).count() >= MIN_PIECES)
}

fn defs(func: &Function) -> Vec<Option<Inst>> {
    let uses = func.use_counts();
    let mut defs = crate::fold::defs(func);
    for (value, def) in defs.iter_mut().enumerate() {
        if uses.get(value) != Some(&1) {
            *def = None;
        }
    }
    defs
}

/// Turns each run of stores that copy loaded fields from one base to the same offsets of another, together one contiguous span, back into one move.
fn regroup(func: &mut Function, prov: &Provenance, cfg: &Cfg, defs: &[Option<Inst>]) -> bool {
    let mut changed = false;
    for index in 0..func.blocks.len() {
        if !cfg.reachable.get(index).copied().unwrap_or(false) {
            continue;
        }
        let chain = pred_chain(cfg, Block(index as u32));
        while let Some(run) = find(func, prov, &chain, defs) {
            apply(func, Block(index as u32), run);
            changed = true;
        }
    }
    changed
}

fn loaded_store(func: &Function, defs: &[Option<Inst>], inst: Inst) -> Option<(Value, i64, Value, i64, u32, Inst)> {
    let Op::Store(value, dst, off) = func.insts[inst.index()].op else { return None };
    let load = defs.get(value.index()).copied().flatten()?;
    let Op::Load(src, src_off) = func.insts[load.index()].op else { return None };
    Some((dst, i64::from(off), src, i64::from(src_off), func.ty(value).bytes(), load))
}

fn find(func: &Function, prov: &Provenance, chain: &[Block], defs: &[Option<Inst>]) -> Option<Run> {
    let block = *chain.last()?;
    let mut run: Option<Run> = None;
    for (at, inst) in func.blocks[block.index()].insts.iter().enumerate() {
        if let Some((dst, off, src, src_off, bytes, load)) = loaded_store(func, defs, *inst) {
            let piece = Piece { at, off, bytes, load };
            match run.as_mut() {
                Some(open) if open.dst == dst && open.src == src && open.delta == src_off - off => open.pieces.push(piece),
                _ => {
                    if let Some(done) = run.take().filter(|done| valid(func, prov, chain, done)) {
                        return Some(done);
                    }
                    run = Some(Run { dst, src, delta: src_off - off, pieces: vec![piece] });
                }
            }
            continue;
        }
        let op = &func.insts[inst.index()].op;
        if !(op.is_speculable() || matches!(op, Op::Load(..))) {
            if let Some(done) = run.take().filter(|done| valid(func, prov, chain, done)) {
                return Some(done);
            }
        }
    }
    run.filter(|done| valid(func, prov, chain, done))
}

fn span(run: &Run) -> Option<(i64, u32)> {
    let mut pieces = run.pieces.clone();
    pieces.sort_by_key(|piece| piece.off);
    let lo = pieces.first()?.off;
    let mut end = lo;
    for piece in &pieces {
        if piece.off != end {
            return None;
        }
        end += i64::from(piece.bytes);
    }
    u32::try_from(end - lo).ok().map(|bytes| (lo, bytes))
}

fn valid(func: &Function, prov: &Provenance, chain: &[Block], run: &Run) -> bool {
    if run.pieces.len() < MIN_PIECES {
        return false;
    }
    let Some((lo, bytes)) = span(run).filter(|(_, bytes)| (MIN_BYTES..=crate::copy::MOVE_INLINE).contains(bytes)) else { return false };
    let dst = prov.loc(run.dst, lo, bytes);
    let src = prov.loc(run.src, lo + run.delta, bytes);
    if may_alias(dst, src, prov) {
        return false;
    }
    let last = chain.len() - 1;
    let first = run.pieces.first().map_or(0, |piece| piece.at);
    let end = run.pieces.last().map_or(0, |piece| piece.at);
    let reads_dst = func.blocks[chain[last].index()].insts[first..end].iter().any(|inst| match func.insts[inst.index()].op {
        Op::Load(addr, off) => func.insts[inst.index()].result.is_some_and(|value| may_alias(prov.loc(addr, i64::from(off), func.ty(value).bytes()), dst, prov)),
        _ => false,
    });
    !reads_dst && run.pieces.iter().all(|piece| unclobbered(func, prov, chain, piece.load, end, src))
}

fn unclobbered(func: &Function, prov: &Provenance, chain: &[Block], load: Inst, end: usize, src: MemLoc) -> bool {
    let Some((from, pos)) = chain.iter().enumerate().find_map(|(index, block)| func.blocks[block.index()].insts.iter().position(|inst| *inst == load).map(|pos| (index, pos))) else {
        return false;
    };
    let last = chain.len() - 1;
    (from..=last).all(|index| {
        let insts = &func.blocks[chain[index].index()].insts;
        let lo = if index == from { pos + 1 } else { 0 };
        let hi = if index == last { end } else { insts.len() };
        lo >= hi || !insts[lo..hi].iter().any(|inst| clobbers_loc(func, (&func.insts[inst.index()].op, None), src, prov))
    })
}

fn apply(func: &mut Function, block: Block, run: Run) {
    let Some((lo, bytes)) = span(&run) else { return };
    let end = run.pieces.last().map_or(0, |piece| piece.at);
    let mut added = Vec::new();
    let dst = offset(func, &mut added, run.dst, lo);
    let src = offset(func, &mut added, run.src, lo + run.delta);
    let last = func.blocks[block.index()].insts[end];
    func.insts[last.index()].op = Op::Move(dst, src, bytes);
    let stores: Vec<usize> = run.pieces.iter().map(|piece| piece.at).filter(|at| *at != end).collect();
    let insts = &mut func.blocks[block.index()].insts;
    let mut kept: Vec<Inst> = insts.iter().enumerate().filter(|(at, _)| !stores.contains(at)).map(|(_, inst)| *inst).collect();
    let at = kept.iter().position(|inst| *inst == last).unwrap_or(kept.len());
    kept.splice(at..at, added);
    *insts = kept;
}

fn offset(func: &mut Function, added: &mut Vec<Inst>, base: Value, off: i64) -> Value {
    if off == 0 {
        return base;
    }
    let imm = func.value(Type::I64);
    added.push(func.add(Op::Const(off), Some(imm)));
    let sum = func.value(Type::I64);
    added.push(func.add(Op::Binary(Binary::Iadd, base, imm), Some(sum)));
    sum
}
