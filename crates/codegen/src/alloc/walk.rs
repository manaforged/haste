use std::iter::Zip;
use std::slice::Iter;

use crate::cfg::Cfg;
use crate::frame::terminator_edges;
use crate::ir::{Block, Function, InstData, Op, Value};
use crate::live::{Live, Sib, for_effective_operands, results};

/// The (parameter, argument) copies one block edge makes.
pub(super) type Copies<'f> = Zip<Iter<'f, Value>, Iter<'f, Value>>;

/// Each instruction in block order, with the weight of its block.
pub(super) fn weighted<'f>(func: &'f Function, blocks: &'f [u64]) -> impl Iterator<Item = (u64, &'f InstData)> + 'f {
    func.blocks.iter().enumerate().flat_map(move |(at, data)| data.insts.iter().map(move |inst| (blocks[at], &func.insts[inst.index()])))
}

/// Calls `visit` for each block edge in block order with its source block, the source's weight (1 without weights), its target and its copies.
pub(super) fn edge_copies<'f>(func: &'f Function, blocks: Option<&[u64]>, mut visit: impl FnMut(Block, u64, Block, Copies<'f>)) {
    for (at, data) in func.blocks.iter().enumerate() {
        let (src, weight) = (Block(at as u32), blocks.map_or(1, |w| w[at]));
        for inst in &data.insts {
            terminator_edges(&func.insts[inst.index()].op, |target, args| visit(src, weight, target, func.blocks[target.index()].params.iter().zip(args)));
        }
    }
}

pub(super) fn weights(func: &Function, blocks: &[u64], sib: &[Option<Sib>], elided: &[bool]) -> Vec<u64> {
    let mut weight = vec![0_u64; func.types.len()];
    for (each, data) in weighted(func, blocks) {
        if data.result.is_some_and(|res| elided[res.index()]) {
            continue;
        }
        for_effective_operands(data, sib, elided, |value| weight[value.index()] += each);
        results(data).for_each(|value| weight[value.index()] += each);
    }
    weight
}

pub(super) fn edge_hints(func: &Function, cfg: Option<&Cfg>, order: &[Block], blocks: Option<&[u64]>) -> (Option<Vec<Vec<Value>>>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let rank = cfg.is_none().then(|| Cfg::ranks(order, func.blocks.len()));
    let is_back = |src: Block, target: Block| match (cfg, rank.as_deref()) {
        (Some(cfg), _) => cfg.reachable[src.index()] && cfg.dominates(target, src),
        (None, Some(rank)) => rank[src.index()] != usize::MAX && rank[target.index()] <= rank[src.index()],
        (None, None) => target.index() <= src.index(),
    };
    let mut incoming = vec![0_u32; func.types.len()];
    if let Some(entry) = func.entry {
        func.blocks[entry.index()].params.iter().for_each(|p| incoming[p.index()] += 1);
    }
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::TryCall { handler, .. } = func.insts[inst.index()].op {
            func.blocks[handler.index()].params.iter().for_each(|p| incoming[p.index()] += 1);
        }
    }
    let mut copied = vec![0_u32; func.types.len()];
    let mut edges: Vec<(bool, u64, usize, usize)> = Vec::new();
    edge_copies(func, blocks, |src, weight, target, copies| {
        let back = is_back(src, target);
        for (&param, &arg) in copies {
            incoming[param.index()] += 1;
            copied[arg.index()] += 1;
            if (cfg.is_some() || back) && param != arg {
                edges.push((back, weight, param.index(), arg.index()));
            }
        }
    });
    edges.sort_unstable_by_key(|&(back, weight, p, a)| (std::cmp::Reverse((back, weight)), p, a));
    let uses = func.use_counts();
    let mut back_args = Vec::new();
    let mut edge_args = Vec::new();
    let mut out = cfg.is_some().then(|| vec![Vec::new(); func.types.len()]);
    for (back, _, p, a) in edges {
        if (back || incoming[p] == 1 || uses[a] == copied[a]) && !back_args.contains(&(p, a)) { back_args.push((p, a)); }
        if cfg.is_some() && !back && !edge_args.contains(&(p, a)) { edge_args.push((p, a)); }
        if let Some(partners) = out.as_mut() {
            let (pv, av) = (Value(p as u32), Value(a as u32));
            if !partners[p].contains(&av) { partners[p].push(av); }
            if !partners[a].contains(&pv) { partners[a].push(pv); }
        }
    }
    (out, back_args, edge_args)
}

/// Gives each split parameter the weight that makes its density the given density of the value it carries before the split, so it fills only registers that value could have had.
pub(super) fn inherit(mut weights: Vec<u64>, live: &Live, split: &[(Value, u64)]) -> Vec<u64> {
    for &(param, density) in split {
        let weight = (u128::from(density) * u128::from(live.total_len(param.index()).max(4))) >> DENSITY_SHIFT;
        weights[param.index()] = u64::try_from(weight).unwrap_or(u64::MAX);
    }
    weights
}

const DENSITY_SHIFT: u32 = 16;

/// Each value's weight per unit of its live length, scaled by `2^DENSITY_SHIFT`.
pub(super) fn densities(weights: &[u64], live: &Live) -> Vec<u64> {
    weights.iter().enumerate().map(|(v, &w)| u64::try_from((u128::from(w) << DENSITY_SHIFT) / u128::from(live.total_len(v).max(4))).unwrap_or(u64::MAX)).collect()
}

/// Calls `visit` with the source, target, parameter and argument of every block-edge copy whose parameter and argument differ.
pub(crate) fn for_edge_copies(func: &Function, mut visit: impl FnMut(Block, Block, Value, Value)) {
    edge_copies(func, None, |src, _, target, copies| copies.filter(|(param, arg)| param != arg).for_each(|(&param, &arg)| visit(src, target, param, arg)));
}
