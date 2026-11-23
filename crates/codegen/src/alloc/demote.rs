use crate::color::{Webs, shareable};
use crate::ir::{Function, Op, Value};
use crate::live::{Elided, Live, for_effective_operands};

use super::Location;
use super::walk::{edge_copies, weighted};

/// Each copy a block edge makes, as its source block's weight, the parameter and the argument, and which blocks take parameters other than from block edges.
fn copies(func: &Function, blocks: &[u64]) -> (Vec<(u64, usize, usize)>, Vec<bool>) {
    let mut fixed = vec![false; func.blocks.len()];
    if let Some(entry) = func.entry {
        fixed[entry.index()] = true;
    }
    for inst in func.blocks.iter().flat_map(|data| &data.insts) {
        let op = &func.insts[inst.index()].op;
        if !matches!(op, Op::Jump(..) | Op::Brif(..)) {
            op.targets().for_each(|target| fixed[target.index()] = true);
        }
    }
    let mut copies = Vec::new();
    edge_copies(func, Some(blocks), |_, weight, _, pairs| copies.extend(pairs.map(|(param, arg)| (weight, param.index(), arg.index()))));
    (copies, fixed)
}

/// The block-weighted count of reads of each value other than block edge copies.
fn reads(func: &Function, blocks: &[u64], eff: Elided<'_>) -> Vec<u64> {
    let mut reads = vec![0_u64; func.types.len()];
    for (weight, data) in weighted(func, blocks) {
        match data.op {
            Op::Jump(..) => {}
            Op::Brif(condition, ..) => reads[condition.index()] += weight,
            _ => for_effective_operands(data, eff, |value| reads[value.index()] += weight),
        }
    }
    reads
}

/// Groups the frame values that block edges copy into one another, then moves to the frame each register block parameter whose copies and reads cost less there.
pub(super) fn webs<R: Copy + PartialEq>(
    func: &Function,
    live: &Live,
    (blocks, partners): (&[u64], &[Vec<Value>]),
    (eff, deferred): (Elided<'_>, &[bool]),
    imm_store: fn(i64) -> u64,
    locations: &mut [Location<R>],
) -> Webs {
    let count = func.types.len();
    let memory = |locations: &[Location<R>], value: usize| shareable(live, locations, deferred, value);
    let joins = |locations: &[Location<R>], webs: &Webs, value: usize, partner: usize| {
        memory(locations, partner) && func.types[partner] == func.types[value] && webs.fits(live, value, partner)
    };
    let mut webs = Webs::new(count);
    for (value, list) in partners.iter().enumerate() {
        for partner in list.iter().map(|partner| partner.index()) {
            if memory(locations, value) && joins(locations, &webs, value, partner) {
                webs.merge(value, partner);
            }
        }
    }
    let (copies, fixed) = copies(func, blocks);
    let reads = reads(func, blocks, eff);
    let mut touching = vec![Vec::new(); count];
    for (at, &(_, param, arg)) in copies.iter().enumerate() {
        if param != arg {
            touching[param].push(at);
            touching[arg].push(at);
        }
    }
    let params = func.blocks.iter().enumerate().filter(|(block, _)| !fixed[*block]).flat_map(|(_, data)| data.params.iter().map(|param| param.index()));
    for value in params {
        let Location::Reg(reg) = locations[value] else { continue };
        if deferred[value] || live.bounds(value).is_none() {
            continue;
        }
        let (mut kept, mut moved, mut joined) = (0_u64, reads[value], Vec::new());
        for &(weight, param, arg) in touching[value].iter().map(|&at| &copies[at]) {
            let partner = if param == value { arg } else { param };
            match locations[partner] {
                Location::Reg(other) => {
                    kept += if other == reg { 0 } else { weight };
                    moved += weight;
                }
                Location::Slot if joins(locations, &webs, value, partner) && joined.iter().all(|&web| webs.fits(live, web, partner)) => {
                    kept += weight;
                    joined.push(partner);
                }
                Location::Imm(imm) => {
                    kept += weight;
                    moved += imm_store(imm) * weight;
                }
                _ => {
                    kept += weight;
                    moved += 2 * weight;
                }
            }
        }
        if moved < kept {
            locations[value] = Location::Slot;
            joined.into_iter().for_each(|partner| webs.merge(value, partner));
        }
    }
    webs
}
