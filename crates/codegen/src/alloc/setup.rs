use crate::cfg::{Cfg, Loops};
use crate::frame::{Abi, Loc};
use crate::ir::{ArgumentPurpose, Block, Function, Inst, Op, Signature, Value};
use crate::live::Live;
use crate::park::{Class, ParkReg};

use super::{Location, Regs};

const BASE_WEIGHT: u64 = 8;
const LOOP_WEIGHT: u64 = 16;
const MAX_DEPTH: u32 = 4;
const SLOW_SHARE: u64 = 16;

fn preserved_clobber_ranges(preserved_calls: &[u32]) -> Vec<crate::live::Range> {
    preserved_calls
        .iter()
        .map(|&at| crate::live::Range {
            start: at - 1,
            end: at - 1,
        })
        .collect()
}

pub(super) fn int_class<'a, R: Copy + PartialEq>(
    func: &Function,
    regs: &'a Regs<R>,
    (order, copied, elided): (&[Block], &[bool], &[bool]),
    params: &[Loc<R>],
    passed: Vec<Option<(R, bool)>>,
    preserved_calls: &[u32],
) -> Class<'a, R> {
    let dict = func
        .signature
        .params
        .iter()
        .any(|param| param.purpose == ArgumentPurpose::Dict);
    let leaf = regs
        .leaf
        .iter()
        .filter(|reg| !(dict && regs.dict == Some(**reg)))
        .copied()
        .collect();
    let mut clobbers = crate::park::scratch_clobbers(
        func,
        (regs.wide_scratch, regs.op_scratch, regs.scratch_mask),
        (order, copied, elided),
    );
    if !preserved_calls.is_empty() && !regs.preserved_clobbers.is_empty() {
        clobbers.push((
            regs.preserved_clobbers,
            preserved_clobber_ranges(preserved_calls),
        ));
    }
    let arrivals = params
        .iter()
        .map(|loc| match *loc {
            Loc::Int(reg) => vec![reg],
            Loc::Pair(low, high) => vec![low, high],
            _ => Vec::new(),
        })
        .collect();
    Class {
        saved: regs.saved,
        leaf,
        arrivals,
        passed,
        clobbers,
    }
}

pub(super) fn float_class<R>(
    regs: &Regs<R>,
    params: &[Loc<R>],
    passed: Vec<Option<(u8, bool)>>,
    preserved_calls: &[u32],
) -> Class<'static, u8> {
    let arrivals = params
        .iter()
        .map(|loc| match *loc {
            Loc::Float(reg) => vec![reg],
            _ => Vec::new(),
        })
        .collect();
    let clobbers = if preserved_calls.is_empty() || regs.float.is_empty() {
        Vec::new()
    } else {
        vec![(regs.float, preserved_clobber_ranges(preserved_calls))]
    };
    Class {
        saved: regs.saved_float,
        leaf: regs.float.to_vec(),
        arrivals,
        passed,
        clobbers,
    }
}

fn slow_blocks(func: &Function, cfg: &Cfg, loops: &Loops) -> (Vec<bool>, Vec<bool>) {
    let count = func.blocks.len();
    let calls: Vec<bool> = func
        .blocks
        .iter()
        .map(|data| {
            data.insts.iter().any(|inst| {
                matches!(
                    func.insts[inst.index()].op,
                    Op::Call { .. } | Op::TryCall { .. } | Op::Copy(..)
                )
            })
        })
        .collect();
    let (cold, mut slow) = crate::live::cold_and_detours(func, cfg, loops);
    for each in &loops.list {
        let mut spine = vec![0_usize; count];
        for &latch in &each.latches {
            let mut block = latch;
            loop {
                spine[block.index()] += 1;
                match cfg.idom[block.index()] {
                    Some(parent) if block != each.head && parent != block => block = parent,
                    _ => break,
                }
            }
        }
        let mut guarded = vec![false; count];
        for &block in &cfg.rpo {
            let at = block.index();
            let under = cfg.idom[at].is_some_and(|parent| guarded[parent.index()]);
            if each.body[at] && (under || (calls[at] && spine[at] != each.latches.len())) {
                guarded[at] = true;
                slow[at] = true;
            }
        }
    }
    (cold, slow)
}

pub(super) fn block_weights(func: &Function, cfg: &Cfg, loops: &Loops) -> Vec<u64> {
    let (cold, slow) = slow_blocks(func, cfg, loops);
    loops
        .depths
        .iter()
        .enumerate()
        .map(|(i, &depth)| {
            if cold[i] {
                1
            } else {
                (BASE_WEIGHT * LOOP_WEIGHT.pow(depth.min(MAX_DEPTH))
                    / if slow[i] { SLOW_SHARE } else { 1 })
                .max(1)
            }
        })
        .collect()
}

fn passed_own<R: Copy + PartialEq>(
    func: &Function,
    inst: Inst,
    value: Value,
    reg: ParkReg<R>,
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> bool {
    let Op::Call { sig, args, .. } = &func.insts[inst.index()].op else {
        return false;
    };
    let Ok(abi) = classify(&func.sigs[sig.index()]) else {
        return false;
    };
    args.iter().zip(&abi.params).any(|(arg, loc)| {
        *arg == value
            && match (reg, *loc) {
                (ParkReg::Int(r), Loc::Int(at)) => r == at,
                (ParkReg::Float(v), Loc::Float(at)) => v == at,
                _ => false,
            }
    })
}

pub(super) fn parked<R: Copy + PartialEq>(
    func: &Function,
    live: &Live,
    locations: &[Location<R>],
    (saved_int, saved_float): (&[R], &[u8]),
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Vec<(Inst, Vec<(Value, ParkReg<R>, bool)>)> {
    let mut at_call: Vec<Vec<(Value, ParkReg<R>, bool)>> = vec![Vec::new(); live.calls.len()];
    for (index, location) in locations.iter().enumerate() {
        let park_reg = match *location {
            Location::Reg(r) if !saved_int.contains(&r) => ParkReg::Int(r),
            Location::Float(v) if !saved_float.contains(&v) => ParkReg::Float(v),
            _ => continue,
        };
        let ranges = live.ranges_of(index);
        if ranges.is_empty() {
            continue;
        }
        let value = Value(index as u32);
        for r in ranges {
            let first = live.calls.partition_point(|at| *at <= r.start);
            let last = live.calls.partition_point(|at| *at <= r.end);
            for call in first..last {
                let at = live.calls[call];
                if at < r.end || !passed_own(func, live.sites[call].0, value, park_reg, classify) {
                    at_call[call].push((value, park_reg, at < r.end));
                }
            }
        }
    }
    for entries in &mut at_call {
        entries.sort_unstable_by_key(|&(v, r, _)| {
            (
                matches!(r, ParkReg::Float(_)),
                func.ty(v).bytes(),
                v.index(),
            )
        });
    }
    let mut out: Vec<(Inst, Vec<(Value, ParkReg<R>, bool)>)> = live
        .sites
        .iter()
        .map(|site| site.0)
        .zip(at_call)
        .filter(|(_, values)| !values.is_empty())
        .collect();
    out.sort_unstable_by_key(|(inst, _)| inst.index());
    out
}
