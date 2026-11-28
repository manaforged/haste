use crate::cfg::{Cfg, Loops};
use crate::frame::{Abi, Loc};
use crate::ir::{ArgumentPurpose, Block, Function, Inst, Op, Signature, Value};
use crate::live::Live;
use crate::park::{Class, ParkReg};

use super::{Location, Regs};

pub(crate) const BASE_WEIGHT: u64 = 8;
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
        let has_call_free_latch = !calls[each.head.index()]
            && each.latches.iter().any(|&latch| {
                let mut block = latch;
                loop {
                    if calls[block.index()] {
                        return false;
                    }
                    match cfg.idom[block.index()] {
                        Some(parent) if block != each.head && parent != block => block = parent,
                        _ => return true,
                    }
                }
            });
        let mut dom_calls = vec![false; count];
        if has_call_free_latch {
            for &block in cfg.rpo.iter().rev() {
                let at = block.index();
                if !each.body[at] {
                    continue;
                }
                if calls[at] {
                    dom_calls[at] = true;
                }
                if dom_calls[at]
                    && block != each.head
                    && let Some(parent) = cfg.idom[at]
                    && each.body[parent.index()]
                {
                    dom_calls[parent.index()] = true;
                }
            }
        }
        let mut guarded = vec![false; count];
        for &block in &cfg.rpo {
            let at = block.index();
            let under = cfg.idom[at].is_some_and(|parent| guarded[parent.index()]);
            let side_call = spine[at] != each.latches.len() && (calls[at] || dom_calls[at]);
            if each.body[at] && (under || side_call) {
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

#[cfg(test)]
mod tests {
    use super::{BASE_WEIGHT, LOOP_WEIGHT, SLOW_SHARE, block_weights};
    use crate::FunctionBuilder;
    use crate::ir::{AbiParam, CallConv, Function, IntCC, Signature, Type};

    #[test]
    fn call_bearing_side_branch_is_marked_slow_when_loop_has_call_free_latch() {
        let mut sig = Signature::new(CallConv::Native);
        sig.params = vec![AbiParam::new(Type::I64), AbiParam::new(Type::I64)];
        let mut callee_sig = Signature::new(CallConv::Native);
        callee_sig.params = vec![AbiParam::new(Type::I64)];
        let mut func = Function::new(sig);
        let (head, fast_latch, side_entry, side_call, side_join) = {
            let mut b = FunctionBuilder::new(&mut func);
            let sig_ref = b.import_signature(callee_sig);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            let head = b.create_block();
            let fast_latch = b.create_block();
            let side_entry = b.create_block();
            let side_call = b.create_block();
            let side_join = b.create_block();
            let exit = b.create_block();
            b.switch_to_block(entry);
            let n = b.block_params(entry)[0];
            let callee = b.block_params(entry)[1];
            let zero = b.ins().iconst(Type::I64, 0);
            b.ins().jump(head, &[zero]);

            b.switch_to_block(head);
            let i = b.append_block_param(head, Type::I64);
            let flag = b.ins().icmp(IntCC::Equal, i, zero);
            b.ins().brif(flag, side_entry, &[], fast_latch, &[]);

            b.switch_to_block(fast_latch);
            let one = b.ins().iconst(Type::I64, 1);
            let next_fast = b.ins().iadd(i, one);
            let cont_fast = b.ins().icmp(IntCC::UnsignedLessThan, next_fast, n);
            b.ins().brif(cont_fast, head, &[next_fast], exit, &[]);

            b.switch_to_block(side_entry);
            let need_grow = b.ins().icmp(IntCC::UnsignedGreaterThan, n, one);
            b.ins().brif(need_grow, side_call, &[], side_join, &[]);

            b.switch_to_block(side_call);
            b.ins().call_indirect(sig_ref, callee, &[i]);
            b.ins().jump(side_join, &[]);

            b.switch_to_block(side_join);
            let next_side = b.ins().iadd(i, one);
            let cont_side = b.ins().icmp(IntCC::UnsignedLessThan, next_side, n);
            b.ins().brif(cont_side, head, &[next_side], exit, &[]);

            b.switch_to_block(exit);
            b.ins().return_(&[]);
            (head, fast_latch, side_entry, side_call, side_join)
        };
        let cfg = func.cfg();
        let loops = cfg.loops();
        let weights = block_weights(&func, &cfg, loops);
        let hot = BASE_WEIGHT * LOOP_WEIGHT;
        let slow = hot / SLOW_SHARE;
        assert_eq!(weights[head.index()], hot);
        assert_eq!(weights[fast_latch.index()], hot);
        assert_eq!(weights[side_entry.index()], slow);
        assert_eq!(weights[side_call.index()], slow);
        assert_eq!(weights[side_join.index()], slow);
    }
}
