use crate::fold::{constant, holds};
use crate::ir::{Binary, Convert, FloatCC, Function, Op, StackSlot, Type, Value};

use super::{CallTarget, FOLD_LIMIT, Location, Uses};

pub(super) fn consts<R>(func: &Function, absolute_calls: bool, locations: &mut [Location<R>]) {
    for data in &func.insts {
        let Some(result) = data.result else { continue };
        match data.op {
            Op::Const(imm) if func.ty(result).is_small_int() => {
                locations[result.index()] = Location::Imm(imm);
            }
            Op::SlotAddr(sym) if absolute_calls => {
                locations[result.index()] = Location::SlotAddr(sym);
            }
            Op::Symbol(sym) if absolute_calls => {
                locations[result.index()] = Location::Symbol(sym);
            }
            _ => {}
        }
    }
}

fn fold<R>(slot: StackSlot, offset: i32, imm: i64) -> Option<Location<R>> {
    let sum = i64::from(offset).checked_add(imm)?;
    (-FOLD_LIMIT..=FOLD_LIMIT)
        .contains(&sum)
        .then_some(Location::Frame(slot, sum as i32))
}

fn static_location<R: Copy>(
    func: &Function,
    locations: &[Location<R>],
    op: &Op,
) -> Option<Location<R>> {
    match *op {
        Op::StackAddr(slot, offset) => Some(Location::Frame(slot, offset)),
        Op::Icmp(cc, x, y) => match (locations[x.index()], locations[y.index()]) {
            (Location::Imm(a), Location::Imm(b)) if func.ty(x).is_small_int() => {
                Some(Location::Imm(i64::from(holds(cc, func.ty(x), a, b))))
            }
            _ => None,
        },
        Op::Binary(op, x, y) => match (locations[x.index()], locations[y.index()]) {
            (Location::Frame(slot, offset), Location::Imm(imm))
            | (Location::Imm(imm), Location::Frame(slot, offset))
                if op == Binary::Iadd =>
            {
                fold(slot, offset, imm)
            }
            (Location::Imm(a), Location::Imm(b)) if func.ty(x).is_small_int() => {
                constant(op, a, b).map(Location::Imm)
            }
            _ => None,
        },
        Op::Convert(Convert::Uextend | Convert::Ireduce, x) => match locations[x.index()] {
            Location::Imm(imm) if func.ty(x).is_small_int() => {
                Some(Location::Imm(crate::fold::extend(imm, func.ty(x), false)))
            }
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn statics<R: Copy>(func: &Function, locations: &mut [Location<R>]) {
    let mut changed = true;
    while changed {
        changed = false;
        for data in &func.insts {
            let Some(result) = data.result else { continue };
            if !matches!(locations[result.index()], Location::Slot)
                || !func.ty(result).is_small_int()
            {
                continue;
            }
            if let Some(location) = static_location(func, locations, &data.op).filter(|location| {
                func.ty(result) == Type::I64 || matches!(location, Location::Imm(_))
            }) {
                locations[result.index()] = location;
                changed = true;
            }
        }
    }
}

pub(super) fn fuses(
    func: &Function,
    census: &Uses,
    value: usize,
    float_branches: &[FloatCC],
    two_address: bool,
) -> bool {
    let (Some((block, at)), Some((use_block, last))) = (census.def[value], census.last_use[value])
    else {
        return false;
    };
    if block != use_block || census.uses[value] != 1 || at >= last {
        return false;
    }
    let insts = &func.blocks[block].insts;
    let flags = match &func.insts[insts[at].index()].op {
        Op::Icmp(_, a, _) | Op::Fits(_, _, a, _) => !func.ty(*a).is_vector(),
        Op::Fcmp(cc, a, _) => !func.ty(*a).is_vector() && float_branches.contains(cc),
        _ => false,
    };
    let pure = insts[at + 1..last].iter().all(|&mid| {
        let d = &func.insts[mid.index()];
        !d.result.is_some_and(|r| func.ty(r).is_vector())
            && matches!(d.op, Op::Const(_) | Op::StackAddr(..) | Op::FramePointer)
            || (two_address
                && !d.result.is_some_and(|r| func.ty(r).is_vector())
                && matches!(d.op, Op::Symbol(_) | Op::SlotAddr(_)))
    });
    flags
        && pure
        && matches!(
            func.insts[insts[last].index()].op,
            Op::Brif(condition, ..) | Op::Trapz(condition) if condition.index() == value
        )
}

pub(super) fn pairs<R>(
    func: &Function,
    locations: &[Location<R>],
    elided: &[bool],
) -> (Vec<Option<Value>>, Vec<bool>) {
    let (mut pairs, mut deferred) = (vec![None; func.types.len()], vec![false; func.types.len()]);
    let dynamic = |value: Value| {
        !elided[value.index()]
            && matches!(locations[value.index()], Location::Slot | Location::Reg(_))
    };
    for block in &func.blocks {
        for window in block.insts.windows(2) {
            let (first, second) = (
                &func.insts[window[0].index()],
                &func.insts[window[1].index()],
            );
            if let (Op::Binary(op, x, y), Op::Fits(check, _, a, b), Some(sum), Some(fits)) =
                (&first.op, &second.op, first.result, second.result)
                && op == check
                && (x, y) == (a, b)
                && func.ty(sum).is_small_int()
                && dynamic(sum)
                && dynamic(fits)
            {
                pairs[fits.index()] = Some(sum);
                deferred[sum.index()] = true;
            }
        }
    }
    (pairs, deferred)
}

pub(super) fn through<R>(
    func: &Function,
    locations: &[Location<R>],
    census: &Uses,
    elided: &mut [bool],
    runtime: bool,
) -> Vec<Option<CallTarget>> {
    let mut through = vec![None; func.types.len()];
    let mut loads = vec![None; func.types.len()];
    for data in &func.insts {
        let Some(result) = data.result else { continue };
        loads[result.index()] = match data.op {
            Op::Load(addr, 0) => match locations[addr.index()] {
                Location::Imm(at) => i32::try_from(at)
                    .ok()
                    .filter(|at| *at > 0)
                    .map(CallTarget::Abs),
                Location::SlotAddr(sym) => Some(CallTarget::Slot(sym)),
                _ => None,
            }
            .map(|target| (target, Some(addr))),
            Op::Runtime(rt) if runtime => Some((CallTarget::Runtime(rt), None)),
            Op::Symbol(sym)
                if runtime && matches!(locations[result.index()], Location::Symbol(_)) =>
            {
                Some((CallTarget::Symbol(sym), None))
            }
            _ => None,
        };
    }
    let mut called = vec![0u32; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } =
            &func.insts[inst.index()].op
            && !args.contains(callee)
        {
            called[callee.index()] += 1;
        }
    }
    let mut left = census.uses.clone();
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } =
            &func.insts[inst.index()].op
            && !args.contains(callee)
            && let Some((target, addr)) = loads[callee.index()]
            && (census.uses[callee.index()] == 1
                || (addr.is_none() && census.uses[callee.index()] == called[callee.index()]))
        {
            through[callee.index()] = Some(target);
            elided[callee.index()] = true;
            if let Some(addr) = addr {
                left[addr.index()] = left[addr.index()].saturating_sub(1);
                elided[addr.index()] |= left[addr.index()] == 0;
            }
        }
    }
    through
}

pub(super) fn const_pairs(func: &Function, census: &Uses, elided: &mut [bool]) {
    let mut kept = vec![false; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let op = &func.insts[inst.index()].op;
        if !crate::wide::reads_halves(func, op) {
            op.for_operands(|value| kept[value.index()] = true);
        }
    }
    let constant = |value: Value| {
        census.def_inst[value.index()]
            .is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::Const(_)))
    };
    for (value, def) in census.def_inst.iter().enumerate() {
        if let Some(inst) = def
            && let Op::Binary(Binary::Iconcat, lo, hi) = func.insts[inst.index()].op
            && !kept[value]
            && constant(lo)
            && constant(hi)
        {
            elided[value] = true;
        }
    }
}
