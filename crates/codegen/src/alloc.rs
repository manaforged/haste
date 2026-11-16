mod demote;
mod hints;
mod walk;

use crate::cfg::{Cfg, Loops};
use crate::fold::{constant, fits_imm32, holds};
use crate::frame::{Abi, Loc};
use crate::ir::{ArgumentPurpose, Binary, Block, Convert, FloatCC, Function, Inst, InstData, Op, Signature, StackSlot, Type, Value};
use crate::live::{Live, Sib, SibBase, live, results};
use crate::park::{Class, Hints, ParkReg, allocate};

pub(crate) struct Regs<R: 'static> {
    pub saved: &'static [R],
    pub saved_float: &'static [u8],
    pub leaf: &'static [R],
    pub wide_scratch: &'static [R],
    pub op_scratch: &'static [R],
    pub uses_scratch: fn(&Function, &Op) -> bool,
    pub dict: Option<R>,
    pub float: &'static [u8],
    pub float_branches: &'static [FloatCC],
    pub vectors: bool,
    pub absolute_calls: bool,
    pub two_address: bool,
    pub park: bool,
    pub fold_sib: bool,
    /// The backend zero-extends 8- and 16-bit call results at the call, and 8- and 16-bit narrowing copies between two registers.
    pub call_zext: bool,
    /// The instructions that store a constant into a frame home.
    pub imm_store: fn(i64) -> u64,
}

const FOLD_LIMIT: i64 = 1 << 24;

/// Reports whether `data` reads or makes a 128-bit integer.
pub(crate) fn touches_i128(func: &Function, data: &InstData) -> bool {
    let mut wide = data.result.is_some_and(|r| func.ty(r) == Type::I128);
    data.op.for_operands(|v| wide |= func.ty(v) == Type::I128);
    wide
}

impl<R: PartialEq + 'static> Regs<R> {
    /// Reports whether emitting `data` may overwrite `reg` as a wide or op scratch register.
    pub(crate) fn scratches(&self, func: &Function, data: &InstData, reg: R) -> bool {
        let wide = touches_i128(func, data);
        (wide && self.wide_scratch.contains(&reg)) || ((wide || (self.uses_scratch)(func, &data.op)) && self.op_scratch.contains(&reg))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Location<R> {
    Slot,
    Imm(i64),
    SlotAddr(u32),
    Symbol(u32),
    Reg(R),
    Float(u8),
    Frame(StackSlot, i32),
}

impl<R> Location<R> {
    pub(crate) fn is_static(&self) -> bool {
        matches!(self, Self::Imm(_) | Self::SlotAddr(_) | Self::Symbol(_) | Self::Frame(..))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CallTarget {
    Abs(i32),
    Slot(u32),
}

pub(crate) struct Plan<R> {
    pub locations: Vec<Location<R>>,
    pub fused: Vec<bool>,
    pub used: Vec<R>,
    pub used_floats: Vec<u8>,
    pub pairs: Vec<Option<Value>>,
    pub deferred: Vec<bool>,
    pub through: Vec<Option<CallTarget>>,
    pub shared: Vec<Option<u32>>,
    pub parked: Vec<(Inst, Vec<(Value, ParkReg<R>, bool)>)>,
    pub homed: Vec<bool>,
    pub sib: Vec<Option<Sib>>,
    pub elided: Vec<bool>,
    pub zero_ext: Vec<bool>,
}

fn registered<R>(regs: &Regs<R>, ty: Type) -> bool {
    ty.is_small_int() || ((ty.is_float() || (regs.vectors && ty.is_vector())) && !regs.float.is_empty())
}

type Position = Option<(usize, usize)>;

struct Uses {
    uses: Vec<u32>,
    last_use: Vec<Position>,
    def: Vec<Position>,
    def_inst: Vec<Option<Inst>>,
}

impl Uses {
    fn visit(&mut self, inst: Inst, data: &crate::ir::InstData, at: (usize, usize)) {
        data.op.for_operands(|value| {
            self.uses[value.index()] += 1;
            self.last_use[value.index()] = Some(at);
        });
        for result in results(data) {
            self.def[result.index()] = Some(at);
            self.def_inst[result.index()] = Some(inst);
        }
    }
}

fn census(func: &Function) -> Uses {
    let count = func.types.len();
    let mut out = Uses { uses: vec![0; count], last_use: vec![None; count], def: vec![None; count], def_inst: vec![None; count] };
    for (b, block) in func.blocks.iter().enumerate() {
        for (i, &inst) in block.insts.iter().enumerate() {
            out.visit(inst, &func.insts[inst.index()], (b, i));
        }
    }
    out
}

fn consts<R>(func: &Function, absolute_calls: bool, locations: &mut [Location<R>]) {
    for data in &func.insts {
        let Some(result) = data.result else { continue };
        match data.op {
            Op::Const(imm) if func.ty(result).is_small_int() => locations[result.index()] = Location::Imm(imm),
            Op::SlotAddr(sym) if absolute_calls => locations[result.index()] = Location::SlotAddr(sym),
            Op::Symbol(sym) if absolute_calls => locations[result.index()] = Location::Symbol(sym),
            _ => {}
        }
    }
}

fn fold<R>(slot: StackSlot, offset: i32, imm: i64) -> Option<Location<R>> {
    let sum = i64::from(offset).checked_add(imm)?;
    (-FOLD_LIMIT..=FOLD_LIMIT).contains(&sum).then_some(Location::Frame(slot, sum as i32))
}

fn static_location<R: Copy>(func: &Function, locations: &[Location<R>], op: &Op) -> Option<Location<R>> {
    match *op {
        Op::StackAddr(slot, offset) => Some(Location::Frame(slot, offset)),
        Op::Icmp(cc, x, y) => match (locations[x.index()], locations[y.index()]) {
            (Location::Imm(a), Location::Imm(b)) if func.ty(x).is_small_int() => Some(Location::Imm(i64::from(holds(cc, func.ty(x), a, b)))),
            _ => None,
        },
        Op::Binary(op, x, y) => match (locations[x.index()], locations[y.index()]) {
            (Location::Frame(slot, offset), Location::Imm(imm)) | (Location::Imm(imm), Location::Frame(slot, offset)) if op == Binary::Iadd => fold(slot, offset, imm),
            (Location::Imm(a), Location::Imm(b)) if func.ty(x).is_small_int() => constant(op, a, b).map(Location::Imm),
            _ => None,
        },
        Op::Convert(Convert::Uextend | Convert::Ireduce, x) => match locations[x.index()] {
            Location::Imm(imm) if func.ty(x).is_small_int() => Some(Location::Imm(crate::fold::extend(imm, func.ty(x), false))),
            _ => None,
        },
        _ => None,
    }
}

fn statics<R: Copy>(func: &Function, locations: &mut [Location<R>]) {
    let mut changed = true;
    while changed {
        changed = false;
        for data in &func.insts {
            let Some(result) = data.result else { continue };
            if !matches!(locations[result.index()], Location::Slot) || !func.ty(result).is_small_int() {
                continue;
            }
            if let Some(location) = static_location(func, locations, &data.op).filter(|location| func.ty(result) == Type::I64 || matches!(location, Location::Imm(_))) {
                locations[result.index()] = location;
                changed = true;
            }
        }
    }
}

fn sib_addrs<R>(func: &Function, locations: &[Location<R>], census: &Uses, zero_ext: &[bool], two_address: bool) -> (Vec<Option<Sib>>, Vec<bool>) {
    let count = func.types.len();
    let (mut sib, mut elided) = (vec![None; count], vec![false; count]);
    let slot_i64 = |v: Value| func.ty(v) == Type::I64 && matches!(locations[v.index()], Location::Slot);
    let single_i64 = |v: Value, block: usize| slot_i64(v) && census.uses[v.index()] == 1 && census.def[v.index()].is_some_and(|(b, _)| b == block);
    let unwrap_idx = |v: Value, block: usize| -> Option<(Value, Option<Value>)> {
        let inner = single_i64(v, block)
            .then(|| census.def_inst[v.index()])
            .flatten()
            .and_then(|def| match func.insts[def.index()].op {
                Op::Convert(Convert::Uextend, inner) if func.ty(inner) == Type::I32 && zero_ext[inner.index()] && matches!(locations[inner.index()], Location::Slot) => Some(inner),
                _ => None,
            });
        inner.map(|i| (i, Some(v))).or_else(|| slot_i64(v).then_some((v, None)))
    };
    let parse_base = |v: Value| match locations[v.index()] {
        Location::Frame(slot, off) => Some((SibBase::Frame(slot), off)),
        Location::Slot if func.ty(v) == Type::I64 => Some((SibBase::Value(v), 0)),
        _ => None,
    };
    let parse_scaled = |v: Value, block: usize| -> Option<(Value, u8, Value, Option<Value>)> {
        let def = single_i64(v, block).then(|| census.def_inst[v.index()])??;
        match func.insts[def.index()].op {
            Op::Binary(Binary::Ishl, idx, sh) => {
                let Location::Imm(s @ 1..=3) = locations[sh.index()] else { return None };
                unwrap_idx(idx, block).map(|(b, e)| (b, s as u8, v, e))
            }
            Op::Binary(Binary::Imul, a, b) => [(a, b), (b, a)].into_iter().find_map(|(idx, sc)| {
                let Location::Imm(m @ (2 | 4 | 8)) = locations[sc.index()] else { return None };
                unwrap_idx(idx, block).map(|(b, e)| (b, m.trailing_zeros() as u8, v, e))
            }),
            _ => None,
        }
    };
    for (block, data) in func.blocks.iter().enumerate() {
        for &inst in &data.insts {
            let addr = match func.insts[inst.index()].op {
                Op::Load(addr, _) => addr,
                Op::Store(val, addr, _) if val != addr => addr,
                _ => continue,
            };
            if sib[addr.index()].is_some() || !slot_i64(addr) || census.uses[addr.index()] == 0 || !census.def[addr.index()].is_some_and(|(b, _)| b == block) {
                continue;
            }
            let Some(def_inst) = census.def_inst[addr.index()] else { continue };
            let Op::Binary(Binary::Iadd, mut left, mut right) = func.insts[def_inst.index()].op else { continue };
            let (mut outer_disp, mut peeled_add) = (0_i64, None);
            for (inner, imm_v) in [(left, right), (right, left)] {
                if let Location::Imm(imm) = locations[imm_v.index()]
                    && (-FOLD_LIMIT..=FOLD_LIMIT).contains(&imm)
                    && single_i64(inner, block)
                    && let Some(inner_inst) = census.def_inst[inner.index()]
                    && let Op::Binary(Binary::Iadd, l2, r2) = func.insts[inner_inst.index()].op
                {
                    (outer_disp, peeled_add, left, right) = (imm, Some(inner), l2, r2);
                    break;
                }
            }
            let matched = [(left, right), (right, left)]
                .into_iter()
                .find_map(|(b, i)| parse_base(b).zip(parse_scaled(i, block)).map(|((base, bd), (idx, sh, sv, ev))| (base, bd, idx, sh, Some(sv), ev)))
                .or_else(|| [(left, right), (right, left)].into_iter().find_map(|(b, i)| parse_base(b).zip(unwrap_idx(i, block)).map(|((base, bd), (idx, ev))| (base, bd, idx, 0, None, ev))));
            let Some((base, base_disp, index, shift, scaled_val, ext_val)) = matched else { continue };
            let total = outer_disp + i64::from(base_disp);
            if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&total) {
                continue;
            }
            let (disp, multi, mut uses) = (total as i32, census.uses[addr.index()] > 1, 0_u32);
            let ok = data.insts.iter().all(|&u| {
                let (off, bytes) = match (&func.insts[u.index()].op, func.insts[u.index()].result) {
                    (&Op::Load(a, off), Some(r)) if a == addr => (off, func.ty(r).bytes()),
                    (&Op::Store(val, a, off), _) if a == addr && val != addr => (off, func.ty(val).bytes()),
                    _ => return true,
                };
                uses += 1;
                let fits = match base {
                    SibBase::Frame(_) => (-FOLD_LIMIT..=FOLD_LIMIT).contains(&(total + i64::from(off))),
                    SibBase::Value(_) => disp.checked_add(off).and_then(|d| d.checked_add(16)).is_some(),
                };
                fits && (two_address || !multi || (total + i64::from(off) == 0 && bytes <= 8 && (shift == 0 || bytes == (1 << shift)) && !matches!(base, SibBase::Frame(_))))
            });
            if !ok || uses != census.uses[addr.index()] {
                continue;
            }
            sib[addr.index()] = Some(Sib { base, index, shift, disp });
            elided[addr.index()] = true;
            [peeled_add, scaled_val, ext_val].into_iter().flatten().for_each(|e| elided[e.index()] = true);
        }
    }
    (sib, elided)
}

fn fuses(func: &Function, census: &Uses, value: usize, float_branches: &[FloatCC], two_address: bool) -> bool {
    let (Some((block, at)), Some((use_block, last))) = (census.def[value], census.last_use[value]) else { return false };
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
            || (two_address && !d.result.is_some_and(|r| func.ty(r).is_vector()) && matches!(d.op, Op::Symbol(_) | Op::SlotAddr(_)))
    });
    flags && pure && matches!(func.insts[insts[last].index()].op, Op::Brif(condition, ..) | Op::Trapz(condition) if condition.index() == value)
}

fn pairs<R>(func: &Function, locations: &[Location<R>], elided: &[bool]) -> (Vec<Option<Value>>, Vec<bool>) {
    let (mut pairs, mut deferred) = (vec![None; func.types.len()], vec![false; func.types.len()]);
    let dynamic = |value: Value| !elided[value.index()] && matches!(locations[value.index()], Location::Slot | Location::Reg(_));
    for block in &func.blocks {
        for window in block.insts.windows(2) {
            let (first, second) = (&func.insts[window[0].index()], &func.insts[window[1].index()]);
            if let (Op::Binary(op, x, y), Op::Fits(check, _, a, b), Some(sum), Some(fits)) = (&first.op, &second.op, first.result, second.result)
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

fn through<R>(func: &Function, locations: &[Location<R>], census: &Uses, elided: &mut [bool]) -> Vec<Option<CallTarget>> {
    let mut through = vec![None; func.types.len()];
    let mut loads = vec![None; func.types.len()];
    for data in &func.insts {
        if let (Op::Load(addr, 0), Some(result)) = (&data.op, data.result) {
            loads[result.index()] = match locations[addr.index()] {
                Location::Imm(at) => i32::try_from(at).ok().filter(|at| *at > 0).map(CallTarget::Abs),
                Location::SlotAddr(sym) => Some(CallTarget::Slot(sym)),
                _ => None,
            }
            .map(|target| (target, *addr));
        }
    }
    let mut left = census.uses.clone();
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } = &func.insts[inst.index()].op
            && census.uses[callee.index()] == 1
            && !args.contains(callee)
            && let Some((target, addr)) = loads[callee.index()]
        {
            through[callee.index()] = Some(target);
            elided[callee.index()] = true;
            left[addr.index()] = left[addr.index()].saturating_sub(1);
            elided[addr.index()] |= left[addr.index()] == 0;
        }
    }
    through
}

fn eligible(live: &Live, wanted: &[bool], value: usize) -> bool {
    wanted[value] && !live.ranges_of(value).is_empty() && !crate::park::crosses_ranges(&live.unwinds, live.ranges_of(value))
}

fn passed<R: Copy>(func: &Function, classify: &impl Fn(&Signature) -> Result<Abi<R>, String>) -> (Vec<Option<(R, bool)>>, Vec<Option<(u8, bool)>>) {
    let (mut ints, mut floats) = (vec![None; func.types.len()], vec![None; func.types.len()]);
    let mark = |vals: &[Value], locs: &[Loc<R>], arg: bool, ints: &mut [Option<(R, bool)>], floats: &mut [Option<(u8, bool)>]| {
        for (val, loc) in vals.iter().zip(locs) {
            match *loc {
                Loc::Int(reg) if arg || ints[val.index()].is_none() => ints[val.index()] = Some((reg, arg)),
                Loc::Float(reg) if arg || floats[val.index()].is_none() => floats[val.index()] = Some((reg, arg)),
                _ => {}
            }
        }
    };
    let ret_abi = classify(&func.signature).ok();
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        match &func.insts[inst.index()].op {
            Op::Call { sig, args, results, .. } => {
                if let Ok(abi) = classify(&func.sigs[sig.index()]) {
                    mark(args, &abi.params, true, &mut ints, &mut floats);
                    mark(results, &abi.returns, false, &mut ints, &mut floats);
                }
            }
            Op::TryCall { sig, args, .. } => {
                if let Ok(abi) = classify(&func.sigs[sig.index()]) {
                    mark(args, &abi.params, true, &mut ints, &mut floats);
                }
            }
            Op::Return(vals) => {
                if let Some(abi) = &ret_abi {
                    mark(vals, &abi.returns, false, &mut ints, &mut floats);
                }
            }
            _ => {}
        }
    }
    (ints, floats)
}

fn int_class<'a, R: Copy + PartialEq>(
    func: &Function,
    regs: &'a Regs<R>,
    (order, copied, elided): (&[Block], &[bool], &[bool]),
    params: &[Loc<R>],
    passed: Vec<Option<(R, bool)>>,
) -> Class<'a, R> {
    let dict = func.signature.params.iter().any(|param| param.purpose == ArgumentPurpose::Dict);
    let leaf = regs.leaf.iter().filter(|reg| !(dict && regs.dict == Some(**reg))).copied().collect();
    let clobbers = crate::park::scratch_clobbers(func, (regs.wide_scratch, regs.op_scratch, regs.uses_scratch), (order, copied, elided));
    let arrivals = params
        .iter()
        .map(|loc| match *loc {
            Loc::Int(reg) => vec![reg],
            Loc::Pair(low, high) => vec![low, high],
            _ => Vec::new(),
        })
        .collect();
    Class { saved: regs.saved, leaf, arrivals, passed, clobbers }
}

fn float_class<R>(regs: &Regs<R>, params: &[Loc<R>], passed: Vec<Option<(u8, bool)>>) -> Class<'static, u8> {
    let arrivals = params
        .iter()
        .map(|loc| match *loc {
            Loc::Float(reg) => vec![reg],
            _ => Vec::new(),
        })
        .collect();
    Class { saved: regs.saved_float, leaf: regs.float.to_vec(), arrivals, passed, clobbers: Vec::new() }
}

const LOOP_WEIGHT: u64 = 16;
const MAX_DEPTH: u32 = 4;
const SLOW_SHARE: u64 = 16;

fn slow_blocks(func: &Function, cfg: &Cfg, loops: &Loops) -> Vec<bool> {
    let count = func.blocks.len();
    let calls: Vec<bool> = func.blocks.iter().map(|data| data.insts.iter().any(|inst| matches!(func.insts[inst.index()].op, Op::Call { .. } | Op::TryCall { .. } | Op::Copy(..)))).collect();
    let mut slow = vec![false; count];
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
    slow
}

fn block_weights(depths: &[u32], slow: &[bool]) -> Vec<u64> {
    depths.iter().zip(slow).map(|(&depth, &slow)| (LOOP_WEIGHT.pow(depth.min(MAX_DEPTH)) / if slow { SLOW_SHARE } else { 1 }).max(1)).collect()
}

fn place<R, F>(locations: &mut [Location<R>], homes: Vec<Option<F>>, wrap: impl Fn(F) -> Location<R>) {
    for (location, home) in locations.iter_mut().zip(homes) {
        if let Some(reg) = home {
            *location = wrap(reg);
        }
    }
}

const MAX_ARM64_LOOP_GP: usize = 18;

fn expensive_i64_const(imm: i64, two_address: bool) -> bool {
    let u = imm as u64;
    if two_address && (u >> 32) == (u & 0xffff_ffff) {
        return false;
    }
    let non_zero = (0..4).filter(|&s| ((u >> (s * 16)) & 0xffff) != 0).count();
    let non_ones = (0..4).filter(|&s| ((u >> (s * 16)) & 0xffff) != 0xffff).count();
    if non_zero < 3 || non_ones < 3 {
        return false;
    }
    two_address || (crate::arm64::int::encode_bitmask_imm(u, true).is_none() && !((u >> 32) == 0 && crate::arm64::int::encode_bitmask_imm(u, false).is_some()))
}

fn x64_alu_imm(ty: Type, imm: i64) -> bool {
    ty == Type::I32 || fits_imm32(imm)
}

fn x64_inline_imm(op: &Op, ty: Type, rhs: bool, imm: i64) -> bool {
    match op {
        Op::Icmp(..) => ty != Type::I64 || fits_imm32(imm),
        Op::Binary(Binary::Iadd | Binary::Bor | Binary::Bxor, ..) => x64_alu_imm(ty, imm),
        Op::Binary(Binary::Band, ..) => x64_alu_imm(ty, imm) || (rhs && imm == 0xFFFF_FFFF),
        Op::Binary(Binary::Isub, ..) => rhs && x64_alu_imm(ty, imm),
        _ => false,
    }
}

fn arm64_inline_imm(op: &Op, ty: Type, rhs: bool, imm: i64) -> bool {
    match *op {
        Op::Binary(Binary::Iadd | Binary::Isub, ..) => rhs && crate::arm64::int::add_imm_parts(false, crate::fold::extend(imm, ty, true)).is_some(),
        Op::Icmp(cc, ..) => rhs && crate::arm64::int::add_imm_parts(true, crate::fold::extend(imm, ty, cc.is_signed())).is_some(),
        Op::Binary(Binary::Band | Binary::Bor | Binary::Bxor, ..) => crate::arm64::int::encode_bitmask_imm(imm as u64, ty == Type::I64).is_some(),
        _ => false,
    }
}

fn loop_gp_pressure<R>(func: &Function, loops: &Loops, elided: &[bool], locations: &[Location<R>]) -> Vec<usize> {
    let mut block_gp = vec![0_usize; func.blocks.len()];
    let mut seen = vec![0_u32; func.types.len()];
    for (idx, each) in loops.list.iter().enumerate() {
        let tag = (idx + 1) as u32;
        let mut gp = 0_usize;
        let mut count_val = |v: Value| {
            let vi = v.index();
            if seen[vi] != tag && !elided[vi] && func.types[vi].is_small_int() && matches!(locations[vi], Location::Slot) {
                seen[vi] = tag;
                gp += 1;
            }
        };
        func.blocks[each.head.index()].params.iter().copied().for_each(&mut count_val);
        for (b, &in_l) in each.body.iter().enumerate() {
            if !in_l { continue; }
            for &inst in &func.blocks[b].insts { func.insts[inst.index()].op.for_operands(&mut count_val); }
        }
        for (b, &in_l) in each.body.iter().enumerate() {
            if in_l { block_gp[b] = block_gp[b].max(gp); }
        }
    }
    block_gp
}

fn scan_const_op<R>(
    func: &Function,
    op: &Op,
    locations: &[Location<R>],
    (two_address, in_loop, outer): (bool, bool, &dyn Fn(Value) -> bool),
    (must_stay_imm, loop_mat): (&mut [bool], &mut [u32]),
) {
    match *op {
        Op::Copy(_, _, bytes) => must_stay_imm[bytes.index()] = true,
        Op::Binary(Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr, _, rhs) => must_stay_imm[rhs.index()] = true,
        Op::Binary(Binary::Imul | Binary::Udiv | Binary::Urem | Binary::Sdiv | Binary::Srem, x, y) => {
            let pow2_ok = matches!(op, Op::Binary(Binary::Imul | Binary::Udiv | Binary::Urem, ..));
            let mul_imm = matches!(op, Op::Binary(Binary::Imul, ..)) && two_address;
            for v in [x, y] {
                if let Location::Imm(imm) = locations[v.index()] {
                    if pow2_ok && ((imm > 0 && imm.count_ones() == 1) || (mul_imm && fits_imm32(imm))) { must_stay_imm[v.index()] = true; }
                    else if two_address && in_loop && outer(v) { loop_mat[v.index()] += 1; }
                }
            }
        }
        Op::Binary(Binary::Iadd | Binary::Isub | Binary::Band | Binary::Bor | Binary::Bxor, x, y) | Op::Icmp(_, x, y) if in_loop => {
            let ty = func.ty(x);
            for (rhs, v) in [(false, x), (true, y)] {
                let Location::Imm(imm) = locations[v.index()] else { continue };
                let inline = if two_address { x64_inline_imm(op, ty, rhs, imm) } else { arm64_inline_imm(op, ty, rhs, imm) };
                if inline { must_stay_imm[v.index()] = true; } else if two_address && outer(v) { loop_mat[v.index()] += 1; }
            }
        }
        Op::Store(val, ..) | Op::StackStore(val, ..) if in_loop => {
            if let Location::Imm(imm) = locations[val.index()] {
                if two_address && (func.ty(val).bytes() < 8 || fits_imm32(imm)) { must_stay_imm[val.index()] = true; }
                else if two_address && imm != 0 && outer(val) { loop_mat[val.index()] += 1; }
            }
        }
        _ => {}
    }
}

fn promote_loop_consts<R>(func: &Function, census: &Uses, loops: &Loops, elided: &[bool], two_address: bool, locations: &mut [Location<R>]) -> Vec<(usize, i64)> {
    let count = func.types.len();
    let (mut must_stay_imm, mut loop_mat, mut max_gp) = (vec![false; count], vec![0_u32; count], vec![0_usize; count]);
    let depths = &loops.depths;
    let block_gp = (!two_address).then(|| loop_gp_pressure(func, loops, elided, locations));
    for (b, block) in func.blocks.iter().enumerate() {
        let depth = depths[b];
        let outer = |v: Value| census.def[v.index()].is_some_and(|(db, _)| depths[db] < depth);
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if data.result.is_some_and(|r| elided[r.index()]) { continue; }
            scan_const_op(func, &data.op, locations, (two_address, depth > 0, &outer), (&mut must_stay_imm, &mut loop_mat));
            data.op.for_operands(|v| {
                let vi = v.index();
                if let Location::Imm(imm) = locations[vi]
                    && func.ty(v) == Type::I64
                    && expensive_i64_const(imm, two_address)
                    && ((depth > 0 && (outer(v) || (!two_address && census.uses[vi] >= 2))) || census.uses[vi] >= 4)
                {
                    loop_mat[vi] += 1;
                    if let Some(bgp) = block_gp.as_deref().filter(|_| depth > 0) { max_gp[vi] = max_gp[vi].max(bgp[b]); }
                }
            });
        }
    }
    let (mut added, mut promoted) = (0_usize, Vec::new());
    for (v, loc) in locations.iter_mut().enumerate() {
        if let Location::Imm(imm) = *loc && imm != 0 && !elided[v] && !must_stay_imm[v] && loop_mat[v] > 0 {
            if !two_address && max_gp[v] > 0 && max_gp[v] + added >= MAX_ARM64_LOOP_GP { continue; }
            if !two_address && max_gp[v] > 0 { added += 1; }
            *loc = Location::Slot;
            promoted.push((v, imm));
        }
    }
    promoted
}

fn promote_loop_syms<R: Copy>(func: &Function, census: &Uses, depths: Option<&[u32]>, elided: &[bool], locations: &mut [Location<R>]) -> Vec<(usize, Location<R>)> {
    let mut in_loop = vec![false; func.types.len()];
    if let Some(depths) = depths {
        let outer = |v: Value, depth: u32| census.def[v.index()].is_some_and(|(db, _)| depths[db] < depth);
        for (b, block) in func.blocks.iter().enumerate().filter(|(b, _)| depths[*b] > 0) {
            let depth = depths[b];
            for &inst in &block.insts {
                let data = &func.insts[inst.index()];
                if !data.result.is_some_and(|r| elided[r.index()]) {
                    data.op.for_operands(|v| if matches!(locations[v.index()], Location::SlotAddr(_) | Location::Symbol(_)) && outer(v, depth) { in_loop[v.index()] = true; });
                }
            }
        }
    }
    let mut promoted = Vec::new();
    for (v, loc) in locations.iter_mut().enumerate() {
        if matches!(*loc, Location::SlotAddr(_) | Location::Symbol(_))
            && !elided[v]
            && (census.uses[v] >= 2 || in_loop[v] || depths.is_some_and(|d| census.def[v].is_some_and(|(db, _)| d[db] > 0)))
        {
            promoted.push((v, *loc));
            *loc = Location::Slot;
        }
    }
    promoted
}

pub(crate) fn plan<R: Copy + PartialEq>(
    func: &Function,
    regs: &Regs<R>,
    (order, copied, weighed): (&[Block], &[bool], bool),
    params: &[Loc<R>],
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Plan<R> {
    let count = func.types.len();
    let mut locations = vec![Location::Slot; count];
    consts(func, regs.absolute_calls, &mut locations);
    statics(func, &mut locations);
    let census = census(func);
    let zero_ext = hints::zero_ext(func, &locations, &census, regs.call_zext);
    let (sib, mut elided) = if regs.fold_sib { sib_addrs(func, &locations, &census, &zero_ext, regs.two_address) } else { (vec![None; count], vec![false; count]) };
    let through = if regs.absolute_calls { through(func, &locations, &census, &mut elided) } else { vec![None; count] };
    let cfg = weighed.then(|| func.cfg());
    let loops = cfg.as_ref().map(|cfg| cfg.loops());
    let loop_depths = loops.as_ref().map(|loops| &loops.depths[..]);
    let promoted_consts = loops.as_deref().map(|loops| promote_loop_consts(func, &census, loops, &elided, regs.two_address, &mut locations)).unwrap_or_default();
    let promoted_syms = regs.absolute_calls.then(|| promote_loop_syms(func, &census, loop_depths, &elided, &mut locations)).unwrap_or_default();
    let wanted: Vec<bool> = (0..count).map(|value| !elided[value] && matches!(locations[value], Location::Slot) && registered(regs, func.types[value])).collect();
    let tracked: Vec<bool> = (0..count).map(|value| !elided[value] && (wanted[value] || (weighed && matches!(locations[value], Location::Slot)))).collect();
    let live = live(func, order, copied, &tracked, &sib, &elided);
    let candidates: Vec<bool> = (0..count).map(|value| eligible(&live, &wanted, value)).collect();
    let fused: Vec<bool> = (0..count).map(|value| candidates[value] && fuses(func, &census, value, regs.float_branches, regs.two_address)).collect();
    let ints: Vec<bool> = (0..count).map(|value| candidates[value] && func.types[value].is_small_int() && !fused[value] && through[value].is_none()).collect();
    let floats: Vec<bool> = (0..count).map(|value| candidates[value] && !fused[value] && (func.types[value].is_float() || (regs.vectors && func.types[value].is_vector()))).collect();
    let reuse = hints::reusable(func, regs, &locations, &live, &elided);
    let blocks = cfg.as_deref().zip(loops).map(|(cfg, loops)| block_weights(&loops.depths, &slow_blocks(func, cfg, loops)));
    let weights = blocks.as_deref().map(|blocks| walk::weights(func, blocks, &sib, &elided));
    let (partners, back_args) = walk::edge_hints(func, cfg.as_deref(), order, blocks.as_deref(), !regs.two_address);
    let park = blocks.as_deref().filter(|_| regs.park).map(|blocks| live.sites.iter().map(|(_, block)| blocks[block.index()]).collect::<Vec<u64>>());
    let copies = if regs.two_address { hints::extend_copies(func, &live, &zero_ext, &ints) } else { Vec::new() };
    let hints = Hints {
        reuse: &reuse,
        weights: weights.as_deref(),
        partners: partners.as_deref(),
        back_args: &back_args,
        park: park.as_deref(),
        two_address: regs.two_address,
        copies: &copies,
    };
    let (passed_ints, passed_floats) = passed(func, classify);
    let (homes, used) = allocate(func, &live, &int_class(func, regs, (order, copied, &elided), params, passed_ints), &ints, &hints);
    place(&mut locations, homes, Location::Reg);
    for (v, loc) in promoted_consts.into_iter().map(|(v, i)| (v, Location::Imm(i))).chain(promoted_syms) {
        if !matches!(locations[v], Location::Reg(_)) {
            locations[v] = loc;
        }
    }
    let (float_homes, used_floats) = allocate(func, &live, &float_class(regs, params, passed_floats), &floats, &hints);
    place(&mut locations, float_homes, Location::Float);
    for data in &func.insts {
        if let (Op::Const(0), Some(res)) = (&data.op, data.result)
            && func.ty(res).is_vector()
            && matches!(locations[res.index()], Location::Slot)
        {
            locations[res.index()] = Location::Imm(0);
        }
    }
    let (pairs, deferred) = pairs(func, &locations, &elided);
    let webs = blocks.as_deref().zip(partners.as_deref()).map(|edges| demote::webs(func, &live, edges, (&sib, &elided, &deferred), regs.imm_store, &mut locations));
    let parked = if park.is_some() { parked(func, &live, &locations, (regs.saved, regs.saved_float), classify) } else { Vec::new() };
    let mut homed = vec![false; if parked.is_empty() { 0 } else { count }];
    parked.iter().flat_map(|(_, values)| values).for_each(|(value, _, _)| homed[value.index()] = true);
    let shared = webs.map_or_else(Vec::new, |webs| crate::color::share(func, &live, &locations, &deferred, &webs));
    let zero_ext = if regs.call_zext && hints::has_reg_reduce(func, &locations) { hints::zero_ext(func, &locations, &census, true) } else { zero_ext };
    Plan { locations, fused, used, used_floats, pairs, deferred, through, shared, parked, homed, sib, elided, zero_ext }
}

impl<R: Copy> Plan<R> {
    pub(crate) fn location(&self, value: Value) -> Location<R> { self.locations[value.index()] }
    pub(crate) fn is_static(&self, value: Value) -> bool { self.elided[value.index()] || self.location(value).is_static() }
    pub(crate) fn const_of(&self, value: Value) -> Option<i64> { if let Location::Imm(imm) = self.location(value) { Some(imm) } else { None } }
    pub(crate) fn reg_of(&self, value: Value) -> Option<R> { if let Location::Reg(reg) = self.location(value) { Some(reg) } else { None } }
    pub(crate) fn parked_at(&self, inst: Inst) -> &[(Value, ParkReg<R>, bool)] {
        self.parked.binary_search_by_key(&inst.index(), |(at, _)| at.index()).map_or(&[], |at| &self.parked[at].1)
    }
    pub(crate) fn float_of(&self, value: Value) -> Option<u8> { if let Location::Float(reg) = self.location(value) { Some(reg) } else { None } }
}

impl<R: Copy + PartialEq> Plan<R> {
    pub(crate) fn same_home(&self, frame: &crate::frame::Frame<R>, a: Value, b: Value) -> bool { crate::frame::same_home(&self.locations, &frame.values, a, b) }
    pub(crate) fn moves(&self, func: &Function, frame: &crate::frame::Frame<R>, block: Block, args: &[Value]) -> bool {
        args.iter().zip(&func.blocks[block.index()].params).any(|(arg, param)| !self.same_home(frame, *arg, *param))
    }
}

fn passed_own<R: Copy + PartialEq>(func: &Function, inst: Inst, value: Value, reg: ParkReg<R>, classify: &impl Fn(&Signature) -> Result<Abi<R>, String>) -> bool {
    let Op::Call { sig, args, .. } = &func.insts[inst.index()].op else { return false };
    let Ok(abi) = classify(&func.sigs[sig.index()]) else { return false };
    args.iter().zip(&abi.params).any(|(arg, loc)| {
        *arg == value
            && match (reg, *loc) {
                (ParkReg::Int(r), Loc::Int(at)) => r == at,
                (ParkReg::Float(v), Loc::Float(at)) => v == at,
                _ => false,
            }
    })
}

fn parked<R: Copy + PartialEq>(
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
        entries.sort_unstable_by_key(|&(v, r, _)| (matches!(r, ParkReg::Float(_)), func.ty(v).bytes(), v.index()));
    }
    let mut out: Vec<(Inst, Vec<(Value, ParkReg<R>, bool)>)> = live.sites.iter().map(|site| site.0).zip(at_call).filter(|(_, values)| !values.is_empty()).collect();
    out.sort_unstable_by_key(|(inst, _)| inst.index());
    out
}
