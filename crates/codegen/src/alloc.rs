use crate::fold::{constant, holds};
use crate::frame::{Abi, Loc};
use crate::ir::{ArgumentPurpose, Binary, Block, Convert, FloatCC, Function, Inst, Op, Signature, StackSlot, Type, Unary, Value};
use crate::live::{Live, Sib, SibBase, for_effective_operands, live, results};
use crate::park::{Class, Hints, ParkReg, allocate, parked};

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
}

const FOLD_LIMIT: i64 = 1 << 24;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Location<R> {
    Slot,
    Imm(i64),
    SlotAddr(u32),
    Reg(R),
    Float(u8),
    Frame(StackSlot, i32),
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
            };
        }
    }
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } = &func.insts[inst.index()].op
            && census.uses[callee.index()] == 1
            && !args.contains(callee)
            && let Some(target) = loads[callee.index()]
        {
            through[callee.index()] = Some(target);
            elided[callee.index()] = true;
        }
    }
    through
}

fn eligible(live: &Live, wanted: &[bool], value: usize) -> bool {
    wanted[value] && !live.ranges_of(value).is_empty() && !crate::park::crosses_ranges(&live.unwinds, live.ranges_of(value))
}

fn passed<R: Copy>(func: &Function, classify: &impl Fn(&Signature) -> Result<Abi<R>, String>) -> (Vec<Option<(R, bool)>>, Vec<Option<(u8, bool)>>) {
    let (mut ints, mut floats) = (vec![None; func.types.len()], vec![None; func.types.len()]);
    let ret_abi = classify(&func.signature).ok();
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        match &func.insts[inst.index()].op {
            Op::Call { sig, args, results, .. } => {
                if let Ok(abi) = classify(&func.sigs[sig.index()]) {
                    for (arg, loc) in args.iter().zip(&abi.params) {
                        match *loc {
                            Loc::Int(reg) => ints[arg.index()] = Some((reg, true)),
                            Loc::Float(reg) => floats[arg.index()] = Some((reg, true)),
                            _ => {}
                        }
                    }
                    for (res, loc) in results.iter().zip(&abi.returns) {
                        match *loc {
                            Loc::Int(reg) if ints[res.index()].is_none() => ints[res.index()] = Some((reg, false)),
                            Loc::Float(reg) if floats[res.index()].is_none() => floats[res.index()] = Some((reg, false)),
                            _ => {}
                        }
                    }
                }
            }
            Op::TryCall { sig, args, .. } => {
                if let Ok(abi) = classify(&func.sigs[sig.index()]) {
                    for (arg, loc) in args.iter().zip(&abi.params) {
                        match *loc {
                            Loc::Int(reg) => ints[arg.index()] = Some((reg, true)),
                            Loc::Float(reg) => floats[arg.index()] = Some((reg, true)),
                            _ => {}
                        }
                    }
                }
            }
            Op::Return(vals) => {
                if let Some(abi) = &ret_abi {
                    for (val, loc) in vals.iter().zip(&abi.returns) {
                        match *loc {
                            Loc::Int(reg) if ints[val.index()].is_none() => ints[val.index()] = Some((reg, false)),
                            Loc::Float(reg) if floats[val.index()].is_none() => floats[val.index()] = Some((reg, false)),
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    (ints, floats)
}

fn int_class<'a, R: Copy + PartialEq>(func: &Function, regs: &'a Regs<R>, order: &[Block], params: &[Loc<R>], passed: Vec<Option<(R, bool)>>) -> Class<'a, R> {
    let (mut wide, mut scratch) = (false, false);
    for data in order.iter().flat_map(|block| &func.blocks[block.index()].insts).map(|inst| &func.insts[inst.index()]) {
        wide |= data.result.is_some_and(|result| func.ty(result) == Type::I128);
        scratch |= (regs.uses_scratch)(func, &data.op);
    }
    let dict = func.signature.params.iter().any(|param| param.purpose == ArgumentPurpose::Dict);
    let kept = |reg: &R| !(wide && regs.wide_scratch.contains(reg)) && !(scratch && regs.op_scratch.contains(reg)) && !(dict && regs.dict == Some(*reg));
    let leaf = regs.leaf.iter().filter(|reg| kept(reg)).copied().collect();
    let arrivals = params
        .iter()
        .map(|loc| match *loc {
            Loc::Int(reg) => vec![reg],
            Loc::Pair(low, high) => vec![low, high],
            _ => Vec::new(),
        })
        .collect();
    Class { saved: regs.saved, leaf, arrivals, passed }
}

fn float_class<R>(regs: &Regs<R>, params: &[Loc<R>], passed: Vec<Option<(u8, bool)>>) -> Class<'static, u8> {
    let arrivals = params
        .iter()
        .map(|loc| match *loc {
            Loc::Float(reg) => vec![reg],
            _ => Vec::new(),
        })
        .collect();
    Class { saved: regs.saved_float, leaf: regs.float.to_vec(), arrivals, passed }
}

const LOOP_WEIGHT: u64 = 16;
const MAX_DEPTH: u32 = 4;

fn block_weights(depths: &[u32]) -> Vec<u64> {
    depths.iter().map(|&depth| LOOP_WEIGHT.pow(depth.min(MAX_DEPTH))).collect()
}

fn weights(func: &Function, blocks: &[u64], sib: &[Option<Sib>], elided: &[bool]) -> Vec<u64> {
    let mut weight = vec![0_u64; func.types.len()];
    for (block, data) in func.blocks.iter().enumerate() {
        let each = blocks[block];
        for inst in &data.insts {
            let data = &func.insts[inst.index()];
            if data.result.is_some_and(|res| elided[res.index()]) {
                continue;
            }
            for_effective_operands(data, sib, elided, |value| weight[value.index()] += each);
            results(data).for_each(|value| weight[value.index()] += each);
        }
    }
    weight
}

fn partners(func: &Function) -> Vec<Vec<Value>> {
    let mut out = vec![Vec::new(); func.types.len()];
    let mut pair = |params: &[Value], args: &[Value]| {
        for (param, arg) in params.iter().zip(args) {
            out[param.index()].push(*arg);
            out[arg.index()].push(*param);
        }
    };
    for want_back in [true, false] {
        for (b_idx, block) in func.blocks.iter().enumerate() {
            let mut edge = |target: Block, args: &[Value]| {
                if (target.index() <= b_idx) == want_back {
                    pair(&func.blocks[target.index()].params, args);
                }
            };
            for inst in &block.insts {
                crate::frame::terminator_edges(&func.insts[inst.index()].op, &mut edge);
            }
        }
    }
    out
}

fn in_place(op: Binary) -> bool {
    matches!(op, Binary::Iadd | Binary::Isub | Binary::Imul | Binary::Band | Binary::Bor | Binary::Bxor | Binary::Fadd | Binary::Fsub | Binary::Fmul | Binary::Fdiv)
}

fn reusable(func: &Function, live: &Live, elided: &[bool]) -> Vec<Option<Value>> {
    let mut reuse = vec![None; func.types.len()];
    for block in &func.blocks {
        for (at, inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let (Op::Binary(op, x, y), Some(result)) = (&data.op, data.result) else { continue };
            if elided[result.index()] {
                continue;
            }
            let next = block.insts.get(at + 1).map(|next| &func.insts[next.index()].op);
            if matches!(next, Some(Op::Fits(check, _, a, b)) if check == op && (a, b) == (x, y)) {
                continue;
            }
            let def = live.defs[result.index()];
            let end_at = def.saturating_sub(1);
            let dies = def > 0
                && live.last_end(x.index()) == Some(end_at)
                && !crate::live::overlaps(live.ranges_of(x.index()), live.ranges_of(result.index()));
            if in_place(*op) && dies && live.ranges_of(result.index()).first().is_some_and(|r| r.start == def) {
                reuse[result.index()] = Some(*x);
            }
        }
    }
    reuse
}

fn place<R, F>(locations: &mut [Location<R>], homes: Vec<Option<F>>, wrap: impl Fn(F) -> Location<R>) {
    for (location, home) in locations.iter_mut().zip(homes) {
        if let Some(reg) = home {
            *location = wrap(reg);
        }
    }
}

fn expensive_i64_const(imm: i64) -> bool {
    let u = imm as u64;
    if (u >> 32) == (u & 0xffff_ffff) {
        return false;
    }
    let non_zero = (0..4).filter(|&s| ((u >> (s * 16)) & 0xffff) != 0).count();
    let non_ones = (0..4).filter(|&s| ((u >> (s * 16)) & 0xffff) != 0xffff).count();
    non_zero >= 3 && non_ones >= 3
}

fn x64_alu_imm(ty: Type, imm: i64) -> bool {
    ty == Type::I32 || i32::try_from(imm).is_ok()
}

fn x64_inline_imm(op: &Op, ty: Type, rhs: bool, imm: i64) -> bool {
    match op {
        Op::Icmp(..) => ty != Type::I64 || i32::try_from(imm).is_ok(),
        Op::Binary(Binary::Iadd | Binary::Bor | Binary::Bxor, ..) => x64_alu_imm(ty, imm),
        Op::Binary(Binary::Band, ..) => x64_alu_imm(ty, imm) || (rhs && imm == 0xFFFF_FFFF),
        Op::Binary(Binary::Isub, ..) => rhs && x64_alu_imm(ty, imm),
        _ => false,
    }
}

fn promote_loop_consts<R>(func: &Function, census: &Uses, depths: &[u32], elided: &[bool], two_address: bool, locations: &mut [Location<R>]) -> Vec<(usize, i64)> {
    let count = func.types.len();
    let mut must_stay_imm = vec![false; count];
    let mut loop_mat = vec![0_u32; count];
    let outer = |v: Value, depth: u32| census.def[v.index()].is_some_and(|(db, _)| depths[db] < depth);
    for (b, block) in func.blocks.iter().enumerate() {
        let depth = depths[b];
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if data.result.is_some_and(|r| elided[r.index()]) {
                continue;
            }
            match data.op {
                Op::Copy(_, _, bytes) => must_stay_imm[bytes.index()] = true,
                Op::Binary(Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr, _, rhs) => must_stay_imm[rhs.index()] = true,
                Op::Binary(Binary::Imul | Binary::Udiv | Binary::Urem, x, y) => {
                    let mul_imm = matches!(data.op, Op::Binary(Binary::Imul, ..)) && two_address;
                    for v in [x, y] {
                        if let Location::Imm(imm) = locations[v.index()] {
                            if (imm > 0 && imm.count_ones() == 1) || (mul_imm && i32::try_from(imm).is_ok()) {
                                must_stay_imm[v.index()] = true;
                            } else if depth > 0 && outer(v, depth) {
                                loop_mat[v.index()] += 1;
                            }
                        }
                    }
                }
                Op::Binary(Binary::Sdiv | Binary::Srem, x, y) if depth > 0 => {
                    for v in [x, y] {
                        if matches!(locations[v.index()], Location::Imm(_)) && outer(v, depth) {
                            loop_mat[v.index()] += 1;
                        }
                    }
                }
                Op::Binary(Binary::Iadd | Binary::Isub | Binary::Band | Binary::Bor | Binary::Bxor, x, y) | Op::Icmp(_, x, y) if depth > 0 => {
                    let ty = func.ty(x);
                    for (rhs, v) in [(false, x), (true, y)] {
                        let Location::Imm(imm) = locations[v.index()] else { continue };
                        let inline = if two_address {
                            x64_inline_imm(&data.op, ty, rhs, imm)
                        } else {
                            matches!(data.op, Op::Binary(Binary::Iadd | Binary::Isub, ..) | Op::Icmp(..)) && (-4095..4096).contains(&imm)
                        };
                        if inline {
                            if two_address || rhs {
                                must_stay_imm[v.index()] = true;
                            }
                        } else if (two_address || matches!(data.op, Op::Binary(Binary::Iadd | Binary::Isub, ..) | Op::Icmp(..))) && outer(v, depth) {
                            loop_mat[v.index()] += 1;
                        }
                    }
                }
                Op::Store(val, ..) | Op::StackStore(val, ..) if depth > 0 => {
                    if let Location::Imm(imm) = locations[val.index()] {
                        if two_address && (func.ty(val).bytes() < 8 || i32::try_from(imm).is_ok()) {
                            must_stay_imm[val.index()] = true;
                        } else if imm != 0 && outer(val, depth) {
                            loop_mat[val.index()] += 1;
                        }
                    }
                }
                _ => {}
            }
            data.op.for_operands(|v| {
                if let Location::Imm(imm) = locations[v.index()]
                    && func.ty(v) == Type::I64
                    && expensive_i64_const(imm)
                    && ((depth > 0 && outer(v, depth)) || census.uses[v.index()] >= 4)
                {
                    loop_mat[v.index()] += 1;
                }
            });
        }
    }
    let mut promoted = Vec::new();
    for v in 0..count {
        if let Location::Imm(imm) = locations[v]
            && imm != 0
            && !elided[v]
            && !must_stay_imm[v]
            && loop_mat[v] > 0
        {
            locations[v] = Location::Slot;
            promoted.push((v, imm));
        }
    }
    promoted
}

fn zero_ext<R>(func: &Function, locations: &[Location<R>], census: &Uses) -> Vec<bool> {
    let count = func.types.len();
    let (mut out, mut incoming) = (vec![false; count], vec![Some(Vec::new()); count]);
    if let Some(entry) = func.entry {
        func.blocks[entry.index()].params.iter().for_each(|p| incoming[p.index()] = None);
    }
    for &inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let op = &func.insts[inst.index()].op;
        if let Op::TryCall { handler, .. } = *op {
            func.blocks[handler.index()].params.iter().for_each(|p| incoming[p.index()] = None);
        }
        crate::frame::terminator_edges(op, &mut |target: Block, args: &[Value]| {
            for (p, &a) in func.blocks[target.index()].params.iter().zip(args) {
                if a != *p && let Some(list) = &mut incoming[p.index()] {
                    list.push(a);
                }
            }
        });
    }
    for (v, ty) in func.types.iter().enumerate() {
        let single_def = census.def_inst[v].is_some_and(|d| func.insts[d.index()].result == Some(Value(v as u32)));
        out[v] = ty.is_small_int() && (single_def || incoming[v].as_ref().is_some_and(|l| !l.is_empty()));
    }
    let const_val = |v: Value| match locations[v.index()] {
        Location::Imm(imm) => Some(imm),
        _ => census.def_inst[v.index()].and_then(|d| match func.insts[d.index()].op { Op::Const(imm) => Some(imm), _ => None }),
    };
    let ty_mask = |ty: Type| if ty == Type::I64 { i64::MAX } else { (1_i64 << ty.bits()) - 1 };
    let fits_mask = |v: Value, mask: i64| const_val(v).is_some_and(|m| (0..=mask).contains(&m));
    let is_icmp = |v: Value| census.def_inst[v.index()].is_some_and(|d| matches!(func.insts[d.index()].op, Op::Icmp(..)));
    let is_cmp = |v: Value| is_icmp(v) || census.def_inst[v.index()].is_some_and(|d| matches!(func.insts[d.index()].op, Op::Convert(Convert::Uextend, inner) if is_icmp(inner)));
    let clean_bits = |out: &[bool], v: Value| -> Option<u32> {
        let imm_bits = |x: i64| (x >= 0).then(|| 64 - (x as u64).leading_zeros());
        if let Some(w) = const_val(v).and_then(imm_bits) {
            return Some(w);
        }
        match census.def_inst[v.index()].map(|d| &func.insts[d.index()].op) {
            Some(&Op::Convert(Convert::Uextend, inner)) => Some(func.types[inner.index()].bits()),
            Some(&Op::Binary(Binary::Band, a, b)) => const_val(a)
                .and_then(imm_bits)
                .or_else(|| const_val(b).and_then(imm_bits))
                .map(|w| w.min(func.types[v.index()].bits()))
                .or_else(|| out[v.index()].then(|| func.types[v.index()].bits())),
            _ => out[v.index()].then(|| func.types[v.index()].bits()),
        }
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &inst in func.blocks.iter().flat_map(|b| &b.insts) {
            let Some(res) = func.insts[inst.index()].result else { continue };
            let ty = func.types[res.index()];
            if !ty.is_small_int() {
                continue;
            }
            let z = match func.insts[inst.index()].op {
                Op::Const(imm) => (0..=ty_mask(ty)).contains(&imm),
                Op::Load(..) | Op::StackLoad(..) | Op::AtomicLoad(_) | Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) | Op::Convert(Convert::Uextend, _) | Op::Unary(Unary::Clz | Unary::Ctz | Unary::Popcnt, _) => true,
                Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Unary(Unary::Bitrev, _) => ty == Type::I32,
                Op::Unary(Unary::Bswap, a) => ty == Type::I32 && out[a.index()],
                Op::Select(_, a, b) => out[a.index()] && out[b.index()],
                Op::Binary(Binary::Bor | Binary::Bxor, a, b) => ty == Type::I32 || (out[a.index()] && out[b.index()]),
                Op::Binary(Binary::Iadd, a, b) => ty == Type::I32 && !is_cmp(a) && !is_cmp(b),
                Op::Binary(Binary::Isub | Binary::Rotl | Binary::Rotr | Binary::Udiv | Binary::Urem | Binary::Umulhi | Binary::Umin | Binary::Umax | Binary::UaddSat | Binary::UsubSat | Binary::AvgRound, _, _) => ty == Type::I32,
                Op::Binary(Binary::Ushr, a, _) => ty == Type::I32 || out[a.index()],
                Op::Binary(Binary::Ishl, a, amt) => const_val(amt).is_some_and(|s| clean_bits(&out, a).is_some_and(|w| w + ((s as u32) & (ty.bits() - 1)) <= ty.bits())),
                Op::Binary(Binary::Band, a, b) => ty == Type::I32 || (matches!(ty, Type::I8 | Type::I16) && (out[a.index()] || out[b.index()] || fits_mask(a, ty_mask(ty)) || fits_mask(b, ty_mask(ty)))),
                Op::Convert(Convert::Ireduce, shifted) => {
                    let (src_ty, dst_bits) = (func.types[shifted.index()], ty.bits());
                    clean_bits(&out, shifted).is_some_and(|w| w <= dst_bits)
                        || (matches!(src_ty, Type::I64 | Type::I32)
                            && census.def_inst[shifted.index()].is_some_and(|def| match func.insts[def.index()].op {
                                Op::Unary(Unary::Clz | Unary::Ctz | Unary::Popcnt, _) => dst_bits >= 8,
                                Op::Binary(Binary::Ushr, _, c) => const_val(c).is_some_and(|s| (0..i64::from(src_ty.bits())).contains(&s) && src_ty.bits() - (s as u32) <= dst_bits),
                                _ => false,
                            }))
                }
                _ => false,
            };
            if out[res.index()] != z {
                out[res.index()] = z;
                changed = true;
            }
        }
        for (p, list) in incoming.iter().enumerate() {
            if out[p] && let Some(list) = list && !list.iter().all(|a| out[a.index()]) {
                out[p] = false;
                changed = true;
            }
        }
    }
    out
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
    let zero_ext = zero_ext(func, &locations, &census);
    let (sib, mut elided) = if regs.fold_sib { sib_addrs(func, &locations, &census, &zero_ext, regs.two_address) } else { (vec![None; count], vec![false; count]) };
    let through = if regs.absolute_calls { through(func, &locations, &census, &mut elided) } else { vec![None; count] };
    let loop_depths = weighed.then(|| func.cfg().loops().depths);
    let promoted_consts = loop_depths.as_deref().map(|depths| promote_loop_consts(func, &census, depths, &elided, regs.two_address, &mut locations)).unwrap_or_default();
    let mut promoted_syms = Vec::new();
    if regs.absolute_calls {
        for (v, loc) in locations.iter_mut().enumerate() {
            if let Location::SlotAddr(sym) = *loc
                && !elided[v]
                && (census.uses[v] > 1 || loop_depths.as_deref().is_some_and(|d| census.def[v].is_some_and(|(b, _)| d[b] > 0)))
            {
                *loc = Location::Slot;
                promoted_syms.push((v, sym));
            }
        }
    }
    let wanted: Vec<bool> = (0..count).map(|value| !elided[value] && matches!(locations[value], Location::Slot) && registered(regs, func.types[value])).collect();
    let tracked: Vec<bool> = (0..count).map(|value| !elided[value] && (wanted[value] || (weighed && matches!(locations[value], Location::Slot)))).collect();
    let live = live(func, order, copied, &tracked, &sib, &elided);
    let candidates: Vec<bool> = (0..count).map(|value| eligible(&live, &wanted, value)).collect();
    let fused: Vec<bool> = (0..count).map(|value| candidates[value] && fuses(func, &census, value, regs.float_branches, regs.two_address)).collect();
    let ints: Vec<bool> = (0..count).map(|value| candidates[value] && func.types[value].is_small_int() && !fused[value] && through[value].is_none()).collect();
    let floats: Vec<bool> = (0..count).map(|value| candidates[value] && !fused[value] && (func.types[value].is_float() || (regs.vectors && func.types[value].is_vector()))).collect();
    let reuse = if regs.two_address { reusable(func, &live, &elided) } else { vec![None; count] };
    let blocks = loop_depths.as_deref().map(block_weights);
    let weights = blocks.as_deref().map(|blocks| weights(func, blocks, &sib, &elided));
    let partners = weighed.then(|| partners(func));
    let park = blocks.as_deref().filter(|_| regs.park).map(|blocks| live.sites.iter().map(|(_, block)| blocks[block.index()]).collect::<Vec<u64>>());
    let hints = Hints { reuse: &reuse, weights: weights.as_deref(), partners: partners.as_deref(), park: park.as_deref() };
    let (passed_ints, passed_floats) = passed(func, classify);
    let (homes, used) = allocate(func, &live, &int_class(func, regs, order, params, passed_ints), &ints, &hints);
    place(&mut locations, homes, Location::Reg);
    for (v, loc) in promoted_consts.into_iter().map(|(v, i)| (v, Location::Imm(i))).chain(promoted_syms.into_iter().map(|(v, s)| (v, Location::SlotAddr(s)))) {
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
    let parked = if park.is_some() { parked(func, &live, &locations, (regs.saved, regs.saved_float), classify) } else { Vec::new() };
    let mut homed = vec![false; if parked.is_empty() { 0 } else { count }];
    parked.iter().flat_map(|(_, values)| values).for_each(|(value, _, _)| homed[value.index()] = true);
    let (pairs, deferred) = pairs(func, &locations, &elided);
    let shared = if weighed { crate::color::share(func, &live, &locations, &deferred) } else { Vec::new() };
    Plan { locations, fused, used, used_floats, pairs, deferred, through, shared, parked, homed, sib, elided, zero_ext }
}

impl<R: Copy> Plan<R> {
    pub(crate) fn location(&self, value: Value) -> Location<R> {
        self.locations[value.index()]
    }

    pub(crate) fn is_static(&self, value: Value) -> bool {
        self.elided[value.index()] || matches!(self.location(value), Location::Imm(_) | Location::SlotAddr(_) | Location::Frame(..))
    }

    pub(crate) fn const_of(&self, value: Value) -> Option<i64> {
        if let Location::Imm(imm) = self.location(value) { Some(imm) } else { None }
    }

    pub(crate) fn reg_of(&self, value: Value) -> Option<R> {
        if let Location::Reg(reg) = self.location(value) { Some(reg) } else { None }
    }

    pub(crate) fn parked_at(&self, inst: Inst) -> &[(Value, ParkReg<R>, bool)] {
        self.parked.binary_search_by_key(&inst.index(), |(at, _)| at.index()).map_or(&[], |at| &self.parked[at].1)
    }

    pub(crate) fn float_of(&self, value: Value) -> Option<u8> {
        if let Location::Float(reg) = self.location(value) { Some(reg) } else { None }
    }
}

impl<R: Copy + PartialEq> Plan<R> {
    pub(crate) fn same_home(&self, frame: &crate::frame::Frame<R>, a: Value, b: Value) -> bool {
        crate::frame::same_home(&self.locations, &frame.values, a, b)
    }

    pub(crate) fn moves(&self, func: &Function, frame: &crate::frame::Frame<R>, block: Block, args: &[Value]) -> bool {
        args.iter().zip(&func.blocks[block.index()].params).any(|(arg, param)| !self.same_home(frame, *arg, *param))
    }
}
