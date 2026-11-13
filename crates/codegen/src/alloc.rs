use crate::fold::{constant, holds};
use crate::frame::{Abi, Loc};
use crate::ir::{ArgumentPurpose, Binary, Block, Convert, FloatCC, Function, Inst, Op, Signature, StackSlot, Type, Value};
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

pub(crate) fn no_scratch(_: &Function, _: &Op) -> bool {
    false
}

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
    ty.is_small_int() || ((ty.is_float() || (regs.vectors && ty == Type::V128)) && !regs.float.is_empty())
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

fn sib_addrs<R>(func: &Function, locations: &[Location<R>], census: &Uses) -> (Vec<Option<Sib>>, Vec<bool>) {
    let count = func.types.len();
    let (mut sib, mut elided) = (vec![None; count], vec![false; count]);
    let single_i64 = |v: Value, block: usize| {
        func.ty(v) == Type::I64
            && matches!(locations[v.index()], Location::Slot)
            && census.uses[v.index()] == 1
            && census.def[v.index()].is_some_and(|(b, _)| b == block)
    };
    let parse_base = |v: Value| match locations[v.index()] {
        Location::Frame(slot, off) => Some((SibBase::Frame(slot), off)),
        Location::Slot if func.ty(v) == Type::I64 => Some((SibBase::Value(v), 0)),
        _ => None,
    };
    let parse_scaled = |v: Value, block: usize| -> Option<(Value, u8, Value)> {
        if !single_i64(v, block) {
            return None;
        }
        let op = &func.insts[census.def_inst[v.index()]?.index()].op;
        match *op {
            Op::Binary(Binary::Ishl, idx, sh) if func.ty(idx) == Type::I64 && matches!(locations[idx.index()], Location::Slot) => {
                let Location::Imm(s @ 1..=3) = locations[sh.index()] else { return None };
                Some((idx, s as u8, v))
            }
            Op::Binary(Binary::Imul, a, b) => [(a, b), (b, a)].into_iter().find_map(|(idx, sc)| {
                if func.ty(idx) != Type::I64 || !matches!(locations[idx.index()], Location::Slot) {
                    return None;
                }
                let Location::Imm(m @ (2 | 4 | 8)) = locations[sc.index()] else { return None };
                Some((idx, m.trailing_zeros() as u8, v))
            }),
            _ => None,
        }
    };
    for (block, data) in func.blocks.iter().enumerate() {
        for &inst in &data.insts {
            let (addr, inst_off) = match func.insts[inst.index()].op {
                Op::Load(addr, off) => (addr, off),
                Op::Store(val, addr, off) if val != addr => (addr, off),
                _ => continue,
            };
            if !single_i64(addr, block) || census.last_use[addr.index()].is_none_or(|(ub, _)| ub != block) {
                continue;
            }
            let Some(def_inst) = census.def_inst[addr.index()] else { continue };
            let Op::Binary(Binary::Iadd, mut left, mut right) = func.insts[def_inst.index()].op else { continue };
            let mut outer_disp = 0_i64;
            let mut peeled_add = None;
            for (inner, imm_v) in [(left, right), (right, left)] {
                if let Location::Imm(imm) = locations[imm_v.index()]
                    && (-FOLD_LIMIT..=FOLD_LIMIT).contains(&imm)
                    && single_i64(inner, block)
                    && let Some(inner_inst) = census.def_inst[inner.index()]
                    && let Op::Binary(Binary::Iadd, l2, r2) = func.insts[inner_inst.index()].op
                {
                    outer_disp = imm;
                    peeled_add = Some(inner);
                    (left, right) = (l2, r2);
                    break;
                }
            }
            let matched = parse_base(left)
                .and_then(|(b, bd)| parse_scaled(right, block).map(|(idx, sh, sv)| (b, bd, idx, sh, Some(sv))))
                .or_else(|| parse_base(right).and_then(|(b, bd)| parse_scaled(left, block).map(|(idx, sh, sv)| (b, bd, idx, sh, Some(sv)))))
                .or_else(|| parse_base(left).filter(|_| func.ty(right) == Type::I64 && matches!(locations[right.index()], Location::Slot)).map(|(b, bd)| (b, bd, right, 0, None)))
                .or_else(|| parse_base(right).filter(|_| func.ty(left) == Type::I64 && matches!(locations[left.index()], Location::Slot)).map(|(b, bd)| (b, bd, left, 0, None)));
            let Some((base, base_disp, index, shift, scaled_val)) = matched else { continue };
            let total = outer_disp + i64::from(base_disp);
            if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&total) {
                continue;
            }
            let disp = total as i32;
            let fits = match base {
                SibBase::Frame(_) => (-FOLD_LIMIT..=FOLD_LIMIT).contains(&(total + i64::from(inst_off))),
                SibBase::Value(_) => disp.checked_add(inst_off).and_then(|d| d.checked_add(16)).is_some(),
            };
            if !fits {
                continue;
            }
            sib[addr.index()] = Some(Sib { base, index, shift, disp });
            elided[addr.index()] = true;
            if let Some(p) = peeled_add {
                elided[p.index()] = true;
            }
            if let Some(s) = scaled_val {
                elided[s.index()] = true;
            }
        }
    }
    (sib, elided)
}

fn fuses(func: &Function, census: &Uses, value: usize, float_branches: &[FloatCC]) -> bool {
    let (Some((block, at)), Some((use_block, last))) = (census.def[value], census.last_use[value]) else { return false };
    if block != use_block {
        return false;
    }
    let insts = &func.blocks[block].insts;
    let op = &func.insts[insts[at].index()].op;
    let next = insts.get(at + 1).map(|inst| &func.insts[inst.index()].op);
    let flags = match op {
        Op::Icmp(..) | Op::Fits(..) => true,
        Op::Fcmp(cc, ..) => float_branches.contains(cc),
        _ => false,
    };
    census.uses[value] == 1
        && last == at + 1
        && flags
        && matches!(next, Some(Op::Brif(condition, ..) | Op::Trapz(condition)) if condition.index() == value)
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

fn passed<R: Copy>(func: &Function, classify: &impl Fn(&Signature) -> Result<Abi<R>, String>) -> (Vec<Option<R>>, Vec<Option<u8>>) {
    let (mut ints, mut floats) = (vec![None; func.types.len()], vec![None; func.types.len()]);
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::Call { sig, args, .. } | Op::TryCall { sig, args, .. } = &func.insts[inst.index()].op
            && let Ok(abi) = classify(&func.sigs[sig.index()])
        {
            for (arg, loc) in args.iter().zip(&abi.params) {
                match *loc {
                    Loc::Int(reg) => ints[arg.index()] = Some(reg),
                    Loc::Float(reg) => floats[arg.index()] = Some(reg),
                    _ => {}
                }
            }
        }
    }
    (ints, floats)
}

fn int_class<'a, R: Copy + PartialEq>(func: &Function, regs: &'a Regs<R>, order: &[Block], params: &[Loc<R>], passed: Vec<Option<R>>) -> Class<'a, R> {
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

fn float_class<R>(regs: &Regs<R>, params: &[Loc<R>], passed: Vec<Option<u8>>) -> Class<'static, u8> {
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

fn block_weights(func: &Function) -> Vec<u64> {
    crate::licm::depths(func).into_iter().map(|depth| LOOP_WEIGHT.pow(depth.min(MAX_DEPTH))).collect()
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
                match &func.insts[inst.index()].op {
                    Op::Jump(target, args) => edge(*target, args),
                    Op::Brif(_, then, then_args, otherwise, else_args) => {
                        edge(*then, then_args);
                        edge(*otherwise, else_args);
                    }
                    _ => {}
                }
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
            let checked = matches!(next, Some(Op::Fits(check, _, a, b)) if check == op && (a, b) == (x, y));
            let def = live.defs[result.index()];
            let end_at = if checked { def + 1 } else { def.saturating_sub(1) };
            let dies = def > 0
                && live.ranges_of(x.index()).iter().any(|r| r.end == end_at)
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

fn promote_loop_consts<R>(func: &Function, census: &Uses, elided: &[bool], locations: &mut [Location<R>]) -> Vec<(usize, i64)> {
    let depths = crate::licm::depths(func);
    if !depths.iter().any(|&d| d > 0) {
        return Vec::new();
    }
    let count = func.types.len();
    let mut must_stay_imm = vec![false; count];
    let mut loop_mat = vec![0_u32; count];
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
                    for v in [x, y] {
                        if let Location::Imm(imm) = locations[v.index()] {
                            if imm > 0 && imm.count_ones() == 1 {
                                must_stay_imm[v.index()] = true;
                            } else if depth > 0 && census.def[v.index()].is_some_and(|(db, _)| depths[db] < depth) {
                                loop_mat[v.index()] += 1;
                            }
                        }
                    }
                }
                Op::Binary(Binary::Sdiv | Binary::Srem, x, y) if depth > 0 => {
                    for v in [x, y] {
                        if matches!(locations[v.index()], Location::Imm(_)) && census.def[v.index()].is_some_and(|(db, _)| depths[db] < depth) {
                            loop_mat[v.index()] += 1;
                        }
                    }
                }
                Op::Binary(Binary::Iadd | Binary::Isub, x, y) | Op::Icmp(_, x, y) if depth > 0 => {
                    if let Location::Imm(imm) = locations[y.index()] {
                        if (-4095..4096).contains(&imm) {
                            must_stay_imm[y.index()] = true;
                        } else if census.def[y.index()].is_some_and(|(db, _)| depths[db] < depth) {
                            loop_mat[y.index()] += 1;
                        }
                    }
                    if let Location::Imm(imm) = locations[x.index()]
                        && !(-4095..4096).contains(&imm)
                        && census.def[x.index()].is_some_and(|(db, _)| depths[db] < depth)
                    {
                        loop_mat[x.index()] += 1;
                    }
                }
                Op::Store(val, ..) | Op::StackStore(val, ..) if depth > 0 => {
                    if let Location::Imm(imm) = locations[val.index()]
                        && imm != 0
                        && census.def[val.index()].is_some_and(|(db, _)| depths[db] < depth)
                    {
                        loop_mat[val.index()] += 1;
                    }
                }
                _ => {}
            }
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

fn zero_ext<R>(func: &Function, order: &[Block], locations: &[Location<R>], census: &Uses) -> Vec<bool> {
    let mut out = vec![false; func.types.len()];
    let fits_mask = |v: Value, mask: i64| matches!(locations[v.index()], Location::Imm(m) if (0..=mask).contains(&m));
    for block in order {
        for &inst in &func.blocks[block.index()].insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            let z = match data.op {
                Op::Load(..) | Op::StackLoad(..) | Op::AtomicLoad(_) | Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) | Op::Convert(Convert::Uextend, _) => true,
                Op::Binary(Binary::Band, a, b) => match func.types[res.index()] {
                    Type::I8 => out[a.index()] || out[b.index()] || fits_mask(a, 0xff) || fits_mask(b, 0xff),
                    Type::I16 => out[a.index()] || out[b.index()] || fits_mask(a, 0xffff) || fits_mask(b, 0xffff),
                    _ => false,
                },
                Op::Convert(Convert::Ireduce, shifted) => {
                    let src_ty = func.types[shifted.index()];
                    let dst_bits = func.types[res.index()].bits();
                    matches!(src_ty, Type::I64 | Type::I32)
                        && census.def_inst[shifted.index()].is_some_and(|def| {
                            matches!(func.insts[def.index()].op, Op::Binary(Binary::Ushr, _, c)
                                if matches!(locations[c.index()], Location::Imm(shift) if (0..i64::from(src_ty.bits())).contains(&shift) && src_ty.bits() - (shift as u32) <= dst_bits))
                        })
                }
                _ => false,
            };
            out[res.index()] = z;
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
    let (sib, mut elided) = if regs.fold_sib { sib_addrs(func, &locations, &census) } else { (vec![None; count], vec![false; count]) };
    let through = if regs.absolute_calls { through(func, &locations, &census, &mut elided) } else { vec![None; count] };
    let promoted_consts = if weighed { promote_loop_consts(func, &census, &elided, &mut locations) } else { Vec::new() };
    let wanted: Vec<bool> = (0..count).map(|value| !elided[value] && matches!(locations[value], Location::Slot) && registered(regs, func.types[value])).collect();
    let tracked: Vec<bool> = (0..count).map(|value| !elided[value] && (wanted[value] || (weighed && matches!(locations[value], Location::Slot)))).collect();
    let live = live(func, order, copied, &tracked, &sib, &elided);
    let candidates: Vec<bool> = (0..count).map(|value| eligible(&live, &wanted, value)).collect();
    let fused: Vec<bool> = (0..count).map(|value| candidates[value] && fuses(func, &census, value, regs.float_branches)).collect();
    let ints: Vec<bool> = (0..count).map(|value| candidates[value] && func.types[value].is_small_int() && !fused[value] && through[value].is_none()).collect();
    let floats: Vec<bool> = (0..count).map(|value| candidates[value] && !fused[value] && (func.types[value].is_float() || (regs.vectors && func.types[value] == Type::V128))).collect();
    let reuse = if regs.two_address { reusable(func, &live, &elided) } else { vec![None; count] };
    let blocks = weighed.then(|| block_weights(func));
    let weights = blocks.as_deref().map(|blocks| weights(func, blocks, &sib, &elided));
    let partners = weighed.then(|| partners(func));
    let park = blocks.as_deref().filter(|_| regs.park).map(|blocks| live.sites.iter().map(|(_, block)| blocks[block.index()]).collect::<Vec<u64>>());
    let hints = Hints { reuse: &reuse, weights: weights.as_deref(), partners: partners.as_deref(), park: park.as_deref() };
    let (passed_ints, passed_floats) = passed(func, classify);
    let (homes, used) = allocate(func, &live, &int_class(func, regs, order, params, passed_ints), &ints, &hints);
    place(&mut locations, homes, Location::Reg);
    for (v, imm) in promoted_consts {
        if !matches!(locations[v], Location::Reg(_)) {
            locations[v] = Location::Imm(imm);
        }
    }
    let (float_homes, used_floats) = allocate(func, &live, &float_class(regs, params, passed_floats), &floats, &hints);
    place(&mut locations, float_homes, Location::Float);
    let parked = if park.is_some() { parked(func, &live, &locations, (regs.saved, regs.saved_float), classify) } else { Vec::new() };
    let mut homed = vec![false; if parked.is_empty() { 0 } else { count }];
    parked.iter().flat_map(|(_, values)| values).for_each(|(value, _, _)| homed[value.index()] = true);
    let (pairs, deferred) = pairs(func, &locations, &elided);
    let shared = if weighed { crate::color::share(func, &live, &locations, &deferred) } else { Vec::new() };
    let zero_ext = zero_ext(func, order, &locations, &census);
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
        match self.location(value) {
            Location::Imm(imm) => Some(imm),
            _ => None,
        }
    }

    pub(crate) fn reg_of(&self, value: Value) -> Option<R> {
        match self.location(value) {
            Location::Reg(reg) => Some(reg),
            _ => None,
        }
    }

    pub(crate) fn parked_at(&self, inst: Inst) -> &[(Value, ParkReg<R>, bool)] {
        self.parked.binary_search_by_key(&inst.index(), |(at, _)| at.index()).map_or(&[], |at| &self.parked[at].1)
    }

    pub(crate) fn float_of(&self, value: Value) -> Option<u8> {
        match self.location(value) {
            Location::Float(reg) => Some(reg),
            _ => None,
        }
    }
}

impl<R: Copy + PartialEq> Plan<R> {
    pub(crate) fn same_home(&self, frame: &crate::frame::Frame<R>, a: Value, b: Value) -> bool {
        a == b
            || matches!((self.location(a), self.location(b)), (Location::Reg(x), Location::Reg(y)) if x == y)
            || matches!((self.location(a), self.location(b)), (Location::Float(x), Location::Float(y)) if x == y)
            || frame.shares(self, a, b)
    }

    pub(crate) fn moves(&self, func: &Function, frame: &crate::frame::Frame<R>, block: Block, args: &[Value]) -> bool {
        args.iter().zip(&func.blocks[block.index()].params).any(|(arg, param)| !self.same_home(frame, *arg, *param))
    }
}
