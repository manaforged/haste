pub(crate) mod consts;
mod demote;
mod hints;
pub(crate) mod hoist;
pub(crate) mod walk;

use crate::cfg::{Cfg, Loops};
use crate::fold::{constant, holds};
use crate::frame::{Abi, Loc};
use crate::ir::{ArgumentPurpose, Binary, Block, Convert, FloatCC, Function, Inst, InstData, Op, Signature, StackSlot, Type, Value};
use crate::live::{Elided, Live, Sib, SibBase, live, results};
use crate::park::{Class, Hints, ParkReg, allocate};

pub(crate) struct Regs<R: 'static> {
    pub saved: &'static [R],
    pub saved_float: &'static [u8],
    pub leaf: &'static [R],
    pub wide_scratch: &'static [R],
    pub op_scratch: &'static [R],
    pub preserved_clobbers: &'static [R],
    pub scratch_mask: fn(&Function, &Op) -> u8,
    pub dict: Option<R>,
    pub float: &'static [u8],
    pub float_branches: &'static [FloatCC],
    pub vectors: bool,
    pub absolute_calls: bool,
    /// The backend calls a runtime helper or a symbol in place instead of loading its address first.
    pub constant_calls: bool,
    pub two_address: bool,
    pub park: bool,
    pub fold_sib: bool,
    /// The backend zero-extends 8- and 16-bit call results at the call, and 8- and 16-bit narrowing copies between two registers.
    pub call_zext: bool,
    /// The instructions that store a constant into a frame home.
    pub imm_store: fn(i64) -> u64,
    /// The register a 64-bit high product lands in, which the product keeps as its home when nothing else needs it.
    pub high_product: Option<R>,
}

pub(crate) const FOLD_LIMIT: i64 = 1 << 24;

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
        if wide && self.wide_scratch.contains(&reg) {
            return true;
        }
        let mask = if wide { u8::MAX } else { (self.scratch_mask)(func, &data.op) };
        self.op_scratch.iter().enumerate().any(|(i, r)| (mask >> i) & 1 != 0 && *r == reg)
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
    Runtime(crate::reloc::RuntimeSymbol),
    Symbol(u32),
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
    pub density: Vec<u64>,
    /// The summed block-weighted accesses of the register candidates left in stack slots.
    pub spill_cost: u64,
    pub hoist: hoist::Hoist<R>,
}

fn registered<R>(regs: &Regs<R>, ty: Type) -> bool {
    ty.is_small_int() || ((ty.is_float() || (regs.vectors && ty.is_vector())) && !regs.float.is_empty())
}

type Position = Option<(usize, usize)>;

pub(super) struct Uses {
    pub(super) uses: Vec<u32>,
    last_use: Vec<Position>,
    pub(super) def: Vec<Position>,
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

/// Extracts `(base, delta)` from an integer `Iadd`/`Isub` with one constant operand supplied by `imm_of`.
pub(crate) fn iadd_sub_const(op: &Op, imm_of: impl Fn(Value) -> Option<i64>) -> Option<(Value, i64)> {
    match *op {
        Op::Binary(Binary::Iadd, a, b) => match (imm_of(a), imm_of(b)) {
            (None, Some(c)) => Some((a, c)),
            (Some(c), None) => Some((b, c)),
            _ => None,
        },
        Op::Binary(Binary::Isub, a, b) if imm_of(a).is_none() => imm_of(b).and_then(i64::checked_neg).map(|c| (a, c)),
        _ => None,
    }
}

/// Counts atomic memory address uses of each value.
pub(crate) fn atomic_addr_uses(func: &Function) -> Vec<u32> {
    let mut atomic_uses = vec![0_u32; func.types.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            let addr = match func.insts[inst.index()].op {
                Op::AtomicLoad(addr, _) => Some(addr),
                Op::AtomicStore(val, addr, _) if val != addr => Some(addr),
                Op::AtomicRmw(_, _, addr, val) if val != addr => Some(addr),
                Op::AtomicCas(_, addr, exp, rep) if exp != addr && rep != addr => Some(addr),
                _ => None,
            };
            if let Some(addr) = addr {
                atomic_uses[addr.index()] += 1;
            }
        }
    }
    atomic_uses
}

/// Returns which `I64` `Iadd`/`Isub` constant-offset values are used exclusively as atomic memory addresses and can fold directly into x64 atomic operands.
pub(crate) fn atomic_elidable_addrs(func: &Function, defs: &[Option<Inst>], uses: &[u32]) -> Vec<bool> {
    let consts = func.const_values();
    let atomic_uses = atomic_addr_uses(func);
    let mut out = vec![false; func.types.len()];
    for (v, &def) in defs.iter().enumerate() {
        let Some(def_inst) = def else { continue };
        if atomic_uses[v] == 0 || atomic_uses[v] != uses[v] || func.types[v] != Type::I64 {
            continue;
        }
        if let Some((base, delta)) = iadd_sub_const(&func.insts[def_inst.index()].op, |x| consts[x.index()])
            && base.index() != v
            && i32::try_from(delta).is_ok()
            && func.ty(base) == Type::I64
        {
            out[v] = true;
        }
    }
    out
}

fn atomic_addrs<R>(func: &Function, locations: &[Location<R>], census: &Uses, sib: &[Option<Sib>], elided: &mut [bool]) -> Vec<Option<Value>> {
    let count = func.types.len();
    let mut atomic_base = vec![None; count];
    let atomic_uses = atomic_addr_uses(func);
    for (v, &def) in census.def_inst.iter().enumerate() {
        let Some(def_inst) = def else { continue };
        if atomic_uses[v] == 0
            || atomic_uses[v] != census.uses[v]
            || elided[v]
            || sib[v].is_some()
            || func.types[v] != Type::I64
            || !matches!(locations[v], Location::Slot)
        {
            continue;
        }
        let Some((base, delta)) = iadd_sub_const(&func.insts[def_inst.index()].op, |x| match locations[x.index()] {
            Location::Imm(c) => Some(c),
            _ => None,
        }) else {
            continue;
        };
        if base.index() == v || i32::try_from(delta).is_err() || elided[base.index()] || sib[base.index()].is_some() || func.ty(base) != Type::I64 {
            continue;
        }
        match locations[base.index()] {
            Location::Slot => {
                elided[v] = true;
                atomic_base[v] = Some(base);
            }
            Location::Frame(_, off) if (-FOLD_LIMIT..=FOLD_LIMIT).contains(&(i64::from(off) + delta)) => {
                elided[v] = true;
            }
            _ => {}
        }
    }
    atomic_base
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

fn through<R>(func: &Function, locations: &[Location<R>], census: &Uses, elided: &mut [bool], runtime: bool) -> Vec<Option<CallTarget>> {
    let mut through = vec![None; func.types.len()];
    let mut loads = vec![None; func.types.len()];
    for data in &func.insts {
        let Some(result) = data.result else { continue };
        loads[result.index()] = match data.op {
            Op::Load(addr, 0) => match locations[addr.index()] {
                Location::Imm(at) => i32::try_from(at).ok().filter(|at| *at > 0).map(CallTarget::Abs),
                Location::SlotAddr(sym) => Some(CallTarget::Slot(sym)),
                _ => None,
            }
            .map(|target| (target, Some(addr))),
            Op::Runtime(rt) if runtime => Some((CallTarget::Runtime(rt), None)),
            Op::Symbol(sym) if runtime && matches!(locations[result.index()], Location::Symbol(_)) => Some((CallTarget::Symbol(sym), None)),
            _ => None,
        };
    }
    let mut called = vec![0u32; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } = &func.insts[inst.index()].op && !args.contains(callee) {
            called[callee.index()] += 1;
        }
    }
    let mut left = census.uses.clone();
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } = &func.insts[inst.index()].op
            && !args.contains(callee)
            && let Some((target, addr)) = loads[callee.index()]
            && (census.uses[callee.index()] == 1 || (addr.is_none() && census.uses[callee.index()] == called[callee.index()]))
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

fn const_pairs(func: &Function, census: &Uses, elided: &mut [bool]) {
    let mut kept = vec![false; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let op = &func.insts[inst.index()].op;
        if !crate::wide::reads_halves(func, op) { op.for_operands(|value| kept[value.index()] = true); }
    }
    let constant = |value: Value| census.def_inst[value.index()].is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::Const(_)));
    for (value, def) in census.def_inst.iter().enumerate() {
        if let Some(inst) = def && let Op::Binary(Binary::Iconcat, lo, hi) = func.insts[inst.index()].op && !kept[value] && constant(lo) && constant(hi) {
            elided[value] = true;
        }
    }
}

fn eligible(live: &Live, wanted: &[bool], value: usize) -> bool {
    wanted[value] && !live.ranges_of(value).is_empty() && !crate::park::crosses_ranges(&live.unwinds, live.ranges_of(value))
}

fn preserved_clobber_ranges(preserved_calls: &[u32]) -> Vec<crate::live::Range> {
    preserved_calls.iter().map(|&at| crate::live::Range { start: at - 1, end: at - 1 }).collect()
}

fn int_class<'a, R: Copy + PartialEq>(
    func: &Function,
    regs: &'a Regs<R>,
    (order, copied, elided): (&[Block], &[bool], &[bool]),
    params: &[Loc<R>],
    passed: Vec<Option<(R, bool)>>,
    preserved_calls: &[u32],
) -> Class<'a, R> {
    let dict = func.signature.params.iter().any(|param| param.purpose == ArgumentPurpose::Dict);
    let leaf = regs.leaf.iter().filter(|reg| !(dict && regs.dict == Some(**reg))).copied().collect();
    let mut clobbers = crate::park::scratch_clobbers(func, (regs.wide_scratch, regs.op_scratch, regs.scratch_mask), (order, copied, elided));
    if !preserved_calls.is_empty() && !regs.preserved_clobbers.is_empty() {
        clobbers.push((regs.preserved_clobbers, preserved_clobber_ranges(preserved_calls)));
    }
    let arrivals = params.iter().map(|loc| match *loc { Loc::Int(reg) => vec![reg], Loc::Pair(low, high) => vec![low, high], _ => Vec::new() }).collect();
    Class { saved: regs.saved, leaf, arrivals, passed, clobbers }
}

/// Returns `(lookup, [slot, size, init])` when a non-try `ThreadCell` call has static arguments eligible for inline TLS with register-preserving cold miss stub.
pub(crate) fn cold_thread_cell<R>(
    func: &Function,
    (callee, args): (Value, &[Value]),
    through: &[Option<CallTarget>],
    locations: &[Location<R>],
    elided: &[bool],
) -> Option<(u32, [Value; 3])> {
    let Some(CallTarget::Symbol(lookup)) = through[callee.index()] else { return None };
    if !func.is_thread_cell(callee) {
        return None;
    }
    let [slot, size, init] = *args else { return None };
    let static_ok = |v: Value| !elided[v.index()] && matches!(locations[v.index()], Location::Imm(_) | Location::Symbol(_) | Location::SlotAddr(_));
    if !(static_ok(slot) && static_ok(size) && static_ok(init)) {
        return None;
    }
    matches!(locations[slot.index()], Location::Symbol(_) | Location::Imm(0)).then_some((lookup, [slot, size, init]))
}

pub(crate) fn cold_thread_cell_op<R>(
    func: &Function,
    op: &Op,
    through: &[Option<CallTarget>],
    locations: &[Location<R>],
    elided: &[bool],
) -> Option<(u32, [Value; 3])> {
    let Op::Call { callee, args, .. } = op else { return None };
    cold_thread_cell(func, (*callee, args), through, locations, elided)
}

fn float_class<R>(regs: &Regs<R>, params: &[Loc<R>], passed: Vec<Option<(u8, bool)>>, preserved_calls: &[u32]) -> Class<'static, u8> {
    let arrivals = params.iter().map(|loc| match *loc { Loc::Float(reg) => vec![reg], _ => Vec::new() }).collect();
    let clobbers = if preserved_calls.is_empty() || regs.float.is_empty() {
        Vec::new()
    } else {
        vec![(regs.float, preserved_clobber_ranges(preserved_calls))]
    };
    Class { saved: regs.saved_float, leaf: regs.float.to_vec(), arrivals, passed, clobbers }
}

const BASE_WEIGHT: u64 = 8;
const LOOP_WEIGHT: u64 = 16;
const MAX_DEPTH: u32 = 4;
const SLOW_SHARE: u64 = 16;

fn slow_blocks(func: &Function, cfg: &Cfg, loops: &Loops) -> (Vec<bool>, Vec<bool>) {
    let count = func.blocks.len();
    let calls: Vec<bool> = func.blocks.iter().map(|data| data.insts.iter().any(|inst| matches!(func.insts[inst.index()].op, Op::Call { .. } | Op::TryCall { .. } | Op::Copy(..)))).collect();
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

fn block_weights(depths: &[u32], (cold, slow): (&[bool], &[bool])) -> Vec<u64> {
    depths
        .iter()
        .enumerate()
        .map(|(i, &depth)| {
            if cold[i] {
                1
            } else {
                (BASE_WEIGHT * LOOP_WEIGHT.pow(depth.min(MAX_DEPTH)) / if slow[i] { SLOW_SHARE } else { 1 }).max(1)
            }
        })
        .collect()
}

fn place<R, F>(locations: &mut [Location<R>], homes: Vec<Option<F>>, wrap: impl Fn(F) -> Location<R>) {
    for (location, home) in locations.iter_mut().zip(homes) {
        if let Some(reg) = home {
            *location = wrap(reg);
        }
    }
}

pub(crate) fn plan<R: Copy + PartialEq>(
    func: &Function,
    regs: &Regs<R>,
    (order, copied, weighed, preserve_tc): (&[Block], &[bool], bool, bool),
    (params, split): (&[Loc<R>], &[(Value, u64)]),
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Plan<R> {
    let count = func.types.len();
    let mut locations = vec![Location::Slot; count];
    consts(func, regs.absolute_calls, &mut locations);
    statics(func, &mut locations);
    let census = census(func);
    let zero_ext = hints::zero_ext(func, &locations, &census, regs.call_zext);
    let (sib, mut elided) = if regs.fold_sib { sib_addrs(func, &locations, &census, &zero_ext, regs.two_address) } else { (vec![None; count], vec![false; count]) };
    let atomic_base = if regs.two_address { atomic_addrs(func, &locations, &census, &sib, &mut elided) } else { vec![None; count] };
    let through = if regs.absolute_calls { through(func, &locations, &census, &mut elided, regs.constant_calls) } else { vec![None; count] };
    const_pairs(func, &census, &mut elided);
    let eff = Elided { sib: &sib, atomic_base: &atomic_base, elided: &elided };
    let cfg = weighed.then(|| func.cfg());
    let loops = cfg.as_ref().map(|cfg| cfg.loops());
    let loop_depths = loops.as_ref().map(|loops| &loops.depths[..]);
    let promoted_consts = loops.as_deref().map(|loops| consts::promote_loop_consts(func, &census, loops, &elided, regs.two_address, &mut locations)).unwrap_or_default();
    let promoted_syms = regs.absolute_calls.then(|| consts::promote_loop_syms(func, &census, loop_depths, &elided, &mut locations)).unwrap_or_default();
    let wanted: Vec<bool> = (0..count).map(|value| !elided[value] && matches!(locations[value], Location::Slot) && registered(regs, func.types[value])).collect();
    let tracked: Vec<bool> = (0..count).map(|value| !elided[value] && (wanted[value] || (weighed && matches!(locations[value], Location::Slot)))).collect();
    let preserved_tc: Vec<bool> = if preserve_tc && func.has_thread_cell() {
        func.insts.iter().map(|d| cold_thread_cell_op(func, &d.op, &through, &locations, &elided).is_some()).collect()
    } else {
        Vec::new()
    };
    let live = live(func, order, copied, &tracked, eff, &preserved_tc);
    let candidates: Vec<bool> = (0..count).map(|value| eligible(&live, &wanted, value)).collect();
    let fused: Vec<bool> = (0..count).map(|value| candidates[value] && fuses(func, &census, value, regs.float_branches, regs.two_address)).collect();
    let ints: Vec<bool> = (0..count).map(|value| candidates[value] && func.types[value].is_small_int() && !fused[value] && through[value].is_none()).collect();
    let floats: Vec<bool> = (0..count).map(|value| candidates[value] && !fused[value] && (func.types[value].is_float() || (regs.vectors && func.types[value].is_vector()))).collect();
    let reuse = hints::reusable(func, regs, &locations, &live, &elided);
    let blocks = cfg.as_deref().zip(loops).map(|(cfg, loops)| {
        let (cold, slow) = slow_blocks(func, cfg, loops);
        block_weights(&loops.depths, (&cold, &slow))
    });
    let natural = blocks.as_deref().map(|blocks| walk::weights(func, blocks, eff));
    let weights = natural.clone().map(|natural| walk::inherit(natural, &live, split));
    let (partners, back_args, edge_args) = walk::edge_hints(func, cfg.as_deref(), order, blocks.as_deref());
    let park = blocks.as_deref().filter(|_| regs.park).map(|blocks| live.sites.iter().map(|(_, block)| blocks[block.index()]).collect::<Vec<u64>>());
    let copies = if regs.two_address { hints::extend_copies(func, &live, &zero_ext, &ints) } else { Vec::new() };
    let hints = Hints { reuse: &reuse, weights: weights.as_deref(), partners: partners.as_deref(), back_args: &back_args, edge_args: &edge_args, park: park.as_deref(), two_address: regs.two_address, copies: &copies };
    let (passed_ints, passed_floats) = hints::passed(func, classify, regs.high_product);
    let (homes, used) = allocate(func, &live, &int_class(func, regs, (order, copied, &elided), params, passed_ints, &live.preserved_calls), &ints, &hints);
    place(&mut locations, homes, Location::Reg);
    for (v, loc) in promoted_consts.into_iter().map(|(v, i)| (v, Location::Imm(i))).chain(promoted_syms) {
        if !matches!(locations[v], Location::Reg(_)) {
            locations[v] = loc;
        }
    }
    let (float_homes, used_floats) = allocate(func, &live, &float_class(regs, params, passed_floats, &live.preserved_calls), &floats, &hints);
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
    let webs = blocks.as_deref().zip(partners.as_deref()).map(|edges| demote::webs(func, &live, edges, (eff, &deferred), regs.imm_store, &mut locations));
    let parked = if park.is_some() { parked(func, &live, &locations, (regs.saved, regs.saved_float), classify) } else { Vec::new() };
    let mut homed = vec![false; if parked.is_empty() { 0 } else { count }];
    parked.iter().flat_map(|(_, values)| values).for_each(|(value, _, _)| homed[value.index()] = true);
    let shared = webs.map_or_else(Vec::new, |webs| crate::color::share(func, &live, &locations, &deferred, &webs));
    let zero_ext = if regs.call_zext && hints::has_reg_reduce(func, &locations) { hints::zero_ext(func, &locations, &census, true) } else { zero_ext };
    let density = weights.as_deref().map_or_else(Vec::new, |weights| walk::densities(weights, &live));
    if crate::trace::enabled() {
        crate::trace::on_alloc(func, &live, (0..count).filter(|&value| ints[value]).map(|value| (value, matches!(locations[value], Location::Slot))));
        if let (Some(loops), Some(blocks)) = (loops, blocks.as_deref()) {
            crate::trace::on_loops(func, &live, (loops, blocks), &ints, |value| matches!(locations[value], Location::Slot));
            crate::trace::on_edge_copies(func, &live, blocks, |value| if let Location::Reg(reg) = locations[value] { Some(reg) } else { None });
        }
    }
    let spill_cost = natural.map_or(0, |natural| (0..count).filter(|&v| (ints[v] || floats[v]) && matches!(locations[v], Location::Slot)).map(|v| natural[v]).sum());
    let hoist = hoist::Hoist::new(func, &live, &locations);
    Plan { locations, fused, used, used_floats, pairs, deferred, through, shared, parked, homed, sib, elided, zero_ext, density, spill_cost, hoist }
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

pub(crate) fn used_preserved_leaf_regs<R: Copy + PartialEq>(regs: &Regs<R>, locations: &[Location<R>], include_dict: bool) -> Vec<R> {
    let mut eligible = [false; 32];
    let mut need_mask = 0u32;
    for (i, &r) in regs.leaf.iter().enumerate() {
        if !regs.preserved_clobbers.contains(&r) {
            eligible[i] = true;
            need_mask |= 1u32 << i;
        }
    }
    let mut used_mask = 0u32;
    if include_dict
        && let Some(dict) = regs.dict
        && let Some(i) = regs.leaf.iter().position(|&r| r == dict)
        && eligible[i]
    {
        used_mask |= 1u32 << i;
    }
    if used_mask != need_mask {
        for loc in locations {
            if let Location::Reg(at) = *loc
                && let Some(i) = regs.leaf.iter().position(|&r| r == at)
                && eligible[i]
            {
                used_mask |= 1u32 << i;
                if used_mask == need_mask {
                    break;
                }
            }
        }
    }
    regs.leaf.iter().enumerate().filter(|&(i, _)| (used_mask & (1u32 << i)) != 0).map(|(_, &r)| r).collect()
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
        if ranges.is_empty() { continue; }
        let value = Value(index as u32);
        for r in ranges {
            let first = live.calls.partition_point(|at| *at <= r.start);
            let last = live.calls.partition_point(|at| *at <= r.end);
            for call in first..last {
                let at = live.calls[call];
                if at < r.end || !passed_own(func, live.sites[call].0, value, park_reg, classify) { at_call[call].push((value, park_reg, at < r.end)); }
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
