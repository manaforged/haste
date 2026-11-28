use crate::ir::{Binary, Convert, Function, Inst, Op, Type, Value};
use crate::live::{Sib, SibBase};

use super::hints::{rcpc2_atomic_uses, rcpc2_delta_fits};
use super::{AtomicAddrMode, FOLD_LIMIT, Location, Uses};

#[derive(Clone, Copy)]
struct MemUses {
    count: u32,
    min_off: i32,
    max_off: i32,
    size_mask: u8,
    cross_block: bool,
    max_depth: u32,
}

fn mem_uses(func: &Function, census: &Uses, loop_depths: Option<&[u32]>) -> Vec<MemUses> {
    let mut out = vec![
        MemUses {
            count: 0,
            min_off: i32::MAX,
            max_off: i32::MIN,
            size_mask: 0,
            cross_block: false,
            max_depth: 0,
        };
        func.types.len()
    ];
    for (b, data) in func.blocks.iter().enumerate() {
        let depth = loop_depths.map_or(0, |d| d[b]);
        for &inst in &data.insts {
            let (addr, off, bytes) = match (
                &func.insts[inst.index()].op,
                func.insts[inst.index()].result,
            ) {
                (&Op::Load(a, off), Some(r)) => (a, off, func.ty(r).bytes()),
                (&Op::Store(val, a, off), _) if val != a => (a, off, func.ty(val).bytes()),
                _ => continue,
            };
            let info = &mut out[addr.index()];
            info.count += 1;
            info.min_off = info.min_off.min(off);
            info.max_off = info.max_off.max(off);
            info.size_mask |= match bytes {
                1 => 1,
                2 => 2,
                4 => 4,
                8 => 8,
                _ => 16,
            };
            info.cross_block |= census.def[addr.index()].is_some_and(|(db, _)| db != b);
            info.max_depth = info.max_depth.max(depth);
        }
    }
    out
}

struct SibCtx<'a, R> {
    func: &'a Function,
    locations: &'a [Location<R>],
    census: &'a Uses,
    zero_ext: &'a [bool],
}

type SibOperands = (SibBase, i32, Value, u8, Option<Value>, Option<Value>, i64);

impl<R> SibCtx<'_, R> {
    fn slot_i64(&self, v: Value) -> bool {
        self.func.ty(v) == Type::I64 && matches!(self.locations[v.index()], Location::Slot)
    }

    fn single_i64(&self, v: Value, block: usize) -> bool {
        self.slot_i64(v)
            && self.census.uses[v.index()] == 1
            && self.census.def[v.index()].is_some_and(|(b, _)| b == block)
    }

    fn unwrap_idx(&self, v: Value, block: usize) -> Option<(Value, Option<Value>)> {
        let inner = self
            .single_i64(v, block)
            .then(|| self.census.def_inst[v.index()])
            .flatten()
            .and_then(|def| match self.func.insts[def.index()].op {
                Op::Convert(Convert::Uextend, inner)
                    if self.func.ty(inner) == Type::I32
                        && self.zero_ext[inner.index()]
                        && matches!(self.locations[inner.index()], Location::Slot) =>
                {
                    Some(inner)
                }
                _ => None,
            });
        inner
            .map(|i| (i, Some(v)))
            .or_else(|| self.slot_i64(v).then_some((v, None)))
    }

    fn parse_base(&self, v: Value) -> Option<(SibBase, i32)> {
        match self.locations[v.index()] {
            Location::Frame(slot, off) => Some((SibBase::Frame(slot), off)),
            Location::Slot if self.func.ty(v) == Type::I64 => Some((SibBase::Value(v), 0)),
            _ => None,
        }
    }

    fn offset_idx(&self, v: Value, block: usize) -> Option<(Value, i64, Value)> {
        let def = self
            .single_i64(v, block)
            .then(|| self.census.def_inst[v.index()])??;
        let (inner, c) = iadd_sub_const(&self.func.insts[def.index()].op, |x| {
            match self.locations[x.index()] {
                Location::Imm(c) => Some(c),
                _ => None,
            }
        })?;
        (inner != v && self.slot_i64(inner)).then_some((inner, c, v))
    }

    fn parse_scaled(
        &self,
        v: Value,
        block: usize,
    ) -> Option<(Value, u8, Value, Option<Value>, i64)> {
        let def = self
            .single_i64(v, block)
            .then(|| self.census.def_inst[v.index()])??;
        match self.func.insts[def.index()].op {
            Op::Binary(Binary::Ishl, idx, sh) => {
                let Location::Imm(s @ 1..=3) = self.locations[sh.index()] else {
                    return None;
                };
                if let Some((inner, c, folded)) = self.offset_idx(idx, block)
                    && let Some(disp) = c.checked_mul(1 << s)
                    && (-FOLD_LIMIT..=FOLD_LIMIT).contains(&disp)
                {
                    return Some((inner, s as u8, v, Some(folded), disp));
                }
                self.unwrap_idx(idx, block)
                    .map(|(b, e)| (b, s as u8, v, e, 0))
            }
            Op::Binary(Binary::Imul, a, b) => [(a, b), (b, a)].into_iter().find_map(|(idx, sc)| {
                let Location::Imm(m @ (2 | 4 | 8)) = self.locations[sc.index()] else {
                    return None;
                };
                self.unwrap_idx(idx, block)
                    .map(|(b, e)| (b, m.trailing_zeros() as u8, v, e, 0))
            }),
            _ => None,
        }
    }

    fn peel_outer_disp(
        &self,
        block: usize,
        left: Value,
        right: Value,
    ) -> (i64, Option<Value>, Value, Value) {
        for (inner, imm_v) in [(left, right), (right, left)] {
            if let Location::Imm(imm) = self.locations[imm_v.index()]
                && (-FOLD_LIMIT..=FOLD_LIMIT).contains(&imm)
                && self.single_i64(inner, block)
                && let Some(inner_inst) = self.census.def_inst[inner.index()]
                && let Op::Binary(Binary::Iadd, l2, r2) = self.func.insts[inner_inst.index()].op
            {
                return (imm, Some(inner), l2, r2);
            }
        }
        (0, None, left, right)
    }

    fn match_operands(&self, block: usize, left: Value, right: Value) -> Option<SibOperands> {
        [(left, right), (right, left)]
            .into_iter()
            .find_map(|(b, i)| {
                self.parse_base(b).zip(self.parse_scaled(i, block)).map(
                    |((base, bd), (idx, sh, sv, ev, id))| (base, bd, idx, sh, Some(sv), ev, id),
                )
            })
            .or_else(|| {
                [(left, right), (right, left)]
                    .into_iter()
                    .find_map(|(b, i)| {
                        self.parse_base(b)
                            .zip(self.unwrap_idx(i, block))
                            .map(|((base, bd), (idx, ev))| (base, bd, idx, 0, None, ev, 0))
                    })
            })
    }
}

fn fits_sib_offset(base: SibBase, total: i64, disp: i32, off: i32) -> bool {
    match base {
        SibBase::Frame(_) => (-FOLD_LIMIT..=FOLD_LIMIT).contains(&(total + i64::from(off))),
        SibBase::Value(_) => disp
            .checked_add(off)
            .and_then(|d| d.checked_add(16))
            .is_some(),
    }
}

fn is_hw_indexed(base: SibBase, total: i64, info: MemUses, shift: u8) -> bool {
    total + i64::from(info.min_off) == 0
        && total + i64::from(info.max_off) == 0
        && (info.size_mask & 16) == 0
        && (shift == 0 || info.size_mask == (1 << shift))
        && !matches!(base, SibBase::Frame(_))
}

pub(super) fn sib_addrs<R>(
    func: &Function,
    locations: &[Location<R>],
    census: &Uses,
    zero_ext: &[bool],
    loop_depths: Option<&[u32]>,
    two_address: bool,
) -> (Vec<Option<Sib>>, Vec<bool>) {
    let count = func.types.len();
    let (mut sib, mut elided) = (vec![None; count], vec![false; count]);
    let uses_of = mem_uses(func, census, loop_depths);
    let ctx = SibCtx {
        func,
        locations,
        census,
        zero_ext,
    };
    for (block, data) in func.blocks.iter().enumerate() {
        let def_depth = loop_depths.map_or(0, |d| d[block]);
        for &inst in &data.insts {
            let Some(addr) = func.insts[inst.index()].result else {
                continue;
            };
            let info = uses_of[addr.index()];
            if info.count == 0
                || info.count != census.uses[addr.index()]
                || !ctx.slot_i64(addr)
                || (info.cross_block && info.max_depth > def_depth)
            {
                continue;
            }
            let Op::Binary(Binary::Iadd, left, right) = func.insts[inst.index()].op else {
                continue;
            };
            let (outer_disp, peeled_add, left, right) = ctx.peel_outer_disp(block, left, right);
            let Some((base, base_disp, index, shift, scaled_val, ext_val, index_disp)) =
                ctx.match_operands(block, left, right)
            else {
                continue;
            };
            let total = outer_disp + i64::from(base_disp) + index_disp;
            if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&total) {
                continue;
            }
            let disp = total as i32;
            if !fits_sib_offset(base, total, disp, info.min_off)
                || !fits_sib_offset(base, total, disp, info.max_off)
            {
                continue;
            }
            if !two_address
                && (info.count > 1 || info.cross_block)
                && !is_hw_indexed(base, total, info, shift)
            {
                continue;
            }
            sib[addr.index()] = Some(Sib {
                base,
                index,
                shift,
                disp,
            });
            elided[addr.index()] = true;
            [peeled_add, scaled_val, ext_val]
                .into_iter()
                .flatten()
                .for_each(|e| elided[e.index()] = true);
        }
    }
    (sib, elided)
}

/// Extracts `(base, delta)` from an integer `Iadd`/`Isub` with one constant operand supplied by `imm_of`.
pub(crate) fn iadd_sub_const(
    op: &Op,
    imm_of: impl Fn(Value) -> Option<i64>,
) -> Option<(Value, i64)> {
    match *op {
        Op::Binary(Binary::Iadd, a, b) => match (imm_of(a), imm_of(b)) {
            (None, Some(c)) => Some((a, c)),
            (Some(c), None) => Some((b, c)),
            _ => None,
        },
        Op::Binary(Binary::Isub, a, b) if imm_of(a).is_none() => {
            imm_of(b).and_then(i64::checked_neg).map(|c| (a, c))
        }
        _ => None,
    }
}

fn atomic_addr_uses(func: &Function) -> Vec<u32> {
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
pub(crate) fn atomic_elidable_addrs(
    func: &Function,
    defs: &[Option<Inst>],
    uses: &[u32],
) -> Vec<bool> {
    let consts = func.const_values();
    let atomic_uses = atomic_addr_uses(func);
    let mut out = vec![false; func.types.len()];
    for (v, &def) in defs.iter().enumerate() {
        let Some(def_inst) = def else { continue };
        if atomic_uses[v] == 0 || atomic_uses[v] != uses[v] || func.types[v] != Type::I64 {
            continue;
        }
        if let Some((base, delta)) =
            iadd_sub_const(&func.insts[def_inst.index()].op, |x| consts[x.index()])
            && base.index() != v
            && i32::try_from(delta).is_ok()
            && func.ty(base) == Type::I64
        {
            out[v] = true;
        }
    }
    out
}

pub(super) fn atomic_addrs<R>(
    func: &Function,
    mode: AtomicAddrMode,
    locations: &[Location<R>],
    census: &Uses,
    sib: &[Option<Sib>],
    elided: &mut [bool],
) -> Vec<Option<Value>> {
    let count = func.types.len();
    let mut atomic_base = vec![None; count];
    let (full_uses, rcpc2_uses) = match mode {
        AtomicAddrMode::None => return atomic_base,
        AtomicAddrMode::Full => (atomic_addr_uses(func), Vec::new()),
        AtomicAddrMode::Rcpc2 => (Vec::new(), rcpc2_atomic_uses(func)),
    };
    for (v, &def) in census.def_inst.iter().enumerate() {
        let Some(def_inst) = def else { continue };
        let (use_count, mask) = match mode {
            AtomicAddrMode::Full => (full_uses[v], 0),
            AtomicAddrMode::Rcpc2 => rcpc2_uses[v],
            AtomicAddrMode::None => unreachable!(),
        };
        if use_count == 0
            || use_count != census.uses[v]
            || elided[v]
            || sib[v].is_some()
            || func.types[v] != Type::I64
            || !matches!(locations[v], Location::Slot)
        {
            continue;
        }
        let Some((base, delta)) = iadd_sub_const(&func.insts[def_inst.index()].op, |x| {
            match locations[x.index()] {
                Location::Imm(c) => Some(c),
                _ => None,
            }
        }) else {
            continue;
        };
        if base.index() == v
            || i32::try_from(delta).is_err()
            || elided[base.index()]
            || sib[base.index()].is_some()
            || func.ty(base) != Type::I64
        {
            continue;
        }
        if mode == AtomicAddrMode::Rcpc2 && (delta == 0 || !rcpc2_delta_fits(delta, mask)) {
            continue;
        }
        match locations[base.index()] {
            Location::Slot => {
                elided[v] = true;
                atomic_base[v] = Some(base);
            }
            Location::Frame(_, off)
                if mode == AtomicAddrMode::Full
                    && (-FOLD_LIMIT..=FOLD_LIMIT).contains(&(i64::from(off) + delta)) =>
            {
                elided[v] = true;
            }
            _ => {}
        }
    }
    atomic_base
}
