use anyhow::Result;
use haste_codegen::{Lane, Target, TargetOpId, Type, Value};

use crate::body::BinOp;
use crate::sema::intrinsic::Simd;
use crate::sema::ty::Ty;

use super::super::Lower;
use super::super::ops::{Num, compare};
use super::super::reloc::SiteOrigin;
use super::{Lanes, vec_int_cc, vec_type};

const HALF_BYTES: i32 = 16;
const WIDE_BYTES: i32 = 32;
const QUAD_BYTES: i32 = 8;
const HALF_ALIGN: u8 = 4;
const QUADS: Type = Type::Vec { lane: Lane::I64, count: 4 };

#[derive(Clone, Copy)]
enum Arg {
    Input,
    Output,
    Other,
}

fn operands(op: Simd) -> Option<([Arg; 3], bool)> {
    use Arg::{Input, Other, Output};
    Some(match op {
        Simd::Lanewise(_) | Simd::Pair(_) | Simd::Shuffle => ([Input, Input, Other], true),
        Simd::Fma => ([Input; 3], true),
        Simd::Select => ([Input, Output, Output], true),
        Simd::Cast | Simd::Unary(_) | Simd::Sqrt | Simd::Insert => ([Input, Other, Other], true),
        Simd::Splat => ([Other; 3], true),
        Simd::Bitmask | Simd::Extract | Simd::Reduce { .. } | Simd::Extreme { .. } | Simd::Fold(_) | Simd::Sum { .. } => ([Input, Other, Other], false),
        Simd::SelectBitmask | Simd::MaskedLoad | Simd::MaskedStore => return None,
    })
}

/// The 32-byte register type that holds a vector of `lanes`, when the host has one: its lanes, or four `I64` lanes for 32 byte lanes.
pub(in crate::lower) fn wide_type(lanes: &Lanes) -> Option<Type> {
    if lanes.bytes() != WIDE_BYTES {
        return None;
    }
    vec_type(&lanes.half())?.double_vec().filter(|ty| ty.is_register_vec(Target::host()))
}

/// The 32-byte register type of four `I64` lanes, when the host has one.
pub(in crate::lower) fn wide_quads() -> Option<Type> {
    QUADS.is_register_vec(Target::host()).then_some(QUADS)
}

fn lane_type(lanes: &Lanes) -> Option<Type> {
    wide_type(lanes).filter(|ty| ty.vec_shape().is_some_and(|(lane, count)| Some(lane) == Lane::from_type(lanes.machine) && i32::from(count) == lanes.count))
}

fn byte_lanes(lanes: &Lanes) -> bool {
    lanes.size == 1 && matches!(lanes.num, Num::Int { .. }) && wide_type(lanes).is_some()
}

fn run_start(lanes: &[u8], group: usize) -> Option<usize> {
    let first = usize::from(*lanes.first()?);
    (group != 0 && first % group == 0 && lanes.iter().enumerate().all(|(at, &lane)| usize::from(lane) == first + at)).then(|| first / group)
}

impl Lower<'_, '_> {
    /// Emits the SIMD intrinsic `op` when one of its vectors fills a 32-byte register: in one instruction where the host has it, else on halves that `simd_halves` reads from a split copy and whose result is joined; `None` when `op` has no such vector.
    pub(super) fn simd_wide(&mut self, origin: &SiteOrigin, op: Simd, types: (Ty, Option<Ty>), (from, to): (Lanes, Lanes), substs: &[Ty], args: &[Value]) -> Result<Option<Option<Value>>> {
        if wide_type(&from).is_none() && wide_type(&to).is_none() {
            return Ok(None);
        }
        if let Some(value) = self.wide_native(origin, op, (&from, &to), args)? {
            return Ok(Some(Some(value)));
        }
        let Some((kinds, vector)) = operands(op) else { return Ok(None) };
        let mut split = args.to_vec();
        for (arg, kind) in split.iter_mut().zip(kinds) {
            let lanes = match kind {
                Arg::Input => &from,
                Arg::Output => &to,
                Arg::Other => continue,
            };
            if let Some(wide) = wide_type(lanes) {
                *arg = self.split_wide(wide, *arg);
            }
        }
        let out = self.simd_halves(origin, op, types, (from, to), substs, &split)?;
        Ok(Some(match (out, wide_type(&to)) {
            (Some(halves), Some(wide)) if vector => Some(self.join_wide(wide, halves)),
            (out, _) => out,
        }))
    }

    fn wide_native(&mut self, origin: &SiteOrigin, op: Simd, (from, to): (&Lanes, &Lanes), args: &[Value]) -> Result<Option<Value>> {
        Ok(match (op, args) {
            (Simd::Lanewise(bin), &[x, y]) if compare(bin).is_some() => self.wide_byte_compare(bin, (from, to), (x, y)),
            (Simd::Lanewise(bin), &[x, y]) => self.wide_lanewise(bin, (from, to), (x, y)),
            (Simd::Bitmask, &[x]) if byte_lanes(from) => {
                let v = self.ins().load(QUADS, x, 0);
                let mask = self.ins().target_op(TargetOpId::X64Pmovmskb256, Type::I32, &[v]);
                Some(self.resize(mask, (Type::I32, false), to.machine))
            }
            (Simd::Splat, &[x]) => match lane_type(from) {
                Some(ty) => {
                    let splat = match self.const_splat_vec(origin, ty) {
                        Some(known) => known,
                        None => self.ins().splat(ty, x),
                    };
                    Some(self.store_vec(splat, ty.bytes()))
                }
                None => None,
            },
            (Simd::Shuffle, &[x, y, _]) => self.wide_permute(origin, (from, to), (x, y))?,
            _ => None,
        })
    }

    fn wide_lanewise(&mut self, bin: BinOp, (from, to): (&Lanes, &Lanes), (x, y): (Value, Value)) -> Option<Value> {
        if from.num != to.num || from.count != to.count {
            return None;
        }
        let bitwise = matches!(bin, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor);
        let ty = if bitwise { wide_type(from) } else { lane_type(from) }?;
        let lane = ty.lane()?;
        let native = match (bin, from.num) {
            (BinOp::Add | BinOp::Sub | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor, Num::Int { .. }) => true,
            (BinOp::Shl, Num::Int { .. }) => matches!(lane, Lane::I32 | Lane::I64),
            (BinOp::Shr, Num::Int { signed, .. }) => lane == Lane::I32 || (lane == Lane::I64 && !signed),
            (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, Num::Float(_)) => true,
            _ => false,
        };
        if !native {
            return None;
        }
        let (a, b) = (self.ins().load(ty, x, 0), self.ins().load(ty, y, 0));
        let out = self.lanewise_value(bin, from.num, (a, b))?;
        Some(self.store_vec(out, ty.bytes()))
    }

    fn wide_byte_compare(&mut self, bin: BinOp, (from, to): (&Lanes, &Lanes), (x, y): (Value, Value)) -> Option<Value> {
        let Num::Int { signed, .. } = from.num else { return None };
        if !byte_lanes(from) || !byte_lanes(to) {
            return None;
        }
        let cc = vec_int_cc(bin, signed);
        let (equal, swap, invert) = cc.packed();
        if !equal && !cc.is_signed() {
            return None;
        }
        let (a, b) = (self.ins().load(QUADS, x, 0), self.ins().load(QUADS, y, 0));
        let (a, b) = if swap { (b, a) } else { (a, b) };
        let id = if equal { TargetOpId::X64Pcmpeqb256 } else { TargetOpId::X64Pcmpgtb256 };
        let mask = self.ins().target_op(id, QUADS, &[a, b]);
        let out = if invert { self.ins().bnot(mask) } else { mask };
        Some(self.store_vec(out, QUADS.bytes()))
    }

    fn wide_permute(&mut self, origin: &SiteOrigin, (from, to): (&Lanes, &Lanes), (x, y): (Value, Value)) -> Result<Option<Value>> {
        if from.size != to.size || from.count <= 0 || to.count <= 0 {
            return Ok(None);
        }
        let (lanes_in, lanes_out) = (from.count as usize, to.count as usize);
        let Some(mask) = self.shuffle_const_mask_wide(origin, lanes_in, lanes_out) else { return Ok(None) };
        let mask = &mask[..lanes_out];
        Ok(match (wide_type(from), wide_type(to)) {
            (Some(_), Some(_)) => self.permute_quads(from.size, mask, (x, y)),
            (Some(wide), None) if to.bytes() == HALF_BYTES => self.take_half(wide, mask, (x, y)),
            (None, Some(wide)) if from.bytes() == HALF_BYTES => self.join_halves(wide, from, mask, (x, y)),
            _ => None,
        })
    }

    fn permute_quads(&mut self, size: i32, mask: &[u8], (x, y): (Value, Value)) -> Option<Value> {
        if size <= 0 || QUAD_BYTES % size != 0 {
            return None;
        }
        let group = (QUAD_BYTES / size) as usize;
        let mut quads = [0u8; 16];
        for (quad, lanes) in quads.iter_mut().zip(mask.chunks_exact(group)) {
            *quad = run_start(lanes, group)? as u8;
        }
        let (a, b) = (self.ins().load(QUADS, x, 0), self.ins().load(QUADS, y, 0));
        let out = self.ins().shuffle(a, b, quads);
        Some(self.store_vec(out, QUADS.bytes()))
    }

    fn take_half(&mut self, wide: Type, mask: &[u8], (x, y): (Value, Value)) -> Option<Value> {
        let half = wide.half_vec()?;
        let (src, id) = match run_start(mask, mask.len())? {
            0 => (x, TargetOpId::X64Low128),
            1 => (x, TargetOpId::X64High128),
            2 => (y, TargetOpId::X64Low128),
            _ => (y, TargetOpId::X64High128),
        };
        let whole = self.ins().load(wide, src, 0);
        let out = self.ins().target_op(id, half, &[whole]);
        Some(self.store_vec(out, half.bytes()))
    }

    fn join_halves(&mut self, wide: Type, from: &Lanes, mask: &[u8], (x, y): (Value, Value)) -> Option<Value> {
        let half = vec_type(from)?;
        let count = from.count as usize;
        if mask.len() != 2 * count {
            return None;
        }
        let (a, b) = (self.ins().load(half, x, 0), self.ins().load(half, y, 0));
        let pick = |lower: &mut Self, lanes: &[u8]| match run_start(lanes, count) {
            Some(0) => a,
            Some(1) => b,
            _ => {
                let mut picked = [0u8; 16];
                picked[..count].copy_from_slice(lanes);
                lower.ins().shuffle(a, b, picked)
            }
        };
        let (low, high) = (pick(self, &mask[..count]), pick(self, &mask[count..]));
        let out = self.ins().target_op(TargetOpId::X64Concat128, wide, &[low, high]);
        Some(self.store_vec(out, wide.bytes()))
    }

    /// The address of a copy of the 32-byte vector at `at` that is stored as two 16-byte halves.
    pub(in crate::lower) fn split_wide(&mut self, wide: Type, at: Value) -> Value {
        let Some(half) = wide.half_vec() else { return at };
        let whole = self.ins().load(wide, at, 0);
        let low = self.ins().target_op(TargetOpId::X64Low128, half, &[whole]);
        let high = self.ins().target_op(TargetOpId::X64High128, half, &[whole]);
        let slot = self.b.create_stack_slot(wide.bytes(), HALF_ALIGN);
        self.ins().stack_store(low, slot, 0);
        self.ins().stack_store(high, slot, HALF_BYTES);
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    /// The address of a 32-byte vector stored whole, joined from the two 16-byte halves at `at`.
    pub(in crate::lower) fn join_wide(&mut self, wide: Type, at: Value) -> Value {
        let Some(half) = wide.half_vec() else { return at };
        let (low, high) = (self.ins().load(half, at, 0), self.ins().load(half, at, HALF_BYTES));
        let whole = self.ins().target_op(TargetOpId::X64Concat128, wide, &[low, high]);
        self.store_vec(whole, wide.bytes())
    }

    /// `vpshufb` on the 32-byte vectors at `table` and `index` in one register, when the host has one.
    pub(in crate::lower) fn wide_byte_permute(&mut self, table: Value, index: Value) -> Option<Value> {
        let wide = wide_quads()?;
        let (bytes, picks) = (self.ins().load(wide, table, 0), self.ins().load(wide, index, 0));
        let out = self.ins().target_op(TargetOpId::X64Pshufb256, wide, &[bytes, picks]);
        Some(self.store_vec(out, wide.bytes()))
    }
}
