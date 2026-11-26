use super::super::{Assembled, State, assemble_const};
use crate::ir::{Binary, Convert, Function, Lane, Op, Type, Value};
use crate::mem::{KnownLoads, MemBase, MemLoc};
use crate::merge::find;

pub(in super::super) fn int_of(ty: Type) -> Option<Type> {
    match ty {
        Type::F32 => Some(Type::I32),
        Type::F64 => Some(Type::I64),
        t if t.is_small_int() => Some(t),
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub(in super::super) enum SubwordPlan {
    Int {
        kty: Type,
        kval: Value,
        target_int: Type,
        k_int: Type,
        shift: i64,
        need_reduce: bool,
        bitcast_out: bool,
    },
    VecBitcast {
        kval: Value,
    },
    VecExtract {
        kty: Type,
        kval: Value,
        lane_vec: Type,
        lane_idx: u8,
        out_vec: Option<Type>,
    },
}

impl SubwordPlan {
    pub(in super::super) fn kval(self) -> Value {
        match self {
            Self::Int { kval, .. } | Self::VecBitcast { kval } | Self::VecExtract { kval, .. } => {
                kval
            }
        }
    }

    pub(in super::super) fn new(
        loc_offset: i64,
        ty: Type,
        koff: i64,
        kty: Type,
        kval: Value,
    ) -> Option<Self> {
        if koff == loc_offset
            && kty.bytes() == ty.bytes()
            && ty != kty
            && (ty.is_vector() || kty.is_vector())
        {
            let ok = (ty.is_vector() && kty.is_vector())
                || (ty.bytes() == 8
                    && (matches!(ty, Type::I64 | Type::F64)
                        || matches!(kty, Type::I64 | Type::F64)));
            return ok.then_some(Self::VecBitcast { kval });
        }
        if kty.is_vector() {
            if kty.bytes() > 16 {
                return None;
            }
            let ty_end = loc_offset.checked_add(i64::from(ty.bytes()))?;
            let k_end = koff.checked_add(i64::from(kty.bytes()))?;
            if loc_offset < koff || ty_end > k_end {
                return None;
            }
            if let Some(lane) = Lane::from_type(ty) {
                let lbytes = i64::from(lane.bytes());
                if (loc_offset - koff) % lbytes != 0 {
                    return None;
                }
                let lane_vec = Type::Vec {
                    lane,
                    count: (kty.bytes() / lane.bytes()) as u8,
                };
                let lane_idx = ((loc_offset - koff) / lbytes) as u8;
                return Some(Self::VecExtract {
                    kty,
                    kval,
                    lane_vec,
                    lane_idx,
                    out_vec: None,
                });
            }
            if ty.is_vector()
                && ty.bytes() == 8
                && kty.bytes() == 16
                && (loc_offset - koff) % 8 == 0
            {
                let lane_vec = Type::Vec {
                    lane: Lane::I64,
                    count: 2,
                };
                let lane_idx = ((loc_offset - koff) / 8) as u8;
                return Some(Self::VecExtract {
                    kty,
                    kval,
                    lane_vec,
                    lane_idx,
                    out_vec: Some(ty),
                });
            }
            return None;
        }
        let target_int = int_of(ty)?;
        let k_int = int_of(kty)?;
        let shift = (loc_offset - koff) * 8;
        let need_shift = shift > 0;
        let need_reduce = ty.bytes() < kty.bytes();
        let bitcast_out = ty.is_float();
        if !(kty.is_float() || need_shift || need_reduce || bitcast_out) {
            return None;
        }
        Some(Self::Int {
            kty,
            kval,
            target_int,
            k_int,
            shift,
            need_reduce,
            bitcast_out,
        })
    }

    pub(in super::super) fn build_op(self, mut emit: impl FnMut(Type, Op) -> Value) -> Op {
        match self {
            Self::VecBitcast { kval } => Op::Convert(Convert::Bitcast, kval),
            Self::VecExtract {
                kty,
                kval,
                lane_vec,
                lane_idx,
                out_vec,
            } => {
                let src = if kty == lane_vec {
                    kval
                } else {
                    emit(lane_vec, Op::Convert(Convert::Bitcast, kval))
                };
                let extract = Op::Extract(src, lane_idx);
                if out_vec.is_some() {
                    let raw = emit(Type::I64, extract);
                    Op::Convert(Convert::Bitcast, raw)
                } else {
                    extract
                }
            }
            Self::Int {
                kty,
                kval,
                target_int,
                k_int,
                shift,
                need_reduce,
                bitcast_out,
            } => {
                let mut cur = kval;
                let need_shift = shift > 0;
                if kty.is_float() {
                    let op = Op::Convert(Convert::Bitcast, cur);
                    if !need_shift && !need_reduce && !bitcast_out {
                        return op;
                    }
                    cur = emit(k_int, op);
                }
                if need_shift {
                    let sh = emit(k_int, Op::Const(shift));
                    let op = Op::Binary(Binary::Ushr, cur, sh);
                    if !need_reduce && !bitcast_out {
                        return op;
                    }
                    cur = emit(k_int, op);
                }
                if need_reduce {
                    let op = Op::Convert(Convert::Ireduce, cur);
                    if !bitcast_out {
                        return op;
                    }
                    cur = emit(target_int, op);
                }
                Op::Convert(Convert::Bitcast, cur)
            }
        }
    }
}

pub(in super::super) fn find_subword_source(
    known: &KnownLoads,
    loc: MemLoc,
    ty: Type,
) -> Option<(i64, Type, Value)> {
    if !(int_of(ty).is_some() || ty.is_vector()) {
        return None;
    }
    let cell_end = loc.offset.checked_add(i64::from(ty.bytes()))?;
    known
        .entries_for(loc.base)
        .iter()
        .filter(|&&(off, kty, kval)| {
            off <= loc.offset
                && off
                    .checked_add(i64::from(kty.bytes()))
                    .is_some_and(|end| cell_end <= end)
                && SubwordPlan::new(loc.offset, ty, off, kty, kval).is_some()
        })
        .min_by_key(|&&(off, kty, _)| (kty.bytes(), !kty.is_small_int(), loc.offset - off))
        .copied()
}

pub(in super::super) fn rodata_const(func: &Function, loc: MemLoc, ty: Type) -> Option<i64> {
    let MemBase::Global(sym) = loc.base else {
        return None;
    };
    func.rodata_read(sym, loc.offset, ty)
}

pub(in super::super) fn const_bits_eq(ty: Type, ca: i64, cb: i64) -> bool {
    match ty {
        t if t.is_small_int() => (ca as u64 & t.unsigned_max()) == (cb as u64 & t.unsigned_max()),
        Type::F32 => (ca as u32) == (cb as u32),
        Type::F64 => ca == cb,
        _ => false,
    }
}

pub(in super::super) fn const_eq_for_ty(
    consts: &[Option<i64>],
    ty: Type,
    a: Value,
    b: Value,
) -> bool {
    consts
        .get(a.index())
        .copied()
        .flatten()
        .zip(consts.get(b.index()).copied().flatten())
        .is_some_and(|(ca, cb)| const_bits_eq(ty, ca, cb))
}

#[derive(Clone, Copy)]
pub(super) enum LocPlan {
    Exact(Value),
    Const(i64),
    Subword(SubwordPlan),
}

pub(super) fn plan_loc(
    func: &Function,
    state: &State<'_>,
    known: &KnownLoads,
    loc: MemLoc,
    ty: Type,
) -> Option<LocPlan> {
    if let Some(v) = known.get(loc.base, loc.offset, ty) {
        return Some(LocPlan::Exact(v));
    }
    if let Some(imm) = rodata_const(func, loc, ty) {
        return Some(LocPlan::Const(imm));
    }
    if let Some(Assembled::Int(imm) | Assembled::FloatBits(imm)) =
        assemble_const(known, loc, ty, state.consts)
    {
        return Some(LocPlan::Const(imm));
    }
    let (koff, kty, kval) = find_subword_source(known, loc, ty)?;
    let kval = find(state.subst, kval);
    SubwordPlan::new(loc.offset, ty, koff, kty, kval).map(LocPlan::Subword)
}

pub(super) fn materialize_plan(
    func: &mut Function,
    state: &mut State<'_>,
    ty: Type,
    plan: LocPlan,
    mut emit: impl FnMut(&mut Function, &mut State<'_>, Type, Op) -> Value,
) -> Value {
    match plan {
        LocPlan::Exact(v) => v,
        LocPlan::Const(imm) => emit(func, state, ty, Op::Const(imm)),
        LocPlan::Subword(sub) => {
            let final_op = sub.build_op(|op_ty, op| emit(func, state, op_ty, op));
            emit(func, state, ty, final_op)
        }
    }
}
