use super::join::{const_bits_eq, const_eq_for_ty, rodata_const};
use super::{State, emit_op};
use crate::ir::{
    Binary, Convert, Function, Inst, Lane, Op, ShuffleMask, TargetOpId, TinyArgs, Type, Value,
};
use crate::mem::{KnownLoads, MemLoc};
use crate::merge::find;

const QUAD: Type = Type::Vec {
    lane: Lane::I64,
    count: 2,
};
const WIDE_QUADS: Type = Type::Vec {
    lane: Lane::I64,
    count: 4,
};

pub(super) enum Assembled {
    Int(i64),
    FloatBits(i64),
    VecZero,
    VecByteSplat(u8),
}

pub(super) fn assemble_const(
    known: &KnownLoads,
    loc: MemLoc,
    ty: Type,
    consts: &[Option<i64>],
) -> Option<Assembled> {
    let entries = known.entries_for(loc.base);
    let byte_at = |byte: i64| -> Option<u8> {
        let at = loc.offset.checked_add(byte)?;
        let (koff, imm) = entries.iter().find_map(|&(off, kty, v)| {
            let end = off.checked_add(i64::from(kty.bytes()))?;
            (kty.is_small_int() && off <= at && at < end)
                .then(|| consts[v.index()].map(|imm| (off, imm)))
                .flatten()
        })?;
        let shift = 8 * (at - koff);
        Some((((imm as u64) >> shift) & 0xFF) as u8)
    };
    if ty.is_vector() {
        let len = i64::from(ty.bytes());
        let first = byte_at(0)?;
        for byte in 1..len {
            if byte_at(byte)? != first {
                return None;
            }
        }
        if ty.bytes() > 16 && first != 0 {
            return None;
        }
        return Some(if first == 0 {
            Assembled::VecZero
        } else {
            Assembled::VecByteSplat(first)
        });
    }
    let width = i64::from(ty.bytes());
    if !((ty.is_small_int() && matches!(width, 1 | 2 | 4 | 8))
        || matches!(ty, Type::F32 | Type::F64))
    {
        return None;
    }
    let mut assembled = 0_u64;
    for byte in 0..width {
        assembled |= u64::from(byte_at(byte)?) << (8 * byte);
    }
    Some(if ty.is_small_int() {
        Assembled::Int(assembled as i64)
    } else {
        Assembled::FloatBits(assembled as i64)
    })
}

pub(super) fn emit_assembled(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    inst: Inst,
    ty: Type,
    result: Value,
    assembled: Assembled,
) {
    match assembled {
        Assembled::Int(imm) => {
            func.insts[inst.index()].op = Op::Const(imm);
            state.consts[result.index()] = Some(imm);
        }
        Assembled::FloatBits(imm) => func.insts[inst.index()].op = Op::Const(imm),
        Assembled::VecZero => func.insts[inst.index()].op = Op::Const(0),
        Assembled::VecByteSplat(b) => {
            let byte_ty = Type::Vec {
                lane: Lane::I8,
                count: ty.bytes() as u8,
            };
            let b_val = emit_op(func, state, kept, Type::I8, Op::Const(i64::from(b)));
            let zero = emit_op(func, state, kept, byte_ty, Op::Const(0));
            let first = emit_op(func, state, kept, byte_ty, Op::Insert(zero, b_val, 0));
            let splat = Op::Shuffle(first, first, ShuffleMask::Const([0; 16]));
            if ty == byte_ty {
                func.insts[inst.index()].op = splat;
            } else {
                let vec = emit_op(func, state, kept, byte_ty, splat);
                func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, vec);
            }
        }
    }
}

pub(super) fn is_silent_store(
    known: &KnownLoads,
    loc: MemLoc,
    ty: Type,
    value: Value,
    consts: &[Option<i64>],
) -> bool {
    if let Some(prev) = known.get(loc.base, loc.offset, ty)
        && (prev == value || const_eq_for_ty(consts, ty, prev, value))
    {
        return true;
    }
    let Some(imm) = consts.get(value.index()).copied().flatten() else {
        return false;
    };
    match assemble_const(known, loc, ty, consts) {
        Some(Assembled::Int(kimm) | Assembled::FloatBits(kimm)) => const_bits_eq(ty, kimm, imm),
        Some(Assembled::VecZero) => ty.is_vector() && imm == 0,
        _ => false,
    }
}

enum ConcatHalf {
    Val(Value, bool),
    Const(i64),
}

fn resolve_concat_half(
    known: &KnownLoads,
    loc: MemLoc,
    offset: i64,
    consts: &[Option<i64>],
) -> Option<ConcatHalf> {
    if let Some(v) = known.get(loc.base, offset, Type::I32) {
        return Some(ConcatHalf::Val(v, false));
    }
    if let Some(v) = known.get(loc.base, offset, Type::F32) {
        return Some(ConcatHalf::Val(v, true));
    }
    match assemble_const(
        known,
        MemLoc {
            offset,
            size: 4,
            ..loc
        },
        Type::I32,
        consts,
    ) {
        Some(Assembled::Int(imm)) => Some(ConcatHalf::Const(imm)),
        _ => None,
    }
}

fn materialize_concat_half(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    half: ConcatHalf,
) -> Value {
    match half {
        ConcatHalf::Val(v, true) => emit_op(
            func,
            state,
            kept,
            Type::I32,
            Op::Convert(Convert::Bitcast, v),
        ),
        ConcatHalf::Val(v, false) => v,
        ConcatHalf::Const(imm) => emit_op(func, state, kept, Type::I32, Op::Const(imm)),
    }
}

enum Word64Plan {
    Val(Value),
    Bitcast(Value),
    Const(i64),
    Pair32(ConcatHalf, ConcatHalf),
}

fn plan_word64(
    func: &Function,
    known: &KnownLoads,
    loc: MemLoc,
    offset: i64,
    consts: &[Option<i64>],
) -> Option<Word64Plan> {
    if let Some(v) = known.get(loc.base, offset, Type::I64) {
        return Some(Word64Plan::Val(v));
    }
    if let Some(v) = known.get(loc.base, offset, Type::F64) {
        return Some(Word64Plan::Bitcast(v));
    }
    if let Some(&(_, _, v)) = known
        .entries_for(loc.base)
        .iter()
        .find(|&&(off, kty, _)| off == offset && kty.is_vector() && kty.bytes() == 8)
    {
        return Some(Word64Plan::Bitcast(v));
    }
    let slice = MemLoc {
        offset,
        size: 8,
        ..loc
    };
    if let Some(imm) = rodata_const(func, slice, Type::I64) {
        return Some(Word64Plan::Const(imm));
    }
    if let Some(Assembled::Int(imm)) = assemble_const(known, slice, Type::I64, consts) {
        return Some(Word64Plan::Const(imm));
    }
    let hi_off = offset.checked_add(4)?;
    Some(Word64Plan::Pair32(
        resolve_concat_half(known, loc, offset, consts)?,
        resolve_concat_half(known, loc, hi_off, consts)?,
    ))
}

fn emit_pack32_op(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    lo_half: ConcatHalf,
    hi_half: ConcatHalf,
) -> Op {
    let lo32 = materialize_concat_half(func, state, kept, lo_half);
    let hi32 = materialize_concat_half(func, state, kept, hi_half);
    let lo64 = emit_op(
        func,
        state,
        kept,
        Type::I64,
        Op::Convert(Convert::Uextend, lo32),
    );
    let hi64 = emit_op(
        func,
        state,
        kept,
        Type::I64,
        Op::Convert(Convert::Uextend, hi32),
    );
    let c32 = emit_op(func, state, kept, Type::I64, Op::Const(32));
    let hi_shifted = emit_op(
        func,
        state,
        kept,
        Type::I64,
        Op::Binary(Binary::Ishl, hi64, c32),
    );
    Op::Binary(Binary::Bor, lo64, hi_shifted)
}

fn materialize_word64(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    plan: Word64Plan,
) -> Value {
    match plan {
        Word64Plan::Val(v) => find(state.subst, v),
        Word64Plan::Bitcast(v) => emit_op(
            func,
            state,
            kept,
            Type::I64,
            Op::Convert(Convert::Bitcast, find(state.subst, v)),
        ),
        Word64Plan::Const(imm) => emit_op(func, state, kept, Type::I64, Op::Const(imm)),
        Word64Plan::Pair32(lo, hi) => {
            let op = emit_pack32_op(func, state, kept, lo, hi);
            emit_op(func, state, kept, Type::I64, op)
        }
    }
}

fn quad_op(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    (lo, hi): (Word64Plan, Word64Plan),
) -> Op {
    let lo64 = materialize_word64(func, state, kept, lo);
    let hi64 = materialize_word64(func, state, kept, hi);
    let zero = emit_op(func, state, kept, QUAD, Op::Const(0));
    let half = emit_op(func, state, kept, QUAD, Op::Insert(zero, lo64, 0));
    Op::Insert(half, hi64, 1)
}

fn set_load(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    (inst, ty): (Inst, Type),
    (built, op): (Type, Op),
) {
    if ty == built {
        func.insts[inst.index()].op = op;
    } else {
        let value = emit_op(func, state, kept, built, op);
        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, value);
    }
}

pub(super) fn forward_concat(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    known: &KnownLoads,
    inst: Inst,
    loc: MemLoc,
    ty: Type,
) -> bool {
    if ty.is_vector()
        && (ty.bytes() == 16 || (ty.bytes() == 32 && ty.is_register_vec(state.target)))
    {
        let Some(plans) = (0..i64::from(ty.bytes()) / 8)
            .map(|word| {
                plan_word64(
                    func,
                    known,
                    loc,
                    loc.offset.checked_add(8 * word)?,
                    state.consts,
                )
            })
            .collect::<Option<Vec<_>>>()
        else {
            return false;
        };
        let mut plans = plans.into_iter();
        let (Some(lo), Some(hi)) = (plans.next(), plans.next()) else {
            return false;
        };
        let low = quad_op(func, state, kept, (lo, hi));
        let built = match (plans.next(), plans.next()) {
            (Some(lo), Some(hi)) => {
                let low = emit_op(func, state, kept, QUAD, low);
                let high = quad_op(func, state, kept, (lo, hi));
                let high = emit_op(func, state, kept, QUAD, high);
                (
                    WIDE_QUADS,
                    Op::TargetOp(TargetOpId::X64Concat128, TinyArgs::two(low, high)),
                )
            }
            _ => (QUAD, low),
        };
        set_load(func, state, kept, (inst, ty), built);
        return true;
    }
    if !(matches!(ty, Type::I64 | Type::F64) || (ty.is_vector() && ty.bytes() == 8)) {
        return false;
    }
    let Some(hi_off) = loc.offset.checked_add(4) else {
        return false;
    };
    let Some(lo_half) = resolve_concat_half(known, loc, loc.offset, state.consts) else {
        return false;
    };
    let Some(hi_half) = resolve_concat_half(known, loc, hi_off, state.consts) else {
        return false;
    };
    let combined = emit_pack32_op(func, state, kept, lo_half, hi_half);
    set_load(func, state, kept, (inst, ty), (Type::I64, combined));
    true
}
