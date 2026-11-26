use super::super::{defs, extend};
use super::bytes::{ByteEval, eval_reduce_const, full_bytes, read_lane};
use crate::ir::{Binary, Convert, Function, Inst, Lane, Op, ShuffleMask, Type, Value};

#[derive(Clone, Copy)]
enum VecPattern {
    Splat(Type, u64),
    One(u8, Type, u64),
    Pair(u64, u64),
}

fn pack_u64(bytes: &[u8; 16], off: usize) -> u64 {
    u64::from_le_bytes([
        bytes[off],
        bytes[off + 1],
        bytes[off + 2],
        bytes[off + 3],
        bytes[off + 4],
        bytes[off + 5],
        bytes[off + 6],
        bytes[off + 7],
    ])
}

fn lane_splat_val(vec_ty: Type, bytes: &[u8; 16]) -> Option<(Type, u64)> {
    let (lane, count) = vec_ty.vec_shape()?;
    if !lane.is_int() {
        return None;
    }
    let eb = lane.bytes() as usize;
    let first = &bytes[..eb];
    if (1..usize::from(count)).any(|i| &bytes[i * eb..(i + 1) * eb] != first) {
        return None;
    }
    let mut raw = [0_u8; 8];
    raw[..eb].copy_from_slice(first);
    Some((lane.to_type(), u64::from_le_bytes(raw)))
}

fn choose_plan(vec_ty: Type, bytes: &[u8; 16], is_shift_rhs: bool) -> (Type, VecPattern) {
    let (lo, hi) = (pack_u64(bytes, 0), pack_u64(bytes, 8));
    if is_shift_rhs && let Some((lane_ty, val)) = lane_splat_val(vec_ty, bytes) {
        return (vec_ty, VecPattern::Splat(lane_ty, val));
    }
    if lo == hi {
        let w32 = u64::from(lo as u32);
        if (w32 | (w32 << 32)) == lo {
            return (
                Type::Vec {
                    lane: Lane::I32,
                    count: 4,
                },
                VecPattern::Splat(Type::I32, w32),
            );
        }
        return (
            Type::Vec {
                lane: Lane::I64,
                count: 2,
            },
            VecPattern::Splat(Type::I64, lo),
        );
    }
    if hi == 0 {
        if (lo >> 32) == 0
            && vec_ty
                != (Type::Vec {
                    lane: Lane::I64,
                    count: 2,
                })
        {
            return (
                Type::Vec {
                    lane: Lane::I32,
                    count: 4,
                },
                VecPattern::One(0, Type::I32, lo),
            );
        }
        return (
            Type::Vec {
                lane: Lane::I64,
                count: 2,
            },
            VecPattern::One(0, Type::I64, lo),
        );
    }
    if lo == 0 {
        return (
            Type::Vec {
                lane: Lane::I64,
                count: 2,
            },
            VecPattern::One(1, Type::I64, hi),
        );
    }
    (
        Type::Vec {
            lane: Lane::I64,
            count: 2,
        },
        VecPattern::Pair(lo, hi),
    )
}

fn is_zero_vec(func: &Function, defs: &[Option<Inst>], val: Value) -> bool {
    func.is_zero_vec(&|value| defs.get(value.index()).copied().flatten(), val)
}

fn const_matches(func: &Function, defs: &[Option<Inst>], val: Value, ty: Type, want: u64) -> bool {
    func.ty(val) == ty && defs.get(val.index()).copied().flatten().is_some_and(|i| {
        matches!(
            func.insts[i.index()].op,
            Op::Const(imm) if extend(imm, ty, false) as u64 == extend(want as i64, ty, false) as u64
        )
    })
}

fn matches_pattern(func: &Function, defs: &[Option<Inst>], inst: Inst, pat: VecPattern) -> bool {
    match (pat, func.insts[inst.index()].op.clone()) {
        (VecPattern::Splat(ty, val), Op::Shuffle(a, b, ShuffleMask::Const(m)))
            if a == b && m == [0; 16] =>
        {
            defs.get(a.index()).copied().flatten().is_some_and(|ai| {
                matches!(
                    func.insts[ai.index()].op,
                    Op::Insert(z, c, 0) if is_zero_vec(func, defs, z) && const_matches(func, defs, c, ty, val)
                )
            })
        }
        (VecPattern::One(idx, ty, val), Op::Insert(z, c, at)) => {
            at == idx && is_zero_vec(func, defs, z) && const_matches(func, defs, c, ty, val)
        }
        (VecPattern::Pair(lo, hi), Op::Insert(v0, c1, 1))
            if const_matches(func, defs, c1, Type::I64, hi) =>
        {
            defs.get(v0.index()).copied().flatten().is_some_and(|i0| {
                matches!(
                    func.insts[i0.index()].op,
                    Op::Insert(z, c0, 0) if is_zero_vec(func, defs, z) && const_matches(func, defs, c0, Type::I64, lo)
                )
            })
        }
        _ => false,
    }
}

fn is_canonical(
    func: &Function,
    defs: &[Option<Inst>],
    inst: Inst,
    vec_ty: Type,
    bytes: &[u8; 16],
    is_shift_rhs: bool,
) -> bool {
    let total = vec_ty.bytes() as usize;
    if bytes[..total].iter().all(|&b| b == 0) {
        return matches!(func.insts[inst.index()].op, Op::Const(0));
    }
    if total == 8 {
        return matches!(
            func.insts[inst.index()].op,
            Op::Convert(Convert::Bitcast, x) if const_matches(func, defs, x, Type::I64, pack_u64(bytes, 0))
        );
    }
    let (target_ty, pat) = choose_plan(vec_ty, bytes, is_shift_rhs);
    if vec_ty == target_ty {
        return matches_pattern(func, defs, inst, pat);
    }
    let Op::Convert(Convert::Bitcast, inner) = func.insts[inst.index()].op else {
        return false;
    };
    func.ty(inner) == target_ty
        && defs
            .get(inner.index())
            .copied()
            .flatten()
            .is_some_and(|i| matches_pattern(func, defs, i, pat))
}

fn build_pattern_op(
    func: &mut Function,
    target_ty: Type,
    pat: VecPattern,
    out: &mut Vec<Inst>,
) -> Op {
    let z = func.value(target_ty);
    out.push(func.add(Op::Const(0), Some(z)));
    match pat {
        VecPattern::Splat(ty, val) => {
            let c = func.value(ty);
            out.push(func.add(Op::Const(extend(val as i64, ty, true)), Some(c)));
            let first = func.value(target_ty);
            out.push(func.add(Op::Insert(z, c, 0), Some(first)));
            Op::Shuffle(first, first, ShuffleMask::Const([0; 16]))
        }
        VecPattern::One(idx, ty, val) => {
            let c = func.value(ty);
            out.push(func.add(Op::Const(extend(val as i64, ty, true)), Some(c)));
            Op::Insert(z, c, idx)
        }
        VecPattern::Pair(lo, hi) => {
            let c0 = func.value(Type::I64);
            out.push(func.add(Op::Const(lo as i64), Some(c0)));
            let v0 = func.value(target_ty);
            out.push(func.add(Op::Insert(z, c0, 0), Some(v0)));
            let c1 = func.value(Type::I64);
            out.push(func.add(Op::Const(hi as i64), Some(c1)));
            Op::Insert(v0, c1, 1)
        }
    }
}

fn emit_canonical(
    func: &mut Function,
    inst: Inst,
    vec_ty: Type,
    bytes: &[u8; 16],
    is_shift_rhs: bool,
    out: &mut Vec<Inst>,
) {
    let total = vec_ty.bytes() as usize;
    if bytes[..total].iter().all(|&b| b == 0) {
        func.insts[inst.index()].op = Op::Const(0);
        out.push(inst);
        return;
    }
    if total == 8 {
        let c = func.value(Type::I64);
        out.push(func.add(Op::Const(pack_u64(bytes, 0) as i64), Some(c)));
        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, c);
        out.push(inst);
        return;
    }
    let (target_ty, pat) = choose_plan(vec_ty, bytes, is_shift_rhs);
    let op = build_pattern_op(func, target_ty, pat, out);
    if target_ty == vec_ty {
        func.insts[inst.index()].op = op;
    } else {
        let tmp = func.value(target_ty);
        out.push(func.add(op, Some(tmp)));
        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, tmp);
    }
    out.push(inst);
}

fn collect_scalar_folds(func: &Function, eval: &mut ByteEval) -> Vec<(Inst, Op)> {
    let mut updates = Vec::new();
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        match data.op {
            Op::Extract(vec, lane_idx)
                if func.ty(vec).bytes() <= 16
                    && (func.ty(res).is_small_int() || func.ty(res).is_float()) =>
            {
                if let Some((lane, count)) = func.ty(vec).vec_shape()
                    && lane_idx < count
                    && let Some(imm) = read_lane(&eval.eval(vec, 0), lane, usize::from(lane_idx))
                {
                    updates.push((*inst, Op::Const(imm)));
                }
            }
            Op::Reduce(rop, vec) if func.ty(res).is_small_int() => {
                let vb = eval.eval(vec, 0);
                if let Some(imm) = eval_reduce_const(&vb, rop, func.ty(vec)) {
                    updates.push((*inst, Op::Const(imm)));
                }
            }
            _ => {}
        }
    }
    updates
}

fn collect_known_vecs(func: &Function, eval: &mut ByteEval) -> Vec<Option<[u8; 16]>> {
    let mut known_vec: Vec<Option<[u8; 16]>> = vec![None; func.types.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let Some(res) = func.insts[inst.index()].result else {
            continue;
        };
        let ty = func.ty(res);
        if ty.is_valid_vec()
            && !matches!(
                func.insts[inst.index()].op,
                Op::Load(..) | Op::StackLoad(..)
            )
        {
            let bytes = eval.eval(res, 0);
            known_vec[res.index()] = full_bytes(&bytes, ty.bytes() as usize);
        }
    }
    known_vec
}

fn rewrite_canonical_vecs(
    func: &mut Function,
    defs: &[Option<Inst>],
    known_vec: &[Option<[u8; 16]>],
) -> bool {
    let (mut root, mut shift_rhs) = (vec![false; func.types.len()], vec![false; func.types.len()]);
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        if let Op::Binary(Binary::Ishl | Binary::Ushr | Binary::Sshr, _, rhs) = data.op {
            shift_rhs[rhs.index()] = true;
        }
        if !data.result.is_some_and(|r| known_vec[r.index()].is_some()) {
            data.op
                .for_operands(|v| root[v.index()] |= known_vec[v.index()].is_some());
        }
    }
    let mut changed = false;
    for b in 0..func.blocks.len() {
        let orig = std::mem::take(&mut func.blocks[b].insts);
        let mut out = Vec::with_capacity(orig.len());
        for inst in orig {
            if let Some(res) = func.insts[inst.index()].result
                && root[res.index()]
                && let Some(bytes) = known_vec[res.index()]
                && !is_canonical(
                    func,
                    defs,
                    inst,
                    func.ty(res),
                    &bytes,
                    shift_rhs[res.index()],
                )
            {
                emit_canonical(
                    func,
                    inst,
                    func.ty(res),
                    &bytes,
                    shift_rhs[res.index()],
                    &mut out,
                );
                changed = true;
                continue;
            }
            out.push(inst);
        }
        func.blocks[b].insts = out;
    }
    changed
}

pub(super) fn fold_const_vectors(func: &mut Function) -> bool {
    let vector = |inst: &Inst| {
        matches!(
            func.insts[inst.index()].op,
            Op::Extract(..) | Op::Reduce(..)
        ) || func.insts[inst.index()]
            .result
            .is_some_and(|res| func.ty(res).is_valid_vec())
    };
    if !func.blocks.iter().flat_map(|b| &b.insts).any(vector) {
        return false;
    }
    let defs = defs(func);
    let mut eval = ByteEval::new(func, &defs);
    let scalar_updates = collect_scalar_folds(func, &mut eval);
    let known_vec = collect_known_vecs(func, &mut eval);
    let mut changed = false;
    for (inst, op) in scalar_updates {
        if func.insts[inst.index()].op != op {
            func.insts[inst.index()].op = op;
            changed = true;
        }
    }
    changed | rewrite_canonical_vecs(func, &defs, &known_vec)
}
