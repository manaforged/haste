use super::{Fold, binary, defs, extend, holds};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Lane, Op, ReduceOp, ShuffleMask, Type, Unary, Value};

pub(super) fn rotate(ty: Type, value: u64, left: u32) -> i64 {
    match ty {
        Type::I8 => i64::from((value as u8).rotate_left(left)),
        Type::I16 => i64::from((value as u16).rotate_left(left)),
        Type::I32 => i64::from((value as u32).rotate_left(left)),
        _ => value.rotate_left(left) as i64,
    }
}

pub(super) fn divide(op: Binary, ty: Type, a: i64, b: i64) -> Option<i64> {
    let signed = matches!(op, Binary::Sdiv | Binary::Srem);
    let (x, y) = (extend(a, ty, signed), extend(b, ty, signed));
    if y == 0 || (signed && y == -1 && x == -1_i64 << (ty.bits() - 1)) {
        return None;
    }
    Some(match op {
        Binary::Sdiv => x / y,
        Binary::Srem => x % y,
        Binary::Udiv => (x as u64 / y as u64) as i64,
        _ => (x as u64 % y as u64) as i64,
    })
}

pub(super) fn sat_binary(op: Binary, ty: Type, sa: i64, sb: i64, ua: u64, ub: u64) -> Option<i64> {
    let bits = ty.bits();
    let (smin, smax) = (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1);
    Some(match op {
        Binary::SaddSat => (i128::from(sa) + i128::from(sb)).clamp(smin, smax) as i64,
        Binary::SsubSat => (i128::from(sa) - i128::from(sb)).clamp(smin, smax) as i64,
        Binary::UaddSat => (u128::from(ua) + u128::from(ub)).min(u128::from(ty.unsigned_max())) as i64,
        Binary::UsubSat => ua.saturating_sub(ub) as i64,
        Binary::AvgRound => ((u128::from(ua) + u128::from(ub) + 1) >> 1) as i64,
        _ => return None,
    })
}

fn clamp_narrow(src: Type, dst: Type, signed: bool, raw: i64) -> i64 {
    let val = extend(raw, src, true);
    let clamped = if signed {
        let max = (1_i64 << (dst.bits() - 1)) - 1;
        val.clamp(-max - 1, max)
    } else {
        val.clamp(0, dst.unsigned_max() as i64)
    };
    extend(clamped, dst, true)
}

pub(super) fn narrow_sat(func: &Function, dst: Type, signed: bool, x: Value, known: &dyn Fn(Value) -> Option<i64>) -> Option<i64> {
    let src = func.ty(x);
    if !src.is_small_int() || dst.bytes() >= src.bytes() {
        return None;
    }
    Some(clamp_narrow(src, dst, signed, known(x)?))
}

fn recast(func: &Function, inst: Inst) -> Option<(Value, Value)> {
    match (&func.insts[inst.index()].op, func.insts[inst.index()].result) {
        (&Op::Convert(Convert::Bitcast, src), Some(res)) if func.ty(src).bytes() == func.ty(res).bytes() && func.ty(src) != func.ty(res) => Some((src, res)),
        _ => None,
    }
}

pub(super) fn fold_bitcast_loads(func: &mut Function) -> bool {
    if !func.blocks.iter().flat_map(|b| &b.insts).any(|&inst| recast(func, inst).is_some()) { return fold_const_vectors(func); }
    let uses = func.use_counts();
    let defs = defs(func);
    let mut remap: Vec<Option<Value>> = vec![None; func.types.len()];
    let mut dropped = vec![false; func.insts.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        if let Some((src, res)) = recast(func, *inst)
            && uses[src.index()] == 1
            && let Some(src_inst) = defs[src.index()]
            && matches!(func.insts[src_inst.index()].op, Op::Load(..))
        {
            func.types[src.index()] = func.ty(res);
            remap[res.index()] = Some(src);
            dropped[inst.index()] = true;
        }
    }
    let mut changed = !remap.iter().all(Option::is_none);
    if changed {
        for block in &mut func.blocks {
            block.insts.retain(|inst| !dropped[inst.index()]);
            for &inst in &block.insts {
                func.insts[inst.index()].op.map_operands(|v| remap[v.index()].unwrap_or(v));
            }
        }
    }
    changed |= fold_const_vectors(func);
    changed
}

fn trace_extract(fold: &Fold, func: &Function, mut vec: Value, mut idx: u8) -> Option<Value> {
    for _ in 0..16 {
        match *fold.def(func, vec)? {
            Op::Convert(Convert::Bitcast, inner) if idx == 0 && func.ty(vec).vec_shape().is_some_and(|(l, _)| func.ty(inner) == l.to_type()) => return Some(inner),
            Op::Insert(_, val, at) if at == idx => return Some(val),
            Op::Insert(prev, _, _) => vec = prev,
            Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
                let count = func.ty(a).vec_shape()?.1;
                let sel = *mask.get(usize::from(idx))?;
                if sel < count { (vec, idx) = (a, sel); } else if sel < 2 * count { (vec, idx) = (b, sel - count); } else { return None; }
            }
            _ => return None,
        }
    }
    None
}

fn splat_shuffle_ident(fold: &Fold, func: &Function, v: Value) -> bool {
    let Some((&Op::Shuffle(a, b, ShuffleMask::Const(mask)), Some((_, count)))) = fold.def(func, v).map(|op| (op, func.ty(v).vec_shape())) else {
        return false;
    };
    let c = usize::from(count);
    let head = mask[0];
    let same = fold.find(a) == fold.find(b);
    usize::from(head) < 2 * c && mask[..c].iter().all(|&i| i == head || (same && i % count == head % count))
}

fn shuffle_identity(fold: &Fold, func: &Function, a: Value, b: Value, mask: [u8; 16]) -> Option<Value> {
    let (_, count) = func.ty(a).vec_shape()?;
    let c = usize::from(count);
    let same_ab = fold.find(a) == fold.find(b);
    if (0..count).all(|i| mask[usize::from(i)] == i || (same_ab && mask[usize::from(i)] == count + i)) {
        return Some(a);
    }
    if !same_ab && (0..count).all(|i| mask[usize::from(i)] == count + i) {
        return Some(b);
    }
    if mask[..c].iter().all(|&i| i < count || (same_ab && i < 2 * count)) && splat_shuffle_ident(fold, func, a) {
        return Some(a);
    }
    (!same_ab && mask[..c].iter().all(|&i| i >= count && i < 2 * count) && splat_shuffle_ident(fold, func, b)).then_some(b)
}

pub(super) fn vec_identity(fold: &Fold, func: &Function, op: &Op) -> Option<Value> {
    match *op {
        Op::Extract(vec, idx) => trace_extract(fold, func, vec, idx),
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => shuffle_identity(fold, func, a, b, mask),
        _ => None,
    }
}

fn vec_zero_binary(func: &Function, fold: &Fold, kind: Binary, x: Value, y: Value) -> Option<Op> {
    match kind {
        Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat | Binary::AvgRound
            if is_zero_bits(func, fold, x) && is_zero_bits(func, fold, y) => Some(Op::Const(0)),
        _ if !func.ty(x).is_vector() => None,
        Binary::Band | Binary::Imul | Binary::Umin if is_zero_bits(func, fold, x) || is_zero_bits(func, fold, y) => Some(Op::Const(0)),
        Binary::Bxor | Binary::Isub | Binary::SsubSat | Binary::UsubSat if fold.find(x) == fold.find(y) => Some(Op::Const(0)),
        Binary::UsubSat | Binary::Ishl | Binary::Ushr | Binary::Sshr if is_zero_bits(func, fold, x) => Some(Op::Const(0)),
        _ => None,
    }
}

fn vec_zero_icmp(func: &Function, fold: &Fold, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    if !func.ty(x).is_vector() {
        return None;
    }
    let eq_false = fold.find(x) == fold.find(y)
        && matches!(cc, IntCC::NotEqual | IntCC::SignedLessThan | IntCC::SignedGreaterThan | IntCC::UnsignedLessThan | IntCC::UnsignedGreaterThan);
    let u_false = (cc == IntCC::UnsignedLessThan && is_zero_bits(func, fold, y))
        || (cc == IntCC::UnsignedGreaterThan && is_zero_bits(func, fold, x));
    (eq_false || u_false).then_some(Op::Const(0))
}

fn vec_zero_other(func: &Function, fold: &Fold, op: &Op) -> Option<Op> {
    match *op {
        Op::Unary(Unary::Ineg | Unary::Bswap | Unary::Bitrev | Unary::Popcnt | Unary::Fabs, x)
            if func.ty(x).is_vector() && is_zero_bits(func, fold, x) => Some(Op::Const(0)),
        Op::Select(_, x, y) if func.ty(x).is_vector() && is_zero_bits(func, fold, x) && is_zero_bits(func, fold, y) => Some(Op::Const(0)),
        Op::Icmp(cc, x, y) => vec_zero_icmp(func, fold, cc, x, y),
        Op::Convert(
            Convert::Sextend | Convert::Uextend | Convert::Ireduce | Convert::NarrowSat { .. } | Convert::FromSint | Convert::FromUint | Convert::ToSintSat | Convert::ToUintSat,
            x,
        ) if func.ty(x).is_vector() && is_zero_bits(func, fold, x) => Some(Op::Const(0)),
        _ => None,
    }
}

fn vec_zero_bitcast(func: &Function, fold: &Fold, x: Value, res: Option<Value>) -> Option<Op> {
    let ty = res.map(|r| func.ty(r))?;
    if (ty.is_vector() || ty.is_small_int() || ty.is_float()) && is_zero_bits(func, fold, x) {
        return Some(Op::Const(0));
    }
    match fold.def(func, x) {
        Some(&Op::Convert(Convert::Bitcast, v)) if func.ty(fold.find(v)) != ty => Some(Op::Convert(Convert::Bitcast, fold.find(v))),
        _ => None,
    }
}

pub(super) fn vec_zero_fold(func: &Function, fold: &Fold, op: &Op, res: Option<Value>) -> Option<Op> {
    match *op {
        Op::Convert(Convert::Bitcast, x) => vec_zero_bitcast(func, fold, x, res),
        Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
            let count = func.ty(a).vec_shape()?.1;
            let (za, zb) = (is_zero_bits(func, fold, a), is_zero_bits(func, fold, b));
            mask[..usize::from(count)].iter().all(|&i| if i < count { za } else { zb }).then_some(Op::Const(0))
        }
        Op::Insert(vec, scalar, _) => (is_zero_bits(func, fold, vec) && is_zero_bits(func, fold, scalar)).then_some(Op::Const(0)),
        Op::Extract(vec, _) => is_zero_bits(func, fold, vec).then_some(Op::Const(0)),
        Op::Reduce(kind, vec) if !matches!(kind, ReduceOp::Fmin | ReduceOp::Fmax) => is_zero_bits(func, fold, vec).then_some(Op::Const(0)),
        Op::Binary(kind, x, y) => vec_zero_binary(func, fold, kind, x, y),
        _ => vec_zero_other(func, fold, op),
    }
}

fn is_zero_bits(func: &Function, fold: &Fold, val: Value) -> bool {
    let val = fold.find(val);
    fold.known(val) == Some(0)
        || matches!(fold.def(func, val), Some(&Op::Const(0)))
        || matches!(fold.def(func, val), Some(&Op::Convert(Convert::Bitcast, z)) if fold.known(z) == Some(0) || matches!(fold.def(func, fold.find(z)), Some(&Op::Const(0))))
}

#[derive(Clone, Copy)]
enum VecPattern {
    Splat(Type, u64),
    One(u8, Type, u64),
    Pair(u64, u64),
}

struct ByteEval<'a> {
    func: &'a Function,
    defs: &'a [Option<Inst>],
    cache: Vec<Option<[Option<u8>; 16]>>,
    visiting: Vec<bool>,
}

impl<'a> ByteEval<'a> {
    fn new(func: &'a Function, defs: &'a [Option<Inst>]) -> Self {
        let n = func.types.len();
        Self { func, defs, cache: vec![None; n], visiting: vec![false; n] }
    }

    fn eval(&mut self, val: Value, depth: u8) -> [Option<u8>; 16] {
        if self.func.ty(val).bytes() > 16 {
            return [None; 16];
        }
        let idx = val.index();
        if let Some(cached) = self.cache[idx] {
            return cached;
        }
        if depth >= 64 || self.visiting[idx] {
            return [None; 16];
        }
        let Some(inst) = self.defs.get(idx).copied().flatten() else {
            return [None; 16];
        };
        self.visiting[idx] = true;
        let out = self.eval_op(&self.func.insts[inst.index()].op, self.func.ty(val), depth + 1);
        self.visiting[idx] = false;
        self.cache[idx] = Some(out);
        out
    }

    fn eval_op(&mut self, op: &Op, ty: Type, depth: u8) -> [Option<u8>; 16] {
        match *op {
            Op::Const(imm) => {
                if ty.is_vector() {
                    return if imm == 0 { [Some(0); 16] } else { [None; 16] };
                }
                if !(ty.is_small_int() || ty.is_float()) {
                    return [None; 16];
                }
                let mut out = [None; 16];
                let le = (imm as u64).to_le_bytes();
                for (dst, &byte) in out[..ty.bytes() as usize].iter_mut().zip(&le) {
                    *dst = Some(byte);
                }
                out
            }
            Op::Convert(Convert::Bitcast, inner) if self.func.ty(inner).bytes() == ty.bytes() => self.eval(inner, depth),
            Op::Binary(Binary::Iconcat, lo, hi) if ty == Type::I128 => {
                let mut out = [None; 16];
                out[..8].copy_from_slice(&self.eval(lo, depth)[..8]);
                out[8..16].copy_from_slice(&self.eval(hi, depth)[..8]);
                out
            }
            Op::Insert(vec, scalar, lane_idx) => {
                let Some((lane, count)) = ty.vec_shape() else { return [None; 16] };
                if lane_idx >= count {
                    return [None; 16];
                }
                let mut out = self.eval(vec, depth);
                let sb = self.eval(scalar, depth);
                let eb = lane.bytes() as usize;
                let start = usize::from(lane_idx) * eb;
                out[start..start + eb].copy_from_slice(&sb[..eb]);
                out
            }
            Op::Shuffle(a, b, ShuffleMask::Const(mask)) => {
                let Some((lane, count)) = ty.vec_shape() else { return [None; 16] };
                let eb = lane.bytes() as usize;
                let ba = self.eval(a, depth);
                let bb = self.eval(b, depth);
                let mut out = [None; 16];
                for (i, &sel) in mask[..usize::from(count)].iter().enumerate() {
                    let dst = &mut out[i * eb..(i + 1) * eb];
                    if sel < count {
                        let s = usize::from(sel) * eb;
                        dst.copy_from_slice(&ba[s..s + eb]);
                    } else if sel < 2 * count {
                        let s = usize::from(sel - count) * eb;
                        dst.copy_from_slice(&bb[s..s + eb]);
                    }
                }
                out
            }
            Op::Extract(vec, lane_idx) => {
                let Some((lane, count)) = self.func.ty(vec).vec_shape() else { return [None; 16] };
                if self.func.ty(vec).bytes() > 16 {
                    return [None; 16];
                }
                if lane_idx >= count {
                    return [None; 16];
                }
                let vb = self.eval(vec, depth);
                let eb = lane.bytes() as usize;
                let start = usize::from(lane_idx) * eb;
                let mut out = [None; 16];
                out[..eb].copy_from_slice(&vb[start..start + eb]);
                out
            }
            Op::Reduce(rop, vec) => {
                let vb = self.eval(vec, depth);
                let Some(imm) = eval_reduce_const(&vb, rop, self.func.ty(vec)) else { return [None; 16] };
                let mut out = [None; 16];
                let le = (imm as u64).to_le_bytes();
                for (dst, &byte) in out[..ty.bytes() as usize].iter_mut().zip(&le) {
                    *dst = Some(byte);
                }
                out
            }
            Op::Unary(kind, x) => self.eval_vec_unary(kind, ty, x, depth),
            Op::Binary(kind, x, y) => self.eval_vec_binary(kind, ty, x, y, depth),
            Op::Icmp(cc, x, y) => self.eval_vec_icmp(cc, ty, x, y, depth),
            Op::Convert(kind, x) => self.eval_vec_convert(kind, ty, x, depth),
            Op::Select(c, x, y) => self.eval_vec_select(ty, (c, x, y), depth),
            _ => [None; 16],
        }
    }

    fn eval_vec_unary(&mut self, op: Unary, ty: Type, x: Value, depth: u8) -> [Option<u8>; 16] {
        let Some((lane, count)) = ty.vec_shape() else { return [None; 16] };
        let bx = self.eval(x, depth);
        let mut out = [None; 16];
        for i in 0..usize::from(count) {
            let Some(lx) = read_lane(&bx, lane, i) else { return [None; 16] };
            let val = match op {
                Unary::Fneg if lane.is_float() => lx ^ lane.to_type().sign_bit(),
                Unary::Fabs if lane.is_float() => lx & !lane.to_type().sign_bit(),
                _ if lane.is_int() => {
                    let bits = lane.bits();
                    let unused = 64 - bits;
                    let u = extend(lx, lane.to_type(), false) as u64;
                    match op {
                        Unary::Ineg => lx.wrapping_neg(),
                        Unary::Bnot => !lx,
                        Unary::Bswap => (u.swap_bytes() >> unused) as i64,
                        Unary::Bitrev => (u.reverse_bits() >> unused) as i64,
                        Unary::Clz => i64::from(u.leading_zeros() - unused),
                        Unary::Ctz => i64::from(u.trailing_zeros().min(bits)),
                        Unary::Popcnt => i64::from(u.count_ones()),
                        _ => return [None; 16],
                    }
                }
                _ => return [None; 16],
            };
            write_lane(&mut out, lane, i, val);
        }
        out
    }

    fn eval_vec_binary(&mut self, op: Binary, ty: Type, x: Value, y: Value, depth: u8) -> [Option<u8>; 16] {
        let Some((lane, count)) = ty.vec_shape() else { return [None; 16] };
        if !lane.is_int() {
            return [None; 16];
        }
        let bx = self.eval(x, depth);
        let by = self.eval(y, depth);
        let lty = lane.to_type();
        let scalar_y = Lane::from_type(self.func.ty(y));
        let mut out = [None; 16];
        for i in 0..usize::from(count) {
            let Some(lx) = read_lane(&bx, lane, i) else { return [None; 16] };
            let ly = match scalar_y {
                Some(sl) => match read_lane(&by, sl, 0) {
                    Some(v) => v,
                    None => return [None; 16],
                },
                None => match read_lane(&by, lane, i) {
                    Some(v) => v,
                    None => return [None; 16],
                },
            };
            let val = match op {
                Binary::Umin => (extend(lx, lty, false) as u64).min(extend(ly, lty, false) as u64) as i64,
                Binary::Umax => (extend(lx, lty, false) as u64).max(extend(ly, lty, false) as u64) as i64,
                _ => match binary(op, lty, lx, ly) {
                    Some(v) => v,
                    None => return [None; 16],
                },
            };
            write_lane(&mut out, lane, i, val);
        }
        out
    }

    fn eval_vec_icmp(&mut self, cc: crate::ir::IntCC, ty: Type, x: Value, y: Value, depth: u8) -> [Option<u8>; 16] {
        let Some((src_lane, count)) = self.func.ty(x).vec_shape() else { return [None; 16] };
        let Some((dst_lane, dst_count)) = ty.vec_shape() else { return [None; 16] };
        if !src_lane.is_int() || count != dst_count {
            return [None; 16];
        }
        let bx = self.eval(x, depth);
        let by = self.eval(y, depth);
        let src_ty = src_lane.to_type();
        let mut out = [None; 16];
        for i in 0..usize::from(count) {
            let (Some(lx), Some(ly)) = (read_lane(&bx, src_lane, i), read_lane(&by, src_lane, i)) else {
                return [None; 16];
            };
            write_lane(&mut out, dst_lane, i, if holds(cc, src_ty, lx, ly) { -1 } else { 0 });
        }
        out
    }

    fn eval_vec_convert(&mut self, op: Convert, ty: Type, x: Value, depth: u8) -> [Option<u8>; 16] {
        let (Some((sl, sc)), Some((dl, dc))) = (self.func.ty(x).vec_shape(), ty.vec_shape()) else {
            return [None; 16];
        };
        if !sl.is_int() || !dl.is_int() || sc != dc {
            return [None; 16];
        }
        let bx = self.eval(x, depth);
        let (sty, dty) = (sl.to_type(), dl.to_type());
        let mut out = [None; 16];
        for i in 0..usize::from(dc) {
            let Some(lx) = read_lane(&bx, sl, i) else { return [None; 16] };
            let val = match op {
                Convert::Uextend if dty.bytes() > sty.bytes() => extend(lx, sty, false),
                Convert::Sextend if dty.bytes() > sty.bytes() => extend(lx, sty, true),
                Convert::Ireduce if dty.bytes() < sty.bytes() => extend(lx, dty, true),
                Convert::NarrowSat { signed } if sty.bytes() == 2 * dty.bytes() => clamp_narrow(sty, dty, signed, lx),
                _ => return [None; 16],
            };
            write_lane(&mut out, dl, i, val);
        }
        out
    }

    fn eval_vec_select(&mut self, ty: Type, (c, x, y): (Value, Value, Value), depth: u8) -> [Option<u8>; 16] {
        let Some((lane, count)) = ty.vec_shape() else { return [None; 16] };
        let Some((c_lane, c_count)) = self.func.ty(c).vec_shape() else { return [None; 16] };
        if count != c_count || c_lane != lane.int_mask() {
            return [None; 16];
        }
        let (bc, bx, by) = (self.eval(c, depth), self.eval(x, depth), self.eval(y, depth));
        let eb = lane.bytes() as usize;
        let mut out = [None; 16];
        for i in 0..usize::from(count) {
            let range = i * eb..(i + 1) * eb;
            let all_ones = bc[range.clone()].iter().all(|&b| b == Some(0xFF));
            let all_zeros = bc[range.clone()].iter().all(|&b| b == Some(0));
            let same_arms = range.clone().all(|j| bx[j].is_some() && bx[j] == by[j]);
            if !all_ones && !all_zeros && !same_arms {
                return [None; 16];
            }
            let src = if all_zeros { &by[range.clone()] } else { &bx[range.clone()] };
            if src.iter().any(Option::is_none) {
                return [None; 16];
            }
            out[range].copy_from_slice(src);
        }
        out
    }
}

fn read_lane(bytes: &[Option<u8>; 16], lane: Lane, idx: usize) -> Option<i64> {
    let eb = lane.bytes() as usize;
    let mut raw = [0_u8; 8];
    for (dst, &byte) in raw[..eb].iter_mut().zip(&bytes[idx * eb..(idx + 1) * eb]) {
        *dst = byte?;
    }
    let u = u64::from_le_bytes(raw) as i64;
    let ty = lane.to_type();
    Some(if ty.is_small_int() { extend(u, ty, true) } else { u })
}

fn write_lane(out: &mut [Option<u8>; 16], lane: Lane, idx: usize, val: i64) {
    let eb = lane.bytes() as usize;
    let le = (val as u64).to_le_bytes();
    for (dst, &byte) in out[idx * eb..(idx + 1) * eb].iter_mut().zip(&le[..eb]) {
        *dst = Some(byte);
    }
}

fn full_bytes(bytes: &[Option<u8>; 16], total: usize) -> Option<[u8; 16]> {
    let mut out = [0_u8; 16];
    for (dst, &src) in out[..total].iter_mut().zip(&bytes[..total]) {
        *dst = src?;
    }
    Some(out)
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
            return (Type::Vec { lane: Lane::I32, count: 4 }, VecPattern::Splat(Type::I32, w32));
        }
        return (Type::Vec { lane: Lane::I64, count: 2 }, VecPattern::Splat(Type::I64, lo));
    }
    if hi == 0 {
        if (lo >> 32) == 0 && vec_ty != (Type::Vec { lane: Lane::I64, count: 2 }) {
            return (Type::Vec { lane: Lane::I32, count: 4 }, VecPattern::One(0, Type::I32, lo));
        }
        return (Type::Vec { lane: Lane::I64, count: 2 }, VecPattern::One(0, Type::I64, lo));
    }
    if lo == 0 {
        return (Type::Vec { lane: Lane::I64, count: 2 }, VecPattern::One(1, Type::I64, hi));
    }
    (Type::Vec { lane: Lane::I64, count: 2 }, VecPattern::Pair(lo, hi))
}

fn is_zero_vec(func: &Function, defs: &[Option<Inst>], val: Value) -> bool {
    func.is_zero_vec(&|value| defs.get(value.index()).copied().flatten(), val)
}

fn const_matches(func: &Function, defs: &[Option<Inst>], val: Value, ty: Type, want: u64) -> bool {
    func.ty(val) == ty
        && defs.get(val.index()).copied().flatten().is_some_and(|i| {
            matches!(func.insts[i.index()].op, Op::Const(imm) if extend(imm, ty, false) as u64 == extend(want as i64, ty, false) as u64)
        })
}

fn matches_pattern(func: &Function, defs: &[Option<Inst>], inst: Inst, pat: VecPattern) -> bool {
    match (pat, func.insts[inst.index()].op.clone()) {
        (VecPattern::Splat(ty, val), Op::Shuffle(a, b, ShuffleMask::Const(m))) if a == b && m == [0; 16] => {
            defs.get(a.index()).copied().flatten().is_some_and(|ai| {
                matches!(func.insts[ai.index()].op, Op::Insert(z, c, 0) if is_zero_vec(func, defs, z) && const_matches(func, defs, c, ty, val))
            })
        }
        (VecPattern::One(idx, ty, val), Op::Insert(z, c, at)) => {
            at == idx && is_zero_vec(func, defs, z) && const_matches(func, defs, c, ty, val)
        }
        (VecPattern::Pair(lo, hi), Op::Insert(v0, c1, 1)) if const_matches(func, defs, c1, Type::I64, hi) => {
            defs.get(v0.index()).copied().flatten().is_some_and(|i0| {
                matches!(func.insts[i0.index()].op, Op::Insert(z, c0, 0) if is_zero_vec(func, defs, z) && const_matches(func, defs, c0, Type::I64, lo))
            })
        }
        _ => false,
    }
}

fn is_canonical(func: &Function, defs: &[Option<Inst>], inst: Inst, vec_ty: Type, bytes: &[u8; 16], is_shift_rhs: bool) -> bool {
    let total = vec_ty.bytes() as usize;
    if bytes[..total].iter().all(|&b| b == 0) {
        return matches!(func.insts[inst.index()].op, Op::Const(0));
    }
    if total == 8 {
        return matches!(func.insts[inst.index()].op, Op::Convert(Convert::Bitcast, x) if const_matches(func, defs, x, Type::I64, pack_u64(bytes, 0)));
    }
    let (target_ty, pat) = choose_plan(vec_ty, bytes, is_shift_rhs);
    if vec_ty == target_ty {
        return matches_pattern(func, defs, inst, pat);
    }
    let Op::Convert(Convert::Bitcast, inner) = func.insts[inst.index()].op else {
        return false;
    };
    func.ty(inner) == target_ty && defs.get(inner.index()).copied().flatten().is_some_and(|i| matches_pattern(func, defs, i, pat))
}

fn build_pattern_op(func: &mut Function, target_ty: Type, pat: VecPattern, out: &mut Vec<Inst>) -> Op {
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

fn emit_canonical(func: &mut Function, inst: Inst, vec_ty: Type, bytes: &[u8; 16], is_shift_rhs: bool, out: &mut Vec<Inst>) {
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

fn reduce_step(op: ReduceOp, lane_ty: Type, acc: i64, v: i64) -> Option<i64> {
    let u = |x| extend(x, lane_ty, false) as u64;
    Some(match op {
        ReduceOp::Add => extend(acc.wrapping_add(v), lane_ty, true),
        ReduceOp::And => acc & v,
        ReduceOp::Or => acc | v,
        ReduceOp::Xor => acc ^ v,
        ReduceOp::Smin => acc.min(v),
        ReduceOp::Smax => acc.max(v),
        ReduceOp::Umin => extend(u(acc).min(u(v)) as i64, lane_ty, true),
        ReduceOp::Umax => extend(u(acc).max(u(v)) as i64, lane_ty, true),
        ReduceOp::Fmin | ReduceOp::Fmax | ReduceOp::Movemask => return None,
    })
}

fn eval_reduce_const(bytes: &[Option<u8>; 16], op: ReduceOp, vec_ty: Type) -> Option<i64> {
    let (lane, count) = (vec_ty.bytes() <= 16).then(|| vec_ty.vec_shape()).flatten()?;
    if !lane.is_int() {
        return None;
    }
    if op == ReduceOp::Movemask {
        let eb = lane.bytes() as usize;
        let mut mask = 0_i64;
        for i in 0..usize::from(count) {
            mask |= i64::from((bytes[i * eb + eb - 1]? >> 7) & 1) << i;
        }
        return Some(extend(mask, Type::I32, true));
    }
    let lane_ty = lane.to_type();
    let mut acc = read_lane(bytes, lane, 0)?;
    for i in 1..usize::from(count) {
        acc = reduce_step(op, lane_ty, acc, read_lane(bytes, lane, i)?)?;
    }
    Some(acc)
}

fn collect_scalar_folds(func: &Function, eval: &mut ByteEval) -> Vec<(Inst, Op)> {
    let mut updates = Vec::new();
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        match data.op {
            Op::Extract(vec, lane_idx) if func.ty(vec).bytes() <= 16 && (func.ty(res).is_small_int() || func.ty(res).is_float()) => {
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
        let Some(res) = func.insts[inst.index()].result else { continue };
        let ty = func.ty(res);
        if ty.is_valid_vec() && !matches!(func.insts[inst.index()].op, Op::Load(..) | Op::StackLoad(..)) {
            let bytes = eval.eval(res, 0);
            known_vec[res.index()] = full_bytes(&bytes, ty.bytes() as usize);
        }
    }
    known_vec
}

fn rewrite_canonical_vecs(func: &mut Function, defs: &[Option<Inst>], known_vec: &[Option<[u8; 16]>]) -> bool {
    let (mut root, mut shift_rhs) = (vec![false; func.types.len()], vec![false; func.types.len()]);
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        if let Op::Binary(Binary::Ishl | Binary::Ushr | Binary::Sshr, _, rhs) = data.op {
            shift_rhs[rhs.index()] = true;
        }
        if !data.result.is_some_and(|r| known_vec[r.index()].is_some()) {
            data.op.for_operands(|v| root[v.index()] |= known_vec[v.index()].is_some());
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
                && !is_canonical(func, defs, inst, func.ty(res), &bytes, shift_rhs[res.index()])
            {
                emit_canonical(func, inst, func.ty(res), &bytes, shift_rhs[res.index()], &mut out);
                changed = true;
                continue;
            }
            out.push(inst);
        }
        func.blocks[b].insts = out;
    }
    changed
}

fn fold_const_vectors(func: &mut Function) -> bool {
    let vector = |inst: &Inst| {
        matches!(func.insts[inst.index()].op, Op::Extract(..) | Op::Reduce(..))
            || func.insts[inst.index()].result.is_some_and(|res| func.ty(res).is_valid_vec())
    };
    if !func.blocks.iter().flat_map(|b| &b.insts).any(vector) { return false; }
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
