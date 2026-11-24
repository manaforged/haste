use super::super::{binary, extend, holds};
use super::clamp_narrow;
use crate::ir::{
    Binary, Convert, Function, Inst, IntCC, Lane, Op, ReduceOp, ShuffleMask, Type, Unary, Value,
};

pub(super) struct ByteEval<'a> {
    func: &'a Function,
    defs: &'a [Option<Inst>],
    cache: Vec<Option<[Option<u8>; 16]>>,
    visiting: Vec<bool>,
}

impl<'a> ByteEval<'a> {
    pub(super) fn new(func: &'a Function, defs: &'a [Option<Inst>]) -> Self {
        let n = func.types.len();
        Self {
            func,
            defs,
            cache: vec![None; n],
            visiting: vec![false; n],
        }
    }

    pub(super) fn eval(&mut self, val: Value, depth: u8) -> [Option<u8>; 16] {
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
        let out = self.eval_op(
            &self.func.insts[inst.index()].op,
            self.func.ty(val),
            depth + 1,
        );
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
            Op::Convert(Convert::Bitcast, inner) if self.func.ty(inner).bytes() == ty.bytes() => {
                self.eval(inner, depth)
            }
            Op::Binary(Binary::Iconcat, lo, hi) if ty == Type::I128 => {
                let mut out = [None; 16];
                out[..8].copy_from_slice(&self.eval(lo, depth)[..8]);
                out[8..16].copy_from_slice(&self.eval(hi, depth)[..8]);
                out
            }
            Op::Insert(vec, scalar, lane_idx) => {
                let Some((lane, count)) = ty.vec_shape() else {
                    return [None; 16];
                };
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
                let Some((lane, count)) = ty.vec_shape() else {
                    return [None; 16];
                };
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
                let Some((lane, count)) = self.func.ty(vec).vec_shape() else {
                    return [None; 16];
                };
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
                let Some(imm) = eval_reduce_const(&vb, rop, self.func.ty(vec)) else {
                    return [None; 16];
                };
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
        let Some((lane, count)) = ty.vec_shape() else {
            return [None; 16];
        };
        let bx = self.eval(x, depth);
        let mut out = [None; 16];
        for i in 0..usize::from(count) {
            let Some(lx) = read_lane(&bx, lane, i) else {
                return [None; 16];
            };
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

    fn eval_vec_binary(
        &mut self,
        op: Binary,
        ty: Type,
        x: Value,
        y: Value,
        depth: u8,
    ) -> [Option<u8>; 16] {
        let Some((lane, count)) = ty.vec_shape() else {
            return [None; 16];
        };
        if !lane.is_int() {
            return [None; 16];
        }
        let bx = self.eval(x, depth);
        let by = self.eval(y, depth);
        let lty = lane.to_type();
        let scalar_y = Lane::from_type(self.func.ty(y));
        let mut out = [None; 16];
        for i in 0..usize::from(count) {
            let Some(lx) = read_lane(&bx, lane, i) else {
                return [None; 16];
            };
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
                Binary::Umin => {
                    (extend(lx, lty, false) as u64).min(extend(ly, lty, false) as u64) as i64
                }
                Binary::Umax => {
                    (extend(lx, lty, false) as u64).max(extend(ly, lty, false) as u64) as i64
                }
                _ => match binary(op, lty, lx, ly) {
                    Some(v) => v,
                    None => return [None; 16],
                },
            };
            write_lane(&mut out, lane, i, val);
        }
        out
    }

    fn eval_vec_icmp(
        &mut self,
        cc: IntCC,
        ty: Type,
        x: Value,
        y: Value,
        depth: u8,
    ) -> [Option<u8>; 16] {
        let Some((src_lane, count)) = self.func.ty(x).vec_shape() else {
            return [None; 16];
        };
        let Some((dst_lane, dst_count)) = ty.vec_shape() else {
            return [None; 16];
        };
        if !src_lane.is_int() || count != dst_count {
            return [None; 16];
        }
        let bx = self.eval(x, depth);
        let by = self.eval(y, depth);
        let src_ty = src_lane.to_type();
        let mut out = [None; 16];
        for i in 0..usize::from(count) {
            let (Some(lx), Some(ly)) = (read_lane(&bx, src_lane, i), read_lane(&by, src_lane, i))
            else {
                return [None; 16];
            };
            write_lane(
                &mut out,
                dst_lane,
                i,
                if holds(cc, src_ty, lx, ly) { -1 } else { 0 },
            );
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
            let Some(lx) = read_lane(&bx, sl, i) else {
                return [None; 16];
            };
            let val = match op {
                Convert::Uextend if dty.bytes() > sty.bytes() => extend(lx, sty, false),
                Convert::Sextend if dty.bytes() > sty.bytes() => extend(lx, sty, true),
                Convert::Ireduce if dty.bytes() < sty.bytes() => extend(lx, dty, true),
                Convert::NarrowSat { signed } if sty.bytes() == 2 * dty.bytes() => {
                    clamp_narrow(sty, dty, signed, lx)
                }
                _ => return [None; 16],
            };
            write_lane(&mut out, dl, i, val);
        }
        out
    }

    fn eval_vec_select(
        &mut self,
        ty: Type,
        (c, x, y): (Value, Value, Value),
        depth: u8,
    ) -> [Option<u8>; 16] {
        let Some((lane, count)) = ty.vec_shape() else {
            return [None; 16];
        };
        let Some((c_lane, c_count)) = self.func.ty(c).vec_shape() else {
            return [None; 16];
        };
        if count != c_count || c_lane != lane.int_mask() {
            return [None; 16];
        }
        let (bc, bx, by) = (
            self.eval(c, depth),
            self.eval(x, depth),
            self.eval(y, depth),
        );
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
            let src = if all_zeros {
                &by[range.clone()]
            } else {
                &bx[range.clone()]
            };
            if src.iter().any(Option::is_none) {
                return [None; 16];
            }
            out[range].copy_from_slice(src);
        }
        out
    }
}

pub(super) fn read_lane(bytes: &[Option<u8>; 16], lane: Lane, idx: usize) -> Option<i64> {
    let eb = lane.bytes() as usize;
    let mut raw = [0_u8; 8];
    for (dst, &byte) in raw[..eb].iter_mut().zip(&bytes[idx * eb..(idx + 1) * eb]) {
        *dst = byte?;
    }
    let u = u64::from_le_bytes(raw) as i64;
    let ty = lane.to_type();
    Some(if ty.is_small_int() {
        extend(u, ty, true)
    } else {
        u
    })
}

fn write_lane(out: &mut [Option<u8>; 16], lane: Lane, idx: usize, val: i64) {
    let eb = lane.bytes() as usize;
    let le = (val as u64).to_le_bytes();
    for (dst, &byte) in out[idx * eb..(idx + 1) * eb].iter_mut().zip(&le[..eb]) {
        *dst = Some(byte);
    }
}

pub(super) fn full_bytes(bytes: &[Option<u8>; 16], total: usize) -> Option<[u8; 16]> {
    let mut out = [0_u8; 16];
    for (dst, &src) in out[..total].iter_mut().zip(&bytes[..total]) {
        *dst = src?;
    }
    Some(out)
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

pub(super) fn eval_reduce_const(
    bytes: &[Option<u8>; 16],
    op: ReduceOp,
    vec_ty: Type,
) -> Option<i64> {
    let (lane, count) = (vec_ty.bytes() <= 16)
        .then(|| vec_ty.vec_shape())
        .flatten()?;
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
