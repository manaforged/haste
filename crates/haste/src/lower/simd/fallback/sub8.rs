use anyhow::Result;
use haste_codegen::{IntCC, Lane, ReduceOp, Type, Value};

use crate::body::BinOp;
use crate::sema::intrinsic::Simd;

use super::super::super::Lower;
use super::super::super::ops::Num;
use super::super::{Lanes, int_for_bytes};

impl Lower<'_, '_> {
    pub(in super::super) fn sub8_lane_intrinsic(
        &mut self,
        op: Simd,
        lanes: &Lanes,
        args: &[Value],
    ) -> Result<Option<Value>> {
        let Some((wlanes, wvec)) = lanes.widened() else {
            return Ok(None);
        };
        match (op, args) {
            (Simd::Sum { mul }, &[x, ref start @ ..]) => {
                if lanes.count == 1 {
                    let v = self.ins().load(lanes.machine, x, 0);
                    let bin = if mul { BinOp::Mul } else { BinOp::Add };
                    let out = match start.first().copied() {
                        Some(init) => self.lane_op(bin, lanes.num, (init, v), lanes.machine)?,
                        None => v,
                    };
                    return Ok(Some(out));
                }
                let mut v = self.widen_sub8(lanes.bytes(), wvec, x);
                if mul {
                    let one = self.lane_identity_one(lanes.num)?;
                    v = self.fill_upper_lanes(wvec, lanes.count as usize, v, one);
                }
                self.vec_sum(&wlanes, wvec, v, (start.first().copied(), mul))
                    .map(Some)
            }
            (Simd::Unary(_) | Simd::Pair(_) | Simd::Fma, _) => {
                let widened: Vec<Value> = args
                    .iter()
                    .map(|&p| {
                        let w = self.widen_sub8(lanes.bytes(), wvec, p);
                        self.store_vec(w, 8)
                    })
                    .collect();
                let Some(wide_res) = self.vec_lane_intrinsic(op, &wlanes, &widened)? else {
                    return Ok(None);
                };
                Ok(Some(self.narrow_sub8(lanes.bytes(), wvec, wide_res)))
            }
            _ => Ok(None),
        }
    }

    pub(super) fn sub8_simd(
        &mut self,
        op: Simd,
        (from, to): (&Lanes, &Lanes),
        args: &[Value],
    ) -> Result<Option<Value>> {
        let Some((wf, wf_ty)) = from.widened() else {
            return Ok(None);
        };
        let Some((wt, _)) = to.widened() else {
            return Ok(None);
        };
        match (op, args) {
            (Simd::Splat, &[scalar]) => {
                let Some(w_res) =
                    self.vec_simd_lanes(None, Simd::Splat, (wf, wf), &[], &[scalar])?
                else {
                    return Ok(None);
                };
                Ok(Some(self.narrow_sub8(from.bytes(), wf_ty, w_res)))
            }
            (Simd::Bitmask, &[x]) => {
                let w = self.widen_sub8(from.bytes(), wf_ty, x);
                let mask = self.ins().reduce(ReduceOp::Movemask, w);
                let mask64 = self.resize(mask, (Type::I32, false), Type::I64);
                let keep = self.ins().iconst(Type::I64, (1i64 << from.count) - 1);
                let trimmed = self.ins().band(mask64, keep);
                Ok(Some(self.resize(trimmed, (Type::I64, false), to.machine)))
            }
            (Simd::Sqrt, &[x]) => {
                let wide_x = self.widen_sub8(from.bytes(), wf_ty, x);
                let w = self.store_vec(wide_x, 8);
                let Some(w_res) = self.vec_simd_lanes(None, Simd::Sqrt, (wf, wf), &[], &[w])?
                else {
                    return Ok(None);
                };
                Ok(Some(self.narrow_sub8(from.bytes(), wf_ty, w_res)))
            }
            (Simd::Reduce { all }, &[x]) => {
                let mut v = self.widen_sub8(from.bytes(), wf_ty, x);
                if all {
                    let first = self.ins().extract(v, 0);
                    v = self.fill_upper_lanes(wf_ty, from.count as usize, v, first);
                }
                let zero = self.ins().iconst(wf_ty, 0);
                let mask = self.ins().icmp(IntCC::NotEqual, v, zero);
                let red = self
                    .ins()
                    .reduce(if all { ReduceOp::And } else { ReduceOp::Or }, mask);
                Ok(Some(self.ins().icmp_imm(IntCC::NotEqual, red, 0)))
            }
            (Simd::Fold(bin), &[x]) if wf_ty.lane().is_some_and(Lane::is_int) => {
                let red = match bin {
                    BinOp::BitAnd => ReduceOp::And,
                    BinOp::BitOr => ReduceOp::Or,
                    BinOp::BitXor => ReduceOp::Xor,
                    BinOp::Add => ReduceOp::Add,
                    _ => return Ok(None),
                };
                let mut v = self.widen_sub8(from.bytes(), wf_ty, x);
                if bin == BinOp::BitAnd {
                    let first = self.ins().extract(v, 0);
                    v = self.fill_upper_lanes(wf_ty, from.count as usize, v, first);
                }
                Ok(Some(self.ins().reduce(red, v)))
            }
            (Simd::Extreme { max }, &[x]) => {
                let v = self.widen_sub8(from.bytes(), wf_ty, x);
                let first = self.ins().extract(v, 0);
                let filled = self.fill_upper_lanes(wf_ty, from.count as usize, v, first);
                let red = match (from.num, max) {
                    (Num::Float(_), true) => ReduceOp::Fmax,
                    (Num::Float(_), false) => ReduceOp::Fmin,
                    (Num::Int { signed: true, .. }, true) => ReduceOp::Smax,
                    (Num::Int { signed: true, .. }, false) => ReduceOp::Smin,
                    (_, true) => ReduceOp::Umax,
                    (_, false) => ReduceOp::Umin,
                };
                Ok(Some(self.ins().reduce(red, filled)))
            }
            (Simd::Lanewise(_), &[x, y]) if from.size == to.size && from.count == to.count => {
                let (ux, uy) = (
                    self.widen_sub8(from.bytes(), wf_ty, x),
                    self.widen_sub8(from.bytes(), wf_ty, y),
                );
                let (wx, wy) = (self.store_vec(ux, 8), self.store_vec(uy, 8));
                let Some(w_res) = self.vec_simd_lanes(None, op, (wf, wt), &[], &[wx, wy])? else {
                    return Ok(None);
                };
                Ok(Some(self.narrow_sub8(from.bytes(), wf_ty, w_res)))
            }
            (Simd::Select, &[mask, yes, no]) if from.size == to.size && from.count == to.count => {
                let Some((_, wt_ty)) = to.widened() else {
                    return Ok(None);
                };
                let (um, uy, un) = (
                    self.widen_sub8(from.bytes(), wf_ty, mask),
                    self.widen_sub8(to.bytes(), wt_ty, yes),
                    self.widen_sub8(to.bytes(), wt_ty, no),
                );
                let (wm, wy, wn) = (
                    self.store_vec(um, 8),
                    self.store_vec(uy, 8),
                    self.store_vec(un, 8),
                );
                let Some(w_res) =
                    self.vec_simd_lanes(None, Simd::Select, (wf, wt), &[], &[wm, wy, wn])?
                else {
                    return Ok(None);
                };
                Ok(Some(self.narrow_sub8(to.bytes(), wt_ty, w_res)))
            }
            _ => Ok(None),
        }
    }

    pub(in super::super) fn widen_sub8(&mut self, bytes: i32, wvec: Type, ptr: Value) -> Value {
        let Some(small_ty) = int_for_bytes(bytes) else {
            return self.ins().load(wvec, ptr, 0);
        };
        let raw = self.ins().load(small_ty, ptr, 0);
        let wide = self.ins().uextend(Type::I64, raw);
        let slot = self.b.create_stack_slot(8, 3);
        self.ins().stack_store(wide, slot, 0);
        self.ins().stack_load(wvec, slot, 0)
    }

    pub(in super::super) fn narrow_sub8(&mut self, bytes: i32, wvec: Type, w_addr: Value) -> Value {
        let w = self.ins().load(wvec, w_addr, 0);
        let slot = self.b.create_stack_slot(8, 3);
        self.ins().stack_store(w, slot, 0);
        let word = self.ins().stack_load(Type::I64, slot, 0);
        let Some(small_ty) = int_for_bytes(bytes) else {
            return w_addr;
        };
        let reduced = self.ins().ireduce(small_ty, word);
        let out_slot = self
            .b
            .create_stack_slot(bytes as u32, bytes.trailing_zeros() as u8);
        self.ins().stack_store(reduced, out_slot, 0);
        self.ins().stack_addr(Type::I64, out_slot, 0)
    }

    fn fill_upper_lanes(
        &mut self,
        wvec: Type,
        keep: usize,
        vec: Value,
        fill_scalar: Value,
    ) -> Value {
        let wcount = wvec.vec_shape().map_or(1, |(_, c)| usize::from(c));
        let fill_vec = self.ins().splat(wvec, fill_scalar);
        let mask = std::array::from_fn(|i| {
            if i >= wcount {
                0
            } else if i < keep {
                i as u8
            } else {
                wcount as u8
            }
        });
        self.ins().shuffle(vec, fill_vec, mask)
    }

    fn lane_identity_one(&mut self, num: Num) -> Result<Value> {
        match num {
            Num::Float(ty) => self.float_const(ty, 1.0),
            Num::Int { ty, .. } => Ok(self.ins().iconst(ty, 1)),
            Num::Bool => Ok(self.ins().iconst(Type::I8, 1)),
        }
    }
}
