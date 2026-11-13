use anyhow::{Result, bail};
use haste_codegen::{IntCC, Lane, ReduceOp, Type, Value};

use crate::body::BinOp;
use crate::sema::intrinsic::{LanePair, LaneUnary, Simd};
use crate::sema::ty::Ty;

use super::super::Lower;
use super::super::ops::Num;
use super::{Lanes, vec_type};

impl Lower<'_, '_> {
    /// Emits the SIMD intrinsic `op` that has no operator of its own, using `Type::Vec` ops where supported.
    pub(super) fn lane_intrinsic(&mut self, op: Simd, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let Some(&vector) = substs.first() else { bail!("a SIMD intrinsic with no vector") };
        let lanes = self.lanes(vector)?;
        if let Some(value) = self.vec_lane_intrinsic(op, &lanes, args)? {
            return Ok(Some(value));
        }
        if let Some(value) = self.wide_lane_intrinsic(op, &lanes, args)? {
            return Ok(Some(value));
        }
        if let Some(value) = self.sub8_lane_intrinsic(op, &lanes, args)? {
            return Ok(Some(value));
        }
        Ok(Some(match (op, args) {
            (Simd::Unary(unary), &[x]) => self.scalar_map(&lanes, &[x], |lower, num, vals| lower.lane_unary(unary, num, vals[0]))?,
            (Simd::Pair(pair), &[x, y]) => self.scalar_map(&lanes, &[x, y], |lower, num, vals| lower.lane_pair(pair, num, (vals[0], vals[1])))?,
            (Simd::Fma, &[x, y, z]) => self.scalar_map(&lanes, &[x, y, z], |lower, num, vals| lower.lane_fma(num, vals))?,
            (Simd::Sum { mul }, &[x, ref start @ ..]) => self.lane_sum(&lanes, x, (start.first().copied(), mul))?,
            (Simd::MaskedLoad, &[mask, pointer, fallback]) => self.masked(substs, mask, pointer, Some(fallback))?,
            (Simd::MaskedStore, &[mask, pointer, values]) => {
                self.masked_store(substs, mask, pointer, values)?;
                return Ok(None);
            }
            (Simd::Unary(_) | Simd::Pair(_) | Simd::Fma | Simd::Sum { .. } | Simd::MaskedLoad | Simd::MaskedStore, _) => bail!("SIMD intrinsic {op:?} takes other arguments"),
            _ => return Ok(None),
        }))
    }

    pub(super) fn vec_lane_intrinsic(&mut self, op: Simd, lanes: &Lanes, args: &[Value]) -> Result<Option<Value>> {
        let Some(vec_ty) = vec_type(lanes) else { return Ok(None) };
        match (op, args) {
            (Simd::Unary(unary), &[x]) => {
                let v = self.ins().load(vec_ty, x, 0);
                let out = match (unary, lanes.num) {
                    (LaneUnary::Neg, Num::Float(_)) => self.ins().fneg(v),
                    (LaneUnary::Neg, Num::Int { .. }) => self.ins().ineg(v),
                    (LaneUnary::Fabs, Num::Float(_)) => self.ins().fabs(v),
                    (LaneUnary::Floor, Num::Float(_)) => self.ins().floor(v),
                    (LaneUnary::Ceil, Num::Float(_)) => self.ins().ceil(v),
                    (LaneUnary::Ctpop, Num::Int { .. }) => self.ins().popcnt(v),
                    (LaneUnary::Ctlz, Num::Int { .. }) => self.ins().clz(v),
                    _ => bail!("SIMD {unary:?} on an unsupported lane"),
                };
                Ok(Some(self.store_vec(out, vec_ty.bytes())))
            }
            (Simd::Pair(pair @ (LanePair::Min | LanePair::Max)), &[x, y]) => {
                let (a, b) = (self.ins().load(vec_ty, x, 0), self.ins().load(vec_ty, y, 0));
                let max = pair == LanePair::Max;
                let out = match lanes.num {
                    Num::Float(_) => if max { self.ins().fmax(a, b) } else { self.ins().fmin(a, b) },
                    Num::Int { signed: true, .. } => if max { self.ins().smax(a, b) } else { self.ins().smin(a, b) },
                    Num::Int { signed: false, .. } => if max { self.ins().umax(a, b) } else { self.ins().umin(a, b) },
                    Num::Bool => bail!("SIMD {pair:?} on boolean lane"),
                };
                Ok(Some(self.store_vec(out, vec_ty.bytes())))
            }
            (Simd::Pair(pair @ (LanePair::SaturatingAdd | LanePair::SaturatingSub)), &[x, y]) => {
                let Num::Int { signed, .. } = lanes.num else { bail!("SIMD {pair:?} on non-integer lane") };
                let (a, b) = (self.ins().load(vec_ty, x, 0), self.ins().load(vec_ty, y, 0));
                let out = match (pair, signed) {
                    (LanePair::SaturatingAdd, true) => self.ins().sadd_sat(a, b),
                    (LanePair::SaturatingAdd, false) => self.ins().uadd_sat(a, b),
                    (LanePair::SaturatingSub, true) => self.ins().ssub_sat(a, b),
                    _ => self.ins().usub_sat(a, b),
                };
                Ok(Some(self.store_vec(out, vec_ty.bytes())))
            }
            (Simd::Fma, &[a, b, c]) if matches!(vec_ty, Type::Vec { lane: Lane::F32, count: 2 | 4 } | Type::Vec { lane: Lane::F64, count: 2 }) => {
                let (va, vb, vc) = (self.ins().load(vec_ty, a, 0), self.ins().load(vec_ty, b, 0), self.ins().load(vec_ty, c, 0));
                let out = self.ins().fma(va, vb, vc);
                Ok(Some(self.store_vec(out, vec_ty.bytes())))
            }
            (Simd::Sum { mul }, &[x, ref start @ ..]) => {
                let v = self.ins().load(vec_ty, x, 0);
                self.vec_sum(lanes, vec_ty, v, (start.first().copied(), mul)).map(Some)
            }
            _ => Ok(None),
        }
    }

    pub(super) fn vec_sum(&mut self, lanes: &Lanes, vec_ty: Type, v: Value, (start, mul): (Option<Value>, bool)) -> Result<Value> {
        let Some((lane, count)) = vec_ty.vec_shape() else { bail!("vec_sum requires a vector type") };
        let op = if mul { BinOp::Mul } else { BinOp::Add };
        if !mul && lane.is_int() {
            let sum = self.ins().reduce(ReduceOp::Add, v);
            return match start {
                Some(init) => self.lane_op(op, lanes.num, (init, sum), lanes.machine),
                None => Ok(sum),
            };
        }
        if lane.is_float()
            && let Some(init) = start
        {
            return self.ordered_float_sum(lanes, count, v, init, mul);
        }
        let total = self.tree_reduce_vec(mul, lane.is_float(), count, v);
        match start {
            Some(init) => self.lane_op(op, lanes.num, (init, total), lanes.machine),
            None => Ok(total),
        }
    }

    fn ordered_float_sum(&mut self, lanes: &Lanes, count: u8, vec: Value, init: Value, mul: bool) -> Result<Value> {
        let op = if mul { BinOp::Mul } else { BinOp::Add };
        let step = |lower: &mut Self, acc: Value, lane_idx: u8| {
            let val = lower.ins().extract(vec, lane_idx);
            lower.lane_op(op, lanes.num, (acc, val), lanes.machine)
        };
        match count {
            1 => step(self, init, 0),
            2 => {
                let a = step(self, init, 0)?;
                step(self, a, 1)
            }
            4 => {
                let a = step(self, init, 0)?;
                let b = step(self, a, 1)?;
                let c = step(self, b, 2)?;
                step(self, c, 3)
            }
            _ => bail!("unsupported float vector lane count {count}"),
        }
    }

    pub(super) fn tree_reduce_vec(&mut self, mul: bool, float: bool, count: u8, mut vec: Value) -> Value {
        let mut step = count / 2;
        while step > 0 {
            let s = usize::from(step);
            let mask = std::array::from_fn(|i| if i < s { (i + s) as u8 } else { 0 });
            let shifted = self.ins().shuffle(vec, vec, mask);
            vec = match (mul, float) {
                (true, true) => self.ins().fmul(vec, shifted),
                (false, true) => self.ins().fadd(vec, shifted),
                (true, false) => self.ins().imul(vec, shifted),
                (false, false) => self.ins().iadd(vec, shifted),
            };
            step /= 2;
        }
        self.ins().extract(vec, 0)
    }

    fn scalar_map(&mut self, lanes: &Lanes, inputs: &[Value], mut each: impl FnMut(&mut Self, Num, &[Value]) -> Result<Value>) -> Result<Value> {
        let (_, address) = self.lanes_slot(*lanes);
        let size = i64::from(lanes.size);
        let num = lanes.num;
        let machine = lanes.machine;
        let ptrs: Vec<Value> = inputs.to_vec();
        self.emit_lane_loop(0, lanes.count, &[], |lower, idx, _| {
            let offset = lower.ins().imul_imm(idx, size);
            let values: Vec<Value> = ptrs
                .iter()
                .map(|&input| {
                    let at = lower.ins().iadd(input, offset);
                    lower.ins().load(machine, at, 0)
                })
                .collect();
            let value = each(lower, num, &values)?;
            let dst = lower.ins().iadd(address, offset);
            lower.ins().store(value, dst, 0);
            Ok(Vec::new())
        })?;
        Ok(address)
    }

    fn lane_unary(&mut self, op: LaneUnary, num: Num, x: Value) -> Result<Value> {
        Ok(match (op, num) {
            (LaneUnary::Neg, Num::Float(_)) => self.ins().fneg(x),
            (LaneUnary::Neg, _) => self.ins().ineg(x),
            (LaneUnary::Fabs, _) => self.ins().fabs(x),
            (LaneUnary::Floor, _) => self.ins().floor(x),
            (LaneUnary::Ceil, _) => self.ins().ceil(x),
            (LaneUnary::Ctpop | LaneUnary::Ctlz, Num::Int { ty, .. }) => self.lane_count(op, ty, x),
            _ => bail!("SIMD {op:?} on a lane that is not an integer"),
        })
    }

    fn lane_count(&mut self, op: LaneUnary, ty: Type, x: Value) -> Value {
        let wide = if ty.bits() < 32 { Type::I32 } else { ty };
        let value = self.resize(x, (ty, false), wide);
        let counted = match op {
            LaneUnary::Ctlz => {
                let zeros = self.ins().clz(value);
                self.ins().iadd_imm(zeros, -i64::from(wide.bits() - ty.bits()))
            }
            _ => self.ins().popcnt(value),
        };
        self.resize(counted, (wide, false), ty)
    }

    fn lane_pair(&mut self, op: LanePair, num: Num, (x, y): (Value, Value)) -> Result<Value> {
        match (op, num) {
            (LanePair::SaturatingAdd, Num::Int { ty, signed }) => {
                if ty != Type::I128 {
                    Ok(if signed { self.ins().sadd_sat(x, y) } else { self.ins().uadd_sat(x, y) })
                } else {
                    self.saturating(BinOp::Add, (ty, signed), x, y)
                }
            }
            (LanePair::SaturatingSub, Num::Int { ty, signed }) => {
                if ty != Type::I128 {
                    Ok(if signed { self.ins().ssub_sat(x, y) } else { self.ins().usub_sat(x, y) })
                } else {
                    self.saturating(BinOp::Sub, (ty, signed), x, y)
                }
            }
            (LanePair::Min, Num::Float(_)) => Ok(self.ins().fmin(x, y)),
            (LanePair::Max, Num::Float(_)) => Ok(self.ins().fmax(x, y)),
            (LanePair::Min | LanePair::Max, Num::Int { .. }) => {
                let wins = self.apply(if op == LanePair::Min { BinOp::Lt } else { BinOp::Gt }, num, (x, y), num)?;
                Ok(self.ins().select(wins, x, y))
            }
            _ => bail!("SIMD {op:?} on a lane it does not take"),
        }
    }

    fn lane_fma(&mut self, num: Num, lanes: &[Value]) -> Result<Value> {
        let Num::Float(_) = num else { bail!("`simd_fma` on a lane that is not a float") };
        let &[a, b, c] = lanes else { bail!("`simd_fma` takes three arguments") };
        Ok(self.ins().fma(a, b, c))
    }

    fn lane_sum(&mut self, lanes: &Lanes, x: Value, (start, mul): (Option<Value>, bool)) -> Result<Value> {
        let op = if mul { BinOp::Mul } else { BinOp::Add };
        let initial = match start {
            Some(start) => start,
            None => self.ins().load(lanes.machine, x, 0),
        };
        let size = i64::from(lanes.size);
        let num = lanes.num;
        let machine = lanes.machine;
        let out = self.emit_lane_loop(i64::from(start.is_none()), lanes.count, &[initial], |lower, idx, acc| {
            let offset = lower.ins().imul_imm(idx, size);
            let at = lower.ins().iadd(x, offset);
            let value = lower.ins().load(machine, at, 0);
            let next = lower.lane_op(op, num, (acc[0], value), machine)?;
            Ok(vec![next])
        })?;
        Ok(out[0])
    }

    fn masked(&mut self, substs: &[Ty], mask: Value, pointer: Value, fallback: Option<Value>) -> Result<Value> {
        let (&[mask_ty, _, vector, ..], Some(fallback)) = (substs, fallback) else { bail!("`simd_masked_load` takes three types") };
        let (from, to) = (self.lanes(mask_ty)?, self.lanes(vector)?);
        let (_, address) = self.vector_slot(vector)?;
        let (from_size, to_size) = (i64::from(from.size), i64::from(to.size));
        let (from_m, to_m) = (from.machine, to.machine);
        self.emit_lane_loop(0, to.count, &[], |lower, idx, _| {
            let mask_off = lower.ins().imul_imm(idx, from_size);
            let mask_at = lower.ins().iadd(mask, mask_off);
            let flag = lower.ins().load(from_m, mask_at, 0);
            let set = lower.ins().icmp_imm(IntCC::NotEqual, flag, 0);
            let elem_off = lower.ins().imul_imm(idx, to_size);
            let ptr_at = lower.ins().iadd(pointer, elem_off);
            let fb_at = lower.ins().iadd(fallback, elem_off);
            let at = lower.ins().select(set, ptr_at, fb_at);
            let value = lower.ins().load(to_m, at, 0);
            let dst = lower.ins().iadd(address, elem_off);
            lower.ins().store(value, dst, 0);
            Ok(Vec::new())
        })?;
        Ok(address)
    }

    fn masked_store(&mut self, substs: &[Ty], mask: Value, pointer: Value, values: Value) -> Result<()> {
        let &[mask_ty, _, vector, ..] = substs else { bail!("`simd_masked_store` takes three types") };
        let (from, to) = (self.lanes(mask_ty)?, self.lanes(vector)?);
        let (_, spare) = self.vector_slot(vector)?;
        let (from_size, to_size) = (i64::from(from.size), i64::from(to.size));
        let (from_m, to_m) = (from.machine, to.machine);
        self.emit_lane_loop(0, to.count, &[], |lower, idx, _| {
            let mask_off = lower.ins().imul_imm(idx, from_size);
            let mask_at = lower.ins().iadd(mask, mask_off);
            let flag = lower.ins().load(from_m, mask_at, 0);
            let set = lower.ins().icmp_imm(IntCC::NotEqual, flag, 0);
            let elem_off = lower.ins().imul_imm(idx, to_size);
            let ptr_at = lower.ins().iadd(pointer, elem_off);
            let sp_at = lower.ins().iadd(spare, elem_off);
            let at = lower.ins().select(set, ptr_at, sp_at);
            let val_at = lower.ins().iadd(values, elem_off);
            let value = lower.ins().load(to_m, val_at, 0);
            lower.ins().store(value, at, 0);
            Ok(Vec::new())
        })?;
        Ok(())
    }
}
