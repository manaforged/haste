use anyhow::{Result, bail};
use haste_codegen::{Arch, IntCC, Lane, ReduceOp, Target, Type, Value};

use crate::body::BinOp;
use crate::host::{HostSymbol, MathSymbol};
use crate::sema::intrinsic::{LanePair, LaneUnary, Simd};
use crate::sema::ty::Ty;

use super::super::Lower;
use super::super::ops::Num;
use super::{Lanes, vec_type};

impl Lower<'_, '_> {
    /// Emits the SIMD intrinsic `op` that has no operator of its own, using `Type::Vec` ops where supported.
    pub(super) fn lane_intrinsic(&mut self, op: Simd, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let Some(&vector) = substs.first() else { bail!("a SIMD intrinsic with no vector") };
        if let Some(value) = self.vec_lane_intrinsic(op, vector, args)? {
            return Ok(Some(value));
        }
        Ok(Some(match (op, args) {
            (Simd::Unary(unary), &[x]) => self.scalar_map(vector, &[x], |lower, num, lanes| lower.lane_unary(unary, num, lanes[0]))?,
            (Simd::Pair(pair), &[x, y]) => self.scalar_map(vector, &[x, y], |lower, num, lanes| lower.lane_pair(pair, num, (lanes[0], lanes[1])))?,
            (Simd::Fma, &[x, y, z]) => self.scalar_map(vector, &[x, y, z], |lower, num, lanes| lower.lane_fma(num, lanes))?,
            (Simd::Sum { mul }, &[x, ref start @ ..]) => self.lane_sum(vector, x, (start.first().copied(), mul))?,
            (Simd::MaskedLoad, &[mask, pointer, fallback]) => self.masked(substs, mask, pointer, Some(fallback))?,
            (Simd::MaskedStore, &[mask, pointer, values]) => {
                self.masked_store(substs, mask, pointer, values)?;
                return Ok(None);
            }
            (Simd::Unary(_) | Simd::Pair(_) | Simd::Fma | Simd::Sum { .. } | Simd::MaskedLoad | Simd::MaskedStore, _) => bail!("SIMD intrinsic {op:?} takes other arguments"),
            _ => return Ok(None),
        }))
    }

    fn vec_lane_intrinsic(&mut self, op: Simd, vector: Ty, args: &[Value]) -> Result<Option<Value>> {
        let lanes = self.lanes(vector)?;
        let Some(vec_ty) = vec_type(&lanes) else { return Ok(None) };
        let arm64 = Target::host().arch == Arch::Arm64;
        match (op, args) {
            (Simd::Unary(unary), &[x]) => {
                let v = self.ins().load(vec_ty, x, 0);
                let out = match (unary, lanes.num) {
                    (LaneUnary::Neg, Num::Float(_)) => self.ins().fneg(v),
                    (LaneUnary::Neg, Num::Int { .. }) => self.ins().ineg(v),
                    (LaneUnary::Fabs, Num::Float(_)) => self.ins().fabs(v),
                    (LaneUnary::Floor, Num::Float(_)) => self.ins().floor(v),
                    (LaneUnary::Ceil, Num::Float(_)) => self.ins().ceil(v),
                    (LaneUnary::Ctpop, Num::Int { .. }) if arm64 => self.ins().popcnt(v),
                    (LaneUnary::Ctlz, Num::Int { .. }) if arm64 => self.ins().clz(v),
                    (LaneUnary::Ctpop | LaneUnary::Ctlz, Num::Int { .. }) => return Ok(None),
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
            (Simd::Sum { mul: false }, &[x, ref start @ ..]) if vec_ty.lane().is_some_and(Lane::is_int) => {
                let v = self.ins().load(vec_ty, x, 0);
                let sum = self.ins().reduce(ReduceOp::Add, v);
                let out = match start.first().copied() {
                    Some(init) => self.lane_op(BinOp::Add, lanes.num, (init, sum), lanes.machine)?,
                    None => sum,
                };
                Ok(Some(out))
            }
            _ => Ok(None),
        }
    }

    fn scalar_map(&mut self, vector: Ty, inputs: &[Value], mut each: impl FnMut(&mut Self, Num, &[Value]) -> Result<Value>) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let (slot, address) = self.vector_slot(vector)?;
        for lane in 0..lanes.count {
            let values: Vec<Value> = inputs.iter().map(|&input| self.ins().load(lanes.machine, input, lane * lanes.size)).collect();
            let value = each(self, lanes.num, &values)?;
            self.ins().stack_store(value, slot, lane * lanes.size);
        }
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
            (LanePair::SaturatingAdd, Num::Int { ty, signed }) => self.saturating(BinOp::Add, (ty, signed), x, y),
            (LanePair::SaturatingSub, Num::Int { ty, signed }) => self.saturating(BinOp::Sub, (ty, signed), x, y),
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
        let Num::Float(ty) = num else { bail!("`simd_fma` on a lane that is not a float") };
        let symbol = if ty == Type::F32 { MathSymbol::Fmaf32 } else { MathSymbol::Fmaf64 };
        self.call_host(HostSymbol::Math(symbol), lanes, Some(ty)).ok_or_else(|| anyhow::anyhow!("a float host call with no result"))
    }

    fn lane_sum(&mut self, vector: Ty, x: Value, (start, mul): (Option<Value>, bool)) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let op = if mul { BinOp::Mul } else { BinOp::Add };
        let mut total = match start {
            Some(start) => start,
            None => self.ins().load(lanes.machine, x, 0),
        };
        for lane in i32::from(start.is_none())..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            total = self.lane_op(op, lanes.num, (total, value), lanes.machine)?;
        }
        Ok(total)
    }

    /// The address `offset` bytes into `pointer` where lane `lane` of `mask` is set, and as far into `otherwise` where not, so a masked-off lane never touches `pointer`.
    fn masked_lane(&mut self, (mask, lanes, lane): (Value, &Lanes, i32), (pointer, otherwise): (Value, Value), offset: i32) -> Value {
        let set = self.ins().load(lanes.machine, mask, lane * lanes.size);
        let set = self.ins().icmp_imm(IntCC::NotEqual, set, 0);
        let (at, spare) = (self.ins().iadd_imm(pointer, i64::from(offset)), self.ins().iadd_imm(otherwise, i64::from(offset)));
        self.ins().select(set, at, spare)
    }

    fn masked(&mut self, substs: &[Ty], mask: Value, pointer: Value, fallback: Option<Value>) -> Result<Value> {
        let (&[mask_ty, _, vector, ..], Some(fallback)) = (substs, fallback) else { bail!("`simd_masked_load` takes three types") };
        let (from, to) = (self.lanes(mask_ty)?, self.lanes(vector)?);
        let (slot, address) = self.vector_slot(vector)?;
        for lane in 0..to.count {
            let at = self.masked_lane((mask, &from, lane), (pointer, fallback), lane * to.size);
            let value = self.ins().load(to.machine, at, 0);
            self.ins().stack_store(value, slot, lane * to.size);
        }
        Ok(address)
    }

    fn masked_store(&mut self, substs: &[Ty], mask: Value, pointer: Value, values: Value) -> Result<()> {
        let &[mask_ty, _, vector, ..] = substs else { bail!("`simd_masked_store` takes three types") };
        let (from, to) = (self.lanes(mask_ty)?, self.lanes(vector)?);
        let (_, spare) = self.vector_slot(vector)?;
        for lane in 0..to.count {
            let at = self.masked_lane((mask, &from, lane), (pointer, spare), lane * to.size);
            let value = self.ins().load(to.machine, values, lane * to.size);
            self.ins().store(value, at, 0);
        }
        Ok(())
    }
}
