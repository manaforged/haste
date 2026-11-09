use anyhow::{Result, bail};
use haste_codegen::{IntCC, StackSlot, Type, Value};

use crate::body::BinOp;
use crate::sema::intrinsic::Simd;
use crate::sema::ty::{Kind, Ty};

use super::ops::{Num, compare};
use super::{Lower, machine, repr};

struct Lanes {
    num: Num,
    machine: Type,
    size: i32,
    count: i32,
}

impl Lower<'_, '_> {
    /// Emits the SIMD intrinsic `op` one lane at a time on vectors held in memory, as a scalar backend does.
    pub(super) fn simd(&mut self, op: Simd, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let (Some(&vector), Some(&first)) = (substs.first(), args.first()) else { bail!("a SIMD intrinsic with no vector") };
        let value = match (op, args, substs.get(1).copied()) {
            (Simd::Splat, _, _) => self.splat(vector, first)?,
            (Simd::Bitmask, _, Some(out)) => self.bitmask(vector, first, out)?,
            (Simd::Lanewise(bin), &[x, y], out) => self.lanewise(bin, (vector, out.unwrap_or(vector)), (x, y))?,
            _ => bail!("SIMD intrinsic {op:?} takes other arguments"),
        };
        Ok(Some(value))
    }

    fn lanes(&mut self, ty: Ty) -> Result<Lanes> {
        let Kind::Adt(item, args) = self.cx.types.kind(ty).clone() else { bail!("a SIMD intrinsic on a type that is not a vector") };
        let fields = crate::sema::infer::field_types(self.cx, item, &args)?;
        let Some(&array) = fields.first().and_then(|variant| variant.first()) else { bail!("a SIMD vector with no field") };
        let Kind::Array(elem, len) = *self.cx.types.kind(array) else { bail!("a SIMD vector whose field is not an array") };
        let Some(lane) = machine(self.cx, elem)? else { bail!("a SIMD lane with no machine type") };
        Ok(Lanes { num: self.num(elem)?, machine: lane, size: lane.bytes() as i32, count: self.len(len)? as i32 })
    }

    fn vector_slot(&mut self, ty: Ty) -> Result<(StackSlot, Value)> {
        let held = repr(self.cx, ty)?;
        let Some(slot) = self.slot(held) else { bail!("a SIMD vector that takes no bytes") };
        Ok((slot, self.ins().stack_addr(Type::I64, slot, 0)))
    }

    fn splat(&mut self, vector: Ty, value: Value) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let (slot, address) = self.vector_slot(vector)?;
        for lane in 0..lanes.count {
            self.ins().stack_store(value, slot, lane * lanes.size);
        }
        Ok(address)
    }

    fn lanewise(&mut self, op: BinOp, (input, output): (Ty, Ty), (x, y): (Value, Value)) -> Result<Value> {
        let (from, to) = (self.lanes(input)?, self.lanes(output)?);
        let (slot, address) = self.vector_slot(output)?;
        for lane in 0..from.count {
            let a = self.ins().load(from.machine, x, lane * from.size);
            let b = self.ins().load(from.machine, y, lane * from.size);
            let value = self.lane_op(op, from.num, (a, b), to.machine)?;
            self.ins().stack_store(value, slot, lane * to.size);
        }
        Ok(address)
    }

    fn lane_op(&mut self, op: BinOp, num: Num, (x, y): (Value, Value), out: Type) -> Result<Value> {
        if compare(op).is_some() {
            let flag = self.apply(op, num, (x, y), num)?;
            let wide = self.resize(flag, (Type::I8, false), out);
            return Ok(self.ins().ineg(wide));
        }
        match (op, num) {
            (BinOp::Add | BinOp::Sub | BinOp::Mul, Num::Int { ty, signed }) => Ok(self.overflowing(op, (ty, signed), x, y)?.0),
            _ => self.apply(op, num, (x, y), num),
        }
    }

    fn bitmask(&mut self, vector: Ty, x: Value, out: Ty) -> Result<Value> {
        let Some(target) = machine(self.cx, out)? else { bail!("a SIMD bitmask into an array is not supported yet") };
        let lanes = self.lanes(vector)?;
        let mut mask = self.ins().iconst(Type::I64, 0);
        for lane in 0..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            let set = self.ins().icmp_imm(IntCC::NotEqual, value, 0);
            let bit = self.ins().uextend(Type::I64, set);
            let shift = self.ins().iconst(Type::I64, i64::from(lane));
            let bit = self.ins().ishl(bit, shift);
            mask = self.ins().bor(mask, bit);
        }
        Ok(self.resize(mask, (Type::I64, false), target))
    }
}
