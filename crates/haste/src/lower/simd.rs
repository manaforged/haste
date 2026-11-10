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
        let out = substs.get(1).copied();
        let value = match (op, out) {
            (Simd::Splat, _) => self.splat(vector, first)?,
            (Simd::Bitmask, Some(out)) => self.bitmask(vector, first, out)?,
            (Simd::Cast, Some(out)) => self.lane_cast((vector, out), first)?,
            (Simd::Reduce { all }, _) => self.reduce(vector, first, all)?,
            _ => self.simd_lanes(op, substs, args)?,
        };
        Ok(Some(value))
    }

    fn simd_lanes(&mut self, op: Simd, substs: &[Ty], args: &[Value]) -> Result<Value> {
        let vector = substs[0];
        Ok(match (op, args) {
            (Simd::Lanewise(bin), &[x, y]) => self.lanewise(bin, (vector, substs.get(1).copied().unwrap_or(vector)), (x, y))?,
            (Simd::Shuffle, &[x, y, idx]) => self.shuffle(substs, (x, y), idx)?,
            (Simd::Extract, &[x, at]) => {
                let (lanes, lane) = self.lane_address(vector, x, at)?;
                self.ins().load(lanes.machine, lane, 0)
            }
            (Simd::Insert, &[x, at, value]) => self.insert(vector, x, (at, value))?,
            _ => bail!("SIMD intrinsic {op:?} takes other arguments"),
        })
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

    fn lane_cast(&mut self, (input, output): (Ty, Ty), x: Value) -> Result<Value> {
        let (from, to) = (self.lanes(input)?, self.lanes(output)?);
        if from.count != to.count {
            bail!("`simd_cast` between vectors of {} and {} lanes", from.count, to.count);
        }
        let (slot, address) = self.vector_slot(output)?;
        for lane in 0..from.count {
            let value = self.ins().load(from.machine, x, lane * from.size);
            let value = self.convert(value, from.num, to.num)?;
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

    /// The vector whose lane `i` is lane `idx[i]` of `x` followed by `y`, as `simd_shuffle` at `[T, U, V]` builds it.
    fn shuffle(&mut self, substs: &[Ty], (x, y): (Value, Value), idx: Value) -> Result<Value> {
        let &[input, indices, output] = substs else { bail!("`simd_shuffle` takes three types") };
        let (from, to) = (self.lanes(input)?, self.lanes(output)?);
        let (both, joined) = self.pair_slot(&from)?;
        for lane in 0..from.count {
            let a = self.ins().load(from.machine, x, lane * from.size);
            let b = self.ins().load(from.machine, y, lane * from.size);
            self.ins().stack_store(a, both, lane * from.size);
            self.ins().stack_store(b, both, (from.count + lane) * from.size);
        }
        let Some(held) = self.spill(Some(idx), indices)? else { bail!("`simd_shuffle` indices take no bytes") };
        let idx = self.address(held);
        let (slot, address) = self.vector_slot(output)?;
        for lane in 0..to.count {
            let at = self.ins().load(Type::I32, idx, lane * 4);
            let at = self.ins().uextend(Type::I64, at);
            let offset = self.ins().imul_imm(at, i64::from(from.size));
            let source = self.ins().iadd(joined, offset);
            let value = self.ins().load(to.machine, source, 0);
            self.ins().stack_store(value, slot, lane * to.size);
        }
        Ok(address)
    }

    fn lane_address(&mut self, vector: Ty, x: Value, at: Value) -> Result<(Lanes, Value)> {
        let lanes = self.lanes(vector)?;
        let at = self.ins().uextend(Type::I64, at);
        let offset = self.ins().imul_imm(at, i64::from(lanes.size));
        let lane = self.ins().iadd(x, offset);
        Ok((lanes, lane))
    }

    fn insert(&mut self, vector: Ty, x: Value, (at, value): (Value, Value)) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let (slot, address) = self.vector_slot(vector)?;
        for lane in 0..lanes.count {
            let held = self.ins().load(lanes.machine, x, lane * lanes.size);
            self.ins().stack_store(held, slot, lane * lanes.size);
        }
        let (_, lane) = self.lane_address(vector, address, at)?;
        self.ins().store(value, lane, 0);
        Ok(address)
    }

    fn pair_slot(&mut self, lanes: &Lanes) -> Result<(StackSlot, Value)> {
        let bytes = (2 * lanes.count * lanes.size) as u32;
        let slot = self.b.create_stack_slot(bytes, lanes.size.trailing_zeros() as u8);
        Ok((slot, self.ins().stack_addr(Type::I64, slot, 0)))
    }

    fn reduce(&mut self, vector: Ty, x: Value, all: bool) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let mut found = self.ins().iconst(Type::I8, i64::from(all));
        for lane in 0..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            let set = self.ins().icmp_imm(IntCC::NotEqual, value, 0);
            found = if all { self.ins().band(found, set) } else { self.ins().bor(found, set) };
        }
        Ok(found)
    }

    /// The `bytes` bytes whose byte `i` is byte `index[i] & 15` of the 16-byte lane of `table` that holds byte `i`, or zero when bit 7 of `index[i]` is set, as `pshufb` gives.
    pub(super) fn byte_permute(&mut self, args: &[Value], bytes: u8) -> Result<Value> {
        let &[table, index] = args else { bail!("`pshufb` takes two vectors") };
        let slot = self.b.create_stack_slot(u32::from(bytes), 5);
        let (low, zero) = (self.ins().iconst(Type::I8, 15), self.ins().iconst(Type::I8, 0));
        for lane in 0..i32::from(bytes) {
            let at = self.ins().load(Type::I8, index, lane);
            let pick = self.ins().band(at, low);
            let pick = self.ins().uextend(Type::I64, pick);
            let base = self.ins().iadd_imm(table, i64::from(lane & !15));
            let source = self.ins().iadd(base, pick);
            let byte = self.ins().load(Type::I8, source, 0);
            let cleared = self.ins().icmp_imm(IntCC::SignedLessThan, at, 0);
            let value = self.ins().select(cleared, zero, byte);
            self.ins().stack_store(value, slot, lane);
        }
        Ok(self.ins().stack_addr(Type::I64, slot, 0))
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
