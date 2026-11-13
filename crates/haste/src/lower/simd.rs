use anyhow::{Result, bail};
use haste_codegen::{Arch, ByteCC, IntCC, StackSlot, Target, Type, Value, VectorOp};

use crate::body::BinOp;
use crate::host::{HostSymbol, X86Symbol};
use crate::sema::intrinsic::Simd;
use crate::sema::ty::{Kind, Ty};

use super::ops::{Num, compare};
use super::{Lower, machine, repr};

mod lanes;

struct Lanes {
    num: Num,
    machine: Type,
    size: i32,
    count: i32,
}

/// The bytes of a vector the code generator holds in one register.
const PACKED_BYTES: i32 = 16;

impl Lanes {
    /// Whether the vector is integer lanes that fill one 128-bit register.
    fn is_packed(&self) -> bool {
        self.size * self.count == PACKED_BYTES && matches!(self.num, Num::Int { .. })
    }
}

impl Lower<'_, '_> {
    /// Emits the SIMD intrinsic `op` one lane at a time on vectors held in memory, as a scalar backend does.
    pub(super) fn simd(&mut self, op: Simd, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let (Some(&vector), Some(&first)) = (substs.first(), args.first()) else { bail!("a SIMD intrinsic with no vector") };
        let out = substs.get(1).copied();
        if matches!(Target::host().arch, Arch::X64 | Arch::Arm64)
            && let Some(value) = self.packed(op, (vector, out.unwrap_or(vector)), args)?
        {
            return Ok(Some(value));
        }
        if let Some(value) = self.packed_swar64(op, (vector, out.unwrap_or(vector)), args)? {
            return Ok(Some(value));
        }
        let value = match (op, out) {
            (Simd::Splat, _) => self.splat(vector, first)?,
            (Simd::Bitmask, Some(out)) => self.bitmask(vector, first, out)?,
            (Simd::Cast, Some(out)) => self.lane_cast((vector, out), first)?,
            (Simd::Reduce { all }, _) => self.reduce(vector, first, all)?,
            (Simd::Extreme { max }, _) => self.extreme(vector, first, max)?,
            (Simd::Fold(op), _) => self.fold(vector, first, op)?,
            (Simd::Sqrt, _) => self.lane_sqrt(vector, first)?,
            (Simd::Unary(_) | Simd::Pair(_) | Simd::Fma | Simd::Sum { .. } | Simd::MaskedLoad | Simd::MaskedStore, _) => return self.lane_intrinsic(op, substs, args),
            _ => self.simd_lanes(op, substs, args)?,
        };
        Ok(Some(value))
    }

    /// Emits `op` as one 128-bit vector operation when it is one the code generator has: a bitwise operation on a register of 8 or 16 integer bytes, or a splat, equality, signed order or bitmask.
    fn packed(&mut self, op: Simd, (input, output): (Ty, Ty), args: &[Value]) -> Result<Option<Value>> {
        let lanes = self.lanes(input)?;
        let is_16 = lanes.is_packed();
        let is_8 = lanes.size * lanes.count == 8 && matches!(lanes.num, Num::Int { .. });
        if !is_16 && !is_8 {
            return Ok(None);
        }
        let is_bytes = lanes.size == 1 && (lanes.count == PACKED_BYTES || lanes.count == 8);
        let signed = matches!(lanes.num, Num::Int { signed: true, .. });
        let load_vec = |lower: &mut Self, addr: Value| {
            if is_8 {
                let f = lower.ins().load(Type::F64, addr, 0);
                lower.ins().widen_vec64(f)
            } else {
                lower.ins().load(Type::V128, addr, 0)
            }
        };
        let vector = match (op, args) {
            (Simd::Splat, &[x]) if is_bytes => self.ins().splat(x),
            (Simd::Lanewise(bin @ (BinOp::Add | BinOp::Sub)), &[x, y]) if is_16 && input == output && lanes.size <= 8 => {
                let (a, b) = (load_vec(self, x), load_vec(self, y));
                self.ins().vector(lane_arithmetic(bin, lanes.size), a, b)
            }
            (Simd::Bitmask, &[x]) if is_16 && is_bytes => {
                let Some(target) = machine(self.cx, output)? else { return Ok(None) };
                let loaded = load_vec(self, x);
                let mask = self.ins().movemask(loaded);
                return Ok(Some(self.resize(mask, (Type::I32, false), target)));
            }
            (Simd::Lanewise(bin @ (BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor)), &[x, y]) if input == output => {
                let (a, b) = (load_vec(self, x), load_vec(self, y));
                match bin {
                    BinOp::BitAnd => self.ins().band(a, b),
                    BinOp::BitOr => self.ins().bor(a, b),
                    _ => self.ins().bxor(a, b),
                }
            }
            (Simd::Lanewise(cmp @ (BinOp::Eq | BinOp::Ne | BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le)), &[x, y])
                if is_bytes
                    && (matches!(cmp, BinOp::Eq | BinOp::Ne) || signed)
                    && {
                        let out = self.lanes(output)?;
                        out.size == 1 && out.count == lanes.count && matches!(out.num, Num::Int { .. })
                    } =>
            {
                let (a, b) = (load_vec(self, x), load_vec(self, y));
                let invert = |v: Value, lower: &mut Self| {
                    let ones = lower.ins().byte_cmp(ByteCC::Equal, a, a);
                    lower.ins().bxor(v, ones)
                };
                match cmp {
                    BinOp::Eq => self.ins().byte_cmp(ByteCC::Equal, a, b),
                    BinOp::Ne => {
                        let eq = self.ins().byte_cmp(ByteCC::Equal, a, b);
                        invert(eq, self)
                    }
                    BinOp::Gt => self.ins().byte_cmp(ByteCC::SignedGreaterThan, a, b),
                    BinOp::Lt => self.ins().byte_cmp(ByteCC::SignedGreaterThan, b, a),
                    BinOp::Ge => {
                        let lt = self.ins().byte_cmp(ByteCC::SignedGreaterThan, b, a);
                        invert(lt, self)
                    }
                    _ => {
                        let gt = self.ins().byte_cmp(ByteCC::SignedGreaterThan, a, b);
                        invert(gt, self)
                    }
                }
            }
            _ => return Ok(None),
        };
        let stored = if is_8 { self.ins().narrow_vec64(vector) } else { vector };
        let (slot, address) = self.vector_slot(if matches!(op, Simd::Splat) { input } else { output })?;
        self.ins().stack_store(stored, slot, 0);
        Ok(Some(address))
    }

    fn packed_swar64(&mut self, op: Simd, (input, output): (Ty, Ty), args: &[Value]) -> Result<Option<Value>> {
        let lanes = self.lanes(input)?;
        if lanes.size * lanes.count != 8 || !matches!(lanes.num, Num::Int { .. }) {
            return Ok(None);
        }
        let signed = matches!(lanes.num, Num::Int { signed: true, .. });
        let is_u8x8 = lanes.size == 1 && lanes.count == 8;
        let word = match (op, args) {
            (Simd::Splat, &[x]) if is_u8x8 => {
                let wide = self.resize(x, (lanes.machine, false), Type::I64);
                let ff = self.ins().iconst(Type::I64, 0xFF);
                let byte = self.ins().band(wide, ff);
                self.ins().imul_imm(byte, 0x0101_0101_0101_0101_u64 as i64)
            }
            (Simd::Extract, &[x, _]) if lanes.size == 8 && lanes.count == 1 => {
                return Ok(Some(self.ins().load(lanes.machine, x, 0)));
            }
            (Simd::Lanewise(bin @ (BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor)), &[x, y]) if input == output => {
                let (a, b) = (self.ins().load(Type::I64, x, 0), self.ins().load(Type::I64, y, 0));
                match bin {
                    BinOp::BitAnd => self.ins().band(a, b),
                    BinOp::BitOr => self.ins().bor(a, b),
                    _ => self.ins().bxor(a, b),
                }
            }
            (Simd::Lanewise(cmp @ (BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)), &[x, y])
                if is_u8x8 && self.lanes(output)?.size == 1 && self.lanes(output)?.count == 8 =>
            {
                let (a, b) = (self.ins().load(Type::I64, x, 0), self.ins().load(Type::I64, y, 0));
                let m7f = self.ins().iconst(Type::I64, 0x7F7F_7F7F_7F7F_7F7F_u64 as i64);
                let m80 = self.ins().iconst(Type::I64, 0x8080_8080_8080_8080_u64 as i64);
                let seven = self.ins().iconst(Type::I64, 7);
                let bits = match cmp {
                    BinOp::Eq | BinOp::Ne => {
                        let d = self.ins().bxor(a, b);
                        let masked = self.ins().band(d, m7f);
                        let low = self.ins().iadd(masked, m7f);
                        let low_or_d = self.ins().bor(low, d);
                        let nz_or_7f = self.ins().bor(low_or_d, m7f);
                        let msb = if cmp == BinOp::Eq {
                            self.ins().bnot(nz_or_7f)
                        } else {
                            self.ins().band(nz_or_7f, m80)
                        };
                        self.ins().ushr(msb, seven)
                    }
                    _ => {
                        let (lhs, rhs, invert) = match cmp {
                            BinOp::Gt => (b, a, false),
                            BinOp::Ge => (a, b, true),
                            BinOp::Le => (b, a, true),
                            _ => (a, b, false),
                        };
                        let (ua, ub) = if signed {
                            (self.ins().bxor(lhs, m80), self.ins().bxor(rhs, m80))
                        } else {
                            (lhs, rhs)
                        };
                        let ua_hi = self.ins().bor(ua, m80);
                        let ub_lo = self.ins().band(ub, m7f);
                        let diff = self.ins().isub(ua_hi, ub_lo);
                        let not_ua = self.ins().bnot(ua);
                        let term1 = self.ins().band(not_ua, ub);
                        let xor_ab = self.ins().bxor(ua, ub);
                        let same_hi = self.ins().bnot(xor_ab);
                        let not_diff = self.ins().bnot(diff);
                        let term2 = self.ins().band(same_hi, not_diff);
                        let lt_hi = self.ins().bor(term1, term2);
                        let raw = if invert { self.ins().bnot(lt_hi) } else { lt_hi };
                        let msb = self.ins().band(raw, m80);
                        self.ins().ushr(msb, seven)
                    }
                };
                self.ins().imul_imm(bits, 0xFF)
            }
            _ => return Ok(None),
        };
        let (slot, address) = self.vector_slot(if matches!(op, Simd::Splat) { input } else { output })?;
        self.ins().stack_store(word, slot, 0);
        Ok(Some(address))
    }

    fn simd_lanes(&mut self, op: Simd, substs: &[Ty], args: &[Value]) -> Result<Value> {
        let vector = substs[0];
        Ok(match (op, args) {
            (Simd::Lanewise(bin), &[x, y]) => self.lanewise(bin, (vector, substs.get(1).copied().unwrap_or(vector)), (x, y))?,
            (Simd::Select, &[mask, yes, no]) => self.lane_select((vector, substs.get(1).copied().unwrap_or(vector)), mask, (yes, no))?,
            (Simd::SelectBitmask, &[mask, yes, no]) => self.bitmask_select((vector, substs.get(1).copied().unwrap_or(vector)), mask, (yes, no))?,
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
        let Kind::Adt(item, args) = *self.cx.types.kind(ty) else { bail!("a SIMD intrinsic on a type that is not a vector") };
        let slice = self.cx.types.substs(args).to_vec();
        let fields = crate::sema::infer::field_types(self.cx, item, &slice)?;
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

    fn lane_sqrt(&mut self, vector: Ty, x: Value) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let (slot, address) = self.vector_slot(vector)?;
        for lane in 0..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            let root = self.ins().sqrt(value);
            self.ins().stack_store(root, slot, lane * lanes.size);
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

    fn fold(&mut self, vector: Ty, x: Value, op: BinOp) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let mut folded = self.ins().load(lanes.machine, x, 0);
        for lane in 1..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            folded = self.apply(op, lanes.num, (folded, value), lanes.num)?;
        }
        Ok(folded)
    }

    fn extreme(&mut self, vector: Ty, x: Value, max: bool) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let better = if max { BinOp::Gt } else { BinOp::Lt };
        let mut best = self.ins().load(lanes.machine, x, 0);
        for lane in 1..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            let wins = self.apply(better, lanes.num, (value, best), lanes.num)?;
            best = self.ins().select(wins, value, best);
        }
        Ok(best)
    }

    /// Calls the host function of the x86 intrinsic `x86` with `args`, each vector by address, and returns the address of the frame slot it writes a vector result to, or the integer it returns.
    pub(super) fn host_vector(&mut self, x86: X86Symbol, args: &[Value]) -> Result<Value> {
        if let Some(value) = self.inline_vector(x86, args) {
            return Ok(value);
        }
        if x86.host_addr().is_none() {
            bail!("{} has no host function on this machine", x86.link_name());
        }
        let slot = self.b.create_stack_slot(x86.out_bytes(), 5);
        let out = self.ins().stack_addr(Type::I64, slot, 0);
        let values: Vec<Value> = std::iter::once(out).chain(args.iter().copied()).collect();
        self.call_host(HostSymbol::X86(x86), &values, None);
        Ok(match x86.scalar_bytes() {
            Some(8) => self.ins().load(Type::I64, out, 0),
            Some(_) => self.ins().load(Type::I32, out, 0),
            None => out,
        })
    }

    /// The `bytes` bytes whose byte `i` is byte `index[i] & 15` of the 16-byte lane of `table` that holds byte `i`, or zero when bit 7 of `index[i]` is set, as `pshufb` gives.
    pub(super) fn byte_permute(&mut self, args: &[Value], bytes: u8) -> Result<Value> {
        let &[table, index] = args else { bail!("`pshufb` takes two vectors") };
        if i32::from(bytes) == PACKED_BYTES && Target::host().arch == Arch::X64 {
            return Ok(self.vector_result(VectorOp::ByteShuffle, (table, index)));
        }
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

    /// The 16 bytes whose byte `i` is the larger of bytes `2i` and `2i + 1` of the first vector followed by the second, as NEON `umaxp` gives.
    pub(super) fn byte_pair_max(&mut self, args: &[Value]) -> Result<Value> {
        let &[first, second] = args else { bail!("`umaxp` takes two vectors") };
        let slot = self.b.create_stack_slot(16, 4);
        for lane in 0..16 {
            let from = if lane < 8 { first } else { second };
            let at = lane % 8 * 2;
            let (x, y) = (self.ins().load(Type::I8, from, at), self.ins().load(Type::I8, from, at + 1));
            let wins = self.ins().icmp(IntCC::UnsignedGreaterThan, x, y);
            let value = self.ins().select(wins, x, y);
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

    fn lane_select(&mut self, (mask_ty, out_ty): (Ty, Ty), mask: Value, (yes, no): (Value, Value)) -> Result<Value> {
        let (from, to) = (self.lanes(mask_ty)?, self.lanes(out_ty)?);
        let (slot, address) = self.vector_slot(out_ty)?;
        for lane in 0..to.count {
            let cond = self.ins().load(from.machine, mask, lane * from.size);
            let set = self.ins().icmp_imm(IntCC::NotEqual, cond, 0);
            let a = self.ins().load(to.machine, yes, lane * to.size);
            let b = self.ins().load(to.machine, no, lane * to.size);
            let chosen = self.ins().select(set, a, b);
            self.ins().stack_store(chosen, slot, lane * to.size);
        }
        Ok(address)
    }

    fn bitmask_select(&mut self, (mask_ty, out_ty): (Ty, Ty), mask: Value, (yes, no): (Value, Value)) -> Result<Value> {
        let Some(from) = machine(self.cx, mask_ty)? else { bail!("a SIMD bitmask select from an array is not supported yet") };
        let to = self.lanes(out_ty)?;
        let (slot, address) = self.vector_slot(out_ty)?;
        let bits = self.resize(mask, (from, false), Type::I64);
        for lane in 0..to.count {
            let shift = self.ins().iconst(Type::I64, i64::from(lane));
            let shifted = self.ins().ushr(bits, shift);
            let one = self.ins().iconst(Type::I64, 1);
            let bit = self.ins().band(shifted, one);
            let set = self.ins().icmp_imm(IntCC::NotEqual, bit, 0);
            let a = self.ins().load(to.machine, yes, lane * to.size);
            let b = self.ins().load(to.machine, no, lane * to.size);
            let chosen = self.ins().select(set, a, b);
            self.ins().stack_store(chosen, slot, lane * to.size);
        }
        Ok(address)
    }

    /// The address of a frame slot holding the x64 vector instruction `op` on the vectors at `x` and `y`.
    fn vector_result(&mut self, op: VectorOp, (x, y): (Value, Value)) -> Value {
        let (a, b) = (self.ins().load(Type::V128, x, 0), self.ins().load(Type::V128, y, 0));
        let vector = self.ins().vector(op, a, b);
        self.store_vector(vector)
    }

    fn store_vector(&mut self, vector: Value) -> Value {
        let slot = self.b.create_stack_slot(PACKED_BYTES as u32, 4);
        self.ins().stack_store(vector, slot, 0);
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    /// The x86 intrinsic `x86` on the vectors at `args` as one instruction in the code, on an x64 host, when it is one the code generator encodes.
    fn inline_vector(&mut self, x86: X86Symbol, args: &[Value]) -> Option<Value> {
        if Target::host().arch != Arch::X64 {
            return None;
        }
        match (x86, args) {
            (X86Symbol::Sha256rnds2, &[x, y, k]) => {
                let (a, b, c) = (self.ins().load(Type::V128, x, 0), self.ins().load(Type::V128, y, 0), self.ins().load(Type::V128, k, 0));
                let vector = self.ins().sha256_rounds(a, b, c);
                Some(self.store_vector(vector))
            }
            (_, &[x, y]) => Some(self.vector_result(instruction(x86)?, (x, y))),
            _ => None,
        }
    }
}

/// The vector instruction that computes the two-operand x86 intrinsic `x86`, when there is one.
fn instruction(x86: X86Symbol) -> Option<VectorOp> {
    Some(match x86 {
        X86Symbol::Sha1nexte => VectorOp::Sha1NextE,
        X86Symbol::Sha1msg1 => VectorOp::Sha1Msg1,
        X86Symbol::Sha1msg2 => VectorOp::Sha1Msg2,
        X86Symbol::Sha256msg1 => VectorOp::Sha256Msg1,
        X86Symbol::Sha256msg2 => VectorOp::Sha256Msg2,
        X86Symbol::Aesenc => VectorOp::AesEnc,
        X86Symbol::Aesenclast => VectorOp::AesEncLast,
        X86Symbol::Aesdec => VectorOp::AesDec,
        X86Symbol::Aesdeclast => VectorOp::AesDecLast,
        _ => return None,
    })
}

/// The packed add or subtract of `op` on lanes of `size` bytes.
fn lane_arithmetic(op: BinOp, size: i32) -> VectorOp {
    match (op == BinOp::Add, size) {
        (true, 1) => VectorOp::AddBytes,
        (true, 2) => VectorOp::AddWords,
        (true, 4) => VectorOp::AddDwords,
        (true, _) => VectorOp::AddQwords,
        (false, 1) => VectorOp::SubBytes,
        (false, 2) => VectorOp::SubWords,
        (false, 4) => VectorOp::SubDwords,
        (false, _) => VectorOp::SubQwords,
    }
}
