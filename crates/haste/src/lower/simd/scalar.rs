use anyhow::{Result, bail};
use haste_codegen::{IntCC, StackSlot, Type, Value};

use crate::body::BinOp;
use crate::sema::intrinsic::Simd;
use crate::sema::ty::Ty;

use super::super::Lower;
use super::super::ops::Num;
use super::{Lanes, machine};

impl Lower<'_, '_> {
    pub(super) fn packed_swar64(&mut self, op: Simd, (input, output): (Ty, Ty), args: &[Value]) -> Result<Option<Value>> {
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

    pub(super) fn simd_lanes(&mut self, op: Simd, substs: &[Ty], args: &[Value]) -> Result<Value> {
        let vector = substs[0];
        Ok(match (op, args) {
            (Simd::Lanewise(bin), &[x, y]) => self.lanewise(bin, (vector, substs.get(1).copied().unwrap_or(vector)), (x, y))?,
            (Simd::Select, &[mask, yes, no]) => self.lane_select((vector, substs.get(1).copied().unwrap_or(vector)), mask, (yes, no))?,
            (Simd::Shuffle, &[x, y, idx]) => self.shuffle(substs, (x, y), idx)?,
            (Simd::Extract, &[x, at]) => {
                let (lanes, lane) = self.lane_address(vector, x, at)?;
                self.ins().load(lanes.machine, lane, 0)
            }
            (Simd::Insert, &[x, at, value]) => self.insert(vector, x, (at, value))?,
            _ => bail!("SIMD intrinsic {op:?} takes other arguments"),
        })
    }

    pub(super) fn splat(&mut self, vector: Ty, value: Value) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let (_, address) = self.vector_slot(vector)?;
        let size = i64::from(lanes.size);
        self.emit_lane_loop(0, lanes.count, &[], |lower, idx, _| {
            let offset = lower.ins().imul_imm(idx, size);
            let dst = lower.ins().iadd(address, offset);
            lower.ins().store(value, dst, 0);
            Ok(Vec::new())
        })?;
        Ok(address)
    }

    pub(super) fn lane_sqrt(&mut self, vector: Ty, x: Value) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let (_, address) = self.vector_slot(vector)?;
        let size = i64::from(lanes.size);
        let m = lanes.machine;
        self.emit_lane_loop(0, lanes.count, &[], |lower, idx, _| {
            let offset = lower.ins().imul_imm(idx, size);
            let src = lower.ins().iadd(x, offset);
            let value = lower.ins().load(m, src, 0);
            let root = lower.ins().sqrt(value);
            let dst = lower.ins().iadd(address, offset);
            lower.ins().store(root, dst, 0);
            Ok(Vec::new())
        })?;
        Ok(address)
    }

    fn lanewise(&mut self, op: BinOp, (input, output): (Ty, Ty), (x, y): (Value, Value)) -> Result<Value> {
        let (from, to) = (self.lanes(input)?, self.lanes(output)?);
        let (_, address) = self.vector_slot(output)?;
        let (fs, ts) = (i64::from(from.size), i64::from(to.size));
        self.emit_lane_loop(0, from.count, &[], |lower, idx, _| {
            let f_off = lower.ins().imul_imm(idx, fs);
            let t_off = lower.ins().imul_imm(idx, ts);
            let ax = lower.ins().iadd(x, f_off);
            let ay = lower.ins().iadd(y, f_off);
            let a = lower.ins().load(from.machine, ax, 0);
            let b = lower.ins().load(from.machine, ay, 0);
            let value = lower.lane_op(op, from.num, (a, b), to.machine)?;
            let dst = lower.ins().iadd(address, t_off);
            lower.ins().store(value, dst, 0);
            Ok(Vec::new())
        })?;
        Ok(address)
    }

    pub(super) fn lane_cast(&mut self, (input, output): (Ty, Ty), x: Value) -> Result<Value> {
        let (from, to) = (self.lanes(input)?, self.lanes(output)?);
        if from.count != to.count {
            bail!("`simd_cast` between vectors of {} and {} lanes", from.count, to.count);
        }
        let (_, address) = self.vector_slot(output)?;
        let (fs, ts) = (i64::from(from.size), i64::from(to.size));
        self.emit_lane_loop(0, from.count, &[], |lower, idx, _| {
            let f_off = lower.ins().imul_imm(idx, fs);
            let t_off = lower.ins().imul_imm(idx, ts);
            let src = lower.ins().iadd(x, f_off);
            let value = lower.ins().load(from.machine, src, 0);
            let value = lower.convert(value, from.num, to.num)?;
            let dst = lower.ins().iadd(address, t_off);
            lower.ins().store(value, dst, 0);
            Ok(Vec::new())
        })?;
        Ok(address)
    }

    fn shuffle(&mut self, substs: &[Ty], (x, y): (Value, Value), idx: Value) -> Result<Value> {
        let &[input, indices, output] = substs else { bail!("`simd_shuffle` takes three types") };
        let (from, to) = (self.lanes(input)?, self.lanes(output)?);
        let (_, joined) = self.pair_slot(&from)?;
        self.copy_bytes(joined, 0, x, 0, from.bytes());
        self.copy_bytes(joined, from.bytes(), y, 0, from.bytes());
        let Some(held) = self.spill(Some(idx), indices)? else { bail!("`simd_shuffle` indices take no bytes") };
        let idx = self.address(held);
        let (_, address) = self.vector_slot(output)?;
        let (fs, ts) = (i64::from(from.size), i64::from(to.size));
        let tm = to.machine;
        self.emit_lane_loop(0, to.count, &[], |lower, lane, _| {
            let i_off = lower.ins().imul_imm(lane, 4);
            let i_at = lower.ins().iadd(idx, i_off);
            let at = lower.ins().load(Type::I32, i_at, 0);
            let at = lower.ins().uextend(Type::I64, at);
            let offset = lower.ins().imul_imm(at, fs);
            let source = lower.ins().iadd(joined, offset);
            let value = lower.ins().load(tm, source, 0);
            let d_off = lower.ins().imul_imm(lane, ts);
            let dst = lower.ins().iadd(address, d_off);
            lower.ins().store(value, dst, 0);
            Ok(Vec::new())
        })?;
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
        let (_, address) = self.vector_slot(vector)?;
        self.copy_bytes(address, 0, x, 0, lanes.bytes());
        let (_, lane) = self.lane_address(vector, address, at)?;
        self.ins().store(value, lane, 0);
        Ok(address)
    }

    fn pair_slot(&mut self, lanes: &Lanes) -> Result<(StackSlot, Value)> {
        let bytes = (2 * lanes.count * lanes.size) as u32;
        let slot = self.b.create_stack_slot(bytes, lanes.size.trailing_zeros() as u8);
        Ok((slot, self.ins().stack_addr(Type::I64, slot, 0)))
    }

    pub(super) fn reduce(&mut self, vector: Ty, x: Value, all: bool) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let init = self.ins().iconst(Type::I8, i64::from(all));
        let size = i64::from(lanes.size);
        let m = lanes.machine;
        let out = self.emit_lane_loop(0, lanes.count, &[init], |lower, idx, acc| {
            let offset = lower.ins().imul_imm(idx, size);
            let src = lower.ins().iadd(x, offset);
            let value = lower.ins().load(m, src, 0);
            let set = lower.ins().icmp_imm(IntCC::NotEqual, value, 0);
            let next = if all { lower.ins().band(acc[0], set) } else { lower.ins().bor(acc[0], set) };
            Ok(vec![next])
        })?;
        Ok(out[0])
    }

    pub(super) fn fold(&mut self, vector: Ty, x: Value, op: BinOp) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let init = self.ins().load(lanes.machine, x, 0);
        let size = i64::from(lanes.size);
        let out = self.emit_lane_loop(1, lanes.count, &[init], |lower, idx, acc| {
            let offset = lower.ins().imul_imm(idx, size);
            let src = lower.ins().iadd(x, offset);
            let value = lower.ins().load(lanes.machine, src, 0);
            let next = lower.apply(op, lanes.num, (acc[0], value), lanes.num)?;
            Ok(vec![next])
        })?;
        Ok(out[0])
    }

    pub(super) fn extreme(&mut self, vector: Ty, x: Value, max: bool) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let better = if max { BinOp::Gt } else { BinOp::Lt };
        let init = self.ins().load(lanes.machine, x, 0);
        let size = i64::from(lanes.size);
        let out = self.emit_lane_loop(1, lanes.count, &[init], |lower, idx, acc| {
            let offset = lower.ins().imul_imm(idx, size);
            let src = lower.ins().iadd(x, offset);
            let value = lower.ins().load(lanes.machine, src, 0);
            let wins = lower.apply(better, lanes.num, (value, acc[0]), lanes.num)?;
            let next = lower.ins().select(wins, value, acc[0]);
            Ok(vec![next])
        })?;
        Ok(out[0])
    }

    pub(super) fn bitmask(&mut self, vector: Ty, x: Value, out: Ty) -> Result<Value> {
        let Some(target) = machine(self.cx, out)? else { bail!("a SIMD bitmask into an array is not supported yet") };
        let lanes = self.lanes(vector)?;
        let init = self.ins().iconst(Type::I64, 0);
        let size = i64::from(lanes.size);
        let m = lanes.machine;
        let done = self.emit_lane_loop(0, lanes.count, &[init], |lower, idx, acc| {
            let offset = lower.ins().imul_imm(idx, size);
            let src = lower.ins().iadd(x, offset);
            let value = lower.ins().load(m, src, 0);
            let set = lower.ins().icmp_imm(IntCC::NotEqual, value, 0);
            let bit = lower.ins().uextend(Type::I64, set);
            let shifted = lower.ins().ishl(bit, idx);
            let next = lower.ins().bor(acc[0], shifted);
            Ok(vec![next])
        })?;
        Ok(self.resize(done[0], (Type::I64, false), target))
    }

    fn lane_select(&mut self, (mask_ty, out_ty): (Ty, Ty), mask: Value, (yes, no): (Value, Value)) -> Result<Value> {
        let (from, to) = (self.lanes(mask_ty)?, self.lanes(out_ty)?);
        let (_, address) = self.vector_slot(out_ty)?;
        let (fs, ts) = (i64::from(from.size), i64::from(to.size));
        self.emit_lane_loop(0, to.count, &[], |lower, idx, _| {
            let f_off = lower.ins().imul_imm(idx, fs);
            let t_off = lower.ins().imul_imm(idx, ts);
            let m_at = lower.ins().iadd(mask, f_off);
            let cond = lower.ins().load(from.machine, m_at, 0);
            let set = lower.ins().icmp_imm(IntCC::NotEqual, cond, 0);
            let ay = lower.ins().iadd(yes, t_off);
            let an = lower.ins().iadd(no, t_off);
            let a = lower.ins().load(to.machine, ay, 0);
            let b = lower.ins().load(to.machine, an, 0);
            let chosen = lower.ins().select(set, a, b);
            let dst = lower.ins().iadd(address, t_off);
            lower.ins().store(chosen, dst, 0);
            Ok(Vec::new())
        })?;
        Ok(address)
    }

    pub(super) fn bitmask_select(&mut self, (mask_ty, out_ty): (Ty, Ty), mask: Value, (yes, no): (Value, Value)) -> Result<Value> {
        let Some(from) = machine(self.cx, mask_ty)? else { bail!("a SIMD bitmask select from an array is not supported yet") };
        let to = self.lanes(out_ty)?;
        let bits = self.resize(mask, (from, false), Type::I64);
        if let Some(vec_res) = self.select_bitmask_vec(&to, bits, (yes, no))? {
            return Ok(vec_res);
        }
        let (_, address) = self.vector_slot(out_ty)?;
        let ts = i64::from(to.size);
        let tm = to.machine;
        self.emit_lane_loop(0, to.count, &[], |lower, idx, _| {
            let shifted = lower.ins().ushr(bits, idx);
            let one = lower.ins().iconst(Type::I64, 1);
            let bit = lower.ins().band(shifted, one);
            let set = lower.ins().icmp_imm(IntCC::NotEqual, bit, 0);
            let t_off = lower.ins().imul_imm(idx, ts);
            let ay = lower.ins().iadd(yes, t_off);
            let an = lower.ins().iadd(no, t_off);
            let a = lower.ins().load(tm, ay, 0);
            let b = lower.ins().load(tm, an, 0);
            let chosen = lower.ins().select(set, a, b);
            let dst = lower.ins().iadd(address, t_off);
            lower.ins().store(chosen, dst, 0);
            Ok(Vec::new())
        })?;
        Ok(address)
    }
}
