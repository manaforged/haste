use anyhow::{Result, bail};
use haste_codegen::{Arch, IntCC, Lane, ReduceOp, StackSlot, Target, TargetOpId, Type, Value};

use crate::host::{HostSymbol, X86Symbol};
use crate::names::Primitive;

use super::super::Lower;
use super::wide::wide_quads;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PtestMode {
    Z,
    C,
    Nzc,
}

impl Lower<'_, '_> {
    pub(in crate::lower) fn shuffle_dyn(&mut self, args: &[Value], tbl: bool) -> Result<Value> {
        let &[table, index] = args else { bail!("byte shuffle takes two vectors") };
        let res = self.shuffle_dyn_16(table, index, tbl, 0);
        Ok(self.store_vec(res, 16))
    }

    fn shuffle_dyn_16(&mut self, table: Value, index: Value, tbl: bool, off: i32) -> Value {
        let ty = Type::Vec { lane: Lane::I8, count: 16 };
        let (t, idx) = (self.ins().load(ty, table, off), self.ins().load(ty, index, off));
        let arch = Target::host().arch;
        if (tbl && arch == Arch::Arm64) || (!tbl && arch == Arch::X64) {
            return self.ins().shuffle_dyn(t, t, idx);
        }
        let adj = if tbl {
            let sixteen_s = self.ins().iconst(Type::I8, 16);
            let sixteen = self.ins().splat(ty, sixteen_s);
            let oob = self.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, idx, sixteen);
            self.ins().bor(idx, oob)
        } else {
            let zero = self.ins().iconst(ty, 0);
            let neg = self.ins().icmp(IntCC::SignedLessThan, idx, zero);
            let fifteen_s = self.ins().iconst(Type::I8, 15);
            let fifteen = self.ins().splat(ty, fifteen_s);
            let masked = self.ins().band(idx, fifteen);
            self.ins().bor(masked, neg)
        };
        self.ins().shuffle_dyn(t, t, adj)
    }

    pub(in crate::lower) fn byte_permute_32(&mut self, args: &[Value]) -> Result<Value> {
        let &[table, index] = args else { bail!("`pshufb` takes two vectors") };
        if let Some(value) = self.wide_binary(TargetOpId::X64Pshufb256, table, index) {
            return Ok(value);
        }
        let slot = self.b.create_stack_slot(32, 5);
        let lo = self.shuffle_dyn_16(table, index, false, 0);
        let hi = self.shuffle_dyn_16(table, index, false, 16);
        self.ins().stack_store(lo, slot, 0);
        self.ins().stack_store(hi, slot, 16);
        Ok(self.ins().stack_addr(Type::I64, slot, 0))
    }

    pub(in crate::lower) fn avg_round_vec(&mut self, ty: Type, args: &[Value]) -> Result<Value> {
        let &[x, y] = args else { bail!("`avg_round` takes two vectors") };
        let (va, vb) = (self.ins().load(ty, x, 0), self.ins().load(ty, y, 0));
        let out = self.ins().avg_round(va, vb);
        Ok(self.store_vec(out, ty.bytes()))
    }

    pub(in crate::lower) fn umaxp_vec(&mut self, args: &[Value]) -> Result<Value> {
        let &[x, y] = args else { bail!("`umaxp` takes two vectors") };
        let ty = Type::Vec { lane: Lane::I8, count: 16 };
        let (va, vb) = (self.ins().load(ty, x, 0), self.ins().load(ty, y, 0));
        let even = self.ins().shuffle(va, vb, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
        let odd = self.ins().shuffle(va, vb, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]);
        let out = self.ins().umax(even, odd);
        Ok(self.store_vec(out, 16))
    }

    pub(in crate::lower) fn sqdmulh_vec(&mut self, ty: Type, args: &[Value]) -> Result<Value> {
        let &[x, y] = args else { bail!("`sqdmulh` takes two vectors") };
        let i32x2 = Type::Vec { lane: Lane::I32, count: 2 };
        let i64x2 = Type::Vec { lane: Lane::I64, count: 2 };
        let i16x4 = Type::Vec { lane: Lane::I16, count: 4 };
        let i32x4 = Type::Vec { lane: Lane::I32, count: 4 };
        let i16x8 = Type::Vec { lane: Lane::I16, count: 8 };
        if ty == i32x2 {
            let out = self.sqdmulh_half(i32x2, i64x2, 31, (x, y), 0);
            return Ok(self.store_vec(out, 8));
        }
        let (half_ty, wide_ty, shift) = match ty {
            _ if ty == i32x4 => (i32x2, i64x2, 31),
            _ if ty == i16x8 => (i16x4, i32x4, 15),
            _ => bail!("unsupported `sqdmulh` type {ty:?}"),
        };
        let lo = self.sqdmulh_half(half_ty, wide_ty, shift, (x, y), 0);
        let hi = self.sqdmulh_half(half_ty, wide_ty, shift, (x, y), 8);
        let slot = self.b.create_stack_slot(16, 4);
        self.ins().stack_store(lo, slot, 0);
        self.ins().stack_store(hi, slot, 8);
        Ok(self.ins().stack_addr(Type::I64, slot, 0))
    }

    fn sqdmulh_half(&mut self, half_ty: Type, wide_ty: Type, shift_bits: i64, (x, y): (Value, Value), off: i32) -> Value {
        let (va, vb) = (self.ins().load(half_ty, x, off), self.ins().load(half_ty, y, off));
        let (wa, wb) = (self.ins().sextend(wide_ty, va), self.ins().sextend(wide_ty, vb));
        let prod = self.ins().imul(wa, wb);
        let wide_lane = wide_ty.lane().map_or(Type::I64, Lane::to_type);
        let sh_s = self.ins().iconst(wide_lane, shift_bits);
        let sh = self.ins().splat(wide_ty, sh_s);
        let shifted = self.ins().sshr(prod, sh);
        self.ins().narrow_sat(half_ty, true, shifted)
    }

    pub(in crate::lower) fn target_vector(&mut self, id: TargetOpId, ty: Type, args: &[Value]) -> Result<Value> {
        let arch = Target::host().arch;
        if is_arm64_target_op(id) && arch != Arch::Arm64 {
            bail!("ARM64 vector intrinsic {id:?} is only supported on Arm64");
        }
        if !is_arm64_target_op(id) && arch != Arch::X64 {
            bail!("x86 vector intrinsic {id:?} is only supported on X64");
        }
        if !(1..=3).contains(&args.len()) {
            bail!("target vector op {id:?} expects 1 to 3 arguments, got {}", args.len());
        }
        let loaded: Vec<Value> = args.iter().map(|&arg| self.ins().load(ty, arg, 0)).collect();
        let vec = self.ins().target_op(id, ty, &loaded);
        Ok(self.store_vec(vec, ty.bytes()))
    }

    pub(in crate::lower) fn host_vector(&mut self, x86: X86Symbol, args: &[Value]) -> Result<Value> {
        if let Some(value) = self.x86_inline(x86, args)? {
            return Ok(match wide_quads().filter(|wide| x86.out_bytes() == wide.bytes() && x86.scalar_bytes().is_none()) {
                Some(wide) => self.join_wide(wide, value),
                None => value,
            });
        }
        if Target::host().arch == Arch::X64
            && let Some((id, ty)) = x86_target_op(x86)
            && (1..=3).contains(&args.len())
        {
            let loaded: Vec<Value> = args.iter().map(|&arg| self.ins().load(ty, arg, 0)).collect();
            let vec = self.ins().target_op(id, ty, &loaded);
            return Ok(self.store_vec(vec, ty.bytes()));
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

    fn x86_inline(&mut self, x86: X86Symbol, args: &[Value]) -> Result<Option<Value>> {
        let i8x16 = Type::Vec { lane: Lane::I8, count: 16 };
        let i8x8 = Type::Vec { lane: Lane::I8, count: 8 };
        let i16x8 = Type::Vec { lane: Lane::I16, count: 8 };
        let i16x4 = Type::Vec { lane: Lane::I16, count: 4 };
        let i32x4 = Type::Vec { lane: Lane::I32, count: 4 };
        let f32x4 = Type::Vec { lane: Lane::F32, count: 4 };
        let f64x2 = Type::Vec { lane: Lane::F64, count: 2 };
        match (x86, args) {
            (X86Symbol::Addcarry64 | X86Symbol::Subborrow64, &[carry, a, b]) if Target::host().arch == Arch::X64 => {
                let (sum, carried) = if x86 == X86Symbol::Addcarry64 { (TargetOpId::X64AddCarry, TargetOpId::X64AddCarryOut) } else { (TargetOpId::X64SubBorrow, TargetOpId::X64SubBorrowOut) };
                let result = self.ins().target_op(sum, Type::I64, &[carry, a, b]);
                let out = self.ins().target_op(carried, Type::I8, &[carry, a, b]);
                let (byte, quad) = (self.cx.types.prim(Primitive::U8), self.cx.types.prim(Primitive::U64));
                Ok(Some(self.pair([byte, quad], [out, result])?))
            }
            (X86Symbol::Addcarry64 | X86Symbol::Subborrow64, &[carry, a, b]) => {
                let set = self.ins().icmp_imm(IntCC::NotEqual, carry, 0);
                let incoming = self.ins().uextend(Type::I128, set);
                let (wide_a, wide_b) = (self.ins().uextend(Type::I128, a), self.ins().uextend(Type::I128, b));
                let sum = if x86 == X86Symbol::Addcarry64 {
                    let partial = self.ins().iadd(wide_a, wide_b);
                    self.ins().iadd(partial, incoming)
                } else {
                    let partial = self.ins().isub(wide_a, wide_b);
                    self.ins().isub(partial, incoming)
                };
                let result = self.ins().ireduce(Type::I64, sum);
                let c64 = self.ins().iconst(Type::I64, 64);
                let high = self.ins().ushr(sum, c64);
                let high = self.ins().ireduce(Type::I64, high);
                let out = self.ins().icmp_imm(IntCC::NotEqual, high, 0);
                let (byte, quad) = (self.cx.types.prim(Primitive::U8), self.cx.types.prim(Primitive::U64));
                Ok(Some(self.pair([byte, quad], [out, result])?))
            }
            (X86Symbol::Pclmulqdq | X86Symbol::Pclmulqdq256 | X86Symbol::Pclmulqdq512, &[a, b, imm]) if Target::host().arch == Arch::X64 && self.b.const_int(imm).is_some() => {
                let id = match self.b.const_int(imm).unwrap_or_default() & 0x11 {
                    0x00 => TargetOpId::X64ClmulLowLow,
                    0x01 => TargetOpId::X64ClmulHighLow,
                    0x10 => TargetOpId::X64ClmulLowHigh,
                    _ => TargetOpId::X64ClmulHighHigh,
                };
                let i64x2 = Type::Vec { lane: Lane::I64, count: 2 };
                let product = |lower: &mut Self, off: i32| {
                    let (x, y) = (lower.ins().load(i64x2, a, off), lower.ins().load(i64x2, b, off));
                    lower.ins().target_op(id, i64x2, &[x, y])
                };
                Ok(Some(self.emit_chunks(x86.out_bytes(), product)))
            }
            (X86Symbol::Lddqu, &[at]) => {
                let v = self.ins().load(i8x16, at, 0);
                Ok(Some(self.store_vec(v, 16)))
            }
            (X86Symbol::Vlddqu, &[at]) => {
                let v = self.emit_256(|lower, off| lower.ins().load(i8x16, at, off));
                Ok(Some(v))
            }
            (X86Symbol::Packsswb | X86Symbol::Packuswb, &[a, b]) => {
                let slot = self.b.create_stack_slot(16, 4);
                self.pack_sat_half(i16x8, i8x8, x86 == X86Symbol::Packsswb, (a, b), (slot, 0));
                Ok(Some(self.ins().stack_addr(Type::I64, slot, 0)))
            }
            (X86Symbol::Packsswb256 | X86Symbol::Packuswb256, &[a, b]) => {
                let signed = x86 == X86Symbol::Packsswb256;
                let slot = self.b.create_stack_slot(32, 5);
                self.pack_sat_half(i16x8, i8x8, signed, (a, b), (slot, 0));
                self.pack_sat_half(i16x8, i8x8, signed, (a, b), (slot, 16));
                Ok(Some(self.ins().stack_addr(Type::I64, slot, 0)))
            }
            (X86Symbol::Packssdw | X86Symbol::Packusdw, &[a, b]) => {
                let slot = self.b.create_stack_slot(16, 4);
                self.pack_sat_half(i32x4, i16x4, x86 == X86Symbol::Packssdw, (a, b), (slot, 0));
                Ok(Some(self.ins().stack_addr(Type::I64, slot, 0)))
            }
            (X86Symbol::Packssdw256 | X86Symbol::Packusdw256, &[a, b]) => {
                let signed = x86 == X86Symbol::Packssdw256;
                let slot = self.b.create_stack_slot(32, 5);
                self.pack_sat_half(i32x4, i16x4, signed, (a, b), (slot, 0));
                self.pack_sat_half(i32x4, i16x4, signed, (a, b), (slot, 16));
                Ok(Some(self.ins().stack_addr(Type::I64, slot, 0)))
            }
            (X86Symbol::Pmaddwd, &[a, b]) => {
                let v = self.pmaddwd_128(a, b, 0);
                Ok(Some(self.store_vec(v, 16)))
            }
            (X86Symbol::Pmaddwd256, &[a, b]) => Ok(Some(self.wide_binary(TargetOpId::X64Pmaddwd256, a, b).unwrap_or_else(|| self.emit_256(|lower, off| lower.pmaddwd_128(a, b, off))))),
            (X86Symbol::Pmaddubsw128, &[a, b]) => {
                let v = self.pmaddubsw_128(a, b, 0);
                Ok(Some(self.store_vec(v, 16)))
            }
            (X86Symbol::Pmaddubsw256, &[a, b]) => Ok(Some(self.wide_binary(TargetOpId::X64Pmaddubsw256, a, b).unwrap_or_else(|| self.emit_256(|lower, off| lower.pmaddubsw_128(a, b, off))))),
            (X86Symbol::Psadbw, &[a, b]) => {
                let slot = self.b.create_stack_slot(16, 4);
                self.psadbw_128((a, b), (slot, 0));
                Ok(Some(self.ins().stack_addr(Type::I64, slot, 0)))
            }
            (X86Symbol::Psadbw256, &[a, b]) => {
                if let Some(v) = self.wide_binary(TargetOpId::X64Psadbw256, a, b) {
                    return Ok(Some(v));
                }
                let slot = self.b.create_stack_slot(32, 5);
                self.psadbw_128((a, b), (slot, 0));
                self.psadbw_128((a, b), (slot, 16));
                Ok(Some(self.ins().stack_addr(Type::I64, slot, 0)))
            }
            (X86Symbol::Minps | X86Symbol::Maxps, &[a, b]) => {
                let v = self.fminmax_128(f32x4, x86 == X86Symbol::Maxps, a, b, 0);
                Ok(Some(self.store_vec(v, 16)))
            }
            (X86Symbol::Minpd | X86Symbol::Maxpd, &[a, b]) => {
                let v = self.fminmax_128(f64x2, x86 == X86Symbol::Maxpd, a, b, 0);
                Ok(Some(self.store_vec(v, 16)))
            }
            (X86Symbol::Minps256 | X86Symbol::Maxps256, &[a, b]) => Ok(Some(self.emit_256(|lower, off| lower.fminmax_128(f32x4, x86 == X86Symbol::Maxps256, a, b, off)))),
            (X86Symbol::Minpd256 | X86Symbol::Maxpd256, &[a, b]) => Ok(Some(self.emit_256(|lower, off| lower.fminmax_128(f64x2, x86 == X86Symbol::Maxpd256, a, b, off)))),
            (
                X86Symbol::Psllw
                | X86Symbol::Pslld
                | X86Symbol::Psllq
                | X86Symbol::Psrlw
                | X86Symbol::Psrld
                | X86Symbol::Psrlq
                | X86Symbol::Psraw
                | X86Symbol::Psrad
                | X86Symbol::Psllw256
                | X86Symbol::Pslld256
                | X86Symbol::Psllq256
                | X86Symbol::Psrlw256
                | X86Symbol::Psrld256
                | X86Symbol::Psrlq256
                | X86Symbol::Psraw256
                | X86Symbol::Psrad256,
                &[a, count],
            ) => {
                let (lane, n, left, signed, wide) = match x86 {
                    X86Symbol::Psllw => (Lane::I16, 8, true, false, false),
                    X86Symbol::Pslld => (Lane::I32, 4, true, false, false),
                    X86Symbol::Psllq => (Lane::I64, 2, true, false, false),
                    X86Symbol::Psrlw => (Lane::I16, 8, false, false, false),
                    X86Symbol::Psrld => (Lane::I32, 4, false, false, false),
                    X86Symbol::Psrlq => (Lane::I64, 2, false, false, false),
                    X86Symbol::Psraw => (Lane::I16, 8, false, true, false),
                    X86Symbol::Psrad => (Lane::I32, 4, false, true, false),
                    X86Symbol::Psllw256 => (Lane::I16, 8, true, false, true),
                    X86Symbol::Pslld256 => (Lane::I32, 4, true, false, true),
                    X86Symbol::Psllq256 => (Lane::I64, 2, true, false, true),
                    X86Symbol::Psrlw256 => (Lane::I16, 8, false, false, true),
                    X86Symbol::Psrld256 => (Lane::I32, 4, false, false, true),
                    X86Symbol::Psrlq256 => (Lane::I64, 2, false, false, true),
                    X86Symbol::Psraw256 => (Lane::I16, 8, false, true, true),
                    _ => (Lane::I32, 4, false, true, true),
                };
                Ok(Some(self.vec_shift_sat(lane, n, left, signed, wide, a, count)))
            }
            (
                X86Symbol::Psignb128
                | X86Symbol::Psignw128
                | X86Symbol::Psignd128
                | X86Symbol::Psignb256
                | X86Symbol::Psignw256
                | X86Symbol::Psignd256,
                &[a, b],
            ) => {
                let (ty, wide) = match x86 {
                    X86Symbol::Psignb128 => (i8x16, false),
                    X86Symbol::Psignw128 => (i16x8, false),
                    X86Symbol::Psignd128 => (i32x4, false),
                    X86Symbol::Psignb256 => (i8x16, true),
                    X86Symbol::Psignw256 => (i16x8, true),
                    _ => (i32x4, true),
                };
                if wide {
                    Ok(Some(self.emit_256(|lower, off| lower.psign_128(ty, a, b, off))))
                } else {
                    let v = self.psign_128(ty, a, b, 0);
                    Ok(Some(self.store_vec(v, 16)))
                }
            }
            (X86Symbol::Minss | X86Symbol::Maxss, &[a, b]) => {
                let va = self.ins().load(f32x4, a, 0);
                let mm = self.fminmax_128(f32x4, x86 == X86Symbol::Maxss, a, b, 0);
                let res = self.ins().shuffle(mm, va, [0, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
                Ok(Some(self.store_vec(res, 16)))
            }
            (X86Symbol::Minsd | X86Symbol::Maxsd, &[a, b]) => {
                let va = self.ins().load(f64x2, a, 0);
                let mm = self.fminmax_128(f64x2, x86 == X86Symbol::Maxsd, a, b, 0);
                let res = self.ins().shuffle(mm, va, [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
                Ok(Some(self.store_vec(res, 16)))
            }
            (
                X86Symbol::Ptestnzc
                | X86Symbol::Ptestnzc256
                | X86Symbol::Vtestnzcps
                | X86Symbol::Vtestnzcpd
                | X86Symbol::Vtestzps256
                | X86Symbol::Vtestcps256
                | X86Symbol::Vtestnzcps256
                | X86Symbol::Vtestzpd256
                | X86Symbol::Vtestcpd256
                | X86Symbol::Vtestnzcpd256,
                &[a, b],
            ) => {
                let i64x2 = Type::Vec { lane: Lane::I64, count: 2 };
                let (ty, wide, mask_lane, mode) = match x86 {
                    X86Symbol::Ptestnzc => (i64x2, false, None, PtestMode::Nzc),
                    X86Symbol::Ptestnzc256 => (i64x2, true, None, PtestMode::Nzc),
                    X86Symbol::Vtestnzcps => (i32x4, false, Some(Lane::I32), PtestMode::Nzc),
                    X86Symbol::Vtestnzcpd => (i64x2, false, Some(Lane::I64), PtestMode::Nzc),
                    X86Symbol::Vtestzps256 => (i32x4, true, Some(Lane::I32), PtestMode::Z),
                    X86Symbol::Vtestcps256 => (i32x4, true, Some(Lane::I32), PtestMode::C),
                    X86Symbol::Vtestnzcps256 => (i32x4, true, Some(Lane::I32), PtestMode::Nzc),
                    X86Symbol::Vtestzpd256 => (i64x2, true, Some(Lane::I64), PtestMode::Z),
                    X86Symbol::Vtestcpd256 => (i64x2, true, Some(Lane::I64), PtestMode::C),
                    _ => (i64x2, true, Some(Lane::I64), PtestMode::Nzc),
                };
                Ok(Some(self.vec_ptest(ty, wide, mask_lane, mode, (a, b))))
            }
            (X86Symbol::Pmulhrsw128 | X86Symbol::Pmulhrsw256, &[a, b]) => {
                let wide = x86 == X86Symbol::Pmulhrsw256;
                let slot = self.b.create_stack_slot(if wide { 32 } else { 16 }, if wide { 5 } else { 4 });
                self.pmulhrsw_128(a, b, slot, 0);
                if wide {
                    self.pmulhrsw_128(a, b, slot, 16);
                }
                Ok(Some(self.ins().stack_addr(Type::I64, slot, 0)))
            }
            _ => Ok(None),
        }
    }

    fn emit_256(&mut self, half: impl FnMut(&mut Self, i32) -> Value) -> Value {
        self.emit_chunks(32, half)
    }

    /// The address of a new `bytes`-byte frame slot that holds each 16-byte chunk `chunk` gives for its offset.
    fn emit_chunks(&mut self, bytes: u32, mut chunk: impl FnMut(&mut Self, i32) -> Value) -> Value {
        let slot = self.b.create_stack_slot(bytes, 5);
        let offsets: Vec<i32> = (0..bytes as i32).step_by(16).collect();
        let values: Vec<Value> = offsets.iter().map(|&off| chunk(self, off)).collect();
        for (off, value) in offsets.into_iter().zip(values) {
            self.ins().stack_store(value, slot, off);
        }
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    fn low_8(&mut self, half_ty: Type, vec16: Value) -> Value {
        let slot = self.b.create_stack_slot(16, 4);
        self.ins().stack_store(vec16, slot, 0);
        self.ins().stack_load(half_ty, slot, 0)
    }

    fn pack_sat_half(&mut self, in_ty: Type, out_half: Type, signed: bool, (a, b): (Value, Value), (slot, off): (StackSlot, i32)) {
        let (va, vb) = (self.ins().load(in_ty, a, off), self.ins().load(in_ty, b, off));
        let lo = self.ins().narrow_sat(out_half, signed, va);
        let hi = self.ins().narrow_sat(out_half, signed, vb);
        self.ins().stack_store(lo, slot, off);
        self.ins().stack_store(hi, slot, off + 8);
    }

    fn pmaddwd_128(&mut self, a: Value, b: Value, off: i32) -> Value {
        let i16x8 = Type::Vec { lane: Lane::I16, count: 8 };
        let i16x4 = Type::Vec { lane: Lane::I16, count: 4 };
        let i32x4 = Type::Vec { lane: Lane::I32, count: 4 };
        let even = [0, 2, 4, 6, 0, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0];
        let odd = [1, 3, 5, 7, 1, 3, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0];
        let (va, vb) = (self.ins().load(i16x8, a, off), self.ins().load(i16x8, b, off));
        let (se_a, se_b) = (self.ins().shuffle(va, va, even), self.ins().shuffle(vb, vb, even));
        let (so_a, so_b) = (self.ins().shuffle(va, va, odd), self.ins().shuffle(vb, vb, odd));
        let (la, lb) = (self.low_8(i16x4, se_a), self.low_8(i16x4, se_b));
        let (ra, rb) = (self.low_8(i16x4, so_a), self.low_8(i16x4, so_b));
        let (ea, eb) = (self.ins().sextend(i32x4, la), self.ins().sextend(i32x4, lb));
        let (oa, ob) = (self.ins().sextend(i32x4, ra), self.ins().sextend(i32x4, rb));
        let pe = self.ins().imul(ea, eb);
        let po = self.ins().imul(oa, ob);
        self.ins().iadd(pe, po)
    }

    fn pmaddubsw_128(&mut self, a: Value, b: Value, off: i32) -> Value {
        let i8x16 = Type::Vec { lane: Lane::I8, count: 16 };
        let i8x8 = Type::Vec { lane: Lane::I8, count: 8 };
        let i16x8 = Type::Vec { lane: Lane::I16, count: 8 };
        let even = [0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14];
        let odd = [1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15];
        let (va, vb) = (self.ins().load(i8x16, a, off), self.ins().load(i8x16, b, off));
        let (se_a, so_a) = (self.ins().shuffle(va, va, even), self.ins().shuffle(va, va, odd));
        let (se_b, so_b) = (self.ins().shuffle(vb, vb, even), self.ins().shuffle(vb, vb, odd));
        let (lea, loa) = (self.low_8(i8x8, se_a), self.low_8(i8x8, so_a));
        let (leb, lob) = (self.low_8(i8x8, se_b), self.low_8(i8x8, so_b));
        let (ea, oa) = (self.ins().uextend(i16x8, lea), self.ins().uextend(i16x8, loa));
        let (eb, ob) = (self.ins().sextend(i16x8, leb), self.ins().sextend(i16x8, lob));
        let pe = self.ins().imul(ea, eb);
        let po = self.ins().imul(oa, ob);
        self.ins().sadd_sat(pe, po)
    }

    fn psadbw_128(&mut self, (a, b): (Value, Value), (slot, off): (StackSlot, i32)) {
        let u8x16 = Type::Vec { lane: Lane::I8, count: 16 };
        let u8x8 = Type::Vec { lane: Lane::I8, count: 8 };
        let i16x8 = Type::Vec { lane: Lane::I16, count: 8 };
        let (va, vb) = (self.ins().load(u8x16, a, off), self.ins().load(u8x16, b, off));
        let d1 = self.ins().usub_sat(va, vb);
        let d2 = self.ins().usub_sat(vb, va);
        let diff = self.ins().bor(d1, d2);
        let tmp = self.b.create_stack_slot(16, 4);
        self.ins().stack_store(diff, tmp, 0);
        let (lo8, hi8) = (self.ins().stack_load(u8x8, tmp, 0), self.ins().stack_load(u8x8, tmp, 8));
        let (lo16, hi16) = (self.ins().uextend(i16x8, lo8), self.ins().uextend(i16x8, hi8));
        let (red_lo, red_hi) = (self.ins().reduce(ReduceOp::Add, lo16), self.ins().reduce(ReduceOp::Add, hi16));
        let (sum_lo, sum_hi) = (self.ins().uextend(Type::I64, red_lo), self.ins().uextend(Type::I64, red_hi));
        self.ins().stack_store(sum_lo, slot, off);
        self.ins().stack_store(sum_hi, slot, off + 8);
    }

    fn fminmax_128(&mut self, ty: Type, max: bool, a: Value, b: Value, off: i32) -> Value {
        let (va, vb) = (self.ins().load(ty, a, off), self.ins().load(ty, b, off));
        if max { self.ins().fmax(va, vb) } else { self.ins().fmin(va, vb) }
    }

    fn vec_ptest(&mut self, ty: Type, wide: bool, mask_lane: Option<Lane>, mode: PtestMode, (a, b): (Value, Value)) -> Value {
        let mask = match mask_lane {
            Some(Lane::I32) => {
                let s = self.ins().iconst(Type::I32, 0x8000_0000_u32 as i64);
                Some(self.ins().splat(ty, s))
            }
            Some(Lane::I64) => {
                let s = self.ins().iconst(Type::I64, i64::MIN);
                Some(self.ins().splat(ty, s))
            }
            _ => None,
        };
        let mut combine = |invert_a: bool| {
            let mut eval = |off: i32| {
                let (va, vb) = (self.ins().load(ty, a, off), self.ins().load(ty, b, off));
                let lhs = if invert_a { self.ins().bnot(va) } else { va };
                let r = self.ins().band(lhs, vb);
                mask.map_or(r, |m| self.ins().band(r, m))
            };
            let lo = eval(0);
            if wide {
                let hi = eval(16);
                self.ins().bor(lo, hi)
            } else {
                lo
            }
        };
        let flag = match mode {
            PtestMode::Z | PtestMode::C => {
                let acc = combine(mode == PtestMode::C);
                let red = self.ins().reduce(ReduceOp::Or, acc);
                self.ins().icmp_imm(IntCC::Equal, red, 0)
            }
            PtestMode::Nzc => {
                let (ab, nab) = (combine(false), combine(true));
                let (r1, r2) = (self.ins().reduce(ReduceOp::Or, ab), self.ins().reduce(ReduceOp::Or, nab));
                let (nz1, nz2) = (self.ins().icmp_imm(IntCC::NotEqual, r1, 0), self.ins().icmp_imm(IntCC::NotEqual, r2, 0));
                self.ins().band(nz1, nz2)
            }
        };
        self.resize(flag, (Type::I8, false), Type::I32)
    }

    fn psign_128(&mut self, ty: Type, a: Value, b: Value, off: i32) -> Value {
        let (va, vb) = (self.ins().load(ty, a, off), self.ins().load(ty, b, off));
        let zero = self.ins().iconst(ty, 0);
        let neg = self.ins().ineg(va);
        let lt0 = self.ins().icmp(IntCC::SignedLessThan, vb, zero);
        let eq0 = self.ins().icmp(IntCC::Equal, vb, zero);
        let signed = self.ins().select(lt0, neg, va);
        self.ins().select(eq0, zero, signed)
    }

    fn pmulhrsw_128(&mut self, a: Value, b: Value, slot: StackSlot, off: i32) {
        let i16x4 = Type::Vec { lane: Lane::I16, count: 4 };
        let i32x4 = Type::Vec { lane: Lane::I32, count: 4 };
        let bias_s = self.ins().iconst(Type::I32, 0x4000);
        let sh_s = self.ins().iconst(Type::I32, 15);
        let (bias, sh) = (self.ins().splat(i32x4, bias_s), self.ins().splat(i32x4, sh_s));
        for delta in [0, 8] {
            let (xa, xb) = (self.ins().load(i16x4, a, off + delta), self.ins().load(i16x4, b, off + delta));
            let (ea, eb) = (self.ins().sextend(i32x4, xa), self.ins().sextend(i32x4, xb));
            let prod = self.ins().imul(ea, eb);
            let sum = self.ins().iadd(prod, bias);
            let shifted = self.ins().sshr(sum, sh);
            let out = self.ins().ireduce(i16x4, shifted);
            self.ins().stack_store(out, slot, off + delta);
        }
    }

    fn vec_shift_sat(&mut self, lane: Lane, count_lanes: u8, left: bool, signed: bool, wide: bool, a: Value, count: Value) -> Value {
        let vec_ty = Type::Vec { lane, count: count_lanes };
        let cnt = self.ins().load(Type::I64, count, 0);
        let lane_bits = lane.bits();
        let lane_ty = lane.to_type();
        let oob = self.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, cnt, i64::from(lane_bits));
        let sh_scalar = self.resize(cnt, (Type::I64, false), lane_ty);
        let sh_vec = self.ins().splat(vec_ty, sh_scalar);
        let sat = if signed {
            let c = self.ins().iconst(lane_ty, i64::from(lane_bits - 1));
            Some(self.ins().splat(vec_ty, c))
        } else {
            None
        };
        let zero = (!signed).then(|| self.ins().iconst(vec_ty, 0));
        let emit_half = |lower: &mut Self, off: i32| {
            let va = lower.ins().load(vec_ty, a, off);
            let shifted = match (left, signed) {
                (true, _) => lower.ins().ishl(va, sh_vec),
                (false, true) => lower.ins().sshr(va, sh_vec),
                (false, false) => lower.ins().ushr(va, sh_vec),
            };
            let saturated = match (sat, zero) {
                (Some(sat_sh), _) => lower.ins().sshr(va, sat_sh),
                (_, Some(z)) => z,
                _ => shifted,
            };
            lower.ins().select(oob, saturated, shifted)
        };
        if wide {
            self.emit_256(|lower, off| emit_half(lower, off))
        } else {
            let res = emit_half(self, 0);
            self.store_vec(res, 16)
        }
    }
}

fn is_arm64_target_op(id: TargetOpId) -> bool {
    matches!(
        id,
        TargetOpId::Arm64Sha256H
            | TargetOpId::Arm64Sha256H2
            | TargetOpId::Arm64Sha256Su0
            | TargetOpId::Arm64Sha256Su1
            | TargetOpId::Arm64AesE
            | TargetOpId::Arm64AesD
            | TargetOpId::Arm64AesMc
            | TargetOpId::Arm64AesImc
    )
}

fn x86_target_op(x86: X86Symbol) -> Option<(TargetOpId, Type)> {
    let i8x16 = Type::Vec { lane: Lane::I8, count: 16 };
    let i32x4 = Type::Vec { lane: Lane::I32, count: 4 };
    Some(match x86 {
        X86Symbol::Sha1nexte => (TargetOpId::X64Sha1NextE, i32x4),
        X86Symbol::Sha1msg1 => (TargetOpId::X64Sha1Msg1, i32x4),
        X86Symbol::Sha1msg2 => (TargetOpId::X64Sha1Msg2, i32x4),
        X86Symbol::Sha256rnds2 => (TargetOpId::X64Sha256Rnds2, i32x4),
        X86Symbol::Sha256msg1 => (TargetOpId::X64Sha256Msg1, i32x4),
        X86Symbol::Sha256msg2 => (TargetOpId::X64Sha256Msg2, i32x4),
        X86Symbol::Aesenc => (TargetOpId::X64AesEnc, i8x16),
        X86Symbol::Aesenclast => (TargetOpId::X64AesEncLast, i8x16),
        X86Symbol::Aesdec => (TargetOpId::X64AesDec, i8x16),
        X86Symbol::Aesdeclast => (TargetOpId::X64AesDecLast, i8x16),
        X86Symbol::Aesimc => (TargetOpId::X64AesImc, i8x16),
        _ => return None,
    })
}
