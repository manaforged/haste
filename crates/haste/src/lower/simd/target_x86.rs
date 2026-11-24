use anyhow::Result;
use haste_codegen::{Arch, IntCC, Lane, ReduceOp, StackSlot, TargetOpId, Type, Value};

use crate::host::X86Symbol;
use crate::names::Primitive;

use super::super::super::Lower;
use super::super::SCRATCH_ALIGN;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PtestMode {
    Z,
    C,
    Nzc,
}

impl Lower<'_, '_> {
    pub(super) fn x86_inline(&mut self, x86: X86Symbol, args: &[Value]) -> Result<Option<Value>> {
        let i8x16 = Type::Vec {
            lane: Lane::I8,
            count: 16,
        };
        match (x86, args) {
            (X86Symbol::Addcarry64 | X86Symbol::Subborrow64, &[carry, a, b]) => self
                .x86_carry_op(x86 == X86Symbol::Addcarry64, carry, a, b)
                .map(Some),
            (
                X86Symbol::Pclmulqdq | X86Symbol::Pclmulqdq256 | X86Symbol::Pclmulqdq512,
                &[a, b, imm],
            ) if crate::compile_target().arch == Arch::X64 => Ok(self
                .b
                .const_int(imm)
                .map(|c| self.x86_pclmulqdq(x86.out_bytes(), a, b, c))),
            (X86Symbol::Lddqu, &[at]) => {
                let v = self.ins().load(i8x16, at, 0);
                Ok(Some(self.store_vec(v, 16)))
            }
            (X86Symbol::Vlddqu, &[at]) => {
                let v = self.emit_256(|lower, off| lower.ins().load(i8x16, at, off));
                Ok(Some(v))
            }
            (_, &[a, b]) => Ok(self.x86_inline_pair(x86, a, b)),
            _ => Ok(None),
        }
    }

    fn x86_inline_pair(&mut self, x86: X86Symbol, a: Value, b: Value) -> Option<Value> {
        if let Some((in_ty, out_half, signed, wide)) = pack_spec(x86) {
            return Some(self.x86_pack_sat(in_ty, out_half, signed, wide, a, b));
        }
        if let Some((ty, max, wide)) = fminmax_spec(x86) {
            return Some(if wide {
                self.emit_256(|lower, off| lower.fminmax_128(ty, max, a, b, off))
            } else {
                let v = self.fminmax_128(ty, max, a, b, 0);
                self.store_vec(v, 16)
            });
        }
        if let Some((lane, n, left, signed, wide)) = shift_spec(x86) {
            return Some(self.vec_shift_sat(lane, n, left, signed, wide, a, b));
        }
        if let Some((ty, wide)) = psign_spec(x86) {
            return Some(if wide {
                self.emit_256(|lower, off| lower.psign_128(ty, a, b, off))
            } else {
                let v = self.psign_128(ty, a, b, 0);
                self.store_vec(v, 16)
            });
        }
        if let Some((ty, wide, mask_lane, mode)) = ptest_spec(x86) {
            return Some(self.vec_ptest(ty, wide, mask_lane, mode, (a, b)));
        }
        let f32x4 = Type::Vec {
            lane: Lane::F32,
            count: 4,
        };
        let f64x2 = Type::Vec {
            lane: Lane::F64,
            count: 2,
        };
        match x86 {
            X86Symbol::Permd => Some(self.x86_permd(a, b)),
            X86Symbol::Pmaddwd => {
                let v = self.pmaddwd_128(a, b, 0);
                Some(self.store_vec(v, 16))
            }
            X86Symbol::Pmaddwd256 => Some(
                self.wide_binary(TargetOpId::X64Pmaddwd256, a, b)
                    .unwrap_or_else(|| self.emit_256(|lower, off| lower.pmaddwd_128(a, b, off))),
            ),
            X86Symbol::Pmaddubsw128 => {
                let v = self.pmaddubsw_128(a, b, 0);
                Some(self.store_vec(v, 16))
            }
            X86Symbol::Pmaddubsw256 => Some(
                self.wide_binary(TargetOpId::X64Pmaddubsw256, a, b)
                    .unwrap_or_else(|| self.emit_256(|lower, off| lower.pmaddubsw_128(a, b, off))),
            ),
            X86Symbol::Psadbw => {
                let slot = self.b.create_stack_slot(16, 4);
                self.psadbw_128((a, b), (slot, 0));
                Some(self.ins().stack_addr(Type::I64, slot, 0))
            }
            X86Symbol::Psadbw256 => {
                if let Some(v) = self.wide_binary(TargetOpId::X64Psadbw256, a, b) {
                    return Some(v);
                }
                let slot = self.b.create_stack_slot(32, SCRATCH_ALIGN);
                self.psadbw_128((a, b), (slot, 0));
                self.psadbw_128((a, b), (slot, 16));
                Some(self.ins().stack_addr(Type::I64, slot, 0))
            }
            X86Symbol::Minss | X86Symbol::Maxss => Some(self.x86_scalar_fminmax(
                f32x4,
                x86 == X86Symbol::Maxss,
                [0, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                a,
                b,
            )),
            X86Symbol::Minsd | X86Symbol::Maxsd => Some(self.x86_scalar_fminmax(
                f64x2,
                x86 == X86Symbol::Maxsd,
                [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                a,
                b,
            )),
            X86Symbol::Pmulhrsw128 | X86Symbol::Pmulhrsw256 => {
                Some(self.x86_pmulhrsw(x86 == X86Symbol::Pmulhrsw256, a, b))
            }
            _ => None,
        }
    }

    fn x86_carry_op(&mut self, add: bool, carry: Value, a: Value, b: Value) -> Result<Value> {
        let (out, result) = if crate::compile_target().arch == Arch::X64 {
            let (sum, carried) = if add {
                (TargetOpId::X64AddCarry, TargetOpId::X64AddCarryOut)
            } else {
                (TargetOpId::X64SubBorrow, TargetOpId::X64SubBorrowOut)
            };
            let result = self.ins().target_op(sum, Type::I64, &[carry, a, b]);
            let out = self.ins().target_op(carried, Type::I8, &[carry, a, b]);
            (out, result)
        } else {
            let set = self.ins().icmp_imm(IntCC::NotEqual, carry, 0);
            let incoming = self.ins().uextend(Type::I128, set);
            let (wide_a, wide_b) = (
                self.ins().uextend(Type::I128, a),
                self.ins().uextend(Type::I128, b),
            );
            let sum = if add {
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
            (out, result)
        };
        let (byte, quad) = (
            self.cx.types.prim(Primitive::U8),
            self.cx.types.prim(Primitive::U64),
        );
        self.pair([byte, quad], [out, result])
    }

    fn x86_pclmulqdq(&mut self, out_bytes: u32, a: Value, b: Value, imm: i64) -> Value {
        let id = match imm & 0x11 {
            0x00 => TargetOpId::X64ClmulLowLow,
            0x01 => TargetOpId::X64ClmulHighLow,
            0x10 => TargetOpId::X64ClmulLowHigh,
            _ => TargetOpId::X64ClmulHighHigh,
        };
        let i64x2 = Type::Vec {
            lane: Lane::I64,
            count: 2,
        };
        self.emit_chunks(out_bytes, |lower, off| {
            let (x, y) = (
                lower.ins().load(i64x2, a, off),
                lower.ins().load(i64x2, b, off),
            );
            lower.ins().target_op(id, i64x2, &[x, y])
        })
    }

    fn x86_permd(&mut self, a: Value, index: Value) -> Value {
        let slot = self.b.create_stack_slot(32, SCRATCH_ALIGN);
        for lane in 0..8 {
            let at = self.ins().load(Type::I32, index, lane * 4);
            let seven = self.ins().iconst(Type::I32, 7);
            let at = self.ins().band(at, seven);
            let at = self.ins().uextend(Type::I64, at);
            let at = self.ins().imul_imm(at, 4);
            let from = self.ins().iadd(a, at);
            let word = self.ins().load(Type::I32, from, 0);
            self.ins().stack_store(word, slot, lane * 4);
        }
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    fn x86_pack_sat(
        &mut self,
        in_ty: Type,
        out_half: Type,
        signed: bool,
        wide: bool,
        a: Value,
        b: Value,
    ) -> Value {
        let (bytes, align) = if wide { (32, SCRATCH_ALIGN) } else { (16, 4) };
        let slot = self.b.create_stack_slot(bytes, align);
        self.pack_sat_half(in_ty, out_half, signed, (a, b), (slot, 0));
        if wide {
            self.pack_sat_half(in_ty, out_half, signed, (a, b), (slot, 16));
        }
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    fn x86_scalar_fminmax(
        &mut self,
        ty: Type,
        max: bool,
        mask: [u8; 16],
        a: Value,
        b: Value,
    ) -> Value {
        let va = self.ins().load(ty, a, 0);
        let mm = self.fminmax_128(ty, max, a, b, 0);
        let res = self.ins().shuffle(mm, va, mask);
        self.store_vec(res, 16)
    }

    fn x86_pmulhrsw(&mut self, wide: bool, a: Value, b: Value) -> Value {
        let slot = self
            .b
            .create_stack_slot(if wide { 32 } else { 16 }, SCRATCH_ALIGN);
        self.pmulhrsw_128(a, b, slot, 0);
        if wide {
            self.pmulhrsw_128(a, b, slot, 16);
        }
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    fn emit_256(&mut self, half: impl FnMut(&mut Self, i32) -> Value) -> Value {
        self.emit_chunks(32, half)
    }

    fn emit_chunks(&mut self, bytes: u32, mut chunk: impl FnMut(&mut Self, i32) -> Value) -> Value {
        let slot = self.b.create_stack_slot(bytes, SCRATCH_ALIGN);
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

    fn pack_sat_half(
        &mut self,
        in_ty: Type,
        out_half: Type,
        signed: bool,
        (a, b): (Value, Value),
        (slot, off): (StackSlot, i32),
    ) {
        let (va, vb) = (
            self.ins().load(in_ty, a, off),
            self.ins().load(in_ty, b, off),
        );
        let lo = self.ins().narrow_sat(out_half, signed, va);
        let hi = self.ins().narrow_sat(out_half, signed, vb);
        self.ins().stack_store(lo, slot, off);
        self.ins().stack_store(hi, slot, off + 8);
    }

    fn pmaddwd_128(&mut self, a: Value, b: Value, off: i32) -> Value {
        let i16x8 = Type::Vec {
            lane: Lane::I16,
            count: 8,
        };
        let i16x4 = Type::Vec {
            lane: Lane::I16,
            count: 4,
        };
        let i32x4 = Type::Vec {
            lane: Lane::I32,
            count: 4,
        };
        let even = [0, 2, 4, 6, 0, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0];
        let odd = [1, 3, 5, 7, 1, 3, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0];
        let (va, vb) = (
            self.ins().load(i16x8, a, off),
            self.ins().load(i16x8, b, off),
        );
        let (se_a, se_b) = (
            self.ins().shuffle(va, va, even),
            self.ins().shuffle(vb, vb, even),
        );
        let (so_a, so_b) = (
            self.ins().shuffle(va, va, odd),
            self.ins().shuffle(vb, vb, odd),
        );
        let (la, lb) = (self.low_8(i16x4, se_a), self.low_8(i16x4, se_b));
        let (ra, rb) = (self.low_8(i16x4, so_a), self.low_8(i16x4, so_b));
        let (ea, eb) = (self.ins().sextend(i32x4, la), self.ins().sextend(i32x4, lb));
        let (oa, ob) = (self.ins().sextend(i32x4, ra), self.ins().sextend(i32x4, rb));
        let pe = self.ins().imul(ea, eb);
        let po = self.ins().imul(oa, ob);
        self.ins().iadd(pe, po)
    }

    fn pmaddubsw_128(&mut self, a: Value, b: Value, off: i32) -> Value {
        let i8x16 = Type::Vec {
            lane: Lane::I8,
            count: 16,
        };
        let i8x8 = Type::Vec {
            lane: Lane::I8,
            count: 8,
        };
        let i16x8 = Type::Vec {
            lane: Lane::I16,
            count: 8,
        };
        let even = [0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14];
        let odd = [1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15];
        let (va, vb) = (
            self.ins().load(i8x16, a, off),
            self.ins().load(i8x16, b, off),
        );
        let (se_a, so_a) = (
            self.ins().shuffle(va, va, even),
            self.ins().shuffle(va, va, odd),
        );
        let (se_b, so_b) = (
            self.ins().shuffle(vb, vb, even),
            self.ins().shuffle(vb, vb, odd),
        );
        let (lea, loa) = (self.low_8(i8x8, se_a), self.low_8(i8x8, so_a));
        let (leb, lob) = (self.low_8(i8x8, se_b), self.low_8(i8x8, so_b));
        let (ea, oa) = (
            self.ins().uextend(i16x8, lea),
            self.ins().uextend(i16x8, loa),
        );
        let (eb, ob) = (
            self.ins().sextend(i16x8, leb),
            self.ins().sextend(i16x8, lob),
        );
        let pe = self.ins().imul(ea, eb);
        let po = self.ins().imul(oa, ob);
        self.ins().sadd_sat(pe, po)
    }

    fn psadbw_128(&mut self, (a, b): (Value, Value), (slot, off): (StackSlot, i32)) {
        let u8x16 = Type::Vec {
            lane: Lane::I8,
            count: 16,
        };
        let u8x8 = Type::Vec {
            lane: Lane::I8,
            count: 8,
        };
        let i16x8 = Type::Vec {
            lane: Lane::I16,
            count: 8,
        };
        let (va, vb) = (
            self.ins().load(u8x16, a, off),
            self.ins().load(u8x16, b, off),
        );
        let d1 = self.ins().usub_sat(va, vb);
        let d2 = self.ins().usub_sat(vb, va);
        let diff = self.ins().bor(d1, d2);
        let tmp = self.b.create_stack_slot(16, 4);
        self.ins().stack_store(diff, tmp, 0);
        let (lo8, hi8) = (
            self.ins().stack_load(u8x8, tmp, 0),
            self.ins().stack_load(u8x8, tmp, 8),
        );
        let (lo16, hi16) = (
            self.ins().uextend(i16x8, lo8),
            self.ins().uextend(i16x8, hi8),
        );
        let (red_lo, red_hi) = (
            self.ins().reduce(ReduceOp::Add, lo16),
            self.ins().reduce(ReduceOp::Add, hi16),
        );
        let (sum_lo, sum_hi) = (
            self.ins().uextend(Type::I64, red_lo),
            self.ins().uextend(Type::I64, red_hi),
        );
        self.ins().stack_store(sum_lo, slot, off);
        self.ins().stack_store(sum_hi, slot, off + 8);
    }

    fn fminmax_128(&mut self, ty: Type, max: bool, a: Value, b: Value, off: i32) -> Value {
        let (va, vb) = (self.ins().load(ty, a, off), self.ins().load(ty, b, off));
        if max {
            self.ins().fmax(va, vb)
        } else {
            self.ins().fmin(va, vb)
        }
    }

    fn vec_ptest(
        &mut self,
        ty: Type,
        wide: bool,
        mask_lane: Option<Lane>,
        mode: PtestMode,
        (a, b): (Value, Value),
    ) -> Value {
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
                let (r1, r2) = (
                    self.ins().reduce(ReduceOp::Or, ab),
                    self.ins().reduce(ReduceOp::Or, nab),
                );
                let (nz1, nz2) = (
                    self.ins().icmp_imm(IntCC::NotEqual, r1, 0),
                    self.ins().icmp_imm(IntCC::NotEqual, r2, 0),
                );
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
        let i16x4 = Type::Vec {
            lane: Lane::I16,
            count: 4,
        };
        let i32x4 = Type::Vec {
            lane: Lane::I32,
            count: 4,
        };
        let bias_s = self.ins().iconst(Type::I32, 0x4000);
        let sh_s = self.ins().iconst(Type::I32, 15);
        let (bias, sh) = (
            self.ins().splat(i32x4, bias_s),
            self.ins().splat(i32x4, sh_s),
        );
        for delta in [0, 8] {
            let (xa, xb) = (
                self.ins().load(i16x4, a, off + delta),
                self.ins().load(i16x4, b, off + delta),
            );
            let (ea, eb) = (self.ins().sextend(i32x4, xa), self.ins().sextend(i32x4, xb));
            let prod = self.ins().imul(ea, eb);
            let sum = self.ins().iadd(prod, bias);
            let shifted = self.ins().sshr(sum, sh);
            let out = self.ins().ireduce(i16x4, shifted);
            self.ins().stack_store(out, slot, off + delta);
        }
    }

    fn vec_shift_sat(
        &mut self,
        lane: Lane,
        count_lanes: u8,
        left: bool,
        signed: bool,
        wide: bool,
        a: Value,
        count: Value,
    ) -> Value {
        let vec_ty = Type::Vec {
            lane,
            count: count_lanes,
        };
        let cnt = self.ins().load(Type::I64, count, 0);
        let lane_bits = lane.bits();
        let lane_ty = lane.to_type();
        let oob = self
            .ins()
            .icmp_imm(IntCC::UnsignedGreaterThanOrEqual, cnt, i64::from(lane_bits));
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

fn pack_spec(x86: X86Symbol) -> Option<(Type, Type, bool, bool)> {
    let i8x8 = Type::Vec {
        lane: Lane::I8,
        count: 8,
    };
    let i16x8 = Type::Vec {
        lane: Lane::I16,
        count: 8,
    };
    let i16x4 = Type::Vec {
        lane: Lane::I16,
        count: 4,
    };
    let i32x4 = Type::Vec {
        lane: Lane::I32,
        count: 4,
    };
    Some(match x86 {
        X86Symbol::Packsswb => (i16x8, i8x8, true, false),
        X86Symbol::Packuswb => (i16x8, i8x8, false, false),
        X86Symbol::Packsswb256 => (i16x8, i8x8, true, true),
        X86Symbol::Packuswb256 => (i16x8, i8x8, false, true),
        X86Symbol::Packssdw => (i32x4, i16x4, true, false),
        X86Symbol::Packusdw => (i32x4, i16x4, false, false),
        X86Symbol::Packssdw256 => (i32x4, i16x4, true, true),
        X86Symbol::Packusdw256 => (i32x4, i16x4, false, true),
        _ => return None,
    })
}

fn fminmax_spec(x86: X86Symbol) -> Option<(Type, bool, bool)> {
    let f32x4 = Type::Vec {
        lane: Lane::F32,
        count: 4,
    };
    let f64x2 = Type::Vec {
        lane: Lane::F64,
        count: 2,
    };
    Some(match x86 {
        X86Symbol::Minps => (f32x4, false, false),
        X86Symbol::Maxps => (f32x4, true, false),
        X86Symbol::Minpd => (f64x2, false, false),
        X86Symbol::Maxpd => (f64x2, true, false),
        X86Symbol::Minps256 => (f32x4, false, true),
        X86Symbol::Maxps256 => (f32x4, true, true),
        X86Symbol::Minpd256 => (f64x2, false, true),
        X86Symbol::Maxpd256 => (f64x2, true, true),
        _ => return None,
    })
}

fn shift_spec(x86: X86Symbol) -> Option<(Lane, u8, bool, bool, bool)> {
    Some(match x86 {
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
        X86Symbol::Psrad256 => (Lane::I32, 4, false, true, true),
        _ => return None,
    })
}

fn psign_spec(x86: X86Symbol) -> Option<(Type, bool)> {
    let i8x16 = Type::Vec {
        lane: Lane::I8,
        count: 16,
    };
    let i16x8 = Type::Vec {
        lane: Lane::I16,
        count: 8,
    };
    let i32x4 = Type::Vec {
        lane: Lane::I32,
        count: 4,
    };
    Some(match x86 {
        X86Symbol::Psignb128 => (i8x16, false),
        X86Symbol::Psignw128 => (i16x8, false),
        X86Symbol::Psignd128 => (i32x4, false),
        X86Symbol::Psignb256 => (i8x16, true),
        X86Symbol::Psignw256 => (i16x8, true),
        X86Symbol::Psignd256 => (i32x4, true),
        _ => return None,
    })
}

fn ptest_spec(x86: X86Symbol) -> Option<(Type, bool, Option<Lane>, PtestMode)> {
    let i32x4 = Type::Vec {
        lane: Lane::I32,
        count: 4,
    };
    let i64x2 = Type::Vec {
        lane: Lane::I64,
        count: 2,
    };
    Some(match x86 {
        X86Symbol::Ptestnzc => (i64x2, false, None, PtestMode::Nzc),
        X86Symbol::Ptestnzc256 => (i64x2, true, None, PtestMode::Nzc),
        X86Symbol::Vtestnzcps => (i32x4, false, Some(Lane::I32), PtestMode::Nzc),
        X86Symbol::Vtestnzcpd => (i64x2, false, Some(Lane::I64), PtestMode::Nzc),
        X86Symbol::Vtestzps256 => (i32x4, true, Some(Lane::I32), PtestMode::Z),
        X86Symbol::Vtestcps256 => (i32x4, true, Some(Lane::I32), PtestMode::C),
        X86Symbol::Vtestnzcps256 => (i32x4, true, Some(Lane::I32), PtestMode::Nzc),
        X86Symbol::Vtestzpd256 => (i64x2, true, Some(Lane::I64), PtestMode::Z),
        X86Symbol::Vtestcpd256 => (i64x2, true, Some(Lane::I64), PtestMode::C),
        X86Symbol::Vtestnzcpd256 => (i64x2, true, Some(Lane::I64), PtestMode::Nzc),
        _ => return None,
    })
}
