use anyhow::{Result, bail};
use haste_codegen::{Arch, IntCC, Lane, Target, TargetOpId, Type, Value};

use crate::host::{HostSymbol, X86Symbol};

use super::Lower;

impl Lower<'_, '_> {
    pub(in crate::lower) fn shuffle_dyn(&mut self, args: &[Value], tbl: bool) -> Result<Value> {
        let &[table, index] = args else { bail!("byte shuffle takes two vectors") };
        let arch = Target::host().arch;
        if (tbl && arch == Arch::Arm64) || (!tbl && arch == Arch::X64) {
            let ty = Type::Vec { lane: Lane::I8, count: 16 };
            let (t, idx) = (self.ins().load(ty, table, 0), self.ins().load(ty, index, 0));
            let res = self.ins().shuffle_dyn(t, t, idx);
            return Ok(self.store_vec(res, 16));
        }
        let slot = self.b.create_stack_slot(16, 4);
        let (mask, zero) = (self.ins().iconst(Type::I8, 15), self.ins().iconst(Type::I8, 0));
        for lane in 0..16 {
            let at = self.ins().load(Type::I8, index, lane);
            let keep = if tbl {
                self.ins().icmp_imm(IntCC::UnsignedLessThan, at, 16)
            } else {
                self.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, at, 0)
            };
            let masked = self.ins().band(at, mask);
            let pick = self.ins().uextend(Type::I64, masked);
            let source = self.ins().iadd(table, pick);
            let byte = self.ins().load(Type::I8, source, 0);
            let value = self.ins().select(keep, byte, zero);
            self.ins().stack_store(value, slot, lane);
        }
        Ok(self.ins().stack_addr(Type::I64, slot, 0))
    }

    pub(in crate::lower) fn byte_permute_32(&mut self, args: &[Value]) -> Result<Value> {
        let &[table, index] = args else { bail!("`pshufb` takes two vectors") };
        let slot = self.b.create_stack_slot(32, 5);
        let (low, zero) = (self.ins().iconst(Type::I8, 15), self.ins().iconst(Type::I8, 0));
        for lane in 0..32 {
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

    pub(in crate::lower) fn target_vector(&mut self, id: TargetOpId, ty: Type, args: &[Value]) -> Result<Value> {
        let arch = Target::host().arch;
        if is_arm64_target_op(id) && arch != Arch::Arm64 {
            return match (id, args) {
                (TargetOpId::Arm64Umaxp, &[x, y]) => Ok(self.byte_pair_max_fallback(x, y)),
                (TargetOpId::Arm64Sqdmulh, &[x, y]) => {
                    let Some((lane, count)) = ty.vec_shape() else { bail!("bad sqdmulh type") };
                    Ok(self.sqdmulh_scalar(i32::from(count), lane.to_type(), (x, y)))
                }
                _ => bail!("ARM64 vector intrinsic {id:?} is only supported on Arm64"),
            };
        }
        if is_x64_target_op(id) && arch != Arch::X64 {
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

    fn byte_pair_max_fallback(&mut self, first: Value, second: Value) -> Value {
        let slot = self.b.create_stack_slot(16, 4);
        for lane in 0..16 {
            let from = if lane < 8 { first } else { second };
            let at = lane % 8 * 2;
            let (x, y) = (self.ins().load(Type::I8, from, at), self.ins().load(Type::I8, from, at + 1));
            let wins = self.ins().icmp(IntCC::UnsignedGreaterThan, x, y);
            let value = self.ins().select(wins, x, y);
            self.ins().stack_store(value, slot, lane);
        }
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    fn sqdmulh_scalar(&mut self, lanes: i32, lane_ty: Type, (x, y): (Value, Value)) -> Value {
        let bytes = lanes * lane_ty.bytes() as i32;
        let slot = self.b.create_stack_slot(bytes as u32, if bytes == 8 { 3 } else { 4 });
        let (wide_ty, shift, min_val, max_val) = if lane_ty == Type::I32 {
            (Type::I64, 32i64, i64::from(i32::MIN), i64::from(i32::MAX))
        } else {
            (Type::I32, 16i64, i64::from(i16::MIN), i64::from(i16::MAX))
        };
        let (one, sh, max_c) = (self.ins().iconst(wide_ty, 1), self.ins().iconst(wide_ty, shift), self.ins().iconst(lane_ty, max_val));
        for lane in 0..lanes {
            let off = lane * lane_ty.bytes() as i32;
            let (a, b) = (self.ins().load(lane_ty, x, off), self.ins().load(lane_ty, y, off));
            let (wa, wb) = (self.ins().sextend(wide_ty, a), self.ins().sextend(wide_ty, b));
            let prod = self.ins().imul(wa, wb);
            let doubled = self.ins().ishl(prod, one);
            let shifted = self.ins().sshr(doubled, sh);
            let narrow = self.ins().ireduce(lane_ty, shifted);
            let eq_a = self.ins().icmp_imm(IntCC::Equal, a, min_val);
            let eq_b = self.ins().icmp_imm(IntCC::Equal, b, min_val);
            let both_min = self.ins().band(eq_a, eq_b);
            let val = self.ins().select(both_min, max_c, narrow);
            self.ins().stack_store(val, slot, off);
        }
        self.ins().stack_addr(Type::I64, slot, 0)
    }
}

fn is_arm64_target_op(id: TargetOpId) -> bool {
    matches!(
        id,
        TargetOpId::Arm64Umaxp
            | TargetOpId::Arm64Sqdmulh
            | TargetOpId::Arm64Sha256H
            | TargetOpId::Arm64Sha256H2
            | TargetOpId::Arm64Sha256Su0
            | TargetOpId::Arm64Sha256Su1
            | TargetOpId::Arm64AesE
            | TargetOpId::Arm64AesD
            | TargetOpId::Arm64AesMc
            | TargetOpId::Arm64AesImc
    )
}

fn is_x64_target_op(id: TargetOpId) -> bool {
    !is_arm64_target_op(id)
}

fn x86_target_op(x86: X86Symbol) -> Option<(TargetOpId, Type)> {
    let i8x16 = Type::Vec { lane: Lane::I8, count: 16 };
    let i16x8 = Type::Vec { lane: Lane::I16, count: 8 };
    let i32x4 = Type::Vec { lane: Lane::I32, count: 4 };
    let i64x2 = Type::Vec { lane: Lane::I64, count: 2 };
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
        X86Symbol::Pmaddwd => (TargetOpId::X64Pmaddwd, i32x4),
        X86Symbol::Pmaddubsw128 => (TargetOpId::X64Pmaddubsw, i16x8),
        X86Symbol::Psadbw => (TargetOpId::X64Psadbw, i64x2),
        X86Symbol::Packsswb => (TargetOpId::X64Packsswb, i8x16),
        X86Symbol::Packssdw => (TargetOpId::X64Packssdw, i16x8),
        X86Symbol::Packuswb => (TargetOpId::X64Packuswb, i8x16),
        X86Symbol::Packusdw => (TargetOpId::X64Packusdw, i16x8),
        _ => return None,
    })
}
