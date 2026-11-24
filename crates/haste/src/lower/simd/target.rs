#[path = "target_x86.rs"]
mod x86;

use anyhow::{Result, bail};
use haste_codegen::{Arch, IntCC, Lane, TargetOpId, Type, Value};

use crate::host::{HostSymbol, X86Symbol};

use super::super::Lower;
use super::SCRATCH_ALIGN;
use super::wide::wide_quads;

impl Lower<'_, '_> {
    pub(in crate::lower) fn shuffle_dyn(&mut self, args: &[Value], tbl: bool) -> Result<Value> {
        let &[table, index] = args else {
            bail!("byte shuffle takes two vectors");
        };
        let res = self.shuffle_dyn_16(table, index, tbl, 0);
        Ok(self.store_vec(res, 16))
    }

    fn shuffle_dyn_16(&mut self, table: Value, index: Value, tbl: bool, off: i32) -> Value {
        let ty = Type::Vec {
            lane: Lane::I8,
            count: 16,
        };
        let (t, idx) = (
            self.ins().load(ty, table, off),
            self.ins().load(ty, index, off),
        );
        let arch = crate::compile_target().arch;
        if (tbl && arch == Arch::Arm64) || (!tbl && arch == Arch::X64) {
            return self.ins().shuffle_dyn(t, t, idx);
        }
        let adj = if tbl {
            let sixteen_s = self.ins().iconst(Type::I8, 16);
            let sixteen = self.ins().splat(ty, sixteen_s);
            let oob = self
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, idx, sixteen);
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
        let &[table, index] = args else {
            bail!("`pshufb` takes two vectors");
        };
        if let Some(value) = self.wide_binary(TargetOpId::X64Pshufb256, table, index) {
            return Ok(value);
        }
        let slot = self.b.create_stack_slot(32, SCRATCH_ALIGN);
        let lo = self.shuffle_dyn_16(table, index, false, 0);
        let hi = self.shuffle_dyn_16(table, index, false, 16);
        self.ins().stack_store(lo, slot, 0);
        self.ins().stack_store(hi, slot, 16);
        Ok(self.ins().stack_addr(Type::I64, slot, 0))
    }

    pub(in crate::lower) fn avg_round_vec(&mut self, ty: Type, args: &[Value]) -> Result<Value> {
        let &[x, y] = args else {
            bail!("`avg_round` takes two vectors");
        };
        let (va, vb) = (self.ins().load(ty, x, 0), self.ins().load(ty, y, 0));
        let out = self.ins().avg_round(va, vb);
        Ok(self.store_vec(out, ty.bytes()))
    }

    pub(in crate::lower) fn umaxp_vec(&mut self, args: &[Value]) -> Result<Value> {
        let &[x, y] = args else {
            bail!("`umaxp` takes two vectors");
        };
        let ty = Type::Vec {
            lane: Lane::I8,
            count: 16,
        };
        let (va, vb) = (self.ins().load(ty, x, 0), self.ins().load(ty, y, 0));
        let even = self.ins().shuffle(
            va,
            vb,
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        );
        let odd = self.ins().shuffle(
            va,
            vb,
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
        );
        let out = self.ins().umax(even, odd);
        Ok(self.store_vec(out, 16))
    }

    pub(in crate::lower) fn sqdmulh_vec(&mut self, ty: Type, args: &[Value]) -> Result<Value> {
        let &[x, y] = args else {
            bail!("`sqdmulh` takes two vectors");
        };
        let i32x2 = Type::Vec {
            lane: Lane::I32,
            count: 2,
        };
        let i64x2 = Type::Vec {
            lane: Lane::I64,
            count: 2,
        };
        let i16x4 = Type::Vec {
            lane: Lane::I16,
            count: 4,
        };
        let i32x4 = Type::Vec {
            lane: Lane::I32,
            count: 4,
        };
        let i16x8 = Type::Vec {
            lane: Lane::I16,
            count: 8,
        };
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

    fn sqdmulh_half(
        &mut self,
        half_ty: Type,
        wide_ty: Type,
        shift_bits: i64,
        (x, y): (Value, Value),
        off: i32,
    ) -> Value {
        let (va, vb) = (
            self.ins().load(half_ty, x, off),
            self.ins().load(half_ty, y, off),
        );
        let (wa, wb) = (
            self.ins().sextend(wide_ty, va),
            self.ins().sextend(wide_ty, vb),
        );
        let prod = self.ins().imul(wa, wb);
        let wide_lane = wide_ty.lane().map_or(Type::I64, Lane::to_type);
        let sh_s = self.ins().iconst(wide_lane, shift_bits);
        let sh = self.ins().splat(wide_ty, sh_s);
        let shifted = self.ins().sshr(prod, sh);
        self.ins().narrow_sat(half_ty, true, shifted)
    }

    pub(in crate::lower) fn target_vector(
        &mut self,
        id: TargetOpId,
        ty: Type,
        args: &[Value],
    ) -> Result<Value> {
        let arch = crate::compile_target().arch;
        if is_arm64_target_op(id) && arch != Arch::Arm64 {
            bail!("ARM64 vector intrinsic {id:?} is only supported on Arm64");
        }
        if !is_arm64_target_op(id) && arch != Arch::X64 {
            bail!("x86 vector intrinsic {id:?} is only supported on X64");
        }
        if !(1..=3).contains(&args.len()) {
            bail!(
                "target vector op {id:?} expects 1 to 3 arguments, got {}",
                args.len()
            );
        }
        let loaded: Vec<Value> = args
            .iter()
            .map(|&arg| self.ins().load(ty, arg, 0))
            .collect();
        let vec = self.ins().target_op(id, ty, &loaded);
        Ok(self.store_vec(vec, ty.bytes()))
    }

    pub(in crate::lower) fn host_vector(
        &mut self,
        x86: X86Symbol,
        args: &[Value],
    ) -> Result<Value> {
        if let Some(value) = self.x86_inline(x86, args)? {
            return Ok(
                match wide_quads()
                    .filter(|wide| x86.out_bytes() == wide.bytes() && x86.scalar_bytes().is_none())
                {
                    Some(wide) => self.join_wide(wide, value),
                    None => value,
                },
            );
        }
        if crate::compile_target().arch == Arch::X64
            && let Some((id, ty)) = x86_target_op(x86)
            && (1..=3).contains(&args.len())
        {
            let loaded: Vec<Value> = args
                .iter()
                .map(|&arg| self.ins().load(ty, arg, 0))
                .collect();
            let vec = self.ins().target_op(id, ty, &loaded);
            return Ok(self.store_vec(vec, ty.bytes()));
        }
        if x86.host_addr().is_none() {
            bail!("{} has no host function on this machine", x86.link_name());
        }
        let slot = self.b.create_stack_slot(x86.out_bytes(), SCRATCH_ALIGN);
        let out = self.ins().stack_addr(Type::I64, slot, 0);
        let values: Vec<Value> = std::iter::once(out).chain(args.iter().copied()).collect();
        self.call_host(HostSymbol::X86(x86), &values, None);
        Ok(match x86.scalar_bytes() {
            Some(8) => self.ins().load(Type::I64, out, 0),
            Some(_) => self.ins().load(Type::I32, out, 0),
            None => out,
        })
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
    let i8x16 = Type::Vec {
        lane: Lane::I8,
        count: 16,
    };
    let i32x4 = Type::Vec {
        lane: Lane::I32,
        count: 4,
    };
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
