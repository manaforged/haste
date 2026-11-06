use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlagsData, Signature, StackSlotData, StackSlotKind, Type, Value, types};
use hir_def::AdtId;
use hir_ty::next_solver::{Ty, TyKind};
use rustc_type_ir::inherent::IntoKind;

use super::{BinOpKind, MemKind};
use crate::lower::{Lower, Place};

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn offset_from(&mut self, ty: Ty<'db>, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let size = self.layout(ty)?.size.bytes().max(1) as i64;
        let left = self.load(types::I64, args[0].addr, 0);
        let right = self.load(types::I64, args[1].addr, 0);
        let bytes = self.builder.ins().isub(left, right);
        let size = self.builder.ins().iconst(types::I64, size);
        let value = self.builder.ins().sdiv(bytes, size);
        self.store(value, dest.addr, 0);
        Ok(())
    }

    pub(super) fn c_call(&mut self, code: usize, params: &[Type], ret: Type, values: &[Value]) -> Value {
        let mut signature = Signature::new(self.call_conv);
        signature.params.extend(params.iter().map(|ty| AbiParam::new(*ty)));
        signature.returns.push(AbiParam::new(ret));
        let signature = self.builder.import_signature(signature);
        let code = self.builder.ins().iconst(types::I64, code as i64);
        let call = self.builder.ins().call_indirect(signature, code, values);
        self.builder.inst_results(call)[0]
    }

    pub(super) fn mem(
        &mut self,
        code: usize,
        kind: MemKind,
        generic: Option<Ty<'db>>,
        dest: Place<'db>,
        args: &[Place<'db>],
    ) -> Result<(), String> {
        let width = match (kind, generic) {
            (MemKind::Compare, _) | (_, None) => 1,
            (_, Some(ty)) => self.layout(ty)?.size.bytes() as i64,
        };
        let count = self.load(types::I64, args[2].addr, 0);
        let bytes = self.builder.ins().imul_imm(count, width);
        match kind {
            MemKind::Move => {
                let from = self.load(types::I64, args[0].addr, 0);
                let to = self.load(types::I64, args[1].addr, 0);
                self.c_call(code, &[types::I64; 3], types::I64, &[to, from, bytes]);
            }
            MemKind::Fill => {
                let to = self.load(types::I64, args[0].addr, 0);
                let value = self.load(types::I8, args[1].addr, 0);
                let value = self.builder.ins().uextend(types::I32, value);
                self.c_call(code, &[types::I64, types::I32, types::I64], types::I64, &[to, value, bytes]);
            }
            MemKind::Compare => {
                let left = self.load(types::I64, args[0].addr, 0);
                let right = self.load(types::I64, args[1].addr, 0);
                let value = self.c_call(code, &[types::I64; 3], types::I32, &[left, right, bytes]);
                self.store(value, dest.addr, 0);
            }
        }
        Ok(())
    }

    pub(super) fn raw_eq(&mut self, ty: Ty<'db>, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let size = self.layout(ty)?.size.bytes() as i64;
        let left = self.load(types::I64, args[0].addr, 0);
        let right = self.load(types::I64, args[1].addr, 0);
        let size = self.builder.ins().iconst(types::I64, size);
        let order = self.c_call(super::memcmp as *const () as usize, &[types::I64; 3], types::I32, &[left, right, size]);
        let value = self.builder.ins().icmp_imm_u(IntCC::Equal, order, 0);
        self.store(value, dest.addr, 0);
        Ok(())
    }

    pub(super) fn compare3(&mut self, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let scalar = self.scalar(args[0].ty.ty)?;
        let left = self.load(scalar.ty, args[0].addr, 0);
        let right = self.load(scalar.ty, args[1].addr, 0);
        let (above, below) = if scalar.signed {
            (IntCC::SignedGreaterThan, IntCC::SignedLessThan)
        } else {
            (IntCC::UnsignedGreaterThan, IntCC::UnsignedLessThan)
        };
        let up = self.builder.ins().icmp(above, left, right);
        let down = self.builder.ins().icmp(below, left, right);
        let value = self.builder.ins().isub(up, down);
        self.store(value, dest.addr, 0);
        Ok(())
    }

    pub(super) fn atomic_exchange(&mut self, ty: Ty<'db>, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let scalar = self.scalar(ty)?;
        let TyKind::Adt(adt, _) = dest.ty.ty.kind() else {
            return Err("compare exchange without a Result".to_owned());
        };
        let AdtId::EnumId(result) = adt.def_id() else {
            return Err("compare exchange without a Result".to_owned());
        };
        let variants = &result.enum_variants(self.db).variants;
        let (ok, err) = (variants[0].0, variants[1].0);
        let pointer = self.load(types::I64, args[0].addr, 0);
        let expected = self.load(scalar.ty, args[1].addr, 0);
        let new = self.load(scalar.ty, args[2].addr, 0);
        let old = self.builder.ins().atomic_cas(MemFlagsData::trusted(), pointer, expected, new);
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 16, 4));
        let holder = self.builder.ins().stack_addr(types::I64, slot, 0);
        self.store(old, holder, 0);
        let value = Place { addr: holder, ty: hir_ty::mir::PlaceTy::from_ty(ty), meta: None };
        let same = self.builder.ins().icmp(IntCC::Equal, old, expected);
        let (hit, miss, done) = (self.builder.create_block(), self.builder.create_block(), self.builder.create_block());
        self.builder.ins().brif(same, hit, &[], miss, &[]);
        self.builder.switch_to_block(hit);
        self.write_variant(dest, ok.into(), &[value])?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(miss);
        self.write_variant(dest, err.into(), &[value])?;
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(())
    }

    pub(super) fn overflow(&mut self, kind: BinOpKind, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let scalar = self.scalar(args[0].ty.ty)?;
        let ty = scalar.ty;
        if ty == types::I128 {
            return Err("128-bit overflow check".to_owned());
        }
        let a = self.load(ty, args[0].addr, 0);
        let b = self.load(ty, args[1].addr, 0);
        let (result, flag) = match kind {
            BinOpKind::Add | BinOpKind::Sub => self.add_sub_overflow(matches!(kind, BinOpKind::Add), a, b, scalar.signed),
            BinOpKind::Mul => self.mul_overflow(a, b, ty, scalar.signed),
            _ => return Err("overflow check of this operator".to_owned()),
        };
        let layout = self.layout(dest.ty.ty)?;
        self.store(result, dest.addr, layout.fields.offset(0).bytes() as i32);
        self.store(flag, dest.addr, layout.fields.offset(1).bytes() as i32);
        Ok(())
    }

    pub(super) fn add_sub_overflow(&mut self, add: bool, a: Value, b: Value, signed: bool) -> (Value, Value) {
        let result = if add { self.builder.ins().iadd(a, b) } else { self.builder.ins().isub(a, b) };
        if !signed {
            let flag = if add {
                self.builder.ins().icmp(IntCC::UnsignedLessThan, result, a)
            } else {
                self.builder.ins().icmp(IntCC::UnsignedLessThan, a, b)
            };
            return (result, flag);
        }
        let left = if add { self.builder.ins().bxor(a, result) } else { self.builder.ins().bxor(a, b) };
        let right = if add { self.builder.ins().bxor(b, result) } else { self.builder.ins().bxor(a, result) };
        let mix = self.builder.ins().band(left, right);
        (result, self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, mix, 0))
    }

    pub(super) fn mul_overflow(&mut self, a: Value, b: Value, ty: Type, signed: bool) -> (Value, Value) {
        let result = self.builder.ins().imul(a, b);
        let ins = self.builder.ins();
        if ty == types::I64 {
            let high = if signed { ins.smulhi(a, b) } else { ins.umulhi(a, b) };
            let expected = if signed {
                self.builder.ins().sshr_imm(result, 63)
            } else {
                self.builder.ins().iconst(types::I64, 0)
            };
            return (result, self.builder.ins().icmp(IntCC::NotEqual, high, expected));
        }
        let wide_a = if signed { ins.sextend(types::I64, a) } else { ins.uextend(types::I64, a) };
        let ins = self.builder.ins();
        let wide_b = if signed { ins.sextend(types::I64, b) } else { ins.uextend(types::I64, b) };
        let product = self.builder.ins().imul(wide_a, wide_b);
        let ins = self.builder.ins();
        let back = if signed { ins.sextend(types::I64, result) } else { ins.uextend(types::I64, result) };
        (result, self.builder.ins().icmp(IntCC::NotEqual, product, back))
    }

    pub(super) fn saturating(&mut self, add: bool, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let scalar = self.scalar(args[0].ty.ty)?;
        let ty = scalar.ty;
        let a = self.load(ty, args[0].addr, 0);
        let b = self.load(ty, args[1].addr, 0);
        let ins = self.builder.ins();
        let result = if add { ins.iadd(a, b) } else { ins.isub(a, b) };
        let bits = ty.bits();
        let value = if scalar.signed {
            let ins = self.builder.ins();
            let mix = if add {
                let left = ins.bxor(a, result);
                let right = self.builder.ins().bxor(b, result);
                self.builder.ins().band(left, right)
            } else {
                let left = ins.bxor(a, b);
                let right = self.builder.ins().bxor(a, result);
                self.builder.ins().band(left, right)
            };
            let overflow = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, mix, 0);
            let negative = self.builder.ins().icmp_imm_s(IntCC::SignedLessThan, a, 0);
            let min = self.builder.ins().iconst(ty, (1_u64 << (bits - 1)) as i64);
            let max_bits = if bits == 64 { i64::MAX } else { (1_i64 << (bits - 1)) - 1 };
            let max = self.builder.ins().iconst(ty, max_bits);
            let bound = self.builder.ins().select(negative, min, max);
            self.builder.ins().select(overflow, bound, result)
        } else if add {
            let overflow = self.builder.ins().icmp(IntCC::UnsignedLessThan, result, a);
            let all = if bits == 64 { -1_i64 } else { (1_i64 << bits) - 1 };
            let max = self.builder.ins().iconst(ty, all);
            self.builder.ins().select(overflow, max, result)
        } else {
            let overflow = self.builder.ins().icmp(IntCC::UnsignedLessThan, a, b);
            let zero = self.builder.ins().iconst(ty, 0);
            self.builder.ins().select(overflow, zero, result)
        };
        self.store(value, dest.addr, 0);
        Ok(())
    }
}
