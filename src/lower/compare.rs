use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, InstBuilder, Signature, Value, types};
use hir_ty::mir::{BinOp, Operand};
use hir_ty::next_solver::{Ty, TyKind};
use rustc_type_ir::inherent::IntoKind;

use super::{Lower, Place, intrinsic, pointee};

const WORD: u64 = 8;
const FAT: u64 = 16;

struct Side<'db> {
    addr: Value,
    ty: Ty<'db>,
    meta: Option<Value>,
}

fn is_compare(op: &BinOp) -> bool {
    matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)
}

fn strict(op: &BinOp, signed: bool) -> IntCC {
    match (op, signed) {
        (BinOp::Lt | BinOp::Le, false) => IntCC::UnsignedLessThan,
        (BinOp::Gt | BinOp::Ge, false) => IntCC::UnsignedGreaterThan,
        (BinOp::Lt | BinOp::Le, true) => IntCC::SignedLessThan,
        _ => IntCC::SignedGreaterThan,
    }
}

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn binary_rvalue(&mut self, op: &BinOp, left: &Operand, right: &Operand) -> Result<Value, String> {
        if !is_compare(op) {
            let (left, scalar) = self.read(left)?;
            let (right, _) = self.read(right)?;
            return self.binary(op, left, right, scalar);
        }
        let (left, right) = (self.peeled(left)?, self.peeled(right)?);
        if matches!(left.ty.kind(), TyKind::Str) {
            return self.str_equal(op, &left, &right);
        }
        if matches!(left.ty.kind(), TyKind::RawPtr(..)) && self.layout(left.ty)?.size.bytes() == FAT {
            return Ok(self.fat_compare(op, &left, &right));
        }
        if let Ok(scalar) = self.scalar(left.ty) {
            let (a, b) = (self.load(scalar.ty, left.addr, 0), self.load(scalar.ty, right.addr, 0));
            return self.binary(op, a, b, scalar);
        }
        self.structural_equal(op, &left, &right)
    }

    fn structural_equal(&mut self, op: &BinOp, left: &Side<'db>, right: &Side<'db>) -> Result<Value, String> {
        let (a, _) = self.size_align(left.ty, left.meta)?;
        let (b, _) = self.size_align(right.ty, right.meta)?;
        let size = Side { addr: a, ty: left.ty, meta: Some(a) };
        let other = Side { addr: b, ty: right.ty, meta: Some(b) };
        self.bytes_equal(op, (left.addr, &size), (right.addr, &other))
    }

    pub(super) fn pointer_order(&mut self, op: &BinOp, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let ty = pointee(args[0].ty.ty)?;
        let left = Side { addr: self.load(types::I64, args[0].addr, 0), ty, meta: None };
        let right = Side { addr: self.load(types::I64, args[1].addr, 0), ty, meta: None };
        let value = if self.layout(ty)?.size.bytes() == FAT {
            self.fat_compare(op, &left, &right)
        } else {
            let scalar = self.scalar(ty)?;
            let (a, b) = (self.load(scalar.ty, left.addr, 0), self.load(scalar.ty, right.addr, 0));
            self.binary(op, a, b, scalar)?
        };
        self.store(value, dest.addr, 0);
        Ok(())
    }

    fn peeled(&mut self, op: &Operand) -> Result<Side<'db>, String> {
        let place = self.operand(op)?;
        let mut side = Side { addr: place.addr, ty: place.ty.ty, meta: None };
        while let TyKind::Ref(_, inner, _) = side.ty.kind() {
            side.meta = self.layout(inner)?.is_unsized().then(|| self.load(types::I64, side.addr, WORD as i32));
            side.addr = self.load(types::I64, side.addr, 0);
            side.ty = inner;
        }
        Ok(side)
    }

    fn str_equal(&mut self, op: &BinOp, left: &Side<'db>, right: &Side<'db>) -> Result<Value, String> {
        self.bytes_equal(op, (left.addr, left), (right.addr, right))
    }

    fn bytes_equal(&mut self, op: &BinOp, left: (Value, &Side<'db>), right: (Value, &Side<'db>)) -> Result<Value, String> {
        let (a, b) = (left.1.meta.ok_or("compare without a length")?, right.1.meta.ok_or("compare without a length")?);
        let same = self.builder.create_block();
        let done = self.builder.create_block();
        self.builder.append_block_param(done, types::I8);
        let hit = self.builder.ins().icmp(IntCC::Equal, a, b);
        let no = self.builder.ins().iconst(types::I8, 0);
        self.builder.ins().brif(hit, same, &[], done, &[no.into()]);
        self.builder.switch_to_block(same);
        let mut signature = Signature::new(self.call_conv);
        signature.params.extend([AbiParam::new(types::I64); 3]);
        signature.returns.push(AbiParam::new(types::I32));
        let signature = self.builder.import_signature(signature);
        let code = self.builder.ins().iconst(types::I64, intrinsic::compare_bytes_code() as i64);
        let call = self.builder.ins().call_indirect(signature, code, &[left.0, right.0, a]);
        let order = self.builder.inst_results(call)[0];
        let equal = self.builder.ins().icmp_imm(IntCC::Equal, order, 0);
        self.builder.ins().jump(done, &[equal.into()]);
        self.builder.switch_to_block(done);
        let equal = self.builder.block_params(done)[0];
        match op {
            BinOp::Eq => Ok(equal),
            BinOp::Ne => Ok(self.builder.ins().bxor_imm(equal, 1)),
            other => Err(format!("{other:?} on bytes")),
        }
    }

    fn fat_compare(&mut self, op: &BinOp, left: &Side<'db>, right: &Side<'db>) -> Value {
        let (a0, a1) = (self.load(types::I64, left.addr, 0), self.load(types::I64, left.addr, WORD as i32));
        let (b0, b1) = (self.load(types::I64, right.addr, 0), self.load(types::I64, right.addr, WORD as i32));
        let first = self.builder.ins().icmp(IntCC::Equal, a0, b0);
        let second = self.builder.ins().icmp(IntCC::Equal, a1, b1);
        match op {
            BinOp::Eq => self.builder.ins().band(first, second),
            BinOp::Ne => {
                let both = self.builder.ins().band(first, second);
                self.builder.ins().bxor_imm(both, 1)
            }
            other => {
                let outer = self.builder.ins().icmp(strict(other, false), a0, b0);
                let inner = match other {
                    BinOp::Le => self.builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, a1, b1),
                    BinOp::Ge => self.builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, a1, b1),
                    _ => self.builder.ins().icmp(strict(other, false), a1, b1),
                };
                self.builder.ins().select(first, inner, outer)
            }
        }
    }
}
