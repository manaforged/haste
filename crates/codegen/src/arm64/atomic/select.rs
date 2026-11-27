use super::super::asm::{Asm, Cond, Reg, Size, ZR};
use super::super::{A, C, D, Emit, V0, V1};
use crate::ir::{Binary, IntCC, Op, Type, Unary, Value};

#[derive(Clone, Copy)]
enum SelectOp {
    Inc,
    Inv,
    Neg,
}

impl Emit<'_> {
    fn select_transform(&self, ty: Type, alt: Value, base: Value) -> Option<SelectOp> {
        if !matches!(ty, Type::I32 | Type::I64) {
            return None;
        }
        let op = &self.func.insts[self.defs[alt.index()]?.index()].op;
        let is_one = |v: Value| {
            self.plan
                .const_of(v)
                .is_some_and(|c| crate::fold::extend(c, ty, false) == 1)
        };
        match *op {
            Op::Unary(Unary::Bnot, inner) if inner == base => Some(SelectOp::Inv),
            Op::Unary(Unary::Ineg, inner) if inner == base => Some(SelectOp::Neg),
            Op::Binary(Binary::Iadd, a, b)
                if (a == base && is_one(b)) || (b == base && is_one(a)) =>
            {
                Some(SelectOp::Inc)
            }
            Op::Binary(Binary::Isub, z, inner)
                if inner == base && self.plan.const_of(z) == Some(0) =>
            {
                Some(SelectOp::Neg)
            }
            _ => None,
        }
    }

    fn emit_select_op(&mut self, kind: SelectOp, size: Size, cond: Cond, dst: Reg, base: Reg) {
        match kind {
            SelectOp::Inc => self.asm.csinc(size, cond, dst, base, base),
            SelectOp::Inv => self.asm.csinv(size, cond, dst, base, base),
            SelectOp::Neg => self.asm.csneg(size, cond, dst, base, base),
        }
    }

    fn elided_icmp_of(&self, condition: Value) -> Option<(IntCC, Value, Value)> {
        super::super::fuse::cinc_operand(self.func, &self.plan, condition, &|v| {
            self.defs[v.index()]
        })
    }

    pub(in crate::arm64) fn preserves_flags(&self, op: &Op, result: Option<Value>) -> bool {
        matches!(*op, Op::Brif(..))
            || matches!(*op, Op::Select(cond, ..) | Op::Trapz(cond) if self.flags.is_some_and(|(v, _)| v == cond))
            || super::super::fuse::preserves_flags(
                self.func,
                &self.plan,
                &|v| self.defs[v.index()],
                op,
                result,
                true,
            )
    }

    pub(in crate::arm64) fn select_cond(&mut self, condition: Value) -> Cond {
        if let Some((val, c)) = self.flags
            && val == condition
        {
            return c;
        }
        if let Some((cc, a, b)) = self.elided_icmp_of(condition) {
            let c = self.compare(cc, a, b);
            self.flags = Some((condition, c));
            return c;
        }
        self.flags = None;
        self.last_cmp = None;
        if !self.tst_small_zero(condition) {
            let flag = self.zext_src(condition, C);
            let size = if self.ty(condition).bytes() <= 4 {
                Size::W
            } else {
                Size::X
            };
            self.asm.add_imm(true, true, size, ZR, flag, 0);
        }
        self.flags = Some((condition, Cond::Ne));
        Cond::Ne
    }

    fn select_csel(
        &mut self,
        cond: Cond,
        dst: Reg,
        (x, y): (Value, Value),
        (xz, yz): (bool, bool),
    ) {
        let (left, right) = match (xz, yz) {
            (true, true) => (ZR, ZR),
            (true, false) => (ZR, self.bits(y, C)),
            (false, true) => (self.bits(x, A), ZR),
            (false, false) => self.bits_pair(x, y),
        };
        self.asm.csel(cond, dst, left, right);
    }

    fn select_const_pair(
        &mut self,
        cond: Cond,
        size: Size,
        dst: Reg,
        (x, y): (Value, Value),
        (kx, ky): (i64, i64),
    ) {
        let (ux, uy) = (kx as u64, ky as u64);
        let mask = if size == Size::W {
            u32::MAX as u64
        } else {
            u64::MAX
        };
        if ux == 0 || uy == 0 {
            self.select_csel(cond, dst, (x, y), (ux == 0, uy == 0));
            return;
        }
        let prefer_y = self.plan.reg_of(x).is_none()
            && (self.plan.reg_of(y).is_some() || Asm::mov_imm_is_single(ky));
        let pick = |this: &mut Self| {
            if prefer_y {
                (this.bits(y, C), cond.invert())
            } else {
                (this.bits(x, A), cond)
            }
        };
        if ((ux.wrapping_add(1)) & mask) == uy {
            let reg = self.bits(x, A);
            self.asm.csinc(size, cond, dst, reg, reg);
        } else if ((uy.wrapping_add(1)) & mask) == ux {
            let reg = self.bits(y, C);
            self.asm.csinc(size, cond.invert(), dst, reg, reg);
        } else if (!ux & mask) == uy {
            let (reg, c) = pick(self);
            self.asm.csinv(size, c, dst, reg, reg);
        } else if (ux.wrapping_neg() & mask) == uy {
            let (reg, c) = pick(self);
            self.asm.csneg(size, c, dst, reg, reg);
        } else {
            self.select_csel(cond, dst, (x, y), (false, false));
        }
    }

    fn select_int_pair(
        &mut self,
        cond: Cond,
        (ty, size): (Type, Size),
        dst: Reg,
        (x, y): (Value, Value),
        (kx, ky): (Option<i64>, Option<i64>),
    ) {
        if let (Some(kx), Some(ky)) = (kx, ky)
            && matches!(ty, Type::I32 | Type::I64)
        {
            self.select_const_pair(cond, size, dst, (x, y), (kx, ky));
        } else if let Some(kind) = self.select_transform(ty, y, x) {
            let reg = self.bits(x, A);
            self.emit_select_op(kind, size, cond, dst, reg);
        } else if let Some(kind) = self.select_transform(ty, x, y) {
            let reg = self.bits(y, C);
            self.emit_select_op(kind, size, cond.invert(), dst, reg);
        } else {
            self.select_csel(cond, dst, (x, y), (kx == Some(0), ky == Some(0)));
        }
    }

    fn select_word_special(
        &mut self,
        (cond, size, ty): (Cond, Size, Type),
        dst: Reg,
        (x, y): (Value, Value),
    ) -> bool {
        let uconst = |v: Value| {
            self.plan
                .const_of(v)
                .map(|imm| crate::fold::extend(imm, ty, false))
        };
        let sconst = |v: Value| {
            self.plan
                .const_of(v)
                .map(|imm| crate::fold::extend(imm, ty, true))
        };
        for ((keep, alt), c, scratch) in [((x, y), cond, A), ((y, x), cond.invert(), C)] {
            let k = uconst(keep);
            if sconst(alt) == Some(-1) {
                let reg = if k == Some(0) {
                    ZR
                } else {
                    self.bits(keep, scratch)
                };
                self.asm.csinv(size, c, dst, reg, ZR);
                return true;
            }
            if uconst(alt) == Some(1) {
                let reg = if k == Some(0) {
                    ZR
                } else {
                    self.bits(keep, scratch)
                };
                self.asm.csinc(size, c, dst, reg, ZR);
                return true;
            }
        }
        false
    }

    fn select_int(&mut self, cond: Cond, ty: Type, (x, y): (Value, Value), result: Value) {
        let dst = self.dst(result);
        let size = if ty == Type::I64 { Size::X } else { Size::W };
        let uconst = |v: Value| {
            self.plan
                .const_of(v)
                .map(|imm| crate::fold::extend(imm, ty, false))
        };
        let (kx, ky) = (uconst(x), uconst(y));
        match (kx, ky) {
            (Some(1), Some(0)) => self.asm.cset(cond, dst),
            (Some(0), Some(1)) => self.asm.cset(cond.invert(), dst),
            _ if matches!(ty, Type::I32 | Type::I64)
                && self.select_word_special((cond, size, ty), dst, (x, y)) => {}
            _ => self.select_int_pair(cond, (ty, size), dst, (x, y), (kx, ky)),
        }
        self.put(result, dst);
    }

    fn select_i128(&mut self, cond: Cond, x: Value, y: Value, result: Value) {
        self.invalidate_spills();
        self.raw(A, x);
        self.raw(D, y);
        self.asm.csel(cond, A, A, D);
        self.put(result, A);
        self.high(A, x);
        self.high(D, y);
        self.asm.csel(cond, A, A, D);
        self.put_high(result, A);
    }

    pub(super) fn select(&mut self, condition: Value, x: Value, y: Value, result: Value) {
        if self.ty(condition).is_vector() {
            self.invalidate_spills();
            self.vec_select_mask(condition, x, y, result);
            return;
        }
        let cond = self.select_cond(condition);
        let ty = self.ty(x);
        if ty.is_vector() {
            self.invalidate_spills();
            self.vec_select_cond(cond, x, y, result);
        } else if ty.is_float() {
            let (left, right) = (self.fin(x, V0), self.fin(y, V1));
            let dst = self.fdst(result);
            self.asm.fcsel(cond, ty == Type::F64, dst, left, right);
            self.fput(result, dst);
        } else if ty == Type::I128 {
            self.select_i128(cond, x, y, result);
        } else {
            self.select_int(cond, ty, (x, y), result);
        }
    }
}
