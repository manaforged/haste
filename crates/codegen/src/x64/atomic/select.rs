use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Value};
use crate::x64::asm::{Alu, Cond, Mem, Reg, Size};
use crate::x64::float::{X0, X1};
use crate::x64::location::LEAF;
use crate::x64::{Emit, int};

impl Emit<'_> {
    fn root(&self, condition: Value) -> (Value, bool) {
        match self.roots[condition.index()] {
            Some((leaf, None, table)) => (leaf, table != LEAF[0]),
            _ => (condition, false),
        }
    }

    /// The flags condition that holds when `condition` is not zero, from the live flags of its compare or a new test.
    pub(in crate::x64) fn truth(&mut self, condition: Value) -> Cond {
        let (root, flip) = self.root(condition);
        let cond = match self.flags.take() {
            Some((value, cond)) if value == root => cond,
            _ => self.test_cond(root),
        };
        if flip { cond.invert() } else { cond }
    }

    fn test_cond(&mut self, condition: Value) -> Cond {
        if self.plan.elided[condition.index()]
            && let Some(inst) = self.defs[condition.index()]
        {
            match self.func.insts[inst.index()].op {
                Op::Binary(Binary::Band, x, y) => return self.emit_test_band(x, y),
                Op::Convert(Convert::Ireduce, inner) if self.plan.elided[inner.index()] => {
                    if let Some(iinst) = self.defs[inner.index()]
                        && let Op::Binary(Binary::Band, x, y) = self.func.insts[iinst.index()].op
                    {
                        return self.emit_test_band(x, y);
                    }
                }
                Op::Icmp(cc, lhs, rhs) => return self.cmp_flags(cc, lhs, rhs, None),
                _ => {}
            }
        }
        if self.held_eq_zero(condition) {
            return Cond::Ne;
        }
        self.test_value(condition);
        Cond::Ne
    }

    fn held_eq_zero(&self, value: Value) -> bool {
        let Some((a, b)) = self.asm.held_eq() else {
            return false;
        };
        let ty = self.ty(value);
        let is_zero = |u: u32| {
            self.const_of(Value(u))
                .is_some_and(|c| crate::fold::extend(c, ty, false) == 0)
        };
        (a == value.index() as u32 && is_zero(b)) || (b == value.index() as u32 && is_zero(a))
    }

    fn test_value(&mut self, value: Value) {
        let size = Size::of(self.ty(value).bytes());
        if self.ty(value).is_small_int()
            && (self.has_elided_mem(value)
                || (matches!(self.location(value), Location::Slot)
                    && self.cached_spill(value).is_none()))
        {
            let rm = self.operand(value);
            self.asm.alu_imm_sized(size, Alu::Cmp, rm, 0);
            return;
        }
        let reg = self
            .reg_of(value)
            .or_else(|| self.cached_spill(value))
            .unwrap_or_else(|| {
                self.get(Reg::Rax, value, Ext::Zero);
                Reg::Rax
            });
        self.asm.test_sized(size, reg, reg);
    }

    fn held_eq_compare(&self, cc: IntCC, x: Value, y: Value) -> Option<Cond> {
        if !matches!(cc, IntCC::Equal | IntCC::NotEqual) {
            return None;
        }
        let (a, b) = self.asm.held_eq()?;
        let ty = self.ty(x);
        let eq = |u: u32, v: Value| {
            u == v.index() as u32
                || self
                    .const_of(Value(u))
                    .zip(self.const_of(v))
                    .is_some_and(|(c1, c2)| {
                        crate::fold::extend(c1, ty, false) == crate::fold::extend(c2, ty, false)
                    })
        };
        ((eq(a, x) && eq(b, y)) || (eq(a, y) && eq(b, x))).then(|| int::cond(cc))
    }

    /// Compares `x` with `y` and returns the flags condition under which `cc` holds.
    pub(in crate::x64) fn cmp_flags(
        &mut self,
        cc: IntCC,
        x: Value,
        y: Value,
        live_flags: Option<(Value, Cond)>,
    ) -> Cond {
        let is_zero = |v: Value| {
            self.const_of(v)
                .is_some_and(|imm| crate::fold::extend(imm, self.ty(v), false) == 0)
        };
        let is_one = |v: Value| {
            self.const_of(v)
                .is_some_and(|imm| crate::fold::extend(imm, self.ty(v), false) == 1)
        };
        let bool_one_var = |a: Value, b: Value| {
            if is_one(b) && self.plan.clean_bits[a.index()] <= 1 {
                Some(a)
            } else if is_one(a) && self.plan.clean_bits[b.index()] <= 1 {
                Some(b)
            } else {
                None
            }
        };
        if self.ty(x) == Type::I128 {
            self.invalidate_spills();
            let cond = self.wide_icmp(cc, x, y);
            self.invalidate_spills();
            cond
        } else if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && live_flags.is_some_and(|(val, fcc)| {
                fcc == Cond::Ne && ((val == x && is_zero(y)) || (val == y && is_zero(x)))
            })
        {
            int::cond(cc)
        } else if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && let Some(bv) = bool_one_var(x, y)
            && live_flags.is_some_and(|(val, fcc)| fcc == Cond::Ne && val == bv)
        {
            if cc == IntCC::Equal {
                Cond::Ne
            } else {
                Cond::E
            }
        } else if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && let Some(bv) = bool_one_var(x, y)
            && let Some(reg) = self.reg_of(bv).or_else(|| self.cached_spill(bv))
        {
            self.asm.test_sized(Size::of(self.ty(bv).bytes()), reg, reg);
            if cc == IntCC::Equal {
                Cond::Ne
            } else {
                Cond::E
            }
        } else if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && let Some((a, b)) = self.elided_band_cmp(x, y)
        {
            let nz = self.emit_test_band(a, b);
            if cc == IntCC::NotEqual {
                nz
            } else {
                nz.invert()
            }
        } else if let Some(cond) = self
            .held_eq_compare(cc, x, y)
            .or_else(|| self.held_compare(cc, x, y))
        {
            cond
        } else {
            let size = Size::of(self.ty(x).bytes());
            let (cc, x, y) = if self.reg_of(x).is_none()
                && (self.reg_of(y).is_some() || self.imm_sized(x, size).is_some())
            {
                (cc.swapped(), y, x)
            } else {
                (cc, x, y)
            };
            self.emit_cmp_pair(
                size,
                x,
                Some(y),
                matches!(cc, IntCC::Equal | IntCC::NotEqual) || cc.is_signed(),
                !cc.is_signed(),
            );
            int::cond(cc)
        }
    }

    fn select_chain(
        &mut self,
        leaves: [Value; 2],
        table: u8,
        (x, y): (Value, Value),
        result: Value,
    ) {
        let (key, first, other) = if table.count_ones() == 1 {
            (table.trailing_zeros(), x, y)
        } else {
            (table.trailing_ones(), y, x)
        };
        let dst = self.dst(result);
        for (at, leaf) in leaves.into_iter().enumerate() {
            let cond = self.test_cond(leaf);
            if at == 0 {
                self.raw(dst, first);
            }
            let source = self.operand(other);
            self.asm.cmov(
                if key >> at & 1 == 1 {
                    cond.invert()
                } else {
                    cond
                },
                dst,
                source,
            );
            self.clobber_reg(dst);
        }
        self.put(result, dst);
    }

    fn select_cond(&mut self, condition: Value, live_flags: Option<(Value, Cond)>) -> Cond {
        if let Some((value, cond)) = live_flags
            && value == condition
        {
            return cond;
        }
        if !self.plan.elided[condition.index()]
            && matches!(self.location(condition), Location::Slot)
            && self.cached_spill(condition).is_none()
        {
            self.asm.alu_imm_sized(
                Size::of(self.ty(condition).bytes()),
                Alu::Cmp,
                self.at(condition),
                0,
            );
            return Cond::Ne;
        }
        self.test_cond(condition)
    }

    fn select_copy(&mut self, value: Value, result: Value) {
        let ty = self.ty(result);
        if ty.in_float_register() {
            let dst = self.xmm_of(value).unwrap_or_else(|| {
                let dst = self.fdst(result);
                self.fget(dst, value);
                dst
            });
            self.fput(result, dst);
            return;
        }
        if ty == Type::I128 {
            self.asm.load(Size::Q, Reg::Rax, self.at(value));
            self.clobber_reg(Reg::Rax);
            self.put(result, Reg::Rax);
            self.asm.load(Size::Q, Reg::Rax, self.at(value).offset(8));
            self.put_high(result, Reg::Rax);
            self.clobber_reg(Reg::Rax);
            return;
        }
        if matches!(self.location(result), Location::Slot)
            && let Some(reg) = self.reg_of(value).or_else(|| self.cached_spill(value))
        {
            self.put(result, reg);
            return;
        }
        let dst = self.dst(result);
        self.raw(dst, value);
        self.put(result, dst);
    }

    pub(super) fn select(&mut self, condition: Value, x: Value, y: Value, result: Value) {
        let live_flags = self.flags.take();
        if self.ty(condition).is_vector() {
            self.flags = live_flags;
            self.spills.clear_fspill();
            self.vector_select(condition, x, y, result);
            if !self.features.avx {
                self.clobber_xmms(&[X0, X1]);
            }
            return;
        }
        if x == y {
            self.flags = live_flags;
            self.select_copy(x, result);
            return;
        }
        if let Some(imm) = self.const_of(condition) {
            let known = crate::fold::extend(imm, self.ty(condition), false);
            self.flags = live_flags;
            self.select_copy(if known != 0 { x } else { y }, result);
            return;
        }
        if let Some((first, Some(second), table)) = self.roots[condition.index()] {
            return self.select_chain([first, second], table, (x, y), result);
        }
        let (condition, flip) = self.root(condition);
        let orig_cond = self.select_cond(condition, live_flags);
        let mut cond = if flip { orig_cond.invert() } else { orig_cond };
        let ty = self.ty(x);
        if ty.in_float_register() {
            let dst = self.fdst(result);
            let (mut x, mut y) = (x, y);
            if self.xmm_of(x) == Some(dst) {
                (x, y) = (y, x);
                cond = cond.invert();
            }
            let done = self.asm.label();
            self.flags = Some((condition, orig_cond));
            self.fget(dst, y);
            self.asm.jcc(cond.invert(), done);
            self.fget(dst, x);
            self.asm.bind(done);
            self.clobber_reg(Reg::Rax);
            self.spills.clear_fspill();
            self.fput(result, dst);
            return;
        }
        if ty == Type::I128 {
            self.asm.load(Size::Q, Reg::Rax, self.at(y));
            self.asm.cmov(cond, Reg::Rax, self.at(x));
            self.clobber_reg(Reg::Rax);
            self.put(result, Reg::Rax);
            self.asm.load(Size::Q, Reg::Rax, self.at(y).offset(8));
            self.asm.cmov(cond, Reg::Rax, self.at(x).offset(8));
            self.put_high(result, Reg::Rax);
            self.clobber_reg(Reg::Rax);
            self.flags = Some((condition, orig_cond));
            return;
        }
        let dst = self.dst(result);
        if self.select_const_pair((condition, orig_cond), cond, (x, y), (ty, dst, result)) {
            return;
        }
        let (mut x, mut y) = (x, y);
        if self.reg_of(x) == Some(dst) {
            (x, y) = (y, x);
            cond = cond.invert();
        }
        self.flags = Some((condition, orig_cond));
        self.raw(dst, y);
        let source = self.operand(x);
        self.asm.cmov(cond, dst, source);
        self.clobber_reg(dst);
        self.put(result, dst);
    }

    fn select_const_pair(
        &mut self,
        root: (Value, Cond),
        cond: Cond,
        values: (Value, Value),
        target: (Type, Reg, Value),
    ) -> bool {
        let (x, y) = values;
        let (ty, dst, result) = target;
        let (Some(xv), Some(yv)) = (self.const_of(x), self.const_of(y)) else {
            return false;
        };
        if !matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64) {
            return false;
        }
        let xu = crate::fold::extend(xv, ty, false) as u64;
        let yu = crate::fold::extend(yv, ty, false) as u64;
        if (xu == 1 && yu == 0) || (xu == 0 && yu == 1) {
            self.emit_setcc_byte(if xu == 1 { cond } else { cond.invert() }, dst);
        } else if !self.select_const_lea(cond, (xu, yu), target) {
            return false;
        }
        self.flags = Some(root);
        self.clobber_reg(dst);
        self.put(result, dst);
        true
    }

    fn select_const_lea(
        &mut self,
        cond: Cond,
        pair: (u64, u64),
        target: (Type, Reg, Value),
    ) -> bool {
        let (xu, yu) = pair;
        let (ty, dst, result) = target;
        if dst == Reg::Rsp {
            return false;
        }
        let mask = u64::MAX >> (64 - ty.bits());
        let subword_zext = matches!(ty, Type::I8 | Type::I16) && self.plan.zero_ext[result.index()];
        let Some((branch_cond, imm32, step)) =
            Self::match_select_lea(cond, (xu, yu), (ty, mask, subword_zext))
        else {
            return false;
        };
        let size = if ty == Type::I64 { Size::Q } else { Size::D };
        let mem = match step {
            1 => Mem::new(dst, imm32),
            2 => Mem::sib(dst, dst, 0, imm32),
            3 => Mem::sib(dst, dst, 1, imm32),
            5 => Mem::sib(dst, dst, 2, imm32),
            _ => Mem::sib(dst, dst, 3, imm32),
        };
        self.emit_setcc_byte(branch_cond, dst);
        self.asm.lea_sized(size, dst, mem);
        true
    }

    fn match_select_lea(
        cond: Cond,
        pair: (u64, u64),
        info: (Type, u64, bool),
    ) -> Option<(Cond, i32, u64)> {
        let (xu, yu) = pair;
        let (ty, mask, subword_zext) = info;
        for step in [1_u64, 2, 3, 5, 9] {
            for (branch_cond, small, hi) in [(cond, yu, xu), (cond.invert(), xu, yu)] {
                if hi != (small.wrapping_add(step) & mask) {
                    continue;
                }
                if subword_zext && small.checked_add(step).is_none_or(|top| top > mask) {
                    continue;
                }
                let small_signed = if subword_zext {
                    small as i64
                } else {
                    crate::fold::extend(small as i64, ty, true)
                };
                if let Ok(imm32) = i32::try_from(small_signed) {
                    return Some((branch_cond, imm32, step));
                }
            }
        }
        None
    }

    fn emit_setcc_byte(&mut self, cond: Cond, dst: Reg) {
        self.asm.setcc(cond, dst);
        self.asm.load(Size::B, dst, dst);
    }
}
