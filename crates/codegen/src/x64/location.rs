use super::asm::{Alu, Mem, Reg, Rm, Shift, Size, Xmm};
use super::features::Features;
use super::float::scalar;
use super::Emit;
use crate::alloc::{Location, Plan};
use crate::frame::Ext;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};
use crate::live::{Sib, SibBase};
use crate::reloc::{RelocKind, StencilReloc};

fn extend(imm: i64, ty: Type, ext: Ext) -> i64 {
    crate::fold::extend(imm, ty, matches!(ext, Ext::Sign))
}

impl Emit<'_> {
    pub(super) fn location(&self, value: Value) -> Location<Reg> {
        self.plan.location(value)
    }

    /// Whether `value` needs no code where it is defined, because its location is a constant or a frame address.
    pub(super) fn is_static(&self, value: Value) -> bool {
        self.plan.is_static(value)
    }

    /// The constant `value` holds, when its location is one.
    pub(super) fn const_of(&self, value: Value) -> Option<i64> {
        self.plan.const_of(value)
    }

    pub(super) fn imm_of(&self, value: Value) -> Option<i32> {
        match self.location(value) {
            Location::Imm(imm) => i32::try_from(imm).ok(),
            _ => None,
        }
    }

    /// The constant `value` as a 32-bit immediate when `size` is narrower than 64 bits, or when it fits a sign-extended 32-bit immediate.
    pub(super) fn imm_sized(&self, value: Value, size: Size) -> Option<i32> {
        if size == Size::Q {
            self.imm_of(value)
        } else {
            self.const_of(value).map(|imm| imm as i32)
        }
    }

    pub(super) fn cached_spill(&self, value: Value) -> Option<Reg> {
        if !matches!(self.location(value), Location::Slot) {
            return None;
        }
        let disp = self.frame.values[value.index()];
        debug_assert_ne!(disp, 0, "unhomed value {value:?}");
        self.cached_spill_for(disp, Reg::Rax)
    }

    /// The constant `value` extended to 64 bits by `ext`, when it fits a sign-extended 32-bit immediate.
    pub(super) fn imm_extended(&self, value: Value, ext: Ext) -> Option<i32> {
        match self.location(value) {
            Location::Imm(imm) => i32::try_from(extend(imm, self.ty(value), ext)).ok(),
            _ => None,
        }
    }

    /// The register that holds `value`, when its location is one.
    pub(super) fn reg_of(&self, value: Value) -> Option<Reg> {
        self.plan.reg_of(value)
    }

    /// The float register that holds `value`, when its location is one.
    pub(super) fn xmm_of(&self, value: Value) -> Option<Xmm> {
        self.plan.float_of(value).map(Xmm)
    }

    /// The register to compute `value` in: its location register, or rax.
    pub(super) fn dst(&self, value: Value) -> Reg {
        self.reg_of(value).unwrap_or(Reg::Rax)
    }

    /// The float register to compute `value` in: its location register, or xmm0.
    pub(super) fn fdst(&self, value: Value) -> Xmm {
        self.xmm_of(value).unwrap_or(Xmm(0))
    }

    /// The memory at `addr + offset`: relative to rbp for a frame address, to the location register of `addr`, or through rcx.
    pub(super) fn address(&mut self, addr: Value, offset: i32) -> Mem {
        self.address_in(addr, offset, Reg::Rcx, Reg::R11)
    }

    pub(super) fn sib_mem(&mut self, mode: Sib, offset: i32, scratch: Reg, temp: Reg) -> Mem {
        let idx_reg = self.reg_of(mode.index).filter(|_| self.ty(mode.index).bytes() >= 8 || self.plan.zero_ext[mode.index.index()]);
        match mode.base {
            SibBase::Frame(slot) => {
                let disp = self.frame.slots[slot.index()] + mode.disp + offset;
                match idx_reg {
                    Some(idx) => Mem::sib(Reg::Rbp, idx, mode.shift, disp),
                    None => {
                        self.get(scratch, mode.index, Ext::Zero);
                        Mem::sib(Reg::Rbp, scratch, mode.shift, disp)
                    }
                }
            }
            SibBase::Value(base) => {
                let disp = mode.disp + offset;
                match (self.reg_of(base), idx_reg) {
                    (Some(b), Some(idx)) => Mem::sib(b, idx, mode.shift, disp),
                    (Some(b), None) => {
                        self.get(scratch, mode.index, Ext::Zero);
                        Mem::sib(b, scratch, mode.shift, disp)
                    }
                    (None, Some(idx)) => {
                        self.raw(scratch, base);
                        Mem::sib(scratch, idx, mode.shift, disp)
                    }
                    (None, None) => {
                        self.get(scratch, mode.index, Ext::Zero);
                        if mode.shift > 0 {
                            self.asm.shift_imm(Size::Q, Shift::Shl, scratch, mode.shift);
                        }
                        match self.location(base) {
                            Location::Slot => self.asm.alu(Alu::Add, scratch, self.at(base)),
                            Location::Imm(imm) if i32::try_from(imm).is_ok() => self.asm.alu_imm(Alu::Add, scratch, imm as i32),
                            Location::Frame(slot, off) => {
                                self.asm.alu(Alu::Add, scratch, Reg::Rbp);
                                self.asm.alu_imm(Alu::Add, scratch, self.frame.slots[slot.index()] + off);
                            }
                            _ => {
                                self.raw(temp, base);
                                self.asm.alu(Alu::Add, scratch, temp);
                            }
                        }
                        self.clobber_reg(scratch);
                        Mem::new(scratch, disp)
                    }
                }
            }
        }
    }

    pub(super) fn address_in(&mut self, addr: Value, offset: i32, scratch: Reg, temp: Reg) -> Mem {
        if self.plan.elided[addr.index()]
            && let Some(mode) = self.plan.sib[addr.index()]
        {
            return self.sib_mem(mode, offset, scratch, temp);
        }
        if let Some(reg) = self.reg_of(addr) {
            return Mem::new(reg, offset);
        }
        if let Location::Frame(slot, base) = self.location(addr)
            && let Some(disp) = self.frame.slots[slot.index()].checked_add(base).and_then(|disp| disp.checked_add(offset))
        {
            return Mem::new(Reg::Rbp, disp);
        }
        self.raw(scratch, addr);
        Mem::new(scratch, offset)
    }

    fn load_ext(&mut self, reg: Reg, source: impl Into<Rm>, bytes: u32, ext: Ext) {
        match ext {
            _ if bytes >= 8 => self.asm.load(Size::Q, reg, source),
            Ext::Zero => self.asm.load(Size::of(bytes), reg, source),
            Ext::Sign => self.asm.load_signed(Size::of(bytes), reg, source),
        }
        self.clobber_reg(reg);
    }

    pub(super) fn get(&mut self, reg: Reg, value: Value, ext: Ext) {
        let ty = self.ty(value);
        let bytes = ty.bytes();
        let full = bytes >= 8 || (!matches!(ext, Ext::Sign) && self.plan.zero_ext[value.index()]);
        match self.location(value) {
            Location::Imm(imm) => {
                self.asm.mov_imm(reg, extend(imm, ty, ext));
                self.clobber_reg(reg);
            }
            Location::SlotAddr(_) | Location::Symbol(_) | Location::Frame(..) => self.raw(reg, value),
            Location::Float(from) => {
                self.asm.movq_out(reg, Xmm(from));
                if full {
                    self.clobber_reg(reg);
                } else {
                    self.load_ext(reg, reg, bytes, ext);
                }
            }
            Location::Reg(from) if full => {
                if from != reg {
                    self.asm.mov(reg, from);
                    self.clobber_reg(reg);
                }
            }
            Location::Reg(from) => self.load_ext(reg, from, bytes, ext),
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in get()");
                if let Some(cached_reg) = self.cached_spill_for(disp, reg) {
                    if full {
                        if cached_reg != reg {
                            self.asm.mov(reg, cached_reg);
                        }
                        self.record_spill(disp, reg);
                    } else {
                        self.load_ext(reg, cached_reg, bytes, ext);
                    }
                    return;
                }
                self.load_ext(reg, Mem::new(Reg::Rbp, disp), bytes, ext);
                if bytes >= 8 {
                    self.record_spill(disp, reg);
                }
            }
        }
    }

    pub(super) fn raw(&mut self, reg: Reg, value: Value) {
        match self.location(value) {
            Location::Imm(imm) => self.asm.mov_imm(reg, imm),
            Location::SlotAddr(sym) if self.direct_calls => {
                let at = self.asm.lea_rip_reloc(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::X64RipRel32));
            }
            Location::SlotAddr(sym) => {
                let at = self.asm.mov_slot_reloc(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Abs32));
            }
            Location::Symbol(sym) => {
                let at = self.asm.mov_abs64(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Abs64));
            }
            Location::Frame(slot, offset) => {
                let at = self.slot(slot, offset);
                self.asm.lea(reg, at);
            }
            Location::Reg(from) if from == reg => return,
            Location::Reg(from) => self.asm.mov(reg, from),
            Location::Float(from) => self.asm.movq_out(reg, Xmm(from)),
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in raw()");
                if let Some(cached_reg) = self.cached_spill_for(disp, reg) {
                    if cached_reg != reg {
                        self.asm.mov(reg, cached_reg);
                    }
                } else {
                    self.asm.load(Size::Q, reg, Mem::new(Reg::Rbp, disp));
                }
                self.record_spill(disp, reg);
                return;
            }
        }
        self.clobber_reg(reg);
    }

    pub(super) fn put(&mut self, value: Value, reg: Reg) {
        match self.location(value) {
            Location::Reg(location) => {
                if location != reg {
                    self.asm.mov(location, reg);
                }
                self.clobber_reg(location);
            }
            Location::Float(location) => {
                self.clobber_xmm(Xmm(location));
                self.asm.movq_in(Xmm(location), reg);
            }
            Location::Slot => self.store_reg_home(self.at(value), reg, self.ty(value).bytes()),
            _ => self.clobber_reg(reg),
        }
    }

    /// Loads the float or vector of type `ty` at `from` into `xmm`; an 8-byte vector by `movq` and a 16-byte vector by `movdqu`, which needs no alignment.
    pub(super) fn xmm_load(&mut self, ty: Type, xmm: Xmm, from: Mem) {
        if ty.is_vector() {
            let (prefix, opcode) = if ty.bytes() == 16 { (0xF3, 0x6F) } else { (0xF3, 0x7E) };
            self.asm.sse(&[prefix], &[0x0F, opcode], xmm.0, from);
        } else {
            self.asm.sse(&[scalar(ty)], &[0x0F, 0x10], xmm.0, from);
        }
    }

    /// Stores the float or vector of type `ty` in `xmm` to `to`.
    pub(super) fn xmm_store(&mut self, ty: Type, to: Mem, xmm: Xmm) {
        if ty.is_vector() {
            let (prefix, opcode) = if ty.bytes() == 16 { (0xF3, 0x7F) } else { (0x66, 0xD6) };
            self.asm.sse(&[prefix], &[0x0F, opcode], xmm.0, to);
        } else {
            self.asm.sse(&[scalar(ty)], &[0x0F, 0x11], xmm.0, to);
        }
    }

    /// Puts float `value` in `xmm`.
    pub(super) fn fget(&mut self, xmm: Xmm, value: Value) {
        let ty = self.ty(value);
        match self.location(value) {
            Location::Float(from) if Xmm(from) == xmm => {}
            Location::Float(from) => {
                self.asm.movaps(xmm, Xmm(from));
                self.clobber_xmm(xmm);
            }
            Location::Imm(0) if ty.is_vector() => self.zero_xmm(xmm),
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in fget()");
                if let Some((cached_disp, cached_reg, cached_ty)) = self.last_fspill
                    && cached_disp == disp
                    && cached_ty == ty
                {
                    if cached_reg != xmm {
                        self.asm.movaps(xmm, cached_reg);
                        self.clobber_xmm(xmm);
                    }
                } else {
                    self.xmm_load(ty, xmm, Mem::new(Reg::Rbp, disp));
                    self.clobber_xmm(xmm);
                    self.last_fspill = Some((disp, xmm, ty));
                }
            }
            _ => {
                self.raw(Reg::Rax, value);
                self.asm.movq_in(xmm, Reg::Rax);
                self.clobber_xmm(xmm);
            }
        }
    }

    /// Writes `xmm` into the location of float `value`.
    pub(super) fn fput(&mut self, value: Value, xmm: Xmm) {
        let ty = self.ty(value);
        match self.location(value) {
            Location::Float(to) => {
                if Xmm(to) != xmm {
                    self.asm.movaps(Xmm(to), xmm);
                }
                self.clobber_xmm(Xmm(to));
            }
            _ => self.store_xmm_home(self.at(value), xmm, ty),
        }
    }

    /// A float register that holds `value`: its location register, or `scratch` loaded with it.
    pub(super) fn fin(&mut self, value: Value, scratch: Xmm) -> Xmm {
        self.xmm_of(value).unwrap_or_else(|| {
            self.fget(scratch, value);
            scratch
        })
    }

    /// A float register or memory operand that holds `value`, using `scratch` when `value` has no float home.
    pub(super) fn foperand(&mut self, value: Value, scratch: Xmm) -> Rm {
        match self.location(value) {
            Location::Float(xmm) => Xmm(xmm).into(),
            Location::Slot => self.at(value).into(),
            _ => {
                self.fget(scratch, value);
                scratch.into()
            }
        }
    }

    /// A register or memory operand that holds `value`, using r11 when `value` has no register or stack home.
    pub(super) fn operand(&mut self, value: Value) -> Rm {
        match self.location(value) {
            Location::Reg(reg) => reg.into(),
            Location::Slot => self.cached_spill(value).map_or_else(|| Rm::from(self.at(value)), Rm::from),
            _ => {
                self.raw(Reg::R11, value);
                Reg::R11.into()
            }
        }
    }

    fn store_reg_home(&mut self, to: Mem, reg: Reg, bytes: u32) {
        self.asm.store(Size::Q, to, reg);
        if to.base == Reg::Rbp && to.index.is_none() {
            if bytes <= 8 {
                self.write_spill(to.disp, reg);
            } else {
                self.invalidate_disp(to.disp);
            }
        }
    }

    fn store_xmm_home(&mut self, to: Mem, xmm: Xmm, ty: Type) {
        self.xmm_store(ty, to, xmm);
        if to.base == Reg::Rbp && to.index.is_none() {
            self.invalidate_disp(to.disp);
            self.last_fspill = Some((to.disp, xmm, ty));
        }
    }

    /// Writes the eight (or sixteen) bytes of `value` to `to`, using rax as scratch; a float in a register writes the bytes of its type.
    pub(super) fn move_to(&mut self, to: Mem, value: Value) {
        let stack_home = to.base == Reg::Rbp && to.index.is_none();
        let ty = self.ty(value);
        match self.location(value) {
            Location::Imm(0) if ty.is_vector() => {
                self.asm.store_imm(to, 0);
                if ty.bytes() == 16 {
                    self.asm.store_imm(to.offset(8), 0);
                }
            }
            Location::Imm(imm) => match i32::try_from(imm) {
                Ok(small) => self.asm.store_imm(to, small),
                Err(_) => {
                    self.asm.mov_imm(Reg::Rax, imm);
                    self.clobber_reg(Reg::Rax);
                    self.asm.store(Size::Q, to, Reg::Rax);
                }
            },
            Location::SlotAddr(sym) if !self.direct_calls => {
                let at = self.asm.store_slot_reloc(to);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Abs32));
            }
            Location::Frame(..) | Location::SlotAddr(_) | Location::Symbol(_) => {
                self.raw(Reg::Rax, value);
                self.asm.store(Size::Q, to, Reg::Rax);
            }
            Location::Reg(reg) => return self.store_reg_home(to, reg, ty.bytes()),
            Location::Float(xmm) => return self.store_xmm_home(to, Xmm(xmm), ty),
            Location::Slot => {
                if ty.bytes() <= 8 && let Some(cached_reg) = self.cached_spill(value) {
                    return self.store_reg_home(to, cached_reg, ty.bytes());
                }
                if let Some((cached_disp, cached_reg, cached_ty)) = self.last_fspill
                    && cached_disp == self.frame.values[value.index()]
                    && cached_ty == ty
                {
                    return self.store_xmm_home(to, cached_reg, cached_ty);
                }
                return self.copy_to(to, self.at(value), ty);
            }
        }
        if stack_home {
            self.invalidate_disp(to.disp);
            if ty.bytes() == 16 {
                self.invalidate_disp(to.disp + 8);
            }
        }
    }
}

const FOLD_LIMIT: i64 = 1 << 24;

struct SibMatch {
    sib: Sib,
    elide: Vec<(Value, usize)>,
    base_start: usize,
    idx_start: usize,
}

struct FuseCtx<'a> {
    func: &'a Function,
    plan: &'a Plan<Reg>,
    uses: &'a [u32],
    pos_of: &'a [Option<(usize, usize, Inst)>],
    b_idx: usize,
}

fn preserved(func: &Function, plan: &Plan<Reg>, b_idx: usize, start: usize, end: usize, skip: &[usize], val: Value) -> bool {
    if plan.elided[val.index()] {
        return false;
    }
    if start + 1 >= end || plan.location(val).is_static() {
        return true;
    }
    if end - start > 128 {
        return false;
    }
    let (reg, slot_id) = match plan.location(val) {
        Location::Reg(reg) => (Some(reg), None),
        Location::Slot => {
            let Some(id) = plan.shared.get(val.index()).copied().flatten() else { return true };
            (None, Some(id))
        }
        _ => return false,
    };
    let clobbers = |r: Value| {
        reg.is_some_and(|reg| plan.reg_of(r) == Some(reg))
            || slot_id.is_some_and(|id| matches!(plan.location(r), Location::Slot) && plan.shared.get(r.index()).copied().flatten() == Some(id))
    };
    for pos in start + 1..end {
        if skip.contains(&pos) {
            continue;
        }
        let data = &func.insts[func.blocks[b_idx].insts[pos].index()];
        if data.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]) {
            continue;
        }
        if (reg.is_some() && matches!(data.op, Op::Call { .. } | Op::TryCall { .. }))
            || data.result.and_then(|r| plan.pairs[r.index()]).is_some_and(clobbers)
            || crate::live::results(data).any(clobbers)
        {
            return false;
        }
    }
    true
}

fn band_ok(func: &Function, plan: &Plan<Reg>, b_idx: usize, (mpos, end, skip): (usize, usize, &[usize]), (x, y): (Value, Value)) -> bool {
    func.ty(x).is_small_int() && preserved(func, plan, b_idx, mpos, end, skip, x) && preserved(func, plan, b_idx, mpos, end, skip, y)
}

fn zero_cmp_band(func: &Function, plan: &Plan<Reg>, uses: &[u32], pos_of: &[Option<(usize, usize, Inst)>], b_idx: usize, lhs: Value, rhs: Value) -> Option<(Value, usize, Value, Value)> {
    let is_zero = |v: Value| plan.const_of(v).is_some_and(|imm| crate::fold::extend(imm, func.ty(v), false) == 0);
    let masked = is_zero(rhs).then_some(lhs).or_else(|| is_zero(lhs).then_some(rhs))?;
    if uses[masked.index()] != 1 || plan.elided[masked.index()] {
        return None;
    }
    let (mb, mpos, minst) = pos_of[masked.index()]?;
    if mb != b_idx {
        return None;
    }
    let Op::Binary(Binary::Band, x, y) = func.insts[minst.index()].op else { return None };
    Some((masked, mpos, x, y))
}

impl FuseCtx<'_> {
    fn peel_const(&self, mut val: Value, mut max_pos: usize, mut peel_cur: bool, elide: &mut Vec<(Value, usize)>) -> Option<(Value, i64, usize)> {
        let (mut disp, mut last_pos) = (0_i64, max_pos);
        loop {
            if self.plan.elided[val.index()] || self.func.ty(val) != Type::I64 {
                break;
            }
            let Some((vb, vpos, vinst)) = self.pos_of[val.index()] else { break };
            if vb != self.b_idx || (peel_cur && (self.uses[val.index()] != 1 || vpos >= max_pos)) {
                break;
            }
            let (next, delta) = match self.func.insts[vinst.index()].op {
                Op::Binary(Binary::Iadd, a, b) => match (self.plan.const_of(a), self.plan.const_of(b)) {
                    (_, Some(c)) => (a, c),
                    (Some(c), None) => (b, c),
                    _ => break,
                },
                Op::Binary(Binary::Isub, a, b) => match self.plan.const_of(b).and_then(|c| c.checked_neg()) {
                    Some(c) => (a, c),
                    None => break,
                },
                _ => break,
            };
            let next_disp = disp.checked_add(delta)?;
            if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&next_disp) {
                break;
            }
            disp = next_disp;
            if peel_cur {
                elide.push((val, vpos));
            }
            last_pos = vpos;
            max_pos = vpos;
            val = next;
            peel_cur = true;
        }
        Some((val, disp, last_pos))
    }

    fn unwrap_uext(&self, v: Value, max_pos: usize, multi_use: bool, elide: &mut Vec<(Value, usize)>) -> (Value, usize) {
        if !self.plan.elided[v.index()]
            && self.uses[v.index()] == 1
            && let Some((eb, epos, einst)) = self.pos_of[v.index()]
            && eb == self.b_idx
            && epos < max_pos
            && let Op::Convert(Convert::Uextend, inner) = self.func.insts[einst.index()].op
            && self.func.ty(inner).is_small_int()
            && self.func.ty(inner).bytes() <= 4
            && !self.plan.elided[inner.index()]
            && (!multi_use || self.plan.zero_ext[inner.index()] || self.plan.reg_of(inner).is_none())
        {
            elide.push((v, epos));
            return (inner, epos);
        }
        (v, max_pos)
    }

    fn parse_index(&self, v: Value, sum_pos: usize, scaled_only: bool, multi_use: bool) -> Option<(Value, u8, i64, usize, Vec<(Value, usize)>)> {
        let mut elide = Vec::new();
        let (core, disp, use_pos) = self.peel_const(v, sum_pos, true, &mut elide)?;
        if !self.plan.elided[core.index()]
            && self.uses[core.index()] == 1
            && self.func.ty(core) == Type::I64
            && let Some((sb, spos, sinst)) = self.pos_of[core.index()]
            && sb == self.b_idx
            && spos < use_pos
        {
            let scaled = match self.func.insts[sinst.index()].op {
                Op::Binary(Binary::Ishl, a, b) => self.plan.const_of(b).filter(|c| matches!(c, 1..=3)).map(|c| (a, c as u8)),
                Op::Binary(Binary::Imul, a, b) => [(a, b), (b, a)].into_iter().find_map(|(x, c)| {
                    self.plan.const_of(c).filter(|&k| matches!(k, 2 | 4 | 8)).map(|k| (x, (k as u64).trailing_zeros() as u8))
                }),
                _ => None,
            };
            if let Some((raw_idx, shift)) = scaled {
                elide.push((core, spos));
                let (idx, idx_start) = self.unwrap_uext(raw_idx, spos, multi_use, &mut elide);
                if !self.plan.elided[idx.index()] && !matches!(self.plan.location(idx), Location::Frame(..) | Location::Imm(..)) {
                    return Some((idx, shift, disp, idx_start, elide));
                }
                return None;
            }
        }
        if scaled_only {
            return None;
        }
        let (idx, idx_start) = self.unwrap_uext(core, use_pos, multi_use, &mut elide);
        if self.plan.elided[idx.index()] || matches!(self.plan.location(idx), Location::Frame(..) | Location::Imm(..)) {
            return None;
        }
        Some((idx, 0, disp, idx_start, elide))
    }

    fn match_sib(&self, root: Value, root_pos: usize, multi_use: bool) -> Option<SibMatch> {
        if self.plan.elided[root.index()] || self.plan.is_static(root) || self.func.ty(root) != Type::I64 {
            return None;
        }
        let mut elide = Vec::new();
        let (sum_node, outer_disp, _) = self.peel_const(root, root_pos, false, &mut elide)?;
        let (sb, sum_pos, sinst) = self.pos_of[sum_node.index()]?;
        if sb != self.b_idx || self.plan.elided[sum_node.index()] {
            return None;
        }
        if sum_node != root {
            if self.uses[sum_node.index()] != 1 || sum_pos >= root_pos {
                return None;
            }
            elide.push((sum_node, sum_pos));
        }
        let Op::Binary(Binary::Iadd, left, right) = self.func.insts[sinst.index()].op else { return None };
        let try_pair = |b_side: Value, i_side: Value, scaled_only: bool| -> Option<SibMatch> {
            let mut pair_elide = elide.clone();
            let (b_core, b_disp, base_start) = self.peel_const(b_side, sum_pos, true, &mut pair_elide)?;
            if self.plan.elided[b_core.index()] {
                return None;
            }
            let (base, frame_off) = match self.plan.location(b_core) {
                Location::Frame(slot, off) => (SibBase::Frame(slot), i64::from(off)),
                Location::Imm(_) => return None,
                _ => (SibBase::Value(b_core), 0),
            };
            let (index, shift, i_disp, idx_start, i_elide) = self.parse_index(i_side, sum_pos, scaled_only, multi_use)?;
            pair_elide.extend(i_elide);
            let total = outer_disp.checked_add(b_disp)?.checked_add(i_disp)?.checked_add(frame_off)?;
            if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&total) {
                return None;
            }
            Some(SibMatch { sib: Sib { base, index, shift, disp: total as i32 }, elide: pair_elide, base_start, idx_start })
        };
        [(left, right), (right, left)]
            .into_iter()
            .find_map(|(b, i)| try_pair(b, i, true))
            .or_else(|| [(left, right), (right, left)].into_iter().find_map(|(b, i)| try_pair(b, i, false)))
    }
}

fn sib_ok(func: &Function, plan: &Plan<Reg>, b_idx: usize, m: &SibMatch, extra_skip: Option<usize>, end_pos: usize) -> bool {
    let mut skip: Vec<usize> = m.elide.iter().map(|&(_, p)| p).collect();
    if let Some(extra) = extra_skip {
        skip.push(extra);
    }
    let base_ok = match m.sib.base {
        SibBase::Frame(_) => true,
        SibBase::Value(base) => preserved(func, plan, b_idx, m.base_start, end_pos, &skip, base),
    };
    base_ok && preserved(func, plan, b_idx, m.idx_start, end_pos, &skip, m.sib.index)
}

pub(super) fn fuse(func: &Function, plan: &mut Plan<Reg>, features: &Features) {
    if func.size() > 8192 {
        return;
    }
    let n = func.types.len();
    let (mut uses, mut pos_of, mut ls_uses) = (vec![0_u32; n], vec![None; n], vec![Vec::new(); n]);
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| uses[v.index()] += 1);
            match data.op {
                Op::Load(addr, off) => ls_uses[addr.index()].push((b_idx, pos, off)),
                Op::Store(val, addr, off) if val != addr => ls_uses[addr.index()].push((b_idx, pos, off)),
                _ => {}
            }
            if let Some(res) = data.result {
                pos_of[res.index()] = Some((b_idx, pos, inst));
            }
        }
    }
    let (mut branch_tst, mut cmp_tst, mut cand_sibs) = (Vec::new(), Vec::new(), Vec::new());
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            match data.op {
                Op::Binary(Binary::Band, x, y) if features.bmi1 && data.result.is_some_and(|r| !plan.elided[r.index()] && func.ty(r).is_small_int()) => {
                    for (cand, other) in [(x, y), (y, x)] {
                        if !plan.const_of(other).is_some_and(|m| matches!(m, 0xFF | 0xFFFF | 0xFFFF_FFFF))
                            && uses[cand.index()] == 1
                            && !plan.elided[cand.index()]
                            && let Some((cb, cpos, cinst)) = pos_of[cand.index()]
                            && cb == b_idx
                            && cpos < pos
                            && let Op::Unary(Unary::Bnot, inner) = func.insts[cinst.index()].op
                            && preserved(func, plan, b_idx, cpos, pos, &[], inner)
                        {
                            plan.elided[cand.index()] = true;
                            break;
                        }
                    }
                }
                Op::Binary(Binary::Iadd | Binary::Isub, ..)
                    if data.result.is_some_and(|r| !plan.elided[r.index()] && plan.sib[r.index()].is_none() && func.ty(r) == Type::I64) =>
                {
                    if let Some(res) = data.result {
                        cand_sibs.push((b_idx, pos, res));
                    }
                }
                Op::Brif(cond_val, ..) | Op::Trapz(cond_val) if uses[cond_val.index()] == 1 && !plan.elided[cond_val.index()] => {
                    if let Some((cb, cpos, cinst)) = pos_of[cond_val.index()]
                        && cb == b_idx
                        && cpos < pos
                    {
                        match func.insts[cinst.index()].op {
                            Op::Binary(Binary::Band, x, y) if band_ok(func, plan, b_idx, (cpos, pos, &[]), (x, y)) => {
                                branch_tst.push((cond_val, None));
                            }
                            Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                                if let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, &uses, &pos_of, b_idx, lhs, rhs)
                                    && mpos < cpos
                                    && band_ok(func, plan, b_idx, (mpos, pos, &[cpos]), (x, y))
                                {
                                    branch_tst.push((cond_val, Some(m)));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) if data.result.is_some_and(|r| !plan.elided[r.index()]) => {
                    if let Some(res) = data.result {
                        cmp_tst.push((b_idx, pos, res, lhs, rhs));
                    }
                }
                _ => {}
            }
        }
    }
    for (cond_val, m) in branch_tst {
        plan.elided[cond_val.index()] = true;
        if let Some(m) = m {
            plan.elided[m.index()] = true;
        }
    }
    for (b_idx, cpos, res, lhs, rhs) in cmp_tst {
        if !plan.elided[res.index()]
            && let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, &uses, &pos_of, b_idx, lhs, rhs)
            && mpos < cpos
            && band_ok(func, plan, b_idx, (mpos, cpos, &[]), (x, y))
        {
            plan.elided[m.index()] = true;
        }
    }
    for &(b_idx, pos, addr) in cand_sibs.iter().rev() {
        let addr_uses = &ls_uses[addr.index()];
        if plan.elided[addr.index()] || addr_uses.is_empty() || addr_uses.len() != uses[addr.index()] as usize {
            continue;
        }
        let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, b_idx };
        let Some(m) = ctx.match_sib(addr, pos, addr_uses.len() > 1) else { continue };
        let fits = addr_uses.iter().all(|&(ub, upos, off)| {
            ub == b_idx && upos > pos && (-FOLD_LIMIT..=FOLD_LIMIT).contains(&(i64::from(m.sib.disp) + i64::from(off)))
        });
        let Some(end_pos) = fits.then(|| addr_uses.iter().map(|&(_, p, _)| p).max()).flatten() else { continue };
        if sib_ok(func, plan, b_idx, &m, Some(pos), end_pos) {
            plan.sib[addr.index()] = Some(m.sib);
            plan.elided[addr.index()] = true;
            for (v, _) in m.elide {
                plan.elided[v.index()] = true;
            }
        }
    }
    for &(b_idx, pos, res) in cand_sibs.iter().rev() {
        if plan.elided[res.index()] || uses[res.index()] == 0 {
            continue;
        }
        let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, b_idx };
        let Some(m) = ctx.match_sib(res, pos, false) else { continue };
        if m.elide.is_empty() && !matches!(m.sib.base, SibBase::Frame(_)) {
            continue;
        }
        if sib_ok(func, plan, b_idx, &m, None, pos) {
            plan.sib[res.index()] = Some(m.sib);
            for (v, _) in m.elide {
                plan.elided[v.index()] = true;
            }
        }
    }
}
