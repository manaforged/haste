pub(super) mod fuse;
mod moves;
mod patterns;
mod sib;

pub(super) use fuse::{fuse, narrows_exactly};
use patterns::odd_scale_source;
pub(super) use patterns::{
    LEAF, Root, adc_fold, adc_plan, bextr_parts, product_halves, shld_parts, unwrap_shift_mask,
};

use super::Emit;
use super::asm::{Alu, Asm, Mem, Reg, Rm, Shift, Size, Xmm};
use crate::alloc::{Location, Plan};
use crate::fold::fits_imm32;
use crate::frame::{Emitter, Ext};
use crate::ir::{Binary, Function, Inst, Lane, Op, ShuffleMask, Type, Value};
use crate::live::{Sib, SibBase};
use crate::reloc::{RelocKind, StencilReloc};

fn extend(imm: i64, ty: Type, ext: Ext) -> i64 {
    crate::fold::extend(imm, ty, matches!(ext, Ext::Sign))
}

const ODD_LEA_TAG: i32 = 0x4000_0000;

fn odd_lea_key(x: Value, lea_scale: u8) -> Option<i32> {
    i32::try_from(x.index())
        .ok()
        .filter(|&i| i < (1 << 28))
        .map(|i| ODD_LEA_TAG | (i << 2) | i32::from(lea_scale & 3))
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
        if self.plan.elided[value.index()] || !matches!(self.location(value), Location::Slot) {
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
        if self.plan.elided[value.index()] {
            return None;
        }
        self.plan.reg_of(value)
    }

    pub(super) fn has_elided_mem(&self, value: Value) -> bool {
        self.plan.elided[value.index()]
            && self.defs[value.index()].is_some_and(|inst| {
                matches!(
                    self.func.insts[inst.index()].op,
                    Op::Load(..) | Op::StackLoad(..) | Op::AtomicLoad(..)
                )
            })
    }

    pub(super) fn elided_mem(&mut self, value: Value) -> Option<Mem> {
        if !self.plan.elided[value.index()] {
            return None;
        }
        let inst = self.defs[value.index()]?;
        match self.func.insts[inst.index()].op {
            Op::Load(addr, off) => Some(self.address(addr, off)),
            Op::StackLoad(slot, off) => Some(self.slot(slot, off)),
            Op::AtomicLoad(addr, _) => Some(if let Some(sym) = self.image_sym(addr) {
                Mem::rip(sym, 0)
            } else if self.plan.elided[addr.index()] && self.plan.sib[addr.index()].is_some() {
                self.address(addr, 0)
            } else {
                self.atomic_target(addr, Reg::Rcx, &[])
            }),
            _ => None,
        }
    }

    /// The float register that holds `value`, when its location is one.
    pub(super) fn xmm_of(&self, value: Value) -> Option<Xmm> {
        if self.plan.elided[value.index()] {
            return None;
        }
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

    /// The frame memory at `disp` from where rbp would point, addressed from the frame base.
    pub(super) fn frame_mem(&self, disp: i32) -> Mem {
        Mem::new(self.base, disp + self.bias)
    }

    /// The displacement of `mem` from where rbp would point when it is a plain frame address.
    pub(super) fn frame_disp(&self, mem: Mem) -> Option<i32> {
        (mem.base == self.base && mem.index.is_none() && mem.rip.is_none())
            .then(|| mem.disp - self.bias)
    }

    /// The symbol of `value` when it is an image address that code reaches relative to `rip`.
    pub(super) fn image_sym(&self, value: Value) -> Option<u32> {
        match self.location(value) {
            Location::Symbol(sym) if self.direct_calls && self.func.is_image_addr(value) => {
                Some(sym)
            }
            _ => None,
        }
    }

    /// The memory at `addr + offset`: relative to the frame base for a frame address, to the location register of `addr`, or through rcx.
    pub(super) fn address(&mut self, addr: Value, offset: i32) -> Mem {
        self.address_in(addr, offset, Reg::Rcx, None)
    }

    fn with_base(&mut self, temp: Option<Reg>, base: Value, emit: impl FnOnce(&mut Asm, Reg)) {
        let reg = temp.unwrap_or(Reg::R11);
        if temp.is_none() {
            self.asm.push(Reg::R11);
        }
        self.raw(reg, base);
        emit(&mut self.asm, reg);
        if temp.is_none() {
            self.asm.pop(Reg::R11);
        }
    }

    fn load_odd_index(&mut self, dst: Reg, x: Value, lea_scale: u8) {
        let key = odd_lea_key(x, lea_scale);
        if let Some(cached) = key.and_then(|k| self.cached_spill_for(k, dst)) {
            if cached != dst {
                self.asm.mov(dst, cached);
                self.clobber_reg(dst);
            }
            return;
        }
        let rx = match self.reg_of(x).filter(|&r| {
            r != Reg::Rsp && (self.ty(x).bytes() >= 8 || self.plan.zero_ext[x.index()])
        }) {
            Some(r) => r,
            None => {
                self.get(dst, x, Ext::Zero);
                dst
            }
        };
        self.asm.lea(dst, Mem::sib(rx, rx, lea_scale, 0));
        match key {
            Some(k) if dst != Reg::R11 => self.record_spill(k, dst),
            _ => self.clobber_reg(dst),
        }
    }

    fn load_index(&mut self, dst: Reg, index: Value) {
        if let Some((x, lea_scale)) =
            odd_scale_source(self.func, &self.plan, |v| self.defs[v.index()], index)
        {
            self.load_odd_index(dst, x, lea_scale);
        } else {
            self.get(dst, index, Ext::Zero);
        }
    }

    fn sib_slow(
        &mut self,
        base: Value,
        mode: Sib,
        disp: i32,
        (scratch, temp): (Reg, Option<Reg>),
    ) -> Mem {
        self.load_index(scratch, mode.index);
        if mode.shift > 0
            && !matches!(self.location(base), Location::Slot | Location::Frame(..))
            && !matches!(self.location(base), Location::Imm(imm) if fits_imm32(imm))
        {
            self.with_base(temp, base, |asm, reg| {
                asm.lea(scratch, Mem::sib(reg, scratch, mode.shift, 0))
            });
            self.clobber_reg(scratch);
            return Mem::new(scratch, disp);
        }
        self.clear_flags();
        if mode.shift > 0 {
            self.asm.shift_imm(Size::Q, Shift::Shl, scratch, mode.shift);
        }
        match self.location(base) {
            Location::Slot => self.asm.alu(Alu::Add, scratch, self.at(base)),
            Location::Imm(imm) if fits_imm32(imm) => {
                self.asm.alu_imm(Alu::Add, scratch, imm as i32)
            }
            Location::Frame(slot, off) => {
                self.asm.alu(Alu::Add, scratch, self.base);
                self.asm.alu_imm(
                    Alu::Add,
                    scratch,
                    self.frame.slots[slot.index()] + off + self.bias,
                );
            }
            _ => self.with_base(temp, base, |asm, reg| asm.alu(Alu::Add, scratch, reg)),
        }
        self.clobber_reg(scratch);
        Mem::new(scratch, disp)
    }

    pub(super) fn sib_mem(
        &mut self,
        mode: Sib,
        offset: i32,
        scratch: Reg,
        temp: Option<Reg>,
    ) -> Mem {
        let idx_reg = self
            .reg_of(mode.index)
            .filter(|_| self.ty(mode.index).bytes() >= 8 || self.plan.zero_ext[mode.index.index()]);
        match mode.base {
            SibBase::Frame(slot) => {
                let disp = self.frame.slots[slot.index()] + mode.disp + offset + self.bias;
                match idx_reg {
                    Some(idx) => Mem::sib(self.base, idx, mode.shift, disp),
                    None => {
                        self.load_index(scratch, mode.index);
                        Mem::sib(self.base, scratch, mode.shift, disp)
                    }
                }
            }
            SibBase::Value(base) => {
                let disp = mode.disp + offset;
                match (self.reg_of(base), idx_reg) {
                    (Some(b), Some(idx)) => Mem::sib(b, idx, mode.shift, disp),
                    (Some(b), None) => {
                        self.load_index(scratch, mode.index);
                        Mem::sib(b, scratch, mode.shift, disp)
                    }
                    (None, Some(idx)) => {
                        self.raw(scratch, base);
                        Mem::sib(scratch, idx, mode.shift, disp)
                    }
                    (None, None) => self.sib_slow(base, mode, disp, (scratch, temp)),
                }
            }
        }
    }

    pub(super) fn address_in(
        &mut self,
        addr: Value,
        offset: i32,
        scratch: Reg,
        temp: Option<Reg>,
    ) -> Mem {
        if self.plan.elided[addr.index()]
            && let Some(mode) = self.plan.sib[addr.index()]
        {
            return self.sib_mem(mode, offset, scratch, temp);
        }
        if let Some(reg) = self.reg_of(addr) {
            return Mem::new(reg, offset);
        }
        if let Location::Offset(base, base_off) = self.location(addr)
            && let Some(disp) = base_off.checked_add(offset)
        {
            return Mem::new(base, disp);
        }
        if let Some(sym) = self.image_sym(addr) {
            return Mem::rip(sym, offset);
        }
        if let Location::Frame(slot, base) = self.location(addr)
            && let Some(disp) = self.frame.slots[slot.index()]
                .checked_add(base)
                .and_then(|disp| disp.checked_add(offset))
        {
            return self.frame_mem(disp);
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
        if let Some(mem) = self.elided_mem(value) {
            self.load_ext(reg, mem, self.ty(value).bytes(), ext);
            return;
        }
        let ty = self.ty(value);
        let bytes = ty.bytes();
        let full = bytes >= 8 || (!matches!(ext, Ext::Sign) && self.plan.zero_ext[value.index()]);
        match self.location(value) {
            Location::Imm(imm) => {
                self.asm.mov_imm(reg, extend(imm, ty, ext));
                self.clobber_reg(reg);
            }
            Location::SlotAddr(_)
            | Location::Symbol(_)
            | Location::Frame(..)
            | Location::Offset(..) => self.raw(reg, value),
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
                self.load_ext(reg, self.frame_mem(disp), bytes, ext);
                if bytes >= 8 {
                    self.record_spill(disp, reg);
                }
            }
        }
    }

    /// Loads `sym` into `reg` relative to `rip`.
    pub(super) fn emit_rip_symbol(&mut self, reg: Reg, sym: u32) {
        let at = self.asm.lea_rip_reloc(reg);
        self.relocs
            .push(StencilReloc::symbol(at, sym, RelocKind::X64RipRel32));
    }

    pub(super) fn raw(&mut self, reg: Reg, value: Value) {
        if let Some(mem) = self.elided_mem(value) {
            self.asm.load(Size::of(self.ty(value).bytes()), reg, mem);
            self.clobber_reg(reg);
            return;
        }
        match self.location(value) {
            Location::Imm(imm) => self.asm.mov_imm(reg, imm),
            Location::SlotAddr(sym) if self.direct_calls => self.emit_rip_symbol(reg, sym),
            Location::SlotAddr(sym) => {
                let at = self.asm.mov_slot_reloc(reg);
                self.relocs
                    .push(StencilReloc::symbol(at, sym, RelocKind::Abs32));
            }
            Location::Symbol(sym) if self.direct_calls && self.func.is_image_addr(value) => {
                self.emit_rip_symbol(reg, sym)
            }
            Location::Symbol(sym) => {
                let at = self.asm.mov_abs64(reg);
                self.relocs
                    .push(StencilReloc::symbol(at, sym, RelocKind::Abs64));
            }
            Location::Frame(slot, offset) => {
                let at = self.slot(slot, offset);
                self.asm.lea(reg, at);
            }
            Location::Offset(base, offset) => self.asm.lea(reg, Mem::new(base, offset)),
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
                    self.asm.load(Size::Q, reg, self.frame_mem(disp));
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
}

type Site = Option<(usize, usize, Inst)>;

/// Elides each integer lane-0 insert into a zero vector whose every use is one later lane-0 broadcast in its block, which then moves the scalar in from its own register.
pub(super) fn elide_broadcast_inserts(
    func: &Function,
    plan: &mut Plan<Reg>,
    defs: &[Option<Inst>],
    (uses, pos_of, use_of): (&[u32], &[Site], &[Site]),
) {
    let zero = |v: Value| {
        defs[v.index()].is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::Const(0)))
    };
    for (v, def) in pos_of.iter().enumerate() {
        let Some((block, start, inst)) = *def else {
            continue;
        };
        let Op::Insert(base, scalar, 0) = func.insts[inst.index()].op else {
            continue;
        };
        let Some((lane, count)) = func.ty(base).vec_shape() else {
            continue;
        };
        if base == scalar
            || !lane.is_int()
            || func.ty(base).bytes() != 16
            || !zero(base)
            || plan.elided[v]
        {
            continue;
        }
        let Some((used_in, end, user)) = use_of[v] else {
            continue;
        };
        let Op::Shuffle(a, b, ShuffleMask::Const(mask)) = func.insts[user.index()].op else {
            continue;
        };
        let reads = u32::from(a.index() == v) + u32::from(b.index() == v);
        if used_in != block
            || a.index() != v
            || uses[v] != reads
            || mask[..usize::from(count)].iter().any(|&at| at != 0)
        {
            continue;
        }
        if fuse::preserved(func, plan, block, start, end, &[], scalar) {
            plan.elided[v] = true;
        }
    }
}

fn shift_imm_ok(func: &Function, op: Binary, x: Value, y: Value) -> bool {
    if x == y {
        return false;
    }
    func.ty(y).vec_shape().is_some_and(|(lane, _)| {
        matches!(lane, Lane::I16 | Lane::I32) || (lane == Lane::I64 && op != Binary::Sshr)
    })
}

pub(super) fn elide_shift_splats(
    func: &Function,
    plan: &mut Plan<Reg>,
    defs: &[Option<Inst>],
    uses: &[u32],
) {
    let mut shift_uses = vec![0_u32; func.types.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            if let Op::Binary(op @ (Binary::Ishl | Binary::Ushr | Binary::Sshr), x, y) =
                func.insts[inst.index()].op
                && shift_imm_ok(func, op, x, y)
            {
                shift_uses[y.index()] += 1;
            }
        }
    }
    for (v, &count) in shift_uses.iter().enumerate() {
        if count == 0 || count != uses[v] || plan.elided[v] {
            continue;
        }
        let Some((_, lanes)) = func.types[v].vec_shape() else {
            continue;
        };
        let Some(def) = defs[v] else {
            continue;
        };
        let Op::Shuffle(a, b, ShuffleMask::Const(m)) = func.insts[def.index()].op else {
            continue;
        };
        let c = usize::from(lanes);
        if !m[..c].iter().all(|&i| i == m[0] && usize::from(i) < c) {
            continue;
        }
        let Some(a_inst) = defs[a.index()] else {
            continue;
        };
        let Op::Insert(_, scalar, idx) = func.insts[a_inst.index()].op else {
            continue;
        };
        if idx != m[0] || plan.const_of(scalar).is_none() {
            continue;
        }
        plan.elided[v] = true;
        let a_reads = 1 + u32::from(b == a);
        if uses[a.index()] == a_reads {
            plan.elided[a.index()] = true;
        }
    }
}
