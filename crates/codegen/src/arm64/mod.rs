mod asm;
mod atomic;
mod float;
mod frame;
mod int;
mod location;
mod wide;

use asm::{Alu, Asm, Cond, FP, LR, Label, Mem, Reg, SP, Size, Vreg, ZR};
use frame::{Abi, Loc, Packing, SAVED, classify};
use crate::alloc::{Location, Plan, Regs, plan};
use crate::frame::{Frame, INCOMING, PAGE, edge_moves, layout, saved};
use crate::ir::{Block, Function, Inst, Op, SigRef, Type, Value};
use crate::reloc::{DictTierUp, StencilReloc};
use crate::target::{Os, Target};
use crate::{Code, Site};

/// The scratch registers the emitter computes in; planning never hands them out.
const A: Reg = Reg(9);
const C: Reg = Reg(10);
const D: Reg = Reg(11);
const S: Reg = Reg(12);
const T: Reg = Reg(13);
/// Holds immediates and callees; the encoder itself uses x17 for large memory offsets.
const IP: Reg = Reg(16);
const V0: Vreg = Vreg(0);
const V1: Vreg = Vreg(1);

/// The registers 128-bit operations use as scratch besides the emitter's own.
const WIDE_SCRATCH: [Reg; 2] = [Reg(14), Reg(15)];

/// The registers planning hands out: the saved pool, and caller-saved registers that are neither scratch, return registers nor the indirect-result register.
const REGS: Regs<Reg> = Regs { saved: &SAVED, leaf: &[Reg(2), Reg(3), Reg(4), Reg(5), Reg(6), Reg(7), Reg(14), Reg(15)], wide_scratch: &WIDE_SCRATCH, dict: None, float: &[], float_branches: &[], absolute_calls: false, two_address: false };

/// Encodes `func` as A64 machine code for `target`.
pub(crate) fn compile(func: &Function, target: &Target) -> Result<Code, String> {
    let packing = if target.os == Os::Darwin { Packing::Natural } else { Packing::Slots };
    let order = crate::live::order(func);
    let abi = classify(&func.signature, packing)?;
    let plan = plan(func, &REGS, &order, &abi.params, |sig| classify(sig, packing));
    let frame = layout(func, saved(func, &plan.used, &SAVED), &plan.locations, |sig| classify(sig, packing))?;
    let mut asm = Asm::default();
    let blocks = func.blocks.iter().map(|_| asm.label()).collect();
    let copied = crate::live::copied_tests(func, &order);
    Emit { func, frame, abi, packing, asm, blocks, sites: Vec::new(), relocs: Vec::new(), plan, flags: None, next: None, copied }.run(order)
}

pub(crate) fn dict_stub(stub_exec: usize, body_exec: usize, dict_addr: usize, tier_up: Option<DictTierUp>) -> Result<Vec<u8>, String> {
    let mut asm = Asm::default();
    if let Some(tier) = tier_up {
        let fast = asm.label();
        asm.mov_fixed64(IP, tier.counter_addr as u64);
        asm.load(Size::W, false, Reg(17), Mem::new(IP, 0));
        asm.add_imm(false, false, Size::W, Reg(17), Reg(17), 1);
        asm.store(Size::W, Mem::new(IP, 0), Reg(17));
        asm.mov_imm(A, i64::from(tier.threshold));
        asm.alu(Alu::Subs, Size::W, ZR, Reg(17), A);
        asm.bcond(Cond::Ne, fast);
        asm.mov_imm(IP, i64::from(tier.encoded_index));
        asm.mov_fixed64(Reg(17), tier.stub_entry as u64);
        asm.br(Reg(17));
        asm.bind(fast);
    }
    asm.mov_fixed64(A, dict_addr as u64);
    let at = stub_exec as i64 + i64::from(asm.pos());
    let delta = (body_exec as i64 - at) / 4;
    if (body_exec as i64 - at) % 4 == 0 && (-(1 << 25)..(1 << 25)).contains(&delta) {
        asm.word(0x1400_0000 | ((delta as u32) & 0x03FF_FFFF));
    } else {
        asm.mov_fixed64(IP, body_exec as u64);
        asm.br(IP);
    }
    asm.finish()
}

#[derive(Clone, Copy)]
pub(super) enum Ext {
    Zero,
    Sign,
}

pub(super) struct Emit<'a> {
    func: &'a Function,
    frame: Frame<Reg>,
    abi: Abi,
    packing: Packing,
    asm: Asm,
    blocks: Vec<Label>,
    sites: Vec<Site>,
    relocs: Vec<StencilReloc>,
    plan: Plan<Reg>,
    flags: Option<(Value, Cond)>,
    next: Option<Block>,
    copied: Vec<bool>,
}

impl Emit<'_> {
    fn run(mut self, order: Vec<Block>) -> Result<Code, String> {
        let entry = self.func.entry.ok_or("the function has no entry block")?;
        self.prologue();
        self.intake(entry);
        for (index, block) in order.iter().enumerate() {
            self.next = order.get(index + 1).copied();
            self.block(*block)?;
        }
        Ok(Code { bytes: self.asm.finish()?, sites: self.sites, relocs: self.relocs })
    }

    fn block(&mut self, block: Block) -> Result<(), String> {
        self.asm.bind(self.blocks[block.index()]);
        self.body(block)
    }

    /// Emits the instructions of `block`, at its label or as a copy in place of a jump to it.
    fn body(&mut self, block: Block) -> Result<(), String> {
        self.flags = None;
        let func = self.func;
        let insts = &func.blocks[block.index()].insts;
        for inst in insts {
            self.inst(*inst)?;
        }
        let ended = insts.last().is_some_and(|inst| func.insts[inst.index()].op.is_terminator());
        if !ended {
            self.asm.brk();
        }
        Ok(())
    }

    fn ty(&self, value: Value) -> Type {
        self.func.ty(value)
    }

    fn save_slot(index: usize) -> Mem {
        Mem::new(FP, -8 * (index as i32 + 1))
    }

    fn prologue(&mut self) {
        self.asm.stp_pre(FP, LR, SP, -16);
        self.asm.mov_sp(FP, SP);
        let size = self.frame.size;
        if size > PAGE {
            self.probe(size);
        } else if size > 0 {
            self.asm.add_imm(true, false, Size::X, SP, SP, size);
        }
        for (index, reg) in self.frame.saves.clone().into_iter().enumerate() {
            self.asm.store(Size::X, Self::save_slot(index), reg);
        }
    }

    /// Moves the stack pointer down by `size` one page at a time, touching each page, so the guard page faults before anything past it is written.
    fn probe(&mut self, size: u32) {
        self.asm.mov_imm(IP, i64::from(size / PAGE));
        let again = self.asm.label();
        self.asm.bind(again);
        self.asm.add_imm(true, false, Size::X, SP, SP, PAGE);
        self.asm.store(Size::X, Mem::new(SP, 0), ZR);
        self.asm.add_imm(true, true, Size::X, IP, IP, 1);
        self.asm.bcond(Cond::Ne, again);
        if size % PAGE != 0 {
            self.asm.add_imm(true, false, Size::X, SP, SP, size % PAGE);
        }
    }

    fn epilogue(&mut self) {
        for (index, reg) in self.frame.saves.clone().into_iter().enumerate() {
            self.asm.load(Size::X, false, reg, Self::save_slot(index));
        }
        self.asm.mov_sp(SP, FP);
        self.asm.ldp_post(FP, LR, SP, 16);
        self.asm.ret();
    }

    fn intake(&mut self, entry: Block) {
        let params = self.func.blocks[entry.index()].params.clone();
        for (value, loc) in params.into_iter().zip(self.abi.params.clone()) {
            match loc {
                Loc::Stack(offset) => {
                    let from = Mem::new(FP, INCOMING + offset as i32);
                    self.load_home(value, from);
                }
                loc => self.store_loc(value, loc),
            }
        }
    }

    fn inst(&mut self, inst: Inst) -> Result<(), String> {
        let func = self.func;
        let data = &func.insts[inst.index()];
        let result = data.result;
        let value = || result.ok_or_else(|| "a value instruction without a result".to_owned());
        match &data.op {
            Op::Call { sig, callee, args, results } => self.call(*sig, *callee, args, results),
            Op::TryCall { sig, callee, args, normal, handler } => self.try_call(*sig, *callee, args, *normal, *handler),
            Op::Jump(block, args) => {
                self.edge(*block, args);
                if self.copied[inst.index()] {
                    return self.body(*block);
                }
                self.goto(*block, true);
                Ok(())
            }
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                self.brif(*condition, (*then, then_args), (*otherwise, else_args));
                Ok(())
            }
            Op::Return(values) => {
                self.ret(values);
                Ok(())
            }
            Op::Trap => {
                self.asm.brk();
                Ok(())
            }
            Op::Trapz(condition) => {
                self.trapz(*condition);
                Ok(())
            }
            op => self.value_op(op, value),
        }
    }

    value_op!();

    /// The condition under which `condition` is nonzero: the fused flags of its compare, or a test of its value.
    fn truth(&mut self, condition: Value) -> Cond {
        match self.flags.take() {
            Some((value, cond)) if value == condition => cond,
            _ => {
                let reg = self.src(condition, A, Ext::Zero);
                let size = Size::of(self.ty(condition).bytes());
                self.asm.alu(Alu::Subs, if size == Size::X { Size::X } else { Size::W }, ZR, reg, ZR);
                Cond::Ne
            }
        }
    }

    fn trapz(&mut self, condition: Value) {
        if let Some(known) = self.plan.const_of(condition) {
            if crate::fold::extend(known, self.ty(condition), false) == 0 {
                self.asm.brk();
            }
            return;
        }
        let skip = self.asm.label();
        let cond = self.truth(condition);
        self.asm.bcond(cond, skip);
        self.asm.brk();
        self.asm.bind(skip);
    }

    /// Whether `a` and `b` live in the same place, so writing one overwrites the other.
    fn same_home(&self, a: Value, b: Value) -> bool {
        a == b || matches!((self.plan.location(a), self.plan.location(b)), (Location::Reg(x), Location::Reg(y)) if x == y)
    }

    /// Writes `value` into the location of `param`.
    fn assign(&mut self, param: Value, value: Value) {
        match self.plan.location(param) {
            Location::Reg(reg) => self.raw(reg, value),
            _ => self.move_to(self.at(param), value),
        }
    }

    fn edge(&mut self, block: Block, args: &[Value]) {
        let (pairs, clash) = edge_moves(args, &self.func.blocks[block.index()].params, |a, b| self.same_home(a, b));
        if !clash {
            for (arg, param) in pairs {
                self.assign(param, arg);
            }
            return;
        }
        for (index, (arg, _)) in pairs.iter().enumerate() {
            self.move_to(Mem::new(FP, self.frame.edge_slot(index)), *arg);
        }
        for (index, (_, param)) in pairs.iter().enumerate() {
            self.load_home(*param, Mem::new(FP, self.frame.edge_slot(index)));
        }
    }

    fn brif(&mut self, condition: Value, then: (Block, &[Value]), otherwise: (Block, &[Value])) {
        if let Some(known) = self.plan.const_of(condition) {
            let (block, args) = if crate::fold::extend(known, self.ty(condition), false) != 0 { then } else { otherwise };
            self.edge(block, args);
            self.goto(block, true);
            return;
        }
        let cond = self.truth(condition);
        let (then_moves, else_moves) = (self.moves(then.0, then.1), self.moves(otherwise.0, otherwise.1));
        let falls = self.next.is_some_and(|next| next == then.0 || next == self.func.jump_target(then.0));
        if !else_moves && (then_moves || falls) {
            self.asm.bcond(cond.invert(), self.blocks[self.func.jump_target(otherwise.0).index()]);
            self.edge(then.0, then.1);
            self.goto(then.0, true);
            return;
        }
        if !then_moves {
            self.asm.bcond(cond, self.blocks[self.func.jump_target(then.0).index()]);
            self.edge(otherwise.0, otherwise.1);
            self.goto(otherwise.0, true);
            return;
        }
        let other = self.asm.label();
        self.asm.bcond(cond.invert(), other);
        self.edge(then.0, then.1);
        self.goto(then.0, false);
        self.asm.bind(other);
        self.edge(otherwise.0, otherwise.1);
        self.goto(otherwise.0, true);
    }

    /// Whether entering `block` with `args` moves any value.
    fn moves(&self, block: Block, args: &[Value]) -> bool {
        args.iter().zip(&self.func.blocks[block.index()].params).any(|(arg, param)| !self.same_home(*arg, *param))
    }

    fn goto(&mut self, block: Block, last: bool) {
        let target = self.func.jump_target(block);
        if last && (self.next == Some(block) || self.next == Some(target)) {
            return;
        }
        self.asm.b(self.blocks[target.index()]);
    }

    fn ret(&mut self, values: &[Value]) {
        for (value, loc) in values.iter().zip(self.abi.returns.clone()) {
            self.load_loc(*value, loc);
        }
        self.epilogue();
    }

    /// Puts `value` where `loc` carries it into a call or out of a return.
    fn load_loc(&mut self, value: Value, loc: Loc) {
        match loc {
            Loc::Int(reg) => self.raw(reg, value),
            Loc::Pair(low, high) => {
                self.raw(low, value);
                self.high(high, value);
            }
            Loc::Float(index) => self.asm.vload(self.ty(value).bytes(), Vreg(index), self.at(value)),
            Loc::Stack(_) | Loc::Struct(..) => {}
        }
    }

    /// Takes `value` from where `loc` carries it out of a call or into the entry.
    fn store_loc(&mut self, value: Value, loc: Loc) {
        match loc {
            Loc::Int(reg) => self.put(value, reg),
            Loc::Pair(low, high) => {
                self.put(value, low);
                self.put_high(value, high);
            }
            Loc::Float(index) => self.asm.vstore(self.ty(value).bytes(), self.at(value), Vreg(index)),
            Loc::Stack(_) | Loc::Struct(..) => {}
        }
    }

    /// Writes a stack argument at `offset` above the stack pointer, in its own size when Apple's packing gives it no full slot.
    fn stack_argument(&mut self, offset: u32, value: Value) {
        let to = Mem::new(SP, offset as i32);
        let bytes = self.ty(value).bytes();
        if self.packing == Packing::Natural && bytes < 8 {
            let reg = self.src(value, A, Ext::Zero);
            self.asm.store(Size::of(bytes), to, reg);
            return;
        }
        self.move_to(to, value);
    }

    fn arguments(&mut self, sig: SigRef, callee: Value, args: &[Value]) -> Result<Abi, String> {
        let abi = classify(&self.func.sigs[sig.index()], self.packing)?;
        for (arg, loc) in args.iter().zip(&abi.params) {
            if let Loc::Stack(offset) = *loc {
                self.stack_argument(offset, *arg);
            }
        }
        for (arg, loc) in args.iter().zip(&abi.params) {
            self.load_loc(*arg, *loc);
        }
        let target = self.plan.reg_of(callee).unwrap_or(IP);
        self.raw(target, callee);
        self.asm.blr(target);
        Ok(abi)
    }

    fn call(&mut self, sig: SigRef, callee: Value, args: &[Value], results: &[Value]) -> Result<(), String> {
        let abi = self.arguments(sig, callee, args)?;
        for (value, loc) in results.iter().zip(abi.returns) {
            self.store_loc(*value, loc);
        }
        Ok(())
    }

    fn try_call(&mut self, sig: SigRef, callee: Value, args: &[Value], normal: Block, handler: Block) -> Result<(), String> {
        self.arguments(sig, callee, args)?;
        let ret = self.asm.pos();
        self.asm.b(self.blocks[normal.index()]);
        let pad = self.asm.pos();
        let func = self.func;
        let [exception] = func.blocks[handler.index()].params[..] else {
            return Err("a try_call handler takes exactly the exception".to_owned());
        };
        self.put(exception, Reg(0));
        self.asm.b(self.blocks[handler.index()]);
        self.sites.push(Site { ret, handler: pad, frame_offset: self.frame.size });
        Ok(())
    }
}
