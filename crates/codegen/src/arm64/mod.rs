mod asm;
mod atomic;
mod float;
mod frame;
mod int;
mod location;
mod wide;

use asm::{Alu, Asm, Cond, FP, LR, Label, Mem, Reg, SP, Size, Vreg, ZR};
use frame::{Abi, Loc, Packing, SAVED, SAVED_FLOAT, classify};
use crate::alloc::{Location, Plan, Regs};
use crate::frame::{ArgStep, Emitter, Ext, Frame, INCOMING, PAGE, Prepared, Step, arg_steps, drive, edge_steps, emit_body, prepare, save_offset};
use crate::ir::{Block, Function, Inst, Op, SigRef, Type, Value};
use crate::park::ParkReg;
use crate::reloc::{DictTierUp, StencilReloc};
use crate::target::{Os, Target};
use crate::{Code, Site};

const A: Reg = Reg(9);
const C: Reg = Reg(10);
const D: Reg = Reg(11);
const S: Reg = Reg(12);
const T: Reg = Reg(13);
const IP: Reg = Reg(16);
const V0: Vreg = Vreg(0);
const V1: Vreg = Vreg(1);

const WIDE_SCRATCH: [Reg; 2] = [Reg(14), Reg(15)];

const REGS: Regs<Reg> = Regs {
    saved: &SAVED,
    saved_float: &SAVED_FLOAT,
    leaf: &[Reg(2), Reg(3), Reg(4), Reg(5), Reg(6), Reg(7), Reg(14), Reg(15)],
    wide_scratch: &WIDE_SCRATCH,
    op_scratch: &[],
    uses_scratch: crate::alloc::no_scratch,
    dict: None,
    float: &float::POOL,
    float_branches: &float::BRANCHES,
    vectors: false,
    absolute_calls: false,
    two_address: false,
    park: true,
    fold_sib: false,
};

pub(crate) fn compile(func: &Function, target: &Target) -> Result<Code, String> {
    let packing = if target.os == Os::Darwin { Packing::Natural } else { Packing::Slots };
    let Prepared { order, copied, abi, plan, frame, framed } = prepare(func, &REGS, target.effort == crate::Effort::Full, |sig| classify(sig, packing))?;
    let mut asm = Asm::default();
    let blocks = func.blocks.iter().map(|_| asm.label()).collect();
    let emit = Emit { func, frame, framed, abi, packing, asm, blocks, sites: Vec::new(), relocs: Vec::new(), plan, flags: None, next: None, copied };
    drive(emit, &order)
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

pub(super) struct Emit<'a> {
    func: &'a Function,
    frame: Frame<Reg>,
    framed: bool,
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

impl<'a> Emitter<'a> for Emit<'a> {
    fn func(&self) -> &'a Function {
        self.func
    }

    fn prologue(&mut self) {
        if !self.framed {
            return;
        }
        self.asm.stp_pre(FP, LR, SP, -16);
        self.asm.mov_sp(FP, SP);
        let size = self.frame.size;
        if size > PAGE {
            self.probe(size);
        } else if size > 0 {
            self.asm.add_imm(true, false, Size::X, SP, SP, size);
        }
        for (index, &reg) in self.frame.saves.iter().enumerate() {
            self.asm.store(Size::X, Mem::new(FP, save_offset(index)), reg);
        }
        let base = self.frame.saves.len();
        for (index, &v) in self.frame.float_saves.iter().enumerate() {
            self.asm.vstore(8, Mem::new(FP, save_offset(base + index)), Vreg(v));
        }
    }

    fn intake(&mut self, entry: Block) -> Result<(), String> {
        let params = self.func.blocks[entry.index()].params.iter().copied().zip(self.abi.params.clone());
        let (registers, memory): (Vec<_>, Vec<_>) = params.partition(|(value, _)| self.plan.reg_of(*value).is_some() || self.vreg_of(*value).is_some());
        for (value, loc) in memory.into_iter().chain(registers) {
            match loc {
                Loc::Stack(offset) => self.load_home(value, Mem::new(FP, INCOMING + offset as i32)),
                _ => self.store_loc(value, loc),
            }
        }
        Ok(())
    }

    fn set_next(&mut self, next: Option<Block>) {
        self.next = next;
    }

    fn bind_block(&mut self, block: Block) {
        self.asm.bind(self.blocks[block.index()]);
    }

    fn clear_flags(&mut self) {
        self.flags = None;
    }

    fn inst(&mut self, inst: Inst) -> Result<(), String> {
        let func = self.func;
        let data = &func.insts[inst.index()];
        let result = data.result;
        let value = || result.ok_or_else(|| "a value instruction without a result".to_owned());
        match &data.op {
            Op::Call { sig, callee, args, results } => self.call(inst, (*sig, *callee), args, results),
            Op::TryCall { sig, callee, args, normal, handler } => self.try_call(*sig, *callee, args, *normal, *handler),
            Op::Jump(block, args) => {
                self.edge(*block, args)?;
                if self.copied[inst.index()] {
                    return emit_body(self, *block);
                }
                self.goto(*block, true);
                Ok(())
            }
            Op::Brif(condition, then, then_args, otherwise, else_args) => self.brif(*condition, (*then, then_args), (*otherwise, else_args)),
            Op::Return(values) => self.ret(values),
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

    fn trap(&mut self) {
        self.asm.brk();
    }

    fn finish(self) -> Result<Code, String> {
        Ok(Code { bytes: self.asm.finish()?, sites: self.sites, relocs: self.relocs })
    }
}

impl Emit<'_> {
    fn ty(&self, value: Value) -> Type {
        self.func.ty(value)
    }

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
        if self.framed {
            for (index, &reg) in self.frame.saves.iter().enumerate() {
                self.asm.load(Size::X, false, reg, Mem::new(FP, save_offset(index)));
            }
            let base = self.frame.saves.len();
            for (index, &v) in self.frame.float_saves.iter().enumerate() {
                self.asm.vload(8, Vreg(v), Mem::new(FP, save_offset(base + index)));
            }
            self.asm.mov_sp(SP, FP);
            self.asm.ldp_post(FP, LR, SP, 16);
        }
        self.asm.ret();
    }

    value_op!();

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

    fn assign(&mut self, param: Value, value: Value) {
        match self.plan.location(param) {
            Location::Reg(reg) => self.raw(reg, value),
            Location::Float(v) => self.fget(Vreg(v), value),
            _ => self.move_to(self.at(param), value),
        }
    }

    fn edge(&mut self, block: Block, args: &[Value]) -> Result<(), String> {
        for step in edge_steps(args, &self.func.blocks[block.index()].params, |a, b| self.plan.same_home(&self.frame, a, b)) {
            match step {
                Step::Move(arg, param) => self.assign(param, arg),
                Step::Save(arg, index) => self.move_to(Mem::new(FP, self.frame.edge_slot(index)), arg),
                Step::Restore(index, param) => self.load_home(param, Mem::new(FP, self.frame.edge_slot(index))),
            }
        }
        Ok(())
    }

    fn brif(&mut self, condition: Value, then: (Block, &[Value]), otherwise: (Block, &[Value])) -> Result<(), String> {
        if let Some(known) = self.plan.const_of(condition) {
            let (block, args) = if crate::fold::extend(known, self.ty(condition), false) != 0 { then } else { otherwise };
            self.edge(block, args)?;
            self.goto(block, true);
            return Ok(());
        }
        let cond = self.truth(condition);
        let (then_moves, else_moves) = (
            self.plan.moves(self.func, &self.frame, then.0, then.1),
            self.plan.moves(self.func, &self.frame, otherwise.0, otherwise.1),
        );
        let falls = self.next.is_some_and(|next| next == then.0 || next == self.func.jump_target(then.0));
        if !else_moves && (then_moves || falls) {
            self.asm.bcond(cond.invert(), self.blocks[self.func.jump_target(otherwise.0).index()]);
            self.edge(then.0, then.1)?;
            self.goto(then.0, true);
            return Ok(());
        }
        if !then_moves {
            self.asm.bcond(cond, self.blocks[self.func.jump_target(then.0).index()]);
            self.edge(otherwise.0, otherwise.1)?;
            self.goto(otherwise.0, true);
            return Ok(());
        }
        let other = self.asm.label();
        self.asm.bcond(cond.invert(), other);
        self.edge(then.0, then.1)?;
        self.goto(then.0, false);
        self.asm.bind(other);
        self.edge(otherwise.0, otherwise.1)?;
        self.goto(otherwise.0, true);
        Ok(())
    }

    fn goto(&mut self, block: Block, last: bool) {
        let target = self.func.jump_target(block);
        if last && (self.next == Some(block) || self.next == Some(target)) {
            return;
        }
        self.asm.b(self.blocks[target.index()]);
    }

    fn ret(&mut self, values: &[Value]) -> Result<(), String> {
        for (value, loc) in values.iter().zip(self.abi.returns.clone()) {
            self.load_loc(*value, loc);
        }
        self.epilogue();
        Ok(())
    }

    fn load_loc(&mut self, value: Value, loc: Loc) {
        match loc {
            Loc::Int(reg) => self.raw(reg, value),
            Loc::Pair(low, high) => {
                self.raw(low, value);
                self.high(high, value);
            }
            Loc::Float(index) => self.fget(Vreg(index), value),
            Loc::Stack(_) | Loc::Struct(..) => {}
        }
    }

    fn store_loc(&mut self, value: Value, loc: Loc) {
        match loc {
            Loc::Int(reg) => self.put(value, reg),
            Loc::Pair(low, high) => {
                self.put(value, low);
                self.put_high(value, high);
            }
            Loc::Float(index) => self.fput(value, Vreg(index)),
            Loc::Stack(_) | Loc::Struct(..) => {}
        }
    }

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
        let target = match self.plan.reg_of(callee).filter(|reg| SAVED.contains(reg)) {
            Some(reg) => reg,
            None => {
                self.raw(T, callee);
                T
            }
        };
        let float_targets: Vec<(Value, u8)> = args.iter().zip(&abi.params).filter_map(|(&arg, &loc)| if let Loc::Float(v) = loc { Some((arg, v)) } else { None }).collect();
        for step in arg_steps(&float_targets, |v| self.plan.float_of(v)) {
            match step {
                ArgStep::Move(src, dst) => self.asm.fmov_reg(true, Vreg(dst), Vreg(src)),
                ArgStep::Save(src) => self.asm.fmov_out(true, A, Vreg(src)),
                ArgStep::Restore(dst) => self.asm.fmov_in(true, Vreg(dst), A),
                ArgStep::Load(val, dst) => self.fget(Vreg(dst), val),
            }
        }
        let int_targets: Vec<(Value, Reg)> = args.iter().zip(&abi.params).filter_map(|(&arg, &loc)| if let Loc::Int(reg) = loc { Some((arg, reg)) } else { None }).collect();
        for step in arg_steps(&int_targets, |v| self.plan.reg_of(v)) {
            match step {
                ArgStep::Move(src, dst) => self.asm.mov(dst, src),
                ArgStep::Save(src) => self.asm.mov(A, src),
                ArgStep::Restore(dst) => self.asm.mov(dst, A),
                ArgStep::Load(val, dst) => self.raw(dst, val),
            }
        }
        for (&arg, &loc) in args.iter().zip(&abi.params) {
            if let Loc::Pair(low, high) = loc {
                self.raw(low, arg);
                self.high(high, arg);
            }
        }
        self.asm.blr(target);
        Ok(abi)
    }

    fn call(&mut self, inst: Inst, (sig, callee): (SigRef, Value), args: &[Value], results: &[Value]) -> Result<(), String> {
        let parked = self.plan.parked_at(inst).to_vec();
        for &(value, reg, _) in &parked {
            let home = self.at(value);
            match reg {
                ParkReg::Int(r) => self.asm.store(Size::X, home, r),
                ParkReg::Float(v) => self.asm.vstore(self.ty(value).bytes(), home, Vreg(v)),
            }
            self.plan.locations[value.index()] = Location::Slot;
        }
        let abi = self.arguments(sig, callee, args)?;
        for (value, loc) in results.iter().zip(abi.returns) {
            self.store_loc(*value, loc);
        }
        for (value, reg, lives) in parked {
            if lives {
                let home = self.at(value);
                match reg {
                    ParkReg::Int(r) => self.asm.load(Size::X, false, r, home),
                    ParkReg::Float(v) => self.asm.vload(self.ty(value).bytes(), Vreg(v), home),
                }
            }
            self.plan.locations[value.index()] = match reg {
                ParkReg::Int(r) => Location::Reg(r),
                ParkReg::Float(v) => Location::Float(v),
            };
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
