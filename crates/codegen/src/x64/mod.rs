mod asm;
mod atomic;
mod float;
mod frame;
mod location;
mod int;
mod live;
mod wide;

use asm::{Alu, Asm, Cond, Label, Mem, Reg, Size, Xmm};
use frame::{Abi, Frame, Loc, classify, layout, saved};
use location::{Location, Plan, plan};

use crate::ir::{ArgumentPurpose, Block, Function, Inst, Op, SigRef, Type, Value};
use crate::{Code, Site};

const PAGE: u32 = 4096;
const INCOMING: i32 = 16;
const MOVSD: (&[u8], &[u8], &[u8]) = (&[0xF2], &[0x0F, 0x10], &[0x0F, 0x11]);

/// Encodes `func` as x86-64 machine code.
pub(crate) fn compile(func: &Function) -> Result<Code, String> {
    let order = live::order(func);
    let abi = classify(&func.signature)?;
    let plan = plan(func, &order, &abi.params);
    let frame = layout(func, saved(func, &plan.used), &plan.locations)?;
    let mut asm = Asm::default();
    let blocks = func.blocks.iter().map(|_| asm.label()).collect();
    Emit { func, frame, abi, asm, blocks, sites: Vec::new(), plan, flags: None, next: None }.run(order)
}

fn save_slot(index: usize) -> Mem {
    Mem::new(Reg::Rbp, -8 * (index as i32 + 1))
}

#[derive(Clone, Copy)]
pub(super) enum Ext {
    Zero,
    Sign,
}

pub(super) struct Emit<'a> {
    func: &'a Function,
    frame: Frame,
    abi: Abi,
    asm: Asm,
    blocks: Vec<Label>,
    sites: Vec<Site>,
    plan: Plan,
    flags: Option<(Value, Cond)>,
    next: Option<Block>,
}

impl Emit<'_> {
    fn run(mut self, order: Vec<Block>) -> Result<Code, String> {
        let entry = self.func.entry.ok_or("the function has no entry block")?;
        self.prologue();
        self.intake(entry)?;
        for (index, block) in order.iter().enumerate() {
            self.next = order.get(index + 1).copied();
            self.block(*block)?;
        }
        Ok(Code { bytes: self.asm.finish()?, sites: self.sites })
    }

    fn block(&mut self, block: Block) -> Result<(), String> {
        self.asm.bind(self.blocks[block.index()]);
        self.flags = None;
        let func = self.func;
        let insts = &func.blocks[block.index()].insts;
        for inst in insts {
            self.inst(*inst)?;
        }
        let ended = insts.last().is_some_and(|inst| func.insts[inst.index()].op.is_terminator());
        if !ended {
            self.ud2();
        }
        Ok(())
    }

    fn at(&self, value: Value) -> Mem {
        Mem::new(Reg::Rbp, self.frame.values[value.index()])
    }

    fn ty(&self, value: Value) -> Type {
        self.func.ty(value)
    }

    fn high(&mut self, reg: Reg, value: Value) {
        let at = self.at(value).offset(8);
        self.asm.load(Size::Q, reg, at);
    }

    fn put_high(&mut self, value: Value, reg: Reg) {
        let at = self.at(value).offset(8);
        self.asm.store(Size::Q, at, reg);
    }

    fn ud2(&mut self) {
        self.asm.bytes(&[0x0F, 0x0B]);
    }

    fn movsd_load(&mut self, xmm: Xmm, from: Mem) {
        self.asm.sse(MOVSD.0, MOVSD.1, xmm.0, from);
    }

    fn movsd_store(&mut self, to: Mem, xmm: Xmm) {
        self.asm.sse(MOVSD.0, MOVSD.2, xmm.0, to);
    }

    fn prologue(&mut self) {
        self.asm.push(Reg::Rbp);
        self.asm.mov(Reg::Rbp, Reg::Rsp);
        let size = self.frame.size;
        if size > PAGE {
            self.probe(size);
        } else if size > 0 {
            self.asm.alu_imm(Alu::Sub, Reg::Rsp, size as i32);
        }
        for (index, reg) in self.frame.saves.iter().enumerate() {
            self.asm.store(Size::Q, save_slot(index), *reg);
        }
    }

    fn probe(&mut self, size: u32) {
        self.asm.mov_imm(Reg::R11, i64::from(size / PAGE));
        let again = self.asm.label();
        self.asm.bind(again);
        self.asm.alu_imm(Alu::Sub, Reg::Rsp, PAGE as i32);
        self.asm.bytes(&[0x80, 0x0C, 0x24, 0x00]);
        self.asm.alu_imm(Alu::Sub, Reg::R11, 1);
        self.asm.jcc(Cond::Ne, again);
        if size % PAGE != 0 {
            self.asm.alu_imm(Alu::Sub, Reg::Rsp, (size % PAGE) as i32);
        }
    }

    fn epilogue(&mut self) {
        for (index, reg) in self.frame.saves.iter().enumerate() {
            self.asm.load(Size::Q, *reg, save_slot(index));
        }
        self.asm.bytes(&[0xC9, 0xC3]);
    }

    fn intake(&mut self, entry: Block) -> Result<(), String> {
        let params = self.func.blocks[entry.index()].params.clone();
        for (value, loc) in params.into_iter().zip(self.abi.params.clone()) {
            let incoming = |offset: u32| Mem::new(Reg::Rbp, INCOMING + offset as i32);
            match loc {
                Loc::Int(reg) => self.put(value, reg),
                Loc::Pair(low, high) => {
                    self.put(value, low);
                    self.put_high(value, high);
                }
                Loc::Float(xmm) => self.movsd_store(self.at(value), Xmm(xmm)),
                Loc::Stack(offset) => self.load_home(value, incoming(offset)),
                Loc::Struct(offset, _) => {
                    self.asm.lea(Reg::Rax, incoming(offset));
                    self.put(value, Reg::Rax);
                }
            }
        }
        Ok(())
    }

    fn copy_to(&mut self, to: Mem, from: Mem, ty: Type) {
        self.asm.load(Size::Q, Reg::Rax, from);
        self.asm.store(Size::Q, to, Reg::Rax);
        if ty == Type::I128 {
            self.asm.load(Size::Q, Reg::Rax, from.offset(8));
            self.asm.store(Size::Q, to.offset(8), Reg::Rax);
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
                self.edge(*block, args)?;
                self.goto(*block, true);
                Ok(())
            }
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                self.brif(*condition, (*then, then_args), (*otherwise, else_args))
            }
            Op::Return(values) => self.ret(values),
            Op::Trap => {
                self.ud2();
                Ok(())
            }
            Op::Trapz(condition) => {
                self.trapz(*condition);
                Ok(())
            }
            op => self.value_op(op, value),
        }
    }

    fn value_op(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
        match op {
            Op::Unary(op, x) => self.unary(*op, *x, result()?),
            Op::Binary(op, x, y) => self.binary(*op, *x, *y, result()?),
            Op::Convert(op, x) => self.convert(*op, *x, result()?),
            Op::Icmp(cc, x, y) => self.icmp(*cc, *x, *y, result()?),
            Op::Fits(op, signed, x, y) => self.fits(*op, *signed, (*x, *y), result()?),
            Op::Fcmp(cc, x, y) => {
                self.fcmp(*cc, *x, *y, result()?);
                Ok(())
            }
            _ => self.memory_op(op, result),
        }
    }

    fn trapz(&mut self, condition: Value) {
        if let Some(known) = self.imm_extended(condition, Ext::Zero) {
            if known == 0 {
                self.ud2();
            }
            return;
        }
        let skip = self.asm.label();
        match self.flags.take() {
            Some((value, cond)) if value == condition => self.asm.jcc(cond, skip),
            _ => {
                self.test_value(condition);
                self.asm.jcc(Cond::Ne, skip);
            }
        }
        self.ud2();
        self.asm.bind(skip);
    }

    /// Sets the zero flag from whether `value` is zero, testing its location register in place when it has one.
    fn test_value(&mut self, value: Value) {
        let reg = self.reg_of(value).unwrap_or_else(|| {
            self.get(Reg::Rax, value, Ext::Zero);
            Reg::Rax
        });
        self.asm.test_sized(Size::of(self.ty(value).bytes()), reg, reg);
    }

    /// Whether `a` and `b` live in the same place, so writing one overwrites the other.
    fn same_home(&self, a: Value, b: Value) -> bool {
        a == b || matches!((self.plan.locations[a.index()], self.plan.locations[b.index()]), (Location::Reg(x), Location::Reg(y)) if x == y)
    }

    /// Writes `value` into the location of `param`.
    fn assign(&mut self, param: Value, value: Value) {
        match self.plan.locations[param.index()] {
            Location::Reg(reg) => self.raw(reg, value),
            _ => self.move_to(self.at(param), value),
        }
    }

    /// Loads the location of `value` from the bytes at `from`.
    fn load_home(&mut self, value: Value, from: Mem) {
        match self.plan.locations[value.index()] {
            Location::Reg(reg) => self.asm.load(Size::Q, reg, from),
            _ => self.copy_to(self.at(value), from, self.ty(value)),
        }
    }

    fn edge(&mut self, block: Block, args: &[Value]) -> Result<(), String> {
        let params = &self.func.blocks[block.index()].params;
        let pairs: Vec<(Value, Value)> = args.iter().copied().zip(params.iter().copied()).filter(|(a, p)| !self.same_home(*a, *p)).collect();
        let clash = pairs.iter().any(|(arg, _)| pairs.iter().any(|(_, param)| self.same_home(*arg, *param)));
        if !clash {
            for (arg, param) in pairs {
                self.assign(param, arg);
            }
            return Ok(());
        }
        let base = self.frame.scratch;
        let scratch = |index: usize| Mem::new(Reg::Rbp, base + 16 * index as i32);
        for (index, (arg, _)) in pairs.iter().enumerate() {
            self.move_to(scratch(index), *arg);
        }
        for (index, (_, param)) in pairs.iter().enumerate() {
            self.load_home(*param, scratch(index));
        }
        Ok(())
    }

    fn brif(&mut self, condition: Value, then: (Block, &[Value]), otherwise: (Block, &[Value])) -> Result<(), String> {
        if let Some(known) = self.imm_extended(condition, Ext::Zero) {
            let (block, args) = if known != 0 { then } else { otherwise };
            self.edge(block, args)?;
            self.goto(block, true);
            return Ok(());
        }
        let cond = match self.flags.take() {
            Some((value, cond)) if value == condition => cond,
            _ => {
                self.test_value(condition);
                Cond::Ne
            }
        };
        let (then_moves, else_moves) = (self.moves(then.0, then.1), self.moves(otherwise.0, otherwise.1));
        let falls = self.next.is_some_and(|next| next == then.0 || next == self.target(then.0));
        if !else_moves && (then_moves || falls) {
            self.asm.jcc(cond.invert(), self.blocks[self.target(otherwise.0).index()]);
            self.edge(then.0, then.1)?;
            self.goto(then.0, true);
            return Ok(());
        }
        if !then_moves {
            self.asm.jcc(cond, self.blocks[self.target(then.0).index()]);
            self.edge(otherwise.0, otherwise.1)?;
            self.goto(otherwise.0, true);
            return Ok(());
        }
        let other = self.asm.label();
        self.asm.jcc(cond.invert(), other);
        self.edge(then.0, then.1)?;
        self.goto(then.0, false);
        self.asm.bind(other);
        self.edge(otherwise.0, otherwise.1)?;
        self.goto(otherwise.0, true);
        Ok(())
    }

    /// Whether entering `block` with `args` moves any value.
    fn moves(&self, block: Block, args: &[Value]) -> bool {
        args.iter().zip(&self.func.blocks[block.index()].params).any(|(arg, param)| !self.same_home(*arg, *param))
    }

    fn goto(&mut self, block: Block, last: bool) {
        let target = self.target(block);
        if last && (self.next == Some(block) || self.next == Some(target)) {
            return;
        }
        self.asm.jmp(self.blocks[target.index()]);
    }

    fn target(&self, mut block: Block) -> Block {
        for _ in 0..self.func.blocks.len() {
            let data = &self.func.blocks[block.index()];
            let [only] = data.insts[..] else { return block };
            match &self.func.insts[only.index()].op {
                Op::Jump(next, args) if data.params.is_empty() && args.is_empty() => block = *next,
                _ => return block,
            }
        }
        block
    }

    fn ret(&mut self, values: &[Value]) -> Result<(), String> {
        for (value, loc) in values.iter().zip(self.abi.returns.clone()) {
            self.load_loc(*value, loc);
        }
        let func = self.func;
        let entry = func.entry.map(|block| &func.blocks[block.index()].params);
        let sret = self.func.signature.params.iter().position(|param| param.purpose == ArgumentPurpose::StructReturn);
        if let (Some(params), Some(index)) = (entry, sret) {
            let pointer = params[index];
            self.raw(Reg::Rax, pointer);
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
            Loc::Float(xmm) => self.movsd_load(Xmm(xmm), self.at(value)),
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
            Loc::Float(xmm) => self.movsd_store(self.at(value), Xmm(xmm)),
            Loc::Stack(_) | Loc::Struct(..) => {}
        }
    }

    fn copy_struct(&mut self, pointer: Value, offset: u32, size: u32) {
        self.raw(Reg::Rax, pointer);
        let mut done = 0;
        for (width, bytes) in [(Size::Q, 8), (Size::D, 4), (Size::W, 2), (Size::B, 1)] {
            while size - done >= bytes {
                self.asm.load(width, Reg::Rcx, Mem::new(Reg::Rax, done as i32));
                self.asm.store(width, Mem::new(Reg::Rsp, (offset + done) as i32), Reg::Rcx);
                done += bytes;
            }
        }
    }

    fn arguments(&mut self, sig: SigRef, callee: Value, args: &[Value]) -> Result<Abi, String> {
        let abi = classify(&self.func.sigs[sig.index()])?;
        for (arg, loc) in args.iter().zip(&abi.params) {
            match *loc {
                Loc::Stack(offset) => self.move_to(Mem::new(Reg::Rsp, offset as i32), *arg),
                Loc::Struct(offset, size) => self.copy_struct(*arg, offset, size),
                _ => {}
            }
        }
        for (arg, loc) in args.iter().zip(&abi.params) {
            self.load_loc(*arg, *loc);
        }
        if let Some(at) = self.plan.through[callee.index()] {
            self.asm.call_absolute(at);
            return Ok(abi);
        }
        let target = self.reg_of(callee).unwrap_or(Reg::R11);
        self.raw(target, callee);
        self.asm.call(target);
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
        self.asm.jmp(self.blocks[normal.index()]);
        let pad = self.asm.pos();
        let func = self.func;
        let [exception] = func.blocks[handler.index()].params[..] else {
            return Err("a try_call handler takes exactly the exception".to_owned());
        };
        self.put(exception, Reg::Rax);
        self.asm.jmp(self.blocks[handler.index()]);
        self.sites.push(Site { ret, handler: pad, frame_offset: self.frame.size });
        Ok(())
    }
}
