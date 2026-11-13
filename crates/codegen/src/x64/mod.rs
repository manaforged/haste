mod asm;
mod atomic;
mod features;
mod float;
mod frame;
mod int;
mod location;
mod vector;
mod wide;

use asm::{Alu, Asm, Cond, Label, Mem, Reg, Size, Xmm};
use features::Features;
use frame::{Abi, Loc, SAVED, classify};
use crate::alloc::{CallTarget, Location, Plan, Regs};
use crate::frame::{ArgStep, Emitter, Ext, Frame, INCOMING, PAGE, Prepared, ShrinkWrap, Step, arg_steps, drive, edge_steps, emit_body, prepare, reg_steps, save_offset, shrink_wrap};
use crate::ir::{ArgumentPurpose, Block, Function, Inst, Op, SigRef, Type, Value};
use crate::park::ParkReg;
use crate::reloc::{DictTierUp, RelocKind, StencilReloc};
use crate::target::{CallAbi, Target};
use crate::{Code, Site};

const REGS: Regs<Reg> = Regs {
    saved: &SAVED,
    saved_float: &[],
    leaf: &[Reg::Rsi, Reg::Rdi, Reg::R9, Reg::R8, Reg::R10],
    wide_scratch: &wide::WIDE_SCRATCH,
    op_scratch: &[Reg::Rsi],
    uses_scratch: atomic::uses_rsi,
    dict: Some(Reg::R10),
    float: &float::POOL,
    float_branches: &float::BRANCHES,
    vectors: true,
    absolute_calls: true,
    two_address: true,
    park: true,
    fold_sib: true,
};

const REGS_WIN64: Regs<Reg> = Regs {
    leaf: &[Reg::R9, Reg::R8, Reg::R10],
    ..REGS
};

pub(crate) fn compile(func: &Function, target: &Target) -> Result<Code, String> {
    let abi_kind = target.abi;
    let regs = if abi_kind == CallAbi::Win64 { &REGS_WIN64 } else { &REGS };
    let Prepared { order, copied, abi, plan, frame, framed } = prepare(func, regs, target.effort == crate::Effort::Full, |sig| classify(sig, abi_kind))?;
    let shrink = shrink_wrap(func, framed, &plan, &frame, &abi.params, &[Reg::Rbp, Reg::Rsp, Reg::R11]);
    let mut asm = Asm::default();
    let blocks = func.blocks.iter().map(|_| asm.label()).collect();
    let slow_thunk = asm.label();
    let emit = Emit {
        func,
        frame,
        framed,
        abi,
        abi_kind,
        asm,
        blocks,
        sites: Vec::new(),
        relocs: Vec::new(),
        plan,
        flags: None,
        next: None,
        features: Features::host(),
        copied,
        direct_calls: target.direct_calls,
        shrink,
        slow_thunk,
        in_shrink_entry: false,
        in_fast_ret: false,
    };
    drive(emit, &order)
}

pub(crate) fn dict_stub(stub_exec: usize, body_exec: usize, dict_addr: usize, tier_up: Option<DictTierUp>) -> Result<Vec<u8>, String> {
    let mut asm = Asm::default();
    if let Some(tier) = tier_up {
        let fast = asm.label();
        asm.mov_abs64_val(Reg::R10, tier.counter_addr as u64);
        asm.encode(&[], false, &[0x83], Alu::Add as u8, Mem::new(Reg::R10, 0).into(), false);
        asm.bytes(&[1]);
        asm.encode(&[], false, &[0x81], Alu::Cmp as u8, Mem::new(Reg::R10, 0).into(), false);
        asm.bytes(&tier.threshold.to_le_bytes());
        asm.jcc(Cond::Ne, fast);
        asm.mov_imm(Reg::R11, i64::from(tier.encoded_index));
        asm.mov_abs64_val(Reg::R10, tier.stub_entry as u64);
        asm.jmp_reg(Reg::R10);
        asm.bind(fast);
    }
    asm.mov_abs64_val(Reg::R10, dict_addr as u64);
    let rel = body_exec as i64 - (stub_exec as i64 + i64::from(asm.pos()) + 5);
    if let Ok(rel32) = i32::try_from(rel) {
        asm.bytes(&[0xE9]);
        asm.bytes(&rel32.to_le_bytes());
    } else {
        asm.mov_abs64_val(Reg::R11, body_exec as u64);
        asm.jmp_reg(Reg::R11);
    }
    asm.finish()
}

pub(super) struct Emit<'a> {
    func: &'a Function,
    frame: Frame<Reg>,
    framed: bool,
    abi: Abi,
    abi_kind: CallAbi,
    asm: Asm,
    blocks: Vec<Label>,
    sites: Vec<Site>,
    relocs: Vec<StencilReloc>,
    plan: Plan<Reg>,
    flags: Option<(Value, Cond)>,
    next: Option<Block>,
    features: Features,
    copied: Vec<bool>,
    direct_calls: bool,
    shrink: Option<ShrinkWrap>,
    slow_thunk: Label,
    in_shrink_entry: bool,
    in_fast_ret: bool,
}

impl<'a> Emitter<'a> for Emit<'a> {
    fn func(&self) -> &'a Function {
        self.func
    }

    fn is_elided(&self, value: Value) -> bool {
        self.plan.elided[value.index()]
    }

    fn prologue(&mut self) {
        if !self.framed {
            return;
        }
        self.asm.push(Reg::Rbp);
        self.asm.mov(Reg::Rbp, Reg::Rsp);
        let size = self.frame.size;
        if size > PAGE {
            self.probe(size);
        } else if size > 0 {
            self.asm.alu_imm(Alu::Sub, Reg::Rsp, size as i32);
        }
        for (index, &reg) in self.frame.saves.iter().enumerate() {
            self.asm.store(Size::Q, Mem::new(Reg::Rbp, save_offset(index)), reg);
        }
    }

    fn intake(&mut self, entry: Block) -> Result<(), String> {
        let func = self.func;
        let params = func.blocks[entry.index()].params.iter().copied().zip(self.abi.params.clone());
        let (registers, memory): (Vec<_>, Vec<_>) = params.partition(|(value, _)| self.reg_of(*value).is_some() || self.xmm_of(*value).is_some());
        for (value, loc) in memory.into_iter().chain(registers) {
            let incoming = |offset: u32| Mem::new(Reg::Rbp, INCOMING + offset as i32);
            match loc {
                Loc::Stack(offset) => self.load_home(value, incoming(offset)),
                Loc::Struct(offset, _) => {
                    self.asm.lea(Reg::Rax, incoming(offset));
                    self.put(value, Reg::Rax);
                }
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

    fn deferred_prologue(&self) -> bool {
        self.shrink.is_some()
    }

    fn blocks_fallthrough_to(&self, block: Block) -> bool {
        self.shrink.is_some_and(|s| s.use_slow_thunk && s.slow_entry == block)
    }

    fn before_block_bind(&mut self, block: Block) -> Result<(), String> {
        if let Some(shrink) = self.shrink
            && shrink.use_slow_thunk
            && block == shrink.slow_entry
        {
            self.asm.bind(self.slow_thunk);
            self.emit_slow_prologue_and_edge(shrink.slow_entry)?;
        }
        Ok(())
    }

    fn on_block_bound(&mut self, block: Block) -> Result<(), String> {
        self.in_shrink_entry = self.shrink.is_some() && self.func.entry == Some(block);
        self.in_fast_ret = self.shrink.is_some_and(|s| s.fast_ret == block);
        if let Some(shrink) = self.shrink
            && !shrink.use_slow_thunk
            && block == shrink.slow_entry
        {
            self.emit_slow_prologue_and_edge(shrink.slow_entry)?;
        }
        Ok(())
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
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                self.brif(*condition, (*then, then_args), (*otherwise, else_args))
            }
            Op::BrTable(index, default, targets) => {
                self.br_table(*index, *default, targets);
                Ok(())
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

    fn trap(&mut self) {
        self.ud2();
    }

    fn finish(self) -> Result<Code, String> {
        Ok(Code { bytes: self.asm.finish()?, sites: self.sites, relocs: self.relocs })
    }
}

impl Emit<'_> {
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
        for (index, &reg) in self.frame.saves.iter().enumerate() {
            self.asm.load(Size::Q, reg, Mem::new(Reg::Rbp, save_offset(index)));
        }
        self.asm.bytes(if self.framed { &[0xC9, 0xC3] } else { &[0xC3] });
    }

    fn copy_to(&mut self, to: Mem, from: Mem, ty: Type) {
        self.asm.load(Size::Q, Reg::Rax, from);
        self.asm.store(Size::Q, to, Reg::Rax);
        if ty.bytes() == 16 {
            self.asm.load(Size::Q, Reg::Rax, from.offset(8));
            self.asm.store(Size::Q, to.offset(8), Reg::Rax);
        }
    }

    value_op!();

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

    fn test_value(&mut self, value: Value) {
        let reg = self.reg_of(value).unwrap_or_else(|| {
            self.get(Reg::Rax, value, Ext::Zero);
            Reg::Rax
        });
        self.asm.test_sized(Size::of(self.ty(value).bytes()), reg, reg);
    }

    fn assign(&mut self, param: Value, value: Value) {
        match self.plan.locations[param.index()] {
            Location::Reg(reg) => self.raw(reg, value),
            Location::Float(xmm) => self.fget(Xmm(xmm), value),
            _ => self.move_to(self.at(param), value),
        }
    }

    fn load_home(&mut self, value: Value, from: Mem) {
        match self.plan.locations[value.index()] {
            Location::Reg(reg) => self.asm.load(Size::Q, reg, from),
            Location::Float(xmm) => self.xmm_load(self.ty(value), Xmm(xmm), from),
            _ => self.copy_to(self.at(value), from, self.ty(value)),
        }
    }

    fn edge(&mut self, block: Block, args: &[Value]) -> Result<(), String> {
        for step in edge_steps(args, &self.func.blocks[block.index()].params, |a, b| self.plan.same_home(&self.frame, a, b)) {
            match step {
                Step::Move(arg, param) => self.assign(param, arg),
                Step::Save(arg, index) => self.move_to(Mem::new(Reg::Rbp, self.frame.edge_slot(index)), arg),
                Step::Restore(index, param) => self.load_home(param, Mem::new(Reg::Rbp, self.frame.edge_slot(index))),
            }
        }
        Ok(())
    }

    fn emit_slow_prologue_and_edge(&mut self, slow_entry: Block) -> Result<(), String> {
        self.prologue();
        if let Some(entry) = self.func.entry
            && let Some(term) = self.func.blocks[entry.index()].insts.last()
            && let Op::Brif(_, then_b, then_args, else_b, else_args) = &self.func.insts[term.index()].op
        {
            let args = if *then_b == slow_entry { then_args.as_slice() } else if *else_b == slow_entry { else_args.as_slice() } else { &[] };
            self.edge(slow_entry, args)?;
        }
        Ok(())
    }

    fn target_label(&self, block: Block) -> Label {
        if self.in_shrink_entry && self.shrink.is_some_and(|s| s.use_slow_thunk && s.slow_entry == block) {
            self.slow_thunk
        } else {
            self.blocks[self.func.jump_target(block).index()]
        }
    }

    fn goto_branch(&mut self, block: Block, last: bool) {
        if self.in_shrink_entry && self.shrink.is_some_and(|s| s.use_slow_thunk && s.slow_entry == block) {
            self.asm.jmp(self.slow_thunk);
        } else {
            self.goto(block, last);
        }
    }

    fn brif(&mut self, condition: Value, then: (Block, &[Value]), otherwise: (Block, &[Value])) -> Result<(), String> {
        let (then, otherwise) = if self.in_shrink_entry {
            let slow = self.shrink.map(|s| s.slow_entry);
            (
                (then.0, if Some(then.0) == slow { &[][..] } else { then.1 }),
                (otherwise.0, if Some(otherwise.0) == slow { &[][..] } else { otherwise.1 }),
            )
        } else {
            (then, otherwise)
        };
        if let Some(known) = self.imm_extended(condition, Ext::Zero) {
            let (block, args) = if known != 0 { then } else { otherwise };
            self.edge(block, args)?;
            self.goto_branch(block, true);
            return Ok(());
        }
        let cond = match self.flags.take() {
            Some((value, cond)) if value == condition => cond,
            _ => {
                self.test_value(condition);
                Cond::Ne
            }
        };
        let (then_moves, else_moves) = (
            self.plan.moves(self.func, &self.frame, then.0, then.1),
            self.plan.moves(self.func, &self.frame, otherwise.0, otherwise.1),
        );
        let falls = self.next.is_some_and(|next| next == then.0 || next == self.func.jump_target(then.0));
        if !else_moves && (then_moves || falls) {
            let target = self.target_label(otherwise.0);
            self.asm.jcc(cond.invert(), target);
            self.edge(then.0, then.1)?;
            self.goto_branch(then.0, true);
            return Ok(());
        }
        if !then_moves {
            let target = self.target_label(then.0);
            self.asm.jcc(cond, target);
            self.edge(otherwise.0, otherwise.1)?;
            self.goto_branch(otherwise.0, true);
            return Ok(());
        }
        let other = self.asm.label();
        self.asm.jcc(cond.invert(), other);
        self.edge(then.0, then.1)?;
        self.goto_branch(then.0, false);
        self.asm.bind(other);
        self.edge(otherwise.0, otherwise.1)?;
        self.goto_branch(otherwise.0, true);
        Ok(())
    }

    fn goto(&mut self, block: Block, last: bool) {
        let target = self.func.jump_target(block);
        if last && (self.next == Some(block) || self.next == Some(target)) {
            return;
        }
        self.asm.jmp(self.blocks[target.index()]);
    }

    fn br_table(&mut self, index: Value, default: Block, targets: &[Block]) {
        if targets.is_empty() {
            self.goto(default, true);
            return;
        }
        self.get(Reg::Rax, index, Ext::Zero);
        self.asm.mov_imm(Reg::Rcx, targets.len() as i64);
        self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
        self.asm.jcc(Cond::Ae, self.blocks[self.func.jump_target(default).index()]);
        let table = self.asm.label();
        self.asm.lea_rip(Reg::Rcx, table);
        self.asm.load_signed(Size::D, Reg::R11, Mem::sib(Reg::Rcx, Reg::Rax, 2, 0));
        self.asm.alu(Alu::Add, Reg::Rcx, Reg::R11);
        self.asm.jmp_reg(Reg::Rcx);
        while self.asm.pos() & 3 != 0 {
            self.asm.bytes(&[0xCC]);
        }
        self.asm.bind(table);
        for &target in targets {
            self.asm.table_entry(table, self.blocks[self.func.jump_target(target).index()]);
        }
    }

    fn ret(&mut self, values: &[Value]) -> Result<(), String> {
        let returns = self.abi.returns.clone();
        self.move_locs(values, &returns, Reg::R11);
        let func = self.func;
        let entry = func.entry.map(|block| &func.blocks[block.index()].params);
        let sret = self.func.signature.params.iter().position(|param| param.purpose == ArgumentPurpose::StructReturn);
        if let (Some(params), Some(index)) = (entry, sret) {
            let pointer = params[index];
            self.raw(Reg::Rax, pointer);
        }
        if self.in_fast_ret {
            self.asm.bytes(&[0xC3]);
        } else {
            self.epilogue();
        }
        Ok(())
    }

    fn move_locs(&mut self, values: &[Value], locs: &[Loc], scratch: Reg) -> usize {
        let float_targets: Vec<(Value, u8)> = values.iter().zip(locs).filter_map(|(&val, &loc)| if let Loc::Float(xmm) = loc { Some((val, xmm)) } else { None }).collect();
        for step in arg_steps(&float_targets, |v| self.plan.float_of(v)) {
            match step {
                ArgStep::Move(src, dst) => self.asm.movaps(Xmm(dst), Xmm(src)),
                ArgStep::Save(src) => {
                    self.asm.alu_imm(Alu::Sub, Reg::Rsp, 16);
                    self.xmm_store(Type::V128, Mem::new(Reg::Rsp, 0), Xmm(src));
                }
                ArgStep::Restore(dst) => {
                    self.xmm_load(Type::V128, Xmm(dst), Mem::new(Reg::Rsp, 0));
                    self.asm.alu_imm(Alu::Add, Reg::Rsp, 16);
                }
                ArgStep::Load(val, dst) => self.fget(Xmm(dst), val),
            }
        }
        let int_targets: Vec<(Value, Reg)> = values.iter().zip(locs).filter_map(|(&val, &loc)| if let Loc::Int(reg) = loc { Some((val, reg)) } else { None }).collect();
        for step in arg_steps(&int_targets, |v| self.reg_of(v)) {
            match step {
                ArgStep::Move(src, dst) => self.asm.mov(dst, src),
                ArgStep::Save(src) => self.asm.mov(scratch, src),
                ArgStep::Restore(dst) => self.asm.mov(dst, scratch),
                ArgStep::Load(val, dst) => self.raw(dst, val),
            }
        }
        for (&val, &loc) in values.iter().zip(locs) {
            if let Loc::Pair(low, high) = loc {
                self.raw(low, val);
                self.high(high, val);
            }
        }
        float_targets.len()
    }

    fn store_locs(&mut self, values: &[Value], locs: &[Loc]) {
        let mut float_pairs = Vec::new();
        let mut int_pairs = Vec::new();
        for (&value, &loc) in values.iter().zip(locs) {
            match (loc, self.plan.location(value)) {
                (Loc::Float(src), Location::Float(dst)) => float_pairs.push((src, dst)),
                (Loc::Int(src), Location::Reg(dst)) => int_pairs.push((src, dst)),
                _ => self.store_loc(value, loc),
            }
        }
        for step in reg_steps(float_pairs) {
            match step {
                ArgStep::Move(src, dst) => self.asm.movaps(Xmm(dst), Xmm(src)),
                ArgStep::Save(src) => {
                    self.asm.alu_imm(Alu::Sub, Reg::Rsp, 16);
                    self.xmm_store(Type::V128, Mem::new(Reg::Rsp, 0), Xmm(src));
                }
                ArgStep::Restore(dst) => {
                    self.xmm_load(Type::V128, Xmm(dst), Mem::new(Reg::Rsp, 0));
                    self.asm.alu_imm(Alu::Add, Reg::Rsp, 16);
                }
                ArgStep::Load(..) => {}
            }
        }
        for step in reg_steps(int_pairs) {
            match step {
                ArgStep::Move(src, dst) => self.asm.mov(dst, src),
                ArgStep::Save(src) => self.asm.mov(Reg::R11, src),
                ArgStep::Restore(dst) => self.asm.mov(dst, Reg::R11),
                ArgStep::Load(..) => {}
            }
        }
    }

    fn store_loc(&mut self, value: Value, loc: Loc) {
        match loc {
            Loc::Int(reg) => self.put(value, reg),
            Loc::Pair(low, high) => {
                self.put(value, low);
                self.put_high(value, high);
            }
            Loc::Float(xmm) => self.fput(value, Xmm(xmm)),
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
        let abi = classify(&self.func.sigs[sig.index()], self.abi_kind)?;
        for (arg, loc) in args.iter().zip(&abi.params) {
            match *loc {
                Loc::Stack(offset) => self.move_to(Mem::new(Reg::Rsp, offset as i32), *arg),
                Loc::Struct(offset, size) => self.copy_struct(*arg, offset, size),
                _ => {}
            }
        }
        let direct = self.plan.through[callee.index()];
        let target = if direct.is_none() {
            match self.reg_of(callee).filter(|reg| SAVED.contains(reg)) {
                Some(reg) => reg,
                None => {
                    self.raw(Reg::R11, callee);
                    Reg::R11
                }
            }
        } else {
            Reg::R11
        };
        let float_count = self.move_locs(args, &abi.params, Reg::Rax);
        if self.abi_kind == CallAbi::SysV && self.func.sigs[sig.index()].params.iter().any(|param| param.purpose == ArgumentPurpose::Variadic) {
            self.asm.mov_imm(Reg::Rax, float_count as i64);
        }
        match direct {
            Some(CallTarget::Abs(at)) => self.asm.call_absolute(at),
            Some(CallTarget::Slot(sym)) if self.direct_calls => {
                let at = self.asm.call_pcrel32();
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::PcRel32));
            }
            Some(CallTarget::Slot(sym)) => {
                let at = self.asm.call_slot_reloc();
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Abs32));
            }
            None => self.asm.call(target),
        }
        Ok(abi)
    }

    fn call(&mut self, inst: Inst, (sig, callee): (SigRef, Value), args: &[Value], results: &[Value]) -> Result<(), String> {
        let parked = self.plan.parked_at(inst).to_vec();
        for &(value, reg, _) in &parked {
            let home = self.at(value);
            match reg {
                ParkReg::Int(r) => self.asm.store(Size::Q, home, r),
                ParkReg::Float(xmm) => self.xmm_store(self.ty(value), home, Xmm(xmm)),
            }
            self.plan.locations[value.index()] = Location::Slot;
        }
        let abi = self.arguments(sig, callee, args)?;
        self.store_locs(results, &abi.returns);
        for (value, reg, lives) in parked {
            if lives {
                let home = self.at(value);
                match reg {
                    ParkReg::Int(r) => self.asm.load(Size::Q, r, home),
                    ParkReg::Float(xmm) => self.xmm_load(self.ty(value), Xmm(xmm), home),
                }
            }
            self.plan.locations[value.index()] = match reg {
                ParkReg::Int(r) => Location::Reg(r),
                ParkReg::Float(xmm) => Location::Float(xmm),
            };
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
