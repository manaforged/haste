mod asm;
mod atomic;
mod features;
mod float;
mod frame;
mod int;
mod location;
mod vector;
mod vector_lanes;
mod wide;

use asm::{Alu, Asm, Cond, Label, Mem, Reg, Size, Xmm};
use features::Features;
use frame::{Abi, Loc, SAVED, classify};
use crate::alloc::{CallTarget, Location, Plan, Regs};
use crate::emit::{Backend, emit_brif, emit_call, emit_goto, emit_slow_prologue, emit_try_call, intake_params};
use crate::frame::{ArgStep, Emitter, Ext, Frame, INCOMING, PAGE, Prepared, ShrinkWrap, Step, arg_steps, drive, edge_steps, emit_body, prepare, reg_steps, save_offset, shrink_wrap};
use crate::ir::{ArgumentPurpose, Block, Function, Inst, Lane, Op, SigRef, Type, Value};
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
    let mut defs = vec![None; func.types.len()];
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Some(res) = func.insts[inst.index()].result {
            defs[res.index()] = Some(inst);
        }
    }
    let last_block = order.last().copied();
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
        defs,
        flags: None,
        last_spill: None,
        last_fspill: None,
        next: None,
        last_block,
        at_end: false,
        trap_label: None,
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
    defs: Vec<Option<Inst>>,
    flags: Option<(Value, Cond)>,
    last_spill: Option<(i32, Reg, (usize, u32))>,
    last_fspill: Option<(i32, Xmm, Type, (usize, u32))>,
    next: Option<Block>,
    last_block: Option<Block>,
    at_end: bool,
    trap_label: Option<Label>,
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
        intake_params(self, entry)
    }

    fn set_next(&mut self, next: Option<Block>) {
        self.next = next;
    }

    fn bind_block(&mut self, block: Block) {
        self.last_spill = None;
        self.last_fspill = None;
        self.at_end = self.last_block == Some(block);
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
            emit_slow_prologue(self, shrink.slow_entry)?;
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
            emit_slow_prologue(self, shrink.slow_entry)?;
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
            Op::Call { sig, callee, args, results } => emit_call(self, inst, (*sig, *callee), args, results),
            Op::TryCall { sig, callee, args, normal, handler } => emit_try_call(self, (*sig, *callee), args, (*normal, *handler)),
            Op::Jump(block, args) => {
                self.edge(*block, args)?;
                if self.copied[inst.index()] {
                    return emit_body(self, *block);
                }
                emit_goto(self, *block, true);
                Ok(())
            }
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                emit_brif(self, *condition, (*then, then_args), (*otherwise, else_args))
            }
            Op::BrTable(index, default, targets) => {
                self.br_table(*index, *default, targets);
                Ok(())
            }
            Op::Return(values) => self.ret(values),
            Op::Trap => {
                self.emit_trap(self.at_end);
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
        self.emit_trap(self.at_end);
    }

    fn finish(mut self) -> Result<Code, String> {
        if let Some(label) = self.trap_label {
            self.asm.bind(label);
            self.ud2();
        }
        Ok(Code { bytes: self.asm.finish()?, sites: self.sites, relocs: self.relocs })
    }
}

impl Emit<'_> {
    fn at(&self, value: Value) -> Mem {
        let disp = self.frame.values[value.index()];
        debug_assert_ne!(disp, 0, "unhomed value {value:?} in at()");
        Mem::new(Reg::Rbp, disp)
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

    fn trap_label(&mut self) -> Label {
        *self.trap_label.get_or_insert_with(|| self.asm.label())
    }

    fn emit_trap(&mut self, fallthrough: bool) {
        let label = self.trap_label();
        if !fallthrough {
            self.asm.jmp(label);
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
        } else if to.base == Reg::Rbp && to.index.is_none() {
            self.last_spill = Some((to.disp, Reg::Rax, self.asm.stamp()));
        }
    }

    value_op!();

    fn trapz(&mut self, condition: Value) {
        if let Some(known) = self.imm_extended(condition, Ext::Zero) {
            if known == 0 {
                self.emit_trap(false);
            }
            return;
        }
        let trap = self.trap_label();
        let cond = match self.flags.take() {
            Some((value, cond)) if value == condition => cond.invert(),
            _ => {
                self.test_value(condition);
                Cond::E
            }
        };
        self.asm.jcc(cond, trap);
    }

    fn test_value(&mut self, value: Value) {
        let reg = self.reg_of(value).or_else(|| self.cached_spill(value)).unwrap_or_else(|| {
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

    fn br_table(&mut self, index: Value, default: Block, targets: &[Block]) {
        if targets.is_empty() {
            emit_goto(self, default, true);
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
                    self.xmm_store(Type::Vec { lane: Lane::I64, count: 2 }, Mem::new(Reg::Rsp, 0), Xmm(src));
                }
                ArgStep::Restore(dst) => {
                    self.xmm_load(Type::Vec { lane: Lane::I64, count: 2 }, Xmm(dst), Mem::new(Reg::Rsp, 0));
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
                    self.xmm_store(Type::Vec { lane: Lane::I64, count: 2 }, Mem::new(Reg::Rsp, 0), Xmm(src));
                }
                ArgStep::Restore(dst) => {
                    self.xmm_load(Type::Vec { lane: Lane::I64, count: 2 }, Xmm(dst), Mem::new(Reg::Rsp, 0));
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
}

impl<'a> Backend<'a, Reg, Label> for Emit<'a> {
    fn plan(&self) -> &Plan<Reg> { &self.plan }
    fn plan_mut(&mut self) -> &mut Plan<Reg> { &mut self.plan }
    fn frame(&self) -> &Frame<Reg> { &self.frame }
    fn abi(&self) -> &Abi { &self.abi }
    fn shrink(&self) -> Option<ShrinkWrap> { self.shrink }
    fn slow_thunk(&self) -> Label { self.slow_thunk }
    fn in_shrink_entry(&self) -> bool { self.in_shrink_entry }
    fn next_block(&self) -> Option<Block> { self.next }
    fn block_label(&self, block: Block) -> Label { self.blocks[block.index()] }
    fn pos(&self) -> u32 { self.asm.pos() }
    fn new_label(&mut self) -> Label { self.asm.label() }
    fn bind_label(&mut self, label: Label) { self.asm.bind(label); }
    fn jump_label(&mut self, label: Label) { self.asm.jmp(label); }
    fn branch_cond(&mut self, condition: Value, invert: bool, label: Label) {
        let cond = match self.flags.take() {
            Some((value, cond)) if value == condition => cond,
            _ => {
                self.test_value(condition);
                Cond::Ne
            }
        };
        self.asm.jcc(if invert { cond.invert() } else { cond }, label);
    }
    fn edge(&mut self, block: Block, args: &[Value]) -> Result<(), String> { self.edge(block, args) }
    fn arguments(&mut self, sig: SigRef, callee: Value, args: &[Value]) -> Result<Abi, String> { self.arguments(sig, callee, args) }
    fn store_locs(&mut self, values: &[Value], locs: &[Loc]) { self.store_locs(values, locs); }
    fn store_loc(&mut self, value: Value, loc: Loc) { self.store_loc(value, loc); }
    fn load_incoming(&mut self, value: Value, offset: u32) { self.load_home(value, Mem::new(Reg::Rbp, INCOMING + offset as i32)); }
    fn intake_struct(&mut self, value: Value, offset: u32) {
        self.asm.lea(Reg::Rax, Mem::new(Reg::Rbp, INCOMING + offset as i32));
        self.put(value, Reg::Rax);
    }
    fn park_spill(&mut self, value: Value, reg: ParkReg<Reg>) {
        let home = self.at(value);
        match reg {
            ParkReg::Int(r) => self.asm.store(Size::Q, home, r),
            ParkReg::Float(xmm) => self.xmm_store(self.ty(value), home, Xmm(xmm)),
        }
    }
    fn park_reload(&mut self, value: Value, reg: ParkReg<Reg>) {
        let home = self.at(value);
        match reg {
            ParkReg::Int(r) => self.asm.load(Size::Q, r, home),
            ParkReg::Float(xmm) => self.xmm_load(self.ty(value), Xmm(xmm), home),
        }
    }
    fn put_exception(&mut self, value: Value) { self.put(value, Reg::Rax); }
    fn push_site(&mut self, site: Site) { self.sites.push(site); }
}
