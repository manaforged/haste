mod asm;
mod atomic;
mod features;
mod float;
mod frame;
mod imm;
mod int;
mod location;
mod vector;
mod vector_lanes;
mod vector_narrow;
mod wide;
mod ymm;

use asm::{Alu, Asm, Cond, Label, Mem, Reg, Rm, Shift, Size, Xmm};
use features::Features;
pub(crate) use ymm::host_avx2;
use frame::{Abi, Loc, SAVED, SAVED_RBP, classify};
use crate::alloc::{CallTarget, Location, Plan, Regs};
use crate::emit::{Backend, Tails, emit_brif, emit_call, emit_goto, emit_slow_prologue, emit_stubs, emit_try_call, intake_params};
use crate::frame::{ArgStep, EdgeScratch, Emitter, Ext, Frame, INCOMING, PAGE, Prepared, ShrinkWrap, Step, arg_steps, drive, edge_steps, emit_body, prepare, reg_steps, save_offset, shrink_wrap};
use crate::ir::{ArgumentPurpose, Binary, Block, Function, Inst, Op, SigRef, TargetOpId, Type, Value};
use crate::park::ParkReg;
use crate::reloc::{DictTierUp, RelocKind, StencilReloc};
use crate::target::{CallAbi, Target};
use crate::{Code, Site};

const REGS: Regs<Reg> = Regs {
    saved: &SAVED,
    saved_float: &[],
    leaf: &[Reg::Rsi, Reg::Rdi, Reg::R9, Reg::R8, Reg::R10, Reg::Rdx, Reg::R11],
    wide_scratch: &wide::WIDE_SCRATCH,
    op_scratch: &[Reg::Rsi, Reg::Rdx, Reg::R11],
    uses_scratch: atomic::uses_scratch,
    dict: Some(Reg::R10),
    float: &float::POOL,
    float_branches: &float::BRANCHES,
    vectors: true,
    absolute_calls: true,
    two_address: true,
    park: true,
    fold_sib: true,
    call_zext: false,
    imm_store,
};

/// One `mov` stores a sign-extended 32-bit immediate; a wider one goes through a register.
fn imm_store(imm: i64) -> u64 {
    if i32::try_from(imm).is_ok() { 1 } else { 2 }
}

const REGS_WIN64: Regs<Reg> = Regs {
    leaf: &[Reg::R9, Reg::R8, Reg::R10, Reg::Rdx, Reg::R11],
    ..REGS
};

const REGS_RBP: Regs<Reg> = Regs { saved: &SAVED_RBP, ..REGS };

const REGS_WIN64_RBP: Regs<Reg> = Regs { saved: &SAVED_RBP, ..REGS_WIN64 };

const INTAKE_XMM: Xmm = Xmm(15);

/// The registers that frame setup and the probe use, which no shrink-wrapped entry may hold a value in.
const FRAME_REGS: [Reg; 3] = [Reg::Rbp, Reg::Rsp, Reg::R11];

/// The masks and shifts of the bit-parallel steps that reverse and count the bits of a word.
const BIT_STEPS: [(u64, u8); 3] = [(0x5555_5555_5555_5555, 1), (0x3333_3333_3333_3333, 2), (0x0F0F_0F0F_0F0F_0F0F, 4)];

pub(crate) fn compile(func: &Function, target: &Target) -> Result<Code, String> {
    let abi_kind = target.abi;
    let keeps = keeps_frame_pointer(func);
    let regs = match (abi_kind, keeps) {
        (CallAbi::Win64, true) => &REGS_WIN64,
        (CallAbi::Win64, false) => &REGS_WIN64_RBP,
        (_, true) => &REGS,
        (_, false) => &REGS_RBP,
    };
    let Prepared { order, copied, abi, mut plan, mut frame, mut framed } = prepare(func, regs, target.effort == crate::Effort::Full, |sig| classify(sig, abi_kind))?;
    let features = Features::host();
    let elided_before = plan.elided.iter().filter(|&&e| e).count();
    let roots = location::fuse(func, &mut plan, &features);
    if plan.elided.iter().filter(|&&e| e).count() != elided_before {
        crate::frame::repack_after_elide(func, &plan, &abi, &mut frame, &mut framed, |sig| classify(sig, abi_kind))?;
    }
    let shrink = shrink_wrap(func, framed, &plan, &frame, &abi.params, &FRAME_REGS);
    let (base, bias) = if framed && !keeps { (Reg::Rsp, frame.size as i32) } else { (Reg::Rbp, 0) };
    let mut asm = Asm::default();
    asm.avx = features.avx;
    let blocks = func.blocks.iter().map(|_| asm.label()).collect();
    let slow_thunk = asm.label();
    let defs = crate::fold::defs(func);
    let overflow_read = wide::overflow_reads(func, &defs);
    let last_block = order.last().copied();
    let emit = Emit {
        func,
        frame,
        framed,
        base,
        bias,
        abi,
        abi_kind,
        asm,
        blocks,
        sites: Vec::new(),
        relocs: Vec::new(),
        plan,
        defs,
        roots,
        flags: None,
        last_spill: [None; 4],
        last_fspill: None,
        next: None,
        last_block,
        at_end: false,
        trap_label: None,
        features,
        copied,
        direct_calls: target.direct_calls,
        shrink,
        slow_thunk,
        in_shrink_entry: false,
        in_fast_ret: false,
        low_product: None,
        overflow_read,
        carry_flags: None,
        tails: Tails::new(),
    };
    drive(emit, &order)
}

/// Whether a framed function addresses its frame from rbp: it reads the frame pointer, catches an unwind or holds a float or vector, whose lowering moves rsp inside the body.
fn keeps_frame_pointer(func: &Function) -> bool {
    func.types.iter().any(|ty| ty.in_float_register())
        || func.blocks.iter().flat_map(|block| &block.insts).any(|inst| matches!(func.insts[inst.index()].op, Op::TryCall { .. } | Op::FramePointer))
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
    base: Reg,
    bias: i32,
    abi: Abi,
    abi_kind: CallAbi,
    asm: Asm,
    blocks: Vec<Label>,
    sites: Vec<Site>,
    relocs: Vec<StencilReloc>,
    plan: Plan<Reg>,
    defs: Vec<Option<Inst>>,
    roots: Vec<Option<location::Root>>,
    flags: Option<(Value, Cond)>,
    last_spill: [Option<(i32, Reg)>; 4],
    last_fspill: Option<(i32, Xmm, Type)>,
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
    low_product: Option<(Value, Value)>,
    overflow_read: Vec<bool>,
    carry_flags: Option<(bool, [Value; 3])>,
    tails: Tails<'a, Label>,
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
        let size = if self.base == Reg::Rsp {
            self.frame.size + 8
        } else {
            self.asm.push(Reg::Rbp);
            self.asm.mov(Reg::Rbp, Reg::Rsp);
            self.frame.size
        };
        if size > PAGE {
            self.probe(size);
        } else if size > 0 {
            self.asm.alu_imm(Alu::Sub, Reg::Rsp, size as i32);
        }
        for (index, &reg) in self.frame.saves.iter().enumerate() {
            self.asm.store(Size::Q, self.frame_mem(save_offset(index)), reg);
        }
    }

    fn intake(&mut self, entry: Block) -> Result<(), String> {
        intake_params(self, entry)
    }

    fn set_next(&mut self, next: Option<Block>) {
        self.next = next;
    }

    fn bind_block(&mut self, block: Block) {
        self.invalidate_spills();
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
        let carry_flags = self.carry_flags.take();
        if matches!(data.op, Op::TargetOp(TargetOpId::X64AddCarryOut | TargetOpId::X64SubBorrowOut, _)) {
            self.carry_flags = carry_flags;
        }
        if let (Some(low), &Op::Binary(Binary::Imul, x, y), Some(result)) = (self.low_product.take(), &data.op, result)
            && (low == (x, y) || low == (y, x))
            && !(self.is_static(result) || self.plan.deferred[result.index()] || self.plan.elided[result.index()])
        {
            self.flags = None;
            self.put(result, Reg::Rax);
            return Ok(());
        }
        match &data.op {
            Op::Call { sig, callee, args, results } => {
                let res = emit_call(self, inst, (*sig, *callee), args, results);
                self.clear_flags();
                self.invalidate_spills();
                res
            }
            Op::TryCall { sig, callee, args, normal, handler } => {
                let res = emit_try_call(self, (*sig, *callee), args, (*normal, *handler));
                self.invalidate_spills();
                res
            }
            Op::Jump(block, args) => {
                self.edge(*block, args)?;
                if self.copied[inst.index()] {
                    return emit_body(self, *block);
                }
                self.invalidate_spills();
                emit_goto(self, *block, true);
                Ok(())
            }
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                let res = emit_brif(self, *condition, (*then, then_args), (*otherwise, else_args));
                self.invalidate_spills();
                res
            }
            Op::BrTable(index, default, targets) => {
                self.br_table(*index, *default, targets);
                self.invalidate_spills();
                Ok(())
            }
            Op::Return(values) => {
                let res = self.ret(values);
                self.invalidate_spills();
                res
            }
            Op::Trap => {
                self.emit_trap(self.at_end);
                self.invalidate_spills();
                Ok(())
            }
            Op::Trapz(condition) => {
                self.trapz(*condition);
                self.invalidate_spills();
                Ok(())
            }
            op => self.value_op(op, value),
        }
    }

    fn trap(&mut self) {
        self.emit_trap(self.at_end);
    }

    fn pos(&self) -> u32 { self.asm.pos() }

    fn finish(mut self, starts: &mut [u32]) -> Result<Code, String> {
        if let Some(label) = self.trap_label {
            self.asm.bind(label);
            self.ud2();
        }
        emit_stubs(&mut self)?;
        self.asm.relax().apply(starts, &mut self.sites, &mut self.relocs);
        Ok(Code { bytes: self.asm.finish()?, sites: self.sites, relocs: self.relocs, loops: Vec::new() })
    }
}

impl Emit<'_> {
    fn at(&self, value: Value) -> Mem {
        let disp = self.frame.values[value.index()];
        debug_assert_ne!(disp, 0, "unhomed value {value:?} in at()");
        self.frame_mem(disp)
    }

    fn ty(&self, value: Value) -> Type {
        self.func.ty(value)
    }

    fn high(&mut self, reg: Reg, value: Value) {
        let at = self.at(value).offset(8);
        self.asm.load(Size::Q, reg, at);
        self.clobber_reg(reg);
    }

    fn put_high(&mut self, value: Value, reg: Reg) {
        let at = self.at(value).offset(8);
        self.asm.store(Size::Q, at, reg);
        self.invalidate_disp(at.disp);
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
            self.asm.load(Size::Q, reg, self.frame_mem(save_offset(index)));
        }
        if self.framed && self.base == Reg::Rsp {
            self.asm.alu_imm(Alu::Add, Reg::Rsp, self.bias + 8);
        }
        self.asm.bytes(if self.framed && self.base == Reg::Rbp { &[0xC9, 0xC3] } else { &[0xC3] });
    }

    fn copy_to(&mut self, to: Mem, from: Mem, ty: Type, scratch: Xmm) {
        if ty.is_vector() && ty.bytes() >= 16 {
            self.xmm_load(ty, scratch, from);
            self.clobber_xmm(scratch);
            return self.store_xmm_home(to, scratch, ty);
        }
        self.copy_word(to, from);
        if ty.bytes() == 16 {
            self.copy_word(to.offset(8), from.offset(8));
            self.clobber_reg(Reg::Rax);
            if let Some(disp) = self.frame_disp(to) {
                self.invalidate_disp(disp);
                self.invalidate_disp(disp + 8);
            }
        } else if let Some(disp) = self.frame_disp(to) {
            self.write_spill(disp, Reg::Rax);
        } else {
            self.clobber_reg(Reg::Rax);
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
        let cond = self.truth(condition).invert();
        self.asm.jcc(cond, trap);
    }

    fn assign(&mut self, param: Value, value: Value) {
        match self.plan.locations[param.index()] {
            Location::Reg(reg) => self.raw(reg, value),
            Location::Float(xmm) => self.fget(Xmm(xmm), value),
            _ => self.move_to(self.at(param), value),
        }
    }

    fn load_home(&mut self, value: Value, from: Mem, scratch: Xmm) {
        match self.plan.locations[value.index()] {
            Location::Reg(reg) => {
                self.asm.load(Size::Q, reg, from);
                self.clobber_reg(reg);
            }
            Location::Float(xmm) => {
                self.xmm_load(self.ty(value), Xmm(xmm), from);
                self.clobber_xmm(Xmm(xmm));
            }
            _ => self.copy_to(self.at(value), from, self.ty(value), scratch),
        }
    }

    fn edge(&mut self, block: Block, args: &[Value]) -> Result<(), String> {
        let float_scratch = [Xmm(0), Xmm(1)];
        for step in edge_steps(self.func, &self.plan.locations, &self.frame.values, args, &self.func.blocks[block.index()].params) {
            match step {
                Step::Move(arg, param) => self.assign(param, arg),
                Step::Save(arg, EdgeScratch::Int(i)) => self.save_edge(i, arg),
                Step::Restore(EdgeScratch::Int(i), param) => self.restore_edge(i, param),
                Step::Save(arg, EdgeScratch::Float(i)) => self.fget(float_scratch[i], arg),
                Step::Restore(EdgeScratch::Float(i), param) => self.fput(param, float_scratch[i]),
                Step::Save(arg, EdgeScratch::Stack(i)) => self.move_to(self.frame_mem(self.frame.edge_slot(i)), arg),
                Step::Restore(EdgeScratch::Stack(i), param) => self.load_home(param, self.frame_mem(self.frame.edge_slot(i)), float::X2),
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
        self.asm.load_signed(Size::D, Reg::Rax, Mem::sib(Reg::Rcx, Reg::Rax, 2, 0));
        self.asm.alu(Alu::Add, Reg::Rcx, Reg::Rax);
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
        self.move_locs(values, &returns, None);
        let func = self.func;
        let entry = func.entry.map(|block| &func.blocks[block.index()].params);
        let sret = self.func.signature.params.iter().position(|param| param.purpose == ArgumentPurpose::StructReturn);
        if let (Some(params), Some(index)) = (entry, sret) {
            let pointer = params[index];
            self.raw(Reg::Rax, pointer);
        }
        self.leave_wide(&func.signature.returns);
        self.asm.flags_overwritten();
        if self.in_fast_ret {
            self.asm.bytes(&[0xC3]);
        } else {
            self.epilogue();
        }
        Ok(())
    }

    fn move_locs(&mut self, values: &[Value], locs: &[Loc], scratch: Option<Reg>) -> usize {
        let float_targets: Vec<(Value, u8)> = values.iter().zip(locs).filter_map(|(&val, &loc)| if let Loc::Float(xmm) = loc { Some((val, xmm)) } else { None }).collect();
        for step in arg_steps(&float_targets, |v| self.plan.float_of(v)) {
            self.float_step(step);
            if let ArgStep::Move(_, dst) | ArgStep::Restore(dst) = step {
                self.clobber_xmm(Xmm(dst));
            }
        }
        let int_targets: Vec<(Value, Reg)> = values.iter().zip(locs).filter_map(|(&val, &loc)| if let Loc::Int(reg) = loc { Some((val, reg)) } else { None }).collect();
        for step in arg_steps(&int_targets, |v| self.reg_of(v)) {
            self.int_step(step, scratch);
        }
        for (&val, &loc) in values.iter().zip(locs) {
            if let Loc::Pair(low, high) = loc {
                self.raw(low, val);
                self.high(high, val);
            }
        }
        self.invalidate_spills();
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
            self.float_step(step);
        }
        for step in reg_steps(int_pairs) {
            self.int_step(step, None);
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
        self.clobber_reg(Reg::Rcx);
    }

    fn arguments(&mut self, sig: SigRef, callee: Value, args: &[Value]) -> Result<Abi, String> {
        self.invalidate_spills();
        let abi = classify(&self.func.sigs[sig.index()], self.abi_kind)?;
        for (arg, loc) in args.iter().zip(&abi.params) {
            match *loc {
                Loc::Stack(offset) => self.move_to(Mem::new(Reg::Rsp, offset as i32), *arg),
                Loc::Struct(offset, size) => self.copy_struct(*arg, offset, size),
                _ => {}
            }
        }
        let direct = self.plan.through[callee.index()];
        let held = self.reg_of(callee).filter(|reg| SAVED.contains(reg));
        let (mut values, mut locs) = (args.to_vec(), abi.params.clone());
        if direct.is_none() && held.is_none() {
            values.push(callee);
            locs.push(Loc::Int(Reg::R11));
        }
        let float_count = self.move_locs(&values, &locs, Some(Reg::Rax));
        let func = self.func;
        self.leave_wide(&func.sigs[sig.index()].params);
        if self.abi_kind == CallAbi::SysV && self.func.sigs[sig.index()].params.iter().any(|param| param.purpose == ArgumentPurpose::Variadic) {
            self.asm.mov_imm(Reg::Rax, float_count as i64);
        }
        self.asm.flags_overwritten();
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
            None => self.asm.call(held.unwrap_or(Reg::R11)),
        }
        self.invalidate_spills();
        Ok(abi)
    }

    pub(super) fn emit_sat_op(&mut self, op: Binary, size: Size, dst: Reg, src: Reg) {
        let (umax, smin, smax) = match size {
            Size::B => (0xFF_i64, 0x80_i64, 0x7F_i64),
            Size::W => (0xFFFF_i64, 0x8000_i64, 0x7FFF_i64),
            Size::D => (0xFFFF_FFFF_i64, 0x8000_0000_i64, 0x7FFF_FFFF_i64),
            Size::Q => (-1_i64, i64::MIN, i64::MAX),
        };
        match op {
            Binary::UaddSat => {
                self.asm.alu_sized(size, Alu::Add, dst, src);
                self.asm.mov_imm(Reg::Rdx, umax);
                self.asm.cmov(Cond::B, dst, Reg::Rdx);
            }
            Binary::UsubSat => {
                self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
                self.asm.alu_sized(size, Alu::Sub, dst, src);
                self.asm.cmov(Cond::B, dst, Reg::Rdx);
            }
            Binary::SaddSat | Binary::SsubSat => {
                let alu = if op == Binary::SaddSat { Alu::Add } else { Alu::Sub };
                self.asm.alu_sized(size, alu, dst, src);
                self.asm.mov_imm(Reg::Rdx, smin);
                self.asm.mov_imm(Reg::R11, smax);
                self.asm.cmov(Cond::S, Reg::Rdx, Reg::R11);
                self.asm.cmov(Cond::O, dst, Reg::Rdx);
            }
            _ => {}
        }
        self.clobber_regs(&[dst, Reg::Rdx, Reg::R11]);
    }

    pub(super) fn popcount(&mut self, dst: Reg, src: Rm) {
        let (part, mask) = (Reg::R11, Reg::Rsi);
        match src {
            Rm::Reg(r) if r == dst as u8 => {}
            _ => self.asm.load(Size::Q, dst, src),
        }
        for (bits, shift) in BIT_STEPS {
            self.asm.mov_imm(mask, bits as i64);
            self.asm.mov(part, dst);
            self.asm.shift_imm(Size::Q, Shift::Shr, part, shift);
            self.asm.alu(Alu::And, part, mask);
            self.asm.alu(Alu::And, dst, mask);
            self.asm.alu(Alu::Add, dst, part);
        }
        self.asm.mov_imm(mask, 0x0101_0101_0101_0101);
        self.asm.imul(dst, mask);
        self.asm.shift_imm(Size::Q, Shift::Shr, dst, 56);
        self.clobber_regs(&[dst, Reg::R11, Reg::Rsi]);
    }

    pub(super) fn reverse_bits(&mut self) {
        for (mask, shift) in BIT_STEPS {
            self.asm.mov_imm(Reg::Rcx, mask as i64);
            self.asm.mov(Reg::Rdx, Reg::Rax);
            self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rdx, shift);
            self.asm.alu(Alu::And, Reg::Rdx, Reg::Rcx);
            self.asm.alu(Alu::And, Reg::Rax, Reg::Rcx);
            self.asm.shift_imm(Size::Q, Shift::Shl, Reg::Rax, shift);
            self.asm.alu(Alu::Or, Reg::Rax, Reg::Rdx);
        }
        self.asm.bswap(Size::Q, Reg::Rax);
        self.clobber_regs(&[Reg::Rax, Reg::Rcx, Reg::Rdx]);
    }

    pub(super) fn clamp_signed(&mut self, reg: Reg, to: Type, signed: bool) {
        let bits = to.bits();
        if bits >= 64 {
            return;
        }
        let (lo, hi) = if signed {
            (-(1_i64 << (bits - 1)), (1_i64 << (bits - 1)) - 1)
        } else {
            (0, (1_i64 << bits) - 1)
        };
        self.asm.mov_imm(Reg::Rcx, lo);
        self.asm.alu(Alu::Cmp, reg, Reg::Rcx);
        self.asm.cmov(Cond::L, reg, Reg::Rcx);
        self.asm.mov_imm(Reg::Rcx, hi);
        self.asm.alu(Alu::Cmp, reg, Reg::Rcx);
        self.asm.cmov(Cond::G, reg, Reg::Rcx);
        self.clobber_regs(&[reg, Reg::Rcx]);
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
    fn new_label(&mut self) -> Label { self.asm.label() }
    fn tails(&mut self) -> &mut Tails<'a, Label> { &mut self.tails }
    fn bind_label(&mut self, label: Label) {
        self.invalidate_spills();
        self.asm.bind(label);
    }
    fn jump_label(&mut self, label: Label) {
        self.invalidate_spills();
        self.asm.flags_overwritten();
        self.asm.jmp(label);
    }
    fn branch_cond(&mut self, condition: Value, invert: bool, label: Label) {
        let cond = self.truth(condition);
        self.asm.jcc(if invert { cond.invert() } else { cond }, label);
    }
    fn edge(&mut self, block: Block, args: &[Value]) -> Result<(), String> { self.edge(block, args) }
    fn arguments(&mut self, sig: SigRef, callee: Value, args: &[Value]) -> Result<Abi, String> { self.arguments(sig, callee, args) }
    fn store_locs(&mut self, values: &[Value], locs: &[Loc]) {
        self.store_locs(values, locs);
        self.invalidate_spills();
    }
    fn store_loc(&mut self, value: Value, loc: Loc) { self.store_loc(value, loc); }
    fn load_incoming(&mut self, value: Value, offset: u32) { self.load_home(value, self.frame_mem(INCOMING + offset as i32), INTAKE_XMM); }
    fn intake_struct(&mut self, value: Value, offset: u32) {
        self.asm.lea(Reg::Rax, self.frame_mem(INCOMING + offset as i32));
        self.clobber_reg(Reg::Rax);
        self.put(value, Reg::Rax);
    }
    fn park_spill(&mut self, value: Value, reg: ParkReg<Reg>) {
        let home = self.at(value);
        let disp = self.frame.values[value.index()];
        match reg {
            ParkReg::Int(r) => {
                self.asm.store(Size::Q, home, r);
                self.write_spill(disp, r);
            }
            ParkReg::Float(xmm) => {
                self.xmm_store(self.ty(value), home, Xmm(xmm));
                self.invalidate_disp(disp);
                self.last_fspill = Some((disp, Xmm(xmm), self.ty(value)));
            }
        }
    }
    fn park_reload(&mut self, value: Value, reg: ParkReg<Reg>) {
        let home = self.at(value);
        let disp = self.frame.values[value.index()];
        match reg {
            ParkReg::Int(r) => {
                self.asm.load(Size::Q, r, home);
                self.record_spill(disp, r);
            }
            ParkReg::Float(xmm) => {
                self.xmm_load(self.ty(value), Xmm(xmm), home);
                self.clobber_xmm(Xmm(xmm));
                self.last_fspill = Some((disp, Xmm(xmm), self.ty(value)));
            }
        }
    }
    fn put_exception(&mut self, value: Value) { self.put(value, Reg::Rax); }
    fn push_site(&mut self, site: Site) { self.sites.push(site); }
}

