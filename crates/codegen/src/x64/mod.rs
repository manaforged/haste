mod asm;
mod atomic;
mod bits;
mod features;
mod float;
mod frame;
mod frames;
mod imm;
mod int;
mod location;
pub(crate) mod text;
mod vector;
mod vector_lanes;
mod vector_reduce;
mod vector_narrow;
mod wide;
mod ymm;

use asm::{Alu, Asm, Cond, Label, Mem, Reg, Size, Xmm};
use features::Features;
pub(crate) use ymm::host_avx2;
use frame::{Abi, Loc, SAVED, SAVED_RBP, classify};
use crate::alloc::{CallTarget, Location, Plan, Regs};
use crate::emit::{Backend, Tails, emit_brif, emit_call, emit_goto, emit_stubs, emit_try_call, intake_params};
use crate::frame::{ArgStep, EdgeScratch, Emitter, Ext, Frame, INCOMING, PAGE, Prepared, ShrinkWrap, Step, arg_steps, drive, edge_steps, emit_body, prepare_split, reg_steps, save_offset, shrink_wrap};
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
    scratch_mask: atomic::scratch_mask,
    dict: Some(Reg::R10),
    float: &float::POOL,
    float_branches: &float::BRANCHES,
    vectors: true,
    absolute_calls: true,
    constant_calls: true,
    two_address: true,
    park: true,
    fold_sib: true,
    call_zext: false,
    imm_store,
    high_product: Some(Reg::Rdx),
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


pub(crate) fn compile(func: &Function, target: &Target) -> Result<Code, String> {
    let abi_kind = target.abi;
    let keeps = keeps_frame_pointer(func);
    let regs = match (abi_kind, keeps) {
        (CallAbi::Win64, true) => &REGS_WIN64,
        (CallAbi::Win64, false) => &REGS_WIN64_RBP,
        (_, true) => &REGS,
        (_, false) => &REGS_RBP,
    };
    let (split, Prepared { order, copied, abi, mut plan, mut frame, mut framed }) = prepare_split(func, regs, target.effort == crate::Effort::Full, |sig| classify(sig, abi_kind))?;
    let func = split.as_ref().unwrap_or(func);
    let features = Features::host();
    let dropped = |plan: &Plan<Reg>| plan.elided.iter().chain(&plan.fused).filter(|&&e| e).count();
    let dropped_before = dropped(&plan);
    let (roots, pure_test) = location::fuse(func, &mut plan, &features);
    if dropped(&plan) != dropped_before {
        crate::frame::repack_after_elide(func, &plan, &abi, &mut frame, &mut framed, |sig| classify(sig, abi_kind))?;
    }
    let shrink = shrink_wrap(func, framed, &plan, &frame, &abi.params, &FRAME_REGS);
    let (base, bias) = if framed && !keeps { (Reg::Rsp, frame.size as i32) } else { (Reg::Rbp, 0) };
    let mut asm = Asm::default();
    asm.avx = features.avx;
    let blocks = func.blocks.iter().map(|_| asm.label()).collect();
    let defs = crate::fold::defs(func);
    let overflow_read = wide::overflow_reads(func);
    let adc_sum = wide::adc_sums(func, &plan, &defs);
    let flag_carry = wide::flag_carries(func, &plan);
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
        pure_test,
        flags: None,
        last_spill: crate::spills::SpillCache::new(),
        last_fspill: None,
        carry: (None, None),
        sole_pred: bits::sole_preds(func, &copied),
        next: None,
        last_block,
        at_end: false,
        trap_label: None,
        lines: Vec::new(),
        features,
        copied,
        direct_calls: target.direct_calls,
        frame_cache: target.direct_calls && target.os == crate::target::Os::Linux,
        cold_calls: Vec::new(),
        shrink,
        in_fast: false,
        low_product: None,
        overflow_read,
        adc_sum,
        flag_carry,
        carry_flags: None,
        tails: Tails::new(),
    };
    drive(emit, &order)
}

/// Whether a framed function addresses its frame from rbp: it reads the frame pointer or catches an unwind.
fn keeps_frame_pointer(func: &Function) -> bool {
    func.keep_fp
        || !func.unwind_drops.is_empty()
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
    pure_test: Vec<bool>,
    flags: Option<(Value, Cond)>,
    last_spill: crate::spills::SpillCache<Reg>,
    last_fspill: Option<(i32, Xmm, Type)>,
    carry: (Option<Block>, Option<crate::spills::SpillCache<Reg>>),
    sole_pred: Vec<Option<Block>>,
    next: Option<Block>,
    last_block: Option<Block>,
    at_end: bool,
    trap_label: Option<Label>,
    lines: Vec<(u32, u64)>,
    features: Features,
    copied: Vec<bool>,
    direct_calls: bool,
    frame_cache: bool,
    cold_calls: Vec<frames::ColdCall>,
    shrink: Option<ShrinkWrap>,
    in_fast: bool,
    low_product: Option<(Value, Value)>,
    overflow_read: Vec<bool>,
    /// The sums whose carry an adc or sbb reads through a compare it absorbed.
    adc_sum: Vec<bool>,
    /// The carry outs that the carry flag alone hands to their only consumers.
    flag_carry: Vec<bool>,
    carry_flags: Option<(bool, [Value; 3])>,
    tails: Tails<'a, Label>,
}

impl<'a> Emitter<'a> for Emit<'a> {
    fn func(&self) -> &'a Function {
        self.func
    }

    fn is_elided(&self, value: Value) -> bool { self.plan.elided[value.index()] }

    fn zero_stack_store(&self, inst: Inst) -> Option<(i32, u32)> { crate::frame::stack_zero_store(self.func, &self.frame.slots, inst, |v| self.const_of(v)) }

    fn emit_zero_frame(&mut self, offset: i32, bytes: u32) {
        let to = self.frame_mem(offset);
        if bytes == 8 { self.zero_to(to); } else { self.asm.store_imm_sized(Size::of(bytes), to, 0); }
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

    fn set_next(&mut self, next: Option<Block>) { self.next = next; }

    fn bind_block(&mut self, block: Block) {
        self.invalidate_spills();
        self.resume_spills(block);
        self.at_end = self.last_block == Some(block);
        self.asm.flags_overwritten();
        self.asm.bind(self.blocks[block.index()]);
    }

    fn bind_entry(&mut self, block: Block) {
        self.carry = (Some(block), None);
        self.at_end = self.last_block == Some(block);
        self.asm.bind(self.blocks[block.index()]);
    }

    fn deferred_prologue(&self) -> bool { self.shrink.is_some() }

    fn on_block_bound(&mut self, block: Block) -> Result<(), String> {
        self.in_fast = self.shrink.as_ref().is_some_and(|s| s.fast[block.index()]);
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
                res
            }
            Op::TryCall { sig, callee, args, normal, handler } => {
                let res = emit_try_call(self, (*sig, *callee), args, (*normal, *handler));
                self.invalidate_spills();
                res
            }
            Op::Jump(block, args) => {
                if self.is_split_ret(*block) {
                    self.edge(*block, args)?;
                    return self.fast_ret(*block);
                }
                if self.exits_fast(*block) {
                    self.prologue();
                    self.in_fast = false;
                }
                self.edge(*block, args)?;
                if self.copied[inst.index()] {
                    return emit_body(self, *block);
                }
                emit_goto(self, *block, true);
                self.carry_spills(self.next == Some(*block));
                self.invalidate_spills();
                Ok(())
            }
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                let res = emit_brif(self, *condition, (*then, then_args), (*otherwise, else_args));
                self.carry_spills(then != otherwise && (self.next == Some(*then) || self.next == Some(*otherwise)));
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
                Ok(())
            }
            op => self.value_op(op, value),
        }
    }

    fn trap(&mut self) {
        self.emit_trap(self.at_end);
    }

    fn pos(&self) -> u32 { self.asm.pos() }

    fn lines(&mut self) -> &mut Vec<(u32, u64)> { &mut self.lines }

    fn finish(mut self, starts: &mut [u32]) -> Result<Code, String> {
        if let Some(label) = self.trap_label {
            self.asm.bind(label);
            self.ud2();
        }
        self.emit_cold_calls();
        emit_stubs(&mut self)?;
        self.relocs.extend(self.asm.rips.drain(..).map(|(at, sym)| StencilReloc::symbol(at, sym, RelocKind::PcRel32)));
        self.asm.relax().apply(starts, &mut self.sites, &mut self.relocs);
        Ok(Code { bytes: self.asm.finish()?, sites: self.sites, relocs: self.relocs, loops: Vec::new(), lines: Vec::new(), unwind_drops: crate::frame::resolve_unwind_drops(self.func, &self.frame) })
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
        self.asm.flags_overwritten();
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
        let params = &self.func.blocks[block.index()].params;
        let (mut rem_a, mut rem_p) = (Vec::new(), Vec::new());
        let (a_slice, p_slice) = if args.iter().any(|&a| self.is_fused_cas(a)) {
            for (&arg, &param) in args.iter().zip(params) {
                if self.is_fused_cas(arg) { self.put(param, Reg::Rax); } else { rem_a.push(arg); rem_p.push(param); }
            }
            (&rem_a[..], &rem_p[..])
        } else {
            (args, &params[..])
        };
        let float_scratch = [Xmm(0), Xmm(1)];
        for step in edge_steps(self.func, &self.plan.locations, &self.frame.values, a_slice, p_slice) {
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
        let slot = match self.reg_of(index) {
            Some(reg) if self.ty(index) == Type::I64 && !matches!(reg, Reg::Rcx | Reg::Rsp) => reg,
            _ => {
                self.get(Reg::Rax, index, Ext::Zero);
                Reg::Rax
            }
        };
        match i32::try_from(targets.len()) {
            Ok(len) => self.asm.alu_imm(Alu::Cmp, slot, len),
            Err(_) => {
                self.asm.mov_imm(Reg::Rcx, targets.len() as i64);
                self.asm.alu(Alu::Cmp, slot, Reg::Rcx);
            }
        }
        self.asm.jcc(Cond::Ae, self.blocks[self.func.jump_target(default).index()]);
        let table = self.asm.label();
        self.asm.lea_rip(Reg::Rcx, table);
        self.asm.load_signed(Size::D, Reg::Rax, Mem::sib(Reg::Rcx, slot, 2, 0));
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
        if self.in_fast {
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

    fn arguments(&mut self, sig: SigRef, callee: Value, args: &[Value], is_try: bool) -> Result<Abi, String> {
        let abi = classify(&self.func.sigs[sig.index()], self.abi_kind)?;
        for (arg, loc) in args.iter().zip(&abi.params) {
            match *loc {
                Loc::Stack(offset) => self.move_to(Mem::new(Reg::Rsp, offset as i32), *arg),
                Loc::Struct(offset, size) => self.copy_struct(*arg, offset, size),
                _ => {}
            }
        }
        let direct = self.plan.through[callee.index()];
        if !is_try
            && let Some(CallTarget::Symbol(lookup)) = direct
            && self.func.is_thread_cell(callee)
            && self.thread_cell_cold(lookup, args)
        {
            self.clobber_call_spills();
            return Ok(abi);
        }
        let cell = match direct {
            Some(CallTarget::Symbol(lookup)) if self.func.is_thread_cell(callee) => args.first().and_then(|&slot| self.thread_cell_hit(slot, lookup)).map(|hit| (lookup, hit)),
            _ => None,
        };
        if cell.is_some() {
            self.clobber_reg(Reg::Rax);
        }
        let held = self.reg_of(callee).filter(|&reg| !matches!(reg, Reg::Rax | Reg::Rcx) && !abi.clobbers(reg));
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
        if let Some((lookup, hit)) = cell {
            let at = self.asm.call_pcrel32();
            self.relocs.push(StencilReloc::symbol(at, lookup, crate::reloc::RelocKind::PcRel32));
            self.asm.bind(hit);
            self.clobber_call_spills();
            return Ok(abi);
        }
        if self.inline_call(direct, args) {
            self.clobber_call_spills();
            return Ok(abi);
        }
        match direct {
            Some(CallTarget::Abs(at)) => self.asm.call_absolute(at),
            Some(CallTarget::Slot(sym)) => {
                let reloc = self.asm.call_symbol(sym, self.direct_calls);
                self.relocs.push(reloc);
            }
            Some(CallTarget::Runtime(rt)) if self.direct_calls => {
                let at = self.asm.call_pcrel32();
                self.relocs.push(StencilReloc::runtime(at, rt, crate::reloc::RelocKind::PcRel32));
            }
            Some(CallTarget::Runtime(rt)) => {
                let at = self.asm.mov_abs64(Reg::R11);
                self.relocs.push(StencilReloc::runtime(at, rt, crate::reloc::RelocKind::Abs64));
                self.asm.call(Reg::R11);
            }
            Some(CallTarget::Symbol(sym)) if self.direct_calls => {
                let at = self.asm.call_pcrel32();
                self.relocs.push(StencilReloc::symbol(at, sym, crate::reloc::RelocKind::PcRel32));
            }
            Some(CallTarget::Symbol(sym)) => {
                let at = self.asm.mov_abs64(Reg::R11);
                self.relocs.push(StencilReloc::symbol(at, sym, crate::reloc::RelocKind::Abs64));
                self.asm.call(Reg::R11);
            }
            None => self.asm.call(held.unwrap_or(Reg::R11)),
        }
        self.clobber_call_spills();
        Ok(abi)
    }
}

impl<'a> Backend<'a, Reg, Label> for Emit<'a> {
    fn plan(&self) -> &Plan<Reg> { &self.plan }
    fn plan_mut(&mut self) -> &mut Plan<Reg> { &mut self.plan }
    fn frame(&self) -> &Frame<Reg> { &self.frame }
    fn abi(&self) -> &Abi { &self.abi }
    fn shrink(&self) -> Option<&ShrinkWrap> { self.shrink.as_ref() }
    fn in_fast(&self) -> bool { self.in_fast }
    fn ret_fast(&mut self, values: &[Value]) -> Result<(), String> {
        self.in_fast = true;
        self.ret(values)
    }
    fn next_block(&self) -> Option<Block> { self.next }
    fn block_label(&self, block: Block) -> Label { self.blocks[block.index()] }
    fn new_label(&mut self) -> Label { self.asm.label() }
    fn tails(&mut self) -> &mut Tails<'a, Label> { &mut self.tails }
    fn bind_label(&mut self, label: Label) {
        self.invalidate_spills();
        self.asm.flags_overwritten();
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
    fn arguments(&mut self, sig: SigRef, callee: Value, args: &[Value], is_try: bool) -> Result<Abi, String> { self.arguments(sig, callee, args, is_try) }
    fn store_locs(&mut self, values: &[Value], locs: &[Loc]) { self.store_locs(values, locs); }
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

