mod asm;
mod atomic;
mod float;
mod frame;
mod fuse;
mod int;
mod location;
mod vector;
mod wide;

use asm::{Alu, Asm, Cond, FP, LR, Label, Mem, Reg, SP, Size, Vreg, ZR};
use frame::{Abi, Loc, Packing, SAVED, SAVED_FLOAT, classify};
use crate::alloc::{CallTarget, Location, Plan, Regs};
use crate::frame::{ArgStep, Emitter, Ext, Frame, INCOMING, PAGE, Prepared, Step, arg_steps, drive, edge_steps, emit_body, prepare, reg_steps, save_offset};
use crate::ir::{ArgumentPurpose, Binary, Block, Function, Inst, IntCC, Op, SigRef, Type, Value};
use crate::park::ParkReg;
use crate::reloc::{DictTierUp, RelocKind, StencilReloc};
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
    vectors: true,
    absolute_calls: true,
    two_address: false,
    park: true,
    fold_sib: true,
};

fn emit_save_pairs<T: Copy>(asm: &mut Asm, regs: &[T], base: usize, mut step: impl FnMut(&mut Asm, i32, T, Option<T>)) {
    let mut i = 0;
    while i < regs.len() {
        if i + 1 < regs.len() {
            let disp = save_offset(base + i + 1);
            if (-512..=504).contains(&disp) && disp % 8 == 0 {
                step(asm, disp, regs[i + 1], Some(regs[i]));
                i += 2;
                continue;
            }
        }
        step(asm, save_offset(base + i), regs[i], None);
        i += 1;
    }
}

pub(crate) fn compile(func: &Function, target: &Target) -> Result<Code, String> {
    let packing = if target.os == Os::Darwin { Packing::Natural } else { Packing::Slots };
    let Prepared { order, copied, abi, mut plan, frame, framed } = prepare(func, &REGS, target.effort == crate::Effort::Full, |sig| classify(sig, packing))?;
    let defs = crate::fold::defs(func);
    fuse::fuse(func, &mut plan);
    let mut asm = Asm::default();
    let blocks = func.blocks.iter().map(|_| asm.label()).collect();
    let emit = Emit {
        func,
        frame,
        framed,
        abi,
        packing,
        asm,
        blocks,
        sites: Vec::new(),
        relocs: Vec::new(),
        plan,
        defs,
        flags: None,
        last_cmp: None,
        next: None,
        copied,
        direct_calls: target.direct_calls,
        sp_bias: 0,
    };
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
    defs: Vec<Option<Inst>>,
    flags: Option<(Value, Cond)>,
    last_cmp: Option<(Size, Value, Value, bool)>,
    next: Option<Block>,
    copied: Vec<bool>,
    direct_calls: bool,
    sp_bias: i32,
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
        self.asm.stp_pre(FP, LR, SP, -16);
        self.asm.mov_sp(FP, SP);
        let size = self.frame.size;
        if size > PAGE {
            self.probe(size);
        } else if size > 0 {
            self.asm.add_imm(true, false, Size::X, SP, SP, size);
        }
        emit_save_pairs(&mut self.asm, &self.frame.saves, 0, |asm, disp, a, b| match b {
            Some(r0) => asm.stp_offset(a, r0, FP, disp),
            None => asm.store(Size::X, Mem::new(FP, disp), a),
        });
        emit_save_pairs(&mut self.asm, &self.frame.float_saves, self.frame.saves.len(), |asm, disp, a, b| match b {
            Some(v0) => asm.fstp_offset(Vreg(a), Vreg(v0), FP, disp),
            None => asm.vstore(8, Mem::new(FP, disp), Vreg(a)),
        });
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
        self.last_cmp = None;
    }

    fn inst(&mut self, inst: Inst) -> Result<(), String> {
        let func = self.func;
        let data = &func.insts[inst.index()];
        let keep_flags = matches!(
            data.op,
            Op::Brif(..)
                | Op::Trapz(..)
                | Op::Load(..)
                | Op::Store(..)
                | Op::StackAddr(..)
                | Op::StackLoad(..)
                | Op::StackStore(..)
                | Op::Const(_)
                | Op::FramePointer
        ) || matches!(data.op, Op::Select(cond, ..) if self.flags.is_some_and(|(v, _)| v == cond));
        if !keep_flags {
            self.flags = None;
            if !matches!(data.op, Op::Icmp(..)) {
                self.last_cmp = None;
            }
        }
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
            Op::BrTable(index, default, targets) => {
                self.br_table(*index, *default, targets);
                Ok(())
            }
            Op::Return(values) => self.ret(values),
            Op::Trap => {
                self.asm.brk();
                Ok(())
            }
            Op::Trapz(condition) => {
                self.trapz(*condition);
                Ok(())
            }
            op => {
                if self.vector_inst(op, value)? {
                    return Ok(());
                }
                self.value_op(op, value)
            }
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
            emit_save_pairs(&mut self.asm, &self.frame.saves, 0, |asm, disp, a, b| match b {
                Some(r0) => asm.ldp_offset(a, r0, FP, disp),
                None => asm.load(Size::X, false, a, Mem::new(FP, disp)),
            });
            emit_save_pairs(&mut self.asm, &self.frame.float_saves, self.frame.saves.len(), |asm, disp, a, b| match b {
                Some(v0) => asm.fldp_offset(Vreg(a), Vreg(v0), FP, disp),
                None => asm.vload(8, Vreg(a), Mem::new(FP, disp)),
            });
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
                self.last_cmp = None;
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
        let steps = edge_steps(args, &self.func.blocks[block.index()].params, |a, b| self.plan.same_home(&self.frame, a, b));
        let reg_only = steps.iter().all(|step| match *step {
            Step::Move(arg, param) => self.plan.reg_of(arg).is_some() && self.plan.reg_of(param).is_some(),
            Step::Save(arg, index) => index < 2 && self.plan.reg_of(arg).is_some(),
            Step::Restore(index, param) => index < 2 && self.plan.reg_of(param).is_some(),
        });
        let scratch = [S, T];
        for step in steps {
            match step {
                Step::Move(arg, param) => self.assign(param, arg),
                Step::Save(arg, index) if reg_only => {
                    if let Some(reg) = self.plan.reg_of(arg) {
                        self.asm.mov(scratch[index], reg);
                    }
                }
                Step::Restore(index, param) if reg_only => {
                    if let Some(reg) = self.plan.reg_of(param) {
                        self.asm.mov(reg, scratch[index]);
                    }
                }
                Step::Save(arg, index) => self.move_to(self.frame_mem(self.frame.edge_slot(index)), arg),
                Step::Restore(index, param) => self.load_home(param, self.frame_mem(self.frame.edge_slot(index))),
            }
        }
        Ok(())
    }

    fn branch_cond(&mut self, condition: Value, invert: bool, label: Label) {
        if self.plan.elided[condition.index()]
            && let Some(inst) = self.defs[condition.index()]
        {
            match self.func.insts[inst.index()].op {
                Op::Binary(Binary::Band, x, y) => {
                    if let Some((val, bit)) = fuse::single_bit_mask(self.func, &self.plan, x, y) {
                        let reg = self.bits(val, A);
                        self.asm.tbz(!invert, reg, bit, label);
                        return;
                    }
                }
                Op::Icmp(cc, lhs, zero) => {
                    let masked = if self.plan.const_of(zero) == Some(0) { lhs } else { zero };
                    if let Some(m_inst) = self.defs[masked.index()]
                        && let Op::Binary(Binary::Band, x, y) = self.func.insts[m_inst.index()].op
                        && let Some((val, bit)) = fuse::single_bit_mask(self.func, &self.plan, x, y)
                    {
                        let eq_zero = cc == IntCC::Equal;
                        let reg = self.bits(val, A);
                        self.asm.tbz(eq_zero ^ !invert, reg, bit, label);
                        return;
                    }
                }
                _ => {}
            }
        }
        if let Some((value, cond)) = self.flags.take()
            && value == condition
        {
            self.asm.bcond(if invert { cond.invert() } else { cond }, label);
            return;
        }
        self.last_cmp = None;
        let bytes = self.ty(condition).bytes();
        let reg = if bytes >= 4 { self.bits(condition, A) } else { self.src(condition, A, Ext::Zero) };
        let size = if bytes >= 8 { Size::X } else { Size::W };
        self.asm.cbz(!invert, size, reg, label);
    }

    fn brif(&mut self, condition: Value, then: (Block, &[Value]), otherwise: (Block, &[Value])) -> Result<(), String> {
        if let Some(known) = self.plan.const_of(condition) {
            let (block, args) = if crate::fold::extend(known, self.ty(condition), false) != 0 { then } else { otherwise };
            self.edge(block, args)?;
            self.goto(block, true);
            return Ok(());
        }
        let (then_moves, else_moves) = (
            self.plan.moves(self.func, &self.frame, then.0, then.1),
            self.plan.moves(self.func, &self.frame, otherwise.0, otherwise.1),
        );
        let falls = self.next.is_some_and(|next| next == then.0 || next == self.func.jump_target(then.0));
        if !else_moves && (then_moves || falls) {
            self.branch_cond(condition, true, self.blocks[self.func.jump_target(otherwise.0).index()]);
            self.edge(then.0, then.1)?;
            self.goto(then.0, true);
            return Ok(());
        }
        if !then_moves {
            self.branch_cond(condition, false, self.blocks[self.func.jump_target(then.0).index()]);
            self.edge(otherwise.0, otherwise.1)?;
            self.goto(otherwise.0, true);
            return Ok(());
        }
        let other = self.asm.label();
        self.branch_cond(condition, true, other);
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

    fn br_table(&mut self, index: Value, default: Block, targets: &[Block]) {
        if targets.is_empty() {
            self.goto(default, true);
            return;
        }
        let idx = self.src(index, A, Ext::Zero);
        self.asm.mov_imm(C, targets.len() as i64);
        self.asm.alu(Alu::Subs, Size::X, ZR, idx, C);
        self.asm.bcond(Cond::Hs, self.blocks[self.func.jump_target(default).index()]);
        let table = self.asm.label();
        self.asm.adr(C, table);
        self.asm.ldrsw_scaled(IP, C, idx);
        self.asm.alu(Alu::Add, Size::X, C, C, IP);
        self.asm.br(C);
        self.asm.bind(table);
        for &target in targets {
            self.asm.table_entry(table, self.blocks[self.func.jump_target(target).index()]);
        }
    }

    fn ret(&mut self, values: &[Value]) -> Result<(), String> {
        let returns = self.abi.returns.clone();
        self.move_locs(values, &returns);
        self.epilogue();
        Ok(())
    }

    fn move_locs(&mut self, values: &[Value], locs: &[Loc]) {
        let float_targets: Vec<(Value, u8)> = values.iter().zip(locs).filter_map(|(&val, &loc)| if let Loc::Float(v) = loc { Some((val, v)) } else { None }).collect();
        for step in arg_steps(&float_targets, |v| self.plan.float_of(v)) {
            match step {
                ArgStep::Move(src, dst) => self.asm.vmov(Vreg(dst), Vreg(src)),
                ArgStep::Save(src) => {
                    self.asm.add_imm(true, false, Size::X, SP, SP, 16);
                    self.sp_bias += 16;
                    self.asm.vstore(16, Mem::new(SP, 0), Vreg(src));
                }
                ArgStep::Restore(dst) => {
                    self.asm.vload(16, Vreg(dst), Mem::new(SP, 0));
                    self.asm.add_imm(false, false, Size::X, SP, SP, 16);
                    self.sp_bias -= 16;
                }
                ArgStep::Load(val, dst) => self.fget(Vreg(dst), val),
            }
        }
        let int_targets: Vec<(Value, Reg)> = values.iter().zip(locs).filter_map(|(&val, &loc)| if let Loc::Int(reg) = loc { Some((val, reg)) } else { None }).collect();
        for step in arg_steps(&int_targets, |v| self.plan.reg_of(v)) {
            match step {
                ArgStep::Move(src, dst) => self.asm.mov(dst, src),
                ArgStep::Save(src) => self.asm.mov(A, src),
                ArgStep::Restore(dst) => self.asm.mov(dst, A),
                ArgStep::Load(val, dst) => self.raw(dst, val),
            }
        }
        for (&val, &loc) in values.iter().zip(locs) {
            if let Loc::Pair(low, high) = loc {
                self.raw(low, val);
                self.high(high, val);
            }
        }
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
                ArgStep::Move(src, dst) => self.asm.vmov(Vreg(dst), Vreg(src)),
                ArgStep::Save(src) => {
                    self.asm.add_imm(true, false, Size::X, SP, SP, 16);
                    self.sp_bias += 16;
                    self.asm.vstore(16, Mem::new(SP, 0), Vreg(src));
                }
                ArgStep::Restore(dst) => {
                    self.asm.vload(16, Vreg(dst), Mem::new(SP, 0));
                    self.asm.add_imm(false, false, Size::X, SP, SP, 16);
                    self.sp_bias -= 16;
                }
                ArgStep::Load(..) => {}
            }
        }
        for step in reg_steps(int_pairs) {
            match step {
                ArgStep::Move(src, dst) => self.asm.mov(dst, src),
                ArgStep::Save(src) => self.asm.mov(A, src),
                ArgStep::Restore(dst) => self.asm.mov(dst, A),
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
            Loc::Float(index) => self.fput(value, Vreg(index)),
            Loc::Stack(_) | Loc::Struct(..) => {}
        }
    }

    fn stack_argument(&mut self, offset: u32, purpose: ArgumentPurpose, value: Value) {
        let to = Mem::new(SP, offset as i32);
        let bytes = self.ty(value).bytes();
        if bytes < 8 && !self.ty(value).is_float() {
            let reg = self.src(value, A, Ext::Zero);
            let width = if self.packing == Packing::Natural && purpose != ArgumentPurpose::Variadic { Size::of(bytes) } else { Size::X };
            self.asm.store(width, to, reg);
            return;
        }
        self.move_to(to, value);
    }

    fn arguments(&mut self, sig: SigRef, callee: Value, args: &[Value]) -> Result<Abi, String> {
        let sig_data = &self.func.sigs[sig.index()];
        let abi = classify(sig_data, self.packing)?;
        for ((arg, loc), param) in args.iter().zip(&abi.params).zip(&sig_data.params) {
            if let Loc::Stack(offset) = *loc {
                self.stack_argument(offset, param.purpose, *arg);
            }
        }
        let direct = self.plan.through[callee.index()];
        let target = if direct.is_none() {
            match self.plan.reg_of(callee).filter(|reg| SAVED.contains(reg)) {
                Some(reg) => reg,
                None => {
                    self.raw(T, callee);
                    T
                }
            }
        } else {
            T
        };
        self.move_locs(args, &abi.params);
        match direct {
            Some(CallTarget::Slot(sym)) if self.direct_calls => {
                let at = self.asm.bl_reloc();
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Arm64Call26));
            }
            Some(CallTarget::Slot(sym)) => {
                let at = self.asm.mov_symbol(T);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Arm64Movw64));
                self.asm.load(Size::X, false, T, Mem::new(T, 0));
                self.asm.blr(T);
            }
            Some(CallTarget::Abs(at)) => {
                self.asm.mov_imm(T, i64::from(at));
                self.asm.blr(T);
            }
            None => self.asm.blr(target),
        }
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
        self.store_locs(results, &abi.returns);
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
