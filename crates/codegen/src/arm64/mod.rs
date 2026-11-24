mod asm;
mod atomic;
mod call;
mod emit;
mod float;
mod frame;
mod frames;
mod fuse;
pub(crate) mod int;
mod location;
pub(crate) mod text;
mod vector;
mod vector_ops;
mod wide;

use crate::alloc::{Location, Plan, Regs};
use crate::emit::{
    Backend, Tails, emit_brif, emit_call, emit_goto, emit_stubs, emit_try_call, intake_params,
};
use crate::frame::{
    Emitter, Frame, PAGE, Prepared, ShrinkWrap, drive, emit_body, prepare_split, save_offset,
    shrink_wrap,
};
use crate::ir::{Block, Function, Inst, Op, Type, Value};
use crate::reloc::{DictTierUp, StencilReloc};
use crate::target::{Os, Target};
use crate::{Code, Site};
use asm::{Alu, Asm, Cond, FP, LR, Label, Mem, Reg, SP, Size, Vreg, ZR};
use frame::{Abi, Packing, SAVED, SAVED_FLOAT, classify};

const A: Reg = Reg(9);
const C: Reg = Reg(10);
const D: Reg = Reg(11);
const S: Reg = Reg(12);
const T: Reg = Reg(13);
const IP: Reg = Reg(16);
const V0: Vreg = Vreg(0);
const V1: Vreg = Vreg(1);

const WIDE_SCRATCH: [Reg; 2] = [Reg(14), Reg(15)];
const OP_SCRATCH: [Reg; 3] = [D, S, T];
const PRESERVED_CLOBBERS: [Reg; 2] = [Reg(0), LR];

const fn make_regs(
    scratch_mask: fn(&Function, &Op) -> u8,
    atomic_addrs: crate::alloc::AtomicAddrMode,
) -> Regs<Reg> {
    Regs {
        saved: &SAVED,
        saved_float: &SAVED_FLOAT,
        leaf: &[
            Reg(0),
            Reg(1),
            Reg(2),
            Reg(3),
            Reg(4),
            Reg(5),
            Reg(6),
            Reg(7),
            Reg(8),
            D,
            S,
            T,
            Reg(14),
            Reg(15),
            LR,
        ],
        wide_scratch: &WIDE_SCRATCH,
        op_scratch: &OP_SCRATCH,
        preserved_clobbers: &PRESERVED_CLOBBERS,
        scratch_mask,
        dict: None,
        float: &float::POOL,
        float_branches: &float::BRANCHES,
        vectors: true,
        absolute_calls: true,
        constant_calls: true,
        two_address: false,
        park: true,
        fold_sib: true,
        atomic_addrs,
        call_zext: true,
        imm_store,
        high_product: None,
    }
}

const REGS: Regs<Reg> = make_regs(atomic::scratch_mask, crate::alloc::AtomicAddrMode::None);
const REGS_LSE: Regs<Reg> = make_regs(
    atomic::scratch_mask_lse,
    crate::alloc::AtomicAddrMode::Rcpc2,
);

fn imm_store(imm: i64) -> u64 {
    if imm == 0 { 1 } else { 2 }
}

fn emit_save_pairs<T: Copy>(
    asm: &mut Asm,
    regs: &[T],
    base: usize,
    mut step: impl FnMut(&mut Asm, i32, T, Option<T>),
) {
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
    let lse = target.os == Os::Darwin;
    let regs = if lse { &REGS_LSE } else { &REGS };
    let packing = if target.os == Os::Darwin {
        Packing::Natural
    } else {
        Packing::Slots
    };
    let full = target.effort == crate::Effort::Full;
    let tlv_cells = target.direct_calls && target.os == Os::Darwin;
    let (
        split,
        Prepared {
            order,
            copied,
            abi,
            mut plan,
            mut frame,
            mut framed,
        },
    ) = prepare_split(func, regs, (full, tlv_cells), |sig| classify(sig, packing))?;
    let func = split.as_ref().unwrap_or(func);
    let defs = crate::fold::defs(func);
    if full {
        let dropped = |plan: &crate::alloc::Plan<Reg>| {
            plan.elided
                .iter()
                .chain(&plan.fused)
                .filter(|&&e| e)
                .count()
        };
        let dropped_before = dropped(&plan);
        fuse::fuse(func, &mut plan, lse);
        if dropped(&plan) != dropped_before {
            crate::frame::repack_after_elide(func, &plan, &abi, &mut frame, &mut framed, |sig| {
                classify(sig, packing)
            })?;
        }
    }
    let tail_calls = if full {
        crate::frame::find_tail_calls(func, &plan, &abi, true, &|sig| classify(sig, packing))
    } else {
        Vec::new()
    };
    let lr = plan
        .locations
        .iter()
        .any(|loc| matches!(loc, Location::Reg(LR)));
    let bracket = !lr && tail_calls.is_empty() && crate::frame::brackets_calls(func, &frame, &abi);
    if !tail_calls.is_empty()
        && framed
        && !lr
        && !crate::frame::needs_frame_without_tails(func, &frame, &abi, &tail_calls)
    {
        framed = false;
    }
    framed = (framed || lr) && !bracket;
    let shrink = full
        .then(|| {
            shrink_wrap(
                func,
                framed,
                &plan,
                &frame,
                &abi.params,
                &[FP, LR, IP],
                |inst, _| crate::frame::is_tail(&tail_calls, inst),
            )
        })
        .flatten();
    let mut asm = Asm::default();
    let tlv_cache = (full && tlv_cells && framed)
        .then(|| frames::plan_tlv_cache(func, &plan, &defs, &mut frame, &mut asm))
        .flatten();
    let blocks = func.blocks.iter().map(|_| asm.label()).collect();
    let emit = Emit {
        func,
        frame,
        framed,
        bracket,
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
        spills: crate::spills::SpillState::new(if full {
            crate::spills::pred_totals(func, &order, &copied)
        } else {
            Vec::new()
        }),
        next: None,
        copied,
        direct_calls: target.direct_calls,
        tlv_cells,
        tlv_cache,
        lse: target.os == Os::Darwin,
        sp_bias: 0,
        shrink,
        trap_label: None,
        cold_calls: Vec::new(),
        lines: Vec::new(),
        in_fast: false,
        tails: Tails::new(),
        tail_calls,
    };
    drive(emit, &order)
}

pub(crate) const DICT_STUB_RELOCS: [(u32, crate::reloc::RelocKind); 2] = [
    (0, crate::reloc::RelocKind::Arm64Movw64),
    (16, crate::reloc::RelocKind::Arm64Movw64),
];

pub(crate) fn dict_stub(
    stub_exec: usize,
    body_exec: usize,
    dict_addr: usize,
    tier_up: Option<DictTierUp>,
) -> Result<Vec<u8>, String> {
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
    bracket: bool,
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
    spills: crate::spills::SpillState<Reg, Vreg>,
    next: Option<Block>,
    copied: Vec<bool>,
    direct_calls: bool,
    tlv_cells: bool,
    tlv_cache: Option<frames::TlvCache>,
    lse: bool,
    sp_bias: i32,
    shrink: Option<ShrinkWrap>,
    trap_label: Option<Label>,
    cold_calls: Vec<frames::ColdCall>,
    lines: Vec<(u32, u64)>,
    in_fast: bool,
    tails: Tails<'a, Label>,
    tail_calls: Vec<bool>,
}

impl<'a> Emitter<'a> for Emit<'a> {
    fn func(&self) -> &'a Function {
        self.func
    }
    fn is_elided(&self, value: Value) -> bool {
        self.plan.elided[value.index()] || matches!(self.plan.location(value), Location::Offset(..))
    }
    fn zero_stack_store(&self, inst: Inst) -> Option<(i32, u32)> {
        crate::frame::stack_zero_store(self.func, &self.frame.slots, inst, |v| {
            self.plan.const_of(v)
        })
    }
    fn emit_zero_frame(&mut self, offset: i32, bytes: u32) {
        self.asm.store(Size::of(bytes), self.frame_mem(offset), ZR);
    }
    fn try_tail_call(&mut self, inst: Inst) -> Result<bool, String> {
        self.emit_tail_call(inst)
    }

    fn prologue(&mut self) {
        if !self.framed {
            return;
        }
        self.push_record();
        let size = self.frame.size;
        if size > PAGE {
            self.probe(size);
        } else if size > 0 {
            self.asm.add_imm(true, false, Size::X, SP, SP, size);
        }
        emit_save_pairs(
            &mut self.asm,
            &self.frame.saves,
            0,
            |asm, disp, a, b| match b {
                Some(r0) => asm.stp_offset(a, r0, FP, disp),
                None => asm.store(Size::X, Mem::new(FP, disp), a),
            },
        );
        emit_save_pairs(
            &mut self.asm,
            &self.frame.float_saves,
            self.frame.saves.len(),
            |asm, disp, a, b| match b {
                Some(v0) => asm.fstp_offset(Vreg(a), Vreg(v0), FP, disp),
                None => asm.vstore(8, Mem::new(FP, disp), Vreg(a)),
            },
        );
        if let Some(slot) = self
            .tlv_cache
            .as_ref()
            .filter(|c| c.needs_zero)
            .map(|c| c.slot)
        {
            let at = self.frame_mem(slot);
            self.asm.store(Size::X, at, ZR);
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
        self.resume_spills(block);
        if let Some(cache) = &mut self.tlv_cache {
            cache.live = cache.dom_known[block.index()];
            cache.may_live = cache.may_known[block.index()];
        }
        self.asm.bind(self.blocks[block.index()]);
    }
    fn bind_entry(&mut self, block: Block) {
        if let Some(cache) = &mut self.tlv_cache {
            cache.live = cache.dom_known[block.index()];
            cache.may_live = cache.may_known[block.index()];
        }
        self.asm.bind(self.blocks[block.index()]);
    }
    fn deferred_prologue(&self) -> bool {
        self.shrink.is_some()
    }
    fn before_block_bind(&mut self, _block: Block) -> Result<(), String> {
        self.asm.maybe_island()?;
        Ok(())
    }
    fn on_block_bound(&mut self, block: Block) -> Result<(), String> {
        self.in_fast = self.shrink.as_ref().is_some_and(|s| s.fast[block.index()]);
        Ok(())
    }
    fn clear_flags(&mut self) {
        self.flags = None;
        self.last_cmp = None;
    }

    fn inst(&mut self, inst: Inst) -> Result<(), String> {
        let func = self.func;
        let data = &func.insts[inst.index()];
        let keep_flags = self.preserves_flags(&data.op, data.result);
        if !keep_flags {
            self.flags = None;
            if !matches!(data.op, Op::Icmp(_, x, _) if !self.ty(x).is_vector())
                && !matches!(data.op, Op::Select(cond, ..) if !self.ty(cond).is_vector())
            {
                self.last_cmp = None;
            }
        }
        let result = data.result;
        let value = || result.ok_or_else(|| "a value instruction without a result".to_owned());
        match &data.op {
            Op::Call {
                sig,
                callee,
                args,
                results,
            } => {
                if self.bracket {
                    self.push_record();
                }
                let res = emit_call(self, inst, (*sig, *callee), args, results);
                if self.bracket {
                    self.asm.ldp_post(FP, LR, SP, 16);
                }
                res
            }
            Op::TryCall {
                sig,
                callee,
                args,
                normal,
                handler,
            } => {
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
                self.record_edge_spill(*block, self.spills.cache);
                self.invalidate_spills();
                emit_goto(self, *block, true);
                Ok(())
            }
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                let res = emit_brif(
                    self,
                    *condition,
                    (*then, then_args),
                    (*otherwise, else_args),
                );
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
                let target = self.trap_target();
                self.asm.b(target);
                self.invalidate_spills();
                Ok(())
            }
            Op::Trapz(condition) => {
                self.trapz(*condition);
                Ok(())
            }
            op => {
                if self.vector_inst(op, value)? {
                    self.invalidate_spills();
                    return Ok(());
                }
                if matches!(op, Op::Fma(..) | Op::Fcmp(..)) {
                    self.invalidate_spills();
                }
                let res = self.value_op(op, value);
                if matches!(op, Op::Fma(..) | Op::Fcmp(..)) {
                    self.invalidate_spills();
                }
                res
            }
        }
    }

    fn trap(&mut self) {
        let target = self.trap_target();
        self.asm.b(target);
    }

    fn pos(&self) -> u32 {
        self.asm.pos()
    }

    fn lines(&mut self) -> &mut Vec<(u32, u64)> {
        &mut self.lines
    }

    fn finish(mut self, _starts: &mut [u32]) -> Result<Code, String> {
        self.emit_cold_calls();
        self.asm.maybe_island()?;
        if let Some(label) = self.trap_label.take() {
            self.asm.bind(label);
            self.asm.brk();
        }
        emit_stubs(&mut self)?;
        Ok(Code {
            bytes: self.asm.finish()?,
            sites: self.sites,
            relocs: self.relocs,
            loops: Vec::new(),
            lines: Vec::new(),
            unwind_drops: crate::frame::resolve_unwind_drops(self.func, &self.frame),
        })
    }
}

impl Emit<'_> {
    fn push_record(&mut self) {
        self.asm.stp_pre(FP, LR, SP, -16);
        self.asm.mov_sp(FP, SP);
    }

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

    pub(super) fn epilogue_body(&mut self) {
        if self.framed {
            emit_save_pairs(
                &mut self.asm,
                &self.frame.saves[..self.frame.restore_saves],
                0,
                |asm, disp, a, b| match b {
                    Some(r0) => asm.ldp_offset(a, r0, FP, disp),
                    None => asm.load(Size::X, false, a, Mem::new(FP, disp)),
                },
            );
            emit_save_pairs(
                &mut self.asm,
                &self.frame.float_saves[..self.frame.restore_floats],
                self.frame.saves.len(),
                |asm, disp, a, b| match b {
                    Some(v0) => asm.fldp_offset(Vreg(a), Vreg(v0), FP, disp),
                    None => asm.vload(8, Vreg(a), Mem::new(FP, disp)),
                },
            );
            if self.frame.size > 0 {
                self.asm.mov_sp(SP, FP);
            }
            self.asm.ldp_post(FP, LR, SP, 16);
        }
    }

    pub(super) fn restore_extra_saves(&mut self) {
        emit_save_pairs(
            &mut self.asm,
            &self.frame.saves[self.frame.restore_saves..],
            self.frame.restore_saves,
            |asm, disp, a, b| match b {
                Some(r0) => asm.ldp_offset(a, r0, FP, disp),
                None => asm.load(Size::X, false, a, Mem::new(FP, disp)),
            },
        );
        emit_save_pairs(
            &mut self.asm,
            &self.frame.float_saves[self.frame.restore_floats..],
            self.frame.saves.len() + self.frame.restore_floats,
            |asm, disp, a, b| match b {
                Some(v0) => asm.fldp_offset(Vreg(a), Vreg(v0), FP, disp),
                None => asm.vload(8, Vreg(a), Mem::new(FP, disp)),
            },
        );
    }

    fn epilogue(&mut self) {
        self.epilogue_body();
        self.asm.ret();
    }

    value_op!();
}
