mod asm;
mod atomic;
mod bits;
mod call;
mod emit;
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
mod vector_narrow;
mod vector_reduce;
mod wide;
mod ymm;

use asm::{Alu, Asm, Cond, Label, Mem, Reg, Xmm};
use features::Features;
use frame::{Abi, SAVED, SAVED_RBP, classify};
pub(crate) use ymm::host_avx2;

use crate::alloc::{Plan, Regs};
use crate::emit::Tails;
use crate::frame::{Frame, Prepared, ShrinkWrap, drive, prepare_split, shrink_wrap};
use crate::ir::{Block, Function, Inst, Op, Value};
use crate::reloc::{DictTierUp, StencilReloc};
use crate::target::{CallAbi, Target};
use crate::{Code, Site};

const REGS: Regs<Reg> = Regs {
    saved: &SAVED,
    saved_float: &[],
    leaf: &[
        Reg::Rsi,
        Reg::Rdi,
        Reg::R9,
        Reg::R8,
        Reg::R10,
        Reg::Rdx,
        Reg::R11,
    ],
    wide_scratch: &wide::WIDE_SCRATCH,
    op_scratch: &[Reg::Rsi, Reg::Rdx, Reg::R11],
    preserved_clobbers: &[],
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
    atomic_addrs: crate::alloc::AtomicAddrMode::Full,
    call_zext: false,
    imm_store,
    high_product: Some(Reg::Rdx),
};

fn imm_store(imm: i64) -> u64 {
    if i32::try_from(imm).is_ok() { 1 } else { 2 }
}

const REGS_WIN64: Regs<Reg> = Regs {
    leaf: &[Reg::R9, Reg::R8, Reg::R10, Reg::Rdx, Reg::R11],
    ..REGS
};

const REGS_RBP: Regs<Reg> = Regs {
    saved: &SAVED_RBP,
    ..REGS
};

const REGS_WIN64_RBP: Regs<Reg> = Regs {
    saved: &SAVED_RBP,
    ..REGS_WIN64
};

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
    let full = target.effort == crate::Effort::Full;
    let frame_cache = target.direct_calls && target.os == crate::target::Os::Linux;
    let preserve_tc = frame_cache && abi_kind == CallAbi::SysV;
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
    ) = prepare_split(func, regs, (full, preserve_tc), |sig| {
        classify(sig, abi_kind)
    })?;
    let func = split.as_ref().unwrap_or(func);
    let features = Features::for_target(target);
    let (roots, pure_test) = if full {
        let dropped_before = plan.dropped_count();
        let fused = location::fuse(func, &mut plan, &features);
        if plan.dropped_count() != dropped_before {
            crate::frame::repack_after_elide(func, &plan, &abi, &mut frame, &mut framed, |sig| {
                classify(sig, abi_kind)
            })?;
        }
        fused
    } else {
        (vec![None; func.types.len()], vec![false; func.types.len()])
    };
    let tail_calls = if full {
        crate::frame::find_tail_calls(func, &plan, &abi, &|sig| classify(sig, abi_kind))
    } else {
        Vec::new()
    };
    if !tail_calls.is_empty()
        && framed
        && !crate::frame::needs_frame_without_tails(func, &frame, &abi, &tail_calls)
    {
        framed = false;
    }
    let shrink = full
        .then(|| {
            shrink_wrap(
                func,
                framed,
                &plan,
                &frame,
                &abi.params,
                &FRAME_REGS,
                |inst, op| {
                    crate::frame::is_tail(&tail_calls, inst)
                        || (preserve_tc
                            && crate::alloc::is_preserved_call_op(
                                func,
                                op,
                                &plan.through,
                                &plan.locations,
                                &plan.elided,
                            ))
                },
            )
        })
        .flatten();
    let (base, bias) = if framed && !keeps {
        (Reg::Rsp, frame.size as i32)
    } else {
        (Reg::Rbp, 0)
    };
    let mut asm = Asm::default();
    asm.avx = features.avx;
    let blocks = func.blocks.iter().map(|_| asm.label()).collect();
    let defs = crate::fold::defs(func);
    let (overflow_read, adc_sum, flag_carry) = if full {
        (
            wide::overflow_reads(func),
            wide::adc_sums(func, &plan, &defs),
            wide::flag_carries(func, &plan, &defs),
        )
    } else {
        let n = func.types.len();
        (vec![false; n], vec![false; n], vec![false; n])
    };
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
        spills: crate::spills::SpillState::new(if full {
            crate::spills::pred_totals(func, &order, &copied)
        } else {
            Vec::new()
        }),
        next: None,
        last_block,
        at_end: false,
        trap_label: None,
        lines: Vec::new(),
        features,
        copied,
        direct_calls: target.direct_calls,
        frame_cache,
        cold_calls: Vec::new(),
        shrink,
        in_fast: false,
        low_product: None,
        overflow_read,
        adc_sum,
        flag_carry,
        carry_flags: None,
        tails: Tails::new(),
        tail_calls,
    };
    drive(emit, &order)
}

fn keeps_frame_pointer(func: &Function) -> bool {
    func.keep_fp
        || func.needs_unwind_frame()
        || func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .any(|inst| {
                matches!(
                    func.insts[inst.index()].op,
                    Op::TryCall { .. } | Op::FramePointer
                )
            })
}

pub(crate) const DICT_STUB_RELOCS: [(u32, crate::reloc::RelocKind); 2] = [
    (2, crate::reloc::RelocKind::Abs64),
    (12, crate::reloc::RelocKind::Abs64),
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
        asm.mov_abs64_val(Reg::R10, tier.counter_addr as u64);
        asm.encode(
            &[],
            false,
            &[0x83],
            Alu::Add as u8,
            Mem::new(Reg::R10, 0).into(),
            false,
        );
        asm.bytes(&[1]);
        asm.encode(
            &[],
            false,
            &[0x81],
            Alu::Cmp as u8,
            Mem::new(Reg::R10, 0).into(),
            false,
        );
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
    spills: crate::spills::SpillState<Reg, Xmm>,
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
    adc_sum: Vec<bool>,
    flag_carry: Vec<bool>,
    carry_flags: Option<(bool, [Value; 3])>,
    tails: Tails<'a, Label>,
    tail_calls: Vec<bool>,
}
