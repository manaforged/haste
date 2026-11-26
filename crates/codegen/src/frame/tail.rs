use crate::alloc::{CallTarget, Plan};
use crate::frame::{Abi, Loc};
use crate::ir::{ArgumentPurpose, Function, Inst, Op, Signature};
use crate::reloc::RuntimeSymbol;

fn has_stack_escape(func: &Function) -> bool {
    func.blocks.iter().flat_map(|b| &b.insts).any(|inst| {
        matches!(
            func.insts[inst.index()].op,
            Op::StackAddr(..) | Op::FramePointer
        )
    })
}

fn returns_results(func: &Function, term: &Op, results: &[crate::ir::Value]) -> bool {
    match term {
        Op::Return(vals) => vals == results,
        Op::Jump(target, args) if args == results => {
            if results
                .iter()
                .enumerate()
                .any(|(i, v)| results[..i].contains(v))
            {
                return false;
            }
            let dst = func.jump_target(*target);
            let block = &func.blocks[dst.index()];
            block.insts.len() == 1
                && matches!(&func.insts[block.insts[0].index()].op, Op::Return(vals) if vals == &block.params)
        }
        _ => false,
    }
}

/// Computes which `Op::Call` instructions are eligible for sibling tail-call emission.
pub(crate) fn find_tail_calls<R: Copy + PartialEq>(
    func: &Function,
    plan: &Plan<R>,
    caller_abi: &Abi<R>,
    allow_direct: bool,
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Vec<bool> {
    if !func.unwind_drops.is_empty()
        || func.signature.params.iter().any(|p| {
            matches!(
                p.purpose,
                ArgumentPurpose::StructReturn | ArgumentPurpose::Dict
            )
        })
        || has_stack_escape(func)
    {
        return Vec::new();
    }
    let mut out = Vec::new();
    for block in &func.blocks {
        if block.insts.len() < 2 {
            continue;
        }
        let call_inst = block.insts[block.insts.len() - 2];
        let term_inst = block.insts[block.insts.len() - 1];
        let Op::Call {
            sig,
            callee,
            results,
            ..
        } = &func.insts[call_inst.index()].op
        else {
            continue;
        };
        if !returns_results(func, &func.insts[term_inst.index()].op, results) {
            continue;
        }
        if func.is_thread_cell(*callee)
            || matches!(
                plan.through[callee.index()],
                Some(CallTarget::Runtime(
                    RuntimeSymbol::Malloc | RuntimeSymbol::Free
                ))
            )
        {
            continue;
        }
        if !allow_direct
            && matches!(
                plan.through[callee.index()],
                Some(CallTarget::Symbol(_) | CallTarget::Slot(_) | CallTarget::Runtime(_))
            )
        {
            continue;
        }
        let callee_sig = &func.sigs[sig.index()];
        if callee_sig.call_conv != func.signature.call_conv
            || callee_sig.returns != func.signature.returns
            || callee_sig.params.iter().any(|p| {
                matches!(
                    p.purpose,
                    ArgumentPurpose::StructReturn | ArgumentPurpose::Dict
                )
            })
        {
            continue;
        }
        let Ok(callee_abi) = classify(callee_sig) else {
            continue;
        };
        if callee_abi.stack != 0
            || callee_abi.returns != caller_abi.returns
            || callee_abi
                .params
                .iter()
                .any(|loc| matches!(loc, Loc::Stack(_) | Loc::Struct(..)))
        {
            continue;
        }
        if out.is_empty() {
            out.resize(func.insts.len(), false);
        }
        out[call_inst.index()] = true;
    }
    out
}

/// Reports whether `func` still needs a frame once `tail_calls` are excluded from call checks.
pub(crate) fn needs_frame_without_tails<R>(
    func: &Function,
    frame: &crate::frame::Frame<R>,
    abi: &Abi<R>,
    tail_calls: &[bool],
) -> bool {
    func.keep_fp
        || abi
            .params
            .iter()
            .any(|loc| matches!(loc, Loc::Stack(_) | Loc::Struct(..)))
        || !func.unwind_drops.is_empty()
        || frame.size > 0
        || !frame.saves.is_empty()
        || !frame.float_saves.is_empty()
        || func.blocks.iter().flat_map(|b| &b.insts).any(|&inst| {
            match func.insts[inst.index()].op {
                Op::Call { .. } => !tail_calls.get(inst.index()).copied().unwrap_or(false),
                Op::TryCall { .. } | Op::FramePointer => true,
                _ => false,
            }
        })
}

/// Helper predicate for checking a `tail_calls` mask.
pub(crate) fn is_tail(tail_calls: &[bool], inst: Inst) -> bool {
    tail_calls.get(inst.index()).copied().unwrap_or(false)
}
