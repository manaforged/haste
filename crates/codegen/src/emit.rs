use crate::alloc::{Location, Plan};
use crate::frame::{Abi, Emitter, Frame, Loc, ShrinkWrap};
use crate::ir::{Block, Op, SigRef, Value};
use crate::park::ParkReg;
use crate::Site;

pub(crate) trait Backend<'a, R: Copy + PartialEq, L: Copy>: Emitter<'a> {
    fn plan(&self) -> &Plan<R>;
    fn plan_mut(&mut self) -> &mut Plan<R>;
    fn frame(&self) -> &Frame<R>;
    fn abi(&self) -> &Abi<R>;
    fn shrink(&self) -> Option<ShrinkWrap>;
    fn slow_thunk(&self) -> L;
    fn in_shrink_entry(&self) -> bool;
    fn next_block(&self) -> Option<Block>;
    fn block_label(&self, block: Block) -> L;
    fn pos(&self) -> u32;
    fn new_label(&mut self) -> L;
    fn bind_label(&mut self, label: L);
    fn jump_label(&mut self, label: L);
    fn branch_cond(&mut self, condition: Value, invert: bool, label: L);
    fn edge(&mut self, block: Block, args: &[Value]) -> Result<(), String>;
    fn arguments(&mut self, sig: SigRef, callee: Value, args: &[Value]) -> Result<Abi<R>, String>;
    fn store_locs(&mut self, values: &[Value], locs: &[Loc<R>]);
    fn store_loc(&mut self, value: Value, loc: Loc<R>);
    fn load_incoming(&mut self, value: Value, offset: u32);
    fn intake_struct(&mut self, value: Value, offset: u32);
    fn park_spill(&mut self, value: Value, reg: ParkReg<R>);
    fn park_reload(&mut self, value: Value, reg: ParkReg<R>);
    fn put_exception(&mut self, value: Value);
    fn push_site(&mut self, site: Site);
}

pub(crate) fn intake_params<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(emit: &mut E, entry: Block) -> Result<(), String> {
    let params = emit.func().blocks[entry.index()].params.iter().copied().zip(emit.abi().params.clone());
    let (registers, memory): (Vec<_>, Vec<_>) = params.partition(|(value, _)| {
        let plan = emit.plan();
        plan.reg_of(*value).is_some() || plan.float_of(*value).is_some()
    });
    for (value, loc) in memory.into_iter().chain(registers) {
        match loc {
            Loc::Stack(offset) => emit.load_incoming(value, offset),
            Loc::Struct(offset, _) => emit.intake_struct(value, offset),
            _ => emit.store_loc(value, loc),
        }
    }
    Ok(())
}

pub(crate) fn emit_slow_prologue<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(emit: &mut E, slow_entry: Block) -> Result<(), String> {
    emit.prologue();
    if let Some(entry) = emit.func().entry
        && let Some(term) = emit.func().blocks[entry.index()].insts.last()
        && let Op::Brif(_, then_b, then_args, else_b, else_args) = &emit.func().insts[term.index()].op
    {
        let args = if *then_b == slow_entry { then_args.as_slice() } else if *else_b == slow_entry { else_args.as_slice() } else { &[] };
        emit.edge(slow_entry, args)?;
    }
    Ok(())
}

pub(crate) fn emit_goto<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(emit: &mut E, block: Block, last: bool) {
    let target = emit.func().jump_target(block);
    if last && (emit.next_block() == Some(block) || emit.next_block() == Some(target)) {
        return;
    }
    let label = emit.block_label(target);
    emit.jump_label(label);
}

fn target_label<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(emit: &E, block: Block) -> L {
    if emit.in_shrink_entry() && emit.shrink().is_some_and(|s| s.use_slow_thunk && s.slow_entry == block) {
        emit.slow_thunk()
    } else {
        emit.block_label(emit.func().jump_target(block))
    }
}

fn goto_branch<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(emit: &mut E, block: Block, last: bool) {
    if emit.in_shrink_entry() && emit.shrink().is_some_and(|s| s.use_slow_thunk && s.slow_entry == block) {
        let slow = emit.slow_thunk();
        emit.jump_label(slow);
    } else {
        emit_goto(emit, block, last);
    }
}

fn shrink_edges<'a, 'b, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(
    emit: &E,
    then: (Block, &'b [Value]),
    otherwise: (Block, &'b [Value]),
) -> ((Block, &'b [Value]), (Block, &'b [Value])) {
    if !emit.in_shrink_entry() {
        return (then, otherwise);
    }
    let slow = emit.shrink().map(|s| s.slow_entry);
    (
        (then.0, if Some(then.0) == slow { &[][..] } else { then.1 }),
        (otherwise.0, if Some(otherwise.0) == slow { &[][..] } else { otherwise.1 }),
    )
}

pub(crate) fn emit_brif<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(
    emit: &mut E,
    condition: Value,
    then: (Block, &[Value]),
    otherwise: (Block, &[Value]),
) -> Result<(), String> {
    let (then, otherwise) = shrink_edges(emit, then, otherwise);
    if let Some(known) = emit.plan().const_of(condition) {
        let (block, args) = if crate::fold::extend(known, emit.func().ty(condition), false) != 0 { then } else { otherwise };
        emit.edge(block, args)?;
        goto_branch(emit, block, true);
        return Ok(());
    }
    let then_moves = emit.plan().moves(emit.func(), emit.frame(), then.0, then.1);
    let else_moves = emit.plan().moves(emit.func(), emit.frame(), otherwise.0, otherwise.1);
    let falls = emit.next_block().is_some_and(|next| next == then.0 || next == emit.func().jump_target(then.0));
    if !else_moves && (then_moves || falls) {
        let target = target_label(emit, otherwise.0);
        emit.branch_cond(condition, true, target);
        emit.edge(then.0, then.1)?;
        goto_branch(emit, then.0, true);
        return Ok(());
    }
    if !then_moves {
        let target = target_label(emit, then.0);
        emit.branch_cond(condition, false, target);
        emit.edge(otherwise.0, otherwise.1)?;
        goto_branch(emit, otherwise.0, true);
        return Ok(());
    }
    let other = emit.new_label();
    emit.branch_cond(condition, true, other);
    emit.edge(then.0, then.1)?;
    goto_branch(emit, then.0, false);
    emit.bind_label(other);
    emit.edge(otherwise.0, otherwise.1)?;
    goto_branch(emit, otherwise.0, true);
    Ok(())
}

pub(crate) fn emit_call<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(
    emit: &mut E,
    inst: crate::ir::Inst,
    (sig, callee): (SigRef, Value),
    args: &[Value],
    results: &[Value],
) -> Result<(), String> {
    let parked = emit.plan().parked_at(inst).to_vec();
    for &(value, reg, _) in &parked {
        emit.park_spill(value, reg);
        emit.plan_mut().locations[value.index()] = Location::Slot;
    }
    let abi = emit.arguments(sig, callee, args)?;
    emit.store_locs(results, &abi.returns);
    for (value, reg, lives) in parked {
        if lives {
            emit.park_reload(value, reg);
        }
        emit.plan_mut().locations[value.index()] = match reg {
            ParkReg::Int(r) => Location::Reg(r),
            ParkReg::Float(f) => Location::Float(f),
        };
    }
    Ok(())
}

pub(crate) fn emit_try_call<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(
    emit: &mut E,
    (sig, callee): (SigRef, Value),
    args: &[Value],
    (normal, handler): (Block, Block),
) -> Result<(), String> {
    emit.arguments(sig, callee, args)?;
    let ret = emit.pos();
    let normal_label = emit.block_label(normal);
    emit.jump_label(normal_label);
    let pad = emit.pos();
    let [exception] = emit.func().blocks[handler.index()].params[..] else {
        return Err("a try_call handler takes exactly the exception".to_owned());
    };
    emit.put_exception(exception);
    let handler_label = emit.block_label(handler);
    emit.jump_label(handler_label);
    let frame_offset = emit.frame().size;
    emit.push_site(Site { ret, handler: pad, frame_offset });
    Ok(())
}
