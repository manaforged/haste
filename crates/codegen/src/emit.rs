use crate::alloc::{Location, Plan, Regs};
use crate::frame::{Abi, Emitter, Frame, Loc, ShrinkWrap};
use crate::ir::{Binary, Block, Function, Inst, Op, SigRef, Value};
use crate::park::ParkReg;
use crate::Site;
use std::ops::Range;

const LOOP_SPAN: u32 = 256;

pub(crate) trait Backend<'a, R: Copy + PartialEq, L: Copy>: Emitter<'a> {
    fn plan(&self) -> &Plan<R>;
    fn plan_mut(&mut self) -> &mut Plan<R>;
    fn frame(&self) -> &Frame<R>;
    fn abi(&self) -> &Abi<R>;
    fn shrink(&self) -> Option<&ShrinkWrap>;
    fn in_fast(&self) -> bool;
    fn is_fast(&self, block: Block) -> bool { self.shrink().is_some_and(|s| s.fast[block.index()]) }
    fn is_split_ret(&self, block: Block) -> bool {
        self.in_fast() && self.shrink().is_some_and(|s| s.split_ret[self.func().jump_target(block).index()])
    }
    fn exits_fast(&self, block: Block) -> bool {
        self.in_fast() && !self.is_fast(block) && !self.is_split_ret(block)
    }
    fn ret_fast(&mut self, values: &[Value]) -> Result<(), String>;
    fn fast_ret(&mut self, block: Block) -> Result<(), String> {
        let Some(Op::Return(values)) = self.func().last(self.func().jump_target(block)) else {
            return Err("split_ret target is not a return".to_owned());
        };
        self.ret_fast(values)
    }
    fn next_block(&self) -> Option<Block>;
    fn block_label(&self, block: Block) -> L;
    fn new_label(&mut self) -> L;
    fn bind_label(&mut self, label: L);
    fn jump_label(&mut self, label: L);
    fn branch_cond(&mut self, condition: Value, invert: bool, label: L);
    fn edge(&mut self, block: Block, args: &[Value]) -> Result<(), String>;
    fn arguments(&mut self, sig: SigRef, callee: Value, args: &[Value], is_try: bool) -> Result<Abi<R>, String>;
    fn store_locs(&mut self, values: &[Value], locs: &[Loc<R>]);
    fn store_loc(&mut self, value: Value, loc: Loc<R>);
    fn load_incoming(&mut self, value: Value, offset: u32);
    fn intake_struct(&mut self, value: Value, offset: u32);
    fn park_spill(&mut self, value: Value, reg: ParkReg<R>);
    fn park_reload(&mut self, value: Value, reg: ParkReg<R>);
    fn put_exception(&mut self, value: Value);
    fn push_site(&mut self, site: Site);
    fn tails(&mut self) -> &mut Tails<'a, L>;
    /// Places a branch island here when pending short branches near their range.
    fn island(&mut self) -> Result<(), String> { Ok(()) }
}

#[derive(Clone, Copy)]
enum StubKind {
    Plain,
    Prologue,
    FastRet,
}

struct Stub<'a, L> {
    label: L,
    block: Block,
    args: &'a [Value],
    kind: StubKind,
}

pub(crate) struct Tails<'a, L> {
    stubs: Vec<Stub<'a, L>>,
    cold: Option<Vec<bool>>,
}

impl<L> Tails<'_, L> {
    pub(crate) fn new() -> Self {
        Self { stubs: Vec::new(), cold: None }
    }

    fn colder(&mut self, func: &Function, condition: Value, edge: Block, other: Block) -> bool {
        let cold = self.cold.get_or_insert_with(|| crate::live::cold(func));
        if cold[edge.index()] != cold[other.index()] {
            return cold[edge.index()];
        }
        let cfg = func.cfg();
        let loops = cfg.loops();
        let branches = |from: &&Block| {
            matches!(func.last(**from), Some(Op::Brif(test, a, _, b, _)) if *test == condition && ((*a == edge && *b == other) || (*a == other && *b == edge)))
        };
        let leaves = |from: &Block| loops.list.iter().any(|each| each.body[from.index()] && each.body[other.index()] && !each.body[edge.index()]);
        let mut froms = cfg.preds_of(other).iter().filter(branches).peekable();
        froms.peek().is_some() && froms.all(leaves)
    }
}

pub(crate) fn loop_spans(func: &Function, order: &[Block], starts: &[u32]) -> Vec<Range<u32>> {
    let cfg = func.cfg();
    let loops = cfg.loops();
    let mut placed = vec![None; func.blocks.len()];
    for (block, span) in order.iter().zip(starts.windows(2)) {
        placed[block.index()] = Some(span[0]..span[1]);
    }
    loops
        .list
        .iter()
        .filter(|each| each.body.iter().zip(&loops.depths).all(|(inside, depth)| !inside || *depth <= each.depth))
        .filter_map(|each| {
            let spans = || each.body.iter().zip(&placed).filter(|(inside, _)| **inside).filter_map(|(_, span)| span.clone());
            let start = spans().map(|span| span.start).min()?;
            let end = spans().map(|span| span.end).max()?;
            (end - start <= LOOP_SPAN).then_some(start..end)
        })
        .collect()
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

pub(crate) fn emit_goto<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(emit: &mut E, block: Block, last: bool) {
    let target = emit.func().jump_target(block);
    if last && (emit.next_block() == Some(block) || emit.next_block() == Some(target)) {
        return;
    }
    let label = emit.block_label(target);
    emit.jump_label(label);
}

fn push_stub<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(
    emit: &mut E,
    block: Block,
    args: &'a [Value],
    kind: StubKind,
) -> L {
    let label = emit.new_label();
    emit.tails().stubs.push(Stub { label, block, args, kind });
    label
}

fn falls<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(emit: &E, block: Block) -> bool {
    emit.next_block().is_some_and(|next| next == block || next == emit.func().jump_target(block))
}

/// Whether `block` closes a loop by branching back to its header.
fn is_latch(func: &Function, block: Block) -> bool {
    func.cfg().loops().list.iter().any(|each| each.latches.contains(&block))
}

fn set_aside<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(
    emit: &mut E,
    condition: Value,
    (then, then_moves): ((Block, &'a [Value]), bool),
    (otherwise, else_moves): ((Block, &'a [Value]), bool),
) -> Result<bool, String> {
    if emit.in_fast() || !(then_moves || else_moves) {
        return Ok(false);
    }
    let func = emit.func();
    let (cold, hot, hot_moves, invert) = if then_moves && emit.tails().colder(func, condition, then.0, otherwise.0) {
        (then, otherwise, else_moves, false)
    } else if else_moves && emit.tails().colder(func, condition, otherwise.0, then.0) {
        (otherwise, then, then_moves, true)
    } else {
        return Ok(false);
    };
    if !hot_moves && !falls(emit, hot.0) {
        return Ok(false);
    }
    let label = push_stub(emit, cold.0, cold.1, StubKind::Plain);
    emit.branch_cond(condition, invert, label);
    emit.edge(hot.0, hot.1)?;
    emit_goto(emit, hot.0, true);
    Ok(true)
}

pub(crate) fn emit_stubs<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(emit: &mut E) -> Result<(), String> {
    for stub in std::mem::take(&mut emit.tails().stubs) {
        emit.island()?;
        emit.clear_flags();
        emit.bind_label(stub.label);
        match stub.kind {
            StubKind::Plain => {
                emit.edge(stub.block, stub.args)?;
                let target = emit.block_label(emit.func().jump_target(stub.block));
                emit.jump_label(target);
            }
            StubKind::Prologue => {
                emit.prologue();
                emit.edge(stub.block, stub.args)?;
                let target = emit.block_label(emit.func().jump_target(stub.block));
                emit.jump_label(target);
            }
            StubKind::FastRet => {
                emit.edge(stub.block, stub.args)?;
                emit.fast_ret(stub.block)?;
            }
        }
    }
    Ok(())
}

fn simple_move_val<R: Copy + PartialEq>(func: &Function, plan: &Plan<R>, v: Value) -> bool {
    let flag = |list: &[bool]| list.get(v.index()).copied().unwrap_or(true);
    !flag(&plan.elided)
        && !flag(&plan.deferred)
        && !flag(&plan.fused)
        && plan.pairs.get(v.index()).is_some_and(Option::is_none)
        && (func.ty(v).is_small_int() || func.ty(v).is_float())
}

fn hoistable_move<R: Copy + PartialEq>(func: &Function, plan: &Plan<R>, arg: Value, param: Value) -> bool {
    simple_move_val(func, plan, arg)
        && simple_move_val(func, plan, param)
        && matches!(plan.locations[arg.index()], Location::Reg(_) | Location::Float(_) | Location::Slot)
        && matches!(plan.locations[param.index()], Location::Reg(_) | Location::Float(_))
}

fn hoist_common_moves<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(
    emit: &mut E,
    condition: Value,
    then: (Block, &'a [Value]),
    otherwise: (Block, &'a [Value]),
) -> Result<bool, String> {
    if emit.in_fast() {
        return Ok(false);
    }
    let (func, plan, frame) = (emit.func(), emit.plan(), emit.frame());
    let then_steps = crate::frame::edge_steps(func, &plan.locations, &frame.values, then.1, &func.blocks[then.0.index()].params);
    let else_steps = crate::frame::edge_steps(func, &plan.locations, &frame.values, otherwise.1, &func.blocks[otherwise.0.index()].params);
    if then_steps.is_empty() || then_steps.len() != else_steps.len() {
        return Ok(false);
    }
    let same = then_steps.iter().zip(&else_steps).all(|(ts, es)| match (ts, es) {
        (crate::frame::Step::Move(ta, tp), crate::frame::Step::Move(ea, ep)) => {
            plan.same_home(frame, *ta, *ea)
                && plan.same_home(frame, *tp, *ep)
                && func.ty(*ta).bytes() == func.ty(*ea).bytes()
                && func.ty(*tp).bytes() == func.ty(*ep).bytes()
                && hoistable_move(func, plan, *ta, *tp)
                && plan.hoist.reads_free(plan.locations[tp.index()], condition)
        }
        _ => false,
    });
    if !same {
        return Ok(false);
    }
    emit.edge(then.0, then.1)?;
    Ok(true)
}

/// Runs the moves of a back edge to loop head `moved` whose other edge leaves that loop ahead of the branch, when each copies a register or stack home into a register that neither `plain` nor the condition reads, so the branch reaches `moved` without a jump of its own.
fn hoist_moves<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(
    emit: &mut E,
    condition: Value,
    (moved, to_then): ((Block, &'a [Value]), bool),
    plain: Block,
) -> Result<bool, String> {
    if emit.in_fast() || falls(emit, moved.0) {
        return Ok(false);
    }
    let closes = emit.func().cfg().loops().list.iter().any(|each| each.head == moved.0 && !each.body.get(plain.index()).copied().unwrap_or(true));
    if !closes {
        return Ok(false);
    }
    let (func, plan, frame) = (emit.func(), emit.plan(), emit.frame());
    let params = &func.blocks[moved.0.index()].params;
    let hoistable = moved.1.iter().zip(params).all(|(&arg, &param)| {
        plan.same_home(frame, arg, param)
            || (hoistable_move(func, plan, arg, param)
                && plan.hoist.free(plan.locations[param.index()], plain, condition))
    });
    if !hoistable {
        return Ok(false);
    }
    emit.edge(moved.0, moved.1)?;
    let target = emit.block_label(emit.func().jump_target(moved.0));
    emit.branch_cond(condition, !to_then, target);
    emit_goto(emit, plain, true);
    Ok(true)
}

pub(crate) fn emit_brif<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(
    emit: &mut E,
    condition: Value,
    then: (Block, &'a [Value]),
    otherwise: (Block, &'a [Value]),
) -> Result<(), String> {
    if let Some(known) = emit.plan().const_of(condition) {
        let (block, args) = if crate::fold::extend(known, emit.func().ty(condition), false) != 0 { then } else { otherwise };
        if emit.is_split_ret(block) {
            emit.edge(block, args)?;
            return emit.fast_ret(block);
        }
        if emit.exits_fast(block) {
            emit.prologue();
        }
        emit.edge(block, args)?;
        emit_goto(emit, block, true);
        return Ok(());
    }
    if emit.in_fast() {
        #[derive(Clone, Copy)]
        enum Dest { Fast, SplitRet, Slow }
        let kind = |b: Block| if emit.is_fast(b) { Dest::Fast } else if emit.is_split_ret(b) { Dest::SplitRet } else { Dest::Slow };
        match (kind(then.0), kind(otherwise.0)) {
            (Dest::Fast, Dest::Fast) => {}
            (Dest::Slow, Dest::Fast) => {
                let label = push_stub(emit, then.0, then.1, StubKind::Prologue);
                emit.branch_cond(condition, false, label);
                emit.edge(otherwise.0, otherwise.1)?;
                emit_goto(emit, otherwise.0, true);
                return Ok(());
            }
            (Dest::Fast, Dest::Slow) => {
                let label = push_stub(emit, otherwise.0, otherwise.1, StubKind::Prologue);
                emit.branch_cond(condition, true, label);
                emit.edge(then.0, then.1)?;
                emit_goto(emit, then.0, true);
                return Ok(());
            }
            (Dest::Slow, Dest::SplitRet) => {
                let label = push_stub(emit, then.0, then.1, StubKind::Prologue);
                emit.branch_cond(condition, false, label);
                emit.edge(otherwise.0, otherwise.1)?;
                return emit.fast_ret(otherwise.0);
            }
            (Dest::SplitRet, Dest::Slow) => {
                let label = push_stub(emit, otherwise.0, otherwise.1, StubKind::Prologue);
                emit.branch_cond(condition, true, label);
                emit.edge(then.0, then.1)?;
                return emit.fast_ret(then.0);
            }
            (Dest::Slow, Dest::Slow) => {
                let label = push_stub(emit, then.0, then.1, StubKind::Prologue);
                emit.branch_cond(condition, false, label);
                emit.prologue();
                emit.edge(otherwise.0, otherwise.1)?;
                emit_goto(emit, otherwise.0, true);
                return Ok(());
            }
            (Dest::SplitRet, Dest::Fast) => {
                let label = push_stub(emit, then.0, then.1, StubKind::FastRet);
                emit.branch_cond(condition, false, label);
                emit.edge(otherwise.0, otherwise.1)?;
                emit_goto(emit, otherwise.0, true);
                return Ok(());
            }
            (Dest::Fast, Dest::SplitRet) => {
                if !falls(emit, then.0) && !emit.plan().moves(emit.func(), emit.frame(), then.0, then.1) {
                    let target = emit.block_label(emit.func().jump_target(then.0));
                    emit.branch_cond(condition, false, target);
                    emit.edge(otherwise.0, otherwise.1)?;
                    return emit.fast_ret(otherwise.0);
                }
                let label = push_stub(emit, otherwise.0, otherwise.1, StubKind::FastRet);
                emit.branch_cond(condition, true, label);
                emit.edge(then.0, then.1)?;
                emit_goto(emit, then.0, true);
                return Ok(());
            }
            (Dest::SplitRet, Dest::SplitRet) => {
                let label = push_stub(emit, then.0, then.1, StubKind::FastRet);
                emit.branch_cond(condition, false, label);
                emit.edge(otherwise.0, otherwise.1)?;
                return emit.fast_ret(otherwise.0);
            }
        }
    }
    let mut then_moves = emit.plan().moves(emit.func(), emit.frame(), then.0, then.1);
    let mut else_moves = emit.plan().moves(emit.func(), emit.frame(), otherwise.0, otherwise.1);
    if then_moves && else_moves && hoist_common_moves(emit, condition, then, otherwise)? {
        then_moves = false;
        else_moves = false;
    }
    if set_aside(emit, condition, (then, then_moves), (otherwise, else_moves))? {
        return Ok(());
    }
    if then_moves && !else_moves && !emit.in_fast() && falls(emit, otherwise.0) && is_latch(emit.func(), otherwise.0) {
        let label = push_stub(emit, then.0, then.1, StubKind::Plain);
        emit.branch_cond(condition, false, label);
        emit.edge(otherwise.0, otherwise.1)?;
        emit_goto(emit, otherwise.0, true);
        return Ok(());
    }
    if then_moves && !else_moves && hoist_moves(emit, condition, (then, true), otherwise.0)? {
        return Ok(());
    }
    if else_moves && !then_moves && hoist_moves(emit, condition, (otherwise, false), then.0)? {
        return Ok(());
    }
    if !else_moves && (then_moves || falls(emit, then.0)) {
        let target = emit.block_label(emit.func().jump_target(otherwise.0));
        emit.branch_cond(condition, true, target);
        if then_moves {
            emit.edge(then.0, then.1)?;
        }
        emit_goto(emit, then.0, true);
        return Ok(());
    }
    if !then_moves {
        let target = emit.block_label(emit.func().jump_target(then.0));
        emit.branch_cond(condition, false, target);
        if else_moves {
            emit.edge(otherwise.0, otherwise.1)?;
        }
        emit_goto(emit, otherwise.0, true);
        return Ok(());
    }
    let other = emit.new_label();
    emit.branch_cond(condition, true, other);
    emit.edge(then.0, then.1)?;
    emit_goto(emit, then.0, false);
    emit.bind_label(other);
    emit.edge(otherwise.0, otherwise.1)?;
    emit_goto(emit, otherwise.0, true);
    Ok(())
}

pub(crate) fn emit_call<'a, R: Copy + PartialEq, L: Copy, E: Backend<'a, R, L>>(
    emit: &mut E,
    inst: crate::ir::Inst,
    (sig, callee): (SigRef, Value),
    args: &[Value],
    results: &[Value],
) -> Result<(), String> {
    let parked: Vec<_> = emit.plan().parked_at(inst).iter().filter(|(value, _, _)| !emit.plan().elided[value.index()]).copied().collect();
    for &(value, reg, _) in &parked {
        emit.park_spill(value, reg);
        emit.plan_mut().locations[value.index()] = Location::Slot;
    }
    let abi = emit.arguments(sig, callee, args, false)?;
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
    emit.arguments(sig, callee, args, true)?;
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

/// Whether `val` keeps its register or shared slot across instructions `start + 1..end` of block `b_idx` that `skip` does not pass over, so a fused use at `end` can still read it; false when the span exceeds `max_span`.
pub(crate) fn preserved<R: Copy + PartialEq + 'static>(func: &Function, (plan, regs): (&Plan<R>, &Regs<R>), (b_idx, start, end, max_span): (usize, usize, usize, usize), skip: impl Fn(usize) -> bool, val: Value) -> bool {
    if plan.elided[val.index()] {
        return false;
    }
    if start + 1 >= end || plan.location(val).is_static() {
        return true;
    }
    if end - start > max_span {
        return false;
    }
    let (reg, slot_id) = match plan.location(val) {
        Location::Reg(reg) => (Some(reg), None),
        Location::Slot => {
            let Some(id) = plan.shared.get(val.index()).copied().flatten() else { return true };
            (None, Some(id))
        }
        _ => return false,
    };
    let clobbers = |r: Value| {
        reg.is_some_and(|reg| plan.reg_of(r) == Some(reg))
            || slot_id.is_some_and(|id| matches!(plan.location(r), Location::Slot) && plan.shared.get(r.index()).copied().flatten() == Some(id))
    };
    for pos in (start + 1..end).filter(|&pos| !skip(pos)) {
        let data = &func.insts[func.blocks[b_idx].insts[pos].index()];
        if data.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]) {
            continue;
        }
        if (reg.is_some() && matches!(data.op, Op::Call { .. } | Op::TryCall { .. }))
            || reg.is_some_and(|reg| regs.scratches(func, data, reg))
            || data.result.and_then(|r| plan.pairs[r.index()]).is_some_and(clobbers)
            || crate::live::results(data).any(clobbers)
        {
            return false;
        }
    }
    true
}

/// Finds a single-use, non-elided `Band(x, y)` in block `b_idx` compared against zero via `is_zero`.
pub(crate) fn zero_cmp_band<R>(
    func: &Function,
    plan: &Plan<R>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    b_idx: usize,
    (lhs, rhs): (Value, Value),
    is_zero: impl Fn(Value) -> bool,
) -> Option<(Value, usize, Value, Value)> {
    let masked = is_zero(rhs).then_some(lhs).or_else(|| is_zero(lhs).then_some(rhs))?;
    if uses[masked.index()] != 1 || plan.elided[masked.index()] {
        return None;
    }
    let (mb, mpos, minst) = pos_of.get(masked.index()).copied().flatten()?;
    if mb != b_idx {
        return None;
    }
    let Op::Binary(Binary::Band, x, y) = func.insts[minst.index()].op else {
        return None;
    };
    Some((masked, mpos, x, y))
}
