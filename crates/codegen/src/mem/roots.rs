use super::{MemRoot, escape};
use crate::cfg::Cfg;
use crate::ir::{Binary, Block, BlockData, Convert, Function, Inst, Op, StackSlot, Type, Value};
use crate::reloc::RuntimeSymbol;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TrackRoot {
    Unset,
    Set(MemRoot),
    Conflict,
}

impl TrackRoot {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unset, x) | (x, Self::Unset) => x,
            (Self::Set(a), Self::Set(b)) if a == b => Self::Set(a),
            (Self::Set(MemRoot::UniqueHeap(_)), Self::Set(MemRoot::UniqueHeap(_))) => {
                Self::Set(MemRoot::UniqueHeap(None))
            }
            _ => Self::Conflict,
        }
    }

    fn across_edge(self, entry: &[Value]) -> Self {
        match self {
            Self::Set(MemRoot::UniqueHeap(Some(owner))) if !entry.contains(&owner) => {
                Self::Set(MemRoot::UniqueHeap(None))
            }
            other => other,
        }
    }
}

pub(super) fn slot_addresses(func: &Function) -> Vec<Option<(StackSlot, i32)>> {
    func.slot_offsets(|_| None)
        .0
        .into_iter()
        .map(|entry| entry.and_then(|(slot, off)| i32::try_from(off).ok().map(|o| (slot, o))))
        .collect()
}

fn step_track(op: &Op, ty: Type, tracked: &[TrackRoot]) -> TrackRoot {
    if ty != Type::I64 {
        return TrackRoot::Conflict;
    }
    match *op {
        Op::StackAddr(slot, _) => TrackRoot::Set(MemRoot::Slot(slot)),
        Op::Symbol(sym) | Op::SlotAddr(sym) => TrackRoot::Set(MemRoot::Global(sym)),
        Op::Convert(Convert::Bitcast, x) => tracked[x.index()],
        Op::Binary(Binary::Iadd, a, b) => match (tracked[a.index()], tracked[b.index()]) {
            (TrackRoot::Unset, _) | (_, TrackRoot::Unset) => TrackRoot::Unset,
            (TrackRoot::Set(r), TrackRoot::Conflict) | (TrackRoot::Conflict, TrackRoot::Set(r)) => {
                TrackRoot::Set(r)
            }
            _ => TrackRoot::Conflict,
        },
        Op::Binary(Binary::Isub, a, b) => match (tracked[a.index()], tracked[b.index()]) {
            (TrackRoot::Unset, _) | (_, TrackRoot::Unset) => TrackRoot::Unset,
            (TrackRoot::Set(r), TrackRoot::Conflict) => TrackRoot::Set(r),
            _ => TrackRoot::Conflict,
        },
        Op::Select(_, a, b) => tracked[a.index()].join(tracked[b.index()]),
        _ => TrackRoot::Conflict,
    }
}

fn join_track_args(
    tracked: &mut [TrackRoot],
    (blocks, entry): (&[BlockData], &[Value]),
    target: Block,
    args: &[Value],
    mut dirty: Option<(&mut [bool], Option<&mut [bool]>)>,
) -> bool {
    let mut changed = false;
    for (&param, &arg) in blocks[target.index()].params.iter().zip(args) {
        if arg != param {
            let next = tracked[param.index()].join(tracked[arg.index()].across_edge(entry));
            if next != tracked[param.index()] {
                tracked[param.index()] = next;
                if let Some((cur_dirty, next_dirty)) = dirty.as_mut() {
                    cur_dirty[param.index()] = true;
                    if let Some(nd) = next_dirty.as_deref_mut() {
                        nd[param.index()] = true;
                    }
                }
                changed = true;
            }
        }
    }
    changed
}

fn seed_entry_roots(func: &Function, cfg: &Cfg, tracked: &mut [TrackRoot]) {
    let Some(entry) = cfg.entry else { return };
    for (i, (&param, abi)) in func.blocks[entry.index()]
        .params
        .iter()
        .zip(&func.signature.params)
        .enumerate()
    {
        let root = u16::try_from(i).ok().and_then(|idx| {
            if abi.noalias {
                Some(MemRoot::NoAliasParam(idx))
            } else if abi.readonly {
                Some(MemRoot::ReadonlyParam(idx))
            } else {
                None
            }
        });
        tracked[param.index()] = root.map_or(TrackRoot::Conflict, TrackRoot::Set);
    }
}

fn is_derived_i64(op: &Op, ty: Type) -> bool {
    ty == Type::I64
        && matches!(
            op,
            Op::Convert(Convert::Bitcast, _)
                | Op::Binary(Binary::Iadd | Binary::Isub, ..)
                | Op::Select(..)
        )
}

fn track_def(
    func: &Function,
    inst: Inst,
    tracked: &mut [TrackRoot],
    affine: &[(Value, i64)],
) -> bool {
    let data = &func.insts[inst.index()];
    let Some(res) = data.result else { return false };
    let ty = func.ty(res);
    let next = if ty == Type::I64 && func.is_unique_ptr(res) {
        let owner = match data.op {
            Op::Load(addr, _) => affine.get(addr.index()).map(|&(root, _)| root),
            _ => None,
        };
        TrackRoot::Set(MemRoot::UniqueHeap(owner))
    } else {
        step_track(&data.op, ty, tracked)
    };
    let changed = next != tracked[res.index()];
    tracked[res.index()] = next;
    changed
}

fn track_edges(
    func: &Function,
    entry: &[Value],
    inst: Inst,
    tracked: &mut [TrackRoot],
    dirty: Option<(&mut [bool], &mut [bool])>,
) -> bool {
    let blocks = (func.blocks.as_slice(), entry);
    match &func.insts[inst.index()].op {
        Op::Jump(t, args) => {
            join_track_args(tracked, blocks, *t, args, dirty.map(|(c, n)| (c, Some(n))))
        }
        Op::Brif(_, t, ta, f, fa) => {
            let mut changed = false;
            if let Some((c, n)) = dirty {
                changed |= join_track_args(tracked, blocks, *t, ta, Some((&mut *c, Some(&mut *n))));
                changed |= join_track_args(tracked, blocks, *f, fa, Some((&mut *c, Some(&mut *n))));
            } else {
                changed |= join_track_args(tracked, blocks, *t, ta, None);
                changed |= join_track_args(tracked, blocks, *f, fa, None);
            }
            changed
        }
        _ => false,
    }
}

pub(super) type TrackedRoots = (
    Vec<TrackRoot>,
    bool,
    bool,
    Vec<Option<u32>>,
    Vec<Option<RuntimeSymbol>>,
);

struct RootScan<'a> {
    tracked: Vec<TrackRoot>,
    symbol_defs: Vec<Option<u32>>,
    frame_runtime: Vec<Option<RuntimeSymbol>>,
    zeros: Vec<bool>,
    consts: Option<Vec<Option<i64>>>,
    mallocs: Vec<Value>,
    frees: Vec<Value>,
    called: Vec<(Value, bool, Block, usize)>,
    has_caller_loc: bool,
    cl_escaped: bool,
    has_mem: bool,
    has_span_op: bool,
    affine: &'a [(Value, i64)],
}

impl RootScan<'_> {
    fn scan_call(
        &mut self,
        func: &Function,
        b: Block,
        pos: usize,
        op: &Op,
        callee: Value,
        results: &[Value],
    ) {
        for &r in results {
            if func.is_unique_ptr(r) {
                self.called.push((r, false, b, pos));
            }
        }
        let malloc = !self.mallocs.is_empty() && self.mallocs.contains(&callee);
        if (func.is_fresh_alloc(callee) || malloc)
            && let Some(&res) = results.first()
        {
            self.tracked[res.index()] = TrackRoot::Set(MemRoot::UniqueHeap(None));
            if malloc {
                self.called.push((res, true, b, pos));
            }
        }
        if !self.zeros.is_empty() && func.is_thread_cell(callee) {
            if let Some(res) = func.caller_loc_result(op, |v| self.zeros[v.index()]) {
                self.tracked[res.index()] = TrackRoot::Set(MemRoot::CallerLoc);
                self.has_caller_loc = true;
            } else {
                let c = self.consts.get_or_insert_with(|| func.const_values());
                if let Some((res, slot)) = func.thread_cell_slot(op, |v| c[v.index()]) {
                    self.tracked[res.index()] = TrackRoot::Set(MemRoot::ThreadCell(slot));
                }
            }
        }
    }

    fn scan_inst(&mut self, func: &Function, b: Block, pos: usize, inst: Inst) {
        let n = self.tracked.len();
        let data = &func.insts[inst.index()];
        match (&data.op, data.result) {
            (&Op::Runtime(rt @ RuntimeSymbol::Malloc), Some(res)) => {
                if self.frame_runtime.is_empty() {
                    self.frame_runtime.resize(n, None);
                }
                self.frame_runtime[res.index()] = Some(rt);
                self.mallocs.push(res);
            }
            (&Op::Runtime(rt @ RuntimeSymbol::Free), Some(res)) => {
                if self.frame_runtime.is_empty() {
                    self.frame_runtime.resize(n, None);
                }
                self.frame_runtime[res.index()] = Some(rt);
                self.frees.push(res);
            }
            (
                Op::Call {
                    callee, results, ..
                },
                _,
            ) => {
                self.scan_call(func, b, pos, &data.op, *callee, results);
            }
            (&Op::StackAddr(slot, _), Some(res)) if func.ty(res) == Type::I64 => {
                self.tracked[res.index()] = TrackRoot::Set(MemRoot::Slot(slot));
            }
            (&(Op::Symbol(sym) | Op::SlotAddr(sym)), Some(res)) => {
                if self.symbol_defs.is_empty() {
                    self.symbol_defs.resize(n, None);
                }
                self.symbol_defs[res.index()] = Some(sym);
                if func.ty(res) == Type::I64 {
                    self.tracked[res.index()] = TrackRoot::Set(MemRoot::Global(sym));
                }
            }
            (op, Some(res)) if func.ty(res) == Type::I64 => {
                if matches!(
                    *op,
                    Op::Load(..)
                        | Op::VolatileLoad(..)
                        | Op::AtomicLoad(..)
                        | Op::AtomicRmw(..)
                        | Op::AtomicCas(..)
                ) {
                    self.has_mem = true;
                } else if matches!(*op, Op::Select(..))
                    || (matches!(*op, Op::Binary(Binary::Iadd | Binary::Isub, ..))
                        && self.affine[res.index()].0 == res)
                {
                    self.has_span_op = true;
                }
                if func.is_unique_ptr(res) {
                    let owner = match *op {
                        Op::Load(addr, _) => self.affine.get(addr.index()).map(|&(r, _)| r),
                        _ => None,
                    };
                    self.tracked[res.index()] = TrackRoot::Set(MemRoot::UniqueHeap(owner));
                } else if is_derived_i64(op, Type::I64) {
                    self.tracked[res.index()] = step_track(op, Type::I64, &self.tracked);
                }
            }
            (
                Op::Load(..)
                | Op::Store(..)
                | Op::VolatileLoad(..)
                | Op::VolatileStore(..)
                | Op::Move(..)
                | Op::Copy(..)
                | Op::AtomicLoad(..)
                | Op::AtomicStore(..)
                | Op::AtomicRmw(..)
                | Op::AtomicCas(..),
                _,
            ) => {
                self.has_mem = true;
            }
            _ => {}
        }
        if self.has_caller_loc
            && !self.cl_escaped
            && escape::inst_escapes_caller_loc(&data.op, &self.tracked)
        {
            self.cl_escaped = true;
        }
    }
}

fn join_edge_back(
    tracked: &mut [TrackRoot],
    blocks: (&[BlockData], &[Value]),
    target: Block,
    args: &[Value],
    is_back: bool,
    dirty: &mut Vec<bool>,
) -> bool {
    if args.is_empty() {
        return false;
    }
    if is_back {
        if dirty.is_empty() {
            dirty.resize(tracked.len(), false);
        }
        join_track_args(
            tracked,
            blocks,
            target,
            args,
            Some((dirty.as_mut_slice(), None)),
        )
    } else {
        join_track_args(tracked, blocks, target, args, None);
        false
    }
}

fn join_block_terminator(
    func: &Function,
    entry: &[Value],
    tracked: &mut [TrackRoot],
    rank: Option<&[usize]>,
    br: usize,
    insts: &[Inst],
    dirty: &mut Vec<bool>,
) -> bool {
    let (Some(rank), Some(&last)) = (rank, insts.last()) else {
        return false;
    };
    let blocks = (func.blocks.as_slice(), entry);
    match &func.insts[last.index()].op {
        Op::Jump(t, args) => {
            join_edge_back(tracked, blocks, *t, args, rank[t.index()] <= br, dirty)
        }
        Op::Brif(_, t, ta, f, fa) if !ta.is_empty() || !fa.is_empty() => {
            join_edge_back(tracked, blocks, *t, ta, rank[t.index()] <= br, dirty)
                | join_edge_back(tracked, blocks, *f, fa, rank[f.index()] <= br, dirty)
        }
        _ => false,
    }
}

fn step_reads_dirty(func: &Function, inst: Inst, edges: bool, dirty: &[bool]) -> bool {
    let op = &func.insts[inst.index()].op;
    let mut any = false;
    if edges {
        op.for_edges(|_, args| any |= args.iter().any(|v| dirty[v.index()]));
    } else {
        op.for_operands(|v| any |= dirty[v.index()]);
    }
    any
}

fn run_root_fixpoint(
    func: &Function,
    cfg: &Cfg,
    entry: &[Value],
    affine: &[(Value, i64)],
    tracked: &mut [TrackRoot],
    dirty: &mut [bool],
) -> bool {
    if dirty.is_empty() {
        return false;
    }
    let mut steps = Vec::new();
    for &b in &cfg.rpo {
        let insts = &func.blocks[b.index()].insts;
        for &inst in insts {
            let data = &func.insts[inst.index()];
            if data.result.is_some_and(|res| {
                !func.is_unique_ptr(res) && is_derived_i64(&data.op, func.ty(res))
            }) {
                steps.push((inst, false));
            }
        }
        if let Some(&last) = insts.last()
            && match &func.insts[last.index()].op {
                Op::Jump(_, args) => !args.is_empty(),
                Op::Brif(_, _, ta, _, fa) => !ta.is_empty() || !fa.is_empty(),
                _ => false,
            }
        {
            steps.push((last, true));
        }
    }
    let mut next_dirty = vec![false; tracked.len()];
    let mut changed = true;
    let mut rounds = 1;
    while changed && rounds < 64 {
        changed = false;
        for &(inst, edges) in &steps {
            if !step_reads_dirty(func, inst, edges, dirty) {
                continue;
            }
            if edges {
                changed |= track_edges(
                    func,
                    entry,
                    inst,
                    tracked,
                    Some((&mut *dirty, &mut next_dirty)),
                );
            } else if let Some(res) = func.insts[inst.index()].result
                && track_def(func, inst, tracked, affine)
            {
                dirty[res.index()] = true;
                next_dirty[res.index()] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        dirty.copy_from_slice(&next_dirty);
        next_dirty.fill(false);
        rounds += 1;
    }
    changed
}

pub(super) fn compute_tracked_roots(
    func: &Function,
    cfg: &Cfg,
    affine: &[(Value, i64)],
) -> TrackedRoots {
    let n = func.types.len();
    let mut tracked = vec![TrackRoot::Conflict; n];
    let mut any_params = false;
    for &b in &cfg.rpo {
        if Some(b) != cfg.entry {
            for &p in &func.blocks[b.index()].params {
                tracked[p.index()] = TrackRoot::Unset;
                any_params = true;
            }
        }
    }
    seed_entry_roots(func, cfg, &mut tracked);
    let entry = cfg.entry.map_or(&[][..], |block| {
        func.blocks[block.index()].params.as_slice()
    });
    let rank = (cfg.rpo.len() > 1).then(|| Cfg::ranks(&cfg.rpo, func.blocks.len()));
    let mut scan = RootScan {
        tracked,
        symbol_defs: Vec::new(),
        frame_runtime: Vec::new(),
        zeros: func.thread_cell_zeros(),
        consts: None,
        mallocs: Vec::new(),
        frees: Vec::new(),
        called: Vec::new(),
        has_caller_loc: false,
        cl_escaped: false,
        has_mem: false,
        has_span_op: false,
        affine,
    };
    let mut dirty = Vec::new();
    let mut need_fixpoint = false;
    for &b in &cfg.rpo {
        let br = rank.as_ref().map_or(0, |r| r[b.index()]);
        let insts = &func.blocks[b.index()].insts;
        for (pos, &inst) in insts.iter().enumerate() {
            scan.scan_inst(func, b, pos, inst);
        }
        need_fixpoint |= join_block_terminator(
            func,
            entry,
            &mut scan.tracked,
            rank.as_deref(),
            br,
            insts,
            &mut dirty,
        );
    }
    if !scan.called.is_empty()
        && escape::finish_fresh_owners(
            func,
            cfg,
            entry,
            affine,
            (&scan.frees, &scan.called),
            &mut scan.tracked,
        )
    {
        dirty.clear();
        dirty.resize(n, true);
        need_fixpoint = true;
    }
    let still_changed =
        need_fixpoint && run_root_fixpoint(func, cfg, entry, affine, &mut scan.tracked, &mut dirty);
    if still_changed {
        scan.tracked.fill(TrackRoot::Conflict);
        scan.has_caller_loc = false;
    } else {
        if any_params {
            scan.tracked
                .iter_mut()
                .filter(|item| **item == TrackRoot::Unset)
                .for_each(|item| *item = TrackRoot::Conflict);
        }
        if scan.has_caller_loc
            && if need_fixpoint {
                escape::caller_loc_escaped(func, cfg, &scan.tracked)
            } else {
                scan.cl_escaped
            }
        {
            scan.tracked
                .iter_mut()
                .filter(|item| **item == TrackRoot::Set(MemRoot::CallerLoc))
                .for_each(|item| *item = TrackRoot::Conflict);
            scan.has_caller_loc = false;
        }
    }
    (
        scan.tracked,
        scan.has_caller_loc,
        scan.has_mem && scan.has_span_op,
        scan.symbol_defs,
        scan.frame_runtime,
    )
}

fn step_affine(
    op: &Op,
    out: &[(Value, i64)],
    consts: &[Option<i64>],
    types: &[Type],
) -> Option<(Value, i64)> {
    match *op {
        Op::Binary(Binary::Iadd, a, b) => {
            let (base, k) = consts[b.index()]
                .map(|k| (a, k))
                .or_else(|| consts[a.index()].map(|k| (b, k)))?;
            let (root, off) = out[base.index()];
            off.checked_add(k).map(|total| (root, total))
        }
        Op::Binary(Binary::Isub, a, b) => {
            let k = consts[b.index()]?.checked_neg()?;
            let (root, off) = out[a.index()];
            off.checked_add(k).map(|total| (root, total))
        }
        Op::Convert(Convert::Bitcast, x) => {
            (types[x.index()] == Type::I64).then_some(out[x.index()])
        }
        Op::Select(_, a, b) => (out[a.index()] == out[b.index()]).then_some(out[a.index()]),
        _ => None,
    }
}

pub(super) fn compute_affine_in(func: &Function, rpo: Option<&[Block]>) -> Vec<(Value, i64)> {
    let count = func.types.len();
    let (mut consts, mut out, mut steps, mut need_more) = (
        vec![None; count],
        (0..count as u32).map(|i| (Value(i), 0)).collect::<Vec<_>>(),
        Vec::new(),
        rpo.is_none(),
    );
    let mut visit = |inst: Inst| {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { return };
        if let Op::Const(imm) = data.op {
            consts[res.index()] = Some(imm);
        } else if is_derived_i64(&data.op, func.ty(res)) {
            if let Some(cand) = step_affine(&data.op, &out, &consts, &func.types) {
                out[res.index()] = cand;
            } else {
                need_more = true;
            }
            steps.push((data, res));
        }
    };
    match rpo {
        Some(blocks) => blocks.iter().for_each(|b| {
            func.blocks[b.index()]
                .insts
                .iter()
                .copied()
                .for_each(&mut visit)
        }),
        None => func
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter().copied())
            .for_each(&mut visit),
    }
    if need_more && !steps.is_empty() {
        for _ in 0..7 {
            let mut changed = false;
            for &(data, res) in &steps {
                if let Some(cand) = step_affine(&data.op, &out, &consts, &func.types)
                    && cand != out[res.index()]
                {
                    (out[res.index()], changed) = (cand, true);
                }
            }
            if !changed {
                break;
            }
        }
    }
    out
}

pub(crate) fn compute_affine(func: &Function) -> Vec<(Value, i64)> {
    compute_affine_in(func, None)
}
