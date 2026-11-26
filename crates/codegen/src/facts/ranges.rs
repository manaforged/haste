use crate::cfg::dominates;
use crate::ir::{Binary, Block, Function, Inst, InstData, Op, Type, Value};

use super::rel::{RelKind, rel_of};
use super::transfer;
use super::{Fact, unsigned, unsigned_order};

/// The range each value's definition proves, computed once in reverse postorder so that operands come before the values that read them.
pub(crate) struct Ranges<'a> {
    defs: &'a [Option<Inst>],
    pub(super) table: Vec<(u64, u64)>,
    pub(super) raised: Vec<Value>,
}

struct CfgInfo<'a> {
    consts: &'a [Option<i64>],
    incoming: &'a [Vec<(Block, &'a [Value])>],
    complete: &'a [bool],
    edge_args: &'a [bool],
    single: &'a [Option<Block>],
    idom: &'a [Option<Block>],
}

pub(super) fn collect_incoming<'a>(
    func: &'a Function,
) -> (Vec<Vec<(Block, &'a [Value])>>, Vec<bool>, Vec<bool>) {
    if !func.blocks.iter().any(|b| !b.params.is_empty()) {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut incoming: Vec<Vec<(Block, &'a [Value])>> = vec![Vec::new(); func.blocks.len()];
    let mut complete = vec![true; func.blocks.len()];
    let mut edge_args = vec![false; func.types.len()];
    for pred_idx in 0..func.blocks.len() {
        let pred = Block(pred_idx as u32);
        let mut edge = |target: Block, args: &'a [Value]| {
            if !func.blocks[target.index()].params.is_empty() {
                incoming[target.index()].push((pred, args));
                for &a in args {
                    edge_args[a.index()] = true;
                }
            }
        };
        match func.last(pred) {
            Some(Op::Jump(target, args)) => edge(*target, args.as_slice()),
            Some(Op::Brif(_, then, then_args, otherwise, else_args)) => {
                edge(*then, then_args.as_slice());
                edge(*otherwise, else_args.as_slice());
            }
            _ => {
                for target in func.successors(pred) {
                    complete[target.index()] = false;
                }
            }
        }
    }
    (incoming, complete, edge_args)
}

fn bump_epoch(epoch: &mut u32, ft: &mut [u32]) {
    *epoch = epoch.wrapping_add(1);
    if *epoch == 0 {
        ft.fill(0);
        *epoch = 1;
    }
}

fn deref_narrowed(info: &CfgInfo<'_>, addr: Value, pos: usize, prev_res: Option<Value>) -> bool {
    info.edge_args.get(addr.index()).copied().unwrap_or(false)
        || (pos > 0 && prev_res != Some(addr))
}

fn combine_bound(upper: bool, a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(if upper { x.min(y) } else { x.max(y) }),
        _ => a.or(b),
    }
}

impl<'a> Ranges<'a> {
    /// The ranges of `func`'s values over the blocks of `order`, with the constants `consts` an earlier fold found.
    pub(crate) fn new(
        func: &Function,
        parts: (&'a [Option<Inst>], &[Option<i64>]),
        blocks: (&[Block], &[Option<Block>], &[Option<Block>]),
    ) -> Self {
        let (incoming, complete, edge_args) = collect_incoming(func);
        Self::over(func, parts, blocks, (&incoming, &complete, &edge_args))
    }

    pub(super) fn over(
        func: &Function,
        (defs, consts): (&'a [Option<Inst>], &[Option<i64>]),
        (order, single, idom): (&[Block], &[Option<Block>], &[Option<Block>]),
        (incoming, complete, edge_args): (&[Vec<(Block, &[Value])>], &[bool], &[bool]),
    ) -> Self {
        let mut table: Vec<(u64, u64)> =
            func.types.iter().map(|ty| (0, ty.unsigned_max())).collect();
        if let Some(entry) = func.entry {
            for (&param, abi) in func.blocks[entry.index()]
                .params
                .iter()
                .zip(&func.signature.params)
            {
                if abi.deref_bytes > 0 && func.ty(param) == Type::I64 {
                    table[param.index()] = (1, u64::MAX - u64::from(abi.deref_bytes - 1));
                }
            }
        }
        let mut ranges = Self {
            defs,
            table,
            raised: Vec::new(),
        };
        let info = CfgInfo {
            consts,
            incoming,
            complete,
            edge_args,
            single,
            idom,
        };
        let mut falls_through = vec![0_u32; func.types.len()];
        let mut epoch = 0_u32;
        if ranges.propagate(func, &info, order, &mut falls_through, &mut epoch) {
            ranges.propagate(func, &info, order, &mut falls_through, &mut epoch);
        }
        ranges
    }

    fn propagate(
        &mut self,
        func: &Function,
        info: &CfgInfo<'_>,
        order: &[Block],
        falls_through: &mut [u32],
        epoch: &mut u32,
    ) -> bool {
        let mut narrowed = false;
        for &block in order {
            narrowed |= self.refine_params(func, info, block);
            bump_epoch(epoch, falls_through);
            for &p in &func.blocks[block.index()].params {
                if func.ty(p).is_small_int() {
                    falls_through[p.index()] = *epoch;
                }
            }
            let mut prev_res = None;
            for (pos, inst) in func.blocks[block.index()].insts.iter().enumerate() {
                let data = &func.insts[inst.index()];
                narrowed |=
                    self.propagate_inst(func, info, data, falls_through, epoch, pos, &mut prev_res);
            }
        }
        narrowed
    }

    fn note_page_zero_deref(
        &mut self,
        func: &Function,
        info: &CfgInfo<'_>,
        (addr, off): (Value, i32),
        (ft, epoch): (&[u32], u32),
        (pos, prev_res): (usize, Option<Value>),
    ) -> bool {
        (0..4096).contains(&off)
            && transfer::note_deref_addr(func, addr, ft, epoch, &mut self.table)
            && deref_narrowed(info, addr, pos, prev_res)
    }

    fn record_known_result(
        &mut self,
        func: &Function,
        info: &CfgInfo<'_>,
        data: &InstData,
        ft: &mut [u32],
        epoch: u32,
    ) {
        if let Some(r) = data.result
            && func.ty(r).is_small_int()
        {
            ft[r.index()] = epoch;
            let ty = func.ty(r);
            self.table[r.index()] = info.consts.get(r.index()).copied().flatten().map_or_else(
                || transfer::known_range(&data.op, ty).unwrap_or((0, ty.unsigned_max())),
                |imm| {
                    let u = unsigned(imm, ty);
                    (u, u)
                },
            );
        }
    }

    fn record_assume(&mut self, func: &Function, value: Value, ft: &[u32], epoch: u32) {
        let (lo, hi) = &mut self.table[value.index()];
        if *lo == 0 && *hi != 0 && func.ty(value) != Type::I8 && ft[value.index()] == epoch {
            *lo = 1;
            if !self.raised.contains(&value) {
                self.raised.push(value);
            }
        }
    }

    fn propagate_inst(
        &mut self,
        func: &Function,
        info: &CfgInfo<'_>,
        data: &InstData,
        ft: &mut [u32],
        epoch: &mut u32,
        pos: usize,
        prev_res: &mut Option<Value>,
    ) -> bool {
        match data.op {
            Op::Const(imm) => {
                *prev_res = data.result;
                if let Some(r) = data.result
                    && func.ty(r).is_small_int()
                {
                    ft[r.index()] = *epoch;
                    let u = unsigned(imm, func.ty(r));
                    self.table[r.index()] = (u, u);
                }
                false
            }
            Op::StackStore(..) => {
                *prev_res = None;
                bump_epoch(epoch, ft);
                false
            }
            Op::Store(_, addr, off) => {
                let narrowed = self.note_page_zero_deref(
                    func,
                    info,
                    (addr, off),
                    (ft, *epoch),
                    (pos, *prev_res),
                );
                *prev_res = None;
                bump_epoch(epoch, ft);
                narrowed
            }
            Op::StackLoad(..) | Op::Load(..) => {
                let narrowed = match data.op {
                    Op::Load(addr, off) => self.note_page_zero_deref(
                        func,
                        info,
                        (addr, off),
                        (ft, *epoch),
                        (pos, *prev_res),
                    ),
                    _ => false,
                };
                *prev_res = data.result;
                bump_epoch(epoch, ft);
                if let Some(r) = data.result
                    && func.ty(r).is_small_int()
                {
                    ft[r.index()] = *epoch;
                    if let Some(imm) = info.consts.get(r.index()).copied().flatten() {
                        let u = unsigned(imm, func.ty(r));
                        self.table[r.index()] = (u, u);
                    }
                }
                narrowed
            }
            Op::Icmp(..)
            | Op::Fcmp(..)
            | Op::Fits(..)
            | Op::StackAddr(..)
            | Op::Symbol(_)
            | Op::SlotAddr(_) => {
                *prev_res = data.result;
                self.record_known_result(func, info, data, ft, *epoch);
                false
            }
            Op::Jump(..) | Op::Brif(..) | Op::Return(..) | Op::BrTable(..) | Op::Trap => {
                *prev_res = None;
                false
            }
            Op::Assume(value) => {
                *prev_res = None;
                self.record_assume(func, value, ft, *epoch);
                false
            }
            Op::Call { ref results, .. } => {
                *prev_res = None;
                bump_epoch(epoch, ft);
                for &r in results {
                    if func.ty(r).is_small_int() {
                        ft[r.index()] = *epoch;
                    }
                }
                false
            }
            _ => self.propagate_general(func, info, data, ft, epoch, pos, prev_res),
        }
    }

    fn propagate_general(
        &mut self,
        func: &Function,
        info: &CfgInfo<'_>,
        data: &InstData,
        ft: &mut [u32],
        epoch: &mut u32,
        pos: usize,
        prev_res: &mut Option<Value>,
    ) -> bool {
        let narrowed = transfer::note_deref(func, &data.op, ft, *epoch, &mut self.table)
            .is_some_and(|addr| deref_narrowed(info, addr, pos, *prev_res));
        *prev_res = data.result;
        if !data.op.is_speculable() {
            bump_epoch(epoch, ft);
        }
        let Some(result) = data.result.filter(|result| func.ty(*result).is_small_int()) else {
            return narrowed;
        };
        ft[result.index()] = *epoch;
        let ty = func.ty(result);
        if let Some(imm) = info.consts.get(result.index()).copied().flatten() {
            let u = unsigned(imm, ty);
            self.table[result.index()] = (u, u);
        } else if let Some(pair) = transfer::fast_range(func, self, &data.op, ty) {
            self.table[result.index()] = pair;
        }
        narrowed
    }

    fn refine_params(&mut self, func: &Function, info: &CfgInfo<'_>, block: Block) -> bool {
        if info.incoming.is_empty()
            || !info.complete[block.index()]
            || info.incoming[block.index()].is_empty()
        {
            return false;
        }
        let mut narrowed = false;
        for (k, &p) in func.blocks[block.index()].params.iter().enumerate() {
            if func.ty(p).is_small_int() {
                narrowed |= self.refine_param_lo(func, info, (block, p, k));
                narrowed |= self.refine_param_hi(func, info, (block, p, k));
            }
        }
        narrowed
    }

    fn refine_param_lo(
        &mut self,
        func: &Function,
        info: &CfgInfo<'_>,
        (block, p, k): (Block, Value, usize),
    ) -> bool {
        let ty = func.ty(p);
        let (mut min_lo, mut has_outside, mut valid) = (u64::MAX, false, true);
        for &(pred, args) in &info.incoming[block.index()] {
            let Some(&arg) = args.get(k) else {
                valid = false;
                break;
            };
            if let Some(step) = self.add_step(func, info.consts, arg, p, ty) {
                let Some(max_p) = ty.unsigned_max().checked_sub(step) else {
                    valid = false;
                    break;
                };
                if !self
                    .edge_bound(func, info, (block, pred), (p, true))
                    .is_some_and(|ub| ub <= max_p)
                {
                    valid = false;
                    break;
                }
            } else {
                has_outside = true;
                min_lo = min_lo.min(self.of(arg).lo);
            }
        }
        if valid && has_outside && min_lo > self.table[p.index()].0 && min_lo <= ty.unsigned_max() {
            self.table[p.index()].0 = min_lo;
            return true;
        }
        false
    }

    fn refine_param_hi(
        &mut self,
        func: &Function,
        info: &CfgInfo<'_>,
        (block, p, k): (Block, Value, usize),
    ) -> bool {
        let (mut max_hi, mut has_bound) = (0_u64, false);
        for &(pred, args) in &info.incoming[block.index()] {
            let Some(&arg) = args.get(k) else {
                return false;
            };
            let Some(edge_hi) = self.edge_arm_hi(func, info, (block, pred), (p, arg, 0)) else {
                return false;
            };
            if let Some(hi) = edge_hi {
                max_hi = max_hi.max(hi);
                has_bound = true;
            }
        }
        if has_bound && max_hi < self.table[p.index()].1 {
            self.table[p.index()].1 = max_hi;
            return true;
        }
        false
    }

    fn edge_arm_hi(
        &self,
        func: &Function,
        info: &CfgInfo<'_>,
        edge: (Block, Block),
        (p, arg, depth): (Value, Value, u8),
    ) -> Option<Option<u64>> {
        if arg == p {
            return Some(None);
        }
        let ty = func.ty(p);
        if let Some(ub) = self.edge_bound(func, info, edge, (arg, true)) {
            return Some(Some(ub));
        }
        if let Some(step) = self.add_step(func, info.consts, arg, p, ty)
            && let Some(ub) = self.edge_bound(func, info, edge, (p, true))
            && let Some(next_hi) = ub.checked_add(step).filter(|&hi| hi <= ty.unsigned_max())
        {
            return Some(Some(next_hi));
        }
        match self.op(func, arg) {
            Some(&Op::Binary(Binary::Ushr, base, _)) if base == p => return Some(None),
            Some(&Op::Binary(Binary::Band, a, b)) if a == p || b == p => return Some(None),
            Some(&Op::Binary(Binary::Isub, base, delta)) if base == p => {
                let sub_shr = matches!(
                    self.op(func, delta),
                    Some(&Op::Binary(Binary::Ushr, inner, _)) if inner == p
                );
                if sub_shr
                    || self
                        .const_val(func, info.consts, delta, ty)
                        .is_some_and(|k| {
                            self.edge_bound(func, info, edge, (p, false))
                                .is_some_and(|lb| lb >= k)
                        })
                {
                    return Some(None);
                }
            }
            Some(&Op::Select(_, a, b)) if depth < 2 => {
                return match (
                    self.edge_arm_hi(func, info, edge, (p, a, depth + 1))?,
                    self.edge_arm_hi(func, info, edge, (p, b, depth + 1))?,
                ) {
                    (None, None) => Some(None),
                    (Some(x), None) | (None, Some(x)) => Some(Some(x)),
                    (Some(x), Some(y)) => Some(Some(x.max(y))),
                };
            }
            _ => {}
        }
        let outside = self
            .const_val(func, info.consts, arg, ty)
            .unwrap_or_else(|| self.of(arg).hi);
        (outside < ty.unsigned_max()).then_some(Some(outside))
    }

    fn const_val(
        &self,
        func: &Function,
        consts: &[Option<i64>],
        v: Value,
        ty: Type,
    ) -> Option<u64> {
        consts
            .get(v.index())
            .copied()
            .flatten()
            .map(|imm| unsigned(imm, ty))
            .or_else(|| match self.op(func, v) {
                Some(&Op::Const(imm)) => Some(unsigned(imm, ty)),
                _ => self.constant(v),
            })
    }

    fn add_step(
        &self,
        func: &Function,
        consts: &[Option<i64>],
        arg: Value,
        p: Value,
        ty: Type,
    ) -> Option<u64> {
        let &Op::Binary(Binary::Iadd, x, y) = self.op(func, arg)? else {
            return None;
        };
        let other = if x == p {
            y
        } else if y == p {
            x
        } else {
            return None;
        };
        self.const_val(func, consts, other, ty).filter(|&k| k > 0)
    }

    fn edge_bound(
        &self,
        func: &Function,
        info: &CfgInfo<'_>,
        (block, pred): (Block, Block),
        (val, upper): (Value, bool),
    ) -> Option<u64> {
        let mut best = None;
        if let Some(&Op::Brif(cond, then, _, otherwise, _)) = func.last(pred)
            && then != otherwise
            && (then == block || otherwise == block)
        {
            best = combine_bound(
                upper,
                best,
                self.proves_bound(func, cond, then == block, (val, upper), 0),
            );
        }
        if let Some(&Op::Brif(cond, then, _, otherwise, _)) = func.last(block)
            && then != otherwise
        {
            if info.single[then.index()] == Some(block) && dominates(info.idom, then, pred) {
                best = combine_bound(
                    upper,
                    best,
                    self.proves_bound(func, cond, true, (val, upper), 0),
                );
            } else if info.single[otherwise.index()] == Some(block)
                && dominates(info.idom, otherwise, pred)
            {
                best = combine_bound(
                    upper,
                    best,
                    self.proves_bound(func, cond, false, (val, upper), 0),
                );
            }
        }
        best
    }

    fn proves_bound(
        &self,
        func: &Function,
        cond: Value,
        taken: bool,
        (val, upper): (Value, bool),
        depth: u8,
    ) -> Option<u64> {
        if depth > 4 {
            return None;
        }
        let ty = func.ty(val);
        match self.op(func, cond) {
            Some(&Op::Binary(Binary::Band, a, b)) if taken => combine_bound(
                upper,
                self.proves_bound(func, a, true, (val, upper), depth + 1),
                self.proves_bound(func, b, true, (val, upper), depth + 1),
            ),
            Some(&Op::Binary(Binary::Bor, a, b)) if !taken => combine_bound(
                upper,
                self.proves_bound(func, a, false, (val, upper), depth + 1),
                self.proves_bound(func, b, false, (val, upper), depth + 1),
            ),
            Some(&Op::Icmp(cc, x, y)) if func.ty(x) == ty => {
                let raw = if taken { cc } else { cc.inverse() };
                let eff = unsigned_order(raw, ty, &self.of(x), &self.of(y))?;
                let (kind, lhs, rhs) = rel_of(eff, x, y);
                match (upper, kind) {
                    (true, RelKind::Eq | RelKind::Ule) if lhs == val => Some(self.of(rhs).hi),
                    (true, RelKind::Ult) if lhs == val => self.of(rhs).hi.checked_sub(1),
                    (false, RelKind::Eq | RelKind::Ule) if rhs == val => Some(self.of(lhs).lo),
                    (false, RelKind::Ult) if rhs == val => {
                        (self.of(lhs).lo < ty.unsigned_max()).then(|| self.of(lhs).lo + 1)
                    }
                    (true, RelKind::Eq) if rhs == val => Some(self.of(lhs).hi),
                    (false, RelKind::Eq) if lhs == val => Some(self.of(rhs).lo),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub(super) fn op<'f>(&self, func: &'f Function, value: Value) -> Option<&'f Op> {
        self.defs
            .get(value.index())
            .copied()
            .flatten()
            .map(|inst| &func.insts[inst.index()].op)
    }

    /// The range of `value` its definition proves.
    pub(crate) fn of(&self, value: Value) -> Fact {
        let (lo, hi) = self.table[value.index()];
        Fact::range(lo, hi)
    }

    /// The constant `value` holds, read unsigned at its width, when its range is one value.
    pub(super) fn constant(&self, value: Value) -> Option<u64> {
        self.of(value).exact()
    }
}
