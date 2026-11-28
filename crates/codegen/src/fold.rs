use crate::ir::{Binary, Block, Convert, Domain, Function, Inst, IntCC, Op, Type, Unary, Value};

mod cfg;
mod eval;
mod exits;
mod range;
mod unused;
mod vector;

pub(crate) use crate::ir::compact;
pub(crate) use eval::{
    Before, binary, const_offset, constant, evaluate_wrapped, extend, fits_imm32, holds,
    is_unit_bool, known_bits, non_null_at,
};
pub(crate) use unused::prune;

pub(super) struct Fold {
    pub(super) consts: Vec<Option<i64>>,
    pub(super) subst: Vec<Value>,
    pub(super) incoming: Vec<Option<Value>>,
    pub(super) mixed: Vec<bool>,
    pub(super) defs: Vec<Option<Inst>>,
}

impl Fold {
    pub(super) fn find(&self, value: Value) -> Value {
        if value.index() < self.subst.len() {
            crate::merge::find(&self.subst, value)
        } else {
            value
        }
    }

    pub(super) fn known(&self, value: Value) -> Option<i64> {
        self.consts.get(self.find(value).index()).copied().flatten()
    }

    fn def_inst(&self, value: Value) -> Option<Inst> {
        self.defs.get(self.find(value).index()).copied().flatten()
    }

    fn both(&self, x: Value, y: Value) -> Option<(i64, i64)> {
        Some((self.known(x)?, self.known(y)?))
    }

    fn evaluate(&self, func: &Function, at: Before, op: &Op, ty: Type) -> Option<i64> {
        evaluate_wrapped(func, op, ty, &|value| self.known(value)).or_else(|| match *op {
            Op::Binary(op, x, y) if ty.is_small_int() => {
                self.absorbed(op, ty, x, y).map(|v| extend(v, ty, true))
            }
            Op::Icmp(cc, x, y) => crate::algebra::non_null_cmp(
                func,
                &|v| self.def_inst(v),
                &|v| self.known(v),
                at,
                (cc, x, y),
            )
            .map(i64::from),
            Op::Load(addr, off) if !func.rodata_bytes.is_empty() && ty.is_small_int() => {
                self.rodata_load(func, ty, addr, off)
            }
            _ => None,
        })
    }

    fn rodata_load(&self, func: &Function, ty: Type, mut addr: Value, off: i32) -> Option<i64> {
        let mut total = i64::from(off);
        for _ in 0..4 {
            match *self.def(func, addr)? {
                Op::Symbol(sym) => return func.rodata_read(sym, total, ty),
                Op::Binary(Binary::Iadd, a, b) => {
                    let (base, c) = if let Some(k) = self.known(b) {
                        (a, k)
                    } else {
                        (b, self.known(a)?)
                    };
                    (total, addr) = (total.checked_add(c)?, base);
                }
                _ => return None,
            }
        }
        None
    }

    pub(super) fn non_null(&self, func: &Function, at: Before, value: Value) -> bool {
        non_null_at(func, &|v| self.def_inst(v), &|v| self.known(v), at, value)
    }

    fn absorbed(&self, op: Binary, ty: Type, x: Value, y: Value) -> Option<i64> {
        let zero = |value: Value| {
            self.known(value)
                .is_some_and(|imm| extend(imm, ty, false) == 0)
        };
        match op {
            Binary::Imul | Binary::Umulhi | Binary::Smulhi | Binary::Band if zero(x) || zero(y) => {
                Some(0)
            }
            Binary::Isub | Binary::Bxor if self.find(x) == self.find(y) => Some(0),
            Binary::Srem | Binary::Urem
                if self.known(y).is_some_and(|imm| {
                    extend(imm, ty, false) == 1
                        || (op == Binary::Srem && extend(imm, ty, true) == -1)
                }) =>
            {
                Some(0)
            }
            _ => None,
        }
    }

    /// The operation of the instruction that defines `value`, when one in a block does.
    pub(super) fn def<'a>(&self, func: &'a Function, value: Value) -> Option<&'a Op> {
        self.def_inst(value)
            .map(|inst| &func.insts[inst.index()].op)
    }

    pub(super) fn boolean(&self, func: &Function, value: Value) -> bool {
        self.known(value).is_none()
            && is_unit_bool(func, &|v| self.def_inst(v), &|v| self.known(v), value)
    }

    pub(super) fn same(&self, ty: Type, x: Value, y: Value) -> bool {
        self.find(x) == self.find(y)
            || matches!(self.both(x, y), Some((a, b)) if extend(a, ty, false) == extend(b, ty, false))
    }

    fn identity(&self, func: &Function, op: &Op, ty: Type) -> Option<Value> {
        match *op {
            Op::Select(condition, x, y) => {
                if self.find(x) == self.find(y) {
                    return Some(x);
                }
                if let Some(known) = self.known(condition) {
                    return Some(if known != 0 { x } else { y });
                }
                if func.ty(condition) == func.ty(x)
                    && self.boolean(func, condition)
                    && self.known(x) == Some(1)
                    && self.known(y) == Some(0)
                {
                    return Some(condition);
                }
                self.equalities(func, condition)
                    .find(|&(var, imm, on_true)| {
                        let is_c = |v: Value| {
                            self.known(v)
                                .is_some_and(|v_imm| extend(v_imm, func.ty(var), false) == imm)
                        };
                        let is_var = |v: Value| self.find(v) == self.find(var);
                        if on_true {
                            is_c(x) && is_var(y)
                        } else {
                            is_var(x) && is_c(y)
                        }
                    })
                    .map(|(var, ..)| var)
            }
            Op::Convert(Convert::Bitcast, x) => {
                (func.ty(x) == ty)
                    .then_some(x)
                    .or_else(|| match *self.def(func, x)? {
                        Op::Convert(Convert::Bitcast, value) if func.ty(value) == ty => Some(value),
                        _ => None,
                    })
            }
            Op::Extract(..) | Op::Shuffle(..) => vector::vec_identity(self, func, op),
            Op::Binary(Binary::Fmin | Binary::Fmax, x, y) if self.find(x) == self.find(y) => {
                Some(x)
            }
            Op::Binary(kind, x, y) if func.ty(x).is_small_int() => self
                .neutral(func, kind, func.ty(x), x, y)
                .or_else(|| self.cancel(func, kind, x, y)),
            Op::Unary(kind @ (Unary::Ineg | Unary::Bnot), x) => match *self.def(func, x)? {
                Op::Unary(inner, value) if inner == kind => Some(value),
                _ => None,
            },
            Op::Icmp(cc, x, y) => [(x, y), (y, x)].into_iter().find_map(|(b, c)| {
                let imm = extend(self.known(c)?, func.ty(b), false);
                (func.ty(b) == Type::I8
                    && self.boolean(func, b)
                    && matches!((cc, imm), (IntCC::NotEqual, 0) | (IntCC::Equal, 1)))
                .then_some(b)
            }),
            _ => None,
        }
    }

    fn neutral(&self, func: &Function, op: Binary, ty: Type, x: Value, y: Value) -> Option<Value> {
        let is = |value: Value, want: i64| {
            self.known(value)
                .is_some_and(|imm| extend(imm, ty, false) == extend(want, ty, false))
        };
        if let Some(unit) = eval::unit(op) {
            if is(y, unit) {
                return Some(x);
            }
            return (!matches!(op, Binary::Isub | Binary::Sdiv | Binary::Udiv) && is(x, unit))
                .then_some(y);
        }
        let shift = matches!(
            op,
            Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr
        );
        if shift
            && self
                .known(y)
                .is_some_and(|count| count & i64::from(ty.bits() - 1) == 0)
        {
            return Some(x);
        }
        if op == Binary::Band {
            for (val, other) in [(x, y), (y, x)] {
                if let Some(mask) = self.known(other) {
                    let (kz, _) =
                        known_bits(func, &|v| self.def_inst(v), &|v| self.known(v), val, 0);
                    let active = !kz & ty.unsigned_max();
                    if (active & (extend(mask, ty, false) as u64)) == active {
                        return Some(val);
                    }
                }
            }
        }
        None
    }

    fn cancel(&self, func: &Function, op: Binary, x: Value, y: Value) -> Option<Value> {
        let ty = func.ty(x);
        if op == Binary::Isub {
            return match *self.def(func, x)? {
                Op::Binary(Binary::Iadd, a, b) if self.same(ty, a, y) => Some(b),
                Op::Binary(Binary::Iadd, a, b) if self.same(ty, b, y) => Some(a),
                _ => None,
            };
        }
        if op != Binary::Bxor {
            return None;
        }
        [(x, y), (y, x)]
            .into_iter()
            .find_map(|(inner, other)| match *self.def(func, inner)? {
                Op::Binary(Binary::Bxor, a, b) if self.same(ty, b, other) => Some(a),
                Op::Binary(Binary::Bxor, a, b) if self.same(ty, a, other) => Some(b),
                _ => None,
            })
    }

    fn constants(&mut self, func: &Function) {
        for (block, data_b) in func.blocks.iter().enumerate() {
            for (pos, inst) in data_b.insts.iter().enumerate() {
                let data = &func.insts[inst.index()];
                let Some(result) = data
                    .result
                    .filter(|r| self.subst[r.index()] == *r && self.consts[r.index()].is_none())
                else {
                    continue;
                };
                let ty = func.ty(result);
                match data.op {
                    Op::Const(imm) => {
                        if ty.is_small_int() {
                            self.consts[result.index()] = Some(extend(imm, ty, true));
                        }
                        continue;
                    }
                    Op::StackLoad(..)
                    | Op::StackAddr(..)
                    | Op::Symbol(_)
                    | Op::SlotAddr(_)
                    | Op::Runtime(_)
                    | Op::Call { .. }
                    | Op::TryCall { .. }
                    | Op::AtomicLoad(..)
                    | Op::AtomicRmw(..)
                    | Op::AtomicCas(..)
                    | Op::Insert(..) => continue,
                    _ => {}
                }
                let at = Before {
                    prefix: &data_b.insts[..pos],
                    block: Some(Block(block as u32)),
                };
                if let Some(imm) = self.evaluate(func, at, &data.op, ty) {
                    self.consts[result.index()] = Some(imm);
                } else if let Some(v) = self.identity(func, &data.op, ty) {
                    self.subst[result.index()] = v;
                }
            }
        }
    }
}

/// Calls `f` with each block the terminator of `block` jumps to with arguments, and those arguments.
pub(crate) fn for_edges<'a>(
    func: &'a Function,
    block: usize,
    mut f: impl FnMut(Block, &'a [Value]),
) {
    let Some(last) = func.blocks[block].insts.last() else {
        return;
    };
    match &func.insts[last.index()].op {
        Op::Jump(target, args) => f(*target, args),
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            f(*then, then_args);
            f(*otherwise, else_args);
        }
        _ => {}
    }
}

/// The instruction in a block that defines each value.
pub(crate) fn defs(func: &Function) -> Vec<Option<Inst>> {
    let mut defs = vec![None; func.types.len()];
    for &i in func.blocks.iter().flat_map(|b| &b.insts) {
        if let Some(r) = func.insts[i.index()].result {
            defs[r.index()] = Some(i);
        }
    }
    defs
}

fn fold_select_op(func: &Function, fold: &Fold, c: Value, x: Value, y: Value) -> Option<Op> {
    match (fold.def(func, x), fold.def(func, y)) {
        (_, Some(&Op::Select(c2, _, z))) if fold.find(c2) == c => {
            Some(Op::Select(c, x, fold.find(z)))
        }
        (Some(&Op::Select(c2, x2, _)), _) if fold.find(c2) == c => {
            Some(Op::Select(c, fold.find(x2), y))
        }
        _ if func.ty(c) == Type::I8
            && fold.boolean(func, c)
            && func.ty(x).is_small_int()
            && func.ty(x) != Type::I8
            && fold
                .known(x)
                .is_some_and(|v| extend(v, func.ty(x), false) == 1)
            && fold
                .known(y)
                .is_some_and(|v| extend(v, func.ty(x), false) == 0) =>
        {
            Some(Op::Convert(Convert::Uextend, c))
        }
        _ => None,
    }
}

fn rewrite_folded_inst(func: &mut Function, fold: &Fold, inst: Inst) -> bool {
    let mut changed = false;
    func.insts[inst.index()].op.map_operands(|v| {
        let n = fold.find(v);
        changed |= n != v;
        n
    });
    let replacement = match func.insts[inst.index()].op {
        Op::Binary(Binary::Fmul, x, y) if crate::algebra::is_float_two(func, &fold.defs, y) => {
            Some(Op::Binary(Binary::Fadd, x, x))
        }
        Op::Binary(Binary::Fmul, x, y) if crate::algebra::is_float_two(func, &fold.defs, x) => {
            Some(Op::Binary(Binary::Fadd, y, y))
        }
        Op::Binary(Binary::Isub, x, y)
            if func.ty(x).is_small_int()
                && fold
                    .known(x)
                    .is_some_and(|imm| extend(imm, func.ty(x), false) == 0) =>
        {
            Some(Op::Unary(Unary::Ineg, y))
        }
        Op::Select(c, x, y) => fold_select_op(func, fold, c, x, y),
        _ => vector::vec_zero_fold(
            func,
            fold,
            &func.insts[inst.index()].op,
            func.insts[inst.index()].result,
        )
        .or_else(|| {
            func.insts[inst.index()]
                .result
                .and_then(|result| fold.consts[result.index()])
                .map(Op::Const)
        }),
    };
    if let Some(op) = replacement
        && func.insts[inst.index()].op != op
    {
        (func.insts[inst.index()].op, changed) = (op, true);
    }
    changed
}

fn settle_cfg_and_params(
    func: &mut Function,
    fold: &mut Fold,
    entry: Block,
    effort: crate::Effort,
    pinned: &[bool],
) -> (bool, bool) {
    let (mut reachable, mut changed, mut cfg_changed) = (Vec::new(), false, false);
    loop {
        fold.constants(func);
        let settled = effort == crate::Effort::Full
            && reachable.is_empty()
            && crate::facts::settle(func, &fold.defs, &mut fold.consts);
        let mut branched = settled | fold.branches(func) | fold.thread(func, pinned);
        if branched {
            func.bump_cfg();
        }
        if branched || reachable.is_empty() {
            let (r, pruned) = unused::prune_changed(func, entry, !branched);
            if pruned {
                func.bump_cfg();
            }
            (reachable, cfg_changed, changed) =
                (r, cfg_changed | branched | pruned, changed | pruned);
        }
        let mut replaced = fold.params(func, &reachable, pinned);
        if !branched
            && !replaced
            && effort == crate::Effort::Full
            && fold.entry_guards(func, &reachable, pinned)
        {
            func.bump_cfg();
            let (r, pruned) = unused::prune_changed(func, entry, false);
            if pruned {
                func.bump_cfg();
            }
            (reachable, branched, cfg_changed, replaced) = (r, true, true, true);
        }
        changed |= branched | replaced;
        if !branched && !replaced {
            return (changed, cfg_changed);
        }
    }
}

/// Resolves a jump-table dispatch for an unsigned `index`, falling back to `default` when out of bounds.
pub(crate) fn table_target(index: u64, default: Block, targets: &[Block]) -> Block {
    usize::try_from(index)
        .ok()
        .and_then(|at| targets.get(at).copied())
        .unwrap_or(default)
}

/// Folds constants, identities and branches on them, settles the compares, traps and branches the facts on each path decide when `effort` is full, empties blocks no path reaches, replaces block parameters every edge agrees on, drops dead pure instructions, and threads the edges through the blocks that leaves holding only a jump; returns the invalidated IR domain.
pub(crate) fn run(func: &mut Function, effort: crate::Effort) -> Domain {
    let Some(entry) = func.entry else {
        return Domain::NONE;
    };
    let count = func.types.len();
    let mut fold = Fold {
        consts: vec![None; count],
        subst: (0..count as u32).map(Value).collect(),
        incoming: Vec::new(),
        mixed: Vec::new(),
        defs: defs(func),
    };
    let pinned = unused::pinned(func, entry);
    let (mut changed, mut cfg_changed) =
        settle_cfg_and_params(func, &mut fold, entry, effort, &pinned);
    let (dropped_params, mut body_changed) = (unused::drop_params(func, &fold.subst), false);
    for b in 0..func.blocks.len() {
        for idx in 0..func.blocks[b].insts.len() {
            body_changed |= rewrite_folded_inst(func, &fold, func.blocks[b].insts[idx]);
        }
    }
    let rev_before = func.cfg_rev;
    if effort == crate::Effort::Full {
        body_changed |= crate::algebra::rewrite(func, &fold.defs)
            | crate::algebra::invert(func)
            | vector::fold_bitcast_loads(func, &fold.defs);
    }
    body_changed |= unused::run(func, &pinned) | unused::dead(func);
    (changed, cfg_changed) = (
        changed | body_changed,
        cfg_changed | (func.cfg_rev != rev_before),
    );
    if (body_changed || dropped_params) && (fold.thread(func, &pinned) | fold.branches(func)) {
        func.bump_cfg();
        unused::prune_changed(func, entry, false);
        unused::dead(func);
        (changed, cfg_changed) = (true, true);
    }
    if effort == crate::Effort::Full
        && (crate::algebra::devirt_vtables(func, &fold.defs)
            | crate::algebra::fold_bxor_conds(func)
            | range::fold_range_checks(func))
    {
        unused::dead(func);
        changed = true;
    }
    if !changed {
        Domain::NONE
    } else if cfg_changed {
        Domain::VALUES | Domain::CFG
    } else {
        Domain::VALUES
    }
}
