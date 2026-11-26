use crate::fold::{const_offset, extend};
use crate::ir::{Binary, Function, Inst, IntCC, Op, Type, Value};

use super::super::icmp::complementary_eq_pair;
use super::super::op_of;
use super::super::select::{build_ord_select, match_ord_select};

fn difference(func: &Function, lcx: &LateCx, v: Value) -> Option<Inst> {
    let inst = lcx.def(v)?;
    let Op::Binary(Binary::Isub | Binary::Bxor, a, b) = func.insts[inst.index()].op else {
        return None;
    };
    let uses = lcx.uses.get(v.index()).copied()?;
    let all_tests = func.ty(a) == Type::I8
        && lcx.konst(func, a).is_some() != lcx.konst(func, b).is_some()
        && lcx.tests.get(v.index()) == Some(&uses);
    (func.ty(a).is_small_int() && (uses == 1 || all_tests)).then_some(inst)
}

struct LateCx {
    defs: Vec<Option<Inst>>,
    uses: Vec<u32>,
    tests: Vec<u32>,
    out: Vec<Inst>,
    dead: Vec<Inst>,
    work: Vec<Value>,
}

impl LateCx {
    fn def(&self, v: Value) -> Option<Inst> {
        self.defs.get(v.index()).copied().flatten()
    }

    fn konst(&self, func: &Function, v: Value) -> Option<i64> {
        match op_of(func, |u| self.def(u), v) {
            Some(&Op::Const(imm)) => Some(imm),
            _ => None,
        }
    }

    fn bump(&mut self, v: Value, delta: i32) {
        let Some(c) = self.uses.get_mut(v.index()) else {
            return;
        };
        if delta > 0 {
            *c += 1;
        } else if *c > 0 {
            *c -= 1;
            if *c == 0 {
                self.work.push(v);
            }
        }
    }

    fn replace_op(&mut self, func: &mut Function, inst: Inst, new_op: Op, start: usize) {
        func.insts[inst.index()]
            .op
            .clone()
            .for_operands(|u| self.bump(u, -1));
        for idx in start..self.out.len() {
            let ni = self.out[idx];
            if let Some(r) = func.insts[ni.index()].result {
                if self.defs.len() <= r.index() {
                    self.defs.resize(r.index() + 1, None);
                    self.uses.resize(r.index() + 1, 0);
                }
                self.defs[r.index()] = Some(ni);
            }
            func.insts[ni.index()].op.for_operands(|u| self.bump(u, 1));
        }
        new_op.for_operands(|u| self.bump(u, 1));
        func.insts[inst.index()].op = new_op;
    }

    fn sweep_orphans(&mut self, func: &mut Function) {
        while let Some(v) = self.work.pop() {
            let Some(inst) = self.def(v) else { continue };
            if self.uses.get(v.index()).copied().unwrap_or(1) != 0 || self.dead.contains(&inst) {
                continue;
            }
            let op = &func.insts[inst.index()].op;
            if !op.is_speculable() {
                continue;
            }
            self.dead.push(inst);
            op.for_operands(|u| self.bump(u, -1));
        }
    }
}

fn late_fold_ord(func: &mut Function, lcx: &mut LateCx, inst: Inst) -> bool {
    let start = lcx.out.len();
    let def_of = |v: Value| lcx.defs.get(v.index()).copied().flatten();
    match func.insts[inst.index()].op {
        Op::Select(cond, t, f) => {
            let Some(m) = match_ord_select(func, &def_of, (cond, t, f)) else {
                return false;
            };
            let new_op = build_ord_select(func.ty(t), m, &mut |uty, op| {
                let val = func.value(uty);
                lcx.out.push(func.add(op, Some(val)));
                val
            });
            lcx.replace_op(func, inst, new_op, start);
            true
        }
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), x, y) => {
            let Some((a, b)) = complementary_eq_pair(func, &def_of, x, y) else {
                return false;
            };
            lcx.replace_op(func, inst, Op::Icmp(cc, a, b), start);
            true
        }
        _ => false,
    }
}

fn offset_target(func: &Function, lcx: &LateCx, op: &Op) -> Option<(Option<IntCC>, Value, i64)> {
    match *op {
        Op::Brif(cond, ..) | Op::Select(cond, ..) => Some((None, cond, 0)),
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), x, y) => {
            match (lcx.konst(func, x), lcx.konst(func, y)) {
                (None, Some(k)) => Some((Some(cc), x, k)),
                (Some(k), None) => Some((Some(cc), y, k)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn shift_offset(
    func: &mut Function,
    lcx: &mut LateCx,
    gone: Inst,
    rhs: i64,
) -> Option<(Value, Value)> {
    let op = func.insts[gone.index()].op.clone();
    let Op::Binary(bin, a, b) = op else {
        return None;
    };
    let ty = func.ty(a);
    if extend(rhs, ty, false) == 0 {
        return Some((a, b));
    }
    let konst = |v: Value| lcx.konst(func, v);
    let (base, imm) = match (bin, konst(a), konst(b)) {
        (Binary::Isub, Some(c), None) => (b, c.wrapping_sub(rhs)),
        (Binary::Isub, ..) => {
            const_offset(&op, konst).map(|(base, delta)| (base, rhs.wrapping_sub(delta)))?
        }
        (_, None, Some(c)) | (_, Some(c), None) => {
            (if konst(a).is_some() { b } else { a }, c ^ rhs)
        }
        _ => return None,
    };
    let val = func.value(ty);
    lcx.out
        .push(func.add(Op::Const(extend(imm, ty, true)), Some(val)));
    Some((base, val))
}

fn offset_compare(func: &mut Function, lcx: &mut LateCx, inst: Inst) -> bool {
    let Some((cc, diff, rhs)) = offset_target(func, lcx, &func.insts[inst.index()].op) else {
        return false;
    };
    let Some(gone) = difference(func, lcx, diff) else {
        return false;
    };
    let start = lcx.out.len();
    let Some((lhs, bound)) = shift_offset(func, lcx, gone, rhs) else {
        return false;
    };
    let (lhs, bound) = complementary_eq_pair(
        func,
        &|v| lcx.defs.get(v.index()).copied().flatten(),
        lhs,
        bound,
    )
    .unwrap_or((lhs, bound));
    let new_op = match cc {
        Some(cc) => Op::Icmp(cc, lhs, bound),
        None => {
            let test = func.value(Type::I8);
            lcx.out
                .push(func.add(Op::Icmp(IntCC::NotEqual, lhs, bound), Some(test)));
            let mut op = func.insts[inst.index()].op.clone();
            if let Op::Brif(cond, ..) | Op::Select(cond, ..) = &mut op {
                *cond = test;
            }
            op
        }
    };
    lcx.replace_op(func, inst, new_op, start);
    if lcx.uses.get(diff.index()) == Some(&0) && !lcx.dead.contains(&gone) {
        lcx.dead.push(gone);
        func.insts[gone.index()]
            .op
            .for_operands(|u| lcx.bump(u, -1));
    }
    true
}

/// The integer compare equivalent to using `op` directly as a non-zero branch condition.
pub(crate) fn branch_cmp(op: &Op) -> Option<(IntCC, Value, Value)> {
    match *op {
        Op::Icmp(cc, a, b) => Some((cc, a, b)),
        Op::Binary(Binary::Isub | Binary::Bxor, a, b) => Some((IntCC::NotEqual, a, b)),
        _ => None,
    }
}

/// Rewrites a branch, a select, or an equality with a constant that reads `a - b` or `a ^ b` into a direct compare of `a`.
pub(crate) fn compare_offsets(func: &mut Function) -> bool {
    let mut lcx = LateCx {
        defs: crate::fold::defs(func),
        uses: func.use_counts(),
        tests: Vec::new(),
        out: Vec::new(),
        dead: Vec::new(),
        work: Vec::new(),
    };
    let mut tests = vec![0_u32; lcx.uses.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        if let Some((_, v, _)) = offset_target(func, &lcx, &func.insts[inst.index()].op)
            && let Some(n) = tests.get_mut(v.index())
        {
            *n += 1;
        }
    }
    lcx.tests = tests;
    let mut changed = false;
    for b in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[b].insts);
        lcx.out = Vec::with_capacity(insts.len());
        for inst in insts {
            changed |= late_fold_ord(func, &mut lcx, inst) || offset_compare(func, &mut lcx, inst);
            lcx.out.push(inst);
        }
        func.blocks[b].insts = std::mem::take(&mut lcx.out);
    }
    lcx.sweep_orphans(func);
    for block in &mut func.blocks {
        block.insts.retain(|inst| !lcx.dead.contains(inst));
    }
    for &inst in &lcx.dead {
        func.insts[inst.index()].op = Op::Trap;
    }
    changed || !lcx.dead.is_empty()
}
