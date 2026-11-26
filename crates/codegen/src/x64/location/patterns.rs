mod idioms;
mod sib;

pub(super) use crate::tailrec::FOLD_LIMIT;
pub(in super::super) use idioms::{
    adc_fold, adc_plan, bextr_parts, product_halves, shld_parts, unwrap_shift_mask,
};
pub(super) use idioms::{adc_parts, adc_sides};
pub(super) use sib::{SibMatch, odd_scale_source, scaled_op};

use super::super::asm::Reg;
use super::fuse::FuseCtx;
use crate::alloc::Plan;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};
use sib::sib_clobbers_flags;

/// A condition read from compare flags: its compare, a second compare when a select reads two, and its truth table over them.
pub(in super::super) type Root = (Value, Option<Value>, u8);

/// The truth tables of the first and the second compare of a condition.
pub(in super::super) const LEAF: [u8; 2] = [0b1010, 0b1100];

pub(super) fn flag_cmp(func: &Function, op: &Op) -> bool {
    match *op {
        Op::Icmp(_, a, _) | Op::Fits(_, _, a, _) => {
            !func.ty(a).is_vector() && func.ty(a) != Type::I128
        }
        Op::Fcmp(cc, a, _) => {
            !func.ty(a).is_vector() && super::super::float::BRANCHES.contains(&cc)
        }
        _ => false,
    }
}

pub(super) fn parked(plan: &Plan<Reg>, v: Value) -> bool {
    plan.homed.get(v.index()) == Some(&true)
}

pub(super) struct Tree {
    pub(super) nodes: Vec<(Value, usize, bool, u8)>,
    pub(super) leaves: Vec<Value>,
    visits: u8,
    pub(super) table: u8,
}

impl FuseCtx<'_> {
    fn truth(&self, v: Value, max_pos: usize, tree: &mut Tree) -> Option<u8> {
        if let Some(imm) = self.plan.const_of(v) {
            return match crate::fold::extend(imm, self.func.ty(v), false) {
                0 => Some(0),
                1 => Some(0xF),
                _ => None,
            };
        }
        if let Some(&(.., table)) = tree.nodes.iter().find(|node| node.0 == v) {
            return Some(table);
        }
        tree.visits += 1;
        let (vb, pos, inst) = self.pos_of[v.index()]?;
        let taken = self.plan.is_static(v)
            || self.plan.fused[v.index()]
            || self.plan.deferred[v.index()]
            || parked(self.plan, v);
        if tree.visits > 32 || vb != self.b_idx || pos >= max_pos || taken {
            return None;
        }
        let mark = (tree.nodes.len(), tree.leaves.len());
        let op = &self.func.insts[inst.index()].op;
        let logic = match *op {
            Op::Convert(Convert::Uextend | Convert::Ireduce, x) => self.truth(x, pos, tree),
            Op::Binary(Binary::Band, x, y) => self.truths(x, y, pos, tree).map(|(a, b)| a & b),
            Op::Binary(Binary::Bor, x, y) => self.truths(x, y, pos, tree).map(|(a, b)| a | b),
            Op::Binary(Binary::Bxor, x, y) | Op::Icmp(IntCC::NotEqual, x, y) => {
                self.truths(x, y, pos, tree).map(|(a, b)| a ^ b)
            }
            Op::Icmp(IntCC::Equal, x, y) => {
                self.truths(x, y, pos, tree).map(|(a, b)| !(a ^ b) & 0xF)
            }
            _ => None,
        };
        let (table, leaf) = match logic {
            Some(table) => (table, false),
            None => {
                tree.nodes.truncate(mark.0);
                tree.leaves.truncate(mark.1);
                if !flag_cmp(self.func, op) || tree.leaves.len() == LEAF.len() {
                    return None;
                }
                tree.leaves.push(v);
                (LEAF[tree.leaves.len() - 1], true)
            }
        };
        tree.nodes.push((v, pos, leaf, table));
        Some(table)
    }

    fn truths(&self, x: Value, y: Value, max_pos: usize, tree: &mut Tree) -> Option<(u8, u8)> {
        Some((self.truth(x, max_pos, tree)?, self.truth(y, max_pos, tree)?))
    }

    pub(super) fn cond_tree(&self, cond: Value, pos: usize) -> Option<Tree> {
        let mut tree = Tree {
            nodes: Vec::new(),
            leaves: Vec::new(),
            visits: 0,
            table: 0,
        };
        tree.table = self.truth(cond, pos, &mut tree)?;
        let reads = match tree.leaves.len() {
            1 => tree.nodes.len() > 1 && (tree.table == LEAF[0] || tree.table == !LEAF[0] & 0xF),
            2 => matches!(tree.table.count_ones(), 1 | 3),
            _ => false,
        };
        let mut refs = vec![0_u32; tree.nodes.len()];
        *refs.last_mut()? += 1;
        for &(v, _, leaf, _) in &tree.nodes {
            if leaf {
                continue;
            }
            let (_, _, inst) = self.pos_of[v.index()]?;
            self.func.insts[inst.index()].op.for_operands(|o| {
                if let Some(at) = tree.nodes.iter().position(|node| node.0 == o) {
                    refs[at] += 1;
                }
            });
        }
        let owned = tree
            .nodes
            .iter()
            .zip(&refs)
            .all(|(node, &count)| self.uses[node.0.index()] == count);
        (reads && owned).then_some(tree)
    }
}

pub(super) fn is_zero_const(func: &Function, plan: &Plan<Reg>, v: Value) -> bool {
    plan.const_of(v)
        .is_some_and(|imm| crate::fold::extend(imm, func.ty(v), false) == 0)
}

fn addr_clobbers_flags(func: &Function, plan: &Plan<Reg>, addr: Value) -> bool {
    plan.elided[addr.index()]
        && plan.sib[addr.index()].is_some_and(|m| sib_clobbers_flags(func, plan, m))
}

fn operand_clobbers_flags(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    v: Value,
) -> bool {
    if !plan.elided[v.index()] {
        return false;
    }
    let Some(inst) = defs[v.index()] else {
        return false;
    };
    matches!(func.insts[inst.index()].op, Op::Load(addr, _) if addr_clobbers_flags(func, plan, addr))
}

fn scalar_val_ok(func: &Function, v: Value) -> bool {
    let ty = func.ty(v);
    !ty.is_vector() && ty != Type::I128
}

fn imul_preserves_flags(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    res: Value,
    (a, b): (Value, Value),
) -> bool {
    matches!(func.ty(res), Type::I32 | Type::I64)
        && [(a, b), (b, a)].into_iter().any(|(x, c)| {
            !operand_clobbers_flags(func, plan, defs, x)
                && plan
                    .const_of(c)
                    .is_some_and(|imm| matches!(imm, 0 | 1 | 3 | 5 | 9))
        })
}

fn select_preserves_flags(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    roots: &[Option<Root>],
    cmp_res: Value,
    (cond_v, x, y): (Value, Value, Value),
) -> bool {
    let cond_ok = cond_v == cmp_res
        || matches!(roots[cond_v.index()], Some((leaf, None, _)) if leaf == cmp_res)
        || x == y
        || plan.const_of(cond_v).is_some();
    cond_ok
        && !func.ty(x).is_vector()
        && !operand_clobbers_flags(func, plan, defs, x)
        && !operand_clobbers_flags(func, plan, defs, y)
}

pub(super) fn mid_preserves_flags(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    roots: &[Option<Root>],
    cmp_res: Value,
    inst: Inst,
    is_last: bool,
) -> bool {
    let data = &func.insts[inst.index()];
    if is_last && matches!(data.op, Op::Brif(..) | Op::Trapz(..)) {
        return true;
    }
    if data
        .result
        .is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()] || plan.is_static(r))
        && !matches!(data.op, Op::Call { .. } | Op::TryCall { .. })
    {
        return true;
    }
    let res_ok = data.result.is_some_and(|r| scalar_val_ok(func, r));
    match data.op {
        Op::Const(_)
        | Op::StackAddr(..)
        | Op::FramePointer
        | Op::Symbol(_)
        | Op::SlotAddr(_)
        | Op::StackLoad(..) => res_ok,
        Op::Load(addr, _) => res_ok && !addr_clobbers_flags(func, plan, addr),
        Op::Store(val, addr, _) => {
            !plan.elided[val.index()]
                && scalar_val_ok(func, val)
                && !addr_clobbers_flags(func, plan, addr)
        }
        Op::StackStore(val, ..) => !plan.elided[val.index()] && scalar_val_ok(func, val),
        Op::Convert(Convert::Uextend | Convert::Sextend | Convert::Ireduce, src) => {
            res_ok && !func.ty(src).is_vector() && !operand_clobbers_flags(func, plan, defs, src)
        }
        Op::Binary(Binary::Iadd | Binary::Isub, ..) => data.result.is_some_and(|r| {
            func.ty(r) == Type::I64
                && !plan.fused[r.index()]
                && plan.sib[r.index()].is_some_and(|m| !sib_clobbers_flags(func, plan, m))
        }),
        Op::Binary(Binary::Imul, a, b) => data
            .result
            .is_some_and(|r| imul_preserves_flags(func, plan, defs, r, (a, b))),
        Op::Select(cond_v, x, y) => {
            select_preserves_flags(func, plan, defs, roots, cmp_res, (cond_v, x, y))
        }
        _ => false,
    }
}
