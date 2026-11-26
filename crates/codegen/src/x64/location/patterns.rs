use super::fuse::FuseCtx;
use super::super::asm::Reg;
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};
use crate::live::{Sib, SibBase};

pub(super) use crate::tailrec::FOLD_LIMIT;

pub(super) struct SibMatch {
    pub(super) sib: Sib,
    pub(super) elide: Vec<(Value, usize)>,
    pub(super) base_start: usize,
    pub(super) idx_start: usize,
}

pub(super) fn scaled_op(func: &Function, plan: &Plan<Reg>, inst: Inst) -> Option<(Value, u8, Option<u8>)> {
    match func.insts[inst.index()].op {
        Op::Binary(Binary::Ishl, a, b) => plan.const_of(b).filter(|c| matches!(c, 1..=3)).map(|c| (a, c as u8, None)),
        Op::Binary(Binary::Imul, a, b) => [(a, b), (b, a)].into_iter().find_map(|(x, c)| {
            let k = plan.const_of(c).filter(|&k| k > 0)?;
            let shift = (k as u64).trailing_zeros() as u8;
            let odd = k >> shift;
            match odd {
                1 if matches!(shift, 1..=3) => Some((x, shift, None)),
                3 | 5 | 9 if shift <= 3 => Some((x, shift, Some((odd - 1).trailing_zeros() as u8))),
                _ => None,
            }
        }),
        _ => None,
    }
}

pub(super) fn odd_scale_source(
    func: &Function,
    plan: &Plan<Reg>,
    def: impl Fn(Value) -> Option<Inst>,
    index: Value,
) -> Option<(Value, u8)> {
    if !plan.elided[index.index()] {
        return None;
    }
    let (raw, _, Some(scale)) = scaled_op(func, plan, def(index)?)? else { return None };
    let mut x = raw;
    if plan.elided[raw.index()]
        && let Some(einst) = def(raw)
        && let Op::Convert(Convert::Uextend | Convert::Sextend, inner) = func.insts[einst.index()].op
    {
        x = inner;
    }
    Some((x, scale))
}

/// A condition read from compare flags: its compare, a second compare when a select reads two, and its truth table over them.
pub(in super::super) type Root = (Value, Option<Value>, u8);

/// The truth tables of the first and the second compare of a condition.
pub(in super::super) const LEAF: [u8; 2] = [0b1010, 0b1100];

pub(super) fn flag_cmp(func: &Function, op: &Op) -> bool {
    match *op {
        Op::Icmp(_, a, _) | Op::Fits(_, _, a, _) => !func.ty(a).is_vector() && func.ty(a) != Type::I128,
        Op::Fcmp(cc, a, _) => !func.ty(a).is_vector() && super::super::float::BRANCHES.contains(&cc),
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
        let taken = self.plan.is_static(v) || self.plan.fused[v.index()] || self.plan.deferred[v.index()] || parked(self.plan, v);
        if tree.visits > 32 || vb != self.b_idx || pos >= max_pos || taken {
            return None;
        }
        let mark = (tree.nodes.len(), tree.leaves.len());
        let op = &self.func.insts[inst.index()].op;
        let logic = match *op {
            Op::Convert(Convert::Uextend | Convert::Ireduce, x) => self.truth(x, pos, tree),
            Op::Binary(Binary::Band, x, y) => self.truths(x, y, pos, tree).map(|(a, b)| a & b),
            Op::Binary(Binary::Bor, x, y) => self.truths(x, y, pos, tree).map(|(a, b)| a | b),
            Op::Binary(Binary::Bxor, x, y) | Op::Icmp(IntCC::NotEqual, x, y) => self.truths(x, y, pos, tree).map(|(a, b)| a ^ b),
            Op::Icmp(IntCC::Equal, x, y) => self.truths(x, y, pos, tree).map(|(a, b)| !(a ^ b) & 0xF),
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
        let mut tree = Tree { nodes: Vec::new(), leaves: Vec::new(), visits: 0, table: 0 };
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
        let owned = tree.nodes.iter().zip(&refs).all(|(node, &count)| self.uses[node.0.index()] == count);
        (reads && owned).then_some(tree)
    }
}

pub(super) fn is_zero_const(func: &Function, plan: &Plan<Reg>, v: Value) -> bool {
    plan.const_of(v).is_some_and(|imm| crate::fold::extend(imm, func.ty(v), false) == 0)
}

fn contiguous_ones(v: u64) -> Option<u8> {
    if v == 0 {
        return None;
    }
    let len = v.trailing_ones();
    (len < 64 && v == (1_u64 << len) - 1).then_some(len as u8)
}

pub(in super::super) fn shld_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    (x, y): (Value, Value),
    child_ok: impl Fn(Value) -> bool,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Value, Value, Value, u8, u8)> {
    if !matches!(ty, Type::I32 | Type::I64) || x == y || !child_ok(x) || !child_ok(y) {
        return None;
    }
    let bits = i64::from(ty.bits());
    for (a, b) in [(x, y), (y, x)] {
        let (ai, bi) = (def(a)?, def(b)?);
        if let (Op::Binary(Binary::Ishl, hi, cl), Op::Binary(Binary::Ushr, lo, cr)) = (&func.insts[ai.index()].op, &func.insts[bi.index()].op)
            && let (Some(l), Some(r)) = (plan.const_of(*cl), plan.const_of(*cr))
            && (1..bits).contains(&l)
            && (1..bits).contains(&r)
            && l + r == bits
        {
            return Some((*hi, *lo, *cl, l as u8, r as u8));
        }
    }
    None
}

pub(in super::super) fn bextr_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    (x, y): (Value, Value),
    child_ok: impl Fn(Value) -> bool,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Option<Value>, Value, u8, u8)> {
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let bits = i64::from(ty.bits());
    match op {
        Binary::Band => {
            for (a, b) in [(x, y), (y, x)] {
                let Some(mask_raw) = plan.const_of(b) else { continue };
                let mask = crate::fold::extend(mask_raw, ty, false) as u64;
                if child_ok(a)
                    && let Some(ai) = def(a)
                    && let Op::Binary(Binary::Ushr, src, cnt) = func.insts[ai.index()].op
                    && let Some(start) = plan.const_of(cnt)
                    && (1..bits).contains(&start)
                    && let Some(len) = contiguous_ones(mask)
                    && i64::from(len) < bits - start
                {
                    return Some((Some(a), src, start as u8, len));
                }
                if ty == Type::I64
                    && i32::try_from(mask_raw).is_err()
                    && mask != 0xFFFF_FFFF
                    && let Some(len) = contiguous_ones(mask)
                    && len < 64
                {
                    return Some((None, a, 0, len));
                }
            }
            None
        }
        Binary::Ushr => {
            let start = plan.const_of(y)?;
            if !(1..bits).contains(&start) || !child_ok(x) {
                return None;
            }
            let xi = def(x)?;
            let Op::Binary(Binary::Band, a, b) = func.insts[xi.index()].op else { return None };
            for (src, mask_v) in [(a, b), (b, a)] {
                let Some(mask_raw) = plan.const_of(mask_v) else { continue };
                let mask = (crate::fold::extend(mask_raw, ty, false) as u64) >> (start as u32);
                if let Some(len) = contiguous_ones(mask)
                    && i64::from(len) < bits - start
                {
                    return Some((Some(x), src, start as u8, len));
                }
            }
            None
        }
        _ => None,
    }
}

pub(in super::super) fn adc_plan(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    (x, y): (Value, Value),
    child_ok: impl Fn(Value) -> bool,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Value, Option<Value>, Value, Value, Option<Value>, bool)> {
    let (cand, inner, cc, a, b) = adc_parts(func, plan, ty, op, (x, y), child_ok, def)?;
    let (invert, left, right) = adc_sides(func, plan, cc, a, b)?;
    Some((cand, inner, left, if cand == y { x } else { y }, right, invert))
}

/// Whether the carry `a cc b` enters an `adc` or `sbb` inverted, and the compare it reads as `left <u right`, or as `left != 0` with no `right`.
pub(super) fn adc_sides(func: &Function, plan: &Plan<Reg>, cc: IntCC, a: Value, b: Value) -> Option<(bool, Value, Option<Value>)> {
    Some(match cc {
        IntCC::UnsignedLessThan => (false, a, Some(b)),
        IntCC::UnsignedGreaterThan if is_zero_const(func, plan, b) => (true, a, None),
        IntCC::UnsignedGreaterThan => (false, b, Some(a)),
        IntCC::UnsignedGreaterThanOrEqual => (true, a, Some(b)),
        IntCC::UnsignedLessThanOrEqual => (true, b, Some(a)),
        IntCC::Equal => (false, if is_zero_const(func, plan, b) { a } else { b }, None),
        IntCC::NotEqual => (true, if is_zero_const(func, plan, b) { a } else { b }, None),
        _ => return None,
    })
}

/// The operands of `other` when it is the add or subtract of the same kind as `op` that an `adc` or `sbb` adding an uninverted carry can compute in the same instruction.
pub(in super::super) fn adc_fold(func: &Function, op: Binary, invert: bool, other: Value, def: impl Fn(Value) -> Option<Inst>) -> Option<(Value, Value)> {
    match func.insts[def(other)?.index()].op {
        Op::Binary(inner, p, q) if !invert && inner == op => Some((p, q)),
        _ => None,
    }
}

pub(in super::super) fn unwrap_shift_mask(func: &Function, plan: &Plan<Reg>, defs: &[Option<Inst>], y: Value, shift_res: Value, bits: u32) -> Value {
    if !matches!(bits, 32 | 64) {
        return y;
    }
    let Some(y_inst) = defs[y.index()] else { return y };
    let Op::Binary(Binary::Band, a, b) = func.insts[y_inst.index()].op else { return y };
    let mask = u64::from(bits - 1);
    let const_val = |v: Value| {
        plan.const_of(v).or_else(|| {
            defs[v.index()].and_then(|i| match func.insts[i.index()].op {
                Op::Const(imm) => Some(imm),
                _ => None,
            })
        })
    };
    let Some(raw) = [(a, b), (b, a)].into_iter().find_map(|(raw, m)| {
        let imm = const_val(m)?;
        ((imm as u64 & mask) == mask).then_some(raw)
    }) else {
        return y;
    };
    if plan.elided[y.index()] {
        return raw;
    }
    let Some(r_raw) = plan.reg_of(raw) else { return y };
    if plan.elided[raw.index()] || plan.reg_of(y) == Some(r_raw) {
        return y;
    }
    let Some(shift_inst) = defs[shift_res.index()] else { return y };
    for block in &func.blocks {
        let Some(ypos) = block.insts.iter().position(|&i| i == y_inst) else { continue };
        let Some(spos) = block.insts.iter().position(|&i| i == shift_inst) else { return y };
        if ypos >= spos {
            return y;
        }
        let ok = block.insts[ypos + 1..spos].iter().all(|&inst| {
            let data = &func.insts[inst.index()];
            if data.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]) {
                return true;
            }
            !matches!(data.op, Op::Call { .. } | Op::TryCall { .. })
                && !data.result.and_then(|r| plan.pairs[r.index()]).is_some_and(|p| plan.reg_of(p) == Some(r_raw))
                && !crate::live::results(data).any(|r| plan.reg_of(r) == Some(r_raw))
        });
        return if ok { raw } else { y };
    }
    y
}

fn adc_cc_ok(func: &Function, plan: &Plan<Reg>, cc: IntCC, a: Value, b: Value) -> bool {
    func.ty(a).is_small_int()
        && (matches!(
            cc,
            IntCC::UnsignedLessThan | IntCC::UnsignedGreaterThan | IntCC::UnsignedLessThanOrEqual | IntCC::UnsignedGreaterThanOrEqual
        ) || (matches!(cc, IntCC::Equal | IntCC::NotEqual) && (is_zero_const(func, plan, a) || is_zero_const(func, plan, b))))
}

pub(super) fn adc_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    (x, y): (Value, Value),
    child_ok: impl Fn(Value) -> bool,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(Value, Option<Value>, IntCC, Value, Value)> {
    if !matches!(op, Binary::Iadd | Binary::Isub) || !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let both = [(y, x), (x, y)];
    let one = [(y, x)];
    let pairs = if op == Binary::Iadd { &both[..] } else { &one[..] };
    for &(cand, _) in pairs {
        if !child_ok(cand) {
            continue;
        }
        let Some(ci) = def(cand) else { continue };
        match func.insts[ci.index()].op {
            Op::Icmp(cc, a, b) if adc_cc_ok(func, plan, cc, a, b) => return Some((cand, None, cc, a, b)),
            Op::Convert(Convert::Uextend, inner) if child_ok(inner) => {
                if let Some(ii) = def(inner)
                    && let Op::Icmp(cc, a, b) = func.insts[ii.index()].op
                    && adc_cc_ok(func, plan, cc, a, b)
                {
                    return Some((cand, Some(inner), cc, a, b));
                }
            }
            _ => {}
        }
    }
    None
}

fn sib_clobbers_flags(func: &Function, plan: &Plan<Reg>, mode: Sib) -> bool {
    let idx_ok = !plan.elided[mode.index.index()]
        && plan.reg_of(mode.index).is_some_and(|_| func.ty(mode.index).bytes() >= 8 || plan.zero_ext[mode.index.index()]);
    match mode.base {
        SibBase::Frame(_) => false,
        SibBase::Value(base) => (plan.elided[base.index()] || plan.reg_of(base).is_none()) && !idx_ok,
    }
}

impl FuseCtx<'_> {
    fn peel_step(&self, vinst: Inst) -> Option<(Value, i64)> {
        match self.func.insts[vinst.index()].op {
            Op::Binary(Binary::Iadd, a, b) => match (self.plan.const_of(a), self.plan.const_of(b)) {
                (_, Some(c)) => Some((a, c)),
                (Some(c), None) => Some((b, c)),
                _ => None,
            },
            Op::Binary(Binary::Isub, a, b) => self.plan.const_of(b).and_then(|c| c.checked_neg()).map(|c| (a, c)),
            _ => None,
        }
    }

    pub(super) fn peel_const(&self, mut val: Value, mut max_pos: usize, mut peel_cur: bool, elide: &mut Vec<(Value, usize)>) -> Option<(Value, i64, usize)> {
        let (mut disp, mut last_pos) = (0_i64, max_pos);
        loop {
            if self.plan.elided[val.index()] || self.func.ty(val) != Type::I64 {
                break;
            }
            let Some((vb, vpos, vinst)) = self.pos_of[val.index()] else { break };
            if vb != self.b_idx || (peel_cur && (self.uses[val.index()] != 1 || vpos >= max_pos)) {
                break;
            }
            let Some((next, delta)) = self.peel_step(vinst) else { break };
            let next_disp = disp.checked_add(delta)?;
            if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&next_disp) {
                break;
            }
            disp = next_disp;
            if peel_cur {
                elide.push((val, vpos));
            }
            (last_pos, max_pos, val, peel_cur) = (vpos, vpos, next, true);
        }
        Some((val, disp, last_pos))
    }

    fn small_op_bounds(&self, v: Value, depth: u8, (bmin, bmax): (u64, u64)) -> (u64, u64) {
        let ty = self.func.ty(v);
        let w = ty.unsigned_max();
        let Some(inst) = self.defs[v.index()] else { return (bmin, bmax) };
        let sub = |u: Value| self.small_bounds(u, depth + 1);
        match self.func.insts[inst.index()].op {
            Op::Convert(Convert::Uextend, a) if self.func.ty(a).is_small_int() => {
                let (l, h) = sub(a);
                let aw = self.func.ty(a).unsigned_max();
                (l.min(aw).max(bmin), h.min(aw).min(bmax))
            }
            Op::Convert(Convert::Sextend, a) if self.func.ty(a).is_small_int() => {
                let sm = (1_u64 << (self.func.ty(a).bits() - 1)) - 1;
                let (l, h) = sub(a);
                if h <= sm { (l.max(bmin), h.min(bmax)) } else { (bmin, bmax) }
            }
            Op::Convert(Convert::Ireduce, a) if self.func.ty(a).is_small_int() => {
                let (l, h) = sub(a);
                if h <= w { (l.max(bmin), h.min(bmax)) } else { (bmin, bmax) }
            }
            Op::Binary(Binary::Iadd, a, b) => {
                let ((l1, h1), (l2, h2)) = (sub(a), sub(b));
                h1.checked_add(h2).filter(|&h| h <= w).map_or((bmin, bmax), |h| ((l1 + l2).max(bmin), h.min(bmax)))
            }
            Op::Binary(Binary::Isub, a, b) => {
                let ((l1, h1), (l2, h2)) = (sub(a), sub(b));
                if l1 >= h2 { ((l1 - h2).max(bmin), (h1 - l2).min(bmax)) } else { (bmin, bmax) }
            }
            Op::Binary(Binary::Umin, a, b) => {
                let ((l1, h1), (l2, h2)) = (sub(a), sub(b));
                (l1.min(l2).max(bmin), h1.min(h2).min(bmax))
            }
            Op::Binary(Binary::Ushr, a, b) => {
                let (l, h) = sub(a);
                self.plan.const_of(b).map_or((bmin, h.min(bmax)), |c| {
                    let k = (c as u32) & (ty.bits() - 1);
                    ((l >> k).max(bmin), (h >> k).min(bmax))
                })
            }
            Op::Select(_, a, b) => {
                let ((l1, h1), (l2, h2)) = (sub(a), sub(b));
                (l1.min(l2).max(bmin), h1.max(h2).min(bmax))
            }
            _ => (bmin, bmax),
        }
    }

    fn small_bounds(&self, v: Value, depth: u8) -> (u64, u64) {
        let ty = self.func.ty(v);
        let w = ty.unsigned_max();
        if !ty.is_small_int() {
            return (0, w);
        }
        let (z, o) = crate::fold::known_bits(self.func, &|u| self.defs[u.index()], &|u| self.plan.const_of(u), v, depth);
        let (bmin, bmax) = (o & w, !z & w);
        if depth >= 6 || bmin == bmax {
            return (bmin, bmax);
        }
        self.small_op_bounds(v, depth, (bmin, bmax))
    }

    pub(super) fn nonneg_small(&self, v: Value) -> bool {
        let ty = self.func.ty(v);
        ty.is_small_int() && ty.bytes() <= 4 && self.small_bounds(v, 0).1 <= (1_u64 << (ty.bits() - 1)) - 1
    }

    fn uext_inner(&self, inst: Inst) -> Option<Value> {
        match self.func.insts[inst.index()].op {
            Op::Convert(Convert::Uextend, inner) => Some(inner),
            Op::Convert(Convert::Sextend, inner) if self.nonneg_small(inner) => Some(inner),
            _ => None,
        }
    }

    fn small_const_step(&self, vinst: Inst, ty: Type) -> Option<(Value, i64)> {
        let w = ty.unsigned_max();
        match self.func.insts[vinst.index()].op {
            Op::Binary(Binary::Iadd, a, b) => [(a, b), (b, a)].into_iter().find_map(|(x, c)| {
                let imm = self.plan.const_of(c)?;
                let (xmin, xmax) = self.small_bounds(x, 0);
                let u = crate::fold::extend(imm, ty, false) as u64;
                if xmax.checked_add(u).is_some_and(|m| m <= w) {
                    return Some((x, u as i64));
                }
                let s = crate::fold::extend(imm, ty, true);
                (s < 0 && xmin >= s.unsigned_abs()).then_some((x, s))
            }),
            Op::Binary(Binary::Isub, a, b) => {
                let u = crate::fold::extend(self.plan.const_of(b)?, ty, false) as u64;
                (self.small_bounds(a, 0).0 >= u).then_some((a, -(u as i64)))
            }
            _ => None,
        }
    }

    fn uext_target_ok(&self, v: Value, multi_use: bool) -> bool {
        self.index_ok(v) && (!multi_use || self.plan.zero_ext[v.index()] || self.plan.reg_of(v).is_none())
    }

    fn peel_small_step(&self, cur: Value, cur_pos: usize, (shift, cur_disp): (u8, i64)) -> Option<(Value, usize, i64)> {
        let cur_ty = self.func.ty(cur);
        if !cur_ty.is_small_int() || cur_ty.bytes() > 4 || self.plan.elided[cur.index()] || self.uses[cur.index()] != 1 {
            return None;
        }
        let (vb, vpos, vinst) = self.pos_of[cur.index()]?;
        if vb != self.b_idx || vpos >= cur_pos {
            return None;
        }
        if let Some(inner) = self.uext_inner(vinst) {
            let inner_ty = self.func.ty(inner);
            return (inner_ty.is_small_int() && inner_ty.bytes() <= 4 && !self.plan.elided[inner.index()]).then_some((inner, vpos, cur_disp));
        }
        let (next, delta) = self.small_const_step(vinst, cur_ty)?;
        let next_disp = delta.checked_mul(1_i64 << shift).and_then(|d| cur_disp.checked_add(d))?;
        ((-FOLD_LIMIT..=FOLD_LIMIT).contains(&next_disp) && !self.plan.elided[next.index()]).then_some((next, vpos, next_disp))
    }

    pub(super) fn peel_small_chain(
        &self,
        inner: Value,
        start_pos: usize,
        (shift, disp): (u8, i64),
        multi_use: bool,
        elide: &mut Vec<(Value, usize)>,
    ) -> Option<(Value, i64, usize)> {
        let (mut cur, mut cur_disp, mut cur_pos) = (inner, disp, start_pos);
        let mut tmp = Vec::new();
        let mut best = self.uext_target_ok(cur, multi_use).then_some((cur, cur_disp, cur_pos, 0_usize));
        while tmp.len() < 16 {
            let Some((next, vpos, next_disp)) = self.peel_small_step(cur, cur_pos, (shift, cur_disp)) else { break };
            tmp.push((cur, vpos));
            (cur, cur_disp, cur_pos) = (next, next_disp, vpos);
            if self.uext_target_ok(cur, multi_use) {
                best = Some((cur, cur_disp, cur_pos, tmp.len()));
            }
        }
        let (idx, out_disp, out_pos, len) = best?;
        elide.extend_from_slice(&tmp[..len]);
        Some((idx, out_disp, out_pos))
    }

    fn local_uext(&self, v: Value, max_pos: usize) -> Option<(Value, usize)> {
        if self.plan.elided[v.index()] || self.uses[v.index()] != 1 {
            return None;
        }
        let (eb, epos, einst) = self.pos_of[v.index()]?;
        let inner = self.uext_inner(einst).filter(|_| eb == self.b_idx && epos < max_pos)?;
        (!self.plan.elided[inner.index()]).then_some((inner, epos))
    }

    pub(super) fn unwrap_base_uext(&self, v: Value, max_pos: usize, elide: &mut Vec<(Value, usize)>) -> (Value, usize) {
        match self.local_uext(v, max_pos) {
            Some((inner, epos)) if self.plan.zero_ext[inner.index()] && self.plan.reg_of(inner).is_some() => {
                elide.push((v, epos));
                (inner, epos)
            }
            _ => (v, max_pos),
        }
    }

    pub(super) fn unwrap_uext_const(
        &self,
        v: Value,
        max_pos: usize,
        (shift, disp): (u8, i64),
        multi_use: bool,
        elide: &mut Vec<(Value, usize)>,
    ) -> (Value, i64, usize) {
        if let Some((inner, epos)) = self.local_uext(v, max_pos)
            && self.func.ty(inner).is_small_int()
            && self.func.ty(inner).bytes() <= 4
        {
            let mut tmp = Vec::new();
            if let Some((idx, out_disp, out_pos)) = self.peel_small_chain(inner, epos, (shift, disp), multi_use, &mut tmp) {
                elide.push((v, epos));
                elide.extend(tmp);
                return (idx, out_disp, out_pos);
            }
        }
        (v, disp, max_pos)
    }

    pub(super) fn unwrap_uext(&self, v: Value, max_pos: usize, multi_use: bool, elide: &mut Vec<(Value, usize)>) -> (Value, usize) {
        let (idx, _, start) = self.unwrap_uext_const(v, max_pos, (0, 0), multi_use, elide);
        (idx, start)
    }

    pub(super) fn index_ok(&self, idx: Value) -> bool {
        !self.plan.elided[idx.index()] && !matches!(self.plan.location(idx), Location::Frame(..) | Location::Imm(..))
    }

    fn peel_scaled(&self, raw_idx: Value, spos: usize, shift: u8, disp: i64) -> (Value, i64, usize, Vec<(Value, usize)>) {
        let mut elide = Vec::new();
        if let Some((core, off, pos)) = self.peel_const(raw_idx, spos, true, &mut elide)
            && off != 0
            && let Some(total) = off.checked_mul(1_i64 << shift).and_then(|d| disp.checked_add(d))
            && (-FOLD_LIMIT..=FOLD_LIMIT).contains(&total)
        {
            return (core, total, pos, elide);
        }
        (raw_idx, disp, spos, Vec::new())
    }

    fn parse_index(&self, v: Value, sum_pos: usize, scaled_only: bool, multi_use: bool) -> Option<(Value, u8, i64, usize, Vec<(Value, usize)>)> {
        let mut elide = Vec::new();
        let (core, disp, use_pos) = self.peel_const(v, sum_pos, true, &mut elide)?;
        if !self.plan.elided[core.index()]
            && self.uses[core.index()] == 1
            && self.func.ty(core) == Type::I64
            && let Some((sb, spos, sinst)) = self.pos_of[core.index()]
            && sb == self.b_idx
            && spos < use_pos
            && let Some((raw_idx, shift, None)) = scaled_op(self.func, self.plan, sinst)
        {
            elide.push((core, spos));
            let (idx_core, total_disp, idx_use_pos, scaled_elide) = self.peel_scaled(raw_idx, spos, shift, disp);
            elide.extend(scaled_elide);
            let (idx, total_disp, idx_start) = self.unwrap_uext_const(idx_core, idx_use_pos, (shift, total_disp), multi_use, &mut elide);
            return self.index_ok(idx).then_some((idx, shift, total_disp, idx_start, elide));
        }
        if scaled_only {
            return None;
        }
        let (idx, total_disp, idx_start) = self.unwrap_uext_const(core, use_pos, (0, disp), multi_use, &mut elide);
        self.index_ok(idx).then_some((idx, 0, total_disp, idx_start, elide))
    }

    fn try_sib_pair(
        &self,
        elide: &[(Value, usize)],
        (sum_pos, outer_disp): (usize, i64),
        (b_side, i_side): (Value, Value),
        scaled_only: bool,
        multi_use: bool,
    ) -> Option<SibMatch> {
        let mut pair_elide = elide.to_vec();
        let (b_core, b_disp, base_start) = self.peel_const(b_side, sum_pos, true, &mut pair_elide)?;
        if self.plan.elided[b_core.index()] {
            return None;
        }
        let (b_core, base_start) = self.unwrap_base_uext(b_core, base_start, &mut pair_elide);
        let (base, frame_off) = match self.plan.location(b_core) {
            Location::Frame(slot, off) => (SibBase::Frame(slot), i64::from(off)),
            Location::Imm(_) => return None,
            _ => (SibBase::Value(b_core), 0),
        };
        let (index, shift, i_disp, idx_start, i_elide) = self.parse_index(i_side, sum_pos, scaled_only, multi_use)?;
        pair_elide.extend(i_elide);
        let total = outer_disp.checked_add(b_disp)?.checked_add(i_disp)?.checked_add(frame_off)?;
        (-FOLD_LIMIT..=FOLD_LIMIT).contains(&total).then_some(SibMatch {
            sib: Sib { base, index, shift, disp: total as i32 },
            elide: pair_elide,
            base_start,
            idx_start,
        })
    }

    fn sib_sum(&self, root: Value, root_pos: usize) -> Option<(Vec<(Value, usize)>, usize, i64, Value, Value)> {
        if self.plan.elided[root.index()] || self.plan.is_static(root) || self.func.ty(root) != Type::I64 {
            return None;
        }
        let mut elide = Vec::new();
        let (sum_node, outer_disp, _) = self.peel_const(root, root_pos, false, &mut elide)?;
        let (sb, sum_pos, sinst) = self.pos_of[sum_node.index()]?;
        if sb != self.b_idx || self.plan.elided[sum_node.index()] {
            return None;
        }
        if sum_node != root {
            if self.uses[sum_node.index()] != 1 || sum_pos >= root_pos {
                return None;
            }
            elide.push((sum_node, sum_pos));
        }
        let Op::Binary(Binary::Iadd, left, right) = self.func.insts[sinst.index()].op else { return None };
        Some((elide, sum_pos, outer_disp, left, right))
    }

    /// Each base-plus-index form of `root`, scaled-index forms first, in both operand orders.
    pub(super) fn sib_matches(&self, root: Value, root_pos: usize, multi_use: bool) -> impl Iterator<Item = SibMatch> + '_ {
        self.sib_sum(root, root_pos).into_iter().flat_map(move |(elide, sum_pos, outer_disp, left, right)| {
            [true, false]
                .into_iter()
                .flat_map(move |scaled_only| [(left, right), (right, left)].into_iter().map(move |pair| (scaled_only, pair)))
                .filter_map(move |(scaled_only, pair)| self.try_sib_pair(&elide, (sum_pos, outer_disp), pair, scaled_only, multi_use))
        })
    }

    pub(super) fn match_sib(&self, root: Value, root_pos: usize, multi_use: bool) -> Option<SibMatch> {
        self.sib_matches(root, root_pos, multi_use).next()
    }
}

fn addr_clobbers_flags(func: &Function, plan: &Plan<Reg>, addr: Value) -> bool {
    plan.elided[addr.index()] && plan.sib[addr.index()].is_some_and(|m| sib_clobbers_flags(func, plan, m))
}

fn operand_clobbers_flags(func: &Function, plan: &Plan<Reg>, defs: &[Option<Inst>], v: Value) -> bool {
    if !plan.elided[v.index()] {
        return false;
    }
    let Some(inst) = defs[v.index()] else { return false };
    matches!(func.insts[inst.index()].op, Op::Load(addr, _) if addr_clobbers_flags(func, plan, addr))
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
    if data.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()] || plan.is_static(r))
        && !matches!(data.op, Op::Call { .. } | Op::TryCall { .. })
    {
        return true;
    }
    let res_ok = data.result.is_some_and(|r| {
        let ty = func.ty(r);
        !ty.is_vector() && ty != Type::I128
    });
    match data.op {
        Op::Const(_) | Op::StackAddr(..) | Op::FramePointer | Op::Symbol(_) | Op::SlotAddr(_) => res_ok,
        Op::Load(addr, _) => res_ok && !addr_clobbers_flags(func, plan, addr),
        Op::StackLoad(..) => res_ok,
        Op::Store(val, addr, _) => !plan.elided[val.index()] && !func.ty(val).is_vector() && func.ty(val) != Type::I128 && !addr_clobbers_flags(func, plan, addr),
        Op::StackStore(val, ..) => !plan.elided[val.index()] && !func.ty(val).is_vector() && func.ty(val) != Type::I128,
        Op::Convert(Convert::Uextend | Convert::Sextend | Convert::Ireduce, src) => {
            res_ok && !func.ty(src).is_vector() && !operand_clobbers_flags(func, plan, defs, src)
        }
        Op::Binary(Binary::Iadd | Binary::Isub, ..) => data.result.is_some_and(|r| {
            func.ty(r) == Type::I64
                && !plan.fused[r.index()]
                && plan.sib[r.index()].is_some_and(|m| !sib_clobbers_flags(func, plan, m))
        }),
        Op::Binary(Binary::Imul, a, b) => data.result.is_some_and(|r| {
            matches!(func.ty(r), Type::I32 | Type::I64)
                && [(a, b), (b, a)].into_iter().any(|(x, c)| {
                    !operand_clobbers_flags(func, plan, defs, x)
                        && plan.const_of(c).is_some_and(|imm| matches!(imm, 0 | 1 | 3 | 5 | 9))
                })
        }),
        Op::Select(cond_v, x, y) => {
            (cond_v == cmp_res || matches!(roots[cond_v.index()], Some((leaf, None, _)) if leaf == cmp_res) || x == y || plan.const_of(cond_v).is_some())
                && !func.ty(x).is_vector()
                && !operand_clobbers_flags(func, plan, defs, x)
                && !operand_clobbers_flags(func, plan, defs, y)
        }
        _ => false,
    }
}

/// The signedness and factors of the 64-bit product whose high and low halves are `x` and `y` in either order.
pub(in super::super) fn product_halves(func: &Function, def: &dyn Fn(Value) -> Option<Inst>, (x, y): (Value, Value)) -> Option<(bool, Value, Value)> {
    let op = |v: Value| def(v).map(|inst| &func.insts[inst.index()].op);
    let ((Some(&Op::Binary(high @ (Binary::Umulhi | Binary::Smulhi), a, b)), Some(&Op::Binary(Binary::Imul, c, d)))
    | (Some(&Op::Binary(Binary::Imul, c, d)), Some(&Op::Binary(high @ (Binary::Umulhi | Binary::Smulhi), a, b)))) = (op(x), op(y))
    else {
        return None;
    };
    (func.ty(x) == Type::I64 && func.ty(y) == Type::I64 && ((a, b) == (c, d) || (a, b) == (d, c))).then_some((high == Binary::Smulhi, a, b))
}
