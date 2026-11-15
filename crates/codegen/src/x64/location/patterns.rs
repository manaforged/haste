use super::fuse::FuseCtx;
use super::super::asm::Reg;
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};
use crate::live::{Sib, SibBase};

pub(super) const FOLD_LIMIT: i64 = 1 << 24;

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
        && let Op::Convert(Convert::Uextend, inner) = func.insts[einst.index()].op
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
    let (invert, left, right) = match cc {
        IntCC::UnsignedLessThan => (false, a, Some(b)),
        IntCC::UnsignedGreaterThan if is_zero_const(func, plan, b) => (true, a, None),
        IntCC::UnsignedGreaterThan => (false, b, Some(a)),
        IntCC::UnsignedGreaterThanOrEqual => (true, a, Some(b)),
        IntCC::UnsignedLessThanOrEqual => (true, b, Some(a)),
        IntCC::Equal => (false, if is_zero_const(func, plan, b) { a } else { b }, None),
        IntCC::NotEqual => (true, if is_zero_const(func, plan, b) { a } else { b }, None),
        _ => return None,
    };
    Some((cand, inner, left, if cand == y { x } else { y }, right, invert))
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

    fn peel_const(&self, mut val: Value, mut max_pos: usize, mut peel_cur: bool, elide: &mut Vec<(Value, usize)>) -> Option<(Value, i64, usize)> {
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

    pub(super) fn unwrap_uext(&self, v: Value, max_pos: usize, multi_use: bool, elide: &mut Vec<(Value, usize)>) -> (Value, usize) {
        if !self.plan.elided[v.index()]
            && self.uses[v.index()] == 1
            && let Some((eb, epos, einst)) = self.pos_of[v.index()]
            && eb == self.b_idx
            && epos < max_pos
            && let Op::Convert(Convert::Uextend, inner) = self.func.insts[einst.index()].op
            && self.func.ty(inner).is_small_int()
            && self.func.ty(inner).bytes() <= 4
            && !self.plan.elided[inner.index()]
            && (!multi_use || self.plan.zero_ext[inner.index()] || self.plan.reg_of(inner).is_none())
        {
            elide.push((v, epos));
            return (inner, epos);
        }
        (v, max_pos)
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
            let (idx, idx_start) = self.unwrap_uext(idx_core, idx_use_pos, multi_use, &mut elide);
            return self.index_ok(idx).then_some((idx, shift, total_disp, idx_start, elide));
        }
        if scaled_only {
            return None;
        }
        let (idx, idx_start) = self.unwrap_uext(core, use_pos, multi_use, &mut elide);
        self.index_ok(idx).then_some((idx, 0, disp, idx_start, elide))
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

    pub(super) fn match_sib(&self, root: Value, root_pos: usize, multi_use: bool) -> Option<SibMatch> {
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
        [(left, right), (right, left)]
            .into_iter()
            .find_map(|pair| self.try_sib_pair(&elide, (sum_pos, outer_disp), pair, true, multi_use))
            .or_else(|| [(left, right), (right, left)].into_iter().find_map(|pair| self.try_sib_pair(&elide, (sum_pos, outer_disp), pair, false, multi_use)))
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
        Op::Store(val, addr, _) => !func.ty(val).is_vector() && func.ty(val) != Type::I128 && !addr_clobbers_flags(func, plan, addr),
        Op::StackStore(val, ..) => !func.ty(val).is_vector() && func.ty(val) != Type::I128,
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
