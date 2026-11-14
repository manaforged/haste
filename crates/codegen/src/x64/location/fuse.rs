use super::patterns::{adc_parts, bextr_parts, is_zero_const, shld_parts};
use super::super::asm::Reg;
use super::super::features::Features;
use super::super::frame::SAVED;
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};
use crate::live::{Sib, SibBase};

const FOLD_LIMIT: i64 = 1 << 24;

struct SibMatch {
    sib: Sib,
    elide: Vec<(Value, usize)>,
    base_start: usize,
    idx_start: usize,
}

struct FuseCtx<'a> {
    func: &'a Function,
    plan: &'a Plan<Reg>,
    uses: &'a [u32],
    pos_of: &'a [Option<(usize, usize, Inst)>],
    b_idx: usize,
}

fn preserved(func: &Function, plan: &Plan<Reg>, b_idx: usize, start: usize, end: usize, skip: &[usize], val: Value) -> bool {
    if plan.elided[val.index()] {
        return false;
    }
    if start + 1 >= end || plan.location(val).is_static() {
        return true;
    }
    if end - start > 128 {
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
    for pos in start + 1..end {
        if skip.contains(&pos) {
            continue;
        }
        let data = &func.insts[func.blocks[b_idx].insts[pos].index()];
        if data.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]) {
            continue;
        }
        if (reg.is_some() && matches!(data.op, Op::Call { .. } | Op::TryCall { .. }))
            || data.result.and_then(|r| plan.pairs[r.index()]).is_some_and(clobbers)
            || crate::live::results(data).any(clobbers)
        {
            return false;
        }
    }
    true
}

fn band_ok(func: &Function, plan: &Plan<Reg>, b_idx: usize, (mpos, end, skip): (usize, usize, &[usize]), (x, y): (Value, Value)) -> bool {
    func.ty(x).is_small_int() && preserved(func, plan, b_idx, mpos, end, skip, x) && preserved(func, plan, b_idx, mpos, end, skip, y)
}

fn zero_cmp_band(
    func: &Function,
    plan: &Plan<Reg>,
    uses: &[u32],
    pos_of: &[Option<(usize, usize, Inst)>],
    b_idx: usize,
    lhs: Value,
    rhs: Value,
) -> Option<(Value, usize, Value, Value)> {
    let masked = is_zero_const(func, plan, rhs).then_some(lhs).or_else(|| is_zero_const(func, plan, lhs).then_some(rhs))?;
    if uses[masked.index()] != 1 || plan.elided[masked.index()] {
        return None;
    }
    let (mb, mpos, minst) = pos_of[masked.index()]?;
    if mb != b_idx {
        return None;
    }
    let Op::Binary(Binary::Band, x, y) = func.insts[minst.index()].op else { return None };
    Some((masked, mpos, x, y))
}

impl FuseCtx<'_> {
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
            let (next, delta) = match self.func.insts[vinst.index()].op {
                Op::Binary(Binary::Iadd, a, b) => match (self.plan.const_of(a), self.plan.const_of(b)) {
                    (_, Some(c)) => (a, c),
                    (Some(c), None) => (b, c),
                    _ => break,
                },
                Op::Binary(Binary::Isub, a, b) => match self.plan.const_of(b).and_then(|c| c.checked_neg()) {
                    Some(c) => (a, c),
                    None => break,
                },
                _ => break,
            };
            let next_disp = disp.checked_add(delta)?;
            if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&next_disp) {
                break;
            }
            disp = next_disp;
            if peel_cur {
                elide.push((val, vpos));
            }
            last_pos = vpos;
            max_pos = vpos;
            val = next;
            peel_cur = true;
        }
        Some((val, disp, last_pos))
    }

    fn unwrap_uext(&self, v: Value, max_pos: usize, multi_use: bool, elide: &mut Vec<(Value, usize)>) -> (Value, usize) {
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

    fn parse_index(&self, v: Value, sum_pos: usize, scaled_only: bool, multi_use: bool) -> Option<(Value, u8, i64, usize, Vec<(Value, usize)>)> {
        let mut elide = Vec::new();
        let (core, disp, use_pos) = self.peel_const(v, sum_pos, true, &mut elide)?;
        if !self.plan.elided[core.index()]
            && self.uses[core.index()] == 1
            && self.func.ty(core) == Type::I64
            && let Some((sb, spos, sinst)) = self.pos_of[core.index()]
            && sb == self.b_idx
            && spos < use_pos
        {
            let scaled = match self.func.insts[sinst.index()].op {
                Op::Binary(Binary::Ishl, a, b) => self.plan.const_of(b).filter(|c| matches!(c, 1..=3)).map(|c| (a, c as u8)),
                Op::Binary(Binary::Imul, a, b) => [(a, b), (b, a)].into_iter().find_map(|(x, c)| {
                    self.plan.const_of(c).filter(|&k| matches!(k, 2 | 4 | 8)).map(|k| (x, (k as u64).trailing_zeros() as u8))
                }),
                _ => None,
            };
            if let Some((raw_idx, shift)) = scaled {
                elide.push((core, spos));
                let (idx, idx_start) = self.unwrap_uext(raw_idx, spos, multi_use, &mut elide);
                if !self.plan.elided[idx.index()] && !matches!(self.plan.location(idx), Location::Frame(..) | Location::Imm(..)) {
                    return Some((idx, shift, disp, idx_start, elide));
                }
                return None;
            }
        }
        if scaled_only {
            return None;
        }
        let (idx, idx_start) = self.unwrap_uext(core, use_pos, multi_use, &mut elide);
        if self.plan.elided[idx.index()] || matches!(self.plan.location(idx), Location::Frame(..) | Location::Imm(..)) {
            return None;
        }
        Some((idx, 0, disp, idx_start, elide))
    }

    fn match_sib(&self, root: Value, root_pos: usize, multi_use: bool) -> Option<SibMatch> {
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
        let try_pair = |b_side: Value, i_side: Value, scaled_only: bool| -> Option<SibMatch> {
            let mut pair_elide = elide.clone();
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
            if !(-FOLD_LIMIT..=FOLD_LIMIT).contains(&total) {
                return None;
            }
            Some(SibMatch { sib: Sib { base, index, shift, disp: total as i32 }, elide: pair_elide, base_start, idx_start })
        };
        [(left, right), (right, left)]
            .into_iter()
            .find_map(|(b, i)| try_pair(b, i, true))
            .or_else(|| [(left, right), (right, left)].into_iter().find_map(|(b, i)| try_pair(b, i, false)))
    }
}

fn sib_ok(func: &Function, plan: &Plan<Reg>, b_idx: usize, m: &SibMatch, extra_skip: Option<usize>, end_pos: usize) -> bool {
    let mut skip: Vec<usize> = m.elide.iter().map(|&(_, p)| p).collect();
    if let Some(extra) = extra_skip {
        skip.push(extra);
    }
    let base_ok = match m.sib.base {
        SibBase::Frame(_) => true,
        SibBase::Value(base) => preserved(func, plan, b_idx, m.base_start, end_pos, &skip, base),
    };
    base_ok && preserved(func, plan, b_idx, m.idx_start, end_pos, &skip, m.sib.index)
}

fn may_barrier_mem(op: &Op) -> bool {
    matches!(
        op,
        Op::Store(..)
            | Op::StackStore(..)
            | Op::Copy(..)
            | Op::Move(..)
            | Op::Fence
            | Op::Call { .. }
            | Op::TryCall { .. }
            | Op::AtomicLoad(..)
            | Op::AtomicStore(..)
            | Op::AtomicRmw(..)
            | Op::AtomicCas(..)
            | Op::Trap
            | Op::Trapz(..)
            | Op::Binary(Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem, ..)
    )
}

fn load_dst_clobber(func: &Function, plan: &Plan<Reg>, linst: Inst, load_res: Value, u_res: Option<Value>, op: Binary, (a, b): (Value, Value)) -> bool {
    let Some(r_dst) = u_res.and_then(|r| plan.reg_of(r)) else { return false };
    if op == Binary::Isub && a == load_res {
        return false;
    }
    let other = if a == load_res { b } else { a };
    if plan.reg_of(other) == Some(r_dst) {
        return false;
    }
    let Op::Load(addr, _) = func.insts[linst.index()].op else { return false };
    if plan.elided[addr.index()] {
        plan.sib[addr.index()].is_some_and(|m| matches!(m.base, SibBase::Value(base) if plan.reg_of(base) == Some(r_dst)) || plan.reg_of(m.index) == Some(r_dst))
    } else {
        plan.reg_of(addr) == Some(r_dst)
    }
}

pub(in super::super) fn fuse(func: &Function, plan: &mut Plan<Reg>, features: &Features) {
    if func.size() > 8192 {
        return;
    }
    let n = func.types.len();
    let (mut uses, mut pos_of, mut ls_uses, mut use_of) = (vec![0_u32; n], vec![None; n], vec![Vec::new(); n], vec![None; n]);
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| {
                uses[v.index()] += 1;
                use_of[v.index()] = Some((b_idx, pos, inst));
            });
            match data.op {
                Op::Load(addr, off) => ls_uses[addr.index()].push((b_idx, pos, off)),
                Op::Store(val, addr, off) if val != addr => ls_uses[addr.index()].push((b_idx, pos, off)),
                _ => {}
            }
            if let Some(res) = data.result {
                pos_of[res.index()] = Some((b_idx, pos, inst));
            }
        }
    }
    let (mut branch_tst, mut cmp_tst, mut cand_sibs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut cand_shld, mut cand_adc, mut cand_bextr, mut cand_loads, mut cand_alu) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else {
                if let Op::Brif(cond_val, ..) | Op::Trapz(cond_val) = data.op
                    && uses[cond_val.index()] == 1
                    && !plan.elided[cond_val.index()]
                    && let Some((cb, cpos, cinst)) = pos_of[cond_val.index()]
                    && cb == b_idx
                    && cpos < pos
                {
                    match func.insts[cinst.index()].op {
                        Op::Binary(Binary::Band, x, y) if band_ok(func, plan, b_idx, (cpos, pos, &[]), (x, y)) => {
                            branch_tst.push((pos, cond_val, None));
                        }
                        Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                            if let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, &uses, &pos_of, b_idx, lhs, rhs)
                                && mpos < cpos
                                && band_ok(func, plan, b_idx, (mpos, pos, &[cpos]), (x, y))
                            {
                                branch_tst.push((pos, cond_val, Some(m)));
                            }
                        }
                        _ => {}
                    }
                }
                continue;
            };
            let ty = func.ty(res);
            match data.op {
                Op::Binary(Binary::Band, x, y) if features.bmi1 && !plan.elided[res.index()] && ty.is_small_int() => {
                    for (cand, other) in [(x, y), (y, x)] {
                        if !plan.const_of(other).is_some_and(|m| matches!(m, 0xFF | 0xFFFF | 0xFFFF_FFFF))
                            && uses[cand.index()] == 1
                            && !plan.elided[cand.index()]
                            && let Some((cb, cpos, cinst)) = pos_of[cand.index()]
                            && cb == b_idx
                            && cpos < pos
                            && let Op::Unary(Unary::Bnot, inner) = func.insts[cinst.index()].op
                            && preserved(func, plan, b_idx, cpos, pos, &[], inner)
                        {
                            plan.elided[cand.index()] = true;
                            break;
                        }
                    }
                    if matches!(ty, Type::I32 | Type::I64) {
                        cand_bextr.push((b_idx, pos, res, Binary::Band, x, y));
                        cand_alu.push((b_idx, pos, res, Binary::Band, x, y));
                    }
                }
                Op::Binary(op @ (Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Bxor), x, y) if !plan.elided[res.index()] => {
                    if matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd) && matches!(ty, Type::I32 | Type::I64) {
                        cand_shld.push((b_idx, pos, res, x, y));
                    }
                    if matches!(op, Binary::Iadd | Binary::Isub) && matches!(ty, Type::I32 | Type::I64) {
                        cand_adc.push((b_idx, pos, res, op, x, y));
                    }
                    if matches!(op, Binary::Iadd | Binary::Isub) && plan.sib[res.index()].is_none() && ty == Type::I64 {
                        cand_sibs.push((b_idx, pos, res));
                    }
                    if matches!(ty, Type::I32 | Type::I64) {
                        cand_alu.push((b_idx, pos, res, op, x, y));
                    }
                }
                Op::Binary(op @ (Binary::Band | Binary::Ushr), x, y) if !plan.elided[res.index()] && matches!(ty, Type::I32 | Type::I64) => {
                    if features.bmi1 {
                        cand_bextr.push((b_idx, pos, res, op, x, y));
                    }
                    if op == Binary::Band {
                        cand_alu.push((b_idx, pos, res, op, x, y));
                    }
                }
                Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) if !plan.elided[res.index()] => {
                    cmp_tst.push((b_idx, pos, res, lhs, rhs));
                }
                Op::Load(..) | Op::StackLoad(..) if uses[res.index()] == 1 && !plan.elided[res.index()] && ty.is_small_int() => {
                    cand_loads.push((b_idx, pos, res));
                }
                _ => {}
            }
        }
    }
    let mut tst_end = vec![None; n];
    for (bpos, cond_val, m) in branch_tst {
        plan.elided[cond_val.index()] = true;
        tst_end[cond_val.index()] = Some(bpos);
        if let Some(m) = m {
            plan.elided[m.index()] = true;
            tst_end[m.index()] = Some(bpos);
        }
    }
    for (b_idx, cpos, res, lhs, rhs) in cmp_tst {
        if !plan.elided[res.index()]
            && let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, &uses, &pos_of, b_idx, lhs, rhs)
            && mpos < cpos
            && band_ok(func, plan, b_idx, (mpos, cpos, &[]), (x, y))
        {
            plan.elided[m.index()] = true;
            tst_end[m.index()] = Some(cpos);
        }
    }
    let mut skip_flags = vec![false; n];
    let def_inst = |v: Value| pos_of[v.index()].map(|(_, _, inst)| inst);
    for (b_idx, pos, res, x, y) in cand_shld {
        if plan.elided[res.index()] || plan.sib[res.index()].is_some() {
            continue;
        }
        let Some((hi, lo, _, _, _)) = shld_parts(func, plan, func.ty(res), (x, y), |v| uses[v.index()] == 1 && !plan.elided[v.index()], def_inst) else { continue };
        let (Some((bx, px, _)), Some((by, py, _))) = (pos_of[x.index()], pos_of[y.index()]) else { continue };
        if bx != b_idx || by != b_idx || px >= pos || py >= pos {
            continue;
        }
        let skip = [px, py];
        let min_p = px.min(py);
        let ok = if hi == lo {
            preserved(func, plan, b_idx, min_p, pos, &skip, hi)
        } else {
            preserved(func, plan, b_idx, min_p, pos, &skip, hi) && preserved(func, plan, b_idx, min_p, pos, &skip, lo)
        };
        if ok {
            plan.elided[x.index()] = true;
            plan.elided[y.index()] = true;
            skip_flags[res.index()] = true;
        }
    }
    for (b_idx, pos, res, op, x, y) in cand_adc {
        if plan.elided[res.index()] || plan.sib[res.index()].is_some() || plan.elided[x.index()] || plan.elided[y.index()] {
            continue;
        }
        let Some((cand, inner, _, a, b)) = adc_parts(func, plan, func.ty(res), op, (x, y), |v| uses[v.index()] == 1 && !plan.elided[v.index()], def_inst) else { continue };
        let Some((cb, cpos, _)) = pos_of[cand.index()] else { continue };
        if cb != b_idx || cpos >= pos {
            continue;
        }
        let (cmp_pos, skip): (usize, &[usize]) = match inner {
            Some(iv) => match pos_of[iv.index()] {
                Some((ib, ipos, _)) if ib == b_idx && ipos < cpos => (ipos, &[cpos]),
                _ => continue,
            },
            None => (cpos, &[]),
        };
        if preserved(func, plan, b_idx, cmp_pos, pos, skip, a) && preserved(func, plan, b_idx, cmp_pos, pos, skip, b) {
            plan.elided[cand.index()] = true;
            if let Some(iv) = inner {
                plan.elided[iv.index()] = true;
            }
            skip_flags[res.index()] = true;
        }
    }
    for (b_idx, pos, res, op, x, y) in cand_bextr {
        if plan.elided[res.index()] || plan.elided[x.index()] || plan.elided[y.index()] {
            continue;
        }
        let Some((Some(mid), src, _, _)) = bextr_parts(func, plan, func.ty(res), op, (x, y), |v| uses[v.index()] == 1 && !plan.elided[v.index()], def_inst) else { continue };
        let Some((mb, mpos, _)) = pos_of[mid.index()] else { continue };
        if mb == b_idx && mpos < pos && preserved(func, plan, b_idx, mpos, pos, &[], src) {
            plan.elided[mid.index()] = true;
        }
    }
    for &(b_idx, pos, addr) in cand_sibs.iter().rev() {
        let addr_uses = &ls_uses[addr.index()];
        if plan.elided[addr.index()] || addr_uses.is_empty() || addr_uses.len() != uses[addr.index()] as usize {
            continue;
        }
        let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, b_idx };
        let Some(m) = ctx.match_sib(addr, pos, addr_uses.len() > 1) else { continue };
        let fits = addr_uses.iter().all(|&(ub, upos, off)| {
            ub == b_idx && upos > pos && (-FOLD_LIMIT..=FOLD_LIMIT).contains(&(i64::from(m.sib.disp) + i64::from(off)))
        });
        let Some(end_pos) = fits.then(|| addr_uses.iter().map(|&(_, p, _)| p).max()).flatten() else { continue };
        if sib_ok(func, plan, b_idx, &m, Some(pos), end_pos) {
            plan.sib[addr.index()] = Some(m.sib);
            plan.elided[addr.index()] = true;
            for (v, _) in m.elide {
                plan.elided[v.index()] = true;
            }
        }
    }
    for &(b_idx, pos, res) in cand_sibs.iter().rev() {
        if plan.elided[res.index()] || uses[res.index()] == 0 || skip_flags[res.index()] {
            continue;
        }
        let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, b_idx };
        let Some(m) = ctx.match_sib(res, pos, false) else { continue };
        if m.elide.is_empty() && !matches!(m.sib.base, SibBase::Frame(_)) {
            continue;
        }
        if sib_ok(func, plan, b_idx, &m, None, pos) {
            plan.sib[res.index()] = Some(m.sib);
            for (v, _) in m.elide {
                plan.elided[v.index()] = true;
            }
        }
    }
    for (b_idx, lpos, load_res) in cand_loads {
        let Some((ub, upos, uinst)) = use_of[load_res.index()] else { continue };
        if ub != b_idx || upos <= lpos {
            continue;
        }
        let udata = &func.insts[uinst.index()];
        let ty = func.ty(load_res);
        let eff_pos = if let Some(ur) = udata.result.filter(|r| plan.elided[r.index()]) {
            let Some(fpos) = tst_end[ur.index()] else { continue };
            fpos
        } else {
            let ok = match udata.op {
                Op::Icmp(..) => true,
                Op::Binary(op @ (Binary::Iadd | Binary::Isub | Binary::Band | Binary::Bor | Binary::Bxor | Binary::Imul), a, b) => {
                    matches!(ty, Type::I32 | Type::I64)
                        && udata.result.is_some_and(|r| plan.sib[r.index()].is_none())
                        && !plan.elided[a.index()]
                        && !plan.elided[b.index()]
                        && !load_dst_clobber(func, plan, func.blocks[b_idx].insts[lpos], load_res, udata.result, op, (a, b))
                }
                Op::Binary(Binary::Rotl | Binary::Rotr, x, cnt) => {
                    features.bmi2
                        && x == load_res
                        && matches!(ty, Type::I32 | Type::I64)
                        && plan.const_of(cnt).is_some_and(|c| (c & i64::from(ty.bits() - 1)) != 0)
                }
                Op::Binary(Binary::Ishl | Binary::Ushr | Binary::Sshr, x, cnt) => {
                    features.bmi2 && x == load_res && matches!(ty, Type::I32 | Type::I64) && plan.const_of(cnt).is_none()
                }
                Op::Unary(uop @ (Unary::Clz | Unary::Ctz | Unary::Popcnt), _) => {
                    matches!(ty, Type::I32 | Type::I64)
                        && match uop {
                            Unary::Clz => features.lzcnt,
                            Unary::Ctz => features.bmi1,
                            _ => features.popcnt,
                        }
                }
                _ => false,
            };
            if !ok {
                continue;
            }
            upos
        };
        if eff_pos <= lpos
            || func.blocks[b_idx].insts[lpos + 1..eff_pos]
                .iter()
                .any(|&mid| may_barrier_mem(&func.insts[mid.index()].op))
        {
            continue;
        }
        let addr_ok = match func.insts[func.blocks[b_idx].insts[lpos].index()].op {
            Op::StackLoad(..) => true,
            Op::Load(addr, _) => {
                if plan.elided[addr.index()] {
                    plan.sib[addr.index()].is_some_and(|m| {
                        let base_ok = match m.base {
                            SibBase::Frame(_) => true,
                            SibBase::Value(b) => preserved(func, plan, b_idx, lpos, eff_pos, &[], b),
                        };
                        base_ok && preserved(func, plan, b_idx, lpos, eff_pos, &[], m.index)
                    })
                } else {
                    preserved(func, plan, b_idx, lpos, eff_pos, &[], addr)
                }
            }
            _ => false,
        };
        if addr_ok {
            plan.elided[load_res.index()] = true;
        }
    }
    for (b_idx, pos, res, op, x, y) in cand_alu {
        if plan.elided[res.index()] || plan.deferred[res.index()] || plan.fused[res.index()] || plan.sib[res.index()].is_some() || skip_flags[res.index()] {
            continue;
        }
        if op == Binary::Band && [x, y].into_iter().any(|v| plan.const_of(v).is_some_and(|m| matches!(m, 0xFF | 0xFFFF | 0xFFFF_FFFF))) {
            continue;
        }
        let shrink_safe = matches!(plan.location(res), Location::Reg(r) if !SAVED.contains(&r) && !matches!(r, Reg::Rbp | Reg::Rsp | Reg::R11))
            || (func.entry != Some(Block(b_idx as u32))
                && !matches!(func.blocks[b_idx].insts.last().map(|i| &func.insts[i.index()].op), Some(Op::Return(_))));
        if !shrink_safe {
            continue;
        }
        let insts = &func.blocks[b_idx].insts;
        for &next_inst in &insts[pos + 1..] {
            let nd = &func.insts[next_inst.index()];
            if nd.result.is_some_and(|r| plan.elided[r.index()]) {
                continue;
            }
            let flagless = nd.result.is_some_and(|r| !func.ty(r).is_vector())
                && matches!(nd.op, Op::Const(_) | Op::StackAddr(..) | Op::FramePointer | Op::Symbol(_) | Op::SlotAddr(_));
            if flagless {
                continue;
            }
            let zf_match = match nd.op {
                Op::Brif(c, ..) | Op::Trapz(c) => c == res,
                Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                    (lhs == res && is_zero_const(func, plan, rhs)) || (rhs == res && is_zero_const(func, plan, lhs))
                }
                _ => false,
            };
            if zf_match {
                plan.fused[res.index()] = true;
            }
            break;
        }
    }
}
