use super::super::Site;
use super::super::patterns::{adc_fold, adc_parts, adc_sides, bextr_parts, shld_parts};
use super::super::sib::{double, find_cand_sib, sib_ok};
use super::{FuseCtx, preserved};
use crate::alloc::Plan;
use crate::ir::{Binary, Function, Inst, Op, Unary, Value};
use crate::live::{Sib, SibBase};
use crate::x64::asm::Reg;

pub(super) fn try_elide_andn_bnot(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Site]),
    (b_idx, pos): (usize, usize),
    (x, y): (Value, Value),
) {
    for (cand, other) in [(x, y), (y, x)] {
        if !plan
            .const_of(other)
            .is_some_and(|m| matches!(m, 0xFF | 0xFFFF | 0xFFFF_FFFF))
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
}

pub(super) fn fuse_shld(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Site]),
    skip_flags: &mut [bool],
    cand_shld: Vec<(usize, usize, Value, Value, Value)>,
) {
    let def_inst = |v: Value| pos_of[v.index()].map(|(_, _, inst)| inst);
    for (b_idx, pos, res, x, y) in cand_shld {
        if plan.elided[res.index()] || plan.sib[res.index()].is_some() {
            continue;
        }
        let Some((hi, lo, _, _, _)) = shld_parts(
            func,
            plan,
            func.ty(res),
            (x, y),
            |v| uses[v.index()] == 1 && !plan.elided[v.index()],
            def_inst,
        ) else {
            continue;
        };
        let (Some((bx, px, _)), Some((by, py, _))) = (pos_of[x.index()], pos_of[y.index()]) else {
            continue;
        };
        if bx != b_idx || by != b_idx || px >= pos || py >= pos {
            continue;
        }
        let skip = [px, py];
        let min_p = px.min(py);
        let ok = if hi == lo {
            preserved(func, plan, b_idx, min_p, pos, &skip, hi)
        } else {
            preserved(func, plan, b_idx, min_p, pos, &skip, hi)
                && preserved(func, plan, b_idx, min_p, pos, &skip, lo)
        };
        if ok {
            plan.elided[x.index()] = true;
            plan.elided[y.index()] = true;
            skip_flags[res.index()] = true;
        }
    }
}

pub(super) fn fuse_adc(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Site]),
    (tst_end, skip_flags): (&mut [Option<usize>], &mut [bool]),
    cand_adc: Vec<(usize, usize, Value, Binary, Value, Value)>,
) {
    let def_inst = |v: Value| pos_of[v.index()].map(|(_, _, inst)| inst);
    for (b_idx, pos, res, op, x, y) in cand_adc {
        if plan.elided[res.index()]
            || plan.sib[res.index()].is_some()
            || plan.elided[x.index()]
            || plan.elided[y.index()]
        {
            continue;
        }
        let Some((cand, inner, cc, a, b)) = adc_parts(
            func,
            plan,
            func.ty(res),
            op,
            (x, y),
            |v| uses[v.index()] == 1 && !plan.elided[v.index()],
            def_inst,
        ) else {
            continue;
        };
        let Some((cb, cpos, _)) = pos_of[cand.index()] else {
            continue;
        };
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
        if preserved(func, plan, b_idx, cmp_pos, pos, skip, a)
            && preserved(func, plan, b_idx, cmp_pos, pos, skip, b)
        {
            plan.elided[cand.index()] = true;
            if let Some(iv) = inner {
                plan.elided[iv.index()] = true;
            }
            tst_end[inner.unwrap_or(cand).index()] = Some(pos);
            skip_flags[res.index()] = true;
            let other = if cand == y { x } else { y };
            let quiet = [cmp_pos, cpos];
            if uses[other.index()] == 1
                && !plan.elided[other.index()]
                && let Some((invert, ..)) = adc_sides(func, plan, cc, a, b)
                && let Some((p, q)) = adc_fold(func, op, invert, other, def_inst)
                && let Some((ob, opos, _)) = pos_of[other.index()]
                && ob == b_idx
                && opos < pos
                && [p, q].into_iter().all(|v| {
                    !plan.elided[v.index()] && preserved(func, plan, b_idx, opos, pos, &quiet, v)
                })
            {
                plan.elided[other.index()] = true;
            }
        }
    }
}

pub(super) fn fuse_bextr(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Site]),
    cand_bextr: Vec<(usize, usize, Value, Binary, Value, Value)>,
) {
    let def_inst = |v: Value| pos_of[v.index()].map(|(_, _, inst)| inst);
    for (b_idx, pos, res, op, x, y) in cand_bextr {
        if plan.elided[res.index()] || plan.elided[x.index()] || plan.elided[y.index()] {
            continue;
        }
        let Some((Some(mid), src, _, _)) = bextr_parts(
            func,
            plan,
            func.ty(res),
            op,
            (x, y),
            |v| uses[v.index()] == 1 && !plan.elided[v.index()],
            def_inst,
        ) else {
            continue;
        };
        let Some((mb, mpos, _)) = pos_of[mid.index()] else {
            continue;
        };
        if mb == b_idx && mpos < pos && preserved(func, plan, b_idx, mpos, pos, &[], src) {
            plan.elided[mid.index()] = true;
        }
    }
}

pub(super) fn apply_cand_sibs(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of, defs): (&[u32], &[Site], &[Option<Inst>]),
    (ls_uses, const_kids): (
        &[Vec<(usize, usize, i32)>],
        &[Vec<(usize, usize, Value, i64)>],
    ),
    cand_sibs: &[(usize, usize, Value)],
) {
    for &(b_idx, pos, addr) in cand_sibs.iter().rev() {
        let ctx = FuseCtx {
            func,
            plan,
            uses,
            pos_of,
            defs,
            b_idx,
        };
        let Some((m, nodes)) = find_cand_sib(&ctx, (pos, addr), ls_uses, const_kids) else {
            continue;
        };
        plan.sib[addr.index()] = Some(m.sib);
        plan.elided[addr.index()] = true;
        for (kid, _, delta) in nodes {
            plan.sib[kid.index()] = Some(Sib {
                disp: (i64::from(m.sib.disp) + delta) as i32,
                ..m.sib
            });
            plan.elided[kid.index()] = true;
        }
        for (v, _) in m.elide {
            plan.elided[v.index()] = true;
        }
    }
}

pub(super) fn apply_lea_sibs(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of, defs): (&[u32], &[Site], &[Option<Inst>]),
    skip_flags: &[bool],
    cand_sibs: &[(usize, usize, Value)],
) {
    for &(b_idx, pos, res) in cand_sibs.iter().rev() {
        if plan.elided[res.index()] || uses[res.index()] == 0 || skip_flags[res.index()] {
            continue;
        }
        let ctx = FuseCtx {
            func,
            plan,
            uses,
            pos_of,
            defs,
            b_idx,
        };
        let Some(m) = ctx
            .sib_matches(res, pos, false)
            .find(|m| {
                (!m.elide.is_empty() || matches!(m.sib.base, SibBase::Frame(_)))
                    && sib_ok(&ctx, m, &[], pos)
            })
            .or_else(|| double(&ctx, res, pos).filter(|m| sib_ok(&ctx, m, &[], pos)))
        else {
            continue;
        };
        plan.sib[res.index()] = Some(m.sib);
        for (v, _) in m.elide {
            plan.elided[v.index()] = true;
        }
    }
}
