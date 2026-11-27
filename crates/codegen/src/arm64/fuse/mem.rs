mod sib;
mod vec;

use self::sib::parse_sib_base_op;
pub(super) use self::sib::try_fuse_sib;
pub(in crate::arm64) use self::vec::encode_neon_movi;
pub(in crate::arm64) use self::vec::pairwise_binary_parts;
pub(super) use self::vec::{const_splat, try_fuse_vec_op};
use super::super::asm::Reg;
use super::{elided_use_end, local_single, preserved};
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};
use crate::live::SibBase;

fn trivial_mul_imm(_func: &Function, plan: &Plan<Reg>, ty: Type, val: Value) -> bool {
    plan.const_of(val)
        .is_some_and(|raw| crate::alloc::consts::arm64_trivial_mul_imm(ty, raw))
}

pub(super) fn try_fuse_madd(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) -> bool {
    if !matches!(alu_op, Binary::Iadd | Binary::Isub) {
        return false;
    }
    let ty = func.ty(res);
    let candidates = if alu_op == Binary::Isub {
        [Some((y, x)), None]
    } else {
        [Some((y, x)), Some((x, y))]
    };
    for (mul, acc) in candidates.into_iter().flatten() {
        if plan.elided[acc.index()] {
            continue;
        }
        let Some((mpos, minst)) = local_single(plan, info, b_idx, pos, mul) else {
            continue;
        };
        let Op::Binary(Binary::Imul, a, b) = func.insts[minst.index()].op else {
            continue;
        };
        if trivial_mul_imm(func, plan, ty, a) || trivial_mul_imm(func, plan, ty, b) {
            continue;
        }
        plan.elided[mul.index()] = true;
        let ok = preserved(func, plan, b_idx, mpos, pos, None, a)
            && preserved(func, plan, b_idx, mpos, pos, None, b);
        if ok {
            return true;
        }
        plan.elided[mul.index()] = false;
    }
    false
}

/// Returns `(sub, a, b, acc)` when `x` and `y` under `op` emit `madd`/`msub` with an elided `Imul`.
pub(in crate::arm64) fn madd_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    x: Value,
    y: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(bool, Value, Value, Value)> {
    if !matches!(op, Binary::Iadd | Binary::Isub) || !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let sub = op == Binary::Isub;
    let candidates = if sub {
        [Some((y, x)), None]
    } else {
        [Some((y, x)), Some((x, y))]
    };
    for (mul, acc) in candidates.into_iter().flatten() {
        if !plan.elided[mul.index()] || plan.elided[acc.index()] {
            continue;
        }
        let Some(inst) = def(mul) else {
            continue;
        };
        if let Op::Binary(Binary::Imul, a, b) = func.insts[inst.index()].op
            && !plan.elided[a.index()]
            && !plan.elided[b.index()]
            && !trivial_mul_imm(func, plan, ty, a)
            && !trivial_mul_imm(func, plan, ty, b)
        {
            return Some((sub, a, b, acc));
        }
    }
    None
}

fn sload_addr_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    (b_idx, lpos, cpos): (usize, usize, usize),
    op: &Op,
) -> bool {
    let Op::Load(addr, _) = *op else {
        return matches!(op, Op::StackLoad(..));
    };
    let Some(sib) = plan.sib[addr.index()] else {
        return !plan.elided[addr.index()] && preserved(func, plan, b_idx, lpos, cpos, None, addr);
    };
    let base_ok = match sib.base {
        SibBase::Frame(_) => true,
        SibBase::Value(b) => preserved(func, plan, b_idx, lpos, cpos, None, b),
    };
    base_ok && preserved(func, plan, b_idx, lpos, cpos, None, sib.index)
}

fn can_elide_sload(
    func: &Function,
    plan: &Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, cpos): (usize, usize),
    val: Value,
) -> bool {
    if !matches!(plan.location(val), Location::Reg(_) | Location::Slot) {
        return false;
    }
    let Some((lpos, linst)) = local_single(plan, info, b_idx, cpos, val) else {
        return false;
    };
    let op = &func.insts[linst.index()].op;
    sload_addr_preserved(func, plan, (b_idx, lpos, cpos), op)
        && !func.blocks[b_idx].insts[lpos + 1..cpos]
            .iter()
            .any(|&i| func.insts[i.index()].op.is_load_barrier())
}

pub(super) fn try_fuse_sload_extend(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    x: Value,
) {
    if matches!(func.ty(x), Type::I8 | Type::I16 | Type::I32)
        && matches!(func.ty(res), Type::I16 | Type::I32 | Type::I64)
        && can_elide_sload(func, plan, info, (b_idx, pos), x)
    {
        plan.elided[x.index()] = true;
    }
}

fn icmp_sload_other_ok(plan: &Plan<Reg>, ty: Type, other: Value) -> bool {
    if plan.elided[other.index()] {
        return false;
    }
    if let Some(imm) = plan.const_of(other) {
        return super::super::int::add_imm_parts(true, crate::fold::extend(imm, ty, true))
            .is_some();
    }
    matches!(plan.location(other), Location::Reg(r) if !matches!(r.0, 9 | 10 | 16 | 17 | 31))
}

pub(super) fn try_fuse_icmp_sload(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos): (usize, usize),
    (cc, a, b): (IntCC, Value, Value),
) {
    let ty = func.ty(a);
    if !cc.is_signed() || !matches!(ty, Type::I8 | Type::I16) {
        return;
    }
    for (load, other) in [(a, b), (b, a)] {
        if icmp_sload_other_ok(plan, ty, other)
            && can_elide_sload(func, plan, info, (b_idx, pos), load)
        {
            plan.elided[load.index()] = true;
            return;
        }
    }
}

fn narrow_store_ok(
    func: &Function,
    plan: &Plan<Reg>,
    uses: &[u32],
    (b_idx, pos, res): (usize, usize, Value),
    inner: Value,
) -> bool {
    if uses[res.index()] != 1
        || plan.elided[res.index()]
        || plan.deferred[res.index()]
        || !func.ty(res).is_small_int()
        || !func.ty(inner).is_small_int()
        || !matches!(plan.location(inner), Location::Reg(_) | Location::Slot)
    {
        return false;
    }
    let insts = &func.blocks[b_idx].insts;
    let store = insts
        .iter()
        .enumerate()
        .skip(pos + 1)
        .find_map(|(at, &inst)| {
            matches!(
                func.insts[inst.index()].op,
                Op::Store(val, addr, _) | Op::AtomicStore(val, addr, _) if val == res && addr != res
            )
            .then_some(at)
        });
    store.is_some_and(|at| {
        elided_use_end(func, plan, insts, at, inner).is_some()
            && preserved(func, plan, b_idx, pos, at, None, inner)
    })
}

pub(super) fn fuse_atomic_addrs(
    func: &Function,
    plan: &mut Plan<Reg>,
    rcpc2: bool,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
) {
    let mut info: Vec<(u32, usize, usize, u8)> = vec![(0, usize::MAX, 0, 0); func.types.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            if let Some((addr, mask)) =
                crate::alloc::rcpc2_atomic_site(func, &func.insts[inst.index()], rcpc2)
            {
                let entry = &mut info[addr.index()];
                if entry.0 == 0 {
                    *entry = (1, b_idx, pos, mask);
                } else if entry.1 == b_idx {
                    entry.0 += 1;
                    entry.2 = entry.2.max(pos);
                    entry.3 |= mask;
                } else {
                    entry.0 = u32::MAX;
                }
            }
        }
    }
    for (v, def) in pos_of.iter().enumerate() {
        let Some((block, start, inst)) = *def else {
            continue;
        };
        let (count, use_block, max_end, mask) = info[v];
        if count == 0 || count != uses[v] || use_block != block || max_end <= start {
            continue;
        }
        let val = Value(v as u32);
        if func.ty(val) != Type::I64
            || plan.elided[v]
            || plan.is_static(val)
            || plan.sib[v].is_some()
        {
            continue;
        }
        let Some((base, delta)) =
            parse_sib_base_op(func, plan, pos_of, &func.insts[inst.index()].op)
        else {
            continue;
        };
        if plan.elided[base.index()] || plan.sib[base.index()].is_some() {
            continue;
        }
        if count > 1
            && plan.reg_of(base).is_none()
            && !matches!(plan.location(base), Location::Frame(..))
        {
            continue;
        }
        if delta == 0 || !crate::alloc::rcpc2_delta_fits(i64::from(delta), mask) {
            continue;
        }
        if preserved(func, plan, block, start, max_end, None, base) {
            plan.elided[v] = true;
        }
    }
}

pub(super) fn fuse_store_ireduce(func: &Function, plan: &mut Plan<Reg>, uses: &[u32]) {
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            if let (&Op::Convert(Convert::Ireduce, inner), Some(res)) = (&data.op, data.result)
                && narrow_store_ok(func, plan, uses, (b_idx, pos, res), inner)
            {
                plan.elided[res.index()] = true;
            }
        }
    }
}

/// The value whose register a store of `val` reads: the source of an elided `Ireduce`, else `val`.
pub(in crate::arm64) fn store_source(
    func: &Function,
    plan: &Plan<Reg>,
    val: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Value {
    match def(val).map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Convert(Convert::Ireduce, inner)) if plan.elided[val.index()] => inner,
        _ => val,
    }
}
