use crate::fold::defs;
use crate::ir::{Binary, Convert, Function, Inst, Op, Value};
use crate::live::results;

mod addr;
mod balance;
mod factor;
mod loads;
mod reuse;

fn count_operand_uses(func: &Function, inst: Inst, val: Value) -> u32 {
    let mut count = 0u32;
    func.insts[inst.index()]
        .op
        .for_operands(|v| count += u32::from(v == val));
    count
}

fn uses_operand(func: &Function, inst: Inst, val: Value) -> bool {
    count_operand_uses(func, inst, val) > 0
}

fn first_use_in_block(func: &Function, insts: &[Inst], pos: usize, val: Value) -> Option<usize> {
    insts[pos + 1..]
        .iter()
        .position(|&i| uses_operand(func, i, val))
        .map(|d| pos + 1 + d)
}

fn inst_latency(op: &Op) -> u32 {
    match op {
        Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_) | Op::StackAddr(..) | Op::FramePointer => 0,
        Op::Load(..)
        | Op::StackLoad(..)
        | Op::AtomicLoad(..)
        | Op::AtomicRmw(..)
        | Op::AtomicCas(..)
        | Op::Fma(..)
        | Op::Call { .. }
        | Op::TryCall { .. } => 3,
        Op::Binary(bin, ..)
            if matches!(
                bin,
                Binary::Imul
                    | Binary::Umulhi
                    | Binary::Smulhi
                    | Binary::Sdiv
                    | Binary::Udiv
                    | Binary::Srem
                    | Binary::Urem
                    | Binary::Fmul
                    | Binary::Fdiv
            ) =>
        {
            3
        }
        _ => 1,
    }
}

fn fill_block_depths(func: &Function, bi: usize, depth: &mut Vec<u32>) {
    depth.clear();
    depth.resize(func.types.len(), 0);
    for &inst in &func.blocks[bi].insts {
        let data = &func.insts[inst.index()];
        let lat = inst_latency(&data.op);
        if lat == 0 {
            continue;
        }
        let mut mx = 0u32;
        data.op.for_operands(|v| mx = mx.max(depth[v.index()]));
        for r in results(data) {
            depth[r.index()] = mx + lat;
        }
    }
}

fn add_use(uses: &mut Vec<u32>, val: Value, delta: i32) {
    if uses.len() <= val.index() {
        uses.resize(val.index() + 1, 0);
    }
    let next = (uses[val.index()] as i64 + i64::from(delta)).max(0);
    uses[val.index()] = next as u32;
}

fn set_bin_depth(depth: &mut [u32], val: Value, a: Value, b: Value) {
    if val.index() < depth.len() && a.index() < depth.len() && b.index() < depth.len() {
        depth[val.index()] = depth[a.index()].max(depth[b.index()]) + 1;
    }
}

fn is_pure_sinkable(func: &Function, op: &Op, res: Value) -> bool {
    let ty = func.ty(res);
    if ty.is_vector() {
        return false;
    }
    match *op {
        Op::Binary(bin, ..) => !matches!(
            bin,
            Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem
        ),
        Op::Unary(..) => true,
        Op::Convert(kind, x) => {
            !ty.is_float()
                && !func.ty(x).is_float()
                && !func.ty(x).is_vector()
                && matches!(
                    kind,
                    Convert::Uextend | Convert::Sextend | Convert::Ireduce | Convert::Bitcast
                )
        }
        _ => false,
    }
}

fn is_static_def(func: &Function, defs: &[Option<Inst>], val: Value) -> bool {
    let Some(inst) = defs.get(val.index()).copied().flatten() else {
        return false;
    };
    matches!(
        func.insts[inst.index()].op,
        Op::Const(_) | Op::StackAddr(..) | Op::Symbol(_) | Op::SlotAddr(_)
    )
}

fn is_fits_paired(func: &Function, a: Inst, b: Inst) -> bool {
    let (f, s) = (&func.insts[a.index()], &func.insts[b.index()]);
    matches!(
        (&f.op, &s.op, f.result),
        (Op::Binary(op, x, y), Op::Fits(chk, _, l, r), Some(sum))
            if op == chk && (x, y) == (l, r) && func.ty(sum).is_small_int()
    )
}

fn operands_live_at(func: &Function, defs: &[Option<Inst>], tail: &[Inst], op: &Op) -> bool {
    let mut ok = true;
    op.for_operands(|v| {
        if !is_static_def(func, defs, v) {
            ok &= tail.iter().any(|&i| uses_operand(func, i, v));
        }
    });
    ok
}

fn sink_pure_block(func: &mut Function, defs: &[Option<Inst>], uses: &[u32], bi: usize) {
    let len = func.blocks[bi].insts.len();
    if len < 32 {
        return;
    }
    for at in (0..len.saturating_sub(1)).rev() {
        let inst = func.blocks[bi].insts[at];
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else {
            continue;
        };
        if uses.get(res.index()).copied().unwrap_or(0) != 1
            || !is_pure_sinkable(func, &data.op, res)
        {
            continue;
        }
        let Some(u_idx) = first_use_in_block(func, &func.blocks[bi].insts, at, res) else {
            continue;
        };
        if u_idx <= at + 4
            || func.insts[func.blocks[bi].insts[u_idx].index()]
                .op
                .is_terminator()
        {
            continue;
        }
        let op = data.op.clone();
        if !operands_live_at(func, defs, &func.blocks[bi].insts[u_idx..], &op) {
            continue;
        }
        let moved = func.blocks[bi].insts.remove(at);
        func.blocks[bi].insts.insert(u_idx - 1, moved);
    }
}

/// Factors single-use bitwise identities, balances addition trees by latency, sinks deferred loads and pure scalar ops, and splits shared address sums.
pub(crate) fn run(func: &mut Function, balance: bool) {
    let (cached_defs, mut uses) = match factor::factor_bitwise(func, balance) {
        Some((d, u)) => (Some(d), u),
        None => (None, func.use_counts()),
    };
    let balanced = balance && balance::balance_add_trees(func, &mut uses);
    let uses = if balanced { func.use_counts() } else { uses };
    for bi in 0..func.blocks.len() {
        loads::sink_loads_block(func, &uses, bi);
    }
    let live_defs = if !balanced && let Some(d) = cached_defs {
        d
    } else {
        defs(func)
    };
    for bi in 0..func.blocks.len() {
        sink_pure_block(func, &live_defs, &uses, bi);
    }
    reuse::after_last_read(func, &uses, &live_defs);
    addr::split_address_sums(func);
}
