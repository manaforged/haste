#[path = "join_slots/coalesce.rs"]
mod coalesce;

use crate::ir::{Binary, Convert, Function, Inst, Op, StackSlot, Type, Value};
use coalesce::coalesce_block_params;

struct SelectEdit {
    sel: Inst,
    cond: Value,
    sa: StackSlot,
    sb: StackSlot,
    loads: Vec<(Inst, Value, i32, i32)>,
}

fn fold_slot_selects(func: &mut Function, addrs: &[Option<(StackSlot, i64)>]) -> bool {
    let mut uses = vec![0_u32; func.types.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        func.insts[inst.index()]
            .op
            .for_operands(|v| uses[v.index()] += 1);
    }
    let mut changed = false;
    for blk in 0..func.blocks.len() {
        let len = func.blocks[blk].insts.len();
        let mut edits = Vec::new();
        for pos in 0..len {
            let sel = func.blocks[blk].insts[pos];
            let data = &func.insts[sel.index()];
            let (Op::Select(cond, a, b), Some(r)) = (&data.op, data.result) else {
                continue;
            };
            if addrs[r.index()].is_some() || uses[r.index()] == 0 || func.ty(*cond).is_vector() {
                continue;
            }
            let (Some((sa, oa)), Some((sb, ob))) = (addrs[a.index()], addrs[b.index()]) else {
                continue;
            };
            let mut loads = Vec::new();
            let mut valid = true;
            for &later in &func.blocks[blk].insts[pos + 1..] {
                let ldata = &func.insts[later.index()];
                let Op::Load(addr, loff) = ldata.op else {
                    ldata.op.for_operands(|v| {
                        if v == r {
                            valid = false;
                        }
                    });
                    if !valid {
                        break;
                    }
                    continue;
                };
                if addr != r {
                    continue;
                }
                let Some(lres) = ldata.result else {
                    valid = false;
                    break;
                };
                let ty = func.ty(lres);
                if !(ty.is_small_int() || ty.is_float() || ty.is_vector() || ty == Type::I128) {
                    valid = false;
                    break;
                }
                let Some(off_a) = oa
                    .checked_add(i64::from(loff))
                    .and_then(|o| i32::try_from(o).ok())
                else {
                    valid = false;
                    break;
                };
                let Some(off_b) = ob
                    .checked_add(i64::from(loff))
                    .and_then(|o| i32::try_from(o).ok())
                else {
                    valid = false;
                    break;
                };
                let bytes = i64::from(ty.bytes());
                if off_a < 0
                    || i64::from(off_a) + bytes > i64::from(func.slots[sa.index()].size)
                    || off_b < 0
                    || i64::from(off_b) + bytes > i64::from(func.slots[sb.index()].size)
                {
                    valid = false;
                    break;
                }
                loads.push((later, lres, off_a, off_b));
            }
            if valid && loads.len() as u32 == uses[r.index()] {
                edits.push(SelectEdit {
                    sel,
                    cond: *cond,
                    sa,
                    sb,
                    loads,
                });
            }
        }
        if edits.is_empty() {
            continue;
        }
        changed = true;
        let mut drops = vec![false; func.insts.len()];
        let mut rewrites: Vec<Option<(Value, StackSlot, i32, StackSlot, i32, Value)>> =
            vec![None; func.insts.len()];
        for edit in edits {
            drops[edit.sel.index()] = true;
            for (inst, res, off_a, off_b) in edit.loads {
                rewrites[inst.index()] = Some((edit.cond, edit.sa, off_a, edit.sb, off_b, res));
            }
        }
        let insts = std::mem::take(&mut func.blocks[blk].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            if drops[inst.index()] {
                continue;
            }
            if let Some((cond, sa, off_a, sb, off_b, res)) = rewrites[inst.index()] {
                let ty = func.ty(res);
                let va = func.value(ty);
                let vb = func.value(ty);
                let ia = func.add(Op::StackLoad(sa, off_a), Some(va));
                let ib = func.add(Op::StackLoad(sb, off_b), Some(vb));
                func.insts[inst.index()].op = Op::Select(cond, va, vb);
                out.push(ia);
                out.push(ib);
            }
            out.push(inst);
        }
        func.blocks[blk].insts = out;
    }
    changed
}

enum AddrOp {
    Add(Value, Value, bool),
    Bitcast(Value),
    Select(Value, Value),
}

/// Resolves slot-relative SSA address values across pointer arithmetic, bitcasts, and equal selects.
pub(super) fn resolve_addrs(
    func: &Function,
    consts: &[Option<i64>],
) -> (Vec<Option<(StackSlot, i64)>>, Vec<(Inst, StackSlot)>) {
    let imm_of = |v: Value| consts.get(v.index()).copied().flatten();
    let (mut addrs, mut addr_insts) = func.slot_offsets(imm_of);
    let mut ops: Vec<(Inst, Value, AddrOp)> = Vec::new();
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        if addrs[res.index()].is_some() {
            continue;
        }
        match data.op {
            Op::Binary(Binary::Iadd, x, c) => ops.push((*inst, res, AddrOp::Add(x, c, false))),
            Op::Binary(Binary::Isub, x, c) => ops.push((*inst, res, AddrOp::Add(x, c, true))),
            Op::Convert(Convert::Bitcast, x)
                if func.ty(res) == Type::I64 && func.ty(x) == Type::I64 =>
            {
                ops.push((*inst, res, AddrOp::Bitcast(x)));
            }
            Op::Select(_, a, b) => ops.push((*inst, res, AddrOp::Select(a, b))),
            _ => {}
        }
    }
    loop {
        let before = ops.len();
        ops.retain(|(inst, res, op)| {
            let resolved = match *op {
                AddrOp::Add(x, c, sub) => {
                    let found = if sub {
                        addrs[x.index()].zip(imm_of(c).and_then(i64::checked_neg))
                    } else {
                        addrs[x.index()]
                            .zip(imm_of(c))
                            .or_else(|| addrs[c.index()].zip(imm_of(x)))
                    };
                    found.and_then(|((slot, base), imm)| Some((slot, base.checked_add(imm)?)))
                }
                AddrOp::Bitcast(x) => addrs[x.index()],
                AddrOp::Select(a, b) => addrs[a.index()].filter(|pa| addrs[b.index()] == Some(*pa)),
            };
            let Some((slot, offset)) = resolved else {
                return true;
            };
            addrs[res.index()] = Some((slot, offset));
            addr_insts.push((*inst, slot));
            false
        });
        if ops.len() == before {
            break;
        }
    }
    (addrs, addr_insts)
}

/// Returns which stack slots still have live address operands outside `dropped` and `addr_insts`.
pub(super) fn live_addr_slots(
    func: &Function,
    addrs: &[Option<(StackSlot, i64)>],
    addr_insts: &[(Inst, StackSlot)],
    dropped: &[bool],
) -> Vec<bool> {
    let mut is_addr = vec![false; func.insts.len()];
    for &(inst, _) in addr_insts {
        is_addr[inst.index()] = true;
    }
    let mut live = vec![false; func.slots.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            if dropped[inst.index()] || is_addr[inst.index()] {
                continue;
            }
            func.insts[inst.index()].op.for_operands(|v| {
                if let Some((slot, _)) = addrs.get(v.index()).copied().flatten() {
                    live[slot.index()] = true;
                }
            });
        }
    }
    live
}

/// Coalesces disjoint arm-local slots at block joins and folds slot-select loads.
pub(super) fn run(func: &mut Function) -> bool {
    let mut changed = false;
    loop {
        let consts = super::subword::eval_consts(func);
        let (mut addrs, addr_insts) = resolve_addrs(func, &consts);
        if addr_insts.is_empty() {
            break;
        }
        let folded = fold_slot_selects(func, &addrs);
        if folded {
            addrs = resolve_addrs(func, &consts).0;
        }
        if !(folded | coalesce_block_params(func, &addrs)) {
            break;
        }
        changed = true;
    }
    changed
}
