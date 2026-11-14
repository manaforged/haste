use super::fits_mem_imm;
use crate::ir::{Binary, Convert, Function, Inst, Op, Type, Unary, Value};
use crate::mem::{MemBase, MemLoc, Provenance, clobbers};

#[derive(Clone, Copy)]
enum RawLoad {
    Stack,
    Addr(Value, i32),
}

#[derive(Clone, Copy)]
struct Leaf {
    pos: usize,
    loc: MemLoc,
    raw: RawLoad,
    shift_byte: i64,
    bytes: i64,
}

struct PackPlan {
    wide_op: Op,
    wide_ty: Type,
    big_endian: bool,
    touched: Vec<Value>,
}

fn collect_leaves(
    func: &Function,
    prov: &Provenance,
    consts: &[Option<i64>],
    def_pos: &[Option<(usize, Inst)>],
    val: Value,
    (leaves, touched): (&mut Vec<Leaf>, &mut Vec<Value>),
) -> Option<()> {
    let (_, inst) = def_pos.get(val.index()).copied().flatten()?;
    match func.insts[inst.index()].op {
        Op::Binary(Binary::Bor | Binary::Iadd, a, b) => {
            touched.push(val);
            collect_leaves(func, prov, consts, def_pos, a, (leaves, touched))?;
            collect_leaves(func, prov, consts, def_pos, b, (leaves, touched))
        }
        _ => {
            let (base_val, shift_byte) = match func.insts[inst.index()].op {
                Op::Binary(Binary::Ishl, inner, amt) => {
                    touched.push(val);
                    let imm = consts.get(amt.index()).copied().flatten()?;
                    if !(0..64).contains(&imm) || imm % 8 != 0 {
                        return None;
                    }
                    (inner, imm / 8)
                }
                _ => (val, 0),
            };
            let load_val = match def_pos.get(base_val.index()).copied().flatten() {
                Some((_, linst)) => match func.insts[linst.index()].op {
                    Op::Convert(Convert::Uextend, inner) => {
                        touched.push(base_val);
                        inner
                    }
                    _ => base_val,
                },
                None => return None,
            };
            let (lpos, linst) = def_pos.get(load_val.index()).copied().flatten()?;
            let load_ty = func.ty(load_val);
            if !matches!(load_ty, Type::I8 | Type::I16) {
                return None;
            }
            let (loc, raw) = match func.insts[linst.index()].op {
                Op::Load(addr, off) => (prov.loc(addr, i64::from(off), load_ty.bytes()), RawLoad::Addr(addr, off)),
                Op::StackLoad(slot, off) => (MemLoc::slot(slot, i64::from(off), load_ty.bytes()), RawLoad::Stack),
                _ => return None,
            };
            touched.push(load_val);
            leaves.push(Leaf { pos: lpos, loc, raw, shift_byte, bytes: i64::from(load_ty.bytes()) });
            Some(())
        }
    }
}

fn build_pack(
    func: &Function,
    prov: &Provenance,
    consts: &[Option<i64>],
    def_pos: &[Option<(usize, Inst)>],
    block_insts: &[Inst],
    (root_pos, root_val, res_ty): (usize, Value, Type),
) -> Option<PackPlan> {
    let mut leaves = Vec::new();
    let mut touched = Vec::new();
    collect_leaves(func, prov, consts, def_pos, root_val, (&mut leaves, &mut touched))?;
    if leaves.len() < 2 || leaves.len() > 8 {
        return None;
    }
    let first = leaves[0];
    if leaves.iter().any(|l| l.loc.base != first.loc.base) {
        return None;
    }
    leaves.sort_by_key(|l| l.loc.offset);
    let min_off = leaves[0].loc.offset;
    let total_w: i64 = leaves.iter().map(|l| l.bytes).sum();
    if !matches!(total_w, 2 | 4 | 8) || total_w > i64::from(res_ty.bytes()) {
        return None;
    }
    let mut cursor = min_off;
    for l in &leaves {
        if l.loc.offset != cursor || l.shift_byte + l.bytes > total_w {
            return None;
        }
        cursor += l.bytes;
    }
    let little = leaves.iter().all(|l| l.shift_byte == l.loc.offset - min_off);
    let big = !little && leaves.iter().all(|l| l.shift_byte == total_w - (l.loc.offset - min_off) - l.bytes);
    if !little && !big {
        return None;
    }
    let wide_ty = match total_w {
        2 => Type::I16,
        4 => Type::I32,
        _ => Type::I64,
    };
    let wide_op = match (first.loc.base, leaves[0].raw) {
        (MemBase::Slot(slot), _) => Op::StackLoad(slot, i32::try_from(min_off).ok()?),
        (_, RawLoad::Addr(addr, off)) if prov.loc(addr, i64::from(off), 1).offset == min_off && fits_mem_imm(wide_ty, i64::from(off)) => {
            Op::Load(addr, off)
        }
        (MemBase::Addr(base), _) if def_pos.get(base.index()).copied().flatten().is_none_or(|(p, _)| p < root_pos) => {
            let off_i32 = i32::try_from(min_off).ok()?;
            if !fits_mem_imm(wide_ty, min_off) {
                return None;
            }
            Op::Load(base, off_i32)
        }
        _ => return None,
    };
    let min_pos = leaves.iter().map(|l| l.pos).min()?;
    let wide_loc = MemLoc::new(first.loc.root, first.loc.base, min_off, total_w as u32);
    for &inst in &block_insts[min_pos + 1..root_pos] {
        if clobbers(func, &func.insts[inst.index()].op, wide_loc, prov) {
            return None;
        }
    }
    Some(PackPlan { wide_op, wide_ty, big_endian: big, touched })
}

pub(super) fn coalesce_loads(func: &mut Function, prov: &Provenance) -> bool {
    let mut consts = vec![None; func.types.len()];
    for data in &func.insts {
        if let (&Op::Const(imm), Some(res)) = (&data.op, data.result) {
            consts[res.index()] = Some(imm);
        }
    }
    let mut uses = func.use_counts();
    let mut def_pos = vec![None; func.types.len()];
    let mut changed = false;
    for b_idx in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[b_idx].insts);
        let mut out = Vec::with_capacity(insts.len());
        let mut dead_pool = vec![false; func.types.len()];
        for (pos, &inst) in insts.iter().enumerate() {
            if let Some(res) = func.insts[inst.index()].result {
                def_pos[res.index()] = Some((pos, inst));
            }
        }
        let mut consumed = vec![false; func.types.len()];
        let mut plans: Vec<Option<(Value, Type, PackPlan)>> = (0..insts.len()).map(|_| None).collect();
        for (pos, &inst) in insts.iter().enumerate().rev() {
            let data = &func.insts[inst.index()];
            if let (Op::Binary(Binary::Bor | Binary::Iadd, ..), Some(res)) = (&data.op, data.result)
                && !consumed[res.index()]
            {
                let res_ty = func.ty(res);
                if matches!(res_ty, Type::I16 | Type::I32 | Type::I64)
                    && let Some(plan) = build_pack(func, prov, &consts, &def_pos, &insts, (pos, res, res_ty))
                {
                    for &v in &plan.touched {
                        if v != res {
                            consumed[v.index()] = true;
                        }
                    }
                    plans[pos] = Some((res, res_ty, plan));
                }
            }
        }
        for (pos, &inst) in insts.iter().enumerate() {
            if let Some((res, res_ty, plan)) = plans[pos].take() {
                func.insts[inst.index()].op.for_operands(|v| uses[v.index()] = uses[v.index()].saturating_sub(1));
                for v in plan.touched {
                    if v != res {
                        dead_pool[v.index()] = true;
                    }
                }
                if !plan.big_endian && plan.wide_ty == res_ty {
                    func.insts[inst.index()].op = plan.wide_op;
                } else {
                    let wide_val = func.value(plan.wide_ty);
                    uses.push(1);
                    def_pos.push(None);
                    dead_pool.push(false);
                    out.push(func.add(plan.wide_op, Some(wide_val)));
                    let swapped = if plan.big_endian && plan.wide_ty == res_ty {
                        Op::Unary(Unary::Bswap, wide_val)
                    } else if plan.big_endian {
                        let bswapped = func.value(plan.wide_ty);
                        uses.push(1);
                        def_pos.push(None);
                        dead_pool.push(false);
                        out.push(func.add(Op::Unary(Unary::Bswap, wide_val), Some(bswapped)));
                        Op::Convert(Convert::Uextend, bswapped)
                    } else {
                        Op::Convert(Convert::Uextend, wide_val)
                    };
                    func.insts[inst.index()].op = swapped;
                }
                changed = true;
            }
            out.push(inst);
        }
        let mut live = Vec::with_capacity(out.len());
        for &inst in out.iter().rev() {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result
                && dead_pool[res.index()]
                && uses[res.index()] == 0
                && data.op.is_removable()
            {
                data.op.for_operands(|v| uses[v.index()] = uses[v.index()].saturating_sub(1));
                continue;
            }
            live.push(inst);
        }
        live.reverse();
        for &inst in &insts {
            if let Some(res) = func.insts[inst.index()].result {
                def_pos[res.index()] = None;
            }
        }
        func.blocks[b_idx].insts = live;
    }
    changed
}
