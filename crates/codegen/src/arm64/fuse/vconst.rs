use super::super::asm::Reg;
use super::def_inst;
use super::mem::const_splat;
use crate::alloc::Plan;
use crate::ir::{Binary, Function, Inst, Op, ShuffleMask, Value};

type PosOf<'a> = &'a [Option<(usize, usize, Inst)>];

fn is_shift_amount(func: &Function, plan: &Plan<Reg>, pos_of: PosOf<'_>, val: Value) -> bool {
    func.ty(val).is_vector()
        && (def_inst(pos_of, val)
            .is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::Const(0)))
            || const_splat(func, plan, pos_of, val).is_some())
}

fn reads_register(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: PosOf<'_>,
    user: Inst,
    val: Value,
) -> bool {
    let data = &func.insts[user.index()];
    let elided = data.result.is_some_and(|res| plan.elided[res.index()]);
    match data.op {
        Op::Binary(Binary::Ishl | Binary::Ushr | Binary::Sshr, x, y) => {
            x == val || y != val || !is_shift_amount(func, plan, pos_of, val)
        }
        Op::Insert(base, scalar, _) => !elided || base != val || scalar == val,
        Op::Shuffle(_, _, ShuffleMask::Const(_)) => !elided,
        _ => true,
    }
}

fn is_vec_const_def(func: &Function, pos_of: PosOf<'_>, val: Value) -> bool {
    func.ty(val).is_vector()
        && def_inst(pos_of, val).is_some_and(|inst| {
            matches!(
                func.insts[inst.index()].op,
                Op::Const(0) | Op::Insert(..) | Op::Shuffle(_, _, ShuffleMask::Const(_))
            )
        })
}

pub(super) fn elide_vec_consts(func: &Function, plan: &mut Plan<Reg>, pos_of: PosOf<'_>) {
    if !func.types.iter().any(|ty| ty.is_vector()) {
        return;
    }
    let mut read = vec![false; func.types.len()];
    loop {
        read.fill(false);
        let view: &Plan<Reg> = plan;
        for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
            func.insts[inst.index()].op.for_operands(|val| {
                read[val.index()] |= reads_register(func, view, pos_of, inst, val)
            });
        }
        let dead: Vec<usize> = (0..read.len())
            .filter(|&v| {
                !read[v] && !view.elided[v] && is_vec_const_def(func, pos_of, Value(v as u32))
            })
            .collect();
        if dead.is_empty() {
            return;
        }
        dead.into_iter().for_each(|v| plan.elided[v] = true);
    }
}
