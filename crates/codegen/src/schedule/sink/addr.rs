use crate::cfg::Loops;
use crate::fold::defs;
use crate::ir::{Binary, Block, Function, Inst, Op, Type, Value};

use super::uses_operand;

struct Sum {
    block: usize,
    res: Value,
    x: Value,
    y: Value,
}

fn collect_sums(func: &Function, loops: &Loops) -> (Vec<Sum>, Vec<Option<u32>>) {
    let defs = defs(func);
    let is_const = |v: Value| {
        defs[v.index()].is_some_and(|i| matches!(func.insts[i.index()].op, Op::Const(_)))
    };
    let (mut sums, mut index) = (Vec::new(), vec![None; func.types.len()]);
    for (bi, block) in func.blocks.iter().enumerate() {
        if loops.depth_of(Block(bi as u32)) == 0 {
            continue;
        }
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let (Op::Binary(Binary::Iadd, x, y), Some(res)) = (&data.op, data.result)
                && func.ty(res) == Type::I64
                && !is_const(*x)
                && !is_const(*y)
            {
                index[res.index()] = Some(sums.len() as u32);
                sums.push(Sum {
                    block: bi,
                    res,
                    x: *x,
                    y: *y,
                });
            }
        }
    }
    (sums, index)
}

fn collect_users(func: &Function, index: &[Option<u32>], count: usize) -> Vec<Vec<(usize, Inst)>> {
    let mut users: Vec<Vec<(usize, Inst)>> = vec![Vec::new(); count];
    for (bi, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            func.insts[inst.index()].op.for_operands(|v| {
                if let Some(Some(k)) = index.get(v.index()) {
                    let list = &mut users[*k as usize];
                    if list.last() != Some(&(bi, inst)) {
                        list.push((bi, inst));
                    }
                }
            });
        }
    }
    users
}

fn reads_as_address(func: &Function, inst: Inst, res: Value) -> bool {
    let data = &func.insts[inst.index()];
    match data.op {
        Op::Load(addr, 0) => addr == res && data.result.is_some_and(|r| func.ty(r) != Type::I128),
        Op::Store(val, addr, 0) => addr == res && val != res && func.ty(val) != Type::I128,
        _ => false,
    }
}

fn runs_at_most_as_often(loops: &Loops, (home, block): (usize, usize)) -> bool {
    loops.list.iter().all(|each| {
        !each.body.get(block).copied().unwrap_or(false)
            || each.body.get(home).copied().unwrap_or(false)
    })
}

fn reads_an_operand(func: &Function, bi: usize, sum: &Sum) -> bool {
    func.blocks[bi]
        .insts
        .iter()
        .any(|&i| uses_operand(func, i, sum.x) || uses_operand(func, i, sum.y))
}

fn clone_blocks(
    func: &Function,
    loops: &Loops,
    sum: &Sum,
    users: &[(usize, Inst)],
) -> Option<Vec<usize>> {
    let mut blocks = Vec::new();
    let mut at_home = false;
    for &(bi, inst) in users {
        if bi == sum.block {
            if !reads_as_address(func, inst, sum.res) {
                return None;
            }
            at_home = true;
        } else if !blocks.contains(&bi) {
            blocks.push(bi);
        }
    }
    let depth = loops.depth_of(Block(sum.block as u32));
    let same_depth = blocks
        .iter()
        .filter(|&&bi| loops.depth_of(Block(bi as u32)) == depth)
        .count();
    let ok = at_home
        && !blocks.is_empty()
        && same_depth <= 1
        && blocks.iter().all(|&bi| {
            runs_at_most_as_often(loops, (sum.block, bi)) && reads_an_operand(func, bi, sum)
        });
    ok.then_some(blocks)
}

fn clone_into(func: &mut Function, sum: &Sum, bi: usize) {
    let insts = &func.blocks[bi].insts;
    let Some(at) = insts.iter().position(|&i| uses_operand(func, i, sum.res)) else {
        return;
    };
    let users: Vec<Inst> = insts[at..]
        .iter()
        .copied()
        .filter(|&i| uses_operand(func, i, sum.res))
        .collect();
    let copy = func.value(Type::I64);
    let inst = func.add(Op::Binary(Binary::Iadd, sum.x, sum.y), Some(copy));
    func.blocks[bi].insts.insert(at, inst);
    for user in users {
        func.insts[user.index()]
            .op
            .map_operands(|v| if v == sum.res { copy } else { v });
    }
}

/// Copies each in-loop register sum that its own block reads only as a memory address into the other blocks that read it, so the backend folds the sum into a register-offset address.
pub(super) fn split_address_sums(func: &mut Function) {
    let cfg = func.cfg();
    let loops = cfg.loops();
    let (sums, index) = collect_sums(func, loops);
    if sums.is_empty() {
        return;
    }
    let users = collect_users(func, &index, sums.len());
    for (sum, list) in sums.iter().zip(&users) {
        if let Some(blocks) = clone_blocks(func, loops, sum, list) {
            blocks.into_iter().for_each(|bi| clone_into(func, sum, bi));
        }
    }
}
