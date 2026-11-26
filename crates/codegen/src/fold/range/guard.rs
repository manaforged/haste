use super::bounds::{sext_candidate, u_bounds};
use super::{RangeState, const_val, def_inst};
use crate::cfg::Cfg;
use crate::fold::{defs, extend};
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Value};

#[derive(Clone, Copy)]
struct UseSite {
    block: Block,
    pos: usize,
    inst: Inst,
    branch_target: Option<Block>,
}

struct GuardCtx {
    cfg: std::sync::Arc<Cfg>,
    def_blocks: Vec<Option<Block>>,
    starts: Vec<u32>,
    sites: Vec<UseSite>,
    has_traps: Vec<bool>,
}

impl GuardCtx {
    fn new(func: &Function) -> Self {
        let cfg = Some(func.cfg())
            .filter(|cfg| cfg.matches(func))
            .unwrap_or_else(|| std::sync::Arc::new(Cfg::new(func)));
        let def_blocks = func.def_blocks();
        let mut starts = vec![0_u32; func.types.len() + 1];
        let has_traps =
            func.blocks
                .iter()
                .zip(&cfg.reachable)
                .map(|(block, &live)| {
                    live && block.insts.iter().any(|inst| {
                        matches!(func.insts[inst.index()].op, Op::Trapz(_) | Op::Assume(_))
                    })
                })
                .collect();
        for_reachable_uses(func, &cfg, |value, _| starts[value.index() + 1] += 1);
        for at in 1..starts.len() {
            starts[at] += starts[at - 1];
        }
        let mut fill = starts.clone();
        let blank = UseSite {
            block: Block(0),
            pos: 0,
            inst: Inst(0),
            branch_target: None,
        };
        let mut sites = vec![blank; starts[starts.len() - 1] as usize];
        for_reachable_uses(func, &cfg, |value, site| {
            let at = &mut fill[value.index()];
            sites[*at as usize] = site;
            *at += 1;
        });
        Self {
            cfg,
            def_blocks,
            starts,
            sites,
            has_traps,
        }
    }

    fn uses(&self, value: Value) -> &[UseSite] {
        &self.sites[self.starts[value.index()] as usize..self.starts[value.index() + 1] as usize]
    }
}

fn for_reachable_uses(func: &Function, cfg: &Cfg, mut each: impl FnMut(Value, UseSite)) {
    for (b_idx, block_data) in func.blocks.iter().enumerate() {
        if !cfg.reachable[b_idx] {
            continue;
        }
        let block = Block(b_idx as u32);
        for (pos, &inst) in block_data.insts.iter().enumerate() {
            record_uses(
                &func.insts[inst.index()].op,
                UseSite {
                    block,
                    pos,
                    inst,
                    branch_target: None,
                },
                &mut each,
            );
        }
    }
}

fn record_uses(op: &Op, site: UseSite, each: &mut impl FnMut(Value, UseSite)) {
    if let Op::Brif(cond, tb, ta, eb, ea) = op {
        each(*cond, site);
        for v in ta {
            each(
                *v,
                UseSite {
                    branch_target: Some(*tb),
                    ..site
                },
            );
        }
        for v in ea {
            each(
                *v,
                UseSite {
                    branch_target: Some(*eb),
                    ..site
                },
            );
        }
        return;
    }
    op.for_operands(|v| each(v, site));
}

fn guard_proves_cmp(cc: IntCC, c_u: u64, sign_bit: u64) -> bool {
    match cc {
        IntCC::UnsignedLessThan => c_u <= sign_bit,
        IntCC::UnsignedLessThanOrEqual | IntCC::Equal => c_u < sign_bit,
        _ => false,
    }
}

fn cmp_operand_const(
    func: &Function,
    state: &RangeState,
    inst: Inst,
    res: Value,
) -> Option<(IntCC, u64)> {
    let Op::Icmp(cc, l, r) = func.insts[inst.index()].op else {
        return None;
    };
    let (cc, other) = if l == res {
        (cc, r)
    } else if r == res {
        (cc.swapped(), l)
    } else {
        return None;
    };
    if other == res {
        return None;
    }
    let c = const_val(state, other)?;
    Some((cc, extend(c, func.ty(res), false) as u64))
}

fn is_sign_inv_cmp(
    func: &Function,
    state: &RangeState,
    inst: Inst,
    res: Value,
    sign_bit: u64,
) -> bool {
    cmp_operand_const(func, state, inst, res).is_some_and(|(cc, c_u)| {
        guard_proves_cmp(cc, c_u, sign_bit) || guard_proves_cmp(cc.inverse(), c_u, sign_bit)
    })
}

fn bool_xor_inner(func: &Function, state: &RangeState, a: Value, b: Value) -> Option<Value> {
    [(a, b), (b, a)].into_iter().find_map(|(inner, one)| {
        (const_val(state, one) == Some(1) && u_bounds(func, state, inner, 0).1 <= 1)
            .then_some(inner)
    })
}

fn guard_proves(
    func: &Function,
    state: &RangeState,
    (cond, taken, depth): (Value, bool, u8),
    res: Value,
    sign_bit: u64,
) -> bool {
    if cond == res {
        return !taken;
    }
    if depth >= 4 {
        return false;
    }
    let Some(inst) = def_inst(state, cond) else {
        return false;
    };
    match func.insts[inst.index()].op {
        Op::Icmp(..) => cmp_operand_const(func, state, inst, res).is_some_and(|(cc, c_u)| {
            let eff = if taken { cc } else { cc.inverse() };
            guard_proves_cmp(eff, c_u, sign_bit)
        }),
        Op::Convert(Convert::Uextend, inner) if func.ty(inner).is_small_int() => {
            guard_proves(func, state, (inner, taken, depth + 1), res, sign_bit)
        }
        Op::Binary(Binary::Bxor, a, b) => bool_xor_inner(func, state, a, b).is_some_and(|inner| {
            guard_proves(func, state, (inner, !taken, depth + 1), res, sign_bit)
        }),
        Op::Binary(Binary::Band, a, b) if taken => {
            guard_proves(func, state, (a, true, depth + 1), res, sign_bit)
                || guard_proves(func, state, (b, true, depth + 1), res, sign_bit)
        }
        Op::Binary(Binary::Bor, a, b) if !taken => {
            guard_proves(func, state, (a, false, depth + 1), res, sign_bit)
                || guard_proves(func, state, (b, false, depth + 1), res, sign_bit)
        }
        _ => false,
    }
}

fn single_pred_guard(
    func: &Function,
    state: &RangeState,
    cfg: &Cfg,
    block: Block,
    (res, sign_bit): (Value, u64),
) -> bool {
    let Some(pred) = cfg.single_pred(block) else {
        return false;
    };
    if pred == block || !cfg.reachable[pred.index()] {
        return false;
    }
    let Some(&term) = func.blocks[pred.index()].insts.last() else {
        return false;
    };
    let Op::Brif(cond, tb, _, eb, _) = func.insts[term.index()].op else {
        return false;
    };
    tb != eb && guard_proves(func, state, (cond, block == tb, 0), res, sign_bit)
}

fn block_has_trap_guard(
    func: &Function,
    state: &RangeState,
    ctx: &GuardCtx,
    (block, limit): (Block, usize),
    (res, sign_bit): (Value, u64),
) -> bool {
    if !ctx.has_traps[block.index()] {
        return false;
    }
    func.blocks[block.index()].insts[..limit].iter().any(|&inst| {
        matches!(
            func.insts[inst.index()].op,
            Op::Trapz(cond) | Op::Assume(cond) if guard_proves(func, state, (cond, true, 0), res, sign_bit)
        )
    })
}

fn brif_edge_guarded(
    func: &Function,
    state: &RangeState,
    site: UseSite,
    (res, sign_bit): (Value, u64),
) -> bool {
    let Some(edge_dest) = site.branch_target else {
        return false;
    };
    let Op::Brif(cond, tb, _, eb, _) = func.insts[site.inst.index()].op else {
        return false;
    };
    tb != eb && guard_proves(func, state, (cond, edge_dest == tb, 0), res, sign_bit)
}

fn is_guarded_use(
    func: &Function,
    state: &RangeState,
    ctx: &GuardCtx,
    site: UseSite,
    (def_block, target): (Block, (Value, u64)),
) -> bool {
    if brif_edge_guarded(func, state, site, target)
        || block_has_trap_guard(func, state, ctx, (site.block, site.pos), target)
        || (site.block != def_block && single_pred_guard(func, state, &ctx.cfg, site.block, target))
    {
        return true;
    }
    let mut anc = site.block;
    while anc != def_block {
        let Some(parent) = ctx.cfg.idom[anc.index()] else {
            return false;
        };
        if parent == anc {
            return false;
        }
        anc = parent;
        if anc != def_block && single_pred_guard(func, state, &ctx.cfg, anc, target) {
            return true;
        }
        let len = func.blocks[anc.index()].insts.len();
        if block_has_trap_guard(func, state, ctx, (anc, len), target) {
            return true;
        }
    }
    false
}

fn can_narrow_guarded(
    func: &Function,
    state: &RangeState,
    ctx: &GuardCtx,
    (res, sign_bit): (Value, u64),
) -> bool {
    let Some(def_block) = ctx.def_blocks[res.index()] else {
        return false;
    };
    ctx.uses(res).iter().all(|&site| {
        is_sign_inv_cmp(func, state, site.inst, res, sign_bit)
            || is_guarded_use(func, state, ctx, site, (def_block, (res, sign_bit)))
    })
}

pub(super) fn fold_guarded_sext(func: &mut Function, state: &mut RangeState) -> bool {
    let has_sext = func
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|&inst| sext_candidate(func, inst).is_some());
    if !has_sext {
        return false;
    }
    state.defs = defs(func);
    let ctx = GuardCtx::new(func);
    let mut changed = false;
    for b_idx in 0..func.blocks.len() {
        if !ctx.cfg.reachable[b_idx] {
            continue;
        }
        for pos in 0..func.blocks[b_idx].insts.len() {
            let inst = func.blocks[b_idx].insts[pos];
            let Some((res, inner, sign_bit)) = sext_candidate(func, inst) else {
                continue;
            };
            if can_narrow_guarded(func, state, &ctx, (res, sign_bit)) {
                func.insts[inst.index()].op = Op::Convert(Convert::Uextend, inner);
                changed = true;
            }
        }
    }
    changed
}
