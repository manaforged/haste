mod affine;
mod caller_loc;
mod caller_loc_sink;
mod coalesce;
mod concat;
mod dom;
mod elide;
mod join;
mod pack;
mod regroup;
mod remat;
mod scan;
mod sink;

use affine::{eliminate_affine_block_params, fold_affine_mem};
use coalesce::ScanBuf;
use concat::{Assembled, assemble_const};
use dom::BlockWrites;
use join::{JoinCtx, eligible_join_preds};
use remat::{FarCseGuard, sink_terminator_defs};
use scan::scan;

use crate::ir::{Domain, Function, Inst, Op, Type, Value};
use crate::mem::{KnownLoads, MemBase, MemLoc, MemRoot, Provenance};
use crate::merge::{find, identity, rename};

struct State<'a> {
    prov: &'a Provenance,
    defs: &'a [Option<Inst>],
    selects: &'a [Option<(Value, Value, Value)>],
    consts: &'a mut Vec<Option<i64>>,
    subst: &'a mut Vec<Value>,
    target: &'a crate::Target,
    post_inline: bool,
}

fn addr_loc(prov: &Provenance, subst: &[Value], addr: Value, offset: i32, size: u32) -> MemLoc {
    let loc = prov.loc(find(subst, addr), i64::from(offset), size);
    if let MemBase::Addr(base) = loc.base {
        let canonical = find(subst, base);
        if canonical != base {
            return prov.loc(canonical, loc.offset, size);
        }
    }
    loc
}

fn emit_op(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    ty: Type,
    op: Op,
) -> Value {
    let val = func.value(ty);
    state.subst.push(val);
    state.consts.push(match op {
        Op::Const(imm) => Some(imm),
        _ => None,
    });
    kept.push(Some(func.add(op, Some(val))));
    val
}

fn fits_mem_imm(ty: Type, off: i64) -> bool {
    let fits_one = |align: i64, o: i64| {
        (-256..=255).contains(&o) || (o >= 0 && o % align == 0 && o <= align * 4095)
    };
    if ty == Type::I128 {
        return off
            .checked_add(8)
            .is_some_and(|hi| fits_one(8, off) && fits_one(8, hi));
    }
    fits_one(i64::from(ty.bytes().max(1)), off)
}

fn collect_scan_meta(
    func: &Function,
) -> Option<(
    Vec<Option<Inst>>,
    Vec<Option<i64>>,
    Vec<Option<(Value, Value, Value)>>,
)> {
    let n = func.types.len();
    let (mut defs, mut consts, mut selects, mut accesses) =
        (vec![None; n], vec![None; n], Vec::new(), false);
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        if let Some(result) = data.result {
            defs[result.index()] = Some(inst);
        }
        match (&data.op, data.result) {
            (&Op::Const(imm), Some(result)) => consts[result.index()] = Some(imm),
            (&Op::Select(c, a, b), Some(result)) => {
                if selects.is_empty() {
                    selects.resize(n, None);
                }
                selects[result.index()] = Some((c, a, b));
            }
            (op, _)
                if matches!(
                    op,
                    Op::Load(..) | Op::StackLoad(..) | Op::Store(..) | Op::StackStore(..)
                ) =>
            {
                accesses = true;
            }
            _ => {}
        }
    }
    accesses.then_some((defs, consts, selects))
}

/// Elides whole-slot moves once, before promotion splits the slots into values.
pub(crate) fn elide_early(func: &mut Function) -> bool {
    if func.entry.is_none()
        || func.coroutine.is_some()
        || !crate::schedule::has_live_op(func, |op| matches!(op, Op::Move(..)))
    {
        return false;
    }
    let cfg = func.cfg();
    elide::elide_moves(func, &Provenance::new(func, &cfg))
}

/// Replaces each load of bytes that an earlier access on every path to it read or wrote with no write of unknown reach between; returns the effect domains modified.
pub(crate) fn run(func: &mut Function, post_inline: bool, target: &crate::Target) -> Domain {
    if func.entry.is_none() || func.coroutine.is_some() {
        return Domain::NONE;
    }
    let domain = |changed: bool| {
        if changed {
            Domain::MEM | Domain::VALUES
        } else {
            Domain::NONE
        }
    };
    let (mut changed, affine) = eliminate_affine_block_params(func);
    if !crate::schedule::has_live_op(func, |op| {
        matches!(
            op,
            Op::Load(..) | Op::StackLoad(..) | Op::Store(..) | Op::StackStore(..)
        )
    }) {
        changed |= sink_terminator_defs(func);
        return domain(changed);
    }
    let cfg = func.cfg();
    let mut prov = Provenance::with_affine(func, &cfg, affine);
    if caller_loc::optimize(func, &cfg, &prov) {
        changed = true;
        if func.types.len() > prov.param_roots.len() {
            prov = Provenance::new(func, &cfg);
        }
    }
    changed |= crate::mem::drop_unread_frame_stores(func, &cfg, &prov);
    changed |= fold_affine_mem(func, &prov.slot_offsets, &prov.affine);
    changed |= pack::coalesce_loads(func, &cfg, &prov, target);
    changed |= sink_terminator_defs(func);
    let Some((defs, mut consts, selects)) = collect_scan_meta(func) else {
        return domain(changed);
    };
    let mut bw = BlockWrites::new(func, &cfg, &prov);
    let mut subst = identity(func.types.len());
    let mut state = State {
        prov: &prov,
        defs: &defs,
        selects: &selects,
        consts: &mut consts,
        subst: &mut subst,
        target,
        post_inline,
    };
    let has_join = cfg.rpo.len() > 1 && cfg.rpo.iter().any(|&b| cfg.pred_count(b) >= 2);
    let (mut renamed, mut scan_changed, mut def_blocks) = (
        false,
        false,
        if has_join {
            func.def_blocks()
        } else {
            Vec::new()
        },
    );
    let (mut visited, mut added_phis) =
        (vec![false; func.blocks.len()], vec![0u8; func.blocks.len()]);
    let mut known_out: Vec<KnownLoads> =
        (0..func.blocks.len()).map(|_| KnownLoads::new()).collect();
    let mut block_bases: Vec<Vec<(MemBase, MemRoot)>> = vec![Vec::new(); func.blocks.len()];
    let mut bufs = ScanBuf {
        kept: Vec::new(),
        pending_stores: Vec::new(),
        const_stores: Vec::new(),
    };
    let (mut clobbers_seen, mut guard) = (Vec::new(), FarCseGuard::empty());
    for &block in &cfg.rpo {
        let (mut known, mut bases) =
            bw.entry_known(func, state.subst, &known_out, &block_bases, block);
        let preds = eligible_join_preds(func, &cfg, &visited, block);
        let mut join = JoinCtx {
            cfg: &cfg,
            bw: &bw,
            known_out: &known_out,
            block_bases: &block_bases,
            visited: &visited,
            def_blocks: &mut def_blocks,
            preds,
            added_phis: &mut added_phis,
        };
        let (bc, br) = scan(
            func,
            block,
            &mut state,
            &mut known,
            &mut bases,
            &mut join,
            (&mut bufs, &mut clobbers_seen, &mut guard),
        );
        if has_join {
            def_blocks.resize(func.types.len(), Some(block));
        }
        visited[block.index()] = true;
        (scan_changed, changed, renamed) = (scan_changed | bc, changed | bc, renamed | br);
        (known_out[block.index()], block_bases[block.index()]) = (known, bases);
    }
    if renamed {
        rename(func, state.subst);
    }
    if scan_changed {
        changed |= pack::coalesce_vec_rescan(func, &cfg, target);
    }
    changed |= remat::split_long_loads(func, &prov) | sink::sink_loads(func, &prov);
    if post_inline {
        if elide::elide_moves(func, &prov) {
            changed = true;
            let cfg = func.cfg();
            changed |= elide::forward_moves(func, &Provenance::new(func, &cfg));
        } else {
            changed |= elide::forward_moves(func, &prov);
        }
    }
    changed |= regroup::run(func, (!changed).then_some(&prov));
    domain(changed)
}
