use crate::cfg::{self, Cfg, Loop};
use crate::ir::{Block, Function, Inst, Op};
use crate::mem::{LoopMem, Provenance};

fn hoist(
    func: &mut Function,
    each: &Loop,
    entry: Block,
    prov: &Provenance,
    home: &mut [Option<Block>],
) -> bool {
    let mem = LoopMem::analyze(func, each, prov);
    let mut moved: Vec<Inst> = Vec::new();
    for block in (0..each.body.len()).filter(|block| each.body[*block]) {
        let head = block == each.head.index();
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut kept = Vec::with_capacity(insts.len());
        let mut head_prefix = head;
        for inst in insts {
            let op = &func.insts[inst.index()].op;
            if matches!(op, Op::Call { .. } | Op::TryCall { .. } | Op::Trap | Op::Trapz(_) | Op::Brif(..)) {
                head_prefix = false;
            }
            let static_loc = matches!(op, Op::StackAddr(..) | Op::SlotAddr(_));
            let movable = (op.is_speculable() && !(mem.calls && static_loc))
                || mem.can_hoist_load(func, prov, head_prefix, inst);
            let mut operands_invariant = true;
            op.for_operands(|v| operands_invariant &= home.get(v.index()).copied().flatten().is_none_or(|at| !each.body.get(at.index()).copied().unwrap_or(false)));
            if movable && operands_invariant && let Some(res) = func.insts[inst.index()].result {
                home[res.index()] = Some(entry);
                moved.push(inst);
            } else {
                kept.push(inst);
            }
        }
        func.blocks[block].insts = kept;
    }
    if moved.is_empty() {
        return false;
    }
    let insts = &mut func.blocks[entry.index()].insts;
    let at = insts.len().saturating_sub(1);
    insts.splice(at..at, moved);
    true
}

/// Moves loop-invariant work of every natural loop that one jump enters out to the block that enters it.
pub(crate) fn run(func: &mut Function) -> bool {
    if func.entry.is_none() || func.coroutine.is_some() {
        return false;
    }
    let cfg = Cfg::new(func);
    let mut found = cfg.loops(func);
    if found.list.is_empty() {
        return false;
    }
    let prov = Provenance::new(func, &cfg);
    let mut home = func.def_blocks();
    let mut changed = false;
    for idx in 0..found.list.len() {
        let before = func.blocks.len();
        let Some(pre) = cfg::preheader(func, &cfg, &found.list[idx]) else { continue };
        if func.blocks.len() > before {
            changed = true;
            home.resize(func.types.len(), Some(pre));
            let head_idx = found.list[idx].head.index();
            found.list[idx].body.resize(pre.index() + 1, false);
            for later in &mut found.list[idx + 1..] {
                let inside = later.body.get(head_idx).copied().unwrap_or(false);
                later.body.resize(pre.index(), false);
                later.body.push(inside);
            }
        }
        changed |= hoist(func, &found.list[idx], pre, &prov, &mut home);
    }
    changed
}
