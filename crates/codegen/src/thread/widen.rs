use crate::cfg::Cfg;
use crate::fold::{defs as collect_defs, evaluate_wrapped, extend, known_bits};
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, Type, Value};

#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeArm {
    Jump,
    BrifThen,
    BrifElse,
}

#[derive(Clone, Copy)]
struct EdgeSite {
    pred: usize,
    term: Inst,
    arm: EdgeArm,
    arg: Value,
    in_loop: bool,
}

struct Candidate {
    block: usize,
    param_idx: usize,
    param: Value,
    narrow_ty: Type,
    wide_ty: Type,
    edges: Vec<EdgeSite>,
    uext_count: usize,
    active: bool,
}

enum EdgePlan {
    Const(i64),
    Param(Value),
    ConvertOnly(Value),
    Mask(Value, u64),
    Extend(Value),
}

/// Widens narrow `I8` and `I16` loop and block parameters to `I32` or `I64` when their upper bits are provably zero or cheaply preserved, eliminating repeated `Uextend` operations inside loops.
pub(super) fn run(func: &mut Function) -> bool {
    let Some(entry) = func.entry else {
        return false;
    };
    if !func
        .blocks
        .iter()
        .any(|block| block.params.iter().any(|&param| is_narrow(func.ty(param))))
    {
        return false;
    }
    let cfg = func.cfg();
    let mut blocked = vec![false; func.blocks.len()];
    blocked[entry.index()] = true;
    for data in &func.insts {
        match &data.op {
            Op::TryCall {
                normal, handler, ..
            } => {
                blocked[normal.index()] = true;
                blocked[handler.index()] = true;
            }
            Op::BrTable(_, default, targets) => {
                blocked[default.index()] = true;
                for &t in targets {
                    blocked[t.index()] = true;
                }
            }
            _ => {}
        }
    }

    let mut cands = collect_candidates(func, &cfg, &blocked);
    if cands.is_empty() {
        return false;
    }
    let defs = collect_defs(func);
    let consts = fold_consts(func, &cfg);
    let mut val_to_cand = vec![None; func.types.len()];
    for (idx, c) in cands.iter().enumerate() {
        val_to_cand[c.param.index()] = Some(idx);
    }
    propagate_wide_ty(&mut cands, &val_to_cand);
    prune_candidates(func, &defs, &consts, &mut cands, &val_to_cand);

    let kept: Vec<usize> = cands
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.active.then_some(i))
        .collect();
    if kept.is_empty() {
        return false;
    }
    apply_widening(func, &defs, &consts, &cands, &val_to_cand, &kept);
    true
}

fn is_narrow(ty: Type) -> bool {
    matches!(ty, Type::I8 | Type::I16)
}

fn fold_consts(func: &Function, cfg: &Cfg) -> Vec<Option<i64>> {
    let mut consts = vec![None; func.types.len()];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result {
                consts[res.index()] =
                    evaluate_wrapped(func, &data.op, func.ty(res), &|v| consts[v.index()]);
            }
        }
    }
    consts
}

fn collect_candidates(func: &Function, cfg: &Cfg, blocked: &[bool]) -> Vec<Candidate> {
    let mut uext_counts = vec![0_usize; func.types.len()];
    let mut uext_has_64 = vec![false; func.types.len()];
    let mut bad_use = vec![false; func.types.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if !cfg.reachable[b_idx] {
                data.op.for_operands(|v| bad_use[v.index()] = true);
                continue;
            }
            if let (Op::Convert(Convert::Uextend, x), Some(res)) = (&data.op, data.result)
                && func.ty(res) != Type::I128
            {
                uext_counts[x.index()] += 1;
                if func.ty(res) == Type::I64 {
                    uext_has_64[x.index()] = true;
                }
            }
        }
    }

    let mut cands = Vec::new();
    let mut slot_to_cand: Vec<Vec<Option<usize>>> = func
        .blocks
        .iter()
        .map(|b| vec![None; b.params.len()])
        .collect();
    for (b_idx, block) in func.blocks.iter().enumerate() {
        if blocked[b_idx] || !cfg.reachable[b_idx] {
            continue;
        }
        for (p_idx, &param) in block.params.iter().enumerate() {
            let ty = func.ty(param);
            if !is_narrow(ty) || uext_counts[param.index()] == 0 || bad_use[param.index()] {
                continue;
            }
            let wide_ty = if uext_has_64[param.index()] {
                Type::I64
            } else {
                Type::I32
            };
            slot_to_cand[b_idx][p_idx] = Some(cands.len());
            cands.push(Candidate {
                block: b_idx,
                param_idx: p_idx,
                param,
                narrow_ty: ty,
                wide_ty,
                edges: Vec::new(),
                uext_count: uext_counts[param.index()],
                active: true,
            });
        }
    }

    for (pred, block) in func.blocks.iter().enumerate() {
        if !cfg.reachable[pred] {
            continue;
        }
        let Some(&term) = block.insts.last() else {
            continue;
        };
        let pred_block = Block(pred as u32);
        let mut push_edge = |target: Block, args: &[Value], arm: EdgeArm| {
            let t_idx = target.index();
            if args.len() != slot_to_cand[t_idx].len() {
                return;
            }
            let in_loop = cfg.dominates(target, pred_block);
            for (p_idx, &arg) in args.iter().enumerate() {
                if let Some(c_idx) = slot_to_cand[t_idx][p_idx] {
                    cands[c_idx].edges.push(EdgeSite {
                        pred,
                        term,
                        arm,
                        arg,
                        in_loop,
                    });
                }
            }
        };
        match &func.insts[term.index()].op {
            Op::Jump(target, args) => push_edge(*target, args, EdgeArm::Jump),
            Op::Brif(_, then_b, then_args, else_b, else_args) => {
                push_edge(*then_b, then_args, EdgeArm::BrifThen);
                push_edge(*else_b, else_args, EdgeArm::BrifElse);
            }
            _ => {}
        }
    }

    for cand in &mut cands {
        if cand.edges.is_empty() {
            cand.active = false;
        }
    }
    cands
}

fn propagate_wide_ty(cands: &mut [Candidate], val_to_cand: &[Option<usize>]) {
    loop {
        let mut changed = false;
        for idx in 0..cands.len() {
            if !cands[idx].active || cands[idx].wide_ty != Type::I64 {
                continue;
            }
            for edge_i in 0..cands[idx].edges.len() {
                let arg = cands[idx].edges[edge_i].arg;
                if let Some(src_idx) = val_to_cand[arg.index()]
                    && cands[src_idx].active
                    && cands[src_idx].wide_ty != Type::I64
                {
                    cands[src_idx].wide_ty = Type::I64;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

fn upper_bits_zero(
    func: &Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    cands: &[Candidate],
    val_to_cand: &[Option<usize>],
    narrow_mask: u64,
    v: Value,
    depth: u8,
) -> bool {
    let ty = func.ty(v);
    if !ty.is_small_int() {
        return false;
    }
    if let Some(c_idx) = val_to_cand[v.index()]
        && cands[c_idx].active
        && (cands[c_idx].narrow_ty.unsigned_max() & !narrow_mask) == 0
    {
        return true;
    }
    if depth < 4
        && let Some(inst) = defs[v.index()]
        && let Op::Select(_, a, b) = func.insts[inst.index()].op
    {
        return upper_bits_zero(
            func,
            defs,
            consts,
            cands,
            val_to_cand,
            narrow_mask,
            a,
            depth + 1,
        ) && upper_bits_zero(
            func,
            defs,
            consts,
            cands,
            val_to_cand,
            narrow_mask,
            b,
            depth + 1,
        );
    }
    let (zeros, _) = known_bits(func, &|u| defs[u.index()], &|u| consts[u.index()], v, depth);
    ((zeros | narrow_mask) & ty.unsigned_max()) == ty.unsigned_max()
}

fn classify_edge(
    func: &Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    cands: &[Candidate],
    val_to_cand: &[Option<usize>],
    cand: &Candidate,
    arg: Value,
) -> (EdgePlan, usize) {
    if let Some(imm) = consts[arg.index()] {
        return (EdgePlan::Const(extend(imm, cand.narrow_ty, false)), 0);
    }
    if let Some(src_idx) = val_to_cand[arg.index()]
        && cands[src_idx].active
        && cands[src_idx].narrow_ty.bits() <= cand.narrow_ty.bits()
    {
        let cost = usize::from(cands[src_idx].wide_ty != cand.wide_ty);
        return (EdgePlan::Param(arg), cost);
    }
    let narrow_mask = cand.narrow_ty.unsigned_max();
    if let Some(inst) = defs[arg.index()]
        && let Op::Convert(Convert::Ireduce, src) = func.insts[inst.index()].op
    {
        let src_ty = func.ty(src);
        if src_ty.is_small_int() {
            if upper_bits_zero(func, defs, consts, cands, val_to_cand, narrow_mask, src, 0) {
                let cost = usize::from(src_ty != cand.wide_ty);
                return (EdgePlan::ConvertOnly(src), cost);
            }
            let cost = 1 + usize::from(src_ty != cand.wide_ty);
            return (EdgePlan::Mask(src, narrow_mask), cost);
        }
    }
    (EdgePlan::Extend(arg), 1)
}

fn can_widen_icmp(
    func: &Function,
    consts: &[Option<i64>],
    cands: &[Candidate],
    val_to_cand: &[Option<usize>],
    cand: &Candidate,
    inst: Inst,
) -> bool {
    let Op::Icmp(cc, lhs, rhs) = func.insts[inst.index()].op else {
        return false;
    };
    if cc.is_signed() {
        return false;
    }
    let other = if lhs == cand.param && rhs == cand.param {
        return true;
    } else if lhs == cand.param {
        rhs
    } else if rhs == cand.param {
        lhs
    } else {
        return false;
    };
    if consts[other.index()].is_some() {
        return true;
    }
    val_to_cand[other.index()]
        .is_some_and(|idx| cands[idx].active && cands[idx].wide_ty == cand.wide_ty)
}

fn candidate_profitable(
    func: &Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    cands: &[Candidate],
    val_to_cand: &[Option<usize>],
    cand: &Candidate,
) -> bool {
    if !cand.active || cand.edges.is_empty() {
        return false;
    }
    let mut loop_inserted = 0_usize;
    let mut outside_inserted = 0_usize;
    let mut has_backedge = false;
    for edge in &cand.edges {
        let (_, cost) = classify_edge(func, defs, consts, cands, val_to_cand, cand, edge.arg);
        if edge.in_loop {
            has_backedge = true;
            loop_inserted += cost;
        } else {
            outside_inserted += cost;
        }
    }

    let mut need_fallback = false;
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            let mut hits = 0_usize;
            data.op.for_operands(|v| {
                if v == cand.param {
                    hits += 1;
                }
            });
            if hits == 0 {
                continue;
            }
            if matches!(data.op, Op::Convert(Convert::Uextend, x) if x == cand.param) {
                continue;
            }
            if can_widen_icmp(func, consts, cands, val_to_cand, cand, inst) {
                continue;
            }
            let handled_as_edge = match &data.op {
                Op::Jump(target, args) => {
                    count_active_edge_hits(target, args, cand.param, val_to_cand, cands)
                }
                Op::Brif(_, then_b, then_args, else_b, else_args) => {
                    count_active_edge_hits(then_b, then_args, cand.param, val_to_cand, cands)
                        + count_active_edge_hits(else_b, else_args, cand.param, val_to_cand, cands)
                }
                _ => 0,
            };
            if hits > handled_as_edge {
                need_fallback = true;
                break;
            }
        }
        if need_fallback {
            break;
        }
    }
    if need_fallback {
        loop_inserted += 1;
    }
    let saved = cand.uext_count;
    if has_backedge {
        loop_inserted < saved && (loop_inserted > 0 || outside_inserted <= 2 * saved)
    } else {
        loop_inserted + outside_inserted < saved
    }
}

fn count_active_edge_hits(
    target: &Block,
    args: &[Value],
    param: Value,
    val_to_cand: &[Option<usize>],
    cands: &[Candidate],
) -> usize {
    let mut hits = 0;
    for (idx, &arg) in args.iter().enumerate() {
        if arg == param
            && let Some(&dst_param) = cands
                .iter()
                .find(|c| c.active && c.block == target.index() && c.param_idx == idx)
                .map(|c| &c.param)
            && val_to_cand[dst_param.index()].is_some()
        {
            hits += 1;
        }
    }
    hits
}

fn prune_candidates(
    func: &Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    cands: &mut [Candidate],
    val_to_cand: &[Option<usize>],
) {
    loop {
        let mut changed = false;
        for idx in 0..cands.len() {
            if cands[idx].active
                && !candidate_profitable(func, defs, consts, cands, val_to_cand, &cands[idx])
            {
                cands[idx].active = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn int_convert_op(from_ty: Type, to_ty: Type) -> Convert {
    if to_ty.bits() > from_ty.bits() {
        Convert::Uextend
    } else {
        Convert::Ireduce
    }
}

fn emit_convert(func: &mut Function, val: Value, to_ty: Type, extra: &mut Vec<Inst>) -> Value {
    let from_ty = func.ty(val);
    if from_ty == to_ty {
        return val;
    }
    let res = func.value(to_ty);
    extra.push(func.add(Op::Convert(int_convert_op(from_ty, to_ty), val), Some(res)));
    res
}

fn materialize_edge(
    func: &mut Function,
    pred: usize,
    wide_ty: Type,
    plan: EdgePlan,
    wide_params: &[Option<Value>],
) -> Value {
    let mut extra = Vec::new();
    let val = match plan {
        EdgePlan::Const(imm) => {
            let v = func.value(wide_ty);
            extra.push(func.add(Op::Const(imm), Some(v)));
            v
        }
        EdgePlan::Param(src_param) => {
            let src_wide = wide_params[src_param.index()].unwrap_or(src_param);
            emit_convert(func, src_wide, wide_ty, &mut extra)
        }
        EdgePlan::ConvertOnly(src) | EdgePlan::Extend(src) => {
            emit_convert(func, src, wide_ty, &mut extra)
        }
        EdgePlan::Mask(src, mask) => {
            let src_w = emit_convert(func, src, wide_ty, &mut extra);
            let m = func.value(wide_ty);
            extra.push(func.add(Op::Const(mask as i64), Some(m)));
            let masked = func.value(wide_ty);
            extra.push(func.add(Op::Binary(Binary::Band, src_w, m), Some(masked)));
            masked
        }
    };
    if !extra.is_empty() {
        let pos = func.blocks[pred].insts.len() - 1;
        func.blocks[pred].insts.splice(pos..pos, extra);
    }
    val
}

fn rewrite_edges(
    func: &mut Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    cands: &[Candidate],
    val_to_cand: &[Option<usize>],
    kept: &[usize],
    wide_params: &[Option<Value>],
) {
    let mut per_pred: Vec<Vec<(Inst, EdgeArm, usize, Value)>> = vec![Vec::new(); func.blocks.len()];
    for &idx in kept {
        let c = &cands[idx];
        for edge in &c.edges {
            let plan = classify_edge(func, defs, consts, cands, val_to_cand, c, edge.arg).0;
            let val = materialize_edge(func, edge.pred, c.wide_ty, plan, wide_params);
            per_pred[edge.pred].push((edge.term, edge.arm, c.param_idx, val));
        }
    }

    for edits in per_pred {
        for (term, arm, param_idx, val) in edits {
            match &mut func.insts[term.index()].op {
                Op::Jump(_, args) if arm == EdgeArm::Jump => args[param_idx] = val,
                Op::Brif(_, _, then_args, _, _) if arm == EdgeArm::BrifThen => {
                    then_args[param_idx] = val
                }
                Op::Brif(_, _, _, _, else_args) if arm == EdgeArm::BrifElse => {
                    else_args[param_idx] = val
                }
                _ => {}
            }
        }
    }
}

fn emit_wide_const(func: &mut Function, cand: &Candidate, imm: i64, out: &mut Vec<Inst>) -> Value {
    let c_val = func.value(cand.wide_ty);
    out.push(func.add(Op::Const(extend(imm, cand.narrow_ty, false)), Some(c_val)));
    c_val
}

fn rewrite_icmp(
    func: &mut Function,
    consts: &[Option<i64>],
    cands: &[Candidate],
    val_to_cand: &[Option<usize>],
    wide_params: &[Option<Value>],
    inst: Inst,
    out: &mut Vec<Inst>,
) {
    let Op::Icmp(cc, lhs, rhs) = func.insts[inst.index()].op else {
        return;
    };
    if cc.is_signed() {
        return;
    }
    let orig_vals = wide_params.len();
    let active_cand = |v: Value| {
        (v.index() < orig_vals)
            .then(|| val_to_cand[v.index()])
            .flatten()
            .filter(|&i| cands[i].active)
    };
    match (active_cand(lhs), active_cand(rhs)) {
        (Some(li), Some(ri)) if cands[li].wide_ty == cands[ri].wide_ty => {
            if let (Some(lw), Some(rw)) = (wide_params[lhs.index()], wide_params[rhs.index()]) {
                func.insts[inst.index()].op = Op::Icmp(cc, lw, rw);
            }
        }
        (Some(li), None) if rhs.index() < orig_vals => {
            if let (Some(lw), Some(imm)) = (wide_params[lhs.index()], consts[rhs.index()]) {
                let c_val = emit_wide_const(func, &cands[li], imm, out);
                func.insts[inst.index()].op = Op::Icmp(cc, lw, c_val);
            }
        }
        (None, Some(ri)) if lhs.index() < orig_vals => {
            if let (Some(rw), Some(imm)) = (wide_params[rhs.index()], consts[lhs.index()]) {
                let c_val = emit_wide_const(func, &cands[ri], imm, out);
                func.insts[inst.index()].op = Op::Icmp(cc, c_val, rw);
            }
        }
        _ => {}
    }
}

fn rewrite_uses(
    func: &mut Function,
    consts: &[Option<i64>],
    cands: &[Candidate],
    val_to_cand: &[Option<usize>],
    wide_params: &[Option<Value>],
) -> (Vec<Option<Value>>, Vec<bool>) {
    let orig_vals = wide_params.len();
    let mut subst = vec![None; orig_vals];
    let mut dead_insts = vec![false; func.insts.len()];
    for b_idx in 0..func.blocks.len() {
        let orig_insts = std::mem::take(&mut func.blocks[b_idx].insts);
        let mut out = Vec::with_capacity(orig_insts.len());
        for inst in orig_insts {
            if let Op::Convert(Convert::Uextend, x) = func.insts[inst.index()].op
                && x.index() < orig_vals
                && let Some(p_wide) = wide_params[x.index()]
                && let Some(res) = func.insts[inst.index()].result
            {
                if func.ty(res) == func.ty(p_wide) {
                    subst[res.index()] = Some(p_wide);
                    dead_insts[inst.index()] = true;
                    func.insts[inst.index()].op = Op::Trap;
                    continue;
                }
                let conv = int_convert_op(func.ty(p_wide), func.ty(res));
                func.insts[inst.index()].op = Op::Convert(conv, p_wide);
            } else {
                rewrite_icmp(
                    func,
                    consts,
                    cands,
                    val_to_cand,
                    wide_params,
                    inst,
                    &mut out,
                );
            }
            out.push(inst);
        }
        func.blocks[b_idx].insts = out;
    }
    (subst, dead_insts)
}

fn insert_fallbacks_and_substitute(
    func: &mut Function,
    cands: &[Candidate],
    kept: &[usize],
    wide_params: &[Option<Value>],
    dead_insts: &[bool],
    subst: &mut [Option<Value>],
) {
    let orig_vals = wide_params.len();
    let mut still_used = vec![false; orig_vals];
    for block in &func.blocks {
        for &inst in &block.insts {
            if inst.index() < dead_insts.len() && dead_insts[inst.index()] {
                continue;
            }
            func.insts[inst.index()].op.for_operands(|v| {
                if v.index() < orig_vals {
                    still_used[v.index()] = true;
                }
            });
        }
    }

    for &idx in kept {
        let c = &cands[idx];
        if still_used[c.param.index()]
            && let Some(p_wide) = wide_params[c.param.index()]
        {
            let narrow_p = func.value(c.narrow_ty);
            let red = func.add(Op::Convert(Convert::Ireduce, p_wide), Some(narrow_p));
            func.blocks[c.block].insts.insert(0, red);
            subst[c.param.index()] = Some(narrow_p);
        }
    }

    for block in &func.blocks {
        for &inst in &block.insts {
            func.insts[inst.index()].op.map_operands(|v| {
                if v.index() < orig_vals {
                    subst[v.index()].unwrap_or(v)
                } else {
                    v
                }
            });
        }
    }
}

fn apply_widening(
    func: &mut Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    cands: &[Candidate],
    val_to_cand: &[Option<usize>],
    kept: &[usize],
) {
    let mut wide_params = vec![None; func.types.len()];
    for &idx in kept {
        let c = &cands[idx];
        wide_params[c.param.index()] = Some(func.value(c.wide_ty));
    }

    rewrite_edges(func, defs, consts, cands, val_to_cand, kept, &wide_params);

    for &idx in kept {
        let c = &cands[idx];
        if let Some(p_wide) = wide_params[c.param.index()] {
            func.blocks[c.block].params[c.param_idx] = p_wide;
        }
    }

    let (mut subst, dead_insts) = rewrite_uses(func, consts, cands, val_to_cand, &wide_params);
    insert_fallbacks_and_substitute(func, cands, kept, &wide_params, &dead_insts, &mut subst);
}
