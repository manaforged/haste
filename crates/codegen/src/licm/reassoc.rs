use rustc_hash::FxHashMap as HashMap;

use crate::cfg::{self, Cfg, Loop};
use crate::ir::{Binary, Block, Function, Inst, Op, Type, Value};

#[derive(Default)]
struct AddChain {
    terms: Vec<(Value, bool)>,
    absorbed: Vec<Inst>,
    const_sum: i64,
    const_count: u8,
    single_const: Option<(Value, bool, i64)>,
    first_const_pos: usize,
    right_expanded: bool,
}

impl AddChain {
    fn push_const(&mut self, val: Value, neg: bool, imm: i64) {
        self.const_sum = if neg { self.const_sum.wrapping_sub(imm) } else { self.const_sum.wrapping_add(imm) };
        self.const_count = self.const_count.saturating_add(1);
        self.single_const = if self.const_count == 1 { Some((val, neg, imm)) } else { None };
        if self.const_count == 1 {
            self.first_const_pos = self.terms.len();
        }
    }
}

struct ReassocCtx<'a> {
    func: &'a Function,
    cfg: &'a Cfg,
    each: &'a Loop,
    uses: &'a [u32],
    rpo_rank: &'a [usize],
    home: &'a [Option<Block>],
    is_const: &'a [bool],
    consts: &'a [Option<i64>],
    const_adds: &'a HashMap<(Value, u64), Vec<(Block, Value)>>,
}

struct AddPlan {
    chain: AddChain,
    masked_const: u64,
    pos_idx: Option<usize>,
}

struct ReassocState<'a> {
    func: &'a mut Function,
    cfg: &'a Cfg,
    each: &'a Loop,
    rpo_rank: &'a [usize],
    defs: Vec<Option<Inst>>,
    uses: Vec<u32>,
    home: &'a mut Vec<Option<Block>>,
    is_const: &'a mut Vec<bool>,
    consts: Vec<Option<i64>>,
    const_adds: HashMap<(Value, u64), Vec<(Block, Value)>>,
}

fn term_rank(ctx: &ReassocCtx<'_>, val: Value) -> (bool, usize) {
    match ctx.home.get(val.index()).copied().flatten() {
        Some(b) => (ctx.each.body.get(b.index()).copied().unwrap_or(false), ctx.rpo_rank[b.index()]),
        None => (false, 0),
    }
}

fn record_block_const_adds(
    func: &Function,
    blk: Block,
    home: &[Option<Block>],
    consts: &[Option<i64>],
    map: &mut HashMap<(Value, u64), Vec<(Block, Value)>>,
) {
    for &inst in &func.blocks[blk.index()].insts {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        let ty = func.ty(res);
        if !ty.is_small_int() {
            continue;
        }
        let Op::Binary(bop @ (Binary::Iadd | Binary::Isub), a, b) = data.op else { continue };
        let mask = ty.unsigned_max();
        let ca = consts.get(a.index()).copied().flatten();
        let cb = consts.get(b.index()).copied().flatten();
        let key = match (bop, ca, cb) {
            (_, _, Some(c)) => (a, (if bop == Binary::Isub { c.wrapping_neg() } else { c }) as u64 & mask),
            (Binary::Iadd, Some(c), _) => (b, (c as u64) & mask),
            _ => continue,
        };
        if home.get(res.index()).copied().flatten() == Some(blk) {
            map.entry(key).or_default().push((blk, res));
        }
    }
}

fn flatten_add(ctx: &ReassocCtx<'_>, defs: &[Option<Inst>], blk: Block, ty: Type, root: Inst) -> Option<AddChain> {
    let Op::Binary(bop @ (Binary::Iadd | Binary::Isub), lhs, rhs) = ctx.func.insts[root.index()].op else { return None };
    let mut work = vec![(rhs, bop == Binary::Isub, true), (lhs, false, false)];
    let mut chain = AddChain::default();
    while let Some((val, neg, is_right)) = work.pop() {
        if chain.terms.len() + usize::from(chain.const_count) >= 12 {
            return None;
        }
        if ctx.is_const.get(val.index()).copied().unwrap_or(false) {
            chain.push_const(val, neg, ctx.consts.get(val.index()).copied().flatten()?);
            continue;
        }
        let child = defs.get(val.index()).copied().flatten().filter(|_| {
            ctx.home.get(val.index()).copied().flatten() == Some(blk)
                && ctx.uses.get(val.index()).copied().unwrap_or(0) == 1
                && ctx.func.ty(val) == ty
        });
        if let Some(c_inst) = child
            && let Op::Binary(cbop @ (Binary::Iadd | Binary::Isub), a, b) = ctx.func.insts[c_inst.index()].op
        {
            chain.absorbed.push(c_inst);
            chain.right_expanded |= is_right;
            work.push((b, if cbop == Binary::Isub { !neg } else { neg }, true));
            work.push((a, neg, false));
        } else {
            chain.terms.push((val, neg));
        }
    }
    (!chain.absorbed.is_empty() && (chain.terms.len() >= 2 || chain.const_count >= 2)).then_some(chain)
}

fn try_reuse_const_add(ctx: &ReassocCtx<'_>, chain: &mut AddChain, blk: Block, masked_const: u64) -> bool {
    let blk_rank = (true, ctx.rpo_rank[blk.index()]);
    for (idx, &(v, neg)) in chain.terms.iter().enumerate() {
        if neg || term_rank(ctx, v) >= blk_rank {
            continue;
        }
        let Some(cands) = ctx.const_adds.get(&(v, masked_const)) else { continue };
        if let Some(&(_, existing)) = cands.iter().find(|&&(b, _)| b != blk && cfg::dominates(&ctx.cfg.idom, b, blk)) {
            chain.terms[idx] = (existing, false);
            chain.const_sum = 0;
            chain.single_const = None;
            return true;
        }
    }
    false
}

fn already_sub_grouped(ctx: &ReassocCtx<'_>, chain: &AddChain, orig: &[(Value, bool)], root: Inst, pos_idx: usize) -> bool {
    let pos_len = orig.len() - pos_idx;
    let Op::Binary(Binary::Isub, _, rhs) = ctx.func.insts[root.index()].op else { return false };
    orig[..pos_len] == chain.terms[pos_idx..]
        && orig[pos_len..] == chain.terms[..pos_idx]
        && chain.absorbed.iter().any(|&a| ctx.func.insts[a.index()].result == Some(rhs))
}

fn add_rewrite_profitable(ctx: &ReassocCtx<'_>, chain: &AddChain, masked_const: u64) -> bool {
    let folds_const = chain.const_count >= 2 || (chain.const_count == 1 && masked_const == 0);
    if folds_const {
        return true;
    }
    let max_rank = chain.terms.iter().map(|&(v, _)| term_rank(ctx, v)).max().unwrap_or((false, 0));
    let early = chain.terms.iter().filter(|&&(v, _)| term_rank(ctx, v) < max_rank).count();
    let has_inv = chain.terms.iter().any(|&(v, _)| !term_rank(ctx, v).0);
    early >= 2 || (masked_const != 0 && early >= 1 && has_inv)
}

fn same_add_shape(ctx: &ReassocCtx<'_>, chain: &AddChain, orig: &[(Value, bool)], root: Inst, pos_idx: Option<usize>) -> bool {
    if let Some(p) = pos_idx {
        return already_sub_grouped(ctx, chain, orig, root, p);
    }
    !chain.right_expanded && orig == chain.terms.as_slice() && (chain.const_count == 0 || chain.first_const_pos <= 1)
}

fn normalize_leading_neg(terms: &mut [(Value, bool)], masked_const: u64) -> Option<Option<usize>> {
    if masked_const != 0 || !terms.first().is_some_and(|t| t.1) {
        return Some(None);
    }
    let p = terms.iter().position(|t| !t.1)?;
    if p == 1 {
        terms.swap(0, 1);
        Some(None)
    } else {
        Some(Some(p))
    }
}

fn plan_add_rewrite(ctx: &ReassocCtx<'_>, mut chain: AddChain, blk: Block, ty: Type, root: Inst) -> Option<AddPlan> {
    let mut masked_const = (chain.const_sum as u64) & ty.unsigned_max();
    if chain.terms.len() <= 1 && masked_const == 0 {
        return None;
    }
    let orig = chain.terms.clone();
    chain.terms.sort_by_key(|&(v, neg)| (term_rank(ctx, v), neg));
    let matched = masked_const != 0 && try_reuse_const_add(ctx, &mut chain, blk, masked_const);
    if matched {
        masked_const = 0;
        chain.terms.sort_by_key(|&(v, neg)| (term_rank(ctx, v), neg));
    }
    if !matched && !add_rewrite_profitable(ctx, &chain, masked_const) {
        return None;
    }
    let pos_idx = normalize_leading_neg(&mut chain.terms, masked_const)?;
    let folds_const = !matched && (chain.const_count >= 2 || (chain.const_count == 1 && masked_const == 0));
    let unchanged = !folds_const && !matched && same_add_shape(ctx, &chain, &orig, root, pos_idx);
    (!unchanged).then_some(AddPlan { chain, masked_const, pos_idx })
}

fn emit_add_step(func: &mut Function, ty: Type, prefix: &mut Vec<Inst>, op: Op) -> Value {
    let val = func.value(ty);
    prefix.push(func.add(op, Some(val)));
    val
}

fn fold_tail_ops(func: &mut Function, prefix: &mut Vec<Inst>, ops: &mut Vec<Op>, tail: &[(Value, bool)], mut acc: Value) {
    let ty = func.ty(acc);
    for &(v, neg) in tail {
        if let Some(prev) = ops.pop() {
            acc = emit_add_step(func, ty, prefix, prev);
        }
        ops.push(Op::Binary(if neg { Binary::Isub } else { Binary::Iadd }, acc, v));
    }
}

fn build_const_add_ops(
    func: &mut Function,
    plan: &AddPlan,
    ty: Type,
    prefix: &mut Vec<Inst>,
    new_const: &mut Option<(Value, i64)>,
) -> Vec<Op> {
    let terms = &plan.chain.terms;
    if terms.is_empty() {
        return vec![Op::Const(crate::fold::extend(plan.masked_const as i64, ty, true))];
    }
    let mut ops = Vec::with_capacity(terms.len() + 1);
    let need_pos = terms[0].1;
    let (cv, cneg) = match plan.chain.single_const {
        Some((v, neg, _)) if !need_pos || !neg => (v, neg),
        _ => {
            let imm = crate::fold::extend(plan.masked_const as i64, ty, true);
            let v = emit_add_step(func, ty, prefix, Op::Const(imm));
            *new_const = Some((v, imm));
            (v, false)
        }
    };
    if need_pos {
        fold_tail_ops(func, prefix, &mut ops, terms, cv);
    } else {
        ops.push(Op::Binary(if cneg { Binary::Isub } else { Binary::Iadd }, terms[0].0, cv));
        fold_tail_ops(func, prefix, &mut ops, &terms[1..], terms[0].0);
    }
    ops
}

fn build_zero_const_ops(func: &mut Function, plan: &AddPlan, ty: Type, prefix: &mut Vec<Inst>) -> Vec<Op> {
    let terms = &plan.chain.terms;
    let mut ops = Vec::with_capacity(terms.len());
    let Some(p) = plan.pos_idx else {
        fold_tail_ops(func, prefix, &mut ops, &terms[1..], terms[0].0);
        return ops;
    };
    let mut neg_acc = terms[0].0;
    for &(v, _) in &terms[1..p] {
        neg_acc = emit_add_step(func, ty, prefix, Op::Binary(Binary::Iadd, neg_acc, v));
    }
    ops.push(Op::Binary(Binary::Isub, terms[p].0, neg_acc));
    fold_tail_ops(func, prefix, &mut ops, &terms[p + 1..], terms[p].0);
    ops
}

fn apply_add_plan(func: &mut Function, plan: AddPlan, root: Inst, ty: Type) -> (Vec<Inst>, Op, Option<(Value, i64)>) {
    let mut prefix = Vec::new();
    let mut new_const = None;
    let mut ops = if plan.masked_const != 0 {
        build_const_add_ops(func, &plan, ty, &mut prefix, &mut new_const)
    } else {
        build_zero_const_ops(func, &plan, ty, &mut prefix)
    };
    let final_op = ops.pop().unwrap_or(Op::Trap);
    func.insts[root.index()].op = final_op.clone();
    (prefix, final_op, new_const)
}

fn try_reassociate_inst(st: &mut ReassocState<'_>, blk: Block, inst: Inst, removed: &mut [bool]) -> Option<Vec<Inst>> {
    let res = st.func.insts[inst.index()].result?;
    let ty = st.func.ty(res);
    if !ty.is_small_int() {
        return None;
    }
    let ctx = ReassocCtx {
        func: st.func,
        cfg: st.cfg,
        each: st.each,
        uses: &st.uses,
        rpo_rank: st.rpo_rank,
        home: st.home,
        is_const: st.is_const,
        consts: &st.consts,
        const_adds: &st.const_adds,
    };
    let chain = flatten_add(&ctx, &st.defs, blk, ty, inst)?;
    let plan = plan_add_rewrite(&ctx, chain, blk, ty, inst)?;
    for &dead in &plan.chain.absorbed {
        removed[dead.index()] = true;
    }
    let (prefix, final_op, new_const) = apply_add_plan(st.func, plan, inst, ty);
    st.home.resize(st.func.types.len(), Some(blk));
    st.is_const.resize(st.func.types.len(), false);
    st.consts.resize(st.func.types.len(), None);
    st.defs.resize(st.func.types.len(), None);
    if let Op::Const(imm) = final_op {
        st.is_const[res.index()] = true;
        st.consts[res.index()] = Some(imm);
    }
    if let Some((cv, imm)) = new_const {
        st.is_const[cv.index()] = true;
        st.consts[cv.index()] = Some(imm);
    }
    Some(prefix)
}

fn reassociate_block_adds(st: &mut ReassocState<'_>, blk: Block) -> bool {
    let old_insts = st.func.blocks[blk.index()].insts.clone();
    let orig_len = st.func.insts.len();
    let mut removed = vec![false; orig_len];
    let mut before_root: Vec<Vec<Inst>> = vec![Vec::new(); orig_len];
    let mut changed = false;
    for &inst in old_insts.iter().rev() {
        if !removed[inst.index()]
            && let Some(prefix) = try_reassociate_inst(st, blk, inst, &mut removed)
        {
            before_root[inst.index()] = prefix;
            changed = true;
        }
    }
    if changed {
        let mut rebuilt = Vec::with_capacity(old_insts.len());
        for inst in old_insts {
            if !removed[inst.index()] {
                rebuilt.extend(before_root[inst.index()].iter().copied());
                rebuilt.push(inst);
            }
        }
        st.func.blocks[blk.index()].insts = rebuilt;
    }
    changed
}

pub(super) fn reassociate_loop_adds(
    func: &mut Function,
    cfg: &Cfg,
    each: &Loop,
    home: &mut Vec<Option<Block>>,
    is_const: &mut Vec<bool>,
) -> bool {
    let uses = func.use_counts();
    let mut defs = vec![None; func.types.len()];
    let mut consts = vec![None; func.types.len()];
    for (idx, data) in func.insts.iter().enumerate() {
        if let Some(res) = data.result {
            defs[res.index()] = Some(Inst(idx as u32));
        }
        if let (&Op::Const(imm), Some(res)) = (&data.op, data.result) {
            consts[res.index()] = Some(imm);
        }
    }
    let rpo_rank = Cfg::ranks(&cfg.rpo, func.blocks.len());
    let mut st = ReassocState { func, cfg, each, rpo_rank: &rpo_rank, defs, uses, home, is_const, consts, const_adds: HashMap::default() };
    let mut changed = false;
    for &blk in &cfg.rpo {
        if each.body.get(blk.index()).copied().unwrap_or(false) {
            changed |= reassociate_block_adds(&mut st, blk);
        }
        record_block_const_adds(st.func, blk, st.home, &st.consts, &mut st.const_adds);
    }
    changed
}
