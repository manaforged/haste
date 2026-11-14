use crate::fold::extend;
use crate::ir::{Binary, Block, BlockData, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::bitmask::{BitmaskPlan, TableLayout, emit_bitmask_test, emit_table_word, known_upper_bound};
use super::emit::push_node;
use super::{Edge, Info, Run, clean_dead_head_insts};

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArgVal {
    Const(i64, Type),
    Invariant(Value),
}

#[derive(Clone)]
enum Outcome {
    Jump { target: Block, args: Vec<ArgVal> },
    Return(Vec<ArgVal>),
    Trap,
}

#[derive(Clone, Copy)]
pub(super) struct PackedLut {
    base: i64,
    span: u64,
    words: [u64; 2],
    two_words: bool,
    signed: bool,
    k: u8,
    bias: u64,
    default_c: Option<i64>,
}

#[derive(Clone, Copy)]
pub(super) enum ColExpr {
    Same(i64),
    OneBit {
        bit_plan: BitmaskPlan,
        val0: i64,
        val1: i64,
        default_is_trap: bool,
    },
    Packed(PackedLut),
}

impl ColExpr {
    pub(super) fn word_count(self) -> usize {
        match self {
            Self::Same(_) => 0,
            Self::OneBit { bit_plan, .. } => 1 + usize::from(bit_plan.two_words),
            Self::Packed(p) => 1 + usize::from(p.two_words),
        }
    }

    fn cost(self, known_u_max: u64) -> (usize, u8, bool, bool) {
        match self {
            Self::Same(_) => (0, 0, false, false),
            Self::OneBit { bit_plan, default_is_trap, .. } => {
                let tier = if bit_plan.two_words { 127 } else { 63 };
                let bounds = !default_is_trap && !(!bit_plan.signed && bit_plan.base == 0 && known_u_max <= tier);
                (self.word_count(), 1, bit_plan.base != 0, bounds)
            }
            Self::Packed(p) => {
                let bounds = p.default_c.is_some() && !(!p.signed && p.base == 0 && known_u_max <= p.span);
                (self.word_count(), p.k, p.base != 0, bounds)
            }
        }
    }
}

fn resolve_val(func: &Function, info: &Info, params: &[Value], args: &[ArgVal], locals: &[(Value, (i64, Type))], v: Value) -> ArgVal {
    if let Some(pos) = params.iter().position(|&p| p == v) {
        return args[pos];
    }
    if let Some(&(_, (imm, ty))) = locals.iter().find(|&&(lv, _)| lv == v) {
        return ArgVal::Const(imm, ty);
    }
    if let Some(imm) = info.consts.get(v.index()).copied().flatten() {
        let ty = func.ty(v);
        return ArgVal::Const(extend(imm, ty, true), ty);
    }
    ArgVal::Invariant(v)
}

fn resolve_edge_candidates(func: &Function, info: &Info, edge: &Edge) -> Vec<Outcome> {
    let init_args: Vec<ArgVal> = if let Some(b) = edge.synth_bool {
        vec![ArgVal::Const(i64::from(b), Type::I8)]
    } else {
        edge.args
            .iter()
            .map(|&v| match info.consts.get(v.index()).copied().flatten() {
                Some(imm) => {
                    let ty = func.ty(v);
                    ArgVal::Const(extend(imm, ty, true), ty)
                }
                None => ArgVal::Invariant(v),
            })
            .collect()
    };
    let mut cands = vec![Outcome::Jump { target: edge.block, args: init_args.clone() }];
    let (mut cur_blk, mut cur_args) = (edge.block, init_args);
    for _ in 0..3 {
        let data = &func.blocks[cur_blk.index()];
        if data.params.len() != cur_args.len() {
            break;
        }
        let Some((&last, body)) = data.insts.split_last() else { break };
        let mut locals: Vec<(Value, (i64, Type))> = Vec::with_capacity(body.len());
        let mut consts_only = true;
        for &inst in body {
            let id = &func.insts[inst.index()];
            let (&Op::Const(imm), Some(res)) = (&id.op, id.result) else {
                consts_only = false;
                break;
            };
            let ty = func.ty(res);
            locals.push((res, (extend(imm, ty, true), ty)));
        }
        if !consts_only {
            break;
        }
        match &func.insts[last.index()].op {
            Op::Trap => {
                cands.push(Outcome::Trap);
                break;
            }
            Op::Return(vals) => {
                let resolved = vals.iter().map(|&v| resolve_val(func, info, &data.params, &cur_args, &locals, v)).collect();
                cands.push(Outcome::Return(resolved));
                break;
            }
            Op::Jump(next, next_args) => {
                let resolved: Vec<ArgVal> = next_args.iter().map(|&v| resolve_val(func, info, &data.params, &cur_args, &locals, v)).collect();
                cands.push(Outcome::Jump { target: *next, args: resolved.clone() });
                cur_blk = *next;
                cur_args = resolved;
            }
            _ => break,
        }
    }
    cands
}

enum UnifiedTarget {
    Return(Vec<Type>),
    Jump(Block, Vec<Type>),
}

enum ColSource {
    Invariant(Value),
    Consts { out_ty: Type, run_vals: Vec<i64>, default_c: Option<i64> },
}

fn match_cols(expected_tys: &[Type], arm_args: &[Vec<ArgVal>], default_args: Option<&[ArgVal]>) -> Option<Vec<ColSource>> {
    let mut cols = Vec::with_capacity(expected_tys.len());
    for (col, &expected_ty) in expected_tys.iter().enumerate() {
        let first = arm_args.first()?.get(col).copied()?;
        match first {
            ArgVal::Invariant(inv) => {
                if arm_args.iter().any(|a| a.get(col) != Some(&ArgVal::Invariant(inv))) {
                    return None;
                }
                if default_args.is_some_and(|da| da.get(col) != Some(&ArgVal::Invariant(inv))) {
                    return None;
                }
                cols.push(ColSource::Invariant(inv));
            }
            ArgVal::Const(_, ty) => {
                if ty != expected_ty || !ty.is_small_int() {
                    return None;
                }
                let mut run_vals = Vec::with_capacity(arm_args.len());
                for a in arm_args {
                    let Some(&ArgVal::Const(imm, aty)) = a.get(col) else { return None };
                    if aty != ty {
                        return None;
                    }
                    run_vals.push(imm);
                }
                let default_c = match default_args {
                    Some(da) => {
                        let Some(&ArgVal::Const(imm, dty)) = da.get(col) else { return None };
                        if dty != ty {
                            return None;
                        }
                        Some(imm)
                    }
                    None => None,
                };
                cols.push(ColSource::Consts { out_ty: ty, run_vals, default_c });
            }
        }
    }
    Some(cols)
}

fn select_unified_target(
    func: &Function,
    arm_cands: &[Vec<Outcome>],
    default_cands: &[Outcome],
    allow_partial: bool,
) -> Option<(UnifiedTarget, Vec<ColSource>, bool)> {
    let first_non_trap = arm_cands.iter().find(|c| !c.iter().all(|o| matches!(o, Outcome::Trap)))?;
    let mut options: Vec<UnifiedTarget> = Vec::new();
    if first_non_trap.iter().any(|o| matches!(o, Outcome::Return(_))) {
        let ret_tys: Vec<Type> = func.signature.returns.iter().map(|p| p.value_type).collect();
        options.push(UnifiedTarget::Return(ret_tys));
    }
    for o in first_non_trap {
        if let Outcome::Jump { target, .. } = o {
            let tys: Vec<Type> = func.blocks[target.index()].params.iter().map(|&p| func.ty(p)).collect();
            options.push(UnifiedTarget::Jump(*target, tys));
        }
    }
    fn find_match<'a>(cands: &'a [Outcome], ut: &UnifiedTarget) -> Option<Option<&'a [ArgVal]>> {
        match ut {
            UnifiedTarget::Return(tys) => cands.iter().find_map(|o| match o {
                Outcome::Return(args) if args.len() == tys.len() => Some(Some(args.as_slice())),
                Outcome::Trap => Some(None),
                _ => None,
            }),
            UnifiedTarget::Jump(t, tys) => cands.iter().find_map(|o| match o {
                Outcome::Jump { target, args } if target == t && args.len() == tys.len() => Some(Some(args.as_slice())),
                Outcome::Trap => Some(None),
                _ => None,
            }),
        }
    }
    for ut in options {
        let expected_tys = match &ut {
            UnifiedTarget::Return(tys) | UnifiedTarget::Jump(_, tys) => tys.as_slice(),
        };
        let Some(arm_args_opt): Option<Vec<Option<&[ArgVal]>>> = arm_cands.iter().map(|c| find_match(c, &ut)).collect() else {
            continue;
        };
        let Some(first_some) = arm_args_opt.iter().flatten().copied().next() else { continue };
        let arm_args: Vec<Vec<ArgVal>> = arm_args_opt.into_iter().map(|opt| opt.unwrap_or(first_some).to_vec()).collect();
        let (default_args, default_matched) = match find_match(default_cands, &ut) {
            Some(Some(args)) => (Some(args), true),
            Some(None) => (None, true),
            None if allow_partial => (None, false),
            None => continue,
        };
        if let Some(cols) = match_cols(expected_tys, &arm_args, default_args) {
            return Some((ut, cols, default_matched));
        }
    }
    None
}

pub(super) fn uval_of(c: i64, ty: Type) -> u64 {
    extend(c, ty, false) as u64
}

fn pack_table(
    out_ty: Type,
    (base, signed, table_len): (i64, bool, usize),
    table: &[i64],
    default_c: Option<i64>,
    known_u_max: u64,
) -> Option<ColExpr> {
    if table_len == 0 || table_len > 128 {
        return None;
    }
    let out_mask = out_ty.unsigned_max();
    let mut distinct: Vec<(u64, i64)> = Vec::new();
    for &v in table {
        let uv = uval_of(v, out_ty);
        if !distinct.iter().any(|&(u, _)| u == uv) {
            distinct.push((uv, v));
            if distinct.len() > 256 {
                return None;
            }
        }
    }
    if let Some(dc) = default_c {
        let ud = uval_of(dc, out_ty);
        if !distinct.iter().any(|&(u, _)| u == ud) {
            distinct.push((ud, dc));
        }
    }
    if distinct.len() == 1 {
        return Some(ColExpr::Same(distinct[0].1));
    }
    let mut best: Option<ColExpr> = None;
    let mut consider = |cand: ColExpr| {
        if best.is_none_or(|b| cand.cost(known_u_max) < b.cost(known_u_max)) {
            best = Some(cand);
        }
    };
    if distinct.len() == 2 {
        let (val0, val1) = match default_c {
            Some(dc) => {
                let ud = uval_of(dc, out_ty);
                let alt = distinct.iter().find(|&&(u, _)| u != ud).map_or(dc, |&(_, v)| v);
                (dc, alt)
            }
            None => (distinct[0].1, distinct[1].1),
        };
        let u1 = uval_of(val1, out_ty);
        let mut words = [0_u64; 2];
        for (i, &val) in table.iter().enumerate() {
            if uval_of(val, out_ty) == u1 {
                words[i >> 6] |= 1_u64 << (i & 63);
            }
        }
        consider(ColExpr::OneBit {
            bit_plan: BitmaskPlan { base, span: (table_len - 1) as u64, words, two_words: table_len > 64, signed },
            val0,
            val1,
            default_is_trap: default_c.is_none(),
        });
    }
    let min_u = table.iter().map(|&v| uval_of(v, out_ty)).min()?;
    let min_s = table.iter().map(|&v| extend(v, out_ty, true) as u64).min()?;
    for bias in [0_u64, min_u, min_s] {
        let max_delta = table.iter().map(|&v| uval_of(v, out_ty).wrapping_sub(bias) & out_mask).max().unwrap_or(0);
        for k in [2_u8, 4, 8] {
            if max_delta < (1_u64 << k) && table_len * usize::from(k) <= 128 {
                let mut words = [0_u64; 2];
                for (i, &v) in table.iter().enumerate() {
                    let delta = uval_of(v, out_ty).wrapping_sub(bias) & out_mask;
                    let bit_pos = i * usize::from(k);
                    words[bit_pos >> 6] |= delta << (bit_pos & 63);
                }
                consider(ColExpr::Packed(PackedLut {
                    base,
                    span: (table_len - 1) as u64,
                    words,
                    two_words: table_len * usize::from(k) > 64,
                    signed,
                    k,
                    bias,
                    default_c,
                }));
                break;
            }
        }
    }
    best
}

pub(super) fn build_col_lut(ty: Type, out_ty: Type, intervals: &[(u64, u64, i64)], default_c: Option<i64>, known_u_max: u64) -> Option<ColExpr> {
    if intervals.is_empty() {
        return None;
    }
    let min_lo = intervals.first()?.0;
    let max_hi = intervals.last()?.1;
    let fallback = default_c.unwrap_or(intervals[0].2);
    let mut best: Option<ColExpr> = None;
    let mut consider = |cand: ColExpr| {
        if best.is_none_or(|b| cand.cost(known_u_max) < b.cost(known_u_max)) {
            best = Some(cand);
        }
    };
    if max_hi - min_lo < 128 {
        let mut bases = vec![min_lo];
        if min_lo > 0 && max_hi < 128 && default_c.is_some() {
            bases.push(0);
        }
        for base in bases {
            let mut lens = vec![(max_hi - base + 1) as usize];
            if base == 0 && default_c.is_some() && max_hi <= known_u_max && known_u_max < 128 {
                lens.push((known_u_max + 1) as usize);
            }
            for len in lens {
                let mut table = vec![fallback; len];
                for &(lo, hi, val) in intervals {
                    let s = (lo - base) as usize;
                    let e = ((hi - base) as usize).min(len - 1);
                    if s <= e {
                        table[s..=e].fill(val);
                    }
                }
                if let Some(expr) = pack_table(out_ty, (base as i64, false, len), &table, default_c, known_u_max) {
                    consider(expr);
                }
            }
        }
    }
    let sign_bit = 1_u64 << (ty.bits() - 1);
    let mut segs: Vec<(i64, i64, i64)> = Vec::with_capacity(intervals.len() + 1);
    for &(lo, hi, val) in intervals {
        if lo < sign_bit && hi >= sign_bit {
            segs.push((extend(lo as i64, ty, true), extend((sign_bit - 1) as i64, ty, true), val));
            segs.push((extend(sign_bit as i64, ty, true), extend(hi as i64, ty, true), val));
        } else {
            segs.push((extend(lo as i64, ty, true), extend(hi as i64, ty, true), val));
        }
    }
    if let (Some(s_min), Some(s_max)) = (segs.iter().map(|&(l, _, _)| l).min(), segs.iter().map(|&(_, h, _)| h).max()) {
        let s_span = s_max.abs_diff(s_min);
        if s_span < 128 {
            let len = (s_span + 1) as usize;
            let mut table = vec![fallback; len];
            for (lo, hi, val) in segs {
                table[lo.abs_diff(s_min) as usize..=hi.abs_diff(s_min) as usize].fill(val);
            }
            if let Some(expr) = pack_table(out_ty, (s_min, true, len), &table, default_c, known_u_max) {
                consider(expr);
            }
        }
    }
    best
}

pub(super) fn emit_col_expr(func: &mut Function, (v, ty): (Value, Type), out_ty: Type, expr: ColExpr, known_u_max: u64, insts: &mut Vec<Inst>) -> Value {
    match expr {
        ColExpr::Same(c) => {
            let res = func.value(out_ty);
            insts.push(func.add(Op::Const(extend(c, out_ty, true)), Some(res)));
            res
        }
        ColExpr::OneBit { bit_plan, val0, val1, default_is_trap } => {
            let cond = emit_bitmask_test(func, (v, ty), bit_plan, known_u_max, !default_is_trap, insts);
            let (u0, u1) = (uval_of(val0, out_ty), uval_of(val1, out_ty));
            if u0 == 0 && u1 == 1 {
                if out_ty == Type::I8 {
                    cond
                } else {
                    let res = func.value(out_ty);
                    insts.push(func.add(Op::Convert(Convert::Uextend, cond), Some(res)));
                    res
                }
            } else if u0 == 1 && u1 == 0 {
                let (one8, flipped) = (func.value(Type::I8), func.value(Type::I8));
                insts.push(func.add(Op::Const(1), Some(one8)));
                insts.push(func.add(Op::Binary(Binary::Bxor, cond, one8), Some(flipped)));
                if out_ty == Type::I8 {
                    flipped
                } else {
                    let res = func.value(out_ty);
                    insts.push(func.add(Op::Convert(Convert::Uextend, flipped), Some(res)));
                    res
                }
            } else {
                let (c1, c0, res) = (func.value(out_ty), func.value(out_ty), func.value(out_ty));
                insts.push(func.add(Op::Const(extend(val1, out_ty, true)), Some(c1)));
                insts.push(func.add(Op::Const(extend(val0, out_ty, true)), Some(c0)));
                insts.push(func.add(Op::Select(cond, c1, c0), Some(res)));
                res
            }
        }
        ColExpr::Packed(p) => {
            let per_word = 64 / u64::from(p.k);
            let layout = TableLayout {
                base: p.base,
                span: p.span,
                words: p.words,
                two_words: p.two_words,
                signed: p.signed,
                per_word,
                bounds_elide_max: p.span,
            };
            let (diff, in_range, word) = emit_table_word(func, (v, ty), layout, known_u_max, p.default_c.is_some(), insts);
            let sh_bits = i64::from(u32::from(p.k).trailing_zeros());
            let (sh_c, shift, shifted) = (func.value(Type::I64), func.value(Type::I64), func.value(Type::I64));
            insts.push(func.add(Op::Const(sh_bits), Some(sh_c)));
            insts.push(func.add(Op::Binary(Binary::Ishl, diff, sh_c), Some(shift)));
            insts.push(func.add(Op::Binary(Binary::Ushr, word, shift), Some(shifted)));
            let raw = if p.k == 8 && p.bias == 0 && out_ty == Type::I8 {
                let res = func.value(Type::I8);
                insts.push(func.add(Op::Convert(Convert::Ireduce, shifted), Some(res)));
                res
            } else {
                let (mask_c, masked) = (func.value(Type::I64), func.value(Type::I64));
                insts.push(func.add(Op::Const(((1_u64 << p.k) - 1) as i64), Some(mask_c)));
                insts.push(func.add(Op::Binary(Binary::Band, shifted, mask_c), Some(masked)));
                let added = if p.bias == 0 {
                    masked
                } else {
                    let (bias_c, sum) = (func.value(Type::I64), func.value(Type::I64));
                    insts.push(func.add(Op::Const(p.bias as i64), Some(bias_c)));
                    insts.push(func.add(Op::Binary(Binary::Iadd, masked, bias_c), Some(sum)));
                    sum
                };
                if out_ty == Type::I64 {
                    added
                } else {
                    let res = func.value(out_ty);
                    insts.push(func.add(Op::Convert(Convert::Ireduce, added), Some(res)));
                    res
                }
            };
            if let (Some(in_range), Some(dc)) = (in_range, p.default_c) {
                let (dc_v, res) = (func.value(out_ty), func.value(out_ty));
                insts.push(func.add(Op::Const(extend(dc, out_ty, true)), Some(dc_v)));
                insts.push(func.add(Op::Select(in_range, raw, dc_v), Some(res)));
                res
            } else {
                raw
            }
        }
    }
}

pub(super) fn try_emit_runs_lut(
    func: &mut Function,
    info: &Info,
    at: (usize, Inst),
    v: Value,
    runs: &[Run],
    default_edge: &Edge,
    head_dead: &[Inst],
) -> bool {
    let ty = func.ty(v);
    if !ty.is_small_int() || runs.is_empty() {
        return false;
    }
    let min_lo = runs[0].lo;
    let max_hi = runs[runs.len() - 1].hi;
    let covered: u64 = runs.iter().map(|r| r.hi.saturating_sub(r.lo).saturating_add(1)).sum();
    let contiguous = max_hi - min_lo < 128 && covered == (max_hi - min_lo + 1);
    let mut unique_edges: Vec<Edge> = Vec::new();
    let mut run_edge_idx = Vec::with_capacity(runs.len());
    for r in runs {
        let idx = if let Some(pos) = unique_edges.iter().position(|e| e.block == r.hit.block && e.args == r.hit.args && e.synth_bool == r.hit.synth_bool) {
            pos
        } else {
            unique_edges.push(r.hit.clone());
            unique_edges.len() - 1
        };
        run_edge_idx.push(idx);
    }
    let edge_cands: Vec<Vec<Outcome>> = unique_edges.iter().map(|e| resolve_edge_candidates(func, info, e)).collect();
    let arm_cands: Vec<Vec<Outcome>> = run_edge_idx.iter().map(|&idx| edge_cands[idx].clone()).collect();
    let default_cands = resolve_edge_candidates(func, info, default_edge);
    let allow_partial = contiguous && runs.len() >= 3;
    let Some((target, cols, default_matched)) = select_unified_target(func, &arm_cands, &default_cands, allow_partial) else {
        return false;
    };
    let known_u_max = known_upper_bound(func, info, v);
    let mut plans = Vec::with_capacity(cols.len());
    let mut total_words = 0_usize;
    for col in cols {
        match col {
            ColSource::Invariant(val) => plans.push(Err(val)),
            ColSource::Consts { out_ty, run_vals, default_c } => {
                let intervals: Vec<(u64, u64, i64)> = runs.iter().zip(run_vals).map(|(r, c)| (r.lo, r.hi, c)).collect();
                let Some(expr) = build_col_lut(ty, out_ty, &intervals, default_c, known_u_max) else {
                    return false;
                };
                total_words += expr.word_count();
                if total_words > 2 {
                    return false;
                }
                plans.push(Ok((out_ty, expr)));
            }
        }
    }
    if total_words > 0 && runs.len() < 3 {
        return false;
    }
    if total_words >= 2 && runs.len() < 4 {
        return false;
    }
    clean_dead_head_insts(func, info, at.0, head_dead);
    let emit_lut_term = |func: &mut Function, insts: &mut Vec<Inst>| {
        let mut out_args = Vec::with_capacity(plans.len());
        for &item in &plans {
            match item {
                Err(val) => out_args.push(val),
                Ok((out_ty, expr)) => out_args.push(emit_col_expr(func, (v, ty), out_ty, expr, known_u_max, insts)),
            }
        }
        let term = match &target {
            UnifiedTarget::Return(_) => Op::Return(out_args),
            UnifiedTarget::Jump(j, _) => Op::Jump(*j, out_args),
        };
        insts.push(func.add(term, None));
    };
    if !default_matched {
        let v64 = if ty == Type::I64 {
            v
        } else {
            let res = func.value(Type::I64);
            let inst = func.add(Op::Convert(Convert::Uextend, v), Some(res));
            super::push_before_last(func, at.0, inst);
            res
        };
        let diff = if min_lo == 0 {
            v64
        } else {
            let (base_c, sub) = (func.value(Type::I64), func.value(Type::I64));
            let c_i = func.add(Op::Const(min_lo as i64), Some(base_c));
            let s_i = func.add(Op::Binary(Binary::Isub, v64, base_c), Some(sub));
            super::push_before_last(func, at.0, c_i);
            super::push_before_last(func, at.0, s_i);
            sub
        };
        let (span_c, in_range) = (func.value(Type::I64), func.value(Type::I8));
        let sc_i = func.add(Op::Const((max_hi - min_lo) as i64), Some(span_c));
        let ir_i = func.add(Op::Icmp(IntCC::UnsignedLessThanOrEqual, diff, span_c), Some(in_range));
        super::push_before_last(func, at.0, sc_i);
        super::push_before_last(func, at.0, ir_i);
        let mut hit_insts = Vec::new();
        emit_lut_term(func, &mut hit_insts);
        let hit_blk = Block(func.blocks.len() as u32);
        func.blocks.push(BlockData { params: Vec::new(), insts: hit_insts });
        let def_args = if let Some(b) = default_edge.synth_bool {
            let val = func.value(Type::I8);
            let inst = func.add(Op::Const(i64::from(b)), Some(val));
            super::push_before_last(func, at.0, inst);
            vec![val]
        } else {
            default_edge.args.clone()
        };
        let brif = func.add(Op::Brif(in_range, hit_blk, Vec::new(), default_edge.block, def_args), None);
        push_node(func, Some(at), vec![brif]);
        return true;
    }
    let mut insts = Vec::new();
    emit_lut_term(func, &mut insts);
    push_node(func, Some(at), insts);
    true
}
