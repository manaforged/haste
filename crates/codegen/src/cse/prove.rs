use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Value};
use crate::merge::rename;

const COND_PEEL_DEPTH: u8 = 4;
const DOM_PROVE_STEPS: u8 = 24;

/// Operand or constant represented on a branch edge.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeVal {
    Const(i64),
    Val(Value),
}

/// Predicate context mapping values through substitutions, constants, and known booleans.
pub(crate) struct CmpEnv<'a> {
    pub(crate) defs: &'a [Option<Inst>],
    pub(crate) consts: &'a [Option<i64>],
    pub(crate) subst: &'a [Value],
    pub(crate) bool_val: Option<&'a [bool]>,
}

impl CmpEnv<'_> {
    /// Resolves `v` to its canonical representative in `subst`.
    pub(crate) fn find(&self, v: Value) -> Value {
        crate::merge::find(self.subst, v)
    }

    fn known(&self, v: Value) -> Option<i64> {
        self.consts.get(self.find(v).index()).copied().flatten()
    }

    /// Returns `v` as a constant `EdgeVal` when known, or its canonical `Value`.
    pub(crate) fn edge_val(&self, v: Value) -> EdgeVal {
        self.known(v)
            .map_or_else(|| EdgeVal::Val(self.find(v)), EdgeVal::Const)
    }

    fn is_bool(&self, func: &Function, v: Value) -> bool {
        self.bool_val.map_or_else(
            || {
                self.defs
                    .get(v.index())
                    .copied()
                    .flatten()
                    .is_some_and(|i| {
                        matches!(func.insts[i.index()].op, Op::Icmp(..) | Op::Fits(..))
                    })
            },
            |b| b.get(v.index()).copied().unwrap_or(false),
        )
    }

    /// Peels zero-extensions and boolean xor-1 inversions off `cond`, returning the inner op, polarity, and depth.
    pub(crate) fn peel_cond<'a>(
        &self,
        func: &'a Function,
        mut cond: Value,
        mut taken: bool,
        mut depth: u8,
    ) -> Option<(&'a Op, bool, u8)> {
        while depth < COND_PEEL_DEPTH {
            let op = &func.insts[self.defs.get(cond.index()).copied().flatten()?.index()].op;
            match *op {
                Op::Convert(Convert::Uextend, inner) if func.ty(inner).is_small_int() => {
                    cond = inner;
                }
                Op::Binary(Binary::Bxor, a, b) => {
                    let (i, _) = [(a, b), (b, a)].into_iter().find(|&(i, k)| {
                        self.is_bool(func, i)
                            && self
                                .known(k)
                                .is_some_and(|v| crate::fold::extend(v, func.ty(k), false) == 1)
                    })?;
                    (cond, taken) = (i, !taken);
                }
                _ => return Some((op, taken, depth)),
            }
            depth += 1;
        }
        None
    }
}

/// Returns whether holding `hcc(hx, hy)` implies `wcc(wx, wy)` at integer type `ty`.
pub(crate) fn cmp_proves(
    ty: Type,
    (mut hcc, mut hx, mut hy): (IntCC, EdgeVal, EdgeVal),
    (mut wcc, mut wx, mut wy): (IntCC, EdgeVal, EdgeVal),
) -> bool {
    use IntCC::*;
    if let (EdgeVal::Const(a), EdgeVal::Const(b)) = (wx, wy) {
        return crate::fold::holds(wcc, ty, a, b);
    }
    if matches!((hx, hy), (EdgeVal::Const(_), EdgeVal::Val(_))) {
        (hcc, hx, hy) = (hcc.swapped(), hy, hx);
    }
    if matches!((wx, wy), (EdgeVal::Const(_), EdgeVal::Val(_))) {
        (wcc, wx, wy) = (wcc.swapped(), wy, wx);
    }
    let cc_ok = |h, w| {
        h == w
            || matches!(
                (h, w),
                (
                    Equal,
                    UnsignedLessThanOrEqual
                        | UnsignedGreaterThanOrEqual
                        | SignedLessThanOrEqual
                        | SignedGreaterThanOrEqual
                ) | (UnsignedLessThan, UnsignedLessThanOrEqual | NotEqual)
                    | (UnsignedGreaterThan, UnsignedGreaterThanOrEqual | NotEqual)
                    | (SignedLessThan, SignedLessThanOrEqual | NotEqual)
                    | (SignedGreaterThan, SignedGreaterThanOrEqual | NotEqual)
            )
    };
    if ((hx, hy) == (wx, wy) && cc_ok(hcc, wcc))
        || ((hy, hx) == (wx, wy) && cc_ok(hcc.swapped(), wcc))
    {
        return true;
    }
    let (EdgeVal::Val(hv), EdgeVal::Const(hc), EdgeVal::Val(wv), EdgeVal::Const(wc)) =
        (hx, hy, wx, wy)
    else {
        return false;
    };
    if hv != wv {
        return false;
    }
    let (um, ch, cw) = (
        ty.unsigned_max(),
        crate::fold::extend(hc, ty, false) as u64,
        crate::fold::extend(wc, ty, false) as u64,
    );
    let (lo, hi) = match hcc {
        Equal => (ch, ch),
        UnsignedLessThan if ch > 0 => (0, ch - 1),
        UnsignedLessThanOrEqual => (0, ch),
        UnsignedGreaterThan if ch < um => (ch + 1, um),
        UnsignedGreaterThanOrEqual => (ch, um),
        NotEqual if ch == 0 && um >= 1 => (1, um),
        NotEqual if ch == um && um >= 1 => (0, um - 1),
        _ => return false,
    };
    matches!((wcc, cw), (Equal, c) if lo == c && hi == c)
        || matches!(wcc, NotEqual if cw < lo || cw > hi)
        || matches!(wcc, UnsignedLessThan if hi < cw)
        || matches!(wcc, UnsignedLessThanOrEqual if hi <= cw)
        || matches!(wcc, UnsignedGreaterThan if lo > cw)
        || matches!(wcc, UnsignedGreaterThanOrEqual if lo >= cw)
}

/// Returns whether assuming `cond == taken` implies comparison `want` at `ty`.
pub(crate) fn cond_proves_want(
    func: &Function,
    env: &CmpEnv<'_>,
    (cond, taken, depth): (Value, bool, u8),
    ty: Type,
    want: (IntCC, EdgeVal, EdgeVal),
) -> bool {
    let Some((op, taken, depth)) = env.peel_cond(func, cond, taken, depth) else {
        return false;
    };
    match *op {
        Op::Binary(Binary::Band, a, b) if taken => {
            cond_proves_want(func, env, (a, true, depth + 1), ty, want)
                || cond_proves_want(func, env, (b, true, depth + 1), ty, want)
        }
        Op::Binary(Binary::Bor, a, b) if !taken => {
            cond_proves_want(func, env, (a, false, depth + 1), ty, want)
                || cond_proves_want(func, env, (b, false, depth + 1), ty, want)
        }
        Op::Icmp(cc, x, y) if func.ty(x) == ty => cmp_proves(
            ty,
            (
                if taken { cc } else { cc.inverse() },
                env.edge_val(x),
                env.edge_val(y),
            ),
            want,
        ),
        _ => false,
    }
}

/// Returns whether a dominating edge or trap/assume proves `want` at `ty` reaching `cur`.
pub(crate) fn dom_proves_want(
    func: &Function,
    cfg: &Cfg,
    env: &CmpEnv<'_>,
    mut cur: Block,
    skip: Option<Block>,
    ty: Type,
    want: (IntCC, EdgeVal, EdgeVal),
) -> bool {
    let start = cur;
    for _ in 0..DOM_PROVE_STEPS {
        if Some(cur) != skip && func.blocks[cur.index()].insts.iter().any(|&i| {
            matches!(
                func.insts[i.index()].op,
                Op::Trapz(c) | Op::Assume(c) if cond_proves_want(func, env, (c, true, 0), ty, want)
            )
        }) {
            return true;
        }
        let Some(parent) = cfg.idom[cur.index()].filter(|&p| p != cur) else {
            break;
        };
        if let Some(&Op::Brif(c, tb, _, eb, _)) = func.last(parent)
            && tb != eb
            && let Some(t) = (cfg.single_pred(tb) == Some(parent)
                && (tb == cur || (tb == parent && cfg.dominates(tb, start))))
            .then_some(true)
            .or_else(|| {
                (cfg.single_pred(eb) == Some(parent)
                    && (eb == cur || (eb == parent && cfg.dominates(eb, start))))
                .then_some(false)
            })
            && cond_proves_want(func, env, (c, t, 0), ty, want)
        {
            return true;
        }
        cur = parent;
    }
    false
}

fn sub_add_const(consts: &[Option<i64>], op: &Op, ty: Type) -> Option<(Value, i64)> {
    let um = ty.unsigned_max() as i64;
    match *op {
        Op::Binary(Binary::Isub, x, y) => {
            consts[y.index()].map(|c| (x, crate::fold::extend(c, ty, false).wrapping_neg() & um))
        }
        Op::Binary(Binary::Iadd, x, y) => consts[y.index()]
            .map(|c| (x, crate::fold::extend(c, ty, false) & um))
            .or_else(|| consts[x.index()].map(|c| (y, crate::fold::extend(c, ty, false) & um))),
        _ => None,
    }
}

fn collect_wide_subs(
    func: &Function,
    cfg: &Cfg,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
) -> Vec<(Value, Type, i64, Value, Block, usize)> {
    let mut wide = Vec::new();
    for &b in &cfg.rpo {
        for (p, &i) in func.blocks[b.index()].insts.iter().enumerate() {
            if let Some(r) = func.insts[i.index()].result
                && matches!(func.ty(r), Type::I32 | Type::I64)
                && let Some((u, d)) = sub_add_const(consts, &func.insts[i.index()].op, func.ty(r))
                && let Some(ui) = defs[u.index()]
                && let Op::Convert(Convert::Uextend, base) = func.insts[ui.index()].op
                && matches!(func.ty(base), Type::I8 | Type::I16)
            {
                wide.push((
                    base,
                    func.ty(base),
                    d & (func.ty(base).unsigned_max() as i64),
                    r,
                    b,
                    p,
                ));
            }
        }
    }
    wide
}

pub(super) fn narrow_widened_subs(
    func: &mut Function,
    cfg: &Cfg,
    consts: &[Option<i64>],
    bool_val: &[bool],
) -> bool {
    let defs = crate::fold::defs(func);
    let wide = collect_wide_subs(func, cfg, &defs, consts);
    if wide.is_empty() {
        return false;
    }
    let id_subst: Vec<Value> = (0..func.types.len() as u32).map(Value).collect();
    let env = CmpEnv {
        defs: &defs,
        consts,
        subst: &id_subst,
        bool_val: Some(bool_val),
    };
    let (mut changed, mut subst, mut has_subst) = (false, id_subst.clone(), false);
    for &block in &cfg.rpo {
        for pos in 0..func.blocks[block.index()].insts.len() {
            let inst = func.blocks[block.index()].insts[pos];
            let Some(res) = func.insts[inst.index()].result else {
                continue;
            };
            let ty = func.ty(res);
            if matches!(ty, Type::I8 | Type::I16)
                && let Some((base, delta)) = sub_add_const(consts, &func.insts[inst.index()].op, ty)
                && let Some(&(_, _, _, dw, _, _)) = wide.iter().find(|&&(bw, st, d, _, wb, wp)| {
                    bw == base
                        && st == ty
                        && d == delta
                        && (wb == block && wp < pos || wb != block && cfg.dominates(wb, block))
                })
            {
                func.insts[inst.index()].op = Op::Convert(Convert::Ireduce, dw);
                changed = true;
            }
            if let Op::Convert(Convert::Uextend, inner) = func.insts[inst.index()].op
                && let Some(ii) = defs[inner.index()]
                && let Op::Convert(Convert::Ireduce, src) = func.insts[ii.index()].op
                && func.ty(inner).bits() < func.ty(src).bits()
                && dom_proves_want(
                    func,
                    cfg,
                    &env,
                    block,
                    Some(block),
                    func.ty(src),
                    (
                        IntCC::UnsignedLessThanOrEqual,
                        EdgeVal::Val(src),
                        EdgeVal::Const(func.ty(inner).unsigned_max() as i64),
                    ),
                )
            {
                func.insts[inst.index()].op = if ty.bits() > func.ty(src).bits() {
                    Op::Convert(Convert::Uextend, src)
                } else if ty.bits() < func.ty(src).bits() {
                    Op::Convert(Convert::Ireduce, src)
                } else {
                    subst[res.index()] = src;
                    has_subst = true;
                    continue;
                };
                changed = true;
            }
        }
    }
    if has_subst {
        rename(func, &subst);
        changed = true;
    }
    if changed {
        let u = func.use_counts();
        for b in &mut func.blocks {
            b.insts.retain(|&i| {
                !matches!(
                    func.insts[i.index()].op,
                    Op::Convert(Convert::Ireduce | Convert::Uextend, _)
                ) || func.insts[i.index()]
                    .result
                    .is_none_or(|r| u[r.index()] > 0)
            });
        }
    }
    changed
}
