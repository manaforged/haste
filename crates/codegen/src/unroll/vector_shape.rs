use super::memset::use_count;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

/// Bytes, and so byte lanes, that one vector iteration covers.
pub(super) const LANES: i64 = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Data,
    Mask,
    Wide,
}

/// A parameter that adds the zero-extended byte `lane` into itself each iteration.
#[derive(Clone, Copy)]
pub(super) struct Sum {
    pub(super) idx: usize,
    pub(super) lane: Value,
}

/// A parameter that adds a stepped `I64` induction variable into itself when an `I8` condition holds.
#[derive(Clone, Copy)]
pub(super) struct IvSum {
    pub(super) idx: usize,
    pub(super) iv_idx: usize,
    pub(super) cond: Value,
    pub(super) invert: bool,
    pub(super) bias: i64,
    two_uses: bool,
}

struct IvCand {
    res: Value,
    iv_idx: usize,
    cond: Value,
    invert: bool,
    bias: i64,
}

struct IvAdd {
    res: Value,
    sum_idx: usize,
    iv_idx: usize,
    bias: i64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Access {
    pub(super) base: Option<Value>,
    pub(super) idx: usize,
    pub(super) off: i32,
}

pub(super) struct Shape {
    pub(super) kinds: Vec<Option<Kind>>,
    pub(super) lanes: Vec<Inst>,
    pub(super) loads: Vec<Access>,
    pub(super) store: Option<Access>,
    pub(super) sums: Vec<Sum>,
    pub(super) iv_sums: Vec<IvSum>,
    pub(super) access: Vec<(Inst, Access)>,
    pub(super) splats: Vec<Value>,
    pub(super) amounts: Vec<(Value, i64)>,
    pub(super) bounds: Vec<(usize, Value)>,
}

fn lane_binary(op: Binary) -> bool {
    matches!(op, Binary::Iadd | Binary::Isub | Binary::Umin | Binary::Umax | Binary::Smin | Binary::Smax | Binary::Band | Binary::Bor | Binary::Bxor)
}

fn mask_binary(op: Binary) -> bool {
    matches!(op, Binary::Band | Binary::Bor | Binary::Bxor)
}

struct Scan<'a> {
    func: &'a Function,
    body: &'a [Inst],
    params: &'a [Value],
    steps: &'a [Option<i64>],
    defs: &'a [bool],
    consts: &'a [Option<i64>],
    kinds: Vec<Option<Kind>>,
    splats: Vec<Value>,
    amounts: Vec<(Value, i64)>,
}

impl Scan<'_> {
    fn kind(&mut self, v: Value) -> Option<Kind> {
        if let Some(kind) = self.kinds[v.index()] {
            return Some(kind);
        }
        let fixed = self.params.iter().position(|&p| p == v).is_some_and(|i| self.steps[i] == Some(0));
        if (self.defs[v.index()] && !fixed) || self.func.ty(v) != Type::I8 {
            return None;
        }
        if !self.splats.contains(&v) {
            self.splats.push(v);
        }
        Some(Kind::Data)
    }

    fn stepped(&self, addr: Value) -> Option<usize> {
        self.params.iter().position(|&p| p == addr).filter(|&i| self.steps[i] == Some(1) && self.func.ty(addr) == Type::I64)
    }

    fn bound(&self, cond: Value) -> Option<(usize, Value)> {
        let def = self.body.iter().map(|&i| &self.func.insts[i.index()]).find(|d| d.result == Some(cond))?;
        let Op::Icmp(IntCC::UnsignedLessThan, at, limit) = def.op else { return None };
        let idx = self.params.iter().position(|&p| p == at).filter(|&i| self.steps[i] == Some(1))?;
        (!self.defs[limit.index()]).then_some((idx, limit))
    }

    fn access(&self, addr: Value, off: i32) -> Option<Access> {
        if let Some(idx) = self.stepped(addr) {
            return Some(Access { base: None, idx, off });
        }
        let def = self.body.iter().map(|&i| &self.func.insts[i.index()]).find(|d| d.result == Some(addr))?;
        let Op::Binary(Binary::Iadd, x, y) = def.op else { return None };
        [(x, y), (y, x)].into_iter().find_map(|(base, at)| {
            let idx = self.stepped(at)?;
            (!self.defs[base.index()]).then_some(Access { base: Some(base), idx, off })
        })
    }

    fn lane(&mut self, op: &Op) -> Option<Kind> {
        match *op {
            Op::Binary(Binary::Ushr | Binary::Ishl, x, y) => {
                let by = self.consts.get(y.index()).copied().flatten().filter(|by| (1..8).contains(by))?;
                self.amounts.push((y, by));
                (self.kind(x)? == Kind::Data).then_some(Kind::Data)
            }
            Op::Binary(op, x, y) if lane_binary(op) => match (self.kind(x)?, self.kind(y)?) {
                (Kind::Data, Kind::Data) => Some(Kind::Data),
                (Kind::Mask, Kind::Mask) if mask_binary(op) => Some(Kind::Mask),
                _ => None,
            },
            Op::Icmp(cc, x, y) => match (self.kind(x)?, self.kind(y)?) {
                (Kind::Data, Kind::Data) => Some(Kind::Mask),
                (Kind::Mask, Kind::Mask) if matches!(cc, IntCC::Equal | IntCC::NotEqual) => Some(Kind::Mask),
                _ => None,
            },
            Op::Select(c, x, y) => {
                let (kc, kx, ky) = (self.kind(c)?, self.kind(x)?, self.kind(y)?);
                (kc == Kind::Mask && kx == ky && kx != Kind::Wide).then_some(kx)
            }
            Op::Unary(Unary::Ineg, x) => (self.kind(x)? == Kind::Data).then_some(Kind::Data),
            _ => None,
        }
    }

    fn is_zero(&self, v: Value) -> bool {
        self.consts.get(v.index()).copied().flatten() == Some(0)
    }

    fn mask_cond(&self, v: Value) -> Option<Value> {
        if matches!(self.kinds[v.index()], Some(Kind::Mask | Kind::Data)) {
            return Some(v);
        }
        let def = self.body.iter().map(|&i| &self.func.insts[i.index()]).find(|d| d.result == Some(v))?;
        let Op::Convert(Convert::Uextend, inner) = def.op else { return None };
        matches!(self.kinds[inner.index()], Some(Kind::Mask | Kind::Data)).then_some(inner)
    }

    fn stepped_iv(&self, v: Value) -> Option<(usize, i64)> {
        if self.func.ty(v) != Type::I64 {
            return None;
        }
        if let Some(idx) = self.stepped(v) {
            return Some((idx, 0));
        }
        let def = self.body.iter().map(|&i| &self.func.insts[i.index()]).find(|d| d.result == Some(v))?;
        match def.op {
            Op::Binary(Binary::Iadd, a, b) => [(a, b), (b, a)].into_iter().find_map(|(p, c)| {
                let idx = self.stepped(p)?;
                let bias = self.consts.get(c.index()).copied().flatten()?;
                Some((idx, bias))
            }),
            Op::Binary(Binary::Isub, a, c) => {
                let idx = self.stepped(a)?;
                let bias = self.consts.get(c.index()).copied().flatten()?.checked_neg()?;
                Some((idx, bias))
            }
            _ => None,
        }
    }

    /// The accumulator that `iadd` adds a widened lane into, when `result` is its back-edge value.
    fn sum(&self, op: &Op, result: Value, (back_args, widened): (&[Value], &[(Value, Value)])) -> Option<Sum> {
        let Op::Binary(Binary::Iadd, x, y) = *op else { return None };
        [(x, y), (y, x)].into_iter().find_map(|(acc, wide)| {
            let idx = self.params.iter().position(|&p| p == acc).filter(|&i| self.steps[i].is_none() && back_args[i] == result)?;
            let lane = widened.iter().find(|&&(w, _)| w == wide).map(|&(_, lane)| lane)?;
            Some(Sum { idx, lane })
        })
    }

    fn control(&self, op: &Op) -> bool {
        let lane = |v: Value| self.kinds[v.index()].is_some();
        match *op {
            Op::Const(_) => true,
            Op::Binary(Binary::Iadd | Binary::Isub, x, y) | Op::Icmp(_, x, y) => !lane(x) && !lane(y) && self.func.ty(x) != Type::I8,
            _ => false,
        }
    }
}

/// Analyzes `body` for byte-lane vectorization and masked `I64` induction-variable reductions.
pub(super) fn analyze(
    func: &Function,
    body: &[Inst],
    (params, back_args): (&[Value], &[Value]),
    (steps, defs): (&[Option<i64>], &[bool]),
    consts: &[Option<i64>],
) -> Option<Shape> {
    let mut scan = Scan { func, body, params, steps, defs, consts, kinds: vec![None; func.types.len()], splats: Vec::new(), amounts: Vec::new() };
    let (mut lanes, mut loads, mut store, mut access, mut bounds) = (Vec::new(), Vec::new(), None, Vec::new(), Vec::new());
    let (mut widened, mut sums) = (Vec::new(), Vec::<Sum>::new());
    let (mut iv_cands, mut iv_adds, mut iv_sums) = (Vec::<IvCand>::new(), Vec::<IvAdd>::new(), Vec::<IvSum>::new());
    let acc_idx_of = |v: Value, back: Option<Value>| {
        params.iter().position(|&p| p == v).filter(|&i| steps[i].is_none() && back.is_none_or(|b| back_args[i] == b))
    };
    let has_sum = |sums: &[Sum], iv_sums: &[IvSum], idx: usize| sums.iter().any(|s| s.idx == idx) || iv_sums.iter().any(|s| s.idx == idx);
    for &inst in body {
        let data = &func.insts[inst.index()];
        if matches!(data.op, Op::Assume(_)) {
            continue;
        }
        if let Op::Trapz(cond) = data.op {
            let bound = scan.bound(cond)?;
            if !bounds.contains(&bound) {
                bounds.push(bound);
            }
            continue;
        }
        if let Op::Store(value, addr, off) = data.op {
            if store.is_some() || func.ty(value) != Type::I8 || scan.kind(value)? != Kind::Data {
                return None;
            }
            let at = scan.access(addr, off)?;
            store = Some(at);
            access.push((inst, at));
            lanes.push(inst);
            continue;
        }
        let result = data.result?;
        if let Op::Convert(Convert::Uextend, x) = data.op
            && scan.kinds[x.index()] == Some(Kind::Data)
        {
            scan.kinds[result.index()] = Some(Kind::Wide);
            widened.push((result, x));
            continue;
        }
        if let Some(sum) = scan.sum(&data.op, result, (back_args, &widened)) {
            if has_sum(&sums, &iv_sums, sum.idx) {
                return None;
            }
            sums.push(sum);
            continue;
        }
        if matches!(func.ty(result), Type::I32 | Type::I64) {
            match data.op {
                Op::Select(c, x, y) if let Some(cond) = scan.mask_cond(c) => {
                    if let Some((sum_idx, pos, invert)) = [(x, y, false), (y, x, true)].into_iter().find_map(|(a, s, inv)| {
                        let sum_idx = acc_idx_of(s, Some(result))?;
                        let pos = iv_adds.iter().position(|add| add.res == a && add.sum_idx == sum_idx)?;
                        Some((sum_idx, pos, inv))
                    }) {
                        if has_sum(&sums, &iv_sums, sum_idx) {
                            return None;
                        }
                        let add = iv_adds.swap_remove(pos);
                        iv_sums.push(IvSum { idx: sum_idx, iv_idx: add.iv_idx, cond, invert, bias: add.bias, two_uses: true });
                        continue;
                    }
                    if use_count(func, body, result) == 1
                        && let Some((iv_idx, bias, invert)) = [(x, y, false), (y, x, true)].into_iter().find_map(|(iv_arm, z_arm, inv)| {
                            scan.is_zero(z_arm).then(|| scan.stepped_iv(iv_arm).map(|(idx, bias)| (idx, bias, inv))).flatten()
                        })
                    {
                        iv_cands.push(IvCand { res: result, iv_idx, cond, invert, bias });
                        continue;
                    }
                }
                Op::Binary(Binary::Iadd, x, y) => {
                    if let Some((sum_idx, pos)) = [(x, y), (y, x)].into_iter().find_map(|(acc, cand)| {
                        let sum_idx = acc_idx_of(acc, Some(result))?;
                        let pos = iv_cands.iter().position(|c| c.res == cand)?;
                        Some((sum_idx, pos))
                    }) {
                        if has_sum(&sums, &iv_sums, sum_idx) {
                            return None;
                        }
                        let cand = iv_cands.swap_remove(pos);
                        iv_sums.push(IvSum { idx: sum_idx, iv_idx: cand.iv_idx, cond: cand.cond, invert: cand.invert, bias: cand.bias, two_uses: false });
                        continue;
                    }
                    if use_count(func, body, result) == 1
                        && let Some((sum_idx, iv_idx, bias)) = [(x, y), (y, x)].into_iter().find_map(|(acc, iv_arm)| {
                            let sum_idx = acc_idx_of(acc, None)?;
                            let (iv_idx, bias) = scan.stepped_iv(iv_arm)?;
                            Some((sum_idx, iv_idx, bias))
                        })
                    {
                        iv_adds.push(IvAdd { res: result, sum_idx, iv_idx, bias });
                        continue;
                    }
                }
                _ => {}
            }
        }
        if func.ty(result) != Type::I8 || (matches!(data.op, Op::Icmp(_, x, _) if func.ty(x) != Type::I8)) {
            if !scan.control(&data.op) {
                return None;
            }
            continue;
        }
        let kind = match data.op {
            Op::Const(_) => {
                scan.splats.push(result);
                Kind::Data
            }
            Op::Load(addr, off) => {
                let at = scan.access(addr, off)?;
                loads.push(at);
                access.push((inst, at));
                Kind::Data
            }
            ref op => scan.lane(op)?,
        };
        scan.kinds[result.index()] = Some(kind);
        if !matches!(data.op, Op::Const(_)) {
            lanes.push(inst);
        }
    }
    let every_sum = (0..params.len()).filter(|&i| steps[i].is_none()).all(|i| {
        let want = if sums.iter().any(|s| s.idx == i) {
            Some(1)
        } else {
            iv_sums.iter().find(|s| s.idx == i).map(|s| if s.two_uses { 2 } else { 1 })
        };
        want.is_some_and(|w| use_count(func, body, params[i]) == w)
    });
    if !every_sum || !iv_cands.is_empty() || !iv_adds.is_empty() || (store.is_none() && sums.is_empty() && iv_sums.is_empty()) {
        return None;
    }
    Some(Shape { kinds: scan.kinds, lanes, loads, store, sums, iv_sums, access, splats: scan.splats, amounts: scan.amounts, bounds })
}
