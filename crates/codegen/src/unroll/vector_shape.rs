use super::memset::use_count;
use crate::ir::{Binary, Convert, Function, Inst, InstData, IntCC, Op, Type, Unary, Value};

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
    matches!(
        op,
        Binary::Iadd
            | Binary::Isub
            | Binary::Umin
            | Binary::Umax
            | Binary::Smin
            | Binary::Smax
            | Binary::Band
            | Binary::Bor
            | Binary::Bxor
    )
}

fn mask_binary(op: Binary) -> bool {
    matches!(op, Binary::Band | Binary::Bor | Binary::Bxor)
}

struct Scan<'a> {
    func: &'a Function,
    body: &'a [Inst],
    params: &'a [Value],
    back_args: &'a [Value],
    steps: &'a [Option<i64>],
    defs: &'a [bool],
    consts: &'a [Option<i64>],
    kinds: Vec<Option<Kind>>,
    lanes: Vec<Inst>,
    loads: Vec<Access>,
    store: Option<Access>,
    access: Vec<(Inst, Access)>,
    bounds: Vec<(usize, Value)>,
    widened: Vec<(Value, Value)>,
    sums: Vec<Sum>,
    iv_cands: Vec<IvCand>,
    iv_adds: Vec<IvAdd>,
    iv_sums: Vec<IvSum>,
    splats: Vec<Value>,
    amounts: Vec<(Value, i64)>,
}

impl Scan<'_> {
    fn kind(&mut self, v: Value) -> Option<Kind> {
        if let Some(kind) = self.kinds[v.index()] {
            return Some(kind);
        }
        let fixed = self
            .params
            .iter()
            .position(|&p| p == v)
            .is_some_and(|i| self.steps[i] == Some(0));
        if (self.defs[v.index()] && !fixed) || self.func.ty(v) != Type::I8 {
            return None;
        }
        if !self.splats.contains(&v) {
            self.splats.push(v);
        }
        Some(Kind::Data)
    }

    fn stepped(&self, addr: Value) -> Option<usize> {
        self.params
            .iter()
            .position(|&p| p == addr)
            .filter(|&i| self.steps[i] == Some(1) && self.func.ty(addr) == Type::I64)
    }

    fn acc_idx(&self, v: Value, back: Option<Value>) -> Option<usize> {
        self.params
            .iter()
            .position(|&p| p == v)
            .filter(|&i| self.steps[i].is_none() && back.is_none_or(|b| self.back_args[i] == b))
    }

    fn has_sum(&self, idx: usize) -> bool {
        self.sums.iter().any(|s| s.idx == idx) || self.iv_sums.iter().any(|s| s.idx == idx)
    }

    fn def_of(&self, v: Value) -> Option<&InstData> {
        self.body
            .iter()
            .map(|&i| &self.func.insts[i.index()])
            .find(|d| d.result == Some(v))
    }

    fn bound(&self, cond: Value) -> Option<(usize, Value)> {
        let def = self.def_of(cond)?;
        let Op::Icmp(IntCC::UnsignedLessThan, at, limit) = def.op else {
            return None;
        };
        let idx = self
            .params
            .iter()
            .position(|&p| p == at)
            .filter(|&i| self.steps[i] == Some(1))?;
        (!self.defs[limit.index()]).then_some((idx, limit))
    }

    fn access(&self, addr: Value, off: i32) -> Option<Access> {
        if let Some(idx) = self.stepped(addr) {
            return Some(Access {
                base: None,
                idx,
                off,
            });
        }
        let def = self.def_of(addr)?;
        let Op::Binary(Binary::Iadd, x, y) = def.op else {
            return None;
        };
        [(x, y), (y, x)].into_iter().find_map(|(base, at)| {
            let idx = self.stepped(at)?;
            (!self.defs[base.index()]).then_some(Access {
                base: Some(base),
                idx,
                off,
            })
        })
    }

    fn lane(&mut self, op: &Op) -> Option<Kind> {
        match *op {
            Op::Binary(Binary::Ushr | Binary::Ishl, x, y) => {
                let by = self
                    .consts
                    .get(y.index())
                    .copied()
                    .flatten()
                    .filter(|by| (1..8).contains(by))?;
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
                (Kind::Mask, Kind::Mask) if matches!(cc, IntCC::Equal | IntCC::NotEqual) => {
                    Some(Kind::Mask)
                }
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
        let def = self.def_of(v)?;
        let Op::Convert(Convert::Uextend, inner) = def.op else {
            return None;
        };
        matches!(self.kinds[inner.index()], Some(Kind::Mask | Kind::Data)).then_some(inner)
    }

    fn stepped_iv(&self, v: Value) -> Option<(usize, i64)> {
        if self.func.ty(v) != Type::I64 {
            return None;
        }
        if let Some(idx) = self.stepped(v) {
            return Some((idx, 0));
        }
        let def = self.def_of(v)?;
        match def.op {
            Op::Binary(Binary::Iadd, a, b) => [(a, b), (b, a)].into_iter().find_map(|(p, c)| {
                let idx = self.stepped(p)?;
                let bias = self.consts.get(c.index()).copied().flatten()?;
                Some((idx, bias))
            }),
            Op::Binary(Binary::Isub, a, c) => {
                let idx = self.stepped(a)?;
                let bias = self
                    .consts
                    .get(c.index())
                    .copied()
                    .flatten()?
                    .checked_neg()?;
                Some((idx, bias))
            }
            _ => None,
        }
    }

    fn sum(&self, op: &Op, result: Value) -> Option<Sum> {
        let Op::Binary(Binary::Iadd, x, y) = *op else {
            return None;
        };
        [(x, y), (y, x)].into_iter().find_map(|(acc, wide)| {
            let idx = self.acc_idx(acc, Some(result))?;
            let lane = self
                .widened
                .iter()
                .find(|&&(w, _)| w == wide)
                .map(|&(_, lane)| lane)?;
            Some(Sum { idx, lane })
        })
    }

    fn control(&self, op: &Op) -> bool {
        let lane = |v: Value| self.kinds[v.index()].is_some();
        match *op {
            Op::Const(_) => true,
            Op::Binary(Binary::Iadd | Binary::Isub, x, y) | Op::Icmp(_, x, y) => {
                !lane(x) && !lane(y) && self.func.ty(x) != Type::I8
            }
            _ => false,
        }
    }

    fn visit_store(&mut self, inst: Inst, value: Value, addr: Value, off: i32) -> Option<()> {
        if self.store.is_some()
            || self.func.ty(value) != Type::I8
            || self.kind(value)? != Kind::Data
        {
            return None;
        }
        let at = self.access(addr, off)?;
        self.store = Some(at);
        self.access.push((inst, at));
        self.lanes.push(inst);
        Some(())
    }

    fn try_iv_select(&mut self, (c, x, y): (Value, Value, Value), result: Value) -> Option<bool> {
        let Some(cond) = self.mask_cond(c) else {
            return Some(false);
        };
        if let Some((sum_idx, pos, invert)) =
            [(x, y, false), (y, x, true)]
                .into_iter()
                .find_map(|(a, s, inv)| {
                    let sum_idx = self.acc_idx(s, Some(result))?;
                    let pos = self
                        .iv_adds
                        .iter()
                        .position(|add| add.res == a && add.sum_idx == sum_idx)?;
                    Some((sum_idx, pos, inv))
                })
        {
            if self.has_sum(sum_idx) {
                return None;
            }
            let add = self.iv_adds.swap_remove(pos);
            self.iv_sums.push(IvSum {
                idx: sum_idx,
                iv_idx: add.iv_idx,
                cond,
                invert,
                bias: add.bias,
                two_uses: true,
            });
            return Some(true);
        }
        if use_count(self.func, self.body, result) == 1
            && let Some((iv_idx, bias, invert)) = [(x, y, false), (y, x, true)]
                .into_iter()
                .find_map(|(iv_arm, z_arm, inv)| {
                    self.is_zero(z_arm)
                        .then(|| self.stepped_iv(iv_arm).map(|(idx, bias)| (idx, bias, inv)))
                        .flatten()
                })
        {
            self.iv_cands.push(IvCand {
                res: result,
                iv_idx,
                cond,
                invert,
                bias,
            });
            return Some(true);
        }
        Some(false)
    }

    fn try_iv_add(&mut self, (x, y): (Value, Value), result: Value) -> Option<bool> {
        if let Some((sum_idx, pos)) = [(x, y), (y, x)].into_iter().find_map(|(acc, cand)| {
            let sum_idx = self.acc_idx(acc, Some(result))?;
            let pos = self.iv_cands.iter().position(|c| c.res == cand)?;
            Some((sum_idx, pos))
        }) {
            if self.has_sum(sum_idx) {
                return None;
            }
            let cand = self.iv_cands.swap_remove(pos);
            self.iv_sums.push(IvSum {
                idx: sum_idx,
                iv_idx: cand.iv_idx,
                cond: cand.cond,
                invert: cand.invert,
                bias: cand.bias,
                two_uses: false,
            });
            return Some(true);
        }
        if use_count(self.func, self.body, result) == 1
            && let Some((sum_idx, iv_idx, bias)) =
                [(x, y), (y, x)].into_iter().find_map(|(acc, iv_arm)| {
                    let sum_idx = self.acc_idx(acc, None)?;
                    let (iv_idx, bias) = self.stepped_iv(iv_arm)?;
                    Some((sum_idx, iv_idx, bias))
                })
        {
            self.iv_adds.push(IvAdd {
                res: result,
                sum_idx,
                iv_idx,
                bias,
            });
            return Some(true);
        }
        Some(false)
    }

    fn try_iv_reduction(&mut self, op: &Op, result: Value) -> Option<bool> {
        match *op {
            Op::Select(c, x, y) => self.try_iv_select((c, x, y), result),
            Op::Binary(Binary::Iadd, x, y) => self.try_iv_add((x, y), result),
            _ => Some(false),
        }
    }

    fn visit_lane(&mut self, inst: Inst, op: &Op, result: Value) -> Option<()> {
        if self.func.ty(result) != Type::I8
            || matches!(*op, Op::Icmp(_, x, _) if self.func.ty(x) != Type::I8)
        {
            return self.control(op).then_some(());
        }
        let kind = match *op {
            Op::Const(_) => {
                self.splats.push(result);
                Kind::Data
            }
            Op::Load(addr, off) => {
                let at = self.access(addr, off)?;
                self.loads.push(at);
                self.access.push((inst, at));
                Kind::Data
            }
            _ => self.lane(op)?,
        };
        self.kinds[result.index()] = Some(kind);
        if !matches!(*op, Op::Const(_)) {
            self.lanes.push(inst);
        }
        Some(())
    }

    fn visit_inst(&mut self, inst: Inst) -> Option<()> {
        let func = self.func;
        let data = &func.insts[inst.index()];
        match data.op {
            Op::Assume(_) => return Some(()),
            Op::Trapz(cond) => {
                let bound = self.bound(cond)?;
                if !self.bounds.contains(&bound) {
                    self.bounds.push(bound);
                }
                return Some(());
            }
            Op::Store(value, addr, off) => return self.visit_store(inst, value, addr, off),
            _ => {}
        }
        let result = data.result?;
        if let Op::Convert(Convert::Uextend, x) = data.op
            && self.kinds[x.index()] == Some(Kind::Data)
        {
            self.kinds[result.index()] = Some(Kind::Wide);
            self.widened.push((result, x));
            return Some(());
        }
        if let Some(sum) = self.sum(&data.op, result) {
            if self.has_sum(sum.idx) {
                return None;
            }
            self.sums.push(sum);
            return Some(());
        }
        if matches!(func.ty(result), Type::I32 | Type::I64)
            && self.try_iv_reduction(&data.op, result)?
        {
            return Some(());
        }
        self.visit_lane(inst, &data.op, result)
    }

    fn finish(self) -> Option<Shape> {
        let every_sum = (0..self.params.len())
            .filter(|&i| self.steps[i].is_none())
            .all(|i| {
                let want = if self.sums.iter().any(|s| s.idx == i) {
                    Some(1)
                } else {
                    self.iv_sums
                        .iter()
                        .find(|s| s.idx == i)
                        .map(|s| if s.two_uses { 2 } else { 1 })
                };
                want.is_some_and(|w| use_count(self.func, self.body, self.params[i]) == w)
            });
        if !every_sum
            || !self.iv_cands.is_empty()
            || !self.iv_adds.is_empty()
            || (self.store.is_none() && self.sums.is_empty() && self.iv_sums.is_empty())
        {
            return None;
        }
        Some(Shape {
            kinds: self.kinds,
            lanes: self.lanes,
            loads: self.loads,
            store: self.store,
            sums: self.sums,
            iv_sums: self.iv_sums,
            access: self.access,
            splats: self.splats,
            amounts: self.amounts,
            bounds: self.bounds,
        })
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
    let mut scan = Scan {
        func,
        body,
        params,
        back_args,
        steps,
        defs,
        consts,
        kinds: vec![None; func.types.len()],
        lanes: Vec::new(),
        loads: Vec::new(),
        store: None,
        access: Vec::new(),
        bounds: Vec::new(),
        widened: Vec::new(),
        sums: Vec::new(),
        iv_cands: Vec::new(),
        iv_adds: Vec::new(),
        iv_sums: Vec::new(),
        splats: Vec::new(),
        amounts: Vec::new(),
    };
    for &inst in body {
        scan.visit_inst(inst)?;
    }
    scan.finish()
}
