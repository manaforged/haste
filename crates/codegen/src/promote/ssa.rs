use super::{Kind, Scan, join_slots};
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, Type, Value};

fn convert(from: Type, to: Type) -> Option<Convert> {
    if from == to {
        return None;
    }
    if from.is_float() || to.is_float() || from.is_vector() || to.is_vector() {
        assert_eq!(from.bytes(), to.bytes());
        return Some(Convert::Bitcast);
    }
    Some(if to.bytes() < from.bytes() {
        Convert::Ireduce
    } else {
        Convert::Uextend
    })
}

#[derive(Clone, Copy)]
struct Load {
    inst: Inst,
    result: Value,
    ty: Type,
}

struct Local {
    var_defs: Vec<Vec<(u32, Value)>>,
    exposed: Vec<Vec<(Block, Load)>>,
    forward: Vec<Vec<(Load, Value)>>,
    store_casts: Vec<Vec<(Inst, Value)>>,
}

impl Local {
    fn new(scan: &Scan, fresh: &mut Fresh, subst: &mut Vec<Value>) -> Self {
        let count = scan.vars.len();
        let mut local = Local {
            var_defs: vec![Vec::new(); count],
            exposed: vec![Vec::new(); count],
            forward: vec![Vec::new(); count],
            store_casts: vec![Vec::new(); count],
        };
        for access in &scan.accesses {
            let block = access.block.0;
            let defs = &mut local.var_defs[access.var];
            match access.kind {
                Kind::Store(value) => {
                    let home = scan.vars[access.var].home();
                    let def = if access.ty != home && access.ty.bytes() == home.bytes() {
                        let cast = fresh.value(subst, home);
                        local.store_casts[access.var].push((access.inst, cast));
                        cast
                    } else {
                        value
                    };
                    if let Some(last) = defs.last_mut().filter(|(b, _)| *b == block) {
                        last.1 = def;
                    } else {
                        defs.push((block, def));
                    }
                }
                Kind::Load(result) => {
                    let load = Load {
                        inst: access.inst,
                        result,
                        ty: access.ty,
                    };
                    if let Some(&(_, def)) = defs.last().filter(|&&(b, _)| b == block) {
                        local.forward[access.var].push((load, def));
                    } else {
                        local.exposed[access.var].push((access.block, load));
                        if access.ty == scan.vars[access.var].home() {
                            defs.push((block, result));
                        }
                    }
                }
            }
        }
        local
    }
}

fn try_targets(func: &Function) -> Vec<bool> {
    let mut try_target = vec![false; func.blocks.len()];
    for index in 0..func.blocks.len() {
        let Some(op) = func.last(Block(index as u32)) else {
            continue;
        };
        if matches!(op, Op::TryCall { .. } | Op::BrTable(..)) {
            op.targets()
                .for_each(|target| try_target[target.index()] = true);
        }
    }
    try_target
}

struct Fresh {
    base: usize,
    types: Vec<Type>,
    zeros: Vec<(Type, Value)>,
}

impl Fresh {
    fn value(&mut self, subst: &mut Vec<Value>, ty: Type) -> Value {
        self.types.push(ty);
        let value = Value((self.base + self.types.len() - 1) as u32);
        subst.push(value);
        value
    }

    fn zero(&mut self, subst: &mut Vec<Value>, ty: Type) -> Value {
        if let Some((_, value)) = self.zeros.iter().find(|(zero, _)| *zero == ty) {
            return *value;
        }
        let value = self.value(subst, ty);
        self.zeros.push((ty, value));
        value
    }

    fn rollback(&mut self, mark: usize) {
        self.types.truncate(mark);
        let limit = self.base + mark;
        self.zeros.retain(|(_, value)| value.index() < limit);
    }
}

#[derive(Clone, Copy)]
struct SlotVar {
    local: usize,
    index: usize,
}

struct Phi {
    value: Value,
    block: Block,
    var: SlotVar,
    args: Vec<Value>,
}

#[derive(Default)]
struct Build {
    phis: Vec<Phi>,
    converts: Vec<(Inst, Convert, Value)>,
}

enum Step {
    Done(Value),
    Up(Block),
}

struct Pass {
    cfg: std::sync::Arc<crate::cfg::Cfg>,
    try_target: Vec<bool>,
    scan: Scan,
    local: Local,
    fresh: Fresh,
    subst: Vec<Value>,
    undo: Vec<(u32, Value)>,
    memo: Vec<(u32, Value)>,
    stamp: u32,
    block_defs: Vec<(u32, Value)>,
    def_stamp: u32,
    path: Vec<Block>,
    promoted_vars: Vec<bool>,
    kept: Vec<bool>,
}

impl Pass {
    fn load_var_defs(&mut self, index: usize) {
        self.def_stamp = self.def_stamp.wrapping_add(1);
        if self.def_stamp == 0 {
            self.block_defs.fill((0, Value(0)));
            self.def_stamp = 1;
        }
        for &(block, val) in &self.local.var_defs[index] {
            self.block_defs[block as usize] = (self.def_stamp, val);
        }
    }

    fn block_def(&self, block: Block) -> Option<Value> {
        let entry = self.block_defs[block.index()];
        (entry.0 == self.def_stamp).then_some(entry.1)
    }

    fn resolve(&mut self, value: Value) -> Value {
        let root = crate::merge::find(&self.subst, value);
        let mut at = value;
        while self.subst[at.index()] != root {
            let next = std::mem::replace(&mut self.subst[at.index()], root);
            self.undo.push((at.0, next));
            at = next;
        }
        root
    }

    fn set(&mut self, from: Value, to: Value) {
        let to = self.resolve(to);
        if from != to {
            self.undo.push((from.0, self.subst[from.index()]));
            self.subst[from.index()] = to;
        }
    }

    fn bind(&mut self, build: &mut Build, home: Type, load: Load, value: Value) {
        match convert(home, load.ty) {
            Some(op) => build.converts.push((load.inst, op, value)),
            None => self.set(load.result, value),
        }
    }

    fn prepare(&mut self, vars: usize) {
        let need = vars * self.cfg.reachable.len();
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.memo.fill((0, Value(0)));
            self.stamp = 1;
        }
        if self.memo.len() < need {
            self.memo.resize(need, (0, Value(0)));
        }
    }

    fn rollback(&mut self, mark: usize) {
        for (index, old) in self.undo.drain(..).rev() {
            self.subst[index as usize] = old;
        }
        self.fresh.rollback(mark);
        self.subst.truncate(self.fresh.base + mark);
    }

    fn step(&mut self, build: &mut Build, var: SlotVar, block: Block) -> Option<Step> {
        let ty = self.scan.vars[var.index].home();
        let preds = self.cfg.preds_of(block);
        if self.cfg.entry == Some(block) && !preds.is_empty() {
            return None;
        }
        if !self.cfg.reachable[block.index()] || preds.is_empty() {
            return Some(Step::Done(self.fresh.zero(&mut self.subst, ty)));
        }
        if let Some(&pred) = preds.first().filter(|&p| preds[1..].iter().all(|q| q == p)) {
            return Some(match self.block_def(pred) {
                Some(def) => Step::Done(def),
                None => Step::Up(pred),
            });
        }
        let value = self.fresh.value(&mut self.subst, ty);
        build.phis.push(Phi {
            value,
            block,
            var,
            args: Vec::new(),
        });
        Some(Step::Done(value))
    }

    fn entry(&mut self, build: &mut Build, var: SlotVar, start: Block) -> Option<Value> {
        let blocks = self.cfg.reachable.len();
        self.path.clear();
        let mut block = start;
        let value = loop {
            let cached = self.memo[var.local * blocks + block.index()];
            if cached.0 == self.stamp {
                break cached.1;
            }
            if self.path.len() > blocks {
                return None;
            }
            self.path.push(block);
            match self.step(build, var, block)? {
                Step::Done(value) => break value,
                Step::Up(pred) => block = pred,
            }
        };
        for &block in &self.path {
            self.memo[var.local * blocks + block.index()] = (self.stamp, value);
        }
        Some(value)
    }

    fn exit(&mut self, build: &mut Build, var: SlotVar, block: Block) -> Option<Value> {
        match self.block_def(block) {
            Some(def) => Some(def),
            None => self.entry(build, var, block),
        }
    }

    fn fill(&mut self, build: &mut Build) -> Option<()> {
        let mut next = 0;
        while next < build.phis.len() {
            let (block, var) = (build.phis[next].block, build.phis[next].var);
            let count = self.cfg.preds_of(block).len();
            let mut args = Vec::with_capacity(count);
            for at in 0..count {
                let pred = self.cfg.preds_of(block)[at];
                args.push(self.exit(build, var, pred)?);
            }
            build.phis[next].args = args;
            next += 1;
        }
        Some(())
    }

    fn trivial(&mut self, phi: &Phi) -> Option<Value> {
        let mut same = None;
        for arg in &phi.args {
            let value = self.resolve(*arg);
            if value == phi.value || same == Some(value) {
                continue;
            }
            if same.is_some() {
                return None;
            }
            same = Some(value);
        }
        let ty = self.scan.vars[phi.var.index].home();
        Some(same.unwrap_or_else(|| self.fresh.zero(&mut self.subst, ty)))
    }

    fn minimize(&mut self, phis: &mut Vec<Phi>) {
        loop {
            let before = phis.len();
            phis.retain(|phi| match self.trivial(phi) {
                Some(value) => {
                    self.set(phi.value, value);
                    false
                }
                None => true,
            });
            if phis.len() == before {
                return;
            }
        }
    }

    fn build_var(&mut self, index: usize) -> Option<Build> {
        self.prepare(1);
        self.load_var_defs(index);
        let mut build = Build::default();
        let var = SlotVar { local: 0, index };
        let home = self.scan.vars[index].home();
        for exposed in 0..self.local.exposed[index].len() {
            let (block, load) = self.local.exposed[index][exposed];
            let value = self.entry(&mut build, var, block)?;
            self.bind(&mut build, home, load, value);
        }
        for forward in 0..self.local.forward[index].len() {
            let (load, def) = self.local.forward[index][forward];
            self.bind(&mut build, home, load, def);
        }
        self.fill(&mut build)?;
        self.minimize(&mut build.phis);
        let blocked = build
            .phis
            .iter()
            .any(|phi| self.try_target[phi.block.index()]);
        (!blocked).then_some(build)
    }

    fn apply_var(&mut self, func: &mut Function, var: usize, build: Build) {
        self.undo.clear();
        for &(inst, cast) in &self.local.store_casts[var] {
            let val = match func.insts[inst.index()].op {
                Op::StackStore(v, ..) | Op::Store(v, ..) => v,
                _ => continue,
            };
            func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, val);
            func.insts[inst.index()].result = Some(cast);
            self.kept[inst.index()] = true;
        }
        for (inst, op, value) in build.converts {
            func.insts[inst.index()].op = Op::Convert(op, value);
            self.kept[inst.index()] = true;
        }
        for phi in build.phis {
            func.blocks[phi.block.index()].params.push(phi.value);
            let preds = self.cfg.preds_of(phi.block);
            for (i, (pred, arg)) in preds.iter().zip(&phi.args).enumerate() {
                if !preds[..i].contains(pred) {
                    func.push_edge_arg(*pred, phi.block, *arg);
                }
            }
        }
        self.promoted_vars[var] = true;
    }

    fn dropped(&self, func: &Function) -> Vec<bool> {
        let mut dropped = vec![false; func.insts.len()];
        for access in &self.scan.accesses {
            if self.promoted_vars[access.var] && !self.kept[access.inst.index()] {
                dropped[access.inst.index()] = true;
            }
        }
        let live_addr_slot =
            join_slots::live_addr_slots(func, &self.scan.addrs, &self.scan.addr_insts, &dropped);
        let all_promoted = |slot: usize| {
            !self.scan.escaped[slot]
                && !live_addr_slot[slot]
                && !self.scan.slot_vars[slot].is_empty()
                && self.scan.slot_vars[slot]
                    .iter()
                    .all(|&var| self.promoted_vars[var])
        };
        for (inst, slot) in &self.scan.addr_insts {
            if all_promoted(slot.index()) {
                dropped[inst.index()] = true;
            }
        }
        dropped
    }

    fn finish(mut self, func: &mut Function, entry: Block) {
        func.types.extend(&self.fresh.types);
        let dropped = self.dropped(func);
        for block in &mut func.blocks {
            block.insts.retain(|inst| !dropped[inst.index()]);
        }
        let zeros = materialize(func, &self.fresh.zeros);
        func.blocks[entry.index()].insts.splice(0..0, zeros);
        for block in &func.blocks {
            for inst in &block.insts {
                func.insts[inst.index()]
                    .op
                    .map_operands(|value| crate::merge::find_compress(&mut self.subst, value));
            }
        }
        let unwritten: Vec<Value> = self.fresh.zeros.iter().map(|&(_, value)| value).collect();
        let insts = &func.insts;
        for block in &mut func.blocks {
            block.insts.retain(|inst| {
                !matches!(
                    insts[inst.index()].op,
                    Op::Store(value, ..) | Op::StackStore(value, ..) if unwritten.contains(&value)
                )
            });
        }
    }
}

fn materialize(func: &mut Function, zeros: &[(Type, Value)]) -> Vec<Inst> {
    let mut insts = Vec::new();
    for &(ty, value) in zeros {
        if ty != Type::I128 {
            insts.push(func.add(Op::Const(0), Some(value)));
            continue;
        }
        let half = func.value(Type::I64);
        insts.push(func.add(Op::Const(0), Some(half)));
        insts.push(func.add(Op::Binary(Binary::Iconcat, half, half), Some(value)));
    }
    insts
}

/// Promotes every promotable variable `scan` found.
pub(super) fn promote(func: &mut Function, entry: Block, scan: Scan) -> bool {
    let count = func.types.len();
    let mut fresh = Fresh {
        base: count,
        types: Vec::new(),
        zeros: Vec::new(),
    };
    let mut subst = crate::merge::identity(count);
    let local = Local::new(&scan, &mut fresh, &mut subst);
    let promoted_vars = vec![false; scan.vars.len()];
    let kept = vec![false; func.insts.len()];
    let blocks = func.blocks.len();
    let mut pass = Pass {
        cfg: func.cfg(),
        try_target: try_targets(func),
        scan,
        local,
        fresh,
        subst,
        undo: Vec::new(),
        memo: Vec::new(),
        stamp: 0,
        block_defs: vec![(0, Value(0)); blocks],
        def_stamp: 0,
        path: Vec::new(),
        promoted_vars,
        kept,
    };
    for slot in 0..func.slots.len() {
        if pass.scan.escaped[slot] {
            continue;
        }
        for idx in 0..pass.scan.slot_vars[slot].len() {
            let var = pass.scan.slot_vars[slot][idx];
            if !pass.scan.var_promotable(slot, var) {
                continue;
            }
            let mark = pass.fresh.types.len();
            match pass.build_var(var) {
                Some(build) => pass.apply_var(func, var, build),
                None => pass.rollback(mark),
            }
        }
    }
    let changed = pass.promoted_vars.iter().any(|&p| p);
    pass.finish(func, entry);
    changed
}
