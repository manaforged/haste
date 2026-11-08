use rustc_hash::FxHashMap as HashMap;

use crate::ir::{Binary, Block, Function, Inst, InstData, Op, StackSlot, Type, Value};

type Var = (StackSlot, i64, Type);

#[derive(Clone, Copy)]
enum Kind {
    Load(Value),
    Store(Value),
}

impl Kind {
    fn value(self) -> Value {
        match self {
            Kind::Load(value) | Kind::Store(value) => value,
        }
    }
}

struct Access {
    block: Block,
    inst: Inst,
    var: usize,
    kind: Kind,
}

struct Touch {
    addr: Option<Value>,
    slot: StackSlot,
    offset: Option<i64>,
    kind: Kind,
}

#[derive(Default)]
struct Scan {
    addrs: HashMap<Value, (StackSlot, i64)>,
    addr_insts: Vec<(Inst, StackSlot)>,
    escaped: Vec<bool>,
    vars: Vec<Var>,
    ids: HashMap<Var, usize>,
    slot_vars: Vec<Vec<usize>>,
    accesses: Vec<Access>,
}

impl Scan {
    fn new(func: &Function) -> Self {
        let slots = func.slots.len();
        let mut scan = Scan { escaped: vec![false; slots], slot_vars: vec![Vec::new(); slots], ..Scan::default() };
        scan.addresses(func);
        for (index, block) in func.blocks.iter().enumerate() {
            for inst in &block.insts {
                scan.classify(func, Block(index as u32), *inst);
            }
        }
        scan
    }

    fn addresses(&mut self, func: &Function) {
        let mut consts = HashMap::default();
        let mut adds = Vec::new();
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            let data = &func.insts[inst.index()];
            let Some(result) = data.result else { continue };
            match &data.op {
                Op::StackAddr(slot, offset) => {
                    self.addrs.insert(result, (*slot, i64::from(*offset)));
                    self.addr_insts.push((*inst, *slot));
                }
                Op::Const(imm) => {
                    consts.insert(result, *imm);
                }
                Op::Binary(Binary::Iadd, x, c) => adds.push((*inst, result, *x, *c)),
                _ => {}
            }
        }
        self.derive(&consts, adds);
    }

    fn derive(&mut self, consts: &HashMap<Value, i64>, mut adds: Vec<(Inst, Value, Value, Value)>) {
        loop {
            let before = adds.len();
            adds.retain(|&(inst, result, x, c)| {
                let (Some(&(slot, base)), Some(&imm)) = (self.addrs.get(&x), consts.get(&c)) else { return true };
                let Some(offset) = base.checked_add(imm) else { return true };
                self.addrs.insert(result, (slot, offset));
                self.addr_insts.push((inst, slot));
                false
            });
            if adds.len() == before {
                return;
            }
        }
    }

    fn classify(&mut self, func: &Function, block: Block, inst: Inst) {
        let data = &func.insts[inst.index()];
        let mut uses = data.op.operands();
        if let Some(touch) = self.touch(data) {
            if let Some(addr) = touch.addr {
                take(&mut uses, addr);
            }
            self.access(func, (block, inst), (touch.slot, touch.offset), touch.kind);
        }
        if let Op::Binary(Binary::Iadd, x, _) = &data.op
            && data.result.is_some_and(|result| self.addrs.contains_key(&result))
        {
            take(&mut uses, *x);
        }
        for value in uses {
            if let Some(&(slot, _)) = self.addrs.get(&value) {
                self.escaped[slot.index()] = true;
            }
        }
    }

    fn touch(&self, data: &InstData) -> Option<Touch> {
        match &data.op {
            Op::StackLoad(slot, offset) => {
                Some(Touch { addr: None, slot: *slot, offset: Some(i64::from(*offset)), kind: Kind::Load(data.result?) })
            }
            Op::StackStore(value, slot, offset) => {
                Some(Touch { addr: None, slot: *slot, offset: Some(i64::from(*offset)), kind: Kind::Store(*value) })
            }
            Op::Load(addr, offset) => self.through(*addr, *offset, Kind::Load(data.result?)),
            Op::Store(value, addr, offset) => self.through(*addr, *offset, Kind::Store(*value)),
            _ => None,
        }
    }

    fn through(&self, addr: Value, offset: i32, kind: Kind) -> Option<Touch> {
        let &(slot, base) = self.addrs.get(&addr)?;
        Some(Touch { addr: Some(addr), slot, offset: base.checked_add(i64::from(offset)), kind })
    }

    fn access(&mut self, func: &Function, at: (Block, Inst), place: (StackSlot, Option<i64>), kind: Kind) {
        let (slot, offset) = place;
        let ty = func.ty(kind.value());
        let size = i64::from(func.slots[slot.index()].size);
        let Some(offset) = offset.filter(|offset| *offset >= 0 && offset + i64::from(ty.bytes()) <= size) else {
            self.escaped[slot.index()] = true;
            return;
        };
        let key = (slot, offset, ty);
        let var = match self.ids.get(&key) {
            Some(var) => *var,
            None => {
                let var = self.vars.len();
                self.vars.push(key);
                self.ids.insert(key, var);
                self.slot_vars[slot.index()].push(var);
                var
            }
        };
        self.accesses.push(Access { block: at.0, inst: at.1, var, kind });
    }

    fn promotable(&self, slot: usize) -> bool {
        if self.escaped[slot] {
            return false;
        }
        let mut spans: Vec<(i64, i64)> = self.slot_vars[slot]
            .iter()
            .map(|var| {
                let (_, offset, ty) = self.vars[*var];
                (offset, offset + i64::from(ty.bytes()))
            })
            .collect();
        spans.sort_unstable();
        !spans.windows(2).any(|pair| pair[1].0 < pair[0].1)
    }
}

fn take(uses: &mut Vec<Value>, value: Value) {
    if let Some(position) = uses.iter().position(|used| *used == value) {
        uses.swap_remove(position);
    }
}

struct Local {
    defs: Vec<HashMap<Block, Value>>,
    exposed: Vec<Vec<(Block, Value)>>,
    forward: Vec<Vec<(Value, Value)>>,
}

impl Local {
    fn new(scan: &Scan) -> Self {
        let count = scan.vars.len();
        let mut local =
            Local { defs: vec![HashMap::default(); count], exposed: vec![Vec::new(); count], forward: vec![Vec::new(); count] };
        for access in &scan.accesses {
            let defs = &mut local.defs[access.var];
            match access.kind {
                Kind::Store(value) => {
                    defs.insert(access.block, value);
                }
                Kind::Load(result) => match defs.get(&access.block) {
                    Some(def) => local.forward[access.var].push((result, *def)),
                    None => {
                        local.exposed[access.var].push((access.block, result));
                        defs.insert(access.block, result);
                    }
                },
            }
        }
        local
    }
}

struct Cfg {
    entry: Block,
    preds: Vec<Vec<Block>>,
    reachable: Vec<bool>,
    try_target: Vec<bool>,
}

fn terminator(func: &Function, block: Block) -> Option<&Op> {
    let inst = func.blocks[block.index()].insts.last()?;
    Some(&func.insts[inst.index()].op).filter(|op| op.is_terminator())
}

fn targets(op: &Op) -> Vec<Block> {
    match op {
        Op::Jump(block, _) => vec![*block],
        Op::Brif(_, then, _, otherwise, _) => vec![*then, *otherwise],
        Op::TryCall { normal, handler, .. } => vec![*normal, *handler],
        _ => Vec::new(),
    }
}

impl Cfg {
    fn new(func: &Function, entry: Block) -> Self {
        let count = func.blocks.len();
        let mut cfg =
            Cfg { entry, preds: vec![Vec::new(); count], reachable: vec![false; count], try_target: vec![false; count] };
        for index in 0..count {
            let block = Block(index as u32);
            let Some(op) = terminator(func, block) else { continue };
            for target in targets(op) {
                let preds = &mut cfg.preds[target.index()];
                if !preds.contains(&block) {
                    preds.push(block);
                }
                if matches!(op, Op::TryCall { .. }) {
                    cfg.try_target[target.index()] = true;
                }
            }
        }
        cfg.reach(func);
        cfg
    }

    fn reach(&mut self, func: &Function) {
        let mut stack = vec![self.entry];
        while let Some(block) = stack.pop() {
            if std::mem::replace(&mut self.reachable[block.index()], true) {
                continue;
            }
            if let Some(op) = terminator(func, block) {
                stack.extend(targets(op));
            }
        }
    }
}

struct Fresh {
    base: usize,
    types: Vec<Type>,
    zeros: Vec<(Type, Value)>,
}

impl Fresh {
    fn value(&mut self, ty: Type) -> Value {
        self.types.push(ty);
        Value((self.base + self.types.len() - 1) as u32)
    }

    fn zero(&mut self, ty: Type) -> Value {
        if let Some((_, value)) = self.zeros.iter().find(|(zero, _)| *zero == ty) {
            return *value;
        }
        let value = self.value(ty);
        self.zeros.push((ty, value));
        value
    }

    fn rollback(&mut self, mark: usize) {
        self.types.truncate(mark);
        let limit = self.base + mark;
        self.zeros.retain(|(_, value)| value.index() < limit);
    }
}

struct Phi {
    value: Value,
    block: Block,
    var: usize,
    args: Vec<Value>,
    live: bool,
}

#[derive(Default)]
struct Build {
    memo: HashMap<(usize, Block), Value>,
    phis: Vec<Phi>,
    staged: HashMap<Value, Value>,
}

enum Step {
    Done(Value),
    Up(Block),
}

struct Pass {
    cfg: Cfg,
    scan: Scan,
    local: Local,
    fresh: Fresh,
    subst: HashMap<Value, Value>,
    promoted: Vec<bool>,
}

impl Pass {
    fn resolve(&self, build: &Build, mut value: Value) -> Value {
        while let Some(next) = build.staged.get(&value).or_else(|| self.subst.get(&value)) {
            value = *next;
        }
        value
    }

    fn step(&mut self, build: &mut Build, var: usize, block: Block) -> Option<Step> {
        let ty = self.scan.vars[var].2;
        let preds = &self.cfg.preds[block.index()];
        if block == self.cfg.entry && !preds.is_empty() {
            return None;
        }
        if !self.cfg.reachable[block.index()] || preds.is_empty() {
            return Some(Step::Done(self.fresh.zero(ty)));
        }
        if let [pred] = preds[..] {
            return Some(match self.local.defs[var].get(&pred) {
                Some(def) => Step::Done(*def),
                None => Step::Up(pred),
            });
        }
        let value = self.fresh.value(ty);
        build.phis.push(Phi { value, block, var, args: Vec::new(), live: true });
        Some(Step::Done(value))
    }

    fn entry(&mut self, build: &mut Build, var: usize, start: Block) -> Option<Value> {
        let mut path = Vec::new();
        let mut block = start;
        let value = loop {
            if let Some(value) = build.memo.get(&(var, block)) {
                break *value;
            }
            if path.len() > self.cfg.preds.len() {
                return None;
            }
            path.push(block);
            match self.step(build, var, block)? {
                Step::Done(value) => break value,
                Step::Up(pred) => block = pred,
            }
        };
        for block in path {
            build.memo.insert((var, block), value);
        }
        Some(value)
    }

    fn exit(&mut self, build: &mut Build, var: usize, block: Block) -> Option<Value> {
        match self.local.defs[var].get(&block) {
            Some(def) => Some(*def),
            None => self.entry(build, var, block),
        }
    }

    fn fill(&mut self, build: &mut Build) -> Option<()> {
        let mut next = 0;
        while next < build.phis.len() {
            let (block, var) = (build.phis[next].block, build.phis[next].var);
            let mut args = Vec::new();
            for pred in self.cfg.preds[block.index()].clone() {
                args.push(self.exit(build, var, pred)?);
            }
            build.phis[next].args = args;
            next += 1;
        }
        Some(())
    }

    fn trivial(&mut self, build: &Build, index: usize) -> Option<Value> {
        let phi = &build.phis[index];
        let mut same = None;
        for arg in &phi.args {
            let value = self.resolve(build, *arg);
            if value == phi.value || same == Some(value) {
                continue;
            }
            if same.is_some() {
                return None;
            }
            same = Some(value);
        }
        let ty = self.scan.vars[phi.var].2;
        Some(same.unwrap_or_else(|| self.fresh.zero(ty)))
    }

    fn minimize(&mut self, build: &mut Build) {
        loop {
            let mut changed = false;
            for index in 0..build.phis.len() {
                if !build.phis[index].live {
                    continue;
                }
                if let Some(value) = self.trivial(build, index) {
                    build.staged.insert(build.phis[index].value, value);
                    build.phis[index].live = false;
                    changed = true;
                }
            }
            if !changed {
                return;
            }
        }
    }

    fn build(&mut self, slot: usize) -> Option<Build> {
        let mut build = Build::default();
        for var in self.scan.slot_vars[slot].clone() {
            for (block, load) in self.local.exposed[var].clone() {
                let value = self.entry(&mut build, var, block)?;
                build.staged.insert(load, value);
            }
            for &(load, def) in &self.local.forward[var] {
                build.staged.insert(load, def);
            }
        }
        self.fill(&mut build)?;
        self.minimize(&mut build);
        let blocked = build.phis.iter().any(|phi| phi.live && self.cfg.try_target[phi.block.index()]);
        (!blocked).then_some(build)
    }

    fn apply(&mut self, func: &mut Function, slot: usize, build: Build) {
        self.subst.extend(build.staged);
        for phi in build.phis.into_iter().filter(|phi| phi.live) {
            func.blocks[phi.block.index()].params.push(phi.value);
            for (pred, arg) in self.cfg.preds[phi.block.index()].iter().zip(&phi.args) {
                push_edge_arg(func, *pred, phi.block, *arg);
            }
        }
        self.promoted[slot] = true;
    }

    fn dropped(&self, count: usize) -> Vec<bool> {
        let mut dropped = vec![false; count];
        for access in &self.scan.accesses {
            if self.promoted[self.scan.vars[access.var].0.index()] {
                dropped[access.inst.index()] = true;
            }
        }
        for (inst, slot) in &self.scan.addr_insts {
            if self.promoted[slot.index()] {
                dropped[inst.index()] = true;
            }
        }
        dropped
    }

    fn finish(self, func: &mut Function, entry: Block) {
        func.types.extend(&self.fresh.types);
        let dropped = self.dropped(func.insts.len());
        for block in &mut func.blocks {
            block.insts.retain(|inst| !dropped[inst.index()]);
        }
        let zeros = materialize(func, &self.fresh.zeros);
        func.blocks[entry.index()].insts.splice(0..0, zeros);
        let empty = Build::default();
        for block in &func.blocks {
            for inst in &block.insts {
                func.insts[inst.index()].op.map_operands(|value| self.resolve(&empty, value));
            }
        }
    }
}

fn push_edge_arg(func: &mut Function, pred: Block, target: Block, arg: Value) {
    let Some(&inst) = func.blocks[pred.index()].insts.last() else { return };
    match &mut func.insts[inst.index()].op {
        Op::Jump(block, args) if *block == target => args.push(arg),
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            if *then == target {
                then_args.push(arg);
            }
            if *otherwise == target {
                else_args.push(arg);
            }
        }
        _ => {}
    }
}

fn push_inst(func: &mut Function, op: Op, result: Value) -> Inst {
    func.insts.push(InstData { op, result: Some(result) });
    Inst(func.insts.len() as u32 - 1)
}

fn materialize(func: &mut Function, zeros: &[(Type, Value)]) -> Vec<Inst> {
    let mut insts = Vec::new();
    for &(ty, value) in zeros {
        if ty != Type::I128 {
            insts.push(push_inst(func, Op::Const(0), value));
            continue;
        }
        func.types.push(Type::I64);
        let half = Value(func.types.len() as u32 - 1);
        insts.push(push_inst(func, Op::Const(0), half));
        insts.push(push_inst(func, Op::Binary(Binary::Iconcat, half, half), value));
    }
    insts
}

/// Promotes every stack slot whose address never escapes into SSA values carried in block parameters.
pub(crate) fn run(func: &mut Function) {
    let Some(entry) = func.entry else { return };
    let scan = Scan::new(func);
    let local = Local::new(&scan);
    let fresh = Fresh { base: func.types.len(), types: Vec::new(), zeros: Vec::new() };
    let promoted = vec![false; func.slots.len()];
    let mut pass = Pass { cfg: Cfg::new(func, entry), scan, local, fresh, subst: HashMap::default(), promoted };
    for slot in 0..func.slots.len() {
        if !pass.scan.promotable(slot) {
            continue;
        }
        let mark = pass.fresh.types.len();
        match pass.build(slot) {
            Some(build) => pass.apply(func, slot, build),
            None => pass.fresh.rollback(mark),
        }
    }
    pass.finish(func, entry);
}
