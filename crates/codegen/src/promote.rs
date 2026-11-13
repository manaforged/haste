use rustc_hash::FxHashMap as HashMap;

#[path = "promote_subword.rs"]
mod subword;

use crate::ir::{Binary, Block, Convert, Function, Inst, InstData, Op, StackSlot, Type, Value};
use crate::split::{End, Layouts};

#[derive(Clone, Copy)]
enum VarMode {
    Uniform { ty: Type, stored: bool },
    NarrowInt { home: Type, has_store: bool },
    Bitcast { home: Type },
    Invalid,
}

fn same_bitcast_family(a: Type, b: Type) -> bool {
    a.bytes() == b.bytes()
        && (matches!((a, b), (Type::F64 | Type::I64, Type::F64 | Type::I64) | (Type::F32 | Type::I32, Type::F32 | Type::I32))
            || (a.is_vector() && b.is_vector()))
}

/// One promoted place: the bytes at `offset` in `slot`, read and written under one mode.
struct Var {
    slot: StackSlot,
    offset: i64,
    end: i64,
    mode: VarMode,
}

impl Var {
    fn new(slot: StackSlot, offset: i64, ty: Type) -> Self {
        Var { slot, offset, end: offset + i64::from(ty.bytes()), mode: VarMode::Uniform { ty, stored: false } }
    }

    fn home(&self) -> Type {
        match self.mode {
            VarMode::Uniform { ty, .. } | VarMode::NarrowInt { home: ty, .. } | VarMode::Bitcast { home: ty } => ty,
            VarMode::Invalid => Type::I8,
        }
    }

    fn sound(&self) -> bool {
        !matches!(self.mode, VarMode::Invalid)
    }

    fn add(&mut self, ty: Type, is_store: bool) {
        self.end = self.end.max(self.offset + i64::from(ty.bytes()));
        self.mode = match self.mode {
            VarMode::Uniform { ty: u, stored } if ty == u => VarMode::Uniform { ty: u, stored: stored || is_store },
            VarMode::Uniform { ty: u, stored } if same_bitcast_family(u, ty) => {
                let home = if stored || !is_store { u } else { ty };
                VarMode::Bitcast { home }
            }
            VarMode::Uniform { ty: u, stored } if u.is_small_int() && ty.is_small_int() => match (stored, is_store) {
                (true, true) => VarMode::Invalid,
                (true, false) if ty.bytes() > u.bytes() => VarMode::Invalid,
                (true, false) => VarMode::NarrowInt { home: u, has_store: true },
                (false, true) if u.bytes() > ty.bytes() => VarMode::Invalid,
                (false, true) => VarMode::NarrowInt { home: ty, has_store: true },
                (false, false) => VarMode::NarrowInt { home: if ty.bytes() > u.bytes() { ty } else { u }, has_store: false },
            },
            VarMode::NarrowInt { home, has_store } if ty.is_small_int() => {
                if is_store {
                    if (has_store && ty != home) || (!has_store && home.bytes() > ty.bytes()) {
                        VarMode::Invalid
                    } else {
                        VarMode::NarrowInt { home: ty, has_store: true }
                    }
                } else if has_store && ty.bytes() > home.bytes() {
                    VarMode::Invalid
                } else {
                    let next = if !has_store && ty.bytes() > home.bytes() { ty } else { home };
                    VarMode::NarrowInt { home: next, has_store }
                }
            }
            VarMode::Bitcast { home } if same_bitcast_family(home, ty) => VarMode::Bitcast { home },
            _ => VarMode::Invalid,
        };
    }
}

fn convert(from: Type, to: Type) -> Option<Convert> {
    if from == to {
        return None;
    }
    if from.is_float() || to.is_float() || from.is_vector() || to.is_vector() {
        assert_eq!(from.bytes(), to.bytes());
        return Some(Convert::Bitcast);
    }
    Some(if to.bytes() < from.bytes() { Convert::Ireduce } else { Convert::Uextend })
}

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
    ty: Type,
}

struct Touch {
    addr: Option<Value>,
    slot: StackSlot,
    offset: Option<i64>,
    kind: Kind,
}

#[derive(Default)]
struct Scan {
    addrs: Vec<Option<(StackSlot, i64)>>,
    addr_insts: Vec<(Inst, StackSlot)>,
    escaped: Vec<bool>,
    vars: Vec<Var>,
    ids: HashMap<(StackSlot, i64), usize>,
    slot_vars: Vec<Vec<usize>>,
    accesses: Vec<Access>,
}

impl Scan {
    fn new(func: &Function) -> Self {
        let slots = func.slots.len();
        let mut scan = Scan { addrs: vec![None; func.types.len()], escaped: vec![false; slots], slot_vars: vec![Vec::new(); slots], ..Scan::default() };
        scan.addresses(func);
        for (index, block) in func.blocks.iter().enumerate() {
            for inst in &block.insts {
                scan.classify(func, Block(index as u32), *inst);
            }
        }
        scan
    }

    fn addresses(&mut self, func: &Function) {
        (self.addrs, self.addr_insts) = func.slot_offsets(|_| None);
    }

    fn classify(&mut self, func: &Function, block: Block, inst: Inst) {
        let data = &func.insts[inst.index()];
        if matches!(data.op, Op::Move(..)) {
            return;
        }
        let mut skip = [None, None];
        if let Some(touch) = self.touch(data) {
            skip[0] = touch.addr;
            self.access(func, (block, inst), (touch.slot, touch.offset), touch.kind);
        }
        if let Op::Binary(Binary::Iadd, x, y) = &data.op
            && data.result.is_some_and(|result| self.addrs[result.index()].is_some())
        {
            skip[1] = if self.addrs[x.index()].is_some() {
                Some(*x)
            } else if self.addrs[y.index()].is_some() {
                Some(*y)
            } else {
                None
            };
        }
        data.op.for_operands(|value| {
            if let Some(at) = skip.iter().position(|skipped| *skipped == Some(value)) {
                skip[at] = None;
            } else if let Some((slot, _)) = self.addrs[value.index()] {
                self.escaped[slot.index()] = true;
            }
        });
    }

    fn touch(&self, data: &InstData) -> Option<Touch> {
        match &data.op {
            Op::StackLoad(slot, offset) => Some(Touch { addr: None, slot: *slot, offset: Some(i64::from(*offset)), kind: Kind::Load(data.result?) }),
            Op::StackStore(value, slot, offset) => Some(Touch { addr: None, slot: *slot, offset: Some(i64::from(*offset)), kind: Kind::Store(*value) }),
            Op::Load(addr, offset) => self.through(*addr, *offset, Kind::Load(data.result?)),
            Op::Store(value, addr, offset) => self.through(*addr, *offset, Kind::Store(*value)),
            _ => None,
        }
    }

    fn through(&self, addr: Value, offset: i32, kind: Kind) -> Option<Touch> {
        let (slot, base) = self.addrs[addr.index()]?;
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
        let var = match self.ids.get(&(slot, offset)) {
            Some(var) => *var,
            None => {
                let var = self.vars.len();
                self.vars.push(Var::new(slot, offset, ty));
                self.ids.insert((slot, offset), var);
                self.slot_vars[slot.index()].push(var);
                var
            }
        };
        self.vars[var].add(ty, matches!(kind, Kind::Store(_)));
        self.accesses.push(Access { block: at.0, inst: at.1, var, kind, ty });
    }

    /// The slot and offset `addr` points into, when it is a slot address.
    fn end(&self, addr: Value) -> End {
        self.addrs[addr.index()].map(|(slot, offset)| (slot.index(), offset))
    }

    /// The layouts the accesses give each slot, spread through the moves of `func`.
    fn layouts(&self, func: &Function) -> Layouts {
        let accesses = self.accesses.iter().map(|access| (self.vars[access.var].slot.index(), self.vars[access.var].offset, access.ty));
        let mut layouts = Layouts::new(func.slots.len(), accesses);
        let moves: Vec<(End, End, u32)> = func.blocks.iter().flat_map(|block| &block.insts).filter_map(|inst| match func.insts[inst.index()].op {
            Op::Move(to, from, bytes) => Some((self.end(to), self.end(from), bytes)),
            _ => None,
        }).collect();
        layouts.spread(&moves);
        layouts
    }

    /// Marks as escaped every slot a move that stayed whole reads or writes, since promotion would drop the stores it reads and miss the bytes it writes.
    fn pin_moves(&mut self, func: &Function) {
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            if let Op::Move(to, from, _) = func.insts[inst.index()].op {
                for addr in [to, from] {
                    if let Some((slot, _)) = self.addrs[addr.index()] {
                        self.escaped[slot.index()] = true;
                    }
                }
            }
        }
    }

    /// Whether `addr` is the address of a slot that nothing but moves lets escape.
    fn open(&self, addr: Value) -> bool {
        self.addrs[addr.index()].is_some_and(|(slot, _)| !self.escaped[slot.index()])
    }

    fn promotable(&self, slot: usize) -> bool {
        if self.escaped[slot] || self.slot_vars[slot].iter().any(|var| !self.vars[*var].sound()) {
            return false;
        }
        let mut spans: Vec<(i64, i64)> = self.slot_vars[slot].iter().map(|var| (self.vars[*var].offset, self.vars[*var].end)).collect();
        spans.sort_unstable();
        !spans.windows(2).any(|pair| pair[1].0 < pair[0].1)
    }
}

/// A load that promotion replaces, with the type it reads.
#[derive(Clone, Copy)]
struct Load {
    inst: Inst,
    result: Value,
    ty: Type,
}

struct Local {
    defs: HashMap<(usize, Block), Value>,
    exposed: Vec<Vec<(Block, Load)>>,
    forward: Vec<Vec<(Load, Value)>>,
    store_casts: Vec<Vec<(Inst, Value)>>,
}

impl Local {
    fn new(scan: &Scan, fresh: &mut Fresh, subst: &mut Vec<Value>) -> Self {
        let count = scan.vars.len();
        let mut local = Local {
            defs: HashMap::default(),
            exposed: vec![Vec::new(); count],
            forward: vec![Vec::new(); count],
            store_casts: vec![Vec::new(); count],
        };
        for access in &scan.accesses {
            let key = (access.var, access.block);
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
                    local.defs.insert(key, def);
                }
                Kind::Load(result) => {
                    let load = Load { inst: access.inst, result, ty: access.ty };
                    match local.defs.get(&key) {
                        Some(def) => local.forward[access.var].push((load, *def)),
                        None => {
                            local.exposed[access.var].push((access.block, load));
                            if access.ty == scan.vars[access.var].home() {
                                local.defs.insert(key, result);
                            }
                        }
                    }
                }
            }
        }
        local
    }

    /// The value `var` holds at the end of `block` when the block writes or reads it.
    fn def(&self, var: usize, block: Block) -> Option<Value> {
        self.defs.get(&(var, block)).copied()
    }
}

fn try_targets(func: &Function) -> Vec<bool> {
    let mut try_target = vec![false; func.blocks.len()];
    for index in 0..func.blocks.len() {
        let Some(op) = func.last(Block(index as u32)) else { continue };
        if matches!(op, Op::TryCall { .. } | Op::BrTable(..)) {
            op.targets().for_each(|target| try_target[target.index()] = true);
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
    cfg: crate::cfg::Cfg,
    try_target: Vec<bool>,
    scan: Scan,
    local: Local,
    fresh: Fresh,
    subst: Vec<Value>,
    undo: Vec<(u32, Value)>,
    memo: Vec<(u32, Value)>,
    stamp: u32,
    path: Vec<Block>,
    promoted: Vec<bool>,
    kept: Vec<bool>,
}

impl Pass {
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
        let need = vars * self.cfg.preds.len();
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
        let preds = &self.cfg.preds[block.index()];
        if self.cfg.entry == Some(block) && !preds.is_empty() {
            return None;
        }
        if !self.cfg.reachable[block.index()] || preds.is_empty() {
            return Some(Step::Done(self.fresh.zero(&mut self.subst, ty)));
        }
        if let [pred] = preds[..] {
            return Some(match self.local.def(var.index, pred) {
                Some(def) => Step::Done(def),
                None => Step::Up(pred),
            });
        }
        let value = self.fresh.value(&mut self.subst, ty);
        build.phis.push(Phi { value, block, var, args: Vec::new() });
        Some(Step::Done(value))
    }

    fn entry(&mut self, build: &mut Build, var: SlotVar, start: Block) -> Option<Value> {
        let blocks = self.cfg.preds.len();
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
        match self.local.def(var.index, block) {
            Some(def) => Some(def),
            None => self.entry(build, var, block),
        }
    }

    fn fill(&mut self, build: &mut Build) -> Option<()> {
        let mut next = 0;
        while next < build.phis.len() {
            let (block, var) = (build.phis[next].block, build.phis[next].var);
            let count = self.cfg.preds[block.index()].len();
            let mut args = Vec::with_capacity(count);
            for at in 0..count {
                let pred = self.cfg.preds[block.index()][at];
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

    fn build(&mut self, slot: usize) -> Option<Build> {
        let vars = self.scan.slot_vars[slot].len();
        self.prepare(vars);
        let mut build = Build::default();
        for local in 0..vars {
            let index = self.scan.slot_vars[slot][local];
            let var = SlotVar { local, index };
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
        }
        self.fill(&mut build)?;
        self.minimize(&mut build.phis);
        let blocked = build.phis.iter().any(|phi| self.try_target[phi.block.index()]);
        (!blocked).then_some(build)
    }

    fn apply(&mut self, func: &mut Function, slot: usize, build: Build) {
        self.undo.clear();
        for &var in &self.scan.slot_vars[slot] {
            for &(inst, cast) in &self.local.store_casts[var] {
                let val = match func.insts[inst.index()].op {
                    Op::StackStore(v, ..) | Op::Store(v, ..) => v,
                    _ => continue,
                };
                func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, val);
                func.insts[inst.index()].result = Some(cast);
                self.kept[inst.index()] = true;
            }
        }
        for (inst, op, value) in build.converts {
            func.insts[inst.index()].op = Op::Convert(op, value);
            self.kept[inst.index()] = true;
        }
        for phi in build.phis {
            func.blocks[phi.block.index()].params.push(phi.value);
            let preds = &self.cfg.preds[phi.block.index()];
            for (i, (pred, arg)) in preds.iter().zip(&phi.args).enumerate() {
                if !preds[..i].contains(pred) {
                    push_edge_arg(func, *pred, phi.block, *arg);
                }
            }
        }
        self.promoted[slot] = true;
    }

    fn dropped(&self, count: usize) -> Vec<bool> {
        let mut dropped = vec![false; count];
        for access in &self.scan.accesses {
            if self.promoted[self.scan.vars[access.var].slot.index()] && !self.kept[access.inst.index()] {
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

    fn finish(mut self, func: &mut Function, entry: Block) {
        func.types.extend(&self.fresh.types);
        let dropped = self.dropped(func.insts.len());
        for block in &mut func.blocks {
            block.insts.retain(|inst| !dropped[inst.index()]);
        }
        let zeros = materialize(func, &self.fresh.zeros);
        func.blocks[entry.index()].insts.splice(0..0, zeros);
        for block in &func.blocks {
            for inst in &block.insts {
                func.insts[inst.index()].op.map_operands(|value| crate::merge::find_compress(&mut self.subst, value));
            }
        }
    }
}

fn push_edge_arg(func: &mut Function, pred: Block, target: Block, arg: Value) {
    let Some(&inst) = func.blocks[pred.index()].insts.last() else {
        return;
    };
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

/// Which moves promotion splits into pieces.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Moves {
    /// Every move that touches a slot only moves let escape.
    Split,
    /// Those of them that an access of either end gives a layout for; a body kept for copying leaves the others whole for the caller, which may know how their bytes are read.
    Guided,
    /// Every move that touches a slot only moves let escape, into words, without reading layouts.
    Words,
}

pub(crate) fn step(func: &mut Function, moves: Moves) -> bool {
    let Some(entry) = func.entry.filter(|_| !func.slots.is_empty()) else {
        return false;
    };
    let mut scan = Scan::new(func);
    let mut splits = Vec::new();
    if func.insts.iter().any(|data| matches!(data.op, Op::Move(to, from, _) if scan.open(to) || scan.open(from))) {
        let layouts = (moves != Moves::Words).then(|| scan.layouts(func));
        let pieces = |to: Value, from: Value, bytes: u32| match layouts.as_ref().and_then(|layouts| layouts.pieces(scan.end(to), scan.end(from), bytes)) {
            None if moves != Moves::Guided => Some(crate::copy::words(bytes)),
            guided => guided,
        };
        splits = crate::copy::expand(func, |to, from, bytes| if scan.open(to) || scan.open(from) { pieces(to, from, bytes) } else { None });
        scan = Scan::new(func);
        scan.pin_moves(func);
    }
    let split_changed = moves == Moves::Split && subword::split(func, &scan);
    if split_changed {
        scan = Scan::new(func);
        scan.pin_moves(func);
    }
    let changed = ((0..func.slots.len()).any(|slot| scan.promotable(slot)) && promote(func, entry, scan)) || split_changed;
    crate::copy::rejoin(func, &splits);
    changed
}

/// Promotes every promotable slot `scan` found.
fn promote(func: &mut Function, entry: Block, scan: Scan) -> bool {
    let count = func.types.len();
    let mut fresh = Fresh { base: count, types: Vec::new(), zeros: Vec::new() };
    let mut subst = crate::merge::identity(count);
    let local = Local::new(&scan, &mut fresh, &mut subst);
    let promoted = vec![false; func.slots.len()];
    let kept = vec![false; func.insts.len()];
    let mut pass = Pass { cfg: crate::cfg::Cfg::from_entry(func, entry), try_target: try_targets(func), scan, local, fresh, subst, undo: Vec::new(), memo: Vec::new(), stamp: 0, path: Vec::new(), promoted, kept };
    for slot in 0..func.slots.len() {
        if !pass.scan.promotable(slot) {
            continue;
        }
        let mark = pass.fresh.types.len();
        match pass.build(slot) {
            Some(build) => pass.apply(func, slot, build),
            None => pass.rollback(mark),
        }
    }
    let changed = pass.promoted.iter().any(|&p| p);
    pass.finish(func, entry);
    changed
}
