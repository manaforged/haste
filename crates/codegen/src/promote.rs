use rustc_hash::FxHashMap as HashMap;

#[path = "promote_subword.rs"]
mod subword;
#[path = "promote/join_slots.rs"]
mod join_slots;
#[path = "promote/vec_slot.rs"]
mod vec_slot;
#[path = "promote/box_fwd.rs"]
mod box_fwd;
#[path = "promote/cold_copy.rs"]
mod cold_copy;

use crate::ir::{Binary, Block, Convert, Domain, Function, Inst, InstData, Op, StackSlot, Type, Value};
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
        && ((a.bytes() == 8 && (a.is_vector() || matches!(a, Type::F64 | Type::I64)) && (b.is_vector() || matches!(b, Type::F64 | Type::I64)))
            || matches!((a, b), (Type::F32 | Type::I32, Type::F32 | Type::I32))
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
    slot: StackSlot,
    offset: Option<i64>,
    kind: Kind,
}

#[derive(Default)]
struct Scan {
    blocks: usize,
    consts: Vec<Option<i64>>,
    addrs: Vec<Option<(StackSlot, i64)>>,
    addr_insts: Vec<(Inst, StackSlot)>,
    escaped: Vec<bool>,
    open_slots: Vec<bool>,
    vars: Vec<Var>,
    ids: HashMap<(StackSlot, i64), usize>,
    slot_vars: Vec<Vec<usize>>,
    accesses: Vec<Access>,
}

impl Scan {
    fn new(func: &Function) -> Self {
        let slots = func.slots.len();
        let prov = crate::mem::Provenance::compute(func);
        let consts = subword::eval_consts(func);
        let escaped = prov.escaped_slots();
        let mut scan = Scan {
            blocks: func.blocks.len(),
            consts,
            addrs: vec![None; func.types.len()],
            escaped,
            open_slots: Vec::new(),
            slot_vars: vec![Vec::new(); slots],
            ..Scan::default()
        };
        scan.addresses(func);
        scan.open_slots = subword::copy_open_slots(func, &scan.addrs, &scan.consts, &prov.param_roots);
        for u in &func.unwind_drops {
            scan.open_slots[u.local.index()] = false;
            scan.open_slots[u.flag.index()] = false;
        }
        for (index, block) in func.blocks.iter().enumerate() {
            for inst in &block.insts {
                let data = &func.insts[inst.index()];
                if let Some(touch) = scan.touch(data) {
                    scan.access(func, (Block(index as u32), *inst), (touch.slot, touch.offset), touch.kind);
                } else if let Op::Load(addr, _) | Op::Store(_, addr, _) = data.op
                    && let Some(&crate::mem::MemRoot::Slot(s)) = prov.param_roots.get(addr.index())
                {
                    scan.escaped[s.index()] = true;
                    scan.open_slots[s.index()] = false;
                }
            }
        }
        scan
    }

    fn addresses(&mut self, func: &Function) {
        (self.addrs, self.addr_insts) = join_slots::resolve_addrs(func, &self.consts);
    }

    fn touch(&self, data: &InstData) -> Option<Touch> {
        match &data.op {
            Op::StackLoad(slot, offset) => Some(Touch { slot: *slot, offset: Some(i64::from(*offset)), kind: Kind::Load(data.result?) }),
            Op::StackStore(value, slot, offset) => Some(Touch { slot: *slot, offset: Some(i64::from(*offset)), kind: Kind::Store(*value) }),
            Op::Load(addr, offset) => self.through(*addr, *offset, Kind::Load(data.result?)),
            Op::Store(value, addr, offset) => self.through(*addr, *offset, Kind::Store(*value)),
            _ => None,
        }
    }

    fn through(&self, addr: Value, offset: i32, kind: Kind) -> Option<Touch> {
        let (slot, base) = self.addrs[addr.index()]?;
        Some(Touch { slot, offset: base.checked_add(i64::from(offset)), kind })
    }

    fn access(&mut self, func: &Function, at: (Block, Inst), place: (StackSlot, Option<i64>), kind: Kind) {
        let (slot, offset) = place;
        let ty = func.ty(kind.value());
        let size = i64::from(func.slots[slot.index()].size);
        let Some(offset) = offset.filter(|offset| *offset >= 0 && offset + i64::from(ty.bytes()) <= size) else {
            self.escaped[slot.index()] = true;
            self.open_slots[slot.index()] = false;
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

    /// The layouts the accesses give each slot, spread through the moves and constant copies of `func`.
    fn layouts(&self, func: &Function) -> Layouts {
        let accesses = self.accesses.iter().map(|access| (self.vars[access.var].slot.index(), self.vars[access.var].offset, access.ty));
        let mut layouts = Layouts::new(func.slots.len(), accesses);
        let moves: Vec<(End, End, u32)> = func.blocks.iter().flat_map(|block| &block.insts).filter_map(|inst| match func.insts[inst.index()].op {
            Op::Move(to, from, bytes) => Some((self.end(to), self.end(from), bytes)),
            Op::Copy(to, from, len) => self.copy_bytes(to, from, len).map(|bytes| (self.end(to), self.end(from), bytes)),
            _ => None,
        }).collect();
        layouts.spread(&moves);
        layouts
    }

    fn copy_bytes(&self, to: Value, from: Value, len: Value) -> Option<u32> {
        let limit = if self.addrs[to.index()].is_some() && self.addrs[from.index()].is_some() {
            subword::both_copy_bounds(self.blocks).0
        } else {
            subword::MAX_COPY_BYTES
        };
        subword::copy_bytes_limit(&self.consts, len, limit)
    }

    /// Whether `addr` is the address of a slot that nothing but moves lets escape.
    fn open(&self, addr: Value) -> bool {
        self.addrs[addr.index()].is_some_and(|(slot, _)| self.open_slots[slot.index()])
    }

    fn var_promotable(&self, slot: usize, var: usize) -> bool {
        if self.escaped[slot] || !self.vars[var].sound() {
            return false;
        }
        let v = &self.vars[var];
        self.slot_vars[slot].iter().all(|&other| other == var || self.vars[other].end <= v.offset || v.end <= self.vars[other].offset)
    }

    fn any_promotable(&self) -> bool {
        (0..self.slot_vars.len()).any(|slot| !self.escaped[slot] && self.slot_vars[slot].iter().any(|&var| self.var_promotable(slot, var)))
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
    cfg: std::sync::Arc<crate::cfg::Cfg>,
    try_target: Vec<bool>,
    scan: Scan,
    local: Local,
    fresh: Fresh,
    subst: Vec<Value>,
    undo: Vec<(u32, Value)>,
    memo: Vec<(u32, Value)>,
    stamp: u32,
    path: Vec<Block>,
    promoted_vars: Vec<bool>,
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
        match self.local.def(var.index, block) {
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
        let blocked = build.phis.iter().any(|phi| self.try_target[phi.block.index()]);
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
                    push_edge_arg(func, *pred, phi.block, *arg);
                }
            }
        }
        self.promoted_vars[var] = true;
    }

    fn dropped(&self, count: usize) -> Vec<bool> {
        let mut dropped = vec![false; count];
        for access in &self.scan.accesses {
            if self.promoted_vars[access.var] && !self.kept[access.inst.index()] {
                dropped[access.inst.index()] = true;
            }
        }
        let all_promoted = |slot: usize| {
            !self.scan.escaped[slot] && !self.scan.slot_vars[slot].is_empty() && self.scan.slot_vars[slot].iter().all(|&var| self.promoted_vars[var])
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
        let unwritten: Vec<Value> = self.fresh.zeros.iter().map(|&(_, value)| value).collect();
        let insts = &func.insts;
        for block in &mut func.blocks {
            block.insts.retain(|inst| !matches!(insts[inst.index()].op, Op::Store(value, ..) | Op::StackStore(value, ..) if unwritten.contains(&value)));
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

fn pre_split(func: &mut Function, moves: Moves) -> bool {
    let joined = moves != Moves::Words && join_slots::run(func);
    let forwarded = moves == Moves::Split && box_fwd::run(func);
    joined || forwarded
}

fn open_move_pieces(
    scan: &Scan,
    layouts: Option<&Layouts>,
    moves: Moves,
    (to, from, bytes): (Value, Value, u32),
) -> Option<Vec<(Type, u32)>> {
    let (to_open, from_open) = (scan.open(to), scan.open(from));
    if !to_open && !from_open {
        return None;
    }
    let both_open = to_open && from_open;
    let (max_b, max_p) = if both_open {
        subword::both_copy_bounds(scan.blocks)
    } else {
        (crate::copy::MOVE_INLINE * 2, subword::PROMOTE_ONE_PIECES)
    };
    if bytes > max_b {
        return None;
    }
    let word_limit = if both_open { max_p } else { crate::copy::EXPAND_WORDS };
    let (found, limit) = match layouts.and_then(|l| l.pieces(scan.end(to), scan.end(from), bytes)) {
        None if moves != Moves::Guided => (Some(crate::copy::vec_words(bytes)), word_limit),
        guided => (guided, max_p),
    };
    found.filter(|p| p.len() <= limit)
}

fn has_open_move(func: &Function, scan: &Scan) -> bool {
    crate::schedule::has_live_op(func, |op| match *op {
        Op::Move(to, from, _) => scan.open(to) || scan.open(from),
        Op::Copy(to, from, len) => scan.copy_bytes(to, from, len).is_some() && (scan.open(to) || scan.open(from)),
        _ => false,
    })
}

fn run_slot_splits(func: &mut Function, scan: &mut Scan, moves: Moves) -> bool {
    let mut split_changed = moves == Moves::Split && subword::split(func, scan);
    if split_changed {
        *scan = Scan::new(func);
    }
    if moves != Moves::Words && vec_slot::split(func, scan) {
        split_changed = true;
        *scan = Scan::new(func);
    }
    split_changed
}

pub(crate) fn step(func: &mut Function, moves: Moves) -> Domain {
    let Some(entry) = func.entry.filter(|_| !func.slots.is_empty()) else {
        return Domain::NONE;
    };
    let params = crate::copy::forward_param_slots(func);
    let returned = crate::copy::forward_sret_slot(func);
    let join_changed = pre_split(func, moves) | cold_copy::run(func);
    let mut scan = Scan::new(func);
    let (mut splits, mut copy_splits, mut dropped_zero) = (Vec::new(), Vec::new(), false);
    if has_open_move(func, &scan) {
        let layouts = (moves != Moves::Words).then(|| scan.layouts(func));
        let pieces = |to, from, bytes| open_move_pieces(&scan, layouts.as_ref(), moves, (to, from, bytes));
        splits = crate::copy::expand(func, pieces);
        (copy_splits, dropped_zero) = subword::expand_copies(func, |to, from, len| pieces(to, from, scan.copy_bytes(to, from, len)?));
        scan = Scan::new(func);
    }
    let split_changed = run_slot_splits(func, &mut scan, moves);
    let changed = (scan.any_promotable() && promote(func, entry, scan)) || join_changed || split_changed || dropped_zero;
    crate::copy::rejoin(func, &splits);
    let all_copies_rejoined = subword::rejoin_copies(func, &copy_splits);
    let cfg = if returned { Domain::CFG } else { Domain::NONE };
    if changed || returned || params || !all_copies_rejoined { Domain::MEM | Domain::VALUES | cfg } else { Domain::NONE }
}

/// Promotes every promotable variable `scan` found.
fn promote(func: &mut Function, entry: Block, scan: Scan) -> bool {
    let count = func.types.len();
    let mut fresh = Fresh { base: count, types: Vec::new(), zeros: Vec::new() };
    let mut subst = crate::merge::identity(count);
    let local = Local::new(&scan, &mut fresh, &mut subst);
    let promoted_vars = vec![false; scan.vars.len()];
    let kept = vec![false; func.insts.len()];
    let mut pass = Pass { cfg: func.cfg(), try_target: try_targets(func), scan, local, fresh, subst, undo: Vec::new(), memo: Vec::new(), stamp: 0, path: Vec::new(), promoted_vars, kept };
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
