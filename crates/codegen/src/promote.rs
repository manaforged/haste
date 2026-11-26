use rustc_hash::FxHashMap as HashMap;

#[path = "promote/box_fwd.rs"]
mod box_fwd;
#[path = "promote/cold_copy.rs"]
mod cold_copy;
#[path = "promote/join_slots.rs"]
mod join_slots;
#[path = "promote/ssa.rs"]
mod ssa;
#[path = "promote_subword.rs"]
mod subword;
#[path = "promote/vec_slot.rs"]
mod vec_slot;

use crate::ir::{Block, Domain, Function, Inst, InstData, Op, StackSlot, Type, Value};
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
        && ((a.bytes() == 8
            && (a.is_vector() || matches!(a, Type::F64 | Type::I64))
            && (b.is_vector() || matches!(b, Type::F64 | Type::I64)))
            || matches!((a, b), (Type::F32 | Type::I32, Type::F32 | Type::I32))
            || (a.is_vector() && b.is_vector()))
}

struct Var {
    slot: StackSlot,
    offset: i64,
    end: i64,
    mode: VarMode,
}

impl Var {
    fn new(slot: StackSlot, offset: i64, ty: Type) -> Self {
        Var {
            slot,
            offset,
            end: offset + i64::from(ty.bytes()),
            mode: VarMode::Uniform { ty, stored: false },
        }
    }

    fn home(&self) -> Type {
        match self.mode {
            VarMode::Uniform { ty, .. }
            | VarMode::NarrowInt { home: ty, .. }
            | VarMode::Bitcast { home: ty } => ty,
            VarMode::Invalid => Type::I8,
        }
    }

    fn sound(&self) -> bool {
        !matches!(self.mode, VarMode::Invalid)
    }

    fn add(&mut self, ty: Type, is_store: bool) {
        self.end = self.end.max(self.offset + i64::from(ty.bytes()));
        self.mode = match self.mode {
            VarMode::Uniform { ty: u, stored } if ty == u => VarMode::Uniform {
                ty: u,
                stored: stored || is_store,
            },
            VarMode::Uniform { ty: u, stored } if same_bitcast_family(u, ty) => {
                let home = if stored || !is_store { u } else { ty };
                VarMode::Bitcast { home }
            }
            VarMode::Uniform { ty: u, stored } if u.is_small_int() && ty.is_small_int() => {
                match (stored, is_store) {
                    (true, true) => VarMode::Invalid,
                    (true, false) if ty.bytes() > u.bytes() => VarMode::Invalid,
                    (true, false) => VarMode::NarrowInt {
                        home: u,
                        has_store: true,
                    },
                    (false, true) if u.bytes() > ty.bytes() => VarMode::Invalid,
                    (false, true) => VarMode::NarrowInt {
                        home: ty,
                        has_store: true,
                    },
                    (false, false) => VarMode::NarrowInt {
                        home: if ty.bytes() > u.bytes() { ty } else { u },
                        has_store: false,
                    },
                }
            }
            VarMode::NarrowInt { home, has_store } if ty.is_small_int() => {
                if is_store {
                    if (has_store && ty != home) || (!has_store && home.bytes() > ty.bytes()) {
                        VarMode::Invalid
                    } else {
                        VarMode::NarrowInt {
                            home: ty,
                            has_store: true,
                        }
                    }
                } else if has_store && ty.bytes() > home.bytes() {
                    VarMode::Invalid
                } else {
                    let next = if !has_store && ty.bytes() > home.bytes() {
                        ty
                    } else {
                        home
                    };
                    VarMode::NarrowInt {
                        home: next,
                        has_store,
                    }
                }
            }
            VarMode::Bitcast { home } if same_bitcast_family(home, ty) => VarMode::Bitcast { home },
            _ => VarMode::Invalid,
        };
    }
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
        scan.open_slots =
            subword::copy_open_slots(func, &scan.addrs, &scan.consts, &prov.param_roots);
        for u in &func.unwind_drops {
            scan.open_slots[u.local.index()] = false;
            scan.open_slots[u.flag.index()] = false;
        }
        for (index, block) in func.blocks.iter().enumerate() {
            for inst in &block.insts {
                let data = &func.insts[inst.index()];
                if let Some(touch) = scan.touch(data) {
                    scan.access(
                        func,
                        (Block(index as u32), *inst),
                        (touch.slot, touch.offset),
                        touch.kind,
                    );
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
            Op::StackLoad(slot, offset) => Some(Touch {
                slot: *slot,
                offset: Some(i64::from(*offset)),
                kind: Kind::Load(data.result?),
            }),
            Op::StackStore(value, slot, offset) => Some(Touch {
                slot: *slot,
                offset: Some(i64::from(*offset)),
                kind: Kind::Store(*value),
            }),
            Op::Load(addr, offset) => self.through(*addr, *offset, Kind::Load(data.result?)),
            Op::Store(value, addr, offset) => self.through(*addr, *offset, Kind::Store(*value)),
            _ => None,
        }
    }

    fn through(&self, addr: Value, offset: i32, kind: Kind) -> Option<Touch> {
        let (slot, base) = self.addrs[addr.index()]?;
        Some(Touch {
            slot,
            offset: base.checked_add(i64::from(offset)),
            kind,
        })
    }

    fn access(
        &mut self,
        func: &Function,
        at: (Block, Inst),
        place: (StackSlot, Option<i64>),
        kind: Kind,
    ) {
        let (slot, offset) = place;
        let ty = func.ty(kind.value());
        let size = i64::from(func.slots[slot.index()].size);
        let Some(offset) =
            offset.filter(|offset| *offset >= 0 && offset + i64::from(ty.bytes()) <= size)
        else {
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
        self.accesses.push(Access {
            block: at.0,
            inst: at.1,
            var,
            kind,
            ty,
        });
    }

    fn end(&self, addr: Value) -> End {
        self.addrs[addr.index()].map(|(slot, offset)| (slot.index(), offset))
    }

    fn layouts(&self, func: &Function) -> Layouts {
        let accesses = self.accesses.iter().map(|access| {
            (
                self.vars[access.var].slot.index(),
                self.vars[access.var].offset,
                access.ty,
            )
        });
        let mut layouts = Layouts::new(func.slots.len(), accesses);
        let moves: Vec<(End, End, u32)> = func
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| match func.insts[inst.index()].op {
                Op::Move(to, from, bytes) => Some((self.end(to), self.end(from), bytes)),
                Op::Copy(to, from, len) => self
                    .copy_bytes(to, from, len)
                    .map(|bytes| (self.end(to), self.end(from), bytes)),
                _ => None,
            })
            .collect();
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

    fn open(&self, addr: Value) -> bool {
        self.addrs[addr.index()].is_some_and(|(slot, _)| self.open_slots[slot.index()])
    }

    fn var_promotable(&self, slot: usize, var: usize) -> bool {
        if self.escaped[slot] || !self.vars[var].sound() {
            return false;
        }
        let v = &self.vars[var];
        self.slot_vars[slot].iter().all(|&other| {
            other == var || self.vars[other].end <= v.offset || v.end <= self.vars[other].offset
        })
    }

    fn any_promotable(&self) -> bool {
        (0..self.slot_vars.len()).any(|slot| {
            !self.escaped[slot]
                && self.slot_vars[slot]
                    .iter()
                    .any(|&var| self.var_promotable(slot, var))
        })
    }
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
    let word_limit = if both_open {
        max_p
    } else {
        crate::copy::EXPAND_WORDS
    };
    let (found, limit) = match layouts.and_then(|l| l.pieces(scan.end(to), scan.end(from), bytes)) {
        None if moves != Moves::Guided => (Some(crate::copy::vec_words(bytes)), word_limit),
        guided => (guided, max_p),
    };
    found.filter(|p| p.len() <= limit)
}

fn has_open_move(func: &Function, scan: &Scan) -> bool {
    crate::schedule::has_live_op(func, |op| match *op {
        Op::Move(to, from, _) => scan.open(to) || scan.open(from),
        Op::Copy(to, from, len) => {
            scan.copy_bytes(to, from, len).is_some() && (scan.open(to) || scan.open(from))
        }
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

pub(crate) fn step(func: &mut Function, moves: Moves, effort: crate::Effort) -> Domain {
    let Some(entry) = func.entry.filter(|_| !func.slots.is_empty()) else {
        return Domain::NONE;
    };
    let opt = effort == crate::Effort::Full;
    let params = opt && crate::copy::forward_param_slots(func);
    let returned = opt && crate::copy::forward_sret_slot(func);
    let join_changed = pre_split(func, moves) | (opt && cold_copy::run(func));
    let mut scan = Scan::new(func);
    let (mut splits, mut copy_splits, mut dropped_zero) = (Vec::new(), Vec::new(), false);
    if has_open_move(func, &scan) {
        let layouts = (moves != Moves::Words).then(|| scan.layouts(func));
        let pieces =
            |to, from, bytes| open_move_pieces(&scan, layouts.as_ref(), moves, (to, from, bytes));
        splits = crate::copy::expand(func, pieces);
        (copy_splits, dropped_zero) = subword::expand_copies(func, |to, from, len| {
            pieces(to, from, scan.copy_bytes(to, from, len)?)
        });
        scan = Scan::new(func);
    }
    let split_changed = run_slot_splits(func, &mut scan, moves);
    let changed = (scan.any_promotable() && ssa::promote(func, entry, scan))
        || join_changed
        || split_changed
        || dropped_zero;
    crate::copy::rejoin(func, &splits, opt);
    let all_copies_rejoined = subword::rejoin_copies(func, &copy_splits);
    let cfg = if returned { Domain::CFG } else { Domain::NONE };
    if changed || returned || params || !all_copies_rejoined {
        Domain::MEM | Domain::VALUES | cfg
    } else {
        Domain::NONE
    }
}
