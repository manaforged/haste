//! Packs isomorphic scalar chains that store to consecutive addresses in a single-block loop into vector operations.

mod emit;

use rustc_hash::FxHashMap as HashMap;

use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Lane, Op, Type, Value};
use crate::mem::{Provenance, may_alias};

#[derive(Clone, Copy)]
enum Kind {
    Splat(Value),
    Param(usize),
    Load(Value, i32),
    Binary(Binary, usize, usize),
    Icmp(IntCC, usize, usize),
    Select(usize, usize, usize),
    Convert(Convert, usize),
}

struct Node {
    lanes: Vec<Value>,
    kind: Kind,
    lane: Lane,
}

struct ParamGroup {
    indices: Vec<usize>,
    back: usize,
}

struct Packer<'a> {
    func: &'a Function,
    block: Block,
    back_args: &'a [Value],
    defs: &'a HashMap<Value, usize>,
    nodes: Vec<Node>,
    memo: HashMap<Vec<Value>, usize>,
    owner: HashMap<Value, (usize, usize)>,
    params: Vec<ParamGroup>,
}

fn lane_binary(op: Binary, lane: Lane) -> bool {
    match op {
        Binary::Iadd | Binary::Isub | Binary::Band | Binary::Bor | Binary::Bxor => true,
        Binary::Imul => matches!(lane, Lane::I16 | Lane::I32),
        Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => lane != Lane::I64,
        _ => false,
    }
}

impl Packer<'_> {
    fn op_of(&self, value: Value) -> Option<&Op> {
        self.defs.get(&value).map(|&pos| {
            &self.func.insts[self.func.blocks[self.block.index()].insts[pos].index()].op
        })
    }

    fn add(&mut self, lanes: &[Value], kind: Kind, lane: Lane) -> Option<usize> {
        let id = self.nodes.len();
        for (k, &value) in lanes.iter().enumerate() {
            if self.defs.contains_key(&value)
                && self
                    .owner
                    .insert(value, (id, k))
                    .is_some_and(|prev| prev != (id, k))
            {
                return None;
            }
        }
        self.nodes.push(Node {
            lanes: lanes.to_vec(),
            kind,
            lane,
        });
        self.memo.insert(lanes.to_vec(), id);
        Some(id)
    }

    fn pack_column(&mut self, ops: &[Op], pick: fn(&Op) -> Option<Value>) -> Option<usize> {
        let values: Vec<Value> = ops.iter().map(pick).collect::<Option<_>>()?;
        self.pack(&values)
    }

    fn param_indices(&self, lanes: &[Value]) -> Option<Vec<usize>> {
        let params = &self.func.blocks[self.block.index()].params;
        lanes
            .iter()
            .map(|v| params.iter().position(|p| p == v))
            .collect()
    }

    fn pack_params(&mut self, lanes: &[Value], indices: Vec<usize>, lane: Lane) -> Option<usize> {
        let group = self.params.len();
        let back: Vec<Value> = indices.iter().map(|&i| self.back_args[i]).collect();
        self.params.push(ParamGroup {
            indices,
            back: usize::MAX,
        });
        let id = self.add(lanes, Kind::Param(group), lane)?;
        self.params[group].back = self.pack(&back)?;
        Some(id)
    }

    fn pack_load(base: Value, off: i32, ty: Type, ops: &[Op]) -> Option<Kind> {
        let step = ty.bytes() as i32;
        let consecutive = ops.iter().enumerate().all(
            |(k, op)| matches!(*op, Op::Load(b, o) if b == base && o == off + k as i32 * step),
        );
        consecutive.then_some(Kind::Load(base, off))
    }

    fn pack_binary(&mut self, bin: Binary, lane: Lane, ops: &[Op]) -> Option<Kind> {
        if !lane_binary(bin, lane)
            || !ops
                .iter()
                .all(|op| matches!(op, Op::Binary(b, ..) if *b == bin))
        {
            return None;
        }
        let x = self.pack_column(ops, |op| match op {
            Op::Binary(_, x, _) => Some(*x),
            _ => None,
        })?;
        let y = self.pack_column(ops, |op| match op {
            Op::Binary(_, _, y) => Some(*y),
            _ => None,
        })?;
        Some(Kind::Binary(bin, x, y))
    }

    fn pack_icmp(&mut self, id: usize, cc: IntCC, ops: &[Op]) -> Option<Kind> {
        if !ops
            .iter()
            .all(|op| matches!(op, Op::Icmp(c, ..) if *c == cc))
        {
            return None;
        }
        let x = self.pack_column(ops, |op| match op {
            Op::Icmp(_, x, _) => Some(*x),
            _ => None,
        })?;
        let y = self.pack_column(ops, |op| match op {
            Op::Icmp(_, _, y) => Some(*y),
            _ => None,
        })?;
        self.nodes[id].lane = self.nodes[x].lane;
        Some(Kind::Icmp(cc, x, y))
    }

    fn pack_select(&mut self, lane: Lane, ops: &[Op]) -> Option<Kind> {
        if !ops.iter().all(|op| matches!(op, Op::Select(..))) {
            return None;
        }
        let c = self.pack_column(ops, |op| match op {
            Op::Select(c, ..) => Some(*c),
            _ => None,
        })?;
        if !matches!(self.nodes[c].kind, Kind::Icmp(..))
            || self.nodes[c].lane.bytes() != lane.bytes()
        {
            return None;
        }
        let x = self.pack_column(ops, |op| match op {
            Op::Select(_, x, _) => Some(*x),
            _ => None,
        })?;
        let y = self.pack_column(ops, |op| match op {
            Op::Select(_, _, y) => Some(*y),
            _ => None,
        })?;
        Some(Kind::Select(c, x, y))
    }

    fn pack_convert(&mut self, conv: Convert, lane: Lane, ops: &[Op]) -> Option<Kind> {
        if !ops
            .iter()
            .all(|op| matches!(op, Op::Convert(c, _) if *c == conv))
        {
            return None;
        }
        let x = self.pack_column(ops, |op| match op {
            Op::Convert(_, x) => Some(*x),
            _ => None,
        })?;
        let from = self.nodes[x].lane;
        if conv == Convert::Ireduce && from.bytes() != 2 * lane.bytes() {
            return None;
        }
        Some(Kind::Convert(conv, x))
    }

    fn pack_ops(&mut self, id: usize, ty: Type, lane: Lane, ops: &[Op]) -> Option<Kind> {
        match &ops[0] {
            Op::Load(base, off) => Self::pack_load(*base, *off, ty, ops),
            Op::Binary(bin, ..) => self.pack_binary(*bin, lane, ops),
            Op::Icmp(cc, ..) => self.pack_icmp(id, *cc, ops),
            Op::Select(..) => self.pack_select(lane, ops),
            Op::Convert(conv @ (Convert::Uextend | Convert::Sextend | Convert::Ireduce), _) => {
                self.pack_convert(*conv, lane, ops)
            }
            _ => None,
        }
    }

    fn pack(&mut self, lanes: &[Value]) -> Option<usize> {
        if let Some(&id) = self.memo.get(lanes) {
            return Some(id);
        }
        let ty = self.func.ty(lanes[0]);
        if !ty.is_small_int() || lanes.iter().any(|&v| self.func.ty(v) != ty) {
            return None;
        }
        let lane = Lane::from_type(ty)?;
        if lanes.iter().all(|&v| v == lanes[0]) {
            let inside = self
                .op_of(lanes[0])
                .is_some_and(|op| !matches!(op, Op::Const(_)));
            return if inside {
                None
            } else {
                self.add(lanes, Kind::Splat(lanes[0]), lane)
            };
        }
        if let Some(indices) = self.param_indices(lanes) {
            return self.pack_params(lanes, indices, lane);
        }
        let ops: Vec<Op> = lanes
            .iter()
            .map(|&v| self.op_of(v).cloned())
            .collect::<Option<_>>()?;
        let id = self.add(lanes, Kind::Splat(lanes[0]), lane)?;
        self.nodes[id].kind = self.pack_ops(id, ty, lane, &ops)?;
        Some(id)
    }

    fn nodes_valid(&self, n: usize, count: u8, at: usize) -> bool {
        let late = |v: Value| self.defs.get(&v).is_some_and(|&pos| pos >= at);
        for node in &self.nodes {
            if let Kind::Splat(v) | Kind::Load(v, _) = node.kind
                && late(v)
            {
                return false;
            }
            if let Kind::Load(..) = node.kind {
                let bytes = node.lane.bytes() * n as u32;
                if bytes != node.lane.bytes() * u32::from(count) && chunk(bytes).is_none() {
                    return false;
                }
            }
        }
        true
    }

    fn memory_safe(&self, prov: &Provenance, seed: &Seed) -> bool {
        let data = &self.func.blocks[self.block.index()];
        for (pos, &inst) in data.insts.iter().enumerate() {
            let op = &self.func.insts[inst.index()].op;
            if writes_memory(op) && !seed.stores.contains(&pos) {
                return false;
            }
        }
        let loads: Vec<usize> = self
            .nodes
            .iter()
            .filter(|node| matches!(node.kind, Kind::Load(..)))
            .flat_map(|node| node.lanes.iter().map(|v| self.defs[v]))
            .collect();
        let step = seed.ty.bytes();
        for (k, &spos) in seed.stores.iter().enumerate() {
            let stored = prov.loc(
                seed.base,
                i64::from(seed.off) + i64::from(step) * k as i64,
                step,
            );
            for &lpos in &loads {
                let Op::Load(addr, off) = self.func.insts[data.insts[lpos].index()].op else {
                    continue;
                };
                let Some(res) = self.func.insts[data.insts[lpos].index()].result else {
                    return false;
                };
                let read = prov.loc(addr, i64::from(off), self.func.ty(res).bytes());
                if spos < lpos && may_alias(stored, read, prov) {
                    return false;
                }
            }
        }
        true
    }

    fn count_extracts(&self, seed: &Seed, term: Inst, at: usize) -> Option<(usize, usize)> {
        let data = &self.func.blocks[self.block.index()];
        let tree: Vec<Inst> = self
            .owner
            .keys()
            .map(|v| data.insts[self.defs[v]])
            .collect();
        let dropped: Vec<usize> = self
            .params
            .iter()
            .flat_map(|g| g.indices.iter().copied())
            .collect();
        let watched: HashMap<Value, usize> = self
            .owner
            .keys()
            .copied()
            .chain(
                self.params
                    .iter()
                    .flat_map(|g| g.indices.iter().map(|&i| data.params[i])),
            )
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        let mut seen = vec![(false, false); watched.len()];
        for (b, bd) in self.func.blocks.iter().enumerate() {
            for (pos, &inst) in bd.insts.iter().enumerate() {
                if tree.contains(&inst) || (b == self.block.index() && seed.stores.contains(&pos)) {
                    continue;
                }
                let early = b == self.block.index() && pos < at;
                let mut mark = |v: Value| {
                    if let Some(&i) = watched.get(&v) {
                        seen[i] = (true, seen[i].1 || early);
                    }
                };
                if inst == term {
                    term_operands(
                        &self.func.insts[inst.index()].op,
                        self.block,
                        &dropped,
                        &mut mark,
                    );
                } else {
                    self.func.insts[inst.index()].op.for_operands(&mut mark);
                }
            }
        }
        let mut extracts = 0;
        for (&value, &i) in &watched {
            let (external, early) = seen[i];
            if let Some(&(id, _)) = self.owner.get(&value)
                && external
                && (early || matches!(self.nodes[id].kind, Kind::Icmp(..)))
            {
                return None;
            }
            extracts += usize::from(external);
        }
        Some((watched.len(), extracts))
    }
}

struct Seed {
    stores: Vec<usize>,
    base: Value,
    off: i32,
    ty: Type,
}

fn seeds(func: &Function, block: Block) -> Vec<Seed> {
    let insts = &func.blocks[block.index()].insts;
    let mut stores: Vec<(Value, Type, i32, usize)> = insts
        .iter()
        .enumerate()
        .filter_map(|(pos, &inst)| match func.insts[inst.index()].op {
            Op::Store(value, base, off) if func.ty(value).is_small_int() => {
                Some((base, func.ty(value), off, pos))
            }
            _ => None,
        })
        .collect();
    stores.sort_by_key(|&(base, ty, off, _)| (base.index(), ty.bytes(), off));
    let mut out = Vec::new();
    let mut at = 0;
    while at < stores.len() {
        let (base, ty, off, _) = stores[at];
        let step = ty.bytes() as i32;
        let mut end = at + 1;
        while end < stores.len()
            && stores[end].0 == base
            && stores[end].1 == ty
            && stores[end].2 == off + (end - at) as i32 * step
        {
            end += 1;
        }
        let len = end - at;
        if len >= 2 && len.is_power_of_two() {
            out.push(Seed {
                stores: stores[at..end].iter().map(|s| s.3).collect(),
                base,
                off,
                ty,
            });
        }
        at = end;
    }
    out
}

fn lanes_per_vector(nodes: &[Node], n: usize) -> Option<u8> {
    let widest = nodes.iter().map(|node| node.lane.bytes()).max()?;
    let narrowest = nodes.iter().map(|node| node.lane.bytes()).min()?;
    let count = 16 / widest;
    (count as usize >= n && count * narrowest >= 8).then_some(count as u8)
}

fn writes_memory(op: &Op) -> bool {
    matches!(
        op,
        Op::Store(..)
            | Op::StackStore(..)
            | Op::Call { .. }
            | Op::TryCall { .. }
            | Op::Copy(..)
            | Op::Move(..)
            | Op::AtomicStore(..)
            | Op::AtomicRmw(..)
            | Op::AtomicCas(..)
            | Op::Fence(..)
    )
}

fn chunk(bytes: u32) -> Option<(Type, Lane)> {
    let ty = Type::int_of_bytes(i64::from(bytes))?;
    Some((ty, Lane::from_type(ty)?))
}

struct Plan {
    nodes: Vec<Node>,
    params: Vec<ParamGroup>,
    seed: Seed,
    root: usize,
    count: u8,
    at: usize,
}

fn back_edge_args(op: &Op, block: Block) -> Option<&[Value]> {
    match op {
        Op::Brif(_, t, ta, e, _) if *t == block && *e != block => Some(ta.as_slice()),
        Op::Brif(_, t, _, e, ea) if *e == block && *t != block => Some(ea.as_slice()),
        Op::Jump(b, args) if *b == block => Some(args.as_slice()),
        _ => None,
    }
}

fn plan(func: &Function, prov: &Provenance, block: Block, seed: Seed) -> Option<Plan> {
    let data = &func.blocks[block.index()];
    let term = *data.insts.last()?;
    let back_args = back_edge_args(&func.insts[term.index()].op, block)?;
    let defs: HashMap<Value, usize> = data
        .insts
        .iter()
        .enumerate()
        .filter_map(|(pos, &inst)| func.insts[inst.index()].result.map(|r| (r, pos)))
        .collect();
    let values: Vec<Value> = seed
        .stores
        .iter()
        .map(|&pos| match func.insts[data.insts[pos].index()].op {
            Op::Store(value, ..) => value,
            _ => unreachable!("seeds hold stores"),
        })
        .collect();
    let mut packer = Packer {
        func,
        block,
        back_args,
        defs: &defs,
        nodes: Vec::new(),
        memo: HashMap::default(),
        owner: HashMap::default(),
        params: Vec::new(),
    };
    let root = packer.pack(&values)?;
    let n = seed.stores.len();
    let count = lanes_per_vector(&packer.nodes, n)?;
    let at = *seed.stores.iter().max()?;
    if !packer.nodes_valid(n, count, at) || !packer.memory_safe(prov, &seed) {
        return None;
    }
    let (watched, extracts) = packer.count_extracts(&seed, term, at)?;
    let scalar = n + watched;
    let vector = packer.nodes.len() + 2 + extracts;
    (vector * 3 < scalar * 2).then_some(Plan {
        nodes: packer.nodes,
        params: packer.params,
        seed,
        root,
        count,
        at,
    })
}

fn term_operands(op: &Op, block: Block, dropped: &[usize], f: &mut impl FnMut(Value)) {
    let scan = |args: &[Value], back: bool, f: &mut dyn FnMut(Value)| {
        for (i, &arg) in args.iter().enumerate() {
            if !(back && dropped.contains(&i)) {
                f(arg);
            }
        }
    };
    match op {
        Op::Brif(c, t, ta, e, ea) => {
            f(*c);
            scan(ta, *t == block, f);
            scan(ea, *e == block, f);
        }
        Op::Jump(b, args) => scan(args, *b == block, f),
        _ => op.for_operands(f),
    }
}

/// Packs store chains in single-block loops into vector operations; true when it changed `func`.
pub(crate) fn run(func: &mut Function) -> bool {
    let mut changed = false;
    for b in 0..func.blocks.len() {
        let block = Block(b as u32);
        let Some(&last) = func.blocks[b].insts.last() else {
            continue;
        };
        if !func.insts[last.index()].op.targets().any(|t| t == block) {
            continue;
        }
        for seed in seeds(func, block) {
            let cfg = func.cfg();
            let prov = Provenance::addresses(func, &cfg);
            if let Some(plan) = plan(func, &prov, block, seed) {
                emit::apply(func, block, &plan);
                changed = true;
                break;
            }
        }
    }
    changed
}
