//! Packs isomorphic scalar chains that store to consecutive addresses in a single-block loop into vector operations.

use rustc_hash::FxHashMap as HashMap;

use crate::ir::{Binary, Block, Convert, Function, Inst, Lane, Op, Type, Value};
use crate::mem::{Provenance, may_alias};
use crate::unroll::{Emit, splat_in};

/// One packed group of lanes and how its vector is formed.
#[derive(Clone, Copy)]
enum Kind {
    Splat(Value),
    Param(usize),
    Load(Value, i32),
    Binary(Binary, usize, usize),
    Icmp(crate::ir::IntCC, usize, usize),
    Select(usize, usize, usize),
    Convert(Convert, usize),
}

struct Node {
    lanes: Vec<Value>,
    kind: Kind,
    lane: Lane,
}

/// A group of loop parameters packed into one vector parameter, with the node its back-edge arguments form.
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
        self.defs.get(&value).map(|&pos| &self.func.insts[self.func.blocks[self.block.index()].insts[pos].index()].op)
    }

    fn add(&mut self, lanes: &[Value], kind: Kind, lane: Lane) -> Option<usize> {
        let id = self.nodes.len();
        for (k, &value) in lanes.iter().enumerate() {
            if self.defs.contains_key(&value) && self.owner.insert(value, (id, k)).is_some_and(|prev| prev != (id, k)) {
                return None;
            }
        }
        self.nodes.push(Node { lanes: lanes.to_vec(), kind, lane });
        self.memo.insert(lanes.to_vec(), id);
        Some(id)
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
            let inside = self.op_of(lanes[0]).is_some_and(|op| !matches!(op, Op::Const(_)));
            return if inside { None } else { self.add(lanes, Kind::Splat(lanes[0]), lane) };
        }
        let params = &self.func.blocks[self.block.index()].params;
        if let Some(indices) = lanes.iter().map(|v| params.iter().position(|p| p == v)).collect::<Option<Vec<_>>>() {
            let group = self.params.len();
            self.params.push(ParamGroup { indices: indices.clone(), back: usize::MAX });
            let id = self.add(lanes, Kind::Param(group), lane)?;
            let back: Vec<Value> = indices.iter().map(|&i| self.back_args[i]).collect();
            self.params[group].back = self.pack(&back)?;
            return Some(id);
        }
        let ops: Vec<Op> = lanes.iter().map(|&v| self.op_of(v).cloned()).collect::<Option<_>>()?;
        let id = self.add(lanes, Kind::Splat(lanes[0]), lane)?;
        let column = |pick: fn(&Op) -> Option<Value>| ops.iter().map(pick).collect::<Option<Vec<Value>>>();
        let kind = match ops[0] {
            Op::Load(base, off) => {
                let step = ty.bytes() as i32;
                let consecutive = ops.iter().enumerate().all(|(k, op)| matches!(*op, Op::Load(b, o) if b == base && o == off + k as i32 * step));
                consecutive.then_some(Kind::Load(base, off))?
            }
            Op::Binary(bin, ..) if lane_binary(bin, lane) && ops.iter().all(|op| matches!(op, Op::Binary(b, ..) if *b == bin)) => {
                let x = self.pack(&column(|op| if let Op::Binary(_, x, _) = *op { Some(x) } else { None })?)?;
                let y = self.pack(&column(|op| if let Op::Binary(_, _, y) = *op { Some(y) } else { None })?)?;
                Kind::Binary(bin, x, y)
            }
            Op::Icmp(cc, ..) if ops.iter().all(|op| matches!(op, Op::Icmp(c, ..) if *c == cc)) => {
                let x = self.pack(&column(|op| if let Op::Icmp(_, x, _) = *op { Some(x) } else { None })?)?;
                let y = self.pack(&column(|op| if let Op::Icmp(_, _, y) = *op { Some(y) } else { None })?)?;
                self.nodes[id].lane = self.nodes[x].lane;
                Kind::Icmp(cc, x, y)
            }
            Op::Select(..) if ops.iter().all(|op| matches!(op, Op::Select(..))) => {
                let c = self.pack(&column(|op| if let Op::Select(c, ..) = *op { Some(c) } else { None })?)?;
                if !matches!(self.nodes[c].kind, Kind::Icmp(..)) || self.nodes[c].lane.bytes() != lane.bytes() {
                    return None;
                }
                let x = self.pack(&column(|op| if let Op::Select(_, x, _) = *op { Some(x) } else { None })?)?;
                let y = self.pack(&column(|op| if let Op::Select(_, _, y) = *op { Some(y) } else { None })?)?;
                Kind::Select(c, x, y)
            }
            Op::Convert(conv @ (Convert::Uextend | Convert::Sextend | Convert::Ireduce), _) if ops.iter().all(|op| matches!(op, Op::Convert(c, _) if *c == conv)) => {
                let x = self.pack(&column(|op| if let Op::Convert(_, x) = *op { Some(x) } else { None })?)?;
                let from = self.nodes[x].lane;
                if conv == Convert::Ireduce && from.bytes() != 2 * lane.bytes() {
                    return None;
                }
                Kind::Convert(conv, x)
            }
            _ => return None,
        };
        self.nodes[id].kind = kind;
        Some(id)
    }
}

/// A run of stores of one scalar type to consecutive offsets from one base.
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
            Op::Store(value, base, off) if func.ty(value).is_small_int() => Some((base, func.ty(value), off, pos)),
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
        while end < stores.len() && stores[end].0 == base && stores[end].1 == ty && stores[end].2 == off + (end - at) as i32 * step {
            end += 1;
        }
        let len = end - at;
        if len >= 2 && len.is_power_of_two() {
            out.push(Seed { stores: stores[at..end].iter().map(|s| s.3).collect(), base, off, ty });
        }
        at = end;
    }
    out
}

/// The lanes per vector for a packed tree, and whether every node has a valid vector type with it.
fn lanes_per_vector(nodes: &[Node], n: usize) -> Option<u8> {
    let widest = nodes.iter().map(|node| node.lane.bytes()).max()?;
    let narrowest = nodes.iter().map(|node| node.lane.bytes()).min()?;
    let count = 16 / widest;
    (count as usize >= n && count * narrowest >= 8).then_some(count as u8)
}

fn writes_memory(op: &Op) -> bool {
    matches!(op, Op::Store(..) | Op::StackStore(..) | Op::Call { .. } | Op::TryCall { .. } | Op::Copy(..) | Op::Move(..) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence(..))
}

/// The scalar integer type `bytes` wide and its lane, for partial vector loads and stores.
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

fn plan(func: &Function, prov: &Provenance, block: Block, seed: Seed) -> Option<Plan> {
    let data = &func.blocks[block.index()];
    let term = *data.insts.last()?;
    let back_args = match &func.insts[term.index()].op {
        Op::Brif(_, t, ta, e, _) if *t == block && *e != block => ta.as_slice(),
        Op::Brif(_, t, _, e, ea) if *e == block && *t != block => ea.as_slice(),
        Op::Jump(b, args) if *b == block => args.as_slice(),
        _ => return None,
    };
    let defs: HashMap<Value, usize> = data.insts.iter().enumerate().filter_map(|(pos, &inst)| func.insts[inst.index()].result.map(|r| (r, pos))).collect();
    let values: Vec<Value> = seed.stores.iter().map(|&pos| match func.insts[data.insts[pos].index()].op {
        Op::Store(value, ..) => value,
        _ => unreachable!("seeds hold stores"),
    }).collect();
    let mut packer = Packer { func, block, back_args, defs: &defs, nodes: Vec::new(), memo: HashMap::default(), owner: HashMap::default(), params: Vec::new() };
    let root = packer.pack(&values)?;
    let Packer { nodes, params, owner, .. } = packer;
    let n = seed.stores.len();
    let count = lanes_per_vector(&nodes, n)?;
    let at = *seed.stores.iter().max()?;
    let loads: Vec<usize> = nodes.iter().filter(|node| matches!(node.kind, Kind::Load(..))).flat_map(|node| node.lanes.iter().map(|v| defs[v])).collect();
    let late = |v: Value| defs.get(&v).is_some_and(|&pos| pos >= at);
    for node in &nodes {
        if let Kind::Splat(v) | Kind::Load(v, _) = node.kind
            && late(v)
        {
            return None;
        }
        if let Kind::Load(..) = node.kind {
            let bytes = node.lane.bytes() * n as u32;
            if bytes != node.lane.bytes() * u32::from(count) && chunk(bytes).is_none() {
                return None;
            }
        }
    }
    for (pos, &inst) in data.insts.iter().enumerate() {
        let op = &func.insts[inst.index()].op;
        if writes_memory(op) && !seed.stores.contains(&pos) {
            return None;
        }
    }
    let step = seed.ty.bytes();
    for (k, &spos) in seed.stores.iter().enumerate() {
        let stored = prov.loc(seed.base, i64::from(seed.off) + i64::from(step) * k as i64, step);
        for &lpos in &loads {
            let Op::Load(addr, off) = func.insts[data.insts[lpos].index()].op else { continue };
            let read = prov.loc(addr, i64::from(off), func.ty(func.insts[data.insts[lpos].index()].result?).bytes());
            if spos < lpos && may_alias(stored, read, prov) {
                return None;
            }
        }
    }
    let tree: Vec<Inst> = owner.keys().map(|v| data.insts[defs[v]]).collect();
    let dropped: Vec<usize> = params.iter().flat_map(|g| g.indices.iter().copied()).collect();
    let watched: HashMap<Value, usize> = owner.keys().copied().chain(params.iter().flat_map(|g| g.indices.iter().map(|&i| data.params[i]))).enumerate().map(|(i, v)| (v, i)).collect();
    let mut seen = vec![(false, false); watched.len()];
    for (b, bd) in func.blocks.iter().enumerate() {
        for (pos, &inst) in bd.insts.iter().enumerate() {
            if tree.contains(&inst) || (b == block.index() && seed.stores.contains(&pos)) {
                continue;
            }
            let early = b == block.index() && pos < at;
            let mut mark = |v: Value| {
                if let Some(&i) = watched.get(&v) {
                    seen[i] = (true, seen[i].1 || early);
                }
            };
            if inst == term {
                term_operands(&func.insts[inst.index()].op, block, &dropped, &mut mark);
            } else {
                func.insts[inst.index()].op.for_operands(&mut mark);
            }
        }
    }
    let mut extracts = 0;
    for (&value, &i) in &watched {
        let (external, early) = seen[i];
        if let Some(&(id, _)) = owner.get(&value)
            && external
            && (early || matches!(nodes[id].kind, Kind::Icmp(..)))
        {
            return None;
        }
        extracts += usize::from(external);
    }
    let scalar = n + watched.len();
    let vector = nodes.len() + 2 + extracts;
    (vector * 3 < scalar * 2).then_some(Plan { nodes, params, seed, root, count, at })
}

/// Calls `f` with each value the terminator `op` reads, leaving out the back-edge argument slots that packed parameters drop.
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

fn vec_ty(lane: Lane, count: u8) -> Type {
    Type::Vec { lane, count }
}

/// Builds the vector of `lanes` from scalars available at the end of a predecessor.
fn gather(e: &mut Emit<'_>, lanes: &[Value], ty: Type) -> Value {
    if lanes.iter().all(|&v| v == lanes[0]) {
        return splat_in(e, ty, lanes[0]);
    }
    let mut acc = e.op(Op::Const(0), ty);
    for (k, &v) in lanes.iter().enumerate() {
        acc = e.op(Op::Insert(acc, v, k as u8), ty);
    }
    acc
}

fn emit_node(e: &mut Emit<'_>, plan: &Plan, vparams: &[Value], done: &mut Vec<Option<Value>>, id: usize) -> Value {
    if let Some(v) = done[id] {
        return v;
    }
    let node = &plan.nodes[id];
    let ty = vec_ty(node.lane, plan.count);
    let value = match node.kind {
        Kind::Splat(v) => match e.func.insts.iter().find(|d| d.result == Some(v)).map(|d| d.op.clone()) {
            Some(Op::Const(0)) => e.op(Op::Const(0), ty),
            _ => splat_in(e, ty, v),
        },
        Kind::Param(group) => vparams[group],
        Kind::Load(base, off) => {
            let bytes = node.lane.bytes() * node.lanes.len() as u32;
            if node.lanes.len() == usize::from(plan.count) {
                e.op(Op::Load(base, off), ty)
            } else {
                let (scalar, lane) = chunk(bytes).expect("checked in plan");
                let loaded = e.op(Op::Load(base, off), scalar);
                let holder = vec_ty(lane, (ty.bytes() / bytes) as u8);
                let zero = e.op(Op::Const(0), holder);
                let filled = e.op(Op::Insert(zero, loaded, 0), holder);
                e.op(Op::Convert(Convert::Bitcast, filled), ty)
            }
        }
        Kind::Binary(op, x, y) => {
            let (vx, vy) = (emit_node(e, plan, vparams, done, x), emit_node(e, plan, vparams, done, y));
            e.op(Op::Binary(op, vx, vy), ty)
        }
        Kind::Icmp(cc, x, y) => {
            let (vx, vy) = (emit_node(e, plan, vparams, done, x), emit_node(e, plan, vparams, done, y));
            e.op(Op::Icmp(cc, vx, vy), ty)
        }
        Kind::Select(c, x, y) => {
            let vc = emit_node(e, plan, vparams, done, c);
            let (vx, vy) = (emit_node(e, plan, vparams, done, x), emit_node(e, plan, vparams, done, y));
            e.op(Op::Select(vc, vx, vy), ty)
        }
        Kind::Convert(op, x) => {
            let vx = emit_node(e, plan, vparams, done, x);
            e.op(Op::Convert(op, vx), ty)
        }
    };
    done[id] = Some(value);
    value
}

fn param_lane(plan: &Plan, group: usize) -> Lane {
    plan.nodes.iter().find(|n| matches!(n.kind, Kind::Param(g) if g == group)).expect("every group has a node").lane
}

fn apply(func: &mut Function, block: Block, plan: &Plan) {
    let b = block.index();
    let preds: Vec<Block> = func.cfg().preds_of(block).iter().copied().filter(|&p| p != block).collect();
    let mut vparams = Vec::new();
    for (g, group) in plan.params.iter().enumerate() {
        let ty = vec_ty(param_lane(plan, g), plan.count);
        let p = func.value(ty);
        func.blocks[b].params.push(p);
        vparams.push(p);
        for &pred in &preds {
            let mut lanes = Vec::new();
            func.for_edge_args_mut(pred, block, |args| lanes = group.indices.iter().map(|&i| args[i]).collect());
            let term = func.blocks[pred.index()].insts.pop().expect("pred has a terminator");
            let mut e = Emit { func, out: Vec::new() };
            let v = gather(&mut e, &lanes, ty);
            let out = e.out;
            func.blocks[pred.index()].insts.extend(out);
            func.blocks[pred.index()].insts.push(term);
            func.push_edge_arg(pred, block, v);
        }
    }
    let insts = func.blocks[b].insts.clone();
    let tree: Vec<Inst> = plan.nodes.iter().filter(|n| !matches!(n.kind, Kind::Param(_) | Kind::Splat(_))).flat_map(|n| n.lanes.iter().map(|&v| insts.iter().copied().find(|&i| func.insts[i.index()].result == Some(v)).expect("lane def"))).collect();
    let mut e = Emit { func, out: Vec::new() };
    let mut done = vec![None; plan.nodes.len()];
    let root = emit_node(&mut e, plan, &vparams, &mut done, plan.root);
    let seed = &plan.seed;
    let n = seed.stores.len();
    let root_ty = vec_ty(plan.nodes[plan.root].lane, plan.count);
    if n == usize::from(plan.count) {
        e.out.push(e.func.add(Op::Store(root, seed.base, seed.off), None));
    } else {
        let bytes = seed.ty.bytes() * n as u32;
        let (_, lane) = chunk(bytes).expect("checked in plan");
        let holder = vec_ty(lane, (root_ty.bytes() / bytes) as u8);
        let cast = e.op(Op::Convert(Convert::Bitcast, root), holder);
        let scalar = e.op(Op::Extract(cast, 0), lane.to_type());
        e.out.push(e.func.add(Op::Store(scalar, seed.base, seed.off), None));
    }
    let mut subst: HashMap<Value, Value> = HashMap::default();
    for (id, node) in plan.nodes.iter().enumerate() {
        if matches!(node.kind, Kind::Splat(_) | Kind::Param(_) | Kind::Icmp(..)) {
            continue;
        }
        let v = emit_node(&mut e, plan, &vparams, &mut done, id);
        for (k, &lane) in node.lanes.iter().enumerate() {
            let x = e.op(Op::Extract(v, k as u8), node.lane.to_type());
            subst.insert(lane, x);
        }
    }
    let back: Vec<Value> = plan.params.iter().map(|g| emit_node(&mut e, plan, &vparams, &mut done, g.back)).collect();
    let body = e.out;
    let mut head = Vec::new();
    for (g, group) in plan.params.iter().enumerate() {
        let lane = param_lane(plan, g);
        for (k, &i) in group.indices.iter().enumerate() {
            let p = func.blocks[b].params[i];
            let x = func.value(lane.to_type());
            head.push(func.add(Op::Extract(vparams[g], k as u8), Some(x)));
            subst.insert(p, x);
        }
    }
    let mut dropped: Vec<usize> = plan.params.iter().flat_map(|g| g.indices.iter().copied()).collect();
    dropped.sort_unstable_by(|a, c| c.cmp(a));
    func.for_edge_args_mut(block, block, |args| {
        for &i in &dropped {
            args.remove(i);
        }
        args.extend(back.iter().copied());
    });
    for &pred in &preds {
        func.for_edge_args_mut(pred, block, |args| {
            for &i in &dropped {
                args.remove(i);
            }
        });
    }
    for &i in &dropped {
        func.blocks[b].params.remove(i);
    }
    let removed: Vec<Inst> = tree.iter().copied().chain(seed.stores.iter().map(|&pos| insts[pos])).collect();
    let mut rebuilt = head;
    for (pos, &inst) in insts.iter().enumerate() {
        if pos == plan.at {
            rebuilt.extend(body.iter().copied());
        }
        if !removed.contains(&inst) {
            rebuilt.push(inst);
        }
    }
    func.blocks[b].insts = rebuilt;
    for bd in 0..func.blocks.len() {
        for idx in 0..func.blocks[bd].insts.len() {
            let inst = func.blocks[bd].insts[idx];
            if body.contains(&inst) {
                continue;
            }
            func.insts[inst.index()].op.map_operands(|v| subst.get(&v).copied().unwrap_or(v));
        }
    }
}

/// Packs store chains in single-block loops into vector operations; true when it changed `func`.
pub(crate) fn run(func: &mut Function) -> bool {
    let mut changed = false;
    for b in 0..func.blocks.len() {
        let block = Block(b as u32);
        let Some(&last) = func.blocks[b].insts.last() else { continue };
        if !func.insts[last.index()].op.targets().any(|t| t == block) {
            continue;
        }
        for seed in seeds(func, block) {
            let cfg = func.cfg();
            let prov = Provenance::addresses(func, &cfg);
            if let Some(plan) = plan(func, &prov, block, seed) {
                apply(func, block, &plan);
                changed = true;
                break;
            }
        }
    }
    changed
}
