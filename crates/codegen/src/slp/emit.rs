use rustc_hash::FxHashMap as HashMap;

use super::{Kind, Plan, Seed, chunk};
use crate::ir::{Block, Convert, Function, Inst, Lane, Op, Type, Value};
use crate::unroll::{Emit, splat_in};

fn vec_ty(lane: Lane, count: u8) -> Type {
    Type::Vec { lane, count }
}

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

fn emit_node(
    e: &mut Emit<'_>,
    plan: &Plan,
    vparams: &[Value],
    done: &mut Vec<Option<Value>>,
    id: usize,
) -> Value {
    if let Some(v) = done[id] {
        return v;
    }
    let node = &plan.nodes[id];
    let ty = vec_ty(node.lane, plan.count);
    let value = match node.kind {
        Kind::Splat(v) => match e
            .func
            .insts
            .iter()
            .find(|d| d.result == Some(v))
            .map(|d| d.op.clone())
        {
            Some(Op::Const(0)) => e.op(Op::Const(0), ty),
            _ => splat_in(e, ty, v),
        },
        Kind::Gather => gather(e, &node.lanes, ty),
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
            let (vx, vy) = (
                emit_node(e, plan, vparams, done, x),
                emit_node(e, plan, vparams, done, y),
            );
            e.op(Op::Binary(op, vx, vy), ty)
        }
        Kind::Icmp(cc, x, y) => {
            let (vx, vy) = (
                emit_node(e, plan, vparams, done, x),
                emit_node(e, plan, vparams, done, y),
            );
            e.op(Op::Icmp(cc, vx, vy), ty)
        }
        Kind::Select(c, x, y) => {
            let vc = emit_node(e, plan, vparams, done, c);
            let (vx, vy) = (
                emit_node(e, plan, vparams, done, x),
                emit_node(e, plan, vparams, done, y),
            );
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
    plan.nodes
        .iter()
        .find(|n| matches!(n.kind, Kind::Param(g) if g == group))
        .expect("every group has a node")
        .lane
}

/// Rewrites `block` in `func` to emit the packed vector instructions described by `plan`.
pub(super) fn apply(func: &mut Function, block: Block, plan: &Plan) {
    let b = block.index();
    let preds: Vec<Block> = func
        .cfg()
        .preds_of(block)
        .iter()
        .copied()
        .filter(|&p| p != block)
        .collect();
    let mut vparams = Vec::new();
    for (g, group) in plan.params.iter().enumerate() {
        let ty = vec_ty(param_lane(plan, g), plan.count);
        let p = func.value(ty);
        func.blocks[b].params.push(p);
        vparams.push(p);
        for &pred in &preds {
            let mut lanes = Vec::new();
            func.for_edge_args_mut(pred, block, |args| {
                lanes = group.indices.iter().map(|&i| args[i]).collect()
            });
            let term = func.blocks[pred.index()]
                .insts
                .pop()
                .expect("pred has a terminator");
            let mut e = Emit {
                func,
                out: Vec::new(),
            };
            let v = gather(&mut e, &lanes, ty);
            let out = e.out;
            func.blocks[pred.index()].insts.extend(out);
            func.blocks[pred.index()].insts.push(term);
            func.push_edge_arg(pred, block, v);
        }
    }
    let insts = func.blocks[b].insts.clone();
    let tree: Vec<Inst> = plan
        .nodes
        .iter()
        .filter(|n| !matches!(n.kind, Kind::Param(_) | Kind::Splat(_) | Kind::Gather))
        .flat_map(|n| {
            n.lanes.iter().map(|&v| {
                insts
                    .iter()
                    .copied()
                    .find(|&i| func.insts[i.index()].result == Some(v))
                    .expect("lane def")
            })
        })
        .collect();
    let mut e = Emit {
        func,
        out: Vec::new(),
    };
    let mut done = vec![None; plan.nodes.len()];
    let root = emit_node(&mut e, plan, &vparams, &mut done, plan.root);
    let seed = &plan.seed;
    if let Seed::Store {
        stores,
        base,
        off,
        ty,
    } = seed
    {
        let n = stores.len();
        let root_ty = vec_ty(plan.nodes[plan.root].lane, plan.count);
        if n == usize::from(plan.count) {
            e.out.push(e.func.add(Op::Store(root, *base, *off), None));
        } else {
            let bytes = ty.bytes() * n as u32;
            let (_, lane) = chunk(bytes).expect("checked in plan");
            let holder = vec_ty(lane, (root_ty.bytes() / bytes) as u8);
            let cast = e.op(Op::Convert(Convert::Bitcast, root), holder);
            let scalar = e.op(Op::Extract(cast, 0), lane.to_type());
            e.out.push(e.func.add(Op::Store(scalar, *base, *off), None));
        }
    }
    let mut subst: HashMap<Value, Value> = HashMap::default();
    for (id, node) in plan.nodes.iter().enumerate() {
        if matches!(
            node.kind,
            Kind::Splat(_) | Kind::Gather | Kind::Param(_) | Kind::Icmp(..)
        ) {
            continue;
        }
        let v = emit_node(&mut e, plan, &vparams, &mut done, id);
        for (k, &lane) in node.lanes.iter().enumerate() {
            let x = e.op(Op::Extract(v, k as u8), node.lane.to_type());
            subst.insert(lane, x);
        }
    }
    let back: Vec<Value> = plan
        .params
        .iter()
        .map(|g| emit_node(&mut e, plan, &vparams, &mut done, g.back))
        .collect();
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
    let mut dropped: Vec<usize> = plan
        .params
        .iter()
        .flat_map(|g| g.indices.iter().copied())
        .collect();
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
    let removed: Vec<Inst> = tree
        .iter()
        .copied()
        .chain(seed.stores().iter().map(|&pos| insts[pos]))
        .collect();
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
            func.insts[inst.index()]
                .op
                .map_operands(|v| subst.get(&v).copied().unwrap_or(v));
        }
    }
}
