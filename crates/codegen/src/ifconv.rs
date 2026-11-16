use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Domain, Function, Inst, Op, Type, Value};
use crate::mem::{MemBase, MemRoot, Provenance};

const MAX_ARM_OPS: usize = 6;
const MAX_ARM_INSTS: usize = 24;
const MAX_SELECTS: usize = 4;
const MAX_MID_BODY: usize = 3;

fn scalar_select(ty: Type) -> bool {
    matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::F32 | Type::F64)
}

struct ArmMatch {
    join: Block,
    t_args: Vec<Value>,
    e_args: Vec<Value>,
    arms: Vec<(Block, Vec<Value>)>,
}

fn remap_val(params: &[Value], args: &[Value], value: Value) -> Value {
    params.iter().position(|&p| p == value).map_or(value, |idx| args[idx])
}

fn pure_arm_ops(func: &Function, body: &[Inst]) -> bool {
    let mut alu = 0usize;
    let mut selects = 0usize;
    for inst in body {
        match func.insts[inst.index()].op {
            Op::Const(_) => {}
            Op::Select(..) => selects += 1,
            _ => alu += 1,
        }
    }
    alu <= MAX_ARM_OPS && alu + selects <= MAX_ARM_OPS + MAX_SELECTS
}

fn safe_arm_op(func: &Function, prov: &Provenance, params: &[Value], incoming: &[Value], inst: Inst) -> bool {
    let op = &func.insts[inst.index()].op;
    if op.is_speculable() {
        return true;
    }
    let Some(bytes) = func.insts[inst.index()].result.map(|r| func.ty(r).bytes()) else { return false };
    match *op {
        Op::StackLoad(slot, off) => {
            off >= 0 && (off as u64).checked_add(u64::from(bytes)).is_some_and(|e| e <= u64::from(func.slots[slot.index()].size))
        }
        Op::Load(addr, off) => {
            let base_addr = remap_val(params, incoming, addr);
            if off >= 0
                && (off as u64)
                    .checked_add(u64::from(bytes))
                    .is_some_and(|e| e <= u64::from(prov.ro_bytes.get(base_addr.index()).copied().unwrap_or(0)))
            {
                return true;
            }
            let loc = prov.loc(base_addr, i64::from(off), bytes);
            match (loc.root, loc.base) {
                (_, MemBase::Slot(slot)) | (MemRoot::Slot(slot), _) => {
                    loc.offset >= 0 && (loc.offset as u64).checked_add(u64::from(bytes)).is_some_and(|e| e <= u64::from(func.slots[slot.index()].size))
                }
                (MemRoot::NoAliasParam(p) | MemRoot::ReadonlyParam(p), MemBase::Addr(base_val)) => {
                    let entry_match = func.entry.and_then(|e| func.blocks[e.index()].params.get(usize::from(p)).copied()) == Some(base_val);
                    let deref_bytes = func.signature.params.get(usize::from(p)).map_or(0, |a| a.deref_bytes);
                    entry_match && loc.offset >= 0 && (loc.offset as u64).checked_add(u64::from(bytes)).is_some_and(|e| e <= u64::from(deref_bytes))
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn pure_arm(func: &Function, prov: &Provenance, block: Block, join: Block, incoming: &[Value]) -> Option<Vec<Value>> {
    let data = &func.blocks[block.index()];
    if data.params.len() != incoming.len() {
        return None;
    }
    let (&last, body) = data.insts.split_last()?;
    if body.len() > MAX_ARM_INSTS {
        return None;
    }
    let Op::Jump(target, args) = &func.insts[last.index()].op else {
        return None;
    };
    if *target != join
        || !pure_arm_ops(func, body)
        || !body.iter().all(|&inst| safe_arm_op(func, prov, &data.params, incoming, inst))
    {
        return None;
    }
    Some(args.iter().map(|&v| remap_val(&data.params, incoming, v)).collect())
}

fn match_diamond(
    func: &Function,
    cfg: &Cfg,
    prov: &Provenance,
    head: Block,
    (then_b, then_args): (Block, &[Value]),
    (else_b, else_args): (Block, &[Value]),
) -> Option<ArmMatch> {
    if cfg.pred_count(then_b) != 1 || cfg.pred_count(else_b) != 1 {
        return None;
    }
    let Some(Op::Jump(join_b, _)) = func.last(then_b).cloned() else { return None };
    if join_b == head || join_b == then_b || join_b == else_b || func.entry == Some(join_b) || cfg.pred_count(join_b) < 2 {
        return None;
    }
    let t_args = pure_arm(func, prov, then_b, join_b, then_args)?;
    let e_args = pure_arm(func, prov, else_b, join_b, else_args)?;
    Some(ArmMatch {
        join: join_b,
        t_args,
        e_args,
        arms: vec![(then_b, then_args.to_vec()), (else_b, else_args.to_vec())],
    })
}

fn match_triangle(func: &Function, cfg: &Cfg, prov: &Provenance, arm: Block, arm_args: &[Value], join: Block) -> Option<Vec<Value>> {
    (cfg.pred_count(arm) == 1 && cfg.pred_count(join) >= 2)
        .then(|| pure_arm(func, prov, arm, join, arm_args))
        .flatten()
}

fn match_arms(func: &Function, cfg: &Cfg, prov: &Provenance, head: Block) -> Option<(Value, ArmMatch)> {
    let &brif_inst = func.blocks[head.index()].insts.last()?;
    let Op::Brif(cond, then_b, then_args, else_b, else_args) = func.insts[brif_inst.index()].op.clone() else {
        return None;
    };
    if !func.ty(cond).is_small_int() {
        return None;
    }
    if then_b == head || else_b == head || func.entry == Some(then_b) || func.entry == Some(else_b) {
        return None;
    }
    let matched = if then_b == else_b {
        ArmMatch { join: then_b, t_args: then_args, e_args: else_args, arms: Vec::new() }
    } else if let Some(m) = match_diamond(func, cfg, prov, head, (then_b, &then_args), (else_b, &else_args)) {
        m
    } else if let Some(t_args) = match_triangle(func, cfg, prov, then_b, &then_args, else_b) {
        ArmMatch { join: else_b, t_args, e_args: else_args, arms: vec![(then_b, then_args)] }
    } else if let Some(e_args) = match_triangle(func, cfg, prov, else_b, &else_args, then_b) {
        ArmMatch { join: then_b, t_args: then_args, e_args, arms: vec![(else_b, else_args)] }
    } else {
        return None;
    };
    Some((cond, matched))
}

fn emit_select(func: &mut Function, head: Block, ty: Type, cond: Value, t_val: Value, e_val: Value) -> Value {
    if t_val == e_val {
        return t_val;
    }
    let val = func.value(ty);
    let inst = func.add(Op::Select(cond, t_val, e_val), Some(val));
    func.blocks[head.index()].insts.push(inst);
    val
}

fn extract_const_leaf(
    func: &Function,
    cfg: &Cfg,
    is_const: &[bool],
    leaf: Block,
    in_args: &[Value],
    join: Block,
) -> Option<(Option<Block>, Value)> {
    if leaf == join {
        return (in_args.len() == 1 && is_const[in_args[0].index()]).then_some((None, in_args[0]));
    }
    if cfg.pred_count(leaf) != 1 || func.entry == Some(leaf) {
        return None;
    }
    let data = &func.blocks[leaf.index()];
    if data.params.len() != in_args.len() {
        return None;
    }
    let (&last, body) = data.insts.split_last()?;
    if body.len() > 2 || !body.iter().all(|&i| matches!(func.insts[i.index()].op, Op::Const(_))) {
        return None;
    }
    let Op::Jump(target, args) = &func.insts[last.index()].op else {
        return None;
    };
    if *target != join || args.len() != 1 {
        return None;
    }
    let val = remap_val(&data.params, in_args, args[0]);
    is_const[val.index()].then_some((Some(leaf), val))
}

fn retire_terminator(func: &mut Function, block: Block) {
    if let Some(term) = func.blocks[block.index()].insts.pop() {
        func.insts[term.index()].op = Op::Trap;
    }
}

fn hoist_block_body(
    func: &mut Function,
    head: Block,
    block: Block,
    in_args: &[Value],
    cleared: &mut [bool],
) -> (Vec<Value>, Vec<Value>) {
    let params = std::mem::take(&mut func.blocks[block.index()].params);
    retire_terminator(func, block);
    let moved = std::mem::take(&mut func.blocks[block.index()].insts);
    if !params.is_empty() {
        for &inst in &moved {
            func.insts[inst.index()].op.map_operands(|v| remap_val(&params, in_args, v));
        }
    }
    func.blocks[head.index()].insts.extend(moved);
    cleared[block.index()] = true;
    (params, in_args.to_vec())
}

type BranchParts = (Value, Block, Vec<Value>, Block, Vec<Value>);

fn branch_block(func: &Function, cfg: &Cfg, (block, in_args): (Block, &[Value]), max_body: usize, keep: impl Fn(&Op) -> bool) -> Option<BranchParts> {
    let data = &func.blocks[block.index()];
    if cfg.pred_count(block) != 1 || func.entry == Some(block) || data.params.len() != in_args.len() {
        return None;
    }
    let (&last, body) = data.insts.split_last()?;
    if body.len() > max_body || !body.iter().all(|&inst| keep(&func.insts[inst.index()].op)) {
        return None;
    }
    let Op::Brif(cond, then_b, then_args, else_b, else_args) = func.insts[last.index()].op.clone() else { return None };
    Some((cond, then_b, then_args, else_b, else_args))
}

fn compare_only(op: &Op) -> bool {
    matches!(op, Op::Const(_) | Op::Icmp(..) | Op::Fcmp(..))
}

fn try_nested_classifier(
    func: &mut Function,
    cfg: &Cfg,
    is_const: &[bool],
    head: Block,
    cleared: &mut [bool],
) -> bool {
    let Some(&brif_inst) = func.blocks[head.index()].insts.last() else { return false };
    let Op::Brif(c1, then_b, then_args, else_b, else_args) = func.insts[brif_inst.index()].op.clone() else {
        return false;
    };
    if !func.ty(c1).is_small_int() || then_b == else_b || then_b == head || else_b == head {
        return false;
    }
    for (mid_is_then, mid, mid_in, other, other_in) in [
        (true, then_b, &then_args, else_b, &else_args),
        (false, else_b, &else_args, then_b, &then_args),
    ] {
        let Some((c2, mt, mt_args, me, me_args)) = branch_block(func, cfg, (mid, mid_in), MAX_MID_BODY, compare_only) else {
            continue;
        };
        if !func.ty(c2).is_small_int() || mt == me || mt == head || mt == mid || me == head || me == mid {
            continue;
        }
        let join = if other != mt && other != me && other_in.len() == 1 && cfg.pred_count(other) >= 2 {
            other
        } else if let Some(Op::Jump(j, _)) = func.last(other) {
            *j
        } else {
            continue;
        };
        if join == head
            || join == mid
            || func.entry == Some(join)
            || cfg.pred_count(join) < 2
            || func.blocks[join.index()].params.len() != 1
        {
            continue;
        }
        let param_ty = func.ty(func.blocks[join.index()].params[0]);
        if !scalar_select(param_ty) || !param_ty.is_small_int() {
            continue;
        }
        let Some((other_leaf, other_val)) = extract_const_leaf(func, cfg, is_const, other, other_in, join) else {
            continue;
        };
        let Some((mt_leaf, mt_val)) = extract_const_leaf(func, cfg, is_const, mt, &mt_args, join) else {
            continue;
        };
        let Some((me_leaf, me_val)) = extract_const_leaf(func, cfg, is_const, me, &me_args, join) else {
            continue;
        };
        retire_terminator(func, head);
        for (leaf_opt, leaf_in) in [(other_leaf, other_in.as_slice()), (mt_leaf, mt_args.as_slice()), (me_leaf, me_args.as_slice())] {
            if let Some(leaf) = leaf_opt {
                hoist_block_body(func, head, leaf, leaf_in, cleared);
            }
        }
        let (mid_params, _) = hoist_block_body(func, head, mid, mid_in, cleared);
        let c2_val = remap_val(&mid_params, mid_in, c2);
        let inner = emit_select(func, head, param_ty, c2_val, mt_val, me_val);
        let (t_val, e_val) = if mid_is_then { (inner, other_val) } else { (other_val, inner) };
        let outer = emit_select(func, head, param_ty, c1, t_val, e_val);
        let jump = func.add(Op::Jump(join, vec![outer]), None);
        func.blocks[head.index()].insts.push(jump);
        return true;
    }
    false
}

fn defines_compare(func: &Function, block: Block, value: Value) -> bool {
    func.blocks[block.index()].insts.iter().any(|&inst| func.insts[inst.index()].result == Some(value) && matches!(func.insts[inst.index()].op, Op::Icmp(..) | Op::Fcmp(..)))
}

fn try_shared_target(func: &mut Function, cfg: &Cfg, region: &[bool], head: Block, cleared: &mut [bool]) -> bool {
    let Some(Op::Brif(c1, t1, ta1, e1, ea1)) = func.last(head).cloned() else { return false };
    if t1 == e1 || !defines_compare(func, head, c1) {
        return false;
    }
    for (mid, mid_in, both) in [(t1, &ta1, true), (e1, &ea1, false)] {
        if !region[mid.index()] || !mid_in.is_empty() || mid == head {
            continue;
        }
        let Some((c2, t2, ta2, e2, ea2)) = branch_block(func, cfg, (mid, mid_in), MAX_MID_BODY, compare_only) else { continue };
        let shared = if both { e2 == e1 && ea2 == ea1 } else { t2 == t1 && ta2 == ta1 };
        if !shared || t2 == e2 || t2 == mid || e2 == mid || func.ty(c1) != func.ty(c2) || !defines_compare(func, mid, c2) {
            continue;
        }
        retire_terminator(func, head);
        hoist_block_body(func, head, mid, mid_in, cleared);
        let cond = func.value(func.ty(c1));
        let joined = func.add(Op::Binary(if both { Binary::Band } else { Binary::Bor }, c1, c2), Some(cond));
        let term = func.add(Op::Brif(cond, t2, ta2, e2, ea2), None);
        func.blocks[head.index()].insts.extend([joined, term]);
        return true;
    }
    false
}

fn try_head(func: &mut Function, cfg: &Cfg, prov: &Provenance, (is_const, region): (&[bool], &[bool]), head: Block, cleared: &mut [bool]) -> bool {
    let Some((cond, m)) = match_arms(func, cfg, prov, head) else {
        return try_nested_classifier(func, cfg, is_const, head, cleared) || try_shared_target(func, cfg, region, head, cleared);
    };
    let join_params = &func.blocks[m.join.index()].params;
    if m.t_args.len() != join_params.len() || m.e_args.len() != join_params.len() {
        return false;
    }
    let mut selects = 0usize;
    let ok = join_params.iter().zip(m.t_args.iter().zip(&m.e_args)).all(|(p, (t, e))| {
        if t == e {
            return true;
        }
        selects += 1;
        selects <= MAX_SELECTS && scalar_select(func.ty(*p))
    });
    if !ok {
        return false;
    }
    let param_tys: Vec<Type> = join_params.iter().map(|p| func.ty(*p)).collect();
    retire_terminator(func, head);
    for (side, in_args) in m.arms {
        let params = std::mem::take(&mut func.blocks[side.index()].params);
        retire_terminator(func, side);
        let moved = std::mem::take(&mut func.blocks[side.index()].insts);
        if param_tys.is_empty() {
            for inst in moved {
                func.insts[inst.index()].op = Op::Trap;
            }
        } else {
            if !params.is_empty() {
                for &inst in &moved {
                    func.insts[inst.index()].op.map_operands(|v| remap_val(&params, &in_args, v));
                }
            }
            func.blocks[head.index()].insts.extend(moved);
        }
        cleared[side.index()] = true;
    }
    let selected: Vec<Value> = param_tys
        .into_iter()
        .zip(m.t_args)
        .zip(m.e_args)
        .map(|((ty, t_val), e_val)| emit_select(func, head, ty, cond, t_val, e_val))
        .collect();
    let jump = func.add(Op::Jump(m.join, selected), None);
    func.blocks[head.index()].insts.push(jump);
    true
}

/// Converts pure diamond and triangle conditional branches into branchless `Op::Select`.
pub(crate) fn run(func: &mut Function) -> Domain {
    if func.entry.is_none() || func.coroutine.is_some() {
        return Domain::NONE;
    }
    let mut any = false;
    loop {
        let cfg = func.cfg();
        let prov = Provenance::new(func, &cfg);
        let mut is_const = vec![false; func.types.len()];
        for block in &func.blocks {
            for &inst in &block.insts {
                let data = &func.insts[inst.index()];
                if matches!(data.op, Op::Const(_))
                    && let Some(res) = data.result
                {
                    is_const[res.index()] = true;
                }
            }
        }
        let region = crate::thread::unpredictable_region(func, &cfg);
        let mut cleared = vec![false; func.blocks.len()];
        let mut changed = false;
        for &block in cfg.rpo.iter().rev() {
            if !cleared[block.index()] {
                changed |= try_head(func, &cfg, &prov, (&is_const, &region), block, &mut cleared);
            }
        }
        if !changed {
            return if any { Domain::CFG | Domain::VALUES } else { Domain::NONE };
        }
        any = true;
        func.bump_cfg();
        crate::merge::run(func);
    }
}
