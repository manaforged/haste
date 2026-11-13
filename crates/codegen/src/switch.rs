use rustc_hash::FxHashMap as HashMap;

use crate::fold::extend;
use crate::ir::{Binary, Block, BlockData, Convert, Function, InstData, IntCC, Op, Type, Value};

struct Info {
    preds: Vec<u32>,
    uses: Vec<u32>,
    consts: Vec<Option<i64>>,
    def_block: Vec<Option<usize>>,
    cmps: Vec<Option<(Value, Value)>>,
}

fn analyze(func: &Function) -> Info {
    let preds = func.pred_counts();
    let uses = func.use_counts();
    let mut consts = vec![None; func.types.len()];
    let mut def_block = vec![None; func.types.len()];
    let mut cmps = vec![None; func.types.len()];
    for (b, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result {
                def_block[res.index()] = Some(b);
                match data.op {
                    Op::Const(imm) => consts[res.index()] = Some(imm),
                    Op::Icmp(IntCC::Equal, a, b) => cmps[res.index()] = Some((a, b)),
                    _ => {}
                }
            }
        }
    }
    Info { preds, uses, consts, def_block, cmps }
}

fn match_eq(func: &Function, info: &Info, cond: Value, expected_val: Option<Value>, expected_block: Option<usize>) -> Option<(Value, i64)> {
    if info.uses[cond.index()] != 1 {
        return None;
    }
    if expected_block.is_some() && info.def_block[cond.index()] != expected_block {
        return None;
    }
    let (a, b) = info.cmps[cond.index()]?;
    let (val, imm) = match (info.consts[a.index()], info.consts[b.index()]) {
        (_, Some(imm)) => (a, imm),
        (Some(imm), None) => (b, imm),
        _ => return None,
    };
    if !func.ty(val).is_small_int() || expected_val.is_some_and(|v| v != val) {
        return None;
    }
    Some((val, imm))
}

fn block_only(func: &Function, info: &Info, block: usize, v: Value) -> Option<(Value, i64, Block, Vec<Value>, Block)> {
    let insts = &func.blocks[block].insts;
    let (&last, body) = insts.split_last()?;
    let Op::Brif(cond, hit, hit_args, next, else_args) = &func.insts[last.index()].op else { return None };
    if !else_args.is_empty() || !func.blocks[next.index()].params.is_empty() {
        return None;
    }
    let (_, imm) = match_eq(func, info, *cond, Some(v), Some(block))?;
    for &inst in body {
        let data = &func.insts[inst.index()];
        if !matches!(data.op, Op::Const(_) | Op::Icmp(IntCC::Equal, ..)) {
            return None;
        }
        if let Some(res) = data.result {
            let mut local = 0_u32;
            for &other in insts {
                func.insts[other.index()].op.for_operands(|op| local += u32::from(op == res));
            }
            if local != info.uses[res.index()] {
                return None;
            }
        }
    }
    Some((v, imm, *hit, hit_args.clone(), *next))
}

fn forward(func: &mut Function, target: Block, args: Vec<Value>) -> Block {
    if args.is_empty() && func.blocks[target.index()].params.is_empty() {
        return target;
    }
    let block = Block(func.blocks.len() as u32);
    let inst = func.add(Op::Jump(target, args), None);
    func.blocks.push(BlockData { params: Vec::new(), insts: vec![inst] });
    block
}

fn push_before_last(func: &mut Function, block: usize, op: Op, ty: Type) -> Value {
    let val = func.value(ty);
    let inst = func.add(op, Some(val));
    let insts = &mut func.blocks[block].insts;
    let pos = insts.len() - 1;
    insts.insert(pos, inst);
    val
}

pub(crate) fn run(func: &mut Function) {
    let info = analyze(func);
    let initial_blocks = func.blocks.len();
    let mut cleared = vec![false; initial_blocks];
    for b0 in 0..initial_blocks {
        if cleared[b0] {
            continue;
        }
        let Some(&last0) = func.blocks[b0].insts.last() else { continue };
        let Op::Brif(cond0, hit0, hit_args0, next0, else_args0) = &func.insts[last0.index()].op else { continue };
        if !else_args0.is_empty() || !func.blocks[next0.index()].params.is_empty() {
            continue;
        }
        let Some((v, k0)) = match_eq(func, &info, *cond0, None, None) else { continue };
        let mut arms = vec![(k0, *hit0, hit_args0.clone())];
        let mut chain = Vec::new();
        let mut cur = next0.index();
        while cur < initial_blocks && cur != b0 && !cleared[cur] && !chain.contains(&cur) && info.preds[cur] == 1 && func.blocks[cur].params.is_empty() {
            let Some((_, imm, hit, hit_args, next)) = block_only(func, &info, cur, v) else { break };
            chain.push(cur);
            arms.push((imm, hit, hit_args));
            cur = next.index();
        }
        let ty = func.ty(v);
        let mut seen: HashMap<u64, (i64, Block, Vec<Value>)> = HashMap::default();
        for (imm, hit, hit_args) in arms {
            let u = extend(imm, ty, false) as u64;
            seen.entry(u).or_insert((imm, hit, hit_args));
        }
        let distinct = seen.len();
        if distinct < 4 {
            continue;
        }
        let (mut u_min, mut u_max) = (u64::MAX, 0_u64);
        let (mut s_min, mut s_max) = (i64::MAX, i64::MIN);
        for &u in seen.keys() {
            u_min = u_min.min(u);
            u_max = u_max.max(u);
            let s = extend(u as i64, ty, true);
            s_min = s_min.min(s);
            s_max = s_max.max(s);
        }
        let u_span = u_max - u_min;
        let s_span = s_max.abs_diff(s_min);
        let signed = s_span < u_span;
        let span = if signed { s_span } else { u_span };
        let Some(table_len) = usize::try_from(span).ok().and_then(|s| s.checked_add(1)) else { continue };
        if table_len > 256 || table_len > distinct * 4 {
            continue;
        }
        let min = if signed { s_min } else { u_min as i64 };
        let v64 = if ty == Type::I64 {
            v
        } else {
            let conv = if signed { Convert::Sextend } else { Convert::Uextend };
            push_before_last(func, b0, Op::Convert(conv, v), Type::I64)
        };
        let idx = if min != 0 {
            let min_val = push_before_last(func, b0, Op::Const(min), Type::I64);
            push_before_last(func, b0, Op::Binary(Binary::Isub, v64, min_val), Type::I64)
        } else {
            v64
        };
        let default_blk = Block(cur as u32);
        let mut table = vec![default_blk; table_len];
        for (u, (_, hit, hit_args)) in seen {
            let key = if signed { extend(u as i64, ty, true) } else { u as i64 };
            let offset = (key as i128 - min as i128) as usize;
            table[offset] = forward(func, hit, hit_args);
        }
        func.insts[last0.index()] = InstData { op: Op::BrTable(idx, default_blk, table), result: None };
        for b in chain {
            func.blocks[b].insts.drain(..).for_each(|inst| func.insts[inst.index()].op = Op::Trap);
            cleared[b] = true;
        }
    }
}
