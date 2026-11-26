mod chain;
mod count;
mod memset;
mod region;
mod self_loop;
mod symbolic;
mod vector;
mod vector_shape;

use crate::cfg::{Cfg, Loop};
use crate::fold::{evaluate_wrapped, extend};
use crate::ir::{Block, Domain, Function, Inst, Op, Type, Value};
use crate::{Arch, Target};
pub(crate) use memset::Emit;
pub(crate) use vector::splat_in;

const MAX_FULL_TRIPS: usize = 64;
const MAX_FULL_INSTS: usize = 1024;
pub(super) const MAX_FP_FULL_TRIPS: usize = 8;
pub(super) const MAX_FP_FULL_INSTS: usize = 96;
pub(super) const MAX_VEC_FULL_TRIPS: usize = 20;
const MAX_CHAIN_BLOCKS: usize = 8;
const MAX_CHAIN_TRIPS: usize = 64;
const MAX_CHAIN_UNROLLED_INSTS: usize = 384;

type Eval = fn(&Function, &Op, Type, &dyn Fn(Value) -> Option<i64>) -> Option<i64>;

/// The value the latest entry of `map` renames `v` to, or `v` when none does.
pub(crate) fn remap_val(map: &[(Value, Value)], v: Value) -> Value {
    map.iter()
        .rev()
        .find(|&&(old, _)| old == v)
        .map_or(v, |&(_, new)| new)
}

fn is_libcall_mem(op: &Op) -> bool {
    matches!(*op, Op::Copy(..))
        || matches!(*op, Op::Move(_, _, bytes) if bytes > crate::copy::MOVE_INLINE)
}

pub(super) fn block_has_libcall_mem(func: &Function, block_idx: usize) -> bool {
    let insts = &func.blocks[block_idx].insts;
    insts[..insts.len().saturating_sub(1)]
        .iter()
        .any(|&i| is_libcall_mem(&func.insts[i.index()].op))
}

fn has_bad_non_term(func: &Function, block_idx: usize) -> bool {
    let insts = &func.blocks[block_idx].insts;
    insts[..insts.len() - 1].iter().any(|&i| {
        matches!(
            func.insts[i.index()].op,
            Op::Call { .. } | Op::TryCall { .. } | Op::BrTable(..) | Op::Jump(..) | Op::Brif(..)
        )
    })
}

fn eval_block(
    func: &Function,
    block_idx: usize,
    params: &[Option<i64>],
    env: &mut [Option<i64>],
    eval: Eval,
) {
    let block = &func.blocks[block_idx];
    for (p, val) in block.params.iter().zip(params) {
        env[p.index()] = *val;
    }
    for &inst in &block.insts[..block.insts.len() - 1] {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        let ty = func.ty(res);
        env[res.index()] = match data.op {
            Op::Select(c, x, y) if ty.is_small_int() => env[c.index()].and_then(|cv| {
                let chosen = if cv != 0 { x } else { y };
                env[chosen.index()].map(|imm| extend(imm, ty, true))
            }),
            _ => eval(func, &data.op, ty, &|v| env[v.index()]),
        };
    }
}

fn clone_body(
    func: &mut Function,
    src_non_term: &[Inst],
    dst: Block,
    map: &mut Vec<(Value, Value)>,
) {
    for &orig_inst in src_non_term {
        let mut op = func.insts[orig_inst.index()].op.clone();
        op.map_operands(|v| remap_val(map, v));
        let new_res = func.insts[orig_inst.index()].result.map(|r| {
            let nr = func.value_like(r);
            map.push((r, nr));
            nr
        });
        let ni = func.add(op, new_res);
        func.blocks[dst.index()].insts.push(ni);
    }
}

fn remap_other_blocks(func: &mut Function, skip: &[usize], map: &[(Value, Value)]) {
    if map.is_empty() {
        return;
    }
    for ob in 0..func.blocks.len() {
        if skip.contains(&ob) {
            continue;
        }
        for idx in 0..func.blocks[ob].insts.len() {
            let inst = func.blocks[ob].insts[idx];
            func.insts[inst.index()]
                .op
                .map_operands(|v| remap_val(map, v));
        }
    }
}

fn pop_terminator(func: &mut Function, blk_idx: usize) {
    if let Some(popped) = func.blocks[blk_idx].insts.pop() {
        func.insts[popped.index()].op = Op::Trap;
    }
}

fn edge_args(func: &Function, pred: Block, head: Block) -> Option<&[Value]> {
    match func.last(pred)? {
        Op::Jump(target, args) if *target == head => Some(args.as_slice()),
        Op::Brif(_, t_blk, t_args, f_blk, _) if *t_blk == head && *f_blk != head => {
            Some(t_args.as_slice())
        }
        Op::Brif(_, t_blk, _, f_blk, f_args) if *f_blk == head && *t_blk != head => {
            Some(f_args.as_slice())
        }
        _ => None,
    }
}

fn meet(known: &mut Option<Vec<Option<i64>>>, args: &[Value], env: &[Option<i64>]) {
    match known {
        None => *known = Some(args.iter().map(|arg| env[arg.index()]).collect()),
        Some(known) => {
            for (slot, arg) in known.iter_mut().zip(args) {
                *slot = slot.filter(|&imm| env[arg.index()] == Some(imm));
            }
        }
    }
}

fn outside_const_params(
    func: &Function,
    cfg: &Cfg,
    head: Block,
    latches: &[Block],
    global_consts: &[Option<i64>],
) -> Option<Vec<Option<i64>>> {
    let param_len = func.blocks[head.index()].params.len();
    let mut acc: Option<Vec<Option<i64>>> = None;
    for p in 0..func.blocks.len() {
        let blk = Block(p as u32);
        if p == head.index()
            || latches.contains(&blk)
            || !cfg.reachable.get(p).copied().unwrap_or(false)
            || !func.successors(blk).any(|t| t == head)
        {
            continue;
        }
        let args = edge_args(func, blk, head).filter(|a| a.len() == param_len)?;
        meet(&mut acc, args, global_consts);
    }
    acc
}

pub(super) fn block_has_vec(func: &Function, blk: usize) -> bool {
    func.blocks[blk].insts.iter().any(|&i| {
        func.insts[i.index()]
            .result
            .is_some_and(|r| func.ty(r).is_vector())
    })
}

pub(super) fn block_has_float(func: &Function, blk: usize) -> bool {
    func.blocks[blk].insts.iter().any(|&i| {
        func.insts[i.index()]
            .result
            .is_some_and(|r| func.ty(r).is_float())
    })
}

fn push_params_map(map: &mut Vec<(Value, Value)>, params: &[Value], args: &[Value]) {
    for (&p, &a) in params.iter().zip(args.iter()) {
        map.push((p, a));
    }
}

fn collect_global_consts(func: &Function, cfg: &Cfg, eval: Eval) -> Vec<Option<i64>> {
    let mut global_consts = vec![None; func.types.len()];
    for &blk in &cfg.rpo {
        for &inst in &func.blocks[blk.index()].insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if let Some(imm) = eval(func, &data.op, func.ty(res), &|v| global_consts[v.index()]) {
                global_consts[res.index()] = Some(imm);
            }
        }
    }
    global_consts
}

fn unroll_single_block(
    func: &mut Function,
    cfg: &Cfg,
    b: usize,
    global_consts: &[Option<i64>],
    sad: bool,
) -> bool {
    if has_bad_non_term(func, b) {
        return false;
    }
    let Some(&last) = func.blocks[b].insts.last() else {
        return false;
    };
    let Op::Brif(cond, t_blk, t_args, f_blk, f_args) = &func.insts[last.index()].op else {
        return false;
    };
    if t_blk == f_blk {
        return false;
    }
    let head = Block(b as u32);
    let (cond, back_on_true, back_args, exit_blk, exit_args) = if *t_blk == head {
        (*cond, true, t_args.clone(), *f_blk, f_args.clone())
    } else if *f_blk == head {
        (*cond, false, f_args.clone(), *t_blk, t_args.clone())
    } else {
        return false;
    };
    memset::run(
        func,
        b,
        (cond, back_on_true),
        &back_args,
        (exit_blk, &exit_args),
        global_consts,
    ) || vector::run(
        func,
        b,
        (cond, back_on_true),
        &back_args,
        (exit_blk, &exit_args),
        (global_consts, sad),
    ) || self_loop::unroll_self(
        func,
        cfg,
        b,
        cond,
        back_on_true,
        back_args,
        exit_blk,
        exit_args,
        global_consts,
    )
}

fn unroll_loop_step(
    func: &mut Function,
    cfg: &Cfg,
    each: &Loop,
    global_consts: &[Option<i64>],
    sad: bool,
) -> bool {
    let head = each.head;
    if each.latches.len() != 1 {
        region::unroll(func, cfg, each)
    } else if each.size == 1 && each.latches[0] == head {
        unroll_single_block(func, cfg, head.index(), global_consts, sad)
    } else if each.size >= 2 && each.latches[0] != head {
        vector::flatten(func, each, (global_consts, sad))
            || chain::unroll_chain(func, cfg, each, global_consts)
            || region::unroll(func, cfg, each)
    } else {
        false
    }
}

/// Unrolls clean 1-block and straight-line inner loops with constant or dynamic trip counts, and constant-trip loops whose bodies branch or leave early.
pub(crate) fn run(func: &mut Function, target: &Target) -> Domain {
    let sad = target.arch == Arch::X64;
    let Some(entry) = func.entry.filter(|_| func.coroutine.is_none()) else {
        return Domain::NONE;
    };
    let initial = func.cfg();
    let loops = initial.loops();
    let mut cfg = std::sync::Arc::clone(&initial);
    let mut global_consts = collect_global_consts(func, &cfg, evaluate_wrapped);
    let (mut changed, mut dirty) = (false, false);
    for each in &loops.list {
        let head = each.head;
        let b = head.index();
        if head == entry || each.size > MAX_CHAIN_BLOCKS || func.blocks[b].insts.is_empty() {
            continue;
        }
        if dirty {
            func.bump_cfg();
            cfg = func.cfg();
            dirty = false;
        }
        if !cfg.reachable.get(b).copied().unwrap_or(false) || cfg.pred_count(head) < 2 {
            continue;
        }
        global_consts.resize(func.types.len(), None);
        let did = unroll_loop_step(func, &cfg, each, &global_consts, sad);
        changed |= did;
        dirty |= did;
    }
    if !changed {
        return Domain::NONE;
    }
    func.bump_cfg();
    Domain::CFG | Domain::VALUES | Domain::MEM
}

#[cfg(test)]
mod tests {
    use super::run;
    use crate::ir::{Binary, IntCC, Op, Type};
    use crate::{
        AbiParam, CallConv, Effort, Function, FunctionBuilder, Signature, Target, compile, verify,
    };

    fn arm64_full() -> Target {
        let mut target = *Target::find("aarch64-apple-darwin").expect("arm64 darwin target");
        target.effort = Effort::Full;
        target
    }

    fn arm64_words(code: crate::Code) -> Vec<u32> {
        code.bytes()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn unrolls_ne_loop_with_const_stride_above_one_and_drops_dead_exit_cmps() {
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(Type::I64));
        sig.params.push(AbiParam::new(Type::I64));
        sig.returns.push(AbiParam::new(Type::I64));
        let mut func = Function::new(sig);
        {
            let mut b = FunctionBuilder::new(&mut func);
            let entry = b.create_block();
            let head = b.create_block();
            let exit = b.create_block();
            b.append_block_params_for_function_params(entry);
            let params = b.block_params(entry).to_vec();
            let cur_ptr = b.append_block_param(head, Type::I64);
            let acc = b.append_block_param(head, Type::I64);
            let out = b.append_block_param(exit, Type::I64);

            b.switch_to_block(entry);
            let empty = b.ins().icmp(IntCC::Equal, params[0], params[1]);
            let zero = b.ins().iconst(Type::I64, 0);
            let step8 = b.ins().iconst(Type::I64, 8);
            let magic = b.ins().iconst(Type::I64, 0xCCCC_CCCC_CCCC_CCCD_u64 as i64);
            let sh3 = b.ins().iconst(Type::I64, 3);
            b.ins().brif(empty, exit, &[zero], head, &[params[0], zero]);

            b.switch_to_block(head);
            let loaded = b.ins().load(Type::I64, cur_ptr, 0);
            let hi = b.ins().umulhi(loaded, magic);
            let q = b.ins().ushr(hi, sh3);
            let next_acc = b.ins().iadd(acc, q);
            let next_ptr = b.ins().iadd(cur_ptr, step8);
            let cont = b.ins().icmp(IntCC::NotEqual, next_ptr, params[1]);
            b.ins()
                .brif(cont, head, &[next_ptr, next_acc], exit, &[next_acc]);

            b.switch_to_block(exit);
            b.ins().return_(&[out]);
        }

        let target = arm64_full();
        verify::verify(&func, &target).expect("pre-unroll div_const_10 IR valid");

        let mut direct = func.clone();
        let domain = run(&mut direct, &target);
        assert!(!domain.is_empty(), "NotEqual stride=8 loop must unroll");
        verify::verify(&direct, &target).expect("post-unroll div_const_10 IR valid");
        assert_eq!(
            direct.blocks.len(),
            5,
            "expected entry, head, exit, guard, unrolled"
        );

        let guard = &direct.blocks[3];
        let has_eq_one_guard = guard.insts.iter().any(|&i| {
            matches!(
                direct.insts[i.index()].op,
                Op::Icmp(IntCC::Equal, _, b)
                    if direct.insts.iter().any(|d| d.result == Some(b) && d.op == Op::Const(1))
            )
        });
        assert!(
            !has_eq_one_guard,
            "guard must not emit bogus stride == 1 check"
        );
        let has_u32_limit_check = guard.insts.iter().any(|&i| {
            matches!(
                direct.insts[i.index()].op,
                Op::Icmp(IntCC::UnsignedLessThanOrEqual, lim_val, _)
                    if direct.insts.iter().any(|d| d.result == Some(lim_val) && d.op == Op::Const(32))
            )
        });
        assert!(
            has_u32_limit_check,
            "guard must compare s_need=32 <= limit directly"
        );

        let unrolled = &direct.blocks[4];
        let umulhi_count = unrolled
            .insts
            .iter()
            .filter(|&&i| matches!(direct.insts[i.index()].op, Op::Binary(Binary::Umulhi, ..)))
            .count();
        assert_eq!(umulhi_count, 4, "expected 4x unrolled Umulhi");

        let icmp_count = unrolled
            .insts
            .iter()
            .filter(|&&i| matches!(direct.insts[i.index()].op, Op::Icmp(..)))
            .count();
        assert_eq!(
            icmp_count, 1,
            "unrolled block must contain only back-edge Icmp, no cloned exit compares"
        );

        let base_ptr = unrolled.params[0];
        let step_adds_from_base = unrolled
            .insts
            .iter()
            .filter(|&&i| {
                matches!(
                    direct.insts[i.index()].op,
                    Op::Binary(Binary::Iadd, a, b) if a == base_ptr || b == base_ptr
                )
            })
            .count();
        assert_eq!(
            step_adds_from_base, 4,
            "all 4 unrolled steps must offset directly from base_ptr"
        );

        let words = arm64_words(compile(func, &target).expect("arm64 div_const_10 compiles"));
        let umulh_words = words
            .iter()
            .filter(|&&w| (w & 0xFFE0_FC00) == 0x9BC0_7C00)
            .count();
        assert!(
            umulh_words >= 4,
            "compiled ARM64 function must retain live 4x unrolled umulh loop: {words:#x?}"
        );
        assert!(
            !words.contains(&0xD280_0009),
            "compiled ARM64 function must not emit dead mov x9, #0 guard: {words:#x?}"
        );
    }

    #[test]
    fn unrolls_ne_loop_with_dynamic_stride_using_nonzero_guard() {
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(Type::I64));
        sig.params.push(AbiParam::new(Type::I64));
        sig.params.push(AbiParam::new(Type::I64));
        sig.returns.push(AbiParam::new(Type::I64));
        let mut func = Function::new(sig);
        {
            let mut b = FunctionBuilder::new(&mut func);
            let entry = b.create_block();
            let head = b.create_block();
            let exit = b.create_block();
            b.append_block_params_for_function_params(entry);
            let params = b.block_params(entry).to_vec();
            let cur_ptr = b.append_block_param(head, Type::I64);
            let acc = b.append_block_param(head, Type::I64);
            let out = b.append_block_param(exit, Type::I64);

            b.switch_to_block(entry);
            let empty = b.ins().icmp(IntCC::Equal, params[0], params[1]);
            let zero = b.ins().iconst(Type::I64, 0);
            b.ins().brif(empty, exit, &[zero], head, &[params[0], zero]);

            b.switch_to_block(head);
            let loaded = b.ins().load(Type::I64, cur_ptr, 0);
            let next_acc = b.ins().iadd(acc, loaded);
            let next_ptr = b.ins().iadd(cur_ptr, params[2]);
            let cont = b.ins().icmp(IntCC::NotEqual, next_ptr, params[1]);
            b.ins()
                .brif(cont, head, &[next_ptr, next_acc], exit, &[next_acc]);

            b.switch_to_block(exit);
            b.ins().return_(&[out]);
        }

        let target = arm64_full();
        let mut direct = func.clone();
        assert!(!run(&mut direct, &target).is_empty());
        verify::verify(&direct, &target).expect("dynamic stride NotEqual IR valid");

        let guard = &direct.blocks[3];
        let has_ugt_zero = guard.insts.iter().any(|&i| {
            matches!(
                direct.insts[i.index()].op,
                Op::Icmp(IntCC::UnsignedGreaterThan, _, z)
                    if direct.insts.iter().any(|d| d.result == Some(z) && d.op == Op::Const(0))
            )
        });
        assert!(has_ugt_zero, "dynamic NotEqual guard must check stride > 0");
        let code = compile(func, &target).expect("dynamic stride NotEqual compiles on ARM64");
        assert!(!code.bytes().is_empty());
    }

    #[test]
    fn skips_trapz_cmps_and_bias_adds_in_unrolled_loop() {
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(Type::I64));
        sig.params.push(AbiParam::new(Type::I64));
        sig.params.push(AbiParam::new(Type::I64));
        sig.returns.push(AbiParam::new(Type::I64));
        let mut func = Function::new(sig);
        {
            let mut b = FunctionBuilder::new(&mut func);
            let entry = b.create_block();
            let head = b.create_block();
            let exit = b.create_block();
            b.append_block_params_for_function_params(entry);
            let params = b.block_params(entry).to_vec();
            let iv = b.append_block_param(head, Type::I64);
            let acc = b.append_block_param(head, Type::I64);
            let out = b.append_block_param(exit, Type::I64);

            b.switch_to_block(entry);
            let zero = b.ins().iconst(Type::I64, 0);
            b.ins().jump(head, &[zero, zero]);

            b.switch_to_block(head);
            let in_len = b.ins().icmp(IntCC::UnsignedLessThan, iv, params[1]);
            b.ins().trapz(in_len);
            let addr = b.ins().iadd(params[0], iv);
            let loaded = b.ins().load(Type::I8, addr, 0);
            let loaded_ext = b.ins().uextend(Type::I64, loaded);
            let next_acc = b.ins().iadd(acc, loaded_ext);
            let next_iv = b.ins().iadd_imm(iv, 1);
            let iv_bias = b.ins().iadd_imm(next_iv, 2);
            let cont = b.ins().icmp(IntCC::UnsignedLessThan, iv_bias, params[2]);
            b.ins()
                .brif(cont, head, &[next_iv, next_acc], exit, &[next_acc]);

            b.switch_to_block(exit);
            b.ins().return_(&[out]);
        }

        let target = arm64_full();
        assert!(!run(&mut func, &target).is_empty());
        verify::verify(&func, &target).expect("trapz bias unroll IR valid");
        let unrolled = &func.blocks[4];
        let trap_count = unrolled
            .insts
            .iter()
            .filter(|&&i| matches!(func.insts[i.index()].op, Op::Trapz(_)))
            .count();
        let icmp_count = unrolled
            .insts
            .iter()
            .filter(|&&i| matches!(func.insts[i.index()].op, Op::Icmp(..)))
            .count();
        assert_eq!(trap_count, 0);
        assert_eq!(
            icmp_count, 1,
            "unrolled block must drop both trapz and exit compares"
        );
    }

    #[test]
    fn rejects_hoisted_non_positive_constant_stride() {
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(Type::I64));
        sig.params.push(AbiParam::new(Type::I64));
        sig.returns.push(AbiParam::new(Type::I64));
        let mut func = Function::new(sig);
        {
            let mut b = FunctionBuilder::new(&mut func);
            let entry = b.create_block();
            let head = b.create_block();
            let exit = b.create_block();
            b.append_block_params_for_function_params(entry);
            let params = b.block_params(entry).to_vec();
            let iv = b.append_block_param(head, Type::I64);
            let acc = b.append_block_param(head, Type::I64);
            let out = b.append_block_param(exit, Type::I64);

            b.switch_to_block(entry);
            let zero = b.ins().iconst(Type::I64, 0);
            b.ins().jump(head, &[params[0], zero]);

            b.switch_to_block(head);
            let loaded = b.ins().load(Type::I64, iv, 0);
            let next_acc = b.ins().iadd(acc, loaded);
            let next_iv = b.ins().iadd(iv, zero);
            let cont = b.ins().icmp(IntCC::NotEqual, next_iv, params[1]);
            b.ins()
                .brif(cont, head, &[next_iv, next_acc], exit, &[next_acc]);

            b.switch_to_block(exit);
            b.ins().return_(&[out]);
        }

        assert!(run(&mut func, &arm64_full()).is_empty());
        assert_eq!(func.blocks.len(), 3);
    }
}
