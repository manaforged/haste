mod flags;
mod idioms;
mod loads;

use super::super::REGS;
use super::super::asm::Reg;
use super::super::features::Features;
use super::patterns::Root;
use super::sib::apply_sib_groups;
use crate::alloc::Plan;
use crate::ir::{Binary, Function, Inst, InstData, IntCC, Op, Type, Value};

pub(super) struct FuseCtx<'a> {
    pub(super) func: &'a Function,
    pub(super) plan: &'a Plan<Reg>,
    pub(super) uses: &'a [u32],
    pub(super) pos_of: &'a [super::Site],
    pub(super) defs: &'a [Option<Inst>],
    pub(super) b_idx: usize,
}

pub(super) fn preserved(
    func: &Function,
    plan: &Plan<Reg>,
    b_idx: usize,
    start: usize,
    end: usize,
    skip: &[usize],
    val: Value,
) -> bool {
    crate::emit::preserved(
        func,
        (plan, &REGS),
        (b_idx, start, end, 128),
        |pos| skip.contains(&pos),
        val,
    )
}

pub(in crate::x64) fn iadd_sub_const(plan: &Plan<Reg>, op: &Op) -> Option<(Value, i64)> {
    crate::alloc::iadd_sub_const(op, |v| plan.const_of(v))
}

/// Whether narrowing `x` to `to` drops only bits known to be zero, so the zero flag of `x` is the zero flag of the narrowed value.
pub(in super::super) fn narrows_exactly(
    func: &Function,
    plan: &Plan<Reg>,
    def: &dyn Fn(Value) -> Option<Inst>,
    x: Value,
    to: Type,
) -> bool {
    let (zeros, _) = crate::fold::known_bits(func, def, &|v| plan.const_of(v), x, 0);
    func.ty(x).unsigned_max() & !to.unsigned_max() & !zeros == 0
}

struct UseInfo {
    defs: Vec<Option<Inst>>,
    uses: Vec<u32>,
    pos_of: Vec<super::Site>,
    ls_uses: Vec<Vec<(usize, usize, i32)>>,
    use_of: Vec<super::Site>,
    last_flag_use: Vec<Option<(usize, usize)>>,
    non_flag_use: Vec<bool>,
    const_kids: Vec<Vec<(usize, usize, Value, i64)>>,
    paired: Vec<(Value, [Value; 2])>,
}

fn collect_use_info(func: &Function, plan: &Plan<Reg>) -> UseInfo {
    let n = func.types.len();
    let defs = crate::fold::defs(func);
    let mut info = UseInfo {
        defs,
        uses: vec![0_u32; n],
        pos_of: vec![None; n],
        ls_uses: vec![Vec::new(); n],
        use_of: vec![None; n],
        last_flag_use: vec![None; n],
        non_flag_use: vec![false; n],
        const_kids: vec![Vec::new(); n],
        paired: Vec::new(),
    };
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let flag_cond = match &data.op {
                Op::Select(c, ..) | Op::Brif(c, ..) | Op::Trapz(c) => Some(*c),
                _ => None,
            };
            let pair = loads::low_of_pair(func, block, pos);
            info.paired.extend(pair);
            let reads_nothing =
                pair.is_some() || crate::live::carry_out_of_pair(func, &block.insts, pos);
            data.op.for_operands(|v| {
                if reads_nothing && loads::is_load(func, &info.defs, v) {
                    return;
                }
                info.uses[v.index()] += 1;
                info.use_of[v.index()] = Some((b_idx, pos, inst));
                if flag_cond == Some(v) && !info.non_flag_use[v.index()] {
                    match info.last_flag_use[v.index()] {
                        Some((pb, ppos)) if pb != b_idx || ppos == pos => {
                            info.non_flag_use[v.index()] = true;
                        }
                        _ => info.last_flag_use[v.index()] = Some((b_idx, pos)),
                    }
                } else {
                    info.non_flag_use[v.index()] = true;
                }
            });
            match data.op {
                Op::Load(addr, off) => info.ls_uses[addr.index()].push((b_idx, pos, off)),
                Op::Store(val, addr, off) if val != addr => {
                    info.ls_uses[addr.index()].push((b_idx, pos, off));
                }
                _ => {}
            }
            if let Some(res) = data.result {
                info.pos_of[res.index()] = Some((b_idx, pos, inst));
                if func.ty(res) == Type::I64
                    && plan.sib[res.index()].is_none()
                    && !plan.elided[res.index()]
                    && let Some((parent, delta)) = iadd_sub_const(plan, &data.op)
                {
                    info.const_kids[parent.index()].push((b_idx, pos, res, delta));
                }
            }
        }
    }
    info
}

#[derive(Default)]
struct Candidates {
    branch_tst: Vec<(usize, Value, Option<Value>)>,
    cmp_tst: Vec<(usize, usize, Value, Value, Value)>,
    cand_sibs: Vec<(usize, usize, Value)>,
    cand_shld: Vec<(usize, usize, Value, Value, Value)>,
    cand_adc: Vec<(usize, usize, Value, Binary, Value, Value)>,
    cand_bextr: Vec<(usize, usize, Value, Binary, Value, Value)>,
    cand_loads: Vec<(usize, usize, Value)>,
    cand_alu: Vec<(usize, usize, Value, Binary, Value, Value)>,
}

fn collect_binary_candidate(
    func: &Function,
    plan: &mut Plan<Reg>,
    features: &Features,
    info: &UseInfo,
    (b_idx, pos, res, ty): (usize, usize, Value, Type),
    (op, x, y): (Binary, Value, Value),
    cands: &mut Candidates,
) {
    if plan.elided[res.index()] {
        return;
    }
    if op == Binary::Band && features.bmi1 && ty.is_small_int() {
        idioms::try_elide_andn_bnot(func, plan, (&info.uses, &info.pos_of), (b_idx, pos), (x, y));
        if matches!(ty, Type::I32 | Type::I64) {
            cands.cand_bextr.push((b_idx, pos, res, Binary::Band, x, y));
            cands.cand_alu.push((b_idx, pos, res, Binary::Band, x, y));
        } else if matches!(ty, Type::I8 | Type::I16) && plan.zero_ext[res.index()] {
            cands.cand_alu.push((b_idx, pos, res, Binary::Band, x, y));
        }
        return;
    }
    match op {
        Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Bxor => {
            if matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
                && matches!(ty, Type::I32 | Type::I64)
            {
                cands.cand_shld.push((b_idx, pos, res, x, y));
            }
            if matches!(op, Binary::Iadd | Binary::Isub) && matches!(ty, Type::I32 | Type::I64) {
                cands.cand_adc.push((b_idx, pos, res, op, x, y));
            }
            if matches!(op, Binary::Iadd | Binary::Isub)
                && plan.sib[res.index()].is_none()
                && ty == Type::I64
            {
                cands.cand_sibs.push((b_idx, pos, res));
            }
            if matches!(ty, Type::I32 | Type::I64)
                || (matches!(op, Binary::Bor | Binary::Bxor)
                    && matches!(ty, Type::I8 | Type::I16)
                    && plan.zero_ext[res.index()])
            {
                cands.cand_alu.push((b_idx, pos, res, op, x, y));
            }
        }
        Binary::Band | Binary::Ushr | Binary::Ishl | Binary::Sshr
            if matches!(ty, Type::I32 | Type::I64) =>
        {
            if features.bmi1 && matches!(op, Binary::Band | Binary::Ushr) {
                cands.cand_bextr.push((b_idx, pos, res, op, x, y));
            }
            let nz_shift = matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr)
                && plan
                    .const_of(y)
                    .is_some_and(|c| (c & i64::from(ty.bits() - 1)) != 0);
            if op == Binary::Band || nz_shift || (op == Binary::Ushr && features.bmi1) {
                cands.cand_alu.push((b_idx, pos, res, op, x, y));
            }
        }
        _ => {}
    }
}

fn load_candidate_ty_ok(features: &Features, atomic: bool, ty: Type) -> bool {
    ty.is_small_int()
        || (!atomic
            && features.avx
            && (ty.is_wide_vec() || (ty.is_valid_vec() && ty.bytes() == 16)))
}

fn collect_result_candidate(
    func: &Function,
    plan: &mut Plan<Reg>,
    features: &Features,
    info: &UseInfo,
    (b_idx, pos, res, data): (usize, usize, Value, &InstData),
    cands: &mut Candidates,
) {
    let ty = func.ty(res);
    match data.op {
        Op::Select(cond_val, x, y) if x != y && func.ty(cond_val).is_small_int() => {
            let ctx = FuseCtx {
                func,
                plan,
                uses: &info.uses,
                pos_of: &info.pos_of,
                defs: &info.defs,
                b_idx,
            };
            if ctx.cond_tree(cond_val, pos).is_none() {
                flags::push_branch_tst(
                    func,
                    plan,
                    (&info.uses, &info.pos_of),
                    (b_idx, pos, cond_val),
                    &mut cands.branch_tst,
                );
            }
        }
        Op::Binary(op, x, y) => {
            collect_binary_candidate(
                func,
                plan,
                features,
                info,
                (b_idx, pos, res, ty),
                (op, x, y),
                cands,
            );
        }
        Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) if !plan.elided[res.index()] => {
            cands.cmp_tst.push((b_idx, pos, res, lhs, rhs));
        }
        Op::Load(..) | Op::StackLoad(..)
            if info.uses[res.index()] == 1
                && !plan.elided[res.index()]
                && load_candidate_ty_ok(features, false, ty) =>
        {
            cands.cand_loads.push((b_idx, pos, res));
        }
        Op::AtomicLoad(..)
            if info.uses[res.index()] == 1
                && !plan.elided[res.index()]
                && load_candidate_ty_ok(features, true, ty) =>
        {
            cands.cand_loads.push((b_idx, pos, res));
        }
        _ => {}
    }
}

fn collect_candidates(
    func: &Function,
    plan: &mut Plan<Reg>,
    features: &Features,
    info: &UseInfo,
) -> Candidates {
    let mut cands = Candidates::default();
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else {
                if let Op::Brif(cond_val, ..) | Op::Trapz(cond_val) = data.op {
                    flags::push_branch_tst(
                        func,
                        plan,
                        (&info.uses, &info.pos_of),
                        (b_idx, pos, cond_val),
                        &mut cands.branch_tst,
                    );
                }
                continue;
            };
            collect_result_candidate(
                func,
                plan,
                features,
                info,
                (b_idx, pos, res, data),
                &mut cands,
            );
        }
    }
    cands
}

pub(in super::super) fn fuse(
    func: &Function,
    plan: &mut Plan<Reg>,
    features: &Features,
) -> (Vec<Option<Root>>, Vec<bool>, Vec<bool>) {
    let n = func.types.len();
    let (mut roots, mut pure_test) = (vec![None; n], vec![false; n]);
    if func.size() > 8192 {
        return (roots, pure_test, vec![false; n]);
    }
    let mut info = collect_use_info(func, plan);
    let cands = collect_candidates(func, plan, features, &info);
    flags::apply_single_leaf_trees(
        func,
        plan,
        (&info.uses, &info.pos_of, &info.defs),
        &mut roots,
        (&mut info.last_flag_use, &mut info.non_flag_use),
    );
    let mut tst_end = flags::apply_tst_fusions(
        func,
        plan,
        (&info.uses, &info.pos_of),
        &roots,
        cands.branch_tst,
        cands.cmp_tst,
    );
    let mut skip_flags = vec![false; n];
    idioms::fuse_shld(
        func,
        plan,
        (&info.uses, &info.pos_of),
        &mut skip_flags,
        cands.cand_shld,
    );
    idioms::fuse_adc(
        func,
        plan,
        (&info.uses, &info.pos_of),
        (&mut tst_end, &mut skip_flags),
        cands.cand_adc,
    );
    idioms::fuse_bextr(func, plan, (&info.uses, &info.pos_of), cands.cand_bextr);
    idioms::apply_cand_sibs(
        func,
        plan,
        (&info.uses, &info.pos_of, &info.defs),
        (&info.ls_uses, &info.const_kids),
        &cands.cand_sibs,
    );
    apply_sib_groups(
        func,
        plan,
        (&info.uses, &info.pos_of, &info.defs),
        &info.ls_uses,
    );
    idioms::apply_lea_sibs(
        func,
        plan,
        (&info.uses, &info.pos_of, &info.defs),
        &skip_flags,
        &cands.cand_sibs,
    );
    flags::apply_select_trees(
        func,
        plan,
        (&info.uses, &info.pos_of, &info.defs),
        &mut tst_end,
        &mut roots,
    );
    super::sib::fuse_mem_rmw(func, plan, (&info.uses, &info.pos_of, &info.defs));
    loads::elide_loads(
        func,
        plan,
        features,
        (&info.uses, &info.pos_of, &info.defs, &info.use_of),
        &tst_end,
        cands.cand_loads,
    );
    flags::fuse_alu_zf(
        func,
        plan,
        (
            &info.uses,
            &info.pos_of,
            &info.defs,
            &info.non_flag_use,
            &skip_flags,
        ),
        cands.cand_alu,
    );
    flags::fuse_flag_cmps(
        func,
        plan,
        (&info.uses, &info.pos_of, &info.defs),
        (&info.last_flag_use, &info.non_flag_use),
        &roots,
        (&mut tst_end, &mut pure_test),
    );
    super::sib::fuse_bt_shifts(func, plan, (&info.uses, &info.pos_of), &tst_end, &pure_test);
    loads::fold_products(func, plan, &info.uses, &info.defs);
    super::elide_shift_splats(func, plan, &info.defs, &info.uses);
    if features.avx2 {
        super::elide_broadcast_inserts(
            func,
            plan,
            &info.defs,
            (&info.uses, &info.pos_of, &info.use_of),
        );
    }
    super::sib::fuse_atomic_rmws(func, plan, (&info.uses, &info.pos_of, &info.use_of));
    loads::elide_zero_bases(func, plan, &info.defs);
    loads::restore_paired_loads(func, plan, &info.defs, &info.paired);
    let narrow_byte = flags::find_narrow_bytes(func, plan, &info.defs);
    (roots, pure_test, narrow_byte)
}
