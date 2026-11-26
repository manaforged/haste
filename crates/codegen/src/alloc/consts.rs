use crate::cfg::Loops;
use crate::fold::fits_imm32;
use crate::frame::Abi;
use crate::ir::{Binary, Function, IntCC, Op, Signature, Type, Value};
use crate::live::{Live, covers, overlaps};

use super::{Location, Regs, Uses};

const MAX_ARM64_LOOP_GP: usize = 18;
const I64_CONST_MIN_HALVES: usize = 3;
const SHIFT_LHS_MIN_HALVES: usize = 2;
const LOOP_CONST_MIN_USES: u32 = 2;
const CONST_PROMOTE_MIN_USES: u32 = 4;

fn half_counts_at_least(u: u64, min: usize) -> bool {
    let non_zero = (0..4).filter(|&s| ((u >> (s * 16)) & 0xffff) != 0).count();
    let non_ones = (0..4)
        .filter(|&s| ((u >> (s * 16)) & 0xffff) != 0xffff)
        .count();
    non_zero >= min && non_ones >= min
}

fn arm64_bitmask(u: u64) -> bool {
    crate::arm64::int::encode_bitmask_imm(u, true).is_some()
        || ((u >> 32) == 0 && crate::arm64::int::encode_bitmask_imm(u, false).is_some())
}

fn expensive_i64_const(imm: i64, two_address: bool) -> bool {
    let u = imm as u64;
    !(two_address && (u >> 32) == (u & 0xffff_ffff))
        && half_counts_at_least(u, I64_CONST_MIN_HALVES)
        && (two_address || !arm64_bitmask(u))
}

fn shift_lhs_expensive(imm: i64, ty: Type, two_address: bool) -> bool {
    if two_address {
        ty == Type::I64 && !fits_imm32(imm) && u32::try_from(imm).is_err()
    } else {
        let u = crate::fold::extend(imm, ty, false) as u64;
        half_counts_at_least(u, SHIFT_LHS_MIN_HALVES) && !arm64_bitmask(u)
    }
}

fn x64_alu_imm(ty: Type, imm: i64) -> bool {
    ty == Type::I32 || fits_imm32(imm)
}

fn x64_inline_imm(op: &Op, ty: Type, rhs: bool, imm: i64, flag_only_band: bool) -> bool {
    match op {
        Op::Icmp(..) => ty != Type::I64 || fits_imm32(imm),
        Op::Binary(Binary::Iadd, ..) => x64_alu_imm(ty, imm),
        Op::Binary(Binary::Bor | Binary::Bxor, ..) => {
            x64_alu_imm(ty, imm) || (ty == Type::I64 && (imm as u64).is_power_of_two())
        }
        Op::Binary(Binary::Band, ..) => {
            x64_alu_imm(ty, imm)
                || imm == 0xFFFF_FFFF
                || (ty == Type::I64
                    && ((!(imm as u64)).is_power_of_two()
                        || (flag_only_band && (imm as u64).is_power_of_two())))
        }
        Op::Binary(Binary::Isub, ..) => rhs && x64_alu_imm(ty, imm),
        _ => false,
    }
}

fn flag_only_bands<R>(func: &Function, census: &Uses, locations: &[Location<R>]) -> Vec<bool> {
    let mut flag = vec![false; func.types.len()];
    let is_zero = |v: Value| matches!(locations[v.index()], Location::Imm(0));
    for block in &func.blocks {
        for &inst in &block.insts {
            match func.insts[inst.index()].op {
                Op::Brif(c, ..) | Op::Trapz(c) | Op::Select(c, ..)
                    if census.uses[c.index()] == 1 =>
                {
                    flag[c.index()] = true
                }
                Op::Icmp(IntCC::Equal | IntCC::NotEqual, a, b) => {
                    if is_zero(b) && census.uses[a.index()] == 1 {
                        flag[a.index()] = true;
                    }
                    if is_zero(a) && census.uses[b.index()] == 1 {
                        flag[b.index()] = true;
                    }
                }
                _ => {}
            }
        }
    }
    flag
}

fn arm64_inline_imm(op: &Op, ty: Type, rhs: bool, imm: i64) -> bool {
    match *op {
        Op::Binary(Binary::Iadd | Binary::Isub, ..) => {
            rhs && crate::arm64::int::add_imm_parts(false, crate::fold::extend(imm, ty, true))
                .is_some()
        }
        Op::Icmp(cc, ..) => {
            rhs && crate::arm64::int::add_imm_parts(
                true,
                crate::fold::extend(imm, ty, cc.is_signed()),
            )
            .is_some()
        }
        Op::Binary(Binary::Band | Binary::Bor | Binary::Bxor, ..) => {
            crate::arm64::int::encode_bitmask_imm(imm as u64, ty == Type::I64).is_some()
        }
        _ => false,
    }
}

/// Returns whether ARM64 lowers an integer multiply by `imm` without materializing a register constant.
pub(crate) fn arm64_trivial_mul_imm(ty: Type, imm: i64) -> bool {
    let uimm = crate::fold::extend(imm, ty, false) as u64;
    let simm = crate::fold::extend(imm, ty, true);
    let max_shift = if ty == Type::I64 { 63 } else { 31 };
    uimm <= 1
        || simm == -1
        || uimm.is_power_of_two()
        || (uimm - 1).is_power_of_two()
        || uimm
            .checked_add(1)
            .is_some_and(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift)
        || simm
            .checked_neg()
            .filter(|&n| n > 2)
            .and_then(|n| (n as u64).checked_add(1))
            .is_some_and(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift)
}

fn arm64_loop_pressure<R>(
    func: &Function,
    census: &Uses,
    loops: &Loops,
    elided: &[bool],
    locations: &[Location<R>],
    is_cand: &[bool],
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let (mut loop_gp, mut cand_loops, mut seen) = (
        vec![0_usize; loops.list.len()],
        vec![Vec::new(); func.types.len()],
        vec![0_u32; func.types.len()],
    );
    for (idx, each) in loops.list.iter().enumerate() {
        let (tag, mut gp) = ((idx + 1) as u32, 0_usize);
        let mut count_val = |v: Value, is_head_param: bool| {
            let vi = v.index();
            if seen[vi] != tag && !elided[vi] {
                let outside = census.def[vi]
                    .is_none_or(|(db, _)| !each.body.get(db).copied().unwrap_or(false));
                if func.types[vi].is_small_int() && matches!(locations[vi], Location::Slot) {
                    if is_head_param || outside {
                        seen[vi] = tag;
                        gp += 1;
                    }
                } else if is_cand[vi] && outside {
                    seen[vi] = tag;
                    cand_loops[vi].push(idx);
                }
            }
        };
        func.blocks[each.head.index()]
            .params
            .iter()
            .copied()
            .for_each(|v| count_val(v, true));
        for (b, &in_l) in each.body.iter().enumerate() {
            if in_l {
                func.blocks[b].insts.iter().for_each(|&inst| {
                    func.insts[inst.index()]
                        .op
                        .for_operands(|v| count_val(v, false))
                });
            }
        }
        loop_gp[idx] = gp;
    }
    (loop_gp, cand_loops)
}

fn scan_const_op<R>(
    func: &Function,
    op: &Op,
    locations: &[Location<R>],
    (two_address, in_loop, flag_only_band, outer): (bool, bool, bool, &dyn Fn(Value) -> bool),
    (must_stay_imm, loop_mat): (&mut [bool], &mut [u32]),
) {
    match *op {
        Op::Copy(_, _, bytes) => must_stay_imm[bytes.index()] = true,
        Op::Binary(
            Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr,
            lhs,
            rhs,
        ) => {
            must_stay_imm[rhs.index()] = true;
            if in_loop
                && outer(lhs)
                && let Location::Imm(imm) = locations[lhs.index()]
                && imm != 0
                && !matches!(locations[rhs.index()], Location::Imm(_))
                && shift_lhs_expensive(imm, func.ty(lhs), two_address)
            {
                loop_mat[lhs.index()] += 1;
            }
        }
        Op::Binary(
            Binary::Imul | Binary::Udiv | Binary::Urem | Binary::Sdiv | Binary::Srem,
            x,
            y,
        ) => {
            let pow2_ok = matches!(
                op,
                Op::Binary(Binary::Imul | Binary::Udiv | Binary::Urem, ..)
            );
            let is_mul = matches!(op, Op::Binary(Binary::Imul, ..));
            for (idx, v) in [x, y].into_iter().enumerate() {
                if let Location::Imm(imm) = locations[v.index()] {
                    let is_pow2 =
                        (crate::fold::extend(imm, func.ty(v), false) as u64).is_power_of_two();
                    let keep_imm = if two_address {
                        (pow2_ok && is_pow2) || (is_mul && fits_imm32(imm))
                    } else {
                        (is_mul && arm64_trivial_mul_imm(func.ty(v), imm))
                            || (!is_mul && idx == 1 && pow2_ok && is_pow2)
                    };
                    if keep_imm {
                        must_stay_imm[v.index()] = true;
                    } else if in_loop && outer(v) {
                        loop_mat[v.index()] += 1;
                    }
                }
            }
        }
        Op::Binary(
            Binary::Iadd | Binary::Isub | Binary::Band | Binary::Bor | Binary::Bxor,
            x,
            y,
        )
        | Op::Icmp(_, x, y)
            if in_loop =>
        {
            let ty = func.ty(x);
            for (rhs, v) in [(false, x), (true, y)] {
                let Location::Imm(imm) = locations[v.index()] else {
                    continue;
                };
                let inline = if two_address {
                    x64_inline_imm(op, ty, rhs, imm, flag_only_band)
                } else {
                    arm64_inline_imm(op, ty, rhs, imm)
                };
                if inline {
                    must_stay_imm[v.index()] = true;
                } else if outer(v) {
                    loop_mat[v.index()] += 1;
                }
            }
        }
        Op::Store(val, ..) | Op::StackStore(val, ..) if in_loop => {
            if let Location::Imm(imm) = locations[val.index()] {
                if two_address && (func.ty(val).bytes() < 8 || fits_imm32(imm)) {
                    must_stay_imm[val.index()] = true;
                } else if two_address && imm != 0 && outer(val) {
                    loop_mat[val.index()] += 1;
                }
            }
        }
        _ => {}
    }
}

pub(super) fn promote_loop_consts<R>(
    func: &Function,
    census: &Uses,
    loops: &Loops,
    elided: &[bool],
    two_address: bool,
    locations: &mut [Location<R>],
) -> Vec<(usize, i64)> {
    let count = func.types.len();
    let (mut must_stay_imm, mut loop_mat) = (vec![false; count], vec![0_u32; count]);
    let depths = &loops.depths;
    let flag_bands = two_address.then(|| flag_only_bands(func, census, locations));
    for (b, block) in func.blocks.iter().enumerate() {
        let depth = depths[b];
        let outer = |v: Value| census.def[v.index()].is_some_and(|(db, _)| depths[db] < depth);
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if data.result.is_some_and(|r| elided[r.index()])
                && !matches!(data.op, Op::Binary(Binary::Imul, ..))
            {
                continue;
            }
            let flag_band = flag_bands
                .as_ref()
                .is_some_and(|fb| data.result.is_some_and(|r| fb[r.index()]));
            scan_const_op(
                func,
                &data.op,
                locations,
                (two_address, depth > 0, flag_band, &outer),
                (&mut must_stay_imm, &mut loop_mat),
            );
            data.op.for_operands(|v| {
                let vi = v.index();
                if let Location::Imm(imm) = locations[vi]
                    && func.ty(v) == Type::I64
                    && expensive_i64_const(imm, two_address)
                    && ((depth > 0
                        && (outer(v) || (!two_address && census.uses[vi] >= LOOP_CONST_MIN_USES)))
                        || census.uses[vi] >= CONST_PROMOTE_MIN_USES)
                {
                    loop_mat[vi] += 1;
                }
            });
        }
    }
    let is_cand: Vec<bool> = (0..count)
        .map(|v| {
            matches!(locations[v], Location::Imm(imm) if imm != 0)
                && !elided[v]
                && !must_stay_imm[v]
                && loop_mat[v] > 0
        })
        .collect();
    let mut pressure = (!two_address)
        .then(|| arm64_loop_pressure(func, census, loops, elided, locations, &is_cand));
    let mut promoted = Vec::new();
    for (v, loc) in locations.iter_mut().enumerate() {
        if is_cand[v]
            && let Location::Imm(imm) = *loc
        {
            if let Some((loop_gp, cand_loops)) = pressure.as_mut() {
                if cand_loops[v]
                    .iter()
                    .any(|&idx| loop_gp[idx] >= MAX_ARM64_LOOP_GP)
                {
                    continue;
                }
                for &idx in &cand_loops[v] {
                    loop_gp[idx] += 1;
                }
            }
            *loc = Location::Slot;
            promoted.push((v, imm));
        }
    }
    promoted
}

pub(super) fn promote_loop_syms<R: Copy>(
    func: &Function,
    census: &Uses,
    depths: Option<&[u32]>,
    elided: &[bool],
    locations: &mut [Location<R>],
) -> Vec<(usize, Location<R>)> {
    let mut in_loop = vec![false; func.types.len()];
    if let Some(depths) = depths {
        let outer =
            |v: Value, depth: u32| census.def[v.index()].is_some_and(|(db, _)| depths[db] < depth);
        for (b, block) in func
            .blocks
            .iter()
            .enumerate()
            .filter(|(b, _)| depths[*b] > 0)
        {
            let depth = depths[b];
            for &inst in &block.insts {
                let data = &func.insts[inst.index()];
                if !data.result.is_some_and(|r| elided[r.index()]) {
                    data.op.for_operands(|v| {
                        if matches!(
                            locations[v.index()],
                            Location::SlotAddr(_) | Location::Symbol(_)
                        ) && outer(v, depth)
                        {
                            in_loop[v.index()] = true;
                        }
                    });
                }
            }
        }
    }
    let folds = addressed_images(func);
    let cold_cell = thread_cell_arg_only(func);
    let mut promoted = Vec::new();
    for (v, loc) in locations.iter_mut().enumerate() {
        if matches!(*loc, Location::SlotAddr(_) | Location::Symbol(_))
            && !elided[v]
            && !folds[v]
            && !cold_cell[v]
            && (census.uses[v] >= 2
                || in_loop[v]
                || depths.is_some_and(|d| census.def[v].is_some_and(|(db, _)| d[db] > 0)))
        {
            promoted.push((v, *loc));
            *loc = Location::Slot;
        }
    }
    promoted
}

fn thread_cell_arg_only(func: &Function) -> Vec<bool> {
    let mut only = vec![true; func.types.len()];
    let mut seen = vec![false; func.types.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            match &func.insts[inst.index()].op {
                Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. }
                    if func.is_thread_cell(*callee) =>
                {
                    only[callee.index()] = false;
                    for &arg in args {
                        seen[arg.index()] = true;
                    }
                }
                op => op.for_operands(|v| only[v.index()] = false),
            }
        }
    }
    only.into_iter()
        .zip(seen)
        .map(|(ok, used)| ok && used)
        .collect()
}

fn addressed_images(func: &Function) -> Vec<bool> {
    let mut folds: Vec<bool> = (0..func.types.len())
        .map(|v| func.is_image_addr(Value(v as u32)))
        .collect();
    let mut small = vec![false; func.types.len()];
    for data in &func.insts {
        if let (Op::Const(k), Some(result)) = (&data.op, data.result) {
            small[result.index()] = i32::try_from(*k).is_ok();
        }
    }
    for block in &func.blocks {
        for &inst in &block.insts {
            match func.insts[inst.index()].op {
                Op::Load(..) | Op::AtomicLoad(..) => {}
                Op::Store(value, ..) => folds[value.index()] = false,
                Op::Binary(Binary::Iadd | Binary::Isub, _, k) if small[k.index()] => {}
                ref op => op.for_operands(|v| folds[v.index()] = false),
            }
        }
    }
    folds
}

/// Promotes spilled `Iadd`/`Isub` of a live register and a single-instruction immediate into `Location::Offset(r, off)`.
pub(super) fn promote_reg_offsets<R: Copy + PartialEq>(
    func: &Function,
    regs: &Regs<R>,
    live: &Live,
    census: &Uses,
    (pairs, deferred): (&[Option<Value>], &[bool]),
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
    elided: &[bool],
    locations: &mut [Location<R>],
) {
    let saved_index = |reg: R| {
        regs.saved.iter().position(|&s| s == reg).filter(|_| {
            !regs.wide_scratch.contains(&reg)
                && !regs.op_scratch.contains(&reg)
                && !regs.preserved_clobbers.contains(&reg)
        })
    };
    let mut owners: Vec<Vec<usize>> = vec![Vec::new(); regs.saved.len()];
    for (w, loc) in locations.iter().enumerate() {
        if let Location::Reg(r) = *loc
            && let Some(idx) = saved_index(r)
        {
            owners[idx].push(w);
        }
    }
    let mut call_sigs: Vec<Vec<usize>> = Vec::new();
    let mut special_use: Vec<bool> = Vec::new();
    let mut use_scratches: Vec<u8> = Vec::new();
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            let mask = (regs.scratch_mask)(func, &data.op);
            if mask != 0 {
                if use_scratches.is_empty() {
                    use_scratches.resize(func.types.len(), 0);
                }
                data.op.for_operands(|v| use_scratches[v.index()] |= mask);
            }
            match &data.op {
                Op::Call {
                    sig, callee, args, ..
                }
                | Op::TryCall {
                    sig, callee, args, ..
                } => {
                    let s = sig.index();
                    for v in std::iter::once(*callee).chain(args.iter().copied()) {
                        if call_sigs.is_empty() {
                            call_sigs.resize(func.types.len(), Vec::new());
                        }
                        if !call_sigs[v.index()].contains(&s) {
                            call_sigs[v.index()].push(s);
                        }
                    }
                }
                op @ (Op::Return(_) | Op::Move(..) | Op::Copy(..)) => {
                    op.for_operands(|v| {
                        if special_use.is_empty() {
                            special_use.resize(func.types.len(), false);
                        }
                        special_use[v.index()] = true;
                    });
                }
                _ => {}
            }
        }
    }
    for v in 0..func.types.len() {
        if !matches!(locations[v], Location::Slot)
            || elided[v]
            || deferred[v]
            || pairs[v].is_some()
            || func.types[v] != Type::I64
        {
            continue;
        }
        let ranges = live.ranges_of(v);
        if ranges.is_empty() {
            continue;
        }
        let Some(inst) = census.def_inst[v] else {
            continue;
        };
        let (base, off) = match func.insts[inst.index()].op {
            Op::Binary(Binary::Iadd, a, b) => match (locations[a.index()], locations[b.index()]) {
                (_, Location::Imm(imm)) => (a, imm),
                (Location::Imm(imm), _) => (b, imm),
                _ => continue,
            },
            Op::Binary(Binary::Isub, a, b) => match locations[b.index()] {
                Location::Imm(imm) => (a, imm.wrapping_neg()),
                _ => continue,
            },
            _ => continue,
        };
        if base.index() == v || elided[base.index()] || func.ty(base) != Type::I64 {
            continue;
        }
        let Ok(off32) = i32::try_from(off) else {
            continue;
        };
        if !regs.two_address && crate::arm64::int::add_imm_parts(false, off).is_none() {
            continue;
        }
        let Location::Reg(r) = locations[base.index()] else {
            continue;
        };
        let saved = saved_index(r);
        let base_live = covers(live.ranges_of(base.index()), ranges);
        let op_scratched = |v: usize, reg: R| {
            let mask = use_scratches.get(v).copied().unwrap_or(0);
            mask != 0
                && regs
                    .op_scratch
                    .iter()
                    .enumerate()
                    .any(|(i, &s)| (mask >> i) & 1 != 0 && s == reg)
        };
        let ok = if base_live {
            saved.is_some()
                || (!regs.wide_scratch.contains(&r)
                    && !regs.preserved_clobbers.contains(&r)
                    && !op_scratched(v, r)
                    && !special_use.get(v).copied().unwrap_or(false)
                    && !crate::park::crosses_ranges(&live.unwinds, ranges)
                    && call_sigs.get(v).is_none_or(|sigs| {
                        sigs.iter()
                            .all(|&s| classify(&func.sigs[s]).is_ok_and(|abi| !abi.clobbers(r)))
                    }))
        } else {
            saved.is_some_and(|idx| {
                owners[idx]
                    .iter()
                    .all(|&w| w == base.index() || !overlaps(ranges, live.ranges_of(w)))
            })
        };
        if !ok {
            continue;
        }
        locations[v] = Location::Offset(r, off32);
    }
}
