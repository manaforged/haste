use crate::fold::{extend, holds};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, ShuffleMask, Type, Unary, Value};

mod arith;
mod bits;
mod cond;
mod icmp;
mod ovf;
mod select;
mod shift;
mod sum;
mod vec;
pub(crate) use cond::{branch_cmp, compare_offsets, fold_bxor_conds, invert};
pub(crate) use icmp::{is_non_null_addr, non_null_cmp};

use arith::const_addend;
use shift::Demand;

struct RewriteCx<'a> {
    func: &'a mut Function,
    defs: Vec<Option<Inst>>,
    consts: Vec<Option<i64>>,
    uses: Vec<u32>,
    out: Vec<Inst>,
    subst: Vec<Value>,
    changed: bool,
}

impl<'a> RewriteCx<'a> {
    fn def(&self, v: Value) -> Option<Inst> {
        self.defs.get(v.index()).copied().flatten()
    }
    fn sconst(&self, v: Value) -> Option<i64> {
        self.consts.get(v.index()).copied().flatten()
    }
    fn uconst(&self, v: Value, ty: Type) -> Option<u64> {
        self.sconst(v).map(|c| extend(c, ty, false) as u64)
    }
    fn known(&self, v: Value) -> (u64, u64) {
        crate::fold::known_bits(self.func, &|x| self.def(x), &|x| self.sconst(x), v, 0)
    }
    fn op(&self, v: Value) -> Option<&Op> {
        op_of(self.func, |x| self.def(x), v)
    }
    fn count(&self, sh: Value, ty: Type) -> Option<u32> {
        self.sconst(sh).map(|imm| (imm as u32) & (ty.bits() - 1))
    }
    fn single_use(&self, v: Value) -> bool {
        self.uses.get(v.index()).copied().unwrap_or(1) <= 1
    }
    fn is_unit_bool(&self, v: Value) -> bool {
        crate::fold::is_unit_bool(self.func, &|x| self.def(x), &|x| self.sconst(x), v)
    }

    fn emit(&mut self, ty: Type, op: Op) -> Value {
        let val = self.func.value(ty);
        if let Op::Const(imm) = op {
            self.consts.resize(self.func.types.len(), None);
            self.consts[val.index()] = Some(imm);
        }
        let inst = self.func.add(op, Some(val));
        self.defs.resize(self.func.types.len(), None);
        self.defs[val.index()] = Some(inst);
        self.out.push(inst);
        self.changed = true;
        val
    }

    fn alias(&mut self, res: Option<Value>, target: Value) {
        if let Some(r) = res {
            let len = self.func.types.len();
            self.subst
                .extend((self.subst.len()..len).map(|i| Value(i as u32)));
            let mapped = crate::merge::find(&self.subst, target);
            self.changed |= mapped != r;
            self.subst[r.index()] = mapped;
        }
    }

    fn redundant_mask(&self, src: Value, ty: Type, mask: u64) -> bool {
        let active = !self.known(src).0 & ty.unsigned_max();
        self.func.ty(src) == ty && (active & mask) == active
    }

    fn unwrap_uextend(&self, v: Value) -> Value {
        peel_uext(self.func, |x| self.def(x), v)
    }
}

pub(super) fn op_of<'f>(
    func: &'f Function,
    def_of: impl Fn(Value) -> Option<Inst>,
    v: Value,
) -> Option<&'f Op> {
    def_of(v).map(|inst| &func.insts[inst.index()].op)
}

pub(super) fn peel_uextend(mut v: Value, step: impl Fn(Value) -> Option<Value>) -> Value {
    while let Some(inner) = step(v) {
        v = inner;
    }
    v
}

pub(super) fn peel_uext(
    func: &Function,
    def_of: impl Fn(Value) -> Option<Inst>,
    v: Value,
) -> Value {
    peel_uextend(v, |u| match op_of(func, &def_of, u) {
        Some(&Op::Convert(Convert::Uextend, inner)) => Some(inner),
        _ => None,
    })
}

pub(super) fn const_u128(
    func: &Function,
    def_of: impl Fn(Value) -> Option<Inst>,
    ty: Type,
    v: Value,
) -> Option<u128> {
    match *op_of(func, &def_of, v)? {
        Op::Const(imm) => Some(if ty.is_small_int() {
            u128::from(extend(imm, ty, false) as u64)
        } else {
            u128::from(u64::try_from(imm).ok()?)
        }),
        Op::Binary(Binary::Iconcat, lo, hi) if ty == Type::I128 => {
            match (op_of(func, &def_of, lo), op_of(func, &def_of, hi)) {
                (Some(&Op::Const(l)), Some(&Op::Const(h))) => {
                    Some(u128::from(l as u64) | (u128::from(h as u64) << 64))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn known_cond(cx: &mut RewriteCx, cond: Value) -> Option<Value> {
    let (ty, (kz, ko)) = (cx.func.ty(cond), cx.known(cond));
    (cx.sconst(cond).is_none() && ty.is_small_int() && (ko != 0 || kz == ty.unsigned_max()))
        .then(|| cx.emit(ty, Op::Const(i64::from(ko != 0))))
}

fn narrow_pair_icmp(cx: &RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    let (&Op::Convert(k1, a), &Op::Convert(k2, b)) = (
        &cx.func.insts[cx.def(x)?.index()].op,
        &cx.func.insts[cx.def(y)?.index()].op,
    ) else {
        return None;
    };
    if k1 != k2
        || cx.func.ty(a) != cx.func.ty(b)
        || !cx.func.ty(a).is_small_int()
        || !cx.single_use(x)
        || !cx.single_use(y)
    {
        return None;
    }
    let ok = match k1 {
        Convert::Uextend => !cc.is_signed(),
        Convert::Sextend => cc.is_signed() || matches!(cc, IntCC::Equal | IntCC::NotEqual),
        _ => false,
    };
    ok.then_some(Op::Icmp(cc, a, b))
}

fn narrow_const_icmp(
    cx: &mut RewriteCx,
    res: Option<Value>,
    cc: IntCC,
    x: Value,
    y: Value,
) -> Option<Op> {
    let ty = cx.func.ty(x);
    for (norm, ext_v, c_v) in [(cc, x, y), (cc.swapped(), y, x)] {
        let Some(c_imm) = cx.sconst(c_v) else {
            continue;
        };
        match shift::demanded(cx, ext_v, ty.unsigned_max(), 0) {
            Demand::Const(k) if res.is_some_and(|r| cx.func.ty(r) == Type::I8) => {
                return Some(Op::Const(i64::from(holds(norm, ty, k as i64, c_imm))));
            }
            Demand::Bits(v) if v != ext_v => return Some(Op::Icmp(norm, v, c_v)),
            _ => {}
        }
        let Some(inst) = cx.def(ext_v) else { continue };
        let Op::Convert(kind @ (Convert::Uextend | Convert::Sextend), a) =
            cx.func.insts[inst.index()].op
        else {
            continue;
        };
        let sub_ty = cx.func.ty(a);
        if !sub_ty.is_small_int() || sub_ty.bits() >= ty.bits() {
            continue;
        }
        let signed = norm.is_signed();
        if kind == Convert::Uextend && !signed {
            let uc = extend(c_imm, ty, false) as u64;
            if uc <= sub_ty.unsigned_max() {
                if !cx.single_use(ext_v) {
                    continue;
                }
                let nc = cx.emit(sub_ty, Op::Const(extend(uc as i64, sub_ty, true)));
                return Some(Op::Icmp(norm, a, nc));
            }
            if res.is_some_and(|r| cx.func.ty(r) == Type::I8) {
                return Some(Op::Const(i64::from(matches!(
                    norm,
                    IntCC::NotEqual | IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual
                ))));
            }
        } else if kind == Convert::Sextend
            && (signed || matches!(norm, IntCC::Equal | IntCC::NotEqual))
        {
            let (sc, half) = (extend(c_imm, ty, true), 1_i64 << (sub_ty.bits() - 1));
            if (-half..half).contains(&sc) && cx.single_use(ext_v) {
                let nc = cx.emit(sub_ty, Op::Const(extend(sc, sub_ty, true)));
                return Some(Op::Icmp(norm, a, nc));
            }
        }
    }
    None
}

fn simplify_icmp(
    cx: &mut RewriteCx,
    res: Option<Value>,
    cc: IntCC,
    x: Value,
    y: Value,
) -> Option<Op> {
    if let Some(op) = icmp::offset_self(cx, cc, x, y)
        .or_else(|| bits::narrow_bit_test(cx, cc, x, y))
        .or_else(|| bits::shifted_bit_test(cx, cc, x, y))
    {
        return Some(op);
    }
    if matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && let Some(op) = [(x, y), (y, x)].into_iter().find_map(|(m, z)| {
            (cx.sconst(z) == Some(0))
                .then(|| icmp::strip_zero_test(cx, m))
                .flatten()
                .map(|w| Op::Icmp(cc, w, z))
        })
    {
        return Some(op);
    }
    if cx.func.ty(x).is_small_int()
        && let Some(op) = narrow_pair_icmp(cx, cc, x, y)
            .or_else(|| narrow_const_icmp(cx, res, cc, x, y))
            .or_else(|| icmp::strip_const_add(cx, cc, x, y))
    {
        return match op {
            Op::Icmp(ncc, nx, ny) => {
                if let Some(folded) = icmp::eval_select(cx, res, ncc, nx, ny)
                    .or_else(|| select::eval_select_zero_cmp(cx, res, ncc, nx, ny))
                {
                    return folded;
                }
                Some(bits::zero_compare(cx, ncc, nx, ny).unwrap_or(Op::Icmp(ncc, nx, ny)))
            }
            other => Some(other),
        };
    }
    if let Some(folded) = icmp::eval_select(cx, res, cc, x, y)
        .or_else(|| select::eval_select_zero_cmp(cx, res, cc, x, y))
    {
        return folded;
    }
    bits::zero_compare(cx, cc, x, y)
}

fn simplify_band(cx: &mut RewriteCx, res: Option<Value>, a: Value, b: Value) -> Option<Op> {
    let ty = cx.func.ty(a);
    if !ty.is_small_int() {
        return None;
    }
    for (lhs, mask_v) in [(a, b), (b, a)] {
        let Some(mask) = cx.uconst(mask_v, ty) else {
            continue;
        };
        if cx.redundant_mask(lhs, ty, mask) {
            cx.alias(res, lhs);
            return None;
        }
        if mask == 0x8080_8080_8080_8080
            && let Some(x) = shift::match_swar_msb_broadcast(cx, lhs)
        {
            return Some(Op::Binary(Binary::Band, x, mask_v));
        }
        match shift::demanded(cx, lhs, mask, 0) {
            Demand::Const(c) => return Some(Op::Const(extend(c as i64, ty, true))),
            Demand::Bits(src) if src == lhs => {}
            Demand::Bits(src) if cx.redundant_mask(src, ty, mask) => {
                cx.alias(res, src);
                return None;
            }
            Demand::Bits(src) => return Some(Op::Binary(Binary::Band, src, mask_v)),
        }
    }
    icmp::fuse_cmp_ops(cx, res, true, a, b)
}

fn simplify_bxor(cx: &mut RewriteCx, a: Value, b: Value) -> Option<Op> {
    let ty = cx.func.ty(a);
    if !ty.is_small_int() {
        return None;
    }
    let all_ones = ty.unsigned_max();
    for (inner, c2_val) in [(a, b), (b, a)] {
        let Some(c2) = cx.sconst(c2_val) else {
            continue;
        };
        if (extend(c2, ty, false) as u64) == all_ones {
            return Some(Op::Unary(Unary::Bnot, inner));
        }
        let Some(inner_inst) = cx.def(inner) else {
            continue;
        };
        let Op::Binary(Binary::Bxor, x, y) = cx.func.insts[inner_inst.index()].op else {
            continue;
        };
        if let Some((src, c1)) = [(x, y), (y, x)]
            .into_iter()
            .find_map(|(s, cv)| cx.sconst(cv).map(|c| (s, c)))
        {
            let xor_imm = extend(c1 ^ c2, ty, true);
            if (xor_imm as u64 & all_ones) == all_ones {
                return Some(Op::Unary(Unary::Bnot, src));
            }
            let combined = cx.emit(ty, Op::Const(xor_imm));
            return Some(Op::Binary(Binary::Bxor, src, combined));
        }
    }
    None
}

fn simplify_ireduce(cx: &mut RewriteCx, res: Option<Value>, dst: Type, x: Value) -> Option<Op> {
    let cur = match shift::demanded(cx, x, dst.unsigned_max(), 0) {
        Demand::Const(c) => return Some(Op::Const(extend(c as i64, dst, true))),
        Demand::Bits(v) => v,
    };
    if let Some(inst) = cx.def(cur) {
        match cx.func.insts[inst.index()].op {
            Op::Convert(k @ (Convert::Uextend | Convert::Sextend), inner) => {
                let ity = cx.func.ty(inner);
                if ity == dst {
                    cx.alias(res, inner);
                    return None;
                }
                if ity.is_small_int() {
                    return Some(Op::Convert(
                        if ity.bits() < dst.bits() {
                            k
                        } else {
                            Convert::Ireduce
                        },
                        inner,
                    ));
                }
            }
            Op::Convert(Convert::Ireduce, inner) if cx.func.ty(inner).is_small_int() => {
                return Some(Op::Convert(Convert::Ireduce, inner));
            }
            Op::Select(cond, t, f) if cx.single_use(cur) => {
                if let (Some(t), Some(f)) = (
                    select::narrow_arm(cx, dst, t),
                    select::narrow_arm(cx, dst, f),
                ) {
                    let t = t.unwrap_or_else(|c| cx.emit(dst, Op::Const(c)));
                    let f = f.unwrap_or_else(|c| cx.emit(dst, Op::Const(c)));
                    return Some(Op::Select(cond, t, f));
                }
            }
            _ => {}
        }
    }
    shift::ireduce_rotate(cx, dst, cur)
        .or_else(|| (cur != x).then_some(Op::Convert(Convert::Ireduce, cur)))
}

fn simplify_convert(cx: &mut RewriteCx, res: Option<Value>, kind: Convert, x: Value) -> Option<Op> {
    let (dst, src) = (cx.func.ty(res?), cx.func.ty(x));
    if !dst.is_small_int() || !src.is_small_int() {
        return None;
    }
    if kind == Convert::Ireduce && dst.bits() < src.bits() {
        return simplify_ireduce(cx, res, dst, x);
    }
    if dst.bits() <= src.bits() {
        return None;
    }
    match (kind, &cx.func.insts[cx.def(x)?.index()].op) {
        (Convert::Uextend, &Op::Convert(Convert::Uextend, inner))
            if cx.func.ty(inner).is_small_int() =>
        {
            Some(Op::Convert(Convert::Uextend, inner))
        }
        (Convert::Sextend, &Op::Convert(Convert::Sextend, inner))
            if cx.func.ty(inner).is_small_int() =>
        {
            Some(Op::Convert(Convert::Sextend, inner))
        }
        (Convert::Uextend, &Op::Convert(Convert::Ireduce, inner)) if cx.func.ty(inner) == dst => {
            let hi = dst.unsigned_max() & !src.unsigned_max();
            if (cx.known(inner).0 & hi) == hi {
                cx.alias(res, inner);
                return None;
            }
            let mask = cx.emit(dst, Op::Const(extend(src.unsigned_max() as i64, dst, true)));
            Some(Op::Binary(Binary::Band, inner, mask))
        }
        _ => None,
    }
}

fn simplify_binary(
    cx: &mut RewriteCx,
    fits_used: &[bool],
    block_udivs: &mut Vec<(Value, Value, Value)>,
    res: Option<Value>,
    (bin, a, b): (Binary, Value, Value),
) -> Option<Op> {
    let out = match bin {
        Binary::Iadd => sum::negated_shift_add(cx, cx.func.ty(a), a, b)
            .map(|(base, idx, shift)| {
                let ty = cx.func.ty(base);
                let k = cx.emit(ty, Op::Const(shift));
                Op::Binary(
                    Binary::Isub,
                    base,
                    cx.emit(ty, Op::Binary(Binary::Ishl, idx, k)),
                )
            })
            .or_else(|| sum::negated_offset_add(cx, cx.func.ty(a), a, b))
            .or_else(|| arith::reassociate_add(cx, fits_used, a, b))
            .or_else(|| shift::simplify_rotate(cx, bin, a, b))
            .or_else(|| sum::cancel_terms(cx, fits_used, res, (bin, a, b))),
        Binary::Isub => bits::low_bits(cx, a, b)
            .or_else(|| arith::reassociate_sub(cx, fits_used, a, b))
            .or_else(|| sum::cancel_terms(cx, fits_used, res, (bin, a, b))),
        Binary::Band => simplify_band(cx, res, a, b),
        Binary::Bor => shift::simplify_rotate(cx, bin, a, b)
            .or_else(|| icmp::fuse_cmp_ops(cx, res, false, a, b))
            .or_else(|| shift::merge_fields(cx, res, a, b)),
        Binary::Bxor => simplify_bxor(cx, a, b).or_else(|| shift::simplify_rotate(cx, bin, a, b)),
        Binary::Ushr | Binary::Ishl => shift::simplify_shift(cx, res, bin, a, b),
        Binary::Imul => sum::scale_sum(cx, a, b),
        Binary::Udiv | Binary::Urem => arith::simplify_div_rem(cx, block_udivs, res, (bin, a, b)),
        _ => None,
    };
    out.or_else(|| arith::fold_select_const_bin(cx, bin, a, b))
}

fn simplify_inst(
    cx: &mut RewriteCx,
    fits_used: &[bool],
    block_udivs: &mut Vec<(Value, Value, Value)>,
    res: Option<Value>,
    inst: Inst,
) -> Option<Op> {
    match cx.func.insts[inst.index()].op {
        Op::Icmp(cc, x, y) => simplify_icmp(cx, res, cc, x, y),
        Op::Binary(bin, a, b) => {
            if res.is_some_and(|r| cx.func.ty(r).is_vector()) {
                vec::simplify_vec_binary(cx, res, bin, a, b)
            } else if matches!(bin, Binary::Band | Binary::Bor | Binary::Bxor)
                && let Some(op) = vec::merge_vec_bitwise(cx, bin, a, b)
            {
                Some(op)
            } else {
                simplify_binary(cx, fits_used, block_udivs, res, (bin, a, b))
            }
        }
        Op::Convert(Convert::Bitcast, x) => vec::simplify_bitcast(cx, res, x),
        Op::Convert(kind, x) => simplify_convert(cx, res, kind, x),
        Op::Insert(v, x, lane) => vec::simplify_insert(cx, res, v, x, lane),
        Op::Extract(v, lane) => vec::simplify_extract(cx, res, v, lane),
        Op::Shuffle(a, b, ShuffleMask::Const(m)) => vec::simplify_shuffle(cx, res, a, b, m),
        Op::Fits(Binary::Imul, false, x, y) => ovf::simplify_mul_fits(cx, x, y),
        Op::Select(cond, t, f) => ovf::gated_product(cx, cond, t, f)
            .or_else(|| vec::byte_abs(cx, cond, t, f))
            .or_else(|| select::simplify_select(cx, res, cond, t, f)),
        Op::Brif(cond, tb, ref ta, eb, ref ea) => {
            let (ta, ea) = (ta.clone(), ea.clone());
            match select::tested_select(cx, cond) {
                Some((test, true)) => Some(Op::Brif(test, tb, ta, eb, ea)),
                Some((test, false)) => {
                    cx.func.bump_cfg();
                    Some(Op::Brif(test, eb, ea, tb, ta))
                }
                None => known_cond(cx, cond)
                    .or_else(|| icmp::strip_zero_test(cx, cond))
                    .or_else(|| {
                        let inner = cx.unwrap_uextend(cond);
                        (inner != cond).then_some(inner)
                    })
                    .map(|c| Op::Brif(c, tb, ta, eb, ea)),
            }
        }
        Op::Load(addr, off) => {
            let ty = cx.func.ty(res?);
            arith::fold_rodata_load(cx, ty, addr, off).or_else(|| {
                arith::fold_disp(cx, addr, off, ty.bytes()).map(|(base, disp)| Op::Load(base, disp))
            })
        }
        Op::Store(value, addr, off) => arith::fold_disp(cx, addr, off, cx.func.ty(value).bytes())
            .map(|(base, disp)| Op::Store(value, base, disp)),
        ref op @ (Op::Reduce(..) | Op::TargetOp(..)) => vec::simplify_movemask(cx, &op.clone()),
        _ => None,
    }
}

/// Rewrites constant unsigned divisions/remainders, unsigned zero compares, negated shift-adds, converts, shifts, selects, and bitwise/SWAR masks, returning whether anything changed.
pub(crate) fn rewrite(func: &mut Function, initial_defs: &[Option<Inst>]) -> bool {
    let len = func.types.len();
    let (mut consts, mut fits_used, mut uses) =
        (vec![None; len], vec![false; len], vec![0_u32; len]);
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        data.op.for_operands(|v| uses[v.index()] += 1);
        if let (Op::Const(imm), Some(res)) = (&data.op, data.result) {
            consts[res.index()] = Some(*imm);
        }
        if let Op::Fits(_, _, x, y) = data.op {
            fits_used[x.index()] = true;
            fits_used[y.index()] = true;
        }
    }
    let mut cx = RewriteCx {
        func,
        defs: initial_defs.to_vec(),
        consts,
        uses,
        out: Vec::new(),
        subst: Vec::new(),
        changed: false,
    };
    for b in 0..cx.func.blocks.len() {
        let old = std::mem::take(&mut cx.func.blocks[b].insts);
        cx.out = Vec::with_capacity(old.len());
        let mut block_udivs = Vec::new();
        for inst in old {
            if !cx.subst.is_empty() {
                cx.func.insts[inst.index()].op.map_operands(|v| {
                    let n = crate::merge::find(&cx.subst, v);
                    cx.changed |= n != v;
                    n
                });
            }
            let res = cx.func.insts[inst.index()].result;
            if let Some(new_op) = simplify_inst(&mut cx, &fits_used, &mut block_udivs, res, inst) {
                if let (Op::Const(imm), Some(r)) = (&new_op, res) {
                    cx.consts.resize(cx.func.types.len(), None);
                    cx.consts[r.index()] = Some(*imm);
                }
                cx.changed |= cx.func.insts[inst.index()].op != new_op;
                cx.func.insts[inst.index()].op = new_op;
            }
            cx.out.push(inst);
        }
        cx.func.blocks[b].insts = std::mem::take(&mut cx.out);
    }
    if !cx.subst.is_empty() {
        for inst in cx.func.blocks.iter().flat_map(|b| &b.insts) {
            cx.func.insts[inst.index()].op.map_operands(|v| {
                let n = crate::merge::find(&cx.subst, v);
                cx.changed |= n != v;
                n
            });
        }
    }
    cx.changed
}

/// Devirtualizes loads of static vtable slots into direct slot/symbol loads.
pub(crate) fn devirt_vtables(func: &mut Function, defs: &[Option<Inst>]) -> bool {
    if func.vtable_slots.is_empty() {
        return false;
    }
    let mut changed = false;
    for b in 0..func.blocks.len() {
        let mut i = 0;
        while i < func.blocks[b].insts.len() {
            let inst = func.blocks[b].insts[i];
            if let (Op::Load(addr, off), Some(res)) = (
                func.insts[inst.index()].op.clone(),
                func.insts[inst.index()].result,
            ) && func.ty(res) == Type::I64
                && let Some(def_inst) = defs.get(addr.index()).copied().flatten()
                && let Op::Symbol(vt_sym) = func.insts[def_inst.index()].op
                && let Some(&(_, _, slot_sym, low)) = func
                    .vtable_slots
                    .iter()
                    .find(|&&(v, o, _, _)| v == vt_sym && o == off)
            {
                let slot_val = func.value(Type::I64);
                let slot_inst = func.add(
                    if low {
                        Op::SlotAddr(slot_sym)
                    } else {
                        Op::Symbol(slot_sym)
                    },
                    Some(slot_val),
                );
                func.blocks[b].insts.insert(i, slot_inst);
                func.insts[inst.index()].op = Op::Load(slot_val, 0);
                changed = true;
                i += 2;
                continue;
            }
            i += 1;
        }
    }
    changed
}

/// Whether `value` is the floating-point constant `2.0`.
pub(crate) fn is_float_two(func: &Function, defs: &[Option<Inst>], value: Value) -> bool {
    let Some(inst) = defs.get(value.index()).copied().flatten() else {
        return false;
    };
    let Op::Const(imm) = func.insts[inst.index()].op else {
        return false;
    };
    match func.ty(value) {
        Type::F64 => imm as u64 == 0x4000_0000_0000_0000,
        Type::F32 => imm as u32 == 0x4000_0000,
        _ => false,
    }
}
