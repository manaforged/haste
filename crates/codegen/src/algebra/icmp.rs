use crate::fold::{extend, holds, known_bits};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::{RewriteCx, swap_cc};

const MAX_DEPTH: u8 = 4;

#[derive(Clone, Copy)]
enum Step {
    Conv(Convert, Type, Type),
    Mask(Type, u64),
}

#[derive(Clone, Copy)]
struct Cmp {
    cc: IntCC,
    ty: Type,
    k: i64,
}

fn stack_base(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, v: Value) -> bool {
    let inner = match def_of(v).map(|inst| &func.insts[inst.index()].op) {
        Some(Op::StackAddr(..)) => return true,
        Some(&Op::Convert(Convert::Bitcast, a)) if func.ty(a) == func.ty(v) => a,
        _ => return false,
    };
    matches!(def_of(inner).map(|inst| &func.insts[inst.index()].op), Some(Op::StackAddr(..)))
}

/// Returns whether `v` denotes a guaranteed non-null stack, symbol, or slot address.
pub(crate) fn is_non_null_addr(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    known: &dyn Fn(Value) -> Option<i64>,
    v: Value,
    depth: u8,
) -> bool {
    if depth >= MAX_DEPTH || func.ty(v).bits() < 32 {
        return false;
    }
    let Some(inst) = def_of(v) else { return false };
    let stack_disp = |base: Value, off: Value| {
        stack_base(func, def_of, base) && known(off).is_some_and(|c| (-0x1000_0000..=0x1000_0000).contains(&c))
    };
    match func.insts[inst.index()].op {
        Op::StackAddr(..) | Op::Symbol(_) | Op::SlotAddr(_) => true,
        Op::Binary(Binary::Iadd, a, b) => stack_disp(a, b) || stack_disp(b, a),
        Op::Binary(Binary::Isub, a, b) => stack_disp(a, b),
        Op::Select(_, a, b) => {
            is_non_null_addr(func, def_of, known, a, depth + 1)
                && is_non_null_addr(func, def_of, known, b, depth + 1)
        }
        Op::Convert(Convert::Bitcast, a) if func.ty(a) == func.ty(v) => {
            is_non_null_addr(func, def_of, known, a, depth + 1)
        }
        _ => false,
    }
}

/// Folds `Icmp(cc, x, y)` against `0` or `1` when one side is a non-null address.
pub(crate) fn non_null_cmp(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    known: &dyn Fn(Value) -> Option<i64>,
    (cc, x, y): (IntCC, Value, Value),
) -> Option<bool> {
    let ty = func.ty(x);
    if !ty.is_small_int() || ty.bits() < 32 {
        return None;
    }
    for (norm, val, rhs) in [(cc, x, y), (swap_cc(cc), y, x)] {
        let Some(imm) = known(rhs) else { continue };
        let u = extend(imm, ty, false) as u64;
        if !is_non_null_addr(func, def_of, known, val, 0) {
            continue;
        }
        match (norm, u) {
            (IntCC::Equal, 0) | (IntCC::UnsignedLessThanOrEqual, 0) | (IntCC::UnsignedLessThan, 1) => return Some(false),
            (IntCC::NotEqual, 0) | (IntCC::UnsignedGreaterThan, 0) | (IntCC::UnsignedGreaterThanOrEqual, 1) => {
                return Some(true);
            }
            _ => {}
        }
    }
    None
}

fn apply_step(c: i64, step: Step) -> i64 {
    match step {
        Step::Conv(Convert::Uextend, src, _) => extend(c, src, false),
        Step::Conv(Convert::Sextend, src, _) => extend(c, src, true),
        Step::Conv(Convert::Ireduce, _, dst) => extend(c, dst, false),
        Step::Conv(_, _, _) => c,
        Step::Mask(ty, mask) => extend(((c as u64) & mask) as i64, ty, false),
    }
}

fn apply_steps(mut c: i64, steps: &[Step]) -> i64 {
    for &step in steps.iter().rev() {
        c = apply_step(c, step);
    }
    c
}

fn step_bits((z, o): (u64, u64), step: Step) -> (u64, u64) {
    match step {
        Step::Conv(Convert::Uextend, src, dst) => {
            let (sw, dw) = (src.unsigned_max(), dst.unsigned_max());
            ((z | !sw) & dw, o & sw)
        }
        Step::Conv(Convert::Ireduce, _, dst) => {
            let dw = dst.unsigned_max();
            (z & dw, o & dw)
        }
        Step::Conv(Convert::Sextend, src, dst) => {
            let (sw, dw) = (src.unsigned_max(), dst.unsigned_max());
            let sign = 1_u64 << (src.bits() - 1);
            let ext = !sw & dw;
            let nz = if (z & sign) != 0 { z | ext } else { z & sw };
            let no = if (o & sign) != 0 { o | ext } else { o & sw };
            (nz & dw, no & dw)
        }
        Step::Conv(_, _, _) => (0, 0),
        Step::Mask(ty, mask) => {
            let w = ty.unsigned_max();
            ((z | !mask) & w, o & mask & w)
        }
    }
}

fn steps_preserve_non_null(src: Type, steps: &[Step]) -> bool {
    src.bits() >= 32
        && steps.iter().all(|s| match *s {
            Step::Conv(Convert::Uextend | Convert::Sextend, from, _) => from.bits() >= 32,
            Step::Mask(ty, mask) => mask >= ty.unsigned_max(),
            _ => false,
        })
}

fn peel_steps(cx: &RewriteCx, mut val: Value) -> (Value, Vec<Step>) {
    let mut steps = Vec::new();
    for _ in 0..MAX_DEPTH {
        let Some(inst) = cx.def(val) else { break };
        match cx.func.insts[inst.index()].op {
            Op::Convert(kind @ (Convert::Uextend | Convert::Sextend | Convert::Ireduce), inner)
                if cx.func.ty(inner).is_small_int() =>
            {
                steps.push(Step::Conv(kind, cx.func.ty(inner), cx.func.ty(val)));
                val = inner;
            }
            Op::Binary(Binary::Band, a, b) => {
                let ty = cx.func.ty(val);
                let masked = [(a, b), (b, a)]
                    .into_iter()
                    .find_map(|(v, m)| cx.uconst(m, ty).map(|mask| (v, mask)));
                let Some((inner, mask)) = masked else { break };
                steps.push(Step::Mask(ty, mask));
                val = inner;
            }
            _ => break,
        }
    }
    (val, steps)
}

fn leaf_holds(cx: &RewriteCx, v: Value, steps: &[Step], cmp: Cmp) -> Option<bool> {
    if let Some(imm) = cx.sconst(v) {
        return Some(holds(cmp.cc, cmp.ty, apply_steps(imm, steps), cmp.k));
    }
    let ty = cx.func.ty(v);
    if steps_preserve_non_null(ty, steps)
        && is_non_null_addr(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), v, 0)
    {
        let u = extend(cmp.k, cmp.ty, false) as u64;
        match (cmp.cc, u) {
            (IntCC::Equal, 0) | (IntCC::UnsignedLessThanOrEqual, 0) | (IntCC::UnsignedLessThan, 1) => return Some(false),
            (IntCC::NotEqual, 0) | (IntCC::UnsignedGreaterThan, 0) | (IntCC::UnsignedGreaterThanOrEqual, 1) => {
                return Some(true);
            }
            _ => {}
        }
    }
    if matches!(cmp.cc, IntCC::Equal | IntCC::NotEqual) {
        let mut bits = known_bits(cx.func, &|x| cx.def(x), &|x| cx.sconst(x), v, 0);
        for &step in steps.iter().rev() {
            bits = step_bits(bits, step);
        }
        let u = (extend(cmp.k, cmp.ty, false) as u64) & cmp.ty.unsigned_max();
        if (u & bits.0) != 0 || (!u & bits.1) != 0 {
            return Some(cmp.cc == IntCC::NotEqual);
        }
    }
    None
}

fn arm_holds(cx: &RewriteCx, v: Value, steps: &mut Vec<Step>, cmp: Cmp, budget: &mut u8) -> Option<bool> {
    if *budget == 0 || steps.len() > 8 {
        return None;
    }
    *budget -= 1;
    if let Some(bit) = leaf_holds(cx, v, steps, cmp) {
        return Some(bit);
    }
    let inst = cx.def(v)?;
    match cx.func.insts[inst.index()].op {
        Op::Convert(kind @ (Convert::Uextend | Convert::Sextend | Convert::Ireduce), inner)
            if cx.func.ty(inner).is_small_int() =>
        {
            steps.push(Step::Conv(kind, cx.func.ty(inner), cx.func.ty(v)));
            let out = arm_holds(cx, inner, steps, cmp, budget);
            steps.pop();
            out
        }
        Op::Binary(Binary::Band, a, b) => {
            let ty = cx.func.ty(v);
            let (inner, mask) = [(a, b), (b, a)]
                .into_iter()
                .find_map(|(u, m)| cx.uconst(m, ty).map(|mask| (u, mask)))?;
            steps.push(Step::Mask(ty, mask));
            let out = arm_holds(cx, inner, steps, cmp, budget);
            steps.pop();
            out
        }
        Op::Select(cond, a, b) if cx.func.ty(cond) == Type::I8 => {
            let ha = arm_holds(cx, a, steps, cmp, budget)?;
            let hb = arm_holds(cx, b, steps, cmp, budget)?;
            (ha == hb).then_some(ha)
        }
        _ => None,
    }
}

fn cc_mask(cc: IntCC) -> u8 {
    match cc {
        IntCC::Equal => 0b010,
        IntCC::NotEqual => 0b101,
        IntCC::SignedLessThan | IntCC::UnsignedLessThan => 0b001,
        IntCC::SignedLessThanOrEqual | IntCC::UnsignedLessThanOrEqual => 0b011,
        IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => 0b100,
        IntCC::SignedGreaterThanOrEqual | IntCC::UnsignedGreaterThanOrEqual => 0b110,
    }
}

fn merge_icmps(cx: &RewriteCx, is_and: bool, (v1, neg1): (Value, bool), (v2, neg2): (Value, bool)) -> Option<Op> {
    let Op::Icmp(raw1, x1, y1) = cx.func.insts[cx.def(v1)?.index()].op else { return None };
    let Op::Icmp(raw2, x2, y2) = cx.func.insts[cx.def(v2)?.index()].op else { return None };
    let cc1 = if neg1 { raw1.inverse() } else { raw1 };
    let cc2 = if neg2 { raw2.inverse() } else { raw2 };
    let aligned2 = if (x2, y2) == (x1, y1) {
        cc2
    } else if (y2, x2) == (x1, y1) {
        swap_cc(cc2)
    } else {
        return None;
    };
    let cmp_cc = |c: IntCC| !matches!(c, IntCC::Equal | IntCC::NotEqual);
    if cmp_cc(cc1) && cmp_cc(aligned2) && cc1.is_signed() != aligned2.is_signed() {
        return None;
    }
    let signed = (cmp_cc(cc1) && cc1.is_signed()) || (cmp_cc(aligned2) && aligned2.is_signed());
    let mask = if is_and { cc_mask(cc1) & cc_mask(aligned2) } else { cc_mask(cc1) | cc_mask(aligned2) };
    let out = match mask {
        0b000 => return Some(Op::Const(0)),
        0b111 => return Some(Op::Const(1)),
        0b010 => IntCC::Equal,
        0b101 => IntCC::NotEqual,
        0b001 if signed => IntCC::SignedLessThan,
        0b001 => IntCC::UnsignedLessThan,
        0b011 if signed => IntCC::SignedLessThanOrEqual,
        0b011 => IntCC::UnsignedLessThanOrEqual,
        0b100 if signed => IntCC::SignedGreaterThan,
        0b100 => IntCC::UnsignedGreaterThan,
        0b110 if signed => IntCC::SignedGreaterThanOrEqual,
        0b110 => IntCC::UnsignedGreaterThanOrEqual,
        _ => return None,
    };
    Some(Op::Icmp(out, x1, y1))
}

fn emit_combo(cx: &mut RewriteCx, res: Option<Value>, is_and: bool, a: (Value, bool), b: (Value, bool)) -> Option<Op> {
    if a.0 == b.0 {
        return match (is_and, a.1 == b.1) {
            (true, false) => Some(Op::Const(0)),
            (false, false) => Some(Op::Const(1)),
            (_, true) if !a.1 => {
                cx.alias(res, a.0);
                None
            }
            (_, true) => {
                let one = cx.emit(Type::I8, Op::Const(1));
                Some(Op::Binary(Binary::Bxor, a.0, one))
            }
        };
    }
    if let Some(op) = merge_icmps(cx, is_and, a, b) {
        return Some(op);
    }
    let mut lit = |(v, neg): (Value, bool)| {
        if neg {
            let one = cx.emit(Type::I8, Op::Const(1));
            cx.emit(Type::I8, Op::Binary(Binary::Bxor, v, one))
        } else {
            v
        }
    };
    let (lhs, rhs) = (lit(a), lit(b));
    Some(Op::Binary(if is_and { Binary::Band } else { Binary::Bor }, lhs, rhs))
}

fn eval_mixed(
    cx: &mut RewriteCx,
    res: Option<Value>,
    (c1, outer_arm, c1_is_then): (Value, bool, bool),
    arm: Value,
    (steps, cmp): (&[Step], Cmp),
) -> Option<Option<Op>> {
    let (inner, extra) = peel_steps(cx, arm);
    let inst = cx.def(inner)?;
    let Op::Select(c2, a2, b2) = cx.func.insts[inst.index()].op else { return None };
    if cx.func.ty(c2) != Type::I8 || !cx.is_unit_bool(c2) {
        return None;
    }
    let mut combined = Vec::with_capacity(steps.len() + extra.len());
    combined.extend_from_slice(steps);
    combined.extend_from_slice(&extra);
    let (mut b_a, mut b_b) = (16, 16);
    let ha = arm_holds(cx, a2, &mut combined, cmp, &mut b_a)?;
    let hb = arm_holds(cx, b2, &mut combined, cmp, &mut b_b)?;
    if ha == hb {
        return None;
    }
    let is_and = !outer_arm;
    let neg1 = c1_is_then ^ outer_arm;
    let neg2 = !ha;
    Some(emit_combo(cx, res, is_and, (c1, neg1), (c2, neg2)))
}

fn eval_one(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, val: Value, k: i64) -> Option<Option<Op>> {
    let cmp = Cmp { cc, ty: cx.func.ty(val), k };
    let (core, mut steps) = peel_steps(cx, val);
    let inst = cx.def(core)?;
    let Op::Select(cond, a, b) = cx.func.insts[inst.index()].op else {
        return None;
    };
    let (mut ba, mut bb) = (16, 16);
    let ha = arm_holds(cx, a, &mut steps, cmp, &mut ba);
    let hb = arm_holds(cx, b, &mut steps, cmp, &mut bb);
    if let (Some(ta), Some(tb)) = (ha, hb) && ta == tb {
        return Some(Some(Op::Const(i64::from(ta))));
    }
    if cx.func.ty(cond) != Type::I8 || !cx.is_unit_bool(cond) {
        return None;
    }
    match (ha, hb) {
        (Some(true), Some(false)) => {
            cx.alias(res, cond);
            Some(None)
        }
        (Some(false), Some(true)) => {
            let one = cx.emit(Type::I8, Op::Const(1));
            Some(Some(Op::Binary(Binary::Bxor, cond, one)))
        }
        (Some(ta), None) => eval_mixed(cx, res, (cond, ta, true), b, (&steps, cmp)),
        (None, Some(tb)) => eval_mixed(cx, res, (cond, tb, false), a, (&steps, cmp)),
        _ => None,
    }
}

pub(super) fn eval_select(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Option<Op>> {
    if !cx.func.ty(x).is_small_int() || !res.is_some_and(|r| cx.func.ty(r) == Type::I8) {
        return None;
    }
    if let Some(bit) = non_null_cmp(cx.func, &|v| cx.def(v), &|v| cx.sconst(v), (cc, x, y)) {
        return Some(Some(Op::Const(i64::from(bit))));
    }
    for (norm, val, c_val) in [(cc, x, y), (swap_cc(cc), y, x)] {
        if let Some(k) = cx.sconst(c_val)
            && let Some(out) = eval_one(cx, res, norm, val, k)
        {
            return Some(out);
        }
    }
    None
}
