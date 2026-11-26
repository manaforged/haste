use super::super::asm::Reg;
use super::bits::peel_elided;
use super::const_of;
use crate::alloc::Plan;
use crate::ir::{Function, Inst, IntCC, Value};

pub(super) fn sign_bit_raw(
    func: &Function,
    plan: &Plan<Reg>,
    (cc, lhs, rhs): (IntCC, Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, u8, bool)> {
    let ty = func.ty(lhs);
    if !ty.is_small_int() || !cc.is_signed() {
        return None;
    }
    let set = match (
        cc,
        crate::fold::extend(const_of(func, plan, rhs, def)?, ty, true),
    ) {
        (IntCC::SignedLessThan, 0) | (IntCC::SignedLessThanOrEqual, -1) => true,
        (IntCC::SignedGreaterThanOrEqual, 0) | (IntCC::SignedGreaterThan, -1) => false,
        _ => return None,
    };
    Some((lhs, (ty.bits() - 1) as u8, set))
}

pub(in crate::arm64) fn sign_bit_test(
    func: &Function,
    plan: &Plan<Reg>,
    cmp: (IntCC, Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, u8, bool)> {
    let (val, bit, set) = sign_bit_raw(func, plan, cmp, def)?;
    let (root, bit) = peel_elided(func, plan, (val, bit), def)?;
    Some((root, bit, set))
}
