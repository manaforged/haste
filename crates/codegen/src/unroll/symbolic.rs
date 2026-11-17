use crate::fold::evaluate_wrapped;
use crate::ir::{Binary, Function, IntCC, Op, Type, Value};

const TAG: i64 = 0x5eed;
const SHIFT: u32 = 48;
const SYM_BITS: u32 = 16;
const OFF_BITS: u32 = 32;

/// Whether `imm` stands for a symbol address plus an offset rather than a number.
pub(super) fn tagged(imm: i64) -> bool {
    imm >> SHIFT == TAG
}

fn tag(sym: u32, off: i64) -> Option<i64> {
    (sym < 1 << SYM_BITS && (0..1 << OFF_BITS).contains(&off)).then(|| (TAG << SHIFT) | (i64::from(sym) << OFF_BITS) | off)
}

fn split(imm: i64) -> (i64, i64) {
    ((imm >> OFF_BITS) & ((1 << SYM_BITS) - 1), imm & ((1 << OFF_BITS) - 1))
}

/// Evaluates `op` like `evaluate_wrapped`, and also tracks symbol addresses moved by constants so loops over read-only tables get a trip count.
pub(super) fn evaluate(func: &Function, op: &Op, ty: Type, known: &dyn Fn(Value) -> Option<i64>) -> Option<i64> {
    if let Op::Symbol(sym) = *op {
        return tag(sym, 0);
    }
    let mut pointer = false;
    op.for_operands(|value| pointer |= known(value).is_some_and(tagged));
    if pointer {
        return if ty == Type::I64 || matches!(op, Op::Icmp(..)) { address(op, known) } else { None };
    }
    evaluate_wrapped(func, op, ty, known).filter(|&imm| !tagged(imm))
}

fn address(op: &Op, known: &dyn Fn(Value) -> Option<i64>) -> Option<i64> {
    match *op {
        Op::Binary(Binary::Iadd, x, y) => {
            let (x, y) = (known(x)?, known(y)?);
            let (base, by) = match (tagged(x), tagged(y)) {
                (true, false) => (x, y),
                (false, true) => (y, x),
                _ => return None,
            };
            let (sym, off) = split(base);
            tag(sym as u32, off.checked_add(by)?)
        }
        Op::Binary(Binary::Isub, x, y) => {
            let (x, y) = (known(x)?, known(y)?);
            if !tagged(x) {
                return None;
            }
            let (sx, ox) = split(x);
            if !tagged(y) {
                return tag(sx as u32, ox.checked_sub(y)?);
            }
            let (sy, oy) = split(y);
            (sx == sy).then_some(ox - oy)
        }
        Op::Icmp(cc, x, y) => {
            let (x, y) = (known(x)?, known(y)?);
            let ((sx, ox), (sy, oy)) = (split(x), split(y));
            if !tagged(x) || !tagged(y) || sx != sy {
                return None;
            }
            let holds = match cc {
                IntCC::Equal => ox == oy,
                IntCC::NotEqual => ox != oy,
                IntCC::UnsignedLessThan => ox < oy,
                IntCC::UnsignedLessThanOrEqual => ox <= oy,
                IntCC::UnsignedGreaterThan => ox > oy,
                IntCC::UnsignedGreaterThanOrEqual => ox >= oy,
                _ => return None,
            };
            Some(i64::from(holds))
        }
        _ => None,
    }
}
