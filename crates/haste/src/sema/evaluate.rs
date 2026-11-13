//! Constant items and anonymous constants whose syntax the folder cannot evaluate, evaluated by running their bodies through the runner that the layer owning code memory installs.

use std::sync::OnceLock;

use anyhow::{Result, bail};

use crate::body::BinOp;

use super::Context;
use super::reach::Instance;
use super::ty::mask;

/// The width and signedness of an integer type.
pub(super) type Int = (u32, bool);

/// A method that constant expressions call: a pure method of the integer types, or the length of a string, byte string or array.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ConstMethod {
    Ilog,
    Ilog2,
    Ilog10,
    Pow,
    CountOnes,
    CountZeros,
    LeadingZeros,
    TrailingZeros,
    Len,
}

#[derive(Clone, Copy)]
pub(super) enum OverflowOp {
    Add,
    Sub,
    Mul,
    Shl,
    Shr,
}

pub(super) const OVERFLOW_OPS: [(&str, OverflowOp); 5] =
    [("overflowing_add", OverflowOp::Add), ("overflowing_sub", OverflowOp::Sub), ("overflowing_mul", OverflowOp::Mul), ("overflowing_shl", OverflowOp::Shl), ("overflowing_shr", OverflowOp::Shr)];

pub(super) const CONST_METHODS: [(&str, ConstMethod); 9] = [
    ("ilog", ConstMethod::Ilog),
    ("ilog2", ConstMethod::Ilog2),
    ("ilog10", ConstMethod::Ilog10),
    ("pow", ConstMethod::Pow),
    ("count_ones", ConstMethod::CountOnes),
    ("count_zeros", ConstMethod::CountZeros),
    ("leading_zeros", ConstMethod::LeadingZeros),
    ("trailing_zeros", ConstMethod::TrailingZeros),
    ("len", ConstMethod::Len),
];

/// What runs constant items: compiles the body of one and each function it calls, runs it, and gives its value.
pub trait Evaluate: Send {
    /// The value of the integer constant `instance`, `bits` wide, from running its body.
    fn evaluate(&mut self, cx: &mut Context<'_>, instance: &Instance, bits: u32) -> Result<u128>;
}

/// Makes a runner of constant items.
pub type Make = fn() -> Result<Box<dyn Evaluate>>;

static MAKE: OnceLock<Make> = OnceLock::new();

/// Installs `make` as the maker of runners of constant items, once, before any pass runs.
pub fn install(make: Make) {
    MAKE.get_or_init(|| make);
}

/// The value of the integer constant `instance`, `bits` wide, from running its body once per pass; a constant whose evaluation needs its own value is an error.
pub fn run(cx: &mut Context<'_>, instance: &Instance, bits: u32) -> Result<u128> {
    if let Some(value) = cx.store.evaluated.get(instance) {
        return Ok(value);
    }
    if let Some(err) = cx.eval_errors.get(instance) {
        bail!("{err}");
    }
    if cx.in_parallel {
        cx.deferred_evals.push((*instance, bits));
        bail!("constant evaluation deferred");
    }
    if cx.eval_stack.contains(instance) {
        bail!("the constant {} depends on itself", cx.name(instance.0));
    }
    cx.eval_stack.push(*instance);
    let mut runner = match cx.store.runner.take() {
        Some(runner) => runner,
        None => {
            let Some(make) = MAKE.get() else {
                cx.eval_stack.pop();
                bail!("no runner of constants is installed");
            };
            match make() {
                Ok(runner) => runner,
                Err(err) => {
                    cx.eval_stack.pop();
                    return Err(err);
                }
            }
        }
    };
    let result = runner.evaluate(cx, instance, bits);
    cx.store.runner.put(runner);
    cx.eval_stack.pop();
    match result {
        Ok(value) => {
            cx.store.evaluated.insert(*instance, value);
            Ok(value)
        }
        Err(err) => {
            cx.eval_errors.insert(*instance, format!("{err:#}"));
            Err(err)
        }
    }
}

pub(super) fn prime(cx: &mut Context<'_>, instance: &Instance, bits: u32) {
    match run(cx, instance, bits) {
        Ok(_) | Err(_) => {}
    }
}

fn ilog(value: u128, base: u128) -> u128 {
    let (mut value, mut log) = (value, 0);
    while value >= base {
        value /= base;
        log += 1;
    }
    log
}

pub(super) fn int_method(method: ConstMethod, x: u128, args: &[u128], int: Int) -> Result<(u128, Int)> {
    let arg = args.first().copied();
    match (method, arg) {
        (ConstMethod::Ilog | ConstMethod::Ilog2 | ConstMethod::Ilog10, _) => int_log(method, x, arg, int).map(|log| (log, (32, false))),
        (ConstMethod::Pow, Some(exp)) => Ok((x.wrapping_pow(exp as u32) & mask(int.0), int)),
        (ConstMethod::Pow, None) => bail!("`pow` in a constant has no exponent"),
        _ => Ok((int_bits(method, x & mask(int.0), int.0), (32, false))),
    }
}

fn int_log(method: ConstMethod, x: u128, base: Option<u128>, (bits, is_signed): Int) -> Result<u128> {
    if x == 0 || (is_signed && signed(x, bits) < 0) {
        bail!("the logarithm of a value that is not positive in a constant");
    }
    Ok(match (method, base) {
        (ConstMethod::Ilog2, _) => u128::from(127 - x.leading_zeros()),
        (ConstMethod::Ilog, Some(base)) if !(is_signed && signed(base, bits) < 2) && base >= 2 => ilog(x, base),
        (ConstMethod::Ilog, _) => bail!("a logarithm of base below 2 in a constant"),
        _ => ilog(x, 10),
    })
}

fn int_bits(method: ConstMethod, x: u128, bits: u32) -> u128 {
    u128::from(match method {
        ConstMethod::CountOnes => x.count_ones(),
        ConstMethod::CountZeros => bits - x.count_ones(),
        ConstMethod::LeadingZeros => x.leading_zeros() - (128 - bits),
        _ => x.trailing_zeros().min(bits),
    })
}

pub(super) fn signed(value: u128, bits: u32) -> i128 {
    let shift = 128 - bits;
    ((value << shift) as i128) >> shift
}

pub(super) fn convert(value: u128, from: Int, to: Int) -> u128 {
    let value = if from.1 { signed(value, from.0) as u128 } else { value };
    value & mask(to.0)
}

pub(super) fn fold(op: BinOp, x: u128, y: u128, (bits, is_signed): Int) -> Result<u128> {
    if matches!(op, BinOp::Div | BinOp::Rem) && y == 0 {
        bail!("a division by zero in a constant");
    }
    Ok(match op {
        BinOp::Div | BinOp::Rem | BinOp::Shr if is_signed => fold_signed(op, signed(x, bits), y, bits),
        BinOp::Div | BinOp::Rem | BinOp::Shr => fold_unsigned(op, x, y),
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::Shl => x.checked_shl(y as u32).unwrap_or(0),
        BinOp::BitAnd => x & y,
        BinOp::BitOr => x | y,
        BinOp::BitXor => x ^ y,
        other => bail!("the operator {other:?} in a constant is not evaluated yet"),
    })
}

fn fold_signed(op: BinOp, x: i128, y: u128, bits: u32) -> u128 {
    match op {
        BinOp::Div => x.wrapping_div(signed(y, bits)) as u128,
        BinOp::Rem => x.wrapping_rem(signed(y, bits)) as u128,
        _ => (x >> (y as u32).min(127)) as u128,
    }
}

fn fold_unsigned(op: BinOp, x: u128, y: u128) -> u128 {
    match op {
        BinOp::Div => x / y,
        BinOp::Rem => x % y,
        _ => x.checked_shr(y as u32).unwrap_or(0),
    }
}

pub(super) fn overflowing_op(op: OverflowOp, x: u128, y: u128, (bits, is_signed): Int) -> (u128, bool) {
    let (xm, ym, shift) = (x & mask(bits), y & mask(bits), (y & 0xffff_ffff) as u32);
    match op {
        OverflowOp::Shl => (x.wrapping_shl(shift % bits) & mask(bits), shift >= bits),
        OverflowOp::Shr if !is_signed => (xm.wrapping_shr(shift % bits), shift >= bits),
        OverflowOp::Shr => ((signed(x, bits).wrapping_shr(shift % bits) as u128) & mask(bits), shift >= bits),
        _ if is_signed => signed_overflow(op, x, y, bits),
        OverflowOp::Add => {
            let (sum, over128) = xm.overflowing_add(ym);
            (sum & mask(bits), if bits == 128 { over128 } else { sum > mask(bits) })
        }
        OverflowOp::Sub => (xm.wrapping_sub(ym) & mask(bits), xm < ym),
        OverflowOp::Mul => {
            let (prod, over) = xm.overflowing_mul(ym);
            (prod & mask(bits), over || (bits < 128 && prod > mask(bits)))
        }
    }
}

fn signed_overflow(op: OverflowOp, x: u128, y: u128, bits: u32) -> (u128, bool) {
    let (sx, sy) = (signed(x, bits), signed(y, bits));
    let (val, over) = match op {
        OverflowOp::Add => sx.overflowing_add(sy),
        OverflowOp::Sub => sx.overflowing_sub(sy),
        _ => sx.overflowing_mul(sy),
    };
    let masked = (val as u128) & mask(bits);
    (masked, over || signed(masked, bits) != val)
}
