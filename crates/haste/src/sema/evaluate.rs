//! Constant items and anonymous constants whose syntax the folder cannot evaluate, evaluated by running their bodies through the runner that the layer owning code memory installs.

use std::sync::OnceLock;

use anyhow::{Result, bail};

use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, PatId, Stmt};

use super::Context;
use super::infer::Res;
use super::reach::Instance;
use super::substs::Substs;
use super::ty::mask;

/// The width and signedness of an integer type.
pub(super) type Int = (u32, bool);

/// A compile-time constant arithmetic failure (overflow, division by zero, or invalid logarithm).
#[derive(Debug)]
pub(super) struct ConstArithmeticError(pub(super) String);

impl std::fmt::Display for ConstArithmeticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConstArithmeticError {}

pub(super) fn is_const_arithmetic_error(err: &anyhow::Error) -> bool {
    err.downcast_ref::<ConstArithmeticError>().is_some()
}

pub(super) fn const_arithmetic_error(msg: impl Into<String>) -> anyhow::Error {
    ConstArithmeticError(msg.into()).into()
}

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

pub(super) const OVERFLOW_OPS: [(&str, OverflowOp); 5] = [
    ("overflowing_add", OverflowOp::Add),
    ("overflowing_sub", OverflowOp::Sub),
    ("overflowing_mul", OverflowOp::Mul),
    ("overflowing_shl", OverflowOp::Shl),
    ("overflowing_shr", OverflowOp::Shr),
];

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

#[derive(Default)]
struct ConstCycleState {
    visiting: Vec<(Instance, bool)>,
    closed: Vec<Instance>,
    min_back: usize,
}

thread_local! {
    static CONST_CYCLE: std::cell::RefCell<ConstCycleState> = const {
        std::cell::RefCell::new(ConstCycleState {
            visiting: Vec::new(),
            closed: Vec::new(),
            min_back: usize::MAX,
        })
    };
}

pub(super) struct ConstCycleFrame {
    instance: Instance,
    depth: usize,
    saved_min: usize,
}

impl Drop for ConstCycleFrame {
    fn drop(&mut self) {
        CONST_CYCLE.with(|cell| {
            let mut st = cell.borrow_mut();
            st.visiting.pop();
            let child_min = st.min_back;
            st.min_back = self.saved_min.min(child_min);
            if child_min >= self.depth && !st.closed.contains(&self.instance) {
                st.closed.push(self.instance);
            }
            if st.visiting.is_empty() {
                st.closed.clear();
                st.min_back = usize::MAX;
            }
        });
    }
}

pub(super) fn const_cycle_active() -> bool {
    CONST_CYCLE.with(|cell| !cell.borrow().visiting.is_empty())
}

pub(super) fn const_cycle_top() -> Option<Instance> {
    CONST_CYCLE.with(|cell| cell.borrow().visiting.last().map(|&(it, _)| it))
}

pub(super) fn enter_const_cycle(
    cx: &Context<'_>,
    instance: Instance,
    is_const: bool,
) -> std::result::Result<Option<ConstCycleFrame>, String> {
    CONST_CYCLE.with(|cell| {
        let mut st = cell.borrow_mut();
        if let Some(pos) = st.visiting.iter().position(|&(it, _)| it == instance) {
            if is_const {
                return Err(format!(
                    "cycle detected when evaluating `{}`",
                    cx.name(instance.0)
                ));
            }
            if let Some(&(bad, _)) = st.visiting[pos + 1..].iter().find(|(_, c)| *c) {
                return Err(format!(
                    "cycle detected when evaluating `{}`",
                    cx.name(bad.0)
                ));
            }
            st.min_back = st.min_back.min(pos);
            return Ok(None);
        }
        if st.closed.contains(&instance) || st.visiting.len() >= crate::RECURSION_LIMIT {
            return Ok(None);
        }
        let depth = st.visiting.len();
        let saved_min = std::mem::replace(&mut st.min_back, usize::MAX);
        st.visiting.push((instance, is_const));
        Ok(Some(ConstCycleFrame {
            instance,
            depth,
            saved_min,
        }))
    })
}

pub(super) fn collect_body_const_deps(
    cx: &Context<'_>,
    body: &Body,
    get_res: impl Fn(ExprId) -> Option<Res>,
    get_op: impl Fn(ExprId) -> Option<Res>,
    get_pat: impl Fn(PatId) -> Option<Res>,
) -> Vec<(Option<ExprId>, crate::names::ItemRef, Substs)> {
    let (exprs, pats) = active_const_nodes(body);
    let mut out = Vec::new();
    for id in exprs {
        let res = match *body.expr(id) {
            Expr::Path(_) => get_res(id).filter(
                |r| matches!(r, Res::Fn(item, _) if cx.item(*item).1.kind == crate::outline::item::ItemKind::Const),
            ),
            Expr::Call(callee, _) => get_res(id).or_else(|| get_res(callee)),
            Expr::Method { .. } => get_res(id),
            Expr::Unary(..) | Expr::Binary(..) | Expr::AssignOp(..) => get_op(id),
            _ => None,
        };
        if let Some(Res::Fn(item, substs)) = res {
            out.push((Some(id), item, substs));
        }
    }
    for pat in pats {
        if let Some(Res::Fn(item, substs)) = get_pat(pat)
            && cx.item(item).1.kind == crate::outline::item::ItemKind::Const
        {
            out.push((None, item, substs));
        }
    }
    out
}

fn active_const_nodes(body: &Body) -> (Vec<ExprId>, Vec<PatId>) {
    let Some(root) = body.root else {
        return (Vec::new(), Vec::new());
    };
    let mut stack = vec![root];
    let (mut exprs, mut pats) = (Vec::new(), Vec::new());
    while let Some(id) = stack.pop() {
        exprs.push(id);
        push_active_children(body, id, &mut stack, &mut pats);
    }
    exprs.sort_by_key(|id| id.0);
    pats.sort_by_key(|id| id.0);
    (exprs, pats)
}

fn push_active_children(body: &Body, id: ExprId, stack: &mut Vec<ExprId>, pats: &mut Vec<PatId>) {
    match *body.expr(id) {
        Expr::Unary(_, a)
        | Expr::Field(a, _)
        | Expr::TupleField(a, _)
        | Expr::Cast(a, _)
        | Expr::Ref { expr: a, .. }
        | Expr::Loop { body: a, .. }
        | Expr::Try(a)
        | Expr::Await(a)
        | Expr::Become(a)
        | Expr::Paren(a) => stack.push(a),
        Expr::Return(a) | Expr::Break(_, a) | Expr::Yield(a) => stack.extend(a),
        Expr::Binary(_, a, b)
        | Expr::Assign(a, b)
        | Expr::AssignOp(_, a, b)
        | Expr::Index(a, b)
        | Expr::Repeat(a, b)
        | Expr::While {
            cond: a, body: b, ..
        } => stack.extend([a, b]),
        Expr::Let(pat, expr) => {
            pats.push(pat);
            stack.push(expr);
        }
        Expr::For {
            pat, iter, body: b, ..
        } => {
            pats.push(pat);
            stack.extend([iter, b]);
        }
        Expr::If(a, b, c) => {
            stack.extend([a, b]);
            stack.extend(c);
        }
        Expr::Range { from, to, .. } => stack.extend(from.into_iter().chain(to)),
        Expr::Call(callee, args) => {
            stack.push(callee);
            stack.extend(body.list(args).iter().map(|&x| ExprId(x)));
        }
        Expr::Method {
            receiver, params, ..
        } => {
            stack.push(receiver);
            stack.extend(body.list(params).iter().map(|&x| ExprId(x)));
        }
        Expr::Tuple(list) | Expr::Array(list) => {
            stack.extend(body.list(list).iter().map(|&x| ExprId(x)));
        }
        Expr::Struct { fields, base, .. } => {
            stack.extend(base);
            stack.extend(
                body.list(fields)
                    .iter()
                    .map(|&idx| body.fields[idx as usize].value),
            );
        }
        Expr::Match(scrut, arms) => {
            stack.push(scrut);
            for &idx in body.list(arms) {
                let arm = body.arms[idx as usize];
                pats.push(arm.pat);
                stack.extend(arm.guard);
                stack.push(arm.body);
            }
        }
        Expr::Block {
            stmts, tail, kind, ..
        } if !matches!(
            kind,
            BlockKind::Async | BlockKind::AsyncMove | BlockKind::Gen
        ) =>
        {
            stack.extend(tail);
            for &stmt_id in body.list(stmts) {
                match body.stmts[stmt_id as usize] {
                    Stmt::Let {
                        pat,
                        init,
                        otherwise,
                        ..
                    } => {
                        pats.push(pat);
                        stack.extend(init);
                        stack.extend(otherwise);
                    }
                    Stmt::Expr(expr, _) => stack.push(expr),
                    Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
                }
            }
        }
        _ => {}
    }
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
        let msg = format!("the constant {} depends on itself", cx.name(instance.0));
        cx.eval_errors.insert(*instance, msg.clone());
        return Err(const_arithmetic_error(msg));
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
    let prev_skip = std::mem::replace(&mut cx.store.lookups.skip_warm_sysroot_bodies, false);
    let result = runner.evaluate(cx, instance, bits);
    cx.store.lookups.skip_warm_sysroot_bodies = prev_skip;
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

pub(crate) fn prime(cx: &mut Context<'_>, instance: &Instance, bits: u32) {
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

const SHIFT_OVERFLOW: &str = "attempt to shift with overflow in a constant";

pub(super) fn is_signed_min(value: u128, bits: u32) -> bool {
    bits > 0 && (value & mask(bits)) == (1_u128 << (bits - 1))
}

pub(super) fn int_method(
    method: ConstMethod,
    x: u128,
    args: &[u128],
    int: Int,
) -> Result<(u128, Int)> {
    let arg = args.first().copied();
    match (method, arg) {
        (ConstMethod::Ilog | ConstMethod::Ilog2 | ConstMethod::Ilog10, _) => {
            int_log(method, x, arg, int).map(|log| (log, (32, false)))
        }
        (ConstMethod::Pow, Some(exp)) => checked_pow(x, exp, int).map(|val| (val, int)),
        (ConstMethod::Pow, None) => bail!("`pow` in a constant has no exponent"),
        _ => Ok((int_bits(method, x & mask(int.0), int.0), (32, false))),
    }
}

fn checked_pow(x: u128, exp: u128, (bits, is_signed): Int) -> Result<u128> {
    let overflow = || const_arithmetic_error("attempt to multiply with overflow in a constant");
    let exp = u32::try_from(exp).map_err(|_| overflow())?;
    if is_signed {
        let sx = signed(x, bits);
        let val = sx.checked_pow(exp).ok_or_else(overflow)?;
        let masked = (val as u128) & mask(bits);
        if signed(masked, bits) != val {
            return Err(overflow());
        }
        Ok(masked)
    } else {
        let val = (x & mask(bits)).checked_pow(exp).ok_or_else(overflow)?;
        if bits < 128 && val > mask(bits) {
            return Err(overflow());
        }
        Ok(val & mask(bits))
    }
}

fn int_log(
    method: ConstMethod,
    x: u128,
    base: Option<u128>,
    (bits, is_signed): Int,
) -> Result<u128> {
    if x == 0 || (is_signed && signed(x, bits) < 0) {
        return Err(const_arithmetic_error(
            "the logarithm of a value that is not positive in a constant",
        ));
    }
    Ok(match (method, base) {
        (ConstMethod::Ilog2, _) => u128::from(127 - x.leading_zeros()),
        (ConstMethod::Ilog, Some(base)) if !(is_signed && signed(base, bits) < 2) && base >= 2 => {
            ilog(x, base)
        }
        (ConstMethod::Ilog, _) => {
            return Err(const_arithmetic_error(
                "a logarithm of base below 2 in a constant",
            ));
        }
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
    let value = if from.1 {
        signed(value, from.0) as u128
    } else {
        value
    };
    value & mask(to.0)
}

pub(super) fn fold(op: BinOp, x: u128, y: u128, (bits, is_signed): Int) -> Result<u128> {
    if matches!(op, BinOp::Div | BinOp::Rem) {
        if (y & mask(bits)) == 0 {
            return Err(const_arithmetic_error(
                "attempt to divide by zero in a constant",
            ));
        }
        if is_signed && is_signed_min(x, bits) && signed(y, bits) == -1 {
            let msg = if op == BinOp::Div {
                "attempt to divide with overflow in a constant"
            } else {
                "attempt to calculate the remainder with overflow in a constant"
            };
            return Err(const_arithmetic_error(msg));
        }
        return Ok(if is_signed {
            fold_signed(op, signed(x, bits), y, bits)
        } else {
            fold_unsigned(op, x & mask(bits), y & mask(bits))
        });
    }
    if matches!(op, BinOp::Shl | BinOp::Shr) {
        if y >= u128::from(bits) {
            return Err(const_arithmetic_error(SHIFT_OVERFLOW));
        }
        let ov = if op == BinOp::Shl {
            OverflowOp::Shl
        } else {
            OverflowOp::Shr
        };
        let (val, over) = overflowing_op(ov, x, y, (bits, is_signed));
        if over {
            return Err(const_arithmetic_error(SHIFT_OVERFLOW));
        }
        return Ok(val);
    }
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul => {
            let ov = match op {
                BinOp::Add => OverflowOp::Add,
                BinOp::Sub => OverflowOp::Sub,
                _ => OverflowOp::Mul,
            };
            let (val, over) = overflowing_op(ov, x, y, (bits, is_signed));
            if over {
                return Err(const_arithmetic_error("arithmetic overflow in a constant"));
            }
            Ok(val)
        }
        BinOp::BitAnd => Ok(x & y),
        BinOp::BitOr => Ok(x | y),
        BinOp::BitXor => Ok(x ^ y),
        other => bail!("the operator {other:?} in a constant is not evaluated yet"),
    }
}

fn fold_signed(op: BinOp, x: i128, y: u128, bits: u32) -> u128 {
    match op {
        BinOp::Div => x.wrapping_div(signed(y, bits)) as u128,
        _ => x.wrapping_rem(signed(y, bits)) as u128,
    }
}

fn fold_unsigned(op: BinOp, x: u128, y: u128) -> u128 {
    match op {
        BinOp::Div => x / y,
        _ => x % y,
    }
}

pub(super) fn overflowing_op(
    op: OverflowOp,
    x: u128,
    y: u128,
    (bits, is_signed): Int,
) -> (u128, bool) {
    let (xm, ym, shift) = (x & mask(bits), y & mask(bits), (y & 0xffff_ffff) as u32);
    match op {
        OverflowOp::Shl => (x.wrapping_shl(shift % bits) & mask(bits), shift >= bits),
        OverflowOp::Shr if !is_signed => (xm.wrapping_shr(shift % bits), shift >= bits),
        OverflowOp::Shr => (
            (signed(x, bits).wrapping_shr(shift % bits) as u128) & mask(bits),
            shift >= bits,
        ),
        _ if is_signed => signed_overflow(op, x, y, bits),
        OverflowOp::Add => {
            let (sum, over128) = xm.overflowing_add(ym);
            (
                sum & mask(bits),
                if bits == 128 {
                    over128
                } else {
                    sum > mask(bits)
                },
            )
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
