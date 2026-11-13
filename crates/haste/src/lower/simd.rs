use anyhow::{Result, bail};
use haste_codegen::{Arch, FloatCC, IntCC, Lane, ReduceOp, StackSlot, Target, Type, Value};

use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, Lit};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::infer::{Res, SiteKey, parse_int};
use crate::sema::intrinsic::Simd;
use crate::sema::ty::{Kind, Ty};

use super::ops::{Num, compare};
use super::reloc::SiteOrigin;
use super::{Lower, machine, repr};

mod fallback;
mod lanes;
mod scalar;
mod target;

#[derive(Clone, Copy)]
pub(super) struct Lanes {
    pub(super) num: Num,
    pub(super) machine: Type,
    pub(super) size: i32,
    pub(super) count: i32,
}

impl Lanes {
    pub(super) fn bytes(self) -> i32 {
        self.size * self.count
    }

    pub(super) fn half(self) -> Self {
        Self { count: self.count / 2, ..self }
    }

    pub(super) fn align_shift(self) -> u8 {
        if self.bytes() >= 32 {
            5
        } else if self.bytes() >= 16 {
            4
        } else {
            3
        }
    }

    pub(super) fn widened(self) -> Option<(Self, Type)> {
        if self.bytes() >= 8 || self.count <= 0 || !(self.count as u32).is_power_of_two() || !matches!(Target::host().arch, Arch::X64 | Arch::Arm64) {
            return None;
        }
        let w = Self { count: 8 / self.size, ..self };
        vec_type(&w).map(|ty| (w, ty))
    }
}

pub(super) fn vec_type(lanes: &Lanes) -> Option<Type> {
    if !matches!(Target::host().arch, Arch::X64 | Arch::Arm64) {
        return None;
    }
    let lane = Lane::from_type(lanes.machine)?;
    let count = u8::try_from(lanes.count).ok()?;
    let ty = Type::Vec { lane, count };
    ty.is_valid_vec().then_some(ty)
}

pub(super) fn vec_for_bytes(bytes: i32) -> Option<Type> {
    match bytes {
        16 => Some(Type::Vec { lane: Lane::I8, count: 16 }),
        8 => Some(Type::Vec { lane: Lane::I8, count: 8 }),
        _ => None,
    }
}

pub(super) fn int_for_bytes(bytes: i32) -> Option<Type> {
    match bytes {
        1 => Some(Type::I8),
        2 => Some(Type::I16),
        4 => Some(Type::I32),
        8 => Some(Type::I64),
        _ => None,
    }
}

impl Lower<'_, '_> {
    /// Emits the SIMD intrinsic `op` using native `Type::Vec` operations, recursive half-splits, or SSA loops.
    pub(super) fn simd(&mut self, origin: &SiteOrigin, op: Simd, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let (Some(&vector), Some(&first)) = (substs.first(), args.first()) else { bail!("a SIMD intrinsic with no vector") };
        let out = substs.get(1).copied();
        if op == Simd::SelectBitmask {
            let &[mask, yes, no] = args else { bail!("SIMD intrinsic {op:?} takes other arguments") };
            return self.bitmask_select((vector, out.unwrap_or(vector)), mask, (yes, no)).map(Some);
        }
        if let Some(value) = self.vec_simd(origin, op, (vector, out.unwrap_or(vector)), substs, args)? {
            return Ok(Some(value));
        }
        if matches!(op, Simd::Unary(_) | Simd::Pair(_) | Simd::Fma | Simd::Sum { .. } | Simd::MaskedLoad | Simd::MaskedStore) {
            return self.lane_intrinsic(op, substs, args);
        }
        if let Some(value) = self.packed_swar64(op, (vector, out.unwrap_or(vector)), args)? {
            return Ok(Some(value));
        }
        if !matches!(op, Simd::Extract | Simd::Insert | Simd::Shuffle) {
            let from = self.lanes(vector)?;
            let to = match op {
                Simd::Bitmask => {
                    let Some(out_ty) = out else { bail!("`simd_bitmask` has no output type") };
                    let Some(target) = machine(self.cx, out_ty)? else { bail!("a SIMD bitmask into an array is not supported yet") };
                    Lanes { num: Num::Int { ty: target, signed: false }, machine: target, size: target.bytes() as i32, count: 1 }
                }
                Simd::Lanewise(_) | Simd::Cast | Simd::Select => self.lanes(out.unwrap_or(vector))?,
                _ => from,
            };
            if let Some(value) = self.wide_simd(op, (&from, &to), args)? {
                return Ok(Some(value));
            }
        }
        let value = match (op, out) {
            (Simd::Splat, _) => self.splat(vector, first)?,
            (Simd::Bitmask, Some(out)) => self.bitmask(vector, first, out)?,
            (Simd::Cast, Some(out)) => self.lane_cast((vector, out), first)?,
            (Simd::Reduce { all }, _) => self.reduce(vector, first, all)?,
            (Simd::Extreme { max }, _) => self.extreme(vector, first, max)?,
            (Simd::Fold(op), _) => self.fold(vector, first, op)?,
            (Simd::Sqrt, _) => self.lane_sqrt(vector, first)?,
            _ => self.simd_lanes(op, substs, args)?,
        };
        Ok(Some(value))
    }

    pub(super) fn store_vec(&mut self, vec: Value, bytes: u32) -> Value {
        let slot = self.b.create_stack_slot(bytes, if bytes <= 8 { 3 } else { 4 });
        self.ins().stack_store(vec, slot, 0);
        self.ins().stack_addr(Type::I64, slot, 0)
    }

    pub(super) fn bitcast_vec(&mut self, to: Type, val: Value) -> Value {
        if self.b.func.ty(val) == to {
            return val;
        }
        let bytes = to.bytes();
        let slot = self.b.create_stack_slot(bytes, bytes.trailing_zeros() as u8);
        self.ins().stack_store(val, slot, 0);
        self.ins().stack_load(to, slot, 0)
    }

    pub(super) fn lanes_slot(&mut self, lanes: Lanes) -> (StackSlot, Value) {
        let slot = self.b.create_stack_slot(lanes.bytes() as u32, lanes.align_shift());
        let addr = self.ins().stack_addr(Type::I64, slot, 0);
        (slot, addr)
    }

    fn vec_simd(&mut self, origin: &SiteOrigin, op: Simd, (input, output): (Ty, Ty), substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let lanes = self.lanes(input)?;
        let out_lanes = match op {
            Simd::Bitmask => {
                let Some(target) = machine(self.cx, output)? else { return Ok(None) };
                Lanes { num: Num::Int { ty: target, signed: false }, machine: target, size: target.bytes() as i32, count: 1 }
            }
            Simd::Lanewise(_) | Simd::Cast | Simd::Select => self.lanes(output)?,
            _ => lanes,
        };
        self.vec_simd_lanes(Some(origin), op, (lanes, out_lanes), substs, args)
    }

    pub(super) fn vec_simd_lanes(&mut self, origin: Option<&SiteOrigin>, op: Simd, (lanes, out_lanes): (Lanes, Lanes), substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        if op == Simd::Select {
            return self.vec_select((&lanes, &out_lanes), args);
        }
        let Some(vec_ty) = vec_type(&lanes) else { return Ok(None) };
        match (op, args) {
            (Simd::Splat, &[x]) => {
                let v = self.ins().splat(vec_ty, x);
                Ok(Some(self.store_vec(v, vec_ty.bytes())))
            }
            (Simd::Bitmask, &[x]) if vec_ty.lane().is_some_and(Lane::is_int) => {
                let v = self.ins().load(vec_ty, x, 0);
                let mask = self.ins().reduce(ReduceOp::Movemask, v);
                Ok(Some(self.resize(mask, (Type::I32, false), out_lanes.machine)))
            }
            (Simd::Sqrt, &[x]) if vec_ty.lane().is_some_and(Lane::is_float) => {
                let v = self.ins().load(vec_ty, x, 0);
                let root = self.ins().sqrt(v);
                Ok(Some(self.store_vec(root, vec_ty.bytes())))
            }
            (Simd::Reduce { all }, &[x]) if vec_ty.lane().is_some_and(Lane::is_int) => {
                let v = self.ins().load(vec_ty, x, 0);
                let zero = self.ins().iconst(vec_ty, 0);
                let mask = self.ins().icmp(IntCC::NotEqual, v, zero);
                let red = self.ins().reduce(if all { ReduceOp::And } else { ReduceOp::Or }, mask);
                Ok(Some(self.ins().icmp_imm(IntCC::NotEqual, red, 0)))
            }
            (Simd::Fold(bin), &[x]) => {
                let v = self.ins().load(vec_ty, x, 0);
                if bin == BinOp::Add && vec_ty.lane().is_some_and(Lane::is_float) {
                    let count = vec_ty.vec_shape().map_or(1, |(_, c)| c);
                    return Ok(Some(self.tree_reduce_vec(false, true, count, v)));
                }
                if !vec_ty.lane().is_some_and(Lane::is_int) {
                    return Ok(None);
                }
                let red = match bin {
                    BinOp::BitAnd => ReduceOp::And,
                    BinOp::BitOr => ReduceOp::Or,
                    BinOp::BitXor => ReduceOp::Xor,
                    BinOp::Add => ReduceOp::Add,
                    _ => return Ok(None),
                };
                Ok(Some(self.ins().reduce(red, v)))
            }
            (Simd::Extreme { max }, &[x]) => {
                let red = match (lanes.num, max) {
                    (Num::Float(_), true) => ReduceOp::Fmax,
                    (Num::Float(_), false) => ReduceOp::Fmin,
                    (Num::Int { signed: true, .. }, true) => ReduceOp::Smax,
                    (Num::Int { signed: true, .. }, false) => ReduceOp::Smin,
                    (_, true) => ReduceOp::Umax,
                    (_, false) => ReduceOp::Umin,
                };
                let v = self.ins().load(vec_ty, x, 0);
                Ok(Some(self.ins().reduce(red, v)))
            }
            (Simd::Lanewise(bin), &[x, y]) => self.vec_lanewise(bin, (&lanes, vec_ty), &out_lanes, (x, y)),
            (Simd::Cast, &[x]) => self.vec_cast((&lanes, vec_ty), &out_lanes, x),
            (Simd::Extract, &[x, _]) => {
                let Some(origin) = origin else { return Ok(None) };
                let Some(idx) = self.const_arg_u8(origin, 1, lanes.count as u8) else { return Ok(None) };
                let v = self.ins().load(vec_ty, x, 0);
                Ok(Some(self.ins().extract(v, idx)))
            }
            (Simd::Insert, &[x, _, val]) => {
                let Some(origin) = origin else { return Ok(None) };
                let Some(idx) = self.const_arg_u8(origin, 1, lanes.count as u8) else { return Ok(None) };
                let v = self.ins().load(vec_ty, x, 0);
                let out = self.ins().insert(v, val, idx);
                Ok(Some(self.store_vec(out, vec_ty.bytes())))
            }
            (Simd::Shuffle, &[x, y, _]) => {
                let (Some(origin), Some(&out_ty)) = (origin, substs.get(2)) else { return Ok(None) };
                let out_sh = self.lanes(out_ty)?;
                if vec_type(&out_sh) != Some(vec_ty) {
                    return Ok(None);
                }
                let Some(mask) = self.shuffle_const_mask(origin, lanes.count as usize) else { return Ok(None) };
                let (a, b) = (self.ins().load(vec_ty, x, 0), self.ins().load(vec_ty, y, 0));
                let res = self.ins().shuffle(a, b, mask);
                Ok(Some(self.store_vec(res, vec_ty.bytes())))
            }
            _ => Ok(None),
        }
    }

    fn vec_lanewise(&mut self, bin: BinOp, (from, vec_ty): (&Lanes, Type), to: &Lanes, (x, y): (Value, Value)) -> Result<Option<Value>> {
        let Some(out_vec) = vec_type(to) else { return Ok(None) };
        if compare(bin).is_some() {
            if out_vec.vec_shape().is_some_and(|(l, c)| l.is_int() && c == from.count as u8 && l.bytes() == from.size as u32) {
                let (a, b) = (self.ins().load(vec_ty, x, 0), self.ins().load(vec_ty, y, 0));
                let out = match from.num {
                    Num::Float(_) => self.ins().fcmp(vec_float_cc(bin), a, b),
                    Num::Int { signed, .. } => self.ins().icmp(vec_int_cc(bin, signed), a, b),
                    Num::Bool => return Ok(None),
                };
                return Ok(Some(self.store_vec(out, out_vec.bytes())));
            }
            return Ok(None);
        }
        if from.num != to.num || from.count != to.count {
            return Ok(None);
        }
        let (a, b) = (self.ins().load(vec_ty, x, 0), self.ins().load(vec_ty, y, 0));
        let out = match (bin, from.num) {
            (BinOp::Add, Num::Int { .. }) => self.ins().iadd(a, b),
            (BinOp::Sub, Num::Int { .. }) => self.ins().isub(a, b),
            (BinOp::Mul, Num::Int { .. }) => self.ins().imul(a, b),
            (BinOp::BitAnd, Num::Int { .. }) => self.ins().band(a, b),
            (BinOp::BitOr, Num::Int { .. }) => self.ins().bor(a, b),
            (BinOp::BitXor, Num::Int { .. }) => self.ins().bxor(a, b),
            (BinOp::Shl, Num::Int { .. }) => self.ins().ishl(a, b),
            (BinOp::Shr, Num::Int { signed: true, .. }) => self.ins().sshr(a, b),
            (BinOp::Shr, Num::Int { signed: false, .. }) => self.ins().ushr(a, b),
            (BinOp::Add, Num::Float(_)) => self.ins().fadd(a, b),
            (BinOp::Sub, Num::Float(_)) => self.ins().fsub(a, b),
            (BinOp::Mul, Num::Float(_)) => self.ins().fmul(a, b),
            (BinOp::Div, Num::Float(_)) => self.ins().fdiv(a, b),
            _ => return Ok(None),
        };
        Ok(Some(self.store_vec(out, vec_ty.bytes())))
    }

    fn vec_cast(&mut self, (from, from_vec): (&Lanes, Type), to: &Lanes, x: Value) -> Result<Option<Value>> {
        let Some(to_vec) = vec_type(to) else { return Ok(None) };
        if from.count != to.count {
            return Ok(None);
        }
        let v = self.ins().load(from_vec, x, 0);
        let out = match (from.num, to.num) {
            (Num::Int { ty: a, signed }, Num::Int { ty: b, .. }) => {
                if a == b {
                    v
                } else if a.bytes() < b.bytes() {
                    if signed { self.ins().sextend(to_vec, v) } else { self.ins().uextend(to_vec, v) }
                } else {
                    self.ins().ireduce(to_vec, v)
                }
            }
            (Num::Float(a), Num::Float(b)) => {
                if a == b {
                    v
                } else if a == Type::F32 {
                    self.ins().fpromote(to_vec, v)
                } else {
                    self.ins().fdemote(to_vec, v)
                }
            }
            (Num::Int { ty: a, signed }, Num::Float(b)) if a.bytes() == b.bytes() || a.bytes() * 2 == b.bytes() => {
                if signed { self.ins().fcvt_from_sint(to_vec, v) } else { self.ins().fcvt_from_uint(to_vec, v) }
            }
            (Num::Float(a), Num::Int { ty: b, signed }) if a.bytes() == b.bytes() => {
                if signed { self.ins().fcvt_to_sint_sat(to_vec, v) } else { self.ins().fcvt_to_uint_sat(to_vec, v) }
            }
            _ => return Ok(None),
        };
        Ok(Some(self.store_vec(out, to_vec.bytes())))
    }

    fn vec_select(&mut self, (from, to): (&Lanes, &Lanes), args: &[Value]) -> Result<Option<Value>> {
        let &[mask, yes, no] = args else { return Ok(None) };
        let (Some(mask_vec), Some(out_vec)) = (vec_type(from), vec_type(to)) else { return Ok(None) };
        if from.count != to.count || from.size != to.size || !mask_vec.lane().is_some_and(Lane::is_int) {
            return Ok(None);
        }
        let (m, a, b) = (self.ins().load(mask_vec, mask, 0), self.ins().load(out_vec, yes, 0), self.ins().load(out_vec, no, 0));
        let zero = self.ins().iconst(mask_vec, 0);
        let norm = self.ins().icmp(IntCC::NotEqual, m, zero);
        let chosen = self.ins().select(norm, a, b);
        Ok(Some(self.store_vec(chosen, out_vec.bytes())))
    }

    fn call_arg_expr(&self, origin: &SiteOrigin, index: usize) -> Option<ExprId> {
        let SiteOrigin::Table(SiteKey::ExprRes(key)) = *origin else { return None };
        let list = match *self.body.expr(key) {
            Expr::Call(_, list) => list,
            _ => self.body.exprs.iter().find_map(|expr| match *expr {
                Expr::Call(callee, list) if callee == key => Some(list),
                _ => None,
            })?,
        };
        self.body.list(list).iter().copied().map(ExprId).filter(|id| !self.typed.const_args.contains(id)).nth(index)
    }

    fn const_arg_u8(&mut self, origin: &SiteOrigin, index: usize, limit: u8) -> Option<u8> {
        let expr = self.call_arg_expr(origin, index)?;
        let val = eval_const_int(self.cx, self.body, self.typed, self.body, self.src, expr, 0)?;
        u8::try_from(val).ok().filter(|&n| n < limit)
    }

    fn shuffle_const_mask(&mut self, origin: &SiteOrigin, count: usize) -> Option<[u8; 16]> {
        if count > 16 {
            return None;
        }
        let expr = self.call_arg_expr(origin, 2)?;
        eval_shuffle_array(self.cx, self.body, self.typed, self.body, self.src, expr, count, 0)
    }
}

fn unwrap_const_expr(body: &Body, mut expr: ExprId) -> ExprId {
    loop {
        match *body.expr(expr) {
            Expr::Paren(inner) | Expr::Cast(inner, _) => expr = inner,
            Expr::Block { stmts, tail: Some(tail), kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const, .. } if stmts.len == 0 => expr = tail,
            _ => return expr,
        }
    }
}

fn eval_const_int<'a>(cx: &mut crate::sema::Context<'a>, origin_body: &Body, typed: &crate::sema::infer::Typed, body: &Body, src: &Source, expr: ExprId, depth: u8) -> Option<u128> {
    if depth > 4 {
        return None;
    }
    let expr = unwrap_const_expr(body, expr);
    match *body.expr(expr) {
        Expr::Lit(Lit::Int, tok) => parse_int(src.text(tok)).ok().map(|(n, _)| n),
        Expr::Path(_) if std::ptr::eq(body, origin_body) => match typed.res.get(&expr).copied()? {
            Res::Const(val) => match *cx.types.kind(val) {
                Kind::Value(n) => Some(n),
                _ => None,
            },
            Res::Fn(item, substs) if cx.item(item).1.kind == ItemKind::Const && cx.types.substs(substs).is_empty() => {
                let item_body = cx.body(item).ok()?;
                let root = item_body.root?;
                let item_src = cx.body_src(item);
                eval_const_int(cx, origin_body, typed, &item_body, item_src, root, depth + 1)
            }
            _ => None,
        },
        _ => None,
    }
}

fn eval_shuffle_array<'a>(cx: &mut crate::sema::Context<'a>, origin_body: &Body, typed: &crate::sema::infer::Typed, body: &Body, src: &Source, expr: ExprId, count: usize, depth: u8) -> Option<[u8; 16]> {
    if depth > 4 {
        return None;
    }
    let expr = unwrap_const_expr(body, expr);
    let limit = u128::try_from(2 * count).ok()?;
    let mut mask = [0u8; 16];
    match *body.expr(expr) {
        Expr::Array(list) => {
            let ids = body.list(list);
            if ids.len() != count {
                return None;
            }
            mask[..count].iter_mut().zip(ids).try_for_each(|(slot, &id)| {
                let n = eval_const_int(cx, origin_body, typed, body, src, ExprId(id), depth + 1)?;
                (n < limit).then(|| *slot = n as u8)
            })?;
            Some(mask)
        }
        Expr::Repeat(elem, _) => {
            let n = eval_const_int(cx, origin_body, typed, body, src, elem, depth + 1)?;
            if n >= limit {
                return None;
            }
            mask[..count].fill(n as u8);
            Some(mask)
        }
        Expr::Path(_) if std::ptr::eq(body, origin_body) => {
            let Res::Fn(item, substs) = typed.res.get(&expr).copied()? else { return None };
            if cx.item(item).1.kind != ItemKind::Const || !cx.types.substs(substs).is_empty() {
                return None;
            }
            let item_body = cx.body(item).ok()?;
            let root = item_body.root?;
            let item_src = cx.body_src(item);
            eval_shuffle_array(cx, origin_body, typed, &item_body, item_src, root, count, depth + 1)
        }
        _ => None,
    }
}

impl Lower<'_, '_> {
    pub(super) fn lanes(&mut self, ty: Ty) -> Result<Lanes> {
        let Kind::Adt(item, args) = *self.cx.types.kind(ty) else { bail!("a SIMD intrinsic on a type that is not a vector") };
        let slice = self.cx.types.substs(args).to_vec();
        let fields = crate::sema::infer::field_types(self.cx, item, &slice)?;
        let Some(&array) = fields.first().and_then(|variant| variant.first()) else { bail!("a SIMD vector with no field") };
        let Kind::Array(elem, len) = *self.cx.types.kind(array) else { bail!("a SIMD vector whose field is not an array") };
        let Some(lane) = machine(self.cx, elem)? else { bail!("a SIMD lane with no machine type") };
        Ok(Lanes { num: self.num(elem)?, machine: lane, size: lane.bytes() as i32, count: self.len(len)? as i32 })
    }

    pub(super) fn vector_slot(&mut self, ty: Ty) -> Result<(StackSlot, Value)> {
        let held = repr(self.cx, ty)?;
        let Some(slot) = self.slot(held) else { bail!("a SIMD vector that takes no bytes") };
        Ok((slot, self.ins().stack_addr(Type::I64, slot, 0)))
    }

    pub(super) fn lane_op(&mut self, op: BinOp, num: Num, (x, y): (Value, Value), out: Type) -> Result<Value> {
        if compare(op).is_some() {
            let flag = self.apply(op, num, (x, y), num)?;
            let wide = self.resize(flag, (Type::I8, false), out);
            return Ok(self.ins().ineg(wide));
        }
        match (op, num) {
            (BinOp::Add | BinOp::Sub | BinOp::Mul, Num::Int { ty, signed }) => Ok(self.overflowing(op, (ty, signed), x, y)?.0),
            _ => self.apply(op, num, (x, y), num),
        }
    }
}

fn vec_int_cc(op: BinOp, signed: bool) -> IntCC {
    match (op, signed) {
        (BinOp::Eq, _) => IntCC::Equal,
        (BinOp::Ne, _) => IntCC::NotEqual,
        (BinOp::Lt, true) => IntCC::SignedLessThan,
        (BinOp::Le, true) => IntCC::SignedLessThanOrEqual,
        (BinOp::Gt, true) => IntCC::SignedGreaterThan,
        (BinOp::Ge, true) => IntCC::SignedGreaterThanOrEqual,
        (BinOp::Lt, false) => IntCC::UnsignedLessThan,
        (BinOp::Le, false) => IntCC::UnsignedLessThanOrEqual,
        (BinOp::Gt, false) => IntCC::UnsignedGreaterThan,
        _ => IntCC::UnsignedGreaterThanOrEqual,
    }
}

fn vec_float_cc(op: BinOp) -> FloatCC {
    match op {
        BinOp::Eq => FloatCC::Equal,
        BinOp::Ne => FloatCC::NotEqual,
        BinOp::Lt => FloatCC::LessThan,
        BinOp::Le => FloatCC::LessThanOrEqual,
        BinOp::Gt => FloatCC::GreaterThan,
        _ => FloatCC::GreaterThanOrEqual,
    }
}
