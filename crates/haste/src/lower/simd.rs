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

mod lanes;
mod target;

pub(super) struct Lanes {
    num: Num,
    machine: Type,
    size: i32,
    count: i32,
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

impl Lower<'_, '_> {
    /// Emits the SIMD intrinsic `op` using native `Type::Vec` operations when possible and scalarizing unsupported shapes.
    pub(super) fn simd(&mut self, origin: &SiteOrigin, op: Simd, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let (Some(&vector), Some(&first)) = (substs.first(), args.first()) else { bail!("a SIMD intrinsic with no vector") };
        let out = substs.get(1).copied();
        if let Some(value) = self.vec_simd(origin, op, (vector, out.unwrap_or(vector)), substs, args)? {
            return Ok(Some(value));
        }
        if matches!(op, Simd::Unary(_) | Simd::Pair(_) | Simd::Fma | Simd::Sum { .. } | Simd::MaskedLoad | Simd::MaskedStore) {
            return self.lane_intrinsic(op, substs, args);
        }
        if let Some(value) = self.packed_swar64(op, (vector, out.unwrap_or(vector)), args)? {
            return Ok(Some(value));
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

    fn vec_simd(&mut self, origin: &SiteOrigin, op: Simd, (input, output): (Ty, Ty), substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let lanes = self.lanes(input)?;
        if op == Simd::SelectBitmask {
            return Ok(None);
        }
        if op == Simd::Select {
            return self.vec_select((input, output), args);
        }
        let Some(vec_ty) = vec_type(&lanes) else { return Ok(None) };
        match (op, args) {
            (Simd::Splat, &[x]) => {
                let v = self.ins().splat(vec_ty, x);
                Ok(Some(self.store_vec(v, vec_ty.bytes())))
            }
            (Simd::Bitmask, &[x]) if vec_ty.lane().is_some_and(Lane::is_int) => {
                let Some(target) = machine(self.cx, output)? else { return Ok(None) };
                let v = self.ins().load(vec_ty, x, 0);
                let mask = self.ins().reduce(ReduceOp::Movemask, v);
                Ok(Some(self.resize(mask, (Type::I32, false), target)))
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
            (Simd::Fold(bin), &[x]) if vec_ty.lane().is_some_and(Lane::is_int) => {
                let red = match bin {
                    BinOp::BitAnd => ReduceOp::And,
                    BinOp::BitOr => ReduceOp::Or,
                    BinOp::BitXor => ReduceOp::Xor,
                    _ => return Ok(None),
                };
                let v = self.ins().load(vec_ty, x, 0);
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
            (Simd::Lanewise(bin), &[x, y]) => self.vec_lanewise(bin, (input, output), (&lanes, vec_ty), (x, y)),
            (Simd::Cast, &[x]) => self.vec_cast((&lanes, vec_ty), output, x),
            (Simd::Extract, &[x, _]) => {
                let Some(idx) = self.const_arg_u8(origin, 1, lanes.count as u8) else { return Ok(None) };
                let v = self.ins().load(vec_ty, x, 0);
                Ok(Some(self.ins().extract(v, idx)))
            }
            (Simd::Insert, &[x, _, val]) => {
                let Some(idx) = self.const_arg_u8(origin, 1, lanes.count as u8) else { return Ok(None) };
                let v = self.ins().load(vec_ty, x, 0);
                let out = self.ins().insert(v, val, idx);
                Ok(Some(self.store_vec(out, vec_ty.bytes())))
            }
            (Simd::Shuffle, &[x, y, _]) => {
                let Some(&out_ty) = substs.get(2) else { return Ok(None) };
                let out_lanes = self.lanes(out_ty)?;
                if vec_type(&out_lanes) != Some(vec_ty) {
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

    fn vec_lanewise(&mut self, bin: BinOp, (input, output): (Ty, Ty), (from, vec_ty): (&Lanes, Type), (x, y): (Value, Value)) -> Result<Option<Value>> {
        let to = self.lanes(output)?;
        let Some(out_vec) = vec_type(&to) else { return Ok(None) };
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
        if input != output {
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

    fn vec_cast(&mut self, (from, from_vec): (&Lanes, Type), output: Ty, x: Value) -> Result<Option<Value>> {
        let to = self.lanes(output)?;
        let Some(to_vec) = vec_type(&to) else { return Ok(None) };
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

    fn vec_select(&mut self, (mask_ty, out_ty): (Ty, Ty), args: &[Value]) -> Result<Option<Value>> {
        let &[mask, yes, no] = args else { return Ok(None) };
        let (from, to) = (self.lanes(mask_ty)?, self.lanes(out_ty)?);
        let (Some(mask_vec), Some(out_vec)) = (vec_type(&from), vec_type(&to)) else { return Ok(None) };
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
            for (slot, &id) in mask[..count].iter_mut().zip(ids) {
                let n = eval_const_int(cx, origin_body, typed, body, src, ExprId(id), depth + 1)?;
                if n >= limit {
                    return None;
                }
                *slot = n as u8;
            }
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

    fn packed_swar64(&mut self, op: Simd, (input, output): (Ty, Ty), args: &[Value]) -> Result<Option<Value>> {
        let lanes = self.lanes(input)?;
        if lanes.size * lanes.count != 8 || !matches!(lanes.num, Num::Int { .. }) {
            return Ok(None);
        }
        let signed = matches!(lanes.num, Num::Int { signed: true, .. });
        let is_u8x8 = lanes.size == 1 && lanes.count == 8;
        let word = match (op, args) {
            (Simd::Splat, &[x]) if is_u8x8 => {
                let wide = self.resize(x, (lanes.machine, false), Type::I64);
                let ff = self.ins().iconst(Type::I64, 0xFF);
                let byte = self.ins().band(wide, ff);
                self.ins().imul_imm(byte, 0x0101_0101_0101_0101_u64 as i64)
            }
            (Simd::Extract, &[x, _]) if lanes.size == 8 && lanes.count == 1 => {
                return Ok(Some(self.ins().load(lanes.machine, x, 0)));
            }
            (Simd::Lanewise(bin @ (BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor)), &[x, y]) if input == output => {
                let (a, b) = (self.ins().load(Type::I64, x, 0), self.ins().load(Type::I64, y, 0));
                match bin {
                    BinOp::BitAnd => self.ins().band(a, b),
                    BinOp::BitOr => self.ins().bor(a, b),
                    _ => self.ins().bxor(a, b),
                }
            }
            (Simd::Lanewise(cmp @ (BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)), &[x, y])
                if is_u8x8 && self.lanes(output)?.size == 1 && self.lanes(output)?.count == 8 =>
            {
                let (a, b) = (self.ins().load(Type::I64, x, 0), self.ins().load(Type::I64, y, 0));
                let m7f = self.ins().iconst(Type::I64, 0x7F7F_7F7F_7F7F_7F7F_u64 as i64);
                let m80 = self.ins().iconst(Type::I64, 0x8080_8080_8080_8080_u64 as i64);
                let seven = self.ins().iconst(Type::I64, 7);
                let bits = match cmp {
                    BinOp::Eq | BinOp::Ne => {
                        let d = self.ins().bxor(a, b);
                        let masked = self.ins().band(d, m7f);
                        let low = self.ins().iadd(masked, m7f);
                        let low_or_d = self.ins().bor(low, d);
                        let nz_or_7f = self.ins().bor(low_or_d, m7f);
                        let msb = if cmp == BinOp::Eq {
                            self.ins().bnot(nz_or_7f)
                        } else {
                            self.ins().band(nz_or_7f, m80)
                        };
                        self.ins().ushr(msb, seven)
                    }
                    _ => {
                        let (lhs, rhs, invert) = match cmp {
                            BinOp::Gt => (b, a, false),
                            BinOp::Ge => (a, b, true),
                            BinOp::Le => (b, a, true),
                            _ => (a, b, false),
                        };
                        let (ua, ub) = if signed {
                            (self.ins().bxor(lhs, m80), self.ins().bxor(rhs, m80))
                        } else {
                            (lhs, rhs)
                        };
                        let ua_hi = self.ins().bor(ua, m80);
                        let ub_lo = self.ins().band(ub, m7f);
                        let diff = self.ins().isub(ua_hi, ub_lo);
                        let not_ua = self.ins().bnot(ua);
                        let term1 = self.ins().band(not_ua, ub);
                        let xor_ab = self.ins().bxor(ua, ub);
                        let same_hi = self.ins().bnot(xor_ab);
                        let not_diff = self.ins().bnot(diff);
                        let term2 = self.ins().band(same_hi, not_diff);
                        let lt_hi = self.ins().bor(term1, term2);
                        let raw = if invert { self.ins().bnot(lt_hi) } else { lt_hi };
                        let msb = self.ins().band(raw, m80);
                        self.ins().ushr(msb, seven)
                    }
                };
                self.ins().imul_imm(bits, 0xFF)
            }
            _ => return Ok(None),
        };
        let (slot, address) = self.vector_slot(if matches!(op, Simd::Splat) { input } else { output })?;
        self.ins().stack_store(word, slot, 0);
        Ok(Some(address))
    }

    fn simd_lanes(&mut self, op: Simd, substs: &[Ty], args: &[Value]) -> Result<Value> {
        let vector = substs[0];
        Ok(match (op, args) {
            (Simd::Lanewise(bin), &[x, y]) => self.lanewise(bin, (vector, substs.get(1).copied().unwrap_or(vector)), (x, y))?,
            (Simd::Select, &[mask, yes, no]) => self.lane_select((vector, substs.get(1).copied().unwrap_or(vector)), mask, (yes, no))?,
            (Simd::SelectBitmask, &[mask, yes, no]) => self.bitmask_select((vector, substs.get(1).copied().unwrap_or(vector)), mask, (yes, no))?,
            (Simd::Shuffle, &[x, y, idx]) => self.shuffle(substs, (x, y), idx)?,
            (Simd::Extract, &[x, at]) => {
                let (lanes, lane) = self.lane_address(vector, x, at)?;
                self.ins().load(lanes.machine, lane, 0)
            }
            (Simd::Insert, &[x, at, value]) => self.insert(vector, x, (at, value))?,
            _ => bail!("SIMD intrinsic {op:?} takes other arguments"),
        })
    }

    fn lanes(&mut self, ty: Ty) -> Result<Lanes> {
        let Kind::Adt(item, args) = *self.cx.types.kind(ty) else { bail!("a SIMD intrinsic on a type that is not a vector") };
        let slice = self.cx.types.substs(args).to_vec();
        let fields = crate::sema::infer::field_types(self.cx, item, &slice)?;
        let Some(&array) = fields.first().and_then(|variant| variant.first()) else { bail!("a SIMD vector with no field") };
        let Kind::Array(elem, len) = *self.cx.types.kind(array) else { bail!("a SIMD vector whose field is not an array") };
        let Some(lane) = machine(self.cx, elem)? else { bail!("a SIMD lane with no machine type") };
        Ok(Lanes { num: self.num(elem)?, machine: lane, size: lane.bytes() as i32, count: self.len(len)? as i32 })
    }

    fn vector_slot(&mut self, ty: Ty) -> Result<(StackSlot, Value)> {
        let held = repr(self.cx, ty)?;
        let Some(slot) = self.slot(held) else { bail!("a SIMD vector that takes no bytes") };
        Ok((slot, self.ins().stack_addr(Type::I64, slot, 0)))
    }

    fn splat(&mut self, vector: Ty, value: Value) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let (slot, address) = self.vector_slot(vector)?;
        for lane in 0..lanes.count {
            self.ins().stack_store(value, slot, lane * lanes.size);
        }
        Ok(address)
    }

    fn lane_sqrt(&mut self, vector: Ty, x: Value) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let (slot, address) = self.vector_slot(vector)?;
        for lane in 0..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            let root = self.ins().sqrt(value);
            self.ins().stack_store(root, slot, lane * lanes.size);
        }
        Ok(address)
    }

    fn lanewise(&mut self, op: BinOp, (input, output): (Ty, Ty), (x, y): (Value, Value)) -> Result<Value> {
        let (from, to) = (self.lanes(input)?, self.lanes(output)?);
        let (slot, address) = self.vector_slot(output)?;
        for lane in 0..from.count {
            let a = self.ins().load(from.machine, x, lane * from.size);
            let b = self.ins().load(from.machine, y, lane * from.size);
            let value = self.lane_op(op, from.num, (a, b), to.machine)?;
            self.ins().stack_store(value, slot, lane * to.size);
        }
        Ok(address)
    }

    fn lane_cast(&mut self, (input, output): (Ty, Ty), x: Value) -> Result<Value> {
        let (from, to) = (self.lanes(input)?, self.lanes(output)?);
        if from.count != to.count {
            bail!("`simd_cast` between vectors of {} and {} lanes", from.count, to.count);
        }
        let (slot, address) = self.vector_slot(output)?;
        for lane in 0..from.count {
            let value = self.ins().load(from.machine, x, lane * from.size);
            let value = self.convert(value, from.num, to.num)?;
            self.ins().stack_store(value, slot, lane * to.size);
        }
        Ok(address)
    }

    fn lane_op(&mut self, op: BinOp, num: Num, (x, y): (Value, Value), out: Type) -> Result<Value> {
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

    /// The vector whose lane `i` is lane `idx[i]` of `x` followed by `y`, as `simd_shuffle` at `[T, U, V]` builds it.
    fn shuffle(&mut self, substs: &[Ty], (x, y): (Value, Value), idx: Value) -> Result<Value> {
        let &[input, indices, output] = substs else { bail!("`simd_shuffle` takes three types") };
        let (from, to) = (self.lanes(input)?, self.lanes(output)?);
        let (both, joined) = self.pair_slot(&from)?;
        if let Some(vt) = vec_type(&from) {
            let (a, b) = (self.ins().load(vt, x, 0), self.ins().load(vt, y, 0));
            self.ins().stack_store(a, both, 0);
            self.ins().stack_store(b, both, from.count * from.size);
        } else {
            for lane in 0..from.count {
                let a = self.ins().load(from.machine, x, lane * from.size);
                let b = self.ins().load(from.machine, y, lane * from.size);
                self.ins().stack_store(a, both, lane * from.size);
                self.ins().stack_store(b, both, (from.count + lane) * from.size);
            }
        }
        let Some(held) = self.spill(Some(idx), indices)? else { bail!("`simd_shuffle` indices take no bytes") };
        let idx = self.address(held);
        let (slot, address) = self.vector_slot(output)?;
        for lane in 0..to.count {
            let at = self.ins().load(Type::I32, idx, lane * 4);
            let at = self.ins().uextend(Type::I64, at);
            let offset = self.ins().imul_imm(at, i64::from(from.size));
            let source = self.ins().iadd(joined, offset);
            let value = self.ins().load(to.machine, source, 0);
            self.ins().stack_store(value, slot, lane * to.size);
        }
        Ok(address)
    }

    fn lane_address(&mut self, vector: Ty, x: Value, at: Value) -> Result<(Lanes, Value)> {
        let lanes = self.lanes(vector)?;
        let at = self.ins().uextend(Type::I64, at);
        let offset = self.ins().imul_imm(at, i64::from(lanes.size));
        let lane = self.ins().iadd(x, offset);
        Ok((lanes, lane))
    }

    fn insert(&mut self, vector: Ty, x: Value, (at, value): (Value, Value)) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let (slot, address) = self.vector_slot(vector)?;
        if let Some(vt) = vec_type(&lanes) {
            let v = self.ins().load(vt, x, 0);
            self.ins().stack_store(v, slot, 0);
        } else {
            for lane in 0..lanes.count {
                let held = self.ins().load(lanes.machine, x, lane * lanes.size);
                self.ins().stack_store(held, slot, lane * lanes.size);
            }
        }
        let (_, lane) = self.lane_address(vector, address, at)?;
        self.ins().store(value, lane, 0);
        Ok(address)
    }

    fn pair_slot(&mut self, lanes: &Lanes) -> Result<(StackSlot, Value)> {
        let bytes = (2 * lanes.count * lanes.size) as u32;
        let slot = self.b.create_stack_slot(bytes, lanes.size.trailing_zeros() as u8);
        Ok((slot, self.ins().stack_addr(Type::I64, slot, 0)))
    }

    fn reduce(&mut self, vector: Ty, x: Value, all: bool) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let mut found = self.ins().iconst(Type::I8, i64::from(all));
        for lane in 0..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            let set = self.ins().icmp_imm(IntCC::NotEqual, value, 0);
            found = if all { self.ins().band(found, set) } else { self.ins().bor(found, set) };
        }
        Ok(found)
    }

    fn fold(&mut self, vector: Ty, x: Value, op: BinOp) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let mut folded = self.ins().load(lanes.machine, x, 0);
        for lane in 1..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            folded = self.apply(op, lanes.num, (folded, value), lanes.num)?;
        }
        Ok(folded)
    }

    fn extreme(&mut self, vector: Ty, x: Value, max: bool) -> Result<Value> {
        let lanes = self.lanes(vector)?;
        let better = if max { BinOp::Gt } else { BinOp::Lt };
        let mut best = self.ins().load(lanes.machine, x, 0);
        for lane in 1..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            let wins = self.apply(better, lanes.num, (value, best), lanes.num)?;
            best = self.ins().select(wins, value, best);
        }
        Ok(best)
    }

    fn bitmask(&mut self, vector: Ty, x: Value, out: Ty) -> Result<Value> {
        let Some(target) = machine(self.cx, out)? else { bail!("a SIMD bitmask into an array is not supported yet") };
        let lanes = self.lanes(vector)?;
        let mut mask = self.ins().iconst(Type::I64, 0);
        for lane in 0..lanes.count {
            let value = self.ins().load(lanes.machine, x, lane * lanes.size);
            let set = self.ins().icmp_imm(IntCC::NotEqual, value, 0);
            let bit = self.ins().uextend(Type::I64, set);
            let shift = self.ins().iconst(Type::I64, i64::from(lane));
            let bit = self.ins().ishl(bit, shift);
            mask = self.ins().bor(mask, bit);
        }
        Ok(self.resize(mask, (Type::I64, false), target))
    }

    fn lane_select(&mut self, (mask_ty, out_ty): (Ty, Ty), mask: Value, (yes, no): (Value, Value)) -> Result<Value> {
        let (from, to) = (self.lanes(mask_ty)?, self.lanes(out_ty)?);
        let (slot, address) = self.vector_slot(out_ty)?;
        for lane in 0..to.count {
            let cond = self.ins().load(from.machine, mask, lane * from.size);
            let set = self.ins().icmp_imm(IntCC::NotEqual, cond, 0);
            let a = self.ins().load(to.machine, yes, lane * to.size);
            let b = self.ins().load(to.machine, no, lane * to.size);
            let chosen = self.ins().select(set, a, b);
            self.ins().stack_store(chosen, slot, lane * to.size);
        }
        Ok(address)
    }

    fn bitmask_select(&mut self, (mask_ty, out_ty): (Ty, Ty), mask: Value, (yes, no): (Value, Value)) -> Result<Value> {
        let Some(from) = machine(self.cx, mask_ty)? else { bail!("a SIMD bitmask select from an array is not supported yet") };
        let to = self.lanes(out_ty)?;
        let (slot, address) = self.vector_slot(out_ty)?;
        let bits = self.resize(mask, (from, false), Type::I64);
        for lane in 0..to.count {
            let shift = self.ins().iconst(Type::I64, i64::from(lane));
            let shifted = self.ins().ushr(bits, shift);
            let one = self.ins().iconst(Type::I64, 1);
            let bit = self.ins().band(shifted, one);
            let set = self.ins().icmp_imm(IntCC::NotEqual, bit, 0);
            let a = self.ins().load(to.machine, yes, lane * to.size);
            let b = self.ins().load(to.machine, no, lane * to.size);
            let chosen = self.ins().select(set, a, b);
            self.ins().stack_store(chosen, slot, lane * to.size);
        }
        Ok(address)
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
