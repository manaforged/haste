use anyhow::{Result, bail};
use haste_codegen::{Arch, FloatCC, IntCC, Lane, ReduceOp, StackSlot, Target, Type, Value};

use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, Lit, Pat, Stmt, Ty as BodyTy, UnOp};
use crate::names::{Primitive, plain};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::infer::{Res, SiteKey, Typed, parse_int};
use crate::sema::intrinsic::Simd;
use crate::sema::ty::{Kind, Ty, int_info, mask as int_mask};

use super::ops::{Num, compare, parse_float};
use super::reloc::SiteOrigin;
use super::{Lower, machine, repr};

mod fallback;
mod lanes;
mod scalar;
mod target;
mod wide;

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
        let (Some(&vector), Some(_)) = (substs.first(), args.first()) else { bail!("a SIMD intrinsic with no vector") };
        let out = if op == Simd::Shuffle { substs.get(2) } else { substs.get(1) }.copied();
        if op == Simd::SelectBitmask {
            let &[mask, yes, no] = args else { bail!("SIMD intrinsic {op:?} takes other arguments") };
            return self.bitmask_select((vector, out.unwrap_or(vector)), mask, (yes, no)).map(Some);
        }
        let from = self.lanes(vector)?;
        let Some(to) = self.simd_out_lanes(op, from, out)? else {
            return Ok(None);
        };
        match self.simd_wide(origin, op, (vector, out), (from, to), substs, args)? {
            Some(done) => Ok(done),
            None => self.simd_halves(origin, op, (vector, out), (from, to), substs, args),
        }
    }

    /// Emits the SIMD intrinsic `op` on vectors of at most 16 bytes, with a wider vector as 16-byte halves in memory.
    fn simd_halves(&mut self, origin: &SiteOrigin, op: Simd, (vector, out): (Ty, Option<Ty>), (from, to): (Lanes, Lanes), substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let Some(&first) = args.first() else { return Ok(None) };
        if let Some(value) = self.vec_simd_lanes(Some(origin), op, (from, to), substs, args)? {
            return Ok(Some(value));
        }
        if matches!(op, Simd::Unary(_) | Simd::Pair(_) | Simd::Fma | Simd::Sum { .. } | Simd::MaskedLoad | Simd::MaskedStore) {
            return self.lane_intrinsic(op, substs, args);
        }
        if let Some(value) = self.packed_swar64(op, (vector, out.unwrap_or(vector)), args)? {
            return Ok(Some(value));
        }
        if let Some(value) = self.wide_lane_op(origin, op, (&from, &to), args)? {
            return Ok(Some(value));
        }
        if let Some(value) = self.wide_simd(op, (&from, &to), args)? {
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

    pub(super) fn bitcast_vec(&mut self, to: Type, val: Value) -> Value {
        if self.b.func.ty(val) == to {
            return val;
        }
        self.ins().bitcast(to, val)
    }

    pub(super) fn lanes_slot(&mut self, lanes: Lanes) -> (StackSlot, Value) {
        let slot = self.b.create_stack_slot(lanes.bytes() as u32, lanes.align_shift());
        let addr = self.ins().stack_addr(Type::I64, slot, 0);
        (slot, addr)
    }

    fn simd_out_lanes(&mut self, op: Simd, from: Lanes, out: Option<Ty>) -> Result<Option<Lanes>> {
        Ok(Some(match op {
            Simd::Bitmask => {
                let Some(out_ty) = out else { bail!("`simd_bitmask` has no output type") };
                let Some(target) = machine(self.cx, out_ty)? else { bail!("a SIMD bitmask into an array is not supported yet") };
                Lanes { num: Num::Int { ty: target, signed: false }, machine: target, size: target.bytes() as i32, count: 1 }
            }
            Simd::Lanewise(_) | Simd::Cast | Simd::Select | Simd::Shuffle => {
                let Some(out_ty) = out else { return Ok(Some(from)) };
                self.lanes(out_ty)?
            }
            _ => from,
        }))
    }

    pub(super) fn vec_simd_lanes(&mut self, origin: Option<&SiteOrigin>, op: Simd, (lanes, out_lanes): (Lanes, Lanes), _substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        if op == Simd::Select {
            return self.vec_select((&lanes, &out_lanes), args);
        }
        let Some(vec_ty) = vec_type(&lanes) else { return Ok(None) };
        match (op, args) {
            (Simd::Splat, &[x]) => {
                let v = origin.and_then(|o| self.const_splat_vec(o, vec_ty)).unwrap_or_else(|| self.ins().splat(vec_ty, x));
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
                let Some(origin) = origin else { return Ok(None) };
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
        let Some(out) = self.lanewise_value(bin, from.num, (a, b)) else { return Ok(None) };
        Ok(Some(self.store_vec(out, vec_ty.bytes())))
    }

    /// The lanewise `bin` of the vectors `a` and `b` of `num` lanes, when one instruction computes it.
    fn lanewise_value(&mut self, bin: BinOp, num: Num, (a, b): (Value, Value)) -> Option<Value> {
        Some(match (bin, num) {
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
            _ => return None,
        })
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

    pub(super) fn const_arg_u8(&mut self, origin: &SiteOrigin, index: usize, limit: u8) -> Option<u8> {
        let expr = self.call_arg_expr(origin, index)?;
        let mut ctx = ConstEval { cx: self.cx, origin_body: self.body, typed: self.typed };
        let val = ctx.int(self.body, self.src, expr, &mut Vec::new(), 0)?;
        u8::try_from(val.val).ok().filter(|&n| (!val.signed || val.to_signed() >= 0) && n < limit)
    }

    fn shuffle_const_mask(&mut self, origin: &SiteOrigin, count: usize) -> Option<[u8; 16]> {
        if count > 16 {
            return None;
        }
        let wide = self.shuffle_const_mask_wide(origin, count, count)?;
        let mut mask = [0u8; 16];
        mask[..count].copy_from_slice(&wide[..count]);
        Some(mask)
    }

    pub(super) fn shuffle_const_mask_wide(&mut self, origin: &SiteOrigin, in_count: usize, out_count: usize) -> Option<[u8; 64]> {
        if out_count == 0 || out_count > 64 || in_count == 0 {
            return None;
        }
        let expr = self.call_arg_expr(origin, 2)?;
        let mut ctx = ConstEval { cx: self.cx, origin_body: self.body, typed: self.typed };
        ctx.shuffle_array(self.body, self.src, expr, (in_count, out_count), &mut Vec::new(), 0)
    }

    pub(super) fn const_splat_vec(&mut self, origin: &SiteOrigin, vec_ty: Type) -> Option<Value> {
        let expr = self.call_arg_expr(origin, 0)?;
        let lane = vec_ty.lane()?;
        let bits = if lane.is_int() {
            let mut ctx = ConstEval { cx: self.cx, origin_body: self.body, typed: self.typed };
            ctx.int(self.body, self.src, expr, &mut Vec::new(), 0)?.val & int_mask(lane.bytes() * 8)
        } else {
            let lane_ty = match lane { Lane::F32 => Type::F32, Lane::F64 => Type::F64, _ => return None };
            let inner = ConstEval { cx: self.cx, origin_body: self.body, typed: self.typed }.unwrap_expr(self.body, expr);
            let Expr::Lit(Lit::Float, tok) = *self.body.expr(inner) else { return None };
            parse_float(lane_ty, self.src.text(tok)).ok()? as u64 as u128
        };
        if bits == 0 {
            return Some(self.ins().iconst(vec_ty, 0));
        }
        let lane_bytes = lane.bytes() as usize;
        let mut bytes = vec![0u8; vec_ty.bytes() as usize];
        for chunk in bytes.chunks_exact_mut(lane_bytes) {
            chunk.copy_from_slice(&bits.to_le_bytes()[..lane_bytes]);
        }
        let addr = self.emit_const_rodata(&bytes);
        Some(self.ins().load(vec_ty, addr, 0))
    }
}

#[derive(Clone, Copy)]
struct ConstVal {
    val: u128,
    bits: u32,
    signed: bool,
    typed: bool,
}

impl ConstVal {
    fn to_signed(self) -> i128 {
        let shift = 128 - self.bits;
        ((self.val << shift) as i128) >> shift
    }

    fn cast(self, (bits, signed): (u32, bool)) -> Self {
        let extended = if self.signed { self.to_signed() as u128 } else { self.val };
        Self { val: extended & int_mask(bits), bits, signed, typed: true }
    }
}

struct ConstEval<'a, 'b> {
    cx: &'b mut crate::sema::Context<'a>,
    origin_body: &'b Body,
    typed: &'b Typed,
}

impl<'a, 'b> ConstEval<'a, 'b> {
    fn unwrap_expr(&self, body: &Body, mut expr: ExprId) -> ExprId {
        loop {
            match *body.expr(expr) {
                Expr::Paren(inner) => expr = inner,
                Expr::Block { stmts, tail: Some(tail), kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const | BlockKind::Expansion, .. }
                    if body.list(stmts).is_empty() =>
                {
                    expr = tail;
                }
                _ => return expr,
            }
        }
    }

    fn unwrap_shuffle(&self, body: &Body, src: &Source, mut expr: ExprId) -> ExprId {
        loop {
            expr = self.unwrap_expr(body, expr);
            match *body.expr(expr) {
                Expr::Call(callee, list) => {
                    let &[arg] = body.list(list) else { return expr };
                    let is_ctor = std::ptr::eq(body, self.origin_body)
                        && matches!(self.typed.res.get(&callee).or_else(|| self.typed.res.get(&expr)), Some(Res::Ctor(..)));
                    let is_idx = matches!(*body.expr(callee), Expr::Path(path) if body.list(body.path(path).segments).last().is_some_and(|&s| plain(src.text(body.segments[s as usize].name)) == "SimdShuffleIdx"));
                    if is_ctor || is_idx {
                        expr = ExprId(arg);
                    } else {
                        return expr;
                    }
                }
                Expr::Struct { path, fields, base: None, .. } => {
                    let &[field] = body.list(fields) else { return expr };
                    let is_idx = body.list(body.path(path).segments).last().is_some_and(|&s| plain(src.text(body.segments[s as usize].name)) == "SimdShuffleIdx");
                    if is_idx {
                        expr = body.fields[field as usize].value;
                    } else {
                        return expr;
                    }
                }
                _ => return expr,
            }
        }
    }

    fn body_ty_int(&self, body: &Body, src: &Source, mut ty: crate::body::TyId) -> Option<(u32, bool)> {
        loop {
            match *body.ty(ty) {
                BodyTy::Paren(inner) => ty = inner,
                BodyTy::Path(path) => {
                    let entry = body.path(path);
                    if entry.global || entry.qself.is_some() {
                        return None;
                    }
                    let &[seg] = body.list(entry.segments) else { return None };
                    return Primitive::named(plain(src.text(body.segments[seg as usize].name))).and_then(int_info);
                }
                _ => return None,
            }
        }
    }

    fn origin_int(&self, expr: ExprId) -> Option<(u32, bool)> {
        match *self.cx.types.kind(self.typed.exprs[expr.0 as usize]) {
            Kind::Prim(prim) => int_info(prim),
            _ => None,
        }
    }

    fn int(&mut self, body: &Body, src: &Source, expr: ExprId, env: &mut Vec<(String, ConstVal)>, depth: u8) -> Option<ConstVal> {
        if depth > 8 {
            return None;
        }
        let expr = self.unwrap_expr(body, expr);
        match *body.expr(expr) {
            Expr::Lit(Lit::Int, tok) => {
                let (val, suffix) = parse_int(src.text(tok)).ok()?;
                let info = suffix.and_then(int_info).or_else(|| std::ptr::eq(body, self.origin_body).then(|| self.origin_int(expr)).flatten());
                let (bits, signed, typed) = match info {
                    Some((b, s)) => (b, s, true),
                    None => (64, false, false),
                };
                Some(ConstVal { val: val & int_mask(bits), bits, signed, typed })
            }
            Expr::Lit(Lit::Bool(flag), _) => Some(ConstVal { val: u128::from(flag), bits: 1, signed: false, typed: true }),
            Expr::Unary(UnOp::Neg, inner) => {
                let v = self.int(body, src, inner, env, depth + 1)?;
                Some(ConstVal { val: v.val.wrapping_neg() & int_mask(v.bits), ..v })
            }
            Expr::Unary(UnOp::Not, inner) => {
                let v = self.int(body, src, inner, env, depth + 1)?;
                Some(ConstVal { val: (!v.val) & int_mask(v.bits), ..v })
            }
            Expr::Cast(inner, ty) => {
                let v = self.int(body, src, inner, env, depth + 1)?;
                let target = self.body_ty_int(body, src, ty)?;
                Some(v.cast(target))
            }
            Expr::Binary(op, lhs, rhs) => self.binary(body, src, (op, lhs, rhs), env, depth + 1),
            Expr::If(cond, then, Some(otherwise)) => {
                let flag = self.int(body, src, cond, env, depth + 1)?;
                self.int(body, src, if flag.val != 0 { then } else { otherwise }, env, depth + 1)
            }
            Expr::Block { stmts, tail: Some(tail), kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const | BlockKind::Expansion, .. } => {
                let mark = self.eval_stmts(body, src, stmts, env, depth + 1)?;
                let out = self.int(body, src, tail, env, depth + 1);
                env.truncate(mark);
                out
            }
            Expr::Index(arr, idx_expr) => {
                let idx_val = self.int(body, src, idx_expr, env, depth + 1)?;
                let idx = usize::try_from(idx_val.val).ok()?;
                self.index_array(body, src, arr, idx, env, depth + 1)
            }
            Expr::Call(callee, list) if std::ptr::eq(body, self.origin_body) => {
                let Res::Fn(item, substs) = self.typed.res.get(&callee).or_else(|| self.typed.res.get(&expr)).copied()? else { return None };
                if self.cx.item(item).1.kind != ItemKind::Fn || !self.cx.types.substs(substs).is_empty() || self.cx.intrinsic(item).is_some() {
                    return None;
                }
                let sig = self.cx.signature(item).ok()?;
                let arg_ids = body.list(list);
                if sig.params.len() != arg_ids.len() {
                    return None;
                }
                let item_body = self.cx.body(item).ok()?;
                let root = item_body.root?;
                let item_src = self.cx.body_src(item);
                let mut sub_env = Vec::with_capacity(arg_ids.len());
                for (&(mut pat, pty), &arg_id) in sig.params.iter().zip(arg_ids) {
                    while let Pat::Paren(inner) = *sig.body.pat(pat) {
                        pat = inner;
                    }
                    let Pat::Bind { name, sub: None, .. } = *sig.body.pat(pat) else { return None };
                    let mut val = self.int(body, src, ExprId(arg_id), env, depth + 1)?;
                    if let Some(target) = self.body_ty_int(&sig.body, item_src, pty) {
                        val = val.cast(target);
                    }
                    sub_env.push((plain(item_src.text(name)).to_owned(), val));
                }
                let mut res = self.int(&item_body, item_src, root, &mut sub_env, depth + 1)?;
                if let Some(target) = sig.ret.and_then(|rt| self.body_ty_int(&sig.body, item_src, rt)) {
                    res = res.cast(target);
                }
                Some(res)
            }
            Expr::Path(path) => {
                let entry = body.path(path);
                if !entry.global
                    && entry.qself.is_none()
                    && let &[seg] = body.list(entry.segments)
                {
                    let name = plain(src.text(body.segments[seg as usize].name));
                    if let Some((_, val)) = env.iter().rev().find(|(bound, _)| bound == name) {
                        return Some(*val);
                    }
                }
                if !std::ptr::eq(body, self.origin_body) {
                    return None;
                }
                match self.typed.res.get(&expr).copied()? {
                    Res::Const(val) => match *self.cx.types.kind(val) {
                        Kind::Value(n) => {
                            let (bits, signed) = self.origin_int(expr).unwrap_or((64, false));
                            Some(ConstVal { val: n & int_mask(bits), bits, signed, typed: true })
                        }
                        _ => None,
                    },
                    Res::Fn(item, substs) if self.cx.item(item).1.kind == ItemKind::Const && self.cx.types.substs(substs).is_empty() => {
                        let item_body = self.cx.body(item).ok()?;
                        let root = item_body.root?;
                        let item_src = self.cx.body_src(item);
                        self.int(&item_body, item_src, root, &mut Vec::new(), depth + 1)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn binary(&mut self, body: &Body, src: &Source, (op, lhs, rhs): (BinOp, ExprId, ExprId), env: &mut Vec<(String, ConstVal)>, depth: u8) -> Option<ConstVal> {
        let mut l = self.int(body, src, lhs, env, depth)?;
        if matches!(op, BinOp::And | BinOp::Or) {
            let short = (l.val != 0) == (op == BinOp::Or);
            let out = if short { l.val != 0 } else { self.int(body, src, rhs, env, depth)?.val != 0 };
            return Some(ConstVal { val: u128::from(out), bits: 1, signed: false, typed: true });
        }
        let mut r = self.int(body, src, rhs, env, depth)?;
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            let sh = u32::try_from(r.val).ok()?;
            let val = match op {
                BinOp::Shl => l.val.checked_shl(sh).unwrap_or(0) & int_mask(l.bits),
                _ if l.signed => ((l.to_signed() >> sh.min(127)) as u128) & int_mask(l.bits),
                _ => l.val.checked_shr(sh).unwrap_or(0),
            };
            return Some(ConstVal { val, ..l });
        }
        let (bits, signed, typed) = match (l.typed, r.typed) {
            (true, _) => (l.bits, l.signed, true),
            (false, true) => (r.bits, r.signed, true),
            (false, false) => (64, false, false),
        };
        l = l.cast((bits, signed));
        r = r.cast((bits, signed));
        if matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
            let ord = if signed { l.to_signed().cmp(&r.to_signed()) } else { l.val.cmp(&r.val) };
            let holds = match op {
                BinOp::Eq => ord.is_eq(),
                BinOp::Ne => ord.is_ne(),
                BinOp::Lt => ord.is_lt(),
                BinOp::Le => ord.is_le(),
                BinOp::Gt => ord.is_gt(),
                _ => ord.is_ge(),
            };
            return Some(ConstVal { val: u128::from(holds), bits: 1, signed: false, typed: true });
        }
        if matches!(op, BinOp::Div | BinOp::Rem) && r.val == 0 {
            return None;
        }
        let val = match op {
            BinOp::Add => l.val.wrapping_add(r.val),
            BinOp::Sub => l.val.wrapping_sub(r.val),
            BinOp::Mul => l.val.wrapping_mul(r.val),
            BinOp::Div if signed => l.to_signed().wrapping_div(r.to_signed()) as u128,
            BinOp::Rem if signed => l.to_signed().wrapping_rem(r.to_signed()) as u128,
            BinOp::Div => l.val / r.val,
            BinOp::Rem => l.val % r.val,
            BinOp::BitAnd => l.val & r.val,
            BinOp::BitOr => l.val | r.val,
            BinOp::BitXor => l.val ^ r.val,
            _ => return None,
        };
        Some(ConstVal { val: val & int_mask(bits), bits, signed, typed })
    }

    fn index_array(&mut self, body: &Body, src: &Source, arr: ExprId, idx: usize, env: &mut Vec<(String, ConstVal)>, depth: u8) -> Option<ConstVal> {
        let arr = self.unwrap_expr(body, arr);
        match *body.expr(arr) {
            Expr::Array(list) => {
                let &elem = body.list(list).get(idx)?;
                self.int(body, src, ExprId(elem), env, depth)
            }
            Expr::Repeat(elem, count_expr) => {
                let count = self.int(body, src, count_expr, env, depth)?;
                (u128::from(idx as u64) < count.val).then(|| self.int(body, src, elem, env, depth)).flatten()
            }
            Expr::Path(_) if std::ptr::eq(body, self.origin_body) => {
                let Res::Fn(item, substs) = self.typed.res.get(&arr).copied()? else { return None };
                if self.cx.item(item).1.kind != ItemKind::Const || !self.cx.types.substs(substs).is_empty() {
                    return None;
                }
                let item_body = self.cx.body(item).ok()?;
                let root = item_body.root?;
                let item_src = self.cx.body_src(item);
                self.index_array(&item_body, item_src, root, idx, &mut Vec::new(), depth)
            }
            _ => None,
        }
    }

    fn eval_stmts(&mut self, body: &Body, src: &Source, stmts: crate::body::List, env: &mut Vec<(String, ConstVal)>, depth: u8) -> Option<usize> {
        let mark = env.len();
        for &stmt_id in body.list(stmts) {
            match body.stmts[stmt_id as usize] {
                Stmt::Empty | Stmt::Item(..) => {}
                Stmt::Let { mut pat, ty, init: Some(init), otherwise: None, .. } => {
                    while let Pat::Paren(inner) = *body.pat(pat) {
                        pat = inner;
                    }
                    let Pat::Bind { name, sub: None, .. } = *body.pat(pat) else {
                        env.truncate(mark);
                        return None;
                    };
                    let Some(mut val) = self.int(body, src, init, env, depth) else {
                        env.truncate(mark);
                        return None;
                    };
                    if let Some(target) = ty.and_then(|t| self.body_ty_int(body, src, t)) {
                        val = val.cast(target);
                    }
                    env.push((plain(src.text(name)).to_owned(), val));
                }
                _ => {
                    env.truncate(mark);
                    return None;
                }
            }
        }
        Some(mark)
    }

    fn shuffle_array(&mut self, body: &Body, src: &Source, expr: ExprId, (in_count, out_count): (usize, usize), env: &mut Vec<(String, ConstVal)>, depth: u8) -> Option<[u8; 64]> {
        if depth > 4 {
            return None;
        }
        let expr = self.unwrap_shuffle(body, src, expr);
        let limit = u128::try_from(2 * in_count).ok()?;
        let mut mask = [0u8; 64];
        match *body.expr(expr) {
            Expr::Array(list) => {
                let ids = body.list(list);
                if ids.len() != out_count {
                    return None;
                }
                mask[..out_count].iter_mut().zip(ids).try_for_each(|(slot, &id)| {
                    let v = self.int(body, src, ExprId(id), env, depth + 1)?;
                    ((!v.signed || v.to_signed() >= 0) && v.val < limit).then(|| *slot = v.val as u8)
                })?;
                Some(mask)
            }
            Expr::Repeat(elem, _) => {
                let v = self.int(body, src, elem, env, depth + 1)?;
                if (v.signed && v.to_signed() < 0) || v.val >= limit {
                    return None;
                }
                mask[..out_count].fill(v.val as u8);
                Some(mask)
            }
            Expr::Block { stmts, tail: Some(tail), kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const | BlockKind::Expansion, .. } => {
                let mark = self.eval_stmts(body, src, stmts, env, depth + 1)?;
                let out = self.shuffle_array(body, src, tail, (in_count, out_count), env, depth + 1);
                env.truncate(mark);
                out
            }
            Expr::Path(_) if std::ptr::eq(body, self.origin_body) => {
                let Res::Fn(item, substs) = self.typed.res.get(&expr).copied()? else { return None };
                if self.cx.item(item).1.kind != ItemKind::Const || !self.cx.types.substs(substs).is_empty() {
                    return None;
                }
                let item_body = self.cx.body(item).ok()?;
                let root = item_body.root?;
                let item_src = self.cx.body_src(item);
                self.shuffle_array(&item_body, item_src, root, (in_count, out_count), &mut Vec::new(), depth + 1)
            }
            _ => None,
        }
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
