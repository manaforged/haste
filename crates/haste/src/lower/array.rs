use anyhow::{Result, bail};
use haste_codegen::{Arch, IntCC, RuntimeSymbol, Type, Value};

use crate::body::{BlockKind, Body, Expr, ExprId, List};
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{Res, SiteKey, Typed};
use crate::sema::ty::{Kind, Ty};

use super::place::{Base, Place};
use super::{Lower, Repr, repr};

fn splat_multiplier(ty: Type) -> Option<u64> {
    match ty {
        Type::I8 => Some(0x0101_0101_0101_0101),
        Type::I16 => Some(0x0001_0001_0001_0001),
        Type::I32 => Some(0x0000_0001_0000_0001),
        _ => None,
    }
}

fn splat_word64(ty: Type, imm: i64) -> Option<u64> {
    let bits = match ty {
        Type::I8 => u64::from(imm as u8),
        Type::I16 => u64::from(imm as u16),
        Type::I32 => u64::from(imm as u32),
        _ => return None,
    };
    splat_multiplier(ty).map(|mul| bits.wrapping_mul(mul))
}

/// Whether `expr` (or `callee`) resolves to `MaybeUninit::uninit()`.
pub(super) fn is_maybe_uninit_fn(cx: &Context<'_>, typed: &Typed, expr: ExprId, callee: ExprId) -> bool {
    let Some((item, _)) = typed.site_fn(SiteKey::ExprRes(expr)).or_else(|| typed.site_fn(SiteKey::ExprRes(callee))) else {
        return false;
    };
    if cx.lang(item) == Some("maybe_uninit_uninit") {
        return true;
    }
    let val_ty = typed.expr_ty(&cx.types, expr);
    val_ty != Ty::NONE && cx.name(item) == "uninit" && matches!(*cx.types.kind(val_ty), Kind::Adt(adt, _) if cx.lang(adt) == Some("maybe_uninit"))
}

/// Extracts `(call_expr, callee)` when `expr` is a zero-argument call or `const { call() }`.
pub(super) fn uninit_call_site(body: &Body, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let expr = match *body.expr(expr) {
        Expr::Block { stmts, tail: Some(tail), kind: BlockKind::Const, .. } if body.list(stmts).is_empty() => tail,
        _ => expr,
    };
    let Expr::Call(callee, args) = *body.expr(expr) else { return None };
    body.list(args).is_empty().then_some((expr, callee))
}

impl Lower<'_, '_> {
    /// Lowers `[items]` into a new frame slot and returns its address.
    pub fn array(&mut self, expr: ExprId, items: List) -> Result<Option<Value>> {
        let ty = self.ty_of(expr);
        let whole = repr(self.cx, ty)?;
        let slot = self.slot(whole);
        let Kind::Array(element, _) = *self.cx.types.kind(ty) else { bail!("array literal of a type that is not an array") };
        let (stride, part) = (self.cx.layout(element)?.size.bytes() as i32, repr(self.cx, element)?);
        let unwind = self.wants_field_unwind()?;
        let mut guards = Vec::new();
        for (at, item) in self.body.list(items).to_vec().into_iter().enumerate() {
            let elem_place = slot.map(|slot| Place::slot(slot, whole).at(at as i32 * stride, part));
            if !self.eval_and_write_part(elem_place, element, ExprId(item), unwind, &mut guards)? {
                return Ok(None);
            }
        }
        self.disarm_owned_guards(guards);
        Ok(slot.and_then(|slot| self.read(Place::slot(slot, whole))))
    }

    /// Lowers `[value; len]` into a new frame slot and returns its address.
    pub fn repeat(&mut self, expr: ExprId, value: ExprId) -> Result<Option<Value>> {
        let ty = self.ty_of(expr);
        let whole = repr(self.cx, ty)?;
        let Kind::Array(element, count) = *self.cx.types.kind(ty) else { bail!("repeat of a type that is not an array") };
        let (stride, part) = (self.cx.layout(element)?.size.bytes() as i32, repr(self.cx, element)?);
        let n = self.len(count)?;
        let total_bytes = n * i64::from(stride);
        let uninit = self.uninit_value(value)?;
        if !uninit && n > 16 && part != Repr::Scalar(Type::I8) && total_bytes >= 32 && self.literal_is_zero(value)? {
            return self.zero_repeat(whole, total_bytes);
        }
        let value = if uninit { None } else { self.expr(value)? };
        let Some(slot) = self.slot(whole) else {
            return Ok(None);
        };
        let base = Place::slot(slot, whole);
        if let Some(val) = value
            && self.open
        {
            self.fill_repeat(base, part, (n, stride, total_bytes), val)?;
        }
        Ok(self.read(base))
    }

    fn zero_repeat(&mut self, whole: Repr, total_bytes: i64) -> Result<Option<Value>> {
        let Some(slot) = self.slot(whole) else {
            return Ok(None);
        };
        let base = Place::slot(slot, whole);
        if self.open {
            let dst = self.address(base);
            let byte = self.ins().iconst(Type::I64, 0);
            let len = self.ins().iconst(Type::I64, total_bytes);
            self.call_libc(RuntimeSymbol::Memset, vec![dst, byte, len], None)?;
        }
        Ok(self.read(base))
    }

    fn fill_repeat(&mut self, base: Place, part: Repr, (n, stride, total_bytes): (i64, i32, i64), val: Value) -> Result<()> {
        if self.try_splat_repeat(base, part, (n, stride, total_bytes), val) {
            return Ok(());
        }
        if n <= 16 {
            for at in 0..n as i32 {
                self.write(base.at(at * stride, part), val);
            }
            return Ok(());
        }
        let dst = self.address(base);
        if part == Repr::Scalar(Type::I8) {
            let byte = self.ins().uextend(Type::I64, val);
            let len = self.ins().iconst(Type::I64, total_bytes);
            self.call_libc(RuntimeSymbol::Memset, vec![dst, byte, len], None)?;
            return Ok(());
        }
        let count_val = self.ins().iconst(Type::I64, n);
        self.each(count_val, |lower, idx| {
            let off = lower.ins().imul_imm(idx, i64::from(stride));
            let elem_addr = lower.ins().iadd(dst, off);
            lower.write(Place { base: Base::Addr(elem_addr), offset: 0, repr: part }, val);
        });
        Ok(())
    }

    fn try_splat_repeat(&mut self, base: Place, part: Repr, (n, stride, total_bytes): (i64, i32, i64), val: Value) -> bool {
        let Repr::Scalar(elem_ty) = part else { return false };
        let Some(mul) = splat_multiplier(elem_ty) else { return false };
        if !matches!(crate::compile_target().arch, Arch::X64 | Arch::Arm64) || stride != elem_ty.bytes() as i32 || total_bytes < 2 || total_bytes > 32 {
            return false;
        }
        let total = total_bytes as usize;
        if let Some(imm) = self.b.const_int(val)
            && let Some(word64) = splat_word64(elem_ty, imm)
        {
            let mut c64 = None;
            self.write_splat_chunks(base, total, val, |lower, ty| match ty {
                Type::I64 => *c64.get_or_insert_with(|| lower.ins().iconst(Type::I64, word64 as i64)),
                _ => lower.ins().iconst(ty, word64 as i64),
            });
            return true;
        }
        if n >= 8 && matches!(elem_ty, Type::I8 | Type::I16) {
            let wide = self.ins().uextend(Type::I64, val);
            let splat = self.ins().imul_imm(wide, mul as i64);
            self.write_splat_chunks(base, total, val, |lower, ty| if ty == Type::I64 { splat } else { lower.ins().ireduce(ty, splat) });
            return true;
        }
        false
    }

    fn write_splat_chunks(&mut self, base: Place, total: usize, orig: Value, mut chunk: impl FnMut(&mut Self, Type) -> Value) {
        let mut off = 0usize;
        for (size, ty) in [(8, Type::I64), (4, Type::I32), (2, Type::I16)] {
            while total - off >= size {
                let v = chunk(self, ty);
                self.write(base.at(off as i32, Repr::Scalar(ty)), v);
                off += size;
            }
        }
        if off < total {
            self.write(base.at(off as i32, Repr::Scalar(Type::I8)), orig);
        }
    }

    /// Whether `expr` is `MaybeUninit::uninit()`, whose bytes are undefined, so a repeat or binding of it need not store them.
    pub(super) fn uninit_call(&self, expr: ExprId) -> bool {
        uninit_call_site(self.body, expr).is_some_and(|(e, callee)| is_maybe_uninit_fn(self.cx, self.typed, e, callee))
    }

    /// Whether the aggregate `expr` holds bytes from `MaybeUninit::uninit()`, which a literal blob would pin to zero.
    pub(super) fn holds_uninit(&mut self, expr: ExprId) -> Result<bool> {
        if self.uninit_value(expr)? {
            return Ok(true);
        }
        let parts: Vec<ExprId> = match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Repeat(inner, _) | Expr::Block { tail: Some(inner), .. } => vec![inner],
            Expr::Array(items) | Expr::Tuple(items) => self.body.list(items).iter().map(|&item| ExprId(item)).collect(),
            Expr::Struct { fields, .. } => self.body.list(fields).iter().map(|&at| self.body.fields[at as usize].value).collect(),
            _ => Vec::new(),
        };
        for part in parts {
            if self.holds_uninit(part)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether `expr` is `MaybeUninit::uninit()` or names a constant whose body is, so a repeat of it need not store its bytes.
    fn uninit_value(&mut self, expr: ExprId) -> Result<bool> {
        if self.uninit_call(expr) {
            return Ok(true);
        }
        let (Expr::Path(_), Some(&Res::Fn(item, substs))) = (self.body.expr(expr), self.typed.res.get(&expr)) else {
            return Ok(false);
        };
        if self.cx.item(item).1.kind != ItemKind::Const {
            return Ok(false);
        }
        let Some((source, typed)) = self.literal_source(item, substs)? else {
            return Ok(false);
        };
        let typed = typed.scoped(&self.cx.types);
        let body = self.cx.body(source)?;
        Ok(body.root.and_then(|root| uninit_call_site(&body, root)).is_some_and(|(e, callee)| is_maybe_uninit_fn(self.cx, &typed, e, callee)))
    }

    /// The place that `base[index]` names after the autoderef steps of `base`: an element of an array or slice after a bounds check, or the target of `Index::index`.
    pub fn index_place(&mut self, expr: ExprId, base: ExprId, index: ExprId) -> Result<Option<Place>> {
        let found = self.autoderef_place(expr, base)?;
        if let Some(&Res::Fn(item, substs)) = self.typed.operators.get(&expr) {
            let receiver = match found {
                Some((place, _)) => self.address(place),
                None => self.address_or_dangling(None, self.ty_of(base))?,
            };
            let values = self.expr(index)?.into_iter();
            let Some(pointer) = self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::Operator(expr)), item, substs, std::iter::once(receiver).chain(values).collect())? else { return Ok(None) };
            return self.pointee_place(pointer, self.ty_of(expr)).map(Some);
        }
        let Some((place, ty)) = found else { return self.empty_index(expr, base, index) };
        if let Kind::Array(_, count) = *self.cx.types.kind(ty) {
            return self.array_index_place(place, count, index, self.ty_of(expr));
        }
        let Some((data, len)) = self.data_len(Some(place), ty)? else {
            return Ok(None);
        };
        let Some(at) = self.expr(index)? else {
            return Ok(None);
        };
        self.check_bounds(at, len)?;
        self.element_place(data, at, self.ty_of(expr)).map(Some)
    }

    fn array_index_place(&mut self, place: Place, count: Ty, index: ExprId, elem_ty: Ty) -> Result<Option<Place>> {
        let Some(at) = self.expr(index)? else { return Ok(None) };
        let count = self.len(count)?;
        let stride = self.cx.layout(elem_ty)?.size.bytes() as i64;
        if let Some(idx) = self.b.const_int(at)
            && (idx as u64) < (count as u64)
            && let Some(off) = idx.checked_mul(stride).and_then(|o| i32::try_from(o).ok())
            && place.offset.checked_add(off).is_some()
        {
            return Ok(Some(place.at(off, repr(self.cx, elem_ty)?)));
        }
        let len = self.ins().iconst(Type::I64, count);
        self.check_bounds(at, len)?;
        let data = self.address(place);
        self.element_place(data, at, elem_ty).map(Some)
    }

    /// The place of element `index` of the array `base` that has no bytes: a dangling one past a bounds check, which rustc also keeps.
    fn empty_index(&mut self, expr: ExprId, base: ExprId, index: ExprId) -> Result<Option<Place>> {
        let ty = self.ty_of(base);
        let Kind::Array(_, count) = *self.cx.types.kind(ty) else {
            return Ok(None);
        };
        let Some(at) = self.expr(index)? else {
            return Ok(None);
        };
        let count = self.len(count)?;
        let len = self.ins().iconst(Type::I64, count);
        self.check_bounds(at, len)?;
        let data = self.address_or_dangling(None, ty)?;
        self.element_place(data, at, self.ty_of(expr)).map(Some)
    }

    fn check_bounds(&mut self, at: Value, len: Value) -> Result<()> {
        let inside = self.ins().icmp(IntCC::UnsignedLessThan, at, len);
        self.check_with(inside, |lower| lower.panic_bounds(at, len))
    }

    /// The data pointer and length of the array or slice of type `ty` at `place`; `None` when it has no bytes.
    pub(super) fn data_len(&mut self, place: Option<Place>, ty: Ty) -> Result<Option<(Value, Value)>> {
        if let Kind::Array(_, count) = *self.cx.types.kind(ty) {
            let Some(at) = place else { return Ok(None) };
            let data = self.address(at);
            let count = self.len(count)?;
            return Ok(Some((data, self.ins().iconst(Type::I64, count))));
        }
        let Some(at) = place else { bail!("slice with no wide pointer") };
        let (data, len) = match at.base {
            Base::Slot(slot) => (self.ins().stack_load(Type::I64, slot, at.offset), self.ins().stack_load(Type::I64, slot, at.offset + 8)),
            Base::Addr(wide) => {
                let data = self.ins().load(Type::I64, wide, at.offset);
                let len = match self.b.wide_const_len(wide).and_then(|n| i64::try_from(n).ok()).filter(|_| at.offset == 0) {
                    Some(n) => self.ins().iconst(Type::I64, n),
                    None => self.ins().load(Type::I64, wide, at.offset + 8),
                };
                (data, len)
            }
        };
        Ok(Some((data, len)))
    }

    /// The place of element `index` of type `element` in the array data at `data`.
    pub(super) fn element_place(&mut self, data: Value, index: Value, element: Ty) -> Result<Place> {
        let stride = self.cx.layout(element)?.size.bytes() as i64;
        if stride == 0 {
            return Ok(Place { base: Base::Addr(data), offset: 0, repr: repr(self.cx, element)? });
        }
        if let Some(idx) = self.b.const_int(index)
            && let Some(off) = idx.checked_mul(stride).and_then(|o| i32::try_from(o).ok())
        {
            return Ok(Place { base: Base::Addr(data), offset: off, repr: repr(self.cx, element)? });
        }
        let address = match stride {
            1 => self.ins().iadd(data, index),
            s if s > 1 && (s as u64).is_power_of_two() => {
                let shift = self.ins().iconst(Type::I64, i64::from((s as u64).trailing_zeros()));
                let offset = self.ins().ishl(index, shift);
                self.ins().iadd(data, offset)
            }
            _ => {
                let offset = self.ins().imul_imm(index, stride);
                self.ins().iadd(data, offset)
            }
        };
        Ok(Place { base: Base::Addr(address), offset: 0, repr: repr(self.cx, element)? })
    }
}
