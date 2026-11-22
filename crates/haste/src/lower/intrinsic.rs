use anyhow::{Result, bail};
use haste_codegen::{AbiParam, CallConv, FloatCC, IntCC, Signature, Type, Value};

use crate::body::BinOp;
use crate::host::HostSymbol;
use crate::names::Primitive;
use crate::sema::intrinsic::Intrinsic;
use crate::sema::ty::{Kind, Ty};

use super::ops::Num;
use super::place::{Base, Place};
use super::reloc::SiteOrigin;
use super::stencil::{RodataReloc, RodataTarget, SymbolSource};
use super::{Lower, Repr, host_is_x64_or_arm64, repr};

impl Lower<'_, '_> {
    /// Emits intrinsic `which` at type arguments `substs` on `args` inline.
    pub(super) fn intrinsic(&mut self, origin: &SiteOrigin, which: Intrinsic, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        match which {
            Intrinsic::Transmute => self.transmute(substs, args),
            Intrinsic::FloatToInt => {
                let (&[from, to], &[value]) = (substs, args) else { bail!("float_to_int_unchecked takes one float") };
                let (from, to) = (self.num(from)?, self.num(to)?);
                self.convert(value, from, to).map(Some)
            }
            Intrinsic::PtrMetadata => self.metadata(substs, args),
            Intrinsic::ConstEvalSelect => self.const_eval_select(origin, substs, args),
            Intrinsic::SelectUnpredictable => self.select_unpredictable(substs, args),
            Intrinsic::WithOverflow(op) => self.with_overflow(op, substs, args).map(Some),
            Intrinsic::Simd(op) => self.simd(origin, op, substs, args),
            Intrinsic::Trap => {
                self.ins().trap();
                self.close();
                Ok(None)
            }
            Intrinsic::Unreachable => {
                self.unreachable();
                Ok(None)
            }
            Intrinsic::Assume => {
                let &[cond] = args else { bail!("assume takes one bool") };
                self.ins().assume(cond);
                Ok(None)
            }
            _ => self.extra_intrinsic(origin, which, substs, args),
        }
    }

    fn extra_intrinsic(&mut self, origin: &SiteOrigin, which: Intrinsic, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        match which {
            Intrinsic::CatchUnwind => self.catch_unwind(args).map(Some),
            Intrinsic::CallerLocation => self.caller_location().map(Some),
            Intrinsic::Cpuid => self.cpuid(args).map(Some),
            Intrinsic::FnAddr => Ok(args.first().copied()),
            Intrinsic::ShuffleDyn { tbl } => self.shuffle_dyn(args, tbl).map(Some),
            Intrinsic::BytePermute32 => self.byte_permute_32(args).map(Some),
            Intrinsic::AvgRound(ty) => self.avg_round_vec(ty, args).map(Some),
            Intrinsic::Umaxp => self.umaxp_vec(args).map(Some),
            Intrinsic::Sqdmulh(ty) => self.sqdmulh_vec(ty, args).map(Some),
            Intrinsic::TargetOp { id, ty } => self.target_vector(id, ty, args).map(Some),
            Intrinsic::X86(x86) => self.host_vector(x86, args).map(Some),
            Intrinsic::CompareBytes => self.compare_const_bytes(origin, substs, args),
            _ => self.memory_intrinsic(origin, which, substs, args),
        }
    }

    /// Ends the block at a point that control never reaches: an assumption that is false there, then a trap.
    pub(super) fn unreachable(&mut self) {
        let never = self.ins().iconst(Type::I8, 0);
        self.ins().assume(never);
        self.ins().trap();
        self.close();
    }

    /// Calls the host function `host` in the platform convention with `args`, and returns its result of machine type `ret`.
    pub(super) fn call_host(&mut self, host: HostSymbol, args: &[Value], ret: Option<Type>) -> Option<Value> {
        let code = self.emit_symbol(SymbolSource::Host(host));
        let mut signature = Signature::new(CallConv::Native);
        signature.params.extend(args.iter().map(|&arg| AbiParam::new(self.b.func.ty(arg))));
        signature.returns.extend(ret.map(AbiParam::new));
        let sig = self.b.import_signature(signature);
        let call = self.ins().call_indirect(sig, code, args);
        self.b.inst_results(call).first().copied()
    }

    pub(super) fn caller_loc_slot(&mut self) -> Result<Value> {
        const CALLER_LOC_SLOT: i64 = 0;
        const CALLER_LOC_CELL_BYTES: i64 = 8;
        let slot = self.ins().iconst(Type::I64, CALLER_LOC_SLOT);
        self.call_thread_cell(slot, CALLER_LOC_CELL_BYTES, None)
    }

    pub(super) fn current_site_location(&mut self) -> Result<Value> {
        let Some(location) = self.cx.lang_item("panic_location") else { bail!("no `Location` lang item") };
        let ty = self.cx.types.adt(location, &[]);
        let layout = self.cx.layout(ty)?;
        let src = self.src;
        let root = src.base.as_deref().map_or(src, |f| &f.source);
        let (line, column) = match self.b.loc() {
            0 => {
                let (item_src, entry) = self.cx.item(self.reached.list[self.index].0.0);
                if item_src.path == src.path {
                    let tok = entry.name.unwrap_or(entry.start);
                    (item_src.line(tok) as u32, item_src.column(tok) as u32)
                } else {
                    (1, 1)
                }
            }
            loc => {
                let mut byte = (loc as u32 as usize).min(root.text.len());
                while byte > 0 && !root.text.is_char_boundary(byte) {
                    byte -= 1;
                }
                let before = &root.text[..byte];
                let line = before.matches('\n').count() as u32 + 1;
                let col = before[before.rfind('\n').map_or(0, |n| n + 1)..].chars().count() as u32 + 1;
                (line, col)
            }
        };
        let mut file = root.path.display().to_string().into_bytes();
        let len = file.len() as u64;
        file.push(0);
        let file_id = self.slots.rodata(&file, &[]);
        let mut bytes = vec![0u8; layout.size.bytes() as usize];
        let mut put = |field: u32, at: i32, value: &[u8]| {
            let from = (crate::sema::layout::field_offset(&layout, 0, field) + at) as usize;
            bytes[from..from + value.len()].copy_from_slice(value);
        };
        put(0, 8, &len.to_le_bytes());
        put(1, 0, &line.to_le_bytes());
        put(2, 0, &column.to_le_bytes());
        let ptr_offset = crate::sema::layout::field_offset(&layout, 0, 0) as u32;
        let loc_id = self.slots.rodata(&bytes, &[RodataReloc { offset: ptr_offset, target: RodataTarget::Rodata(file_id) }]);
        Ok(self.emit_symbol(SymbolSource::Rodata(loc_id)))
    }

    pub(super) fn expr_site_place(&self, expr: crate::body::ExprId) -> Option<u64> {
        self.expr_site_token(expr)
            .map(|tok| self.src.place(tok))
            .filter(|&loc| loc != 0)
            .or_else(|| self.body.expr_ends.get(expr.0 as usize).map(|&end| self.src.place(end.saturating_sub(1))).filter(|&loc| loc != 0))
    }

    fn expr_site_token(&self, expr: crate::body::ExprId) -> Option<u32> {
        use crate::body::Expr;
        match self.body.exprs[expr.0 as usize] {
            Expr::Method { name, .. } => Some(name),
            Expr::Call(callee, _) | Expr::Index(callee, _) => self.expr_site_token(callee),
            Expr::Path(path) => Some(self.body.paths[path.0 as usize].start),
            _ => None,
        }
    }

    pub(super) fn enter_tracked_caller_loc(&mut self) -> Result<()> {
        let slot = self.caller_loc_slot()?;
        let prev = self.ins().load(Type::I64, slot, 0);
        let has = self.ins().icmp_imm(IntCC::NotEqual, prev, 0);
        let own = self.current_site_location()?;
        let cur = self.ins().select(has, prev, own);
        self.ins().store(cur, slot, 0);
        self.caller_loc_guard = Some((slot, prev));
        Ok(())
    }

    pub(super) fn wrap_inlined_caller_loc(&mut self, from_blk: usize, to_blk: usize) -> Result<()> {
        for b_idx in from_blk..to_blk {
            let block = self.b.block_at(b_idx);
            let Some((first_call, can_fall_through)) = self.b.block_first_tracked(block, self.dict_param, |sym, arg0| match self.relocs.get(sym as usize) {
                Some(SymbolSource::Slot(idx)) => self.cx.track_caller(self.reached.list[*idx].0.0),
                Some(SymbolSource::Host(HostSymbol::ThreadCell)) => arg0 == Some(0),
                _ => false,
            }) else {
                continue;
            };
            let (saved, tail) = self.b.splice_open(block, first_call);
            let slot = self.caller_loc_slot()?;
            let prev = self.ins().load(Type::I64, slot, 0);
            let loc = self.current_site_location()?;
            self.ins().store(loc, slot, 0);
            if can_fall_through && let Some((&term, mid)) = tail.split_last() {
                self.b.splice_append(mid, None);
                self.ins().store(prev, slot, 0);
                self.b.splice_append(&[term], saved);
            } else {
                self.b.splice_append(&tail, saved);
            }
        }
        Ok(())
    }

    pub(super) fn with_caller_loc<T>(&mut self, track: Option<bool>, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let Some(set_site) = track else {
            return run(self);
        };
        let slot = self.caller_loc_slot()?;
        let prev = self.ins().load(Type::I64, slot, 0);
        let next = if set_site { self.current_site_location()? } else { self.ins().iconst(Type::I64, 0) };
        self.ins().store(next, slot, 0);
        let res = run(self)?;
        if self.open {
            self.ins().store(prev, slot, 0);
        }
        Ok(res)
    }

    fn caller_location(&mut self) -> Result<Value> {
        let fallback = self.current_site_location()?;
        let slot = self.caller_loc_slot()?;
        let cur = self.ins().load(Type::I64, slot, 0);
        let has = self.ins().icmp_imm(IntCC::NotEqual, cur, 0);
        Ok(self.ins().select(has, cur, fallback))
    }

    fn catch_unwind(&mut self, args: &[Value]) -> Result<Value> {
        let &[try_fn, data, catch_fn] = args else { bail!("catch_unwind takes three arguments") };
        let res = self.call_host(HostSymbol::CatchUnwind, &[try_fn, data, catch_fn], Some(Type::I32)).expect("invariant: catch_unwind returns i32");
        if self.open {
            let caught = self.b.create_block();
            let done = self.b.create_block();
            self.brif(res, (caught, &[]), (done, &[]));
            self.enter(caught);
            let (slot, restore) = match self.caller_loc_guard {
                Some((slot, prev)) => (slot, prev),
                None => (self.caller_loc_slot()?, self.ins().iconst(Type::I64, 0)),
            };
            self.ins().store(restore, slot, 0);
            self.jump(done, &[]);
            self.enter(done);
        }
        Ok(res)
    }

    fn cpuid(&mut self, args: &[Value]) -> Result<Value> {
        let &[leaf, sub_leaf] = args else { bail!("`__cpuid_count` takes a leaf and a subleaf") };
        let slot = self.b.create_stack_slot(16, 3);
        let out = self.ins().stack_addr(Type::I64, slot, 0);
        self.call_host(HostSymbol::Cpuid, &[out, leaf, sub_leaf], None);
        Ok(out)
    }

    fn transmute(&mut self, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let (&[from, to], value) = (substs, args.first().copied()) else { bail!("transmute takes two type arguments") };
        let (from, to) = (repr(self.cx, from)?, repr(self.cx, to)?);
        Ok(match (from, to, value) {
            (_, Repr::Zst, _) | (Repr::Zst, _, _) | (_, _, None) => None,
            (Repr::Scalar(a), Repr::Scalar(b), Some(value)) if a == b => Some(value),
            (Repr::Scalar(a), Repr::Scalar(b), Some(value)) if can_bitcast(a, b) => Some(self.ins().bitcast(b, value)),
            (Repr::Mem { .. }, Repr::Mem { .. }, Some(address)) => Some(address),
            (Repr::Mem { .. }, Repr::Scalar(ty), Some(address)) => Some(self.ins().load(ty, address, 0)),
            (Repr::Scalar(ty), _, Some(value)) => {
                let slot = self.b.create_stack_slot(ty.bytes(), ty.bytes().trailing_zeros() as u8);
                self.ins().stack_store(value, slot, 0);
                match to {
                    Repr::Scalar(other) => Some(self.ins().stack_load(other, slot, 0)),
                    _ => Some(self.ins().stack_addr(Type::I64, slot, 0)),
                }
            }
        })
    }

    fn with_overflow(&mut self, op: BinOp, substs: &[Ty], args: &[Value]) -> Result<Value> {
        let (Some(&ty), &[x, y]) = (substs.first(), args) else { bail!("{op:?} with overflow takes one type and two values") };
        let Num::Int { ty: machine, signed } = self.num(ty)? else { bail!("{op:?} with overflow on a non-integer") };
        let (result, ok) = self.overflowing(op, (machine, signed), x, y)?;
        let overflowed = self.ins().icmp_imm(IntCC::Equal, ok, 0);
        self.flagged(ty, result, overflowed)
    }

    /// The address of a `(ty, bool)` pair in a new frame slot that holds `value` and `flag`.
    pub(super) fn flagged(&mut self, ty: Ty, value: Value, flag: Value) -> Result<Value> {
        let bool_ty = self.cx.types.prim(Primitive::Bool);
        self.pair([ty, bool_ty], [value, flag])
    }

    /// The address of a stack tuple of types `tys` that holds `values`.
    pub(super) fn pair(&mut self, tys: [Ty; 2], values: [Value; 2]) -> Result<Value> {
        let tuple = self.cx.types.tuple(&tys);
        let tuple_repr = repr(self.cx, tuple)?;
        let Some(slot) = self.slot(tuple_repr) else { bail!("a pair takes no bytes") };
        for (index, value) in (0..).zip(values) {
            let at = self.field_offset(tuple, index)?;
            self.ins().stack_store(value, slot, at);
        }
        Ok(self.ins().stack_addr(Type::I64, slot, 0))
    }

    fn const_eval_select(&mut self, origin: &SiteOrigin, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let (Some(&tuple), Some(&runtime)) = (substs.first(), substs.get(2)) else { bail!("const_eval_select takes four type arguments") };
        let (Kind::Tuple(fields), Kind::FnDef(item, fn_substs)) = (self.cx.types.kind(tuple).clone(), self.cx.types.kind(runtime).clone()) else { bail!("const_eval_select needs a tuple of arguments and a function item") };
        let fields = self.cx.types.substs(fields).to_vec();
        let mut values = Vec::new();
        if let Some(&address) = args.first() {
            for (index, field) in fields.into_iter().enumerate() {
                let offset = self.field_offset(tuple, index as u32)?;
                let place = Place { base: Base::Addr(address), offset, repr: repr(self.cx, field)? };
                values.extend(self.read(place));
            }
        }
        self.invoke(SiteOrigin::ConstEvalSelect(Box::new(origin.clone())), item, fn_substs, values)
    }

    fn select_unpredictable(&mut self, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let Some(&ty) = substs.first() else { bail!("select_unpredictable takes one type argument") };
        let whole = repr(self.cx, ty)?;
        if whole == Repr::Zst {
            return Ok(None);
        }
        let &[cond, yes, no] = args else { bail!("select_unpredictable takes three arguments") };
        match whole {
            Repr::Zst => Ok(None),
            Repr::Scalar(_) => Ok(Some(self.unpredictable(cond, yes, no))),
            Repr::Mem { parts, .. } => {
                let Some(slot) = self.slot(whole) else { return Ok(None) };
                let out = self.ins().stack_addr(Type::I64, slot, 0);
                let mut pick = |m: Type, at: i32| {
                    let a = self.ins().load(m, yes, at);
                    let b = self.ins().load(m, no, at);
                    let sel = self.unpredictable(cond, a, b);
                    self.ins().stack_store(sel, slot, at);
                };
                match parts {
                    super::Parts::Bytes => {
                        let chosen = self.unpredictable(cond, yes, no);
                        self.copy(out, chosen, whole);
                    }
                    other => {
                        for (m, at) in other.scalars() {
                            pick(m, at);
                        }
                    }
                }
                Ok(Some(out))
            }
        }
    }

    fn unpredictable(&mut self, cond: Value, yes: Value, no: Value) -> Value {
        let chosen = self.ins().select(cond, yes, no);
        self.b.func.mark_unpredictable(chosen);
        chosen
    }

    fn metadata(&mut self, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let (Some(&pointee), Some(&pointer)) = (substs.first(), args.first()) else { bail!("ptr_metadata takes a pointer") };
        let tail = self.cx.tail(pointee)?;
        match self.cx.types.kind(tail) {
            Kind::Dyn(..) => Ok(Some(self.ins().iadd_imm(pointer, 8))),
            kind if kind.is_unsized() => Ok(Some(self.ins().load(Type::I64, pointer, 8))),
            _ => Ok(None),
        }
    }

    pub(super) fn value_intrinsic(&mut self, which: Intrinsic, substs: &[Ty], args: &[Value]) -> Result<Value> {
        use Intrinsic::*;
        match (which, args) {
            (Fabs, &[x]) => Ok(self.ins().fabs(x)),
            (Sqrt, &[x]) => Ok(self.ins().sqrt(x)),
            (Floor, &[x]) => Ok(self.ins().floor(x)),
            (Ceil, &[x]) => Ok(self.ins().ceil(x)),
            (Trunc, &[x]) => Ok(self.ins().trunc(x)),
            (Copysign, &[x, y]) => Ok(self.ins().fcopysign(x, y)),
            (Fmin, &[x, y]) => Ok(self.ins().fmin(x, y)),
            (Fmax, &[x, y]) => Ok(self.ins().fmax(x, y)),
            (Fma, &[a, b, c]) => Ok(self.ins().fma(a, b, c)),
            (Round(bits), &[x]) => self.round(if bits == 32 { Type::F32 } else { Type::F64 }, x),
            _ => {
                let Some(&ty) = substs.first() else { bail!("intrinsic {which:?} has no type argument") };
                let Num::Int { ty, signed } = self.num(ty)? else { bail!("intrinsic {which:?} on a non-integer") };
                self.int_intrinsic(which, (ty, signed), args)
            }
        }
    }

    fn int_intrinsic(&mut self, which: Intrinsic, num: (Type, bool), args: &[Value]) -> Result<Value> {
        use Intrinsic::*;
        Ok(match (which, args) {
            (WrappingAdd, &[x, y]) => self.ins().iadd(x, y),
            (WrappingSub, &[x, y]) => self.ins().isub(x, y),
            (WrappingMul, &[x, y]) => self.ins().imul(x, y),
            (SaturatingAdd, &[x, y]) => self.saturating(BinOp::Add, num, x, y)?,
            (SaturatingSub, &[x, y]) => self.saturating(BinOp::Sub, num, x, y)?,
            (RotateLeft, &[x, by]) => self.ins().rotl(x, by),
            (RotateRight, &[x, by]) => self.ins().rotr(x, by),
            (Bswap, &[x]) => self.ins().bswap(x),
            (Bitreverse, &[x]) => self.ins().bitrev(x),
            (Ctpop, &[x]) => self.count(num.0, |lower| lower.ins().popcnt(x)),
            (Ctlz, &[x]) => self.count(num.0, |lower| lower.ins().clz(x)),
            (Cttz, &[x]) => self.count(num.0, |lower| lower.ins().ctz(x)),
            (CttzNonzero, &[x]) if num.0.bits() < 32 => {
                let word = self.ins().uextend(Type::I32, x);
                self.ins().ctz(word)
            }
            (CttzNonzero, &[x]) => self.count(num.0, |lower| lower.ins().ctz(x)),
            _ => bail!("intrinsic {which:?} with {} arguments", args.len()),
        })
    }

    fn count(&mut self, ty: Type, op: impl FnOnce(&mut Self) -> Value) -> Value {
        let value = op(self);
        self.resize(value, (ty, false), Type::I32)
    }

    pub(super) fn saturating(&mut self, op: BinOp, (ty, signed): (Type, bool), x: Value, y: Value) -> Result<Value> {
        let (result, ok) = self.overflowing(op, (ty, signed), x, y)?;
        let bound = match (op, signed) {
            (BinOp::Add, false) => self.int_const(ty, u128::MAX),
            (_, false) => self.int_const(ty, 0),
            (_, true) => {
                let min = self.min_value(ty);
                let max = self.ins().bnot(min);
                let negative = self.ins().icmp_imm(IntCC::SignedLessThan, x, 0);
                self.ins().select(negative, min, max)
            }
        };
        Ok(self.ins().select(ok, result, bound))
    }

    fn round(&mut self, ty: Type, x: Value) -> Result<Value> {
        let whole = self.ins().trunc(x);
        let rest = self.ins().fsub(x, whole);
        let distance = self.ins().fabs(rest);
        let half = self.float_const(ty, 0.5)?;
        let one = self.float_const(ty, 1.0)?;
        let step = self.ins().fcopysign(one, x);
        let away = self.ins().fadd(whole, step);
        let up = self.ins().fcmp(FloatCC::GreaterThanOrEqual, distance, half);
        Ok(self.ins().select(up, away, whole))
    }

    fn compare_const_bytes(&mut self, origin: &SiteOrigin, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        let &[x, y, count] = args else {
            return self.memory_intrinsic(origin, Intrinsic::CompareBytes, substs, args);
        };
        let ty = match self.b.const_int(count) {
            Some(2) if host_is_x64_or_arm64() => Type::I16,
            Some(4) if host_is_x64_or_arm64() => Type::I32,
            Some(8) if host_is_x64_or_arm64() => Type::I64,
            _ => return self.memory_intrinsic(origin, Intrinsic::CompareBytes, substs, args),
        };
        let (a, b) = (self.ins().load(ty, x, 0), self.ins().load(ty, y, 0));
        let (a, b) = (self.ins().bswap(a), self.ins().bswap(b));
        let (lt, gt) = (self.ins().icmp(IntCC::UnsignedLessThan, a, b), self.ins().icmp(IntCC::UnsignedGreaterThan, a, b));
        let diff = self.ins().isub(gt, lt);
        Ok(Some(self.ins().sextend(Type::I32, diff)))
    }
}

fn can_bitcast(a: Type, b: Type) -> bool {
    a.bytes() == b.bytes()
        && ((a.bytes() == 8 && (a.is_vector() || matches!(a, Type::F64 | Type::I64)) && (b.is_vector() || matches!(b, Type::F64 | Type::I64)))
            || matches!((a, b), (Type::F32 | Type::I32, Type::F32 | Type::I32))
            || (a.is_vector() && b.is_vector()))
}
