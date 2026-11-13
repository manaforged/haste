use anyhow::{Result, bail};
use haste_codegen::{AbiParam, CallConv, FloatCC, IntCC, Signature, Type, Value};

use crate::body::BinOp;
use crate::host::HostSymbol;
use crate::names::Primitive;
use crate::sema::intrinsic::Intrinsic;
use crate::sema::ty::{Kind, Ty};

use super::ops::Num;
use super::place::{Base, Place};
use super::stencil::{RodataReloc, RodataTarget, SymbolSource};
use super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Emits intrinsic `which` at type arguments `substs` on `args` inline.
    pub fn intrinsic(&mut self, which: Intrinsic, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
        match which {
            Intrinsic::Transmute => self.transmute(substs, args),
            Intrinsic::FloatToInt => {
                let (&[from, to], &[value]) = (substs, args) else { bail!("float_to_int_unchecked takes one float") };
                let (from, to) = (self.num(from)?, self.num(to)?);
                self.convert(value, from, to).map(Some)
            }
            Intrinsic::PtrMetadata => self.metadata(substs, args),
            Intrinsic::ConstEvalSelect => self.const_eval_select(substs, args),
            Intrinsic::WithOverflow(op) => self.with_overflow(op, substs, args).map(Some),
            Intrinsic::Simd(op) => self.simd(op, substs, args),
            Intrinsic::Trap => {
                self.ins().trap();
                self.close();
                Ok(None)
            }
            Intrinsic::CatchUnwind => self.catch_unwind(args).map(Some),
            Intrinsic::CallerLocation => self.caller_location().map(Some),
            Intrinsic::Cpuid => self.cpuid(args).map(Some),
            Intrinsic::FnAddr => Ok(args.first().copied()),
            Intrinsic::BytePermute(bytes) => self.byte_permute(args, bytes).map(Some),
            Intrinsic::BytePairMax => self.byte_pair_max(args).map(Some),
            Intrinsic::X86(x86) => self.host_vector(x86, args).map(Some),
            _ => self.memory_intrinsic(which, substs, args),
        }
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

    fn caller_location(&mut self) -> Result<Value> {
        let Some(location) = self.cx.lang_item("panic_location") else { bail!("no `Location` lang item") };
        let ty = self.cx.types.adt(location, &[]);
        let layout = self.cx.layout(ty)?;
        let (src, entry) = self.cx.item(self.reached.list[self.index].0.0);
        let mut file = src.path.display().to_string().into_bytes();
        let (len, line, column) = (file.len() as u64, src.line(entry.start) as u32, src.column(entry.start) as u32);
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

    fn catch_unwind(&mut self, args: &[Value]) -> Result<Value> {
        let &[try_fn, data, _] = args else { bail!("catch_unwind takes three arguments") };
        let byte = self.cx.types.prim(Primitive::U8);
        let pointer = self.cx.types.intern(Kind::Ptr(crate::sema::ty::Mut::Mut, byte));
        let unit = self.cx.types.unit();
        self.call_code(try_fn, (vec![pointer], Some(unit)), vec![data])?;
        Ok(self.ins().iconst(Type::I32, 0))
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
        let tuple = self.cx.types.tuple(&[ty, bool_ty]);
        let tuple_repr = repr(self.cx, tuple)?;
        let Some(slot) = self.slot(tuple_repr) else { bail!("a flagged pair takes no bytes") };
        let (first, second) = (self.field_offset(tuple, 0)?, self.field_offset(tuple, 1)?);
        self.ins().stack_store(value, slot, first);
        self.ins().stack_store(flag, slot, second);
        Ok(self.ins().stack_addr(Type::I64, slot, 0))
    }

    fn const_eval_select(&mut self, substs: &[Ty], args: &[Value]) -> Result<Option<Value>> {
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
        self.invoke(item, fn_substs, values)
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
}
