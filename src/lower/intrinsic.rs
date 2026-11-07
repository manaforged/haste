use haste_codegen::{AbiParam, AtomicRmwOp, Signature, Type, Value};
use hir_def::AdtId;
use hir_ty::next_solver::{GenericArgs, Ty, TyKind};
use rustc_type_ir::inherent::IntoKind;

use super::{Lower, Place};

mod ops;

use crate::lower::BinOp;

unsafe extern "C" {
    fn dlsym(handle: *mut std::ffi::c_void, name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
    fn __divti3(a: i128, b: i128) -> i128;
    fn __modti3(a: i128, b: i128) -> i128;
    fn __udivti3(a: u128, b: u128) -> u128;
    fn __umodti3(a: u128, b: u128) -> u128;
    fn fmod(a: f64, b: f64) -> f64;
    fn fmodf(a: f32, b: f32) -> f32;
    fn memcpy(to: *mut u8, from: *const u8, size: usize) -> *mut u8;
    fn memmove(to: *mut u8, from: *const u8, size: usize) -> *mut u8;
    fn memset(to: *mut u8, value: i32, size: usize) -> *mut u8;
    fn memcmp(left: *const u8, right: *const u8, size: usize) -> i32;
    fn round(x: f64) -> f64;
    fn roundf(x: f32) -> f32;
    fn fma(a: f64, b: f64, c: f64) -> f64;
    fn fmaf(a: f32, b: f32, c: f32) -> f32;
    fn pow(a: f64, b: f64) -> f64;
    fn powf(a: f32, b: f32) -> f32;
    fn exp(a: f64) -> f64;
    fn expf(a: f32) -> f32;
    fn fmax(a: f64, b: f64) -> f64;
    fn fmaxf(a: f32, b: f32) -> f32;
    fn fmin(a: f64, b: f64) -> f64;
    fn fminf(a: f32, b: f32) -> f32;
    fn sin(a: f64) -> f64;
    fn sinf(a: f32) -> f32;
    fn cos(a: f64) -> f64;
    fn cosf(a: f32) -> f32;
    fn exp2(a: f64) -> f64;
    fn exp2f(a: f32) -> f32;
    fn log(a: f64) -> f64;
    fn logf(a: f32) -> f32;
    fn log2(a: f64) -> f64;
    fn log2f(a: f32) -> f32;
    fn log10(a: f64) -> f64;
    fn log10f(a: f32) -> f32;
    fn rint(a: f64) -> f64;
    fn rintf(a: f32) -> f32;
}

fn bin_op(kind: BinOpKind) -> BinOp {
    match kind {
        BinOpKind::Add => BinOp::Add,
        BinOpKind::Sub => BinOp::Sub,
        BinOpKind::Mul => BinOp::Mul,
        BinOpKind::Div => BinOp::Div,
        BinOpKind::Rem => BinOp::Rem,
        BinOpKind::Shl => BinOp::Shl,
        BinOpKind::Shr => BinOp::Shr,
    }
}

pub(super) fn wide_call(op: &BinOp, ty: Type, signed: bool) -> Option<usize> {
    let code = match (op, ty, signed) {
        (BinOp::Div, Type::I128, true) => __divti3 as *const () as usize,
        (BinOp::Rem, Type::I128, true) => __modti3 as *const () as usize,
        (BinOp::Div, Type::I128, false) => __udivti3 as *const () as usize,
        (BinOp::Rem, Type::I128, false) => __umodti3 as *const () as usize,
        (BinOp::Rem, Type::F64, _) => fmod as *const () as usize,
        (BinOp::Rem, Type::F32, _) => fmodf as *const () as usize,
        _ => return None,
    };
    Some(code)
}

extern "C" fn rust_alloc(size: usize, align: usize) -> *mut u8 {
    // SAFETY: the caller is std's allocator shim, which passes a valid non-zero layout.
    unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, align)) }
}

extern "C" fn rust_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    // SAFETY: as in rust_alloc.
    unsafe { std::alloc::alloc_zeroed(std::alloc::Layout::from_size_align_unchecked(size, align)) }
}

extern "C" fn rust_dealloc(pointer: *mut u8, size: usize, align: usize) {
    // SAFETY: pointer came from rust_alloc with this same layout, per the allocator contract.
    unsafe { std::alloc::dealloc(pointer, std::alloc::Layout::from_size_align_unchecked(size, align)) }
}

extern "C" fn rust_realloc(pointer: *mut u8, size: usize, align: usize, new_size: usize) -> *mut u8 {
    // SAFETY: as in rust_dealloc.
    unsafe { std::alloc::realloc(pointer, std::alloc::Layout::from_size_align_unchecked(size, align), new_size) }
}

extern "C" fn no_shim() {}

extern "C" fn powi_f64(base: f64, power: i32) -> f64 {
    base.powi(power)
}

extern "C" fn powi_f32(base: f32, power: i32) -> f32 {
    base.powi(power)
}

std::thread_local! {
    static THREAD_STATICS: std::cell::Cell<*mut Vec<usize>> = const { std::cell::Cell::new(std::ptr::null_mut()) };
}

extern "C" fn thread_local_address(index: usize, template: *const u8, size: usize) -> usize {
    THREAD_STATICS.with(|cell| {
        if cell.get().is_null() {
            cell.set(Box::into_raw(Box::new(Vec::new())));
        }
        // SAFETY: the table is leaked and only this thread holds its pointer, and the program's thread-local destructors may still reach it at thread exit.
        let statics = unsafe { &mut *cell.get() };
        if statics.len() <= index {
            statics.resize(index + 1, 0);
        }
        if statics[index] == 0 {
            let storage = Box::leak(vec![0_u128; size.div_ceil(16).max(1)].into_boxed_slice());
            // SAFETY: template points at the leaked initial value of this static, which is size bytes, and storage is at least that long.
            unsafe { std::ptr::copy_nonoverlapping(template, storage.as_mut_ptr().cast::<u8>(), size) };
            statics[index] = storage.as_ptr() as usize;
        }
        statics[index]
    })
}

pub(super) fn thread_local_code() -> usize {
    thread_local_address as *const () as usize
}

pub(super) fn compare_bytes_code() -> usize {
    memcmp as *const () as usize
}

const ALLOCATOR: &[(&str, fn() -> usize)] = &[
    ("__rust_alloc", || rust_alloc as *const () as usize),
    ("__rust_alloc_zeroed", || rust_alloc_zeroed as *const () as usize),
    ("__rust_dealloc", || rust_dealloc as *const () as usize),
    ("__rust_realloc", || rust_realloc as *const () as usize),
    ("__rust_no_alloc_shim_is_unstable_v2", || no_shim as *const () as usize),
    ("miri_promise_symbolic_alignment", || no_shim as *const () as usize),
];

const UNKNOWN_FILE: &str = "<engine>";

pub fn is_known(name: &str) -> bool {
    TABLE.iter().any(|(known, _)| *known == name)
}

pub fn is_core_override(name: &str) -> bool {
    OVERRIDES.contains(&name) || (name.starts_with("atomic_") && is_known(name))
}

static DSO_HANDLE: u8 = 0;

pub(crate) fn symbol(name: &str) -> Option<usize> {
    #[cfg(target_os = "macos")]
    if let Some(code) = crate::park::symbol(name) {
        return Some(code);
    }
    if name == "__dso_handle" {
        return Some(&raw const DSO_HANDLE as usize);
    }
    if let Some((_, code)) = ALLOCATOR.iter().find(|(known, _)| *known == name) {
        return Some(code());
    }
    let name = std::ffi::CString::new(name).ok()?;
    // SAFETY: name is a valid C string and RTLD_DEFAULT is the platform's default-search handle.
    let code = unsafe { dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
    (!code.is_null()).then_some(code as usize)
}

const OVERRIDES: &[&str] = &["read_unaligned", "write_unaligned"];

#[derive(Clone, Copy)]
enum MemKind {
    Move,
    Fill,
    Compare,
}

#[derive(Clone, Copy)]
enum Count {
    Ones,
    LeadingZeros,
    TrailingZeros,
}

#[derive(Clone, Copy)]
enum Float {
    Abs,
    Sqrt,
    Floor,
    Ceil,
    Trunc,
}

#[derive(Clone, Copy)]
enum Layout {
    Size,
    Align,
}

#[derive(Clone, Copy)]
enum Op {
    Trap,
    Nop,
    False,
    CopyFirst,
    Write,
    WriteBox,
    Read,
    CopyMemory,
    LayoutOf(Layout),
    LayoutOfVal(Layout),
    NeedsDrop,
    DiscriminantValue,
    SliceItem,
    AtomicLoad,
    AtomicStore,
    AtomicRmw(AtomicRmwOp),
    AtomicExchange,
    AtomicFence,
    OffsetFrom,
    Mem(fn() -> usize, MemKind),
    RawEq,
    Compare3,
    SwapBytes,
    ReverseBits,
    PointerMask,
    VariantCount,
    TypeId,
    TypeName,
    CatchUnwind,
    Offset,
    RawPtr,
    Metadata,
    Arith(BinOpKind),
    SaturatingAdd,
    SaturatingSub,
    Overflow(BinOpKind),
    RotateLeft,
    RotateRight,
    CountBits(Count),
    FloatUnary(Float),
    CopySign,
    Libm(fn() -> usize, Type, usize),
    Powi(fn() -> usize, Type),
    FloatToInt,
    CallerLocation,
    PointerOrder(Order),
}

#[derive(Clone, Copy)]
enum Order {
    Lt,
    Le,
    Gt,
    Ge,
}

const POINTER_ORDER: &[(&str, &str)] =
    &[("lt", "pointer_lt"), ("le", "pointer_le"), ("gt", "pointer_gt"), ("ge", "pointer_ge")];

pub fn pointer_order_name(method: &str) -> Option<&'static str> {
    POINTER_ORDER.iter().find(|(known, _)| *known == method).map(|(_, name)| *name)
}

#[derive(Clone, Copy)]
enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Shl,
    Shr,
}

macro_rules! libm {
    ($name:ident) => {
        || $name as *const () as usize
    };
}

const TABLE: &[(&str, Op)] = &[
    ("abort", Op::Trap),
    ("unreachable", Op::Trap),
    ("assume", Op::Nop),
    ("assert_inhabited", Op::Nop),
    ("assert_zero_valid", Op::Nop),
    ("assert_mem_uninitialized_valid", Op::Nop),
    ("cold_path", Op::Nop),
    ("forget", Op::Nop),
    ("ub_checks", Op::False),
    ("contract_checks", Op::False),
    ("overflow_checks", Op::False),
    ("likely", Op::CopyFirst),
    ("unlikely", Op::CopyFirst),
    ("black_box", Op::CopyFirst),
    ("transmute", Op::CopyFirst),
    ("transmute_unchecked", Op::CopyFirst),
    ("write_via_move", Op::Write),
    ("write_box_via_move", Op::WriteBox),
    ("read_via_copy", Op::Read),
    ("copy_nonoverlapping", Op::CopyMemory),
    ("size_of", Op::LayoutOf(Layout::Size)),
    ("min_align_of", Op::LayoutOf(Layout::Align)),
    ("align_of", Op::LayoutOf(Layout::Align)),
    ("pref_align_of", Op::LayoutOf(Layout::Align)),
    ("size_of_val", Op::LayoutOfVal(Layout::Size)),
    ("min_align_of_val", Op::LayoutOfVal(Layout::Align)),
    ("align_of_val", Op::LayoutOfVal(Layout::Align)),
    ("needs_drop", Op::NeedsDrop),
    ("discriminant_value", Op::DiscriminantValue),
    ("slice_get_unchecked", Op::SliceItem),
    ("atomic_load", Op::AtomicLoad),
    ("atomic_store", Op::AtomicStore),
    ("atomic_swap", Op::AtomicRmw(AtomicRmwOp::Xchg)),
    ("atomic_add", Op::AtomicRmw(AtomicRmwOp::Add)),
    ("atomic_sub", Op::AtomicRmw(AtomicRmwOp::Sub)),
    ("atomic_and", Op::AtomicRmw(AtomicRmwOp::And)),
    ("atomic_nand", Op::AtomicRmw(AtomicRmwOp::Nand)),
    ("atomic_or", Op::AtomicRmw(AtomicRmwOp::Or)),
    ("atomic_xor", Op::AtomicRmw(AtomicRmwOp::Xor)),
    ("atomic_max", Op::AtomicRmw(AtomicRmwOp::Smax)),
    ("atomic_min", Op::AtomicRmw(AtomicRmwOp::Smin)),
    ("atomic_umax", Op::AtomicRmw(AtomicRmwOp::Umax)),
    ("atomic_umin", Op::AtomicRmw(AtomicRmwOp::Umin)),
    ("atomic_compare_exchange", Op::AtomicExchange),
    ("atomic_compare_exchange_weak", Op::AtomicExchange),
    ("atomic_fence", Op::AtomicFence),
    ("atomic_compiler_fence", Op::AtomicFence),
    ("ptr_offset_from", Op::OffsetFrom),
    ("ptr_offset_from_unsigned", Op::OffsetFrom),
    ("copy", Op::Mem(libm!(memmove), MemKind::Move)),
    ("write_bytes", Op::Mem(libm!(memset), MemKind::Fill)),
    ("compare_bytes", Op::Mem(libm!(memcmp), MemKind::Compare)),
    ("raw_eq", Op::RawEq),
    ("three_way_compare", Op::Compare3),
    ("bswap", Op::SwapBytes),
    ("bitreverse", Op::ReverseBits),
    ("ptr_mask", Op::PointerMask),
    ("variant_count", Op::VariantCount),
    ("type_id", Op::TypeId),
    ("type_name", Op::TypeName),
    ("catch_unwind", Op::CatchUnwind),
    ("read_unaligned", Op::Read),
    ("write_unaligned", Op::Write),
    ("volatile_load", Op::Read),
    ("unaligned_volatile_load", Op::Read),
    ("volatile_store", Op::Write),
    ("unaligned_volatile_store", Op::Write),
    ("is_val_statically_known", Op::False),
    ("offset", Op::Offset),
    ("arith_offset", Op::Offset),
    ("aggregate_raw_ptr", Op::RawPtr),
    ("ptr_metadata", Op::Metadata),
    ("wrapping_add", Op::Arith(BinOpKind::Add)),
    ("unchecked_add", Op::Arith(BinOpKind::Add)),
    ("wrapping_sub", Op::Arith(BinOpKind::Sub)),
    ("unchecked_sub", Op::Arith(BinOpKind::Sub)),
    ("wrapping_mul", Op::Arith(BinOpKind::Mul)),
    ("unchecked_mul", Op::Arith(BinOpKind::Mul)),
    ("unchecked_div", Op::Arith(BinOpKind::Div)),
    ("exact_div", Op::Arith(BinOpKind::Div)),
    ("unchecked_rem", Op::Arith(BinOpKind::Rem)),
    ("unchecked_shl", Op::Arith(BinOpKind::Shl)),
    ("unchecked_shr", Op::Arith(BinOpKind::Shr)),
    ("add_with_overflow", Op::Overflow(BinOpKind::Add)),
    ("sub_with_overflow", Op::Overflow(BinOpKind::Sub)),
    ("mul_with_overflow", Op::Overflow(BinOpKind::Mul)),
    ("saturating_add", Op::SaturatingAdd),
    ("saturating_sub", Op::SaturatingSub),
    ("rotate_left", Op::RotateLeft),
    ("rotate_right", Op::RotateRight),
    ("ctpop", Op::CountBits(Count::Ones)),
    ("ctlz", Op::CountBits(Count::LeadingZeros)),
    ("ctlz_nonzero", Op::CountBits(Count::LeadingZeros)),
    ("cttz", Op::CountBits(Count::TrailingZeros)),
    ("cttz_nonzero", Op::CountBits(Count::TrailingZeros)),
    ("fabs", Op::FloatUnary(Float::Abs)),
    ("fabsf32", Op::FloatUnary(Float::Abs)),
    ("fabsf64", Op::FloatUnary(Float::Abs)),
    ("sqrtf32", Op::FloatUnary(Float::Sqrt)),
    ("sqrtf64", Op::FloatUnary(Float::Sqrt)),
    ("floorf32", Op::FloatUnary(Float::Floor)),
    ("floorf64", Op::FloatUnary(Float::Floor)),
    ("ceilf32", Op::FloatUnary(Float::Ceil)),
    ("ceilf64", Op::FloatUnary(Float::Ceil)),
    ("truncf32", Op::FloatUnary(Float::Trunc)),
    ("truncf64", Op::FloatUnary(Float::Trunc)),
    ("copysignf32", Op::CopySign),
    ("copysignf64", Op::CopySign),
    ("roundf64", Op::Libm(libm!(round), Type::F64, 1)),
    ("roundf32", Op::Libm(libm!(roundf), Type::F32, 1)),
    ("fmaf64", Op::Libm(libm!(fma), Type::F64, 3)),
    ("fmaf32", Op::Libm(libm!(fmaf), Type::F32, 3)),
    ("powf64", Op::Libm(libm!(pow), Type::F64, 2)),
    ("powf32", Op::Libm(libm!(powf), Type::F32, 2)),
    ("expf64", Op::Libm(libm!(exp), Type::F64, 1)),
    ("expf32", Op::Libm(libm!(expf), Type::F32, 1)),
    ("sinf64", Op::Libm(libm!(sin), Type::F64, 1)),
    ("sinf32", Op::Libm(libm!(sinf), Type::F32, 1)),
    ("cosf64", Op::Libm(libm!(cos), Type::F64, 1)),
    ("cosf32", Op::Libm(libm!(cosf), Type::F32, 1)),
    ("exp2f64", Op::Libm(libm!(exp2), Type::F64, 1)),
    ("exp2f32", Op::Libm(libm!(exp2f), Type::F32, 1)),
    ("logf64", Op::Libm(libm!(log), Type::F64, 1)),
    ("logf32", Op::Libm(libm!(logf), Type::F32, 1)),
    ("log2f64", Op::Libm(libm!(log2), Type::F64, 1)),
    ("log2f32", Op::Libm(libm!(log2f), Type::F32, 1)),
    ("log10f64", Op::Libm(libm!(log10), Type::F64, 1)),
    ("log10f32", Op::Libm(libm!(log10f), Type::F32, 1)),
    ("round_ties_even_f64", Op::Libm(libm!(rint), Type::F64, 1)),
    ("round_ties_even_f32", Op::Libm(libm!(rintf), Type::F32, 1)),
    ("powif64", Op::Powi(|| powi_f64 as *const () as usize, Type::F64)),
    ("powif32", Op::Powi(|| powi_f32 as *const () as usize, Type::F32)),
    ("float_to_int_unchecked", Op::FloatToInt),
    ("caller_location", Op::CallerLocation),
    ("pointer_lt", Op::PointerOrder(Order::Lt)),
    ("pointer_le", Op::PointerOrder(Order::Le)),
    ("pointer_gt", Op::PointerOrder(Order::Gt)),
    ("pointer_ge", Op::PointerOrder(Order::Ge)),
    ("maximum_number_nsz_f64", Op::Libm(libm!(fmax), Type::F64, 2)),
    ("maximum_number_nsz_f32", Op::Libm(libm!(fmaxf), Type::F32, 2)),
    ("minimum_number_nsz_f64", Op::Libm(libm!(fmin), Type::F64, 2)),
    ("minimum_number_nsz_f32", Op::Libm(libm!(fminf), Type::F32, 2)),
];

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn intrinsic(
        &mut self,
        name: &str,
        generics: GenericArgs<'db>,
        dest: Place<'db>,
        args: &[Place<'db>],
    ) -> Result<(), String> {
        let op = TABLE.iter().find(|(known, _)| *known == name).map(|(_, op)| *op);
        let op = op.ok_or_else(|| format!("intrinsic {name}"))?;
        let generic = generics.as_slice().first().and_then(|arg| arg.ty());
        match op {
            Op::Trap => self.trap_and_continue(),
            Op::Nop => {}
            Op::False => {
                let zero = self.builder.ins().iconst(Type::I8, 0);
                self.store(zero, dest.addr, 0);
            }
            Op::CopyFirst => self.copy_sized(dest, args[0].addr)?,
            Op::Write => self.write_through(args)?,
            Op::WriteBox => {
                self.write_through(args)?;
                self.copy_sized(dest, args[0].addr)?;
            }
            Op::Read => {
                let pointer = self.load(Type::I64, args[0].addr, 0);
                self.copy_sized(dest, pointer)?;
            }
            Op::CopyMemory => self.copy_memory(generic.ok_or("copy without a type")?, args)?,
            Op::LayoutOf(kind) => self.layout_value(generic.ok_or("layout without a type")?, kind, dest)?,
            Op::LayoutOfVal(kind) => {
                let ty = generic.ok_or("layout without a type")?;
                let meta = self.layout(ty)?.is_unsized().then(|| self.load(Type::I64, args[0].addr, 8));
                let (size, align) = self.size_align(ty, meta)?;
                let value = if matches!(kind, Layout::Size) { size } else { align };
                self.store(value, dest.addr, 0);
            }
            Op::NeedsDrop => {
                let needs = self.needs_drop(generic.ok_or("needs_drop without a type")?);
                let value = self.builder.ins().iconst(Type::I8, i64::from(needs));
                self.store(value, dest.addr, 0);
            }
            Op::DiscriminantValue => {
                let pointer = self.load(Type::I64, args[0].addr, 0);
                let ty = generic.ok_or("discriminant without a type")?;
                let place = Place { addr: pointer, ty: hir_ty::mir::PlaceTy::from_ty(ty), meta: None };
                self.discriminant(dest, place)?;
            }
            Op::SliceItem => {
                let item = generics.as_slice().get(2).and_then(|arg| arg.ty()).ok_or("slice item without a type")?;
                let base = self.load(Type::I64, args[0].addr, 0);
                let index = self.load(Type::I64, args[1].addr, 0);
                let size = self.layout(item)?.size.bytes() as i64;
                let delta = self.builder.ins().imul_imm(index, size);
                let value = self.builder.ins().iadd(base, delta);
                self.store(value, dest.addr, 0);
            }
            Op::AtomicLoad => {
                let scalar = self.scalar(generic.ok_or("atomic without a type")?)?;
                let pointer = self.load(Type::I64, args[0].addr, 0);
                let value = self.builder.ins().atomic_load(scalar.ty, pointer);
                self.store(value, dest.addr, 0);
            }
            Op::AtomicStore => {
                let scalar = self.scalar(generic.ok_or("atomic without a type")?)?;
                let pointer = self.load(Type::I64, args[0].addr, 0);
                let value = self.load(scalar.ty, args[1].addr, 0);
                self.builder.ins().atomic_store(value, pointer);
            }
            Op::AtomicRmw(op) => {
                let scalar = self.scalar(generic.ok_or("atomic without a type")?)?;
                let pointer = self.load(Type::I64, args[0].addr, 0);
                let value = self.load(scalar.ty, args[1].addr, 0);
                let old = self.builder.ins().atomic_rmw(scalar.ty, op, pointer, value);
                self.store(old, dest.addr, 0);
            }
            Op::AtomicExchange => self.atomic_exchange(generic.ok_or("atomic without a type")?, dest, args)?,
            Op::AtomicFence => {
                self.builder.ins().fence();
            }
            Op::OffsetFrom => self.offset_from(generic.ok_or("offset_from without a type")?, dest, args)?,
            Op::Mem(code, kind) => self.mem(code(), kind, generic, dest, args)?,
            Op::RawEq => self.raw_eq(generic.ok_or("raw_eq without a type")?, dest, args)?,
            Op::Compare3 => self.compare3(dest, args)?,
            Op::SwapBytes => {
                let scalar = self.scalar(args[0].ty.ty)?;
                let mut value = self.load(scalar.ty, args[0].addr, 0);
                if scalar.ty != Type::I8 {
                    value = self.builder.ins().bswap(value);
                }
                self.store(value, dest.addr, 0);
            }
            Op::ReverseBits => {
                let scalar = self.scalar(args[0].ty.ty)?;
                let value = self.load(scalar.ty, args[0].addr, 0);
                let value = self.builder.ins().bitrev(value);
                self.store(value, dest.addr, 0);
            }
            Op::PointerMask => {
                let pointer = self.load(Type::I64, args[0].addr, 0);
                let mask = self.load(Type::I64, args[1].addr, 0);
                let value = self.builder.ins().band(pointer, mask);
                self.store(value, dest.addr, 0);
            }
            Op::VariantCount => {
                let count = match generic.map(|ty| ty.kind()) {
                    Some(TyKind::Adt(adt, _)) => match adt.def_id() {
                        AdtId::EnumId(id) => id.enum_variants(self.db).variants.len(),
                        _ => 0,
                    },
                    _ => 0,
                };
                let value = self.builder.ins().iconst(Type::I64, count as i64);
                self.store(value, dest.addr, 0);
            }
            Op::TypeId => {
                let erased = hir_ty::next_solver::Region::new_erased(self.infcx.interner);
                let generic = generic.ok_or("type_id without a type")?;
                let text = format!("{:?}", rustc_type_ir::fold_regions(self.infcx.interner, generic, |region, _| {
                    if matches!(region.kind(), rustc_type_ir::RegionKind::ReBound(..)) { region } else { erased }
                }));
                let mut hasher = std::hash::DefaultHasher::new();
                std::hash::Hash::hash(&text, &mut hasher);
                let low = std::hash::Hasher::finish(&hasher);
                std::hash::Hash::hash(&1_u8, &mut hasher);
                let high = std::hash::Hasher::finish(&hasher);
                let low = self.builder.ins().iconst(Type::I64, low as i64);
                let high = self.builder.ins().iconst(Type::I64, high as i64);
                self.store(low, dest.addr, 0);
                self.store(high, dest.addr, 8);
            }
            Op::TypeName => {
                let text: &'static str = Box::leak(format!("{:?}", generic.ok_or("type_name without a type")?).into_boxed_str());
                let data = self.builder.ins().iconst(Type::I64, text.as_ptr() as i64);
                let len = self.builder.ins().iconst(Type::I64, text.len() as i64);
                self.store(data, dest.addr, 0);
                self.store(len, dest.addr, 8);
            }
            Op::CatchUnwind => self.catch_unwind(dest, args)?,
            Op::Offset => self.offset(dest, args)?,
            Op::RawPtr => self.raw_ptr(dest, args)?,
            Op::Metadata => {
                if self.layout(dest.ty.ty)?.size.bytes() != 0 {
                    let meta = self.load(Type::I64, args[0].addr, 8);
                    self.store(meta, dest.addr, 0);
                }
            }
            Op::Arith(kind) => self.arith(&bin_op(kind), dest, args)?,
            Op::Overflow(kind) => self.overflow(kind, dest, args)?,
            Op::SaturatingAdd => self.saturating(true, dest, args)?,
            Op::SaturatingSub => self.saturating(false, dest, args)?,
            Op::RotateLeft => self.rotate(true, dest, args)?,
            Op::RotateRight => self.rotate(false, dest, args)?,
            Op::CountBits(kind) => self.count_bits(kind, dest, args)?,
            Op::FloatUnary(kind) => self.float_unary(kind, dest, args)?,
            Op::CopySign => {
                let scalar = self.scalar(args[0].ty.ty)?;
                let left = self.load(scalar.ty, args[0].addr, 0);
                let right = self.load(scalar.ty, args[1].addr, 0);
                let result = self.builder.ins().fcopysign(left, right);
                self.store(result, dest.addr, 0);
            }
            Op::Libm(code, ty, arity) => self.libm_call(code(), (ty, arity), dest, args)?,
            Op::Powi(code, ty) => self.powi(code(), ty, dest, args),
            Op::FloatToInt => self.float_to_int(dest, args[0])?,
            Op::CallerLocation => self.caller_location(dest)?,
            Op::PointerOrder(order) => {
                let op = match order {
                    Order::Lt => BinOp::Lt,
                    Order::Le => BinOp::Le,
                    Order::Gt => BinOp::Gt,
                    Order::Ge => BinOp::Ge,
                };
                self.pointer_order(&op, dest, args)?;
            }
        }
        Ok(())
    }

    fn powi(&mut self, code: usize, ty: Type, dest: Place<'db>, args: &[Place<'db>]) {
        let mut signature = Signature::new(self.call_conv);
        signature.params.extend([AbiParam::new(ty), AbiParam::new(Type::I32)]);
        signature.returns.push(AbiParam::new(ty));
        let signature = self.builder.import_signature(signature);
        let code = self.builder.ins().iconst(Type::I64, code as i64);
        let base = self.load(ty, args[0].addr, 0);
        let power = self.load(Type::I32, args[1].addr, 0);
        let call = self.builder.ins().call_indirect(signature, code, &[base, power]);
        let result = self.builder.inst_results(call)[0];
        self.store(result, dest.addr, 0);
    }

    fn caller_location(&mut self, dest: Place<'db>) -> Result<(), String> {
        let location = super::pointee(dest.ty.ty)?;
        let layout = self.layout(location)?;
        let mut storage = vec![0_u64; (layout.size.bytes() as usize).div_ceil(8)].into_boxed_slice();
        let (text, size) = crate::backend::leak_text(UNKNOWN_FILE);
        let base = storage.as_mut_ptr().cast::<u8>();
        let (file, line, column) = (layout.fields.offset(0).bytes() as usize, layout.fields.offset(1).bytes() as usize, layout.fields.offset(2).bytes() as usize);
        // SAFETY: the offsets come from the layout of Location, which sized this buffer, and its fields are a fat pointer and two u32 values.
        unsafe {
            base.add(file).cast::<[i64; 2]>().write_unaligned([text, size]);
            base.add(line).cast::<u32>().write_unaligned(0);
            base.add(column).cast::<u32>().write_unaligned(0);
        }
        let value = self.builder.ins().iconst(Type::I64, Box::leak(storage).as_ptr() as i64);
        self.store(value, dest.addr, 0);
        Ok(())
    }

    pub(super) fn trap_and_continue(&mut self) {
        self.trap();
        let next = self.builder.create_block();
        self.builder.switch_to_block(next);
    }

    fn copy_sized(&mut self, dest: Place<'db>, from: Value) -> Result<(), String> {
        let size = self.layout(dest.ty.ty)?.size.bytes();
        self.copy(dest.addr, from, size);
        Ok(())
    }

    fn write_through(&mut self, args: &[Place<'db>]) -> Result<(), String> {
        let pointer = self.load(Type::I64, args[0].addr, 0);
        let size = self.layout(args[1].ty.ty)?.size.bytes();
        self.copy(pointer, args[1].addr, size);
        Ok(())
    }

    fn copy_memory(&mut self, ty: Ty<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let size = self.layout(ty)?.size.bytes();
        let from = self.load(Type::I64, args[0].addr, 0);
        let to = self.load(Type::I64, args[1].addr, 0);
        let count = self.load(Type::I64, args[2].addr, 0);
        let bytes = self.builder.ins().imul_imm(count, size as i64);
        let mut signature = Signature::new(self.call_conv);
        signature.params.extend([AbiParam::new(Type::I64); 3]);
        signature.returns.push(AbiParam::new(Type::I64));
        let signature = self.builder.import_signature(signature);
        let code = self.builder.ins().iconst(Type::I64, memcpy as *const () as usize as i64);
        self.builder.ins().call_indirect(signature, code, &[to, from, bytes]);
        Ok(())
    }

    fn layout_value(&mut self, ty: Ty<'db>, kind: Layout, dest: Place<'db>) -> Result<(), String> {
        let layout = self.layout(ty)?;
        let value = match kind {
            Layout::Size => layout.size.bytes(),
            Layout::Align => layout.align.abi.bytes(),
        };
        let value = self.builder.ins().iconst(Type::I64, value as i64);
        self.store(value, dest.addr, 0);
        Ok(())
    }

    fn offset(&mut self, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let pointer = self.load(Type::I64, args[0].addr, 0);
        let index_scalar = self.scalar(args[1].ty.ty)?;
        let index = self.load(index_scalar.ty, args[1].addr, 0);
        let pointee = crate::lower::pointee(args[0].ty.ty)?;
        let size = self.layout(pointee)?.size.bytes();
        let delta = self.builder.ins().imul_imm(index, size as i64);
        let value = self.builder.ins().iadd(pointer, delta);
        self.store(value, dest.addr, 0);
        Ok(())
    }

    fn raw_ptr(&mut self, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let data = self.load(Type::I64, args[0].addr, 0);
        self.store(data, dest.addr, 0);
        if self.layout(args[1].ty.ty)?.size.bytes() != 0 {
            let meta = self.load(Type::I64, args[1].addr, 0);
            self.store(meta, dest.addr, 8);
        }
        Ok(())
    }

    fn rotate(&mut self, left: bool, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let scalar = self.scalar(args[0].ty.ty)?;
        let value = self.load(scalar.ty, args[0].addr, 0);
        let shift = self.load(Type::I32, args[1].addr, 0);
        let ins = self.builder.ins();
        let result = if left { ins.rotl(value, shift) } else { ins.rotr(value, shift) };
        self.store(result, dest.addr, 0);
        Ok(())
    }

    fn count_bits(&mut self, kind: Count, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let scalar = self.scalar(args[0].ty.ty)?;
        let value = self.load(scalar.ty, args[0].addr, 0);
        let ins = self.builder.ins();
        let counted = match kind {
            Count::Ones => ins.popcnt(value),
            Count::LeadingZeros => ins.clz(value),
            Count::TrailingZeros => ins.ctz(value),
        };
        let result = self.to_u32(counted, scalar.ty);
        self.store(result, dest.addr, 0);
        Ok(())
    }

    fn float_unary(&mut self, kind: Float, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let scalar = self.scalar(args[0].ty.ty)?;
        let value = self.load(scalar.ty, args[0].addr, 0);
        let ins = self.builder.ins();
        let result = match kind {
            Float::Abs => ins.fabs(value),
            Float::Sqrt => ins.sqrt(value),
            Float::Floor => ins.floor(value),
            Float::Ceil => ins.ceil(value),
            Float::Trunc => ins.trunc(value),
        };
        self.store(result, dest.addr, 0);
        Ok(())
    }

    fn to_u32(&mut self, value: Value, from: Type) -> Value {
        let ins = self.builder.ins();
        if from.bits() > 32 {
            ins.ireduce(Type::I32, value)
        } else if from.bits() < 32 {
            ins.uextend(Type::I32, value)
        } else {
            value
        }
    }

    fn arith(&mut self, op: &BinOp, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let scalar = self.scalar(args[0].ty.ty)?;
        let right_scalar = self.scalar(args[1].ty.ty)?;
        let left = self.load(scalar.ty, args[0].addr, 0);
        let right = self.load(right_scalar.ty, args[1].addr, 0);
        let value = self.binary(op, left, right, scalar)?;
        self.store(value, dest.addr, 0);
        Ok(())
    }

    fn libm_call(
        &mut self,
        code: usize,
        (ty, arity): (Type, usize),
        dest: Place<'db>,
        args: &[Place<'db>],
    ) -> Result<(), String> {
        let mut values = Vec::with_capacity(arity);
        for arg in args.iter().take(arity) {
            values.push(self.load(ty, arg.addr, 0));
        }
        let result = self.scalar_call(code, ty, &values);
        self.store(result, dest.addr, 0);
        Ok(())
    }
}
