use haste_codegen::AtomicRmwOp;

use crate::body::BinOp;
use crate::names::ItemRef;
use crate::outline::item::{Attr, ItemKind};

use super::Context;

/// A compiler intrinsic that lowering emits inline instead of calling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    WrappingAdd,
    WrappingSub,
    WrappingMul,
    SaturatingAdd,
    SaturatingSub,
    RotateLeft,
    RotateRight,
    Ctpop,
    Ctlz,
    Cttz,
    Bswap,
    Bitreverse,
    Fabs,
    Sqrt,
    Floor,
    Ceil,
    Trunc,
    Copysign,
    Round(u8),
    Transmute,
    PtrMetadata,
    WithOverflow(BinOp),
    ConstEvalSelect,
    Trap,
    Cpuid,
    /// SSSE3 or AVX2 `pshufb` on a vector of this many bytes.
    BytePermute(u8),
    CatchUnwind,
    SizeOf,
    AlignOf,
    SizeOfVal,
    AlignOfVal,
    NeedsDrop,
    TypeId,
    TypeName,
    NoOp,
    BlackBox,
    Offset(bool),
    /// A thin pointer with only the address bits a mask keeps.
    PtrMask,
    OffsetFrom(bool),
    Read,
    Write,
    WriteBox,
    MemCopy(bool),
    WriteBytes,
    Unchecked(BinOp),
    ThreeWayCompare,
    Fence,
    RawPointer,
    DiscriminantValue,
    CompareBytes,
    RawEq,
    Atomic(Atomic),
    SliceElement,
    Simd(Simd),
}

/// A SIMD intrinsic, which lowering runs one lane at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Simd {
    Splat,
    Bitmask,
    Lanewise(BinOp),
    /// Each lane converted as `as` converts it.
    Cast,
    Shuffle,
    Extract,
    Insert,
    /// Whether every lane, or any lane, of a mask is set.
    Reduce { all: bool },
    /// The largest, or smallest, lane of a vector.
    Extreme { max: bool },
}

/// An atomic access, which lowering makes sequentially consistent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Atomic {
    Load,
    Cache,
    Rmw(AtomicRmwOp),
    Exchange,
}

/// The LLVM intrinsics that foreign declarations link to and lowering emits inline.
const LLVM: [(&str, Intrinsic); 2] = [("llvm.x86.ssse3.pshuf.b.128", Intrinsic::BytePermute(16)), ("llvm.x86.avx2.pshuf.b", Intrinsic::BytePermute(32))];

const NAMES: [(&str, Intrinsic); 130] = {
    use Intrinsic::*;
    [
        ("wrapping_add", WrappingAdd),
        ("wrapping_sub", WrappingSub),
        ("wrapping_mul", WrappingMul),
        ("unchecked_add", WrappingAdd),
        ("unchecked_sub", WrappingSub),
        ("unchecked_mul", WrappingMul),
        ("saturating_add", SaturatingAdd),
        ("saturating_sub", SaturatingSub),
        ("rotate_left", RotateLeft),
        ("rotate_right", RotateRight),
        ("ctpop", Ctpop),
        ("ctlz", Ctlz),
        ("ctlz_nonzero", Ctlz),
        ("cttz", Cttz),
        ("cttz_nonzero", Cttz),
        ("bswap", Bswap),
        ("bitreverse", Bitreverse),
        ("fabsf32", Fabs),
        ("fabsf64", Fabs),
        ("sqrtf32", Sqrt),
        ("sqrtf64", Sqrt),
        ("floorf32", Floor),
        ("floorf64", Floor),
        ("ceilf32", Ceil),
        ("ceilf64", Ceil),
        ("truncf32", Trunc),
        ("truncf64", Trunc),
        ("copysignf32", Copysign),
        ("copysignf64", Copysign),
        ("fabs", Fabs),
        ("sqrt", Sqrt),
        ("floor", Floor),
        ("roundf32", Round(32)),
        ("roundf64", Round(64)),
        ("transmute", Transmute),
        ("transmute_unchecked", Transmute),
        ("atomic_fence", Fence),
        ("aggregate_raw_ptr", RawPointer),
        ("atomic_singlethreadfence", NoOp),
        ("ptr_metadata", PtrMetadata),
        ("add_with_overflow", WithOverflow(BinOp::Add)),
        ("sub_with_overflow", WithOverflow(BinOp::Sub)),
        ("mul_with_overflow", WithOverflow(BinOp::Mul)),
        ("const_eval_select", ConstEvalSelect),
        ("abort", Trap),
        ("catch_unwind", CatchUnwind),
        ("unreachable", Trap),
        ("caller_location", Trap),
        ("size_of", SizeOf),
        ("align_of", AlignOf),
        ("min_align_of", AlignOf),
        ("pref_align_of", AlignOf),
        ("size_of_val", SizeOfVal),
        ("discriminant_value", DiscriminantValue),
        ("compare_bytes", CompareBytes),
        ("raw_eq", RawEq),
        ("align_of_val", AlignOfVal),
        ("min_align_of_val", AlignOfVal),
        ("needs_drop", NeedsDrop),
        ("type_id", TypeId),
        ("type_name", TypeName),
        ("forget", NoOp),
        ("assert_inhabited", NoOp),
        ("assert_zero_valid", NoOp),
        ("assert_mem_uninitialized_valid", NoOp),
        ("black_box", BlackBox),
        ("offset", Offset(true)),
        ("arith_offset", Offset(false)),
        ("ptr_mask", PtrMask),
        ("ptr_offset_from", OffsetFrom(true)),
        ("ptr_offset_from_unsigned", OffsetFrom(false)),
        ("read_via_copy", Read),
        ("volatile_load", Read),
        ("write_via_move", Write),
        ("volatile_store", Write),
        ("write_box_via_move", WriteBox),
        ("copy_nonoverlapping", MemCopy(false)),
        ("copy", MemCopy(true)),
        ("write_bytes", WriteBytes),
        ("unchecked_div", Unchecked(BinOp::Div)),
        ("exact_div", Unchecked(BinOp::Div)),
        ("unchecked_rem", Unchecked(BinOp::Rem)),
        ("unchecked_shl", Unchecked(BinOp::Shl)),
        ("unchecked_shr", Unchecked(BinOp::Shr)),
        ("three_way_compare", ThreeWayCompare),
        ("slice_get_unchecked", SliceElement),
        ("atomic_load", Atomic(self::Atomic::Load)),
        ("atomic_store", Atomic(self::Atomic::Cache)),
        ("atomic_cxchg", Atomic(self::Atomic::Exchange)),
        ("atomic_cxchgweak", Atomic(self::Atomic::Exchange)),
        ("atomic_xchg", Atomic(self::Atomic::Rmw(AtomicRmwOp::Xchg))),
        ("atomic_xadd", Atomic(self::Atomic::Rmw(AtomicRmwOp::Add))),
        ("atomic_xsub", Atomic(self::Atomic::Rmw(AtomicRmwOp::Sub))),
        ("atomic_and", Atomic(self::Atomic::Rmw(AtomicRmwOp::And))),
        ("atomic_nand", Atomic(self::Atomic::Rmw(AtomicRmwOp::Nand))),
        ("atomic_or", Atomic(self::Atomic::Rmw(AtomicRmwOp::Or))),
        ("atomic_xor", Atomic(self::Atomic::Rmw(AtomicRmwOp::Xor))),
        ("atomic_max", Atomic(self::Atomic::Rmw(AtomicRmwOp::Smax))),
        ("atomic_min", Atomic(self::Atomic::Rmw(AtomicRmwOp::Smin))),
        ("atomic_umax", Atomic(self::Atomic::Rmw(AtomicRmwOp::Umax))),
        ("atomic_umin", Atomic(self::Atomic::Rmw(AtomicRmwOp::Umin))),
        ("simd_splat", Simd(self::Simd::Splat)),
        ("simd_bitmask", Simd(self::Simd::Bitmask)),
        ("simd_shuffle", Simd(self::Simd::Shuffle)),
        ("simd_cast", Simd(self::Simd::Cast)),
        ("simd_as", Simd(self::Simd::Cast)),
        ("simd_extract", Simd(self::Simd::Extract)),
        ("simd_extract_dyn", Simd(self::Simd::Extract)),
        ("simd_insert", Simd(self::Simd::Insert)),
        ("simd_insert_dyn", Simd(self::Simd::Insert)),
        ("simd_reduce_all", Simd(self::Simd::Reduce { all: true })),
        ("simd_reduce_any", Simd(self::Simd::Reduce { all: false })),
        ("simd_reduce_max", Simd(self::Simd::Extreme { max: true })),
        ("simd_reduce_min", Simd(self::Simd::Extreme { max: false })),
        ("simd_add", Simd(self::Simd::Lanewise(BinOp::Add))),
        ("simd_sub", Simd(self::Simd::Lanewise(BinOp::Sub))),
        ("simd_mul", Simd(self::Simd::Lanewise(BinOp::Mul))),
        ("simd_and", Simd(self::Simd::Lanewise(BinOp::BitAnd))),
        ("simd_or", Simd(self::Simd::Lanewise(BinOp::BitOr))),
        ("simd_xor", Simd(self::Simd::Lanewise(BinOp::BitXor))),
        ("simd_shl", Simd(self::Simd::Lanewise(BinOp::Shl))),
        ("simd_shr", Simd(self::Simd::Lanewise(BinOp::Shr))),
        ("simd_div", Simd(self::Simd::Lanewise(BinOp::Div))),
        ("simd_rem", Simd(self::Simd::Lanewise(BinOp::Rem))),
        ("simd_eq", Simd(self::Simd::Lanewise(BinOp::Eq))),
        ("simd_ne", Simd(self::Simd::Lanewise(BinOp::Ne))),
        ("simd_lt", Simd(self::Simd::Lanewise(BinOp::Lt))),
        ("simd_le", Simd(self::Simd::Lanewise(BinOp::Le))),
        ("simd_gt", Simd(self::Simd::Lanewise(BinOp::Gt))),
        ("simd_ge", Simd(self::Simd::Lanewise(BinOp::Ge))),
    ]
};

impl Context<'_> {
    /// The intrinsic that the function `item` is, when it carries `#[rustc_intrinsic]` and lowering knows it, or when it is the panic handler. `Trap` stands for the panic runtime, which is not lowered yet, and for code that must never run.
    pub fn intrinsic(&self, item: ItemRef) -> Option<Intrinsic> {
        let (src, entry) = self.item(item);
        if self.lang(item) == Some("panic_impl") {
            return Some(Intrinsic::Trap);
        }
        let marked = self.rustc_intrinsic(item);
        let name = src.text(entry.name?);
        if let Some(link) = self.attr_value(item, "link_name")
            && let Some(&(_, which)) = LLVM.iter().find(|(known, _)| *known == link)
        {
            return Some(which);
        }
        if entry.kind == ItemKind::Fn && name == "__cpuid_count" && self.outline.roots[item.krate as usize].name == "core" {
            return Some(Intrinsic::Cpuid);
        }
        marked.then(|| NAMES.iter().find(|(known, _)| *known == name).map(|(_, which)| *which)).flatten()
    }

    /// Whether the function `item` carries `#[rustc_intrinsic]`.
    pub fn rustc_intrinsic(&self, item: ItemRef) -> bool {
        let (src, entry) = self.item(item);
        entry.kind == ItemKind::Fn && entry.attrs.iter().any(|attr| matches!(attr, Attr::Other(at) if src.text(*at) == "rustc_intrinsic"))
    }
}
