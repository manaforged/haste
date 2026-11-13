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
    FloatToInt,
    Transmute,
    PtrMetadata,
    WithOverflow(BinOp),
    ConstEvalSelect,
    SelectUnpredictable,
    Trap,
    CallerLocation,
    Cpuid,
    /// `FnPtr::addr`, the address of a function pointer.
    FnAddr,
    /// SSSE3 or AVX2 `pshufb` on a vector of this many bytes.
    BytePermute(u8),
    /// NEON `umaxp` on two vectors of 16 bytes.
    BytePairMax,
    /// An ARM64 NEON or SHA-256 intrinsic emitted inline.
    Arm64(Arm64Intrinsic),
    /// An LLVM x86 intrinsic that a host function computes.
    X86(crate::host::X86Symbol),
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
    Select,
    SelectBitmask,
    Shuffle,
    Extract,
    Insert,
    /// Whether every lane, or any lane, of a mask is set.
    Reduce {
        all: bool,
    },
    /// The largest, or smallest, lane of a vector.
    Extreme {
        max: bool,
    },
    /// Every lane of a vector combined by one bitwise operator.
    Fold(BinOp),
    Sqrt,
    /// Each lane through one operation on that lane alone.
    Unary(LaneUnary),
    /// Each pair of lanes through one operation that has no operator.
    Pair(LanePair),
    /// Each lane of the first vector times the second plus the third, rounded once.
    Fma,
    /// The sum, or product, of every lane in order, after the start value when one is given.
    Sum {
        mul: bool,
    },
    /// Each lane read from the pointer where the mask lane is set, and taken from the fallback where not.
    MaskedLoad,
    /// Each lane written through the pointer where the mask lane is set.
    MaskedStore,
}

/// An operation on one SIMD lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneUnary {
    Neg,
    Ctpop,
    Ctlz,
    Fabs,
    Floor,
    Ceil,
}

/// An operation on two SIMD lanes that no binary operator spells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanePair {
    Min,
    Max,
    SaturatingAdd,
    SaturatingSub,
}

/// An atomic access, which lowering makes sequentially consistent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Atomic {
    Load,
    Cache,
    Rmw(AtomicRmwOp),
    Exchange,
}

/// An ARM64 NEON or crypto intrinsic emitted inline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm64Intrinsic {
    Tbl1,
    Sqdmulh2S,
    Sqdmulh4S,
    Sqdmulh8H,
    Sha256H,
    Sha256H2,
    Sha256Su0,
    Sha256Su1,
}

/// The LLVM intrinsics that foreign declarations link to and lowering emits inline.
const LLVM: [(&str, Intrinsic); 11] = [
    ("llvm.x86.ssse3.pshuf.b.128", Intrinsic::BytePermute(16)),
    ("llvm.x86.avx2.pshuf.b", Intrinsic::BytePermute(32)),
    ("llvm.aarch64.neon.umaxp.v16i8", Intrinsic::BytePairMax),
    ("llvm.aarch64.neon.tbl1.v16i8", Intrinsic::Arm64(Arm64Intrinsic::Tbl1)),
    ("llvm.aarch64.neon.sqdmulh.v2i32", Intrinsic::Arm64(Arm64Intrinsic::Sqdmulh2S)),
    ("llvm.aarch64.neon.sqdmulh.v4i32", Intrinsic::Arm64(Arm64Intrinsic::Sqdmulh4S)),
    ("llvm.aarch64.neon.sqdmulh.v8i16", Intrinsic::Arm64(Arm64Intrinsic::Sqdmulh8H)),
    ("llvm.aarch64.crypto.sha256h", Intrinsic::Arm64(Arm64Intrinsic::Sha256H)),
    ("llvm.aarch64.crypto.sha256h2", Intrinsic::Arm64(Arm64Intrinsic::Sha256H2)),
    ("llvm.aarch64.crypto.sha256su0", Intrinsic::Arm64(Arm64Intrinsic::Sha256Su0)),
    ("llvm.aarch64.crypto.sha256su1", Intrinsic::Arm64(Arm64Intrinsic::Sha256Su1)),
];

const NAMES: [(&str, Intrinsic); 161] = {
    use Intrinsic::*;
    [
        ("cold_path", NoOp),
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
        ("float_to_int_unchecked", FloatToInt),
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
        ("select_unpredictable", SelectUnpredictable),
        ("abort", Trap),
        ("catch_unwind", CatchUnwind),
        ("unreachable", Trap),
        ("caller_location", CallerLocation),
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
        ("simd_select", Simd(self::Simd::Select)),
        ("simd_select_bitmask", Simd(self::Simd::SelectBitmask)),
        ("simd_extract", Simd(self::Simd::Extract)),
        ("simd_extract_dyn", Simd(self::Simd::Extract)),
        ("simd_insert", Simd(self::Simd::Insert)),
        ("simd_insert_dyn", Simd(self::Simd::Insert)),
        ("simd_reduce_all", Simd(self::Simd::Reduce { all: true })),
        ("simd_reduce_any", Simd(self::Simd::Reduce { all: false })),
        ("simd_reduce_max", Simd(self::Simd::Extreme { max: true })),
        ("simd_reduce_min", Simd(self::Simd::Extreme { max: false })),
        ("simd_reduce_and", Simd(self::Simd::Fold(BinOp::BitAnd))),
        ("simd_reduce_or", Simd(self::Simd::Fold(BinOp::BitOr))),
        ("simd_reduce_xor", Simd(self::Simd::Fold(BinOp::BitXor))),
        ("simd_fsqrt", Simd(self::Simd::Sqrt)),
        ("simd_neg", Simd(self::Simd::Unary(LaneUnary::Neg))),
        ("simd_ctpop", Simd(self::Simd::Unary(LaneUnary::Ctpop))),
        ("simd_ctlz", Simd(self::Simd::Unary(LaneUnary::Ctlz))),
        ("simd_fabs", Simd(self::Simd::Unary(LaneUnary::Fabs))),
        ("simd_floor", Simd(self::Simd::Unary(LaneUnary::Floor))),
        ("simd_ceil", Simd(self::Simd::Unary(LaneUnary::Ceil))),
        ("simd_imin", Simd(self::Simd::Pair(LanePair::Min))),
        ("simd_imax", Simd(self::Simd::Pair(LanePair::Max))),
        ("simd_fmin", Simd(self::Simd::Pair(LanePair::Min))),
        ("simd_fmax", Simd(self::Simd::Pair(LanePair::Max))),
        ("simd_minimum_number_nsz", Simd(self::Simd::Pair(LanePair::Min))),
        ("simd_maximum_number_nsz", Simd(self::Simd::Pair(LanePair::Max))),
        ("simd_saturating_add", Simd(self::Simd::Pair(LanePair::SaturatingAdd))),
        ("simd_saturating_sub", Simd(self::Simd::Pair(LanePair::SaturatingSub))),
        ("simd_fma", Simd(self::Simd::Fma)),
        ("simd_relaxed_fma", Simd(self::Simd::Fma)),
        ("simd_reduce_add_ordered", Simd(self::Simd::Sum { mul: false })),
        ("simd_reduce_add_unordered", Simd(self::Simd::Sum { mul: false })),
        ("simd_reduce_mul_ordered", Simd(self::Simd::Sum { mul: true })),
        ("simd_reduce_mul_unordered", Simd(self::Simd::Sum { mul: true })),
        ("simd_masked_load", Simd(self::Simd::MaskedLoad)),
        ("simd_masked_store", Simd(self::Simd::MaskedStore)),
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
        match self.lang(item) {
            Some("panic_impl") => return Some(Intrinsic::Trap),
            Some("fn_ptr_addr") => return Some(Intrinsic::FnAddr),
            _ => {}
        }
        let marked = self.rustc_intrinsic(item);
        let name = src.text(entry.name?);
        if let Some(link) = self.attr_value(item, "link_name")
            && let Some(&(_, which)) = LLVM.iter().find(|(known, _)| *known == link)
        {
            return Some(which);
        }
        if let Some(link) = self.attr_value(item, "link_name")
            && let Some(x86) = crate::host::X86Symbol::from_link(link)
        {
            return Some(Intrinsic::X86(x86));
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
