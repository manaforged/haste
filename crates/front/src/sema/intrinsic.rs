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
    SizeOf,
    AlignOf,
    SizeOfVal,
    AlignOfVal,
    NeedsDrop,
    NoOp,
    BlackBox,
    Offset(bool),
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
}

const NAMES: [(&str, Intrinsic); 78] = {
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
        ("unreachable", Trap),
        ("caller_location", Trap),
        ("size_of", SizeOf),
        ("align_of", AlignOf),
        ("min_align_of", AlignOf),
        ("pref_align_of", AlignOf),
        ("size_of_val", SizeOfVal),
        ("align_of_val", AlignOfVal),
        ("min_align_of_val", AlignOfVal),
        ("needs_drop", NeedsDrop),
        ("forget", NoOp),
        ("assert_inhabited", NoOp),
        ("assert_zero_valid", NoOp),
        ("assert_mem_uninitialized_valid", NoOp),
        ("black_box", BlackBox),
        ("offset", Offset(true)),
        ("arith_offset", Offset(false)),
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
    ]
};

impl Context<'_> {
    /// The intrinsic that the function `item` is, when it carries `#[rustc_intrinsic]` and lowering knows it, or when it is the panic handler. `Trap` stands for the panic runtime, which is not lowered yet, and for code that must never run.
    pub fn intrinsic(&self, item: ItemRef) -> Option<Intrinsic> {
        let (src, entry) = self.item(item);
        if self.lang(item) == Some("panic_impl") {
            return Some(Intrinsic::Trap);
        }
        let marked = entry.kind == ItemKind::Fn && entry.attrs.iter().any(|attr| matches!(attr, Attr::Other(at) if src.text(*at) == "rustc_intrinsic"));
        let name = src.text(entry.name?);
        marked.then(|| NAMES.iter().find(|(known, _)| *known == name).map(|(_, which)| *which)).flatten()
    }
}
