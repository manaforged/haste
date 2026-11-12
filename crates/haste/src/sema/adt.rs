use std::sync::Arc;

use anyhow::{Result, anyhow};
use rustc_abi::{Align, Integer, IntegerType, LayoutCalculator, NicheOptimizations, ReprFlags, ReprOptions, TargetDataLayout};
use rustc_index::IndexVec;

use crate::names::{ItemRef, Primitive};
use crate::outline::attr::split_commas;
use crate::outline::item::{Attr, ItemKind};
use crate::outline::source::Source;
use crate::outline::tree::Truth;

use super::Context;
use super::infer::field_types;
use super::layout::{Field, Layout, Variant};

const REPR_INTS: [(&str, Primitive); 12] = {
    use Primitive::*;
    [("u8", U8), ("u16", U16), ("u32", U32), ("u64", U64), ("u128", U128), ("usize", Usize), ("i8", I8), ("i16", I16), ("i32", I32), ("i64", I64), ("i128", I128), ("isize", Isize)]
};

impl<'a> Context<'a> {
    /// The discriminant of each variant of the enum `item`: its written value, or one more than the previous.
    pub fn discriminants(&mut self, item: ItemRef) -> Result<Arc<[i128]>> {
        if let Some(found) = self.store.lookups.discriminants.get(&item) {
            return Ok(found);
        }
        let before = self.quiet_mark();
        let primitive = self.discriminant_primitive(item);
        let found: Arc<[i128]> = super::infer::discriminants(self, item, primitive)?.into();
        if self.quiet_mark() == before {
            self.store.lookups.discriminants.insert(item, Arc::clone(&found));
        }
        Ok(found)
    }

    /// The integer type of the discriminants of the enum `item`: its `#[repr]` integer, or `isize`.
    pub fn discriminant_primitive(&self, item: ItemRef) -> Primitive {
        self.repr(item).int.map_or(Primitive::Isize, repr_primitive)
    }

    /// The `#[repr]` options of `item`, written directly or through a `cfg_attr` that holds.
    pub fn repr(&self, item: ItemRef) -> ReprOptions {
        let (src, entry) = self.item(item);
        let mut repr = ReprOptions::default();
        self.read_repr(item, (src, &entry.attrs), &mut repr);
        repr
    }

    fn read_repr(&self, item: ItemRef, (src, attrs): (&Source, &[Attr]), repr: &mut ReprOptions) {
        for attr in attrs {
            match attr {
                Attr::CfgAttr(cfg, inner) if self.outline.crates[item.krate as usize].env.eval(cfg) == Truth::Yes => self.read_repr(item, (src, inner), repr),
                &Attr::Other(at) if src.text(at) == "repr" && src.kind(at + 1) == rustc_lexer::TokenKind::OpenParen => read_repr_words(src, at, repr),
                _ => {}
            }
        }
    }

    /// The argument positions that `#[rustc_legacy_const_generics(..)]` on the function `item` gives its const generics, in order.
    pub fn legacy_const_args(&self, item: ItemRef) -> Vec<usize> {
        let (src, entry) = self.item(item);
        let named = entry.attrs.iter().find_map(|attr| match *attr {
            Attr::Other(at) if src.text(at) == "rustc_legacy_const_generics" && src.kind(at + 1) == rustc_lexer::TokenKind::OpenParen => Some(at),
            _ => None,
        });
        let Some(at) = named else { return Vec::new() };
        split_commas(src, at + 2, src.pair(at + 1)).into_iter().filter_map(|(from, _)| src.text(from).parse().ok()).collect()
    }

    /// The value of `#[lang = "..."]` on `item`.
    pub fn lang(&self, item: ItemRef) -> Option<&'a str> {
        self.attr_value(item, "lang")
    }

    /// The value of the name-value attribute `#[key = "..."]` on `item`, written directly or through a `cfg_attr` that holds.
    pub fn attr_value(&self, item: ItemRef, key: &str) -> Option<&'a str> {
        let at = self.attr_at(item, key)?;
        Some(self.item(item).0.text(at + 2).trim_matches('"'))
    }

    /// Whether the static `item` is `#[thread_local]`, with one cell per thread.
    pub fn thread_local(&self, item: ItemRef) -> bool {
        self.attr_at(item, "thread_local").is_some()
    }

    /// Whether the function `item` is `#[inline(always)]`.
    pub fn inline_always(&self, item: ItemRef) -> bool {
        self.attr_at(item, "inline").is_some_and(|at| self.item(item).0.text(at + 2) == "always")
    }

    /// Whether the const function `item` is `#[rustc_promotable]`, so a reference to its call on promotable arguments lives for the whole program.
    pub fn rustc_promotable(&self, item: ItemRef) -> bool {
        self.attr_at(item, "rustc_promotable").is_some()
    }

    /// The token of the attribute `key` that `item` carries, through `cfg_attr` whose cfg holds.
    pub(super) fn attr_at(&self, item: ItemRef, key: &str) -> Option<u32> {
        let file = self.file(item);
        let index = item.outer().map_or(item.item, |(_, index)| index);
        if !file.items.may_carry(index, key) {
            return None;
        }
        let (src, entry) = (&file.source, &file.items.items[index as usize]);
        let pick = |attr: &Attr| {
            if let Attr::Other(at) = attr { (src.text(*at) == key).then_some(*at) } else { None }
        };
        self.outline.crates[item.krate as usize].env.find_attr(&entry.attrs, crate::outline::tree::Truth::Yes, &pick)
    }

    /// The layout of the struct, enum or union `item` at `args`.
    pub fn adt_layout(&mut self, item: ItemRef, args: &[super::ty::Ty]) -> Result<Layout> {
        let fields = field_types(self, item, args)?;
        let sized = !super::infer::maybe_unsized(self, item)?;
        let mut layouts = Vec::with_capacity(fields.len());
        for variant in fields.iter() {
            layouts.push(variant.iter().map(|ty| self.layout(*ty)).collect::<Result<Vec<_>>>()?);
        }
        let refs: Vec<Vec<&Layout>> = layouts.iter().map(|variant| variant.iter().map(|layout| &**layout).collect()).collect();
        let variants: IndexVec<Variant, IndexVec<Field, &&Layout>> = refs.iter().map(|variant| variant.iter().collect()).collect();
        let dl = super::layout::target()?;
        let calc = LayoutCalculator::new(dl);
        let repr = self.repr(item);
        let kind = self.item(item).1.kind;
        let result = match kind {
            ItemKind::Union => calc.layout_of_union(&repr, &variants),
            _ => {
                let discriminants = if kind == ItemKind::Enum { self.discriminants(item)? } else { Arc::from([0]) };
                let niches = if matches!(self.lang(item), Some("unsafe_cell" | "unsafe_pinned")) { NicheOptimizations::Disabled } else { NicheOptimizations::Enabled };
                let bits = Integer::from_attr(dl, repr.discr_type()).size();
                let tagged = discriminants.iter().enumerate().map(|(at, &value)| (rustc_index::Idx::new(at), bits.truncate(value as u128)));
                calc.layout_of_struct_or_enum(&repr, &variants, kind == ItemKind::Enum, niches, |min, max| discr_range(dl, &repr, min.start, max.last), tagged, sized)
            }
        };
        result.map_err(|error| anyhow!("layout of {}: {error:?}", self.name(item)))
    }
}

/// Adds the words of the `#[repr(..)]` attribute at `at` to `repr`.
fn read_repr_words(src: &Source, at: u32, repr: &mut ReprOptions) {
    for (from, _) in split_commas(src, at + 2, src.pair(at + 1)) {
        match src.text(from) {
            "C" => repr.flags |= ReprFlags::IS_C,
            "transparent" => repr.flags |= ReprFlags::IS_TRANSPARENT,
            "packed" => repr.pack = Some(repr_align(src, from).unwrap_or(Align::ONE)),
            "align" => repr.align = repr_align(src, from).or(repr.align),
            word => repr.int = REPR_INTS.iter().find(|(name, _)| *name == word).map(|(_, int)| integer_type(*int)).or(repr.int),
        }
    }
}

/// The alignment that the `(N)` after the `#[repr]` word at `at` gives, as in `packed(2)` or `align(16)`.
fn repr_align(src: &crate::outline::source::Source, at: u32) -> Option<Align> {
    if src.kind(at + 1) != rustc_lexer::TokenKind::OpenParen {
        return None;
    }
    let bytes = super::infer::parse_int(src.text(at + 2)).ok()?.0;
    Align::from_bytes(u64::try_from(bytes).ok()?).ok()
}

/// The integer type that the `#[repr]` integer `int` names.
pub(super) fn repr_primitive(int: IntegerType) -> Primitive {
    REPR_INTS.iter().map(|(_, primitive)| *primitive).find(|primitive| integer_type(*primitive) == int).unwrap_or(Primitive::Isize)
}

fn integer_type(primitive: Primitive) -> IntegerType {
    let (bits, signed) = super::ty::int_info(primitive).expect("repr integers are integers");
    match primitive {
        Primitive::Usize | Primitive::Isize => IntegerType::Pointer(signed),
        _ => IntegerType::Fixed(Integer::from_size(rustc_abi::Size::from_bits(bits)).expect("a repr integer has a size"), signed),
    }
}

fn discr_range(dl: &TargetDataLayout, repr: &ReprOptions, min: i128, max: u128) -> (Integer, bool) {
    match repr.int {
        Some(IntegerType::Fixed(integer, signed)) => return (integer, signed),
        Some(IntegerType::Pointer(signed)) => return (dl.ptr_sized_integer(), signed),
        None => {}
    }
    let unsigned = Integer::fit_unsigned(max);
    let signed = std::cmp::max(Integer::fit_signed(min), Integer::fit_signed(i128::try_from(max).unwrap_or(i128::MAX)));
    let at_least = if repr.c() { Integer::I32 } else { Integer::I8 };
    if min >= 0 { (std::cmp::max(unsigned, at_least), false) } else { (std::cmp::max(signed, at_least), true) }
}
