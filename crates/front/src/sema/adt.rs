use std::rc::Rc;

use anyhow::{Result, anyhow};
use rustc_abi::{Align, Integer, IntegerType, LayoutCalculator, ReprFlags, ReprOptions, TargetDataLayout};
use rustc_index::IndexVec;

use crate::names::{ItemRef, Primitive};
use crate::outline::attr::split_commas;
use crate::outline::item::{Attr, ItemKind};

use super::Context;
use super::infer::field_types;
use super::layout::{Field, Layout, Variant};

const REPR_INTS: [(&str, Primitive); 12] = {
    use Primitive::*;
    [("u8", U8), ("u16", U16), ("u32", U32), ("u64", U64), ("u128", U128), ("usize", Usize), ("i8", I8), ("i16", I16), ("i32", I32), ("i64", I64), ("i128", I128), ("isize", Isize)]
};

impl<'a> Context<'a> {
    /// The discriminant of each variant of the enum `item`: its written value, or one more than the previous.
    pub fn discriminants(&mut self, item: ItemRef) -> Result<Rc<[i128]>> {
        if let Some(found) = self.discriminants.get(&item) {
            return Ok(found.clone());
        }
        let primitive = self.discriminant_primitive(item);
        let found: Rc<[i128]> = super::infer::discriminants(self, item, primitive)?.into();
        self.discriminants.insert(item, found.clone());
        Ok(found)
    }

    /// The integer type of the discriminants of the enum `item`: its `#[repr]` integer, or `isize`.
    pub fn discriminant_primitive(&self, item: ItemRef) -> Primitive {
        self.repr(item).int.map_or(Primitive::Isize, repr_primitive)
    }

    /// The `#[repr]` options of `item`.
    pub fn repr(&self, item: ItemRef) -> ReprOptions {
        let (src, entry) = self.item(item);
        let mut repr = ReprOptions::default();
        for attr in &entry.attrs {
            let Attr::Other(at) = *attr else { continue };
            if src.text(at) != "repr" || src.kind(at + 1) != rustc_lexer::TokenKind::OpenParen {
                continue;
            }
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
        repr
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
        if let Some(&known) = self.lang_of.borrow().get(&item) {
            return known;
        }
        let found = self.attr_value(item, "lang");
        self.lang_of.borrow_mut().insert(item, found);
        found
    }

    /// The value of the name-value attribute `#[key = "..."]` on `item`, written directly or through a `cfg_attr` that holds.
    pub fn attr_value(&self, item: ItemRef, key: &str) -> Option<&'a str> {
        let (src, entry) = self.item(item);
        let env = &self.outline.crates[item.krate as usize].env;
        let at = named_attr((src, env), &entry.attrs, key)?;
        Some(src.text(at + 2).trim_matches('"'))
    }

    /// The layout of the struct, enum or union `item` at `args`.
    pub fn adt_layout(&mut self, item: ItemRef, args: &[super::ty::Ty]) -> Result<Layout> {
        let fields = field_types(self, item, args)?;
        let sized = !super::infer::maybe_unsized(self, item)?;
        let mut layouts = Vec::with_capacity(fields.len());
        for variant in &fields {
            layouts.push(variant.iter().map(|ty| self.layout(*ty)).collect::<Result<Vec<_>>>()?);
        }
        let refs: Vec<Vec<&Layout>> = layouts.iter().map(|variant| variant.iter().map(|layout| &**layout).collect()).collect();
        let variants: IndexVec<Variant, IndexVec<Field, &&Layout>> = refs.iter().map(|variant| variant.iter().collect()).collect();
        let dl = super::layout::target()?;
        let calc = LayoutCalculator::new(&dl);
        let repr = self.repr(item);
        let kind = self.item(item).1.kind;
        let result = match kind {
            ItemKind::Union => calc.layout_of_union(&repr, &variants),
            _ => {
                let discriminants = if kind == ItemKind::Enum { self.discriminants(item)? } else { Rc::from([0]) };
                let no_niche = matches!(self.lang(item), Some("unsafe_cell" | "unsafe_pinned"));
                let tagged = discriminants.iter().enumerate().map(|(at, &value)| (rustc_index::Idx::new(at), value));
                calc.layout_of_struct_or_enum(&repr, &variants, kind == ItemKind::Enum, no_niche, |min, max| discr_range(&dl, &repr, min, max), tagged, sized)
            }
        };
        result.map_err(|error| anyhow!("layout of {}: {error:?}", self.name(item)))
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

fn discr_range(dl: &TargetDataLayout, repr: &ReprOptions, min: i128, max: i128) -> (Integer, bool) {
    match repr.int {
        Some(IntegerType::Fixed(integer, signed)) => return (integer, signed),
        Some(IntegerType::Pointer(signed)) => return (dl.ptr_sized_integer(), signed),
        None => {}
    }
    let unsigned = Integer::fit_unsigned(std::cmp::max(min as u128, max as u128));
    let signed = std::cmp::max(Integer::fit_signed(min), Integer::fit_signed(max));
    let at_least = if repr.c() { Integer::I32 } else { Integer::I8 };
    if min >= 0 { (std::cmp::max(unsigned, at_least), false) } else { (std::cmp::max(signed, at_least), true) }
}

fn named_attr((src, env): (&crate::outline::source::Source, &crate::outline::tree::Env), attrs: &[Attr], key: &str) -> Option<u32> {
    attrs.iter().find_map(|attr| match attr {
        Attr::Other(at) if src.text(*at) == key => Some(*at),
        Attr::CfgAttr(cfg, inner) if env.eval(cfg) == crate::outline::tree::Truth::Yes => named_attr((src, env), inner, key),
        _ => None,
    })
}
