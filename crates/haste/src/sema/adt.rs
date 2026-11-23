use std::ops::ControlFlow;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use rustc_abi::{Align, FieldsShape, Integer, IntegerType, LayoutCalculator, NicheOptimizations, ReprFlags, ReprOptions, TargetDataLayout};
use rustc_hash::FxHashMap as HashMap;
use rustc_index::{Idx, IndexVec};

use crate::names::{ItemRef, Primitive};
use crate::outline::attr::{repr_words, split_commas};
use crate::outline::item::{Attr, ItemKind, MARK_CFG_ATTR, MARK_INLINE_SPEC, mark};
use crate::outline::source::Source;
use crate::outline::tree::{Env, Truth};

use super::Context;
use super::infer::field_types;
use super::layout::{Field, Layout, Variant};
use super::lookups::ActiveAttrs;
use super::ty::Kind;

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
        if entry.attrs.is_empty() || !self.has_mark(item, "repr") {
            return ReprOptions::default();
        }
        self.store.lookups.reprs.get_or_insert_with(item, || {
            let mut repr = ReprOptions::default();
            read_repr_words(src, repr_words(src, &entry.attrs, &self.outline.crates[item.krate as usize].env), &mut repr);
            repr
        })
    }

    /// The argument positions that `#[rustc_legacy_const_generics(..)]` on the function `item` gives its const generics, in order.
    pub fn legacy_const_args(&self, item: ItemRef) -> Vec<usize> {
        let (src, entry) = self.item(item);
        if entry.attrs.is_empty() || !self.has_mark(item, "rustc_legacy_const_generics") {
            return Vec::new();
        }
        let named = entry.attrs.iter().find_map(|attr| match *attr {
            Attr::Other(at) if src.text(at) == "rustc_legacy_const_generics" && src.kind(at + 1) == rustc_lexer::TokenKind::OpenParen => Some(at),
            _ => None,
        });
        let Some(at) = named else { return Vec::new() };
        split_commas(src, at + 2, src.pair(at + 1)).into_iter().filter_map(|(from, _)| src.text(from).parse().ok()).collect()
    }

    /// The value of `#[lang = "..."]` on `item`.
    pub fn lang(&self, item: ItemRef) -> Option<&'a str> {
        let (src, entry) = self.item(item);
        if entry.attrs.is_empty() {
            return None;
        }
        if item.outer().is_some() {
            let at = active_attr_at(&self.outline.crates[item.krate as usize].env, src, &entry.attrs, "lang")?;
            return Some(src.text(at + 2).trim_matches('"'));
        }
        let map = self.store.lookups.lang_items.get_or_init(|| self.build_lang_map());
        let &val = map.get(&item)?;
        Some(src.text(val).trim_matches('"'))
    }

    fn build_lang_map(&self) -> HashMap<ItemRef, u32> {
        let mut map = HashMap::default();
        let lang_bit = mark("lang");
        for (krate, tree) in self.outline.crates.iter().enumerate() {
            for (file, entry) in tree.files.iter().enumerate() {
                for &(at, bits) in &entry.items.marks {
                    if bits & lang_bit != 0
                        && let Some(attr_tok) = active_attr_at(&tree.env, &entry.source, &entry.items.items[at as usize].attrs, "lang")
                    {
                        map.insert(ItemRef { krate: krate as u32, file: file as u32, item: at }, attr_tok + 2);
                    }
                }
            }
        }
        map
    }

    /// The symbol that the foreign `item` links to: its `#[link_name]` with escapes read and without the leading `\u{1}` that asks for the name verbatim, else its own name.
    pub fn symbol_name(&self, item: ItemRef) -> Option<String> {
        let (src, entry) = self.item(item);
        let Some(at) = self.attr_at(item, "link_name") else { return entry.name.map(|at| src.text(at).to_owned()) };
        let text = src.literal_text(at + 2).ok().flatten()?;
        Some(text.strip_prefix('\u{1}').map(str::to_owned).unwrap_or(text))
    }

    /// The value of `#[inline]` on `item`, returning `Some("")` for plain `#[inline]` without scanning attributes.
    pub fn inline_attr(&self, item: ItemRef) -> Option<&'a str> {
        let (_, entry) = self.item(item);
        if entry.attrs.is_empty() {
            return None;
        }
        if let Some(bits) = self.item_marks(item) {
            if bits & mark("inline") == 0 {
                return None;
            }
            if bits & (MARK_CFG_ATTR | MARK_INLINE_SPEC) == 0 {
                return Some("");
            }
        }
        self.attr_value(item, "inline")
    }

    /// The value of the name-value attribute `#[key = "..."]` on `item`, written directly or through a `cfg_attr` that holds.
    pub fn attr_value(&self, item: ItemRef, key: &str) -> Option<&'a str> {
        let at = self.attr_at(item, key)?;
        Some(self.item(item).0.text(at + 2).trim_matches('"'))
    }

    /// Whether the static `item` is `#[thread_local]`, with one cell per thread.
    pub fn thread_local(&self, item: ItemRef) -> bool {
        self.has_attr(item, "thread_local")
    }

    /// Whether arithmetic overflow checks are enabled for crate `krate`.
    pub fn overflow_checks(&self, krate: u32) -> bool {
        self.outline.crates[krate as usize].env.overflow_checks()
    }

    /// Whether the function `item` carries `#[track_caller]`, directly or inherited from its trait method declaration.
    pub fn track_caller(&mut self, item: ItemRef) -> bool {
        if self.has_attr(item, "track_caller") {
            return true;
        }
        let Some(block) = self.owner_impl(item) else {
            return false;
        };
        let Some(tr) = self.impl_trait(block) else {
            return false;
        };
        let Some(member) = self.trait_member(tr, self.plain_name(item)) else {
            return false;
        };
        self.has_attr(member, "track_caller")
    }

    /// Whether the const function `item` is `#[rustc_promotable]`, so a reference to its call on promotable arguments lives for the whole program.
    pub fn rustc_promotable(&self, item: ItemRef) -> bool {
        self.has_attr(item, "rustc_promotable")
    }

    #[inline]
    fn item_marks(&self, item: ItemRef) -> Option<u16> {
        if item.outer().is_some() {
            return None;
        }
        let marks = &self.file(item).items.marks;
        Some(marks.binary_search_by_key(&item.item, |(at, _)| *at).map_or(0, |pos| marks[pos].1))
    }

    fn has_mark(&self, item: ItemRef, key: &str) -> bool {
        let bit = mark(key);
        bit == 0 || self.item_marks(item).is_none_or(|bits| bits & bit != 0)
    }

    /// Whether `item` carries attribute `key`, directly or through an active `cfg_attr`.
    pub fn has_attr(&self, item: ItemRef, key: &str) -> bool {
        let (_, entry) = self.item(item);
        if entry.attrs.is_empty() {
            return false;
        }
        let bit = mark(key);
        if bit != 0
            && let Some(bits) = self.item_marks(item)
        {
            if bits & bit == 0 {
                return false;
            }
            if bits & MARK_CFG_ATTR == 0 {
                return true;
            }
        }
        self.attr_at(item, key).is_some()
    }

    /// The token of the attribute `key` that `item` carries, through `cfg_attr` whose cfg holds.
    pub(super) fn attr_at(&self, item: ItemRef, key: &str) -> Option<u32> {
        let (src, entry) = self.item(item);
        if entry.attrs.is_empty() {
            return None;
        }
        let bit = mark(key);
        let no_cfg = if bit != 0
            && let Some(bits) = self.item_marks(item)
        {
            if bits & bit == 0 {
                return None;
            }
            bits & MARK_CFG_ATTR == 0
        } else {
            !entry.attrs.iter().any(|attr| matches!(attr, Attr::CfgAttr(..)))
        };
        if no_cfg {
            return entry.attrs.iter().find_map(|attr| match *attr {
                Attr::Other(at) if src.text(at) == key => Some(at),
                _ => None,
            });
        }
        let active = self.store.lookups.attrs.get_or_insert_with(item, || {
            let mut out = Vec::new();
            collect_active_attrs(&self.outline.crates[item.krate as usize].env, &entry.attrs, &mut out);
            ActiveAttrs::from_vec(out)
        });
        active.as_slice().iter().copied().find(|&at| src.text(at) == key)
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
        if repr.simd() && kind == ItemKind::Struct && fields.len() == 1 && !fields[0].is_empty() {
            if fields[0].len() == 1
                && let Kind::Array(elem_ty, len_ty) = *self.types.kind(fields[0][0])
            {
                let elt_layout = self.layout(elem_ty)?;
                let Some(count) = self.types.len(len_ty) else { bail!("simd array {} has unknown length", self.name(item)) };
                return calc.layout_of_simd_type(elt_layout, count, repr.packed()).map_err(|error| anyhow!("layout of {}: {error:?}", self.name(item)));
            }
            let k = fields[0].len();
            let elt_layout = &layouts[0][0];
            let mut out = calc.layout_of_simd_type(elt_layout, k as u64, repr.packed()).map_err(|error| anyhow!("layout of {}: {error:?}", self.name(item)))?;
            out.fields = FieldsShape::Arbitrary {
                offsets: (0..k).map(|i| elt_layout.size * (i as u64)).collect(),
                in_memory_order: (0..k).map(Field::new).collect(),
            };
            return Ok(out);
        }
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

/// Adds the `#[repr]` words that start at the tokens `words` to `repr`.
fn read_repr_words(src: &Source, words: Vec<u32>, repr: &mut ReprOptions) {
    for from in words {
        match src.text(from) {
            "C" => repr.flags |= ReprFlags::IS_C,
            "simd" => repr.flags |= ReprFlags::IS_SIMD,
            "transparent" => repr.flags |= ReprFlags::IS_TRANSPARENT,
            "packed" => repr.pack = Some(repr_align(src, from).unwrap_or(Align::ONE)),
            "align" => repr.align = repr_align(src, from).or(repr.align),
            word => repr.int = REPR_INTS.iter().find(|(name, _)| *name == word).map(|(_, int)| integer_type(*int)).or(repr.int),
        }
    }
}

fn visit_active_attrs<T>(env: &Env, attrs: &[Attr], f: &mut impl FnMut(u32) -> ControlFlow<T>) -> ControlFlow<T> {
    for attr in attrs {
        match attr {
            Attr::CfgAttr(cfg, inner) if env.eval(cfg) == Truth::Yes => visit_active_attrs(env, inner, f)?,
            &Attr::Other(at) => f(at)?,
            _ => {}
        }
    }
    ControlFlow::Continue(())
}

fn collect_active_attrs(env: &Env, attrs: &[Attr], out: &mut Vec<u32>) {
    let _ = visit_active_attrs(env, attrs, &mut |at| {
        out.push(at);
        ControlFlow::<()>::Continue(())
    });
}

fn active_attr_at(env: &Env, src: &Source, attrs: &[Attr], key: &str) -> Option<u32> {
    match visit_active_attrs(env, attrs, &mut |at| if src.text(at) == key { ControlFlow::Break(at) } else { ControlFlow::Continue(()) }) {
        ControlFlow::Break(at) => Some(at),
        ControlFlow::Continue(()) => None,
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
