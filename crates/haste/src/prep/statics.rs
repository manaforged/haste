use std::sync::Mutex;

use rustc_hash::FxHashMap as HashMap;

use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty};

use super::form;
use super::indirect::{Indirect, Mode};
use super::key::Keys;
use super::record::{Record, Static};
use super::source::{nested, parent_of};

/// The item kinds whose definitions a key that names a static, or a type a function pointer passes, may name.
const DEFINED: [ItemKind; 5] = [ItemKind::Fn, ItemKind::Static, ItemKind::Struct, ItemKind::Enum, ItemKind::Union];

/// The statics holding state that a library defines once, by key and symbol, with Haste's definitions by key once worked out for a crate, and the address each of Haste's statics resolves to.
pub(super) struct Statics {
    facts: Vec<Static>,
    by_key: HashMap<String, usize>,
    by_symbol: HashMap<String, usize>,
    defs: Mutex<HashMap<u32, HashMap<String, ItemRef>>>,
}

/// What Haste's code needs to touch the library's statics: per static of the manifest the address Haste's code uses for it, when it may share the library's copy.
pub(super) struct Shared {
    pub address: Vec<Option<usize>>,
}

impl Shared {
    /// Whether the library's static `symbol` is one Haste's code shares, so that rustc's code touching it may run.
    pub(super) fn holds(&self, statics: &Statics, symbol: &str) -> bool {
        statics.by_symbol.get(symbol).is_some_and(|&at| self.address[at].is_some())
    }
}

impl Statics {
    /// The statics of a library's manifest.
    pub(super) fn new(facts: Vec<Static>) -> Statics {
        let by_key = facts.iter().enumerate().map(|(at, fact)| (fact.key.clone(), at)).collect();
        let by_symbol = facts.iter().enumerate().map(|(at, fact)| (fact.symbol.clone(), at)).collect();
        Statics { facts, by_key, by_symbol, defs: Mutex::default() }
    }

    /// Which statics Haste's code shares with the library, decided once: each one a single Haste static names, whose layouts match, whose values carry no code pointer that cannot cross, which no record touching it calls through a function pointer type whose calls differ, not per thread, and whose symbol the open library exports.
    pub(super) fn share(&self, cx: &mut Context<'_>, (keys, indirect): (&Keys, &Indirect), records: &[Record]) -> Shared {
        let mut calls: Vec<Vec<Option<&[String]>>> = vec![Vec::new(); self.facts.len()];
        for record in records {
            for at in record.statics.iter().filter_map(|symbol| self.by_symbol.get(symbol)) {
                calls[*at].push(record.calls.as_deref());
            }
        }
        let address = self.facts.iter().zip(&calls).map(|(fact, calls)| self.decide(cx, (keys, indirect), fact, calls)).collect();
        Shared { address }
    }

    fn decide(&self, cx: &mut Context<'_>, (keys, indirect): (&Keys, &Indirect), fact: &Static, calls: &[Option<&[String]>]) -> Option<usize> {
        self.same(cx, (keys, indirect), fact)?;
        for called in calls {
            for key in (*called)? {
                let pointer = self.parse(cx, keys, key)?;
                matches!(indirect.mode(cx, keys, pointer), Mode::Free).then_some(())?;
            }
        }
        super::exported(&fact.symbol)
    }

    /// Whether `fact` names one static of Haste's, shared by every thread on both sides, whose layouts match and whose code pointers may cross.
    fn same(&self, cx: &mut Context<'_>, (keys, indirect): (&Keys, &Indirect), fact: &Static) -> Option<()> {
        (!fact.thread_local).then_some(())?;
        let item = self.resolve(cx, keys, &fact.key)?;
        (cx.item(item).1.kind == ItemKind::Static && !cx.foreign(item) && !cx.thread_local(item)).then_some(())?;
        (self.key(cx, keys, item).as_deref() == Some(fact.key.as_str())).then_some(())?;
        let ty = crate::sema::infer::infer(cx, item, Substs::EMPTY).ok()?.ret?;
        let (printed, found) = super::conv::layouts(cx, &[ty]).ok()?;
        (fact.layouts.as_ref() == Some(&printed) && indirect.cross(cx, keys, &found)).then_some(())
    }

    /// The address Haste's code uses for static `item` when it shares the library's copy.
    pub(super) fn address(&self, cx: &mut Context<'_>, keys: &Keys, shared: &Shared, item: ItemRef) -> Option<usize> {
        let at = *self.by_key.get(&self.key(cx, keys, item)?)?;
        shared.address[at]
    }

    /// The key of Haste's static `item` as the driver keys one: its definition's, or for a static declared in a function's body, the function's key and the static's name when no other static of the body has that name.
    fn key(&self, cx: &mut Context<'_>, keys: &Keys, item: ItemRef) -> Option<String> {
        let Some((owner, _)) = item.outer() else { return keys.def(cx, item).ok() };
        let name = cx.name(item);
        self.declared(cx, owner, name)?;
        Some(nested(&keys.def(cx, owner).ok()?, name))
    }

    /// The one static named `name` that the body of `owner` declares, at any depth.
    fn declared(&self, cx: &mut Context<'_>, owner: ItemRef, name: &str) -> Option<ItemRef> {
        cx.body(owner).ok()?;
        let found = cx.expansion(owner)?;
        let items = &found.file.items.items;
        let mut named = items.iter().enumerate().filter(|(_, entry)| entry.kind == ItemKind::Static && entry.name.is_some_and(|at| found.file.source.text(at) == name));
        let (at, _) = named.next()?;
        named.next().is_none().then(|| owner.inner(at as u32))
    }

    /// Haste's definition that `key` names: a function, static or type a key of its crate names, or a static that a function's body declares.
    fn resolve(&self, cx: &mut Context<'_>, keys: &Keys, key: &str) -> Option<ItemRef> {
        if let Some((parent, name)) = parent_of(key) {
            let owner = self.resolve(cx, keys, parent)?;
            return self.declared(cx, owner, name);
        }
        let krate = key.split(':').next()?;
        let mut found = None;
        for at in keys.crates_named(krate) {
            let mut defs = self.defs.lock().ok()?;
            let defs = defs.entry(at as u32).or_insert_with(|| crate_defs(cx, keys, at));
            found = found.or_else(|| defs.get(key).copied());
        }
        found
    }

    /// The type that type key `key` names, when it holds only primitives, pointers, references, tuples, arrays, slices, function pointers and types whose definitions Haste finds.
    fn parse(&self, cx: &mut Context<'_>, keys: &Keys, key: &str) -> Option<Ty> {
        let (ty, rest) = self.ty(cx, keys, key)?;
        rest.is_empty().then_some(ty)
    }

    fn ty<'k>(&self, cx: &mut Context<'_>, keys: &Keys, text: &'k str) -> Option<(Ty, &'k str)> {
        let pointers: [(&str, fn(Ty) -> Kind); 4] = [(form::MUT_REF, |ty| Kind::Ref(Mut::Mut, ty)), (form::SHARED_REF, |ty| Kind::Ref(Mut::Not, ty)), (form::CONST_PTR, |ty| Kind::Ptr(Mut::Not, ty)), (form::MUT_PTR, |ty| Kind::Ptr(Mut::Mut, ty))];
        for (prefix, make) in pointers {
            if let Some(rest) = text.strip_prefix(prefix) {
                let (inner, rest) = self.ty(cx, keys, rest)?;
                return Some((cx.types.intern(make(inner)), rest));
            }
        }
        if let Some(rest) = text.strip_prefix(form::FN_PTR) {
            let (params, rest) = self.items(cx, keys, rest, form::RETURNS)?;
            let (ret, rest) = self.ty(cx, keys, rest)?;
            return Some((cx.types.fn_ptr(&params, ret), rest));
        }
        if let Some(rest) = text.strip_prefix('(') {
            let (items, rest) = self.items(cx, keys, rest, ")")?;
            return Some((cx.types.tuple(&items), rest));
        }
        if let Some(rest) = text.strip_prefix('[') {
            return self.sequence(cx, keys, rest);
        }
        if let Some(rest) = text.strip_prefix(form::NEVER) {
            return Some((cx.types.intern(Kind::Never), rest));
        }
        self.named(cx, keys, text)
    }

    /// Types each followed by a comma up to `end`, and the text after `end`.
    fn items<'k>(&self, cx: &mut Context<'_>, keys: &Keys, mut text: &'k str, end: &str) -> Option<(Vec<Ty>, &'k str)> {
        let mut items = Vec::new();
        while !text.starts_with(end) {
            let (item, rest) = self.ty(cx, keys, text)?;
            items.push(item);
            text = rest.strip_prefix(',')?;
        }
        Some((items, &text[end.len()..]))
    }

    /// An array `T;N]` or a slice `T]` after its opening bracket.
    fn sequence<'k>(&self, cx: &mut Context<'_>, keys: &Keys, text: &'k str) -> Option<(Ty, &'k str)> {
        let (element, rest) = self.ty(cx, keys, text)?;
        if let Some(rest) = rest.strip_prefix(']') {
            return Some((cx.types.intern(Kind::Slice(element)), rest));
        }
        let (len, rest) = rest.strip_prefix(';')?.split_once(']')?;
        Some((cx.types.array(element, len.parse().ok()?), rest))
    }

    /// A primitive, or a type definition with its type arguments.
    fn named<'k>(&self, cx: &mut Context<'_>, keys: &Keys, text: &'k str) -> Option<(Ty, &'k str)> {
        let end = text.find(['<', ',', ')', ']', ';', '>']).unwrap_or(text.len());
        let (name, rest) = text.split_at(end);
        if let Some(primitive) = Primitive::named(name) {
            return Some((cx.types.prim(primitive), rest));
        }
        let item = self.resolve(cx, keys, name)?;
        matches!(cx.item(item).1.kind, ItemKind::Struct | ItemKind::Enum | ItemKind::Union).then_some(())?;
        let mut rest = rest.strip_prefix('<')?;
        let mut args = Vec::new();
        while !rest.starts_with('>') {
            let (arg, after) = self.ty(cx, keys, rest)?;
            args.push(arg);
            rest = after.strip_prefix(',').unwrap_or(after);
        }
        Some((cx.types.adt(item, &args), &rest[1..]))
    }
}

/// Each function, static and type definition of crate `krate` that a key names, by its key.
fn crate_defs(cx: &Context<'_>, keys: &Keys, krate: usize) -> HashMap<String, ItemRef> {
    let mut defs = HashMap::default();
    for (file, tree) in cx.outline.crates[krate].files.iter().enumerate() {
        for (at, entry) in tree.items.items.iter().enumerate() {
            if !DEFINED.contains(&entry.kind) {
                continue;
            }
            let item = ItemRef { krate: krate as u32, file: file as u32, item: at as u32 };
            if let Ok(key) = keys.def(cx, item) {
                defs.insert(key, item);
            }
        }
    }
    defs
}
