//! Semantic analysis of reached function bodies at concrete types, lowered to `haste-codegen` SSA.

pub mod adt;
mod body_scope;
mod builtin;
mod expand;
mod format;
pub mod impls;
pub mod infer;
pub mod intrinsic;
pub mod layout;
pub mod reach;
pub mod traits;
pub mod ty;

use rustc_hash::FxHashMap as HashMap;
use std::rc::Rc;

use anyhow::{Context as _, Result, anyhow, bail};

use crate::body::sig::{self, Signature};
use crate::body::adt::{AdtDef, adt as parse_adt};
use crate::body::Body;
use crate::names::collect::items_of;
use crate::names::path::{Asks, Paths};
use crate::names::{ItemRef, ModRef, Names};
use crate::outline::Outline;
use crate::outline::item::{Item, ItemKind};
use crate::outline::source::Source;

use ty::Types;

/// What Sema keeps between passes over a growing outline: interned types and parsed headers and bodies.
#[derive(Default)]
pub struct Store {
    types: Types,
    signatures: HashMap<ItemRef, Rc<Signature>>,
    bodies: HashMap<ItemRef, Rc<Body>>,
    adts: HashMap<ItemRef, Rc<AdtDef>>,
    layouts: HashMap<ty::Ty, Rc<layout::Layout>>,
    tails: HashMap<ty::Ty, ty::Ty>,
    headers: HashMap<impls::ImplRef, Rc<sig::ImplHeader>>,
    finals: HashMap<reach::Instance, Rc<infer::Typed>>,
    closures: HashMap<ty::Ty, Rc<infer::ClosureInfo>>,
    lookups: Lookups,
}

/// The lookups Sema caches about one outline, which hold while only function bodies change.
#[derive(Default)]
struct Lookups {
    modules: HashMap<u32, Rc<HashMap<(u32, u32), u32>>>,
    heads: HashMap<impls::Holder, Rc<impls::Index>>,
    owners: HashMap<u32, Rc<HashMap<(u32, Option<u32>), u32>>>,
    trait_defs: HashMap<ItemRef, Rc<traits::TraitGenerics>>,
    discriminants: HashMap<ItemRef, Rc<[i128]>>,
    langs: Option<(Rc<HashMap<String, ItemRef>>, Rc<[u32]>)>,
    drops: HashMap<ty::Ty, bool>,
    reveals: HashMap<reach::Instance, Option<ty::Ty>>,
    futures: HashMap<ty::Ty, ty::Ty>,
    lang_of: std::cell::RefCell<HashMap<ItemRef, Option<u32>>>,
    scope_traits: std::cell::RefCell<HashMap<ModRef, Rc<[ItemRef]>>>,
    type_defs: std::cell::RefCell<HashMap<TypeKey, crate::names::Def>>,
    impl_traits: HashMap<impls::ImplRef, Option<ItemRef>>,
}

impl Store {
    /// Drops the cached lookups, which no longer hold once the outline loads more files or resolves more names.
    pub fn forget_lookups(&mut self) {
        self.lookups = Lookups::default();
    }

    /// Drops what the store keeps about the items that `edited` holds, and about the instances, types and lookups that name one of them.
    pub fn forget(&mut self, outline: &Outline, edited: Edited<'_>) {
        let mine = |item: ItemRef| edited.holds(item.krate, item.file);
        let block_kept = |block: &impls::ImplRef| match *block {
            impls::ImplRef::Module { krate, index } => outline.crates[krate as usize].impls.get(index as usize).is_some_and(|entry| !edited.holds(krate, entry.file as u32) && !entry.parts.iter().any(|&part| edited.holds(krate, part as u32))),
            impls::ImplRef::Body(item) => !mine(item),
        };
        self.signatures.retain(|item, _| !mine(*item));
        self.bodies.retain(|item, _| !mine(*item));
        self.adts.retain(|item, _| !mine(*item));
        self.headers.retain(|block, _| block_kept(block));
        let mut naming = Naming { types: &self.types, mine: &mine, memo: HashMap::default() };
        self.finals.retain(|instance, _| !naming.touches(instance));
        self.layouts.retain(|ty, _| !naming.names(*ty));
        self.tails.retain(|ty, _| !naming.names(*ty));
        self.closures.retain(|ty, _| !naming.names(*ty));
        let kept = &mut self.lookups;
        kept.drops.retain(|ty, _| !naming.names(*ty));
        kept.reveals.retain(|instance, _| !naming.touches(instance));
        kept.futures.retain(|ty, _| !naming.names(*ty));
        kept.impl_traits.retain(|block, _| block_kept(block));
        kept.heads.retain(|holder, _| match *holder {
            impls::Holder::Crate(krate) => !edited.touches_crate(krate),
            impls::Holder::Body(item) => !mine(item),
        });
        kept.modules.retain(|krate, _| !edited.touches_crate(*krate));
        kept.owners.retain(|krate, _| !edited.touches_crate(*krate));
        kept.scope_traits.get_mut().retain(|home, _| !edited.touches_crate(home.krate));
        kept.type_defs.get_mut().retain(|key, _| !edited.touches_crate(key.0.krate));
        kept.trait_defs.retain(|item, _| !mine(*item));
        kept.discriminants.retain(|item, _| !mine(*item));
        kept.lang_of.get_mut().retain(|item, _| !mine(*item));
    }

    /// The indexes of the instances in `list` that are functions `edited` holds or take a type that names one of its items.
    pub fn touched<'l>(&self, list: impl Iterator<Item = &'l reach::Instance>, edited: Edited<'_>) -> Vec<usize> {
        let mine = |item: ItemRef| edited.holds(item.krate, item.file);
        let mut naming = Naming { types: &self.types, mine: &mine, memo: HashMap::default() };
        list.enumerate().filter(|(_, instance)| naming.touches(instance)).map(|(at, _)| at).collect()
    }
}

/// What an edit changed: function bodies in some `(crate, file)` slots, or whole crates read again.
#[derive(Clone, Copy, Debug)]
pub enum Edited<'e> {
    /// Function bodies in these `(crate, file)` slots.
    Files(&'e [(u32, u32)]),
    /// Every file of these crates.
    Crates(&'e [u32]),
}

impl Edited<'_> {
    fn holds(self, krate: u32, file: u32) -> bool {
        match self {
            Edited::Files(files) => files.contains(&(krate, file)),
            Edited::Crates(crates) => crates.contains(&krate),
        }
    }

    fn touches_crate(self, krate: u32) -> bool {
        match self {
            Edited::Files(files) => files.iter().any(|&(owner, _)| owner == krate),
            Edited::Crates(crates) => crates.contains(&krate),
        }
    }
}

/// Which types name an item for which `mine` holds, remembered per type.
struct Naming<'t, F> {
    types: &'t Types,
    mine: &'t F,
    memo: HashMap<ty::Ty, bool>,
}

impl<F: Fn(ItemRef) -> bool> Naming<'_, F> {
    fn touches(&mut self, (item, substs): &reach::Instance) -> bool {
        (self.mine)(*item) || substs.iter().any(|ty| self.names(*ty))
    }

    /// Whether `ty` names, at any depth, an item or a closure of an item for which `mine` holds.
    fn names(&mut self, ty: ty::Ty) -> bool {
        if let Some(&known) = self.memo.get(&ty) {
            return known;
        }
        let kind = self.types.kind(ty);
        let own = match kind {
            ty::Kind::Adt(item, _) | ty::Kind::FnDef(item, _) | ty::Kind::Closure(item, ..) | ty::Kind::Dyn(item, ..) => (self.mine)(*item),
            _ => false,
        };
        let closure_args = match kind {
            ty::Kind::Closure(_, args, _) => args.clone(),
            _ => Vec::new(),
        };
        let inner: Vec<ty::Ty> = kind.children().chain(closure_args).collect();
        let found = own || inner.into_iter().any(|inner| self.names(inner));
        self.memo.insert(ty, found);
        found
    }
}

/// Everything Sema reads in one pass, with its caches and the lookups that found nothing loaded yet.
pub struct Context<'a> {
    pub outline: &'a Outline,
    pub names: &'a Names,
    pub types: Types,
    pub asks: Asks,
    store: Store,
    kids: std::cell::RefCell<HashMap<usize, Rc<Kids>>>,
    trait_members: std::cell::RefCell<HashMap<ItemRef, Rc<HashMap<&'a str, ItemRef>>>>,
    impl_members: std::cell::RefCell<HashMap<impls::ImplRef, Rc<HashMap<&'a str, ItemRef>>>>,
}

/// Where a type path is resolved and what it says: home module, body root, nested instance, segments, leading `::` and `$crate`.
pub type TypeKey = (ModRef, Option<ItemRef>, Option<ItemRef>, Vec<crate::names::Sym>, bool, Option<u32>);

/// The items of one file or body grouped by the item they are declared inside, in order.
type Kids = HashMap<Option<u32>, Rc<[u32]>>;

impl<'a> Context<'a> {
    /// A pass over an outline and its resolved names that reuses `store`.
    pub fn new(outline: &'a Outline, names: &'a Names, mut store: Store) -> Context<'a> {
        let types = std::mem::take(&mut store.types);
        let (kids, trait_members, impl_members, asks) = Default::default();
        Context { outline, names, types, asks, store, kids, trait_members, impl_members }
    }

    /// Whether no lookup in this pass asked the front end for anything, so another pass would find the same.
    pub fn settled(&self) -> bool {
        self.asks.count() == 0
    }

    /// Ends the pass: the store for the next one, and what lookups asked the front end to load or resolve.
    pub fn release(mut self) -> (Store, Asks) {
        self.store.types = self.types;
        (self.store, self.asks)
    }

    /// The type-namespace items that paths name, by home module, body root, nested instance, segments, `::` and `$crate`.
    pub fn type_defs(&self) -> &std::cell::RefCell<HashMap<TypeKey, crate::names::Def>> {
        &self.store.lookups.type_defs
    }

    /// The source file and outline entry of `item`.
    pub fn item(&self, item: ItemRef) -> (&'a Source, &'a Item) {
        let file = self.file(item);
        let index = item.outer().map_or(item.item, |(_, index)| index);
        (&file.source, &file.items.items[index as usize])
    }

    /// The item that `item` is declared directly inside, in the same file or body.
    pub fn parent_item(&self, item: ItemRef) -> Option<ItemRef> {
        let parent = self.item(item).1.parent?;
        Some(match item.declaring_body() {
            Some(root) => root.inner(parent),
            None => ItemRef { item: parent, ..item },
        })
    }

    /// The items declared directly inside `of`: the members of a trait or impl, in the file or body that holds it.
    pub fn children(&self, of: ItemRef) -> impl Iterator<Item = ItemRef> + 'a {
        let at = of.outer().map_or(of.item, |(_, at)| at);
        let list = self.kids_of(&self.file(of).items.items, Some(at));
        let body = of.declaring_body();
        (0..list.len()).map(move |index| match body {
            Some(root) => root.inner(list[index]),
            None => ItemRef { item: list[index], ..of },
        })
    }

    /// The indices of the items in `items` declared directly inside item `parent`, or at the top level for none.
    pub fn kids_of(&self, items: &[Item], parent: Option<u32>) -> Rc<[u32]> {
        let key = items.as_ptr() as usize;
        let cached = self.kids.borrow().get(&key).cloned();
        let kids = cached.unwrap_or_else(|| {
            let mut grouped: HashMap<Option<u32>, Vec<u32>> = HashMap::default();
            for (at, item) in items.iter().enumerate() {
                grouped.entry(item.parent).or_default().push(at as u32);
            }
            let kids: Rc<Kids> = Rc::new(grouped.into_iter().map(|(parent, list)| (parent, list.into())).collect());
            self.kids.borrow_mut().insert(key, kids.clone());
            kids
        });
        kids.get(&parent).cloned().unwrap_or_else(|| Rc::from([]))
    }

    /// Whether `item` is a function or static declared in an `extern` block, which the host links by name.
    pub fn foreign(&self, item: ItemRef) -> bool {
        let (_, entry) = self.item(item);
        matches!(entry.kind, ItemKind::Fn | ItemKind::Static) && entry.body.is_none() && self.parent_item(item).is_some_and(|parent| self.item(parent).1.kind == ItemKind::ExternBlock)
    }

    /// What the body that defines the closure type `ty` inferred for it.
    pub fn closure(&self, ty: ty::Ty) -> Option<Rc<infer::ClosureInfo>> {
        self.store.closures.get(&ty).cloned()
    }

    /// The output of the async fn future `ty` that a call named while its body was still being inferred, from the declared return type.
    pub fn future_output(&self, ty: ty::Ty) -> Option<ty::Ty> {
        self.store.lookups.futures.get(&ty).copied()
    }

    /// Records what the body that defines the closure type `ty` inferred for it.
    pub fn set_closure(&mut self, ty: ty::Ty, info: infer::ClosureInfo) {
        self.store.closures.insert(ty, Rc::new(info));
    }

    /// The tuple of the captures of the closure type `ty`, which is how the closure is laid out and dropped.
    pub fn closure_env(&self, ty: ty::Ty) -> Option<ty::Ty> {
        self.store.closures.get(&ty).map(|info| info.env)
    }

    /// Whether the code of the closure type `ty` takes the address of its environment first, which only a closure that captures something does.
    pub fn closure_takes_env(&self, ty: ty::Ty) -> bool {
        self.store.closures.get(&ty).is_some_and(|info| !info.closure.captures.is_empty())
    }

    /// The type that `ty` owns when it is a `Box`.
    pub fn boxed(&self, ty: ty::Ty) -> Option<ty::Ty> {
        match self.types.kind(ty) {
            ty::Kind::Adt(item, args) if self.lang(*item) == Some("owned_box") => args.first().copied(),
            _ => None,
        }
    }

    /// Whether a call of the closure type `ty` consumes the closure, because its body moves a captured value out.
    pub fn closure_consumes(&self, ty: ty::Ty) -> bool {
        self.store.closures.get(&ty).is_some_and(|info| info.closure.consumes)
    }

    /// A short spelling of `ty`, each type in it first passed through `resolve`.
    pub fn type_text(&self, ty: ty::Ty, resolve: &dyn Fn(ty::Ty) -> ty::Ty) -> String {
        use ty::{Kind, Mut};
        let list = |tys: &[ty::Ty]| tys.iter().map(|&ty| self.type_text(ty, resolve)).collect::<Vec<_>>().join(", ");
        match self.types.kind(resolve(ty)).clone() {
            Kind::Adt(item, args) if args.is_empty() => self.name(item).to_owned(),
            Kind::Adt(item, args) => format!("{}<{}>", self.name(item), list(&args)),
            Kind::Ref(Mut::Not, inner) => format!("&{}", self.type_text(inner, resolve)),
            Kind::Ref(Mut::Mut, inner) => format!("&mut {}", self.type_text(inner, resolve)),
            Kind::Tuple(items) => format!("({})", list(&items)),
            Kind::Slice(element) => format!("[{}]", self.type_text(element, resolve)),
            Kind::Prim(primitive) => format!("{primitive:?}").to_lowercase(),
            other => format!("{other:?}"),
        }
    }

    /// `FnOnce::call_once`, which every call of a closure value is an instance of.
    pub fn call_once(&mut self) -> Option<ItemRef> {
        let of = self.lang_item("fn_once")?;
        self.trait_member(of, "call_once")
    }

    /// Whether `item` is `FnOnce::call_once` or the `call_once` of an `FnOnce` impl, which take the callee by value.
    pub fn takes_by_value(&mut self, item: ItemRef) -> bool {
        let once = self.lang_item("fn_once");
        self.call_once() == Some(item) || self.owner_impl(item).and_then(|block| self.impl_trait(block)).is_some_and(|of| Some(of) == once)
    }

    /// Whether `instance` names, in its function or its type arguments, an item that a body declared before an edit and declares no longer.
    pub fn undeclared(&self, instance: &reach::Instance) -> bool {
        let gone = |item: ItemRef| item.outer().is_some_and(|(outer, index)| self.expansion(outer).is_none_or(|found| index as usize >= found.file.items.items.len()));
        Naming { types: &self.types, mine: &gone, memo: HashMap::default() }.touches(instance)
    }

    /// The outline file that holds `item`, or the expansion of the body that declares it.
    pub fn file(&self, item: ItemRef) -> &'a crate::outline::tree::File {
        match item.outer() {
            Some((outer, _)) => &self.expansion(outer).expect("a body that declares items is read").file,
            None => &self.outline.crates[item.krate as usize].files[item.file as usize],
        }
    }

    /// Path resolution inside crate `krate`, recording what it finds not loaded yet.
    pub fn paths(&self, krate: u32) -> Paths<'_> {
        Paths { demands: Some(&self.asks), ..self.names.paths(self.outline, krate) }
    }

    /// The module that holds `item`, through any impl, trait or extern block around it.
    pub fn module_of(&mut self, item: ItemRef) -> Result<ModRef> {
        if let Some((outer, _)) = item.outer() {
            return self.module_of(outer);
        }
        if let Some(block) = self.owner_impl(item) {
            return self.impl_module(block);
        }
        let map = self.modules_of(item.krate);
        let (_, mut entry) = self.item(item);
        let mut top = item.item;
        let file = &self.outline.crates[item.krate as usize].files[item.file as usize];
        while let Some(parent) = entry.parent.filter(|parent| file.items.items[*parent as usize].kind != ItemKind::Mod) {
            top = parent;
            entry = &file.items.items[parent as usize];
        }
        let module = map.get(&(item.file, top)).copied().with_context(|| format!("no module holds item {item:?}"))?;
        Ok(ModRef { krate: item.krate, module })
    }

    fn modules_of(&mut self, krate: u32) -> Rc<HashMap<(u32, u32), u32>> {
        let tree = &self.outline.crates[krate as usize];
        self.store.lookups.modules
            .entry(krate)
            .or_insert_with(|| {
                let pairs = (0..tree.modules.len()).flat_map(|module| items_of(tree, module).into_iter().map(move |pair| (pair, module as u32)));
                Rc::new(pairs.collect())
            })
            .clone()
    }

    /// The parsed header of the function `item`.
    pub fn signature(&mut self, item: ItemRef) -> Result<Rc<Signature>> {
        if let Some(found) = self.store.signatures.get(&item) {
            return Ok(found.clone());
        }
        let (src, entry) = self.item(item);
        let parsed = match (entry.kind, entry.name) {
            (ItemKind::Const | ItemKind::Static, Some(name)) => sig::const_signature(src, name, entry.end)?,
            _ => sig::signature(src, entry.start, entry.body.unwrap_or(entry.end), &self.outline.crates[item.krate as usize].env)?,
        };
        let parsed = Rc::new(parsed);
        self.store.signatures.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The parsed body of the function `item`.
    pub fn body(&mut self, item: ItemRef) -> Result<Rc<Body>> {
        if let Some(found) = self.store.bodies.get(&item) {
            return Ok(found.clone());
        }
        let mut parsed = self.parse_body(item)?;
        let sig = self.signature(item)?;
        if sig.params.iter().any(|&(pat, _)| !sig.body.plain_param(pat)) {
            parsed = crate::body::parse_params(self.body_src(item), parsed, &sig.param_starts)?;
        }
        let parsed = Rc::new(parsed);
        self.store.bodies.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The position of the variant with declaration index `index` among the configured variants of the enum `item`.
    pub fn variant_position(&mut self, item: ItemRef, index: u32) -> Result<usize> {
        match self.adt_def(item)?.position(index) {
            Some(at) => Ok(at),
            None => bail!("variant {index} of {} is configured out", self.name(item)),
        }
    }

    /// The parsed definition of the struct, enum or union `item`.
    pub fn adt_def(&mut self, item: ItemRef) -> Result<Rc<AdtDef>> {
        if let Some(found) = self.store.adts.get(&item) {
            return Ok(found.clone());
        }
        let (src, entry) = self.item(item);
        let name = entry.name.ok_or_else(|| anyhow!("type item has no name"))?;
        let mut parsed = parse_adt(src, name, entry.end, entry.kind == ItemKind::Enum)?;
        let env = &self.outline.crates[item.krate as usize].env;
        parsed.configure(src, &|attrs| env.enabled(attrs));
        let parsed = Rc::new(parsed);
        self.store.adts.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The type parameters, arena and target type of the type alias `item`.
    pub fn alias_parts(&self, item: ItemRef) -> Result<(Vec<sig::Generic>, Body, crate::body::TyId)> {
        let (src, entry) = self.item(item);
        let (_, eq) = self.alias_eq(item)?;
        let params = self.alias_generics(item)?.1.params.into_iter().filter(|generic| generic.kind != sig::GenericKind::Lifetime).collect();
        let (arena, ty) = crate::body::parse_ty(src, eq + 1, entry.end - 1)?;
        Ok((params, arena, ty))
    }

    /// The generic parameters and where clause of the type alias `item`, with the bounds that its target may rely on.
    pub fn alias_generics(&self, item: ItemRef) -> Result<traits::TraitGenerics> {
        let (src, _) = self.item(item);
        let (at, eq) = self.alias_eq(item)?;
        match src.text(at + 1) {
            "<" => sig::generics(src, at + 1, eq),
            _ => Ok(Default::default()),
        }
    }

    fn alias_eq(&self, item: ItemRef) -> Result<(u32, u32)> {
        let (src, entry) = self.item(item);
        let Some(at) = entry.name else { bail!("type alias with no name") };
        let mut depth = 0i32;
        let mut found = None;
        for token in at + 1..entry.end {
            match src.kind(token) {
                rustc_lexer::TokenKind::Lt => depth += 1,
                rustc_lexer::TokenKind::Gt if src.kind(token - 1) != rustc_lexer::TokenKind::Minus => depth -= 1,
                rustc_lexer::TokenKind::Eq if depth == 0 => {
                    found = Some(token);
                    break;
                }
                _ => {}
            }
        }
        let Some(eq) = found else { bail!("type `{}` has no `=`", src.text(at)) };
        Ok((at, eq))
    }

    /// The name of `item`, for messages.
    pub fn name(&self, item: ItemRef) -> &'a str {
        let (src, entry) = self.item(item);
        entry.name.map_or("<anonymous>", |name| src.text(name))
    }
}
