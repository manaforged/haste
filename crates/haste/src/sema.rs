//! Semantic analysis of reached function bodies at concrete types, lowered to `haste-codegen` SSA.

pub mod adt;
mod body_scope;
mod builtin;
pub mod census;
pub mod evaluate;
mod expand;
pub(crate) mod format;
pub mod impls;
pub mod infer;
pub mod intrinsic;
pub mod layout;
mod lookups;
pub mod reach;
pub mod substs;
pub mod summary;
pub mod table;
pub mod traits;
pub mod ty;

pub use lookups::{Answer, Cache, Edited};
use lookups::{Naming, SegKey, TypeScope};

use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::body::Body;
use crate::body::adt::{AdtDef, adt as parse_adt};
use crate::body::sig::{self, Signature};
use crate::names::collect::items_of;
use crate::names::path::{Paths, Requests};
use crate::names::{ItemRef, ModRef, Names};
use crate::outline::Outline;
use crate::outline::item::{Item, ItemKind};
use crate::outline::source::Source;

use table::Table;
use ty::Types;

pub(super) use crate::sync::{read_lock, write_lock};

/// Everything Sema reads in one pass, with its caches and the lookups that found nothing loaded yet.
pub struct Context<'a> {
    pub outline: &'a Outline,
    pub names: &'a Names,
    pub types: Types,
    pub asks: Requests,
    pub(crate) in_parallel: bool,
    pub(crate) deferred_evals: Vec<(reach::Instance, u32)>,
    store: Cache,
    kids: Table<usize, Arc<Kids>>,
    trait_members: Table<ItemRef, Arc<HashMap<&'a str, ItemRef>>>,
    impl_members: Table<impls::ImplRef, Arc<HashMap<&'a str, ItemRef>>>,
    aliases: Table<ItemRef, Arc<Alias>>,
    pub(super) eval_errors: Table<reach::Instance, String>,
    pub(super) eval_stack: Vec<reach::Instance>,
    pub(super) drop_stack: Vec<ty::Ty>,
    pub(super) form_stack: Vec<(infer::Form, usize)>,
}

/// A parsed type alias: its type parameters, its generics and where clause, and its target type.
#[derive(Debug)]
pub struct Alias {
    pub params: Vec<sig::Generic>,
    pub generics: Arc<traits::TraitGenerics>,
    pub arena: Body,
    pub ty: crate::body::TyId,
}

/// The items of one file or body grouped by the item they are declared inside, in order.
type Kids = HashMap<Option<u32>, Arc<[u32]>>;

impl<'a> Context<'a> {
    /// Returns the shared inference cache attached to this context.
    pub(crate) fn store(&self) -> &Cache {
        &self.store
    }

    /// Drops cached reach-only tables and sysroot bodies to cut RSS before one-shot codegen.
    pub(crate) fn compact_for_oneshot_codegen(&mut self) {
        self.store.finals.clear();
        self.store.bodies.retain(|item, _| !self.outline.is_sysroot(item.krate));
        self.store.lookups.summaries = table::Flight::default();
        self.store.lookups.selections.clear();
        self.store.lookups.projections.clear();
        self.store.lookups.trait_calls.clear();
        self.store.lookups.assoc_types.clear();
        self.store.lookups.type_defs.clear();
        self.store.lookups.scope_traits.clear();
        self.store.lookups.reveals = table::Flight::default();
        self.store.lookups.futures.clear();
        self.store.lookups.forms.clear();
        self.store.lookups.sysroot_index = Arc::new(std::sync::OnceLock::new());
        self.kids.clear();
        self.trait_members.clear();
        self.impl_members.clear();
        self.aliases.clear();
    }

    /// A pass over an outline and its resolved names that reuses `store`.
    pub fn new(outline: &'a Outline, names: &'a Names, mut store: Cache) -> Context<'a> {
        let types = store.types.clone();
        store.lookups.check(outline);
        let (kids, trait_members, impl_members, aliases, asks) = Default::default();
        Context {
            outline,
            names,
            types,
            asks,
            in_parallel: false,
            deferred_evals: Vec::new(),
            store,
            kids,
            trait_members,
            impl_members,
            aliases,
            eval_errors: Table::default(),
            eval_stack: Vec::new(),
            drop_stack: Vec::new(),
            form_stack: Vec::new(),
        }
    }

    /// A sibling context over the same outline, names, types and caches, with its own empty request list so a worker thread can run without contending on asks.
    pub fn fork(&self) -> Context<'a> {
        Context {
            outline: self.outline,
            names: self.names,
            types: self.types.clone(),
            asks: Requests::default(),
            in_parallel: true,
            deferred_evals: Vec::new(),
            store: self.store.clone(),
            kids: self.kids.clone(),
            trait_members: self.trait_members.clone(),
            impl_members: self.impl_members.clone(),
            aliases: self.aliases.clone(),
            eval_errors: self.eval_errors.clone(),
            eval_stack: Vec::new(),
            drop_stack: Vec::new(),
            form_stack: Vec::new(),
        }
    }

    /// Takes the constant evaluations that a parallel pass deferred until worker threads joined.
    pub fn take_deferred_evals(&mut self) -> Vec<(reach::Instance, u32)> {
        std::mem::take(&mut self.deferred_evals)
    }

    /// The lengths of `asks` and `deferred_evals`, unchanged only when a lookup neither asked for unloaded names nor deferred a constant evaluation.
    pub(crate) fn quiet_mark(&self) -> (usize, usize) {
        (self.asks.count(), self.deferred_evals.len())
    }

    /// Whether no lookup in this pass asked for anything to load or resolve, so another pass would find the same.
    pub fn resolved(&self) -> bool {
        self.asks.count() == 0
    }

    /// Ends the pass: the store for the next one, and what lookups asked to load or resolve.
    pub fn release(mut self) -> (Cache, Requests) {
        self.store.types = self.types;
        (self.store, self.asks)
    }

    /// The cached type-namespace definition for `segments` in `scope`.
    pub fn type_def(&self, scope: TypeScope, segments: &[crate::names::Sym]) -> Option<crate::names::Def> {
        self.store.lookups.type_defs.get(&(scope, SegKey::from_slice(segments)))
    }

    /// Caches `def` as the type-namespace definition for `segments` in `scope`.
    pub fn set_type_def(&self, scope: TypeScope, segments: &[crate::names::Sym], def: crate::names::Def) {
        self.store.lookups.type_defs.insert((scope, SegKey::from_slice(segments)), def);
    }

    /// The source file and outline entry of `item`.
    pub fn item(&self, item: ItemRef) -> (&'a Source, &'a Item) {
        let file = self.file(item);
        let index = item.outer().map_or(item.item, |(_, index)| index);
        (&file.source, &file.items.items[index as usize])
    }

    /// The item that `item` is declared directly inside, in the same file or body.
    pub fn parent_item(&self, item: ItemRef) -> Option<ItemRef> {
        if let Some(parent) = self.item(item).1.parent {
            return Some(match item.declaring_body() {
                Some(root) => root.inner(parent),
                None => ItemRef { item: parent, ..item },
            });
        }
        if item.declaring_body().is_some() || !self.file(item).expanded {
            return None;
        }
        let entry = self.outline.crates[item.krate as usize].traits.iter().find(|entry| entry.parts.contains(&(item.file as usize)))?;
        Some(ItemRef { krate: item.krate, file: entry.file as u32, item: entry.at })
    }

    /// The items declared directly inside `of`: the members of a trait or impl, in the file, expanded parts or body that holds it.
    pub fn children(&self, of: ItemRef) -> Vec<ItemRef> {
        if of.outer().is_none()
            && self.item(of).1.kind == ItemKind::Trait
            && let Some(entry) = self.outline.crates[of.krate as usize].traits.iter().find(|entry| entry.file == of.file as usize && entry.at == of.item)
        {
            return self.block_parts(of.krate, entry);
        }
        let at = of.outer().map_or(of.item, |(_, at)| at);
        let file = self.file(of);
        let env = &self.outline.crates[of.krate as usize].env;
        let list = self.kids_of(&file.items.items, Some(at));
        let body = of.declaring_body();
        list.iter().copied().filter(|&index| env.enabled(&file.items.items[index as usize].attrs)).map(|index| body.map_or(ItemRef { item: index, ..of }, |root| root.inner(index))).collect()
    }

    pub(super) fn block_parts(&self, krate: u32, entry: &crate::outline::tree::Impl) -> Vec<ItemRef> {
        let tree = &self.outline.crates[krate as usize];
        let parts = std::iter::once((entry.file, Some(entry.at))).chain(entry.parts.iter().map(|&part| (part, None)));
        let mut found = Vec::new();
        for (file, owner) in parts {
            let items = &tree.files[file].items.items;
            for &at in self.kids_of(items, owner).iter() {
                if tree.env.enabled(&items[at as usize].attrs) {
                    found.push(ItemRef { krate, file: file as u32, item: at });
                }
            }
        }
        found
    }

    /// The indices of the items in `items` declared directly inside item `parent`, or at the top level for none.
    pub fn kids_of(&self, items: &[Item], parent: Option<u32>) -> Arc<[u32]> {
        let key = items.as_ptr() as usize;
        let kids = self.kids.get_or_insert_with(key, || {
            let mut grouped: HashMap<Option<u32>, Vec<u32>> = HashMap::default();
            for (at, item) in items.iter().enumerate() {
                grouped.entry(item.parent).or_default().push(at as u32);
            }
            Arc::new(grouped.into_iter().map(|(owner, list)| (owner, list.into())).collect())
        });
        kids.get(&parent).cloned().unwrap_or_else(|| Arc::from([]))
    }

    /// Whether `item` is a function, static or type declared in an `extern` block, or a bodyless float intrinsic with a host function, which the host links by name.
    pub fn foreign(&self, item: ItemRef) -> bool {
        let (src, entry) = self.item(item);
        if !matches!(entry.kind, ItemKind::Fn | ItemKind::Static | ItemKind::Type) || entry.body.is_some() {
            return false;
        }
        let external = self.parent_item(item).is_some_and(|parent| self.item(parent).1.kind == ItemKind::ExternBlock);
        external || (self.rustc_intrinsic(item) && entry.name.is_some_and(|at| crate::host::intrinsic(src.text(at)).is_some()))
    }

    /// What the body that defines the closure type `ty` inferred for it.
    pub fn closure(&self, ty: ty::Ty) -> Option<Arc<infer::ClosureInfo>> {
        self.store.closures.get(&ty)
    }

    /// Whether the body that defines the closure type `ty` has recorded it.
    pub fn has_closure(&self, ty: ty::Ty) -> bool {
        self.store.closures.contains_key(&ty)
    }

    /// The output of the async fn future `ty` that a call named while its body was still being inferred, from the declared return type.
    pub fn future_output(&self, ty: ty::Ty) -> Option<ty::Ty> {
        self.store.lookups.futures.get(&ty)
    }

    /// Records what the body that defines the closure type `ty` inferred for it.
    pub fn set_closure(&mut self, ty: ty::Ty, info: infer::ClosureInfo) {
        if info.typed.exprs.is_empty() && self.store.closures.get(&ty).is_some_and(|cur| !cur.typed.exprs.is_empty()) {
            return;
        }
        self.store.closures.insert(ty, Arc::new(info));
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
        if ty == ty::Ty::NONE {
            return None;
        }
        match self.types.kind(ty) {
            ty::Kind::Adt(item, args) if self.lang(*item) == Some("owned_box") => self.types.substs(*args).first().copied(),
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
            Kind::Adt(item, args) => format!("{}<{}>", self.name(item), list(self.types.substs(args))),
            Kind::Ref(Mut::Not, inner) => format!("&{}", self.type_text(inner, resolve)),
            Kind::Ref(Mut::Mut, inner) => format!("&mut {}", self.type_text(inner, resolve)),
            Kind::Tuple(items) => format!("({})", list(self.types.substs(items))),
            Kind::Slice(element) => format!("[{}]", self.type_text(element, resolve)),
            Kind::Prim(primitive) => format!("{primitive:?}").to_lowercase(),
            Kind::Param(index) => format!("Param({index})"),
            Kind::Proj(id) => {
                let def = self.types.proj(id);
                format!("<{} as {}>::{}", self.type_text(def.self_ty, resolve), self.name(def.of), self.name(def.assoc))
            }
            other => self.callable_text(other, resolve),
        }
    }

    fn callable_text(&self, kind: ty::Kind, resolve: &dyn Fn(ty::Ty) -> ty::Ty) -> String {
        use ty::{Kind, Mut};
        let list = |tys: &[ty::Ty]| tys.iter().map(|&ty| self.type_text(ty, resolve)).collect::<Vec<_>>().join(", ");
        let angled = |args: substs::Substs| if self.types.substs(args).is_empty() { String::new() } else { format!("<{}>", list(self.types.substs(args))) };
        match kind {
            Kind::FnDef(item, args) => format!("fn {}{}", self.name(item), angled(args)),
            Kind::Closure(item, args, expr) => format!("{{closure#{expr} in {}{}}}", self.name(item), angled(args)),
            Kind::FnPtr(params, ret) => format!("fn({}) -> {}", list(self.types.substs(params)), self.type_text(ret, resolve)),
            Kind::Dyn(item, args, ..) => format!("dyn {}{}", self.name(item), angled(args)),
            Kind::Ptr(Mut::Not, inner) => format!("*const {}", self.type_text(inner, resolve)),
            Kind::Ptr(Mut::Mut, inner) => format!("*mut {}", self.type_text(inner, resolve)),
            Kind::Array(element, len) => format!("[{}; {}]", self.type_text(element, resolve), self.type_text(len, resolve)),
            Kind::Never => "!".to_owned(),
            Kind::Value(bits) => bits.to_string(),
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
        if let Some(of) = self.owner_trait(item) {
            return self.module_of(of);
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

    fn modules_of(&mut self, krate: u32) -> Arc<HashMap<(u32, u32), u32>> {
        let tree = &self.outline.crates[krate as usize];
        self.store.lookups.modules.get_or_insert_with(krate, || {
            let pairs = (0..tree.modules.len()).flat_map(|module| items_of(tree, module).into_iter().map(move |pair| (pair, module as u32)));
            Arc::new(pairs.collect())
        })
    }

    /// The parsed header of the function `item`.
    pub fn signature(&mut self, item: ItemRef) -> Result<Arc<Signature>> {
        if let Some(found) = self.store.signatures.get(&item) {
            return Ok(found);
        }
        let parsed = Arc::new(self.parse_signature(item)?);
        self.store.signatures.insert(item, Arc::clone(&parsed));
        Ok(parsed)
    }

    /// The parsed body of the function `item`.
    pub fn body(&mut self, item: ItemRef) -> Result<Arc<Body>> {
        if let Some(found) = self.store.bodies.get(&item) {
            return Ok(found);
        }
        let mut parsed = self.parse_body(item)?;
        let sig = self.signature(item)?;
        if sig.params.iter().any(|&(pat, _)| !sig.body.plain_param(pat)) {
            parsed = crate::body::parse_params(self.body_src(item), parsed, &sig.param_starts)?;
        }
        let parsed = Arc::new(parsed);
        self.store.bodies.insert(item, Arc::clone(&parsed));
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
    pub fn adt_def(&mut self, item: ItemRef) -> Result<Arc<AdtDef>> {
        if let Some(found) = self.store.adts.get(&item) {
            return Ok(found);
        }
        let (src, entry) = self.item(item);
        let name = entry.name.ok_or_else(|| anyhow!("type item has no name"))?;
        let mut parsed = parse_adt(src, name, entry.end, entry.kind == ItemKind::Enum)?;
        let env = &self.outline.crates[item.krate as usize].env;
        parsed.configure(src, &|attrs| env.enabled(attrs));
        let parsed = Arc::new(parsed);
        self.store.adts.insert(item, Arc::clone(&parsed));
        Ok(parsed)
    }

    /// The type alias `item`, parsed once per pass.
    pub fn alias_def(&self, item: ItemRef) -> Result<Arc<Alias>> {
        if let Some(found) = self.aliases.get(&item) {
            return Ok(found);
        }
        let (src, entry) = self.item(item);
        let (at, eq) = self.alias_eq(item)?;
        let generics: traits::TraitGenerics = match src.text(at + 1) {
            "<" => sig::generics(src, at + 1, eq)?,
            _ => Default::default(),
        };
        let params = generics.1.params.iter().filter(|generic| generic.kind != sig::GenericKind::Lifetime).cloned().collect();
        let (arena, ty) = crate::body::parse_ty(src, eq + 1, entry.end - 1)?;
        let alias = Arc::new(Alias { params, generics: Arc::new(generics), arena, ty });
        self.aliases.insert(item, Arc::clone(&alias));
        Ok(alias)
    }

    fn alias_eq(&self, item: ItemRef) -> Result<(u32, u32)> {
        let (src, entry) = self.item(item);
        let Some(at) = entry.name else { bail!("type alias with no name") };
        let (mut depth, mut found) = (0i32, None);
        for token in at + 1..entry.end {
            match src.kind(token) {
                rustc_lexer::TokenKind::Lt => depth += 1,
                rustc_lexer::TokenKind::Gt if src.kind(token - 1) != rustc_lexer::TokenKind::Minus => depth -= 1,
                rustc_lexer::TokenKind::Eq if depth == 0 => { found = Some(token); break; }
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

    /// The name of `item` as a path or method call names it, without an `r#` prefix.
    pub fn plain_name(&self, item: ItemRef) -> &'a str {
        crate::names::plain(self.name(item))
    }
}
