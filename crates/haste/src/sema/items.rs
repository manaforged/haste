use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::body::Body;
use crate::body::adt::{AdtDef, adt as parse_adt};
use crate::body::sig::{self, Signature};
use crate::names::collect::items_of;
use crate::names::path::Paths;
use crate::names::{ItemRef, ModRef};
use crate::outline::item::{Item, ItemKind};
use crate::outline::source::Source;

use super::{Alias, Context, reach, traits, ty};

fn extern_abi_tokens_may_unwind(src: &Source, start: u32, end: u32) -> bool {
    for tok in start..end {
        if src.text(tok) == "extern" {
            let next = src.text(tok + 1);
            return matches!(
                next,
                "\"Rust\"" | "\"rust-call\"" | "\"rust-cold\"" | "\"rust-intrinsic\""
            ) || (next.starts_with('"') && next.ends_with("-unwind\""));
        }
    }
    true
}

impl<'a> Context<'a> {
    /// The source file and outline entry of `item`.
    #[inline]
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
                None => ItemRef {
                    item: parent,
                    ..item
                },
            });
        }
        if item.declaring_body().is_some() || !self.file(item).expanded {
            return None;
        }
        let entry = self.outline.crates[item.krate as usize]
            .traits
            .iter()
            .find(|entry| entry.parts.contains(&(item.file as usize)))?;
        Some(ItemRef {
            krate: item.krate,
            file: entry.file as u32,
            item: entry.at,
        })
    }

    /// The items declared directly inside `of`: the members of a trait or impl, in the file, expanded parts or body that holds it.
    pub fn children(&self, of: ItemRef) -> Vec<ItemRef> {
        if of.outer().is_none()
            && self.item(of).1.kind == ItemKind::Trait
            && let Some(entry) = self.outline.crates[of.krate as usize]
                .traits
                .iter()
                .find(|entry| entry.file == of.file as usize && entry.at == of.item)
        {
            return self.block_parts(of.krate, entry);
        }
        let at = of.outer().map_or(of.item, |(_, at)| at);
        let file = self.file(of);
        let env = &self.outline.crates[of.krate as usize].env;
        let list = self.kids_of(&file.items.items, Some(at));
        let body = of.declaring_body();
        list.iter()
            .copied()
            .filter(|&index| env.enabled(&file.items.items[index as usize].attrs))
            .map(|index| body.map_or(ItemRef { item: index, ..of }, |root| root.inner(index)))
            .collect()
    }

    pub(crate) fn block_parts(
        &self,
        krate: u32,
        entry: &crate::outline::tree::Impl,
    ) -> Vec<ItemRef> {
        let tree = &self.outline.crates[krate as usize];
        let parts = std::iter::once((entry.file, Some(entry.at)))
            .chain(entry.parts.iter().map(|&part| (part, None)));
        let mut found = Vec::new();
        for (file, owner) in parts {
            let items = &tree.files[file].items.items;
            for &at in self.kids_of(items, owner).iter() {
                if tree.env.enabled(&items[at as usize].attrs) {
                    found.push(ItemRef {
                        krate,
                        file: file as u32,
                        item: at,
                    });
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
            Arc::new(
                grouped
                    .into_iter()
                    .map(|(owner, list)| (owner, list.into()))
                    .collect(),
            )
        });
        kids.get(&parent).cloned().unwrap_or_else(|| Arc::from([]))
    }

    pub(crate) fn is_extern_fn(&self, item: ItemRef) -> bool {
        let (_, entry) = self.item(item);
        entry.kind == ItemKind::Fn
            && entry.body.is_none()
            && self
                .parent_item(item)
                .is_some_and(|parent| self.item(parent).1.kind == ItemKind::ExternBlock)
    }

    /// Whether the `extern` block declaring foreign item `item` uses an unwinding ABI.
    pub(crate) fn extern_block_may_unwind(&self, item: ItemRef) -> bool {
        let Some(parent) = self.parent_item(item) else {
            return true;
        };
        let (src, entry) = self.item(parent);
        if entry.kind != ItemKind::ExternBlock {
            return true;
        }
        let Some(open) = entry.body else {
            return true;
        };
        extern_abi_tokens_may_unwind(src, entry.start, open)
    }

    /// Whether a Rust-defined function item's ABI allows unwinding across its boundary.
    pub(crate) fn fn_abi_may_unwind(&self, item: ItemRef) -> bool {
        let (src, entry) = self.item(item);
        if entry.kind != ItemKind::Fn {
            return true;
        }
        let end = entry.name.or(entry.body).unwrap_or(entry.end);
        extern_abi_tokens_may_unwind(src, entry.start, end)
    }

    /// Whether `item` is a function, static or type declared in an `extern` block, or a bodyless float intrinsic with a host function, which the host links by name.
    pub fn foreign(&self, item: ItemRef) -> bool {
        let (src, entry) = self.item(item);
        if !matches!(entry.kind, ItemKind::Fn | ItemKind::Static | ItemKind::Type)
            || entry.body.is_some()
        {
            return false;
        }
        let external = self
            .parent_item(item)
            .is_some_and(|parent| self.item(parent).1.kind == ItemKind::ExternBlock);
        if external && entry.kind == ItemKind::Fn && self.global_alloc_shim(item).is_some() {
            return false;
        }
        external
            || (self.rustc_intrinsic(item)
                && entry
                    .name
                    .is_some_and(|at| crate::host::intrinsic(src.text(at)).is_some()))
    }

    /// The outline file that holds `item`, or the expansion of the body that declares it.
    #[inline]
    pub fn file(&self, item: ItemRef) -> &'a crate::outline::tree::File {
        match item.outer() {
            Some((outer, _)) => self.expanded_file(outer),
            None => &self.outline.crates[item.krate as usize].files[item.file as usize],
        }
    }

    #[cold]
    fn expanded_file(&self, outer: ItemRef) -> &'a crate::outline::tree::File {
        &self
            .expansion(outer)
            .expect("a body that declares items is read")
            .file
    }

    /// Path resolution inside crate `krate`, recording what it finds not loaded yet.
    pub fn paths(&self, krate: u32) -> Paths<'_> {
        Paths {
            demands: Some(&self.asks),
            ..self.names.paths(self.outline, krate)
        }
    }

    /// The path of the module that holds `item`, or its crate's name when no module is found.
    pub fn module_text(&mut self, item: ItemRef) -> String {
        self.module_of(item).map_or_else(
            |_| self.outline.roots[item.krate as usize].name.clone(),
            |at| self.outline.module_path(at),
        )
    }

    /// The name of `item` under the path of the module that holds it.
    pub fn qualified(&mut self, item: ItemRef) -> String {
        format!("{}::{}", self.module_text(item), self.name(item))
    }

    /// The name of `instance` the way a symbolizer spells it: an impl method under its self type, and a closure or async body as `{closure}` under the function that holds it.
    pub fn symbol_text(&mut self, (item, substs): reach::Instance) -> String {
        let types = self.types.substs(substs).to_vec();
        if let Some(&ty::Kind::Closure(parent, args, _)) =
            types.first().map(|&ty| self.types.kind(ty))
        {
            return format!("{}::{{closure}}", self.symbol_text((parent, args)));
        }
        let Some(name) = self
            .owner_impl(item)
            .and_then(|block| self.impl_self_name(block))
        else {
            return self.qualified(item);
        };
        format!("{}::{name}::{}", self.module_text(item), self.name(item))
    }

    /// The module that holds `item`, through any impl, trait or extern block around it.
    pub fn module_of(&mut self, item: ItemRef) -> Result<ModRef> {
        if let Some(&found) = self.module_of_local.get(&item) {
            return Ok(found);
        }
        let module = self.module_of_uncached(item)?;
        self.module_of_local.insert(item, module);
        Ok(module)
    }

    fn module_of_uncached(&mut self, item: ItemRef) -> Result<ModRef> {
        if let Some((outer, _)) = item.outer() {
            return self.module_of(outer);
        }
        let file = &self.outline.crates[item.krate as usize].files[item.file as usize];
        let mut entry = &file.items.items[item.item as usize];
        match entry.parent {
            Some(parent) => {
                let kind = file.items.items[parent as usize].kind;
                if kind == ItemKind::Impl
                    && let Some(block) = self.owner_impl(item)
                {
                    return self.impl_module(block);
                }
                if kind == ItemKind::Trait {
                    return self.module_of(ItemRef {
                        krate: item.krate,
                        file: item.file,
                        item: parent,
                    });
                }
            }
            None if file.expanded => {
                if let Some(block) = self.owner_impl(item) {
                    return self.impl_module(block);
                }
                if let Some(owner) = self.owner_trait(item) {
                    return self.module_of(owner);
                }
            }
            None => {}
        }
        let mut top = item.item;
        let file = &self.outline.crates[item.krate as usize].files[item.file as usize];
        while let Some(parent) = entry
            .parent
            .filter(|parent| file.items.items[*parent as usize].kind != ItemKind::Mod)
        {
            top = parent;
            entry = &file.items.items[parent as usize];
        }
        let module = self
            .modules_of(item.krate)
            .get(&(item.file, top))
            .copied()
            .with_context(|| format!("no module holds item {item:?}"))?;
        Ok(ModRef {
            krate: item.krate,
            module,
        })
    }

    fn modules_of(&mut self, krate: u32) -> &HashMap<(u32, u32), u32> {
        let tree = &self.outline.crates[krate as usize];
        let store = &self.store.lookups.modules;
        self.crate_modules[krate as usize].get_or_insert_with(|| {
            store.get_or_insert_with(krate, || {
                let pairs = (0..tree.modules.len()).flat_map(|module| {
                    items_of(tree, module)
                        .into_iter()
                        .map(move |pair| (pair, module as u32))
                });
                Arc::new(pairs.collect())
            })
        })
    }

    /// The parsed header of the function `item`.
    pub fn signature(&mut self, item: ItemRef) -> Result<Arc<Signature>> {
        if let Some(found) = self.store.signatures.get(&item) {
            return Ok(found);
        }
        let mut parsed = self.parse_signature(item)?;
        let (src, entry) = self.item(item);
        let env = &self.outline.crates[item.krate as usize].env;
        if crate::outline::attr::has_attr(src, &entry.attrs, env, "rustc_deprecated_safe_2024") {
            parsed.deprecated_safe_2024 = true;
        }
        let parsed = Arc::new(parsed);
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
        if sig
            .params
            .iter()
            .any(|&(pat, _)| !sig.body.plain_param(pat))
        {
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
        let params = generics
            .1
            .params
            .iter()
            .filter(|generic| generic.kind != sig::GenericKind::Lifetime)
            .cloned()
            .collect();
        let (arena, ty) = crate::body::parse_ty(src, eq + 1, entry.end - 1)?;
        let alias = Arc::new(Alias {
            params,
            generics: Arc::new(generics),
            arena,
            ty,
        });
        self.aliases.insert(item, Arc::clone(&alias));
        Ok(alias)
    }

    fn alias_eq(&self, item: ItemRef) -> Result<(u32, u32)> {
        let (src, entry) = self.item(item);
        let Some(at) = entry.name else {
            bail!("type alias with no name")
        };
        let (mut depth, mut found) = (0i32, None);
        for token in at + 1..entry.end {
            match src.kind(token) {
                rustc_lexer::TokenKind::Lt => depth += 1,
                rustc_lexer::TokenKind::Gt
                    if src.kind(token - 1) != rustc_lexer::TokenKind::Minus =>
                {
                    depth -= 1
                }
                rustc_lexer::TokenKind::Eq if depth == 0 => {
                    found = Some(token);
                    break;
                }
                _ => {}
            }
        }
        let Some(eq) = found else {
            bail!("type `{}` has no `=`", src.text(at))
        };
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

    /// External libraries declared by active `#[link(...)]` extern blocks in crate `krate`.
    pub fn crate_link_libs(&self, krate: u32) -> &[String] {
        &self.outline.crates[krate as usize].link_libs
    }

    /// The declared edition of crate `krate`, if non-empty.
    pub fn crate_edition(&self, krate: u32) -> Option<&str> {
        self.names
            .crates
            .get(krate as usize)
            .map(|c| c.edition.as_str())
            .filter(|e| !e.is_empty())
            .or_else(|| {
                self.outline
                    .roots
                    .get(krate as usize)
                    .map(|r| r.edition.as_str())
                    .filter(|e| !e.is_empty())
            })
    }
}
