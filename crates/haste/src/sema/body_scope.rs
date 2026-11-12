//! Names inside a function body: the items that its blocks and the modules it declares hold, found from the innermost scope out, as rustc resolves them.

use anyhow::{Result, anyhow};
use rustc_hash::FxHashMap as HashMap;

use crate::body::{self as syntax, Body, Stmt, sig};
use crate::names::collect::{extern_crate_target, import_parts, namespaces};
use crate::names::path::{Path as NamePath, Paths};
use crate::names::{Binding, Def, ItemRef, ModRef, Ns, Origin, PerNs, SELF, Sym, Symbols, plain};
use crate::outline::item::{Item, ItemKind, Items, Vis};
use crate::outline::scan;
use crate::outline::source::Source;
use crate::outline::tree::{File, Index};

use super::Context;

/// Where a path written inside a body leads.
pub(super) enum BodyPath {
    /// A definition that the body declares or links through `extern crate`.
    Def(Def),
    /// An enum that the body declares, whose variant the last segment names.
    Variant(ItemRef),
    /// Out of the body's modules through `super`: the rest of the path, after this many segments, resolves from the home module.
    Outside(usize),
}

/// The rest of `path` once its first `skip` segments led out of the body's modules, to resolve from the home module.
pub(super) fn outside<'p>(path: &NamePath<'p>, skip: usize) -> NamePath<'p> {
    NamePath { segments: &path.segments[skip..], absolute: false, dollar: path.dollar }
}

/// The namespace that the first of `len` path segments is looked up in, for a path that names something in `ns`.
pub(super) fn head_ns(len: usize, ns: Ns) -> Ns {
    if len == 1 { ns } else { Ns::Types }
}

/// Whether the item `entry` binds a name in namespace `ns`.
pub(super) fn in_namespace(entry: &Item, ns: Ns) -> bool {
    if matches!(entry.kind, ItemKind::Mod | ItemKind::ExternCrate) { ns == Ns::Types } else { namespaces(entry).contains(&ns) }
}

impl<'a> Context<'a> {
    fn body_entries(&self, root: ItemRef) -> &'a [Item] {
        self.expansion(root).map_or(&[][..], |found| &found.file.items.items[..])
    }

    /// The item named `name`, in namespace `ns` or any when none, that the body item `level` of `root`, or the body itself when none, holds directly or in an extern block, visible at body item `from`.
    fn body_child(&self, root: ItemRef, level: Option<u32>, from: Option<u32>, name: &str, ns: Option<Ns>) -> Option<ItemRef> {
        let found = self.expansion(root)?;
        let items = &found.file.items.items;
        let held = self.body_index(&found.file).items.get(&level)?.get(name)?;
        let matches = |at: u32| ns.is_none_or(|ns| in_namespace(&items[at as usize], ns));
        if level.is_some_and(|at| items[at as usize].kind == ItemKind::Mod) || from.is_none() {
            return held.iter().copied().find(|&at| matches(at)).map(|at| root.inner(at));
        }
        let mut cur = from.and_then(|at| found.item_scopes.get(at as usize).copied());
        while let Some(scope) = cur {
            if let Some(&at) = held.iter().rev().find(|&&at| found.item_scopes.get(at as usize) == Some(&scope) && matches(at)) {
                return Some(root.inner(at));
            }
            cur = found.scopes.get(scope as usize).copied().flatten();
        }
        None
    }

    /// The name index of the body expansion `file`, made on first use and kept with it.
    fn body_index(&self, file: &'a File) -> &'a Index {
        file.index(|file| index_of(file, &self.names.symbols))
    }

    /// The item named `name` that code at the body item `from` of `root`, or in the body's own blocks when none, sees: in the blocks around it, then in its module, innermost first.
    fn scope_item(&self, root: ItemRef, from: Option<u32>, name: &str, ns: Option<Ns>) -> Option<ItemRef> {
        let items = self.body_entries(root);
        let mut level = from;
        loop {
            let searched = level.is_none_or(|at| !matches!(items[at as usize].kind, ItemKind::Impl | ItemKind::Trait));
            if searched && let Some(found) = self.body_child(root, level, from, name, ns) {
                return Some(found);
            }
            match level {
                Some(at) if items[at as usize].kind != ItemKind::Mod => level = items[at as usize].parent,
                _ => return None,
            }
        }
    }

    /// Where `path`, written at `at` in a body of module `home`, leads among the items that the body of `at.0` declares, where `at.1` is the body item it is written in, or none when it does not start among them.
    pub(super) fn local_path(&self, home: ModRef, at: Option<(ItemRef, Option<u32>)>, path: &NamePath<'_>, ns: Ns) -> Option<BodyPath> {
        let (root, from) = at.filter(|_| !path.absolute)?;
        let texts: Vec<&str> = path.segments.iter().map(|&segment| self.names.symbols.text(segment)).collect();
        self.body_path((root, home), from, &texts, ns)
    }

    /// Where `rest` leads from the body item `item` of a body in module `home`, which a block around the path declares.
    pub(super) fn local_descend(&self, home: ModRef, item: ItemRef, rest: &[crate::names::Sym], ns: Ns) -> Option<BodyPath> {
        let (root, at) = item.outer()?;
        let texts: Vec<&str> = rest.iter().map(|&segment| self.names.symbols.text(segment)).collect();
        self.body_descend((root, home), at, &texts, ns)
    }

    /// Where `segments`, written at the body item `from` of `root`, lead through the items and modules that the body declares, or none when the path does not start among them.
    fn body_path(&self, (root, home): (ItemRef, ModRef), from: Option<u32>, segments: &[&str], ns: Ns) -> Option<BodyPath> {
        let items = self.body_entries(root);
        let enclosing = |mut at: Option<u32>| {
            while let Some(index) = at.filter(|&index| items[index as usize].kind != ItemKind::Mod) {
                at = items[index as usize].parent;
            }
            at
        };
        let supers = segments.iter().take_while(|segment| **segment == "super").count();
        let (start, rest) = match (supers, segments.first()) {
            (0, None) => return None,
            (0, Some(&"self")) => (enclosing(from)?, &segments[1..]),
            (0, Some(first)) => {
                (self.scope_item(root, from, first, Some(head_ns(segments.len(), ns)))?.outer()?.1, &segments[1..])
            }
            _ => {
                let mut module = enclosing(from)?;
                for done in 1..=supers {
                    match enclosing(items[module as usize].parent) {
                        Some(outer) => module = outer,
                        None => return Some(BodyPath::Outside(done)),
                    }
                }
                (module, &segments[supers..])
            }
        };
        self.body_descend((root, home), start, rest, ns)
    }

    fn body_descend(&self, (root, home): (ItemRef, ModRef), mut at: u32, rest: &[&str], ns: Ns) -> Option<BodyPath> {
        let items = self.body_entries(root);
        if items[at as usize].kind == ItemKind::ExternCrate {
            return self.extern_descend((root, home), at, rest, ns);
        }
        for (index, name) in rest.iter().enumerate() {
            let last = index + 1 == rest.len();
            match items[at as usize].kind {
                ItemKind::Enum if last => return Some(BodyPath::Variant(root.inner(at))),
                ItemKind::Mod => {
                    let Some(next) = self.body_child(root, Some(at), None, name, Some(if last { ns } else { Ns::Types })) else {
                        return self.module_import((root, home), at, &rest[index..], ns);
                    };
                    let next = next.outer()?.1;
                    if items[next as usize].kind == ItemKind::ExternCrate {
                        return self.extern_descend((root, home), next, &rest[index + 1..], ns);
                    }
                    at = next;
                }
                _ => return None,
            }
        }
        Some(BodyPath::Def(Def::Item(root.inner(at))))
    }

    fn extern_descend(&self, (root, home): (ItemRef, ModRef), at: u32, rest: &[&str], ns: Ns) -> Option<BodyPath> {
        let found = self.expansion(root)?;
        let item = &found.file.items.items[at as usize];
        let deps = &self.outline.deps[root.krate as usize];
        let def = Def::Module(extern_crate_target(&found.file.source, item, root.krate, deps)?);
        if rest.is_empty() {
            return (ns == Ns::Types).then_some(BodyPath::Def(def));
        }
        self.def_descend(home, def, rest, ns)
    }

    /// Where `rest` leads through the `use` declaration of the body module `at` that binds its first segment.
    fn module_import(&self, (root, home): (ItemRef, ModRef), at: u32, rest: &[&str], ns: Ns) -> Option<BodyPath> {
        let file = &self.expansion(root)?.file;
        let (first, more) = rest.split_first()?;
        let word = self.names.symbols.intern(first);
        let paths = self.paths(home.krate);
        let held = |import: &&u32| file.items.items[file.items.imports[**import as usize].item as usize].parent == Some(at);
        let imports = self.body_index(file).imports.get(*first)?;
        let (_, entry) = imports.iter().filter(held).find_map(|&import| self.import_entry((root, home), &paths, (file, import), Some(word)))?;
        let binding = if more.is_empty() { entry.get(ns) } else { entry.types };
        self.def_descend(home, binding?.def, more, ns)
    }

    /// Where `rest` leads from `def` through the members of each definition it names.
    fn def_descend(&self, home: ModRef, mut def: Def, rest: &[&str], ns: Ns) -> Option<BodyPath> {
        for (index, name) in rest.iter().enumerate() {
            let entry = self.path_member(home, def, self.names.symbols.intern(name));
            let binding = if index + 1 == rest.len() { entry.get(ns) } else { entry.types };
            def = binding?.def;
        }
        Some(BodyPath::Def(def))
    }

    /// The bindings that the `use` declarations of the body `root` in module `home` give `word`, or every name they import when none, as code at token `here` sees them: named imports first, then globs.
    pub(super) fn body_imports(&self, (root, home): (ItemRef, ModRef), word: Option<Sym>, (here, hidden): (Option<u32>, &[u32])) -> Vec<PerNs> {
        let Some(found) = self.expansion(root) else {
            return Vec::new();
        };
        let file = &found.file;
        if file.items.imports.is_empty() {
            return Vec::new();
        }
        let (paths, index) = (self.paths(home.krate), self.body_index(file));
        let named = word.and_then(|word| index.imports.get(self.names.symbols.text(word))).map_or(&[][..], |list| &list[..]);
        let globbed = if word.is_some() { &index.globs[..] } else { &[][..] };
        let every = if word.is_none() { 0..file.items.imports.len() as u32 } else { 0..0 };
        let (mut out, mut globs) = (Vec::new(), Vec::new());
        for import in named.iter().chain(globbed).copied().chain(every) {
            if !import_visible(file, index, import, here) || hidden.contains(&file.items.imports[import as usize].item) {
                continue;
            }
            match self.import_entry((root, home), &paths, (file, import), word) {
                Some((true, entry)) => out.push(entry),
                Some((false, entry)) => globs.push(entry),
                None => {}
            }
        }
        out.extend(globs);
        out
    }

    /// What the import `import` of the body expansion `file` gives `word`, or every name it imports when none, and whether it is a named import rather than a glob.
    fn import_entry(&self, (root, home): (ItemRef, ModRef), paths: &Paths<'_>, (file, import): (&File, u32), word: Option<Sym>) -> Option<(bool, PerNs)> {
        let (segments, absolute, dollar, name) = import_parts(file, import, &self.names.symbols);
        let segments = segments.strip_suffix(&[SELF]).unwrap_or(&segments);
        let absolute = paths.use_path(NamePath { segments, absolute, dollar }).absolute;
        let from = file.items.items[file.items.imports[import as usize].item as usize].parent;
        let (head, leaf) = match (name, segments.split_last(), word) {
            (None, _, Some(word)) => (segments, word),
            (Some(_), Some((&leaf, head)), _) => (head, leaf),
            (None, Some(_), None) => {
                let path = NamePath { segments, absolute, dollar };
                let def = match self.local_path(home, Some((root, from)), &path, Ns::Types) {
                    Some(BodyPath::Outside(skip)) if skip == segments.len() => Some(Def::Module(home)),
                    Some(BodyPath::Outside(skip)) => paths.resolve(&outside(&path, skip), home),
                    found => self.body_binding(found, "").map(|binding| binding.def).or_else(|| paths.resolve(&path, home)),
                }?;
                return Some((false, PerNs { types: Some(Binding { def, vis: Vis::Pub, origin: Origin::Glob }), values: None, macros: None }));
            }
            _ => return None,
        };
        let local = if absolute { None } else { self.body_import((root, home), from, head, leaf) };
        let entry = match head {
            _ if let Some(local) = local => local,
            [] => paths.first(leaf, home, &NamePath { segments: &[leaf], absolute, dollar }),
            _ => self.path_member(home, paths.resolve(&NamePath { segments: head, absolute, dollar }, home)?, leaf),
        };
        (entry.types.is_some() || entry.values.is_some()).then_some((name.is_some(), entry))
    }

    /// The bindings of `leaf` in the item that `head`, written at the body item `from` of `root`, names among the items the body declares: a variant of an enum or an item of a module.
    fn body_import(&self, (root, home): (ItemRef, ModRef), from: Option<u32>, head: &[Sym], leaf: Sym) -> Option<PerNs> {
        if head.is_empty() {
            return None;
        }
        let texts: Vec<&str> = head.iter().chain([&leaf]).map(|&segment| self.names.symbols.text(segment)).collect();
        let found = |ns| self.body_binding(self.body_path((root, home), from, &texts, ns), texts[texts.len() - 1]);
        let entry = PerNs { types: found(Ns::Types), values: found(Ns::Values), macros: None };
        (entry.types.is_some() || entry.values.is_some()).then_some(entry)
    }

    /// The bindings of `name` inside `def`, an enum's variants or a module's items, for a path written in module `home`, where a body may declare `def`.
    pub(super) fn path_member(&self, home: ModRef, def: Def, name: Sym) -> PerNs {
        let Some((root, at)) = (match def {
            Def::Item(item) => item.outer(),
            _ => None,
        }) else {
            return self.paths(home.krate).member(def, name);
        };
        let text = self.names.symbols.text(name);
        let found = |ns| self.body_binding(self.body_descend((root, home), at, &[text], ns), text);
        PerNs { types: found(Ns::Types), values: found(Ns::Values), macros: None }
    }

    /// The binding of the body item or enum variant `name` that `path` reached.
    fn body_binding(&self, path: Option<BodyPath>, name: &str) -> Option<Binding> {
        let def = match path? {
            BodyPath::Def(def) => def,
            BodyPath::Variant(item) => self.body_variant(item, name)?,
            BodyPath::Outside(_) => return None,
        };
        Some(Binding { def, vis: Vis::Pub, origin: Origin::Item })
    }

    /// The variant named `name` of the enum `item` that a body declares.
    fn body_variant(&self, item: ItemRef, name: &str) -> Option<Def> {
        let (root, at) = item.outer()?;
        let found = self.expansion(root)?;
        let open = found.file.items.items[at as usize].body?;
        let index = crate::names::path::variant_names(&found.file.source, open).iter().position(|variant| *variant == name)?;
        Some(Def::Variant(item, index as u32))
    }

    /// Resolves `path` in the type namespace through the `use` declarations of the body `root` in module `home`, as code at token `here` sees them, when one binds its first segment.
    pub(super) fn body_use_path(&self, (root, home): (ItemRef, ModRef), path: &NamePath<'_>, here: Option<u32>) -> Option<Def> {
        let (first, rest) = path.segments.split_first().filter(|_| !path.absolute)?;
        let binding = self.body_imports((root, home), Some(*first), (here, &[])).into_iter().find_map(|entry| entry.types)?;
        rest.iter().try_fold(binding.def, |def, segment| self.path_member(home, def, *segment).types.map(|found| found.def))
    }
}

/// Whether the `use` declaration `import` of a body's expansion `file` is in scope at token `here`, or in the body itself when `here` is none: it is unless a function or module declared in the body holds it and not `here`.
fn import_visible(file: &File, index: &Index, import: u32, here: Option<u32>) -> bool {
    index.holders[import as usize].iter().all(|&at| here.is_some_and(|here| holds(&file.items.items[at as usize], here)))
}

/// Whether the tokens of `entry` hold token `token`.
fn holds(entry: &Item, token: u32) -> bool {
    entry.start <= token && token < entry.end
}

/// The name index of a body's expansion `file`: its items by holder and plain name, its imports by the name `symbols` spells for them, and the functions around each import.
fn index_of(file: &File, symbols: &Symbols) -> Index {
    let (src, items) = (&file.source, &file.items.items);
    let mut index = Index::default();
    for (at, entry) in items.iter().enumerate() {
        let Some(name) = entry.name else { continue };
        let block = entry.parent.filter(|&parent| items[parent as usize].kind == ItemKind::ExternBlock);
        for level in [Some(entry.parent), block.map(|parent| items[parent as usize].parent)].into_iter().flatten() {
            index.items.entry(level).or_default().entry(plain(src.text(name)).into()).or_default().push(at as u32);
        }
    }
    let scopes: Vec<u32> = (0..items.len() as u32).filter(|&at| matches!(items[at as usize].kind, ItemKind::Fn | ItemKind::Mod)).collect();
    for import in 0..file.items.imports.len() as u32 {
        match import_parts(file, import, symbols).3 {
            Some(name) => index.imports.entry(symbols.text(name).into()).or_default().push(import),
            None => index.globs.push(import),
        }
        let start = items[file.items.imports[import as usize].item as usize].start;
        index.holders.push(scopes.iter().copied().filter(|&at| holds(&items[at as usize], start)).collect());
    }
    index
}

pub(super) fn parse_in(src: &Source, config: syntax::Config<'_>, entry: &Item) -> Result<Body> {
    match (entry.kind, entry.name) {
        (ItemKind::Const | ItemKind::Static, Some(name)) => {
            let from = sig::initializer(src, name, entry.end).ok_or_else(|| anyhow!("constant {} has no value", src.text(name)))?;
            syntax::parse_expr(src, from, entry.end - 1, config)
        }
        _ => syntax::parse(src, entry.body.ok_or_else(|| anyhow!("function {} has no body", src.text(entry.name.unwrap_or(entry.start))))?, config),
    }
}

/// The items that `body` declares, and those that the bodies of its functions and constants declare in turn, with their block scopes.
pub(super) fn declared(source: &Source, config: syntax::Config<'_>, body: &Body) -> Result<(Items, Vec<Option<u32>>, Vec<u32>)> {
    let mut items = Items::default();
    let mut scopes = vec![None];
    let mut item_scopes = Vec::new();
    let starts = body_starts(body, 0, &mut scopes).into_iter().map(|(at, scope)| (at, None, scope)).collect();
    scan_nested(source, config, starts, &mut items, &mut scopes, &mut item_scopes)?;
    Ok((items, scopes, item_scopes))
}

pub(super) fn scan_nested(source: &Source, config: syntax::Config<'_>, mut starts: Vec<(u32, Option<u32>, u32)>, items: &mut Items, scopes: &mut Vec<Option<u32>>, item_scopes: &mut Vec<u32>) -> Result<()> {
    while !starts.is_empty() {
        let first = items.items.len();
        let by_lead: HashMap<u32, u32> = starts.iter().map(|&(at, _, scope)| (at, scope)).collect();
        let pairs = std::mem::take(&mut starts).into_iter().map(|(at, owner, _)| (at, owner));
        *items = scan::body_items(source, pairs, (std::mem::take(items), config.calls))?;
        for entry in &items.items[first..] {
            let scope = by_lead.get(&entry.lead).copied().or_else(|| entry.parent.map(|parent| item_scopes[parent as usize])).unwrap_or(0);
            item_scopes.push(scope);
        }
        for index in first..items.items.len() {
            let entry = &items.items[index];
            let has_body = matches!(entry.kind, ItemKind::Const | ItemKind::Static) || (entry.kind == ItemKind::Fn && entry.body.is_some());
            if has_body && let Ok(inner) = parse_in(source, config, entry) {
                starts.extend(body_starts(&inner, item_scopes[index], scopes).into_iter().map(|(at, scope)| (at, Some(index as u32), scope)));
            }
        }
    }
    Ok(())
}

fn body_starts(body: &Body, base: u32, scopes: &mut Vec<Option<u32>>) -> Vec<(u32, u32)> {
    let mut starts = Vec::new();
    if let Some(root) = body.root {
        ScopeWalk { body, scopes, starts: &mut starts }.expr(root, base);
    }
    starts
}

struct ScopeWalk<'b> {
    body: &'b Body,
    scopes: &'b mut Vec<Option<u32>>,
    starts: &'b mut Vec<(u32, u32)>,
}

impl ScopeWalk<'_> {
    fn expr_list(&mut self, list: syntax::List, scope: u32) {
        for &id in self.body.list(list) {
            self.expr(syntax::ExprId(id), scope);
        }
    }

    fn expr(&mut self, id: syntax::ExprId, scope: u32) {
        use syntax::{BlockKind, Expr};
        match *self.body.expr(id) {
            Expr::Block { stmts, tail, kind, .. } => {
                let inner = if kind == BlockKind::Expansion {
                    scope
                } else {
                    let next = self.scopes.len() as u32;
                    self.scopes.push(Some(scope));
                    next
                };
                for &stmt in self.body.list(stmts) {
                    self.stmt(stmt, inner);
                }
                if let Some(tail) = tail {
                    self.expr(tail, inner);
                }
            }
            Expr::If(cond, then, otherwise) => {
                self.expr(cond, scope);
                self.expr(then, scope);
                if let Some(otherwise) = otherwise {
                    self.expr(otherwise, scope);
                }
            }
            Expr::Match(scrutinee, arms) => {
                self.expr(scrutinee, scope);
                for &arm in self.body.list(arms) {
                    let arm = self.body.arms[arm as usize];
                    self.pat(arm.pat, scope);
                    if let Some(guard) = arm.guard {
                        self.expr(guard, scope);
                    }
                    self.expr(arm.body, scope);
                }
            }
            other => self.expr_compound(other, scope),
        }
    }

    fn expr_compound(&mut self, expr: syntax::Expr, scope: u32) {
        use syntax::Expr;
        match expr {
            Expr::Unary(_, a) | Expr::Field(a, _) | Expr::TupleField(a, _) | Expr::Ref { expr: a, .. } | Expr::Loop { body: a, .. } | Expr::Try(a) | Expr::Await(a) | Expr::Become(a) | Expr::Paren(a) => {
                self.expr(a, scope);
            }
            Expr::Binary(_, a, b) | Expr::Assign(a, b) | Expr::AssignOp(_, a, b) | Expr::Index(a, b) | Expr::While { cond: a, body: b, .. } | Expr::Repeat(a, b) => {
                self.expr(a, scope);
                self.expr(b, scope);
            }
            Expr::Let(pat, init) => {
                self.pat(pat, scope);
                self.expr(init, scope);
            }
            Expr::For { pat, iter, body, .. } => {
                self.pat(pat, scope);
                self.expr(iter, scope);
                self.expr(body, scope);
            }
            Expr::Closure { params, ret, body, .. } => {
                for &param in self.body.list(params) {
                    let param = self.body.params[param as usize];
                    self.pat(param.pat, scope);
                    if let Some(ty) = param.ty {
                        self.ty(ty, scope);
                    }
                }
                if let Some(ret) = ret {
                    self.ty(ret, scope);
                }
                self.expr(body, scope);
            }
            other => self.expr_leaf(other, scope),
        }
    }

    fn expr_leaf(&mut self, expr: syntax::Expr, scope: u32) {
        use syntax::Expr;
        match expr {
            Expr::Call(callee, args) => {
                self.expr(callee, scope);
                self.expr_list(args, scope);
            }
            Expr::Method { receiver, params, .. } => {
                self.expr(receiver, scope);
                self.expr_list(params, scope);
            }
            Expr::Tuple(list) | Expr::Array(list) => self.expr_list(list, scope),
            Expr::Struct { fields, base, .. } => {
                for &field in self.body.list(fields) {
                    self.expr(self.body.fields[field as usize].value, scope);
                }
                if let Some(base) = base {
                    self.expr(base, scope);
                }
            }
            Expr::Range { from, to, .. } => {
                if let Some(from) = from {
                    self.expr(from, scope);
                }
                if let Some(to) = to {
                    self.expr(to, scope);
                }
            }
            Expr::Cast(inner, ty) => {
                self.expr(inner, scope);
                self.ty(ty, scope);
            }
            Expr::Return(Some(a)) | Expr::Break(_, Some(a)) | Expr::Yield(Some(a)) => self.expr(a, scope),
            _ => {}
        }
    }

    fn stmt(&mut self, index: u32, scope: u32) {
        match self.body.stmts[index as usize] {
            Stmt::Item(start, _) => self.starts.push((start, scope)),
            Stmt::Expr(expr, _) => self.expr(expr, scope),
            Stmt::Let { pat, ty, init, otherwise, .. } => {
                self.pat(pat, scope);
                if let Some(ty) = ty {
                    self.ty(ty, scope);
                }
                if let Some(init) = init {
                    self.expr(init, scope);
                }
                if let Some(otherwise) = otherwise {
                    self.expr(otherwise, scope);
                }
            }
            Stmt::Macro { .. } | Stmt::Empty => {}
        }
    }

    fn pat(&mut self, id: syntax::PatId, scope: u32) {
        use syntax::Pat;
        match *self.body.pat(id) {
            Pat::Lit(expr) => self.expr(expr, scope),
            Pat::Range { from, to, .. } => {
                if let Some(from) = from {
                    self.expr(from, scope);
                }
                if let Some(to) = to {
                    self.expr(to, scope);
                }
            }
            Pat::Bind { sub: Some(inner), .. } | Pat::Ref { pat: inner, .. } | Pat::Box(inner) | Pat::Paren(inner) | Pat::Deref(inner) => self.pat(inner, scope),
            Pat::TupleStruct(_, list) | Pat::Tuple(list) | Pat::Slice(list) | Pat::Or(list) => {
                for &pat in self.body.list(list) {
                    self.pat(syntax::PatId(pat), scope);
                }
            }
            Pat::Struct { fields, .. } => {
                for &field in self.body.list(fields) {
                    self.pat(self.body.field_pats[field as usize].pat, scope);
                }
            }
            _ => {}
        }
    }

    fn ty(&mut self, id: syntax::TyId, scope: u32) {
        use syntax::Ty;
        match *self.body.ty(id) {
            Ty::Array(inner, len) => {
                self.ty(inner, scope);
                self.expr(len, scope);
            }
            Ty::Const(expr) => self.expr(expr, scope),
            Ty::Ref { ty: inner, .. } | Ty::Ptr { ty: inner, .. } | Ty::Slice(inner) | Ty::Paren(inner) | Ty::Pat(inner, _) => self.ty(inner, scope),
            Ty::Tuple(list) => {
                for &ty in self.body.list(list) {
                    self.ty(syntax::TyId(ty), scope);
                }
            }
            _ => {}
        }
    }
}
