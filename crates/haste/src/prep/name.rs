use std::collections::{BTreeSet, VecDeque};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::graph::{Side, Workspace};
use crate::names::{Def, ItemRef, ModRef, Names, Ns};
use crate::outline::item::Vis;
use crate::outline::{Outline, Role};
use crate::sema::Context;
use crate::sema::impls::ImplRef;
use crate::sema::infer::{detached, type_params};
use crate::sema::reach::Instance;
use crate::sema::ty::{Kind, Mut, Ty};

const SYSROOT_ROOTS: [&str; 3] = ["core", "alloc", "std"];
const NAMESPACES: [Ns; 2] = [Ns::Types, Ns::Values];
const ALLOCATOR: (&str, &str) = ("alloc", "Global");
const IMPL_TRAIT: &str = "impl";
const ROOT_MODULE: u32 = 0;
const KEY_MARK: &str = " #";

/// The name of the function that the manifest text `key` calls: the last segment of its path outside angle brackets, past any turbofish.
pub(super) fn function(key: &str) -> &str {
    let path = key.split_once(KEY_MARK).map_or(key, |(path, _)| path);
    let (mut depth, mut start, mut name) = (0i32, 0, path);
    for (at, byte) in path.bytes().enumerate() {
        match byte {
            b'<' => depth += 1,
            b'>' => depth -= 1,
            b':' if depth == 0 && path[at + 1..].starts_with(':') => {
                name = &path[start..at];
                start = at + 2;
            }
            _ => {}
        }
    }
    let last = &path[start..];
    if last.starts_with('<') { name } else { last }
}

/// The crate name the shim gives the package of `unit`.
pub(super) fn alias(unit: usize) -> String {
    format!("u{unit}")
}

/// The shortest public path of every item that a crate the shim can name exports, found from each crate root through public modules and re-exports.
pub(super) struct Paths {
    items: HashMap<ItemRef, String>,
}

/// A callee as the shim calls it: its path with any turbofish, and the text that names the instance in a manifest.
pub(super) struct Callee {
    pub path: String,
    pub key: String,
}

impl Paths {
    /// The paths of the items that `names` resolved, for the crates of `workspace` the shim can depend on.
    pub(super) fn new(outline: &Outline, names: &Names, workspace: &Workspace) -> Paths {
        let mut paths = Paths { items: HashMap::default() };
        let mut seen = HashSet::default();
        for (krate, prefix) in roots(outline, workspace) {
            paths.walk(names, (ModRef { krate, module: ROOT_MODULE }, prefix), &mut seen);
        }
        paths
    }

    fn walk(&mut self, names: &Names, start: (ModRef, String), seen: &mut HashSet<ModRef>) {
        let mut queue = VecDeque::from([start]);
        while let Some((module, path)) = queue.pop_front() {
            if !seen.insert(module) {
                continue;
            }
            for (name, def) in exports(names, module) {
                let path = format!("{path}::{name}");
                match def {
                    Def::Module(inner) => queue.push_back((inner, path)),
                    Def::Item(item) => {
                        self.items.entry(item).or_insert(path);
                    }
                    Def::Variant(..) | Def::Primitive(_) => {}
                }
            }
        }
    }

    fn get(&self, item: ItemRef) -> Option<&str> {
        self.items.get(&item).map(String::as_str)
    }
}

fn roots(outline: &Outline, workspace: &Workspace) -> Vec<(u32, String)> {
    let library = |krate: &usize| outline.roots[*krate].role == Role::Lib && workspace.graph.units[outline.roots[*krate].unit].side == Side::Target;
    let crates: Vec<usize> = (0..outline.roots.len()).filter(library).collect();
    let mut found = Vec::new();
    for name in SYSROOT_ROOTS {
        let named = crates.iter().filter(|&&krate| outline.package(workspace, krate).sysroot && outline.roots[krate].name == name);
        found.extend(named.map(|&krate| (krate as u32, format!("::{name}"))));
    }
    let registry = crates.iter().filter(|&&krate| !outline.package(workspace, krate).sysroot && !outline.package(workspace, krate).local());
    found.extend(registry.map(|&krate| (krate as u32, format!("::{}", alias(outline.roots[krate].unit)))));
    found
}

fn exports(names: &Names, module: ModRef) -> Vec<(&'static str, Def)> {
    let Some(scope) = names.crates.get(module.krate as usize).and_then(|krate| krate.scopes.get(module.module as usize)) else {
        return Vec::new();
    };
    let public = |entry: &crate::names::PerNs| NAMESPACES.iter().filter_map(|&ns| entry.get(ns)).filter(|binding| binding.vis == Vis::Pub).map(|binding| binding.def).collect::<Vec<_>>();
    let mut found: Vec<(&'static str, Def)> = scope.names.iter().flat_map(|(name, entry)| public(entry).into_iter().map(|def| (names.symbols.text(*name), def))).collect();
    found.sort_by(|left, right| left.0.cmp(right.0));
    found
}

/// Prints types and callees as Rust source the shim compiles, and records the crates the printed paths name.
pub(super) struct Printer<'p> {
    paths: &'p Paths,
    pub krates: BTreeSet<u32>,
}

impl<'p> Printer<'p> {
    /// A printer over `paths` that has named no crate yet.
    pub(super) fn new(paths: &'p Paths) -> Printer<'p> {
        Printer { paths, krates: BTreeSet::new() }
    }

    /// `ty` as the shim spells it, or none when the shim cannot name it.
    pub(super) fn ty(&mut self, cx: &Context<'_>, ty: Ty) -> Option<String> {
        Some(match cx.types.kind(ty).clone() {
            Kind::Prim(primitive) => format!("{primitive:?}").to_lowercase(),
            Kind::Tuple(items) => self.tuple(cx, cx.types.substs(items))?,
            Kind::Ref(Mut::Not, inner) => format!("&{}", self.ty(cx, inner)?),
            Kind::Ref(Mut::Mut, inner) => format!("&mut {}", self.ty(cx, inner)?),
            Kind::Ptr(Mut::Not, inner) => format!("*const {}", self.ty(cx, inner)?),
            Kind::Ptr(Mut::Mut, inner) => format!("*mut {}", self.ty(cx, inner)?),
            Kind::Array(element, len) => format!("[{}; {}]", self.ty(cx, element)?, cx.types.len(len)?),
            Kind::Slice(element) => format!("[{}]", self.ty(cx, element)?),
            Kind::Adt(item, args) => self.named(cx, item, cx.types.substs(args))?,
            _ => return None,
        })
    }

    fn tuple(&mut self, cx: &Context<'_>, items: &[Ty]) -> Option<String> {
        let printed = self.list(cx, items)?;
        Some(if items.len() == 1 { format!("({printed},)") } else { format!("({printed})") })
    }

    fn list(&mut self, cx: &Context<'_>, items: &[Ty]) -> Option<String> {
        let printed: Option<Vec<String>> = items.iter().map(|&item| self.arg(cx, item)).collect();
        Some(printed?.join(", "))
    }

    fn arg(&mut self, cx: &Context<'_>, ty: Ty) -> Option<String> {
        match *cx.types.kind(ty) {
            Kind::Value(bits) => Some(bits.to_string()),
            _ => self.ty(cx, ty),
        }
    }

    fn named(&mut self, cx: &Context<'_>, item: ItemRef, args: &[Ty]) -> Option<String> {
        let path = self.paths.get(item)?.to_owned();
        self.krates.insert(item.krate);
        let args = match args.split_last() {
            Some((&last, rest)) if allocator(cx, last) => rest,
            _ => args,
        };
        Some(if args.is_empty() { path } else { format!("{path}<{}>", self.list(cx, args)?) })
    }

    /// The callee that `instance` names, or none when the shim cannot call it by a path.
    pub(super) fn callee(&mut self, cx: &mut Context<'_>, (item, substs): Instance) -> Option<Callee> {
        if item.declaring_body().is_some() {
            return None;
        }
        let substs = cx.types.substs(substs).to_vec();
        let (head, own) = match (cx.owner_impl(item), cx.owner_trait(item)) {
            (Some(block), _) => self.impl_head(cx, item, block, &substs)?,
            (None, Some(of)) => self.trait_head(cx, item, of, &substs)?,
            (None, None) => (self.named(cx, item, &[])?, 0),
        };
        let own = substs.get(own..)?;
        let turbofish = if own.is_empty() || impl_trait_params(cx, item) { String::new() } else { format!("::<{}>", self.list(cx, own)?) };
        let path = format!("{head}{turbofish}");
        let key = format!("{path}{KEY_MARK}{}", self.list(cx, &substs)?);
        Some(Callee { path, key })
    }

    fn impl_head(&mut self, cx: &mut Context<'_>, item: ItemRef, block: ImplRef, substs: &[Ty]) -> Option<(String, usize)> {
        let count = detached(cx, item, |infer| infer.impl_params(block)).ok()?;
        let outer = substs.get(..count)?.to_vec();
        let self_ty = detached(cx, item, |infer| infer.impl_frame(block, &outer).map(|(_, ty)| ty)).ok()?;
        let name = cx.name(item);
        let printed = self.ty(cx, self_ty)?;
        let Some(of) = cx.impl_trait(block) else {
            return Some((format!("<{printed}>::{name}"), count));
        };
        let args = detached(cx, item, |infer| infer.impl_trait_args(block, &outer, self_ty)).ok()?;
        let bound = self.named(cx, of, &args)?;
        Some((format!("<{printed} as {bound}>::{name}"), count))
    }

    fn trait_head(&mut self, cx: &mut Context<'_>, item: ItemRef, of: ItemRef, substs: &[Ty]) -> Option<(String, usize)> {
        let generics = cx.trait_generics(of).ok()?;
        let count = 1 + type_params(&generics.1.params).count();
        let (self_ty, args) = substs.get(..count)?.split_first()?;
        let printed = self.ty(cx, *self_ty)?;
        let bound = self.named(cx, of, args)?;
        Some((format!("<{printed} as {bound}>::{}", cx.name(item)), count))
    }
}

/// Whether `ty` is the global allocator, which a type argument list leaves to its stable default.
fn allocator(cx: &Context<'_>, ty: Ty) -> bool {
    let Kind::Adt(item, _) = *cx.types.kind(ty) else { return false };
    cx.outline.roots[item.krate as usize].name == ALLOCATOR.0 && cx.name(item) == ALLOCATOR.1
}

/// Whether function `item` takes `impl Trait` arguments, whose calls take no turbofish.
fn impl_trait_params(cx: &mut Context<'_>, item: ItemRef) -> bool {
    let Ok(signature) = cx.signature(item) else { return true };
    let source = &cx.file(item).source;
    signature.generics.iter().any(|generic| source.text(generic.name) == IMPL_TRAIT)
}
