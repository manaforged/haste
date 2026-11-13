use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::bail;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::Deserialize;

use crate::graph::Workspace;
use crate::names::collect::{items_of, namespaces};
use crate::names::{ItemRef, plain};
use crate::outline::Outline;
use crate::outline::item::ItemKind;
use crate::outline::source::{Mark, Source};
use crate::outline::tree::Crate;
use crate::sema::Context;
use crate::sema::impls::ImplRef;
use crate::sema::infer::{Shape, Typed, detached};
use crate::sema::reach::Instance;
use crate::sema::ty::{Kind, Mut, Ty};

use super::edit::Snapshot;
use super::form;
use super::ident;
use super::source::{crate_key, normalize};

const BYTE_ORDER_MARK: char = '\u{feff}';

/// Where a definition is: its file from the parent of its last `src` directory on, and the byte offset of its name there.
#[derive(Deserialize, Clone, PartialEq, Eq, Debug)]
pub(super) struct Place {
    pub file: String,
    pub lo: u32,
}

/// Why an instance or a type has no key: a part of it has none.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Miss {
    Unnamed,
}

/// What naming the definitions of a build needs: each crate's name, whether it is local, and whether another package's crate shares its name; the snapshot prep took, by which a local definition is named where it was then; each file on disk as a key names it, once read; and per crate, once worked out, the places on disk that more than one definition's name came from and the module keys more than one item has.
pub(super) struct Keys {
    names: Vec<String>,
    local: Vec<bool>,
    doubled: Vec<bool>,
    then: Snapshot,
    files: Mutex<HashMap<PathBuf, Option<Disk>>>,
    copied: Mutex<HashMap<u32, HashSet<(PathBuf, u32)>>>,
    crowded: Mutex<HashMap<u32, HashSet<String>>>,
}

impl Keys {
    /// The keys of the crates of `outline`, naming local definitions as the snapshot `then` placed them.
    pub(super) fn new(outline: &Outline, workspace: &Workspace, then: Snapshot) -> Keys {
        let package = |krate: usize| workspace.graph.units[outline.roots[krate].unit].package;
        let names: Vec<String> = (0..outline.roots.len()).map(|krate| crate_key(&outline.roots[krate].name.replace('-', "_"), outline.package(workspace, krate).sysroot)).collect();
        let doubled = (0..names.len()).map(|krate| (0..names.len()).any(|other| names[other] == names[krate] && package(other) != package(krate))).collect();
        let local = (0..names.len()).map(|krate| outline.package(workspace, krate).local()).collect();
        Keys { names, local, doubled, then, files: Mutex::default(), copied: Mutex::default(), crowded: Mutex::default() }
    }

    /// The key of `instance` with body `typed`, and for a closure's call whether it calls by value.
    pub(super) fn instance(&self, cx: &mut Context<'_>, (item, substs): Instance, typed: &Typed) -> Result<(String, Option<bool>), Miss> {
        let substs = cx.types.substs(substs).to_vec();
        match &typed.shape {
            Shape::Closure(_) => {
                let once = ident::once(cx, item).ok_or(Miss::Unnamed)?;
                Ok((self.ty(cx, *substs.first().ok_or(Miss::Unnamed)?)?, Some(once)))
            }
            Shape::Body if cx.item(item).1.kind == ItemKind::Fn => Ok((form::named(&self.function(cx, item)?, &self.list(cx, &substs)?), None)),
            _ => Err(Miss::Unnamed),
        }
    }

    /// The crates whose key names them `name`.
    pub(super) fn crates_named<'n>(&'n self, name: &'n str) -> impl Iterator<Item = usize> + 'n {
        (0..self.names.len()).filter(move |&at| self.names[at] == name)
    }

    /// `ty` as a key.
    pub(super) fn ty(&self, cx: &mut Context<'_>, ty: Ty) -> Result<String, Miss> {
        let kind = cx.types.kind(ty).clone();
        match kind {
            Kind::Prim(primitive) => Ok(format!("{primitive:?}").to_lowercase()),
            Kind::Never => Ok(form::NEVER.to_owned()),
            Kind::Tuple(..) | Kind::Ref(..) | Kind::Ptr(..) | Kind::Array(..) | Kind::Slice(_) => self.built(cx, kind),
            _ => self.named(cx, ty, kind),
        }
    }

    /// The key of a tuple, reference, raw pointer, array or slice type of kind `kind`, built from the keys of its parts.
    fn built(&self, cx: &mut Context<'_>, kind: Kind) -> Result<String, Miss> {
        Ok(match kind {
            Kind::Tuple(items) => form::tuple(&self.types(cx, &cx.types.substs(items).to_vec())?),
            Kind::Ref(mutable, inner) => form::reference(mutable == Mut::Mut, &self.ty(cx, inner)?),
            Kind::Ptr(mutable, inner) => form::raw(mutable == Mut::Mut, &self.ty(cx, inner)?),
            Kind::Array(inner, len) => form::array(&self.ty(cx, inner)?, &self.arg(cx, len)?),
            Kind::Slice(inner) => form::slice(&self.ty(cx, inner)?),
            _ => return Err(Miss::Unnamed),
        })
    }

    /// The key of type `ty` of kind `kind` that a definition names: a struct, enum or union, a function item, a closure, a function pointer or a trait object.
    fn named(&self, cx: &mut Context<'_>, ty: Ty, kind: Kind) -> Result<String, Miss> {
        Ok(match kind {
            Kind::Adt(item, args) => form::named(&self.def(cx, item)?, &self.list(cx, &cx.types.substs(args).to_vec())?),
            Kind::FnDef(item, args) => form::fn_def(&self.function(cx, item)?, &self.list(cx, &cx.types.substs(args).to_vec())?),
            Kind::Closure(..) => self.closure(cx, ty)?,
            Kind::FnPtr(params, ret) => form::fn_ptr(&self.types(cx, &cx.types.substs(params).to_vec())?, &self.ty(cx, ret)?),
            Kind::Dyn(of, args, members, _) => self.object(cx, of, (&cx.types.substs(args).to_vec(), &members))?,
            _ => return Err(Miss::Unnamed),
        })
    }

    /// The key of a trait object type of trait `of` with arguments `args`, the last of which give the associated types `members` fixes.
    fn object(&self, cx: &mut Context<'_>, of: ItemRef, (args, members): (&[Ty], &[ItemRef])) -> Result<String, Miss> {
        let (given, bound) = crate::sema::ty::object_parts(args, members);
        let (given, bound): (Vec<Ty>, Vec<(ItemRef, Ty)>) = (given.to_vec(), bound.collect());
        let mut fixed = Vec::new();
        for (member, ty) in bound {
            fixed.push((cx.name(member).to_owned(), self.ty(cx, ty)?));
        }
        Ok(form::object(&self.def(cx, of)?, &self.list(cx, &given)?, &fixed))
    }

    fn closure(&self, cx: &mut Context<'_>, ty: Ty) -> Result<String, Miss> {
        let at = ident::closure(cx, ty).ok_or(Miss::Unnamed)?;
        Ok(form::closure(&self.function(cx, at.root)?, (at.order, at.count), &self.list(cx, &at.outer)?))
    }

    fn arg(&self, cx: &mut Context<'_>, ty: Ty) -> Result<String, Miss> {
        match *cx.types.kind(ty) {
            Kind::Value(bits) => Ok(bits.to_string()),
            _ => self.ty(cx, ty),
        }
    }

    fn types(&self, cx: &mut Context<'_>, items: &[Ty]) -> Result<Vec<String>, Miss> {
        items.iter().map(|&item| self.ty(cx, item)).collect()
    }

    fn list(&self, cx: &mut Context<'_>, items: &[Ty]) -> Result<Vec<String>, Miss> {
        items.iter().map(|&item| self.arg(cx, item)).collect()
    }

    /// The key of definition `item`: its crate, file, the byte offset of its name and the name, where prep saw it for a local definition; else `written`'s.
    pub(super) fn def(&self, cx: &mut Context<'_>, item: ItemRef) -> Result<String, Miss> {
        let krate = item.krate as usize;
        if self.doubled[krate] {
            return Err(Miss::Unnamed);
        }
        let place = if self.local[krate] {
            ident::identity(cx.outline, item).and_then(|identity| self.then.place(&identity).cloned())
        } else {
            self.place(cx, item)
        };
        match place {
            Some(place) => Ok(format!("{}:{}:{}:{}", self.names[krate], place.file, place.lo, cx.name(item))),
            None if !self.local[krate] => self.written(cx, item).ok_or(Miss::Unnamed),
            None => Err(Miss::Unnamed),
        }
    }

    /// The key of function `item`: `def`'s, or for a method of an impl of a crate no edit changes whose name a macro wrote, `member`'s.
    pub(super) fn function(&self, cx: &mut Context<'_>, item: ItemRef) -> Result<String, Miss> {
        if let Ok(key) = self.def(cx, item) {
            return Ok(key);
        }
        let (source, entry) = cx.item(item);
        if !entry.name.is_some_and(|name| written_plain(source, name)) || item.outer().is_some() || self.local[item.krate as usize] || self.doubled[item.krate as usize] {
            return Err(Miss::Unnamed);
        }
        let block = cx.owner_impl(item).ok_or(Miss::Unnamed)?;
        self.member(cx, item, block).ok_or(Miss::Unnamed)
    }

    /// The key of definition `item` of a crate no edit changes whose name a macro wrote, when no `macro` item's hygiene keeps that name apart, by the module path, for one a module holds when no other item a module holds has that key.
    fn written(&self, cx: &mut Context<'_>, item: ItemRef) -> Option<String> {
        let (source, entry) = cx.item(item);
        let name = entry.name?;
        if item.outer().is_some() || !written_plain(source, name) {
            return None;
        }
        let file = cx.file(item);
        let in_module = entry.parent.is_none_or(|parent| file.items.items[parent as usize].kind == ItemKind::Mod);
        if !in_module || cx.owner_impl(item).is_some() || cx.owner_trait(item).is_some() || cx.foreign(item) {
            return None;
        }
        let module = cx.module_of(item).ok()?.module;
        let key = form::in_module(&self.names[item.krate as usize], &plain_path(&cx.outline.crates[item.krate as usize], module as usize), plain(source.text(name)));
        (!self.crowded(cx, item.krate, &key)?).then_some(key)
    }

    /// The key of member `item` of impl `block`, which takes no type arguments, by the key of the trait it implements with that trait's arguments and the key of its self type.
    fn member(&self, cx: &mut Context<'_>, item: ItemRef, block: ImplRef) -> Option<String> {
        let of = cx.impl_trait(block);
        let (self_ty, args) = detached(cx, item, |infer| {
            if infer.impl_params(block)? != 0 {
                bail!("a generic impl");
            }
            let (_, self_ty) = infer.impl_frame(block, &[])?;
            let args = if of.is_some() { infer.impl_trait_args(block, &[], self_ty)? } else { Vec::new() };
            Ok((self_ty, args))
        })
        .ok()?;
        let self_ty = self.ty(cx, self_ty).ok()?;
        let of = match of {
            Some(of) => Some(form::named(&self.def(cx, of).ok()?, &self.list(cx, &args).ok()?)),
            None => None,
        };
        form::member(&self.names[item.krate as usize], of.as_deref(), &self_ty, cx.name(item))
    }

    /// Whether more than one item that a module of crate `krate` holds has the module key `key`, as two modules of one path may hold when hygiene keeps their names apart.
    fn crowded(&self, cx: &Context<'_>, krate: u32, key: &str) -> Option<bool> {
        let mut crowded = self.crowded.lock().ok()?;
        let found = crowded.entry(krate).or_insert_with(|| {
            let tree = &cx.outline.crates[krate as usize];
            let (mut seen, mut twice) = (HashSet::default(), HashSet::default());
            for module in 0..tree.modules.len() {
                let path = plain_path(tree, module);
                for (file, at) in items_of(tree, module) {
                    let file = &tree.files[file as usize];
                    let entry = &file.items.items[at as usize];
                    let Some(name) = entry.name.filter(|_| !namespaces(entry).is_empty()) else { continue };
                    let named = form::in_module(&self.names[krate as usize], &path, plain(file.source.text(name)));
                    if !seen.insert(named.clone()) {
                        twice.insert(named);
                    }
                }
            }
            twice
        });
        Some(found.contains(key))
    }

    /// Where definition `item` of a crate no edit changes is: where its name lies in the file on disk, or where the token a macro copied it from lies, when no other definition of the crate took its name from that token.
    fn place(&self, cx: &Context<'_>, item: ItemRef) -> Option<Place> {
        if item.outer().is_some() {
            return None;
        }
        let source = &cx.file(item).source;
        let name = cx.item(item).1.name?;
        let lo = source.origin(name)?;
        if self.copied_twice(cx, item.krate, (&source.path, lo))? {
            return None;
        }
        let mut files = self.files.lock().ok()?;
        let disk = files.entry(source.path.clone()).or_insert_with(|| Disk::read(&source.path)).as_ref()?;
        (lo as usize + source.text(name).len() <= disk.len).then(|| Place { file: disk.name.clone(), lo: disk.offset(lo) })
    }

    /// Whether the names of more than one definition of crate `krate` lie at, or were copied by macros from, the token at byte `lo` of file `path`.
    fn copied_twice(&self, cx: &Context<'_>, krate: u32, (path, lo): (&Path, u32)) -> Option<bool> {
        let mut copied = self.copied.lock().ok()?;
        let found = copied.entry(krate).or_insert_with(|| {
            let (mut seen, mut twice) = (HashSet::default(), HashSet::default());
            for file in &cx.outline.crates[krate as usize].files {
                for name in file.items.items.iter().filter_map(|entry| entry.name) {
                    let Some(origin) = file.source.origin(name) else { continue };
                    if !seen.insert((file.source.path.clone(), origin)) {
                        twice.insert((file.source.path.clone(), origin));
                    }
                }
            }
            twice
        });
        Some(found.contains(&(path.to_path_buf(), lo)))
    }
}

/// The name a key gives file `path` and the length of its text on disk, when `text` begins with that text and rustc counts byte offsets in it as Haste does.
pub(super) fn on_disk(path: &Path, text: &str) -> Option<(String, usize)> {
    let read = std::fs::read(path).ok()?;
    (clean(text) && text.as_bytes().starts_with(&read)).then_some(())?;
    Some((normalize(path)?, read.len()))
}

/// The names of module `module` of `tree` and its parents below the crate root, as written, outermost first.
pub(super) fn module_path(tree: &Crate, module: usize) -> Vec<String> {
    let mut names = Vec::new();
    let mut at = module;
    while let Some(parent) = tree.modules[at].parent {
        names.push(tree.modules[at].name.clone());
        at = parent;
    }
    names.reverse();
    names
}

/// `module_path` as rustc names the modules, without raw identifier prefixes.
fn plain_path(tree: &Crate, module: usize) -> Vec<String> {
    module_path(tree, module).iter().map(|name| plain(name).to_owned()).collect()
}

/// Whether a macro wrote name token `name` of `source`, copied from no file on disk, with no `macro` item's hygiene to keep it apart.
fn written_plain(source: &Source, name: u32) -> bool {
    source.origin(name).is_none() && !matches!(source.mark(name), Some(Mark::Home { .. }))
}

/// A file on disk as a key names it: its name, its length, and what rustc drops from its text before it counts byte offsets: the length of a byte order mark, and the offsets of the carriage returns before line feeds.
struct Disk {
    name: String,
    len: usize,
    mark: u32,
    returns: Vec<u32>,
}

impl Disk {
    /// The file at `path`, when a key can name it.
    fn read(path: &Path) -> Option<Disk> {
        let text = std::fs::read_to_string(path).ok()?;
        let mark = if text.starts_with(BYTE_ORDER_MARK) { BYTE_ORDER_MARK.len_utf8() as u32 } else { 0 };
        let returns = text.match_indices("\r\n").map(|(at, _)| at as u32).collect();
        Some(Disk { name: normalize(path)?, len: text.len(), mark, returns })
    }

    /// The byte offset rustc counts for byte `lo` of the file's text.
    fn offset(&self, lo: u32) -> u32 {
        lo - self.mark - self.returns.partition_point(|&at| at < lo) as u32
    }
}

/// Whether byte offsets in `text` are those rustc counts, which it counts after dropping a byte order mark and carriage returns.
fn clean(text: &str) -> bool {
    !text.starts_with(BYTE_ORDER_MARK) && !text.contains('\r')
}
