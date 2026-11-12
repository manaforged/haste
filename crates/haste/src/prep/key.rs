use std::sync::Mutex;

use rustc_hash::FxHashMap as HashMap;
use serde::Deserialize;

use crate::graph::Workspace;
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{Shape, Typed};
use crate::sema::reach::Instance;
use crate::sema::ty::{Kind, Mut, Ty};

use super::edit::Snapshot;
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

/// What naming the definitions of a build needs: each crate's name, whether it is local, and whether another package's crate shares its name; the snapshot prep took, by which a local definition is named where it was then; and each file's name once worked out.
pub(super) struct Keys {
    names: Vec<String>,
    local: Vec<bool>,
    doubled: Vec<bool>,
    then: Snapshot,
    files: Mutex<HashMap<(u32, u32), Option<String>>>,
}

impl Keys {
    /// The keys of the crates of `outline`, naming local definitions as the snapshot `then` placed them.
    pub(super) fn new(outline: &Outline, workspace: &Workspace, then: Snapshot) -> Keys {
        let package = |krate: usize| workspace.graph.units[outline.roots[krate].unit].package;
        let names: Vec<String> = (0..outline.roots.len()).map(|krate| crate_key(&outline.roots[krate].name.replace('-', "_"), outline.package(workspace, krate).sysroot)).collect();
        let doubled = (0..names.len()).map(|krate| (0..names.len()).any(|other| names[other] == names[krate] && package(other) != package(krate))).collect();
        let local = (0..names.len()).map(|krate| outline.package(workspace, krate).local()).collect();
        Keys { names, local, doubled, then, files: Mutex::default() }
    }

    /// The key of `instance` with body `typed`, and for a closure's call whether it calls by value.
    pub(super) fn instance(&self, cx: &mut Context<'_>, (item, substs): Instance, typed: &Typed) -> Result<(String, Option<bool>), Miss> {
        let substs = cx.types.substs(substs).to_vec();
        match &typed.shape {
            Shape::Closure(_) => {
                let once = ident::once(cx, item).ok_or(Miss::Unnamed)?;
                Ok((self.ty(cx, *substs.first().ok_or(Miss::Unnamed)?)?, Some(once)))
            }
            Shape::Body if cx.item(item).1.kind == ItemKind::Fn => Ok((format!("{}<{}>", self.def(cx, item)?, self.list(cx, &substs)?), None)),
            _ => Err(Miss::Unnamed),
        }
    }

    /// `ty` as a key.
    pub(super) fn ty(&self, cx: &mut Context<'_>, ty: Ty) -> Result<String, Miss> {
        Ok(match cx.types.kind(ty).clone() {
            Kind::Prim(primitive) => format!("{primitive:?}").to_lowercase(),
            Kind::Never => "!".to_owned(),
            Kind::Tuple(items) => {
                let items = cx.types.substs(items).to_vec();
                format!("({})", items.iter().map(|&item| self.ty(cx, item).map(|key| key + ",")).collect::<Result<String, Miss>>()?)
            }
            Kind::Ref(Mut::Not, inner) => format!("&{}", self.ty(cx, inner)?),
            Kind::Ref(Mut::Mut, inner) => format!("&mut {}", self.ty(cx, inner)?),
            Kind::Ptr(Mut::Not, inner) => format!("*const {}", self.ty(cx, inner)?),
            Kind::Ptr(Mut::Mut, inner) => format!("*mut {}", self.ty(cx, inner)?),
            Kind::Array(inner, len) => format!("[{};{}]", self.ty(cx, inner)?, self.arg(cx, len)?),
            Kind::Slice(inner) => format!("[{}]", self.ty(cx, inner)?),
            Kind::Adt(item, args) => format!("{}<{}>", self.def(cx, item)?, self.list(cx, &cx.types.substs(args).to_vec())?),
            Kind::FnDef(item, args) => format!("fn {}<{}>", self.def(cx, item)?, self.list(cx, &cx.types.substs(args).to_vec())?),
            Kind::Closure(..) => self.closure(cx, ty)?,
            Kind::FnPtr(params, ret) => {
                let params = cx.types.substs(params).to_vec();
                format!("fn({})->{}", params.iter().map(|&param| self.ty(cx, param).map(|key| key + ",")).collect::<Result<String, Miss>>()?, self.ty(cx, ret)?)
            }
            Kind::Dyn(of, args, members, _) => self.object(cx, of, (&cx.types.substs(args).to_vec(), &members))?,
            _ => return Err(Miss::Unnamed),
        })
    }

    /// The key of a trait object type of trait `of` with arguments `args`, the last of which give the associated types `members` fixes, as the driver prints one: those in name order.
    fn object(&self, cx: &mut Context<'_>, of: ItemRef, (args, members): (&[Ty], &[ItemRef])) -> Result<String, Miss> {
        let (given, bound) = crate::sema::ty::object_parts(args, members);
        let (given, bound): (Vec<Ty>, Vec<(ItemRef, Ty)>) = (given.to_vec(), bound.collect());
        let mut fixed = Vec::new();
        for (member, ty) in bound {
            fixed.push(format!("+{}={}", cx.name(member), self.ty(cx, ty)?));
        }
        fixed.sort();
        Ok(format!("dyn {}<{}>{}", self.def(cx, of)?, self.list(cx, &given)?, fixed.concat()))
    }

    fn closure(&self, cx: &mut Context<'_>, ty: Ty) -> Result<String, Miss> {
        let at = ident::closure(cx, ty).ok_or(Miss::Unnamed)?;
        Ok(format!("closure {}#{}/{}<{}>", self.def(cx, at.root)?, at.order, at.count, self.list(cx, &at.outer)?))
    }

    fn arg(&self, cx: &mut Context<'_>, ty: Ty) -> Result<String, Miss> {
        match *cx.types.kind(ty) {
            Kind::Value(bits) => Ok(bits.to_string()),
            _ => self.ty(cx, ty),
        }
    }

    fn list(&self, cx: &mut Context<'_>, items: &[Ty]) -> Result<String, Miss> {
        let keys: Result<Vec<String>, Miss> = items.iter().map(|&item| self.arg(cx, item)).collect();
        Ok(keys?.join(","))
    }

    /// The key of definition `item`: its crate, file, the byte offset of its name and the name, where prep saw it for a local definition.
    pub(super) fn def(&self, cx: &Context<'_>, item: ItemRef) -> Result<String, Miss> {
        let krate = item.krate as usize;
        if self.doubled[krate] {
            return Err(Miss::Unnamed);
        }
        let place = if self.local[krate] {
            ident::identity(cx.outline, item).and_then(|identity| self.then.place(&identity).cloned())
        } else {
            self.place(cx, item)
        };
        let place = place.ok_or(Miss::Unnamed)?;
        Ok(format!("{}:{}:{}:{}", self.names[krate], place.file, place.lo, cx.name(item)))
    }

    /// Where definition `item` of a crate no edit changes is, when its name is in its file's own text.
    fn place(&self, cx: &Context<'_>, item: ItemRef) -> Option<Place> {
        if item.outer().is_some() {
            return None;
        }
        let file = cx.file(item);
        let name = cx.item(item).1.name?;
        let (text, span) = (file.source.text.as_str(), file.source.span(name, name));
        let lo = (span.as_ptr() as usize).checked_sub(text.as_ptr() as usize)?;
        (file.source.base.is_none() && lo + span.len() <= text.len()).then_some(())?;
        let mut files = self.files.lock().ok()?;
        let named = files.entry((item.krate, item.file)).or_insert_with(|| clean(text).then(|| normalize(&file.source.path)).flatten());
        Some(Place { file: named.clone()?, lo: lo as u32 })
    }
}

/// Whether byte offsets in `text` are those rustc counts, which it counts after dropping a byte order mark and carriage returns.
pub(super) fn clean(text: &str) -> bool {
    !text.starts_with(BYTE_ORDER_MARK) && !text.contains('\r')
}
