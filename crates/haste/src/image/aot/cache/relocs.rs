use std::hash::{Hash as _, Hasher as _};
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::host::{HostSymbol, MathSymbol, X86Symbol};
use crate::lower::reloc::{RelocPlan, RelocStep, SiteOrigin};
use crate::lower::{Slots, SymbolSource};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::Context as SemaContext;
use crate::sema::infer::{Res, Shape, Typed};
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::summary::codec::{Cursor, read_str, write_str, write_u32, write_u64};
use crate::substrate::file_key;

use super::codec::Stamps;

#[derive(Default)]
pub(super) struct LocalFileIndex {
    files: HashMap<u64, (u32, u32)>,
}

impl LocalFileIndex {
    pub(super) fn build(outline: &Outline) -> Self {
        let mut files = HashMap::default();
        for (krate, tree) in outline.crates.iter().enumerate() {
            for file in 0..tree.files.len() as u32 {
                if let Some(key) = stable_file_key(outline, krate as u32, file) {
                    files.entry(key).or_insert((krate as u32, file));
                }
            }
        }
        Self { files }
    }

    pub(super) fn resolve(
        &self,
        outline: &Outline,
        stamps: &Stamps,
        record: LocalItemRecord,
    ) -> Option<ItemRef> {
        let &(krate, file) = self.files.get(&record.file)?;
        let item = ItemRef {
            krate,
            file,
            item: record.item,
        };
        (item.outer().is_none()
            && item_check_key(outline, stamps, record.file, item)? == record.key)
            .then_some(item)
    }
}

#[derive(Clone, Copy)]
pub(super) struct LocalItemRecord {
    file: u64,
    item: u32,
    key: u64,
}

impl LocalItemRecord {
    pub(super) fn of(outline: &Outline, stamps: &Stamps, item: ItemRef) -> Option<Self> {
        let file = stable_file_key(outline, item.krate, item.file)?;
        let key = item_check_key(outline, stamps, file, item)?;
        Some(Self {
            file,
            item: item.item,
            key,
        })
    }

    pub(super) fn write(self, out: &mut Vec<u8>) {
        write_u64(out, self.file);
        write_u32(out, self.item);
        write_u64(out, self.key);
    }

    pub(super) fn read(cur: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            file: cur.u64()?,
            item: cur.u32()?,
            key: cur.u64()?,
        })
    }
}

pub(super) fn hash_str(s: &str) -> u64 {
    let mut h = FxHasher::default();
    s.hash(&mut h);
    h.finish()
}

pub(super) fn stable_file_key(outline: &Outline, krate: u32, file: u32) -> Option<u64> {
    let tree = outline.crates.get(krate as usize)?;
    let root = outline.roots.get(krate as usize)?;
    if tree.env.fixed {
        let base = file_key(outline, krate, file)?;
        let pin = outline.pins.get(krate as usize).copied().flatten();
        let mut h = FxHasher::default();
        (base, pin, &root.path, root.role as u8).hash(&mut h);
        return Some(h.finish());
    }
    let entry = tree.files.get(file as usize)?;
    if entry.expanded && entry.expand_origin == 0 {
        return None;
    }
    let base_dir = root.path.parent().unwrap_or(Path::new(""));
    let rel_path = entry
        .source
        .path
        .strip_prefix(base_dir)
        .unwrap_or(&entry.source.path);
    let mut h = FxHasher::default();
    0xA07F_11E0_u64.hash(&mut h);
    root.name.hash(&mut h);
    (root.role as u8).hash(&mut h);
    root.path.hash(&mut h);
    tree.env.target.triple.hash(&mut h);
    tree.env.test.hash(&mut h);
    tree.env.features.hash(&mut h);
    rel_path.hash(&mut h);
    entry.expanded.hash(&mut h);
    if entry.expanded {
        entry.expand_origin.hash(&mut h);
        entry.source.offset.hash(&mut h);
        hash_str(&entry.source.text).hash(&mut h);
    }
    Some(h.finish())
}

pub(super) fn item_check_key(
    outline: &Outline,
    stamps: &Stamps,
    file_digest: u64,
    item: ItemRef,
) -> Option<u64> {
    let krate = outline.crates.get(item.krate as usize)?;
    let file = krate.files.get(item.file as usize)?;
    let (base_item, sub_idx) = match item.outer() {
        Some((outer, sub)) => (outer.item, Some(sub)),
        None => (item.item, None),
    };
    let entry = file.items.items.get(base_item as usize)?;
    let mut h = FxHasher::default();
    stamps.iface(item.krate)?.hash(&mut h);
    file_digest.hash(&mut h);
    item.item.hash(&mut h);
    sub_idx.hash(&mut h);
    if krate.env.fixed {
        entry.start.hash(&mut h);
    }
    entry.kind.hash(&mut h);
    if let Some(name_tok) = entry.name {
        file.source.text(name_tok).hash(&mut h);
    }
    Some(h.finish())
}

struct InlineWalk<'a, 'b> {
    cx: &'a SemaContext<'b>,
    reached: &'a Reached,
    relocs: &'a [SymbolSource],
    visited: HashSet<usize>,
    out: Vec<(u32, u32)>,
}

impl InlineWalk<'_, '_> {
    fn visit_plan(&mut self, plan: &RelocPlan, index: usize, base: usize) {
        let mut pos = base;
        for step in &plan.steps {
            let len = step.symbols();
            if let RelocStep::SiteInline {
                origin,
                plan: sub_plan,
            } = step
            {
                if let Some(callee_idx) =
                    resolve_inline_origin(self.reached, index, self.relocs, base, origin)
                {
                    if let Some(((c_item, _), _)) = self.reached.list.get(callee_idx)
                        && !self.cx.outline.is_fixed(c_item.krate)
                    {
                        self.out.push((c_item.krate, c_item.body_owner().file));
                    }
                    if self.visited.insert(callee_idx) {
                        self.visit_plan(sub_plan, callee_idx, pos);
                    }
                } else {
                    self.record_fallback(origin, sub_plan, index, pos, len);
                }
            }
            pos += len;
        }
    }

    fn record_fallback(
        &mut self,
        origin: &SiteOrigin,
        sub_plan: &RelocPlan,
        index: usize,
        pos: usize,
        len: usize,
    ) {
        for sym in self.relocs.get(pos..pos + len).into_iter().flatten() {
            if let SymbolSource::Slot(c_idx) = *sym
                && let Some(((c_item, _), _)) = self.reached.list.get(c_idx)
                && !self.cx.outline.is_fixed(c_item.krate)
            {
                self.out.push((c_item.krate, c_item.body_owner().file));
            }
        }
        let Some(typed) = self
            .reached
            .list
            .get(index)
            .and_then(|(_, r)| r.as_ref().ok())
        else {
            return;
        };
        match origin {
            SiteOrigin::DropGlue(_) | SiteOrigin::DropFn(_) => {
                if matches!(origin, SiteOrigin::DropGlue(_)) && !plan_has_inline(sub_plan) {
                    return;
                }
                for res in &typed.drops {
                    if let Res::Fn(item, substs) = *res {
                        if !self.cx.outline.is_fixed(item.krate) {
                            self.out.push((item.krate, item.body_owner().file));
                        }
                        if let Some(&idx) = self.reached.index.get(&(item, substs))
                            && self.visited.insert(idx)
                        {
                            self.record_direct_local_callees(idx);
                        }
                    }
                }
            }
            SiteOrigin::PointerFnDef { .. }
            | SiteOrigin::ConstEvalSelect(_)
            | SiteOrigin::VtableMethod { .. } => {
                self.record_direct_local_callees(index);
            }
            _ => {}
        }
    }

    fn record_direct_local_callees(&mut self, root: usize) {
        for &v in self.reached.direct_calls(root).into_iter().flatten() {
            if let Some(((item, _), _)) = self.reached.list.get(v)
                && !self.cx.outline.is_fixed(item.krate)
            {
                self.out.push((item.krate, item.body_owner().file));
            }
        }
    }
}

fn plan_has_inline(plan: &RelocPlan) -> bool {
    plan.steps
        .iter()
        .any(|s| matches!(s, RelocStep::SiteInline { .. }))
}

pub(super) fn collect_inlined_local_files(
    cx: &SemaContext<'_>,
    reached: &Reached,
    plan: &RelocPlan,
    index: usize,
    relocs: &[SymbolSource],
) -> Vec<(u32, u32)> {
    let owner = reached
        .list
        .get(index)
        .map(|((item, _), _)| (item.krate, item.body_owner().file));
    let mut walk = InlineWalk {
        cx,
        reached,
        relocs,
        visited: HashSet::default(),
        out: Vec::new(),
    };
    walk.visit_plan(plan, index, 0);
    let mut files = walk.out;
    if let Some(own) = owner {
        files.retain(|&kf| kf != own);
    }
    files.sort_unstable();
    files.dedup();
    files
}

fn resolve_inline_origin(
    reached: &Reached,
    index: usize,
    relocs: &[SymbolSource],
    base: usize,
    origin: &SiteOrigin,
) -> Option<usize> {
    if let SiteOrigin::Symbol(at) = origin {
        return match relocs.get(base + *at as usize) {
            Some(&SymbolSource::Slot(idx)) => Some(idx),
            _ => None,
        };
    }
    let typed = reached.list.get(index)?.1.as_ref().ok()?;
    let (item, substs) = resolve_site_from_typed(typed, origin)?;
    reached.index.get(&(item, substs)).copied()
}

fn resolve_site_from_typed(typed: &Typed, origin: &SiteOrigin) -> Option<(ItemRef, Substs)> {
    match origin {
        SiteOrigin::Table(key) => typed.site_fn(*key),
        SiteOrigin::Known(item, substs) => Some((*item, *substs)),
        SiteOrigin::ShapeGlueDrop => match &typed.shape {
            Shape::Glue(glue) => match glue.drop {
                Some(Res::Fn(item, substs)) => Some((item, substs)),
                _ => None,
            },
            _ => None,
        },
        SiteOrigin::ShapeGlueResume => match &typed.shape {
            Shape::Glue(glue) => match glue.resume {
                Some(Res::Fn(item, substs)) => Some((item, substs)),
                _ => None,
            },
            _ => None,
        },
        SiteOrigin::ShapeUntuple => match &typed.shape {
            Shape::Untuple(Res::Fn(item, substs)) => Some((*item, *substs)),
            _ => None,
        },
        SiteOrigin::ShapeClone(idx) => match &typed.shape {
            Shape::Clone(calls) => match calls.get(*idx as usize) {
                Some(Res::Fn(item, substs)) => Some((*item, *substs)),
                _ => None,
            },
            _ => None,
        },
        SiteOrigin::DropGlue(_)
        | SiteOrigin::PointerFnDef { .. }
        | SiteOrigin::DropFn(_)
        | SiteOrigin::ConstEvalSelect(_)
        | SiteOrigin::VtableMethod { .. }
        | SiteOrigin::Symbol(_) => None,
    }
}

pub(super) fn resolve_or_want(
    reached: &Reached,
    slots: &dyn Slots,
    instance: (ItemRef, Substs),
) -> Option<usize> {
    reached
        .index
        .get(&instance)
        .copied()
        .filter(|&idx| !reached.is_untyped(idx))
        .or_else(|| {
            slots.want(instance);
            None
        })
}

pub(super) fn write_host(out: &mut Vec<u8>, host: &HostSymbol) -> Option<()> {
    match host {
        HostSymbol::Alloc => out.push(0),
        HostSymbol::AllocZeroed => out.push(1),
        HostSymbol::Dealloc => out.push(2),
        HostSymbol::Realloc => out.push(3),
        HostSymbol::Noop => out.push(4),
        HostSymbol::Barrier => out.push(5),
        HostSymbol::Cpuid => out.push(6),
        HostSymbol::ExtendedState => out.push(7),
        HostSymbol::DsoHandle => out.push(8),
        HostSymbol::Abort => out.push(9),
        HostSymbol::FillOnce => out.push(10),
        HostSymbol::ThreadCell => out.push(11),
        HostSymbol::Syscall => out.push(12),
        HostSymbol::Math(m) => {
            out.push(13);
            write_str(out, m.link_name())?;
        }
        HostSymbol::X86(x) => {
            out.push(14);
            write_str(out, x.link_name())?;
        }
        HostSymbol::Dynamic { name, lib } => {
            out.push(15);
            write_str(out, name)?;
            write_str(out, lib.as_deref().unwrap_or(""))?;
        }
        HostSymbol::CatchUnwind => out.push(16),
        HostSymbol::StartPanic => out.push(17),
        HostSymbol::PanicCleanup => out.push(18),
        HostSymbol::DropPush => out.push(19),
        HostSymbol::DropPop => out.push(20),
    }
    Some(())
}

pub(super) fn read_host(cur: &mut Cursor<'_>) -> Result<HostSymbol> {
    Ok(match cur.u8()? {
        0 => HostSymbol::Alloc,
        1 => HostSymbol::AllocZeroed,
        2 => HostSymbol::Dealloc,
        3 => HostSymbol::Realloc,
        4 => HostSymbol::Noop,
        5 => HostSymbol::Barrier,
        6 => HostSymbol::Cpuid,
        7 => HostSymbol::ExtendedState,
        8 => HostSymbol::DsoHandle,
        9 => HostSymbol::Abort,
        10 => HostSymbol::FillOnce,
        11 => HostSymbol::ThreadCell,
        12 => HostSymbol::Syscall,
        13 => {
            HostSymbol::Math(MathSymbol::from_name(read_str(cur)?).context("unknown math symbol")?)
        }
        14 => HostSymbol::X86(X86Symbol::from_link(read_str(cur)?).context("unknown x86 symbol")?),
        15 => {
            let name: Box<str> = read_str(cur)?.into();
            let raw_lib = read_str(cur)?;
            let lib = (!raw_lib.is_empty()).then(|| raw_lib.into());
            HostSymbol::Dynamic { name, lib }
        }
        16 => HostSymbol::CatchUnwind,
        17 => HostSymbol::StartPanic,
        18 => HostSymbol::PanicCleanup,
        19 => HostSymbol::DropPush,
        20 => HostSymbol::DropPop,
        _ => bail!("invalid host symbol tag"),
    })
}
