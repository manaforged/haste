use std::hash::{Hash as _, Hasher as _};
use std::path::Path;
use std::sync::{Arc, OnceLock};

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
pub(crate) struct LocalFileIndex {
    files: HashMap<u64, (u32, u32)>,
}

impl LocalFileIndex {
    pub(crate) fn build(outline: &Outline) -> Self {
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

    pub(crate) fn resolve(
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
pub(crate) struct LocalItemRecord {
    file: u64,
    item: u32,
    key: u64,
}

impl LocalItemRecord {
    pub(crate) fn of(outline: &Outline, stamps: &Stamps, item: ItemRef) -> Option<Self> {
        let file = stable_file_key(outline, item.krate, item.file)?;
        let key = item_check_key(outline, stamps, file, item)?;
        Some(Self {
            file,
            item: item.item,
            key,
        })
    }

    pub(crate) fn write(self, out: &mut Vec<u8>) {
        write_u64(out, self.file);
        write_u32(out, self.item);
        write_u64(out, self.key);
    }

    pub(crate) fn read(cur: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            file: cur.u64()?,
            item: cur.u32()?,
            key: cur.u64()?,
        })
    }
}

pub(crate) use crate::graph::store::hash_str;

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

pub(super) struct TransitiveLocalFiles {
    files: Vec<(u32, u32)>,
    words: usize,
    scc_of: Vec<u32>,
    scc_bits: Vec<u64>,
}

impl TransitiveLocalFiles {
    pub(super) fn build(cx: &SemaContext<'_>, reached: &Reached) -> Self {
        let n = reached.list.len();
        let mut files: Vec<(u32, u32)> = Vec::new();
        let mut file_id: HashMap<(u32, u32), u32> = HashMap::default();
        let mut slot_file = vec![u32::MAX; n];
        for (i, ((item, _), _)) in reached.list.iter().enumerate() {
            if !cx.outline.is_fixed(item.krate) {
                let kf = (item.krate, item.body_owner().file);
                let id = *file_id.entry(kf).or_insert_with(|| {
                    let next = files.len() as u32;
                    files.push(kf);
                    next
                });
                slot_file[i] = id;
            }
        }
        if files.is_empty() {
            return Self {
                files,
                words: 0,
                scc_of: Vec::new(),
                scc_bits: Vec::new(),
            };
        }
        let words = files.len().div_ceil(64);
        let (scc_of, scc_bits) = build_scc_bits(reached, &slot_file, words);
        Self {
            files,
            words,
            scc_of,
            scc_bits,
        }
    }

    fn push_callees(&self, reached: &Reached, root: usize, out: &mut Vec<(u32, u32)>) {
        if self.words == 0 {
            return;
        }
        let Some(callees) = reached.direct_calls(root) else {
            return;
        };
        if callees.is_empty() {
            return;
        }
        if self.words == 1 {
            let mut bits = 0_u64;
            for &v in callees {
                if let Some(&c) = self.scc_of.get(v) {
                    bits |= self.scc_bits[c as usize];
                }
            }
            push_bits_word(bits, 0, &self.files, out);
            return;
        }
        let mut acc = vec![0_u64; self.words];
        for &v in callees {
            if let Some(&c) = self.scc_of.get(v) {
                let base = c as usize * self.words;
                for (dst, &src) in acc.iter_mut().zip(&self.scc_bits[base..base + self.words]) {
                    *dst |= src;
                }
            }
        }
        for (w, bits) in acc.into_iter().enumerate() {
            push_bits_word(bits, w * 64, &self.files, out);
        }
    }
}

fn push_bits_word(mut bits: u64, base_idx: usize, files: &[(u32, u32)], out: &mut Vec<(u32, u32)>) {
    while bits != 0 {
        let tz = bits.trailing_zeros() as usize;
        out.push(files[base_idx + tz]);
        bits &= bits - 1;
    }
}

fn finalize_scc(
    reached: &Reached,
    slot_file: &[u32],
    words: usize,
    members: &[usize],
    scc_id: u32,
    scc_of: &mut [u32],
    scc_bits: &mut Vec<u64>,
) {
    let base = scc_bits.len();
    scc_bits.resize(base + words, 0);
    for &u in members {
        scc_of[u] = scc_id;
    }
    for &u in members {
        let fid = slot_file[u];
        if fid != u32::MAX {
            let idx = fid as usize;
            scc_bits[base + (idx / 64)] |= 1_u64 << (idx % 64);
        }
        for &v in reached.direct_calls(u).into_iter().flatten() {
            if let Some(&cv) = scc_of.get(v)
                && cv != scc_id
            {
                let src_base = cv as usize * words;
                for w in 0..words {
                    scc_bits[base + w] |= scc_bits[src_base + w];
                }
            }
        }
    }
}

fn build_scc_bits(reached: &Reached, slot_file: &[u32], words: usize) -> (Vec<u32>, Vec<u64>) {
    let n = slot_file.len();
    let mut disc = vec![0_u32; n];
    let mut low = vec![0_u32; n];
    let mut on_stack = vec![false; n];
    let mut tarjan_stack: Vec<usize> = Vec::new();
    let mut call_stack: Vec<(usize, usize)> = Vec::new();
    let mut members: Vec<usize> = Vec::new();
    let mut scc_of = vec![u32::MAX; n];
    let mut scc_bits: Vec<u64> = Vec::new();
    let mut next_disc = 0_u32;
    let mut next_scc = 0_u32;

    for start in 0..n {
        if disc[start] != 0 {
            continue;
        }
        next_disc += 1;
        disc[start] = next_disc;
        low[start] = next_disc;
        on_stack[start] = true;
        tarjan_stack.push(start);
        call_stack.push((start, 0));

        while let Some((u, ei)) = call_stack.last_mut() {
            let u = *u;
            let edges = reached.direct_calls(u).unwrap_or(&[]);
            if *ei < edges.len() {
                let v = edges[*ei];
                *ei += 1;
                if v >= n {
                    continue;
                }
                if disc[v] == 0 {
                    next_disc += 1;
                    disc[v] = next_disc;
                    low[v] = next_disc;
                    on_stack[v] = true;
                    tarjan_stack.push(v);
                    call_stack.push((v, 0));
                } else if on_stack[v] && disc[v] < low[u] {
                    low[u] = disc[v];
                }
            } else {
                call_stack.pop();
                if let Some(&(parent, _)) = call_stack.last()
                    && low[u] < low[parent]
                {
                    low[parent] = low[u];
                }
                if low[u] == disc[u] {
                    members.clear();
                    while let Some(w) = tarjan_stack.pop() {
                        on_stack[w] = false;
                        members.push(w);
                        if w == u {
                            break;
                        }
                    }
                    finalize_scc(
                        reached,
                        slot_file,
                        words,
                        &members,
                        next_scc,
                        &mut scc_of,
                        &mut scc_bits,
                    );
                    next_scc += 1;
                }
            }
        }
    }
    (scc_of, scc_bits)
}

struct InlineWalk<'a, 'b> {
    cx: &'a SemaContext<'b>,
    reached: &'a Reached,
    relocs: &'a [SymbolSource],
    transitive: Option<&'a OnceLock<TransitiveLocalFiles>>,
    visited_plans: HashSet<(usize, usize)>,
    visited_callees: Vec<u64>,
    out: Vec<(u32, u32)>,
}

impl InlineWalk<'_, '_> {
    fn push_item_file(&mut self, item: ItemRef) {
        if !self.cx.outline.is_fixed(item.krate) {
            self.out.push((item.krate, item.body_owner().file));
        }
    }

    fn push_slot_file(&mut self, idx: usize) {
        if let Some(((item, _), _)) = self.reached.list.get(idx) {
            self.push_item_file(*item);
        }
    }

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
                    resolve_inline_origin(self.reached, index, plan, self.relocs, base, origin)
                {
                    self.push_slot_file(callee_idx);
                    if self
                        .visited_plans
                        .insert((callee_idx, Arc::as_ptr(sub_plan) as usize))
                    {
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
            if let SymbolSource::Slot(c_idx) = *sym {
                self.push_slot_file(c_idx);
            }
        }
        if matches!(origin, SiteOrigin::DropGlue(_)) && !plan_has_inline(sub_plan) {
            return;
        }
        if self
            .visited_plans
            .insert((index, std::ptr::from_ref(sub_plan) as usize))
        {
            self.visit_plan(sub_plan, index, pos);
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
                for res in &typed.drops {
                    if let Res::Fn(item, substs) = *res {
                        self.push_item_file(item);
                        if let Some(&idx) = self.reached.index.get(&(item, substs)) {
                            self.record_transitive_local_callees(idx);
                        }
                    }
                }
            }
            SiteOrigin::PointerFnDef { .. }
            | SiteOrigin::ConstEvalSelect(_)
            | SiteOrigin::VtableMethod { .. } => {
                self.record_transitive_local_callees(index);
            }
            _ => {}
        }
    }

    fn record_transitive_local_callees(&mut self, root: usize) {
        if let Some(cell) = self.transitive {
            let cache = cell.get_or_init(|| TransitiveLocalFiles::build(self.cx, self.reached));
            cache.push_callees(self.reached, root, &mut self.out);
            return;
        }
        let n = self.reached.list.len();
        let words = n.div_ceil(64);
        if self.visited_callees.len() < words {
            self.visited_callees.resize(words, 0);
        }
        let mut stack = vec![root];
        while let Some(u) = stack.pop() {
            for &v in self.reached.direct_calls(u).into_iter().flatten() {
                if v < n {
                    let mask = 1_u64 << (v % 64);
                    let slot = &mut self.visited_callees[v / 64];
                    if (*slot & mask) == 0 {
                        *slot |= mask;
                        self.push_slot_file(v);
                        stack.push(v);
                    }
                }
            }
        }
    }
}

fn plan_has_inline(plan: &RelocPlan) -> bool {
    plan.steps
        .iter()
        .any(|s| matches!(s, RelocStep::SiteInline { .. }))
}

pub(super) fn relocs_unwind_digest(
    cx: &mut SemaContext<'_>,
    reached: &Reached,
    relocs: &[SymbolSource],
) -> u64 {
    let mut h = FxHasher::default();
    relocs.len().hash(&mut h);
    for src in relocs {
        crate::lower::stencil::sym_may_unwind(cx, reached, Some(src)).hash(&mut h);
    }
    h.finish()
}

#[cfg(test)]
pub(super) fn collect_inlined_local_files(
    cx: &SemaContext<'_>,
    reached: &Reached,
    plan: &RelocPlan,
    index: usize,
    relocs: &[SymbolSource],
) -> Vec<(u32, u32)> {
    collect_inlined_local_files_cached(cx, reached, plan, index, relocs, None)
}

pub(super) fn collect_inlined_local_files_cached(
    cx: &SemaContext<'_>,
    reached: &Reached,
    plan: &RelocPlan,
    index: usize,
    relocs: &[SymbolSource],
    transitive: Option<&OnceLock<TransitiveLocalFiles>>,
) -> Vec<(u32, u32)> {
    let owner = reached
        .list
        .get(index)
        .map(|((item, _), _)| (item.krate, item.body_owner().file));
    let mut walk = InlineWalk {
        cx,
        reached,
        relocs,
        transitive,
        visited_plans: HashSet::default(),
        visited_callees: Vec::new(),
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

fn find_vtable_slot_callee(
    plan: &RelocPlan,
    relocs: &[SymbolSource],
    base: usize,
    target_expr: crate::body::ExprId,
    target_slot: u16,
) -> Option<usize> {
    let mut step_pos = base;
    for step in &plan.steps {
        if let RelocStep::VtableSlot { expr, slot } = *step
            && expr == target_expr
            && slot == target_slot
        {
            return match relocs.get(step_pos) {
                Some(&SymbolSource::Slot(idx)) => Some(idx),
                _ => None,
            };
        }
        step_pos += step.symbols();
    }
    None
}

fn resolve_inline_origin(
    reached: &Reached,
    index: usize,
    plan: &RelocPlan,
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
    if let SiteOrigin::VtableMethod { expr, slot } = *origin {
        return find_vtable_slot_callee(plan, relocs, base, expr, slot);
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
