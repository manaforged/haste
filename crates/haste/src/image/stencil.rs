use std::sync::Arc;

use anyhow::{Result, bail};
use haste_codegen::{Code, Effort, Site, Target};
use rustc_hash::FxHashMap as HashMap;

use crate::lower::{self, FootprintId, RelocPlan, RodataId, RodataReloc, RodataTarget, Slots, SymbolSource, VtableEntry, VtableId, Word};
use crate::names::ItemRef;
use crate::outline::tree::Profile;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sync::lock;

use super::aot::AotFn;
use super::{Image, ImageMode, Pass};

/// Minimum template code size in bytes before sharing a dictionary body.
pub const DICT_MIN_BYTES: usize = 512;

/// Minimum distinct relocation vectors before sharing a dictionary body.
pub const DICT_MIN_FANOUT: usize = 8;

const DEFAULT_TIER_UP: u32 = 64;
const DICT_STUB_SPACE: usize = 48;

#[derive(Clone, Copy)]
pub(super) struct DictConfig {
    pub min_bytes: usize,
    pub min_fanout: usize,
    pub tier_up: u32,
    pub verify: bool,
}

impl DictConfig {
    pub fn from_env() -> Self {
        let min_bytes = env_usize("HASTE_DICT_MIN_BYTES").unwrap_or(DICT_MIN_BYTES);
        let default_fanout = if min_bytes == 0 { 1 } else { DICT_MIN_FANOUT };
        let min_fanout = env_usize("HASTE_DICT_MIN_FANOUT").unwrap_or(default_fanout).max(1);
        let tier_up = env_usize("HASTE_JIT_TIER_UP").map_or(DEFAULT_TIER_UP, |v| v as u32);
        let verify = std::env::var_os("HASTE_VERIFY_STENCIL").is_some_and(|v| v == "1");
        Self { min_bytes, min_fanout, tier_up, verify }
    }
}

pub(super) fn target() -> Target {
    crate::host_target()
}

impl From<Profile> for Effort {
    fn from(profile: Profile) -> Effort {
        match profile {
            Profile::Debug => Effort::Quick,
            Profile::Release => Effort::Full,
        }
    }
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse().ok()
}

#[derive(Clone)]
pub(super) enum CachedTemplate {
    Stencil { code: Arc<Code>, reloc_plan: Arc<RelocPlan> },
    Dict { body_exec: usize, code: Arc<Code>, reloc_plan: Arc<RelocPlan> },
}

impl CachedTemplate {
    fn reloc_plan(&self) -> &Arc<RelocPlan> {
        match self {
            Self::Stencil { reloc_plan, .. } | Self::Dict { reloc_plan, .. } => reloc_plan,
        }
    }
}

pub(super) struct JitTemplate {
    pub stencil: Arc<Code>,
    pub dict: Option<(usize, Arc<Code>)>,
    pub fanout: usize,
    pub reloc_plan: Arc<RelocPlan>,
}

pub(super) struct TierUpEntry {
    pub item: ItemRef,
    pub footprint: FootprintId,
    pub symbols: Vec<u64>,
}

type FirstJit = (usize, Arc<Code>, Vec<u64>, usize, Arc<RelocPlan>);

pub(super) struct StencilState {
    pub cfg: DictConfig,
    pub aliases: HashMap<(ItemRef, FootprintId, Vec<u64>), (usize, bool)>,
    pub templates: HashMap<(ItemRef, FootprintId), CachedTemplate>,
    pub jit_templates: HashMap<(ItemRef, FootprintId), JitTemplate>,
    pub first_jit: HashMap<ItemRef, Option<FirstJit>>,
    pub tier_ups: HashMap<usize, TierUpEntry>,
}

impl Default for StencilState {
    fn default() -> Self {
        Self { cfg: DictConfig::from_env(), aliases: HashMap::default(), templates: HashMap::default(), jit_templates: HashMap::default(), first_jit: HashMap::default(), tier_ups: HashMap::default() }
    }
}

impl StencilState {
    pub fn clear(&mut self) {
        self.cfg = DictConfig::from_env();
        self.aliases.clear();
        self.templates.clear();
        self.jit_templates.clear();
        self.first_jit.clear();
        self.tier_ups.clear();
    }
}

pub(super) struct DistinctInst {
    pub leader: usize,
    pub followers: Vec<usize>,
    pub symbols: Vec<u64>,
    pub relocs: Option<Vec<SymbolSource>>,
}

pub(super) struct GroupWork {
    pub item: ItemRef,
    pub footprint: FootprintId,
    pub distinct: Vec<DistinctInst>,
    pub leader_out: Option<lower::StencilOutput>,
}

impl GroupWork {
    pub fn representative(&self) -> usize {
        self.distinct[0].leader
    }

    pub fn all_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.distinct.iter().flat_map(|inst| std::iter::once(inst.leader).chain(inst.followers.iter().copied()))
    }
}

pub(super) enum VerifyRecord {
    Patched { exec: usize, len: usize, sites: Vec<Site> },
    Dict { body_exec: usize, stub_exec: usize, stub_bytes: Vec<u8>, code: Arc<Code> },
}

pub(super) struct FreshSlots<'a>(pub(super) &'a Image, pub(super) lower::Bodies);

impl Slots for FreshSlots<'_> {
    fn entry(&self, index: usize) -> i64 { self.0.entry(index) }
    fn data(&self, bytes: &[u8]) -> i64 { self.0.data(bytes) }
    fn zeroed(&self, len: usize) -> i64 { self.0.zeroed(len) }
    fn table(&self, words: &[Word]) -> i64 { self.0.table(words) }
    fn cell(&self, index: usize, size: u32) -> i64 { self.0.cell(index, size) }
    fn want(&self, instance: crate::sema::reach::Instance) { self.0.want(instance); }
    fn wanted_len(&self) -> usize { self.0.wanted_len() }
    fn bodies(&self) -> &lower::Bodies { &self.1 }
    fn allow_inline(&self) -> bool { self.0.allow_inline() }
    fn low_entries(&self) -> bool { self.0.low_entries() }
    fn rodata(&self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId { self.0.rodata(bytes, relocs) }
    fn vtable_id(&self, entries: &[VtableEntry]) -> VtableId { self.0.vtable_id(entries) }
    fn resolve_symbol(&self, sym: &SymbolSource) -> u64 { self.0.resolve_symbol(sym) }
}

fn item_err(cx: &Context<'_>, item: ItemRef, err: impl std::fmt::Display) -> String {
    format!("{}: {err:#}", cx.name(item))
}

impl Image {
    pub(super) fn intern_rodata(&self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
        let key = (bytes.to_vec(), relocs.to_vec());
        let mut map = lock(&self.rodata_map);
        if let Some(&id) = map.get(&key) {
            return id;
        }
        let address = self.data(bytes) as usize as u64;
        for reloc in relocs {
            self.apply_rodata_reloc(address, *reloc);
        }
        let mut addrs = lock(&self.rodata_addrs);
        let id = RodataId(addrs.len() as u32);
        addrs.push(address);
        if let Some(aot) = &self.aot {
            lock(aot).rodata.push((bytes.to_vec(), relocs.to_vec()));
        }
        map.insert(key, id);
        id
    }

    fn apply_rodata_reloc(&self, base: u64, reloc: RodataReloc) {
        let at = (base + u64::from(reloc.offset)) as usize;
        match reloc.target {
            RodataTarget::Rodata(target) => {
                let ptr = lock(&self.rodata_addrs)[target.0 as usize];
                // SAFETY: at..at + 8 lies within the freshly allocated rodata blob in shared memory.
                unsafe { (at as *mut u64).write_unaligned(ptr) };
            }
        }
    }

    pub(super) fn intern_vtable(&self, entries: &[VtableEntry]) -> VtableId {
        let mut map = lock(&self.vtable_map);
        if let Some(&id) = map.get(entries) {
            return id;
        }
        let words: Vec<Word> = entries
            .iter()
            .map(|entry| match *entry {
                VtableEntry::Word(word) => word,
                VtableEntry::Super(sub) => Word::Int(lock(&self.vtable_addrs)[sub.0 as usize]),
            })
            .collect();
        let address = self.table(&words) as usize as u64;
        let mut addrs = lock(&self.vtable_addrs);
        let id = VtableId(addrs.len() as u32);
        addrs.push(address);
        if let Some(aot) = &self.aot {
            lock(aot).vtables.push(entries.to_vec());
        }
        map.insert(entries.to_vec(), id);
        id
    }

    pub(super) fn promoted_cell(&self, owner: usize, seq: u32, size: u32) -> u64 {
        let mut cells = lock(&self.promoted_cells);
        if let Some(&addr) = cells.get(&(owner, seq)) {
            return addr;
        }
        let addr = self.zeroed(size as usize) as usize as u64;
        cells.insert((owner, seq), addr);
        if let Some(aot) = &self.aot {
            let mut state = lock(aot);
            if !state.promoted_seen.contains_key(&(owner, seq)) {
                let at = state.promoted.len();
                state.promoted_seen.insert((owner, seq), at);
                state.promoted.push(((owner, seq), size));
            }
        }
        addr
    }

    pub(super) fn symbol_value(&self, sym: &SymbolSource) -> u64 {
        match *sym {
            SymbolSource::Slot(index) => self.entry(index) as usize as u64,
            SymbolSource::Vtable(id) => lock(&self.vtable_addrs)[id.0 as usize],
            SymbolSource::Rodata(id) => lock(&self.rodata_addrs)[id.0 as usize],
            SymbolSource::StaticCell { index, size } => self.cell(index, size) as usize as u64,
            SymbolSource::PromotedCell { owner, seq, size } => self.promoted_cell(owner, seq, size),
            SymbolSource::Host(ref host) => host.host_addr().unwrap_or(0) as u64,
            SymbolSource::Word(word) => word,
            SymbolSource::ThreadSlot(index) => self.thread_slot(index),
        }
    }

    pub(super) fn reset_caches(&mut self) {
        self.bodies.clear();
        self.pass.stencils.clear();
    }

    pub(super) fn compile_bucket(&self, cx: &mut Context<'_>, reached: &Reached, indices: &[usize], cfg: DictConfig, (record_cache, allow_singleton): (bool, bool), pass: &mut Pass) -> Vec<(usize, String)> {
        if allow_singleton && indices.len() == 1 {
            let index = indices[0];
            return self.compile_singleton(cx, reached, index, pass).err().map(|err| (index, err)).into_iter().collect();
        }
        let (groups, mut errors) = self.partition_work(cx, reached, indices, cfg, pass);
        for mut group in groups {
            if let Err(err) = self.compile_group(cx, reached, &mut group, cfg, record_cache, pass) {
                errors.extend(group.all_indices().map(|idx| (idx, err.clone())));
            }
        }
        errors
    }

    fn compile_singleton(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, pass: &mut Pass) -> Result<(), String> {
        let started = std::time::Instant::now();
        let item = reached.list[index].0.0;
        let out = lower::lower_stencil(cx, reached, self, index).map_err(|e| item_err(cx, item, e))?;
        let code = haste_codegen::compile(out.func, &target().direct_calls(self.mode == ImageMode::Aot)).map_err(|e| item_err(cx, item, e))?;
        if self.mode == ImageMode::Jit {
            let symbols: Vec<u64> = out.relocs.iter().map(|sym| self.symbol_value(sym)).collect();
            let exec = pass.memory.write_patched(&code, &symbols).map_err(|e| item_err(cx, item, e))?;
            self.publish(index, exec);
        } else {
            for sym in &out.relocs {
                self.symbol_value(sym);
            }
        }
        let code_len = code.bytes().len() as u32;
        self.record_aot(index, AotFn::Direct { code: Arc::new(code), relocs: out.relocs }, &[]);
        pass.record_size(index, code_len, started.elapsed().as_nanos() as u64);
        Ok(())
    }

    pub(super) fn partition_work(&self, cx: &mut Context<'_>, reached: &Reached, work: &[usize], cfg: DictConfig, pass: &mut Pass) -> (Vec<GroupWork>, Vec<(usize, String)>) {
        let mut errors = Vec::new();
        let mut order: Vec<(ItemRef, FootprintId)> = Vec::new();
        let mut by_fp: HashMap<(ItemRef, FootprintId), Vec<usize>> = HashMap::default();
        for &index in work {
            let item = reached.list[index].0.0;
            let footprint = match lower::eval_footprint(cx, reached, self, index) {
                Ok(fp) => fp,
                Err(err) => {
                    errors.push((index, item_err(cx, item, &err)));
                    continue;
                }
            };
            by_fp.entry((item, footprint)).or_insert_with(|| {
                order.push((item, footprint));
                Vec::new()
            }).push(index);
        }
        let mut verify_hits = Vec::new();
        let mut groups = Vec::new();
        for (item, footprint) in order {
            let Some(indices) = by_fp.remove(&(item, footprint)) else { continue };
            if let Some(reloc_plan) = pass.stencils.templates.get(&(item, footprint)).map(|t| Arc::clone(t.reloc_plan())) {
                for index in indices {
                    let relocs = match lower::eval_relocs(cx, reached, self, &reloc_plan, index) {
                        Ok(r) => r,
                        Err(err) => {
                            errors.push((index, item_err(cx, item, &err)));
                            continue;
                        }
                    };
                    let symbols: Vec<u64> = relocs.iter().map(|sym| self.symbol_value(sym)).collect();
                    if let Some(hit) = self.reuse_cached((item, footprint), index, &symbols, self.aot.is_some().then_some(&relocs), pass)
                        && cfg.verify
                    {
                        verify_hits.push((index, hit));
                    }
                }
                continue;
            }
            let Some((&rep, rest)) = indices.split_first() else { continue };
            let out = match lower::lower_stencil(cx, reached, self, rep) {
                Ok(out) => out,
                Err(err) => {
                    let msg = item_err(cx, item, &err);
                    errors.extend(indices.into_iter().map(|idx| (idx, msg.clone())));
                    continue;
                }
            };
            let mut sym_pos: HashMap<Vec<u64>, usize> = HashMap::default();
            let mut distinct: Vec<DistinctInst> = Vec::new();
            let rep_syms: Vec<u64> = out.relocs.iter().map(|sym| self.symbol_value(sym)).collect();
            sym_pos.insert(rep_syms.clone(), 0);
            distinct.push(DistinctInst {
                leader: rep,
                followers: Vec::new(),
                symbols: rep_syms,
                relocs: self.aot.is_some().then(|| out.relocs.to_vec()),
            });
            for &index in rest {
                let relocs = match lower::eval_relocs(cx, reached, self, &out.reloc_plan, index) {
                    Ok(r) => r,
                    Err(err) => {
                        errors.push((index, item_err(cx, item, &err)));
                        continue;
                    }
                };
                let symbols: Vec<u64> = relocs.iter().map(|sym| self.symbol_value(sym)).collect();
                if let Some(&pos) = sym_pos.get(&symbols) {
                    distinct[pos].followers.push(index);
                } else {
                    sym_pos.insert(symbols.clone(), distinct.len());
                    distinct.push(DistinctInst {
                        leader: index,
                        followers: Vec::new(),
                        symbols,
                        relocs: self.aot.is_some().then(|| relocs.to_vec()),
                    });
                }
            }
            groups.push(GroupWork { item, footprint, distinct, leader_out: Some(out) });
        }
        if cfg.verify {
            errors.extend(self.verify_records(cx, reached, &verify_hits));
        }
        (groups, errors)
    }

    fn reuse_cached(&self, (item, footprint): (ItemRef, FootprintId), index: usize, symbols: &[u64], aot_relocs: Option<&Vec<SymbolSource>>, pass: &mut Pass) -> Option<VerifyRecord> {
        let alias_key = (item, footprint, symbols.to_vec());
        if let Some(&(addr, is_dict)) = pass.stencils.aliases.get(&alias_key) {
            if self.mode == ImageMode::Jit {
                self.publish(index, addr);
            }
            if let Some(relocs) = aot_relocs
                && let Some(template) = pass.stencils.templates.get(&(item, footprint))
            {
                match template {
                    CachedTemplate::Stencil { code, .. } => self.record_aot(index, AotFn::Direct { code: Arc::clone(code), relocs: relocs.clone() }, &[]),
                    CachedTemplate::Dict { code, .. } => self.record_aot(index, AotFn::Dict { body: Arc::clone(code), relocs: relocs.clone() }, &[]),
                }
            }
            return self.cached_verify_record((item, footprint), addr, is_dict, pass);
        }
        let template = pass.stencils.templates.get(&(item, footprint)).cloned()?;
        match template {
            CachedTemplate::Stencil { code, .. } => {
                let exec = if self.mode != ImageMode::Jit { 0 } else { let e = pass.memory.write_patched(&code, symbols).ok()?; self.publish(index, e); e };
                if let Some(relocs) = aot_relocs {
                    self.record_aot(index, AotFn::Direct { code: Arc::clone(&code), relocs: relocs.clone() }, &[]);
                }
                pass.record_size(index, code.bytes().len() as u32, 0);
                if self.mode != ImageMode::Check {
                    pass.stencils.aliases.insert(alias_key, (exec, false));
                }
                Some(VerifyRecord::Patched { exec, len: code.bytes().len(), sites: code.sites().to_vec() })
            }
            CachedTemplate::Dict { body_exec, code, .. } => {
                let (stub_exec, stub_bytes) = if self.mode != ImageMode::Jit { (0, Vec::new()) } else { let (e, b) = self.emit_dict_stub(body_exec, symbols, pass).ok()?; self.publish(index, e); (e, b) };
                if let Some(relocs) = aot_relocs {
                    self.record_aot(index, AotFn::Dict { body: Arc::clone(&code), relocs: relocs.clone() }, &[]);
                }
                pass.record_size(index, stub_bytes.len() as u32, 0);
                if self.mode != ImageMode::Check {
                    pass.stencils.aliases.insert(alias_key, (stub_exec, true));
                }
                Some(VerifyRecord::Dict { body_exec, stub_exec, stub_bytes, code })
            }
        }
    }

    fn cached_verify_record(&self, key: (ItemRef, FootprintId), addr: usize, is_dict: bool, pass: &Pass) -> Option<VerifyRecord> {
        match (is_dict, pass.stencils.templates.get(&key)?) {
            (false, CachedTemplate::Stencil { code, .. }) => Some(VerifyRecord::Patched { exec: addr, len: code.bytes().len(), sites: code.sites().to_vec() }),
            (true, CachedTemplate::Dict { body_exec, code, .. }) => Some(VerifyRecord::Dict { body_exec: *body_exec, stub_exec: addr, stub_bytes: Vec::new(), code: code.clone() }),
            _ => None,
        }
    }

    fn try_frozen_group(&self, cx: &mut Context<'_>, reached: &Reached, group: &GroupWork, cfg: DictConfig, pass: &mut Pass) -> Option<Result<(), String>> {
        let tree = &cx.outline.crates[group.item.krate as usize];
        if !tree.env.release || !tree.env.fixed { return None; }
        let substrate = cx.store().substrate()?;
        let item_key = crate::substrate::sysroot_item_key(cx.outline, group.item)?;
        let slice = substrate.stencil(item_key, group.footprint.0)?;
        let sites = cfg.verify.then(|| slice.decode_sites()).transpose().ok()?;
        let mut records = Vec::new();
        for inst in &group.distinct {
            if self.mode == ImageMode::Check {
                pass.record_size(inst.leader, slice.code().len() as u32, 0);
                continue;
            }
            let patched = crate::substrate::patch_stencil(&slice, &inst.symbols).ok()?;
            let exec = match pass.memory.write(&patched) { Ok(exec) => exec, Err(e) => return Some(Err(item_err(cx, group.item, e))) };
            self.store_instance(inst, exec);
            pass.record_size(inst.leader, patched.len() as u32, 0);
            if let Some(sites) = &sites {
                for idx in std::iter::once(inst.leader).chain(inst.followers.iter().copied()) {
                    records.push((idx, VerifyRecord::Patched { exec, len: slice.code().len(), sites: sites.clone() }));
                }
            }
        }
        Some(self.finish_verify(cx, reached, &records))
    }

    pub(super) fn compile_group(&self, cx: &mut Context<'_>, reached: &Reached, group: &mut GroupWork, cfg: DictConfig, record_cache: bool, pass: &mut Pass) -> Result<(), String> {
        let rep = group.representative();
        if cfg.min_bytes == 0 && group.distinct.len() >= cfg.min_fanout {
            return self.compile_dict_group(cx, reached, group, cfg, record_cache, pass);
        }
        if !self.jit_mode
            && self.mode != ImageMode::Aot
            && let Some(exec_ok) = self.try_frozen_group(cx, reached, group, cfg, pass)
        {
            return exec_ok;
        }
        let started = std::time::Instant::now();
        let out = match group.leader_out.take() {
            Some(out) => out,
            None => lower::lower_stencil(cx, reached, self, rep).map_err(|e| item_err(cx, group.item, e))?,
        };
        let reloc_plan = out.reloc_plan;
        let code = Arc::new(haste_codegen::compile(out.func, &target().direct_calls(self.mode == ImageMode::Aot)).map_err(|e| item_err(cx, group.item, e))?);
        let tree = &cx.outline.crates[group.item.krate as usize];
        if self.mode != ImageMode::Aot
            && tree.env.release
            && tree.env.fixed
            && code.relocs().iter().all(|r| r.symbol_index().is_some())
            && let Some(item_key) = crate::substrate::sysroot_item_key(cx.outline, group.item)
        {
            pass.frozen_stencils.push((item_key, group.footprint.0, Arc::clone(&code)));
        }
        if code.bytes().len() > cfg.min_bytes && group.distinct.len() >= cfg.min_fanout {
            pass.record_size(rep, 0, started.elapsed().as_nanos() as u64);
            return self.compile_dict_group(cx, reached, group, cfg, record_cache, pass);
        }
        let mut records = Vec::new();
        for inst in &group.distinct {
            let exec = if self.mode != ImageMode::Jit { 0 } else { let e = pass.memory.write_patched(&code, &inst.symbols).map_err(|e| item_err(cx, group.item, e))?; self.store_instance(inst, e); e };
            if let Some(relocs) = &inst.relocs {
                self.record_aot(inst.leader, AotFn::Direct { code: Arc::clone(&code), relocs: relocs.clone() }, &inst.followers);
            }
            pass.record_size(inst.leader, code.bytes().len() as u32, 0);
            if record_cache && self.mode != ImageMode::Check {
                pass.stencils.aliases.insert((group.item, group.footprint, inst.symbols.clone()), (exec, false));
            }
            if cfg.verify {
                for idx in std::iter::once(inst.leader).chain(inst.followers.iter().copied()) {
                    records.push((idx, VerifyRecord::Patched { exec, len: code.bytes().len(), sites: code.sites().to_vec() }));
                }
            }
        }
        pass.record_size(rep, 0, started.elapsed().as_nanos() as u64);
        if record_cache {
            pass.stencils.templates.insert((group.item, group.footprint), CachedTemplate::Stencil { code, reloc_plan });
        }
        self.finish_verify(cx, reached, &records)
    }

    pub(super) fn write_dict_body(&self, code: &Code, pass: &mut Pass) -> Result<usize> {
        if !code.relocs().iter().all(|reloc| reloc.as_runtime().is_some()) {
            bail!("dictionary body has non-runtime relocations");
        }
        pass.memory.write_patched(code, &[]).map_err(|err| anyhow::anyhow!("{err}"))
    }

    fn compile_dict_group(&self, cx: &mut Context<'_>, reached: &Reached, group: &GroupWork, cfg: DictConfig, record_cache: bool, pass: &mut Pass) -> Result<(), String> {
        let rep = group.representative();
        let started = std::time::Instant::now();
        let (func, _, reloc_plan) = lower::lower_dict(cx, reached, self, rep).map_err(|e| item_err(cx, group.item, e))?;
        let code = Arc::new(haste_codegen::compile(func, &target()).map_err(|e| item_err(cx, group.item, e))?);
        let body_exec = if self.mode != ImageMode::Jit { 0 } else { self.write_dict_body(&code, pass).map_err(|e| item_err(cx, group.item, e))? };
        let mut records = Vec::new();
        for inst in &group.distinct {
            let (stub_exec, stub_bytes) = if self.mode != ImageMode::Jit { (0, Vec::new()) } else { let (e, b) = self.emit_dict_stub(body_exec, &inst.symbols, pass).map_err(|e| item_err(cx, group.item, e))?; self.store_instance(inst, e); (e, b) };
            if let Some(relocs) = &inst.relocs {
                self.record_aot(inst.leader, AotFn::Dict { body: Arc::clone(&code), relocs: relocs.clone() }, &inst.followers);
            }
            pass.record_size(inst.leader, stub_bytes.len() as u32, 0);
            if record_cache && self.mode != ImageMode::Check {
                pass.stencils.aliases.insert((group.item, group.footprint, inst.symbols.clone()), (stub_exec, true));
            }
            if cfg.verify {
                for idx in std::iter::once(inst.leader).chain(inst.followers.iter().copied()) {
                    records.push((idx, VerifyRecord::Dict { body_exec, stub_exec, stub_bytes: stub_bytes.clone(), code: code.clone() }));
                }
            }
        }
        pass.record_size(rep, code.bytes().len() as u32, started.elapsed().as_nanos() as u64);
        if record_cache {
            pass.stencils.templates.insert((group.item, group.footprint), CachedTemplate::Dict { body_exec, code, reloc_plan });
        }
        self.finish_verify(cx, reached, &records)
    }

    fn store_instance(&self, inst: &DistinctInst, addr: usize) {
        self.publish(inst.leader, addr);
        for &follower in &inst.followers {
            self.publish(follower, addr);
        }
    }

    fn alloc_dict(&self, symbols: &[u64]) -> usize {
        if symbols.is_empty() {
            return 0;
        }
        let bytes: Vec<u8> = symbols.iter().flat_map(|word| word.to_le_bytes()).collect();
        self.data(&bytes) as usize
    }

    fn emit_dict_stub(&self, body_exec: usize, symbols: &[u64], pass: &mut Pass) -> Result<(usize, Vec<u8>)> {
        self.emit_dict_stub_with_tier(body_exec, symbols, None, pass)
    }

    pub(super) fn emit_dict_stub_with_tier(&self, body_exec: usize, symbols: &[u64], tier_up: Option<haste_codegen::DictTierUp>, pass: &mut Pass) -> Result<(usize, Vec<u8>)> {
        let dict_addr = self.alloc_dict(symbols);
        let arch = target().arch;
        let space = if tier_up.is_some() { 128 } else { DICT_STUB_SPACE };
        let mut written = Vec::new();
        let exec = pass.memory.write_with(space, |stub_exec| {
            let bytes = haste_codegen::dict_entry_stub(arch, stub_exec, body_exec, dict_addr, tier_up)?;
            written = bytes.clone();
            Ok(bytes)
        })?;
        Ok((exec, written))
    }
}
