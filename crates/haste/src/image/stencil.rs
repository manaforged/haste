use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use haste_codegen::{Code, DictTierUp, Effort, Site, Target};
use rustc_hash::FxHashMap as HashMap;

use crate::lower::{self, FootprintId, RelocPlan, RodataId, RodataReloc, RodataTarget, Slots, SymbolSource, VtableEntry, VtableId, Word};
use crate::names::ItemRef;
use crate::outline::tree::Profile;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sync::lock;

use super::aot::AotFn;
use super::{Image, Pass, stub};

/// Minimum template code size in bytes before sharing a dictionary body.
pub const DICT_MIN_BYTES: usize = 512;

/// Minimum distinct relocation vectors before sharing a dictionary body.
pub const DICT_MIN_FANOUT: usize = 8;

const DEFAULT_TIER_UP: u32 = 64;
const DICT_STUB_SPACE: usize = 48;
const TIER_STUB_SPACE: usize = 128;

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

/// The host target with the code generator's effort set by the build profile.
fn target() -> Target {
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

enum VerifyRecord {
    Patched { exec: usize, len: usize, sites: Vec<Site> },
    Dict { body_exec: usize, stub_exec: usize, stub_bytes: Vec<u8>, code: Arc<Code> },
}

struct FreshSlots<'a>(&'a Image, lower::Bodies);

impl Slots for FreshSlots<'_> {
    fn entry(&self, index: usize) -> i64 { self.0.entry(index) }
    fn data(&self, bytes: &[u8]) -> i64 { self.0.data(bytes) }
    fn zeroed(&self, len: usize) -> i64 { self.0.zeroed(len) }
    fn table(&self, words: &[Word]) -> i64 { self.0.table(words) }
    fn cell(&self, index: usize, size: u32) -> i64 { self.0.cell(index, size) }
    fn want(&self, instance: crate::sema::reach::Instance) { self.0.want(instance); }
    fn wanted_len(&self) -> usize { self.0.wanted_len() }
    fn bodies(&self) -> &lower::Bodies { &self.1 }
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
        let symbols: Vec<u64> = out.relocs.iter().map(|sym| self.symbol_value(sym)).collect();
        let code = haste_codegen::compile(out.func, &target()).map_err(|e| item_err(cx, item, e))?;
        let exec = pass.memory.write_patched(&code, &symbols).map_err(|e| item_err(cx, item, e))?;
        self.publish(index, exec);
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
            self.publish(index, addr);
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
                let exec = pass.memory.write_patched(&code, symbols).ok()?;
                self.publish(index, exec);
                if let Some(relocs) = aot_relocs {
                    self.record_aot(index, AotFn::Direct { code: Arc::clone(&code), relocs: relocs.clone() }, &[]);
                }
                pass.record_size(index, code.bytes().len() as u32, 0);
                pass.stencils.aliases.insert(alias_key, (exec, false));
                Some(VerifyRecord::Patched { exec, len: code.bytes().len(), sites: code.sites().to_vec() })
            }
            CachedTemplate::Dict { body_exec, code, .. } => {
                let (stub_exec, stub_bytes) = self.emit_dict_stub(body_exec, symbols, None, pass).ok()?;
                self.publish(index, stub_exec);
                if let Some(relocs) = aot_relocs {
                    self.record_aot(index, AotFn::Dict { body: Arc::clone(&code), relocs: relocs.clone() }, &[]);
                }
                pass.record_size(index, stub_bytes.len() as u32, 0);
                pass.stencils.aliases.insert(alias_key, (stub_exec, true));
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
            && self.aot.is_none()
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
        let code = Arc::new(haste_codegen::compile(out.func, &target()).map_err(|e| item_err(cx, group.item, e))?);
        let tree = &cx.outline.crates[group.item.krate as usize];
        if tree.env.release
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
            let exec = pass.memory.write_patched(&code, &inst.symbols).map_err(|e| item_err(cx, group.item, e))?;
            self.store_instance(inst, exec);
            if let Some(relocs) = &inst.relocs {
                self.record_aot(inst.leader, AotFn::Direct { code: Arc::clone(&code), relocs: relocs.clone() }, &inst.followers);
            }
            pass.record_size(inst.leader, code.bytes().len() as u32, 0);
            if record_cache {
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

    fn write_dict_body(&self, code: &Code, pass: &mut Pass) -> Result<usize> {
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
        let body_exec = self.write_dict_body(&code, pass).map_err(|e| item_err(cx, group.item, e))?;
        let mut records = Vec::new();
        for inst in &group.distinct {
            let (stub_exec, stub_bytes) = self.emit_dict_stub(body_exec, &inst.symbols, None, pass).map_err(|e| item_err(cx, group.item, e))?;
            self.store_instance(inst, stub_exec);
            if let Some(relocs) = &inst.relocs {
                self.record_aot(inst.leader, AotFn::Dict { body: Arc::clone(&code), relocs: relocs.clone() }, &inst.followers);
            }
            pass.record_size(inst.leader, stub_bytes.len() as u32, 0);
            if record_cache {
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

    fn emit_dict_stub(&self, body_exec: usize, symbols: &[u64], tier_up: Option<DictTierUp>, pass: &mut Pass) -> Result<(usize, Vec<u8>)> {
        let dict_addr = self.alloc_dict(symbols);
        let arch = target().arch;
        let space = if tier_up.is_some() { TIER_STUB_SPACE } else { DICT_STUB_SPACE };
        let mut written = Vec::new();
        let exec = pass.memory.write_with(space, |stub_exec| {
            let bytes = haste_codegen::dict_entry_stub(arch, stub_exec, body_exec, dict_addr, tier_up)?;
            written = bytes.clone();
            Ok(bytes)
        })?;
        Ok((exec, written))
    }

    pub(super) fn compile_jit_one(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, pass: &mut Pass) -> Result<usize> {
        let cfg = pass.stencils.cfg;
        let item = reached.list[index].0.0;
        let wanted0 = self.wanted_len();
        if !cfg.verify
            && cfg.min_fanout > 1
            && let Some(exec) = self.try_first_jit(cx, reached, index, item, wanted0, pass)?
        {
            return Ok(exec);
        }
        self.compile_jit_cached(cx, reached, index, item, wanted0, cfg, pass)
    }

    fn try_first_jit(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, item: ItemRef, wanted0: usize, pass: &mut Pass) -> Result<Option<usize>> {
        match pass.stencils.first_jit.get(&item).cloned() {
            None => {
                let out = lower::lower_stencil(cx, reached, self, index)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                let symbols: Vec<u64> = out.relocs.iter().map(|sym| self.symbol_value(sym)).collect();
                let code = Arc::new(haste_codegen::compile(out.func, &target())?);
                let exec = pass.memory.write_patched(&code, &symbols)?;
                pass.stencils.first_jit.insert(item, Some((index, code, symbols, exec, out.reloc_plan)));
                Ok(Some(exec))
            }
            Some(Some((idx1, code1, syms1, exec1, plan1))) => {
                let fp1 = lower::eval_footprint(cx, reached, self, idx1)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                pass.stencils.jit_templates.insert((item, fp1), JitTemplate { stencil: code1, dict: None, fanout: 1, reloc_plan: plan1 });
                pass.stencils.aliases.insert((item, fp1, syms1), (exec1, false));
                pass.stencils.first_jit.insert(item, None);
                Ok(None)
            }
            Some(None) => Ok(None),
        }
    }

    fn compile_jit_cached(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, item: ItemRef, wanted0: usize, cfg: DictConfig, pass: &mut Pass) -> Result<usize> {
        let footprint = lower::eval_footprint(cx, reached, self, index)?;
        if self.wanted_len() > wanted0 {
            bail!("unreached inline callee");
        }
        let existing_plan = pass.stencils.jit_templates.get(&(item, footprint)).map(|jt| Arc::clone(&jt.reloc_plan));
        let (symbols, prelowered) = match existing_plan {
            Some(plan) => {
                let relocs = lower::eval_relocs(cx, reached, self, &plan, index)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                (relocs.iter().map(|sym| self.symbol_value(sym)).collect::<Vec<u64>>(), None)
            }
            None => {
                let out = lower::lower_stencil(cx, reached, self, index)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                let symbols: Vec<u64> = out.relocs.iter().map(|sym| self.symbol_value(sym)).collect();
                (symbols, Some(out))
            }
        };
        let alias_key = (item, footprint, symbols.clone());
        if let Some(&(addr, is_dict)) = pass.stencils.aliases.get(&alias_key) {
            if cfg.verify
                && let Some(jt) = pass.stencils.jit_templates.get(&(item, footprint))
            {
                let rec = match (is_dict, &jt.dict) {
                    (true, Some((body_exec, dict_code))) => VerifyRecord::Dict { body_exec: *body_exec, stub_exec: addr, stub_bytes: Vec::new(), code: dict_code.clone() },
                    _ => VerifyRecord::Patched { exec: addr, len: jt.stencil.bytes().len(), sites: jt.stencil.sites().to_vec() },
                };
                self.verify_single(cx, reached, index, &rec)?;
            }
            return Ok(addr);
        }
        let stencil = self.ensure_jit_stencil(cx, reached, index, (item, footprint), prelowered, pass)?;
        let fanout = pass.stencils.jit_templates.get(&(item, footprint)).map_or(1, |jt| jt.fanout);
        if stencil.bytes().len() > cfg.min_bytes && fanout >= cfg.min_fanout {
            return self.instantiate_jit_dict(cx, reached, index, (item, footprint), symbols, cfg, pass);
        }
        let exec = pass.memory.write_patched(&stencil, &symbols)?;
        pass.stencils.aliases.insert(alias_key, (exec, false));
        if cfg.verify {
            let rec = VerifyRecord::Patched { exec, len: stencil.bytes().len(), sites: stencil.sites().to_vec() };
            self.verify_single(cx, reached, index, &rec)?;
        }
        Ok(exec)
    }

    fn ensure_jit_stencil(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, key: (ItemRef, FootprintId), prelowered: Option<lower::StencilOutput>, pass: &mut Pass) -> Result<Arc<Code>> {
        if let Some(entry) = pass.stencils.jit_templates.get_mut(&key) {
            entry.fanout += 1;
            return Ok(entry.stencil.clone());
        }
        let out = match prelowered {
            Some(out) => out,
            None => lower::lower_stencil(cx, reached, self, index)?,
        };
        let stencil = Arc::new(haste_codegen::compile(out.func, &target())?);
        pass.stencils.jit_templates.insert(key, JitTemplate { stencil: stencil.clone(), dict: None, fanout: 1, reloc_plan: out.reloc_plan });
        Ok(stencil)
    }

    fn instantiate_jit_dict(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, key: (ItemRef, FootprintId), symbols: Vec<u64>, cfg: DictConfig, pass: &mut Pass) -> Result<usize> {
        let (body_exec, dict_code) = match pass.stencils.jit_templates.get(&key).and_then(|jt| jt.dict.clone()) {
            Some(pair) => pair,
            None => {
                let (func, _, _) = lower::lower_dict(cx, reached, self, index)?;
                let code = Arc::new(haste_codegen::compile(func, &target())?);
                let exec = self.write_dict_body(&code, pass)?;
                if let Some(jt) = pass.stencils.jit_templates.get_mut(&key) {
                    jt.dict = Some((exec, code.clone()));
                }
                (exec, code)
            }
        };
        let tier = self.make_tier_up(index, key, &symbols, cfg.tier_up, pass)?;
        let (stub_exec, stub_bytes) = self.emit_dict_stub(body_exec, &symbols, tier, pass)?;
        if tier.is_none() {
            pass.stencils.aliases.insert((key.0, key.1, symbols), (stub_exec, true));
        }
        if cfg.verify {
            let rec = VerifyRecord::Dict { body_exec, stub_exec, stub_bytes, code: dict_code };
            self.verify_single(cx, reached, index, &rec)?;
        }
        Ok(stub_exec)
    }

    fn make_tier_up(&self, index: usize, key: (ItemRef, FootprintId), symbols: &[u64], threshold: u32, pass: &mut Pass) -> Result<Option<DictTierUp>> {
        if !self.jit_mode || threshold == 0 || index >= (stub::TIER_UP_BIT as usize) {
            return Ok(None);
        }
        let counter_addr = lock(&self.shared).alloc(8)?;
        pass.stencils.tier_ups.insert(index, TierUpEntry { item: key.0, footprint: key.1, symbols: symbols.to_vec() });
        Ok(Some(DictTierUp { counter_addr, threshold, stub_entry: stub::entry_addr(), encoded_index: (index as u32) | stub::TIER_UP_BIT }))
    }

    pub(super) fn tier_up(&mut self, index: usize) -> Result<usize> {
        let Some(entry) = self.pass.stencils.tier_ups.remove(&index) else {
            let existing = self.slot(index).load(Ordering::Acquire);
            if existing != 0 {
                return Ok(existing);
            }
            bail!("missing tier-up entry for slot {index}");
        };
        let Some(jt) = self.pass.stencils.jit_templates.get(&(entry.item, entry.footprint)) else {
            bail!("missing stencil template for slot {index}");
        };
        let exec = self.pass.memory.write_patched(&jt.stencil, &entry.symbols)?;
        self.pass.stencils.aliases.insert((entry.item, entry.footprint, entry.symbols), (exec, false));
        self.publish(index, exec);
        Ok(exec)
    }

    fn finish_verify(&self, cx: &mut Context<'_>, reached: &Reached, records: &[(usize, VerifyRecord)]) -> Result<(), String> {
        let errs = self.verify_records(cx, reached, records);
        if let Some((_, msg)) = errs.into_iter().next() {
            return Err(msg);
        }
        Ok(())
    }

    fn verify_records(&self, cx: &mut Context<'_>, reached: &Reached, records: &[(usize, VerifyRecord)]) -> Vec<(usize, String)> {
        let mut errs = Vec::new();
        let fresh = FreshSlots(self, lower::Bodies::default());
        for (index, record) in records {
            fresh.1.clear_transient();
            if let Err(err) = Self::verify_with_fresh(cx, reached, &fresh, *index, record) {
                errs.push((*index, format!("{}: {err:#}", cx.name(reached.list[*index].0.0))));
            }
        }
        errs
    }

    fn verify_single(&self, cx: &mut Context<'_>, reached: &Reached, index: usize, record: &VerifyRecord) -> Result<()> {
        let fresh = FreshSlots(self, lower::Bodies::default());
        Self::verify_with_fresh(cx, reached, &fresh, index, record)
    }

    fn verify_with_fresh(cx: &mut Context<'_>, reached: &Reached, fresh: &FreshSlots<'_>, index: usize, record: &VerifyRecord) -> Result<()> {
        match record {
            VerifyRecord::Patched { exec, len, sites } => Self::verify_patched(cx, reached, fresh, index, (*exec, *len), sites),
            VerifyRecord::Dict { body_exec, stub_exec, stub_bytes, code } => Self::verify_dict(cx, reached, fresh, index, (*body_exec, *stub_exec), stub_bytes, code),
        }
    }

    fn verify_patched(cx: &mut Context<'_>, reached: &Reached, fresh: &FreshSlots<'_>, index: usize, (exec, len): (usize, usize), sites: &[Site]) -> Result<()> {
        let out = lower::lower_stencil(cx, reached, fresh, index)?;
        let slots: &dyn Slots = fresh;
        let symbols: Vec<u64> = out.relocs.iter().map(|sym| slots.resolve_symbol(sym)).collect();
        let fresh_code = haste_codegen::compile(out.func, &target())?;
        let mut expected = vec![0u8; fresh_code.bytes().len()];
        fresh_code.patch(&symbols, &mut expected);
        // SAFETY: exec..exec + len is mapped readable executable code in this image.
        let actual = unsafe { std::slice::from_raw_parts(exec as *const u8, len) };
        if actual != expected.as_slice() {
            bail!("stencil verification byte mismatch ({} vs {} bytes)", actual.len(), expected.len());
        }
        if sites != fresh_code.sites() {
            bail!("stencil verification Site table mismatch");
        }
        Ok(())
    }

    fn verify_dict(cx: &mut Context<'_>, reached: &Reached, fresh: &FreshSlots<'_>, index: usize, (body_exec, stub_exec): (usize, usize), stub_bytes: &[u8], code: &Code) -> Result<()> {
        let (func, _, _) = lower::lower_dict(cx, reached, fresh, index)?;
        let fresh_code = haste_codegen::compile(func, &target())?;
        if !fresh_code.relocs().iter().all(|reloc| reloc.as_runtime().is_some()) {
            bail!("dict body verification found non-runtime relocations");
        }
        let mut expected = vec![0u8; fresh_code.bytes().len()];
        fresh_code.patch(&[], &mut expected);
        // SAFETY: body_exec..body_exec + code.bytes().len() is mapped readable executable code in this image.
        let actual_body = unsafe { std::slice::from_raw_parts(body_exec as *const u8, code.bytes().len()) };
        if actual_body != expected.as_slice() {
            bail!("dict body verification byte mismatch ({} vs {} bytes)", actual_body.len(), expected.len());
        }
        if code.sites() != fresh_code.sites() {
            bail!("dict body verification Site table mismatch");
        }
        if !stub_bytes.is_empty() {
            // SAFETY: stub_exec..stub_exec + stub_bytes.len() is mapped readable executable code in this image.
            let actual_stub = unsafe { std::slice::from_raw_parts(stub_exec as *const u8, stub_bytes.len()) };
            if actual_stub != stub_bytes {
                bail!("dict entry stub byte mismatch");
            }
        }
        Ok(())
    }
}
