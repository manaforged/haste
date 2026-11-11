use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use haste_codegen::{Code, DictTierUp, Site, Target};
use rustc_hash::FxHashMap as HashMap;

use crate::lower::{
    self, FootprintId, RodataId, RodataReloc, RodataTarget, Slots, SymbolSource, VtableEntry, VtableId, Word,
};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;

use super::{Image, stub};

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

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse().ok()
}

#[derive(Clone)]
pub(super) enum CachedTemplate {
    Stencil(Arc<Code>),
    Dict { body_exec: usize, code: Arc<Code> },
}

pub(super) struct JitTemplate {
    pub stencil: Arc<Code>,
    pub dict: Option<(usize, Arc<Code>)>,
    pub fanout: usize,
}

pub(super) struct TierUpEntry {
    pub item: ItemRef,
    pub footprint: FootprintId,
    pub symbols: Vec<u64>,
}

type FirstJit = (usize, Arc<Code>, Vec<u64>, usize);

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
        Self {
            cfg: DictConfig::from_env(),
            aliases: HashMap::default(),
            templates: HashMap::default(),
            jit_templates: HashMap::default(),
            first_jit: HashMap::default(),
            tier_ups: HashMap::default(),
        }
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
}

pub(super) struct GroupWork {
    pub item: ItemRef,
    pub footprint: FootprintId,
    pub distinct: Vec<DistinctInst>,
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
    fn entry(&self, index: usize) -> i64 {
        self.0.entry(index)
    }

    fn data(&self, bytes: &[u8]) -> i64 {
        self.0.data(bytes)
    }

    fn zeroed(&self, len: usize) -> i64 {
        self.0.zeroed(len)
    }

    fn table(&self, words: &[Word]) -> i64 {
        self.0.table(words)
    }

    fn cell(&self, index: usize, size: u32) -> i64 {
        self.0.cell(index, size)
    }

    fn want(&self, instance: crate::sema::reach::Instance) {
        self.0.want(instance);
    }

    fn wanted_len(&self) -> usize {
        self.0.wanted_len()
    }

    fn bodies(&self) -> &lower::Bodies {
        &self.1
    }

    fn low_entries(&self) -> bool {
        self.0.low_entries()
    }

    fn rodata(&self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
        self.0.rodata(bytes, relocs)
    }

    fn vtable_id(&self, entries: &[VtableEntry]) -> VtableId {
        self.0.vtable_id(entries)
    }

    fn resolve_symbol(&self, sym: &SymbolSource) -> u64 {
        self.0.resolve_symbol(sym)
    }
}

fn item_err(cx: &Context<'_>, item: ItemRef, err: impl std::fmt::Display) -> String {
    format!("{}: {err:#}", cx.name(item))
}

impl Image {
    pub(super) fn intern_rodata(&self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
        let key = (bytes.to_vec(), relocs.to_vec());
        if let Some(&id) = self.rodata_map.borrow().get(&key) {
            return id;
        }
        let address = self.data(bytes) as usize as u64;
        for reloc in relocs {
            self.apply_rodata_reloc(address, *reloc);
        }
        let mut addrs = self.rodata_addrs.borrow_mut();
        let id = RodataId(addrs.len() as u32);
        addrs.push(address);
        self.rodata_map.borrow_mut().insert(key, id);
        id
    }

    fn apply_rodata_reloc(&self, base: u64, reloc: RodataReloc) {
        let at = (base + u64::from(reloc.offset)) as usize;
        match reloc.target {
            RodataTarget::Rodata(target) => {
                let ptr = self.rodata_addrs.borrow()[target.0 as usize];
                // SAFETY: at..at + 8 lies within the freshly allocated rodata blob in shared memory.
                unsafe { (at as *mut u64).write_unaligned(ptr) };
            }
        }
    }

    pub(super) fn intern_vtable(&self, entries: &[VtableEntry]) -> VtableId {
        if let Some(&id) = self.vtable_map.borrow().get(entries) {
            return id;
        }
        let words: Vec<Word> = entries
            .iter()
            .map(|entry| match *entry {
                VtableEntry::Word(word) => word,
                VtableEntry::Super(sub) => Word::Int(self.vtable_addrs.borrow()[sub.0 as usize]),
            })
            .collect();
        let address = self.table(&words) as usize as u64;
        let mut addrs = self.vtable_addrs.borrow_mut();
        let id = VtableId(addrs.len() as u32);
        addrs.push(address);
        self.vtable_map.borrow_mut().insert(entries.to_vec(), id);
        id
    }

    pub(super) fn promoted_cell(&self, owner: usize, seq: u32, size: u32) -> u64 {
        if let Some(&addr) = self.promoted_cells.borrow().get(&(owner, seq)) {
            return addr;
        }
        let addr = self.zeroed(size as usize) as usize as u64;
        self.promoted_cells.borrow_mut().insert((owner, seq), addr);
        addr
    }

    pub(super) fn symbol_value(&self, sym: &SymbolSource) -> u64 {
        match *sym {
            SymbolSource::Slot(index) => self.entry(index) as usize as u64,
            SymbolSource::Vtable(id) => self.vtable_addrs.borrow()[id.0 as usize],
            SymbolSource::Rodata(id) => self.rodata_addrs.borrow()[id.0 as usize],
            SymbolSource::StaticCell { index, size } => self.cell(index, size) as usize as u64,
            SymbolSource::PromotedCell { owner, seq, size } => self.promoted_cell(owner, seq, size),
            SymbolSource::Host(ref host) => host.host_addr().unwrap_or(0) as u64,
            SymbolSource::Word(word) => word,
        }
    }

    pub(super) fn reset_caches(&mut self) {
        self.bodies.clear();
        self.stencils.clear();
    }

    pub(super) fn compile_bucket(&mut self, cx: &mut Context<'_>, reached: &Reached, indices: &[usize], cfg: DictConfig, (record_cache, allow_singleton): (bool, bool)) -> Vec<(usize, String)> {
        if allow_singleton && indices.len() == 1 {
            let index = indices[0];
            return self.compile_singleton(cx, reached, index).err().map(|err| (index, err)).into_iter().collect();
        }
        let (groups, mut errors) = self.partition_work(cx, reached, indices, cfg);
        for group in &groups {
            if let Err(err) = self.compile_group(cx, reached, group, cfg, record_cache) {
                errors.extend(group.all_indices().map(|idx| (idx, err.clone())));
            }
        }
        errors
    }

    fn compile_singleton(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize) -> Result<(), String> {
        let started = std::time::Instant::now();
        let item = reached.list[index].0.0;
        let out = lower::lower_stencil(cx, reached, self, index).map_err(|e| item_err(cx, item, e))?;
        let symbols: Vec<u64> = out.relocs.iter().map(|sym| self.symbol_value(sym)).collect();
        let code = haste_codegen::compile(out.func, Target::host()).map_err(|e| item_err(cx, item, e))?;
        let exec = self.memory.write_patched(&code, &symbols).map_err(|e| item_err(cx, item, e))?;
        self.slot(index).store(exec, Ordering::Release);
        self.record_size(index, code.bytes().len() as u32, started.elapsed().as_nanos() as u64);
        Ok(())
    }

    pub(super) fn partition_work(&mut self, cx: &mut Context<'_>, reached: &Reached, work: &[usize], cfg: DictConfig) -> (Vec<GroupWork>, Vec<(usize, String)>) {
        let mut errors = Vec::new();
        let mut order: Vec<(ItemRef, FootprintId)> = Vec::new();
        let mut by_fp: HashMap<(ItemRef, FootprintId), (HashMap<Vec<u64>, usize>, Vec<DistinctInst>)> = HashMap::default();
        let mut verify_hits = Vec::new();
        for &index in work {
            let item = reached.list[index].0.0;
            let (footprint, relocs) = match lower::probe_footprint(cx, reached, self, index) {
                Ok(pair) => pair,
                Err(err) => {
                    errors.push((index, format!("{}: {err:#}", cx.name(item))));
                    continue;
                }
            };
            let symbols: Vec<u64> = relocs.iter().map(|sym| self.symbol_value(sym)).collect();
            if let Some(hit) = self.reuse_cached((item, footprint), index, &symbols) {
                if cfg.verify {
                    verify_hits.push((index, hit));
                }
                continue;
            }
            let entry = by_fp.entry((item, footprint)).or_insert_with(|| {
                order.push((item, footprint));
                (HashMap::default(), Vec::new())
            });
            if let Some(&pos) = entry.0.get(&symbols) {
                entry.1[pos].followers.push(index);
            } else {
                entry.0.insert(symbols.clone(), entry.1.len());
                entry.1.push(DistinctInst { leader: index, followers: Vec::new(), symbols });
            }
        }
        if cfg.verify {
            errors.extend(self.verify_records(cx, reached, &verify_hits));
        }
        let groups = order.into_iter().filter_map(|key| by_fp.remove(&key).map(|(_, distinct)| GroupWork { item: key.0, footprint: key.1, distinct })).collect();
        (groups, errors)
    }

    fn reuse_cached(&mut self, (item, footprint): (ItemRef, FootprintId), index: usize, symbols: &[u64]) -> Option<VerifyRecord> {
        let alias_key = (item, footprint, symbols.to_vec());
        if let Some(&(addr, is_dict)) = self.stencils.aliases.get(&alias_key) {
            self.slot(index).store(addr, Ordering::Release);
            return self.cached_verify_record((item, footprint), addr, is_dict);
        }
        let template = self.stencils.templates.get(&(item, footprint)).cloned()?;
        match template {
            CachedTemplate::Stencil(code) => {
                let exec = self.memory.write_patched(&code, symbols).ok()?;
                self.slot(index).store(exec, Ordering::Release);
                self.record_size(index, code.bytes().len() as u32, 0);
                self.stencils.aliases.insert(alias_key, (exec, false));
                Some(VerifyRecord::Patched { exec, len: code.bytes().len(), sites: code.sites().to_vec() })
            }
            CachedTemplate::Dict { body_exec, code } => {
                let (stub_exec, stub_bytes) = self.emit_dict_stub(body_exec, symbols, None).ok()?;
                self.slot(index).store(stub_exec, Ordering::Release);
                self.record_size(index, stub_bytes.len() as u32, 0);
                self.stencils.aliases.insert(alias_key, (stub_exec, true));
                Some(VerifyRecord::Dict { body_exec, stub_exec, stub_bytes, code })
            }
        }
    }

    fn cached_verify_record(&self, key: (ItemRef, FootprintId), addr: usize, is_dict: bool) -> Option<VerifyRecord> {
        match (is_dict, self.stencils.templates.get(&key)?) {
            (false, CachedTemplate::Stencil(code)) => Some(VerifyRecord::Patched { exec: addr, len: code.bytes().len(), sites: code.sites().to_vec() }),
            (true, CachedTemplate::Dict { body_exec, code }) => Some(VerifyRecord::Dict { body_exec: *body_exec, stub_exec: addr, stub_bytes: Vec::new(), code: code.clone() }),
            _ => None,
        }
    }

    pub(super) fn compile_group(&mut self, cx: &mut Context<'_>, reached: &Reached, group: &GroupWork, cfg: DictConfig, record_cache: bool) -> Result<(), String> {
        let rep = group.representative();
        if cfg.min_bytes == 0 && group.distinct.len() >= cfg.min_fanout {
            return self.compile_dict_group(cx, reached, group, cfg, record_cache);
        }
        let started = std::time::Instant::now();
        let out = lower::lower_stencil(cx, reached, self, rep).map_err(|e| item_err(cx, group.item, e))?;
        let code = Arc::new(haste_codegen::compile(out.func, Target::host()).map_err(|e| item_err(cx, group.item, e))?);
        if code.bytes().len() > cfg.min_bytes && group.distinct.len() >= cfg.min_fanout {
            self.record_size(rep, 0, started.elapsed().as_nanos() as u64);
            return self.compile_dict_group(cx, reached, group, cfg, record_cache);
        }
        let mut records = Vec::new();
        for inst in &group.distinct {
            let exec = self.memory.write_patched(&code, &inst.symbols).map_err(|e| item_err(cx, group.item, e))?;
            self.store_instance(inst, exec);
            self.record_size(inst.leader, code.bytes().len() as u32, 0);
            if record_cache {
                self.stencils.aliases.insert((group.item, group.footprint, inst.symbols.clone()), (exec, false));
            }
            if cfg.verify {
                for idx in std::iter::once(inst.leader).chain(inst.followers.iter().copied()) {
                    records.push((idx, VerifyRecord::Patched { exec, len: code.bytes().len(), sites: code.sites().to_vec() }));
                }
            }
        }
        self.record_size(rep, 0, started.elapsed().as_nanos() as u64);
        if record_cache {
            self.stencils.templates.insert((group.item, group.footprint), CachedTemplate::Stencil(code));
        }
        self.finish_verify(cx, reached, &records)
    }

    fn write_dict_body(&mut self, code: &Code) -> Result<usize> {
        if !code.relocs().iter().all(|reloc| reloc.as_runtime().is_some()) {
            bail!("dictionary body has non-runtime relocations");
        }
        self.memory.write_patched(code, &[]).map_err(|err| anyhow::anyhow!("{err}"))
    }

    fn compile_dict_group(&mut self, cx: &mut Context<'_>, reached: &Reached, group: &GroupWork, cfg: DictConfig, record_cache: bool) -> Result<(), String> {
        let rep = group.representative();
        let started = std::time::Instant::now();
        let (func, _) = lower::lower_dict(cx, reached, self, rep).map_err(|e| item_err(cx, group.item, e))?;
        let code = Arc::new(haste_codegen::compile(func, Target::host()).map_err(|e| item_err(cx, group.item, e))?);
        let body_exec = self.write_dict_body(&code).map_err(|e| item_err(cx, group.item, e))?;
        let mut records = Vec::new();
        for inst in &group.distinct {
            let (stub_exec, stub_bytes) = self.emit_dict_stub(body_exec, &inst.symbols, None).map_err(|e| item_err(cx, group.item, e))?;
            self.store_instance(inst, stub_exec);
            self.record_size(inst.leader, stub_bytes.len() as u32, 0);
            if record_cache {
                self.stencils.aliases.insert((group.item, group.footprint, inst.symbols.clone()), (stub_exec, true));
            }
            if cfg.verify {
                for idx in std::iter::once(inst.leader).chain(inst.followers.iter().copied()) {
                    records.push((idx, VerifyRecord::Dict { body_exec, stub_exec, stub_bytes: stub_bytes.clone(), code: code.clone() }));
                }
            }
        }
        self.record_size(rep, code.bytes().len() as u32, started.elapsed().as_nanos() as u64);
        if record_cache {
            self.stencils.templates.insert((group.item, group.footprint), CachedTemplate::Dict { body_exec, code });
        }
        self.finish_verify(cx, reached, &records)
    }

    fn store_instance(&self, inst: &DistinctInst, addr: usize) {
        self.slot(inst.leader).store(addr, Ordering::Release);
        for &follower in &inst.followers {
            self.slot(follower).store(addr, Ordering::Release);
        }
    }

    fn alloc_dict(&self, symbols: &[u64]) -> usize {
        if symbols.is_empty() {
            return 0;
        }
        let bytes: Vec<u8> = symbols.iter().flat_map(|word| word.to_le_bytes()).collect();
        self.data(&bytes) as usize
    }

    fn emit_dict_stub(&mut self, body_exec: usize, symbols: &[u64], tier_up: Option<DictTierUp>) -> Result<(usize, Vec<u8>)> {
        let dict_addr = self.alloc_dict(symbols);
        let arch = Target::host().arch;
        let space = if tier_up.is_some() { TIER_STUB_SPACE } else { DICT_STUB_SPACE };
        let mut written = Vec::new();
        let exec = self.memory.write_with(space, |stub_exec| {
            let bytes = haste_codegen::dict_entry_stub(arch, stub_exec, body_exec, dict_addr, tier_up)?;
            written = bytes.clone();
            Ok(bytes)
        })?;
        Ok((exec, written))
    }

    pub(super) fn compile_jit_one(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize) -> Result<usize> {
        let cfg = self.stencils.cfg;
        let item = reached.list[index].0.0;
        let wanted0 = self.wanted_len();
        if !cfg.verify && cfg.min_fanout > 1 && let Some(exec) = self.try_first_jit(cx, reached, index, item, wanted0)? {
            return Ok(exec);
        }
        self.compile_jit_cached(cx, reached, index, item, wanted0, cfg)
    }

    fn try_first_jit(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize, item: ItemRef, wanted0: usize) -> Result<Option<usize>> {
        match self.stencils.first_jit.get(&item).cloned() {
            None => {
                let out = lower::lower_stencil(cx, reached, self, index)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                let symbols: Vec<u64> = out.relocs.iter().map(|sym| self.symbol_value(sym)).collect();
                let code = Arc::new(haste_codegen::compile(out.func, Target::host())?);
                let exec = self.memory.write_patched(&code, &symbols)?;
                self.stencils.first_jit.insert(item, Some((index, code, symbols, exec)));
                Ok(Some(exec))
            }
            Some(Some((idx1, code1, syms1, exec1))) => {
                let (fp1, _) = lower::probe_footprint(cx, reached, self, idx1)?;
                if self.wanted_len() > wanted0 {
                    bail!("unreached inline callee");
                }
                self.stencils.jit_templates.insert((item, fp1), JitTemplate { stencil: code1, dict: None, fanout: 1 });
                self.stencils.aliases.insert((item, fp1, syms1), (exec1, false));
                self.stencils.first_jit.insert(item, None);
                Ok(None)
            }
            Some(None) => Ok(None),
        }
    }

    fn compile_jit_cached(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize, item: ItemRef, wanted0: usize, cfg: DictConfig) -> Result<usize> {
        let (footprint, relocs) = lower::probe_footprint(cx, reached, self, index)?;
        if self.wanted_len() > wanted0 {
            bail!("unreached inline callee");
        }
        let symbols: Vec<u64> = relocs.iter().map(|sym| self.symbol_value(sym)).collect();
        let alias_key = (item, footprint, symbols.clone());
        if let Some(&(addr, is_dict)) = self.stencils.aliases.get(&alias_key) {
            if cfg.verify && let Some(jt) = self.stencils.jit_templates.get(&(item, footprint)) {
                let rec = match (is_dict, &jt.dict) {
                    (true, Some((body_exec, dict_code))) => VerifyRecord::Dict { body_exec: *body_exec, stub_exec: addr, stub_bytes: Vec::new(), code: dict_code.clone() },
                    _ => VerifyRecord::Patched { exec: addr, len: jt.stencil.bytes().len(), sites: jt.stencil.sites().to_vec() },
                };
                self.verify_single(cx, reached, index, &rec)?;
            }
            return Ok(addr);
        }
        let stencil = self.ensure_jit_stencil(cx, reached, index, (item, footprint))?;
        let fanout = self.stencils.jit_templates.get(&(item, footprint)).map_or(1, |jt| jt.fanout);
        if stencil.bytes().len() > cfg.min_bytes && fanout >= cfg.min_fanout {
            return self.instantiate_jit_dict(cx, reached, index, (item, footprint), symbols, cfg);
        }
        let exec = self.memory.write_patched(&stencil, &symbols)?;
        self.stencils.aliases.insert(alias_key, (exec, false));
        if cfg.verify {
            let rec = VerifyRecord::Patched { exec, len: stencil.bytes().len(), sites: stencil.sites().to_vec() };
            self.verify_single(cx, reached, index, &rec)?;
        }
        Ok(exec)
    }

    fn ensure_jit_stencil(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize, key: (ItemRef, FootprintId)) -> Result<Arc<Code>> {
        if let Some(entry) = self.stencils.jit_templates.get_mut(&key) {
            entry.fanout += 1;
            return Ok(entry.stencil.clone());
        }
        let out = lower::lower_stencil(cx, reached, self, index)?;
        let stencil = Arc::new(haste_codegen::compile(out.func, Target::host())?);
        self.stencils.jit_templates.insert(key, JitTemplate { stencil: stencil.clone(), dict: None, fanout: 1 });
        Ok(stencil)
    }

    fn instantiate_jit_dict(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize, key: (ItemRef, FootprintId), symbols: Vec<u64>, cfg: DictConfig) -> Result<usize> {
        let (body_exec, dict_code) = match self.stencils.jit_templates.get(&key).and_then(|jt| jt.dict.clone()) {
            Some(pair) => pair,
            None => {
                let (func, _) = lower::lower_dict(cx, reached, self, index)?;
                let code = Arc::new(haste_codegen::compile(func, Target::host())?);
                let exec = self.write_dict_body(&code)?;
                if let Some(jt) = self.stencils.jit_templates.get_mut(&key) {
                    jt.dict = Some((exec, code.clone()));
                }
                (exec, code)
            }
        };
        let tier = self.make_tier_up(index, key, &symbols, cfg.tier_up)?;
        let (stub_exec, stub_bytes) = self.emit_dict_stub(body_exec, &symbols, tier)?;
        if tier.is_none() {
            self.stencils.aliases.insert((key.0, key.1, symbols), (stub_exec, true));
        }
        if cfg.verify {
            let rec = VerifyRecord::Dict { body_exec, stub_exec, stub_bytes, code: dict_code };
            self.verify_single(cx, reached, index, &rec)?;
        }
        Ok(stub_exec)
    }

    fn make_tier_up(&mut self, index: usize, key: (ItemRef, FootprintId), symbols: &[u64], threshold: u32) -> Result<Option<DictTierUp>> {
        if !self.jit_mode || threshold == 0 || index >= (stub::TIER_UP_BIT as usize) {
            return Ok(None);
        }
        let counter_addr = self.shared.borrow_mut().alloc(8)?;
        self.stencils.tier_ups.insert(index, TierUpEntry { item: key.0, footprint: key.1, symbols: symbols.to_vec() });
        Ok(Some(DictTierUp {
            counter_addr,
            threshold,
            stub_entry: stub::entry_addr(),
            encoded_index: (index as u32) | stub::TIER_UP_BIT,
        }))
    }

    pub(super) fn tier_up(&mut self, index: usize) -> Result<usize> {
        let Some(entry) = self.stencils.tier_ups.remove(&index) else {
            let existing = self.slot(index).load(Ordering::Acquire);
            if existing != 0 {
                return Ok(existing);
            }
            bail!("missing tier-up entry for slot {index}");
        };
        let Some(jt) = self.stencils.jit_templates.get(&(entry.item, entry.footprint)) else {
            bail!("missing stencil template for slot {index}");
        };
        let exec = self.memory.write_patched(&jt.stencil, &entry.symbols)?;
        self.stencils.aliases.insert((entry.item, entry.footprint, entry.symbols), (exec, false));
        self.slot(index).store(exec, Ordering::Release);
        self.link();
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
            VerifyRecord::Dict { body_exec, stub_exec, stub_bytes, code } => {
                Self::verify_dict(cx, reached, fresh, index, (*body_exec, *stub_exec), stub_bytes, code)
            }
        }
    }

    fn verify_patched(cx: &mut Context<'_>, reached: &Reached, fresh: &FreshSlots<'_>, index: usize, (exec, len): (usize, usize), sites: &[Site]) -> Result<()> {
        let out = lower::lower_stencil(cx, reached, fresh, index)?;
        let slots: &dyn Slots = fresh;
        let symbols: Vec<u64> = out.relocs.iter().map(|sym| slots.resolve_symbol(sym)).collect();
        let fresh_code = haste_codegen::compile(out.func, Target::host())?;
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
        let (func, _) = lower::lower_dict(cx, reached, fresh, index)?;
        let fresh_code = haste_codegen::compile(func, Target::host())?;
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
