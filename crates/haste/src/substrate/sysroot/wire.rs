use std::hash::Hasher as _;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::graph::Workspace;
use crate::macros::Compilation;
use crate::names::{Crate as NamesCrate, Sym, Symbols};
use crate::outline::Outline;
use crate::outline::source::{Outputs, Source};
use crate::outline::tree::{Crate as OutlineCrate, Decl, File, Held, Impl, Macro, MemberBlock, Module, State, Textual, Waiting};
use crate::sema::summary::codec::{Cursor, write_u32, write_u64};

use super::names::{read_names_crate, write_names_crate};
use super::tokens::{
    frag_from_tag, read_items, read_mark, read_opt_u32, read_origins, read_path, read_paths, read_rules, read_str, read_tokens, read_u32s_map,
    read_vis, u32_le, write_frag, write_items, write_mark, write_opt_u32, write_origins, write_path, write_paths, write_rules, write_str, write_tokens, write_usize,
    write_vis,
};
use super::{CrateIdentity, HEADER_LEN, SYSROOT_MAGIC, SYSROOT_VERSION, identity_of, match_crates};

const FIXED_SYM_COUNT: u32 = 23;

pub(super) struct Tables<'a> {
    to_local: Vec<Option<u32>>,
    symbols: &'a Symbols,
    sym_map: HashMap<Sym, u32>,
    extra_syms: Vec<&'a str>,
    file_map: HashMap<usize, u32>,
    files: Vec<&'a Arc<File>>,
    macro_map: HashMap<usize, u32>,
    macros: Vec<&'a Arc<Macro>>,
    textual_map: HashMap<usize, u32>,
    textuals: Vec<&'a Arc<Textual>>,
}

impl<'a> Tables<'a> {
    fn new(crate_count: usize, sysroot_crates: &[usize], symbols: &'a Symbols) -> Self {
        let mut to_local = vec![None; crate_count];
        for (local, &krate) in sysroot_crates.iter().enumerate() {
            to_local[krate] = Some(local as u32);
        }
        Self {
            to_local,
            symbols,
            sym_map: HashMap::default(),
            extra_syms: Vec::new(),
            file_map: HashMap::default(),
            files: Vec::new(),
            macro_map: HashMap::default(),
            macros: Vec::new(),
            textual_map: HashMap::default(),
            textuals: Vec::new(),
        }
    }

    pub(super) fn sym(&mut self, sym: Sym) -> u32 {
        if sym.0 < FIXED_SYM_COUNT {
            return sym.0;
        }
        if let Some(&id) = self.sym_map.get(&sym) {
            return id;
        }
        let id = FIXED_SYM_COUNT + self.extra_syms.len() as u32;
        self.extra_syms.push(self.symbols.text(sym));
        self.sym_map.insert(sym, id);
        id
    }

    fn file(&mut self, file: &'a Arc<File>) -> u32 {
        let ptr = Arc::as_ptr(file) as usize;
        if let Some(&id) = self.file_map.get(&ptr) {
            return id;
        }
        if let Some(base) = &file.source.base {
            self.file(base);
        }
        let id = self.files.len() as u32;
        self.files.push(file);
        self.file_map.insert(ptr, id);
        id
    }

    fn mac(&mut self, mac: &'a Arc<Macro>) -> u32 {
        let ptr = Arc::as_ptr(mac) as usize;
        if let Some(&id) = self.macro_map.get(&ptr) {
            return id;
        }
        self.file(&mac.file);
        let id = self.macros.len() as u32;
        self.macros.push(mac);
        self.macro_map.insert(ptr, id);
        id
    }

    fn textual(&mut self, node: &'a Arc<Textual>) -> u32 {
        let ptr = Arc::as_ptr(node) as usize;
        if let Some(&id) = self.textual_map.get(&ptr) {
            return id;
        }
        if let Some(next) = &node.next {
            self.textual(next);
        }
        self.mac(&node.def);
        let id = self.textuals.len() as u32;
        self.textuals.push(node);
        self.textual_map.insert(ptr, id);
        id
    }

    pub(super) fn local_krate(&self, krate: usize) -> Option<u32> {
        *self.to_local.get(krate)?
    }
}

fn write_chunked<T>(
    items: &[T],
    mut encode: impl FnMut(&mut Vec<u8>, &T) -> Option<()>,
) -> Option<(Vec<u32>, Vec<u8>)> {
    let mut ends = Vec::with_capacity(items.len());
    let mut blob = Vec::new();
    for item in items {
        encode(&mut blob, item)?;
        ends.push(u32::try_from(blob.len()).ok()?);
    }
    Some((ends, blob))
}

fn read_slices<'a>(cur: &mut Cursor<'a>, count: usize) -> Result<Vec<&'a [u8]>> {
    let table_bytes = count.checked_mul(4).context("slice table overflow")?;
    let table = cur.take(table_bytes)?;
    let mut ranges = Vec::with_capacity(count);
    let mut prev = 0usize;
    for c in table.chunks_exact(4) {
        let end = u32_le(c, 0) as usize;
        if end < prev {
            bail!("non-monotonic sysroot slice offset");
        }
        ranges.push((prev, end));
        prev = end;
    }
    let blob = cur.take(prev)?;
    Ok(ranges.into_iter().map(|(s, e)| &blob[s..e]).collect())
}

fn read_file_slices<'a>(cur: &mut Cursor<'a>, count: usize) -> Result<Vec<(&'a [u8], Option<u32>)>> {
    let table_bytes = count.checked_mul(8).context("file table overflow")?;
    let table = cur.take(table_bytes)?;
    let mut entries = Vec::with_capacity(count);
    let mut prev = 0usize;
    for c in table.chunks_exact(8) {
        let end = u32_le(c, 0) as usize;
        let base_raw = u32_le(c, 4);
        if end < prev {
            bail!("non-monotonic sysroot file offset");
        }
        entries.push((prev, end, (base_raw != u32::MAX).then_some(base_raw)));
        prev = end;
    }
    let blob = cur.take(prev)?;
    Ok(entries.into_iter().map(|(s, e, base)| (&blob[s..e], base)).collect())
}

fn hash_chunk(chunk: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(chunk);
    hasher.finish()
}

fn payload_checksum(payload: &[u8]) -> u64 {
    const CHUNK: usize = 1024 * 1024;
    let chunks: Vec<&[u8]> = payload.chunks(CHUNK).collect();
    let hashes = crate::sync::parallel_map(chunks, hash_chunk);
    let mut hasher = FxHasher::default();
    hasher.write_usize(payload.len());
    for h in hashes {
        hasher.write_u64(h);
    }
    hasher.finish()
}

pub(super) fn encode_sysroot(key: u64, workspace: &Workspace, compilation: &Compilation, sysroot_crates: &[usize]) -> Option<Vec<u8>> {
    let mut tables = Tables::new(compilation.outline.crates.len(), sysroot_crates, &compilation.names.symbols);
    let (crate_ends, crates_blob) = write_chunked(sysroot_crates, |buf, &krate| {
        write_outline_crate(buf, &compilation.outline.crates[krate], &mut tables)
    })?;
    let (name_ends, names_blob) = write_chunked(sysroot_crates, |buf, &krate| {
        write_names_crate(buf, &compilation.names.crates[krate], &mut tables)
    })?;

    let mut files_blob = Vec::new();
    let mut file_table = Vec::with_capacity(tables.files.len() * 8);
    for file in &tables.files {
        write_file(&mut files_blob, file, &tables.to_local)?;
        let end = u32::try_from(files_blob.len()).ok()?;
        let base_id = file.source.base.as_ref().and_then(|b| tables.file_map.get(&(Arc::as_ptr(b) as usize)).copied());
        write_u32(&mut file_table, end);
        write_opt_u32(&mut file_table, base_id);
    }

    let mut out = Vec::with_capacity(crates_blob.len() + names_blob.len() + files_blob.len() + file_table.len() + 65536);
    out.extend_from_slice(&SYSROOT_MAGIC);
    write_u32(&mut out, SYSROOT_VERSION);
    write_u32(&mut out, u32::try_from(sysroot_crates.len()).ok()?);
    write_u64(&mut out, key);
    let checksum_at = out.len();
    write_u64(&mut out, 0);

    for &krate in sysroot_crates {
        let ident = identity_of(workspace, &compilation.outline, krate, &tables.to_local)?;
        write_identity(&mut out, &ident)?;
    }

    write_usize(&mut out, tables.extra_syms.len())?;
    for sym in &tables.extra_syms {
        write_str(&mut out, sym)?;
    }

    write_usize(&mut out, tables.files.len())?;
    out.extend_from_slice(&file_table);
    out.extend_from_slice(&files_blob);

    write_usize(&mut out, tables.macros.len())?;
    for mac in &tables.macros {
        let file_id = *tables.file_map.get(&(Arc::as_ptr(&mac.file) as usize))?;
        write_u32(&mut out, file_id);
        write_u32(&mut out, tables.local_krate(mac.krate)?);
        write_rules(&mut out, &mac.rules, &tables.to_local)?;
    }

    write_usize(&mut out, tables.textuals.len())?;
    for node in &tables.textuals {
        write_str(&mut out, &node.name)?;
        write_u32(&mut out, *tables.macro_map.get(&(Arc::as_ptr(&node.def) as usize))?);
        let next_id = node.next.as_ref().and_then(|n| tables.textual_map.get(&(Arc::as_ptr(n) as usize)).copied());
        write_opt_u32(&mut out, next_id);
    }

    for end in crate_ends {
        write_u32(&mut out, end);
    }
    out.extend_from_slice(&crates_blob);
    for end in name_ends {
        write_u32(&mut out, end);
    }
    out.extend_from_slice(&names_blob);
    let checksum = payload_checksum(&out[HEADER_LEN..]);
    out[checksum_at..checksum_at + 8].copy_from_slice(&checksum.to_le_bytes());
    Some(out)
}

pub(super) fn decode_sysroot(
    data: &[u8],
    expected_key: u64,
    workspace: &Workspace,
    outline: &Outline,
) -> Result<Option<(Vec<(usize, OutlineCrate)>, Symbols, Vec<(usize, NamesCrate)>)>> {
    let mut cur = Cursor::new(data);
    if cur.take(8)? != SYSROOT_MAGIC {
        bail!("invalid sysroot cache magic");
    }
    if cur.u32()? != SYSROOT_VERSION {
        return Ok(None);
    }
    let crate_count = cur.count()?;
    if cur.u64()? != expected_key {
        return Ok(None);
    }
    let expected_checksum = cur.u64()?;
    if payload_checksum(&data[HEADER_LEN..]) != expected_checksum {
        return Ok(None);
    }

    let mut identities = Vec::with_capacity(crate_count);
    for _ in 0..crate_count {
        identities.push(read_identity(&mut cur)?);
    }
    let Some(local_crates) = match_crates(&identities, workspace, outline) else {
        return Ok(None);
    };
    let remap: Vec<u32> = local_crates.iter().map(|&k| k as u32).collect();

    let sym_count = cur.count()?;
    let symbols = Symbols::default();
    for idx in 0..sym_count {
        let text = read_str(&mut cur)?;
        let expected = FIXED_SYM_COUNT + idx as u32;
        if symbols.intern(text).0 != expected {
            bail!("sysroot symbol table mismatch");
        }
    }

    let file_count = cur.count()?;
    let file_slices = read_file_slices(&mut cur, file_count)?;
    let mut indep = Vec::with_capacity(file_count);
    let mut deps = Vec::new();
    for (idx, &(slice, base)) in file_slices.iter().enumerate() {
        match base {
            None => indep.push((idx, slice)),
            Some(base_id) => deps.push((idx, slice, base_id)),
        }
    }
    indep.sort_unstable_by(|a, b| b.1.len().cmp(&a.1.len()));
    let decoded_indep = crate::sync::parallel_map(indep, |(idx, slice)| -> Result<(usize, Arc<File>)> {
        let mut fcur = Cursor::new(slice);
        let file = read_file(&mut fcur, None, &remap)?;
        if fcur.pos != slice.len() {
            bail!("trailing bytes in sysroot file");
        }
        Ok((idx, Arc::new(file)))
    });
    let mut file_slots: Vec<Option<Arc<File>>> = vec![None; file_count];
    for item in decoded_indep {
        let (idx, file) = item?;
        file_slots[idx] = Some(file);
    }
    for (idx, slice, base_id) in deps {
        let base = Arc::clone(file_slots.get(base_id as usize).and_then(Option::as_ref).context("base file out of bounds")?);
        let mut fcur = Cursor::new(slice);
        let file = read_file(&mut fcur, Some(base), &remap)?;
        if fcur.pos != slice.len() {
            bail!("trailing bytes in sysroot file");
        }
        file_slots[idx] = Some(Arc::new(file));
    }
    let files: Vec<Arc<File>> = file_slots.into_iter().map(|f| f.expect("every file index decoded")).collect();

    let macro_count = cur.count()?;
    let mut macros = Vec::with_capacity(macro_count);
    for _ in 0..macro_count {
        let file = Arc::clone(files.get(cur.u32()? as usize).context("macro file out of bounds")?);
        let krate = *remap.get(cur.u32()? as usize).context("macro crate out of bounds")? as usize;
        let rules = read_rules(&mut cur, &remap)?;
        macros.push(Arc::new(Macro { file, rules, krate }));
    }

    let textual_count = cur.count()?;
    let mut textuals: Vec<Arc<Textual>> = Vec::with_capacity(textual_count);
    for _ in 0..textual_count {
        let name = read_str(&mut cur)?.to_owned();
        let def = Arc::clone(macros.get(cur.u32()? as usize).context("textual macro out of bounds")?);
        let next = match read_opt_u32(&mut cur)? {
            Some(id) => Some(Arc::clone(textuals.get(id as usize).context("textual next out of bounds")?)),
            None => None,
        };
        textuals.push(Arc::new(Textual { name, def, next }));
    }

    let outline_slices = read_slices(&mut cur, crate_count)?;
    let name_slices = read_slices(&mut cur, crate_count)?;
    if cur.pos != data.len() {
        bail!("trailing bytes in sysroot cache");
    }

    enum CrateTask<'a> {
        Outline(usize, &'a [u8]),
        Names(usize, &'a [u8]),
    }
    enum CrateOut {
        Outline(usize, OutlineCrate),
        Names(usize, NamesCrate),
    }

    let mut tasks = Vec::with_capacity(crate_count * 2);
    for (pos, slice) in outline_slices.into_iter().enumerate() {
        tasks.push(CrateTask::Outline(pos, slice));
    }
    for (pos, slice) in name_slices.into_iter().enumerate() {
        tasks.push(CrateTask::Names(pos, slice));
    }
    tasks.sort_unstable_by(|a, b| {
        let la = match a { CrateTask::Outline(_, s) | CrateTask::Names(_, s) => s.len() };
        let lb = match b { CrateTask::Outline(_, s) | CrateTask::Names(_, s) => s.len() };
        lb.cmp(&la)
    });

    let decoded = crate::sync::parallel_map(tasks, |task| -> Result<CrateOut> {
        match task {
            CrateTask::Outline(pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let krate = read_outline_crate(&mut ccur, &files, &macros, &textuals, &remap)?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in sysroot outline crate");
                }
                Ok(CrateOut::Outline(pos, krate))
            }
            CrateTask::Names(pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let krate = read_names_crate(&mut ccur, &remap)?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in sysroot names crate");
                }
                Ok(CrateOut::Names(pos, krate))
            }
        }
    });

    let mut outline_slots: Vec<Option<OutlineCrate>> = (0..crate_count).map(|_| None).collect();
    let mut name_slots: Vec<Option<NamesCrate>> = (0..crate_count).map(|_| None).collect();
    for item in decoded {
        match item? {
            CrateOut::Outline(pos, krate) => outline_slots[pos] = Some(krate),
            CrateOut::Names(pos, krate) => name_slots[pos] = Some(krate),
        }
    }
    let outline_crates = local_crates.iter().zip(outline_slots).map(|(&krate, slot)| (krate, slot.expect("every outline crate decoded"))).collect();
    let name_crates = local_crates.iter().zip(name_slots).map(|(&krate, slot)| (krate, slot.expect("every names crate decoded"))).collect();
    Ok(Some((outline_crates, symbols, name_crates)))
}

fn write_identity(out: &mut Vec<u8>, ident: &CrateIdentity) -> Option<()> {
    write_str(out, &ident.unit_name)?;
    write_str(out, &ident.pkg_name)?;
    write_str(out, &ident.pkg_version)?;
    out.push(u8::from(ident.side_host));
    out.push(ident.role);
    write_usize(out, ident.features.len())?;
    for feat in &ident.features {
        write_str(out, feat)?;
    }
    write_usize(out, ident.deps.len())?;
    for (name, dep) in &ident.deps {
        write_str(out, name)?;
        write_u32(out, *dep);
    }
    Some(())
}

fn read_identity(cur: &mut Cursor<'_>) -> Result<CrateIdentity> {
    let unit_name = read_str(cur)?.to_owned();
    let pkg_name = read_str(cur)?.to_owned();
    let pkg_version = read_str(cur)?.to_owned();
    let side_host = cur.u8()? != 0;
    let role = cur.u8()?;
    let feat_len = cur.count()?;
    let mut features = Vec::with_capacity(feat_len);
    for _ in 0..feat_len {
        features.push(read_str(cur)?.to_owned());
    }
    let dep_len = cur.count()?;
    let mut deps = Vec::with_capacity(dep_len);
    for _ in 0..dep_len {
        deps.push((read_str(cur)?.to_owned(), cur.u32()?));
    }
    Ok(CrateIdentity { unit_name, pkg_name, pkg_version, side_host, role, features, deps })
}

fn write_file(out: &mut Vec<u8>, file: &Arc<File>, to_local: &[Option<u32>]) -> Option<()> {
    let src = &file.source;
    write_path(out, &src.path)?;
    write_str(out, &src.text)?;
    write_tokens(out, &src.tokens)?;
    write_usize(out, src.marks.len())?;
    for &(at, mark) in &src.marks {
        write_u32(out, at);
        write_mark(out, mark, to_local)?;
    }
    write_usize(out, src.opaque.len())?;
    for &(from, to, frag) in &src.opaque {
        write_u32(out, from);
        write_u32(out, to);
        write_frag(out, frag);
    }
    write_u32(out, src.offset);
    write_origins(out, &src.origins)?;
    write_items(out, &file.items)?;
    out.push(u8::from(file.expanded));
    write_u64(out, file.expand_origin);
    Some(())
}

fn read_file(cur: &mut Cursor<'_>, base: Option<Arc<File>>, remap: &[u32]) -> Result<File> {
    let path = read_path(cur)?;
    let text = read_str(cur)?.to_owned();
    let tokens = read_tokens(cur)?;
    let mark_len = cur.count()?;
    let mut marks = Vec::with_capacity(mark_len);
    for _ in 0..mark_len {
        marks.push((cur.u32()?, read_mark(cur, remap)?));
    }
    let opaque_len = cur.count()?;
    let opaque_bytes = cur.take(opaque_len.checked_mul(9).context("opaque overflow")?)?;
    let mut opaque = Vec::with_capacity(opaque_len);
    for c in opaque_bytes.chunks_exact(9) {
        opaque.push((u32_le(c, 0), u32_le(c, 4), frag_from_tag(c[8])?));
    }
    let offset = cur.u32()?;
    let origins = read_origins(cur)?;
    let items = read_items(cur)?;
    let expanded = cur.u8()? != 0;
    let expand_origin = cur.u64()?;
    let source = Source { path, text, tokens, marks, opaque, base, offset, outputs: Outputs::default(), origins };
    let mut file = File::new(source, items, expanded);
    file.expand_origin = expand_origin;
    Ok(file)
}

fn write_waiting<'a>(out: &mut Vec<u8>, w: &'a Waiting, tables: &mut Tables<'a>) -> Option<()> {
    write_u32(out, tables.file(&w.file));
    write_usize(out, w.owner)?;
    write_u32(out, w.at);
    let scope = w.scope.as_ref().map(|s| tables.textual(s));
    write_opt_u32(out, scope);
    Some(())
}

fn read_waiting(cur: &mut Cursor<'_>, files: &[Arc<File>], textuals: &[Arc<Textual>]) -> Result<Waiting> {
    let file = Arc::clone(files.get(cur.u32()? as usize).context("waiting file out of bounds")?);
    let owner = cur.u32()? as usize;
    let at = cur.u32()?;
    let scope = match read_opt_u32(cur)? {
        Some(id) => Some(Arc::clone(textuals.get(id as usize).context("waiting scope out of bounds")?)),
        None => None,
    };
    Ok(Waiting { file, owner, at, scope })
}

fn write_held<'a>(out: &mut Vec<u8>, h: &'a Held, tables: &mut Tables<'a>) -> Option<()> {
    write_u32(out, tables.file(&h.file));
    write_usize(out, h.owner)?;
    write_u32(out, h.at);
    write_u32(out, h.path.0);
    write_u32(out, h.path.1);
    Some(())
}

fn read_held(cur: &mut Cursor<'_>, files: &[Arc<File>]) -> Result<Held> {
    let file = Arc::clone(files.get(cur.u32()? as usize).context("held file out of bounds")?);
    let owner = cur.u32()? as usize;
    let at = cur.u32()?;
    let path = (cur.u32()?, cur.u32()?);
    Ok(Held { file, owner, at, path })
}

fn write_impls(out: &mut Vec<u8>, impls: &[Impl]) -> Option<()> {
    write_usize(out, impls.len())?;
    for block in impls {
        write_usize(out, block.file)?;
        write_u32(out, block.at);
        write_usize(out, block.module)?;
        write_usize(out, block.parts.len())?;
        for &part in &block.parts {
            write_usize(out, part)?;
        }
    }
    Some(())
}

fn read_impls(cur: &mut Cursor<'_>) -> Result<Vec<Impl>> {
    let count = cur.count()?;
    let mut impls = Vec::with_capacity(count);
    for _ in 0..count {
        let file = cur.u32()? as usize;
        let at = cur.u32()?;
        let module = cur.u32()? as usize;
        let parts = read_u32s_map(cur, |v| v as usize)?;
        impls.push(Impl { file, at, module, parts });
    }
    Ok(impls)
}

fn write_local_krates(out: &mut Vec<u8>, krates: &[usize], tables: &Tables<'_>) -> Option<()> {
    write_usize(out, krates.len())?;
    for &dep in krates {
        write_u32(out, tables.local_krate(dep)?);
    }
    Some(())
}

fn read_remapped_krates(cur: &mut Cursor<'_>, remap: &[u32], err: &'static str) -> Result<Vec<usize>> {
    let count = cur.count()?;
    let bytes = cur.take(count.checked_mul(4).context("crate list overflow")?)?;
    let mut out = Vec::with_capacity(count);
    for c in bytes.chunks_exact(4) {
        out.push(*remap.get(u32_le(c, 0) as usize).context(err)? as usize);
    }
    Ok(out)
}

fn write_outline_crate<'a>(out: &mut Vec<u8>, krate: &'a OutlineCrate, tables: &mut Tables<'a>) -> Option<()> {
    write_usize(out, krate.files.len())?;
    for file in &krate.files {
        let id = tables.file(file);
        write_u32(out, id);
    }
    write_usize(out, krate.modules.len())?;
    for module in &krate.modules {
        write_opt_u32(out, (module.file != usize::MAX).then_some(module.file as u32));
        write_opt_u32(out, module.item);
        write_opt_u32(out, module.parent.map(|p| p as u32));
        write_str(out, &module.name)?;
        write_path(out, &module.dir)?;
        write_usize(out, module.parts.len())?;
        for &part in &module.parts {
            write_usize(out, part)?;
        }
        write_vis(out, module.vis);
        out.push(u8::from(module.macro_use));
        match &module.state {
            State::Declared(Decl::Inline { file, at }) => { out.push(0); write_usize(out, *file)?; write_u32(out, *at); }
            State::Declared(Decl::File { candidates, by_attr }) => {
                out.push(1);
                write_paths(out, candidates)?;
                out.push(u8::from(*by_attr));
            }
            State::Loaded => out.push(2),
            State::Missing => out.push(3),
            State::Gated => out.push(4),
        }
        let textual = module.textual.as_ref().map(|t| tables.textual(t));
        write_opt_u32(out, textual);
    }
    write_paths(out, &krate.missing)?;
    write_paths(out, &krate.gated)?;
    write_usize(out, krate.errors.len())?;
    for e in &krate.errors { write_str(out, e)?; }
    write_usize(out, krate.exports.len())?;
    for (name, mac) in &krate.exports {
        write_str(out, name)?;
        write_u32(out, tables.mac(mac));
    }
    write_usize(out, krate.defined.len())?;
    for (&(file, item), mac) in &krate.defined {
        write_u32(out, file);
        write_u32(out, item);
        write_u32(out, tables.mac(mac));
    }
    write_usize(out, krate.deferred.len())?;
    for w in &krate.deferred { write_waiting(out, w, tables)?; }
    write_impls(out, &krate.impls)?;
    write_impls(out, &krate.traits)?;
    write_usize(out, krate.impl_bodies.len())?;
    for &(file, at) in &krate.impl_bodies {
        write_usize(out, file)?;
        write_u32(out, at);
    }
    write_usize(out, krate.members.len())?;
    for (block, w) in &krate.members {
        match *block {
            MemberBlock::Impl(idx) => { out.push(0); write_usize(out, idx)?; }
            MemberBlock::Trait(idx) => { out.push(1); write_usize(out, idx)?; }
        }
        write_waiting(out, w, tables)?;
    }
    write_local_krates(out, &krate.fallback, tables)?;
    write_usize(out, krate.foreign.len())?;
    for w in &krate.foreign { write_waiting(out, w, tables)?; }
    write_usize(out, krate.global_asm.len())?;
    for w in &krate.global_asm { write_waiting(out, w, tables)?; }
    write_usize(out, krate.derived.len())?;
    for h in &krate.derived { write_held(out, h, tables)?; }
    write_usize(out, krate.attributed.len())?;
    for h in &krate.attributed { write_held(out, h, tables)?; }
    write_usize(out, krate.consumed.len())?;
    for &(file, item) in &krate.consumed {
        write_u32(out, file);
        write_u32(out, item);
    }
    write_local_krates(out, &krate.macro_use, tables)?;
    write_usize(out, krate.expansions)?;
    Some(())
}

fn read_outline_crate(
    cur: &mut Cursor<'_>,
    files: &[Arc<File>],
    macros: &[Arc<Macro>],
    textuals: &[Arc<Textual>],
    remap: &[u32],
) -> Result<OutlineCrate> {
    let file_count = cur.count()?;
    let file_bytes = cur.take(file_count.checked_mul(4).context("crate files overflow")?)?;
    let mut crate_files = Vec::with_capacity(file_count);
    for c in file_bytes.chunks_exact(4) {
        crate_files.push(Arc::clone(files.get(u32_le(c, 0) as usize).context("crate file out of bounds")?));
    }
    let mod_count = cur.count()?;
    let mut modules = Vec::with_capacity(mod_count);
    for _ in 0..mod_count {
        let file = read_opt_u32(cur)?.map_or(usize::MAX, |v| v as usize);
        let item = read_opt_u32(cur)?;
        let parent = read_opt_u32(cur)?.map(|v| v as usize);
        let name = read_str(cur)?.to_owned();
        let dir = read_path(cur)?;
        let parts = read_u32s_map(cur, |v| v as usize)?;
        let vis = read_vis(cur)?;
        let macro_use = cur.u8()? != 0;
        let state = match cur.u8()? {
            0 => State::Declared(Decl::Inline { file: cur.u32()? as usize, at: cur.u32()? }),
            1 => State::Declared(Decl::File { candidates: read_paths(cur)?, by_attr: cur.u8()? != 0 }),
            2 => State::Loaded,
            3 => State::Missing,
            4 => State::Gated,
            _ => bail!("invalid module state tag"),
        };
        let textual = match read_opt_u32(cur)? {
            Some(id) => Some(Arc::clone(textuals.get(id as usize).context("module textual out of bounds")?)),
            None => None,
        };
        modules.push(Module { file, item, parent, name, dir, parts, vis, macro_use, state, textual });
    }
    let missing = read_paths(cur)?;
    let gated = read_paths(cur)?;
    let err_len = cur.count()?;
    let mut errors = Vec::with_capacity(err_len);
    for _ in 0..err_len { errors.push(read_str(cur)?.to_owned()); }
    let export_len = cur.count()?;
    let mut exports = HashMap::with_capacity_and_hasher(export_len, Default::default());
    for _ in 0..export_len {
        let name = read_str(cur)?.to_owned();
        let mac = Arc::clone(macros.get(cur.u32()? as usize).context("export macro out of bounds")?);
        exports.insert(name, mac);
    }
    let def_len = cur.count()?;
    let def_bytes = cur.take(def_len.checked_mul(12).context("defined macros overflow")?)?;
    let mut defined = HashMap::with_capacity_and_hasher(def_len, Default::default());
    for c in def_bytes.chunks_exact(12) {
        let key = (u32_le(c, 0), u32_le(c, 4));
        let mac = Arc::clone(macros.get(u32_le(c, 8) as usize).context("defined macro out of bounds")?);
        defined.insert(key, mac);
    }
    let deferred = read_waitings(cur, files, textuals)?;
    let impls = read_impls(cur)?;
    let traits = read_impls(cur)?;
    let body_len = cur.count()?;
    let body_bytes = cur.take(body_len.checked_mul(8).context("impl bodies overflow")?)?;
    let impl_bodies = body_bytes.chunks_exact(8).map(|c| (u32_le(c, 0) as usize, u32_le(c, 4))).collect();
    let member_len = cur.count()?;
    let mut members = Vec::with_capacity(member_len);
    for _ in 0..member_len {
        let block = match cur.u8()? {
            0 => MemberBlock::Impl(cur.u32()? as usize),
            1 => MemberBlock::Trait(cur.u32()? as usize),
            _ => bail!("invalid member block tag"),
        };
        members.push((block, read_waiting(cur, files, textuals)?));
    }
    let fallback = read_remapped_krates(cur, remap, "fallback crate out of bounds")?;
    let foreign = read_waitings(cur, files, textuals)?;
    let global_asm = read_waitings(cur, files, textuals)?;
    let derived = read_helds(cur, files)?;
    let attributed = read_helds(cur, files)?;
    let consumed_len = cur.count()?;
    let consumed_bytes = cur.take(consumed_len.checked_mul(8).context("consumed overflow")?)?;
    let mut consumed = HashSet::with_capacity_and_hasher(consumed_len, Default::default());
    for c in consumed_bytes.chunks_exact(8) {
        consumed.insert((u32_le(c, 0), u32_le(c, 4)));
    }
    let macro_use = read_remapped_krates(cur, remap, "macro_use crate out of bounds")?;
    let expansions = cur.u32()? as usize;
    Ok(OutlineCrate {
        env: Default::default(),
        files: crate_files,
        modules,
        missing,
        gated,
        errors,
        exports,
        defined,
        deferred,
        impls,
        traits,
        impl_bodies,
        members,
        fallback,
        foreign,
        global_asm,
        derived,
        attributed,
        consumed,
        macro_use,
        expansions,
    })
}

fn read_waitings(cur: &mut Cursor<'_>, files: &[Arc<File>], textuals: &[Arc<Textual>]) -> Result<Vec<Waiting>> {
    let count = cur.count()?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count { out.push(read_waiting(cur, files, textuals)?); }
    Ok(out)
}

fn read_helds(cur: &mut Cursor<'_>, files: &[Arc<File>]) -> Result<Vec<Held>> {
    let count = cur.count()?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count { out.push(read_held(cur, files)?); }
    Ok(out)
}
