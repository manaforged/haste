mod decode;

use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use crate::graph::Workspace;
use crate::macros::Compilation;
use crate::names::{Sym, Symbols};
use crate::outline::tree::{File, Macro, Textual};
use crate::sema::impls::Index as ImplIndex;
use crate::sema::summary::codec::{write_u32, write_u64};

use super::impls::{build_sysroot_heads, write_impl_index};
use super::names::write_names_crate;
use super::outline::write_outline_crate;
use super::tokens::{
    u32_le, write_compact_tokens, write_frag, write_items, write_marks, write_opt_u32,
    write_origins, write_path, write_rules, write_str, write_usize,
};
use super::{CrateIdentity, SYSROOT_MAGIC, SYSROOT_VERSION, identity_of};

pub(super) use decode::decode_sysroot;

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

    pub(super) fn file(&mut self, file: &'a Arc<File>) -> u32 {
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

    pub(super) fn mac(&mut self, mac: &'a Arc<Macro>) -> u32 {
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

    pub(super) fn textual(&mut self, node: &'a Arc<Textual>) -> u32 {
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

pub(super) fn opt_u32_le(b: &[u8], at: usize) -> Option<u32> {
    let v = u32_le(b, at);
    (v != u32::MAX).then_some(v)
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

pub(super) fn encode_sysroot(
    key: u64,
    workspace: &Workspace,
    compilation: &Compilation,
    sysroot_crates: &[usize],
) -> Option<(Vec<u8>, Arc<HashMap<u32, Arc<ImplIndex>>>)> {
    let mut tables = Tables::new(
        compilation.outline.crates.len(),
        sysroot_crates,
        &compilation.names.symbols,
    );
    let (crate_ends, crates_blob) = write_chunked(sysroot_crates, |buf, &krate| {
        write_outline_crate(buf, &compilation.outline.crates[krate], &mut tables)
    })?;
    let (name_ends, names_blob) = write_chunked(sysroot_crates, |buf, &krate| {
        write_names_crate(buf, &compilation.names.crates[krate], &mut tables)
    })?;
    let built_heads = build_sysroot_heads(compilation, sysroot_crates);
    let (impl_ends, impls_blob) = write_chunked(
        &sysroot_crates
            .iter()
            .copied()
            .zip(built_heads.iter())
            .collect::<Vec<_>>(),
        |buf, &(krate, index)| write_impl_index(buf, index.as_ref(), krate as u32, &tables),
    )?;
    let mut heads_map = HashMap::default();
    for (&krate, index) in sysroot_crates.iter().zip(built_heads) {
        if let Some(index) = index {
            heads_map.insert(krate as u32, Arc::new(index));
        }
    }

    let mut files_blob = Vec::new();
    let mut file_table = Vec::with_capacity(tables.files.len() * 8);
    for file in &tables.files {
        write_file(&mut files_blob, file, &tables.to_local)?;
        let end = u32::try_from(files_blob.len()).ok()?;
        let base_id = file
            .source
            .base
            .as_ref()
            .and_then(|b| tables.file_map.get(&(Arc::as_ptr(b) as usize)).copied());
        write_u32(&mut file_table, end);
        write_opt_u32(&mut file_table, base_id);
    }

    let (macro_ends, macros_blob) = write_chunked(&tables.macros, |buf, mac| {
        let file_id = *tables.file_map.get(&(Arc::as_ptr(&mac.file) as usize))?;
        write_u32(buf, file_id);
        write_u32(buf, tables.local_krate(mac.krate)?);
        write_rules(buf, &mac.rules, &tables.to_local)
    })?;

    let sym_blob_len: usize = tables.extra_syms.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(
        crates_blob.len()
            + names_blob.len()
            + files_blob.len()
            + file_table.len()
            + macros_blob.len()
            + sym_blob_len
            + 65536,
    );
    out.extend_from_slice(&SYSROOT_MAGIC);
    write_u32(&mut out, SYSROOT_VERSION);
    write_u32(&mut out, u32::try_from(sysroot_crates.len()).ok()?);
    write_u64(&mut out, key);
    let len_at = out.len();
    write_u64(&mut out, 0);

    for &krate in sysroot_crates {
        let ident = identity_of(workspace, &compilation.outline, krate, &tables.to_local)?;
        write_identity(&mut out, &ident)?;
    }

    write_usize(&mut out, tables.extra_syms.len())?;
    write_usize(&mut out, sym_blob_len)?;
    let mut sym_end = 0usize;
    for sym in &tables.extra_syms {
        sym_end += sym.len();
        write_usize(&mut out, sym_end)?;
    }
    for sym in &tables.extra_syms {
        out.extend_from_slice(sym.as_bytes());
    }

    write_usize(&mut out, tables.files.len())?;
    out.extend_from_slice(&file_table);
    out.extend_from_slice(&files_blob);

    write_usize(&mut out, macro_ends.len())?;
    for end in macro_ends {
        write_u32(&mut out, end);
    }
    out.extend_from_slice(&macros_blob);

    write_usize(&mut out, tables.textuals.len())?;
    for node in &tables.textuals {
        write_str(&mut out, &node.name)?;
        write_u32(
            &mut out,
            *tables.macro_map.get(&(Arc::as_ptr(&node.def) as usize))?,
        );
        let next_id = node
            .next
            .as_ref()
            .and_then(|n| tables.textual_map.get(&(Arc::as_ptr(n) as usize)).copied());
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
    for end in impl_ends {
        write_u32(&mut out, end);
    }
    out.extend_from_slice(&impls_blob);
    out.extend_from_slice(&SYSROOT_MAGIC);
    let total_len = out.len() as u64;
    out[len_at..len_at + 8].copy_from_slice(&total_len.to_le_bytes());
    Some((out, Arc::new(heads_map)))
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

fn disk_text_tail(file: &File) -> Option<&str> {
    let src = &file.source;
    if file.expanded
        || src.base.is_some()
        || !matches!(src.origins, crate::outline::source::Origins::Read)
        || src.path.starts_with("/haste/")
    {
        return None;
    }
    let meta = std::fs::metadata(&src.path).ok()?;
    if !meta.is_file() {
        return None;
    }
    let disk_len = usize::try_from(meta.len()).ok()?;
    if src.text.len() < disk_len || !src.text.is_char_boundary(disk_len) {
        return None;
    }
    Some(&src.text[disk_len..])
}

fn write_file(out: &mut Vec<u8>, file: &Arc<File>, to_local: &[Option<u32>]) -> Option<()> {
    let src = &file.source;
    write_path(out, &src.path)?;
    if let Some(tail) = disk_text_tail(file) {
        out.push(1);
        write_usize(out, src.text.len())?;
        write_str(out, tail)?;
    } else {
        out.push(0);
        write_str(out, &src.text)?;
    }
    write_compact_tokens(out, &src.tokens)?;
    write_marks(out, &src.marks, to_local)?;
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
