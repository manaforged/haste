use anyhow::{Context as _, Result, bail};

use crate::graph::cfg::Cfg;
use crate::outline::item::{Attr, Import, Item, ItemKind, Items, Leaf};
use crate::sema::summary::codec::{Cursor, write_u32};

use super::super::wire::opt_u32_le;
use super::{read_str, u32_le, vis_from_tag, vis_tag, write_opt_u32, write_str, write_usize};

const ITEM_KINDS: [ItemKind; 17] = [
    ItemKind::Mod,
    ItemKind::Use,
    ItemKind::ExternCrate,
    ItemKind::Fn,
    ItemKind::Struct,
    ItemKind::Enum,
    ItemKind::Union,
    ItemKind::Trait,
    ItemKind::TraitAlias,
    ItemKind::Impl,
    ItemKind::Type,
    ItemKind::Const,
    ItemKind::Static,
    ItemKind::MacroRules,
    ItemKind::Macro,
    ItemKind::MacroCall,
    ItemKind::ExternBlock,
];

fn write_cfg_list(out: &mut Vec<u8>, list: &[Cfg]) -> Option<()> {
    write_usize(out, list.len())?;
    for item in list {
        write_cfg(out, item)?;
    }
    Some(())
}

fn read_cfg_list(cur: &mut Cursor<'_>) -> Result<Vec<Cfg>> {
    let len = cur.count()?;
    let mut list = Vec::with_capacity(len);
    for _ in 0..len {
        list.push(read_cfg(cur)?);
    }
    Ok(list)
}

fn write_cfg(out: &mut Vec<u8>, cfg: &Cfg) -> Option<()> {
    match cfg {
        Cfg::Name(name) => {
            out.push(0);
            write_str(out, name)?;
        }
        Cfg::KeyValue(k, v) => {
            out.push(1);
            write_str(out, k)?;
            write_str(out, v)?;
        }
        Cfg::All(list) => {
            out.push(2);
            write_cfg_list(out, list)?;
        }
        Cfg::Any(list) => {
            out.push(3);
            write_cfg_list(out, list)?;
        }
        Cfg::Not(inner) => {
            out.push(4);
            write_cfg(out, inner)?;
        }
    }
    Some(())
}

fn read_cfg(cur: &mut Cursor<'_>) -> Result<Cfg> {
    Ok(match cur.u8()? {
        0 => Cfg::Name(read_str(cur)?.to_owned()),
        1 => Cfg::KeyValue(read_str(cur)?.to_owned(), read_str(cur)?.to_owned()),
        2 => Cfg::All(read_cfg_list(cur)?),
        3 => Cfg::Any(read_cfg_list(cur)?),
        4 => Cfg::Not(Box::new(read_cfg(cur)?)),
        _ => bail!("invalid cfg tag"),
    })
}

fn write_attrs(out: &mut Vec<u8>, attrs: &[Attr]) -> Option<()> {
    write_usize(out, attrs.len())?;
    for attr in attrs {
        match attr {
            Attr::Cfg(cfg) => {
                out.push(0);
                write_cfg(out, cfg)?;
            }
            Attr::CfgAttr(cfg, inner) => {
                out.push(1);
                write_cfg(out, cfg)?;
                write_attrs(out, inner)?;
            }
            Attr::Path(p) => {
                out.push(2);
                write_str(out, p)?;
            }
            Attr::MacroUse => out.push(3),
            Attr::MacroExport(local) => {
                out.push(4);
                out.push(u8::from(*local));
            }
            Attr::Derive(at) => {
                out.push(5);
                write_u32(out, *at);
            }
            Attr::NoStd => out.push(6),
            Attr::NoCore => out.push(7),
            Attr::NoImplicitPrelude => out.push(8),
            Attr::Other(at) => {
                out.push(9);
                write_u32(out, *at);
            }
        }
    }
    Some(())
}

fn read_attrs(cur: &mut Cursor<'_>) -> Result<Vec<Attr>> {
    let len = cur.count()?;
    let mut attrs = Vec::with_capacity(len);
    for _ in 0..len {
        attrs.push(match cur.u8()? {
            0 => Attr::Cfg(read_cfg(cur)?),
            1 => Attr::CfgAttr(read_cfg(cur)?, read_attrs(cur)?),
            2 => Attr::Path(read_str(cur)?.to_owned()),
            3 => Attr::MacroUse,
            4 => Attr::MacroExport(cur.u8()? != 0),
            5 => Attr::Derive(cur.u32()?),
            6 => Attr::NoStd,
            7 => Attr::NoCore,
            8 => Attr::NoImplicitPrelude,
            9 => Attr::Other(cur.u32()?),
            _ => bail!("invalid attr tag"),
        });
    }
    Ok(attrs)
}

pub(in super::super) fn write_items(out: &mut Vec<u8>, items: &Items) -> Option<()> {
    write_attrs(out, &items.inner)?;
    write_usize(out, items.items.len())?;
    out.reserve(items.items.len() * 30);
    for item in &items.items {
        let (vtag, vis_at) = vis_tag(item.vis);
        let has_attrs = u8::from(!item.attrs.is_empty()) << 7;
        out.push(ITEM_KINDS.iter().position(|&k| k == item.kind)? as u8);
        out.push(vtag | has_attrs);
        write_opt_u32(out, item.name);
        write_u32(out, item.lead);
        write_u32(out, item.start);
        write_u32(out, item.end);
        write_opt_u32(out, item.body);
        write_opt_u32(out, item.parent);
        write_u32(out, vis_at);
    }
    for item in &items.items {
        if !item.attrs.is_empty() {
            write_attrs(out, &item.attrs)?;
        }
    }
    write_usize(out, items.imports.len())?;
    let total_paths: usize = items.imports.iter().map(|i| i.path.len()).sum();
    write_usize(out, total_paths)?;
    out.reserve(items.imports.len() * 13 + total_paths * 4);
    for import in &items.imports {
        let (leaf_tag, alias) = match import.leaf {
            Leaf::Name(alias) => (0u8, alias.unwrap_or(u32::MAX)),
            Leaf::Underscore => (1u8, u32::MAX),
            Leaf::Glob => (2u8, u32::MAX),
        };
        out.push(leaf_tag | (u8::from(import.absolute) << 2));
        write_u32(out, import.item);
        write_u32(out, alias);
        write_usize(out, import.path.len())?;
    }
    for import in &items.imports {
        for &seg in &import.path {
            write_u32(out, seg);
        }
    }
    write_usize(out, items.marks.len())?;
    out.reserve(items.marks.len() * 6);
    for &(at, bits) in &items.marks {
        write_u32(out, at);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Some(())
}

pub(in super::super) fn read_items(cur: &mut Cursor<'_>) -> Result<Items> {
    let inner = read_attrs(cur)?;
    let item_count = cur.count()?;
    let item_bytes = cur.take(item_count.checked_mul(30).context("items overflow")?)?;
    let mut items = Vec::with_capacity(item_count);
    let mut attr_indices = Vec::new();
    for (idx, b) in item_bytes.chunks_exact(30).enumerate() {
        let kind = ITEM_KINDS
            .get(b[0] as usize)
            .copied()
            .context("invalid item kind tag")?;
        let flags = b[1];
        let vis = vis_from_tag(flags & 0x7f, u32_le(b, 26))?;
        if flags & 0x80 != 0 {
            attr_indices.push(idx);
        }
        items.push(Item {
            kind,
            name: opt_u32_le(b, 2),
            vis,
            attrs: Vec::new(),
            lead: u32_le(b, 6),
            start: u32_le(b, 10),
            end: u32_le(b, 14),
            body: opt_u32_le(b, 18),
            parent: opt_u32_le(b, 22),
        });
    }
    for idx in attr_indices {
        items[idx].attrs = read_attrs(cur)?;
    }
    let import_count = cur.count()?;
    let total_paths = cur.count()?;
    let import_bytes = cur.take(import_count.checked_mul(13).context("imports overflow")?)?;
    let path_bytes = cur.take(
        total_paths
            .checked_mul(4)
            .context("import paths overflow")?,
    )?;
    let mut imports = Vec::with_capacity(import_count);
    let mut path_pos = 0usize;
    for b in import_bytes.chunks_exact(13) {
        let flags = b[0];
        let item = u32_le(b, 1);
        let alias = u32_le(b, 5);
        let path_len = u32_le(b, 9) as usize;
        let path_byte_len = path_len
            .checked_mul(4)
            .context("import path length overflow")?;
        let path_end = path_pos
            .checked_add(path_byte_len)
            .context("import path overflow")?;
        let seg_slice = path_bytes
            .get(path_pos..path_end)
            .context("import path out of bounds")?;
        path_pos = path_end;
        let path = seg_slice.chunks_exact(4).map(|p| u32_le(p, 0)).collect();
        let leaf = match flags & 3 {
            0 => Leaf::Name((alias != u32::MAX).then_some(alias)),
            1 => Leaf::Underscore,
            2 => Leaf::Glob,
            _ => bail!("invalid import leaf tag"),
        };
        imports.push(Import {
            item,
            absolute: flags & 4 != 0,
            path,
            leaf,
        });
    }
    if path_pos != path_bytes.len() {
        bail!("trailing bytes in import paths");
    }
    let mark_count = cur.count()?;
    let mark_bytes = cur.take(mark_count.checked_mul(6).context("item marks overflow")?)?;
    let marks = mark_bytes
        .chunks_exact(6)
        .map(|c| (u32_le(c, 0), u16::from_le_bytes([c[4], c[5]])))
        .collect();
    Ok(Items {
        inner,
        items,
        imports,
        marks,
    })
}
