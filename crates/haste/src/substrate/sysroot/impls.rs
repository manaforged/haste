use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::macros::Compilation;
use crate::names::{ItemRef, PRIMITIVES};
use crate::sema::impls::{Head, ImplRef, Index, TraitBucket};
use crate::sema::summary::codec::{Cursor, write_u32};
use crate::sema::ty::Mut;
use crate::sema::{Cache, Context};

use super::tokens::{read_u32s_map, write_usize};
use super::wire::Tables;

pub(super) fn build_module_heads(
    compilation: &Compilation,
    crates: &[usize],
) -> Vec<Option<Index>> {
    let mut cx = Context::new(&compilation.outline, &compilation.names, Cache::default());
    crates
        .iter()
        .map(|&krate| {
            let index = cx.index_module_impls(krate as u32);
            index.complete().then_some(index)
        })
        .collect()
}

fn write_item_ref(out: &mut Vec<u8>, item: ItemRef, tables: &Tables<'_>) -> Option<()> {
    if item.outer().is_some() {
        return None;
    }
    write_u32(out, tables.local_krate(item.krate as usize)?);
    write_u32(out, item.file);
    write_u32(out, item.item);
    Some(())
}

fn read_item_ref(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<ItemRef> {
    let local = cur.u32()? as usize;
    let krate = *remap.get(local).context("impl item crate out of bounds")?;
    let file = cur.u32()?;
    let item = cur.u32()?;
    Ok(ItemRef { krate, file, item })
}

fn write_head(out: &mut Vec<u8>, head: Head, tables: &Tables<'_>) -> Option<()> {
    match head {
        Head::Prim(prim) => {
            out.push(0);
            out.push(PRIMITIVES.iter().position(|&(_, p)| p == prim)? as u8);
        }
        Head::Adt(item) => {
            out.push(1);
            write_item_ref(out, item, tables)?;
        }
        Head::Ref(Mut::Not) => out.push(2),
        Head::Ref(Mut::Mut) => out.push(3),
        Head::Ptr(Mut::Not) => out.push(4),
        Head::Ptr(Mut::Mut) => out.push(5),
        Head::Slice => out.push(6),
        Head::Array => out.push(7),
        Head::Tuple => out.push(8),
        Head::Never => out.push(9),
        Head::Dyn(item) => {
            out.push(10);
            write_item_ref(out, item, tables)?;
        }
    }
    Some(())
}

fn read_head(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Head> {
    Ok(match cur.u8()? {
        0 => Head::Prim(
            PRIMITIVES
                .get(cur.u8()? as usize)
                .map(|&(_, p)| p)
                .context("invalid impl head primitive")?,
        ),
        1 => Head::Adt(read_item_ref(cur, remap)?),
        2 => Head::Ref(Mut::Not),
        3 => Head::Ref(Mut::Mut),
        4 => Head::Ptr(Mut::Not),
        5 => Head::Ptr(Mut::Mut),
        6 => Head::Slice,
        7 => Head::Array,
        8 => Head::Tuple,
        9 => Head::Never,
        10 => Head::Dyn(read_item_ref(cur, remap)?),
        _ => bail!("invalid impl head tag"),
    })
}

fn write_impl_refs(out: &mut Vec<u8>, refs: &[ImplRef], krate: u32) -> Option<()> {
    write_usize(out, refs.len())?;
    for &block in refs {
        let ImplRef::Module {
            krate: block_krate,
            index,
        } = block
        else {
            return None;
        };
        if block_krate != krate {
            return None;
        }
        write_u32(out, index);
    }
    Some(())
}

fn read_impl_refs(cur: &mut Cursor<'_>, krate: u32) -> Result<Vec<ImplRef>> {
    read_u32s_map(cur, |index| ImplRef::Module { krate, index })
}

fn encode_index(out: &mut Vec<u8>, index: &Index, krate: u32, tables: &Tables<'_>) -> Option<()> {
    write_usize(out, index.inherent.len())?;
    for (&head, refs) in &index.inherent {
        write_head(out, head, tables)?;
        write_impl_refs(out, refs, krate)?;
    }
    write_usize(out, index.traits.len())?;
    for (&of, bucket) in &index.traits {
        write_item_ref(out, of, tables)?;
        write_impl_refs(out, &bucket.all, krate)?;
        write_impl_refs(out, &bucket.blanket, krate)?;
        write_usize(out, bucket.by_head.len())?;
        for (&head, refs) in &bucket.by_head {
            write_head(out, head, tables)?;
            write_impl_refs(out, refs, krate)?;
        }
    }
    write_usize(out, index.negative.len())?;
    for (&of, heads) in &index.negative {
        write_item_ref(out, of, tables)?;
        write_usize(out, heads.len())?;
        for &head in heads {
            write_head(out, head, tables)?;
        }
    }
    Some(())
}

pub(super) fn write_impl_index(
    out: &mut Vec<u8>,
    index: Option<&Index>,
    krate: u32,
    tables: &Tables<'_>,
) -> Option<()> {
    let Some(index) = index else {
        out.push(0);
        return Some(());
    };
    let mut tmp = Vec::new();
    if encode_index(&mut tmp, index, krate, tables).is_none() {
        out.push(0);
        return Some(());
    }
    out.push(1);
    out.extend_from_slice(&tmp);
    Some(())
}

pub(super) fn read_impl_index(
    cur: &mut Cursor<'_>,
    krate: u32,
    remap: &[u32],
) -> Result<Option<Arc<Index>>> {
    match cur.u8()? {
        0 => return Ok(None),
        1 => {}
        _ => bail!("invalid impl index presence tag"),
    }
    let inherent_len = cur.count()?;
    let mut inherent = HashMap::with_capacity_and_hasher(inherent_len, Default::default());
    for _ in 0..inherent_len {
        let head = read_head(cur, remap)?;
        let refs = read_impl_refs(cur, krate)?;
        inherent.insert(head, refs);
    }
    let traits_len = cur.count()?;
    let mut traits = HashMap::with_capacity_and_hasher(traits_len, Default::default());
    for _ in 0..traits_len {
        let of = read_item_ref(cur, remap)?;
        let all = read_impl_refs(cur, krate)?;
        let blanket = read_impl_refs(cur, krate)?;
        let by_head_len = cur.count()?;
        let mut by_head = HashMap::with_capacity_and_hasher(by_head_len, Default::default());
        for _ in 0..by_head_len {
            let head = read_head(cur, remap)?;
            let refs = read_impl_refs(cur, krate)?;
            by_head.insert(head, refs);
        }
        traits.insert(
            of,
            TraitBucket {
                all,
                blanket,
                by_head,
            },
        );
    }
    let negative_len = cur.count()?;
    let mut negative = HashMap::with_capacity_and_hasher(negative_len, Default::default());
    for _ in 0..negative_len {
        let of = read_item_ref(cur, remap)?;
        let heads_len = cur.count()?;
        let mut heads = Vec::with_capacity(heads_len);
        for _ in 0..heads_len {
            heads.push(read_head(cur, remap)?);
        }
        negative.insert(of, heads);
    }
    Ok(Some(Arc::new(Index::from_parts(
        inherent, traits, negative,
    ))))
}
