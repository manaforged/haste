use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::body::sig::ImplHeader;
use crate::names::{Def, ItemRef, ModRef, Sym, Symbols};
use crate::outline::Outline;
use crate::sema::impls::{Holder, ImplRef, Index as ImplIndex};
use crate::sema::infer::{Fields, Form};
use crate::sema::lookups::{ActiveAttrs, LangMap, SegKey, TypeScope};
use crate::sema::substs::Substs;
use crate::sema::ty::Ty;

use super::SummaryKind;
use super::codec::{Cursor, Decoder, FileIndex, decode_prim, read_opt, read_str, read_vec};
use super::warm::{
    WARM_DEF_SHARDS, WARM_HDR_SHARDS, WARM_NON_SHARD_SECTIONS, WARM_SECTIONS, WarmFlights,
    WarmTimings,
};
use super::warm_abi::{read_impl_index, read_impl_ref, read_layout, read_repr_options};
use super::warm_body::read_body;
use super::warm_expr::read_opt_u32;
use super::warm_sig::{read_adt_def, read_generics, read_impl_header, read_signature};

fn read_mod_ref(cur: &mut Cursor<'_>) -> Result<ModRef> {
    Ok(ModRef {
        krate: cur.u32()?,
        module: cur.u32()?,
    })
}

fn read_def(dec: &Decoder<'_>, cur: &mut Cursor<'_>) -> Result<Def> {
    Ok(match cur.u8()? {
        0 => Def::Module(read_mod_ref(cur)?),
        1 => Def::Item(dec.read_item_ref(cur)?),
        2 => Def::Variant(dec.read_item_ref(cur)?, cur.u32()?),
        3 => Def::Primitive(decode_prim(cur.u8()?)?),
        _ => bail!("invalid def tag"),
    })
}

fn impl_index_has_body(idx: &ImplIndex) -> bool {
    idx.inherent
        .values()
        .flatten()
        .any(|r| matches!(r, ImplRef::Body(_)))
        || idx.traits.values().any(|b| {
            b.all
                .iter()
                .chain(&b.blanket)
                .chain(b.by_head.values().flatten())
                .any(|r| matches!(r, ImplRef::Body(_)))
        })
}

fn read_fields(dec: &Decoder<'_>, cur: &mut Cursor<'_>) -> Result<Vec<Vec<Ty>>> {
    read_vec(cur, |c| dec.read_tys(c))
}

#[derive(Default)]
struct SectionOut {
    reveals: Vec<(crate::sema::reach::Instance, Ty)>,
    checked_impls: Vec<(ImplRef, bool)>,
    summaries: Vec<(ItemRef, SummaryKind)>,
    body_heads: Vec<(Holder, Arc<ImplIndex>)>,
    body_headers: Vec<(ImplRef, Arc<ImplHeader>)>,
    body_forms: Vec<((Form, usize), Option<Fields>)>,
    body_impl_traits: Vec<(ImplRef, Option<ItemRef>)>,
    body_selections: Vec<((ItemRef, Ty, Substs), Option<(ImplRef, Substs)>)>,
}

fn finish_cursor(cur: &Cursor<'_>, bytes: &[u8]) -> Result<()> {
    if cur.pos != bytes.len() {
        bail!("trailing bytes in warm cache section");
    }
    Ok(())
}

fn decode_sec_defs(
    bytes: &[u8],
    dec: &Decoder<'_>,
    store: &crate::sema::Cache,
) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    let lk = &store.lookups;
    for _ in 0..cur.count()? {
        let k = cur.u32()?;
        let n = cur.count()?;
        let mut map = HashMap::with_capacity_and_hasher(n, Default::default());
        for _ in 0..n {
            map.insert((cur.u32()?, cur.u32()?), cur.u32()?);
        }
        lk.modules.insert(k, Arc::new(map));
    }
    for _ in 0..cur.count()? {
        let k = cur.u32()?;
        let n = cur.count()?;
        let mut map = HashMap::with_capacity_and_hasher(n, Default::default());
        for _ in 0..n {
            map.insert((cur.u32()?, read_opt_u32(&mut cur)?), cur.u32()?);
        }
        lk.owners.insert(k, Arc::new(map));
    }
    for _ in 0..cur.count()? {
        let m = read_mod_ref(&mut cur)?;
        let items: Vec<ItemRef> = read_vec(&mut cur, |c| dec.read_item_ref(c))?;
        lk.scope_traits.insert(m, Arc::from(items));
    }
    for _ in 0..cur.count()? {
        lk.trait_autos
            .insert(dec.read_item_ref(&mut cur)?, cur.u8()? != 0);
    }
    finish_cursor(&cur, bytes)?;
    Ok(SectionOut::default())
}

fn decode_sec_type_defs(
    bytes: &[u8],
    dec: &Decoder<'_>,
    syms: &[Sym],
    store: &crate::sema::Cache,
) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    let lk = &store.lookups;
    for _ in 0..cur.count()? {
        let scope: TypeScope = (
            read_mod_ref(&mut cur)?,
            read_opt(&mut cur, |c| dec.read_item_ref(c))?,
            read_opt(&mut cur, |c| dec.read_item_ref(c))?,
            cur.u8()?,
            read_opt_u32(&mut cur)?,
        );
        let seg_syms = read_vec(&mut cur, |c| {
            syms.get(c.u32()? as usize)
                .copied()
                .context("sym index out of bounds")
        })?;
        let def = read_def(dec, &mut cur)?;
        lk.type_defs
            .insert((scope, SegKey::from_slice(&seg_syms)), def);
    }
    finish_cursor(&cur, bytes)?;
    Ok(SectionOut::default())
}

fn decode_sec_heads(
    bytes: &[u8],
    dec: &Decoder<'_>,
    store: &crate::sema::Cache,
) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    let mut out = SectionOut::default();
    for _ in 0..cur.count()? {
        let holder = match cur.u8()? {
            0 => Holder::Crate(cur.u32()?),
            1 => Holder::Body(dec.read_item_ref(&mut cur)?),
            _ => bail!("invalid impl holder tag"),
        };
        let idx = read_impl_index(&mut cur, &|id| dec.item(id))?;
        if matches!(holder, Holder::Body(_)) || impl_index_has_body(&idx) {
            out.body_heads.push((holder, Arc::new(idx)));
        } else {
            store.lookups.heads.insert(holder, Arc::new(idx));
        }
    }
    finish_cursor(&cur, bytes)?;
    Ok(out)
}

fn decode_sec_sigs(
    bytes: &[u8],
    dec: &Decoder<'_>,
    store: &crate::sema::Cache,
) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        store
            .signatures
            .insert(item, Arc::new(read_signature(&mut cur)?));
    }
    finish_cursor(&cur, bytes)?;
    Ok(SectionOut::default())
}

fn decode_sec_adts(
    bytes: &[u8],
    dec: &Decoder<'_>,
    store: &crate::sema::Cache,
) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    let lk = &store.lookups;
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        store.adts.insert(item, Arc::new(read_adt_def(&mut cur)?));
    }
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        let vals: Vec<i128> = read_vec(&mut cur, |c| Ok(c.u128()? as i128))?;
        lk.discriminants.insert(item, Arc::from(vals));
    }
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        lk.reprs.insert(item, read_repr_options(&mut cur)?);
    }
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        let attrs = read_vec(&mut cur, |c| c.u32())?;
        lk.attrs.insert(item, ActiveAttrs::from_vec(attrs));
    }
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        let body = read_body(&mut cur)?;
        let gens = read_generics(&mut cur)?;
        lk.trait_defs.insert(item, Arc::new((body, gens)));
    }
    for _ in 0..cur.count()? {
        lk.verdicts.insert(dec.read_item_ref(&mut cur)?, ());
    }
    finish_cursor(&cur, bytes)?;
    Ok(SectionOut::default())
}

fn decode_sec_impl_meta(
    bytes: &[u8],
    dec: &Decoder<'_>,
    store: &crate::sema::Cache,
) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    let mut out = SectionOut::default();
    for _ in 0..cur.count()? {
        let r = read_impl_ref(&mut cur)?;
        let tr = read_opt(&mut cur, |c| dec.read_item_ref(c))?;
        if matches!(r, ImplRef::Body(_)) {
            out.body_impl_traits.push((r, tr));
        } else {
            store.lookups.impl_traits.insert(r, tr);
        }
    }
    let checked_n = cur.count()?;
    out.checked_impls.reserve(checked_n);
    for _ in 0..checked_n {
        out.checked_impls
            .push((read_impl_ref(&mut cur)?, cur.u8()? != 0));
    }
    finish_cursor(&cur, bytes)?;
    Ok(out)
}

fn decode_sec_headers(bytes: &[u8], store: &crate::sema::Cache) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    let mut out = SectionOut::default();
    for _ in 0..cur.count()? {
        let r = read_impl_ref(&mut cur)?;
        let hdr = Arc::new(read_impl_header(&mut cur)?);
        if matches!(r, ImplRef::Body(_)) {
            out.body_headers.push((r, hdr));
        } else {
            store.headers.insert(r, hdr);
        }
    }
    finish_cursor(&cur, bytes)?;
    Ok(out)
}

fn decode_sec_layouts(
    bytes: &[u8],
    dec: &Decoder<'_>,
    store: &crate::sema::Cache,
) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    let mut out = SectionOut::default();
    let lk = &store.lookups;
    for _ in 0..cur.count()? {
        let ty = dec.read_ty_ref(&mut cur)?;
        store.layouts.insert(ty, Arc::new(read_layout(&mut cur)?));
    }
    for _ in 0..cur.count()? {
        let ty = dec.read_ty_ref(&mut cur)?;
        store.tails.insert(ty, dec.read_ty_ref(&mut cur)?);
    }
    for _ in 0..cur.count()? {
        let ty = dec.read_ty_ref(&mut cur)?;
        lk.fields.insert(ty, Arc::from(read_fields(dec, &mut cur)?));
    }
    for _ in 0..cur.count()? {
        let form = match cur.u8()? {
            0 => Form::Adt(dec.read_item_ref(&mut cur)?),
            1 => Form::Fn(dec.read_item_ref(&mut cur)?),
            2 => Form::Impl(read_impl_ref(&mut cur)?),
            _ => bail!("invalid form tag"),
        };
        let arity = cur.u32()? as usize;
        let val: Option<Fields> = read_opt(&mut cur, |c| read_fields(dec, c))?.map(Arc::from);
        if matches!(form, Form::Impl(ImplRef::Body(_))) {
            out.body_forms.push(((form, arity), val));
        } else {
            lk.forms.insert((form, arity), val);
        }
    }
    finish_cursor(&cur, bytes)?;
    Ok(out)
}

fn decode_sec_selections(
    bytes: &[u8],
    dec: &Decoder<'_>,
    store: &crate::sema::Cache,
) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    let mut out = SectionOut::default();
    for _ in 0..cur.count()? {
        let tr = dec.read_item_ref(&mut cur)?;
        let ty = dec.read_ty_ref(&mut cur)?;
        let s = dec.substs(&mut cur)?;
        let picked = read_opt(&mut cur, |c| Ok((read_impl_ref(c)?, dec.substs(c)?)))?;
        if matches!(picked, Some((ImplRef::Body(_), _))) {
            out.body_selections.push(((tr, ty, s), picked));
        } else {
            store.lookups.selections.insert((tr, ty, s), picked);
        }
    }
    finish_cursor(&cur, bytes)?;
    Ok(out)
}

fn decode_sec_misc(
    bytes: &[u8],
    dec: &Decoder<'_>,
    syms: &[Sym],
    store: &crate::sema::Cache,
) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    let mut out = SectionOut::default();
    decode_misc_ty_facts(&mut cur, dec, store)?;
    decode_misc_reveals(&mut cur, dec, store, &mut out)?;
    decode_misc_trait_facts(&mut cur, dec, syms, store)?;
    finish_cursor(&cur, bytes)?;
    Ok(out)
}

fn decode_misc_ty_facts(
    cur: &mut Cursor<'_>,
    dec: &Decoder<'_>,
    store: &crate::sema::Cache,
) -> Result<()> {
    let lk = &store.lookups;
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(cur)?;
        let substs = dec.substs(cur)?;
        store.evaluated.insert((item, substs), cur.u128()?);
    }
    for _ in 0..cur.count()? {
        lk.drops.insert(dec.read_ty_ref(cur)?, cur.u8()? != 0);
    }
    for _ in 0..cur.count()? {
        lk.own_drops.insert(dec.read_ty_ref(cur)?, cur.u8()? != 0);
    }
    for _ in 0..cur.count()? {
        lk.unpins
            .insert(dec.read_ty_ref(cur)?, (cur.u8()? != 0, cur.u32()?));
    }
    for _ in 0..cur.count()? {
        lk.freezes
            .insert(dec.read_ty_ref(cur)?, (cur.u8()? != 0, cur.u32()?));
    }
    for _ in 0..cur.count()? {
        let ty = dec.read_ty_ref(cur)?;
        lk.glues.insert(ty, read_opt(cur, |c| dec.res(c))?);
    }
    for _ in 0..cur.count()? {
        let ty = dec.read_ty_ref(cur)?;
        lk.futures.insert(ty, dec.read_ty_ref(cur)?);
    }
    Ok(())
}

fn decode_misc_reveals(
    cur: &mut Cursor<'_>,
    dec: &Decoder<'_>,
    store: &crate::sema::Cache,
    out: &mut SectionOut,
) -> Result<()> {
    let reveals_n = cur.count()?;
    out.reveals.reserve(reveals_n);
    for _ in 0..reveals_n {
        let item = dec.read_item_ref(cur)?;
        let s = dec.substs(cur)?;
        let ty = dec.read_ty_ref(cur)?;
        out.reveals.push(((item, s), ty));
    }
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(cur)?;
        let s = dec.substs(cur)?;
        let ty = dec.read_ty_ref(cur)?;
        store.lookups.provisional_rets.insert((item, s), ty);
    }
    Ok(())
}

fn decode_misc_trait_facts(
    cur: &mut Cursor<'_>,
    dec: &Decoder<'_>,
    syms: &[Sym],
    store: &crate::sema::Cache,
) -> Result<()> {
    let lk = &store.lookups;
    for _ in 0..cur.count()? {
        let tr = dec.read_item_ref(cur)?;
        let ty = dec.read_ty_ref(cur)?;
        let a1 = dec.substs(cur)?;
        let sym = syms
            .get(cur.u32()? as usize)
            .copied()
            .context("sym index out of bounds")?;
        let a2 = dec.substs(cur)?;
        let out_ty = read_opt(cur, |c| dec.read_ty_ref(c))?;
        lk.projections.insert((tr, ty, a1, sym, a2), out_ty);
    }
    for _ in 0..cur.count()? {
        let m = dec.read_item_ref(cur)?;
        let s = dec.substs(cur)?;
        lk.trait_calls.insert((m, s), dec.res(cur)?);
    }
    for _ in 0..cur.count()? {
        let tr = dec.read_item_ref(cur)?;
        let ty = dec.read_ty_ref(cur)?;
        let a1 = dec.substs(cur)?;
        let assoc = dec.read_item_ref(cur)?;
        let a2 = dec.substs(cur)?;
        let val = dec.read_ty_ref(cur)?;
        lk.assoc_types.insert((tr, ty, a1, assoc, a2), val);
    }
    for _ in 0..cur.count()? {
        let tr = dec.read_item_ref(cur)?;
        let ty = dec.read_ty_ref(cur)?;
        lk.auto_traits.insert((tr, ty), cur.u8()? != 0);
    }
    Ok(())
}

fn decode_sec_summaries(bytes: &[u8], dec: &Decoder<'_>) -> Result<SectionOut> {
    let mut cur = Cursor::new(bytes);
    let n = cur.count()?;
    let mut out = SectionOut::default();
    out.summaries.reserve(n);
    for _ in 0..n {
        let item = dec.read_item_ref(&mut cur)?;
        let kind = match cur.u8()? {
            0 => SummaryKind::Poly(Arc::new(dec.read_summary(&mut cur)?)),
            1 => dec.read_verdict(&mut cur)?,
            _ => bail!("invalid summary tag"),
        };
        out.summaries.push((item, kind));
    }
    finish_cursor(&cur, bytes)?;
    Ok(out)
}

fn decode_section(
    idx: usize,
    bytes: &[u8],
    dec: &Decoder<'_>,
    syms: &[Sym],
    store: &crate::sema::Cache,
) -> Result<SectionOut> {
    let defs_end = WARM_NON_SHARD_SECTIONS + WARM_DEF_SHARDS;
    let hdrs_end = defs_end + WARM_HDR_SHARDS;
    match idx {
        0 => decode_sec_defs(bytes, dec, store),
        1 => decode_sec_heads(bytes, dec, store),
        2 => decode_sec_sigs(bytes, dec, store),
        3 => decode_sec_adts(bytes, dec, store),
        4 => decode_sec_impl_meta(bytes, dec, store),
        5 => decode_sec_layouts(bytes, dec, store),
        6 => decode_sec_selections(bytes, dec, store),
        7 => decode_sec_misc(bytes, dec, syms, store),
        i if (WARM_NON_SHARD_SECTIONS..defs_end).contains(&i) => {
            decode_sec_type_defs(bytes, dec, syms, store)
        }
        i if (defs_end..hdrs_end).contains(&i) => decode_sec_headers(bytes, store),
        i if (hdrs_end..WARM_SECTIONS).contains(&i) => decode_sec_summaries(bytes, dec),
        _ => bail!("invalid warm cache section index"),
    }
}

pub(super) fn decode_payload(
    bytes: &[u8],
    store: &mut crate::sema::Cache,
    outline: &Outline,
    symbols: &Symbols,
    allow_lang_map: bool,
    settled_fixed: &dyn Fn(u32) -> bool,
) -> Result<WarmFlights> {
    let t_start = std::time::Instant::now();
    let mut cur = Cursor::new(bytes);
    let item_count = cur.count()?;
    let item_bytes = cur.take(item_count.checked_mul(12).context("warm items overflow")?)?;
    let mut items = Vec::with_capacity(item_count);
    for c in item_bytes.chunks_exact(12) {
        items.push(ItemRef {
            krate: crate::substrate::read_u32(c, 0),
            file: crate::substrate::read_u32(c, 4),
            item: crate::substrate::read_u32(c, 8),
        });
    }
    let ty_count = cur.count()?;
    let mut tys = Vec::with_capacity(ty_count);
    for _ in 0..ty_count {
        tys.push(super::codec::read_ty(&mut cur, &store.types, &items, &tys)?);
    }
    let sym_count = cur.count()?;
    let mut syms = Vec::with_capacity(sym_count);
    for _ in 0..sym_count {
        syms.push(symbols.intern(read_str(&mut cur)?));
    }
    let dec = Decoder::new(&store.types, &items, &tys);

    let file_n = cur.count()?;
    let file_bytes = cur.take(file_n.checked_mul(16).context("warm file map overflow")?)?;
    let mut file_map: HashMap<u64, (u32, u32)> =
        HashMap::with_capacity_and_hasher(file_n, Default::default());
    for c in file_bytes.chunks_exact(16) {
        let key = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        file_map.insert(
            key,
            (
                crate::substrate::read_u32(c, 8),
                crate::substrate::read_u32(c, 12),
            ),
        );
    }
    for (krate, tree) in outline.crates.iter().enumerate() {
        if tree.env.fixed && !settled_fixed(krate as u32) {
            for file in 0..tree.files.len() as u32 {
                if let Some(key) = crate::substrate::file_key(outline, krate as u32, file) {
                    file_map.entry(key).or_insert((krate as u32, file));
                }
            }
        }
    }
    let _ = store
        .lookups
        .file_index
        .set(Arc::new(FileIndex::from_files(file_map)));

    if let Some(map) = read_opt(&mut cur, |c| {
        let count = c.count()?;
        let mut map: HashMap<ItemRef, Box<str>> =
            HashMap::with_capacity_and_hasher(count, Default::default());
        for _ in 0..count {
            let item = dec.read_item_ref(c)?;
            let name = read_str(c)?.to_owned().into_boxed_str();
            map.insert(item, name);
        }
        Ok(map)
    })? && allow_lang_map
    {
        let _ = store
            .lookups
            .lang_items
            .set(Arc::new(LangMap::from_map(map)));
    }

    if let Some(prims) = read_opt(&mut cur, |c| read_vec(c, |c2| c2.u32()))? {
        *crate::sema::write_lock(&store.lookups.state.primitives) = Some(Arc::from(prims));
    }

    let sec_count = cur.count()?;
    if sec_count != WARM_SECTIONS {
        bail!("unexpected warm section count {sec_count}");
    }
    let mut section_bytes = [0u32; WARM_SECTIONS];
    let mut dir = [(0usize, 0usize); WARM_SECTIONS];
    let mut max_end = 0usize;
    for i in 0..WARM_SECTIONS {
        let off = cur.u32()? as usize;
        let len = cur.u32()? as usize;
        let end = off
            .checked_add(len)
            .context("warm cache section offset overflow")?;
        max_end = max_end.max(end);
        section_bytes[i] = len as u32;
        dir[i] = (off, end);
    }
    let sec_blob = cur.take(max_end)?;
    if cur.pos != bytes.len() {
        bail!("trailing bytes in warm cache payload");
    }
    let slices: Vec<&[u8]> = dir
        .iter()
        .map(|&(start, end)| {
            sec_blob
                .get(start..end)
                .context("warm cache section out of bounds")
        })
        .collect::<Result<Vec<_>>>()?;

    let prelude_us = t_start.elapsed().as_micros() as u64;
    let t_secs = std::time::Instant::now();
    let workers = crate::jobs().min(WARM_SECTIONS);
    let outs: Vec<Result<SectionOut>> = if workers <= 1 || sec_blob.len() < 16 * 1024 {
        slices
            .iter()
            .enumerate()
            .map(|(i, s)| decode_section(i, s, &dec, &syms, store))
            .collect()
    } else {
        let mut order: Vec<usize> = (0..WARM_SECTIONS).collect();
        order.sort_unstable_by(|&a, &b| slices[b].len().cmp(&slices[a].len()));
        let order_ref = &order;
        let next = AtomicUsize::new(0);
        let shared_store: &crate::sema::Cache = store;
        crate::sync::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut local = Vec::new();
                        loop {
                            let pos = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&i) = order_ref.get(pos) else {
                                break;
                            };
                            local.push(decode_section(i, slices[i], &dec, &syms, shared_store));
                        }
                        local
                    })
                })
                .collect();
            let mut merged = Vec::with_capacity(WARM_SECTIONS);
            for h in handles {
                match h.join() {
                    Ok(chunk) => merged.extend(chunk),
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            }
            merged
        })
    };

    let mut flights = WarmFlights {
        reveals: Vec::new(),
        checked_impls: Vec::new(),
        summaries: Vec::new(),
        body_heads: Vec::new(),
        body_headers: Vec::new(),
        body_forms: Vec::new(),
        body_impl_traits: Vec::new(),
        body_selections: Vec::new(),
        timings: WarmTimings {
            payload_bytes: bytes.len(),
            item_count: items.len(),
            ty_count: tys.len(),
            sym_count: syms.len(),
            prelude_us,
            sections_us: 0,
            section_bytes,
        },
    };
    for res in outs {
        let part = res?;
        flights.reveals.extend(part.reveals);
        flights.checked_impls.extend(part.checked_impls);
        flights.summaries.extend(part.summaries);
        flights.body_heads.extend(part.body_heads);
        flights.body_headers.extend(part.body_headers);
        flights.body_forms.extend(part.body_forms);
        flights.body_impl_traits.extend(part.body_impl_traits);
        flights.body_selections.extend(part.body_selections);
    }
    flights.timings.sections_us = t_secs.elapsed().as_micros() as u64;
    Ok(flights)
}
