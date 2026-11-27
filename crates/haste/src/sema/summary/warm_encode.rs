use std::cell::RefCell;

use rustc_abi::Variants;
use rustc_hash::FxHashMap as HashMap;

use crate::names::{Def, ItemRef, ModRef, Sym, Symbols};
use crate::outline::Outline;
use crate::sema::impls::Holder;
use crate::sema::infer::Form;
use crate::sema::substs::Substs;
use crate::sema::table::{Flight, Table};
use crate::sema::ty::Ty;

use super::SummaryKind;
use super::codec::{Encoder, encode_prim, write_opt, write_str, write_u32, write_u64, write_u128};
use super::warm::{
    WARM_DEF_SHARDS, WARM_HDR_SHARDS, WARM_NON_SHARD_SECTIONS, WARM_SECTIONS, WARM_SUM_SHARDS,
    WarmFilter,
};
use super::warm_abi::{write_impl_index, write_impl_ref, write_layout, write_repr_options};
use super::warm_body::write_body;
use super::warm_expr::{write_opt_u32, write_slice};
use super::warm_sig::{write_adt_def, write_generics, write_impl_header, write_signature};

mod facts;

use facts::{encode_trait_facts, encode_ty_facts};

struct SymPool {
    map: HashMap<Sym, u32>,
    list: Vec<Sym>,
}

impl SymPool {
    fn new() -> Self {
        Self {
            map: HashMap::default(),
            list: Vec::new(),
        }
    }

    fn id(&mut self, sym: Sym) -> u32 {
        if let Some(&idx) = self.map.get(&sym) {
            return idx;
        }
        let idx = self.list.len() as u32;
        self.list.push(sym);
        self.map.insert(sym, idx);
        idx
    }

    fn write(&self, out: &mut Vec<u8>, symbols: &Symbols) -> Option<()> {
        write_u32(out, self.list.len() as u32);
        for &sym in &self.list {
            write_str(out, symbols.text(sym))?;
        }
        Some(())
    }
}

fn snap_table<K: Clone, V: Clone>(
    table: &Table<K, V>,
    mut ok: impl FnMut(&K, &V) -> bool,
) -> Vec<(K, V)> {
    let mut out = Vec::new();
    table.retain(|k, v| {
        if ok(k, v) {
            out.push((k.clone(), v.clone()));
        }
        true
    });
    out
}

fn snap_flight<K: Clone + Eq + std::hash::Hash, V: Clone>(
    flight: &Flight<K, V>,
    mut ok: impl FnMut(&K, &V) -> bool,
) -> Vec<(K, V)> {
    let mut out = Vec::new();
    flight.retain_entries(|k, v| {
        if ok(k, v) {
            out.push((k.clone(), v.clone()));
        }
        true
    });
    out
}

fn write_mod_ref(out: &mut Vec<u8>, m: ModRef) {
    write_u32(out, m.krate);
    write_u32(out, m.module);
}

fn write_def<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    enc: &mut Encoder<'_, F>,
    out: &mut Vec<u8>,
    def: Def,
) -> Option<()> {
    match def {
        Def::Module(m) => {
            out.push(0);
            write_mod_ref(out, m);
        }
        Def::Item(i) => {
            out.push(1);
            write_u32(out, enc.item_id(i)?);
        }
        Def::Variant(i, v) => {
            out.push(2);
            write_u32(out, enc.item_id(i)?);
            write_u32(out, v);
        }
        Def::Primitive(p) => {
            out.push(3);
            out.push(encode_prim(p)?);
        }
    }
    Some(())
}

fn write_fields<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    enc: &mut Encoder<'_, F>,
    out: &mut Vec<u8>,
    fields: &[Vec<Ty>],
) -> Option<()> {
    write_u32(out, fields.len() as u32);
    for variant in fields {
        enc.write_tys(out, variant)?;
    }
    Some(())
}

fn write_table_entries<T>(
    out: &mut Vec<u8>,
    entries: &[T],
    mut encode_one: impl FnMut(&T, &mut Vec<u8>) -> Option<()>,
) {
    let count_at = out.len();
    write_u32(out, 0);
    let mut written = 0u32;
    let mut scratch = Vec::new();
    for entry in entries {
        scratch.clear();
        if encode_one(entry, &mut scratch).is_some() {
            out.extend_from_slice(&scratch);
            written += 1;
        }
    }
    out[count_at..count_at + 4].copy_from_slice(&written.to_le_bytes());
}

struct Snap<'a> {
    store: &'a crate::sema::Cache,
    outline: &'a Outline,
    filter: &'a WarmFilter<'a>,
    memo: RefCell<Vec<u8>>,
}

impl Snap<'_> {
    fn ty(&self, t: Ty) -> bool {
        self.filter
            .ty(&self.store.types, t, &mut self.memo.borrow_mut())
    }

    fn substs(&self, s: Substs) -> bool {
        self.filter
            .substs(&self.store.types, s, &mut self.memo.borrow_mut())
    }
}

struct Sink<'e, F> {
    enc: Encoder<'e, F>,
    syms: SymPool,
    secs: [Vec<u8>; WARM_SECTIONS],
}

pub(super) fn encode_payload(
    store: &crate::sema::Cache,
    outline: &Outline,
    symbols: &Symbols,
    filter: &WarmFilter<'_>,
) -> Option<(Vec<u8>, Vec<ItemRef>, u32)> {
    let mut allow = |_: &Outline, item: ItemRef| filter.item(item).then_some(0);
    let snap = Snap {
        store,
        outline,
        filter,
        memo: RefCell::new(Vec::new()),
    };
    let mut sink = Sink {
        enc: Encoder::new_raw(outline, &store.types, &mut allow),
        syms: SymPool::new(),
        secs: std::array::from_fn(|_| Vec::new()),
    };
    let prelude = encode_prelude(&snap, &mut sink.enc)?;
    let mut raw_count = 0usize;
    raw_count += encode_scopes(&snap, &mut sink);
    raw_count += encode_type_defs(&snap, &mut sink);
    raw_count += encode_item_tables(&snap, &mut sink);
    raw_count += encode_headers(&snap, &mut sink);
    raw_count += encode_impl_meta(&snap, &mut sink);
    raw_count += encode_layouts(&snap, &mut sink);
    raw_count += encode_selections(&snap, &mut sink);
    raw_count += encode_ty_facts(&snap, &mut sink);
    raw_count += encode_trait_facts(&snap, &mut sink);
    raw_count += encode_summaries(&snap, &mut sink);
    let (out, items) = assemble_payload(sink, &prelude, symbols)?;
    Some((out, items, raw_count as u32))
}

fn encode_prelude<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    enc: &mut Encoder<'_, F>,
) -> Option<Vec<u8>> {
    let (store, outline, filter) = (snap.store, snap.outline, snap.filter);
    let mut prelude = Vec::new();
    let file_entries: Vec<(u64, u32, u32)> = match store.lookups.file_index.get() {
        Some(idx) => idx
            .iter()
            .filter(|&(_, &(k, f))| filter.file(k, f))
            .map(|(&key, &(k, f))| (key, k, f))
            .collect(),
        None => (0..outline.crates.len() as u32)
            .filter(|&k| filter.krate(k))
            .flat_map(|k| {
                let lim = filter.pre_files(k);
                (0..lim)
                    .filter_map(move |f| Some((crate::substrate::file_key(outline, k, f)?, k, f)))
            })
            .collect(),
    };
    write_slice(&mut prelude, &file_entries, |o, &(key, k, f)| {
        write_u64(o, key);
        write_u32(o, k);
        write_u32(o, f);
    });

    let langs: Option<Vec<(ItemRef, String)>> = store.lookups.lang_items.get().map(|lm| {
        lm.map
            .iter()
            .filter(|&(&item, _)| filter.item(item))
            .map(|(&item, name)| (item, name.as_ref().to_owned()))
            .collect()
    });
    write_opt(&mut prelude, langs.as_ref(), |o, entries| {
        write_table_entries(o, entries, |(item, name), b| {
            write_u32(b, enc.item_id(*item)?);
            write_str(b, name)
        });
        Some(())
    })?;

    let prims: Option<Vec<u32>> = super::super::read_lock(&store.lookups.state.primitives)
        .as_ref()
        .map(|p| p.iter().copied().filter(|&k| filter.krate(k)).collect());
    write_opt(&mut prelude, prims.as_ref(), |o, p| {
        write_slice(o, p, |o2, v| write_u32(o2, *v));
        Some(())
    })?;
    Some(prelude)
}

fn encode_scopes<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    sink: &mut Sink<'_, F>,
) -> usize {
    let (lk, filter) = (&snap.store.lookups, snap.filter);
    let Sink { enc, secs, .. } = sink;
    let modules = snap_table(&lk.modules, |&k, _| filter.krate(k));
    write_slice(&mut secs[0], &modules, |o, (k, map)| {
        write_u32(o, *k);
        write_u32(o, map.len() as u32);
        for (&(a, b), &v) in map.iter() {
            write_u32(o, a);
            write_u32(o, b);
            write_u32(o, v);
        }
    });

    let owners = snap_table(&lk.owners, |&k, _| filter.krate(k));
    write_slice(&mut secs[0], &owners, |o, (k, map)| {
        write_u32(o, *k);
        write_u32(o, map.len() as u32);
        for (&(a, b), &v) in map.iter() {
            write_u32(o, a);
            write_opt_u32(o, b);
            write_u32(o, v);
        }
    });

    let scope_traits = snap_table(&lk.scope_traits, |&m, _| filter.mod_ref(m));
    write_table_entries(&mut secs[0], &scope_traits, |(m, items), b| {
        write_mod_ref(b, *m);
        write_u32(b, items.len() as u32);
        for &item in items.iter() {
            write_u32(b, enc.item_id(item)?);
        }
        Some(())
    });

    let trait_autos = snap_table(&lk.trait_autos, |k, _| filter.item(*k));
    write_table_entries(&mut secs[0], &trait_autos, |(k, v), b| {
        write_u32(b, enc.item_id(*k)?);
        b.push(u8::from(*v));
        Some(())
    });
    modules.len() + owners.len() + scope_traits.len() + trait_autos.len()
}

fn encode_type_defs<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    sink: &mut Sink<'_, F>,
) -> usize {
    let filter = snap.filter;
    let Sink { enc, syms, secs } = sink;
    let type_defs = snap_table(&snap.store.lookups.type_defs, |&(sc, _), &def| {
        filter.mod_ref(sc.0)
            && sc.1.is_none_or(|i| filter.item(i))
            && sc.2.is_none_or(|i| filter.item(i))
            && filter.def(&def)
    });
    let mut seg_buf = Vec::new();
    let def_base = WARM_NON_SHARD_SECTIONS;
    let def_chunk = type_defs.len().div_ceil(WARM_DEF_SHARDS).max(1);
    for shard in 0..WARM_DEF_SHARDS {
        let start = (shard * def_chunk).min(type_defs.len());
        let end = ((shard + 1) * def_chunk).min(type_defs.len());
        write_table_entries(
            &mut secs[def_base + shard],
            &type_defs[start..end],
            |((sc, seg), def), b| {
                write_mod_ref(b, sc.0);
                write_opt(b, sc.1.as_ref(), |o, i| {
                    write_u32(o, enc.item_id(*i)?);
                    Some(())
                })?;
                write_opt(b, sc.2.as_ref(), |o, i| {
                    write_u32(o, enc.item_id(*i)?);
                    Some(())
                })?;
                b.push(sc.3);
                write_opt_u32(b, sc.4);
                seg_buf.clear();
                seg.push_syms(&mut seg_buf);
                write_u32(b, seg_buf.len() as u32);
                for &s in &seg_buf {
                    write_u32(b, syms.id(s));
                }
                write_def(enc, b, *def)
            },
        );
    }
    type_defs.len()
}

fn encode_item_tables<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    sink: &mut Sink<'_, F>,
) -> usize {
    let (store, lk, filter) = (snap.store, &snap.store.lookups, snap.filter);
    let Sink { enc, secs, .. } = sink;
    let heads = snap_table(&lk.heads, |h, idx| filter.holder(*h) && idx.complete());
    write_table_entries(&mut secs[1], &heads, |(h, idx), b| {
        match *h {
            Holder::Crate(k) => {
                b.push(0);
                write_u32(b, k);
            }
            Holder::Body(root) => {
                b.push(1);
                write_u32(b, enc.item_id(root)?);
            }
        }
        write_impl_index(b, idx, &mut |i| enc.item_id(i), &|r| filter.impl_ref(r))
    });

    let sigs = snap_table(&store.signatures, |k, _| filter.item(*k));
    write_table_entries(&mut secs[2], &sigs, |(k, s), b| {
        write_u32(b, enc.item_id(*k)?);
        write_signature(b, s);
        Some(())
    });

    let adts = snap_table(&store.adts, |k, _| filter.item(*k));
    write_table_entries(&mut secs[3], &adts, |(k, a), b| {
        write_u32(b, enc.item_id(*k)?);
        write_adt_def(b, a);
        Some(())
    });

    let discrs = snap_table(&lk.discriminants, |k, _| filter.item(*k));
    write_table_entries(&mut secs[3], &discrs, |(k, d), b| {
        write_u32(b, enc.item_id(*k)?);
        write_slice(b, d, |o, v| write_u128(o, *v as u128));
        Some(())
    });

    let reprs = snap_table(&lk.reprs, |k, _| filter.item(*k));
    write_table_entries(&mut secs[3], &reprs, |(k, r), b| {
        write_u32(b, enc.item_id(*k)?);
        write_repr_options(b, r);
        Some(())
    });

    let attrs = snap_table(&lk.attrs, |k, _| filter.item(*k));
    write_table_entries(&mut secs[3], &attrs, |(k, a), b| {
        write_u32(b, enc.item_id(*k)?);
        write_slice(b, a.as_slice(), |o, v| write_u32(o, *v));
        Some(())
    });

    let trait_defs = snap_table(&lk.trait_defs, |k, _| filter.item(*k));
    write_table_entries(&mut secs[3], &trait_defs, |(k, tg), b| {
        write_u32(b, enc.item_id(*k)?);
        write_body(b, &tg.0);
        write_generics(b, &tg.1);
        Some(())
    });

    let verdicts = snap_table(&lk.verdicts, |k, _| filter.item(*k));
    write_table_entries(&mut secs[3], &verdicts, |(k, _), b| {
        write_u32(b, enc.item_id(*k)?);
        Some(())
    });
    heads.len()
        + sigs.len()
        + adts.len()
        + discrs.len()
        + reprs.len()
        + attrs.len()
        + trait_defs.len()
        + verdicts.len()
}

fn encode_headers<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    sink: &mut Sink<'_, F>,
) -> usize {
    let filter = snap.filter;
    let Sink { enc, secs, .. } = sink;
    let headers = snap_table(&snap.store.headers, |k, _| filter.impl_ref(*k));
    let hdr_base = WARM_NON_SHARD_SECTIONS + WARM_DEF_SHARDS;
    let hdr_chunk = headers.len().div_ceil(WARM_HDR_SHARDS).max(1);
    for shard in 0..WARM_HDR_SHARDS {
        let start = (shard * hdr_chunk).min(headers.len());
        let end = ((shard + 1) * hdr_chunk).min(headers.len());
        write_table_entries(
            &mut secs[hdr_base + shard],
            &headers[start..end],
            |(k, h), b| {
                write_impl_ref(b, *k, &mut |i| enc.item_id(i))?;
                write_impl_header(b, h);
                Some(())
            },
        );
    }
    headers.len()
}

fn encode_impl_meta<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    sink: &mut Sink<'_, F>,
) -> usize {
    let (lk, filter) = (&snap.store.lookups, snap.filter);
    let Sink { enc, secs, .. } = sink;
    let impl_traits = snap_table(&lk.impl_traits, |k, v| {
        filter.impl_ref(*k) && v.is_none_or(|i| filter.item(i))
    });
    write_table_entries(&mut secs[4], &impl_traits, |(k, v), b| {
        write_impl_ref(b, *k, &mut |i| enc.item_id(i))?;
        write_opt(b, v.as_ref(), |o, i| {
            write_u32(o, enc.item_id(*i)?);
            Some(())
        })
    });

    let checked_impls = snap_flight(&lk.checked_impls, |k, _| filter.impl_ref(*k));
    write_table_entries(&mut secs[4], &checked_impls, |(k, v), b| {
        write_impl_ref(b, *k, &mut |i| enc.item_id(i))?;
        b.push(u8::from(*v));
        Some(())
    });
    impl_traits.len() + checked_impls.len()
}

fn encode_layouts<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    sink: &mut Sink<'_, F>,
) -> usize {
    let (store, lk, filter) = (snap.store, &snap.store.lookups, snap.filter);
    let Sink { enc, secs, .. } = sink;
    let layouts = snap_table(&store.layouts, |&t, l| {
        !matches!(l.variants, Variants::Multiple { .. }) && snap.ty(t)
    });
    write_table_entries(&mut secs[5], &layouts, |(t, l), b| {
        write_u32(b, enc.ty_id(*t)?);
        write_layout(b, l)
    });

    let tails = snap_table(&store.tails, |&t, &v| snap.ty(t) && snap.ty(v));
    write_table_entries(&mut secs[5], &tails, |(t, v), b| {
        write_u32(b, enc.ty_id(*t)?);
        write_u32(b, enc.ty_id(*v)?);
        Some(())
    });

    let fields = snap_table(&lk.fields, |&t, _| snap.ty(t));
    write_table_entries(&mut secs[5], &fields, |(t, f), b| {
        write_u32(b, enc.ty_id(*t)?);
        write_fields(enc, b, f)
    });

    let forms = snap_table(&lk.forms, |&(f, _), _| match f {
        Form::Adt(i) | Form::Fn(i) => filter.item(i),
        Form::Impl(b) => filter.impl_ref(b),
    });
    write_table_entries(&mut secs[5], &forms, |((f, arity), val), b| {
        match *f {
            Form::Adt(i) => {
                b.push(0);
                write_u32(b, enc.item_id(i)?);
            }
            Form::Fn(i) => {
                b.push(1);
                write_u32(b, enc.item_id(i)?);
            }
            Form::Impl(bl) => {
                b.push(2);
                write_impl_ref(b, bl, &mut |i| enc.item_id(i))?;
            }
        }
        write_u32(b, *arity as u32);
        write_opt(b, val.as_ref(), |o, fld| write_fields(enc, o, fld))
    });
    layouts.len() + tails.len() + fields.len() + forms.len()
}

fn encode_selections<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    sink: &mut Sink<'_, F>,
) -> usize {
    let filter = snap.filter;
    let Sink { enc, secs, .. } = sink;
    let selections = snap_table(&snap.store.lookups.selections, |&(tr, ty, s), v| {
        filter.item(tr)
            && snap.ty(ty)
            && snap.substs(s)
            && v.is_none_or(|(bl, ss)| filter.impl_ref(bl) && snap.substs(ss))
    });
    write_table_entries(&mut secs[6], &selections, |((tr, ty, s), val), b| {
        write_u32(b, enc.item_id(*tr)?);
        write_u32(b, enc.ty_id(*ty)?);
        enc.write_substs(b, *s)?;
        write_opt(b, val.as_ref(), |o, &(bl, ss)| {
            write_impl_ref(o, bl, &mut |i| enc.item_id(i))?;
            enc.write_substs(o, ss)
        })
    });
    selections.len()
}

fn encode_summaries<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    sink: &mut Sink<'_, F>,
) -> usize {
    let filter = snap.filter;
    let Sink { enc, secs, .. } = sink;
    let summaries = snap_flight(&snap.store.lookups.summaries, |k, v| {
        filter.item(*k) && !matches!(v, SummaryKind::Error(_))
    });
    let sum_base = WARM_NON_SHARD_SECTIONS + WARM_DEF_SHARDS + WARM_HDR_SHARDS;
    let sum_chunk = summaries.len().div_ceil(WARM_SUM_SHARDS).max(1);
    for shard in 0..WARM_SUM_SHARDS {
        let start = (shard * sum_chunk).min(summaries.len());
        let end = ((shard + 1) * sum_chunk).min(summaries.len());
        write_table_entries(
            &mut secs[sum_base + shard],
            &summaries[start..end],
            |(k, kind), b| {
                write_u32(b, enc.item_id(*k)?);
                match kind {
                    SummaryKind::Poly(s) => {
                        b.push(0);
                        enc.write_summary(b, s)?;
                    }
                    SummaryKind::MonomorphicHints(_) | SummaryKind::MonomorphicFallback => {
                        b.push(1);
                        enc.write_verdict(b, kind)?;
                    }
                    SummaryKind::Error(_) => return None,
                }
                Some(())
            },
        );
    }
    summaries.len()
}

fn assemble_payload<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    sink: Sink<'_, F>,
    prelude: &[u8],
    symbols: &Symbols,
) -> Option<(Vec<u8>, Vec<ItemRef>)> {
    let Sink { enc, syms, secs } = sink;
    let (items, tys) = enc.finish_raw();
    let total_sec_bytes: usize = secs.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(
        items.len() * 12 + tys.bytes().len() + prelude.len() + total_sec_bytes + 256,
    );
    write_slice(&mut out, &items, |o, i| {
        write_u32(o, i.krate);
        write_u32(o, i.file);
        write_u32(o, i.item);
    });
    write_u32(&mut out, tys.count());
    out.extend_from_slice(tys.bytes());
    syms.write(&mut out, symbols)?;
    out.extend_from_slice(prelude);
    write_u32(&mut out, WARM_SECTIONS as u32);
    let mut offset = 0u32;
    for sec in &secs {
        let len = u32::try_from(sec.len()).ok()?;
        write_u32(&mut out, offset);
        write_u32(&mut out, len);
        offset = offset.checked_add(len)?;
    }
    for sec in &secs {
        out.extend_from_slice(sec);
    }
    Some((out, items))
}
