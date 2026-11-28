use std::sync::Arc;

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
use super::warm::WarmFlights;
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

pub(super) fn decode_payload(
    bytes: &[u8],
    store: &mut crate::sema::Cache,
    outline: &Outline,
    symbols: &Symbols,
    allow_lang_map: bool,
    settled_fixed: &dyn Fn(u32) -> bool,
) -> Result<WarmFlights> {
    let mut cur = Cursor::new(bytes);
    let items = read_vec(&mut cur, |c| {
        Ok(ItemRef {
            krate: c.u32()?,
            file: c.u32()?,
            item: c.u32()?,
        })
    })?;
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
    let sym_at = |id: u32| -> Result<Sym> {
        syms.get(id as usize)
            .copied()
            .context("sym index out of bounds")
    };
    let dec = Decoder::new(&store.types, &items, &tys);

    let mut file_map: HashMap<u64, (u32, u32)> = HashMap::default();
    let file_n = cur.count()?;
    file_map.reserve(file_n);
    for _ in 0..file_n {
        file_map.insert(cur.u64()?, (cur.u32()?, cur.u32()?));
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

    if let Some(pairs) = read_opt(&mut cur, |c| {
        read_vec(c, |c2| {
            Ok((
                dec.read_item_ref(c2)?,
                read_str(c2)?.to_owned().into_boxed_str(),
            ))
        })
    })? && allow_lang_map
    {
        let map: HashMap<ItemRef, Box<str>> = pairs.into_iter().collect();
        let _ = store
            .lookups
            .lang_items
            .set(Arc::new(LangMap::from_map(map)));
    }

    if let Some(prims) = read_opt(&mut cur, |c| read_vec(c, |c2| c2.u32()))? {
        *crate::sema::write_lock(&store.lookups.state.primitives) = Some(Arc::from(prims));
    }

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
    let mut body_heads: Vec<(Holder, Arc<ImplIndex>)> = Vec::new();
    for _ in 0..cur.count()? {
        let holder = match cur.u8()? {
            0 => Holder::Crate(cur.u32()?),
            1 => Holder::Body(dec.read_item_ref(&mut cur)?),
            _ => bail!("invalid impl holder tag"),
        };
        let idx = read_impl_index(&mut cur, &|id| dec.item(id))?;
        if matches!(holder, Holder::Body(_)) || impl_index_has_body(&idx) {
            body_heads.push((holder, Arc::new(idx)));
        } else {
            lk.heads.insert(holder, Arc::new(idx));
        }
    }
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        store
            .signatures
            .insert(item, Arc::new(read_signature(&mut cur)?));
    }
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        store.adts.insert(item, Arc::new(read_adt_def(&mut cur)?));
    }
    for _ in 0..cur.count()? {
        let ty = dec.read_ty_ref(&mut cur)?;
        store.layouts.insert(ty, Arc::new(read_layout(&mut cur)?));
    }
    for _ in 0..cur.count()? {
        let ty = dec.read_ty_ref(&mut cur)?;
        store.tails.insert(ty, dec.read_ty_ref(&mut cur)?);
    }
    let mut body_headers: Vec<(ImplRef, Arc<ImplHeader>)> = Vec::new();
    for _ in 0..cur.count()? {
        let r = read_impl_ref(&mut cur)?;
        let hdr = Arc::new(read_impl_header(&mut cur)?);
        if matches!(r, ImplRef::Body(_)) {
            body_headers.push((r, hdr));
        } else {
            store.headers.insert(r, hdr);
        }
    }
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        let substs = dec.substs(&mut cur)?;
        store.evaluated.insert((item, substs), cur.u128()?);
    }
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        let body = read_body(&mut cur)?;
        let gens = read_generics(&mut cur)?;
        lk.trait_defs.insert(item, Arc::new((body, gens)));
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
        lk.drops.insert(dec.read_ty_ref(&mut cur)?, cur.u8()? != 0);
    }
    for _ in 0..cur.count()? {
        lk.own_drops
            .insert(dec.read_ty_ref(&mut cur)?, cur.u8()? != 0);
    }
    for _ in 0..cur.count()? {
        lk.unpins
            .insert(dec.read_ty_ref(&mut cur)?, (cur.u8()? != 0, cur.u32()?));
    }
    for _ in 0..cur.count()? {
        lk.freezes
            .insert(dec.read_ty_ref(&mut cur)?, (cur.u8()? != 0, cur.u32()?));
    }
    for _ in 0..cur.count()? {
        let ty = dec.read_ty_ref(&mut cur)?;
        lk.glues.insert(ty, read_opt(&mut cur, |c| dec.res(c))?);
    }
    for _ in 0..cur.count()? {
        let ty = dec.read_ty_ref(&mut cur)?;
        lk.fields
            .insert(ty, Arc::from(read_fields(&dec, &mut cur)?));
    }
    let mut body_forms: Vec<((Form, usize), Option<Fields>)> = Vec::new();
    for _ in 0..cur.count()? {
        let form = match cur.u8()? {
            0 => Form::Adt(dec.read_item_ref(&mut cur)?),
            1 => Form::Fn(dec.read_item_ref(&mut cur)?),
            2 => Form::Impl(read_impl_ref(&mut cur)?),
            _ => bail!("invalid form tag"),
        };
        let arity = cur.u32()? as usize;
        let val: Option<Fields> = read_opt(&mut cur, |c| read_fields(&dec, c))?.map(Arc::from);
        if matches!(form, Form::Impl(ImplRef::Body(_))) {
            body_forms.push(((form, arity), val));
        } else {
            lk.forms.insert((form, arity), val);
        }
    }
    for _ in 0..cur.count()? {
        let ty = dec.read_ty_ref(&mut cur)?;
        lk.futures.insert(ty, dec.read_ty_ref(&mut cur)?);
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
    for _ in 0..cur.count()? {
        let scope: TypeScope = (
            read_mod_ref(&mut cur)?,
            read_opt(&mut cur, |c| dec.read_item_ref(c))?,
            read_opt(&mut cur, |c| dec.read_item_ref(c))?,
            cur.u8()?,
            read_opt_u32(&mut cur)?,
        );
        let seg_syms = read_vec(&mut cur, |c| sym_at(c.u32()?))?;
        let def = read_def(&dec, &mut cur)?;
        lk.type_defs
            .insert((scope, SegKey::from_slice(&seg_syms)), def);
    }
    let mut body_impl_traits: Vec<(ImplRef, Option<ItemRef>)> = Vec::new();
    for _ in 0..cur.count()? {
        let r = read_impl_ref(&mut cur)?;
        let tr = read_opt(&mut cur, |c| dec.read_item_ref(c))?;
        if matches!(r, ImplRef::Body(_)) {
            body_impl_traits.push((r, tr));
        } else {
            lk.impl_traits.insert(r, tr);
        }
    }
    let mut body_selections: Vec<((ItemRef, Ty, Substs), Option<(ImplRef, Substs)>)> = Vec::new();
    for _ in 0..cur.count()? {
        let tr = dec.read_item_ref(&mut cur)?;
        let ty = dec.read_ty_ref(&mut cur)?;
        let s = dec.substs(&mut cur)?;
        let picked = read_opt(&mut cur, |c| Ok((read_impl_ref(c)?, dec.substs(c)?)))?;
        if matches!(picked, Some((ImplRef::Body(_), _))) {
            body_selections.push(((tr, ty, s), picked));
        } else {
            lk.selections.insert((tr, ty, s), picked);
        }
    }
    for _ in 0..cur.count()? {
        let tr = dec.read_item_ref(&mut cur)?;
        let ty = dec.read_ty_ref(&mut cur)?;
        lk.auto_traits.insert((tr, ty), cur.u8()? != 0);
    }
    let checked_n = cur.count()?;
    let mut checked_impls = Vec::with_capacity(checked_n);
    for _ in 0..checked_n {
        checked_impls.push((read_impl_ref(&mut cur)?, cur.u8()? != 0));
    }
    let reveals_n = cur.count()?;
    let mut reveals = Vec::with_capacity(reveals_n);
    for _ in 0..reveals_n {
        let item = dec.read_item_ref(&mut cur)?;
        let s = dec.substs(&mut cur)?;
        let ty = dec.read_ty_ref(&mut cur)?;
        reveals.push(((item, s), ty));
    }
    for _ in 0..cur.count()? {
        let item = dec.read_item_ref(&mut cur)?;
        let s = dec.substs(&mut cur)?;
        let ty = dec.read_ty_ref(&mut cur)?;
        lk.provisional_rets.insert((item, s), ty);
    }
    for _ in 0..cur.count()? {
        let tr = dec.read_item_ref(&mut cur)?;
        let ty = dec.read_ty_ref(&mut cur)?;
        let a1 = dec.substs(&mut cur)?;
        let sym = sym_at(cur.u32()?)?;
        let a2 = dec.substs(&mut cur)?;
        let out_ty = read_opt(&mut cur, |c| dec.read_ty_ref(c))?;
        lk.projections.insert((tr, ty, a1, sym, a2), out_ty);
    }
    for _ in 0..cur.count()? {
        let m = dec.read_item_ref(&mut cur)?;
        let s = dec.substs(&mut cur)?;
        lk.trait_calls.insert((m, s), dec.res(&mut cur)?);
    }
    for _ in 0..cur.count()? {
        let tr = dec.read_item_ref(&mut cur)?;
        let ty = dec.read_ty_ref(&mut cur)?;
        let a1 = dec.substs(&mut cur)?;
        let assoc = dec.read_item_ref(&mut cur)?;
        let a2 = dec.substs(&mut cur)?;
        let val = dec.read_ty_ref(&mut cur)?;
        lk.assoc_types.insert((tr, ty, a1, assoc, a2), val);
    }
    for _ in 0..cur.count()? {
        lk.verdicts.insert(dec.read_item_ref(&mut cur)?, ());
    }
    let sum_n = cur.count()?;
    let mut summaries = Vec::with_capacity(sum_n);
    for _ in 0..sum_n {
        let item = dec.read_item_ref(&mut cur)?;
        let kind = match cur.u8()? {
            0 => SummaryKind::Poly(Arc::new(dec.read_summary(&mut cur)?)),
            1 => dec.read_verdict(&mut cur)?,
            _ => bail!("invalid summary tag"),
        };
        summaries.push((item, kind));
    }
    if cur.pos != bytes.len() {
        bail!("trailing bytes in warm cache payload");
    }
    Ok(WarmFlights {
        reveals,
        checked_impls,
        summaries,
        body_heads,
        body_headers,
        body_forms,
        body_impl_traits,
        body_selections,
    })
}
