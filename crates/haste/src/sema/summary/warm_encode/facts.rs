use crate::names::ItemRef;
use crate::outline::Outline;

use super::super::codec::{write_opt, write_u32, write_u128};
use super::{Sink, Snap, snap_flight, snap_table, write_table_entries};

pub(super) fn encode_ty_facts<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    sink: &mut Sink<'_, F>,
) -> usize {
    let (store, lk, filter) = (snap.store, &snap.store.lookups, snap.filter);
    let Sink { enc, secs, .. } = sink;
    let evals = snap_table(&store.evaluated, |&(i, s), _| {
        filter.item(i) && snap.substs(s)
    });
    write_table_entries(&mut secs[7], &evals, |((i, s), v), b| {
        write_u32(b, enc.item_id(*i)?);
        enc.write_substs(b, *s)?;
        write_u128(b, *v);
        Some(())
    });

    let drops = snap_table(&lk.drops, |&t, _| snap.ty(t));
    write_table_entries(&mut secs[7], &drops, |(t, v), b| {
        write_u32(b, enc.ty_id(*t)?);
        b.push(u8::from(*v));
        Some(())
    });

    let own_drops = snap_table(&lk.own_drops, |&t, _| snap.ty(t));
    write_table_entries(&mut secs[7], &own_drops, |(t, v), b| {
        write_u32(b, enc.ty_id(*t)?);
        b.push(u8::from(*v));
        Some(())
    });

    let unpins = snap_table(&lk.unpins, |&t, _| snap.ty(t));
    write_table_entries(&mut secs[7], &unpins, |(t, (ok, d)), b| {
        write_u32(b, enc.ty_id(*t)?);
        b.push(u8::from(*ok));
        write_u32(b, *d);
        Some(())
    });

    let freezes = snap_table(&lk.freezes, |&t, _| snap.ty(t));
    write_table_entries(&mut secs[7], &freezes, |(t, (ok, d)), b| {
        write_u32(b, enc.ty_id(*t)?);
        b.push(u8::from(*ok));
        write_u32(b, *d);
        Some(())
    });

    let glues = snap_table(&lk.glues, |&t, _| snap.ty(t));
    write_table_entries(&mut secs[7], &glues, |(t, r), b| {
        write_u32(b, enc.ty_id(*t)?);
        write_opt(b, r.as_ref(), |o, res| enc.write_res(o, res))
    });

    let futures = snap_table(&lk.futures, |&t, &v| snap.ty(t) && snap.ty(v));
    write_table_entries(&mut secs[7], &futures, |(t, v), b| {
        write_u32(b, enc.ty_id(*t)?);
        write_u32(b, enc.ty_id(*v)?);
        Some(())
    });
    evals.len()
        + drops.len()
        + own_drops.len()
        + unpins.len()
        + freezes.len()
        + glues.len()
        + futures.len()
}

pub(super) fn encode_trait_facts<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    snap: &Snap<'_>,
    sink: &mut Sink<'_, F>,
) -> usize {
    let (lk, filter) = (&snap.store.lookups, snap.filter);
    let Sink { enc, syms, secs } = sink;
    let reveals = snap_flight(&lk.reveals, |&(i, s), &t| {
        filter.item(i) && snap.substs(s) && snap.ty(t)
    });
    write_table_entries(&mut secs[7], &reveals, |((i, s), t), b| {
        write_u32(b, enc.item_id(*i)?);
        enc.write_substs(b, *s)?;
        write_u32(b, enc.ty_id(*t)?);
        Some(())
    });

    let provisional_rets = snap_table(&lk.provisional_rets, |&(i, s), &t| {
        filter.item(i) && snap.substs(s) && snap.ty(t)
    });
    write_table_entries(&mut secs[7], &provisional_rets, |((i, s), t), b| {
        write_u32(b, enc.item_id(*i)?);
        enc.write_substs(b, *s)?;
        write_u32(b, enc.ty_id(*t)?);
        Some(())
    });

    let projections = snap_table(&lk.projections, |&(tr, ty, a1, _, a2), v| {
        filter.item(tr)
            && snap.ty(ty)
            && snap.substs(a1)
            && snap.substs(a2)
            && v.is_none_or(|t| snap.ty(t))
    });
    write_table_entries(
        &mut secs[7],
        &projections,
        |((tr, ty, a1, sym, a2), val), b| {
            write_u32(b, enc.item_id(*tr)?);
            write_u32(b, enc.ty_id(*ty)?);
            enc.write_substs(b, *a1)?;
            write_u32(b, syms.id(*sym));
            enc.write_substs(b, *a2)?;
            write_opt(b, val.as_ref(), |o, t| {
                write_u32(o, enc.ty_id(*t)?);
                Some(())
            })
        },
    );

    let trait_calls = snap_table(&lk.trait_calls, |&(m, s), _| {
        filter.item(m) && snap.substs(s)
    });
    write_table_entries(&mut secs[7], &trait_calls, |((m, s), res), b| {
        write_u32(b, enc.item_id(*m)?);
        enc.write_substs(b, *s)?;
        enc.write_res(b, res)
    });

    let assoc_types = snap_table(&lk.assoc_types, |&(tr, ty, a1, assoc, a2), &val| {
        filter.item(tr)
            && snap.ty(ty)
            && snap.substs(a1)
            && filter.item(assoc)
            && snap.substs(a2)
            && snap.ty(val)
    });
    write_table_entries(
        &mut secs[7],
        &assoc_types,
        |((tr, ty, a1, assoc, a2), val), b| {
            write_u32(b, enc.item_id(*tr)?);
            write_u32(b, enc.ty_id(*ty)?);
            enc.write_substs(b, *a1)?;
            write_u32(b, enc.item_id(*assoc)?);
            enc.write_substs(b, *a2)?;
            write_u32(b, enc.ty_id(*val)?);
            Some(())
        },
    );

    let auto_traits = snap_table(&lk.auto_traits, |&(tr, ty), _| {
        filter.item(tr) && snap.ty(ty)
    });
    write_table_entries(&mut secs[7], &auto_traits, |((tr, ty), v), b| {
        write_u32(b, enc.item_id(*tr)?);
        write_u32(b, enc.ty_id(*ty)?);
        b.push(u8::from(*v));
        Some(())
    });
    reveals.len()
        + provisional_rets.len()
        + projections.len()
        + trait_calls.len()
        + assoc_types.len()
        + auto_traits.len()
}
