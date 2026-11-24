use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::body::ExprId;
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::infer::{Res, SiteKey, Vtable};
use crate::sema::substs::{PackedMap, ProjDef, ProjId, Substs};
use crate::sema::ty::{Ty, Types};

mod files;
mod hash;
mod mono;
mod summary;
mod tables;
mod wire;

pub use files::FileIndex;
pub(crate) use files::ItemRecord;
use files::{read_items, write_items};
pub use hash::{substs_key, type_id_key};
pub use mono::{decode_typed, decode_verdict, encode_typed, encode_verdict};
pub use summary::{decode_instance, decode_summary, encode_instance, encode_summary};
use tables::{read_syntax, write_syntax};
pub(crate) use wire::{
    Cursor, TyTable, read_mut, read_site_key, read_str, read_substs, read_ty, write_site_key,
    write_str, write_u16, write_u32, write_u32s, write_u64,
};

fn framed<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    (outline, types, key_of): (&Outline, &Types, &mut F),
    magic: [u8; 4],
    write: impl FnOnce(&mut Encoder<'_, F>, &mut Vec<u8>) -> Option<()>,
) -> Option<Vec<u8>> {
    let mut enc = Encoder {
        outline,
        types,
        key_of,
        item_map: HashMap::default(),
        items: Vec::new(),
        tys: TyTable::default(),
    };
    let mut body = Vec::new();
    write(&mut enc, &mut body)?;
    let mut out = Vec::with_capacity(
        16 + enc.items.len() * ItemRecord::LEN + enc.tys.bytes().len() + body.len(),
    );
    out.extend_from_slice(&magic);
    write_items(&mut out, &enc.items);
    write_u32(&mut out, enc.tys.count());
    out.extend_from_slice(enc.tys.bytes());
    out.extend_from_slice(&body);
    Some(out)
}

fn unframed<T>(
    (outline, index, types): (&Outline, &FileIndex, &Types),
    magic: [u8; 4],
    bytes: &[u8],
    read: impl FnOnce(&Decoder<'_>, &mut Cursor<'_>) -> Result<Option<T>>,
) -> Result<Option<T>> {
    let mut cur = Cursor::new(bytes);
    if cur.take(4)? != magic {
        bail!("invalid substrate entry magic");
    }
    let Some(items) = read_items(&mut cur, outline, index)? else {
        return Ok(None);
    };
    let ty_count = cur.count()?;
    let mut tys = Vec::with_capacity(ty_count);
    for _ in 0..ty_count {
        tys.push(read_ty(&mut cur, types, &items, &tys)?);
    }
    read(
        &Decoder {
            types,
            items: &items,
            tys: &tys,
        },
        &mut cur,
    )
}

struct Encoder<'a, F> {
    outline: &'a Outline,
    types: &'a Types,
    key_of: &'a mut F,
    item_map: HashMap<ItemRef, u32>,
    items: Vec<ItemRecord>,
    tys: TyTable,
}

fn intern_item<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    outline: &Outline,
    key_of: &mut F,
    item_map: &mut HashMap<ItemRef, u32>,
    items: &mut Vec<ItemRecord>,
    item: ItemRef,
) -> Option<u32> {
    if let Some(&idx) = item_map.get(&item) {
        return Some(idx);
    }
    let record = ItemRecord::of(outline, item, key_of)?;
    let idx = items.len() as u32;
    items.push(record);
    item_map.insert(item, idx);
    Some(idx)
}

impl<F: FnMut(&Outline, ItemRef) -> Option<u64>> Encoder<'_, F> {
    fn item_id(&mut self, item: ItemRef) -> Option<u32> {
        intern_item(
            self.outline,
            self.key_of,
            &mut self.item_map,
            &mut self.items,
            item,
        )
    }

    fn substs_ids(&mut self, substs: Substs) -> Option<Vec<u32>> {
        let (outline, key_of, item_map, items) = (
            self.outline,
            &mut *self.key_of,
            &mut self.item_map,
            &mut self.items,
        );
        self.tys.substs_ids(self.types, substs, &mut |i| {
            intern_item(outline, key_of, item_map, items, i)
        })
    }

    fn ty_id(&mut self, ty: Ty) -> Option<u32> {
        let (outline, key_of, item_map, items) = (
            self.outline,
            &mut *self.key_of,
            &mut self.item_map,
            &mut self.items,
        );
        self.tys.ty_id(self.types, ty, &mut |i| {
            intern_item(outline, key_of, item_map, items, i)
        })
    }

    fn write_substs(&mut self, out: &mut Vec<u8>, s: Substs) -> Option<()> {
        let ids = self.substs_ids(s)?;
        write_u32s(out, &ids);
        Some(())
    }

    fn write_tys(&mut self, out: &mut Vec<u8>, tys: &[Ty]) -> Option<()> {
        write_u32(out, tys.len() as u32);
        for &t in tys {
            write_u32(out, self.ty_id(t)?);
        }
        Some(())
    }

    fn write_proj_def(&mut self, out: &mut Vec<u8>, pid: ProjId) -> Option<()> {
        let d = self.types.proj(pid);
        write_u32(out, self.item_id(d.of)?);
        write_u32(out, self.ty_id(d.self_ty)?);
        self.write_substs(out, d.args)?;
        write_u32(out, self.item_id(d.assoc)?);
        self.write_substs(out, d.own)
    }

    fn write_site_calls(
        &mut self,
        out: &mut Vec<u8>,
        calls: &[(SiteKey, ItemRef, Substs)],
    ) -> Option<()> {
        write_u32(out, calls.len() as u32);
        for &(site, item, substs) in calls {
            write_site_key(out, site);
            write_u32(out, self.item_id(item)?);
            self.write_substs(out, substs)?;
        }
        Some(())
    }

    fn write_res(&mut self, out: &mut Vec<u8>, r: &Res) -> Option<()> {
        match *r {
            Res::Local(id) => {
                out.push(0);
                write_u32(out, id);
            }
            Res::Fn(item, s) => {
                out.push(1);
                write_u32(out, self.item_id(item)?);
                self.write_substs(out, s)?;
            }
            Res::Ctor(item, idx, s) => {
                out.push(2);
                write_u32(out, self.item_id(item)?);
                write_u32(out, idx);
                self.write_substs(out, s)?;
            }
            Res::Const(ty) => {
                out.push(3);
                write_u32(out, self.ty_id(ty)?);
            }
        }
        Some(())
    }

    fn write_vtable(&mut self, out: &mut Vec<u8>, v: &Vtable) -> Option<()> {
        write_u32(out, self.ty_id(v.ty)?);
        write_u32(out, self.ty_id(v.object)?);
        write_opt(out, v.drop.as_ref(), |out, r| self.write_res(out, r))?;
        write_u32(out, v.methods.len() as u32);
        for m in &v.methods {
            write_opt(out, m.as_ref(), |out, r| self.write_res(out, r))?;
        }
        write_u32(out, v.supers.len() as u32);
        for s in &v.supers {
            self.write_vtable(out, s)?;
        }
        Some(())
    }
}

struct Decoder<'a> {
    types: &'a Types,
    items: &'a [ItemRef],
    tys: &'a [Ty],
}

impl Decoder<'_> {
    fn ty(&self, idx: u32) -> Result<Ty> {
        if idx == u32::MAX {
            return Ok(Ty::NONE);
        }
        self.tys
            .get(idx as usize)
            .copied()
            .context("type index out of bounds")
    }

    fn read_ty_ref(&self, cur: &mut Cursor<'_>) -> Result<Ty> {
        self.ty(cur.u32()?)
    }

    fn item(&self, idx: u32) -> Result<ItemRef> {
        self.items
            .get(idx as usize)
            .copied()
            .context("item index out of bounds")
    }

    fn read_item_ref(&self, cur: &mut Cursor<'_>) -> Result<ItemRef> {
        self.item(cur.u32()?)
    }

    fn substs(&self, cur: &mut Cursor<'_>) -> Result<Substs> {
        let ids = cur
            .u32s()?
            .into_iter()
            .map(|i| self.ty(i))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.types.intern_substs(&ids))
    }

    fn read_tys(&self, cur: &mut Cursor<'_>) -> Result<Vec<Ty>> {
        read_vec(cur, |c| self.read_ty_ref(c))
    }

    fn read_proj_def(&self, cur: &mut Cursor<'_>) -> Result<ProjId> {
        Ok(self.types.intern_proj(ProjDef {
            of: self.read_item_ref(cur)?,
            self_ty: self.read_ty_ref(cur)?,
            args: self.substs(cur)?,
            assoc: self.read_item_ref(cur)?,
            own: self.substs(cur)?,
        }))
    }

    fn read_site_calls(&self, cur: &mut Cursor<'_>) -> Result<Box<[(SiteKey, ItemRef, Substs)]>> {
        Ok(read_vec(cur, |c| {
            Ok((read_site_key(c)?, self.read_item_ref(c)?, self.substs(c)?))
        })?
        .into_boxed_slice())
    }

    fn res(&self, cur: &mut Cursor<'_>) -> Result<Res> {
        Ok(match cur.u8()? {
            0 => Res::Local(cur.u32()?),
            1 => Res::Fn(self.read_item_ref(cur)?, self.substs(cur)?),
            2 => Res::Ctor(self.read_item_ref(cur)?, cur.u32()?, self.substs(cur)?),
            3 => Res::Const(self.read_ty_ref(cur)?),
            _ => bail!("invalid res tag"),
        })
    }

    fn vtable(&self, cur: &mut Cursor<'_>) -> Result<Vtable> {
        let ty = self.read_ty_ref(cur)?;
        let object = self.read_ty_ref(cur)?;
        let drop = read_opt(cur, |c| self.res(c))?;
        let methods = read_vec(cur, |c| read_opt(c, |c2| self.res(c2)))?;
        let supers = read_vec(cur, |c| self.vtable(c))?;
        Ok(Vtable {
            ty,
            object,
            drop,
            methods,
            supers,
        })
    }
}

fn write_opt<T>(
    out: &mut Vec<u8>,
    opt: Option<&T>,
    mut f: impl FnMut(&mut Vec<u8>, &T) -> Option<()>,
) -> Option<()> {
    match opt {
        Some(v) => {
            out.push(1);
            f(out, v)
        }
        None => {
            out.push(0);
            Some(())
        }
    }
}

fn read_opt<T>(
    cur: &mut Cursor<'_>,
    f: impl FnOnce(&mut Cursor<'_>) -> Result<T>,
) -> Result<Option<T>> {
    Ok(if cur.u8()? != 0 { Some(f(cur)?) } else { None })
}

fn read_vec<T>(
    cur: &mut Cursor<'_>,
    mut f: impl FnMut(&mut Cursor<'_>) -> Result<T>,
) -> Result<Vec<T>> {
    (0..cur.u32()?).map(|_| f(cur)).collect()
}

fn read_expr_ids(cur: &mut Cursor<'_>) -> Result<Vec<ExprId>> {
    read_vec(cur, |c| c.u32().map(ExprId))
}

fn write_map<K, V>(
    out: &mut Vec<u8>,
    map: &PackedMap<K, V>,
    mut f: impl FnMut(&mut Vec<u8>, &K, &V),
) {
    write_u32(out, map.len() as u32);
    for (k, v) in map {
        f(out, k, v);
    }
}

fn write_map_opt<K, V>(
    out: &mut Vec<u8>,
    map: &PackedMap<K, V>,
    mut f: impl FnMut(&mut Vec<u8>, &K, &V) -> Option<()>,
) -> Option<()> {
    write_u32(out, map.len() as u32);
    for (k, v) in map {
        f(out, k, v)?;
    }
    Some(())
}

fn read_map<K: Ord, V>(
    cur: &mut Cursor<'_>,
    mut f: impl FnMut(&mut Cursor<'_>) -> Result<(K, V)>,
) -> Result<PackedMap<K, V>> {
    let len = cur.u32()? as usize;
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(f(cur)?);
    }
    Ok(PackedMap::from_sorted(vec))
}
