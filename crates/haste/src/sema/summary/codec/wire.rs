//! Shared little-endian wire primitives and workspace-invariant type/site codecs.

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::body::{ExprId, PatId};
use crate::names::{ItemRef, PRIMITIVES, Primitive};
use crate::sema::infer::SiteKey;
use crate::sema::substs::{ProjDef, Substs};
use crate::sema::ty::{Kind, Mut, Ty, Types};

/// A little-endian reader over the bytes of an encoded summary, sysroot, or relocation table.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pub(crate) pos: usize,
}

impl<'a> Cursor<'a> {
    /// A reader at the start of `data`.
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    /// A `u32` count, refused when it exceeds the bytes left.
    pub(crate) fn count(&mut self) -> Result<usize> {
        let n = self.u32()? as usize;
        if n > self.data.len() - self.pos {
            bail!("count exceeds remaining bytes");
        }
        Ok(n)
    }

    /// Advances the cursor by `n` bytes and returns that slice.
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("cursor overflow")?;
        let slice = self.data.get(self.pos..end).context("unexpected eof")?;
        self.pos = end;
        Ok(slice)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    pub(crate) fn u128(&mut self) -> Result<u128> {
        Ok(u128::from_le_bytes(self.take(16)?.try_into()?))
    }

    pub(crate) fn u32s(&mut self) -> Result<Vec<u32>> {
        let n = self.count()?;
        (0..n).map(|_| self.u32()).collect()
    }
}

pub(crate) fn write_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u128(out: &mut Vec<u8>, v: u128) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn write_u32s(out: &mut Vec<u8>, s: &[u32]) {
    write_u32(out, s.len() as u32);
    s.iter().for_each(|&v| write_u32(out, v));
}

pub(crate) fn write_str(out: &mut Vec<u8>, s: &str) -> Option<()> {
    write_u32(out, u32::try_from(s.len()).ok()?);
    out.extend_from_slice(s.as_bytes());
    Some(())
}

pub(crate) fn read_str<'a>(cur: &mut Cursor<'a>) -> Result<&'a str> {
    let len = cur.count()?;
    std::str::from_utf8(cur.take(len)?).context("invalid utf-8 string")
}

pub(crate) fn read_mut(cur: &mut Cursor<'_>) -> Result<Mut> {
    Ok(if cur.u8()? != 0 { Mut::Mut } else { Mut::Not })
}

pub(crate) fn encode_prim(p: Primitive) -> Option<u8> {
    u8::try_from(PRIMITIVES.iter().position(|&(_, prim)| prim == p)?).ok()
}

pub(crate) fn decode_prim(raw: u8) -> Result<Primitive> {
    PRIMITIVES.get(raw as usize).map(|&(_, prim)| prim).context("invalid primitive tag")
}

/// An append-only table of interned type nodes serialized into a byte buffer.
#[derive(Default)]
pub(crate) struct TyTable {
    map: HashMap<Ty, u32>,
    count: u32,
    buf: Vec<u8>,
}

impl TyTable {
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub(crate) fn substs_ids(&mut self, types: &Types, substs: Substs, item_id: &mut impl FnMut(ItemRef) -> Option<u32>) -> Option<Vec<u32>> {
        types.substs(substs).iter().map(|&ty| self.ty_id(types, ty, item_id)).collect()
    }

    pub(crate) fn ty_id(&mut self, types: &Types, ty: Ty, item_id: &mut impl FnMut(ItemRef) -> Option<u32>) -> Option<u32> {
        if ty == Ty::NONE {
            return Some(u32::MAX);
        }
        if let Some(&idx) = self.map.get(&ty) {
            return Some(idx);
        }
        let mut rec = Vec::new();
        match types.kind(ty).clone() {
            Kind::Prim(p) => {
                rec.push(0);
                rec.push(encode_prim(p)?);
            }
            Kind::Never => rec.push(1),
            Kind::Tuple(s) => {
                rec.push(2);
                write_u32s(&mut rec, &self.substs_ids(types, s, item_id)?);
            }
            Kind::Ref(m, i) => {
                rec.push(3);
                rec.push(u8::from(m == Mut::Mut));
                write_u32(&mut rec, self.ty_id(types, i, item_id)?);
            }
            Kind::Ptr(m, i) => {
                rec.push(4);
                rec.push(u8::from(m == Mut::Mut));
                write_u32(&mut rec, self.ty_id(types, i, item_id)?);
            }
            Kind::Array(e, l) => {
                rec.push(5);
                write_u32(&mut rec, self.ty_id(types, e, item_id)?);
                write_u32(&mut rec, self.ty_id(types, l, item_id)?);
            }
            Kind::Slice(e) => {
                rec.push(6);
                write_u32(&mut rec, self.ty_id(types, e, item_id)?);
            }
            Kind::Adt(i, s) => {
                rec.push(7);
                write_u32(&mut rec, item_id(i)?);
                write_u32s(&mut rec, &self.substs_ids(types, s, item_id)?);
            }
            Kind::FnDef(i, s) => {
                rec.push(8);
                write_u32(&mut rec, item_id(i)?);
                write_u32s(&mut rec, &self.substs_ids(types, s, item_id)?);
            }
            Kind::FnPtr(p, r) => {
                rec.push(9);
                write_u32s(&mut rec, &self.substs_ids(types, p, item_id)?);
                write_u32(&mut rec, self.ty_id(types, r, item_id)?);
            }
            Kind::Closure(p, s, at) => {
                rec.push(10);
                write_u32(&mut rec, item_id(p)?);
                write_u32s(&mut rec, &self.substs_ids(types, s, item_id)?);
                write_u32(&mut rec, at);
            }
            Kind::Dyn(o, s, m, a) => {
                rec.push(11);
                write_u32(&mut rec, item_id(o)?);
                write_u32s(&mut rec, &self.substs_ids(types, s, item_id)?);
                let ms = types.items(m).iter().map(|&x| item_id(x)).collect::<Option<Vec<_>>>()?;
                let aus = types.items(a).iter().map(|&x| item_id(x)).collect::<Option<Vec<_>>>()?;
                write_u32s(&mut rec, &ms);
                write_u32s(&mut rec, &aus);
            }
            Kind::Pat(b, range) => {
                let (lo, hi) = types.pat(range);
                rec.push(12);
                write_u32(&mut rec, self.ty_id(types, b, item_id)?);
                write_u128(&mut rec, lo);
                write_u128(&mut rec, hi);
            }
            Kind::Value(v) => {
                rec.push(13);
                write_u128(&mut rec, v);
            }
            Kind::Param(p) => {
                rec.push(14);
                write_u16(&mut rec, p);
            }
            Kind::Proj(id) => {
                let d = types.proj(id);
                rec.push(15);
                write_u32(&mut rec, item_id(d.of)?);
                write_u32(&mut rec, self.ty_id(types, d.self_ty, item_id)?);
                write_u32s(&mut rec, &self.substs_ids(types, d.args, item_id)?);
                write_u32(&mut rec, item_id(d.assoc)?);
                write_u32s(&mut rec, &self.substs_ids(types, d.own, item_id)?);
            }
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => return None,
        }
        let idx = self.count;
        self.count = idx.checked_add(1)?;
        self.buf.extend_from_slice(&rec);
        self.map.insert(ty, idx);
        Some(idx)
    }
}

fn ty_at(tys: &[Ty], id: u32) -> Result<Ty> {
    if id == u32::MAX {
        Ok(Ty::NONE)
    } else {
        tys.get(id as usize).copied().context("ty ref out of bounds")
    }
}

pub(crate) fn read_substs(cur: &mut Cursor<'_>, types: &Types, tys: &[Ty]) -> Result<Substs> {
    let count = cur.count()?;
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        ids.push(ty_at(tys, cur.u32()?)?);
    }
    Ok(types.intern_substs(&ids))
}

pub(crate) fn read_ty(cur: &mut Cursor<'_>, types: &Types, items: &[ItemRef], tys: &[Ty]) -> Result<Ty> {
    let item_at = |id: u32| items.get(id as usize).copied().context("item ref out of bounds");
    let kind = match cur.u8()? {
        0 => Kind::Prim(decode_prim(cur.u8()?)?),
        1 => Kind::Never,
        2 => Kind::Tuple(read_substs(cur, types, tys)?),
        3 => Kind::Ref(read_mut(cur)?, ty_at(tys, cur.u32()?)?),
        4 => Kind::Ptr(read_mut(cur)?, ty_at(tys, cur.u32()?)?),
        5 => Kind::Array(ty_at(tys, cur.u32()?)?, ty_at(tys, cur.u32()?)?),
        6 => Kind::Slice(ty_at(tys, cur.u32()?)?),
        7 => Kind::Adt(item_at(cur.u32()?)?, read_substs(cur, types, tys)?),
        8 => Kind::FnDef(item_at(cur.u32()?)?, read_substs(cur, types, tys)?),
        9 => Kind::FnPtr(read_substs(cur, types, tys)?, ty_at(tys, cur.u32()?)?),
        10 => Kind::Closure(item_at(cur.u32()?)?, read_substs(cur, types, tys)?, cur.u32()?),
        11 => {
            let (of, s) = (item_at(cur.u32()?)?, read_substs(cur, types, tys)?);
            let m_count = cur.count()?;
            let mut members = Vec::with_capacity(m_count);
            for _ in 0..m_count {
                members.push(item_at(cur.u32()?)?);
            }
            let a_count = cur.count()?;
            let mut autos = Vec::with_capacity(a_count);
            for _ in 0..a_count {
                autos.push(item_at(cur.u32()?)?);
            }
            Kind::Dyn(of, s, types.intern_items(&members), types.intern_items(&autos))
        }
        12 => Kind::Pat(ty_at(tys, cur.u32()?)?, types.intern_pat((cur.u128()?, cur.u128()?))),
        13 => Kind::Value(cur.u128()?),
        14 => Kind::Param(cur.u16()?),
        15 => Kind::Proj(types.intern_proj(ProjDef {
            of: item_at(cur.u32()?)?,
            self_ty: ty_at(tys, cur.u32()?)?,
            args: read_substs(cur, types, tys)?,
            assoc: item_at(cur.u32()?)?,
            own: read_substs(cur, types, tys)?,
        })),
        _ => bail!("invalid type node tag"),
    };
    Ok(types.intern(kind))
}

pub(crate) fn write_site_key(out: &mut Vec<u8>, s: SiteKey) {
    match s {
        SiteKey::ExprRes(e) => { out.push(0); write_u32(out, e.0); }
        SiteKey::PatRes(p) => { out.push(1); write_u32(out, p.0); }
        SiteKey::Deref(e, i) => { out.push(2); write_u32(out, e.0); write_u32(out, i); }
        SiteKey::Operator(e) => { out.push(3); write_u32(out, e.0); }
        SiteKey::TryBlock(e) => { out.push(4); write_u32(out, e.0); }
        SiteKey::BoxFree(e) => { out.push(5); write_u32(out, e.0); }
        SiteKey::ForInto(e) => { out.push(6); write_u32(out, e.0); }
        SiteKey::ForNext(e) => { out.push(7); write_u32(out, e.0); }
        SiteKey::TryBranch(e) => { out.push(8); write_u32(out, e.0); }
        SiteKey::TryFrom(e) => { out.push(9); write_u32(out, e.0); }
        SiteKey::AwaitInto(e) => { out.push(10); write_u32(out, e.0); }
        SiteKey::AwaitPoll(e) => { out.push(11); write_u32(out, e.0); }
        SiteKey::CoercionReify(e) => { out.push(12); write_u32(out, e.0); }
        SiteKey::CoercionDeref(e, i) => { out.push(13); write_u32(out, e.0); write_u16(out, i); }
    }
}

pub(crate) fn read_site_key(cur: &mut Cursor<'_>) -> Result<SiteKey> {
    Ok(match cur.u8()? {
        0 => SiteKey::ExprRes(ExprId(cur.u32()?)),
        1 => SiteKey::PatRes(PatId(cur.u32()?)),
        2 => SiteKey::Deref(ExprId(cur.u32()?), cur.u32()?),
        3 => SiteKey::Operator(ExprId(cur.u32()?)),
        4 => SiteKey::TryBlock(ExprId(cur.u32()?)),
        5 => SiteKey::BoxFree(ExprId(cur.u32()?)),
        6 => SiteKey::ForInto(ExprId(cur.u32()?)),
        7 => SiteKey::ForNext(ExprId(cur.u32()?)),
        8 => SiteKey::TryBranch(ExprId(cur.u32()?)),
        9 => SiteKey::TryFrom(ExprId(cur.u32()?)),
        10 => SiteKey::AwaitInto(ExprId(cur.u32()?)),
        11 => SiteKey::AwaitPoll(ExprId(cur.u32()?)),
        12 => SiteKey::CoercionReify(ExprId(cur.u32()?)),
        13 => SiteKey::CoercionDeref(ExprId(cur.u32()?), cur.u16()?),
        _ => bail!("invalid site key tag"),
    })
}
