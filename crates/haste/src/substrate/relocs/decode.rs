use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap;

use super::PLAN_MAGIC;
use crate::body::ExprId;
use crate::host::{HostSymbol, MathSymbol, X86Symbol};
use crate::lower::reloc::{RelocPlan, RelocStep, SiteOrigin, TyOrigin};
use crate::lower::{
    RodataId, RodataReloc, RodataTarget, Slots, SymbolSource, VtableEntry, VtableId, Word,
};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::summary::codec::{
    Cursor, FileIndex, ItemRecord, read_site_key, read_str, read_substs, read_ty,
};
use crate::sema::ty::{Ty, Types};

/// Reconstructs a `RelocPlan` from `bytes` in `reached`, returning `None` cleanly if any dependency or item is absent.
pub fn decode_reloc_plan(
    outline: &Outline,
    types: &Types,
    reached: &Reached,
    slots: &dyn Slots,
    index: &FileIndex,
    sysroot_only: bool,
    bytes: &[u8],
) -> Result<Option<RelocPlan>> {
    let mut cur = Cursor::new(bytes);
    if cur.take(4)? != PLAN_MAGIC {
        bail!("invalid portable reloc plan magic");
    }
    let include_deps = cur.u8()? != 0;
    if sysroot_only && include_deps {
        return Ok(None);
    }
    let Some(items) = decode_items(&mut cur, outline, index, include_deps)? else {
        return Ok(None);
    };
    let tys = decode_types(&mut cur, types, &items)?;
    let Some(rodatas) = decode_rodatas(&mut cur, types, reached, slots, &items, &tys)? else {
        return Ok(None);
    };
    let Some(vtables) = decode_vtables(&mut cur, types, reached, slots, &items, &tys)? else {
        return Ok(None);
    };

    let mut dec = Decoder {
        types,
        reached,
        slots,
        items: &items,
        tys: &tys,
        rodatas: &rodatas,
        vtables: &vtables,
        bytes,
        seen: Seen::default(),
    };
    let Some(plan) = dec.read_plan(&mut cur)? else {
        return Ok(None);
    };
    if cur.pos != bytes.len() {
        bail!("trailing bytes in portable reloc plan");
    }
    Ok(Some(plan))
}

fn decode_items(
    cur: &mut Cursor<'_>,
    outline: &Outline,
    index: &FileIndex,
    include_deps: bool,
) -> Result<Option<Vec<ItemRef>>> {
    let key_of = if include_deps {
        crate::substrate::sysroot_item_key
    } else {
        crate::substrate::sysroot_only_item_key
    };
    let item_count = cur.count()?;
    let mut items = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let record = ItemRecord::read(cur)?;
        let sub = cur.u32()?;
        let Some(base) = index.resolve(outline, record, key_of) else {
            return Ok(None);
        };
        let item = if sub == 0 {
            base
        } else {
            let sub_idx = sub - 1;
            if base.item >= (1 << 15) || sub_idx >= (1 << 16) {
                bail!("invalid sub-item index");
            }
            base.inner(sub_idx)
        };
        items.push(item);
    }
    Ok(Some(items))
}

fn decode_types(cur: &mut Cursor<'_>, types: &Types, items: &[ItemRef]) -> Result<Vec<Ty>> {
    let ty_count = cur.count()?;
    let ty_len = cur.u32()? as usize;
    let ty_data = cur.take(ty_len)?;
    let mut ty_cur = Cursor::new(ty_data);
    let mut tys = Vec::with_capacity(ty_count);
    for _ in 0..ty_count {
        tys.push(read_ty(&mut ty_cur, types, items, &tys)?);
    }
    if ty_cur.pos != ty_data.len() {
        bail!("trailing bytes in reloc plan type table");
    }
    Ok(tys)
}

fn decode_rodatas(
    cur: &mut Cursor<'_>,
    types: &Types,
    reached: &Reached,
    slots: &dyn Slots,
    items: &[ItemRef],
    tys: &[Ty],
) -> Result<Option<Vec<RodataId>>> {
    let rodata_count = cur.count()?;
    let mut rodatas = Vec::with_capacity(rodata_count);
    for _ in 0..rodata_count {
        let len = cur.u32()? as usize;
        let data = cur.take(len)?;
        let reloc_count = cur.count()?;
        let mut relocs = Vec::with_capacity(reloc_count);
        for _ in 0..reloc_count {
            let offset = cur.u32()?;
            let target = match cur.u8()? {
                0 => RodataTarget::Rodata(
                    *rodatas
                        .get(cur.u32()? as usize)
                        .context("rodata reloc out of bounds")?,
                ),
                1 => {
                    let item = *items
                        .get(cur.u32()? as usize)
                        .context("rodata fn item out of bounds")?;
                    let substs = read_substs(cur, types, tys)?;
                    let Some(idx) = resolve_or_want(reached, slots, (item, substs)) else {
                        return Ok(None);
                    };
                    RodataTarget::Fn(idx)
                }
                _ => bail!("invalid rodata reloc tag"),
            };
            relocs.push(RodataReloc { offset, target });
        }
        rodatas.push(slots.rodata(data, &relocs));
    }
    Ok(Some(rodatas))
}

fn decode_vtables(
    cur: &mut Cursor<'_>,
    types: &Types,
    reached: &Reached,
    slots: &dyn Slots,
    items: &[ItemRef],
    tys: &[Ty],
) -> Result<Option<Vec<VtableId>>> {
    let vtable_count = cur.count()?;
    let mut vtables = Vec::with_capacity(vtable_count);
    for _ in 0..vtable_count {
        let entry_count = cur.count()?;
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            match cur.u8()? {
                0 => entries.push(VtableEntry::Word(Word::Int(cur.u64()?))),
                1 => {
                    let item = *items
                        .get(cur.u32()? as usize)
                        .context("vtable item out of bounds")?;
                    let substs = read_substs(cur, types, tys)?;
                    let Some(idx) = resolve_or_want(reached, slots, (item, substs)) else {
                        return Ok(None);
                    };
                    entries.push(VtableEntry::Word(Word::Code(idx)));
                }
                2 => {
                    let super_id = *vtables
                        .get(cur.u32()? as usize)
                        .context("super vtable out of bounds")?;
                    entries.push(VtableEntry::Super(super_id));
                }
                _ => bail!("invalid vtable entry tag"),
            }
        }
        vtables.push(slots.vtable_id(&entries));
    }
    Ok(Some(vtables))
}

struct Decoder<'a> {
    types: &'a Types,
    reached: &'a Reached,
    slots: &'a dyn Slots,
    items: &'a [ItemRef],
    tys: &'a [Ty],
    rodatas: &'a [RodataId],
    vtables: &'a [VtableId],
    bytes: &'a [u8],
    seen: Seen<'a>,
}

#[derive(Default)]
struct Seen<'a> {
    long: FxHashMap<[u8; 8], Vec<(&'a [u8], Arc<RelocPlan>)>>,
    short: Vec<(&'a [u8], Arc<RelocPlan>)>,
}

impl<'a> Seen<'a> {
    fn find(&self, rest: &[u8]) -> Option<(usize, Arc<RelocPlan>)> {
        let bucket = rest
            .first_chunk::<8>()
            .and_then(|prefix| self.long.get(prefix));
        self.short
            .iter()
            .chain(bucket.into_iter().flatten())
            .find(|(enc, _)| rest.starts_with(enc))
            .map(|(enc, plan)| (enc.len(), Arc::clone(plan)))
    }

    fn insert(&mut self, enc: &'a [u8], plan: &Arc<RelocPlan>) {
        let entry = (enc, Arc::clone(plan));
        match enc.first_chunk::<8>() {
            Some(prefix) => self.long.entry(*prefix).or_default().push(entry),
            None => self.short.push(entry),
        }
    }
}

impl Decoder<'_> {
    fn item_at(&self, id: u32) -> Result<ItemRef> {
        self.items
            .get(id as usize)
            .copied()
            .context("reloc plan item out of bounds")
    }

    fn read_instance(&self, cur: &mut Cursor<'_>) -> Result<Option<usize>> {
        let item = self.item_at(cur.u32()?)?;
        let substs = read_substs(cur, self.types, self.tys)?;
        Ok(resolve_or_want(self.reached, self.slots, (item, substs)))
    }

    fn read_plan(&mut self, cur: &mut Cursor<'_>) -> Result<Option<RelocPlan>> {
        let count = cur.count()?;
        let mut steps = Vec::with_capacity(count);
        for _ in 0..count {
            let Some(step) = self.read_step(cur)? else {
                return Ok(None);
            };
            steps.push(step);
        }
        Ok(Some(RelocPlan::from_vec(steps)))
    }

    fn read_subplan(&mut self, cur: &mut Cursor<'_>) -> Result<Option<Arc<RelocPlan>>> {
        let (start, bytes) = (cur.pos, self.bytes);
        if let Some((len, plan)) = self
            .seen
            .find(bytes.get(start..).context("reloc subplan out of bounds")?)
        {
            cur.take(len)?;
            return Ok(Some(plan));
        }
        let Some(plan) = self.read_plan(cur)? else {
            return Ok(None);
        };
        let plan = Arc::new(plan);
        self.seen.insert(&bytes[start..cur.pos], &plan);
        Ok(Some(plan))
    }

    fn read_step(&mut self, cur: &mut Cursor<'_>) -> Result<Option<RelocStep>> {
        Ok(Some(match cur.u8()? {
            0 => {
                let Some(sym) = self.read_symbol(cur)? else {
                    return Ok(None);
                };
                RelocStep::Const(sym)
            }
            1 => RelocStep::Promoted {
                seq: cur.u32()?,
                size: cur.u32()?,
            },
            2 => RelocStep::VtableAt(ExprId(cur.u32()?)),
            3 => RelocStep::VtableSlot {
                expr: ExprId(cur.u32()?),
                slot: cur.u16()?,
            },
            4 => RelocStep::SiteSlot(self.read_site_origin(cur)?),
            5 => {
                let origin = self.read_site_origin(cur)?;
                let Some(plan) = self.read_subplan(cur)? else {
                    return Ok(None);
                };
                RelocStep::SiteInline { origin, plan }
            }
            6 => RelocStep::TypeId(self.read_ty_origin(cur)?),
            7 => RelocStep::TypeName(self.read_ty_origin(cur)?),
            _ => bail!("invalid reloc step tag"),
        }))
    }

    fn read_symbol(&self, cur: &mut Cursor<'_>) -> Result<Option<SymbolSource>> {
        Ok(Some(match cur.u8()? {
            0 => match self.read_instance(cur)? {
                Some(idx) => SymbolSource::Slot(idx),
                None => return Ok(None),
            },
            1 => SymbolSource::Vtable(
                *self
                    .vtables
                    .get(cur.u32()? as usize)
                    .context("vtable ref out of bounds")?,
            ),
            2 => SymbolSource::Rodata(
                *self
                    .rodatas
                    .get(cur.u32()? as usize)
                    .context("rodata ref out of bounds")?,
            ),
            3 => {
                let resolved = self.read_instance(cur)?;
                let size = cur.u32()?;
                let Some(index) = resolved else {
                    return Ok(None);
                };
                SymbolSource::StaticCell { index, size }
            }
            4 => {
                let resolved = self.read_instance(cur)?;
                let (seq, size) = (cur.u32()?, cur.u32()?);
                let Some(owner) = resolved else {
                    return Ok(None);
                };
                SymbolSource::PromotedCell { owner, seq, size }
            }
            5 => SymbolSource::Host(read_host(cur)?),
            6 => SymbolSource::Word(cur.u64()?),
            7 => match self.read_instance(cur)? {
                Some(index) => SymbolSource::ThreadSlot(index),
                None => return Ok(None),
            },
            _ => bail!("invalid symbol source tag"),
        }))
    }

    fn read_ty_origin(&self, cur: &mut Cursor<'_>) -> Result<TyOrigin> {
        Ok(match cur.u8()? {
            0 => TyOrigin::Subst(cur.u16()?),
            1 => TyOrigin::Local(cur.u32()?),
            2 => TyOrigin::Expr(ExprId(cur.u32()?)),
            3 => TyOrigin::ForIter(ExprId(cur.u32()?)),
            4 => TyOrigin::AwaitFuture(ExprId(cur.u32()?)),
            5 => TyOrigin::SiteSubst(Box::new(self.read_site_origin(cur)?), cur.u8()?),
            6 => TyOrigin::Step {
                base: Box::new(self.read_ty_origin(cur)?),
                variant: cur.u32()?,
                field: cur.u32()?,
            },
            _ => bail!("invalid ty origin tag"),
        })
    }

    fn read_site_origin(&self, cur: &mut Cursor<'_>) -> Result<SiteOrigin> {
        Ok(match cur.u8()? {
            0 => SiteOrigin::Table(read_site_key(cur)?),
            1 => SiteOrigin::PointerFnDef {
                expr: ExprId(cur.u32()?),
                callee: ExprId(cur.u32()?),
            },
            2 => SiteOrigin::DropGlue(self.read_ty_origin(cur)?),
            3 => SiteOrigin::DropFn(self.read_ty_origin(cur)?),
            4 => SiteOrigin::ConstEvalSelect(Box::new(self.read_site_origin(cur)?)),
            5 => SiteOrigin::ShapeGlueDrop,
            6 => SiteOrigin::ShapeGlueResume,
            7 => SiteOrigin::ShapeUntuple,
            8 => SiteOrigin::ShapeClone(cur.u16()?),
            9 => SiteOrigin::VtableMethod {
                expr: ExprId(cur.u32()?),
                slot: cur.u16()?,
            },
            10 => SiteOrigin::Known(
                self.item_at(cur.u32()?)?,
                read_substs(cur, self.types, self.tys)?,
            ),
            11 => SiteOrigin::Symbol(cur.u32()?),
            _ => bail!("invalid site origin tag"),
        })
    }
}

fn resolve_or_want(
    reached: &Reached,
    slots: &dyn Slots,
    instance: (ItemRef, Substs),
) -> Option<usize> {
    reached
        .index
        .get(&instance)
        .copied()
        .filter(|&idx| !reached.is_untyped(idx))
        .or_else(|| {
            slots.want(instance);
            None
        })
}

fn read_host(cur: &mut Cursor<'_>) -> Result<HostSymbol> {
    Ok(match cur.u8()? {
        0 => HostSymbol::Alloc,
        1 => HostSymbol::AllocZeroed,
        2 => HostSymbol::Dealloc,
        3 => HostSymbol::Realloc,
        4 => HostSymbol::Noop,
        5 => HostSymbol::Barrier,
        6 => HostSymbol::Cpuid,
        7 => HostSymbol::ExtendedState,
        8 => HostSymbol::DsoHandle,
        9 => HostSymbol::Abort,
        10 => HostSymbol::FillOnce,
        11 => HostSymbol::ThreadCell,
        12 => HostSymbol::Syscall,
        13 => {
            HostSymbol::Math(MathSymbol::from_name(read_str(cur)?).context("unknown math symbol")?)
        }
        14 => HostSymbol::X86(X86Symbol::from_link(read_str(cur)?).context("unknown x86 symbol")?),
        15 => {
            let name: Box<str> = read_str(cur)?.into();
            let raw_lib = read_str(cur)?;
            let lib = (!raw_lib.is_empty()).then(|| raw_lib.into());
            HostSymbol::Dynamic { name, lib }
        }
        16 => HostSymbol::CatchUnwind,
        17 => HostSymbol::StartPanic,
        18 => HostSymbol::PanicCleanup,
        19 => HostSymbol::DropPush,
        20 => HostSymbol::DropPop,
        _ => bail!("invalid host symbol tag"),
    })
}
