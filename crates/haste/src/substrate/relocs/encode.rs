use std::collections::HashMap;

use super::PLAN_MAGIC;
use crate::host::HostSymbol;
use crate::lower::reloc::{RelocPlan, RelocStep, SiteOrigin, TyOrigin};
use crate::lower::{
    RodataId, RodataReloc, RodataTarget, SymbolSource, VtableEntry, VtableId, Word,
};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::summary::codec::{
    ItemRecord, TyTable, write_site_key, write_str, write_u16, write_u32, write_u32s, write_u64,
};
use crate::sema::ty::Types;

/// Encodes `plan` into workspace-invariant bytes, or `None` if any symbol or type leaves the allowed crate scope.
pub fn encode_reloc_plan<F, FR, FV>(
    outline: &Outline,
    types: &Types,
    reached: &Reached,
    plan: &RelocPlan,
    include_deps: bool,
    mut key_of: F,
    mut rodata_of: FR,
    mut vtable_of: FV,
) -> Option<Vec<u8>>
where
    F: FnMut(&Outline, ItemRef) -> Option<u64>,
    FR: FnMut(RodataId) -> Option<(Vec<u8>, Vec<RodataReloc>)>,
    FV: FnMut(VtableId) -> Option<Vec<VtableEntry>>,
{
    let mut enc = Encoder {
        outline,
        types,
        reached,
        key_of: &mut key_of,
        rodata_of: &mut rodata_of,
        vtable_of: &mut vtable_of,
        item_map: HashMap::new(),
        items: Vec::new(),
        tys: TyTable::default(),
        rodata_map: HashMap::new(),
        rodatas: Vec::new(),
        vtable_map: HashMap::new(),
        vtables: Vec::new(),
    };
    let mut steps_buf = Vec::new();
    enc.write_plan(&mut steps_buf, plan)?;

    let mut out = Vec::with_capacity(
        32 + enc.items.len() * (ItemRecord::LEN + 4) + enc.tys.bytes().len() + steps_buf.len(),
    );
    out.extend_from_slice(&PLAN_MAGIC);
    out.push(u8::from(include_deps));
    write_u32(&mut out, u32::try_from(enc.items.len()).ok()?);
    for &(record, sub) in &enc.items {
        record.write(&mut out);
        write_u32(&mut out, sub);
    }
    write_u32(&mut out, enc.tys.count());
    write_u32(&mut out, u32::try_from(enc.tys.bytes().len()).ok()?);
    out.extend_from_slice(enc.tys.bytes());
    write_u32(&mut out, u32::try_from(enc.rodatas.len()).ok()?);
    out.extend_from_slice(&enc.rodatas_bytes()?);
    write_u32(&mut out, u32::try_from(enc.vtables.len()).ok()?);
    out.extend_from_slice(&enc.vtables_bytes()?);
    out.extend_from_slice(&steps_buf);
    Some(out)
}

enum EncRodataTarget {
    Rodata(u32),
    Fn(u32, Vec<u32>),
}

struct EncRodata {
    bytes: Vec<u8>,
    relocs: Vec<(u32, EncRodataTarget)>,
}

enum EncVtableEntry {
    Int(u64),
    Code(u32, Vec<u32>),
    Super(u32),
}

struct Encoder<'a, F, FR, FV> {
    outline: &'a Outline,
    types: &'a Types,
    reached: &'a Reached,
    key_of: &'a mut F,
    rodata_of: &'a mut FR,
    vtable_of: &'a mut FV,
    item_map: HashMap<ItemRef, u32>,
    items: Vec<(ItemRecord, u32)>,
    tys: TyTable,
    rodata_map: HashMap<RodataId, u32>,
    rodatas: Vec<EncRodata>,
    vtable_map: HashMap<VtableId, u32>,
    vtables: Vec<Vec<EncVtableEntry>>,
}

impl<F, FR, FV> Encoder<'_, F, FR, FV>
where
    F: FnMut(&Outline, ItemRef) -> Option<u64>,
    FR: FnMut(RodataId) -> Option<(Vec<u8>, Vec<RodataReloc>)>,
    FV: FnMut(VtableId) -> Option<Vec<VtableEntry>>,
{
    fn rodatas_bytes(&self) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for entry in &self.rodatas {
            write_u32(&mut out, u32::try_from(entry.bytes.len()).ok()?);
            out.extend_from_slice(&entry.bytes);
            write_u32(&mut out, u32::try_from(entry.relocs.len()).ok()?);
            for (offset, target) in &entry.relocs {
                write_u32(&mut out, *offset);
                match target {
                    EncRodataTarget::Rodata(id) => {
                        out.push(0);
                        write_u32(&mut out, *id);
                    }
                    EncRodataTarget::Fn(item, substs) => {
                        out.push(1);
                        write_u32(&mut out, *item);
                        write_u32s(&mut out, substs);
                    }
                }
            }
        }
        Some(out)
    }

    fn vtables_bytes(&self) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for entries in &self.vtables {
            write_u32(&mut out, u32::try_from(entries.len()).ok()?);
            for entry in entries {
                match entry {
                    EncVtableEntry::Int(val) => {
                        out.push(0);
                        write_u64(&mut out, *val);
                    }
                    EncVtableEntry::Code(item, substs) => {
                        out.push(1);
                        write_u32(&mut out, *item);
                        write_u32s(&mut out, substs);
                    }
                    EncVtableEntry::Super(id) => {
                        out.push(2);
                        write_u32(&mut out, *id);
                    }
                }
            }
        }
        Some(out)
    }

    fn item_id(&mut self, item: ItemRef) -> Option<u32> {
        intern_sub_item(
            self.outline,
            self.key_of,
            &mut self.item_map,
            &mut self.items,
            item,
        )
    }

    fn reached_ref(&mut self, index: usize) -> Option<(u32, Vec<u32>)> {
        let ((item, substs), _) = self.reached.list.get(index)?;
        Some((self.item_id(*item)?, self.substs_ids(*substs)?))
    }

    fn substs_ids(&mut self, substs: Substs) -> Option<Vec<u32>> {
        let (outline, key_of, item_map, items) = (
            self.outline,
            &mut *self.key_of,
            &mut self.item_map,
            &mut self.items,
        );
        self.tys.substs_ids(self.types, substs, &mut |i| {
            intern_sub_item(outline, key_of, item_map, items, i)
        })
    }

    fn rodata_id(&mut self, id: RodataId) -> Option<u32> {
        if let Some(&idx) = self.rodata_map.get(&id) {
            return Some(idx);
        }
        let (bytes, relocs) = (self.rodata_of)(id)?;
        let mut enc_relocs = Vec::with_capacity(relocs.len());
        for reloc in relocs {
            let target = match reloc.target {
                RodataTarget::Rodata(child) => {
                    if child == id {
                        return None;
                    }
                    EncRodataTarget::Rodata(self.rodata_id(child)?)
                }
                RodataTarget::Fn(fn_idx) => {
                    let (item, substs) = self.reached_ref(fn_idx)?;
                    EncRodataTarget::Fn(item, substs)
                }
            };
            enc_relocs.push((reloc.offset, target));
        }
        let idx = u32::try_from(self.rodatas.len()).ok()?;
        self.rodatas.push(EncRodata {
            bytes,
            relocs: enc_relocs,
        });
        self.rodata_map.insert(id, idx);
        Some(idx)
    }

    fn vtable_id(&mut self, id: VtableId) -> Option<u32> {
        if let Some(&idx) = self.vtable_map.get(&id) {
            return Some(idx);
        }
        let entries = (self.vtable_of)(id)?;
        let mut enc_entries = Vec::with_capacity(entries.len());
        for entry in entries {
            match entry {
                VtableEntry::Word(Word::Int(val)) => enc_entries.push(EncVtableEntry::Int(val)),
                VtableEntry::Word(Word::Code(idx)) => {
                    let (item, substs) = self.reached_ref(idx)?;
                    enc_entries.push(EncVtableEntry::Code(item, substs));
                }
                VtableEntry::Super(super_id) => {
                    if super_id == id {
                        return None;
                    }
                    enc_entries.push(EncVtableEntry::Super(self.vtable_id(super_id)?));
                }
            }
        }
        let idx = u32::try_from(self.vtables.len()).ok()?;
        self.vtables.push(enc_entries);
        self.vtable_map.insert(id, idx);
        Some(idx)
    }

    fn write_plan(&mut self, out: &mut Vec<u8>, plan: &RelocPlan) -> Option<()> {
        write_u32(out, u32::try_from(plan.steps.len()).ok()?);
        for step in &plan.steps {
            self.write_step(out, step)?;
        }
        Some(())
    }

    fn write_step(&mut self, out: &mut Vec<u8>, step: &RelocStep) -> Option<()> {
        match step {
            RelocStep::Const(src) => {
                out.push(0);
                self.write_symbol(out, src)?;
            }
            RelocStep::Promoted { seq, size } => {
                out.push(1);
                write_u32(out, *seq);
                write_u32(out, *size);
            }
            RelocStep::VtableAt(expr) => {
                out.push(2);
                write_u32(out, expr.0);
            }
            RelocStep::VtableSlot { expr, slot } => {
                out.push(3);
                write_u32(out, expr.0);
                write_u16(out, *slot);
            }
            RelocStep::SiteSlot(origin) => {
                out.push(4);
                self.write_site_origin(out, origin)?;
            }
            RelocStep::SiteInline { origin, plan } => {
                out.push(5);
                self.write_site_origin(out, origin)?;
                self.write_plan(out, plan)?;
            }
            RelocStep::TypeId(origin) => {
                out.push(6);
                self.write_ty_origin(out, origin)?;
            }
            RelocStep::TypeName(origin) => {
                out.push(7);
                self.write_ty_origin(out, origin)?;
            }
        }
        Some(())
    }

    fn write_symbol(&mut self, out: &mut Vec<u8>, src: &SymbolSource) -> Option<()> {
        match src {
            SymbolSource::Slot(index) => {
                out.push(0);
                let (item, substs) = self.reached_ref(*index)?;
                write_u32(out, item);
                write_u32s(out, &substs);
            }
            SymbolSource::Vtable(id) => {
                out.push(1);
                write_u32(out, self.vtable_id(*id)?);
            }
            SymbolSource::Rodata(id) => {
                out.push(2);
                write_u32(out, self.rodata_id(*id)?);
            }
            SymbolSource::StaticCell { index, size } => {
                out.push(3);
                let (item, substs) = self.reached_ref(*index)?;
                write_u32(out, item);
                write_u32s(out, &substs);
                write_u32(out, *size);
            }
            SymbolSource::PromotedCell { owner, seq, size } => {
                out.push(4);
                let (item, substs) = self.reached_ref(*owner)?;
                write_u32(out, item);
                write_u32s(out, &substs);
                write_u32(out, *seq);
                write_u32(out, *size);
            }
            SymbolSource::Host(host) => {
                out.push(5);
                write_host(out, host)?;
            }
            SymbolSource::Word(word) => {
                out.push(6);
                write_u64(out, *word);
            }
            SymbolSource::ThreadSlot(index) => {
                out.push(7);
                let (item, substs) = self.reached_ref(*index)?;
                write_u32(out, item);
                write_u32s(out, &substs);
            }
        }
        Some(())
    }

    fn write_ty_origin(&mut self, out: &mut Vec<u8>, origin: &TyOrigin) -> Option<()> {
        match origin {
            TyOrigin::Subst(idx) => {
                out.push(0);
                write_u16(out, *idx);
            }
            TyOrigin::Local(idx) => {
                out.push(1);
                write_u32(out, *idx);
            }
            TyOrigin::Expr(expr) => {
                out.push(2);
                write_u32(out, expr.0);
            }
            TyOrigin::ForIter(expr) => {
                out.push(3);
                write_u32(out, expr.0);
            }
            TyOrigin::AwaitFuture(expr) => {
                out.push(4);
                write_u32(out, expr.0);
            }
            TyOrigin::SiteSubst(site, idx) => {
                out.push(5);
                self.write_site_origin(out, site)?;
                out.push(*idx);
            }
            TyOrigin::Step {
                base,
                variant,
                field,
            } => {
                out.push(6);
                self.write_ty_origin(out, base)?;
                write_u32(out, *variant);
                write_u32(out, *field);
            }
        }
        Some(())
    }

    fn write_site_origin(&mut self, out: &mut Vec<u8>, origin: &SiteOrigin) -> Option<()> {
        match origin {
            SiteOrigin::Table(key) => {
                out.push(0);
                write_site_key(out, *key);
            }
            SiteOrigin::PointerFnDef { expr, callee } => {
                out.push(1);
                write_u32(out, expr.0);
                write_u32(out, callee.0);
            }
            SiteOrigin::DropGlue(ty) => {
                out.push(2);
                self.write_ty_origin(out, ty)?;
            }
            SiteOrigin::DropFn(ty) => {
                out.push(3);
                self.write_ty_origin(out, ty)?;
            }
            SiteOrigin::ConstEvalSelect(inner) => {
                out.push(4);
                self.write_site_origin(out, inner)?;
            }
            SiteOrigin::ShapeGlueDrop => out.push(5),
            SiteOrigin::ShapeGlueResume => out.push(6),
            SiteOrigin::ShapeUntuple => out.push(7),
            SiteOrigin::ShapeClone(idx) => {
                out.push(8);
                write_u16(out, *idx);
            }
            SiteOrigin::VtableMethod { expr, slot } => {
                out.push(9);
                write_u32(out, expr.0);
                write_u16(out, *slot);
            }
            SiteOrigin::Known(item, substs) => {
                out.push(10);
                write_u32(out, self.item_id(*item)?);
                write_u32s(out, &self.substs_ids(*substs)?);
            }
            SiteOrigin::Symbol(at) => {
                out.push(11);
                write_u32(out, *at);
            }
        }
        Some(())
    }
}

fn intern_sub_item<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    outline: &Outline,
    key_of: &mut F,
    item_map: &mut HashMap<ItemRef, u32>,
    items: &mut Vec<(ItemRecord, u32)>,
    item: ItemRef,
) -> Option<u32> {
    if let Some(&idx) = item_map.get(&item) {
        return Some(idx);
    }
    let (base, sub) = match item.outer() {
        Some((outer, sub)) => (outer, sub.checked_add(1)?),
        None => (item, 0),
    };
    let record = ItemRecord::of(outline, base, key_of)?;
    let idx = u32::try_from(items.len()).ok()?;
    items.push((record, sub));
    item_map.insert(item, idx);
    Some(idx)
}

fn write_host(out: &mut Vec<u8>, host: &HostSymbol) -> Option<()> {
    match host {
        HostSymbol::Alloc => out.push(0),
        HostSymbol::AllocZeroed => out.push(1),
        HostSymbol::Dealloc => out.push(2),
        HostSymbol::Realloc => out.push(3),
        HostSymbol::Noop => out.push(4),
        HostSymbol::Barrier => out.push(5),
        HostSymbol::Cpuid => out.push(6),
        HostSymbol::ExtendedState => out.push(7),
        HostSymbol::DsoHandle => out.push(8),
        HostSymbol::Abort => out.push(9),
        HostSymbol::FillOnce => out.push(10),
        HostSymbol::ThreadCell => out.push(11),
        HostSymbol::Syscall => out.push(12),
        HostSymbol::Math(m) => {
            out.push(13);
            write_str(out, m.link_name())?;
        }
        HostSymbol::X86(x) => {
            out.push(14);
            write_str(out, x.link_name())?;
        }
        HostSymbol::Dynamic { name, lib } => {
            out.push(15);
            write_str(out, name)?;
            write_str(out, lib.as_deref().unwrap_or(""))?;
        }
        HostSymbol::CatchUnwind => out.push(16),
        HostSymbol::StartPanic => out.push(17),
        HostSymbol::PanicCleanup => out.push(18),
        HostSymbol::DropPush => out.push(19),
        HostSymbol::DropPop => out.push(20),
    }
    Some(())
}
