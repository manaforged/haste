use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::body::ExprId;
use crate::lower::reloc::{RelocPlan, RelocStep, SiteOrigin, TyOrigin};
use crate::lower::stencil::Form;
use crate::lower::{RodataId, Slots, SymbolSource, VtableId};
use crate::sema::reach::Reached;
use crate::sema::summary::codec::{
    Cursor, read_site_key, write_site_key, write_u16, write_u32, write_u32s,
};
use crate::sema::ty::Types;

use super::codec::read_u32_le;
use super::tables::{BundleEncoder, SharedTables, decode_symbol};

pub(super) fn encode_reloc_plan(enc: &mut BundleEncoder<'_>, plan: &RelocPlan) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    write_plan(enc, &mut out, plan)?;
    Some(out)
}

fn write_plan(enc: &mut BundleEncoder<'_>, out: &mut Vec<u8>, plan: &RelocPlan) -> Option<()> {
    write_u32(out, u32::try_from(plan.steps.len()).ok()?);
    for step in &plan.steps {
        write_step(enc, out, step)?;
    }
    Some(())
}

fn write_step(enc: &mut BundleEncoder<'_>, out: &mut Vec<u8>, step: &RelocStep) -> Option<()> {
    match step {
        RelocStep::Const(src) => {
            out.push(0);
            enc.write_symbol(out, src)?;
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
            write_site_origin(enc, out, origin)?;
        }
        RelocStep::SiteInline { origin, plan } => {
            out.push(5);
            write_site_origin(enc, out, origin)?;
            write_plan(enc, out, plan)?;
        }
        RelocStep::TypeId(origin) => {
            out.push(6);
            write_ty_origin(enc, out, origin)?;
        }
        RelocStep::TypeName(origin) => {
            out.push(7);
            write_ty_origin(enc, out, origin)?;
        }
    }
    Some(())
}

fn write_ty_origin(
    enc: &mut BundleEncoder<'_>,
    out: &mut Vec<u8>,
    origin: &TyOrigin,
) -> Option<()> {
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
            write_site_origin(enc, out, site)?;
            out.push(*idx);
        }
        TyOrigin::Step {
            base,
            variant,
            field,
        } => {
            out.push(6);
            write_ty_origin(enc, out, base)?;
            write_u32(out, *variant);
            write_u32(out, *field);
        }
    }
    Some(())
}

fn write_site_origin(
    enc: &mut BundleEncoder<'_>,
    out: &mut Vec<u8>,
    origin: &SiteOrigin,
) -> Option<()> {
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
            write_ty_origin(enc, out, ty)?;
        }
        SiteOrigin::DropFn(ty) => {
            out.push(3);
            write_ty_origin(enc, out, ty)?;
        }
        SiteOrigin::ConstEvalSelect(inner) => {
            out.push(4);
            write_site_origin(enc, out, inner)?;
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
            write_u32(out, enc.item_id(*item)?);
            write_u32s(out, &enc.substs_ids(*substs)?);
        }
        SiteOrigin::Symbol(at) => {
            out.push(11);
            write_u32(out, *at);
        }
    }
    Some(())
}

#[derive(Default)]
struct Seen<'a> {
    long: FxHashMap<[u8; 8], Vec<(&'a [u8], Arc<RelocPlan>)>>,
    short: Vec<(&'a [u8], Arc<RelocPlan>)>,
}

impl<'a> Seen<'a> {
    fn find(&self, rest: &[u8]) -> Option<(usize, Arc<RelocPlan>)> {
        let bucket = rest.first_chunk::<8>().and_then(|p| self.long.get(p));
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

struct PlanDecoder<'a, FR, FV> {
    tables: &'a SharedTables,
    types: &'a Types,
    reached: &'a Reached,
    slots: &'a dyn Slots,
    bytes: &'a [u8],
    seen: Seen<'a>,
    decode_rodata: FR,
    decode_vtable: FV,
}

pub(super) fn decode_reloc_plan(
    tables: &SharedTables,
    types: &Types,
    reached: &Reached,
    slots: &dyn Slots,
    bytes: &[u8],
    decode_rodata: impl FnMut(u32) -> Option<RodataId>,
    decode_vtable: impl FnMut(u32) -> Option<VtableId>,
) -> Option<RelocPlan> {
    let mut dec = PlanDecoder {
        tables,
        types,
        reached,
        slots,
        bytes,
        seen: Seen::default(),
        decode_rodata,
        decode_vtable,
    };
    let mut cur = Cursor::new(bytes);
    let plan = dec.read_plan(&mut cur)?;
    (cur.pos == bytes.len()).then_some(plan)
}

impl<'a, FR, FV> PlanDecoder<'a, FR, FV>
where
    FR: FnMut(u32) -> Option<RodataId>,
    FV: FnMut(u32) -> Option<VtableId>,
{
    fn read_plan(&mut self, cur: &mut Cursor<'a>) -> Option<RelocPlan> {
        let count = cur.count().ok()?;
        let mut steps = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            steps.push(self.read_step(cur)?);
        }
        Some(RelocPlan::from_vec(steps))
    }

    fn read_subplan(&mut self, cur: &mut Cursor<'a>) -> Option<Arc<RelocPlan>> {
        let start = cur.pos;
        if let Some((len, plan)) = self.seen.find(self.bytes.get(start..)?) {
            cur.take(len).ok()?;
            return Some(plan);
        }
        let plan = Arc::new(self.read_plan(cur)?);
        self.seen.insert(&self.bytes[start..cur.pos], &plan);
        Some(plan)
    }

    fn read_step(&mut self, cur: &mut Cursor<'a>) -> Option<RelocStep> {
        Some(match cur.u8().ok()? {
            0 => RelocStep::Const(self.read_symbol(cur)?),
            1 => RelocStep::Promoted {
                seq: cur.u32().ok()?,
                size: cur.u32().ok()?,
            },
            2 => RelocStep::VtableAt(ExprId(cur.u32().ok()?)),
            3 => RelocStep::VtableSlot {
                expr: ExprId(cur.u32().ok()?),
                slot: cur.u16().ok()?,
            },
            4 => RelocStep::SiteSlot(self.read_site_origin(cur)?),
            5 => RelocStep::SiteInline {
                origin: self.read_site_origin(cur)?,
                plan: self.read_subplan(cur)?,
            },
            6 => RelocStep::TypeId(self.read_ty_origin(cur)?),
            7 => RelocStep::TypeName(self.read_ty_origin(cur)?),
            _ => return None,
        })
    }

    fn read_symbol(&mut self, cur: &mut Cursor<'_>) -> Option<SymbolSource> {
        decode_symbol(
            self.tables,
            self.types,
            self.reached,
            self.slots,
            cur,
            &mut self.decode_rodata,
            &mut self.decode_vtable,
        )
    }

    fn read_ty_origin(&self, cur: &mut Cursor<'_>) -> Option<TyOrigin> {
        Some(match cur.u8().ok()? {
            0 => TyOrigin::Subst(cur.u16().ok()?),
            1 => TyOrigin::Local(cur.u32().ok()?),
            2 => TyOrigin::Expr(ExprId(cur.u32().ok()?)),
            3 => TyOrigin::ForIter(ExprId(cur.u32().ok()?)),
            4 => TyOrigin::AwaitFuture(ExprId(cur.u32().ok()?)),
            5 => TyOrigin::SiteSubst(Box::new(self.read_site_origin(cur)?), cur.u8().ok()?),
            6 => TyOrigin::Step {
                base: Box::new(self.read_ty_origin(cur)?),
                variant: cur.u32().ok()?,
                field: cur.u32().ok()?,
            },
            _ => return None,
        })
    }

    fn read_site_origin(&self, cur: &mut Cursor<'_>) -> Option<SiteOrigin> {
        Some(match cur.u8().ok()? {
            0 => SiteOrigin::Table(read_site_key(cur).ok()?),
            1 => SiteOrigin::PointerFnDef {
                expr: ExprId(cur.u32().ok()?),
                callee: ExprId(cur.u32().ok()?),
            },
            2 => SiteOrigin::DropGlue(self.read_ty_origin(cur)?),
            3 => SiteOrigin::DropFn(self.read_ty_origin(cur)?),
            4 => SiteOrigin::ConstEvalSelect(Box::new(self.read_site_origin(cur)?)),
            5 => SiteOrigin::ShapeGlueDrop,
            6 => SiteOrigin::ShapeGlueResume,
            7 => SiteOrigin::ShapeUntuple,
            8 => SiteOrigin::ShapeClone(cur.u16().ok()?),
            9 => SiteOrigin::VtableMethod {
                expr: ExprId(cur.u32().ok()?),
                slot: cur.u16().ok()?,
            },
            10 => SiteOrigin::Known(
                self.tables.item_at(cur.u32().ok()?)?,
                self.tables.read_substs_checked(self.types, cur)?,
            ),
            11 => SiteOrigin::Symbol(cur.u32().ok()?),
            _ => return None,
        })
    }
}

const INL_BODY_HEAD: usize = 13;
const FORMS: [Form; 4] = [Form::Lean, Form::Loop, Form::Compact, Form::Dense];

pub(super) fn pack_inline_blob(
    returns: bool,
    height: u8,
    size: u16,
    form: Form,
    plan_bytes: &[u8],
    func_bytes: &[u8],
) -> Option<Vec<u8>> {
    let plan_len = u32::try_from(plan_bytes.len()).ok()?;
    let func_len = u32::try_from(func_bytes.len()).ok()?;
    let mut out = Vec::with_capacity(INL_BODY_HEAD + plan_bytes.len() + func_bytes.len());
    out.push(u8::from(returns));
    out.push(height);
    out.extend_from_slice(&size.to_le_bytes());
    out.push(form as u8);
    write_u32(&mut out, plan_len);
    write_u32(&mut out, func_len);
    out.extend_from_slice(plan_bytes);
    out.extend_from_slice(func_bytes);
    Some(out)
}

pub(super) fn decode_inline_shape(blob: &[u8]) -> Option<(bool, u8, u16)> {
    let head = blob.get(..INL_BODY_HEAD)?;
    let returns = match head[0] {
        0 => false,
        1 => true,
        _ => return None,
    };
    let height = head[1];
    let size = u16::from_le_bytes([head[2], head[3]]);
    Some((returns, height, size))
}

pub(super) fn split_inline_blob(blob: &[u8]) -> Option<(bool, u8, u16, Form, &[u8], &[u8])> {
    let (returns, height, size) = decode_inline_shape(blob)?;
    let form_tag = blob[4];
    let form = *FORMS
        .get(usize::from(form_tag))
        .filter(|found| **found as u8 == form_tag)?;
    let plan_len = read_u32_le(blob, 5) as usize;
    let func_len = read_u32_le(blob, 9) as usize;
    let plan_end = INL_BODY_HEAD.checked_add(plan_len)?;
    let func_end = plan_end.checked_add(func_len)?;
    if func_end != blob.len() {
        return None;
    }
    Some((
        returns,
        height,
        size,
        form,
        &blob[INL_BODY_HEAD..plan_end],
        &blob[plan_end..func_end],
    ))
}
