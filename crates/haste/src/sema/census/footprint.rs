use rustc_abi::{Primitive as AbiPrim, TagEncoding, Variants};
use rustc_hash::FxHashSet;
use rustc_index::Idx;

use crate::body::{Body, Expr, ExprId, Pat, PatId, UnOp};
use crate::lower::{self, Repr};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{Arena, Res, Shape, Typed, field_types, needs_drop};
use crate::sema::layout::{field_offset, target};
use crate::sema::reach::Reached;
use crate::sema::ty::{Kind, Ty};

use super::calls::{collect_calls, hash_control_and_coercions};
use super::digest::{Digest, Mix, TraitFacts, hash_num_tag, hash_repr, hash_signature, hash_unsized_kind, inline_flags, outer_shape_digest};

#[derive(Clone, Copy)]
pub(super) struct Edge {
    pub tag: u8,
    pub key: u64,
    pub callee: usize,
    pub item: ItemRef,
    pub foreign: bool,
}

struct BaseRecord {
    valid: bool,
    foreign: bool,
    generic: bool,
    inlined: bool,
    expr_count: usize,
    outer_shape: Digest,
    k_base: Digest,
    d_base: Digest,
    d_slots: u32,
    niche: Digest,
    offset: Digest,
    assoc_abi: Digest,
    spec_callee: Digest,
    reflection: Digest,
    edges: Vec<Edge>,
}

#[derive(Clone, Copy, Default)]
pub(super) struct InstanceRecord {
    pub valid: bool,
    pub foreign: bool,
    pub generic: bool,
    pub expr_count: usize,
    pub outer_shape: Digest,
    pub k_no_inline: Digest,
    pub k_inline: Digest,
    pub d_inline: Digest,
    pub d_slots: u32,
    pub niche: Digest,
    pub offset: Digest,
    pub assoc_abi: Digest,
    pub spec_callee: Digest,
    pub reflection: Digest,
}

pub(super) struct HashBins {
    pub k: Mix,
    pub d: Mix,
    pub d_slots: u32,
    pub niche: Mix,
    pub offset: Mix,
    pub assoc_abi: Mix,
    pub spec_callee: Mix,
    pub reflection: Mix,
}

impl HashBins {
    fn new() -> Self {
        Self { k: Mix::new(1), d: Mix::new(2), d_slots: 0, niche: Mix::new(3), offset: Mix::new(4), assoc_abi: Mix::new(5), spec_callee: Mix::new(6), reflection: Mix::new(7) }
    }
}

type InlineMemo = (Digest, Digest, Digest, u32);

pub(super) fn analyze_all(cx: &mut Context<'_>, reached: &Reached) -> Vec<InstanceRecord> {
    let inlined = inline_flags(cx, reached);
    let mut facts = TraitFacts::build(cx);
    let mut bases = Vec::with_capacity(reached.list.len());
    for (index, &flag) in inlined.iter().enumerate() {
        bases.push(analyze_base(cx, reached, index, flag, &mut facts));
    }
    let mut memo: Vec<Option<InlineMemo>> = vec![None; bases.len()];
    let mut visiting_set = FxHashSet::default();
    let mut out = Vec::with_capacity(bases.len());
    for index in 0..bases.len() {
        let base = &bases[index];
        if !base.valid {
            out.push(InstanceRecord::default());
            continue;
        }
        let (k_inline, d_inline, reflection, d_slots) = resolve_inline(index, &bases, &mut memo, &mut visiting_set);
        out.push(InstanceRecord {
            valid: true,
            foreign: base.foreign,
            generic: base.generic,
            expr_count: base.expr_count,
            outer_shape: base.outer_shape,
            k_no_inline: base.k_base,
            k_inline,
            d_inline,
            d_slots,
            niche: base.niche,
            offset: base.offset,
            assoc_abi: base.assoc_abi,
            spec_callee: base.spec_callee,
            reflection,
        });
    }
    out
}

fn resolve_inline(index: usize, bases: &[BaseRecord], memo: &mut [Option<InlineMemo>], visiting_set: &mut FxHashSet<usize>) -> InlineMemo {
    if let Some(found) = memo[index] {
        return found;
    }
    let base = &bases[index];
    let (mut k_mix, mut d_mix, mut r_mix) = (Mix::new(10), Mix::new(20), Mix::new(30));
    k_mix.digest(base.k_base);
    d_mix.digest(base.d_base);
    r_mix.digest(base.reflection);
    let mut slots = base.d_slots;
    visiting_set.insert(index);
    for edge in &base.edges {
        if !edge.foreign && bases[edge.callee].inlined && !visiting_set.contains(&edge.callee) {
            let (ck, cd, cr, cslots) = resolve_inline(edge.callee, bases, memo, visiting_set);
            k_mix.byte(1);
            k_mix.byte(edge.tag);
            k_mix.u64(edge.key);
            k_mix.item(edge.item);
            k_mix.digest(ck);
            d_mix.byte(1);
            d_mix.byte(edge.tag);
            d_mix.u64(edge.key);
            d_mix.digest(cd);
            r_mix.byte(edge.tag);
            r_mix.u64(edge.key);
            r_mix.digest(cr);
            slots = slots.saturating_add(cslots);
        } else {
            d_mix.byte(0);
            d_mix.byte(edge.tag);
            d_mix.u64(edge.key);
            d_mix.usize(edge.callee);
            slots = slots.saturating_add(1);
        }
    }
    visiting_set.remove(&index);
    let computed = (k_mix.finish(), d_mix.finish(), r_mix.finish(), slots);
    memo[index] = Some(computed);
    computed
}

fn empty_base() -> BaseRecord {
    BaseRecord { valid: false, foreign: false, generic: false, inlined: false, expr_count: 0, outer_shape: Digest::default(), k_base: Digest::default(), d_base: Digest::default(), d_slots: 0, niche: Digest::default(), offset: Digest::default(), assoc_abi: Digest::default(), spec_callee: Digest::default(), reflection: Digest::default(), edges: Vec::new() }
}

fn analyze_base(cx: &mut Context<'_>, reached: &Reached, index: usize, inlined: bool, facts: &mut TraitFacts) -> BaseRecord {
    let ((item, substs), result) = &reached.list[index];
    let Ok(typed) = result else { return empty_base() };
    let typed = typed.scoped(&cx.types);
    let typed: &Typed = &typed;
    let generic = !substs.is_empty();
    if cx.foreign(*item) {
        let (mut k, mut d) = (Mix::new(1), Mix::new(2));
        k.byte(0xff);
        k.item(*item);
        d.item(*item);
        return BaseRecord { valid: true, foreign: true, generic, k_base: k.finish(), d_base: d.finish(), ..empty_base() };
    }
    let outer_shape = if generic { outer_shape_digest(cx, *substs) } else { Digest::default() };
    let source = match &typed.shape {
        Shape::Closure(closure) => closure.parent,
        _ => *item,
    };
    let mut bins = HashBins::new();
    let mut edges = Vec::new();
    hash_header(cx, *item, typed, &mut bins);
    hash_locals_and_pats(cx, typed, &mut bins);
    hash_body_ast(cx, reached, *item, typed, &mut bins);
    hash_control_and_coercions(cx, reached, typed, &mut bins, &mut edges);
    collect_calls(cx, reached, (source, typed), (&mut bins, &mut edges), facts);
    BaseRecord {
        valid: true,
        foreign: false,
        generic,
        inlined,
        expr_count: typed.exprs.len(),
        outer_shape,
        k_base: bins.k.finish(),
        d_base: bins.d.finish(),
        d_slots: bins.d_slots,
        niche: bins.niche.finish(),
        offset: bins.offset.finish(),
        assoc_abi: bins.assoc_abi.finish(),
        spec_callee: bins.spec_callee.finish(),
        reflection: bins.reflection.finish(),
        edges,
    }
}

fn is_enum_ty(cx: &Context<'_>, ty: Ty) -> bool {
    matches!(cx.types.kind(ty), Kind::Adt(item, _) if cx.item(*item).1.kind == ItemKind::Enum)
}

pub(super) fn hash_type_use(cx: &mut Context<'_>, ty: Ty, bins: &mut HashBins) {
    if ty == Ty::NONE {
        bins.k.byte(0);
        return;
    }
    bins.k.byte(1);
    let r = lower::repr(cx, ty).unwrap_or(Repr::Zst);
    hash_repr(&r, &mut bins.k);
    let is_enum = is_enum_ty(cx, ty);
    if !is_enum {
        hash_repr(&r, &mut bins.assoc_abi);
    }
    bins.k.flag(cx.boxed(ty).is_some());
    bins.k.flag(needs_drop(cx, ty).unwrap_or(false));
    match cx.types.kind(ty).clone() {
        Kind::Array(elem, len) => {
            bins.k.byte(1);
            let count = cx.types.len(len).unwrap_or(0);
            let stride = cx.layout(elem).map_or(0, |l| l.size.bytes());
            bins.k.u64(count);
            bins.k.u64(stride);
            bins.offset.u64(stride);
            if let Ok(er) = lower::repr(cx, elem) {
                hash_repr(&er, &mut bins.k);
            }
        }
        Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) if cx.is_unsized(pointee).unwrap_or(false) => {
            bins.k.byte(2);
            hash_unsized_kind(cx, pointee, &mut bins.k);
        }
        Kind::Adt(adt_item, _) if is_enum => {
            bins.k.byte(3);
            hash_enum_encoding(cx, adt_item, ty, &r, bins);
        }
        _ => bins.k.byte(0),
    }
}

fn hash_enum_encoding(cx: &mut Context<'_>, item: ItemRef, ty: Ty, repr: &Repr, bins: &mut HashBins) {
    let Ok(layout) = cx.layout(ty) else { return };
    hash_repr(repr, &mut bins.niche);
    bins.niche.u64(layout.size.bytes());
    bins.niche.u64(layout.align.bytes());
    match &layout.variants {
        Variants::Empty => {
            bins.k.byte(0);
            bins.niche.byte(0);
        }
        Variants::Single { index } => {
            bins.k.byte(1);
            bins.k.usize(index.index());
            bins.niche.byte(1);
            bins.niche.usize(index.index());
        }
        Variants::Multiple { tag, tag_encoding, tag_field, .. } => {
            let offset = layout.fields.offset(tag_field.index()).bytes() as i32;
            let tag_bytes = target().map_or(0, |dl| tag.size(dl).bytes());
            let signed = matches!(tag.primitive(), AbiPrim::Int(_, true));
            for out in [&mut bins.k, &mut bins.niche] {
                out.byte(2);
                out.i32(offset);
                out.u64(tag_bytes);
                out.flag(signed);
            }
            hash_tag_encoding(cx, item, tag_encoding, bins);
        }
    }
}

fn hash_tag_encoding(cx: &mut Context<'_>, item: ItemRef, encoding: &TagEncoding<crate::sema::layout::Variant>, bins: &mut HashBins) {
    match encoding {
        TagEncoding::Direct => {
            bins.k.byte(0);
            bins.niche.byte(0);
            if let Ok(discs) = cx.discriminants(item) {
                bins.k.usize(discs.len());
                bins.niche.usize(discs.len());
                for &disc in discs.iter() {
                    bins.k.u128(disc as u128);
                    bins.niche.u128(disc as u128);
                }
            }
        }
        TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
            for out in [&mut bins.k, &mut bins.niche] {
                out.byte(1);
                out.usize(untagged_variant.index());
                out.usize(niche_variants.start.index());
                out.usize(niche_variants.last.index());
                out.u128(*niche_start);
            }
        }
    }
}

fn hash_header(cx: &mut Context<'_>, item: ItemRef, typed: &Typed, bins: &mut HashBins) {
    let static_item = lower::is_static(cx, item);
    let source = match &typed.shape {
        Shape::Closure(closure) => closure.parent,
        _ => item,
    };
    bins.k.flag(static_item);
    bins.k.flag(cx.outline.crates[source.krate as usize].env.overflow_checks());
    let sig = if static_item { lower::signature_of(cx, None, []) } else { lower::signature(cx, typed) };
    if let Ok(sig) = sig {
        hash_signature(&sig, &mut bins.k);
    }
    for &param in &typed.params {
        let ty = typed.locals[param as usize].ty;
        hash_type_use(cx, ty, bins);
        bins.k.byte(lower::passed_parts(cx, ty).ok().flatten().map_or(0, |parts| match parts {
            lower::Parts::One(_) => 1,
            lower::Parts::Pair(..) => 2,
            lower::Parts::Bytes => 3,
        }));
    }
    if let Some(ret) = typed.ret {
        hash_type_use(cx, ret, bins);
        bins.k.flag(matches!(cx.types.kind(ret), Kind::Never));
        bins.k.flag(lower::passed_parts(cx, ret).ok().flatten().is_some());
    }
    hash_shape_kind(cx, typed, bins);
}

fn hash_shape_kind(cx: &mut Context<'_>, typed: &Typed, bins: &mut HashBins) {
    match &typed.shape {
        Shape::Body => bins.k.byte(0),
        Shape::Glue(glue) => {
            bins.k.byte(1);
            bins.k.flag(glue.drop.is_some());
            bins.k.usize(glue.parts.len());
            bins.k.flag(glue.resume.is_some());
            if let Some(&first) = typed.params.first()
                && let Kind::Ref(_, target_ty) = *cx.types.kind(typed.locals[first as usize].ty)
            {
                hash_type_use(cx, target_ty, bins);
                hash_all_field_offsets(cx, target_ty, bins);
            }
        }
        Shape::Closure(cb) => {
            bins.k.byte(2);
            bins.k.flag(cb.env.is_some());
            bins.k.flag(cb.closure.consumes);
            bins.k.flag(cb.closure.coroutine);
            bins.k.usize(cb.closure.captures.len());
            for cap in &cb.closure.captures {
                bins.k.byte(cap.by as u8);
            }
            if let Some(env_local) = cb.env {
                hash_closure_env_offsets(cx, typed.locals[env_local as usize].ty, bins);
            }
        }
        Shape::Virtual(slot) => {
            bins.k.byte(3);
            bins.k.u32(*slot);
        }
        Shape::Untuple(_) => bins.k.byte(4),
        Shape::Ctor(variant) => {
            bins.k.byte(5);
            bins.k.u32(*variant);
        }
        Shape::Clone(calls) => {
            bins.k.byte(6);
            bins.k.usize(calls.len());
        }
        Shape::Anon(_) => bins.k.byte(7),
    }
}

fn hash_all_field_offsets(cx: &mut Context<'_>, ty: Ty, bins: &mut HashBins) {
    if let Ok(layout) = cx.layout(ty) {
        for idx in 0..layout.fields.count() {
            let off = layout.fields.offset(idx).bytes() as i32;
            bins.k.i32(off);
            bins.offset.i32(off);
        }
    }
}

pub(super) fn hash_closure_env_offsets(cx: &mut Context<'_>, raw_ty: Ty, bins: &mut HashBins) {
    let mut ty = raw_ty;
    while let Kind::Ref(_, inner) | Kind::Ptr(_, inner) = *cx.types.kind(ty) {
        ty = inner;
    }
    let env = cx.ensure_closure(ty).map_or(ty, |info| info.env);
    hash_all_field_offsets(cx, env, bins);
}

fn hash_locals_and_pats(cx: &mut Context<'_>, typed: &Typed, bins: &mut HashBins) {
    bins.k.usize(typed.locals.len());
    for local in &typed.locals {
        hash_type_use(cx, local.ty, bins);
    }
    bins.k.usize(typed.drops.len());
    bins.k.usize(typed.unpacks.len());
    for (&local, &pat) in &typed.unpacks {
        bins.k.u32(local);
        bins.k.u32(pat.0);
    }
    bins.k.usize(typed.pats.len());
    for (&pat, &ty) in &typed.pats {
        bins.k.u32(pat.0);
        hash_type_use(cx, ty, bins);
    }
}

fn body_arena(cx: &mut Context<'_>, item: ItemRef, typed: &Typed) -> Option<Arena> {
    let source = match &typed.shape {
        Shape::Closure(closure) => closure.parent,
        Shape::Body => item,
        Shape::Anon(anon) => return Some(anon.arena.clone()),
        _ => return None,
    };
    cx.body(source).ok().map(Arena::Body)
}

fn hash_body_ast(cx: &mut Context<'_>, reached: &Reached, item: ItemRef, typed: &Typed, bins: &mut HashBins) {
    bins.k.usize(typed.exprs.len());
    for &ty in &typed.exprs {
        hash_type_use(cx, ty, bins);
    }
    let Some(arena) = body_arena(cx, item, typed) else { return };
    let body = arena.body();
    for (idx, expr) in body.exprs.iter().enumerate() {
        hash_expr_site(cx, (reached, body, typed), (ExprId(idx as u32), expr), bins);
    }
    for (idx, pat) in body.pats.iter().enumerate() {
        hash_pat_site(cx, (body, typed), (PatId(idx as u32), pat), bins);
    }
}

fn hash_expr_site(cx: &mut Context<'_>, (reached, body, typed): (&Reached, &Body, &Typed), (id, expr): (ExprId, &Expr), bins: &mut HashBins) {
    let Some(&expr_ty) = typed.exprs.get(id.0 as usize) else { return };
    if expr_ty == Ty::NONE {
        return;
    }
    match *expr {
        Expr::Unary(op, operand) if op != UnOp::Deref => hash_operand_num(cx, typed, operand, bins),
        Expr::Binary(_, lhs, rhs) | Expr::AssignOp(_, lhs, rhs) if !typed.operators.contains_key(&id) => {
            hash_operand_num(cx, typed, lhs, bins);
            hash_operand_num(cx, typed, rhs, bins);
        }
        Expr::Cast(inner, _) => {
            hash_operand_num(cx, typed, inner, bins);
            hash_num_tag(cx, expr_ty, &mut bins.k);
        }
        Expr::TupleField(base, field_idx) => {
            if let Some(container) = autoderef_ty(cx, reached, typed, id, base) {
                hash_field_at(cx, container, 0, field_idx, bins);
            }
        }
        Expr::Field(base, _) => {
            if let (Some(&field_idx), Some(container)) = (typed.fields.get(&id), autoderef_ty(cx, reached, typed, id, base)) {
                hash_field_at(cx, container, 0, field_idx, bins);
            }
        }
        Expr::Struct { fields, .. } => hash_struct_lit(cx, (body, typed), (id, expr_ty, fields), bins),
        Expr::Call(callee, list) => {
            if let Some(Res::Ctor(_, variant, _)) = typed.res.get(&callee) {
                for idx in 0..body.list(list).len() {
                    hash_field_at(cx, expr_ty, *variant, idx as u32, bins);
                }
            }
        }
        _ => {}
    }
}

fn hash_operand_num(cx: &mut Context<'_>, typed: &Typed, expr: ExprId, bins: &mut HashBins) {
    if let Some(&ty) = typed.exprs.get(expr.0 as usize) {
        hash_num_tag(cx, ty, &mut bins.k);
    }
}

fn hash_struct_lit(cx: &mut Context<'_>, (body, typed): (&Body, &Typed), (id, container, fields): (ExprId, Ty, crate::body::List), bins: &mut HashBins) {
    let variant = match typed.res.get(&id) {
        Some(Res::Ctor(_, v, _)) => *v,
        _ => 0,
    };
    for &at in body.list(fields) {
        let val_expr = body.fields[at as usize].value;
        if let Some(&field_idx) = typed.inits.get(&val_expr) {
            hash_field_at(cx, container, variant, field_idx, bins);
        }
    }
}

fn hash_pat_site(cx: &mut Context<'_>, (body, typed): (&Body, &Typed), (id, pat): (PatId, &Pat), bins: &mut HashBins) {
    let Some(&ty) = typed.pats.get(&id) else { return };
    let variant = match typed.pat_res.get(&id) {
        Some(Res::Ctor(_, v, _)) => *v,
        _ => 0,
    };
    match *pat {
        Pat::Range { .. } => hash_num_tag(cx, ty, &mut bins.k),
        Pat::Tuple(items) | Pat::TupleStruct(_, items) => hash_tuple_pat_fields(cx, body, (ty, variant), items, bins),
        Pat::Struct { fields, .. } => {
            for &at in body.list(fields) {
                let sub = body.field_pats[at as usize].pat;
                if let Some(&field_idx) = typed.field_pats.get(&sub) {
                    hash_field_at(cx, ty, variant, field_idx, bins);
                }
            }
        }
        _ => {}
    }
}

fn autoderef_ty(cx: &mut Context<'_>, reached: &Reached, typed: &Typed, expr: ExprId, base: ExprId) -> Option<Ty> {
    let mut ty = *typed.exprs.get(base.0 as usize)?;
    if ty == Ty::NONE {
        return None;
    }
    let steps = typed.autoderefs.get(&expr).copied().unwrap_or(0);
    for step in 0..steps {
        if let Some(Res::Fn(item, substs)) = typed.derefs.get(&(expr, step)) {
            let idx = *reached.index.get(&(*item, *substs))?;
            let ret = reached.list[idx].1.as_ref().ok()?.ret?;
            let Kind::Ref(_, target_ty) = *cx.types.kind(ret) else { return None };
            ty = target_ty;
        } else {
            ty = match cx.types.kind(ty).clone() {
                Kind::Ref(_, inner) | Kind::Ptr(_, inner) => inner,
                _ => cx.boxed(ty)?,
            };
        }
    }
    Some(ty)
}

fn hash_tuple_pat_fields(cx: &mut Context<'_>, body: &Body, (ty, variant): (Ty, u32), items: crate::body::List, bins: &mut HashBins) {
    let subs = body.list(items);
    let rest_at = subs.iter().position(|&p| matches!(body.pat(PatId(p)), Pat::Rest));
    let total = match *cx.types.kind(ty) {
        Kind::Tuple(ts) => cx.types.substs(ts).len(),
        Kind::Adt(item, args) => {
            let slice = cx.types.substs(args).to_vec();
            field_types(cx, item, &slice).map_or(subs.len(), |v| v.get(variant as usize).map_or(subs.len(), Vec::len))
        }
        _ => subs.len(),
    };
    for (at, _) in subs.iter().enumerate() {
        let field_idx = match rest_at {
            None => Some(at),
            Some(r) if at < r => Some(at),
            Some(r) if at > r => total.checked_sub(subs.len() - at),
            Some(_) => None,
        };
        if let Some(idx) = field_idx {
            hash_field_at(cx, ty, variant, idx as u32, bins);
        }
    }
}

pub(super) fn hash_field_at(cx: &mut Context<'_>, container: Ty, variant: u32, field_idx: u32, bins: &mut HashBins) {
    if let Ok(layout) = cx.layout(container) {
        let off = field_offset(&layout, variant, field_idx);
        bins.k.i32(off);
        bins.offset.i32(off);
    }
}
