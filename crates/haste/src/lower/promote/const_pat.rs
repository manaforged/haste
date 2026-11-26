use rustc_abi::{TagEncoding, Variants};
use rustc_index::Idx;

use crate::body::{List, Pat, PatId};
use crate::sema::infer::{Res, field_types, position, rest_split};
use crate::sema::ty::{Kind, Ty, mask};

use super::const_val::{ConstEnv, Frame, PlaceRef, PtrTarget};
use super::literal::Scope;

impl<'l, 'a, 'r> ConstEnv<'l, 'a, 'r> {
    /// Tests pattern `pat` against `scrut` (and binds frame locals when `bind` is true).
    pub(super) fn match_and_bind(
        &mut self,
        scope: Scope<'_>,
        pat: PatId,
        scrut: &PlaceRef,
        frame: &mut Frame,
        bind: bool,
    ) -> Option<bool> {
        let place = self.peel_pat_place(scope, pat, scrut)?;
        if !self.match_pat_res(scope, pat, &place)? {
            return Some(false);
        }
        self.match_pat_body(scope, pat, &place, frame, bind)
    }

    fn match_pat_res(&mut self, scope: Scope<'_>, pat: PatId, place: &PlaceRef) -> Option<bool> {
        if let Some(Res::Ctor(_, variant, _)) = scope.typed.pat_res.get(&pat).copied()
            && !self.adt_holds_variant(place, *scope.typed.pats.get(&pat)?, variant)?
        {
            return Some(false);
        }
        if let Some(&Res::Fn(item, substs)) = scope.typed.pat_res.get(&pat) {
            let pat_ty = *scope.typed.pats.get(&pat)?;
            let expected = self.eval_const_item_val(item, substs, pat_ty)?;
            let actual = self.read_place(place)?;
            return Some(expected.bytes == actual.bytes);
        }
        Some(true)
    }

    fn match_pat_body(
        &mut self,
        scope: Scope<'_>,
        pat: PatId,
        place: &PlaceRef,
        frame: &mut Frame,
        bind: bool,
    ) -> Option<bool> {
        match *scope.body.pat(pat) {
            Pat::Wild | Pat::Path(_) => Some(true),
            Pat::Paren(inner) => self.match_and_bind(scope, inner, place, frame, bind),
            Pat::Bind { sub, by_ref, .. } if !scope.typed.pat_res.contains_key(&pat) => {
                self.match_bind_pat(scope, (pat, sub, by_ref), place, frame, bind)
            }
            Pat::Bind { .. } => Some(true),
            Pat::Ref { pat: inner, .. } => {
                let ptr_val = self.read_place(place)?;
                let Kind::Ref(_, pointee) = *self.lower.cx.types.kind(*scope.typed.pats.get(&pat)?)
                else {
                    return None;
                };
                let inner_place = self.place_from_ref_val(&ptr_val, pointee)?;
                self.match_and_bind(scope, inner, &inner_place, frame, bind)
            }
            Pat::Lit(lit_expr) => {
                let exp = self.eval_expr_val(scope, lit_expr, frame)?;
                let act = self.read_place(place)?;
                Some(exp.bytes == act.bytes)
            }
            _ => self.match_pat_compound(scope, pat, place, frame, bind),
        }
    }

    fn match_pat_compound(
        &mut self,
        scope: Scope<'_>,
        pat: PatId,
        place: &PlaceRef,
        frame: &mut Frame,
        bind: bool,
    ) -> Option<bool> {
        match *scope.body.pat(pat) {
            Pat::Or(list) => self.match_or_pat(scope, list, place, frame, bind),
            Pat::Slice(list) => self.match_slice_pat(scope, list, place, frame, bind),
            Pat::Tuple(_) | Pat::TupleStruct(..) | Pat::Struct { .. } => {
                self.match_subpats(scope, pat, place, frame, bind)
            }
            _ => None,
        }
    }

    fn match_bind_pat(
        &mut self,
        scope: Scope<'_>,
        (pat, sub, by_ref): (PatId, Option<PatId>, bool),
        place: &PlaceRef,
        frame: &mut Frame,
        bind: bool,
    ) -> Option<bool> {
        if let Some(inner) = sub
            && !self.match_and_bind(scope, inner, place, frame, bind)?
        {
            return Some(false);
        }
        if bind {
            self.bind_local_pat(scope, pat, by_ref, place, frame)?;
        }
        Some(true)
    }

    fn match_or_pat(
        &mut self,
        scope: Scope<'_>,
        list: List,
        place: &PlaceRef,
        frame: &mut Frame,
        bind: bool,
    ) -> Option<bool> {
        for &raw in scope.body.list(list) {
            let alt = PatId(raw);
            if self.match_and_bind(scope, alt, place, frame, false)? {
                if bind {
                    self.match_and_bind(scope, alt, place, frame, true)?;
                }
                return Some(true);
            }
        }
        Some(false)
    }

    fn match_subpats(
        &mut self,
        scope: Scope<'_>,
        pat: PatId,
        place: &PlaceRef,
        frame: &mut Frame,
        bind: bool,
    ) -> Option<bool> {
        let variant = match scope.typed.pat_res.get(&pat) {
            Some(&Res::Ctor(_, v, _)) => v,
            _ => 0,
        };
        let subs: Vec<(PatId, u32)> = match *scope.body.pat(pat) {
            Pat::Struct { fields, .. } => scope
                .body
                .list(fields)
                .iter()
                .map(|&raw| {
                    let sub = scope.body.field_pats[raw as usize].pat;
                    Some((sub, *scope.typed.field_pats.get(&sub)?))
                })
                .collect::<Option<Vec<_>>>()?,
            Pat::Tuple(list) | Pat::TupleStruct(_, list) => {
                let count = self.adt_subpat_count(place.ty, variant)?;
                let (pats, rest) = rest_split(scope.body, list).ok()?;
                pats.iter()
                    .enumerate()
                    .filter_map(|(at, &sub)| {
                        position(at, pats.len(), rest, count).map(|idx| (sub, idx as u32))
                    })
                    .collect()
            }
            _ => return None,
        };
        for (sub, idx) in subs {
            let sub_place = self.part_place_ref(place, (variant, idx))?;
            if !self.match_and_bind(scope, sub, &sub_place, frame, bind)? {
                return Some(false);
            }
        }
        Some(true)
    }

    fn adt_subpat_count(&mut self, ty: Ty, variant: u32) -> Option<usize> {
        match *self.lower.cx.types.kind(ty) {
            Kind::Tuple(items) => Some(self.lower.cx.types.substs(items).len()),
            Kind::Adt(item, args) => {
                let s = self.lower.cx.types.substs(args).to_vec();
                Some(
                    field_types(self.lower.cx, item, &s)
                        .ok()?
                        .get(variant as usize)?
                        .len(),
                )
            }
            _ => None,
        }
    }

    fn match_slice_pat(
        &mut self,
        scope: Scope<'_>,
        list: List,
        place: &PlaceRef,
        frame: &mut Frame,
        bind: bool,
    ) -> Option<bool> {
        let (elem_ty, total) = self.slice_pat_target(place)?;
        let (pats, rest) = rest_split(scope.body, list).ok()?;
        let min_len = pats.len() - usize::from(rest.is_some());
        if (rest.is_some() && total < min_len) || (rest.is_none() && total != min_len) {
            return Some(false);
        }
        let stride = self.lower.cx.layout(elem_ty).ok()?.size.bytes();
        for (at, &sub) in pats.iter().enumerate() {
            let sub_place = self.slice_elem_place_pat(
                scope,
                sub,
                (at, pats.len(), rest, total, min_len),
                (place, elem_ty, stride),
            )?;
            if let Some(elem_place) = sub_place
                && !self.match_and_bind(scope, sub, &elem_place, frame, bind)?
            {
                return Some(false);
            }
        }
        Some(true)
    }

    fn slice_pat_target(&self, place: &PlaceRef) -> Option<(Ty, usize)> {
        match *self.lower.cx.types.kind(place.ty) {
            Kind::Array(elem, len_ty) => Some((
                elem,
                usize::try_from(self.lower.cx.types.len(len_ty)?).ok()?,
            )),
            Kind::Slice(elem) => Some((elem, usize::try_from(place.slice_len?).ok()?)),
            _ => None,
        }
    }

    fn slice_elem_place_pat(
        &self,
        scope: Scope<'_>,
        sub: PatId,
        (at, pat_len, rest, total, min_len): (usize, usize, Option<usize>, usize, usize),
        (place, elem_ty, stride): (&PlaceRef, Ty, u64),
    ) -> Option<Option<PlaceRef>> {
        if let Some(idx) = position(at, pat_len, rest, total) {
            let target = place.target.offset(stride.checked_mul(idx as u64)?)?;
            return Some(Some(PlaceRef {
                target,
                ty: elem_ty,
                slice_len: None,
            }));
        }
        if matches!(*scope.body.pat(sub), Pat::Rest) {
            return Some(None);
        }
        let start = place.target.offset(stride.checked_mul(at as u64)?)?;
        let slice_ty = *scope.typed.pats.get(&sub)?;
        Some(Some(PlaceRef {
            target: start,
            ty: slice_ty,
            slice_len: Some((total - min_len) as u64),
        }))
    }

    fn peel_pat_place(
        &mut self,
        scope: Scope<'_>,
        pat: PatId,
        scrut: &PlaceRef,
    ) -> Option<PlaceRef> {
        let derefs = scope.typed.pat_derefs.get(&pat).copied().unwrap_or(0);
        let mut place = scrut.clone();
        for level in 0..derefs {
            let ptr_val = self.read_place(&place)?;
            let inner = if level + 1 == derefs {
                *scope.typed.pats.get(&pat)?
            } else if let Kind::Ref(_, inner) = *self.lower.cx.types.kind(place.ty) {
                inner
            } else {
                return None;
            };
            place = self.place_from_ref_val(&ptr_val, inner)?;
        }
        Some(place)
    }

    fn bind_local_pat(
        &mut self,
        scope: Scope<'_>,
        pat: PatId,
        by_ref: bool,
        place: &PlaceRef,
        frame: &mut Frame,
    ) -> Option<()> {
        let &local = scope.typed.binds.get(&pat)?;
        let referenced = by_ref || scope.typed.ref_binds.contains_key(&pat);
        let val = if referenced {
            self.make_ref(place)
        } else {
            self.read_place(place)?
        };
        let id = self.alloc_val(val)?;
        *frame.locals.get_mut(local as usize)? = Some(id);
        Some(())
    }

    fn adt_holds_variant(&mut self, place: &PlaceRef, ty: Ty, variant: u32) -> Option<bool> {
        let Kind::Adt(item, _) = *self.lower.cx.types.kind(ty) else {
            return None;
        };
        let layout = self.lower.cx.layout(ty).ok()?;
        match &layout.variants {
            Variants::Empty => Some(false),
            Variants::Single { index } => Some(index.index() == variant as usize),
            Variants::Multiple {
                tag,
                tag_encoding,
                tag_field,
                ..
            } => {
                let tag_bytes = tag.size(crate::sema::layout::target().ok()?).bytes() as usize;
                let off = u64::try_from(layout.fields.offset(tag_field.index()).bytes()).ok()?;
                self.check_multiple_tag(
                    item,
                    variant,
                    place.target.offset(off)?,
                    tag_bytes,
                    tag_encoding,
                )
            }
        }
    }

    fn check_multiple_tag(
        &mut self,
        item: crate::names::ItemRef,
        variant: u32,
        target: PtrTarget,
        tag_bytes: usize,
        tag_encoding: &TagEncoding<crate::sema::layout::Variant>,
    ) -> Option<bool> {
        let m = mask(u32::try_from(tag_bytes * 8).ok()?);
        let raw = self.read_raw_u128(target, tag_bytes)? & m;
        match tag_encoding {
            TagEncoding::Direct => {
                let exp = *self
                    .lower
                    .cx
                    .discriminants(item)
                    .ok()?
                    .get(variant as usize)? as u128
                    & m;
                Some(raw == exp)
            }
            TagEncoding::Niche {
                untagged_variant,
                niche_variants,
                niche_start,
            } => {
                let first = niche_variants.start.index() as u128;
                let span = niche_variants.last.index() as u128 - first;
                let key = raw.wrapping_sub(niche_start & m) & m;
                if untagged_variant.index() as u128 == u128::from(variant) {
                    Some(key > span)
                } else {
                    let exp = u128::from(variant).checked_sub(first)?;
                    Some(key == (exp & m))
                }
            }
        }
    }

    fn read_raw_u128(&self, target: PtrTarget, size: usize) -> Option<u128> {
        let PtrTarget::Obj { id, addend } = target else {
            return None;
        };
        let obj = self.objs.get(id as usize)?;
        let start = usize::try_from(addend).ok()?;
        let slice = obj.bytes.get(start..start + size)?;
        let mut buf = [0u8; 16];
        buf[..size].copy_from_slice(slice);
        Some(u128::from_le_bytes(buf))
    }
}
