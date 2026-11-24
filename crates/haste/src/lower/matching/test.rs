use anyhow::{Result, bail};
use haste_codegen::{Block, FloatCC, IntCC, Type, Value};
use rustc_abi::{TagEncoding, Variants};
use rustc_index::Idx;

use crate::body::{Body, ExprId, List, Pat, PatId};
use crate::sema::infer::{Res, SiteKey, Typed};
use crate::sema::layout::target;
use crate::sema::ty::{Kind, Ty, mask};

use super::super::adt::tag_type;
use super::super::moves::{MovePath, Root};
use super::super::ops::Num;
use super::super::place::{Base, Place};
use super::super::reloc::SiteOrigin;
use super::super::{Lower, Repr};

#[derive(Clone, Copy)]
pub(super) struct Hoisted {
    derefs: u32,
    peeled: Option<Place>,
    tag: Option<(Ty, Type, Value)>,
}

fn pat_equality_tests(body: &Body, typed: &Typed, pat: PatId) -> usize {
    let ctor = usize::from(matches!(
        typed.pat_res.get(&pat),
        Some(Res::Ctor(..) | Res::Fn(..))
    ));
    match *body.pat(pat) {
        Pat::Or(list) => body
            .list(list)
            .iter()
            .map(|&at| pat_equality_tests(body, typed, PatId(at)))
            .sum(),
        Pat::Paren(inner)
        | Pat::Bind {
            sub: Some(inner), ..
        } => pat_equality_tests(body, typed, inner),
        Pat::Lit(_) => 1,
        _ => ctor,
    }
}

fn concrete_pat(body: &Body, pat: PatId) -> Option<PatId> {
    match *body.pat(pat) {
        Pat::Wild | Pat::Bind { sub: None, .. } => None,
        Pat::Paren(inner)
        | Pat::Bind {
            sub: Some(inner), ..
        } => concrete_pat(body, inner),
        Pat::Or(list) => body
            .list(list)
            .iter()
            .find_map(|&at| concrete_pat(body, PatId(at))),
        _ => Some(pat),
    }
}

impl Lower<'_, '_> {
    pub(super) fn hoist_match_scrutinee(
        &mut self,
        arm_ids: &[u32],
        place: Option<Place>,
    ) -> Result<Option<Hoisted>> {
        let total_tests: usize = arm_ids
            .iter()
            .map(|&at| pat_equality_tests(self.body, self.typed, self.body.arms[at as usize].pat))
            .sum();
        if total_tests < 2 {
            return Ok(None);
        }
        let Some(first_pat) = arm_ids
            .iter()
            .find_map(|&at| concrete_pat(self.body, self.body.arms[at as usize].pat))
        else {
            return Ok(None);
        };
        let derefs = self.typed.pat_derefs.get(&first_pat).copied().unwrap_or(0);
        let Some(peeled) = self.peeled(first_pat, place)? else {
            return Ok(None);
        };
        if matches!(peeled.repr, Repr::Scalar(_)) && matches!(peeled.base, Base::Addr(_)) {
            let (Some(v), Some(slot)) = (self.read(peeled), self.slot(peeled.repr)) else {
                return Ok(None);
            };
            let tmp = Place::slot(slot, peeled.repr);
            self.write(tmp, v);
            return Ok(Some(Hoisted {
                derefs,
                peeled: Some(tmp),
                tag: None,
            }));
        }
        let Ok(ty) = self.pat_ty(first_pat) else {
            return Ok(None);
        };
        let Kind::Adt(..) = *self.cx.types.kind(ty) else {
            return Ok(None);
        };
        let layout = self.cx.layout(ty)?;
        let Variants::Multiple {
            tag,
            tag_encoding: TagEncoding::Direct,
            tag_field,
            ..
        } = &layout.variants
        else {
            return Ok(Some(Hoisted {
                derefs,
                peeled: Some(peeled),
                tag: None,
            }));
        };
        let bytes = tag.size(target()?).bytes();
        let Ok(tag_ty) = tag_type(bytes) else {
            return Ok(Some(Hoisted {
                derefs,
                peeled: Some(peeled),
                tag: None,
            }));
        };
        let offset = layout.fields.offset(tag_field.index()).bytes() as i32;
        let Some(val) = self.read(peeled.at(offset, Repr::Scalar(tag_ty))) else {
            return Ok(Some(Hoisted {
                derefs,
                peeled: Some(peeled),
                tag: None,
            }));
        };
        Ok(Some(Hoisted {
            derefs,
            peeled: Some(peeled),
            tag: Some((ty, tag_ty, val)),
        }))
    }

    /// Branches to `fail` unless the value at `place` matches `pat`.
    pub(super) fn test(&mut self, pat: PatId, place: Option<Place>, fail: Block) -> Result<()> {
        self.test_with_at(pat, place, None, None, fail)
    }

    pub(super) fn test_at(
        &mut self,
        pat: PatId,
        place: Option<Place>,
        at: Option<(Root, MovePath)>,
        fail: Block,
    ) -> Result<()> {
        self.test_with_at(pat, place, at, None, fail)
    }

    fn or_same_derefs(&self, list: List, derefs: u32) -> bool {
        self.body.list(list).iter().all(|&at| {
            let alt = PatId(at);
            match *self.body.pat(alt) {
                Pat::Or(sub) => self.or_same_derefs(sub, derefs),
                Pat::Wild | Pat::Bind { sub: None, .. } => true,
                _ => self.typed.pat_derefs.get(&alt).copied().unwrap_or(0) == derefs,
            }
        })
    }

    fn hoisted_for(&self, pat: PatId, hoisted: Option<Hoisted>) -> Option<Hoisted> {
        hoisted.filter(|h| match *self.body.pat(pat) {
            Pat::Wild | Pat::Bind { sub: None, .. } => false,
            Pat::Or(list) => self.or_same_derefs(list, h.derefs),
            _ => self.typed.pat_derefs.get(&pat).copied().unwrap_or(0) == h.derefs,
        })
    }

    pub(super) fn test_with_at(
        &mut self,
        pat: PatId,
        place: Option<Place>,
        at: Option<(Root, MovePath)>,
        hoisted: Option<Hoisted>,
        fail: Block,
    ) -> Result<()> {
        let at = at.filter(|_| self.typed.pat_derefs.get(&pat).copied().unwrap_or(0) == 0);
        let h = self.hoisted_for(pat, hoisted);
        let place = match h {
            Some(h) => h.peeled,
            None => self.peeled(pat, place)?,
        };
        if let Some((hty, tag_ty, tag_val)) = h.and_then(|h| h.tag)
            && let Some(Res::Ctor(_, variant, _)) = self.typed.pat_res.get(&pat).cloned()
            && self.pat_ty(pat)? == hty
            && let Kind::Adt(item, _) = *self.cx.types.kind(hty)
        {
            let expected =
                self.cx.discriminants(item)?[variant as usize] as u128 & mask(tag_ty.bits());
            let expected = self.int_const(tag_ty, expected);
            let ok = self.ins().icmp(IntCC::Equal, tag_val, expected);
            self.pass(ok, fail);
        } else {
            self.test_variant(pat, place, fail)?;
        }
        self.test_const(pat, place, None, fail)?;
        match *self.body.pat(pat) {
            Pat::Bind { sub: Some(sub), .. } => self.test_at(sub, place, at, fail),
            Pat::Paren(inner) => self.test_at(inner, place, at, fail),
            Pat::Lit(lit) => self.test_lit(pat, lit, place, None, fail),
            Pat::Range {
                from,
                to,
                inclusive,
            } => self.test_range(pat, (from, to, inclusive), place, fail),
            Pat::Ref { pat: inner, .. } => {
                let pointee = self.ref_target(pat, place)?;
                self.test(inner, Some(pointee), fail)
            }
            Pat::Or(list) => self.test_or(list, place, at, hoisted, fail),
            Pat::Tuple(_) | Pat::TupleStruct(..) | Pat::Struct { .. } | Pat::Slice(_) => {
                for (sub, sub_place, sub_at, _) in
                    self.subpatterns_at(pat, place, at, Some(fail))?
                {
                    self.test_at(sub, sub_place, sub_at, fail)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn test_variant(&mut self, pat: PatId, place: Option<Place>, fail: Block) -> Result<()> {
        let Some(Res::Ctor(_, variant, _)) = self.typed.pat_res.get(&pat).cloned() else {
            return Ok(());
        };
        let ty = self.pat_ty(pat)?;
        if let Some(ok) = self.is_variant(place, ty, variant)? {
            self.pass(ok, fail);
        }
        Ok(())
    }

    fn try_leaf_or(
        &mut self,
        mut pat: PatId,
        place: Option<Place>,
        hoisted: Option<Hoisted>,
        matched: Block,
        fail: Block,
    ) -> Result<bool> {
        while let Pat::Paren(inner) = *self.body.pat(pat) {
            pat = inner;
        }
        let h = self.hoisted_for(pat, hoisted);
        let place = match h {
            Some(h) => h.peeled,
            None => self.peeled(pat, place)?,
        };
        match *self.body.pat(pat) {
            Pat::Lit(lit) => {
                self.test_lit(pat, lit, place, Some(matched), fail)?;
                Ok(true)
            }
            Pat::Path(_) if matches!(self.typed.pat_res.get(&pat), Some(Res::Fn(..))) => {
                self.test_const(pat, place, Some(matched), fail)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn test_or(
        &mut self,
        list: List,
        place: Option<Place>,
        at: Option<(Root, MovePath)>,
        hoisted: Option<Hoisted>,
        fail: Block,
    ) -> Result<()> {
        let matched = self.b.create_block();
        let alts: Vec<PatId> = self.body.list(list).iter().map(|&at| PatId(at)).collect();
        for (idx, &alt) in alts.iter().enumerate() {
            let next = if idx + 1 == alts.len() {
                fail
            } else {
                self.b.create_block()
            };
            if !self.try_leaf_or(alt, place, hoisted, matched, next)? {
                self.test_with_at(alt, place, at.clone(), hoisted, next)?;
                self.jump(matched, &[]);
            }
            if next != fail {
                self.enter(next);
            }
        }
        self.enter(matched);
        Ok(())
    }

    fn scalar(&mut self, pat: PatId, place: Option<Place>) -> Result<(Value, Num)> {
        let num = self.num(self.pat_ty(pat)?)?;
        let Some(value) = place.and_then(|place| self.read(place)) else {
            bail!("literal pattern on a value with no bytes");
        };
        Ok((value, num))
    }

    fn test_const(
        &mut self,
        pat: PatId,
        place: Option<Place>,
        on_match: Option<Block>,
        fail: Block,
    ) -> Result<()> {
        let Some(&Res::Fn(item, substs)) = self.typed.pat_res.get(&pat) else {
            return Ok(());
        };
        let Some(expected) = self.invoke(
            SiteOrigin::Table(SiteKey::PatRes(pat)),
            item,
            substs,
            Vec::new(),
        )?
        else {
            bail!("constant pattern with no value");
        };
        self.test_equal(pat, place, expected, on_match, fail)
    }

    fn test_lit(
        &mut self,
        pat: PatId,
        lit: ExprId,
        place: Option<Place>,
        on_match: Option<Block>,
        fail: Block,
    ) -> Result<()> {
        let Some(expected) = self.expr(lit)? else {
            bail!("literal with no value");
        };
        self.test_equal(pat, place, expected, on_match, fail)
    }

    fn test_equal(
        &mut self,
        pat: PatId,
        place: Option<Place>,
        expected: Value,
        on_match: Option<Block>,
        fail: Block,
    ) -> Result<()> {
        let ty = self.pat_ty(pat)?;
        let Some(value) = place.and_then(|place| self.read(place)) else {
            bail!("literal pattern on a value with no bytes");
        };
        let ok = self.equal(ty, value, expected)?;
        match on_match {
            Some(matched) => self.brif(ok, (matched, &[]), (fail, &[])),
            None => self.pass(ok, fail),
        }
        Ok(())
    }

    fn test_range(
        &mut self,
        pat: PatId,
        (from, to, inclusive): (Option<ExprId>, Option<ExprId>, bool),
        place: Option<Place>,
        fail: Block,
    ) -> Result<()> {
        let (value, num) = self.scalar(pat, place)?;
        let signed = matches!(num, Num::Int { signed: true, .. });
        let (low, high) = if signed {
            (IntCC::SignedGreaterThanOrEqual, IntCC::SignedLessThan)
        } else {
            (IntCC::UnsignedGreaterThanOrEqual, IntCC::UnsignedLessThan)
        };
        let high = match (inclusive, high) {
            (false, cc) => cc,
            (true, IntCC::SignedLessThan) => IntCC::SignedLessThanOrEqual,
            (true, _) => IntCC::UnsignedLessThanOrEqual,
        };
        for (end, cc, fcc) in [
            (from, low, FloatCC::GreaterThanOrEqual),
            (
                to,
                high,
                if inclusive {
                    FloatCC::LessThanOrEqual
                } else {
                    FloatCC::LessThan
                },
            ),
        ] {
            let Some(end) = end else { continue };
            let Some(bound) = self.expr(end)? else {
                bail!("range end with no value");
            };
            let ok = match num {
                Num::Float(_) => self.ins().fcmp(fcc, value, bound),
                _ => self.ins().icmp(cc, value, bound),
            };
            self.pass(ok, fail);
        }
        Ok(())
    }
}
