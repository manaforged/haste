use anyhow::{Result, bail};
use haste_codegen::{Block, IntCC, Type, Value};

use crate::body::{ExprId, List, Pat, PatId};
use crate::sema::infer::{Res, position, rest_split};
use crate::sema::ty::{Kind, Ty};

use super::drop::Pending;
use super::moves::{MovePath, Root};
use super::ops::Num;
use super::place::{Base, Place};
use super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Lowers `match`: each arm tests its pattern and falls through to the next one when it fails.
    pub fn match_expr(&mut self, expr: ExprId, scrutinee: ExprId, arms: List) -> Result<Option<Value>> {
        let place = self.place_or_temp(scrutinee)?;
        if !self.open {
            return Ok(None);
        }
        let from = self.scrutinee_path(scrutinee);
        let (join, joined, slot) = self.join_block(expr)?;
        for at in self.body.list(arms).to_vec() {
            if !self.open {
                break;
            }
            let arm = self.body.arms[at as usize];
            let next = self.b.create_block();
            self.test(arm.pat, place, next)?;
            let mark = self.owned.len();
            self.bind_matched(arm.pat, place, from.clone())?;
            if let Some(guard) = arm.guard {
                let Some(ok) = self.expr(guard)? else { bail!("guard with no value") };
                self.guard_pass(ok, mark, next);
            }
            self.commit_moves()?;
            let value = self.expr(arm.body)?;
            self.end_scope(mark)?;
            self.branch_into(value, (join, slot));
            self.enter(next);
        }
        if self.open {
            self.ins().trap();
            self.close();
        }
        self.enter(join);
        Ok(self.joined(joined, slot))
    }

    /// Lowers the condition `let pat = init` to whether it matched, binding its names on the way that did.
    pub fn let_expr(&mut self, pat: PatId, init: ExprId) -> Result<Option<Value>> {
        let place = self.place_or_temp(init)?;
        let (fail, done) = (self.b.create_block(), self.b.create_block());
        let matched = self.b.append_block_param(done, Type::I8);
        self.test(pat, place, fail)?;
        self.bind_matched(pat, place, self.scrutinee_path(init))?;
        self.commit_moves()?;
        let yes = self.ins().iconst(Type::I8, 1);
        self.jump(done, &[yes]);
        self.enter(fail);
        let no = self.ins().iconst(Type::I8, 0);
        self.jump(done, &[no]);
        self.enter(done);
        Ok(Some(matched))
    }

    /// Lowers `let pat = init else { otherwise };`: binds the names when `pat` matches, and runs the diverging `otherwise` when it does not.
    pub fn let_else(&mut self, pat: PatId, init: ExprId, otherwise: ExprId) -> Result<()> {
        let place = self.place_or_temp(init)?;
        if !self.open {
            return Ok(());
        }
        let (fail, matched) = (self.b.create_block(), self.b.create_block());
        self.test(pat, place, fail)?;
        self.jump(matched, &[]);
        self.enter(fail);
        self.expr(otherwise)?;
        if self.open {
            self.ins().trap();
            self.close();
        }
        self.enter(matched);
        self.bind_matched(pat, place, self.scrutinee_path(init))?;
        self.commit_moves()
    }

    fn guard_pass(&mut self, ok: Value, mark: usize, next: Block) {
        let (pass, fail) = (self.b.create_block(), self.b.create_block());
        self.brif(ok, (pass, &[]), (fail, &[]));
        self.enter(fail);
        self.forget_from(mark);
        self.jump(next, &[]);
        self.enter(pass);
    }

    pub(super) fn pass(&mut self, ok: Value, fail: Block) {
        let pass = self.b.create_block();
        self.brif(ok, (pass, &[]), (fail, &[]));
        self.enter(pass);
    }

    fn peeled(&mut self, pat: PatId, mut place: Option<Place>) -> Result<Option<Place>> {
        let derefs = self.typed.pat_derefs.get(&pat).copied().unwrap_or(0);
        for level in 0..derefs {
            let Some(pointer) = place.and_then(|place| self.read(place)) else { bail!("reference with no value") };
            place = Some(match level + 1 == derefs {
                true => self.pointee_place(pointer, self.pat_ty(pat)?)?,
                false => Place { base: Base::Addr(pointer), offset: 0, repr: Repr::Scalar(Type::I64) },
            });
        }
        Ok(place)
    }

    /// The place that the reference matched by the `&` pattern `pat` points to, laid out as the reference's pointee type, which the inner pattern may deref again.
    fn ref_target(&mut self, pat: PatId, place: Option<Place>) -> Result<Place> {
        let Kind::Ref(_, pointee) = *self.cx.types.kind(self.pat_ty(pat)?) else { bail!("a `&` pattern on a value that is not a reference") };
        let repr = repr(self.cx, pointee)?;
        let Some(pointer) = place.and_then(|place| self.read(place)) else { bail!("reference with no value") };
        Ok(Place { base: Base::Addr(pointer), offset: 0, repr })
    }

    pub(super) fn pat_ty(&self, pat: PatId) -> Result<Ty> {
        self.typed.pats.get(&pat).copied().ok_or_else(|| anyhow::anyhow!("pattern has no type"))
    }

    /// Branches to `fail` unless the value at `place` matches `pat`.
    fn test(&mut self, pat: PatId, place: Option<Place>, fail: Block) -> Result<()> {
        let place = self.peeled(pat, place)?;
        self.test_variant(pat, place, fail)?;
        self.test_const(pat, place, fail)?;
        match *self.body.pat(pat) {
            Pat::Bind { sub: Some(sub), .. } => self.test(sub, place, fail),
            Pat::Paren(inner) => self.test(inner, place, fail),
            Pat::Lit(lit) => self.test_lit(pat, lit, place, fail),
            Pat::Range { from, to, inclusive } => self.test_range(pat, (from, to, inclusive), place, fail),
            Pat::Ref { pat: inner, .. } => {
                let pointee = self.ref_target(pat, place)?;
                self.test(inner, Some(pointee), fail)
            }
            Pat::Or(list) => self.test_or(list, place, fail),
            Pat::Tuple(_) | Pat::TupleStruct(..) | Pat::Struct { .. } | Pat::Slice(_) => {
                for (sub, sub_place, _) in self.subpatterns(pat, place, Some(fail))? {
                    self.test(sub, sub_place, fail)?;
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

    fn test_or(&mut self, list: List, place: Option<Place>, fail: Block) -> Result<()> {
        let matched = self.b.create_block();
        let alts: Vec<PatId> = self.body.list(list).iter().map(|&at| PatId(at)).collect();
        for (at, alt) in alts.iter().enumerate() {
            let next = if at + 1 == alts.len() { fail } else { self.b.create_block() };
            self.test(*alt, place, next)?;
            self.jump(matched, &[]);
            if next != fail {
                self.enter(next);
            }
        }
        self.enter(matched);
        Ok(())
    }

    fn scalar(&mut self, pat: PatId, place: Option<Place>) -> Result<(Value, Num)> {
        let num = self.num(self.pat_ty(pat)?)?;
        let Some(value) = place.and_then(|place| self.read(place)) else { bail!("literal pattern on a value with no bytes") };
        Ok((value, num))
    }

    fn test_const(&mut self, pat: PatId, place: Option<Place>, fail: Block) -> Result<()> {
        let Some(Res::Fn(item, substs)) = self.typed.pat_res.get(&pat).cloned() else {
            return Ok(());
        };
        let Some(expected) = self.invoke(item, &substs, Vec::new())? else { bail!("constant pattern with no value") };
        self.test_equal(pat, place, expected, fail)
    }

    fn test_lit(&mut self, pat: PatId, lit: ExprId, place: Option<Place>, fail: Block) -> Result<()> {
        let Some(expected) = self.expr(lit)? else { bail!("literal with no value") };
        self.test_equal(pat, place, expected, fail)
    }

    fn test_equal(&mut self, pat: PatId, place: Option<Place>, expected: Value, fail: Block) -> Result<()> {
        let ty = self.pat_ty(pat)?;
        let Some(value) = place.and_then(|place| self.read(place)) else { bail!("literal pattern on a value with no bytes") };
        let ok = self.equal(ty, value, expected)?;
        self.pass(ok, fail);
        Ok(())
    }

    fn test_range(&mut self, pat: PatId, (from, to, inclusive): (Option<ExprId>, Option<ExprId>, bool), place: Option<Place>, fail: Block) -> Result<()> {
        let (value, num) = self.scalar(pat, place)?;
        let signed = matches!(num, Num::Int { signed: true, .. });
        let (low, high) = if signed { (IntCC::SignedGreaterThanOrEqual, IntCC::SignedLessThan) } else { (IntCC::UnsignedGreaterThanOrEqual, IntCC::UnsignedLessThan) };
        let high = match (inclusive, high) {
            (false, cc) => cc,
            (true, IntCC::SignedLessThan) => IntCC::SignedLessThanOrEqual,
            (true, _) => IntCC::UnsignedLessThanOrEqual,
        };
        for (end, cc) in [(from, low), (to, high)] {
            let Some(end) = end else { continue };
            let Some(bound) = self.expr(end)? else { bail!("range end with no value") };
            let ok = match num {
                Num::Float(_) => bail!("float range patterns are not supported yet"),
                _ => self.ins().icmp(cc, value, bound),
            };
            self.pass(ok, fail);
        }
        Ok(())
    }

    /// The sub-patterns of a tuple, tuple-struct, struct or slice pattern `pat`, each with its place and the field it matches; a slice tests its length against `fail` first.
    fn subpatterns(&mut self, pat: PatId, place: Option<Place>, fail: Option<Block>) -> Result<Vec<(PatId, Option<Place>, Option<(u32, u32)>)>> {
        if let Pat::Slice(list) = *self.body.pat(pat) {
            return Ok(self.slice_parts((pat, list), place, fail)?.into_iter().map(|(sub, at)| (sub, at, None)).collect());
        }
        let ty = self.pat_ty(pat)?;
        let mut found = Vec::new();
        for (sub, (variant, index)) in self.part_indices(pat)? {
            let sub_place = match place {
                Some(place) => Some(self.part_place(place, ty, (variant, index))?),
                None => None,
            };
            found.push((sub, sub_place, Some((variant, index))));
        }
        Ok(found)
    }

    /// The sub-patterns of a tuple, tuple-struct or struct pattern `pat`, each with the variant and field it matches.
    pub(super) fn part_indices(&mut self, pat: PatId) -> Result<Vec<(PatId, (u32, u32))>> {
        let ty = self.pat_ty(pat)?;
        let variant = match self.typed.pat_res.get(&pat) {
            Some(Res::Ctor(_, variant, _)) => *variant,
            _ => 0,
        };
        let subs: Vec<(PatId, u32)> = match *self.body.pat(pat) {
            Pat::Struct { fields, .. } => self.body.list(fields).iter().map(|&at| self.body.field_pats[at as usize].pat).map(|sub| (sub, self.typed.field_pats.get(&sub).copied().unwrap_or(u32::MAX))).collect(),
            Pat::Tuple(list) | Pat::TupleStruct(_, list) => {
                let count = self.field_count(ty, variant)?;
                let (pats, rest) = rest_split(self.body, list)?;
                pats.iter().enumerate().filter_map(|(at, sub)| position(at, pats.len(), rest, count).map(|index| (*sub, index as u32))).collect()
            }
            _ => Vec::new(),
        };
        Ok(subs.into_iter().map(|(sub, index)| (sub, (variant, index))).collect())
    }

    fn field_count(&mut self, ty: Ty, variant: u32) -> Result<usize> {
        Ok(match self.cx.types.kind(ty).clone() {
            Kind::Tuple(items) => items.len(),
            Kind::Adt(item, args) => crate::sema::infer::field_types(self.cx, item, &args)?[variant as usize].len(),
            _ => bail!("positional pattern on a type with no fields"),
        })
    }

    /// Declares the bindings of `pat`, which matched the value at `place`, and notes which parts of `at` its by-value bindings move out.
    pub fn bind_matched(&mut self, pat: PatId, place: Option<Place>, at: Option<(Root, MovePath)>) -> Result<()> {
        let at = at.filter(|_| self.typed.pat_derefs.get(&pat).copied().unwrap_or(0) == 0);
        let place = self.peeled(pat, place)?;
        match *self.body.pat(pat) {
            Pat::Bind { sub, by_ref, .. } if !self.typed.pat_res.contains_key(&pat) => {
                self.bind_local(pat, by_ref, place, at)?;
                sub.map_or(Ok(()), |sub| self.bind_matched(sub, place, None))
            }
            Pat::Paren(inner) => self.bind_matched(inner, place, at),
            Pat::Or(list) if self.typed.or_binds.contains(&pat) => self.bind_or(list, place, at),
            Pat::Ref { pat: inner, .. } => {
                let pointee = self.ref_target(pat, place)?;
                self.bind_matched(inner, Some(pointee), None)
            }
            Pat::Tuple(_) | Pat::TupleStruct(..) | Pat::Struct { .. } | Pat::Slice(_) => {
                for (sub, sub_place, step) in self.subpatterns(pat, place, None)? {
                    let sub_at = self.sub_at(at.clone(), step, sub)?;
                    self.bind_matched(sub, sub_place, sub_at)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn bind_or(&mut self, list: List, place: Option<Place>, from: Option<(Root, MovePath)>) -> Result<()> {
        let join = self.b.create_block();
        let which = self.b.append_block_param(join, Type::I8);
        let alts: Vec<PatId> = self.body.list(list).iter().map(|&at| PatId(at)).collect();
        for (at, &alt) in alts.iter().enumerate() {
            let next = (at + 1 < alts.len()).then(|| self.b.create_block());
            if let Some(next) = next {
                self.test(alt, place, next)?;
            }
            let mark = self.pending.len();
            self.bind_matched(alt, place, from.clone())?;
            self.pending[mark..].iter_mut().for_each(|pending| pending.when.push((which, at as i64)));
            let index = self.ins().iconst(Type::I8, at as i64);
            self.jump(join, &[index]);
            if let Some(next) = next {
                self.enter(next);
            }
        }
        self.enter(join);
        Ok(())
    }

    fn sub_at(&mut self, at: Option<(Root, MovePath)>, step: Option<(u32, u32)>, sub: PatId) -> Result<Option<(Root, MovePath)>> {
        match (at, step) {
            (Some((root, mut path)), Some(step)) => {
                path.push(step);
                Ok(Some((root, path)))
            }
            (Some(_), None) if crate::sema::infer::needs_drop(self.cx, self.pat_ty(sub)?)? => {
                bail!("moving out of an array by a slice pattern is not supported yet")
            }
            _ => Ok(None),
        }
    }

    fn bind_local(&mut self, pat: PatId, by_ref: bool, place: Option<Place>, at: Option<(Root, MovePath)>) -> Result<()> {
        let Some(&local) = self.typed.binds.get(&pat) else { bail!("binding has no local") };
        let referenced = by_ref || self.typed.ref_binds.contains_key(&pat);
        let value = if referenced { Some(self.address_or_dangling(place, self.pat_ty(pat)?)?) } else { place.and_then(|place| self.read(place)) };
        let moves = !referenced && crate::sema::infer::needs_drop(self.cx, self.pat_ty(pat)?)?;
        match self.locals[local as usize] {
            Some(bound) => {
                value.into_iter().for_each(|value| self.write(bound, value));
                self.reinit(Root::Local(local), &[]);
            }
            None => self.declare(local, value)?,
        }
        if let Some((root, path)) = at.filter(|_| moves) {
            self.pending.push(Pending { root, path, when: Vec::new() });
        }
        Ok(())
    }
}
