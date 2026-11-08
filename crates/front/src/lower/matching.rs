use anyhow::{Result, bail};
use haste_codegen::{Block, FloatCC, IntCC, Type, Value};

use crate::body::{ExprId, List, Pat, PatId};
use crate::sema::infer::{Res, position, rest_split};
use crate::sema::ty::{Kind, Ty};

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
        let join = self.b.create_block();
        let joined = Self::carrier(self.repr_of(expr)?).map(|ty| self.b.append_block_param(join, ty));
        for at in self.body.list(arms).to_vec() {
            let arm = self.body.arms[at as usize];
            for alt in self.alternatives(arm.pat) {
                let next = self.b.create_block();
                self.test(alt, place, next)?;
                self.bind_matched(alt, place)?;
                if let Some(guard) = arm.guard {
                    let Some(ok) = self.expr(guard)? else { bail!("guard with no value") };
                    self.pass(ok, next);
                }
                let value = self.expr(arm.body)?;
                self.branch_value(value, join);
                self.b.switch_to_block(next);
                self.open = true;
            }
        }
        self.ins().trap();
        self.close();
        self.b.switch_to_block(join);
        self.open = true;
        Ok(joined)
    }

    /// Lowers the condition `let pat = init` to whether it matched, binding its names on the way that did.
    pub fn let_expr(&mut self, pat: PatId, init: ExprId) -> Result<Option<Value>> {
        let place = self.place_or_temp(init)?;
        let (fail, done) = (self.b.create_block(), self.b.create_block());
        let matched = self.b.append_block_param(done, Type::I8);
        self.test(pat, place, fail)?;
        self.bind_matched(pat, place)?;
        let yes = self.ins().iconst(Type::I8, 1);
        self.ins().jump(done, &[yes]);
        self.b.switch_to_block(fail);
        self.open = true;
        let no = self.ins().iconst(Type::I8, 0);
        self.ins().jump(done, &[no]);
        self.b.switch_to_block(done);
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
        self.ins().jump(matched, &[]);
        self.b.switch_to_block(fail);
        self.open = true;
        self.expr(otherwise)?;
        if self.open {
            self.ins().trap();
            self.close();
        }
        self.b.switch_to_block(matched);
        self.open = true;
        self.bind_matched(pat, place)
    }

    fn alternatives(&self, pat: PatId) -> Vec<PatId> {
        match *self.body.pat(pat) {
            Pat::Or(list) => self.body.list(list).iter().map(|&at| PatId(at)).collect(),
            _ => vec![pat],
        }
    }

    fn pass(&mut self, ok: Value, fail: Block) {
        let pass = self.b.create_block();
        self.ins().brif(ok, pass, &[], fail, &[]);
        self.b.switch_to_block(pass);
    }

    fn peeled(&mut self, pat: PatId, mut place: Option<Place>) -> Result<Option<Place>> {
        let derefs = self.typed.pat_derefs.get(&pat).copied().unwrap_or(0);
        for level in 0..derefs {
            let Some(pointer) = place.and_then(|place| self.read(place)) else { bail!("reference with no value") };
            let inner = if level + 1 == derefs { self.pat_repr(pat)? } else { Repr::Scalar(Type::I64) };
            place = Some(Place { base: Base::Addr(pointer), offset: 0, repr: inner });
        }
        Ok(place)
    }

    fn pat_ty(&self, pat: PatId) -> Result<Ty> {
        self.typed.pats.get(&pat).copied().ok_or_else(|| anyhow::anyhow!("pattern has no type"))
    }

    fn pat_repr(&mut self, pat: PatId) -> Result<Repr> {
        let ty = self.pat_ty(pat)?;
        repr(self.cx, ty)
    }

    /// Branches to `fail` unless the value at `place` matches `pat`.
    fn test(&mut self, pat: PatId, place: Option<Place>, fail: Block) -> Result<()> {
        let place = self.peeled(pat, place)?;
        self.test_variant(pat, place, fail)?;
        match *self.body.pat(pat) {
            Pat::Bind { sub: Some(sub), .. } => self.test(sub, place, fail),
            Pat::Paren(inner) => self.test(inner, place, fail),
            Pat::Lit(lit) => self.test_lit(pat, lit, place, fail),
            Pat::Range { from, to, inclusive } => self.test_range(pat, (from, to, inclusive), place, fail),
            Pat::Ref { pat: inner, .. } => {
                let pointee = self.pat_repr(inner)?;
                let Some(pointer) = place.and_then(|place| self.read(place)) else { bail!("reference with no value") };
                self.test(inner, Some(Place { base: Base::Addr(pointer), offset: 0, repr: pointee }), fail)
            }
            Pat::Or(list) => self.test_or(list, place, fail),
            Pat::Tuple(_) | Pat::TupleStruct(..) | Pat::Struct { .. } => {
                for (sub, sub_place) in self.parts(pat, place)? {
                    self.test(sub, sub_place, fail)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn test_variant(&mut self, pat: PatId, place: Option<Place>, fail: Block) -> Result<()> {
        let (Some(Res::Ctor(_, variant, _)), Some(place)) = (self.typed.pat_res.get(&pat).cloned(), place) else { return Ok(()) };
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
            self.ins().jump(matched, &[]);
            if next != fail {
                self.b.switch_to_block(next);
            }
        }
        self.b.switch_to_block(matched);
        Ok(())
    }

    fn scalar(&mut self, pat: PatId, place: Option<Place>) -> Result<(Value, Num)> {
        let num = self.num(self.pat_ty(pat)?)?;
        let Some(value) = place.and_then(|place| self.read(place)) else { bail!("literal pattern on a value with no bytes") };
        Ok((value, num))
    }

    fn test_lit(&mut self, pat: PatId, lit: ExprId, place: Option<Place>, fail: Block) -> Result<()> {
        let (value, num) = self.scalar(pat, place)?;
        let Some(expected) = self.expr(lit)? else { bail!("literal with no value") };
        let ok = match num {
            Num::Float(_) => self.ins().fcmp(FloatCC::Equal, value, expected),
            _ => self.ins().icmp(IntCC::Equal, value, expected),
        };
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

    /// The sub-patterns of a tuple, tuple-struct or struct pattern `pat`, each with the place of the field it matches.
    fn parts(&mut self, pat: PatId, place: Option<Place>) -> Result<Vec<(PatId, Option<Place>)>> {
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
        let mut found = Vec::with_capacity(subs.len());
        for (sub, index) in subs {
            let sub_place = match place {
                Some(place) => Some(self.part_place(place, ty, (variant, index))?),
                None => None,
            };
            found.push((sub, sub_place));
        }
        Ok(found)
    }

    fn field_count(&mut self, ty: Ty, variant: u32) -> Result<usize> {
        Ok(match self.cx.types.kind(ty).clone() {
            Kind::Tuple(items) => items.len(),
            Kind::Adt(item, args) => crate::sema::infer::field_types(self.cx, item, &args)?[variant as usize].len(),
            _ => bail!("positional pattern on a type with no fields"),
        })
    }

    fn part_place(&mut self, place: Place, ty: Ty, (variant, index): (u32, u32)) -> Result<Place> {
        match self.cx.types.kind(ty).clone() {
            Kind::Tuple(items) => {
                let layout = self.cx.layout(ty)?;
                let field = repr(self.cx, items[index as usize])?;
                Ok(place.at(layout.fields.offset(index as usize).bytes() as i32, field))
            }
            _ => self.variant_field(place, ty, (variant, index)),
        }
    }

    /// Declares the bindings of `pat`, which matched the value at `place`.
    pub fn bind_matched(&mut self, pat: PatId, place: Option<Place>) -> Result<()> {
        let place = self.peeled(pat, place)?;
        match *self.body.pat(pat) {
            Pat::Bind { sub, .. } if !self.typed.pat_res.contains_key(&pat) => {
                let Some(&local) = self.typed.binds.get(&pat) else { bail!("binding has no local") };
                let value = if self.typed.ref_binds.contains_key(&pat) { Some(self.address_or_dangling(place, self.pat_ty(pat)?)?) } else { place.and_then(|place| self.read(place)) };
                self.declare(local, value)?;
                sub.map_or(Ok(()), |sub| self.bind_matched(sub, place))
            }
            Pat::Paren(inner) => self.bind_matched(inner, place),
            Pat::Ref { pat: inner, .. } => {
                let pointee = self.pat_repr(inner)?;
                let Some(pointer) = place.and_then(|place| self.read(place)) else { bail!("reference with no value") };
                self.bind_matched(inner, Some(Place { base: Base::Addr(pointer), offset: 0, repr: pointee }))
            }
            Pat::Tuple(_) | Pat::TupleStruct(..) | Pat::Struct { .. } => {
                for (sub, sub_place) in self.parts(pat, place)? {
                    self.bind_matched(sub, sub_place)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
