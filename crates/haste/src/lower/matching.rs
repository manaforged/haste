mod bind;
mod test;

use anyhow::{Result, bail};
use haste_codegen::{Block, Type, Value};

use crate::body::{ExprId, List, Pat, PatId};
use crate::sema::infer::{Res, position, rest_split};
use crate::sema::ty::{Kind, Ty};

use super::moves::{MovePath, Root};
use super::place::{Base, Place};
use super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Lowers `match`: each arm tests its pattern and falls through to the next one when it fails.
    pub fn match_expr(
        &mut self,
        expr: ExprId,
        scrutinee: ExprId,
        arms: List,
    ) -> Result<Option<Value>> {
        let place = self.place_or_temp(scrutinee)?;
        if !self.open {
            return Ok(None);
        }
        let from = self.scrutinee_path(scrutinee);
        let (join, joined, slot) = self.join_block(expr)?;
        let arm_ids = self.body.list(arms).to_vec();
        let hoisted = self.hoist_match_scrutinee(&arm_ids, place)?;
        for at in arm_ids {
            if !self.open {
                break;
            }
            let arm = self.body.arms[at as usize];
            let next = self.b.create_block();
            self.test_with_at(arm.pat, place, from.clone(), hoisted, next)?;
            let mark = self.owned.len();
            self.bind_matched(arm.pat, place, from.clone())?;
            if let Some(guard) = arm.guard {
                let outer_pending = std::mem::take(&mut self.pending);
                let guard_mark = self.owned.len();
                let ok = self.expr(guard)?;
                self.end_scope(guard_mark)?;
                if !self.open {
                    self.owned.truncate(mark);
                    self.enter(next);
                    continue;
                }
                self.pending = outer_pending;
                let Some(ok) = ok else {
                    bail!("guard with no value");
                };
                self.guard_pass(ok, mark, next);
            }
            self.commit_moves()?;
            let value = self.expr(arm.body)?;
            let value = self.snapshot_before_drops(mark, arm.body, value)?;
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
        let from = self.scrutinee_path(init);
        let (fail, done) = (self.b.create_block(), self.b.create_block());
        let matched = self.b.append_block_param(done, Type::I8);
        self.test_at(pat, place, from.clone(), fail)?;
        self.bind_matched(pat, place, from)?;
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
        let from = self.scrutinee_path(init);
        if !self.open {
            return Ok(());
        }
        let (fail, matched) = (self.b.create_block(), self.b.create_block());
        self.test_at(pat, place, from.clone(), fail)?;
        self.jump(matched, &[]);
        self.enter(fail);
        self.expr(otherwise)?;
        if self.open {
            self.ins().trap();
            self.close();
        }
        self.enter(matched);
        self.bind_matched(pat, place, from)?;
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
            let Some(pointer) = place.and_then(|place| self.read(place)) else {
                bail!("reference with no value");
            };
            place = Some(match level + 1 == derefs {
                true => self.pointee_place(pointer, self.pat_ty(pat)?)?,
                false => Place {
                    base: Base::Addr(pointer),
                    offset: 0,
                    repr: Repr::Scalar(Type::I64),
                },
            });
        }
        Ok(place)
    }

    fn ref_target(&mut self, pat: PatId, place: Option<Place>) -> Result<Place> {
        let Kind::Ref(_, pointee) = *self.cx.types.kind(self.pat_ty(pat)?) else {
            bail!("a `&` pattern on a value that is not a reference");
        };
        let repr = repr(self.cx, pointee)?;
        let Some(pointer) = place.and_then(|place| self.read(place)) else {
            bail!("reference with no value");
        };
        Ok(Place {
            base: Base::Addr(pointer),
            offset: 0,
            repr,
        })
    }

    pub(super) fn pat_ty(&self, pat: PatId) -> Result<Ty> {
        self.typed
            .pats
            .get(&pat)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("pattern has no type"))
    }

    pub(super) fn slice_rest_range(
        &self,
        pat: PatId,
    ) -> Result<Option<(PatId, std::ops::Range<u32>)>> {
        let Pat::Slice(list) = *self.body.pat(pat) else {
            return Ok(None);
        };
        let Kind::Array(_, len_ty) = *self.cx.types.kind(self.pat_ty(pat)?) else {
            return Ok(None);
        };
        let count = self.len(len_ty)? as usize;
        let (pats, Some(split)) = rest_split(self.body, list)? else {
            return Ok(None);
        };
        let rest_pat = pats[split];
        if matches!(self.body.pat(rest_pat), Pat::Rest) {
            return Ok(None);
        }
        let after = pats.len() - 1 - split;
        Ok(Some((rest_pat, split as u32..(count - after) as u32)))
    }

    fn subpatterns_at(
        &mut self,
        pat: PatId,
        place: Option<Place>,
        at: Option<(Root, MovePath)>,
        fail: Option<Block>,
    ) -> Result<
        Vec<(
            PatId,
            Option<Place>,
            Option<(Root, MovePath)>,
            Option<SubStep>,
        )>,
    > {
        if let Pat::Slice(list) = *self.body.pat(pat) {
            let steps = self.part_indices(pat)?;
            let rest = self.slice_rest_range(pat)?;
            let step = |sub: PatId| {
                if let Some((rest_sub, range)) = &rest
                    && *rest_sub == sub
                {
                    return Some(SubStep::Range(range.clone()));
                }
                steps
                    .iter()
                    .find(|(part, _)| *part == sub)
                    .map(|&(_, step)| SubStep::Field(step))
            };
            let mut out = Vec::new();
            for (sub, sub_place) in self.slice_parts((pat, list), place, fail)? {
                let s = step(sub);
                let sub_at = match &s {
                    Some(SubStep::Field(_)) => self.sub_at(at.clone(), s.clone(), sub)?,
                    _ => None,
                };
                out.push((sub, sub_place, sub_at, s));
            }
            return Ok(out);
        }
        let ty = self.pat_ty(pat)?;
        let mut found = Vec::new();
        for (sub, (variant, index)) in self.part_indices(pat)? {
            let mut sub_place = match place {
                Some(place) => Some(self.part_place(place, ty, (variant, index))?),
                None => None,
            };
            let step = Some(SubStep::Field((variant, index)));
            let mut sub_at = self.sub_at(at.clone(), step.clone(), sub)?;
            if sub_place.is_none()
                && let Some((Root::Local(local), ref path)) = sub_at
                && let Some((cap_local, cap_place)) = self.capture_by_path(local, path)
            {
                sub_place = cap_place;
                sub_at = Some((Root::Local(cap_local), Vec::new()));
            }
            found.push((sub, sub_place, sub_at, step));
        }
        Ok(found)
    }

    /// The sub-patterns of a tuple, tuple-struct, struct or array pattern `pat`, each with the variant and field or element it matches.
    pub(super) fn part_indices(&mut self, pat: PatId) -> Result<Vec<(PatId, (u32, u32))>> {
        let ty = self.pat_ty(pat)?;
        let variant = match self.typed.pat_res.get(&pat) {
            Some(Res::Ctor(_, variant, _)) => *variant,
            _ => 0,
        };
        let subs: Vec<(PatId, u32)> = match *self.body.pat(pat) {
            Pat::Struct { fields, .. } => self
                .body
                .list(fields)
                .iter()
                .map(|&at| self.body.field_pats[at as usize].pat)
                .map(|sub| {
                    (
                        sub,
                        self.typed.field_pats.get(&sub).copied().unwrap_or(u32::MAX),
                    )
                })
                .collect(),
            Pat::Tuple(list) | Pat::TupleStruct(_, list) | Pat::Slice(list)
                if !matches!(self.cx.types.kind(ty), Kind::Slice(_)) =>
            {
                let count = self.field_count(ty, variant)?;
                let (pats, rest) = rest_split(self.body, list)?;
                pats.iter()
                    .enumerate()
                    .filter_map(|(at, sub)| {
                        position(at, pats.len(), rest, count).map(|index| (*sub, index as u32))
                    })
                    .collect()
            }
            _ => Vec::new(),
        };
        Ok(subs
            .into_iter()
            .map(|(sub, index)| (sub, (variant, index)))
            .collect())
    }

    fn field_count(&mut self, ty: Ty, variant: u32) -> Result<usize> {
        Ok(match *self.cx.types.kind(ty) {
            Kind::Tuple(items) => self.cx.types.substs(items).len(),
            Kind::Array(_, len) => self.len(len)? as usize,
            Kind::Adt(item, args) => {
                let slice = self.cx.types.substs(args).to_vec();
                crate::sema::infer::field_types(self.cx, item, &slice)?[variant as usize].len()
            }
            _ => bail!("positional pattern on a type with no fields"),
        })
    }

    fn sub_at(
        &mut self,
        at: Option<(Root, MovePath)>,
        step: Option<SubStep>,
        sub: PatId,
    ) -> Result<Option<(Root, MovePath)>> {
        match (at, step) {
            (Some((root, mut path)), Some(SubStep::Field(step))) => {
                path.push(step);
                Ok(Some((root, path)))
            }
            (Some(_), None) if crate::sema::infer::needs_drop(self.cx, self.pat_ty(sub)?)? => {
                bail!("moving out of an array by a slice pattern is not supported yet")
            }
            _ => Ok(None),
        }
    }
}

#[derive(Clone)]
enum SubStep {
    Field((u32, u32)),
    Range(std::ops::Range<u32>),
}
