use anyhow::{Result, bail};
use haste_codegen::Type;

use crate::body::{List, Pat, PatId};
use crate::sema::infer::needs_drop;

use super::super::Lower;
use super::super::drop::Pending;
use super::super::moves::{MovePath, Root};
use super::super::place::Place;
use super::SubStep;

impl Lower<'_, '_> {
    fn moves_sub_array(&mut self, mut pat: PatId) -> Result<bool> {
        while let Pat::Paren(inner) = *self.body.pat(pat) {
            pat = inner;
        }
        Ok(
            matches!(*self.body.pat(pat), Pat::Bind { by_ref: false, .. })
                && !self.typed.pat_res.contains_key(&pat)
                && !self.typed.ref_binds.contains_key(&pat)
                && needs_drop(self.cx, self.pat_ty(pat)?)?,
        )
    }

    /// Declares the bindings of `pat`, which matched the value at `place`, and notes which parts of `at` its by-value bindings move out.
    pub fn bind_matched(
        &mut self,
        pat: PatId,
        place: Option<Place>,
        at: Option<(Root, MovePath)>,
    ) -> Result<()> {
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
                for (sub, sub_place, sub_at, step) in
                    self.subpatterns_at(pat, place, at.clone(), None)?
                {
                    match step {
                        Some(SubStep::Range(range)) => {
                            self.bind_matched(sub, sub_place, None)?;
                            if let Some((root, base)) = &at
                                && self.moves_sub_array(sub)?
                            {
                                for index in range {
                                    let mut path = base.clone();
                                    path.push((0, index));
                                    self.pending.push(Pending {
                                        root: *root,
                                        path,
                                        when: Vec::new(),
                                    });
                                }
                            }
                        }
                        _ => {
                            self.bind_matched(sub, sub_place, sub_at)?;
                        }
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn bind_or(
        &mut self,
        list: List,
        place: Option<Place>,
        from: Option<(Root, MovePath)>,
    ) -> Result<()> {
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
            self.pending[mark..]
                .iter_mut()
                .for_each(|pending| pending.when.push((which, at as i64)));
            let index = self.ins().iconst(Type::I8, at as i64);
            self.jump(join, &[index]);
            if let Some(next) = next {
                self.enter(next);
            }
        }
        self.enter(join);
        Ok(())
    }

    fn bind_local(
        &mut self,
        pat: PatId,
        by_ref: bool,
        place: Option<Place>,
        at: Option<(Root, MovePath)>,
    ) -> Result<()> {
        let Some(&local) = self.typed.binds.get(&pat) else {
            bail!("binding has no local");
        };
        let referenced = by_ref || self.typed.ref_binds.contains_key(&pat);
        let value = if referenced {
            Some(self.address_or_dangling(place, self.pat_ty(pat)?)?)
        } else {
            place.and_then(|place| self.read(place))
        };
        let moves = !referenced && needs_drop(self.cx, self.pat_ty(pat)?)?;
        match self.locals[local as usize] {
            Some(bound) => {
                value.into_iter().for_each(|value| self.write(bound, value));
                self.reinit(Root::Local(local), &[]);
            }
            None => self.declare(local, value)?,
        }
        if let Some((root, path)) = at.filter(|_| moves) {
            self.pending.push(Pending {
                root,
                path,
                when: Vec::new(),
            });
        }
        Ok(())
    }
}
