use anyhow::{Result, bail};
use haste_codegen::{Block, IntCC, Type, Value};

use crate::body::{List, Pat, PatId};
use crate::sema::infer::{position, rest_split};
use crate::sema::ty::Kind;

use super::ops::WIDE;
use super::place::{Base, Place};
use super::{Lower, repr};

impl Lower<'_, '_> {
    /// The items of slice pattern `pat` that match an element of the array or slice at `place`, each with that element's place; branches to `fail` first when a slice's length cannot match.
    pub(super) fn slice_parts(&mut self, (pat, list): (PatId, List), place: Option<Place>, fail: Option<Block>) -> Result<Vec<(PatId, Option<Place>)>> {
        let ty = self.pat_ty(pat)?;
        let element = match *self.cx.types.kind(ty) {
            Kind::Array(element, _) | Kind::Slice(element) => element,
            _ => bail!("slice pattern on a type that is not an array or slice"),
        };
        let (pats, rest) = rest_split(self.body, list)?;
        let count = pats.len();
        let Some((data, len)) = self.data_len(place, ty)? else { return Ok(Vec::new()) };
        if let (Some(fail), Kind::Slice(_)) = (fail, self.cx.types.kind(ty)) {
            let given = self.ins().iconst(Type::I64, (count - usize::from(rest.is_some())) as i64);
            let cc = if rest.is_some() { IntCC::UnsignedGreaterThanOrEqual } else { IntCC::Equal };
            let ok = self.ins().icmp(cc, len, given);
            self.pass(ok, fail);
        }
        let mut found = Vec::new();
        for (at, &sub) in pats.iter().enumerate() {
            let Some(_) = position(at, count, rest, count) else {
                if !matches!(self.body.pat(sub), Pat::Rest) {
                    found.push((sub, Some(self.rest_place(sub, (data, len), at, count - 1)?)));
                }
                continue;
            };
            let index = match rest {
                Some(split) if at > split => self.ins().iadd_imm(len, at as i64 - count as i64),
                _ => self.ins().iconst(Type::I64, at as i64),
            };
            found.push((sub, Some(self.element_place(data, index, element)?)));
        }
        Ok(found)
    }

    fn rest_place(&mut self, sub: PatId, (data, len): (Value, Value), split: usize, given: usize) -> Result<Place> {
        let ty = self.pat_ty(sub)?;
        let (Kind::Array(element, _) | Kind::Slice(element)) = *self.cx.types.kind(ty) else { bail!("the rest of a slice pattern is not an array or slice") };
        let stride = self.cx.layout(element)?.size.bytes() as i64;
        let start = self.ins().iadd_imm(data, stride * split as i64);
        if let Kind::Array(..) = self.cx.types.kind(ty) {
            return Ok(Place { base: Base::Addr(start), offset: 0, repr: repr(self.cx, ty)? });
        }
        let left = self.ins().iadd_imm(len, -(given as i64));
        Ok(Place { base: Base::Addr(self.wide(start, left)), offset: 0, repr: WIDE })
    }
}
