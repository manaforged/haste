use std::sync::Arc;

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::body::ExprId;
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::infer::{BodyTables, Glue, LazyTables, Local, Res, Shape, Typed};
use crate::sema::substs::Substs;
use crate::sema::ty::Types;

use super::super::SummaryKind;
use super::{Cursor, Decoder, Encoder, FileIndex, framed, read_opt, read_syntax, unframed, write_opt, write_syntax, write_u32, write_u32s};

const TYPED_MAGIC: [u8; 4] = *b"HMT1";
const VERDICT_MAGIC: [u8; 4] = *b"HPV1";

/// Encodes the eagerly typed body of a concrete fixed instance at `substs`, or `None` if it is lazy or names a closure or local item.
pub fn encode_typed(outline: &Outline, types: &Types, substs: Substs, typed: &Typed, mut key_of: impl FnMut(&Outline, ItemRef) -> Option<u64>) -> Result<Option<Vec<u8>>> {
    let LazyTables::Eager(tables) = &typed.tables else { return Ok(None) };
    Ok(framed((outline, types, &mut key_of), TYPED_MAGIC, |enc, body| enc.write_typed(body, (substs, typed), tables)))
}

/// Decodes an eagerly typed fixed instance body from `bytes`, or `None` if `expected_substs` or any item mismatches.
pub fn decode_typed(outline: &Outline, index: &FileIndex, types: &Types, expected_substs: Substs, bytes: &[u8]) -> Result<Option<Typed>> {
    unframed((outline, index, types), TYPED_MAGIC, bytes, |dec, cur| dec.read_typed(cur, expected_substs))
}

/// Encodes a settled non-polymorphic summary verdict, or `None` for a polymorphic summary or a hint naming a local item.
pub fn encode_verdict(outline: &Outline, types: &Types, kind: &SummaryKind, mut key_of: impl FnMut(&Outline, ItemRef) -> Option<u64>) -> Result<Option<Vec<u8>>> {
    Ok(framed((outline, types, &mut key_of), VERDICT_MAGIC, |enc, body| enc.write_verdict(body, kind)))
}

/// Decodes a non-polymorphic summary verdict from `bytes`, or `None` if any hinted item is absent.
pub fn decode_verdict(outline: &Outline, index: &FileIndex, types: &Types, bytes: &[u8]) -> Result<Option<SummaryKind>> {
    unframed((outline, index, types), VERDICT_MAGIC, bytes, |dec, cur| dec.read_verdict(cur).map(Some))
}

impl<F: FnMut(&Outline, ItemRef) -> Option<u64>> Encoder<'_, F> {
    fn write_typed(&mut self, out: &mut Vec<u8>, (substs, typed): (Substs, &Typed), tables: &BodyTables) -> Option<()> {
        self.write_substs(out, substs)?;
        self.write_shape(out, &typed.shape)?;
        write_u32(out, typed.locals.len() as u32);
        for local in &typed.locals {
            write_u32(out, self.ty_id(local.ty)?);
        }
        write_u32s(out, &typed.params);
        write_opt(out, typed.ret.as_ref(), |out, t| {
            write_u32(out, self.ty_id(*t)?);
            Some(())
        })?;
        self.write_res_list(out, &typed.drops)?;
        write_syntax(out, &tables.syntax);
        self.write_tables(out, tables)
    }

    fn write_res_list(&mut self, out: &mut Vec<u8>, list: &[Res]) -> Option<()> {
        write_u32(out, list.len() as u32);
        for res in list {
            self.write_res(out, res)?;
        }
        Some(())
    }

    fn write_shape(&mut self, out: &mut Vec<u8>, shape: &Shape) -> Option<()> {
        match shape {
            Shape::Body => out.push(0),
            Shape::Glue(glue) => {
                out.push(1);
                write_opt(out, glue.drop.as_ref(), |o, r| self.write_res(o, r))?;
                self.write_res_list(out, &glue.parts)?;
                write_opt(out, glue.resume.as_ref(), |o, r| self.write_res(o, r))?;
            }
            Shape::Virtual(slot) => {
                out.push(2);
                write_u32(out, *slot);
            }
            Shape::Untuple(res) => {
                out.push(3);
                self.write_res(out, res)?;
            }
            Shape::Ctor(variant) => {
                out.push(4);
                write_u32(out, *variant);
            }
            Shape::Clone(calls) => {
                out.push(5);
                self.write_res_list(out, calls)?;
            }
            Shape::Closure(_) | Shape::Anon(_) | Shape::Asm(_) | Shape::GlobalAlloc(_) => return None,
        }
        Some(())
    }

    fn write_verdict(&mut self, out: &mut Vec<u8>, kind: &SummaryKind) -> Option<()> {
        match kind {
            SummaryKind::Poly(_) => return None,
            SummaryKind::MonomorphicFallback => out.push(0),
            SummaryKind::MonomorphicHints(hints) => {
                out.push(1);
                let mut pairs: Vec<(ExprId, ItemRef)> = hints.iter().map(|(&expr, &item)| (expr, item)).collect();
                pairs.sort_unstable_by_key(|&(expr, _)| expr.0);
                write_u32(out, pairs.len() as u32);
                for (expr, item) in pairs {
                    write_u32(out, expr.0);
                    write_u32(out, self.item_id(item)?);
                }
            }
        }
        Some(())
    }
}

impl Decoder<'_> {
    fn read_typed(&self, cur: &mut Cursor<'_>, expected_substs: Substs) -> Result<Option<Typed>> {
        if self.substs(cur)? != expected_substs {
            return Ok(None);
        }
        let shape = self.read_shape(cur)?;
        let locals = (0..cur.u32()?).map(|_| Ok(Local { ty: self.ty(cur.u32()?)? })).collect::<Result<Vec<_>>>()?;
        let params = cur.u32s()?;
        let ret = read_opt(cur, |c| self.ty(c.u32()?))?;
        let drops = self.res_list(cur)?;
        let syntax = Arc::new(read_syntax(cur)?);
        let tables = Arc::new(self.read_tables(cur, syntax)?);
        Ok(Some(Typed { tables: LazyTables::Eager(tables), locals, params, ret, drops, shape }))
    }

    fn res_list(&self, cur: &mut Cursor<'_>) -> Result<Vec<Res>> {
        (0..cur.u32()?).map(|_| self.res(cur)).collect()
    }

    fn read_shape(&self, cur: &mut Cursor<'_>) -> Result<Shape> {
        Ok(match cur.u8()? {
            0 => Shape::Body,
            1 => Shape::Glue(Glue { drop: read_opt(cur, |c| self.res(c))?, parts: self.res_list(cur)?, resume: read_opt(cur, |c| self.res(c))? }),
            2 => Shape::Virtual(cur.u32()?),
            3 => Shape::Untuple(self.res(cur)?),
            4 => Shape::Ctor(cur.u32()?),
            5 => Shape::Clone(self.res_list(cur)?),
            _ => bail!("invalid shape tag"),
        })
    }

    fn read_verdict(&self, cur: &mut Cursor<'_>) -> Result<SummaryKind> {
        Ok(match cur.u8()? {
            0 => SummaryKind::MonomorphicFallback,
            1 => {
                let hints = (0..cur.u32()?).map(|_| Ok((ExprId(cur.u32()?), self.item(cur.u32()?)?))).collect::<Result<HashMap<_, _>>>()?;
                SummaryKind::MonomorphicHints(Arc::new(hints))
            }
            _ => bail!("invalid verdict tag"),
        })
    }
}
