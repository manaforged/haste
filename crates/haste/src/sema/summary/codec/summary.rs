use std::sync::Arc;

use anyhow::{Context as _, Result, bail};

use crate::body::{BinOp, ExprId};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::infer::{Local, Res, SiteKey};
use crate::sema::substs::{ProjId, Substs};
use crate::sema::ty::{Ty, Types};

use super::super::{CallOverride, ConstHole, OpCheck, PolyInst, PolyObligation, PolySummary};
use super::{
    Cursor, Decoder, Encoder, FileIndex, framed, read_expr_ids, read_opt, read_syntax, read_vec,
    unframed, write_opt, write_syntax, write_u16, write_u32, write_u32s, write_u64,
};

const CODEC_MAGIC: [u8; 4] = *b"HPS2";
const INST_MAGIC: [u8; 4] = *b"HFI2";

/// Encodes `summary` into a workspace-invariant byte vector, or `None` if it references non-sysroot items.
pub fn encode_summary(
    outline: &Outline,
    types: &Types,
    summary: &PolySummary,
    mut key_of: impl FnMut(&Outline, ItemRef) -> Option<u64>,
) -> Result<Option<Vec<u8>>> {
    Ok(framed(
        (outline, types, &mut key_of),
        CODEC_MAGIC,
        |enc, body| enc.write_summary(body, summary),
    ))
}

/// Decodes a frozen `PolySummary` from `bytes` into `types`, or `None` if any sysroot item is absent.
pub fn decode_summary(
    outline: &Outline,
    index: &FileIndex,
    types: &Types,
    bytes: &[u8],
) -> Result<Option<PolySummary>> {
    unframed((outline, index, types), CODEC_MAGIC, bytes, |dec, cur| {
        dec.read_summary(cur).map(Some)
    })
}

/// Decoded concrete sysroot instance resolution data from a substrate.
pub struct FrozenInst {
    pub projs: Box<[(ProjId, Ty)]>,
    pub resolved_calls: Box<[(SiteKey, ItemRef, Substs)]>,
    pub builtin_ops: Box<[ExprId]>,
    pub call_overrides: Box<[CallOverride]>,
    pub offset_overrides: Box<[(ExprId, u64)]>,
    pub drops: Vec<Res>,
}

/// Encodes a concrete sysroot `PolyInst` and its `drops`, or `None` if it references non-sysroot items.
pub fn encode_instance(
    outline: &Outline,
    types: &Types,
    substs: Substs,
    inst: &PolyInst,
    drops: &[Res],
    mut key_of: impl FnMut(&Outline, ItemRef) -> Option<u64>,
) -> Result<Option<Vec<u8>>> {
    Ok(framed(
        (outline, types, &mut key_of),
        INST_MAGIC,
        |enc, body| enc.write_instance(body, substs, inst, drops),
    ))
}

/// Decodes a frozen concrete sysroot instance from `bytes`, returning `None` if `expected_substs` or any item mismatches.
pub fn decode_instance(
    outline: &Outline,
    index: &FileIndex,
    types: &Types,
    expected_substs: Substs,
    bytes: &[u8],
) -> Result<Option<FrozenInst>> {
    unframed((outline, index, types), INST_MAGIC, bytes, |dec, cur| {
        dec.read_instance(cur, expected_substs)
    })
}

impl<F: FnMut(&Outline, ItemRef) -> Option<u64>> Encoder<'_, F> {
    pub(in crate::sema::summary) fn write_summary(
        &mut self,
        out: &mut Vec<u8>,
        s: &PolySummary,
    ) -> Option<()> {
        write_u16(out, s.param_count);
        write_opt(out, s.ret.as_ref(), |out, t| {
            write_u32(out, self.ty_id(*t)?);
            Some(())
        })?;
        write_u32(out, s.locals.len() as u32);
        for l in &s.locals {
            write_u32(out, self.ty_id(l.ty)?);
        }
        write_u32s(out, &s.params);
        write_syntax(out, &s.template.syntax);
        self.write_tables(out, &s.template)?;
        write_u32(out, s.obligations.len() as u32);
        for ob in &s.obligations {
            self.write_obligation(out, ob)?;
        }
        write_u32(out, s.projections.len() as u32);
        for &p in &s.projections {
            self.write_proj_def(out, p)?;
        }
        self.write_tys(out, &s.layout_tys)?;
        write_u32(out, s.const_holes.len() as u32);
        for ch in &s.const_holes {
            self.write_const_hole(out, ch)?;
        }
        write_u32(out, s.fixed_calls.len() as u32);
        for &(item, substs) in &s.fixed_calls {
            write_u32(out, self.item_id(item)?);
            self.write_substs(out, substs)?;
        }
        self.write_site_calls(out, &s.direct_calls)?;
        self.write_site_calls(out, &s.call_holes)?;
        write_u32(out, s.op_holes.len() as u32);
        for &(expr, op, res) in &s.op_holes {
            write_u32(out, expr.0);
            self.write_op_check(out, op)?;
            self.write_res(out, &res)?;
        }
        write_u32(out, s.fixed_drops.len() as u32);
        for r in &s.fixed_drops {
            self.write_res(out, r)?;
        }
        self.write_tys(out, &s.drop_tys)?;
        self.write_tys(out, &s.parametric_tys)
    }

    fn write_obligation(&mut self, out: &mut Vec<u8>, ob: &PolyObligation) -> Option<()> {
        match ob {
            PolyObligation::Prove { of, self_ty, args } => {
                out.push(0);
                write_u32(out, self.item_id(*of)?);
                write_u32(out, self.ty_id(*self_ty)?);
                self.write_substs(out, *args)?;
            }
            PolyObligation::Call { expr, found, tuple } => {
                out.push(1);
                write_u32(out, expr.0);
                write_u32(out, self.ty_id(*found)?);
                write_u32(out, self.ty_id(*tuple)?);
            }
        }
        Some(())
    }

    fn write_const_hole(&mut self, out: &mut Vec<u8>, ch: &ConstHole) -> Option<()> {
        match ch {
            ConstHole::Value(ty) => {
                out.push(0);
                write_u32(out, self.ty_id(*ty)?);
            }
            ConstHole::OffsetOf { expr, steps } => {
                out.push(1);
                write_u32(out, expr.0);
                write_u32(out, steps.len() as u32);
                for &(ty, f) in steps {
                    write_u32(out, self.ty_id(ty)?);
                    write_u32(out, f);
                }
            }
            ConstHole::Intrinsic { item, substs } => {
                out.push(2);
                write_u32(out, self.item_id(*item)?);
                self.write_substs(out, *substs)?;
            }
        }
        Some(())
    }

    fn write_instance(
        &mut self,
        out: &mut Vec<u8>,
        substs: Substs,
        inst: &PolyInst,
        drops: &[Res],
    ) -> Option<()> {
        self.write_substs(out, substs)?;
        write_u32(out, inst.projs.len() as u32);
        for &(pid, ty) in &inst.projs {
            self.write_proj_def(out, pid)?;
            write_u32(out, self.ty_id(ty)?);
        }
        self.write_site_calls(out, &inst.resolved_calls)?;
        write_u32(out, inst.builtin_ops.len() as u32);
        for e in &inst.builtin_ops {
            write_u32(out, e.0);
        }
        write_u32(out, inst.call_overrides.len() as u32);
        for ov in &inst.call_overrides {
            self.write_call_override(out, ov)?;
        }
        write_u32(out, inst.offset_overrides.len() as u32);
        for &(e, v) in &inst.offset_overrides {
            write_u32(out, e.0);
            write_u64(out, v);
        }
        write_u32(out, drops.len() as u32);
        for r in drops {
            self.write_res(out, r)?;
        }
        Some(())
    }

    fn write_call_override(&mut self, out: &mut Vec<u8>, ov: &CallOverride) -> Option<()> {
        write_u32(out, ov.expr.0);
        write_opt(out, ov.res.as_ref(), |o, r| self.write_res(o, r))?;
        write_opt(out, ov.autoderefs.as_ref(), |o, a| {
            write_u32(o, *a);
            Some(())
        })?;
        out.push(u8::from(ov.overloaded));
        write_opt(out, ov.box_free.as_ref(), |o, r| self.write_res(o, r))
    }

    fn write_op_check(&mut self, out: &mut Vec<u8>, op: OpCheck) -> Option<()> {
        match op {
            OpCheck::Unary(t) => {
                out.push(0);
                write_u32(out, self.ty_id(t)?);
            }
            OpCheck::Binary(b, l, r) => {
                out.push(1);
                out.push(b as u8);
                write_u32(out, self.ty_id(l)?);
                write_u32(out, self.ty_id(r)?);
            }
            OpCheck::AssignOp(b, l, r) => {
                out.push(2);
                out.push(b as u8);
                write_u32(out, self.ty_id(l)?);
                write_u32(out, self.ty_id(r)?);
            }
            OpCheck::Index(b, i) => {
                out.push(3);
                write_opt(out, b.as_ref(), |o, t| {
                    write_u32(o, self.ty_id(*t)?);
                    Some(())
                })?;
                write_u32(out, self.ty_id(i)?);
            }
            OpCheck::Never => out.push(4),
        }
        Some(())
    }
}

impl Decoder<'_> {
    pub(in crate::sema::summary) fn read_summary(
        &self,
        cur: &mut Cursor<'_>,
    ) -> Result<PolySummary> {
        let param_count = cur.u16()?;
        let ret = read_opt(cur, |c| self.read_ty_ref(c))?;
        let locals = read_vec(cur, |c| {
            Ok(Local {
                ty: self.read_ty_ref(c)?,
            })
        })?
        .into_boxed_slice();
        let params = cur.u32s()?.into_boxed_slice();
        let syntax = Arc::new(read_syntax(cur)?);
        let template = Arc::new(self.read_tables(cur, syntax)?);
        let obligations = read_vec(cur, |c| self.read_obligation(c))?.into_boxed_slice();
        let projections = read_vec(cur, |c| self.read_proj_def(c))?.into_boxed_slice();
        let layout_tys = self.read_tys(cur)?.into_boxed_slice();
        let const_holes = read_vec(cur, |c| self.read_const_hole(c))?.into_boxed_slice();
        let fixed_calls =
            read_vec(cur, |c| Ok((self.read_item_ref(c)?, self.substs(c)?)))?.into_boxed_slice();
        let direct_calls = self.read_site_calls(cur)?;
        let call_holes = self.read_site_calls(cur)?;
        let op_holes = read_vec(cur, |c| {
            Ok((ExprId(c.u32()?), self.read_op_check(c)?, self.res(c)?))
        })?
        .into_boxed_slice();
        let fixed_drops = read_vec(cur, |c| self.res(c))?.into_boxed_slice();
        let drop_tys = self.read_tys(cur)?.into_boxed_slice();
        let parametric_tys = self.read_tys(cur)?.into_boxed_slice();
        Ok(PolySummary {
            param_count,
            template,
            locals,
            params,
            ret,
            obligations,
            projections,
            layout_tys,
            const_holes,
            fixed_calls,
            direct_calls,
            call_holes,
            op_holes,
            fixed_drops,
            drop_tys,
            parametric_tys,
        })
    }

    fn read_obligation(&self, cur: &mut Cursor<'_>) -> Result<PolyObligation> {
        Ok(match cur.u8()? {
            0 => PolyObligation::Prove {
                of: self.read_item_ref(cur)?,
                self_ty: self.read_ty_ref(cur)?,
                args: self.substs(cur)?,
            },
            1 => PolyObligation::Call {
                expr: ExprId(cur.u32()?),
                found: self.read_ty_ref(cur)?,
                tuple: self.read_ty_ref(cur)?,
            },
            _ => bail!("invalid obligation tag"),
        })
    }

    fn read_const_hole(&self, cur: &mut Cursor<'_>) -> Result<ConstHole> {
        Ok(match cur.u8()? {
            0 => ConstHole::Value(self.read_ty_ref(cur)?),
            1 => ConstHole::OffsetOf {
                expr: ExprId(cur.u32()?),
                steps: read_vec(cur, |c| Ok((self.read_ty_ref(c)?, c.u32()?)))?.into_boxed_slice(),
            },
            2 => ConstHole::Intrinsic {
                item: self.read_item_ref(cur)?,
                substs: self.substs(cur)?,
            },
            _ => bail!("invalid const hole tag"),
        })
    }

    fn read_instance(
        &self,
        cur: &mut Cursor<'_>,
        expected_substs: Substs,
    ) -> Result<Option<FrozenInst>> {
        if self.substs(cur)? != expected_substs {
            return Ok(None);
        }
        let mut projs = read_vec(cur, |c| Ok((self.read_proj_def(c)?, self.read_ty_ref(c)?)))?;
        projs.sort_unstable_by_key(|&(pid, _)| pid.0);
        let projs = projs.into_boxed_slice();
        let resolved_calls = self.read_site_calls(cur)?;
        let builtin_ops = read_expr_ids(cur)?.into_boxed_slice();
        let call_overrides = read_vec(cur, |c| self.read_call_override(c))?.into_boxed_slice();
        let offset_overrides =
            read_vec(cur, |c| Ok((ExprId(c.u32()?), c.u64()?)))?.into_boxed_slice();
        let drops = read_vec(cur, |c| self.res(c))?;
        Ok(Some(FrozenInst {
            projs,
            resolved_calls,
            builtin_ops,
            call_overrides,
            offset_overrides,
            drops,
        }))
    }

    fn read_call_override(&self, cur: &mut Cursor<'_>) -> Result<CallOverride> {
        Ok(CallOverride {
            expr: ExprId(cur.u32()?),
            res: read_opt(cur, |c| self.res(c))?,
            autoderefs: read_opt(cur, |c| c.u32())?,
            overloaded: cur.u8()? != 0,
            box_free: read_opt(cur, |c| self.res(c))?,
        })
    }

    fn read_op_check(&self, cur: &mut Cursor<'_>) -> Result<OpCheck> {
        Ok(match cur.u8()? {
            0 => OpCheck::Unary(self.read_ty_ref(cur)?),
            1 => OpCheck::Binary(
                decode_bin_op(cur.u8()?)?,
                self.read_ty_ref(cur)?,
                self.read_ty_ref(cur)?,
            ),
            2 => OpCheck::AssignOp(
                decode_bin_op(cur.u8()?)?,
                self.read_ty_ref(cur)?,
                self.read_ty_ref(cur)?,
            ),
            3 => OpCheck::Index(
                read_opt(cur, |c| self.read_ty_ref(c))?,
                self.read_ty_ref(cur)?,
            ),
            4 => OpCheck::Never,
            _ => bail!("invalid op check tag"),
        })
    }
}

fn decode_bin_op(raw: u8) -> Result<BinOp> {
    const OPS: [BinOp; 18] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Rem,
        BinOp::And,
        BinOp::Or,
        BinOp::BitXor,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::Eq,
        BinOp::Lt,
        BinOp::Le,
        BinOp::Ne,
        BinOp::Ge,
        BinOp::Gt,
    ];
    OPS.get(raw as usize).copied().context("invalid bin op tag")
}
