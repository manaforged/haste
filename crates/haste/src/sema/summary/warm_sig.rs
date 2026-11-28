use anyhow::{Result, bail};

use crate::body::adt::{AdtDef, FieldDef, FieldVis, Shape as AdtShape, VariantDef};
use crate::body::sig::{
    Generic, GenericKind, Generics, ImplHeader, Predicate, SelfKind, Signature,
};
use crate::body::{ExprId, PatId, TyId};

use super::codec::{Cursor, read_opt, read_vec, write_opt, write_u32};
use super::warm_body::{read_body, write_body};
use super::warm_expr::{read_list, read_opt_u32, write_list, write_opt_u32, write_slice};

fn write_generics_slice(out: &mut Vec<u8>, generics: &[Generic]) {
    write_slice(out, generics, |o, g| {
        write_u32(o, g.name);
        match g.kind {
            GenericKind::Type => o.push(0),
            GenericKind::Lifetime => o.push(1),
            GenericKind::Const(t) => {
                o.push(2);
                write_u32(o, t.0);
            }
        }
        write_list(o, g.bounds);
        write_opt_u32(o, g.default.map(|t| t.0));
        write_u32(o, g.span.0);
        write_u32(o, g.span.1);
    });
}

fn read_generics_slice(cur: &mut Cursor<'_>) -> Result<Vec<Generic>> {
    read_vec(cur, |c| {
        let name = c.u32()?;
        let kind = match c.u8()? {
            0 => GenericKind::Type,
            1 => GenericKind::Lifetime,
            2 => GenericKind::Const(TyId(c.u32()?)),
            _ => bail!("invalid generic kind"),
        };
        Ok(Generic {
            name,
            kind,
            bounds: read_list(c)?,
            default: read_opt_u32(c)?.map(TyId),
            span: (c.u32()?, c.u32()?),
        })
    })
}

fn write_predicates(out: &mut Vec<u8>, predicates: &[Predicate]) {
    write_slice(out, predicates, |o, p| {
        write_opt_u32(o, p.ty.map(|t| t.0));
        write_opt_u32(o, p.lifetime);
        write_list(o, p.bounds);
    });
}

fn read_predicates(cur: &mut Cursor<'_>) -> Result<Vec<Predicate>> {
    read_vec(cur, |c| {
        Ok(Predicate {
            ty: read_opt_u32(c)?.map(TyId),
            lifetime: read_opt_u32(c)?,
            bounds: read_list(c)?,
        })
    })
}

pub(super) fn write_generics(out: &mut Vec<u8>, g: &Generics) {
    write_generics_slice(out, &g.params);
    write_predicates(out, &g.predicates);
    write_list(out, g.supers);
}

pub(super) fn read_generics(cur: &mut Cursor<'_>) -> Result<Generics> {
    Ok(Generics {
        params: read_generics_slice(cur)?,
        predicates: read_predicates(cur)?,
        supers: read_list(cur)?,
    })
}

pub(super) fn write_signature(out: &mut Vec<u8>, s: &Signature) {
    write_body(out, &s.body);
    write_generics_slice(out, &s.generics);
    let _ = write_opt(out, s.receiver.as_ref(), |o, &(kind, mutable)| {
        match kind {
            SelfKind::Value => o.push(0),
            SelfKind::Ref => o.push(1),
            SelfKind::RefMut => o.push(2),
            SelfKind::Typed(t) => {
                o.push(3);
                write_u32(o, t.0);
            }
        }
        o.push(u8::from(mutable));
        Some(())
    });
    write_opt_u32(out, s.receiver_lifetime);
    write_slice(out, &s.params, |o, &(p, t)| {
        write_u32(o, p.0);
        write_u32(o, t.0);
    });
    write_opt_u32(out, s.ret.map(|t| t.0));
    write_predicates(out, &s.predicates);
    out.push(
        u8::from(s.variadic)
            | (u8::from(s.opaque) << 1)
            | (u8::from(s.is_async) << 2)
            | (u8::from(s.is_unsafe) << 3)
            | (u8::from(s.is_safe) << 4),
    );
    write_slice(out, &s.param_starts, |o, v| write_u32(o, *v));
}

pub(super) fn read_signature(cur: &mut Cursor<'_>) -> Result<Signature> {
    let body = read_body(cur)?;
    let generics = read_generics_slice(cur)?;
    let receiver = read_opt(cur, |c| {
        let kind = match c.u8()? {
            0 => SelfKind::Value,
            1 => SelfKind::Ref,
            2 => SelfKind::RefMut,
            3 => SelfKind::Typed(TyId(c.u32()?)),
            _ => bail!("invalid self kind"),
        };
        Ok((kind, c.u8()? != 0))
    })?;
    let receiver_lifetime = read_opt_u32(cur)?;
    let params = read_vec(cur, |c| Ok((PatId(c.u32()?), TyId(c.u32()?))))?;
    let ret = read_opt_u32(cur)?.map(TyId);
    let predicates = read_predicates(cur)?;
    let flags = cur.u8()?;
    let param_starts = read_vec(cur, |c| c.u32())?;
    Ok(Signature {
        body,
        generics,
        receiver,
        receiver_lifetime,
        params,
        ret,
        predicates,
        variadic: flags & 1 != 0,
        opaque: flags & 2 != 0,
        is_async: flags & 4 != 0,
        is_unsafe: flags & 8 != 0,
        is_safe: flags & 16 != 0,
        param_starts,
    })
}

pub(super) fn write_adt_def(out: &mut Vec<u8>, adt: &AdtDef) {
    write_body(out, &adt.body);
    write_generics(out, &adt.generics);
    write_slice(out, &adt.variants, |o, v| {
        write_u32(o, v.name);
        o.push(v.shape as u8);
        write_slice(o, &v.fields, |o2, f| {
            write_opt_u32(o2, f.name);
            write_u32(o2, f.ty.0);
            o2.push(f.vis as u8);
            write_opt_u32(o2, f.attrs);
        });
        write_opt_u32(o, v.discriminant.map(|e| e.0));
        write_opt_u32(o, v.attrs);
        write_u32(o, v.index);
    });
    write_slice(out, &adt.wheres, |o, &(a, b)| {
        write_u32(o, a);
        write_u32(o, b);
    });
}

pub(super) fn read_adt_def(cur: &mut Cursor<'_>) -> Result<AdtDef> {
    let body = read_body(cur)?;
    let generics = read_generics(cur)?;
    let variants = read_vec(cur, |c| {
        let name = c.u32()?;
        let shape = match c.u8()? {
            0 => AdtShape::Unit,
            1 => AdtShape::Tuple,
            2 => AdtShape::Named,
            _ => bail!("invalid adt shape"),
        };
        let fields = read_vec(c, |c2| {
            let name = read_opt_u32(c2)?;
            let ty = TyId(c2.u32()?);
            let vis = match c2.u8()? {
                0 => FieldVis::Public,
                1 => FieldVis::Crate,
                2 => FieldVis::Super,
                3 => FieldVis::Private,
                _ => bail!("invalid field vis"),
            };
            Ok(FieldDef {
                name,
                ty,
                vis,
                attrs: read_opt_u32(c2)?,
            })
        })?;
        Ok(VariantDef {
            name,
            shape,
            fields,
            discriminant: read_opt_u32(c)?.map(ExprId),
            attrs: read_opt_u32(c)?,
            index: c.u32()?,
        })
    })?;
    let wheres = read_vec(cur, |c| Ok((c.u32()?, c.u32()?)))?;
    Ok(AdtDef {
        body,
        generics,
        variants,
        wheres,
    })
}

pub(super) fn write_impl_header(out: &mut Vec<u8>, h: &ImplHeader) {
    write_body(out, &h.body);
    write_generics_slice(out, &h.generics);
    out.push(u8::from(h.negative));
    write_opt_u32(out, h.trait_ref.map(|t| t.0));
    write_u32(out, h.self_ty.0);
    write_predicates(out, &h.predicates);
}

pub(super) fn read_impl_header(cur: &mut Cursor<'_>) -> Result<ImplHeader> {
    Ok(ImplHeader {
        body: read_body(cur)?,
        generics: read_generics_slice(cur)?,
        negative: cur.u8()? != 0,
        trait_ref: read_opt_u32(cur)?.map(TyId),
        self_ty: TyId(cur.u32()?),
        predicates: read_predicates(cur)?,
    })
}
