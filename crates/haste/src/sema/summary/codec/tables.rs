use std::sync::Arc;

use anyhow::{Result, bail};

use crate::body::{ExprId, PatId};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::infer::{
    Adjust, Awaited, BodyTables, By, Capture, Closure, Coercion, For, Res, SyntaxTables, Tried,
};
use crate::sema::substs::{PackedMap, PackedSet};
use crate::sema::ty::Mut;

use super::{
    Cursor, Decoder, Encoder, read_expr_ids, read_map, read_mut, read_opt, read_vec, write_map,
    write_map_opt, write_opt, write_u32, write_u32s, write_u64,
};

impl<F: FnMut(&Outline, ItemRef) -> Option<u64>> Encoder<'_, F> {
    pub(super) fn write_tables(&mut self, out: &mut Vec<u8>, t: &BodyTables) -> Option<()> {
        self.write_tys(out, &t.exprs)?;
        self.write_expr_res_map(out, &t.res)?;
        write_map_opt(out, &t.autoderefs, |out, k, v| {
            write_u32(out, k.0);
            write_u32(out, *v);
            Some(())
        })?;
        write_map_opt(out, &t.derefs, |out, k, v| {
            write_u32(out, k.0.0);
            write_u32(out, k.1);
            self.write_res(out, v)
        })?;
        self.write_expr_res_map(out, &t.operators)?;
        write_map_opt(out, &t.coercions, |out, k, v| {
            write_u32(out, k.0);
            self.write_coercion(out, v)
        })?;
        write_map_opt(out, &t.offsets, |out, k, v| {
            write_u32(out, k.0);
            write_u64(out, *v);
            Some(())
        })?;
        write_map_opt(out, &t.pat_res, |out, k, v| {
            write_u32(out, k.0);
            self.write_res(out, v)
        })?;
        write_u32(out, t.overloaded.as_slice().len() as u32);
        for k in t.overloaded.as_slice() {
            write_u32(out, k.0);
        }
        write_map_opt(out, &t.pats, |out, k, v| {
            write_u32(out, k.0);
            write_u32(out, self.ty_id(*v)?);
            Some(())
        })?;
        write_map_opt(out, &t.fors, |out, k, v| {
            write_u32(out, k.0);
            self.write_for(out, v)
        })?;
        write_map_opt(out, &t.tries, |out, k, v| {
            write_u32(out, k.0);
            self.write_tried(out, v)
        })?;
        write_map_opt(out, &t.awaits, |out, k, v| {
            write_u32(out, k.0);
            self.write_awaited(out, v)
        })?;
        write_map_opt(out, &t.closures, |out, k, v| {
            write_u32(out, k.0);
            self.write_closure(out, v)
        })?;
        self.write_expr_res_map(out, &t.try_blocks)?;
        self.write_expr_res_map(out, &t.box_frees)
    }

    fn write_expr_res_map(
        &mut self,
        out: &mut Vec<u8>,
        map: &PackedMap<ExprId, Res>,
    ) -> Option<()> {
        write_map_opt(out, map, |out, k, v| {
            write_u32(out, k.0);
            self.write_res(out, v)
        })
    }

    fn write_for(&mut self, out: &mut Vec<u8>, v: &For) -> Option<()> {
        self.write_res(out, &v.into_iter)?;
        self.write_res(out, &v.next)?;
        write_u32(out, self.ty_id(v.iter)?);
        write_u32(out, self.ty_id(v.option)?);
        write_u32(out, v.some);
        Some(())
    }

    fn write_tried(&mut self, out: &mut Vec<u8>, v: &Tried) -> Option<()> {
        self.write_res(out, &v.branch)?;
        self.write_res(out, &v.from_residual)?;
        write_u32(out, self.ty_id(v.flow)?);
        write_u32(out, v.proceed);
        write_u32(out, v.stop);
        write_opt(out, v.target.as_ref(), |out, e| {
            write_u32(out, e.0);
            Some(())
        })
    }

    fn write_awaited(&mut self, out: &mut Vec<u8>, v: &Awaited) -> Option<()> {
        self.write_res(out, &v.into_future)?;
        write_u32(out, self.ty_id(v.future)?);
        self.write_res(out, &v.poll)?;
        write_u32(out, self.ty_id(v.polled)?);
        Some(())
    }

    fn write_closure(&mut self, out: &mut Vec<u8>, v: &Closure) -> Option<()> {
        write_u32s(out, &v.params);
        write_u32(out, self.ty_id(v.ret)?);
        write_u32(out, v.body.0);
        write_u32(out, v.captures.len() as u32);
        for cap in &v.captures {
            write_u32(out, cap.local);
            out.push(match cap.by {
                By::Ref => 0,
                By::Mut => 1,
                By::Value => 2,
            });
            write_u32(out, cap.path.len() as u32);
            for &(variant, index) in &cap.path {
                write_u32(out, variant);
                write_u32(out, index);
            }
        }
        out.push(u8::from(v.consumes));
        out.push(u8::from(v.coroutine));
        Some(())
    }

    fn write_coercion(&mut self, out: &mut Vec<u8>, v: &Coercion) -> Option<()> {
        match v {
            Coercion::Len(ty) => {
                out.push(0);
                write_u32(out, self.ty_id(*ty)?);
            }
            Coercion::Vtable(vt) => {
                out.push(1);
                self.write_vtable(out, vt)?;
            }
            Coercion::Reify(r) => {
                out.push(2);
                self.write_res(out, r)?;
            }
            Coercion::Deref(steps, ty) => {
                out.push(3);
                write_u32(out, steps.len() as u32);
                for (st, sr) in steps {
                    write_u32(out, self.ty_id(*st)?);
                    write_opt(out, sr.as_ref(), |out, r| self.write_res(out, r))?;
                }
                write_u32(out, self.ty_id(*ty)?);
            }
            Coercion::Autos => out.push(4),
            Coercion::Upcast(idx) => {
                out.push(5);
                write_u32(out, *idx);
            }
            Coercion::Field { target, inner } => {
                out.push(6);
                write_u32(out, self.ty_id(*target)?);
                self.write_coercion(out, inner)?;
            }
        }
        Some(())
    }
}

pub(super) fn write_syntax(out: &mut Vec<u8>, syn: &SyntaxTables) {
    write_map(out, &syn.binds, |o, k, v| {
        write_u32(o, k.0);
        write_u32(o, *v);
    });
    write_map(out, &syn.adjusts, |o, k, v| {
        write_u32(o, k.0);
        write_u32(o, v.derefs);
        match v.autoref {
            Some(m) => {
                o.push(1);
                o.push(u8::from(m == Mut::Mut));
            }
            None => o.push(0),
        }
    });
    write_map(out, &syn.fields, |o, k, v| {
        write_u32(o, k.0);
        write_u32(o, *v);
    });
    write_map(out, &syn.inits, |o, k, v| {
        write_u32(o, k.0);
        write_u32(o, *v);
    });
    write_map(out, &syn.ranges, |o, k, v| {
        write_u32(o, k.0);
        write_u32s(o, v);
    });
    write_map(out, &syn.field_pats, |o, k, v| {
        write_u32(o, k.0);
        write_u32(o, *v);
    });
    write_map(out, &syn.pat_derefs, |o, k, v| {
        write_u32(o, k.0);
        write_u32(o, *v);
    });
    write_map(out, &syn.ref_binds, |o, k, v| {
        write_u32(o, k.0);
        o.push(u8::from(*v == Mut::Mut));
    });
    write_u32(out, syn.or_binds.as_slice().len() as u32);
    syn.or_binds
        .as_slice()
        .iter()
        .for_each(|k| write_u32(out, k.0));
    write_u32(out, syn.const_args.as_slice().len() as u32);
    syn.const_args
        .as_slice()
        .iter()
        .for_each(|k| write_u32(out, k.0));
    write_map(out, &syn.unpacks, |o, k, v| {
        write_u32(o, *k);
        write_u32(o, v.0);
    });
}

impl Decoder<'_> {
    pub(super) fn read_tables(
        &self,
        cur: &mut Cursor<'_>,
        syntax: Arc<SyntaxTables>,
    ) -> Result<BodyTables> {
        let exprs = self.read_tys(cur)?.into_boxed_slice();
        let res = self.read_expr_res_map(cur)?;
        let autoderefs = read_map(cur, |c| Ok((ExprId(c.u32()?), c.u32()?)))?;
        let derefs = read_map(cur, |c| Ok(((ExprId(c.u32()?), c.u32()?), self.res(c)?)))?;
        let operators = self.read_expr_res_map(cur)?;
        let coercions = read_map(cur, |c| Ok((ExprId(c.u32()?), self.read_coercion(c)?)))?;
        let offsets = read_map(cur, |c| Ok((ExprId(c.u32()?), c.u64()?)))?;
        let pat_res = read_map(cur, |c| Ok((PatId(c.u32()?), self.res(c)?)))?;
        let overloaded = PackedSet::pack(read_expr_ids(cur)?);
        let pats = read_map(cur, |c| Ok((PatId(c.u32()?), self.read_ty_ref(c)?)))?;
        let fors = read_map(cur, |c| Ok((ExprId(c.u32()?), self.read_for(c)?)))?;
        let tries = read_map(cur, |c| Ok((ExprId(c.u32()?), self.read_tried(c)?)))?;
        let awaits = read_map(cur, |c| Ok((ExprId(c.u32()?), self.read_awaited(c)?)))?;
        let closures = read_map(cur, |c| Ok((ExprId(c.u32()?), self.read_closure(c)?)))?;
        let try_blocks = self.read_expr_res_map(cur)?;
        let box_frees = self.read_expr_res_map(cur)?;
        Ok(BodyTables {
            syntax,
            exprs,
            res,
            autoderefs,
            derefs,
            operators,
            coercions,
            offsets,
            pat_res,
            overloaded,
            pats,
            fors,
            tries,
            awaits,
            closures,
            try_blocks,
            box_frees,
        })
    }

    fn read_expr_res_map(&self, cur: &mut Cursor<'_>) -> Result<PackedMap<ExprId, Res>> {
        read_map(cur, |c| Ok((ExprId(c.u32()?), self.res(c)?)))
    }

    fn read_for(&self, c: &mut Cursor<'_>) -> Result<For> {
        Ok(For {
            into_iter: self.res(c)?,
            next: self.res(c)?,
            iter: self.read_ty_ref(c)?,
            option: self.read_ty_ref(c)?,
            some: c.u32()?,
        })
    }

    fn read_tried(&self, c: &mut Cursor<'_>) -> Result<Tried> {
        Ok(Tried {
            branch: self.res(c)?,
            from_residual: self.res(c)?,
            flow: self.read_ty_ref(c)?,
            proceed: c.u32()?,
            stop: c.u32()?,
            target: read_opt(c, |c2| c2.u32().map(ExprId))?,
        })
    }

    fn read_awaited(&self, c: &mut Cursor<'_>) -> Result<Awaited> {
        Ok(Awaited {
            into_future: self.res(c)?,
            future: self.read_ty_ref(c)?,
            poll: self.res(c)?,
            polled: self.read_ty_ref(c)?,
        })
    }

    fn read_closure(&self, c: &mut Cursor<'_>) -> Result<Closure> {
        let params = c.u32s()?;
        let ret = self.read_ty_ref(c)?;
        let body = ExprId(c.u32()?);
        let captures = read_vec(c, read_capture)?;
        Ok(Closure {
            params,
            ret,
            body,
            captures,
            consumes: c.u8()? != 0,
            coroutine: c.u8()? != 0,
        })
    }

    fn read_coercion(&self, c: &mut Cursor<'_>) -> Result<Coercion> {
        Ok(match c.u8()? {
            0 => Coercion::Len(self.read_ty_ref(c)?),
            1 => Coercion::Vtable(self.vtable(c)?),
            2 => Coercion::Reify(self.res(c)?),
            3 => Coercion::Deref(
                read_vec(c, |c2| {
                    Ok((self.read_ty_ref(c2)?, read_opt(c2, |c3| self.res(c3))?))
                })?,
                self.read_ty_ref(c)?,
            ),
            4 => Coercion::Autos,
            5 => Coercion::Upcast(c.u32()?),
            6 => Coercion::Field {
                target: self.read_ty_ref(c)?,
                inner: Box::new(self.read_coercion(c)?),
            },
            _ => bail!("invalid coercion tag"),
        })
    }
}

fn read_capture(c: &mut Cursor<'_>) -> Result<Capture> {
    let local = c.u32()?;
    let by = match c.u8()? {
        0 => By::Ref,
        1 => By::Mut,
        2 => By::Value,
        _ => bail!("invalid capture by"),
    };
    let path = read_vec(c, |c2| Ok((c2.u32()?, c2.u32()?)))?;
    Ok(Capture { local, by, path })
}

pub(super) fn read_syntax(cur: &mut Cursor<'_>) -> Result<SyntaxTables> {
    let binds = read_map(cur, |c| Ok((PatId(c.u32()?), c.u32()?)))?;
    let adjusts = read_map(cur, |c| {
        Ok((
            ExprId(c.u32()?),
            Adjust {
                derefs: c.u32()?,
                autoref: read_opt(c, read_mut)?,
            },
        ))
    })?;
    let fields = read_map(cur, |c| Ok((ExprId(c.u32()?), c.u32()?)))?;
    let inits = read_map(cur, |c| Ok((ExprId(c.u32()?), c.u32()?)))?;
    let ranges = read_map(cur, |c| {
        Ok((ExprId(c.u32()?), c.u32s()?.into_boxed_slice()))
    })?;
    let field_pats = read_map(cur, |c| Ok((PatId(c.u32()?), c.u32()?)))?;
    let pat_derefs = read_map(cur, |c| Ok((PatId(c.u32()?), c.u32()?)))?;
    let ref_binds = read_map(cur, |c| Ok((PatId(c.u32()?), read_mut(c)?)))?;
    let or_binds = PackedSet::pack(read_vec(cur, |c| c.u32().map(PatId))?);
    let const_args = PackedSet::pack(read_expr_ids(cur)?);
    let unpacks = read_map(cur, |c| Ok((c.u32()?, PatId(c.u32()?))))?;
    Ok(SyntaxTables {
        binds,
        adjusts,
        fields,
        inits,
        ranges,
        field_pats,
        pat_derefs,
        ref_binds,
        or_binds,
        const_args,
        unpacks,
    })
}
