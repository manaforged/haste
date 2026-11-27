use std::sync::OnceLock;

use anyhow::{Result, bail};

use crate::body::{
    Args, Arm, Body, Bound, ExprId, FieldInit, FieldPat, GenericArg, Node, Param, PatId, Path,
    PathId, Segment, Stmt, Ty as SyntaxTy, TyId,
};

use super::codec::{Cursor, read_opt, read_vec, write_opt, write_u32};
use super::warm_expr::{
    read_expr, read_list, read_opt_u32, read_pat, write_expr, write_list, write_opt_u32, write_pat,
    write_slice,
};

fn write_syntax_ty(out: &mut Vec<u8>, ty: &SyntaxTy) {
    match *ty {
        SyntaxTy::Path(p) => {
            out.push(0);
            write_u32(out, p.0);
        }
        SyntaxTy::Ref {
            lifetime,
            mutable,
            ty,
        } => {
            out.push(1);
            write_opt_u32(out, lifetime);
            out.push(u8::from(mutable));
            write_u32(out, ty.0);
        }
        SyntaxTy::Ptr { mutable, ty } => {
            out.push(2);
            out.push(u8::from(mutable));
            write_u32(out, ty.0);
        }
        SyntaxTy::Slice(t) => {
            out.push(3);
            write_u32(out, t.0);
        }
        SyntaxTy::Array(t, e) => {
            out.push(4);
            write_u32(out, t.0);
            write_u32(out, e.0);
        }
        SyntaxTy::Tuple(l) => {
            out.push(5);
            write_list(out, l);
        }
        SyntaxTy::Fn {
            params,
            ret,
            variadic,
            is_unsafe,
        } => {
            out.push(6);
            write_list(out, params);
            write_opt_u32(out, ret.map(|t| t.0));
            out.push(u8::from(variadic) | (u8::from(is_unsafe) << 1));
        }
        SyntaxTy::ImplTrait(n, l) => {
            out.push(7);
            write_u32(out, n);
            write_list(out, l);
        }
        SyntaxTy::DynTrait(l) => {
            out.push(8);
            write_list(out, l);
        }
        SyntaxTy::Never => out.push(9),
        SyntaxTy::Infer => out.push(10),
        SyntaxTy::Macro { path, open } => {
            out.push(11);
            write_u32(out, path.0);
            write_u32(out, open);
        }
        SyntaxTy::Paren(t) => {
            out.push(12);
            write_u32(out, t.0);
        }
        SyntaxTy::Pat(t, p) => {
            out.push(13);
            write_u32(out, t.0);
            write_u32(out, p.0);
        }
        SyntaxTy::Const(e) => {
            out.push(14);
            write_u32(out, e.0);
        }
    }
}

fn read_syntax_ty(cur: &mut Cursor<'_>) -> Result<SyntaxTy> {
    Ok(match cur.u8()? {
        0 => SyntaxTy::Path(PathId(cur.u32()?)),
        1 => SyntaxTy::Ref {
            lifetime: read_opt_u32(cur)?,
            mutable: cur.u8()? != 0,
            ty: TyId(cur.u32()?),
        },
        2 => SyntaxTy::Ptr {
            mutable: cur.u8()? != 0,
            ty: TyId(cur.u32()?),
        },
        3 => SyntaxTy::Slice(TyId(cur.u32()?)),
        4 => SyntaxTy::Array(TyId(cur.u32()?), ExprId(cur.u32()?)),
        5 => SyntaxTy::Tuple(read_list(cur)?),
        6 => {
            let params = read_list(cur)?;
            let ret = read_opt_u32(cur)?.map(TyId);
            let flags = cur.u8()?;
            SyntaxTy::Fn {
                params,
                ret,
                variadic: flags & 1 != 0,
                is_unsafe: flags & 2 != 0,
            }
        }
        7 => SyntaxTy::ImplTrait(cur.u32()?, read_list(cur)?),
        8 => SyntaxTy::DynTrait(read_list(cur)?),
        9 => SyntaxTy::Never,
        10 => SyntaxTy::Infer,
        11 => SyntaxTy::Macro {
            path: PathId(cur.u32()?),
            open: cur.u32()?,
        },
        12 => SyntaxTy::Paren(TyId(cur.u32()?)),
        13 => SyntaxTy::Pat(TyId(cur.u32()?), PatId(cur.u32()?)),
        14 => SyntaxTy::Const(ExprId(cur.u32()?)),
        _ => bail!("invalid syntax ty tag"),
    })
}

pub(super) fn write_args(out: &mut Vec<u8>, args: Option<Args>) {
    let _ = write_opt(out, args.as_ref(), |o, a| {
        match *a {
            Args::Angle(l) => {
                o.push(0);
                write_list(o, l);
            }
            Args::Paren { inputs, output } => {
                o.push(1);
                write_list(o, inputs);
                write_opt_u32(o, output.map(|t| t.0));
            }
            Args::ReturnType => o.push(2),
        }
        Some(())
    });
}

pub(super) fn read_args(cur: &mut Cursor<'_>) -> Result<Option<Args>> {
    read_opt(cur, |c| {
        Ok(match c.u8()? {
            0 => Args::Angle(read_list(c)?),
            1 => Args::Paren {
                inputs: read_list(c)?,
                output: read_opt_u32(c)?.map(TyId),
            },
            2 => Args::ReturnType,
            _ => bail!("invalid args tag"),
        })
    })
}

fn write_stmt(out: &mut Vec<u8>, s: &Stmt) {
    match *s {
        Stmt::Let {
            pat,
            ty,
            init,
            otherwise,
            extend,
        } => {
            out.push(0);
            write_u32(out, pat.0);
            write_opt_u32(out, ty.map(|t| t.0));
            write_opt_u32(out, init.map(|e| e.0));
            write_opt_u32(out, otherwise.map(|e| e.0));
            out.push(u8::from(extend));
        }
        Stmt::Expr(e, semi) => {
            out.push(1);
            write_u32(out, e.0);
            out.push(u8::from(semi));
        }
        Stmt::Item(a, b) => {
            out.push(2);
            write_u32(out, a);
            write_u32(out, b);
        }
        Stmt::Macro { path, open, semi } => {
            out.push(3);
            write_u32(out, path.0);
            write_u32(out, open);
            out.push(u8::from(semi));
        }
        Stmt::Empty => out.push(4),
    }
}

fn read_stmt(cur: &mut Cursor<'_>) -> Result<Stmt> {
    Ok(match cur.u8()? {
        0 => Stmt::Let {
            pat: PatId(cur.u32()?),
            ty: read_opt_u32(cur)?.map(TyId),
            init: read_opt_u32(cur)?.map(ExprId),
            otherwise: read_opt_u32(cur)?.map(ExprId),
            extend: cur.u8()? != 0,
        },
        1 => Stmt::Expr(ExprId(cur.u32()?), cur.u8()? != 0),
        2 => Stmt::Item(cur.u32()?, cur.u32()?),
        3 => Stmt::Macro {
            path: PathId(cur.u32()?),
            open: cur.u32()?,
            semi: cur.u8()? != 0,
        },
        4 => Stmt::Empty,
        _ => bail!("invalid stmt tag"),
    })
}

pub(super) fn write_body(out: &mut Vec<u8>, body: &Body) {
    let flags: u32 = ((u32::from(!body.exprs.is_empty())) << 0)
        | ((u32::from(!body.expr_starts.is_empty())) << 1)
        | ((u32::from(!body.expr_ends.is_empty())) << 2)
        | ((u32::from(!body.pats.is_empty())) << 3)
        | ((u32::from(!body.pat_spans.is_empty())) << 4)
        | ((u32::from(!body.tys.is_empty())) << 5)
        | ((u32::from(!body.paths.is_empty())) << 6)
        | ((u32::from(!body.stmts.is_empty())) << 7)
        | ((u32::from(!body.segments.is_empty())) << 8)
        | ((u32::from(!body.args.is_empty())) << 9)
        | ((u32::from(!body.bounds.is_empty())) << 10)
        | ((u32::from(!body.arms.is_empty())) << 11)
        | ((u32::from(!body.fields.is_empty())) << 12)
        | ((u32::from(!body.field_pats.is_empty())) << 13)
        | ((u32::from(!body.params.is_empty())) << 14)
        | ((u32::from(!body.ids.is_empty())) << 15)
        | ((u32::from(!body.attrs.is_empty())) << 16)
        | ((u32::from(!body.fn_params.is_empty())) << 17)
        | ((u32::from(body.root.is_some())) << 18);
    write_u32(out, flags);
    if flags & (1 << 0) != 0 {
        write_slice(out, &body.exprs, write_expr);
    }
    if flags & (1 << 1) != 0 {
        write_slice(out, &body.expr_starts, |o, v| write_u32(o, *v));
    }
    if flags & (1 << 2) != 0 {
        write_slice(out, &body.expr_ends, |o, v| write_u32(o, *v));
    }
    if flags & (1 << 3) != 0 {
        write_slice(out, &body.pats, write_pat);
    }
    if flags & (1 << 4) != 0 {
        write_slice(out, &body.pat_spans, |o, &(a, b)| {
            write_u32(o, a);
            write_u32(o, b);
        });
    }
    if flags & (1 << 5) != 0 {
        write_slice(out, &body.tys, write_syntax_ty);
    }
    if flags & (1 << 6) != 0 {
        write_slice(out, &body.paths, |o, p| {
            let _ = write_opt(o, p.qself.as_ref(), |o2, &(t, tr)| {
                write_u32(o2, t.0);
                write_opt_u32(o2, tr.map(|x| x.0));
                Some(())
            });
            o.push(u8::from(p.global));
            write_u32(o, p.start);
            write_list(o, p.segments);
        });
    }
    if flags & (1 << 7) != 0 {
        write_slice(out, &body.stmts, write_stmt);
    }
    if flags & (1 << 8) != 0 {
        write_slice(out, &body.segments, |o, s| {
            write_u32(o, s.name);
            write_args(o, s.args);
        });
    }
    if flags & (1 << 9) != 0 {
        write_slice(out, &body.args, |o, a| match *a {
            GenericArg::Ty(t) => {
                o.push(0);
                write_u32(o, t.0);
            }
            GenericArg::Lifetime(l) => {
                o.push(1);
                write_u32(o, l);
            }
            GenericArg::Const(e) => {
                o.push(2);
                write_u32(o, e.0);
            }
            GenericArg::Binding { name, args, ty } => {
                o.push(3);
                write_u32(o, name);
                write_args(o, args);
                write_u32(o, ty.0);
            }
            GenericArg::Constraint { name, bounds } => {
                o.push(4);
                write_u32(o, name);
                write_list(o, bounds);
            }
        });
    }
    if flags & (1 << 10) != 0 {
        write_slice(out, &body.bounds, |o, b| match *b {
            Bound::Trait {
                path,
                maybe,
                constness,
            } => {
                o.push(0);
                write_u32(o, path.0);
                o.push(u8::from(maybe) | (u8::from(constness) << 1));
            }
            Bound::Lifetime(l) => {
                o.push(1);
                write_u32(o, l);
            }
            Bound::Use(u) => {
                o.push(2);
                write_u32(o, u);
            }
        });
    }
    if flags & (1 << 11) != 0 {
        write_slice(out, &body.arms, |o, a| {
            write_u32(o, a.pat.0);
            write_opt_u32(o, a.guard.map(|g| g.0));
            write_u32(o, a.body.0);
        });
    }
    if flags & (1 << 12) != 0 {
        write_slice(out, &body.fields, |o, f| {
            write_u32(o, f.name);
            write_u32(o, f.value.0);
        });
    }
    if flags & (1 << 13) != 0 {
        write_slice(out, &body.field_pats, |o, f| {
            write_u32(o, f.name);
            write_u32(o, f.pat.0);
        });
    }
    if flags & (1 << 14) != 0 {
        write_slice(out, &body.params, |o, p| {
            write_u32(o, p.pat.0);
            write_opt_u32(o, p.ty.map(|t| t.0));
        });
    }
    if flags & (1 << 15) != 0 {
        write_slice(out, &body.ids, |o, v| write_u32(o, *v));
    }
    if flags & (1 << 16) != 0 {
        write_slice(out, &body.attrs, |o, &(node, a, b)| {
            match node {
                Node::Stmt(x) => {
                    o.push(0);
                    write_u32(o, x);
                }
                Node::Arm(x) => {
                    o.push(1);
                    write_u32(o, x);
                }
                Node::Field(x) => {
                    o.push(2);
                    write_u32(o, x);
                }
                Node::Expr(x) => {
                    o.push(3);
                    write_u32(o, x.0);
                }
            }
            write_u32(o, a);
            write_u32(o, b);
        });
    }
    if let Some(root) = body.root {
        write_u32(out, root.0);
    }
    if flags & (1 << 17) != 0 {
        write_slice(out, &body.fn_params, |o, p| write_u32(o, p.0));
    }
}

fn opt_vec<T>(
    cur: &mut Cursor<'_>,
    flags: u32,
    bit: u32,
    f: impl FnMut(&mut Cursor<'_>) -> Result<T>,
) -> Result<Vec<T>> {
    if flags & (1 << bit) != 0 {
        read_vec(cur, f)
    } else {
        Ok(Vec::new())
    }
}

pub(super) fn read_body(cur: &mut Cursor<'_>) -> Result<Body> {
    let flags = cur.u32()?;
    Ok(Body {
        exprs: opt_vec(cur, flags, 0, read_expr)?,
        expr_starts: opt_vec(cur, flags, 1, |c| c.u32())?,
        expr_ends: opt_vec(cur, flags, 2, |c| c.u32())?,
        pats: opt_vec(cur, flags, 3, read_pat)?,
        pat_spans: opt_vec(cur, flags, 4, |c| Ok((c.u32()?, c.u32()?)))?,
        tys: opt_vec(cur, flags, 5, read_syntax_ty)?,
        paths: opt_vec(cur, flags, 6, |c| {
            Ok(Path {
                qself: read_opt(c, |c2| Ok((TyId(c2.u32()?), read_opt_u32(c2)?.map(PathId))))?,
                global: c.u8()? != 0,
                start: c.u32()?,
                segments: read_list(c)?,
            })
        })?,
        stmts: opt_vec(cur, flags, 7, read_stmt)?,
        segments: opt_vec(cur, flags, 8, |c| {
            Ok(Segment {
                name: c.u32()?,
                args: read_args(c)?,
            })
        })?,
        args: opt_vec(cur, flags, 9, |c| {
            Ok(match c.u8()? {
                0 => GenericArg::Ty(TyId(c.u32()?)),
                1 => GenericArg::Lifetime(c.u32()?),
                2 => GenericArg::Const(ExprId(c.u32()?)),
                3 => GenericArg::Binding {
                    name: c.u32()?,
                    args: read_args(c)?,
                    ty: TyId(c.u32()?),
                },
                4 => GenericArg::Constraint {
                    name: c.u32()?,
                    bounds: read_list(c)?,
                },
                _ => bail!("invalid generic arg tag"),
            })
        })?,
        bounds: opt_vec(cur, flags, 10, |c| {
            Ok(match c.u8()? {
                0 => {
                    let path = PathId(c.u32()?);
                    let f = c.u8()?;
                    Bound::Trait {
                        path,
                        maybe: f & 1 != 0,
                        constness: f & 2 != 0,
                    }
                }
                1 => Bound::Lifetime(c.u32()?),
                2 => Bound::Use(c.u32()?),
                _ => bail!("invalid bound tag"),
            })
        })?,
        arms: opt_vec(cur, flags, 11, |c| {
            Ok(Arm {
                pat: PatId(c.u32()?),
                guard: read_opt_u32(c)?.map(ExprId),
                body: ExprId(c.u32()?),
            })
        })?,
        fields: opt_vec(cur, flags, 12, |c| {
            Ok(FieldInit {
                name: c.u32()?,
                value: ExprId(c.u32()?),
            })
        })?,
        field_pats: opt_vec(cur, flags, 13, |c| {
            Ok(FieldPat {
                name: c.u32()?,
                pat: PatId(c.u32()?),
            })
        })?,
        params: opt_vec(cur, flags, 14, |c| {
            Ok(Param {
                pat: PatId(c.u32()?),
                ty: read_opt_u32(c)?.map(TyId),
            })
        })?,
        ids: opt_vec(cur, flags, 15, |c| c.u32())?,
        attrs: opt_vec(cur, flags, 16, |c| {
            let node = match c.u8()? {
                0 => Node::Stmt(c.u32()?),
                1 => Node::Arm(c.u32()?),
                2 => Node::Field(c.u32()?),
                3 => Node::Expr(ExprId(c.u32()?)),
                _ => bail!("invalid attr node tag"),
            };
            Ok((node, c.u32()?, c.u32()?))
        })?,
        root: (flags & (1 << 18) != 0)
            .then(|| cur.u32().map(ExprId))
            .transpose()?,
        fn_params: opt_vec(cur, flags, 17, |c| c.u32().map(PatId))?,
        syms: OnceLock::new(),
    })
}
