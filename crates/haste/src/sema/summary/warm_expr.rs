use anyhow::{Context as _, Result, bail};

use crate::body::{BinOp, BlockKind, Expr, ExprId, List, Lit, Pat, PatId, PathId, TyId, UnOp};

use super::codec::{Cursor, read_opt, write_opt, write_u32};

const BOPS: [BinOp; 18] = [
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

const BLOCK_KINDS: [BlockKind; 8] = [
    BlockKind::Plain,
    BlockKind::Expansion,
    BlockKind::Unsafe,
    BlockKind::Const,
    BlockKind::Async,
    BlockKind::AsyncMove,
    BlockKind::Gen,
    BlockKind::Try,
];

pub(super) fn write_slice<T>(out: &mut Vec<u8>, slice: &[T], mut f: impl FnMut(&mut Vec<u8>, &T)) {
    write_u32(out, slice.len() as u32);
    for item in slice {
        f(out, item);
    }
}

pub(super) fn write_opt_u32(out: &mut Vec<u8>, val: Option<u32>) {
    let _ = write_opt(out, val.as_ref(), |o, v| {
        write_u32(o, *v);
        Some(())
    });
}

pub(super) fn read_opt_u32(cur: &mut Cursor<'_>) -> Result<Option<u32>> {
    read_opt(cur, |c| c.u32())
}

pub(super) fn write_list(out: &mut Vec<u8>, list: List) {
    write_u32(out, list.start);
    write_u32(out, list.len);
}

pub(super) fn read_list(cur: &mut Cursor<'_>) -> Result<List> {
    Ok(List {
        start: cur.u32()?,
        len: cur.u32()?,
    })
}

fn write_opt_list(out: &mut Vec<u8>, list: Option<List>) {
    let _ = write_opt(out, list.as_ref(), |o, l| {
        write_list(o, *l);
        Some(())
    });
}

fn read_opt_list(cur: &mut Cursor<'_>) -> Result<Option<List>> {
    read_opt(cur, read_list)
}

pub(super) fn write_expr(out: &mut Vec<u8>, expr: &Expr) {
    match *expr {
        Expr::Lit(l, at) => {
            out.push(0);
            out.push(match l {
                Lit::Int => 0,
                Lit::Float => 1,
                Lit::Char => 2,
                Lit::Byte => 3,
                Lit::Str => 4,
                Lit::ByteStr => 5,
                Lit::CStr => 6,
                Lit::Bool(true) => 7,
                Lit::Bool(false) => 8,
            });
            write_u32(out, at);
        }
        Expr::Path(p) => {
            out.push(1);
            write_u32(out, p.0);
        }
        Expr::Unary(u, e) => {
            out.push(2);
            out.push(u as u8);
            write_u32(out, e.0);
        }
        Expr::Binary(b, l, r) => {
            out.push(3);
            out.push(b as u8);
            write_u32(out, l.0);
            write_u32(out, r.0);
        }
        Expr::Assign(l, r) => {
            out.push(4);
            write_u32(out, l.0);
            write_u32(out, r.0);
        }
        Expr::AssignOp(b, l, r) => {
            out.push(5);
            out.push(b as u8);
            write_u32(out, l.0);
            write_u32(out, r.0);
        }
        Expr::Call(c, a) => {
            out.push(6);
            write_u32(out, c.0);
            write_list(out, a);
        }
        Expr::Method {
            receiver,
            name,
            args,
            params,
        } => {
            out.push(7);
            write_u32(out, receiver.0);
            write_u32(out, name);
            write_opt_list(out, args);
            write_list(out, params);
        }
        Expr::Field(e, n) => {
            out.push(8);
            write_u32(out, e.0);
            write_u32(out, n);
        }
        Expr::TupleField(e, n) => {
            out.push(9);
            write_u32(out, e.0);
            write_u32(out, n);
        }
        Expr::Index(b, i) => {
            out.push(10);
            write_u32(out, b.0);
            write_u32(out, i.0);
        }
        Expr::Cast(e, t) => {
            out.push(11);
            write_u32(out, e.0);
            write_u32(out, t.0);
        }
        Expr::Ref { mutable, raw, expr } => {
            out.push(12);
            out.push(u8::from(mutable) | (u8::from(raw) << 1));
            write_u32(out, expr.0);
        }
        Expr::Block {
            stmts,
            tail,
            label,
            kind,
        } => {
            out.push(13);
            write_list(out, stmts);
            write_opt_u32(out, tail.map(|e| e.0));
            write_opt_u32(out, label);
            out.push(kind as u8);
        }
        Expr::If(c, t, f) => {
            out.push(14);
            write_u32(out, c.0);
            write_u32(out, t.0);
            write_opt_u32(out, f.map(|e| e.0));
        }
        Expr::Let(p, e) => {
            out.push(15);
            write_u32(out, p.0);
            write_u32(out, e.0);
        }
        Expr::While { label, cond, body } => {
            out.push(16);
            write_opt_u32(out, label);
            write_u32(out, cond.0);
            write_u32(out, body.0);
        }
        Expr::Loop { label, body } => {
            out.push(17);
            write_opt_u32(out, label);
            write_u32(out, body.0);
        }
        Expr::For {
            label,
            pat,
            iter,
            body,
        } => {
            out.push(18);
            write_opt_u32(out, label);
            write_u32(out, pat.0);
            write_u32(out, iter.0);
            write_u32(out, body.0);
        }
        Expr::Match(s, a) => {
            out.push(19);
            write_u32(out, s.0);
            write_list(out, a);
        }
        Expr::Closure {
            params,
            ret,
            body,
            capture,
        } => {
            out.push(20);
            write_list(out, params);
            write_opt_u32(out, ret.map(|t| t.0));
            write_u32(out, body.0);
            out.push(u8::from(capture));
        }
        Expr::Tuple(l) => {
            out.push(21);
            write_list(out, l);
        }
        Expr::Array(l) => {
            out.push(22);
            write_list(out, l);
        }
        Expr::Repeat(v, n) => {
            out.push(23);
            write_u32(out, v.0);
            write_u32(out, n.0);
        }
        Expr::Struct {
            path,
            fields,
            base,
            rest,
        } => {
            out.push(24);
            write_u32(out, path.0);
            write_list(out, fields);
            write_opt_u32(out, base.map(|e| e.0));
            out.push(u8::from(rest));
        }
        Expr::Range {
            from,
            to,
            inclusive,
        } => {
            out.push(25);
            write_opt_u32(out, from.map(|e| e.0));
            write_opt_u32(out, to.map(|e| e.0));
            out.push(u8::from(inclusive));
        }
        Expr::Try(e) => {
            out.push(26);
            write_u32(out, e.0);
        }
        Expr::Await(e) => {
            out.push(27);
            write_u32(out, e.0);
        }
        Expr::Return(e) => {
            out.push(28);
            write_opt_u32(out, e.map(|x| x.0));
        }
        Expr::Break(l, e) => {
            out.push(29);
            write_opt_u32(out, l);
            write_opt_u32(out, e.map(|x| x.0));
        }
        Expr::Continue(l) => {
            out.push(30);
            write_opt_u32(out, l);
        }
        Expr::Yield(e) => {
            out.push(31);
            write_opt_u32(out, e.map(|x| x.0));
        }
        Expr::Become(e) => {
            out.push(32);
            write_u32(out, e.0);
        }
        Expr::Macro { path, open } => {
            out.push(33);
            write_u32(out, path.0);
            write_u32(out, open);
        }
        Expr::Paren(e) => {
            out.push(34);
            write_u32(out, e.0);
        }
        Expr::Underscore => out.push(35),
        Expr::FormatArgs(o) => {
            out.push(36);
            write_u32(out, o);
        }
        Expr::OffsetOf { ty, fields } => {
            out.push(37);
            write_u32(out, ty.0);
            write_list(out, fields);
        }
    }
}

fn read_binop(cur: &mut Cursor<'_>) -> Result<BinOp> {
    BOPS.get(cur.u8()? as usize)
        .copied()
        .context("invalid binop")
}

pub(super) fn read_expr(cur: &mut Cursor<'_>) -> Result<Expr> {
    Ok(match cur.u8()? {
        0 => Expr::Lit(
            match cur.u8()? {
                0 => Lit::Int,
                1 => Lit::Float,
                2 => Lit::Char,
                3 => Lit::Byte,
                4 => Lit::Str,
                5 => Lit::ByteStr,
                6 => Lit::CStr,
                7 => Lit::Bool(true),
                8 => Lit::Bool(false),
                _ => bail!("invalid literal tag"),
            },
            cur.u32()?,
        ),
        1 => Expr::Path(PathId(cur.u32()?)),
        2 => Expr::Unary(
            match cur.u8()? {
                0 => UnOp::Neg,
                1 => UnOp::Not,
                2 => UnOp::Deref,
                _ => bail!("invalid unop"),
            },
            ExprId(cur.u32()?),
        ),
        3 => Expr::Binary(read_binop(cur)?, ExprId(cur.u32()?), ExprId(cur.u32()?)),
        4 => Expr::Assign(ExprId(cur.u32()?), ExprId(cur.u32()?)),
        5 => Expr::AssignOp(read_binop(cur)?, ExprId(cur.u32()?), ExprId(cur.u32()?)),
        6 => Expr::Call(ExprId(cur.u32()?), read_list(cur)?),
        7 => Expr::Method {
            receiver: ExprId(cur.u32()?),
            name: cur.u32()?,
            args: read_opt_list(cur)?,
            params: read_list(cur)?,
        },
        8 => Expr::Field(ExprId(cur.u32()?), cur.u32()?),
        9 => Expr::TupleField(ExprId(cur.u32()?), cur.u32()?),
        10 => Expr::Index(ExprId(cur.u32()?), ExprId(cur.u32()?)),
        11 => Expr::Cast(ExprId(cur.u32()?), TyId(cur.u32()?)),
        12 => {
            let flags = cur.u8()?;
            Expr::Ref {
                mutable: flags & 1 != 0,
                raw: flags & 2 != 0,
                expr: ExprId(cur.u32()?),
            }
        }
        13 => Expr::Block {
            stmts: read_list(cur)?,
            tail: read_opt_u32(cur)?.map(ExprId),
            label: read_opt_u32(cur)?,
            kind: BLOCK_KINDS
                .get(cur.u8()? as usize)
                .copied()
                .context("invalid block kind")?,
        },
        14 => Expr::If(
            ExprId(cur.u32()?),
            ExprId(cur.u32()?),
            read_opt_u32(cur)?.map(ExprId),
        ),
        15 => Expr::Let(PatId(cur.u32()?), ExprId(cur.u32()?)),
        16 => Expr::While {
            label: read_opt_u32(cur)?,
            cond: ExprId(cur.u32()?),
            body: ExprId(cur.u32()?),
        },
        17 => Expr::Loop {
            label: read_opt_u32(cur)?,
            body: ExprId(cur.u32()?),
        },
        18 => Expr::For {
            label: read_opt_u32(cur)?,
            pat: PatId(cur.u32()?),
            iter: ExprId(cur.u32()?),
            body: ExprId(cur.u32()?),
        },
        19 => Expr::Match(ExprId(cur.u32()?), read_list(cur)?),
        20 => Expr::Closure {
            params: read_list(cur)?,
            ret: read_opt_u32(cur)?.map(TyId),
            body: ExprId(cur.u32()?),
            capture: cur.u8()? != 0,
        },
        21 => Expr::Tuple(read_list(cur)?),
        22 => Expr::Array(read_list(cur)?),
        23 => Expr::Repeat(ExprId(cur.u32()?), ExprId(cur.u32()?)),
        24 => Expr::Struct {
            path: PathId(cur.u32()?),
            fields: read_list(cur)?,
            base: read_opt_u32(cur)?.map(ExprId),
            rest: cur.u8()? != 0,
        },
        25 => Expr::Range {
            from: read_opt_u32(cur)?.map(ExprId),
            to: read_opt_u32(cur)?.map(ExprId),
            inclusive: cur.u8()? != 0,
        },
        26 => Expr::Try(ExprId(cur.u32()?)),
        27 => Expr::Await(ExprId(cur.u32()?)),
        28 => Expr::Return(read_opt_u32(cur)?.map(ExprId)),
        29 => Expr::Break(read_opt_u32(cur)?, read_opt_u32(cur)?.map(ExprId)),
        30 => Expr::Continue(read_opt_u32(cur)?),
        31 => Expr::Yield(read_opt_u32(cur)?.map(ExprId)),
        32 => Expr::Become(ExprId(cur.u32()?)),
        33 => Expr::Macro {
            path: PathId(cur.u32()?),
            open: cur.u32()?,
        },
        34 => Expr::Paren(ExprId(cur.u32()?)),
        35 => Expr::Underscore,
        36 => Expr::FormatArgs(cur.u32()?),
        37 => Expr::OffsetOf {
            ty: TyId(cur.u32()?),
            fields: read_list(cur)?,
        },
        _ => bail!("invalid expr tag"),
    })
}

pub(super) fn write_pat(out: &mut Vec<u8>, pat: &Pat) {
    match *pat {
        Pat::Wild => out.push(0),
        Pat::Rest => out.push(1),
        Pat::Bind {
            name,
            by_ref,
            mutable,
            sub,
        } => {
            out.push(2);
            write_u32(out, name);
            out.push(u8::from(by_ref) | (u8::from(mutable) << 1));
            write_opt_u32(out, sub.map(|p| p.0));
        }
        Pat::Lit(e) => {
            out.push(3);
            write_u32(out, e.0);
        }
        Pat::Range {
            from,
            to,
            inclusive,
        } => {
            out.push(4);
            write_opt_u32(out, from.map(|e| e.0));
            write_opt_u32(out, to.map(|e| e.0));
            out.push(u8::from(inclusive));
        }
        Pat::Path(p) => {
            out.push(5);
            write_u32(out, p.0);
        }
        Pat::TupleStruct(p, l) => {
            out.push(6);
            write_u32(out, p.0);
            write_list(out, l);
        }
        Pat::Struct { path, fields, rest } => {
            out.push(7);
            write_u32(out, path.0);
            write_list(out, fields);
            out.push(u8::from(rest));
        }
        Pat::Tuple(l) => {
            out.push(8);
            write_list(out, l);
        }
        Pat::Slice(l) => {
            out.push(9);
            write_list(out, l);
        }
        Pat::Ref { mutable, pat } => {
            out.push(10);
            out.push(u8::from(mutable));
            write_u32(out, pat.0);
        }
        Pat::Box(p) => {
            out.push(11);
            write_u32(out, p.0);
        }
        Pat::Or(l) => {
            out.push(12);
            write_list(out, l);
        }
        Pat::Macro { path, open } => {
            out.push(13);
            write_u32(out, path.0);
            write_u32(out, open);
        }
        Pat::Paren(p) => {
            out.push(14);
            write_u32(out, p.0);
        }
        Pat::Never => out.push(15),
        Pat::Deref(p) => {
            out.push(16);
            write_u32(out, p.0);
        }
    }
}

pub(super) fn read_pat(cur: &mut Cursor<'_>) -> Result<Pat> {
    Ok(match cur.u8()? {
        0 => Pat::Wild,
        1 => Pat::Rest,
        2 => {
            let name = cur.u32()?;
            let flags = cur.u8()?;
            Pat::Bind {
                name,
                by_ref: flags & 1 != 0,
                mutable: flags & 2 != 0,
                sub: read_opt_u32(cur)?.map(PatId),
            }
        }
        3 => Pat::Lit(ExprId(cur.u32()?)),
        4 => Pat::Range {
            from: read_opt_u32(cur)?.map(ExprId),
            to: read_opt_u32(cur)?.map(ExprId),
            inclusive: cur.u8()? != 0,
        },
        5 => Pat::Path(PathId(cur.u32()?)),
        6 => Pat::TupleStruct(PathId(cur.u32()?), read_list(cur)?),
        7 => Pat::Struct {
            path: PathId(cur.u32()?),
            fields: read_list(cur)?,
            rest: cur.u8()? != 0,
        },
        8 => Pat::Tuple(read_list(cur)?),
        9 => Pat::Slice(read_list(cur)?),
        10 => Pat::Ref {
            mutable: cur.u8()? != 0,
            pat: PatId(cur.u32()?),
        },
        11 => Pat::Box(PatId(cur.u32()?)),
        12 => Pat::Or(read_list(cur)?),
        13 => Pat::Macro {
            path: PathId(cur.u32()?),
            open: cur.u32()?,
        },
        14 => Pat::Paren(PatId(cur.u32()?)),
        15 => Pat::Never,
        16 => Pat::Deref(PatId(cur.u32()?)),
        _ => bail!("invalid pat tag"),
    })
}
