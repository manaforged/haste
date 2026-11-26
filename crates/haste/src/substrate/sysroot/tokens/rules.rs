use anyhow::{Context as _, Result, bail};

use crate::outline::rules::{Meta, Op, Piece, Rule};
use crate::sema::summary::codec::{Cursor, write_u32, write_u32s};

use super::{read_frag, read_opt_u32, read_u32s_map, write_frag, write_opt_u32, write_usize};

const OP_KINDS: [Op; 3] = [Op::Many, Op::Some, Op::Maybe];

pub(in super::super) fn write_rules(
    out: &mut Vec<u8>,
    rules: &[Rule],
    to_local: &[Option<u32>],
) -> Option<()> {
    write_usize(out, rules.len())?;
    for rule in rules {
        write_pieces(out, &rule.matcher, to_local)?;
        write_pieces(out, &rule.body, to_local)?;
        write_usize(out, rule.slots)?;
    }
    Some(())
}

pub(in super::super) fn read_rules(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Vec<Rule>> {
    let count = cur.count()?;
    let mut rules = Vec::with_capacity(count);
    for _ in 0..count {
        rules.push(Rule {
            matcher: read_pieces(cur, remap)?,
            body: read_pieces(cur, remap)?,
            slots: cur.u32()? as usize,
        });
    }
    Ok(rules)
}

fn write_pieces(out: &mut Vec<u8>, pieces: &[Piece], to_local: &[Option<u32>]) -> Option<()> {
    write_usize(out, pieces.len())?;
    for piece in pieces {
        match piece {
            Piece::Token(t) => {
                out.push(0);
                write_u32(out, *t);
            }
            Piece::Group(open, inner) => {
                out.push(1);
                write_u32(out, *open);
                write_pieces(out, inner, to_local)?;
            }
            Piece::Var(at, frag, slot) => {
                out.push(2);
                write_u32(out, *at);
                match frag {
                    Some(f) => {
                        out.push(1);
                        write_frag(out, *f);
                    }
                    None => out.push(0),
                }
                write_opt_u32(out, *slot);
            }
            Piece::Crate(krate) => {
                out.push(3);
                let mapped = match krate {
                    Some(k) => Some((*to_local.get(*k as usize)?)?),
                    None => None,
                };
                write_opt_u32(out, mapped);
            }
            Piece::CratePath => out.push(4),
            Piece::Meta(meta) => {
                out.push(5);
                write_meta(out, meta)?;
            }
            Piece::Repeat(inner, sep, op, slots) => {
                out.push(6);
                write_pieces(out, inner, to_local)?;
                match sep {
                    Some((a, b)) => {
                        out.push(1);
                        write_u32(out, *a);
                        write_u32(out, *b);
                    }
                    None => out.push(0),
                }
                out.push(OP_KINDS.iter().position(|&o| o == *op)? as u8);
                write_u32s(out, slots);
            }
        }
    }
    Some(())
}

fn read_pieces(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Vec<Piece>> {
    let count = cur.count()?;
    let mut pieces = Vec::with_capacity(count);
    for _ in 0..count {
        pieces.push(match cur.u8()? {
            0 => Piece::Token(cur.u32()?),
            1 => Piece::Group(cur.u32()?, read_pieces(cur, remap)?),
            2 => {
                let at = cur.u32()?;
                let frag = if cur.u8()? != 0 {
                    Some(read_frag(cur)?)
                } else {
                    None
                };
                let slot = read_opt_u32(cur)?;
                Piece::Var(at, frag, slot)
            }
            3 => {
                let mapped = match read_opt_u32(cur)? {
                    Some(k) => Some(*remap.get(k as usize).context("macro crate out of bounds")?),
                    None => None,
                };
                Piece::Crate(mapped)
            }
            4 => Piece::CratePath,
            5 => Piece::Meta(read_meta(cur)?),
            6 => {
                let inner = read_pieces(cur, remap)?;
                let sep = if cur.u8()? != 0 {
                    Some((cur.u32()?, cur.u32()?))
                } else {
                    None
                };
                let op = OP_KINDS
                    .get(cur.u8()? as usize)
                    .copied()
                    .context("invalid repeat op")?;
                let slots = read_u32s_map(cur, |v| v)?.into_boxed_slice();
                Piece::Repeat(inner, sep, op, slots)
            }
            _ => bail!("invalid piece tag"),
        });
    }
    Ok(pieces)
}

fn write_meta(out: &mut Vec<u8>, meta: &Meta) -> Option<()> {
    match meta {
        Meta::Ignore(slot) => {
            out.push(0);
            write_opt_u32(out, *slot);
        }
        Meta::Index(depth) => {
            out.push(1);
            write_usize(out, *depth)?;
        }
        Meta::Len(depth) => {
            out.push(2);
            write_usize(out, *depth)?;
        }
        Meta::Count(slot) => {
            out.push(3);
            write_opt_u32(out, *slot);
        }
        Meta::Concat(parts) => {
            out.push(4);
            write_usize(out, parts.len())?;
            for &(slot, token) in parts {
                write_opt_u32(out, slot);
                write_u32(out, token);
            }
        }
    }
    Some(())
}

fn read_meta(cur: &mut Cursor<'_>) -> Result<Meta> {
    Ok(match cur.u8()? {
        0 => Meta::Ignore(read_opt_u32(cur)?),
        1 => Meta::Index(cur.u32()? as usize),
        2 => Meta::Len(cur.u32()? as usize),
        3 => Meta::Count(read_opt_u32(cur)?),
        4 => {
            let count = cur.count()?;
            let mut parts = Vec::with_capacity(count);
            for _ in 0..count {
                parts.push((read_opt_u32(cur)?, cur.u32()?));
            }
            Meta::Concat(parts)
        }
        _ => bail!("invalid meta tag"),
    })
}
