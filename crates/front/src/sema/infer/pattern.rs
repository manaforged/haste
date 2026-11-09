use anyhow::{Result, bail};
use rustc_lexer::TokenKind;

use crate::body::{self as syntax, Body, PathId};
use crate::names::Primitive;
use crate::names::path::Path as NamePath;
use crate::names::resolve::macro_path;

use super::super::builtin::Builtin;
use super::super::ty::{Kind, Ty, int_info};
use super::Infer;
use super::path::parse_int;

/// The bit patterns of an integer type of `bits` bits, and its smallest and largest values as bits.
#[derive(Clone, Copy)]
struct Bits {
    mask: u128,
    min: u128,
    max: u128,
}

impl Bits {
    fn of(bits: u32, signed: bool) -> Bits {
        let mask = crate::sema::ty::mask(bits);
        match signed {
            true => Bits { mask, min: 1 << (bits - 1), max: mask >> 1 },
            false => Bits { mask, min: 0, max: mask },
        }
    }
}

impl Infer<'_, '_> {
    /// The type that the type macro call `path!` with its group opening at `open` names: a pattern type of `pattern_type!`.
    pub(super) fn macro_ty(&mut self, arena: &Body, path: PathId, open: u32) -> Result<Ty> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        let found = macro_path(&self.cx.paths(self.home.krate), &NamePath { segments: &segments, absolute, dollar }, self.home);
        let (src, close) = (self.src, self.src.pair(open));
        match found.and_then(|item| self.cx.builtin(item)) {
            Some(Builtin::PatternType) => {}
            Some(Builtin::CfgSelect) => {
                let env = &self.cx.outline.crates[self.home.krate as usize].env;
                let Some(arm) = crate::outline::mbe::select_arm(src, open, env)? else { bail!("no cfg_select! arm applies") };
                let (inner, ty) = syntax::parse_ty(src, arm.from, arm.to)?;
                return self.lower_ty(&inner, ty);
            }
            _ => {
                let Some(output) = found.map(|item| self.cx.macro_output(src, item, open)).transpose()?.flatten() else { bail!("type macro `{}` is not supported", self.spell(&segments)) };
                let (inner, ty) = syntax::parse_ty(output, 0, output.len())?;
                return self.in_source(output, |infer| infer.lower_ty(&inner, ty));
            }
        }
        let Some(is) = (open + 1..close).find(|&at| src.kind(at) == TokenKind::Ident && src.text(at) == "is") else { bail!("pattern_type! needs `is`") };
        let (inner, base) = syntax::parse_ty(src, open + 1, is)?;
        let base = self.lower_ty(&inner, base)?;
        let (start, end) = self.valid_range(base, (is + 1, close))?;
        Ok(self.cx.types.intern(Kind::Pat(base, start, end)))
    }

    fn valid_range(&mut self, base: Ty, (from, to): (u32, u32)) -> Result<(u128, u128)> {
        let src = self.src;
        if to == from + 2 && src.text(from) == "!" && src.text(from + 1) == "null" {
            return Ok((1, u64::MAX as u128));
        }
        let Some((bits, signed)) = (match *self.cx.types.kind(base) {
            Kind::Prim(primitive) => int_info(primitive),
            _ => None,
        }) else {
            bail!("a pattern type on {}", self.show(base))
        };
        let bits = Bits::of(bits, signed);
        let mut ranges = Vec::new();
        let mut start = from;
        for at in from..=to {
            if at == to || src.kind(at) == TokenKind::Or {
                ranges.push(self.range(bits, (start, at))?);
                start = at + 1;
            }
        }
        match ranges[..] {
            [one] => Ok(one),
            [(s1, e1), (s2, e2)] if e2.wrapping_add(1) & bits.mask == s1 => Ok((s2, e1)),
            [(s1, e1), (s2, e2)] if e1.wrapping_add(1) & bits.mask == s2 => Ok((s1, e2)),
            _ => bail!("pattern type ranges that do not join into one are not supported"),
        }
    }

    fn range(&mut self, bits: Bits, (from, to): (u32, u32)) -> Result<(u128, u128)> {
        let src = self.src;
        let Some(dots) = (from..to).find(|&at| src.kind(at) == TokenKind::Dot && src.kind(at + 1) == TokenKind::Dot && src.joint(at)) else {
            let value = self.bound(bits, (from, to))?;
            return Ok((value, value));
        };
        let inclusive = src.kind(dots + 2) == TokenKind::Eq && src.joint(dots + 1);
        let low = if dots == from { bits.min } else { self.bound(bits, (from, dots))? };
        let after = dots + 2 + u32::from(inclusive);
        let high = match (after == to, inclusive) {
            (true, _) => bits.max,
            (false, true) => self.bound(bits, (after, to))?,
            (false, false) => self.bound(bits, (after, to))?.wrapping_sub(1) & bits.mask,
        };
        Ok((low, high))
    }

    fn bound(&mut self, bits: Bits, (from, to): (u32, u32)) -> Result<u128> {
        let src = self.src;
        let negative = src.text(from) == "-";
        let at = from + u32::from(negative);
        let value = match (to - at, src.kind(at)) {
            (1, TokenKind::Literal { .. }) => parse_int(src.text(at))?.0,
            (4, TokenKind::Ident) if Primitive::named(src.text(at)).is_some() && matches!(src.text(at + 3), "MAX" | "MIN") => {
                return Ok(if src.text(at + 3) == "MAX" { bits.max } else { bits.min });
            }
            (1, TokenKind::Ident) => {
                let name = self.cx.names.symbols.intern(src.text(at));
                self.const_named(&[name], false, None)?.0
            }
            _ => bail!("pattern type bound `{}` is not a literal", src.span(from, to - 1)),
        };
        Ok(if negative { value.wrapping_neg() & bits.mask } else { value & bits.mask })
    }
}
