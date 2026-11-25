mod backprop;
mod exits;
mod live;
mod loops;
mod ranges;
mod rel;
mod settle;
mod transfer;

use crate::fold::{extend, holds};
use crate::ir::{IntCC, Type};
pub(crate) use ranges::Ranges;
pub(crate) use settle::settle;
use settle::{DomState, SelectRec};
pub(crate) use transfer::deref_ptr;

/// The most facts a path keeps; past it the oldest goes.
pub(super) const MAX_FACTS: usize = 64;

const MAX_NOT: usize = 32;

/// The most symbolic relations a path keeps; past it the oldest goes.
pub(super) const MAX_RELS: usize = 64;

/// What is known about an integer value: its unsigned range at its type's width, and constants it differs from.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Fact {
    pub(super) lo: u64,
    pub(super) hi: u64,
    not: Vec<u64>,
}

fn unsigned(imm: i64, ty: Type) -> u64 {
    extend(imm, ty, false) as u64
}

impl Fact {
    fn full(ty: Type) -> Self {
        Self {
            lo: 0,
            hi: ty.unsigned_max(),
            not: Vec::new(),
        }
    }

    pub(super) fn range(lo: u64, hi: u64) -> Self {
        Self {
            lo,
            hi,
            not: Vec::new(),
        }
    }

    /// The single value the fact allows, when it allows one.
    pub(crate) fn exact(&self) -> Option<u64> {
        (self.lo == self.hi).then_some(self.lo)
    }

    /// Whether `self` already implies every constraint in `other`.
    #[inline]
    pub(super) fn subsumes(&self, other: &Fact) -> bool {
        self.lo >= other.lo
            && self.hi <= other.hi
            && (other.not.is_empty() || other.not.iter().all(|&v| !self.admits(v)))
    }

    /// What both facts say.
    pub(super) fn meet(mut self, other: &Fact) -> Self {
        self.lo = self.lo.max(other.lo);
        self.hi = self.hi.min(other.hi);
        for value in &other.not {
            self.exclude(*value);
        }
        self
    }

    fn exclude(&mut self, value: u64) {
        if self.lo == value && self.lo < self.hi {
            self.lo += 1;
        } else if self.hi == value && self.lo < self.hi {
            self.hi -= 1;
        } else if (self.lo..=self.hi).contains(&value)
            && !self.not.contains(&value)
            && self.not.len() < MAX_NOT
        {
            self.not.push(value);
        }
    }

    /// Whether the value can equal `value`.
    pub(super) fn admits(&self, value: u64) -> bool {
        (self.lo..=self.hi).contains(&value) && !self.not.contains(&value)
    }

    /// Whether the fact holds for both signed and unsigned readings alike: its range sits below the sign bit of `ty`.
    pub(super) fn non_negative(&self, ty: Type) -> bool {
        self.hi <= ty.unsigned_max() >> 1
    }
}

/// `cc` as the unsigned order that agrees with it on `x` and `y`: itself when unsigned, its unsigned twin when both sit below the sign bit of `ty`.
pub(super) fn unsigned_order(cc: IntCC, ty: Type, x: &Fact, y: &Fact) -> Option<IntCC> {
    match cc.is_signed() {
        true if x.non_negative(ty) && y.non_negative(ty) => Some(cc.as_unsigned()),
        true => None,
        false => Some(cc),
    }
}

/// `x cc y` when the facts settle it for every pair of values they allow.
pub(crate) fn decide(cc: IntCC, ty: Type, x: &Fact, y: &Fact) -> Option<bool> {
    if let (Some(a), Some(b)) = (x.exact(), y.exact()) {
        return Some(holds(cc, ty, a as i64, b as i64));
    }
    let cc = unsigned_order(cc, ty, x, y)?;
    let disjoint = x.hi < y.lo
        || y.hi < x.lo
        || y.exact().is_some_and(|b| !x.admits(b))
        || x.exact().is_some_and(|a| !y.admits(a));
    match cc {
        IntCC::Equal => disjoint.then_some(false),
        IntCC::NotEqual => disjoint.then_some(true),
        _ => ordered(cc, x, y),
    }
}

fn ordered(cc: IntCC, x: &Fact, y: &Fact) -> Option<bool> {
    match cc {
        IntCC::UnsignedLessThan => order(x.hi < y.lo, x.lo >= y.hi),
        IntCC::UnsignedGreaterThanOrEqual => order(x.lo >= y.hi, x.hi < y.lo),
        IntCC::UnsignedGreaterThan => order(x.lo > y.hi, x.hi <= y.lo),
        IntCC::UnsignedLessThanOrEqual => order(x.hi <= y.lo, x.lo > y.hi),
        _ => None,
    }
}

fn order(always: bool, never: bool) -> Option<bool> {
    if always {
        Some(true)
    } else if never {
        Some(false)
    } else {
        None
    }
}

/// The fact on `x` that `x cc y` being `taken` gives, for an equality or unsigned `cc`.
pub(super) fn compared_range(cc: IntCC, ty: Type, rhs: &Fact, taken: bool) -> Option<Fact> {
    let cc = if taken { cc } else { cc.inverse() };
    let max = ty.unsigned_max();
    let mut fact = Fact::full(ty);
    match cc {
        IntCC::Equal => fact = rhs.clone(),
        IntCC::NotEqual => fact.exclude(rhs.exact()?),
        IntCC::UnsignedLessThan if rhs.hi > 0 => fact.hi = rhs.hi - 1,
        IntCC::UnsignedLessThanOrEqual => fact.hi = rhs.hi,
        IntCC::UnsignedGreaterThan if rhs.lo < max => fact.lo = rhs.lo + 1,
        IntCC::UnsignedGreaterThanOrEqual => fact.lo = rhs.lo,
        _ => return None,
    }
    Some(fact)
}

/// Whether a value with `fact` is nonzero, when the fact settles it.
pub(crate) fn truth(fact: &Fact) -> Option<bool> {
    match fact.exact() {
        Some(known) => Some(known != 0),
        None => (!fact.admits(0)).then_some(true),
    }
}
