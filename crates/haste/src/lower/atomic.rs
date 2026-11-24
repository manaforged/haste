use anyhow::{Result, bail};
use haste_codegen::{AtomicOrd, IntCC, Value};

use crate::sema::intrinsic::Atomic;
use crate::sema::ty::{Kind, Ty};

use super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Extracts the strongest `AtomicOrd` const-generic argument present in `substs`, defaulting to `SeqCst`.
    pub(super) fn atomic_ord(&self, substs: &[Ty]) -> AtomicOrd {
        let mut found: Option<AtomicOrd> = None;
        for &ty in substs {
            if let Kind::Value(bits) = *self.cx.types.kind(ty)
                && let Some(ord) = u8::try_from(bits).ok().and_then(AtomicOrd::from_u8)
            {
                found = Some(match found {
                    Some(prev) => prev.join(ord),
                    None => ord,
                });
            }
        }
        found.unwrap_or(AtomicOrd::SeqCst)
    }

    /// Emits the atomic access `which` of the `ty` that the first argument points at, honoring the ordering in `substs`.
    pub(super) fn atomic(
        &mut self,
        which: Atomic,
        ty: Ty,
        substs: &[Ty],
        args: &[Value],
    ) -> Result<Option<Value>> {
        let Repr::Scalar(machine) = repr(self.cx, ty)? else {
            bail!("an atomic access of a type that is not a scalar")
        };
        let ord = self.atomic_ord(substs);
        match (which, args) {
            (Atomic::Load, &[at]) => Ok(Some(self.ins().atomic_load(machine, at, ord))),
            (Atomic::Cache, &[at, value]) => {
                self.ins().atomic_store(value, at, ord);
                Ok(None)
            }
            (Atomic::Rmw(op), &[at, value]) => {
                Ok(Some(self.ins().atomic_rmw(machine, op, ord, at, value)))
            }
            (Atomic::Exchange, &[at, expected, replacement]) => self
                .exchange(ty, ord, at, (expected, replacement))
                .map(Some),
            _ => bail!("atomic {which:?} with {} arguments", args.len()),
        }
    }

    fn exchange(
        &mut self,
        ty: Ty,
        ord: AtomicOrd,
        at: Value,
        (expected, replacement): (Value, Value),
    ) -> Result<Value> {
        let old = self.ins().atomic_cas(ord, at, expected, replacement);
        let swapped = self.ins().icmp(IntCC::Equal, old, expected);
        self.flagged(ty, old, swapped)
    }
}
