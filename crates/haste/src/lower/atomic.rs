use anyhow::{Result, bail};
use haste_codegen::{IntCC, Value};

use crate::sema::intrinsic::Atomic;
use crate::sema::ty::Ty;

use super::{Lower, Repr, repr};

impl Lower<'_, '_> {
    /// Emits the atomic access `which` of the `ty` that the first argument points at, sequentially consistent whatever ordering the call names.
    pub(super) fn atomic(&mut self, which: Atomic, ty: Ty, args: &[Value]) -> Result<Option<Value>> {
        let Repr::Scalar(machine) = repr(self.cx, ty)? else { bail!("an atomic access of a type that is not a scalar") };
        match (which, args) {
            (Atomic::Load, &[at]) => Ok(Some(self.ins().atomic_load(machine, at))),
            (Atomic::Cache, &[at, value]) => {
                self.ins().atomic_store(value, at);
                Ok(None)
            }
            (Atomic::Rmw(op), &[at, value]) => Ok(Some(self.ins().atomic_rmw(machine, op, at, value))),
            (Atomic::Exchange, &[at, expected, replacement]) => self.exchange(ty, at, (expected, replacement)).map(Some),
            _ => bail!("atomic {which:?} with {} arguments", args.len()),
        }
    }

    fn exchange(&mut self, ty: Ty, at: Value, (expected, replacement): (Value, Value)) -> Result<Value> {
        let old = self.ins().atomic_cas(at, expected, replacement);
        let swapped = self.ins().icmp(IntCC::Equal, old, expected);
        self.flagged(ty, old, swapped)
    }
}
