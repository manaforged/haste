use anyhow::{Result, bail};

use super::super::ty::{Kind, Mut, Ty};
use super::object::Vtable;
use super::{Infer, Res};

/// How the value of an expression changes as it coerces to the type that it is used at.
#[derive(Clone, Debug)]
pub enum Coercion {
    /// A pointer to an array of this length becomes a pointer to a slice.
    Len(Ty),
    /// A pointer to a sized value becomes a pointer to a trait object, with this vtable.
    Vtable(Vtable),
    /// A function item, or a closure that captures nothing, becomes a pointer to the code of this function.
    Reify(Res),
}

impl Coercion {
    /// Whether the coerced value is a pointer that carries metadata after its address.
    pub fn widens(&self) -> bool {
        !matches!(self, Coercion::Reify(_))
    }
}

impl Infer<'_, '_> {
    /// The coercion of a value of type `found` to `expected`, when one applies.
    pub(super) fn coercion(&mut self, found: Ty, expected: Ty) -> Result<Option<Coercion>> {
        let (found, expected) = (self.resolve(found), self.resolve(expected));
        match (self.cx.types.kind(found).clone(), self.cx.types.kind(expected).clone()) {
            (Kind::Ref(from, a) | Kind::Ptr(from, a), Kind::Ref(to, b) | Kind::Ptr(to, b)) => {
                let unsize = self.unsize_pointee(a, b)?;
                if unsize.is_some() && from == Mut::Not && to == Mut::Mut {
                    bail!("cannot coerce a shared pointer to a mutable one");
                }
                Ok(unsize)
            }
            (Kind::Adt(_, from), Kind::Adt(_, to)) if self.cx.boxed(found).is_some() && self.cx.boxed(expected).is_some() => self.unsize_box(&from, &to),
            (Kind::FnDef(item, substs), Kind::FnPtr(params, ret)) => {
                let have = self.fn_types(item, &substs)?;
                self.unify_fn(have, (params, ret))?;
                Ok(Some(Coercion::Reify(Res::Fn(item, substs))))
            }
            (Kind::Closure(..), Kind::FnPtr(params, ret)) => self.reify_closure(found, (params, ret)),
            _ => Ok(None),
        }
    }

    fn unsize_box(&mut self, from: &[Ty], to: &[Ty]) -> Result<Option<Coercion>> {
        let unsize = self.unsize_pointee(from[0], to[0])?;
        if unsize.is_some() {
            from[1..].iter().zip(&to[1..]).try_for_each(|(&a, &b)| self.unify(a, b))?;
        }
        Ok(unsize)
    }

    fn reify_closure(&mut self, found: Ty, (params, ret): (Vec<Ty>, Ty)) -> Result<Option<Coercion>> {
        let (inputs, output, captures) = self.closure_sig(found)?;
        if captures {
            bail!("a closure that captures variables cannot become a function pointer");
        }
        self.unify_fn((inputs.clone(), output), (params, ret))?;
        let Some(call_once) = self.cx.call_once() else { bail!("core has no `FnOnce::call_once`") };
        let tuple = self.cx.types.intern(Kind::Tuple(inputs));
        Ok(Some(Coercion::Reify(Res::Fn(call_once, vec![found, tuple]))))
    }

    fn unify_fn(&mut self, (params, ret): (Vec<Ty>, Ty), (wanted, out): (Vec<Ty>, Ty)) -> Result<()> {
        if params.len() != wanted.len() {
            bail!("a function of {} parameters where {} are wanted", params.len(), wanted.len());
        }
        params.into_iter().zip(wanted).try_for_each(|(param, wanted)| self.unify(param, wanted))?;
        self.unify(ret, out)
    }

    /// The coercion `coercion` once its types are known.
    pub(super) fn zonk_coercion(&mut self, coercion: Coercion) -> Result<Coercion> {
        match coercion {
            Coercion::Len(len) => Ok(Coercion::Len(self.zonk(len))),
            Coercion::Vtable(vtable) => self.zonk_vtable(vtable).map(Coercion::Vtable),
            Coercion::Reify(res) => self.zonk_res(res).map(Coercion::Reify),
        }
    }
}
