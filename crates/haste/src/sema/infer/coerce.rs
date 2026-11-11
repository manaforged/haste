use anyhow::{Result, bail};

use super::super::ty::{Kind, Mut, Ty};
use super::object::Vtable;
use super::{Infer, Res};
use crate::RECURSION_LIMIT;

/// How the value of an expression changes as it coerces to the type that it is used at.
#[derive(Clone, Debug)]
pub enum Coercion {
    /// A pointer to an array of this length becomes a pointer to a slice.
    Len(Ty),
    /// A pointer to a sized value becomes a pointer to a trait object, with this vtable.
    Vtable(Vtable),
    /// A function item, or a closure that captures nothing, becomes a pointer to the code of this function.
    Reify(Res),
    /// A reference becomes a reference to what dereferencing its pointee reaches: each step names the type it dereferences and, unless the compiler dereferences it, the `Deref::deref` call; then the type reached.
    Deref(Vec<(Ty, Option<Res>)>, Ty),
    /// A pointer to a trait object becomes a pointer to the same trait object with fewer auto traits, with its value unchanged.
    Autos,
    /// A pointer to a trait object becomes a pointer to a trait object of a supertrait, whose vtable this slot of the old one holds.
    Upcast(u32),
}

impl Coercion {
    /// Whether the coerced value is a pointer that carries metadata after its address.
    pub fn widens(&self) -> bool {
        !matches!(self, Coercion::Reify(_) | Coercion::Deref(..))
    }
}

impl Infer<'_, '_> {
    /// The coercion of a value of type `found` to `expected`, when one applies.
    pub(super) fn coercion(&mut self, found: Ty, expected: Ty) -> Result<Option<Coercion>> {
        let (found, expected) = (self.resolve(found), self.resolve(expected));
        match (self.cx.types.kind(found).clone(), self.cx.types.kind(expected).clone()) {
            (Kind::Ref(from, a) | Kind::Ptr(from, a), Kind::Ref(to, b) | Kind::Ptr(to, b)) => self.pointer_coercion((found, expected), (from, a), (to, b)),
            (Kind::Adt(a, from), Kind::Adt(b, to)) if a == b && from != to && self.coerces(found, expected)? => {
                let (from, to) = (self.cx.types.substs(from).to_vec(), self.cx.types.substs(to).to_vec());
                self.unsize_args(&from, &to)
            }
            (Kind::FnDef(item, substs), Kind::FnPtr(params, ret)) => {
                let args = self.cx.types.substs(substs).to_vec();
                let wanted = self.cx.types.substs(params).to_vec();
                let have = self.fn_types(item, &args)?;
                self.unify_fn(have, (wanted, ret))?;
                Ok(Some(Coercion::Reify(Res::Fn(item, substs))))
            }
            (Kind::Closure(..), Kind::FnPtr(params, ret)) => {
                let wanted = self.cx.types.substs(params).to_vec();
                self.reify_closure(found, (wanted, ret))
            }
            _ => Ok(None),
        }
    }

    fn coerces(&mut self, found: Ty, expected: Ty) -> Result<bool> {
        let Some(of) = self.cx.lang_item("coerce_unsized") else {
            return Ok(false);
        };
        self.may_implement(of, found, &[expected])
    }

    fn unsize_args(&mut self, from: &[Ty], to: &[Ty]) -> Result<Option<Coercion>> {
        let Some((at, unsize)) = self.unsized_arg(from, to)? else {
            return Ok(None);
        };
        let rest = from.iter().zip(to).enumerate().filter(|(index, _)| *index != at);
        rest.map(|(_, pair)| pair).try_for_each(|(&a, &b)| self.unify(a, b))?;
        Ok(Some(unsize))
    }

    fn unsized_arg(&mut self, from: &[Ty], to: &[Ty]) -> Result<Option<(usize, Coercion)>> {
        for (at, (&a, &b)) in from.iter().zip(to).enumerate() {
            if let Some(unsize) = self.unsize_pointee(a, b)? {
                return Ok(Some((at, unsize)));
            }
            let nested = matches!((self.cx.types.kind(self.resolve(a)), self.cx.types.kind(self.resolve(b))), (Kind::Adt(x, _), Kind::Adt(y, _)) if x == y);
            if nested && let Some(unsize) = self.coercion(a, b)? {
                return Ok(Some((at, unsize)));
            }
        }
        Ok(None)
    }

    fn pointer_coercion(&mut self, (found, expected): (Ty, Ty), (from, a): (Mut, Ty), (to, b): (Mut, Ty)) -> Result<Option<Coercion>> {
        let unsize = self.unsize_pointee(a, b)?;
        let loosens = from == Mut::Not && to == Mut::Mut;
        if unsize.is_some() && loosens {
            bail!("cannot coerce a shared pointer to a mutable one");
        }
        let both_refs = matches!(self.cx.types.kind(found), Kind::Ref(..)) && matches!(self.cx.types.kind(expected), Kind::Ref(..));
        match unsize {
            None if both_refs && !loosens => self.deref_coercion(a, b, to),
            unsize => Ok(unsize),
        }
    }

    fn deref_coercion(&mut self, from: Ty, to: Ty, mutability: Mut) -> Result<Option<Coercion>> {
        let (mut ty, to) = (self.resolve(from), self.resolve(to));
        let mut steps = Vec::new();
        while steps.len() < RECURSION_LIMIT {
            if self.same_head(ty, to) && (ty == to || self.trial_unify(ty, to)) {
                return Ok((!steps.is_empty()).then_some(Coercion::Deref(steps, to)));
            }
            let next = match self.cx.types.kind(ty).clone() {
                Kind::Ref(_, inner) => Some((inner, None)),
                Kind::Adt(..) if self.cx.boxed(ty).is_some() => self.cx.boxed(ty).map(|inner| (inner, None)),
                Kind::Adt(..) => self.overloaded_deref(ty, mutability)?.map(|(res, target)| (target, Some(res))),
                Kind::Param(_) | Kind::Proj(_) => {
                    self.poly_fallback();
                    self.overloaded_deref(ty, mutability)?.map(|(res, target)| (target, Some(res)))
                }
                _ => None,
            };
            let Some((inner, res)) = next else {
                return Ok(None);
            };
            steps.push((ty, res));
            ty = self.resolve(inner);
        }
        Ok(None)
    }

    fn same_head(&self, a: Ty, b: Ty) -> bool {
        let (a, b) = (self.cx.types.kind(a), self.cx.types.kind(b));
        match (a, b) {
            (Kind::Var(_), _) | (_, Kind::Var(_)) => true,
            (Kind::Adt(x, _), Kind::Adt(y, _)) => x == y,
            _ => std::mem::discriminant(a) == std::mem::discriminant(b),
        }
    }

    pub(super) fn trial_unify(&mut self, a: Ty, b: Ty) -> bool {
        matches!(self.trial(|infer| Ok(infer.unifies(a, b).then_some(()))), Ok(Some(())))
    }

    fn reify_closure(&mut self, found: Ty, (params, ret): (Vec<Ty>, Ty)) -> Result<Option<Coercion>> {
        let (inputs, output, captures) = self.closure_sig(found)?;
        if captures {
            bail!("a closure that captures variables cannot become a function pointer");
        }
        self.unify_fn((inputs.clone(), output), (params, ret))?;
        let Some(call_once) = self.cx.call_once() else { bail!("core has no `FnOnce::call_once`") };
        let tuple = self.cx.types.tuple(&inputs);
        Ok(Some(Coercion::Reify(Res::fn_of(&self.cx.types, call_once, &[found, tuple]))))
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
            Coercion::Autos | Coercion::Upcast(_) => Ok(coercion),
            Coercion::Vtable(vtable) => self.zonk_vtable(vtable).map(Coercion::Vtable),
            Coercion::Reify(res) => self.zonk_res(res).map(Coercion::Reify),
            Coercion::Deref(steps, to) => {
                let mut zonked = Vec::new();
                for (ty, res) in steps {
                    zonked.push((self.zonk(ty), res.map(|res| self.zonk_res(res)).transpose()?));
                }
                Ok(Coercion::Deref(zonked, self.zonk(to)))
            }
        }
    }
}
