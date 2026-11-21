use anyhow::{Result, bail};

use crate::RECURSION_LIMIT;
use crate::names::ItemRef;

use super::super::ty::{Kind, Mut, Ty};
use super::object::Vtable;
use super::poly::PolyPhase;
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
            if self.same_head(ty, to) && self.deref_step_fits(ty, to) {
                if steps.is_empty() {
                    return Ok(None);
                }
                self.unify(ty, to)?;
                return Ok(Some(Coercion::Deref(steps, self.resolve(to))));
            }
            let next = match self.cx.types.kind(ty).clone() {
                Kind::Ref(m, inner) if mutability == Mut::Not || m == Mut::Mut => Some((inner, None)),
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
}

pub(super) type DeferredSnap = (
    Vec<(ItemRef, Vec<Ty>)>,
    Vec<super::project::Projection>,
    Option<(PolyPhase, usize)>,
);

impl Infer<'_, '_> {
    pub(super) fn save_deferred(&self) -> DeferredSnap {
        (
            self.obligations.clone(),
            self.projections.clone(),
            self.poly.as_ref().map(|p| (p.phase, p.obligations.len())),
        )
    }

    pub(super) fn restore_deferred(&mut self, (obligations, projections, poly): DeferredSnap) {
        self.obligations = obligations;
        self.projections = projections;
        if let (Some(state), Some((phase, len))) = (&mut self.poly, poly) {
            state.phase = phase;
            state.obligations.truncate(len);
        }
    }

    fn deref_step_fits(&mut self, ty: Ty, to: Ty) -> bool {
        if ty == to {
            return true;
        }
        if !self.has_vars(to) || self.obligations.is_empty() {
            return self.trial_unify(ty, to);
        }
        let saved = self.save_deferred();
        let saved_poly = saved.2;
        let stuck_before = self.stuck_obligations();
        let ok = self.probe(|infer| {
            if !infer.unifies(ty, to) {
                return false;
            }
            if infer.settle_projections().is_err() || infer.settle_obligations().is_err() {
                return false;
            }
            let hit_fallback = saved_poly.is_some_and(|(was, _)| {
                was != PolyPhase::Fallback && infer.poly.as_ref().is_some_and(|p| p.phase == PolyPhase::Fallback)
            });
            !hit_fallback && infer.stuck_obligations() <= stuck_before
        });
        self.restore_deferred(saved);
        ok
    }

    fn stuck_obligations(&mut self) -> usize {
        let mut stuck = 0;
        for i in 0..self.obligations.len() {
            let (of, substs) = self.obligations[i].clone();
            if self.obligation_stuck(of, &substs) {
                stuck += 1;
            }
        }
        stuck
    }

    fn obligation_stuck(&mut self, of: ItemRef, substs: &[Ty]) -> bool {
        if !substs.iter().any(|&t| self.has_vars(t)) {
            return true;
        }
        let self_ty = self.resolve(substs[0]);
        if self.has_vars(self_ty)
            || self.is_callable_trait(of)
            || matches!(self.cx.lang(of), Some("discriminant_kind" | "bikeshed_guaranteed_no_drop"))
        {
            return false;
        }
        let args = &substs[1..];
        let known_holds = self.compiler_decides(of, self_ty)
            || self.compiler_impl(of, self_ty)
            || self.object_holds(of, self_ty, args) == Some(true);
        !known_holds && self.probe(|infer| infer.may_implement(of, self_ty, args).ok() == Some(false))
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

    /// The type both branches coerce to when they differ: an unsized slice, a function pointer, or one branch's reference or pointer of the same type, such as a `Box`, that the other coerces to.
    pub(super) fn branch_target(&mut self, a: Ty, b: Ty) -> Result<Option<Ty>> {
        if let Some(target) = self.slice_branch(a, b) {
            return Ok(Some(target));
        }
        if let Some(target) = self.fn_branch(a, b)? {
            return Ok(Some(target));
        }
        let (ka, kb) = (self.cx.types.kind(self.resolve(a)), self.cx.types.kind(self.resolve(b)));
        let pointers = match (ka, kb) {
            (Kind::Ref(..), Kind::Ref(..)) => true,
            (Kind::Adt(x, _), Kind::Adt(y, _)) => x == y,
            _ => false,
        };
        if !pointers {
            return Ok(None);
        }
        if self.trial(|infer| Ok(infer.coercion(b, a).ok().flatten()))?.is_some() {
            return Ok(Some(a));
        }
        Ok(self.trial(|infer| Ok(infer.coercion(a, b).ok().flatten()))?.map(|_| b))
    }

    /// The branch type that the other branch unsizes to, when one branch points to an array and the other to a slice.
    fn slice_branch(&self, a: Ty, b: Ty) -> Option<Ty> {
        let pointee = |ty: Ty| match self.cx.types.kind(self.resolve(ty)) {
            Kind::Ref(_, inner) | Kind::Ptr(_, inner) => Some(self.cx.types.kind(self.resolve(*inner)).clone()),
            _ => None,
        };
        match (pointee(a)?, pointee(b)?) {
            (Kind::Array(..), Kind::Slice(_)) => Some(b),
            (Kind::Slice(_), Kind::Array(..)) => Some(a),
            _ => None,
        }
    }

    fn fn_def_ptr(&mut self, item: ItemRef, substs: super::super::substs::Substs) -> Result<Ty> {
        let slice = self.cx.types.substs(substs).to_vec();
        let (params, ret) = self.fn_types(item, &slice)?;
        Ok(self.cx.types.fn_ptr(&params, ret))
    }

    fn non_cap_closure_sig(&mut self, ty: Ty) -> Result<Option<(Vec<Ty>, Ty)>> {
        let (inputs, output, captures) = self.closure_sig(ty)?;
        Ok((!captures).then_some((inputs, output)))
    }

    /// The function pointer type both branches reify to for distinct function items, non-capturing closures, or a function pointer.
    fn fn_branch(&mut self, a: Ty, b: Ty) -> Result<Option<Ty>> {
        let (a, b) = (self.resolve(a), self.resolve(b));
        let (a_kind, b_kind) = (self.cx.types.kind(a).clone(), self.cx.types.kind(b).clone());
        match (&a_kind, &b_kind) {
            (Kind::FnDef(..), Kind::FnPtr(..)) => Ok(Some(b)),
            (Kind::FnPtr(..), Kind::FnDef(..)) => Ok(Some(a)),
            (Kind::Closure(..), Kind::FnPtr(..)) => Ok(self.non_cap_closure_sig(a)?.map(|_| b)),
            (Kind::FnPtr(..), Kind::Closure(..)) => Ok(self.non_cap_closure_sig(b)?.map(|_| a)),
            (Kind::FnDef(item, substs), Kind::FnDef(..)) if a_kind != b_kind => self.fn_def_ptr(*item, *substs).map(Some),
            (Kind::FnDef(item, substs), Kind::Closure(..)) => Ok(self.non_cap_closure_sig(b)?.is_some().then(|| self.fn_def_ptr(*item, *substs)).transpose()?),
            (Kind::Closure(..), Kind::FnDef(item, substs)) => Ok(self.non_cap_closure_sig(a)?.is_some().then(|| self.fn_def_ptr(*item, *substs)).transpose()?),
            (Kind::Closure(..), Kind::Closure(..)) if a_kind != b_kind => {
                let (Some((in_a, out_a)), Some((in_b, out_b))) = (self.non_cap_closure_sig(a)?, self.non_cap_closure_sig(b)?) else {
                    return Ok(None);
                };
                if in_a.len() != in_b.len() {
                    return Ok(None);
                }
                let params: Vec<Ty> = in_a.iter().zip(&in_b).map(|(&x, &y)| if self.has_vars(x) && !self.has_vars(y) { y } else { x }).collect();
                let ret = if self.has_vars(out_a) && !self.has_vars(out_b) { out_b } else { out_a };
                Ok(Some(self.cx.types.fn_ptr(&params, ret)))
            }
            _ => Ok(None),
        }
    }

    pub(super) fn shared_pointer(&mut self, a: Ty, b: Ty) -> Result<Option<Ty>> {
        let (a, b) = (self.cx.types.kind(self.resolve(a)).clone(), self.cx.types.kind(self.resolve(b)).clone());
        let (shared, inner, other) = match (a, b) {
            (Kind::Ptr(x, inner), Kind::Ptr(y, other)) if x != y => (Kind::Ptr(Mut::Not, inner), inner, other),
            (Kind::Ref(x, inner), Kind::Ref(y, other)) if x != y => (Kind::Ref(Mut::Not, inner), inner, other),
            _ => return Ok(None),
        };
        self.unify(inner, other)?;
        Ok(Some(self.cx.types.intern(shared)))
    }

    /// Deduces a callable `FnPtr` signature from a trait object's supertraits when the object's principal trait is not itself a callable lang trait.
    pub(super) fn dyn_callable_super_sig(&mut self, ty: Ty) -> Result<Option<Ty>> {
        let ty = self.resolve(ty);
        let Kind::Dyn(of, parts, members, _) = self.cx.types.kind(ty).clone() else {
            return Ok(None);
        };
        if self.is_callable_trait(of) {
            return Ok(None);
        }
        let (obj_output, init_substs) = {
            let (given, mut bound) = super::super::ty::object_parts(self.cx.types.substs(parts), &members);
            let obj_output = bound.find(|&(m, _)| self.cx.name(m) == "Output").map(|(_, out)| out);
            let init_substs = std::iter::once(ty).chain(given.iter().copied()).collect::<Vec<Ty>>();
            (obj_output, init_substs)
        };
        let mut queue = vec![(of, init_substs, 0u32)];
        let mut seen = Vec::new();
        while let Some((trait_of, substs, depth)) = queue.pop() {
            if depth >= 8 || seen.contains(&trait_of) {
                continue;
            }
            seen.push(trait_of);
            for ((parent, parent_substs), bindings) in self.super_bindings(trait_of, &substs)? {
                if self.is_callable_trait(parent)
                    && let Some(&tuple) = parent_substs.get(1)
                    && let Kind::Tuple(inputs) = *self.cx.types.kind(self.resolve(tuple))
                {
                    let inputs = self.cx.types.substs(inputs).to_vec();
                    let output = if self.is_async_callable_trait(parent) {
                        self.fresh(Kind::Var)
                    } else {
                        bindings
                            .into_iter()
                            .find(|(name, _)| name == "Output")
                            .map(|(_, out)| out)
                            .or(obj_output)
                            .unwrap_or_else(|| self.fresh(Kind::Var))
                    };
                    return Ok(Some(self.cx.types.fn_ptr(&inputs, output)));
                }
                queue.push((parent, parent_substs, depth + 1));
            }
        }
        Ok(None)
    }

    /// Unifies `found` and `target` for structural/identity `as` casts that cannot be primitive conversions (such as `Result<(), _>`).
    pub(super) fn finish_identity_cast(&mut self, found: Ty, target: Ty) -> Result<()> {
        let (f, t) = (self.resolve(found), self.resolve(target));
        let (fk, tk) = (self.cx.types.kind(f), self.cx.types.kind(t));
        let structural = matches!(
            (fk, tk),
            (Kind::Adt(a, _), Kind::Adt(b, _)) if a == b
        ) || matches!(
            (fk, tk),
            (Kind::Tuple(_), Kind::Tuple(_))
                | (Kind::Array(..), Kind::Array(..))
                | (Kind::Slice(_), Kind::Slice(_))
                | (Kind::FnPtr(..), Kind::FnPtr(..))
                | (Kind::Never, _)
                | (Kind::Var(_), Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..) | Kind::Slice(_) | Kind::FnPtr(..))
        );
        if structural {
            self.unify(f, t)?;
            return self.settle_pending();
        }
        if let (Kind::Ref(m1, _), Kind::Ref(m2, _)) = (fk, tk)
            && (*m1 == *m2 || *m2 == Mut::Not)
        {
            return self.coerce(f, t);
        }
        Ok(())
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
