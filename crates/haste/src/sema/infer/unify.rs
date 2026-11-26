use anyhow::Result;

use crate::body::{Expr, ExprId};
use crate::names::ItemRef;

use super::super::substs::{ItemList, Substs};
use super::super::ty::{Kind, Mut, Ty, float_bits, int_info};
use super::Infer;

#[derive(Clone, Copy)]
struct Clash {
    a: Ty,
    b: Ty,
    cycle: bool,
}

impl Infer<'_, '_> {
    pub(super) fn trial<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<Option<T>>,
    ) -> Result<Option<T>> {
        let mark = (
            self.mark_vars(),
            self.projections.len(),
            self.obligations.len(),
        );
        let out = f(self);
        if matches!(out, Ok(Some(_))) {
            self.commit_vars(mark.0);
        } else {
            self.rollback_vars(mark.0);
            self.projections.truncate(mark.1);
            self.obligations.truncate(mark.2);
        }
        out
    }

    pub(super) fn probe<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let mark = (
            self.mark_vars(),
            self.projections.len(),
            self.obligations.len(),
        );
        let out = f(self);
        self.rollback_vars(mark.0);
        self.projections.truncate(mark.1);
        self.obligations.truncate(mark.2);
        out
    }

    pub(super) fn mark_vars(&mut self) -> (usize, usize) {
        self.trail_depth += 1;
        (self.vars.len(), self.var_trail.len())
    }

    pub(super) fn set_var(&mut self, index: usize, value: Option<Ty>) {
        if self.trail_depth > 0 {
            self.var_trail.push((index as u32, self.vars[index]));
        }
        self.vars[index] = value;
    }

    pub(super) fn rollback_vars(&mut self, (vars_len, trail_len): (usize, usize)) {
        while self.var_trail.len() > trail_len {
            let Some((index, prev)) = self.var_trail.pop() else {
                break;
            };
            if (index as usize) < vars_len {
                self.vars[index as usize] = prev;
            }
        }
        while self
            .diverge_trail
            .last()
            .is_some_and(|&(depth, _)| depth >= self.trail_depth)
        {
            if let Some((_, index)) = self.diverge_trail.pop()
                && (index as usize) < vars_len
                && (index as usize) < self.var_diverges.len()
            {
                self.var_diverges[index as usize] = false;
            }
        }
        self.vars.truncate(vars_len);
        self.var_diverges.truncate(vars_len);
        self.trail_depth -= 1;
    }

    pub(super) fn commit_vars(&mut self, (_, trail_len): (usize, usize)) {
        for (depth, _) in &mut self.diverge_trail {
            if *depth >= self.trail_depth {
                *depth = self.trail_depth - 1;
            }
        }
        if self.trail_depth == 1 {
            self.diverge_trail.clear();
        }
        self.trail_depth -= 1;
        if self.trail_depth == 0 {
            self.var_trail.truncate(trail_len);
        }
    }

    pub(super) fn capture_var_delta(
        &self,
        (vars_len, trail_len): (usize, usize),
    ) -> (Vec<(u32, Option<Ty>)>, Vec<Option<Ty>>, Vec<u32>) {
        let mut updates: Vec<(u32, Option<Ty>)> = self.var_trail[trail_len..]
            .iter()
            .filter_map(|&(i, _)| ((i as usize) < vars_len).then_some((i, None)))
            .collect();
        updates.sort_unstable_by_key(|&(i, _)| i);
        updates.dedup_by_key(|&mut (i, _)| i);
        for (i, val) in &mut updates {
            *val = self.vars[*i as usize];
        }
        let diverges = self
            .diverge_trail
            .iter()
            .rev()
            .take_while(|&&(d, _)| d >= self.trail_depth)
            .map(|&(_, i)| i)
            .collect();
        (updates, self.vars[vars_len..].to_vec(), diverges)
    }

    pub(super) fn apply_var_delta(
        &mut self,
        base_len: usize,
        updates: &[(u32, Option<Ty>)],
        added: &[Option<Ty>],
        diverges: &[u32],
    ) {
        for &(index, value) in updates {
            self.set_var(index as usize, value);
        }
        self.vars.truncate(base_len);
        self.var_diverges.truncate(base_len);
        self.vars.extend_from_slice(added);
        self.var_diverges.resize(self.vars.len(), false);
        for &index in diverges {
            self.set_diverging(index as usize);
        }
    }

    pub(super) fn fresh(&mut self, kind: fn(u32) -> Kind) -> Ty {
        self.vars.push(None);
        self.var_diverges.push(false);
        self.cx.types.intern(kind(self.vars.len() as u32 - 1))
    }

    fn set_diverging(&mut self, index: usize) {
        if self.var_diverges.len() <= index {
            self.var_diverges
                .resize(self.vars.len().max(index + 1), false);
        }
        if self.var_diverges[index] {
            return;
        }
        if self.trail_depth > 0 {
            self.diverge_trail.push((self.trail_depth, index as u32));
        }
        self.var_diverges[index] = true;
    }

    pub(super) fn var_of(&self, ty: Ty) -> Option<u32> {
        match self.cx.types.kind(ty) {
            Kind::Var(index) | Kind::IntVar(index) | Kind::FloatVar(index) => Some(*index),
            _ => None,
        }
    }

    /// `ty` with its outer inference variables followed; a variable of another inference stays as it is.
    pub fn resolve(&self, mut ty: Ty) -> Ty {
        while let Some(bound) = self
            .var_of(ty)
            .and_then(|index| self.vars.get(index as usize).copied().flatten())
        {
            ty = bound;
        }
        ty
    }

    fn bind(&mut self, var: Ty, to: Ty) -> Result<(), Clash> {
        let index = self.var_of(var).unwrap_or(u32::MAX) as usize;
        if index >= self.vars.len() {
            return Err(Clash {
                a: var,
                b: to,
                cycle: false,
            });
        }
        let accepts = match (self.cx.types.kind(var), self.cx.types.kind(to)) {
            (Kind::Var(_), _) => true,
            (Kind::IntVar(_), Kind::IntVar(_)) | (Kind::FloatVar(_), Kind::FloatVar(_)) => true,
            (Kind::IntVar(_), Kind::Prim(primitive)) => int_info(*primitive).is_some(),
            (Kind::FloatVar(_), Kind::Prim(primitive)) => float_bits(*primitive).is_some(),
            _ => false,
        };
        if !accepts {
            return Err(Clash {
                a: var,
                b: to,
                cycle: false,
            });
        }
        if self.occurs(index as u32, to) {
            return Err(Clash {
                a: var,
                b: to,
                cycle: true,
            });
        }
        self.set_var(index, Some(to));
        if self.var_diverges.get(index).copied().unwrap_or(false)
            && let Some(target) = self.var_of(to)
        {
            self.set_diverging(target as usize);
        }
        Ok(())
    }

    fn occurs(&self, var: u32, ty: Ty) -> bool {
        let ty = self.resolve(ty);
        if self.var_of(ty) == Some(var) {
            return true;
        }
        self.cx
            .types
            .kind(ty)
            .children(&self.cx.types)
            .any(|item| self.occurs(var, item))
    }

    /// Makes `a` and `b` the same type.
    pub fn unify(&mut self, a: Ty, b: Ty) -> Result<()> {
        self.unify_quiet(a, b)
            .map_err(|clash| self.clash_error(clash))
    }

    pub(super) fn unifies(&mut self, a: Ty, b: Ty) -> bool {
        self.unify_quiet(a, b).is_ok()
    }

    fn clash_error(&self, Clash { a, b, cycle }: Clash) -> anyhow::Error {
        match cycle {
            true => anyhow::anyhow!("cyclic type: {} occurs in {}", self.show(a), self.show(b)),
            false => anyhow::anyhow!("mismatched types: {} and {}", self.show(a), self.show(b)),
        }
    }

    fn unify_quiet(&mut self, a: Ty, b: Ty) -> Result<(), Clash> {
        let (a, b) = (self.resolve(a), self.resolve(b));
        if a == b {
            return Ok(());
        }
        enum Action {
            Bind(Ty, Ty),
            One(Ty, Ty),
            Two(Ty, Ty, Ty, Ty),
            Substs(Substs, Substs),
            FnPtr(Substs, Substs, Ty, Ty),
        }
        let action = match (self.cx.types.kind(a), self.cx.types.kind(b)) {
            (Kind::Var(_), _) => Action::Bind(a, b),
            (_, Kind::Var(_)) => Action::Bind(b, a),
            (Kind::IntVar(_) | Kind::FloatVar(_), _) => Action::Bind(a, b),
            (_, Kind::IntVar(_) | Kind::FloatVar(_)) => Action::Bind(b, a),
            (Kind::Tuple(x), Kind::Tuple(y)) => Action::Substs(*x, *y),
            (Kind::Ref(m, x), Kind::Ref(n, y)) | (Kind::Ptr(m, x), Kind::Ptr(n, y)) if m == n => {
                Action::One(*x, *y)
            }
            (Kind::Array(x, m), Kind::Array(y, n)) => Action::Two(*x, *y, *m, *n),
            (Kind::Slice(x), Kind::Slice(y)) => Action::One(*x, *y),
            (Kind::Pat(x, r1), Kind::Pat(y, r2)) if r1 == r2 => Action::One(*x, *y),
            (
                ka @ (Kind::Adt(i, x) | Kind::FnDef(i, x) | Kind::Dyn(i, x, ..)),
                kb @ (Kind::Adt(j, y) | Kind::FnDef(j, y) | Kind::Dyn(j, y, ..)),
            ) if std::mem::discriminant(ka) == std::mem::discriminant(kb)
                && i == j
                && dyn_members(ka) == dyn_members(kb) =>
            {
                Action::Substs(*x, *y)
            }
            (Kind::FnPtr(x, r), Kind::FnPtr(y, s)) => Action::FnPtr(*x, *y, *r, *s),
            _ => return Err(Clash { a, b, cycle: false }),
        };
        let clash = || Clash { a, b, cycle: false };
        match action {
            Action::Bind(var, ty) => self.bind(var, ty),
            Action::One(x, y) => self.unify_quiet(x, y).map_err(|_| clash()),
            Action::Two(x0, y0, x1, y1) => self
                .unify_quiet(x0, y0)
                .and_then(|()| self.unify_quiet(x1, y1))
                .map_err(|_| clash()),
            Action::Substs(x, y) => self.unify_substs(x, y).map_err(|()| clash()),
            Action::FnPtr(x, y, r, s) => self
                .unify_substs(x, y)
                .map_err(|()| clash())
                .and_then(|()| self.unify_quiet(r, s).map_err(|_| clash())),
        }
    }

    fn unify_substs(&mut self, a: Substs, b: Substs) -> Result<(), ()> {
        if a == b {
            return Ok(());
        }
        let len = self.cx.types.substs(a).len();
        if len != self.cx.types.substs(b).len() {
            return Err(());
        }
        for i in 0..len {
            let (x, y) = (self.cx.types.substs(a)[i], self.cx.types.substs(b)[i]);
            self.unify_quiet(x, y).map_err(drop)?;
        }
        Ok(())
    }

    /// Checks that a value of type `found` can be used where `expected` is wanted, through the coercions that keep its bits.
    pub fn coerce(&mut self, found: Ty, expected: Ty) -> Result<()> {
        let (found, expected) = (self.resolve(found), self.resolve(expected));
        match (self.cx.types.kind(found), self.weakened(found, expected)) {
            (Kind::Never, _) => {
                if let Kind::Var(index) = self.cx.types.kind(expected) {
                    self.set_diverging(*index as usize);
                }
                Ok(())
            }
            (_, Some((a, b))) => self.unify(a, b),
            _ => self.unify(found, expected),
        }?;
        self.settle_pending()
    }

    pub(super) fn weakened(&self, found: Ty, expected: Ty) -> Option<(Ty, Ty)> {
        match (
            self.cx.types.kind(self.resolve(found)),
            self.cx.types.kind(self.resolve(expected)),
        ) {
            (&Kind::Ref(from, a), &Kind::Ptr(to, b))
            | (&Kind::Ref(from, a), &Kind::Ref(to, b))
            | (&Kind::Ptr(from, a), &Kind::Ptr(to, b))
                if from == Mut::Mut && to == Mut::Not =>
            {
                Some((a, b))
            }
            (&Kind::Ref(from, a), &Kind::Ptr(to, b)) if from == to => Some((a, b)),
            _ => None,
        }
    }

    pub(super) fn settle_pending(&mut self) -> Result<()> {
        for _ in 0..crate::RECURSION_LIMIT {
            let before = self.backlog();
            self.settle_metadata()?;
            self.settle_projections()?;
            self.settle_opaques(false)?;
            self.settle_operators(false)?;
            self.settle_obligations()?;
            if self.backlog() >= before {
                return Ok(());
            }
        }
        Ok(())
    }

    pub(super) fn backlog(&self) -> usize {
        self.metadata.len()
            + self.projections.len()
            + self.opaques.len()
            + self.operands.len()
            + self.obligations.len()
    }

    pub(super) fn known_ty(&mut self, ty: Ty) -> Result<Ty> {
        if matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_)) {
            self.settle_pending()?;
        }
        Ok(self.resolve(ty))
    }

    /// Coerces the value of `expr`, of type `found`, to `expected`, at each branch value it ends in, recording an array-to-slice unsizing.
    pub fn coerce_expr(&mut self, expr: ExprId, found: Ty, expected: Ty) -> Result<()> {
        if self.known(Some(expected)).is_none() {
            return self.coerce(found, expected);
        }
        if self.resolve(found) == self.resolve(expected) {
            return self.settle_pending();
        }
        if let Some(ends) = self.branch_ends(expr) {
            for end in ends {
                let found = self.out.exprs[end.0 as usize];
                self.coerce_expr(end, found, expected)?;
            }
            self.out.exprs[expr.0 as usize] = expected;
            return Ok(());
        }
        match self.coercion(found, expected)? {
            Some(coercion) => {
                self.out.coercions.insert(expr, coercion);
                self.settle_pending()
            }
            None => self.coerce(found, expected),
        }
    }

    fn branch_ends(&self, expr: ExprId) -> Option<Vec<ExprId>> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => Some(vec![inner]),
            Expr::Block {
                tail: Some(tail),
                label: None,
                kind,
                ..
            } if kind.is_plain_scope() => Some(vec![tail]),
            Expr::If(_, then, Some(otherwise)) => Some(vec![then, otherwise]),
            Expr::Match(_, arms) => Some(
                self.body
                    .list(arms)
                    .iter()
                    .map(|&at| self.body.arms[at as usize].body)
                    .collect(),
            ),
            _ => None,
        }
    }

    pub(super) fn known(&self, expected: Option<Ty>) -> Option<Ty> {
        expected.filter(|ty| {
            !matches!(
                self.cx.types.kind(self.resolve(*ty)),
                Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)
            )
        })
    }

    pub(super) fn assume(&mut self, of: ItemRef, substs: &[Ty]) -> Result<bool> {
        let mut matched = Vec::new();
        for (held, types) in self.assumptions()? {
            if held != of || matched.contains(&types) {
                continue;
            }
            let fits = |infer: &mut Self| -> Result<Option<()>> {
                Ok(substs
                    .iter()
                    .zip(&types)
                    .all(|(&a, &b)| infer.unifies(a, b))
                    .then_some(()))
            };
            if self.probe(fits)?.is_some() {
                matched.push(types);
            }
        }
        if matched.len() > 1 {
            self.receiver_bounds(of, &mut matched)?;
        }
        let [types] = &matched[..] else {
            return Ok(false);
        };
        substs
            .iter()
            .zip(types)
            .try_for_each(|(&a, &b)| self.unify(a, b))?;
        Ok(true)
    }

    fn receiver_bounds(&mut self, of: ItemRef, matched: &mut Vec<Vec<Ty>>) -> Result<()> {
        let Some(param) = self.receiver_param else {
            return Ok(());
        };
        let Some(ty) = self.param_ty(param) else {
            return Ok(());
        };
        let own = self.header_bounds((param, ty), &mut |_, bound| Ok(bound == of), false)?;
        matched.retain(|types| {
            own.iter()
                .any(|(_, args)| types[0] == ty && types[1..] == args[..])
        });
        Ok(())
    }

    fn settle_metadata(&mut self) -> Result<()> {
        let mut at = 0;
        while at < self.metadata.len() {
            let (pointee, result) = self.metadata[at];
            if matches!(self.cx.types.kind(self.resolve(pointee)), Kind::Var(_)) {
                at += 1;
                continue;
            }
            self.metadata.swap_remove(at);
            let found = self.metadata_of(pointee)?;
            self.unify(found, result)?;
        }
        Ok(())
    }

    pub(super) fn unsettled(&self) -> ((usize, usize), usize, usize) {
        (
            self.cx.quiet_mark(),
            self.projections.len(),
            self.obligations.len(),
        )
    }

    pub(super) fn cache_state(&self) -> (((usize, usize), usize, usize), u32) {
        (self.unsettled(), self.cycle_hits)
    }

    pub(super) fn bodiless(&self) -> bool {
        self.instance.is_none()
            && self.items.is_empty()
            && self.scopes.is_empty()
            && self.const_args.is_empty()
            && self.evaluating.is_empty()
            && self.selecting.is_empty()
    }

    /// A short spelling of `ty` for messages.
    pub fn show(&self, ty: Ty) -> String {
        self.cx.type_text(ty, &|ty| self.resolve(ty))
    }
}

fn dyn_members(kind: &Kind) -> (ItemList, ItemList) {
    match *kind {
        Kind::Dyn(_, _, members, autos) => (members, autos),
        _ => (ItemList::EMPTY, ItemList::EMPTY),
    }
}
