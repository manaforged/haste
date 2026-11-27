use anyhow::{Result, bail};

use crate::body::{Expr, ExprId, List};
use crate::names::ItemRef;

use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Res};

impl Infer<'_, '_> {
    /// Infers the call `expr` of the `dyn Fn` or `dyn FnMut` value of type `ty`, recording it as a call of `call` or `call_mut` on the trait object.
    pub(super) fn object_call(
        &mut self,
        (expr, found): (ExprId, Ty),
        ty: Ty,
        args: List,
    ) -> Result<Ty> {
        let Kind::Dyn(of, parts, members, _) = *self.cx.types.kind(ty) else {
            bail!("an object call of a type that is not a trait object")
        };
        let (tuple, bound_ret) = {
            let (given, mut bound) = super::super::ty::object_parts(
                self.cx.types.substs(parts),
                self.cx.types.items(members),
            );
            let Some(&tuple) = given.first() else {
                bail!("a callable trait object with no argument tuple")
            };
            (tuple, bound.next().map(|(_, ret)| ret))
        };
        let ret = bound_ret.unwrap_or_else(|| self.cx.types.unit());
        let Kind::Tuple(params) = *self.cx.types.kind(self.resolve(tuple)) else {
            bail!("a callable trait object whose arguments are not a tuple")
        };
        let params = self.cx.types.substs(params).to_vec();
        self.infer_call_args(args, &params, None)?;
        let lang = self.cx.lang(of);
        let name = super::callable::callable_method_name(lang);
        if matches!(name, "call_once" | "async_call_once") {
            self.free_box(expr, found)?;
        }
        let Some(method) = self.cx.trait_member(of, name) else {
            bail!("`{}` has no `{name}`", self.cx.name(of))
        };
        self.out
            .res
            .insert(expr, Res::fn_of(&self.cx.types, method, &[ty, tuple]));
        Ok(ret)
    }

    /// Infers the call `expr` of a value of type `found`, stepping through references, `Box`es and `Deref` impls to a callable impl or callable target.
    pub(super) fn overloaded_call(
        &mut self,
        expr: ExprId,
        found: Ty,
        args_list: List,
    ) -> Result<Ty> {
        let params: Vec<Ty> = self
            .body
            .list(args_list)
            .iter()
            .map(|_| self.fresh(Kind::Var))
            .collect();
        let tuple = self.cx.types.tuple(&params);
        let (mut ty, mut steps) = (self.resolve(found), 0);
        loop {
            if let Some(method) = self.callable_impl(ty, tuple)? {
                return self.finish_impl_call(
                    (expr, found, steps),
                    (method, ty, tuple),
                    (args_list, &params),
                );
            }
            if let Some(ret) = self.step_target_call((expr, found, steps), ty, args_list)? {
                return Ok(ret);
            }
            let next = match self.builtin_deref(ty) {
                Some(inner) => Some(inner),
                None => self.deref_step(expr, steps, ty)?,
            };
            let Some(next) = next else {
                return Err(self.diag_not_callable(expr, found));
            };
            if steps + 1 >= crate::RECURSION_LIMIT as u32 {
                return Err(self.diag_deref_limit(expr, found));
            }
            (ty, steps) = (self.resolve(next), steps + 1);
        }
    }

    fn finish_impl_call(
        &mut self,
        step: (ExprId, Ty, u32),
        (method, ty, tuple): (ItemRef, Ty, Ty),
        (args, params): (List, &[Ty]),
    ) -> Result<Ty> {
        self.record_call_steps(step, super::by_reference(self.cx, method) == Some(Mut::Mut))?;
        self.infer_call_args(args, params, None)?;
        self.out
            .res
            .insert(step.0, Res::fn_of(&self.cx.types, method, &[ty, tuple]));
        self.out.overloaded.insert(step.0);
        let Some(once) = self.cx.lang_item("fn_once") else {
            bail!("core has no `FnOnce`")
        };
        self.project(once, ty, vec![tuple], "Output")
    }

    fn step_target_call(
        &mut self,
        step: (ExprId, Ty, u32),
        ty: Ty,
        args: List,
    ) -> Result<Option<Ty>> {
        match self.cx.types.kind(ty).clone() {
            Kind::Dyn(of, ..) if self.is_callable_trait(of) => {
                self.record_call_steps(
                    step,
                    matches!(self.cx.lang(of), Some("fn_mut" | "async_fn_mut")),
                )?;
                self.object_call((step.0, step.1), ty, args).map(Some)
            }
            Kind::Dyn(..) => self.dyn_super_call(step, ty, args),
            Kind::Closure(..) => {
                let mutably = self.closure_mutates(ty)?;
                self.record_call_steps(step, mutably)?;
                self.closure_call(step.0, ty, args).map(Some)
            }
            Kind::FnDef(item, substs) => {
                self.record_call_steps(step, false)?;
                self.out.res.insert(step.0, Res::Fn(item, substs));
                let given = self.cx.types.substs(substs).to_vec();
                let (params, ret) = self.fn_types(item, &given)?;
                self.infer_call_args(args, &params, Some((item, &given)))?;
                Ok(Some(ret))
            }
            Kind::FnPtr(params, ret) => {
                self.record_call_steps(step, false)?;
                let params = self.cx.types.substs(params).to_vec();
                self.infer_call_args(args, &params, None)?;
                Ok(Some(ret))
            }
            Kind::UnsafeFnPtr(params, ret) => {
                self.check_unsafe_fn_ptr_call(step.0)?;
                self.record_call_steps(step, false)?;
                let params = self.cx.types.substs(params).to_vec();
                self.infer_call_args(args, &params, None)?;
                Ok(Some(ret))
            }
            Kind::Param(_) | Kind::Proj(_) if self.poly.is_some() => {
                if let Some((method, ..)) = self.poly_callable_target(ty) {
                    self.record_call_steps(
                        step,
                        super::by_reference(self.cx, method) == Some(Mut::Mut),
                    )?;
                }
                self.poly_call(step.0, step.1, ty, args).map(Some)
            }
            _ => Ok(None),
        }
    }

    pub(super) fn record_call_steps(
        &mut self,
        (expr, found, steps): (ExprId, Ty, u32),
        mutably: bool,
    ) -> Result<()> {
        self.out.autoderefs.remove(&expr);
        if steps > 0 {
            self.out.autoderefs.insert(expr, steps);
        }
        if mutably {
            self.steps_mutably(expr, found, steps)?;
            if let Expr::Call(callee, _) = *self.body.expr(expr) {
                self.note_mut(callee)?;
            }
        }
        Ok(())
    }

    pub(super) fn infer_call_args(
        &mut self,
        args: List,
        params: &[Ty],
        def: Option<(ItemRef, &[Ty])>,
    ) -> Result<()> {
        let args = self.body.list(args).to_vec();
        let variadic = match def {
            Some((item, _)) => self.cx.foreign(item) && self.cx.signature(item)?.variadic,
            None => false,
        };
        if args.len() < params.len() || (!variadic && args.len() != params.len()) {
            return Err(self.diag_arity(None, params.len(), args.len()));
        }
        for (&arg, &param) in args.iter().zip(params) {
            self.call_arg(def, ExprId(arg), (param, None), &mut None)?;
        }
        for &arg in &args[params.len()..] {
            self.expr(ExprId(arg), None)?;
        }
        if let Some((item, substs)) = def {
            self.check_transmute(self.current_expr, item, substs)?;
        }
        Ok(())
    }

    /// `Fn::call`, `FnMut::call_mut`, `FnOnce::call_once`, or their `AsyncFn*` counterparts, the first that an impl gives `ty` at the argument tuple `tuple`.
    pub(in crate::sema) fn callable_impl(&mut self, ty: Ty, tuple: Ty) -> Result<Option<ItemRef>> {
        for (lang, name) in super::callable::CALLABLE_METHODS {
            let Some(of) = self.cx.lang_item(lang) else {
                continue;
            };
            if self
                .trial(|infer| Ok(infer.select(of, ty, &[tuple])?.map(|_| ())))?
                .is_some()
            {
                return Ok(self.cx.trait_member(of, name));
            }
        }
        Ok(None)
    }

    /// Takes back the `Self` obligation of a call of trait item `item` at `substs` while a trait argument is still unknown, so the arguments are typed before an impl is picked, as rustc selects a callee's obligations only after its arguments that are not closures.
    pub(super) fn hold_open_obligation(
        &mut self,
        def: Option<&(ItemRef, Vec<Ty>)>,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let Some((of, substs)) =
            def.and_then(|(item, substs)| Some((self.cx.owner_trait(*item)?, substs)))
        else {
            return Ok(None);
        };
        let count = 1 + super::scope::type_params(&self.cx.trait_generics(of)?.1.params).count();
        let Some(outer) = substs.get(..count) else {
            return Ok(None);
        };
        let open = outer[1..]
            .iter()
            .any(|&ty| matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_)));
        let Some(at) = self
            .obligations
            .iter()
            .position(|(held, tys)| open && *held == of && tys[..] == *outer)
        else {
            return Ok(None);
        };
        Ok(Some(self.obligations.remove(at)))
    }

    /// Gives back the held obligation once an argument of type `found` is typed and before it is coerced to the open parameter `param`, as rustc selects pending obligations there, when the impl that picks takes that argument; an open trait argument may have impls here that selection does not see, so a pick the argument does not fit stays held until the arguments are coerced.
    pub(super) fn release_if_fits(
        &mut self,
        held: &mut Option<(ItemRef, Vec<Ty>)>,
        found: Ty,
        param: Ty,
    ) -> Result<()> {
        let Some(entry) = held.clone() else {
            return Ok(());
        };
        if !matches!(self.cx.types.kind(self.resolve(param)), Kind::Var(_)) {
            return Ok(());
        }
        let saved = self.save_deferred();
        let res = self.trial(|infer| {
            infer.obligations.push(entry);
            if infer.settle_obligations().is_err()
                || matches!(infer.cx.types.kind(infer.resolve(param)), Kind::Var(_))
            {
                return Ok(None);
            }
            Ok(infer.coerces_to(found, param).then_some(()))
        });
        if matches!(res, Ok(Some(()))) {
            *held = None;
        } else {
            self.restore_deferred(saved);
        }
        res.map(drop)
    }

    /// Gives back an obligation that `hold_open_obligation` took and settles what it now decides.
    pub(super) fn release_obligation(
        &mut self,
        held: &mut Option<(ItemRef, Vec<Ty>)>,
    ) -> Result<()> {
        let Some(entry) = held.take() else {
            return Ok(());
        };
        self.obligations.push(entry);
        self.settle_pending()
    }

    /// Infers the arguments `args` of a call of a function with `params` that returns `ret`, the function item `def` when it is one, and returns `ret`.
    pub(super) fn call_args(
        &mut self,
        args: Vec<u32>,
        (params, ret, def): (Vec<Ty>, Ty, Option<(ItemRef, Vec<Ty>)>),
        (expected, mut held): (Option<Ty>, Option<(ItemRef, Vec<Ty>)>),
    ) -> Result<Ty> {
        let extra = self.variadic_extra(
            def.as_ref().map(|(item, _)| *item),
            args.len(),
            params.len(),
        )?;
        let value_count = params.len();
        let def_ref = def.as_ref().map(|(item, substs)| (*item, &substs[..]));
        self.call_args_ordered(
            &args[..value_count],
            (&params, ret, expected),
            def_ref,
            &mut held,
        )?;
        for &arg in &args[args.len() - extra..] {
            self.expr(ExprId(arg), None)?;
        }
        if let Some((item, substs)) = def {
            self.check_transmute(self.current_expr, item, &substs)?;
        }
        Ok(ret)
    }

    pub(super) fn call_args_ordered(
        &mut self,
        args: &[u32],
        (params, ret, expected): (&[Ty], Ty, Option<Ty>),
        def: Option<(ItemRef, &[Ty])>,
        held: &mut Option<(ItemRef, Vec<Ty>)>,
    ) -> Result<()> {
        let hints = self.expected_inputs(params, ret, expected);
        let has_deferred = args
            .iter()
            .any(|&arg| self.is_deferred_closure_arg(ExprId(arg)));
        for (&arg, (&param, &hint)) in args.iter().zip(params.iter().zip(&hints)) {
            if !self.is_deferred_closure_arg(ExprId(arg)) {
                self.call_arg(def, ExprId(arg), (param, hint), held)?;
            }
        }
        if has_deferred || def.is_some_and(|(item, _)| self.is_transmute(item)) {
            self.unify_expected_ret(ret, expected);
        }
        self.release_obligation(held)?;
        for (&arg, (&param, &hint)) in args.iter().zip(params.iter().zip(&hints)) {
            if self.is_deferred_closure_arg(ExprId(arg)) {
                self.call_arg(def, ExprId(arg), (param, hint), held)?;
            }
        }
        Ok(())
    }

    fn unify_expected_ret(&mut self, ret: Ty, expected: Option<Ty>) {
        let Some(wanted) = expected.map(|ty| self.resolve(ty)) else {
            return;
        };
        let ret = self.resolve(ret);
        if !matches!(self.cx.types.kind(ret), Kind::Var(_))
            || matches!(
                self.cx.types.kind(wanted),
                Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)
            )
            || self.cx.types.kind(wanted).is_unsized()
        {
            return;
        }
        if matches!(
            self.probe(|infer| -> Result<bool> { Ok(infer.unifies(ret, wanted)) }),
            Ok(true)
        ) {
            let _ = self.unify(ret, wanted);
        }
    }

    fn is_deferred_closure_arg(&self, expr: ExprId) -> bool {
        self.is_closure(expr)
            || matches!(*self.body.expr(expr), Expr::Ref { expr: inner, .. } if self.is_closure(inner))
    }

    /// How many arguments past its `params` a call of `callee` with `given` arguments passes, which only a C-variadic foreign function takes.
    fn variadic_extra(
        &mut self,
        callee: Option<ItemRef>,
        given: usize,
        params: usize,
    ) -> Result<usize> {
        let variadic = match callee {
            Some(item) => self.cx.foreign(item) && self.cx.signature(item)?.variadic,
            None => false,
        };
        match given.checked_sub(params) {
            Some(0) => Ok(0),
            Some(extra) if variadic => Ok(extra),
            _ => Err(self.diag_arity(None, params, given)),
        }
    }

    pub(super) fn expected_inputs(
        &mut self,
        params: &[Ty],
        ret: Ty,
        expected: Option<Ty>,
    ) -> Vec<Option<Ty>> {
        let Some(expected) = expected else {
            return vec![None; params.len()];
        };
        self.probe(|infer| match infer.unifies(ret, expected) {
            true => params
                .iter()
                .map(|&param| Some(infer.substitute(param)))
                .collect(),
            false => vec![None; params.len()],
        })
    }

    pub(super) fn free_box(&mut self, expr: ExprId, found: Ty) -> Result<()> {
        let found = self.resolve(found);
        if self.cx.boxed(found).is_none() || self.out.autoderefs.get(&expr) != Some(&1) {
            bail!("a `dyn FnOnce` is called by value only through a `Box`");
        }
        let Some((block, args)) = self.drop_impl(found)? else {
            bail!("`Box` has no `Drop` impl")
        };
        let Some(member) = self.cx.member(block, "drop") else {
            bail!("the `Drop` impl of `Box` has no `drop`")
        };
        let args: Vec<Ty> = args.into_iter().map(|arg| self.zonk(arg)).collect();
        self.out
            .box_frees
            .insert(expr, Res::fn_of(&self.cx.types, member, &args));
        Ok(())
    }

    fn is_transmute(&mut self, item: ItemRef) -> bool {
        self.cx.outline.is_sysroot(item.krate)
            && self.cx.name(item) == "transmute"
            && (self.cx.intrinsic(item) == Some(super::super::intrinsic::Intrinsic::Transmute)
                || matches!(
                    self.cx.qualified(item).as_str(),
                    "core::mem::transmute" | "core::intrinsics::transmute" | "std::mem::transmute"
                ))
    }

    pub(super) fn check_transmute(
        &mut self,
        expr: Option<ExprId>,
        item: ItemRef,
        substs: &[Ty],
    ) -> Result<()> {
        if !self.is_transmute(item) {
            return Ok(());
        }
        let [src, dst, ..] = substs else {
            return Ok(());
        };
        let (src, dst) = (self.substitute(*src), self.substitute(*dst));
        if !self.cx.types.is_ground(src) || !self.cx.types.is_ground(dst) {
            if self.has_params(src) || self.has_params(dst) {
                self.poly_fallback();
            }
            return Ok(());
        }
        let (src_layout, dst_layout) = (self.cx.layout(src)?, self.cx.layout(dst)?);
        if src_layout.size != dst_layout.size {
            let (src_bytes, dst_bytes) = (src_layout.size.bytes(), dst_layout.size.bytes());
            return Err(self.diag_expr(
                crate::diag::DiagnosticKind::Type,
                expr,
                format!(
                    "cannot transmute between types of different sizes, or dependently-sized types: `{}` ({src_bytes} bytes) to `{}` ({dst_bytes} bytes)",
                    self.show(src),
                    self.show(dst),
                ),
                format!("cannot transmute between types of different sizes ({src_bytes} vs {dst_bytes} bytes)"),
            ));
        }
        Ok(())
    }

    pub(super) fn check_deferred_transmutes(&mut self) -> Result<()> {
        let mut sites: Vec<(ExprId, ItemRef, super::super::substs::Substs)> = self
            .out
            .res
            .iter()
            .filter_map(|(&expr, res)| match *res {
                Res::Fn(item, substs) if self.cx.name(item) == "transmute" => {
                    Some((expr, item, substs))
                }
                _ => None,
            })
            .collect();
        sites.sort_by_key(|(expr, _, _)| expr.0);
        for (expr, item, substs) in sites {
            let slice = self.cx.types.substs(substs).to_vec();
            self.check_transmute(Some(expr), item, &slice)?;
        }
        Ok(())
    }

    pub(super) fn enter_const_root_cycle(
        &self,
        expr: ExprId,
        is_root: bool,
    ) -> Result<Option<super::super::evaluate::ConstCycleFrame>> {
        if !is_root || self.poly.is_some() || self.sig.ret.is_none() {
            return Ok(None);
        }
        let Some((item, substs)) = self.instance else {
            return Ok(None);
        };
        if self.cx.outline.is_sysroot(item.krate)
            || super::super::evaluate::const_cycle_top() == Some((item, substs))
        {
            return Ok(None);
        }
        let is_const = self.cx.item(item).1.kind == crate::outline::item::ItemKind::Const;
        if !is_const && !super::super::evaluate::const_cycle_active() {
            return Ok(None);
        }
        if self
            .cx
            .types
            .substs(substs)
            .iter()
            .any(|&ty| !self.cx.types.is_ground(ty))
        {
            return Ok(None);
        }
        super::super::evaluate::enter_const_cycle(self.cx, (item, substs), is_const).map_err(
            |msg| {
                self.diag_expr(
                    crate::diag::DiagnosticKind::Type,
                    Some(expr),
                    msg,
                    "cyclic constant value",
                )
            },
        )
    }

    pub(super) fn check_const_deps(&mut self, root_expr: ExprId) -> Result<()> {
        let deps = super::super::evaluate::collect_body_const_deps(
            self.cx,
            self.body,
            |id| self.out.res.get(&id).copied(),
            |id| self.out.operators.get(&id).copied(),
            |id| self.out.pat_res.get(&id).copied(),
        );
        for (site, raw_item, raw_substs) in deps {
            let raw_args = self.cx.types.substs(raw_substs).to_vec();
            let resolved: Vec<Ty> = raw_args.into_iter().map(|t| self.substitute(t)).collect();
            if resolved.iter().any(|&t| !self.cx.types.is_ground(t)) {
                continue;
            }
            let substs = self.cx.types.intern_substs(&resolved);
            let Res::Fn(item, substs) = self.trait_call(raw_item, substs)? else {
                continue;
            };
            self.inspect_const_dep(site.or(Some(root_expr)), item, substs)?;
        }
        Ok(())
    }

    fn inspect_const_dep(
        &mut self,
        site: Option<ExprId>,
        item: ItemRef,
        substs: super::super::substs::Substs,
    ) -> Result<()> {
        use crate::outline::item::ItemKind;
        if self.cx.outline.is_sysroot(item.krate) || self.cx.foreign(item) {
            return Ok(());
        }
        let kind = self.cx.item(item).1.kind;
        if !matches!(kind, ItemKind::Const | ItemKind::Fn)
            || (kind == ItemKind::Fn && self.cx.item(item).1.body.is_none())
        {
            return Ok(());
        }
        let Some(_frame) = super::super::evaluate::enter_const_cycle(
            self.cx,
            (item, substs),
            kind == ItemKind::Const,
        )
        .map_err(|msg| {
            self.diag_expr(
                crate::diag::DiagnosticKind::Type,
                site,
                msg,
                "cyclic constant value",
            )
        })?
        else {
            return Ok(());
        };
        if let Some(typed) = self.cx.store.finals.get(&(item, substs))
            && let super::LazyTables::Eager(tables) = &typed.tables
        {
            let body = self.cx.body(item)?;
            let deps = super::super::evaluate::collect_body_const_deps(
                self.cx,
                &body,
                |id| tables.res.get(&id).copied(),
                |id| tables.operators.get(&id).copied(),
                |id| tables.pat_res.get(&id).copied(),
            );
            for (_, dep_item, dep_substs) in deps {
                if self
                    .cx
                    .types
                    .substs(dep_substs)
                    .iter()
                    .any(|&t| !self.cx.types.is_ground(t))
                {
                    continue;
                }
                let Res::Fn(dep_item, dep_substs) = self.trait_call(dep_item, dep_substs)? else {
                    continue;
                };
                self.inspect_const_dep(site, dep_item, dep_substs)?;
            }
            return Ok(());
        }
        let sig = self.cx.signature(item)?;
        let body = self.cx.body(item)?;
        if body.root.is_none() {
            return Ok(());
        }
        let src = self.cx.body_src(item);
        let home = self.cx.module_of(item)?;
        let args = self.cx.types.substs(substs).to_vec();
        let before = self.cx.quiet_mark();
        let mut child = Box::new(Infer::start(self.cx, (&body, &sig, src), home));
        (child.root, child.instance) = (Some(item.body_owner()), Some((item, substs)));
        let mut frame = child.frame(item, &args)?;
        (frame.src, frame.root) = (src, child.root);
        child.swap_frame(frame);
        child.out.exprs.resize(body.exprs.len(), Ty::NONE);
        let ran = child.run(false);
        if child.cx.quiet_mark() != before {
            return Ok(());
        }
        ran?;
        if let Some(dep_root) = child.body.root {
            child.check_const_deps(dep_root)?;
        }
        Ok(())
    }

    pub(super) fn in_unsafe_ctx(&self) -> bool {
        UNSAFE_STATE.with(|cell| {
            cell.borrow()
                .depth
                .unwrap_or_else(|| u32::from(self.sig.is_unsafe))
                > 0
        })
    }

    pub(super) fn check_unsafe_fn_ptr_call(&self, expr: ExprId) -> Result<()> {
        if self.cx.outline.is_sysroot(self.home.krate) || self.in_unsafe_ctx() {
            return Ok(());
        }
        Err(self.diag_expr(
            crate::diag::DiagnosticKind::Unsafe,
            Some(expr),
            "call to unsafe function is unsafe and requires unsafe function or block",
            "call to unsafe function",
        ))
    }

    pub(super) fn check_union_field(&self, expr: ExprId, item: ItemRef) -> Result<()> {
        if self.cx.outline.is_sysroot(self.home.krate)
            || self.cx.item(item).1.kind != crate::outline::item::ItemKind::Union
        {
            return Ok(());
        }
        let body_ptr = self.body as *const crate::body::Body as usize;
        let allowed = self.in_unsafe_ctx()
            || UNSAFE_STATE.with(|cell| cell.borrow().safe_fields.contains(&(body_ptr, expr)));
        if allowed {
            return Ok(());
        }
        Err(self.diag_expr(
            crate::diag::DiagnosticKind::Unsafe,
            Some(expr),
            "access to union field is unsafe and requires unsafe function or block",
            "access to union field",
        ))
    }

    pub(super) fn push_unsafe_ctx(&self, expr: ExprId, is_root: bool) -> UnsafeGuard {
        use crate::body::BlockKind;
        let body_ptr = self.body as *const crate::body::Body as usize;
        UNSAFE_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            let guard = UnsafeGuard {
                prev_depth: st.depth,
                safe_mark: st.safe_fields.len(),
            };
            let base = if is_root {
                u32::from(self.sig.is_unsafe)
            } else {
                st.depth.unwrap_or_else(|| u32::from(self.sig.is_unsafe))
            };
            st.depth = Some(match *self.body.expr(expr) {
                Expr::Block { kind, .. } => match kind {
                    BlockKind::Unsafe => base + 1,
                    BlockKind::Const | BlockKind::Async | BlockKind::AsyncMove => 0,
                    _ => base,
                },
                Expr::Assign(lhs, _) => {
                    if let Some(target) = unwrap_paren_field(self.body, lhs) {
                        st.safe_fields.push((body_ptr, target));
                    }
                    base
                }
                Expr::Ref {
                    raw: true,
                    expr: inner,
                    ..
                } => {
                    push_raw_safe_fields(self.body, body_ptr, inner, &mut st.safe_fields);
                    base
                }
                _ => base,
            });
            guard
        })
    }
}

#[derive(Default)]
struct UnsafeState {
    depth: Option<u32>,
    safe_fields: Vec<(usize, ExprId)>,
}

thread_local! {
    static UNSAFE_STATE: std::cell::RefCell<UnsafeState> = const {
        std::cell::RefCell::new(UnsafeState {
            depth: None,
            safe_fields: Vec::new(),
        })
    };
}

pub(super) struct UnsafeGuard {
    prev_depth: Option<u32>,
    safe_mark: usize,
}

impl Drop for UnsafeGuard {
    fn drop(&mut self) {
        UNSAFE_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            st.depth = self.prev_depth;
            st.safe_fields.truncate(self.safe_mark);
        });
    }
}

fn unwrap_paren_field(body: &crate::body::Body, mut expr: ExprId) -> Option<ExprId> {
    loop {
        match *body.expr(expr) {
            Expr::Paren(inner) => expr = inner,
            Expr::Field(..) => return Some(expr),
            _ => return None,
        }
    }
}

fn push_raw_safe_fields(
    body: &crate::body::Body,
    body_ptr: usize,
    mut expr: ExprId,
    out: &mut Vec<(usize, ExprId)>,
) {
    loop {
        match *body.expr(expr) {
            Expr::Paren(inner) => expr = inner,
            Expr::Field(base, _) => {
                out.push((body_ptr, expr));
                expr = base;
            }
            Expr::TupleField(base, _) | Expr::Index(base, _) => expr = base,
            _ => break,
        }
    }
}
