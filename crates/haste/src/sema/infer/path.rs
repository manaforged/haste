use anyhow::{Result, bail};

use crate::body::sig::{Generic, GenericKind};
use crate::body::{self as syntax, Args, Body, GenericArg, PathId, TyId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, Ns, PerNs, Primitive, Sym, Symbols, plain};
use crate::outline::item::ItemKind;
use crate::outline::source::{Mark, Source};

use super::super::body_scope::{BodyPath, outside};
use super::super::impls::ImplRef;
use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Res};

/// The interned segments of `path` in `arena`, whether it starts with `::`, and the crate a leading `$crate` names.
pub fn segments(symbols: &Symbols, src: &Source, arena: &Body, path: PathId) -> (Vec<Sym>, bool, Option<u32>) {
    let path = arena.path(path);
    let segments: Vec<Sym> = arena.list(path.segments).iter().map(|&at| symbols.intern(plain(src.text(arena.segments[at as usize].name)))).collect();
    let first = arena.list(path.segments).first().map(|&at| arena.segments[at as usize].name).unwrap_or(0);
    let dollar = first.checked_sub(1).and_then(|at| src.dollar(at));
    (segments, path.global, dollar)
}

impl<'a> Infer<'_, 'a> {
    pub(super) fn segments(&self, arena: &Body, path: PathId) -> (Vec<Sym>, bool, Option<u32>) {
        segments(&self.cx.names.symbols, self.src, arena, path)
    }

    /// The item a value path names, resolved from the body's module, then for a single name that a `macro` item wrote, from that macro's module, with a foreign declaration taken to the function the linker binds it to.
    pub fn value_item(&mut self, arena: &Body, path: PathId) -> Result<Def> {
        let found = self.value_named(arena, path)?;
        Ok(match found {
            Def::Item(item) => Def::Item(self.cx.linked(item)),
            other => other,
        })
    }

    fn value_named(&mut self, arena: &Body, path: PathId) -> Result<Def> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        let found = self.value_def(&segments, absolute, dollar);
        let [name] = segments[..] else { return found };
        match found {
            Err(error) => self.def_site(arena, path, name).and_then(|found| found.values).map(|binding| binding.def).ok_or(error),
            found => found,
        }
    }

    /// The bindings of the single name `name` at `path` in the module of the `macro` item that wrote it, when one did.
    fn def_site(&mut self, arena: &Body, path: PathId, name: Sym) -> Option<PerNs> {
        let first = arena.list(arena.path(path).segments)[0];
        let Some(Mark::Home { krate, file, item }) = self.src.mark(arena.segments[first as usize].name) else { return None };
        let module = self.cx.module_of(ItemRef { krate, file, item }).ok()?;
        let path = NamePath { segments: &[name], absolute: false, dollar: None };
        Some(self.cx.paths(module.krate).first(name, module, &path))
    }

    /// The item in the value namespace that the path `segments` names.
    pub(super) fn value_def(&mut self, segments: &[Sym], absolute: bool, dollar: Option<u32>) -> Result<Def> {
        let (segments, inside) = match self.local_def(&NamePath { segments, absolute, dollar }, Ns::Values) {
            Some(Ok(def)) => return Ok(def),
            Some(Err(skip)) => (&segments[skip..], false),
            None => (segments, true),
        };
        let absolute = absolute && inside;
        let Some((last, prefix)) = segments.split_last() else { bail!("empty path") };
        let found = if prefix.is_empty() {
            let local = if absolute || !inside { None } else { self.body_use(*last).values };
            local.or_else(|| self.cx.paths(self.home.krate).first(*last, self.home, &NamePath { segments, absolute, dollar }).values)
        } else {
            let path = NamePath { segments: prefix, absolute, dollar };
            let head = if inside { self.type_def(&path) } else { self.cx.paths(self.home.krate).resolve(&path, self.home) };
            head.map(|def| self.cx.path_member(self.home.krate, def, *last)).and_then(|entry| entry.values)
        };
        match found {
            Some(binding) => Ok(binding.def),
            None => bail!("cannot find value `{}`", self.spell(segments)),
        }
    }

    /// The type-namespace item that `path` names, a single segment first among the items that the body declares.
    pub(super) fn type_def(&mut self, path: &NamePath<'_>) -> Option<Def> {
        if !self.items.is_empty() {
            return self.type_def_here(path);
        }
        let here = self.anchor().and_then(|(root, from)| from.map(|at| root.inner(at)));
        let key = (self.home, self.root, here, path.segments.to_vec(), path.absolute, path.dollar);
        if let Some(&found) = self.cx.type_defs().borrow().get(&key) {
            return Some(found);
        }
        let found = self.type_def_here(path)?;
        self.cx.type_defs().borrow_mut().insert(key, found);
        Some(found)
    }

    fn type_def_here(&mut self, path: &NamePath<'_>) -> Option<Def> {
        match self.local_def(path, Ns::Types) {
            Some(Ok(def)) => Some(def),
            Some(Err(skip)) => self.cx.paths(self.home.krate).resolve(&outside(path, skip), self.home),
            None => self.resolve_local(path),
        }
    }

    /// The body item that a path written here starts from: the current item when the body declares it, otherwise the body itself.
    fn anchor(&self) -> Option<(ItemRef, Option<u32>)> {
        let here = self.frame_item.or_else(|| self.block.and_then(ImplRef::item));
        match here.and_then(ItemRef::outer) {
            Some((root, at)) => Some((root, Some(at))),
            None => self.root.map(|root| (root, None)),
        }
    }

    /// Where `path` leads among the items that the body declares: the item or variant it names, or `Err` with the count of segments that led out through `super`.
    fn local_def(&mut self, path: &NamePath<'_>, ns: Ns) -> Option<Result<Def, usize>> {
        let head = path.segments.first().filter(|_| !path.absolute).and_then(|first| self.block_item(self.cx.names.symbols.text(*first)));
        let found = match head {
            Some(item) => self.cx.local_descend(item, &path.segments[1..], ns)?,
            None => self.cx.local_path(self.anchor().filter(|(_, from)| from.is_some()), path, ns)?,
        };
        Some(match found {
            BodyPath::Item(item) => Ok(Def::Item(item)),
            BodyPath::Variant(item) => Ok(self.variant_def(item, *path.segments.last()?)?),
            BodyPath::Outside(skip) => Err(skip),
        })
    }

    /// The variant named `name` of the enum `item`.
    fn variant_def(&mut self, item: ItemRef, name: Sym) -> Option<Def> {
        let def = self.cx.adt_def(item).ok()?;
        let (src, text) = (self.cx.item(item).0, self.cx.names.symbols.text(name));
        def.variants.iter().find(|variant| plain(src.text(variant.name)) == text).map(|variant| Def::Variant(item, variant.index))
    }

    /// Resolves every segment of `path` in the type namespace, its first through the body's own `use` declarations when one binds it.
    pub(super) fn resolve_local(&mut self, path: &NamePath<'_>) -> Option<Def> {
        let (first, rest) = path.segments.split_first()?;
        let local = if path.absolute { None } else { self.body_use(*first).types };
        let paths = self.cx.paths(self.home.krate);
        let Some(binding) = local else { return paths.resolve(path, self.home) };
        rest.iter().try_fold(binding.def, |def, segment| self.cx.path_member(self.home.krate, def, *segment).types.map(|found| found.def))
    }

    /// The bindings that the `use` declarations inside the body give `word`, each namespace from the first one that binds it.
    fn body_use(&mut self, word: Sym) -> PerNs {
        self.body_imports(Some(word)).into_iter().fold(PerNs::default(), |found, entry| PerNs {
            types: found.types.or(entry.types),
            values: found.values.or(entry.values),
            macros: found.macros.or(entry.macros),
        })
    }

    /// The traits in scope through the body: those it declares and those its `use` declarations name.
    pub(super) fn body_traits(&mut self) -> Vec<ItemRef> {
        let mut traits = self.root.map(|root| self.cx.body_declared_traits(root)).unwrap_or_default();
        let found = self.body_imports(None).into_iter().filter_map(|entry| entry.types);
        let home = self.home;
        for binding in found.collect::<Vec<_>>() {
            match binding.def {
                Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Trait => traits.push(item),
                Def::Module(module) => self.cx.paths(home.krate).traits_of(module, &mut traits),
                _ => {}
            }
        }
        traits
    }

    fn body_imports(&mut self, word: Option<Sym>) -> Vec<PerNs> {
        let Some(root) = self.root.filter(|&root| self.cx.expansion(root).is_some_and(|found| !found.file.items.imports.is_empty())) else { return Vec::new() };
        let Ok(home) = self.cx.module_of(root) else { return Vec::new() };
        let here = self.instance.as_ref().map(|(item, _)| *item).filter(|item| item.outer().is_some()).map(|item| self.cx.item(item).1.start);
        self.cx.body_imports((root, home), word, here)
    }

    /// The item named `name` that a block around the use declares.
    fn block_item(&self, name: &str) -> Option<ItemRef> {
        self.items.iter().rev().find(|(bound, _)| *bound == name).map(|(_, item)| *item)
    }

    /// Resolves the value path expression `expr` to a local or an item, and returns its type.
    pub fn path_expr(&mut self, expr: syntax::ExprId, path: PathId) -> Result<Ty> {
        let (ty, res) = self.value_path(path)?;
        if let Res::Local(local) = res {
            self.note_read(expr, local);
        }
        if let Res::Ctor(item, variant, args) = &res
            && matches!(self.cx.types.kind(ty), Kind::FnPtr(..))
        {
            let reify = super::adt::ctor_fn(self.cx, (*item, *variant, args.clone()));
            self.out.coercions.insert(expr, super::coerce::Coercion::Reify(reify));
        }
        self.out.res.insert(expr, res);
        Ok(ty)
    }

    /// The type and resolution of the value path `path`: a local, a constructor, a function or a constant.
    pub(super) fn value_path(&mut self, path: PathId) -> Result<(Ty, Res)> {
        let body = self.body;
        let entry = body.path(path);
        let single = entry.qself.is_none() && !entry.global && entry.segments.len == 1;
        let first = body.segments[body.list(entry.segments)[0] as usize].name;
        if single && let Some(found) = self.bound_value(crate::names::plain(self.src.text(first)), self.src.scope(first))? {
            return Ok(found);
        }
        if let Some(ctor) = self.ctor(body, path, true)? {
            return self.ctor_value(ctor);
        }
        self.path_value(body, path)
    }

    /// The function or constant that the value path `path` of `body` names: an item, or an associated item of a type or trait, which a qualified path names first.
    pub(super) fn path_value(&mut self, body: &Body, path: PathId) -> Result<(Ty, Res)> {
        if body.path(path).qself.is_some()
            && let Some(found) = self.assoc_value(body, path)?
        {
            return Ok(found);
        }
        match self.value_item(body, path) {
            Ok(def) => self.item_value(body, path, def),
            Err(error) => self.assoc_value(body, path)?.ok_or(error),
        }
    }

    fn item_value(&mut self, body: &Body, path: PathId, def: Def) -> Result<(Ty, Res)> {
        match def {
            Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Fn => {
                let given = self.segment_args(body, path, body.path(path).segments.len as usize)?;
                self.fn_item(item, Vec::new(), given)
            }
            Def::Item(item) if matches!(self.cx.item(item).1.kind, ItemKind::Const | ItemKind::Static) => self.const_item(item, Vec::new()),
            other => bail!("unsupported value {other:?}"),
        }
    }

    /// The local that `name` written in hygiene scope `(id, parent)` finds: one its own expansion bound, else one of the scope its macro was defined in, else one the program bound; or a const parameter.
    fn bound_value(&mut self, name: &str, (id, parent): (u32, u32)) -> Result<Option<(Ty, Res)>> {
        let found = [id, parent, 0].into_iter().find_map(|scope| self.scopes.iter().rev().find(|(bound, _)| *bound == (name, scope)));
        if let Some(&(_, local)) = found {
            self.note_use(local);
            return Ok(Some((self.out.locals[local as usize].ty, Res::Local(local))));
        }
        Ok(self.const_param(name)?.map(|(ty, value)| (ty, Res::Const(value))))
    }

    pub(super) fn const_item(&mut self, item: ItemRef, outer: Vec<Ty>) -> Result<(Ty, Res)> {
        let ty = self.fn_types(item, &outer)?.1;
        Ok((ty, Res::Fn(item, outer)))
    }

    /// The function item `item` inside the type arguments `outer`, with its own type parameters set to `given`.
    pub(super) fn fn_item(&mut self, item: ItemRef, outer: Vec<Ty>, given: Vec<Ty>) -> Result<(Ty, Res)> {
        let mut substs = outer;
        substs.extend(self.own_args(item, given)?);
        self.bind_bounds(item, &substs)?;
        Ok((self.cx.types.intern(Kind::FnDef(item, substs.clone())), Res::Fn(item, substs)))
    }

    /// Runs `run` as if inside the type item `item`, with its type parameters set to `args`.
    fn within<T>(&mut self, item: ItemRef, params: &[Generic], args: &[Ty], run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let frame = self.item_frame(item, (params, args), (None, None))?;
        self.with_frame(frame, run)
    }

    /// The type and const arguments of segment `count` of `path`, skipping lifetimes.
    pub(super) fn segment_args(&mut self, arena: &Body, path: PathId, count: usize) -> Result<Vec<Ty>> {
        let Some(&last) = arena.list(arena.path(path).segments)[..count].last() else { return Ok(Vec::new()) };
        let list = match arena.segments[last as usize].args {
            None => return Ok(Vec::new()),
            Some(Args::Angle(list)) => list,
            Some(Args::Paren { inputs, output }) => {
                let (inputs, _) = self.paren_sig(arena, inputs, output)?;
                return Ok(vec![self.cx.types.intern(Kind::Tuple(inputs))]);
            }
            Some(Args::ReturnType) => bail!("return type notation is not supported yet"),
        };
        self.arg_list(arena, list)
    }

    /// The inputs and output of the `Fn(A, B) -> C` arguments `inputs` and `output`.
    pub(super) fn paren_sig(&mut self, arena: &Body, inputs: syntax::List, output: Option<TyId>) -> Result<(Vec<Ty>, Ty)> {
        let inputs = arena.list(inputs).iter().map(|&at| self.lower_ty(arena, TyId(at))).collect::<Result<Vec<_>>>()?;
        let output = match output {
            Some(ty) => self.lower_ty(arena, ty)?,
            None => self.cx.types.unit(),
        };
        Ok((inputs, output))
    }

    /// The associated types that the last segment of `path` binds, as `Name = T` or the output of `Fn(A) -> T`.
    pub(super) fn path_bindings(&mut self, arena: &Body, path: PathId) -> Result<Vec<(String, Ty)>> {
        let Some(&last) = arena.list(arena.path(path).segments).last() else { return Ok(Vec::new()) };
        match arena.segments[last as usize].args {
            Some(Args::Paren { inputs, output }) => Ok(vec![("Output".to_string(), self.paren_sig(arena, inputs, output)?.1)]),
            Some(Args::Angle(list)) => {
                let mut found = Vec::new();
                for &at in arena.list(list) {
                    let GenericArg::Binding { name, args: None, ty } = arena.args[at as usize] else { continue };
                    found.push((self.src.text(name).to_string(), self.lower_ty(arena, ty)?));
                }
                Ok(found)
            }
            Some(Args::ReturnType) => bail!("return type notation is not supported yet"),
            None => Ok(Vec::new()),
        }
    }

    /// The own type arguments of function `item`: the turbofish arguments `given`, then fresh variables for the parameters they leave out.
    pub(super) fn own_args(&mut self, item: ItemRef, mut given: Vec<Ty>) -> Result<Vec<Ty>> {
        let count = super::scope::type_params(&self.cx.signature(item)?.generics).count();
        if given.len() > count {
            bail!("`{}` takes {count} type arguments, {} given", self.cx.name(item), given.len());
        }
        given.extend((given.len()..count).map(|_| self.fresh(Kind::Var)));
        Ok(given)
    }

    /// The types and const values that the generic argument list `list` gives, skipping lifetimes.
    pub(super) fn arg_list(&mut self, arena: &Body, list: syntax::List) -> Result<Vec<Ty>> {
        let mut tys = Vec::new();
        for &at in arena.list(list) {
            match arena.args[at as usize] {
                GenericArg::Ty(ty) => tys.push(self.ty_or_const_arg(arena, ty)?),
                GenericArg::Lifetime(_) => {}
                GenericArg::Const(expr) => tys.push(self.const_arg(arena, expr)?),
                GenericArg::Binding { .. } | GenericArg::Constraint { .. } => {}
            }
        }
        Ok(tys)
    }

    /// The type arguments of the struct, enum or union `item`: `given`, then the defaults of the rest.
    pub(super) fn adt_args(&mut self, item: ItemRef, given: Vec<Ty>, fresh: bool) -> Result<Vec<Ty>> {
        let def = self.cx.adt_def(item)?;
        let (arena, parsed) = (&def.body, &def.generics);
        let params: Vec<&Generic> = parsed.params.iter().filter(|param| param.kind != GenericKind::Lifetime).collect();
        if given.len() > params.len() {
            bail!("`{}` takes {} type arguments, {} given", self.cx.name(item), params.len(), given.len());
        }
        let mut args = given;
        if fresh && args.is_empty() {
            return Ok(params.iter().map(|_| self.fresh(Kind::Var)).collect());
        }
        for param in &params[args.len()..] {
            let (GenericKind::Type, Some(default)) = (param.kind, param.default) else { bail!("`{}` needs {} type arguments, {} given", self.cx.name(item), params.len(), args.len()) };
            let known = args.clone();
            args.push(self.within(item, &parsed.params, &known, |infer| infer.lower_ty(arena, default))?);
        }
        Ok(args)
    }

    /// The type that the syntax type `ty` in `arena` names.
    pub fn lower_ty(&mut self, arena: &Body, ty: TyId) -> Result<Ty> {
        match *arena.ty(ty) {
            syntax::Ty::Path(path) => self.path_ty(arena, path),
            syntax::Ty::Paren(inner) => self.lower_ty(arena, inner),
            syntax::Ty::Infer => Ok(self.fresh(Kind::Var)),
            syntax::Ty::Macro { path, open } => self.macro_ty(arena, path, open),
            syntax::Ty::ImplTrait(at, _) => Ok(self.impl_param(at)),
            _ => {
                let kind = self.lower_kind(arena, ty)?;
                Ok(self.cx.types.intern(kind))
            }
        }
    }

    fn lower_kind(&mut self, arena: &Body, ty: TyId) -> Result<Kind> {
        match *arena.ty(ty) {
            syntax::Ty::Ref { mutable, ty, .. } => self.lower_ty(arena, ty).map(|inner| Kind::Ref(mutability(mutable), inner)),
            syntax::Ty::Ptr { mutable, ty } => self.lower_ty(arena, ty).map(|inner| Kind::Ptr(mutability(mutable), inner)),
            syntax::Ty::Slice(ty) => self.lower_ty(arena, ty).map(Kind::Slice),
            syntax::Ty::Array(ty, len) => self.array_ty(arena, ty, len),
            syntax::Ty::Tuple(list) => self.lower_list(arena, list).map(Kind::Tuple),
            syntax::Ty::Never => Ok(Kind::Never),
            syntax::Ty::DynTrait(bounds) => self.object_ty(arena, bounds),
            syntax::Ty::Fn { params, ret, variadic } => self.fn_ptr_ty(arena, (params, ret), variadic),
            other => bail!("unsupported type {other:?}"),
        }
    }

    fn fn_ptr_ty(&mut self, arena: &Body, (params, ret): (syntax::List, Option<TyId>), variadic: bool) -> Result<Kind> {
        if variadic {
            bail!("variadic function pointer types are not supported yet");
        }
        let params = self.lower_list(arena, params)?;
        let ret = match ret {
            Some(ret) => self.lower_ty(arena, ret)?,
            None => self.cx.types.unit(),
        };
        Ok(Kind::FnPtr(params, ret))
    }

    fn object_ty(&mut self, arena: &Body, bounds: syntax::List) -> Result<Kind> {
        let (mut principal, mut autos) = (None, Vec::new());
        for &bound in arena.list(bounds) {
            let syntax::Bound::Trait { path, maybe: false, .. } = arena.bounds[bound as usize] else { continue };
            let Some(of) = self.trait_named(arena, path, self.home) else { bail!("a trait object bound that names no trait") };
            if self.cx.is_auto_trait(of) {
                autos.push(of);
            } else if principal.replace((of, path)).is_some() {
                bail!("a trait object with two traits that are not auto traits");
            }
        }
        autos.sort_by_key(|item| (item.krate, item.file, item.item));
        autos.dedup();
        let Some((of, path)) = principal else {
            if autos.is_empty() {
                bail!("a trait object with no trait");
            }
            let first = autos.remove(0);
            return Ok(Kind::Dyn(first, Vec::new(), Vec::new(), autos));
        };
        let count = arena.list(arena.path(path).segments).len();
        let given = self.segment_args(arena, path, count)?;
        let wanted = super::scope::type_params(&self.cx.trait_generics(of)?.1.params).count();
        if given.len() != wanted {
            bail!("a trait object of `{}` that leaves trait arguments to their defaults", self.cx.name(of));
        }
        let object = self.fresh(Kind::Var);
        let mut bound = Vec::new();
        for (name, ty) in self.path_bindings(arena, path)? {
            let Some((owner, _)) = self.assoc_owner((of, given.clone()), object, &name, 0)? else { bail!("`{}` has no associated type `{name}`", self.cx.name(of)) };
            let Some(member) = self.cx.trait_member(owner, &name) else { bail!("`{}` has no member `{name}`", self.cx.name(owner)) };
            bound.push((member, ty));
        }
        bound.sort_by_key(|(member, _)| (member.krate, member.file, member.item));
        let (members, tys): (Vec<ItemRef>, Vec<Ty>) = bound.into_iter().unzip();
        Ok(Kind::Dyn(of, given.into_iter().chain(tys).collect(), members, autos))
    }

    /// The type that the associated type `name` of impl `block` is, read in the current frame over the alias's own source, when the impl has one.
    pub(super) fn impl_alias(&mut self, block: ImplRef, (name, own): (&str, &[Ty])) -> Result<Option<Ty>> {
        let Some(member) = self.cx.member(block, name) else { return Ok(None) };
        if self.cx.item(member).1.kind != ItemKind::Type {
            return Ok(None);
        }
        let (params, arena, ty) = self.cx.alias_parts(member)?;
        let src = &self.cx.file(member).source;
        let outer = self.generics.len();
        self.generics.extend(params.iter().map(|param| src.text(param.name)).zip(own.iter().copied()));
        let found = self.in_source(src, |infer| infer.lower_ty(&arena, ty));
        self.generics.truncate(outer);
        found.map(Some)
    }

    /// The type that the type alias `item` names at the type arguments `given`, then the defaults of the rest, or fresh ones in a value path that gives none.
    pub(super) fn alias(&mut self, item: ItemRef, given: Vec<Ty>, fresh: bool) -> Result<Ty> {
        let (params, arena, ty) = self.cx.alias_parts(item)?;
        if let Some(definer) = self.opaque_alias(item, &params, &arena) {
            return super::opaque::revealed(self.cx, definer, &[]);
        }
        let declared = self.cx.alias_generics(item)?;
        let file = self.cx.file(item);
        let mut args = given;
        let infer_all = fresh && args.is_empty();
        for param in &params[args.len().min(params.len())..] {
            let arg = match param.default.filter(|_| !infer_all) {
                Some(default) => {
                    let known = args.clone();
                    self.within(item, &params[..known.len()], &known, |infer| infer.lower_ty(&declared.0, default))?
                }
                None => self.fresh(Kind::Var),
            };
            args.push(arg);
        }
        let names: Vec<&str> = params.iter().map(|generic| file.source.text(generic.name)).collect();
        let generics = names.into_iter().zip(args).collect();
        let frame = super::scope::Frame { src: &file.source, home: self.cx.module_of(item)?, generics, self_ty: None, block: None, item: Some(item), root: item.declaring_body() };
        self.with_frame(frame, |infer| infer.lower_ty(&arena, ty))
    }

    /// The function that defines the hidden types of the alias `item` when its body holds `impl Trait`, it has no type parameters, and this body is not that function, which binds them itself.
    fn opaque_alias(&self, item: ItemRef, params: &[crate::body::sig::Generic], arena: &Body) -> Option<ItemRef> {
        if !params.is_empty() || item.outer().is_some() || !arena.tys.iter().any(|ty| matches!(ty, syntax::Ty::ImplTrait(..))) {
            return None;
        }
        let definer = super::opaque::opaque_definer(self.cx, item)?;
        self.instance.as_ref().is_none_or(|(at, _)| *at != definer).then_some(definer)
    }

    fn impl_param(&mut self, at: u32) -> Ty {
        let key = self.src.text(at);
        match self.generics.iter().find(|(name, _)| super::scope::same_param(name, key)) {
            Some(&(_, ty)) => ty,
            None => {
                let hidden = self.fresh(Kind::Var);
                self.hidden.push((at, hidden));
                hidden
            }
        }
    }

    fn array_ty(&mut self, arena: &Body, ty: TyId, len: syntax::ExprId) -> Result<Kind> {
        Ok(Kind::Array(self.lower_ty(arena, ty)?, self.const_arg(arena, len)?))
    }

    fn lower_list(&mut self, arena: &Body, list: syntax::List) -> Result<Vec<Ty>> {
        arena.list(list).iter().map(|&item| self.lower_ty(arena, TyId(item))).collect()
    }


    fn path_ty(&mut self, arena: &Body, path: PathId) -> Result<Ty> {
        if let Some((qself, Some(trait_path))) = arena.path(path).qself {
            let Some(&last) = arena.list(arena.path(path).segments).last() else { bail!("a projection with no name") };
            return self.projection(arena, (qself, trait_path), last);
        }
        let count = arena.path(path).segments.len as usize;
        self.named_ty(arena, path, count, false)
    }

    /// The type that the first `count` segments of `path` name, with fresh arguments for an ADT given none when `fresh` holds.
    pub(super) fn named_ty(&mut self, arena: &Body, path: PathId, count: usize, fresh: bool) -> Result<Ty> {
        let (mut segments, absolute, dollar) = self.segments(arena, path);
        segments.truncate(count);
        if let Some(ty) = self.scoped_ty(&segments)? {
            return Ok(ty);
        }
        let def = self.type_def(&NamePath { segments: &segments, absolute, dollar });
        let def = match (def, &segments[..]) {
            (Some(Def::Module(_)), [name]) => crate::names::Symbols::primitive(*name).map(Def::Primitive).or(def),
            (None, [name]) => self.def_site(arena, path, *name).and_then(|found| found.types).map(|binding| binding.def),
            _ => def,
        };
        match def {
            Some(Def::Primitive(primitive)) => Ok(self.cx.types.prim(primitive)),
            Some(Def::Item(item)) if matches!(self.cx.item(item).1.kind, ItemKind::Type | ItemKind::Struct | ItemKind::Enum | ItemKind::Union) => {
                let given = self.segment_args(arena, path, count)?;
                self.item_ty(item, given, fresh)
            }
            other => bail!("cannot lower type `{}` ({other:?})", self.spell(&segments)),
        }
    }

    fn item_ty(&mut self, item: ItemRef, given: Vec<Ty>, fresh: bool) -> Result<Ty> {
        if self.cx.item(item).1.kind == ItemKind::Type {
            return self.alias(item, given, fresh);
        }
        let args = self.adt_args(item, given, fresh)?;
        Ok(self.cx.types.intern(Kind::Adt(item, args)))
    }

    /// The type that `segments` names through the frame: `Self`, a type parameter, or an associated type `Param::Name` of one.
    fn scoped_ty(&mut self, segments: &[Sym]) -> Result<Option<Ty>> {
        let symbols = &self.cx.names.symbols;
        match *segments {
            [one] if symbols.text(one) == "Self" && self.self_ty.is_some() => Ok(self.self_ty),
            [one] => Ok(self.generics.iter().find(|(generic, _)| *generic == symbols.text(one)).map(|&(_, ty)| ty)),
            [param, name] if symbols.text(param) == "Self" || self.generics.iter().any(|(generic, _)| *generic == symbols.text(param)) => {
                let (param, name) = (symbols.text(param).to_string(), symbols.text(name).to_string());
                self.param_projection(&param, &name)
            }
            _ => Ok(None),
        }
    }

    /// The type `<qself as Trait>::Name` names: the impl's associated type, or the pointer metadata the compiler provides.
    /// The associated type that segment `segment` names on `<qself as trait_path>`.
    pub(super) fn projection(&mut self, arena: &Body, (qself, trait_path): (TyId, PathId), segment: u32) -> Result<Ty> {
        let ty = self.lower_ty(arena, qself)?;
        let (segments, absolute, dollar) = self.segments(arena, trait_path);
        let found = self.trait_at(&NamePath { segments: &segments, absolute, dollar }, self.home);
        let name = plain(self.src.text(arena.segments[segment as usize].name));
        let Some(item) = found else { bail!("projection through `{}`, which is not a trait", self.spell(&segments)) };
        let own = match arena.segments[segment as usize].args {
            Some(Args::Angle(list)) => self.arg_list(arena, list)?,
            _ => Vec::new(),
        };
        if self.cx.lang(item) == Some("pointee_trait") && name == "Metadata" {
            return self.metadata_of(ty);
        }
        let args = self.bound_args(item, arena, trait_path, ty)?;
        self.project_own(item, ty, args, (name, &own))
    }

    /// The pointer metadata of `ty`: a length for slices and `str`, nothing for sized types, or a variable until `ty` is known.
    pub fn metadata_of(&mut self, ty: Ty) -> Result<Ty> {
        match self.cx.types.kind(self.resolve(ty)) {
            Kind::Slice(_) | Kind::Prim(Primitive::Str) => Ok(self.cx.types.prim(Primitive::Usize)),
            Kind::Dyn(..) => {
                let Some(metadata) = self.cx.lang_item("dyn_metadata") else { bail!("core has no `DynMetadata`") };
                Ok(self.cx.types.intern(Kind::Adt(metadata, vec![ty])))
            }
            Kind::Var(_) => {
                let result = self.fresh(Kind::Var);
                self.metadata.push((ty, result));
                Ok(result)
            }
            _ => Ok(self.cx.types.unit()),
        }
    }

    pub(super) fn spell(&self, segments: &[Sym]) -> String {
        segments.iter().map(|sym| self.cx.names.symbols.text(*sym)).collect::<Vec<_>>().join("::")
    }
}

fn mutability(mutable: bool) -> Mut {
    if mutable { Mut::Mut } else { Mut::Not }
}

/// The value of an integer literal and its suffix type, if it has one.
pub fn parse_int(text: &str) -> Result<(u128, Option<Primitive>)> {
    let digits: String = text.chars().filter(|ch| *ch != '_').collect();
    let (radix, rest) = match digits.get(..2) {
        Some("0x") => (16, &digits[2..]),
        Some("0o") => (8, &digits[2..]),
        Some("0b") => (2, &digits[2..]),
        _ => (10, &digits[..]),
    };
    let split = rest.find(|ch: char| !ch.is_digit(radix)).unwrap_or(rest.len());
    let (number, suffix) = rest.split_at(split);
    let value = u128::from_str_radix(number, radix)?;
    let suffix = if suffix.is_empty() { None } else { Some(Primitive::named(suffix).ok_or_else(|| anyhow::anyhow!("bad suffix {suffix}"))?) };
    Ok((value, suffix))
}
