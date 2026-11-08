use anyhow::{Result, bail};

use crate::body::sig::{Generic, GenericKind};
use crate::body::{self as syntax, Args, Body, GenericArg, Lit, PathId, TyId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, Primitive, Sym};
use crate::outline::item::ItemKind;

use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Res};

impl<'a> Infer<'_, 'a> {
    fn segments(&self, arena: &Body, path: PathId) -> (Vec<Sym>, bool, Option<u32>) {
        let path = arena.path(path);
        let src = self.src;
        let symbols = &self.cx.names.symbols;
        let segments: Vec<Sym> = arena.list(path.segments).iter().map(|&at| symbols.intern(crate::names::plain(src.text(arena.segments[at as usize].name)))).collect();
        let first = arena.list(path.segments).first().map(|&at| arena.segments[at as usize].name).unwrap_or(0);
        let dollar = first.checked_sub(1).and_then(|at| src.dollar(at));
        (segments, path.global, dollar)
    }

    /// The item a value path names, resolved from the body's module.
    pub fn value_item(&mut self, arena: &Body, path: PathId) -> Result<Def> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        let paths = self.cx.paths(self.home.krate);
        let Some((last, prefix)) = segments.split_last() else { bail!("empty path") };
        let found = if prefix.is_empty() {
            paths.first(*last, self.home, &NamePath { segments: &segments, absolute, dollar }).values
        } else {
            let head = NamePath { segments: prefix, absolute, dollar };
            paths.resolve(&head, self.home).map(|def| paths.member(def, *last)).and_then(|entry| entry.values)
        };
        match found {
            Some(binding) => Ok(binding.def),
            None => bail!("cannot find value `{}`", self.spell(&segments)),
        }
    }

    /// Resolves the value path expression `expr` to a local or an item, and returns its type.
    pub fn path_expr(&mut self, expr: syntax::ExprId, path: PathId) -> Result<Ty> {
        let body = self.body;
        let entry = body.path(path);
        let single = entry.qself.is_none() && !entry.global && entry.segments.len == 1;
        if single {
            let name = crate::names::plain(self.src.text(body.segments[body.list(entry.segments)[0] as usize].name));
            if let Some(&(_, local)) = self.scopes.iter().rev().find(|(bound, _)| *bound == name) {
                self.out.res.insert(expr, Res::Local(local));
                return Ok(self.out.locals[local as usize].ty);
            }
        }
        match self.value_item(body, path)? {
            Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Fn => self.fn_item(expr, item),
            other => bail!("unsupported value {other:?}"),
        }
    }

    fn fn_item(&mut self, expr: syntax::ExprId, item: ItemRef) -> Result<Ty> {
        let sig = self.cx.signature(item)?;
        let count = sig.generics.iter().filter(|generic| generic.kind != GenericKind::Lifetime).count();
        let substs: Vec<Ty> = (0..count).map(|_| self.fresh(Kind::Var)).collect();
        self.out.res.insert(expr, Res::Fn(item, substs.clone()));
        Ok(self.cx.types.intern(Kind::FnDef(item, substs)))
    }

    /// The parameter and return types of the function `item` at `substs`.
    pub fn fn_types(&mut self, item: ItemRef, substs: &[Ty]) -> Result<(Vec<Ty>, Ty)> {
        let sig = self.cx.signature(item)?;
        self.within(item, &sig.generics, substs, |infer| infer.sig_types(&sig))
    }

    /// Runs `run` as if inside `item`, with its non-lifetime generic parameters set to `args`.
    fn within<T>(&mut self, item: ItemRef, params: &[Generic], args: &[Ty], run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let file = self.cx.file(item);
        let home = self.cx.module_of(item)?;
        let names = params.iter().filter(|generic| generic.kind != GenericKind::Lifetime).map(|generic| file.source.text(generic.name));
        let generics: Vec<(&str, Ty)> = names.zip(args.iter().copied()).collect();
        let saved = (std::mem::replace(&mut self.file, file), std::mem::replace(&mut self.src, &file.source), std::mem::replace(&mut self.home, home), std::mem::replace(&mut self.generics, generics));
        let result = run(self);
        (self.file, self.src, self.home, self.generics) = saved;
        result
    }

    fn segment_args(&mut self, arena: &Body, path: PathId) -> Result<Vec<Ty>> {
        let Some(&last) = arena.list(arena.path(path).segments).last() else { return Ok(Vec::new()) };
        let list = match arena.segments[last as usize].args {
            None => return Ok(Vec::new()),
            Some(Args::Angle(list)) => list,
            Some(other) => bail!("unsupported generic arguments {other:?}"),
        };
        let mut tys = Vec::new();
        for &at in arena.list(list) {
            match arena.args[at as usize] {
                GenericArg::Ty(ty) => tys.push(self.lower_ty(arena, ty)?),
                GenericArg::Lifetime(_) => {}
                other => bail!("unsupported generic argument {other:?}"),
            }
        }
        Ok(tys)
    }

    /// The type arguments of the struct, enum or union `item`: `given`, then the defaults of the rest.
    fn adt_args(&mut self, item: ItemRef, given: Vec<Ty>) -> Result<Vec<Ty>> {
        let generics = self.cx.adt_generics(item)?;
        let (arena, parsed) = (&generics.0, &generics.1);
        let params: Vec<&Generic> = parsed.params.iter().filter(|param| param.kind != GenericKind::Lifetime).collect();
        if given.len() > params.len() {
            bail!("`{}` takes {} type arguments, {} given", self.cx.name(item), params.len(), given.len());
        }
        let mut args = given;
        for param in &params[args.len()..] {
            let (GenericKind::Type, Some(default)) = (param.kind, param.default) else { bail!("`{}` needs {} type arguments, {} given", self.cx.name(item), params.len(), args.len()) };
            let known = args.clone();
            args.push(self.within(item, &parsed.params, &known, |infer| infer.lower_ty(arena, default))?);
        }
        Ok(args)
    }

    fn sig_types(&mut self, sig: &crate::body::sig::Signature) -> Result<(Vec<Ty>, Ty)> {
        let params = sig.params.iter().map(|&(_, ty)| self.lower_ty(&sig.body, ty)).collect::<Result<Vec<_>>>()?;
        let ret = match sig.ret {
            Some(ty) => self.lower_ty(&sig.body, ty)?,
            None => self.cx.types.unit(),
        };
        Ok((params, ret))
    }

    /// The type that the syntax type `ty` in `arena` names.
    pub fn lower_ty(&mut self, arena: &Body, ty: TyId) -> Result<Ty> {
        let kind = match *arena.ty(ty) {
            syntax::Ty::Path(path) => return self.path_ty(arena, path),
            syntax::Ty::Paren(inner) => return self.lower_ty(arena, inner),
            syntax::Ty::Infer => return Ok(self.fresh(Kind::Var)),
            syntax::Ty::Ref { mutable, ty, .. } => self.lower_ty(arena, ty).map(|inner| Kind::Ref(mutability(mutable), inner)),
            syntax::Ty::Ptr { mutable, ty } => self.lower_ty(arena, ty).map(|inner| Kind::Ptr(mutability(mutable), inner)),
            syntax::Ty::Slice(ty) => self.lower_ty(arena, ty).map(Kind::Slice),
            syntax::Ty::Array(ty, len) => self.array_ty(arena, ty, len),
            syntax::Ty::Tuple(list) => self.lower_list(arena, list).map(Kind::Tuple),
            syntax::Ty::Never => Ok(Kind::Never),
            other => bail!("unsupported type {other:?}"),
        };
        Ok(self.cx.types.intern(kind?))
    }

    fn array_ty(&mut self, arena: &Body, ty: TyId, len: syntax::ExprId) -> Result<Kind> {
        Ok(Kind::Array(self.lower_ty(arena, ty)?, self.const_len(arena, len)?))
    }

    fn lower_list(&mut self, arena: &Body, list: syntax::List) -> Result<Vec<Ty>> {
        arena.list(list).iter().map(|&item| self.lower_ty(arena, TyId(item))).collect()
    }

    fn const_len(&self, arena: &Body, len: syntax::ExprId) -> Result<u64> {
        match *arena.expr(len) {
            syntax::Expr::Lit(Lit::Int, at) => parse_int(self.src.text(at)).map(|(value, _)| value as u64),
            _ => bail!("array lengths other than literals are not supported yet"),
        }
    }

    fn path_ty(&mut self, arena: &Body, path: PathId) -> Result<Ty> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        if let [name] = segments[..] {
            let text = self.cx.names.symbols.text(name);
            if let Some(&(_, ty)) = self.generics.iter().find(|(generic, _)| *generic == text) {
                return Ok(ty);
            }
        }
        let paths = self.cx.paths(self.home.krate);
        let def = paths.resolve(&NamePath { segments: &segments, absolute, dollar }, self.home);
        match def {
            Some(Def::Primitive(primitive)) => Ok(self.cx.types.prim(primitive)),
            Some(Def::Item(item)) if matches!(self.cx.item(item).1.kind, ItemKind::Struct | ItemKind::Enum | ItemKind::Union) => {
                let given = self.segment_args(arena, path)?;
                let args = self.adt_args(item, given)?;
                Ok(self.cx.types.intern(Kind::Adt(item, args)))
            }
            other => bail!("cannot lower type `{}` ({other:?})", self.spell(&segments)),
        }
    }

    fn spell(&self, segments: &[Sym]) -> String {
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
