use anyhow::{Result, bail};

use crate::body::{Body, Expr, List, Lit, Pat, PatId, PathId};
use crate::names::plain;
use crate::outline::item::ItemKind;

use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Res};

/// How the bindings of a pattern take their values: by value, or by reference after the pattern matched through one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Value,
    Ref(Mut),
}

/// The field that item `at` of a positional pattern with `count` items and a `..` at `rest` matches in a value of `fields` fields, or none for the `..`.
pub fn position(at: usize, count: usize, rest: Option<usize>, fields: usize) -> Option<usize> {
    match rest {
        Some(split) if at == split => None,
        Some(split) if at > split => Some(fields - (count - at)),
        _ => Some(at),
    }
}

/// The items of pattern list `list` in `body`, and where its one `..` is.
pub fn rest_split(body: &Body, list: List) -> Result<(Vec<PatId>, Option<usize>)> {
    let pats: Vec<PatId> = body.list(list).iter().map(|&at| PatId(at)).collect();
    let is_rest = |pat: PatId| match *body.pat(pat) {
        Pat::Rest => true,
        Pat::Bind { sub: Some(sub), .. } => matches!(body.pat(sub), Pat::Rest),
        _ => false,
    };
    let rests: Vec<usize> = pats.iter().enumerate().filter(|(_, pat)| is_rest(**pat)).map(|(at, _)| at).collect();
    match rests[..] {
        [] => Ok((pats, None)),
        [at] => Ok((pats, Some(at))),
        _ => bail!("more than one `..` in a pattern"),
    }
}

impl Infer<'_, '_> {
    /// Checks pattern `pat` against a value of type `ty` and binds its names.
    pub fn bind_pat(&mut self, pat: PatId, ty: Ty) -> Result<()> {
        self.pat(pat, ty, Mode::Value)
    }

    fn pat(&mut self, pat: PatId, ty: Ty, mode: Mode) -> Result<()> {
        let (ty, mode) = self.peel(pat, ty, mode);
        self.out.pats.insert(pat, ty);
        match *self.body.pat(pat) {
            Pat::Bind { name, mutable, by_ref, sub } => self.bind_name(pat, (name, mutable, by_ref), sub, (ty, mode)),
            Pat::Wild | Pat::Rest => Ok(()),
            Pat::Paren(inner) => self.pat(inner, ty, mode),
            Pat::Lit(expr) => self.value_pat(expr, ty),
            Pat::Range { from, to, .. } => from.into_iter().chain(to).try_for_each(|end| self.value_pat(end, ty)),
            Pat::Tuple(list) => self.tuple_pat(list, ty, mode),
            Pat::Slice(list) => self.slice_pat(list, ty, mode),
            Pat::Ref { mutable, pat: inner } => self.ref_pat(inner, mutable, ty),
            Pat::Path(path) => self.path_pat(pat, path, (ty, mode)),
            Pat::TupleStruct(path, list) => self.variant_pat(pat, path, ty, (list, mode)),
            Pat::Struct { path, fields, .. } => self.struct_pat(pat, path, ty, (fields, mode)),
            Pat::Or(list) => self.or_pat(pat, list, ty, mode),
            ref other => bail!("unsupported pattern {other:?}"),
        }
    }

    fn bind_name(&mut self, pat: PatId, (name, mutable, by_ref): (u32, bool, bool), sub: Option<PatId>, (ty, mode): (Ty, Mode)) -> Result<()> {
        if !mutable
            && !by_ref
            && sub.is_none()
            && let Some((found, res)) = self.named_pat(name)?
        {
            return self.named_value(pat, (found, res), ty, mode);
        }
        let explicit = if mutable { Mut::Mut } else { Mut::Not };
        let reference = match (by_ref, mode) {
            (true, _) => Some(explicit),
            (false, Mode::Ref(_)) if mutable => None,
            (false, Mode::Ref(mutability)) => Some(mutability),
            (false, Mode::Value) => None,
        };
        let local_ty = match reference {
            Some(mutability) => {
                self.out.ref_binds.insert(pat, mutability);
                self.cx.types.intern(Kind::Ref(mutability, ty))
            }
            None => ty,
        };
        let local = self.new_local(local_ty);
        self.out.binds.insert(pat, local);
        self.scope_name(name, local);
        sub.map_or(Ok(()), |sub| self.pat(sub, ty, mode))
    }

    /// Puts `local` in scope under the name at token `name`, without its `r#` prefix.
    pub(super) fn scope_name(&mut self, name: u32, local: u32) {
        self.scopes.push(((plain(self.src.text(name)), self.src.scope(name).0), local));
    }

    fn value_pat(&mut self, expr: crate::body::ExprId, ty: Ty) -> Result<()> {
        let found = self.expr(expr, Some(ty))?;
        self.coerce_expr(expr, found, ty)
    }

    fn ref_pat(&mut self, inner: PatId, mutable: bool, ty: Ty) -> Result<()> {
        let pointee = self.fresh(Kind::Var);
        let reference = self.cx.types.intern(Kind::Ref(if mutable { Mut::Mut } else { Mut::Not }, pointee));
        self.unify(ty, reference)?;
        self.pat(inner, pointee, Mode::Value)
    }

    fn peel(&mut self, pat: PatId, ty: Ty, mode: Mode) -> (Ty, Mode) {
        let keeps = match *self.body.pat(pat) {
            Pat::Bind { .. } | Pat::Path(_) | Pat::Wild | Pat::Rest | Pat::Ref { .. } | Pat::Paren(_) | Pat::Or(_) => true,
            Pat::Lit(expr) => matches!(self.body.expr(expr), Expr::Lit(Lit::Str | Lit::ByteStr | Lit::CStr, _)),
            _ => false,
        };
        if keeps {
            return (ty, mode);
        }
        self.peel_refs(pat, ty, mode)
    }

    fn peel_refs(&mut self, pat: PatId, mut ty: Ty, mut mode: Mode) -> (Ty, Mode) {
        let mut derefs = 0;
        while let Kind::Ref(mutability, inner) = *self.cx.types.kind(self.resolve(ty)) {
            mode = match mode {
                Mode::Ref(Mut::Not) => Mode::Ref(Mut::Not),
                _ => Mode::Ref(mutability),
            };
            ty = inner;
            derefs += 1;
        }
        if derefs > 0 {
            self.out.pat_derefs.insert(pat, derefs);
        }
        (self.resolve(ty), mode)
    }

    fn tuple_pat(&mut self, list: List, ty: Ty, mode: Mode) -> Result<()> {
        let (pats, rest) = rest_split(self.body, list)?;
        let items = match (self.cx.types.kind(ty).clone(), rest) {
            (Kind::Tuple(items), _) => self.cx.types.substs(items).to_vec(),
            (_, None) => {
                let items: Vec<Ty> = (0..pats.len()).map(|_| self.fresh(Kind::Var)).collect();
                let tuple = self.cx.types.tuple(&items);
                self.unify(ty, tuple)?;
                items
            }
            _ => bail!("the type of a tuple pattern with `..` must be known here"),
        };
        self.positional(&pats, rest, &items, mode)
    }

    fn slice_pat(&mut self, list: List, ty: Ty, mode: Mode) -> Result<()> {
        let (pats, rest) = rest_split(self.body, list)?;
        let given = pats.len() - usize::from(rest.is_some());
        let element = self.slice_element(ty, given, rest.is_some())?;
        for (at, &pat) in pats.iter().enumerate() {
            let item = if Some(at) == rest { self.rest_ty(ty, element, given) } else { element };
            self.pat(pat, item, mode)?;
        }
        Ok(())
    }

    fn rest_ty(&mut self, ty: Ty, element: Ty, given: usize) -> Ty {
        match *self.cx.types.kind(ty) {
            Kind::Array(_, len) => {
                let left = self.cx.types.len(len).unwrap_or(0) - given as u64;
                self.cx.types.array(element, left)
            }
            _ => self.cx.types.intern(Kind::Slice(element)),
        }
    }

    fn slice_element(&mut self, ty: Ty, given: usize, rest: bool) -> Result<Ty> {
        let (element, len) = match *self.cx.types.kind(ty) {
            Kind::Slice(element) => return Ok(element),
            Kind::Array(element, len) => (element, len),
            _ => bail!("slice pattern on {}", self.show(ty)),
        };
        let resolved_len = self.resolve(len);
        if matches!(self.cx.types.kind(resolved_len), Kind::Param(_) | Kind::Proj(_)) {
            self.poly_fallback();
        }
        match self.cx.types.len(resolved_len) {
            Some(len) if given as u64 > len || (!rest && given as u64 != len) => bail!("pattern has {given} elements, the array has {len}"),
            Some(_) => Ok(element),
            None if !rest => {
                let known = self.cx.types.intern(Kind::Value(given as u128));
                self.unify(len, known).map(|()| element)
            }
            None => bail!("a slice pattern with `..` on an array of unknown length"),
        }
    }

    fn positional(&mut self, pats: &[PatId], rest: Option<usize>, items: &[Ty], mode: Mode) -> Result<()> {
        let given = pats.len() - usize::from(rest.is_some());
        if given > items.len() || (rest.is_none() && given != items.len()) {
            bail!("pattern has {given} fields, the type has {}", items.len());
        }
        for (at, &pat) in pats.iter().enumerate() {
            let Some(index) = position(at, pats.len(), rest, items.len()) else {
                continue;
            };
            self.pat(pat, items[index], mode)?;
        }
        Ok(())
    }

    fn path_pat(&mut self, pat: PatId, path: PathId, (ty, mode): (Ty, Mode)) -> Result<()> {
        let (found, res) = self.value_path(path)?;
        if !matches!(res, Res::Ctor(..)) && !self.is_const(&res) {
            bail!("pattern path is not a unit constructor or a constant");
        }
        self.named_value(pat, (found, res), ty, mode)
    }

    fn is_const(&self, res: &Res) -> bool {
        matches!(res, Res::Fn(item, _) if self.cx.item(*item).1.kind == ItemKind::Const)
    }

    fn named_value(&mut self, pat: PatId, (found, res): (Ty, Res), ty: Ty, mode: Mode) -> Result<()> {
        let peeled = if self.is_const(&res) { ty } else { self.peel_refs(pat, ty, mode).0 };
        self.out.pats.insert(pat, peeled);
        self.out.pat_res.insert(pat, res);
        self.unify(peeled, found)
    }

    fn variant_pat(&mut self, pat: PatId, path: PathId, ty: Ty, (list, mode): (List, Mode)) -> Result<()> {
        let body = self.body;
        let Some((item, variant, args)) = self.ctor(body, path, true)? else { bail!("pattern path is not a tuple constructor") };
        let adt = self.cx.types.intern(Kind::Adt(item, args));
        self.unify(ty, adt)?;
        let slice = self.cx.types.substs(args).to_vec();
        let fields = self.adt_fields(item, &slice)?[variant as usize].clone();
        self.out.pat_res.insert(pat, Res::Ctor(item, variant, args));
        let (pats, rest) = rest_split(self.body, list)?;
        self.positional(&pats, rest, &fields, mode)
    }

    fn struct_pat(&mut self, pat: PatId, path: PathId, ty: Ty, (list, mode): (List, Mode)) -> Result<()> {
        let body = self.body;
        let Some((item, variant, args)) = self.ctor(body, path, false)? else { bail!("struct pattern of something that is not a struct") };
        let adt = self.cx.types.intern(Kind::Adt(item, args));
        self.unify(ty, adt)?;
        let slice = self.cx.types.substs(args).to_vec();
        let fields = self.adt_fields(item, &slice)?[variant as usize].clone();
        self.out.pat_res.insert(pat, Res::Ctor(item, variant, args));
        for &at in body.list(list) {
            let field = body.field_pats[at as usize];
            let name = plain(self.src.text(field.name));
            let Some(index) = self.field_index(item, variant, name)? else { bail!("no field `{name}` on {}", self.cx.name(item)) };
            self.out.field_pats.insert(field.pat, index);
            self.pat(field.pat, fields[index as usize], mode)?;
        }
        Ok(())
    }

    fn or_pat(&mut self, pat: PatId, list: List, ty: Ty, mode: Mode) -> Result<()> {
        let depth = self.scopes.len();
        let mut first: Option<Vec<(super::Name, u32)>> = None;
        for &alt in self.body.list(list) {
            self.pat(PatId(alt), ty, mode)?;
            let bound = self.scopes.split_off(depth);
            match &first {
                None => first = Some(bound),
                Some(names) => self.merge_alt(names, bound)?,
            }
        }
        let names = first.unwrap_or_default();
        if !names.is_empty() {
            self.out.or_binds.insert(pat);
        }
        self.scopes.extend(names);
        Ok(())
    }

    fn merge_alt(&mut self, names: &[(super::Name, u32)], bound: Vec<(super::Name, u32)>) -> Result<()> {
        if names.len() != bound.len() {
            bail!("the alternatives of an or-pattern bind different names");
        }
        for (name, local) in bound {
            let Some(&(_, kept)) = names.iter().find(|(other, _)| *other == name) else { bail!("`{}` is not bound in every alternative", name.0) };
            self.unify(self.out.locals[local as usize].ty, self.out.locals[kept as usize].ty)?;
            self.out.binds.values_mut().filter(|at| **at == local).for_each(|at| *at = kept);
        }
        Ok(())
    }

    fn named_pat(&mut self, name: u32) -> Result<Option<(Ty, Res)>> {
        let sym = self.cx.names.symbols.intern(plain(self.src.text(name)));
        let found = self.value_def(&[sym], false, None).ok();
        let (item, variant) = match found {
            Some(crate::names::Def::Variant(item, index)) => {
                let Some(at) = self.cx.adt_def(item)?.position(index) else {
                    return Ok(None);
                };
                (item, at as u32)
            }
            Some(crate::names::Def::Item(item)) if self.cx.item(item).1.kind == ItemKind::Struct => (item, 0),
            Some(crate::names::Def::Item(item)) if self.cx.item(item).1.kind == ItemKind::Const => {
                return self.const_item(item, Vec::new()).map(Some);
            }
            _ => return Ok(None),
        };
        if self.cx.adt_def(item)?.variants[variant as usize].shape != crate::body::adt::Shape::Unit {
            return Ok(None);
        }
        let args = self.adt_args(item, Vec::new(), true)?;
        let interned = self.cx.types.intern_substs(&args);
        let adt = self.cx.types.intern(Kind::Adt(item, interned));
        Ok(Some((adt, Res::Ctor(item, variant, interned))))
    }
}
