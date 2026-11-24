mod lower;

use anyhow::{Result, bail};

use crate::body::{self as syntax, Body, PathId};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, Ns, PerNs, Primitive, Sym, Symbols, plain};
use crate::outline::item::ItemKind;
use crate::outline::source::{Mark, Source};

use super::super::body_scope::{BodyPath, head_ns, in_namespace, outside};
use super::super::impls::ImplRef;
use super::super::ty::{Kind, Ty};
use super::{Infer, Res};

/// Stack-inline path segment symbols avoiding heap allocation for paths of up to 4 segments.
pub struct PathSyms {
    inline: [Sym; 4],
    len: u8,
    overflow: Vec<Sym>,
}

impl PathSyms {
    /// An empty segment list.
    pub const EMPTY: Self = Self {
        inline: [Sym(0); 4],
        len: 0,
        overflow: Vec::new(),
    };

    /// Shortens the segment list to at most `len` elements.
    pub fn truncate(&mut self, len: usize) {
        if self.len == u8::MAX {
            self.overflow.truncate(len);
        } else if len < usize::from(self.len) {
            self.len = len as u8;
        }
    }
}

impl std::ops::Deref for PathSyms {
    type Target = [Sym];

    #[inline]
    fn deref(&self) -> &[Sym] {
        if self.len == u8::MAX {
            &self.overflow
        } else {
            &self.inline[..usize::from(self.len)]
        }
    }
}

/// The interned segments of `path` in `arena`, whether it starts with `::`, and the crate a leading `$crate` names.
#[inline]
pub fn segments(
    symbols: &Symbols,
    src: &Source,
    arena: &Body,
    path: PathId,
) -> (PathSyms, bool, Option<u32>) {
    let path = arena.path(path);
    let syms = arena.syms(symbols, src);
    let list = arena.list(path.segments);
    let segments = if list.len() <= 4 {
        let mut inline = [Sym(0); 4];
        for (slot, &at) in inline.iter_mut().zip(list) {
            *slot = syms[at as usize];
        }
        PathSyms {
            inline,
            len: list.len() as u8,
            overflow: Vec::new(),
        }
    } else {
        PathSyms {
            inline: [Sym(0); 4],
            len: u8::MAX,
            overflow: list.iter().map(|&at| syms[at as usize]).collect(),
        }
    };
    let first = list
        .first()
        .map(|&at| arena.segments[at as usize].name)
        .unwrap_or(0);
    let dollar = first.checked_sub(1).and_then(|at| src.dollar(at));
    (segments, path.global, dollar)
}

impl<'a> Infer<'_, 'a> {
    pub(super) fn segments(&self, arena: &Body, path: PathId) -> (PathSyms, bool, Option<u32>) {
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
        let [name] = segments[..] else {
            return found;
        };
        match found {
            Err(error) => self
                .def_site(arena, path, name)
                .and_then(|found| found.values)
                .map(|binding| binding.def)
                .ok_or(error),
            found => found,
        }
    }

    fn def_site(&mut self, arena: &Body, path: PathId, name: Sym) -> Option<PerNs> {
        let first = arena.list(arena.path(path).segments)[0];
        let Some(Mark::Home { krate, file, item }) =
            self.src.mark(arena.segments[first as usize].name)
        else {
            return None;
        };
        let module = self.cx.module_of(ItemRef { krate, file, item }).ok()?;
        let path = NamePath {
            segments: &[name],
            absolute: false,
            dollar: None,
        };
        Some(self.cx.paths(module.krate).first(name, module, &path))
    }

    pub(super) fn value_def(
        &mut self,
        segments: &[Sym],
        absolute: bool,
        dollar: Option<u32>,
    ) -> Result<Def> {
        let (segments, inside) = match self.local_def(
            &NamePath {
                segments,
                absolute,
                dollar,
            },
            Ns::Values,
        ) {
            Some(Ok(def)) => return Ok(def),
            Some(Err(skip)) => (&segments[skip..], false),
            None => (segments, true),
        };
        let absolute = absolute && inside;
        let Some((last, prefix)) = segments.split_last() else {
            bail!("empty path");
        };
        let found = if prefix.is_empty() {
            let local = if absolute || !inside {
                None
            } else {
                self.body_use(*last).values
            };
            local.or_else(|| {
                self.cx
                    .paths(self.home.krate)
                    .first(
                        *last,
                        self.home,
                        &NamePath {
                            segments,
                            absolute,
                            dollar,
                        },
                    )
                    .values
            })
        } else {
            let path = NamePath {
                segments: prefix,
                absolute,
                dollar,
            };
            let head = if inside {
                self.type_def(&path)
            } else {
                self.cx.paths(self.home.krate).resolve(&path, self.home)
            };
            head.map(|def| self.cx.path_member(self.home, def, *last))
                .and_then(|entry| entry.values)
        };
        match found {
            Some(binding) => Ok(binding.def),
            None => Err(self.diag_expr(
                crate::diag::DiagnosticKind::Name,
                None,
                format!("cannot find value `{}` in this scope", self.spell(segments)),
                "not found in this scope",
            )),
        }
    }

    pub(super) fn type_def(&mut self, path: &NamePath<'_>) -> Option<Def> {
        if !self.items.is_empty() || !self.block_uses.is_empty() {
            return self.type_def_here(path);
        }
        let here = self
            .anchor()
            .and_then(|(root, from)| from.map(|at| root.inner(at)));
        let scope = (self.home, self.root, here, path.absolute, path.dollar);
        if let Some(found) = self.cx.type_def(scope, path.segments) {
            return Some(found);
        }
        let before = self.cx.quiet_mark();
        let found = self.type_def_here(path)?;
        if self.cx.quiet_mark() == before {
            self.cx.set_type_def(scope, path.segments, found);
        }
        Some(found)
    }

    fn type_def_here(&mut self, path: &NamePath<'_>) -> Option<Def> {
        match self.local_def(path, Ns::Types) {
            Some(Ok(def)) => Some(def),
            Some(Err(skip)) => self
                .cx
                .paths(self.home.krate)
                .resolve(&outside(path, skip), self.home),
            None => self.resolve_local(path),
        }
    }

    fn anchor(&self) -> Option<(ItemRef, Option<u32>)> {
        let here = self
            .frame_item
            .or_else(|| self.block.and_then(ImplRef::item));
        match here.and_then(ItemRef::outer) {
            Some((root, at)) => Some((root, Some(at))),
            None => self.root.map(|root| (root, None)),
        }
    }

    fn local_def(&mut self, path: &NamePath<'_>, ns: Ns) -> Option<Result<Def, usize>> {
        let wanted = head_ns(path.segments.len(), ns);
        let head = path
            .segments
            .first()
            .filter(|_| !path.absolute)
            .and_then(|first| self.block_item(self.cx.names.symbols.text(*first), wanted));
        let found = match head {
            Some(item) => self
                .cx
                .local_descend(self.home, item, &path.segments[1..], ns)?,
            None => self.cx.local_path(
                self.home,
                self.anchor().filter(|(_, from)| from.is_some()),
                path,
                ns,
            )?,
        };
        Some(match found {
            BodyPath::Def(def) => Ok(def),
            BodyPath::Variant(item) => Ok(self.variant_def(item, *path.segments.last()?)?),
            BodyPath::Outside(skip) => Err(skip),
        })
    }

    fn variant_def(&mut self, item: ItemRef, name: Sym) -> Option<Def> {
        let def = self.cx.adt_def(item).ok()?;
        let (src, text) = (self.cx.item(item).0, self.cx.names.symbols.text(name));
        def.variants
            .iter()
            .find(|variant| plain(src.text(variant.name)) == text)
            .map(|variant| Def::Variant(item, variant.index))
    }

    pub(super) fn resolve_local(&mut self, path: &NamePath<'_>) -> Option<Def> {
        let (first, rest) = path.segments.split_first()?;
        let local = if path.absolute {
            None
        } else {
            self.body_use(*first).types
        };
        let paths = self.cx.paths(self.home.krate);
        let Some(binding) = local else {
            return paths.resolve(path, self.home);
        };
        rest.iter().try_fold(binding.def, |def, segment| {
            self.cx
                .path_member(self.home, def, *segment)
                .types
                .map(|found| found.def)
        })
    }

    fn body_use(&mut self, word: Sym) -> PerNs {
        self.body_imports(Some(word))
            .into_iter()
            .fold(PerNs::default(), |found, entry| PerNs {
                types: found.types.or(entry.types),
                values: found.values.or(entry.values),
                macros: found.macros.or(entry.macros),
            })
    }

    pub(super) fn body_traits(&mut self) -> Vec<ItemRef> {
        let mut traits = self
            .root
            .map(|root| self.cx.body_declared_traits(root))
            .unwrap_or_default();
        let found = self
            .body_imports(None)
            .into_iter()
            .filter_map(|entry| entry.types);
        let home = self.home;
        for binding in found.collect::<Vec<_>>() {
            match binding.def {
                Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Trait => {
                    traits.push(item)
                }
                Def::Module(module) => self.cx.paths(home.krate).traits_of(module, &mut traits),
                _ => {}
            }
        }
        traits
    }

    fn body_imports(&mut self, word: Option<Sym>) -> Vec<PerNs> {
        let Some(root) = self.root.filter(|&root| {
            self.cx
                .expansion(root)
                .is_some_and(|found| !found.file.items.imports.is_empty())
        }) else {
            return Vec::new();
        };
        let hidden = self.closed_uses(root);
        let here = self.reader().and_then(ItemRef::outer).map(|(_, at)| at);
        if let Some(found) = self.imports.get(&(root, word, here, hidden.clone())) {
            return found.clone();
        }
        let Ok(home) = self.cx.module_of(root) else {
            return Vec::new();
        };
        let before = self.cx.asks.mark();
        let found = self.cx.body_imports((root, home), word, (here, &hidden));
        if self.cx.asks.mark() == before {
            self.imports
                .insert((root, word, here, hidden), found.clone());
        }
        found
    }

    fn reader(&self) -> Option<ItemRef> {
        let declared =
            |item: &ItemRef| item.declaring_body().is_some() && item.declaring_body() == self.root;
        let frame = self
            .frame_item
            .or_else(|| self.block.and_then(ImplRef::item))
            .filter(declared);
        frame.or_else(|| {
            self.instance
                .as_ref()
                .map(|(item, _)| *item)
                .filter(|item| item.outer().is_some())
        })
    }

    fn closed_uses(&self, root: ItemRef) -> Vec<u32> {
        let Some(found) = self.cx.expansion(root) else {
            return Vec::new();
        };
        let items = &found.file.items.items;
        let declared = self.body.stmts.iter().filter_map(|stmt| match *stmt {
            crate::body::Stmt::Item(start, end) => {
                Some(super::super::expand::stmt_items(items, (start, end)))
            }
            _ => None,
        });
        declared
            .flatten()
            .map(|at| at as u32)
            .filter(|at| items[*at as usize].kind == ItemKind::Use && !self.block_uses.contains(at))
            .collect()
    }

    fn block_item(&self, name: &str, ns: Ns) -> Option<ItemRef> {
        self.items
            .iter()
            .rev()
            .find(|(bound, item)| *bound == name && in_namespace(self.cx.item(*item).1, ns))
            .map(|(_, item)| *item)
    }

    /// Resolves the value path expression `expr` to a local or an item, and returns its type.
    pub fn path_expr(&mut self, expr: syntax::ExprId, path: PathId) -> Result<Ty> {
        let (ty, res) = self.value_path(path)?;
        if let Res::Local(local) = res {
            self.note_read(expr, local);
        }
        self.out.res.insert(expr, res);
        Ok(ty)
    }

    pub(super) fn value_path(&mut self, path: PathId) -> Result<(Ty, Res)> {
        let body = self.body;
        let entry = body.path(path);
        let single = entry.qself.is_none() && !entry.global && entry.segments.len == 1;
        let first = body.segments[body.list(entry.segments)[0] as usize].name;
        let mut scope = self.src.scope(first);
        if single && scope == (0, 0) && self.src.mark(first).is_none() {
            scope = self.proc_call_scope(first);
        }
        if single && let Some(found) = self.bound_value(plain(self.src.text(first)), scope)? {
            return Ok(found);
        }
        if let Some(ctor) = self.ctor(body, path, true)? {
            return self.ctor_value(ctor);
        }
        self.path_value(body, path)
    }

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
            Def::Item(item)
                if matches!(
                    self.cx.item(item).1.kind,
                    ItemKind::Const | ItemKind::Static
                ) =>
            {
                self.const_item(item, Vec::new())
            }
            other => bail!("unsupported value {other:?}"),
        }
    }

    fn proc_call_scope(&self, token: u32) -> (u32, u32) {
        let Some(root) = self.root else {
            return (0, 0);
        };
        let Some(raw) = self
            .cx
            .outline
            .crates
            .get(root.krate as usize)
            .and_then(|c| c.files.get(root.file as usize))
        else {
            return (0, 0);
        };
        if token < raw.source.len() {
            return (0, 0);
        }
        let Some(exp) = raw.expansion(root.krate, root.item) else {
            return (0, 0);
        };
        let enclosing = |pos: u32| {
            exp.calls
                .iter()
                .filter_map(|(&co, &bo)| (pos > bo && pos < self.src.pair(bo)).then_some((co, bo)))
                .max_by_key(|&(_, bo)| bo)
        };
        let Some((mut call_open, _)) = enclosing(token) else {
            return (0, 0);
        };
        for _ in 0..8 {
            let scope = self.src.scope(call_open.saturating_sub(2));
            if scope != (0, 0) {
                return scope;
            }
            let Some((outer, _)) = enclosing(call_open) else {
                break;
            };
            call_open = outer;
        }
        (0, 0)
    }

    fn bound_value(&mut self, name: &str, (id, parent): (u32, u32)) -> Result<Option<(Ty, Res)>> {
        let found = [id, parent, 0].into_iter().find_map(|scope| {
            self.scopes
                .iter()
                .rev()
                .find(|(bound, _)| *bound == (name, scope))
        });
        if let Some(&(_, local)) = found {
            self.note_use(local);
            return Ok(Some((
                self.out.locals[local as usize].ty,
                Res::Local(local),
            )));
        }
        Ok(self
            .const_param(name)?
            .map(|(ty, value)| (ty, Res::Const(value))))
    }

    pub(super) fn const_item(&mut self, item: ItemRef, outer: Vec<Ty>) -> Result<(Ty, Res)> {
        let ty = self.fn_types(item, &outer)?.1;
        Ok((ty, Res::fn_of(&self.cx.types, item, &outer)))
    }

    pub(super) fn fn_item(
        &mut self,
        item: ItemRef,
        outer: Vec<Ty>,
        given: Vec<Ty>,
    ) -> Result<(Ty, Res)> {
        let mut substs = outer;
        substs.extend(self.own_args(item, given)?);
        self.bind_bounds(item, &substs)?;
        let interned = self.cx.types.intern_substs(&substs);
        Ok((
            self.cx.types.intern(Kind::FnDef(item, interned)),
            Res::Fn(item, interned),
        ))
    }

    pub(super) fn spell(&self, segments: &[Sym]) -> String {
        segments
            .iter()
            .map(|sym| self.cx.names.symbols.text(*sym))
            .collect::<Vec<_>>()
            .join("::")
    }
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
    let split = rest
        .find(|ch: char| !ch.is_digit(radix))
        .unwrap_or(rest.len());
    let (number, suffix) = rest.split_at(split);
    let value = u128::from_str_radix(number, radix)?;
    let suffix = if suffix.is_empty() {
        None
    } else {
        Some(Primitive::named(suffix).ok_or_else(|| anyhow::anyhow!("bad suffix {suffix}"))?)
    };
    Ok((value, suffix))
}
