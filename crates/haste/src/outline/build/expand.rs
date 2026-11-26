use std::sync::Arc;

use anyhow::Result;
use rustc_lexer::TokenKind;

use super::super::item::{Attr, Item, ItemKind, Vis};
use super::super::rules::Frag;
use super::super::source::Source;
use super::super::tree::{Env, File, Macro, MemberBlock, Textual, Waiting, textual};
use super::super::{mbe, rules, scan};
use super::{Loader, Others, child_origin};

const BUILTIN: [&str; 6] = [
    "include",
    "global_asm",
    "compile_error",
    "concat",
    "env",
    "asm",
];

enum Found {
    Macro(Arc<Macro>),
    NotYet(usize),
    Missing,
}

impl Loader<'_> {
    pub(super) fn members(
        &mut self,
        file: &Arc<File>,
        block: MemberBlock,
        module: usize,
        owner: Option<u32>,
    ) -> Result<()> {
        for (at, item) in file.items.items.iter().enumerate() {
            if item.parent == owner
                && item.kind == ItemKind::MacroCall
                && self.out.env.enabled(&item.attrs)
            {
                self.member_call(file, block, module, at as u32)?;
            }
        }
        Ok(())
    }

    fn member_call(
        &mut self,
        file: &Arc<File>,
        block: MemberBlock,
        module: usize,
        at: u32,
    ) -> Result<()> {
        let item = &file.items.items[at as usize];
        let (Some(name), Some(_)) = (item.name, item.body) else {
            return Ok(());
        };
        let name = crate::names::plain(file.source.text(name));
        let Some(definition) = self.lookup(file, item, name) else {
            self.out
                .members
                .push((block, self.waiting(file, module, at)));
            return Ok(());
        };
        self.member_at(file, block, module, at, &definition)
    }

    /// Expands the waiting macro call `call` in an impl or trait body with `definition`, in the textual scope it was written in.
    pub fn expand_member(
        &mut self,
        block: MemberBlock,
        call: Waiting,
        definition: &Macro,
    ) -> Result<()> {
        self.scope = call.scope;
        self.member_at(&call.file, block, call.owner, call.at, definition)
    }

    fn member_at(
        &mut self,
        file: &Arc<File>,
        block: MemberBlock,
        module: usize,
        at: u32,
        definition: &Macro,
    ) -> Result<()> {
        let Some((name, expanded)) = self.expand_call(file, at, definition) else {
            return Ok(());
        };
        let Some(expanded) = self.scanned(expanded, name, child_origin(file, 2, at)) else {
            return Ok(());
        };
        let parts = match block {
            MemberBlock::Impl(index) => &mut self.out.impls[index].parts,
            MemberBlock::Trait(index) => &mut self.out.traits[index].parts,
        };
        parts.push(self.out.files.len() - 1);
        self.depth += 1;
        let result = self.members(&expanded, block, module, None);
        self.depth -= 1;
        result
    }

    pub(super) fn define(&mut self, file: &Arc<File>, place: (u32, u32), item: &Item) {
        let (Some(name), Some(open)) = (item.name, item.body) else {
            return;
        };
        let builtin = item
            .attrs
            .iter()
            .any(|attr| matches!(attr, Attr::Other(at) if file.source.text(*at) == "rustc_builtin_macro"));
        if builtin {
            return;
        }
        let parsed = if item.kind == ItemKind::Macro {
            rules::parse_macro(&file.source, open)
        } else {
            rules::parse(&file.source, open)
        };
        match parsed {
            Ok(rules) => {
                let rules = if item.attrs.contains(&Attr::MacroExport(true)) {
                    rules::local_inner(&file.source, rules)
                } else {
                    rules
                };
                let name = crate::names::plain(file.source.text(name)).to_owned();
                let definition = Arc::new(Macro {
                    file: file.clone(),
                    rules,
                    krate: self.krate,
                });
                self.out.defined.insert(place, definition.clone());
                let public = item.kind == ItemKind::Macro && item.vis != Vis::Private;
                if public
                    || item
                        .attrs
                        .iter()
                        .any(|attr| matches!(attr, Attr::MacroExport(_)))
                {
                    self.out.exports.insert(name.clone(), definition.clone());
                }
                if item.kind == ItemKind::MacroRules {
                    self.scope = Some(Arc::new(Textual {
                        name,
                        def: definition,
                        next: self.scope.take(),
                    }));
                }
            }
            Err(error) => self.out.errors.push(format!("{error:#}")),
        }
    }

    fn exported(&self, krate: usize, name: &str) -> Found {
        let (exports, unsettled) = match self.others {
            _ if krate == self.krate => (&self.out.exports, self.out.unsettled()),
            Others::Crates(crates) => (&crates[krate].exports, crates[krate].unsettled()),
            Others::Views(views) => match views.get(krate).and_then(Option::as_ref) {
                Some(view) => (&view.exports, view.partial),
                None => return Found::NotYet(krate),
            },
        };
        match exports.get(name) {
            Some(found) => Found::Macro(found.clone()),
            None if unsettled => Found::NotYet(krate),
            None => Found::Missing,
        }
    }

    fn lookup(&mut self, file: &File, item: &Item, name: &str) -> Option<Arc<Macro>> {
        if name == "cfg_select" {
            return textual(self.scope.as_ref(), name).filter(|_| item.name == Some(item.start));
        }
        let crates = match self.qualified(file, item) {
            Some(crates) => crates?,
            None => {
                if let Some(found) = textual(self.scope.as_ref(), name) {
                    return Some(found);
                }
                self.out.macro_use.clone()
            }
        };
        for krate in crates {
            match self.exported(krate, name) {
                Found::Macro(found) => return Some(found),
                Found::NotYet(krate) => self.out.fallback.push(krate),
                Found::Missing => {}
            }
        }
        None
    }

    fn qualified(&self, file: &File, item: &Item) -> Option<Option<Vec<usize>>> {
        let src = &file.source;
        let first = item.start;
        if src.kind(first) == TokenKind::Dollar {
            return Some(
                (item.name == Some(first + 4))
                    .then(|| src.dollar(first).map(|krate| vec![krate as usize]))
                    .flatten(),
            );
        }
        if item.name == Some(first) || src.kind(first) == TokenKind::Colon {
            return None;
        }
        if item.name != Some(first + 3) {
            return Some(None);
        }
        Some(match src.text(first) {
            "crate" | "self" | "super" => Some(vec![self.krate]),
            segment => self
                .deps
                .iter()
                .find(|(dep, _)| dep == segment)
                .map(|(_, krate)| vec![*krate]),
        })
    }

    /// Expands again the calls that found no definition, now that more modules are read.
    pub fn retry(&mut self) -> Result<()> {
        self.out.fallback.clear();
        for call in std::mem::take(&mut self.out.deferred) {
            self.scope = call.scope;
            self.call(&call.file, call.owner, call.at)?;
        }
        for (block, call) in std::mem::take(&mut self.out.members) {
            self.scope = call.scope;
            self.member_call(&call.file, block, call.owner, call.at)?;
        }
        self.scope = None;
        Ok(())
    }

    fn waiting(&self, file: &Arc<File>, owner: usize, at: u32) -> Waiting {
        Waiting {
            file: file.clone(),
            owner,
            at,
            scope: self.scope.clone(),
        }
    }

    pub(super) fn call(&mut self, file: &Arc<File>, module: usize, at: u32) -> Result<()> {
        let item = &file.items.items[at as usize];
        let (Some(name), Some(open)) = (item.name, item.body) else {
            return Ok(());
        };
        let name = crate::names::plain(file.source.text(name));
        if name == "include" {
            return self.include(file, module, open);
        }
        if name == "global_asm" {
            self.out.global_asm.push(self.waiting(file, module, at));
        }
        if BUILTIN.contains(&name) {
            return Ok(());
        }
        let Some(definition) = self.lookup(file, item, name) else {
            if name == "cfg_select" {
                return self.cfg_select(file, module, open);
            }
            self.out.deferred.push(self.waiting(file, module, at));
            return Ok(());
        };
        self.expand_at(file, module, at, &definition)
    }

    /// Expands the waiting macro call `call` with `definition`, in the textual scope it was written in.
    pub fn expand(&mut self, call: Waiting, definition: &Macro) -> Result<()> {
        self.scope = call.scope;
        self.expand_at(&call.file, call.owner, call.at, definition)
    }

    /// Adds `source`, the output of procedural macro `name`, to module `module` in the module's textual scope.
    pub fn insert(&mut self, module: usize, source: Source, name: &str) -> Result<()> {
        self.scope = self.out.modules[module].textual.clone();
        let result = self.splice(module, Ok(source), name, 0);
        self.scope = None;
        result
    }

    fn expand_call<'f>(
        &mut self,
        file: &'f Arc<File>,
        at: u32,
        definition: &Macro,
    ) -> Option<(&'f str, Result<Source>)> {
        let item = &file.items.items[at as usize];
        let (Some(name), Some(open)) = (item.name, item.body) else {
            return None;
        };
        let name = crate::names::plain(file.source.text(name));
        let limit = crate_recursion_limit(self.out);
        if self.depth >= limit {
            self.out.errors.push(format!(
                "{}: macro {name} recursed past {limit} levels",
                file.source.path.display(),
            ));
            return None;
        }
        Some((
            name,
            mbe::expand(
                &definition.file.source,
                &definition.rules,
                &file.source,
                (open, definition.krate as u32),
                None,
            ),
        ))
    }

    fn expand_at(
        &mut self,
        file: &Arc<File>,
        module: usize,
        at: u32,
        definition: &Macro,
    ) -> Result<()> {
        let Some((name, expanded)) = self.expand_call(file, at, definition) else {
            return Ok(());
        };
        self.splice(module, expanded, name, child_origin(file, 3, at))
    }

    fn cfg_select(&mut self, file: &Arc<File>, module: usize, open: u32) -> Result<()> {
        let selected = match mbe::select(&file.source, open, &self.out.env) {
            Ok(Some(selected)) => selected,
            Ok(None) => return Ok(()),
            Err(error) => {
                self.out.errors.push(format!("cfg_select!: {error:#}"));
                return Ok(());
            }
        };
        self.splice(
            module,
            Ok(selected),
            "cfg_select",
            child_origin(file, 4, open),
        )
    }

    fn include(&mut self, file: &Arc<File>, module: usize, open: u32) -> Result<()> {
        let src = &file.source;
        let path = path_text(src, (open + 1, src.pair(open)), &self.out.env)
            .ok_or_else(|| anyhow::anyhow!("a path that is not a literal, `concat!` or `env!`"));
        let source = path.and_then(|path| Source::read(&src.beside(&path)));
        self.splice(module, source, "include", child_origin(file, 5, open))
    }

    fn scanned(&mut self, source: Result<Source>, name: &str, origin: u64) -> Option<Arc<File>> {
        let expanded = source.and_then(|source| {
            scan::scan(&source).map(|items| File::new_expanded(source, items, origin))
        });
        match expanded {
            Ok(expanded) => {
                let expanded = Arc::new(expanded);
                self.out.files.push(expanded.clone());
                Some(expanded)
            }
            Err(error) => {
                self.out.errors.push(format!("{name}!: {error:#}"));
                None
            }
        }
    }

    pub(super) fn splice(
        &mut self,
        module: usize,
        source: Result<Source>,
        name: &str,
        origin: u64,
    ) -> Result<()> {
        let Some(expanded) = self.scanned(source, name, origin) else {
            return Ok(());
        };
        let index = self.out.files.len() - 1;
        self.out.modules[module].parts.push(index);
        self.out.expansions += 1;
        self.depth += 1;
        let result = self.children(&expanded, index, module, None);
        self.depth -= 1;
        result
    }
}

/// The text of the tokens from `from` up to `to` when they are a string literal or a `concat!`, `env!` or `stringify!` of them, inside the parentheses that a macro puts around an expression fragment.
pub(crate) fn path_text(src: &Source, (from, to): (u32, u32), env: &Env) -> Option<String> {
    if to == from + 1 {
        return src.literal_text(from).ok().flatten();
    }
    if src.text(from) == "(" && src.pair(from) == to - 1 {
        return path_text(src, (from + 1, to - 1), env);
    }
    if to < from + 3 || src.text(from + 1) != "!" {
        return None;
    }
    let parts = src.args(from + 2);
    match src.text(from) {
        "concat" => parts
            .into_iter()
            .map(|range| path_text(src, range, env))
            .collect(),
        "env" => env.var(&path_text(src, *parts.first()?, env)?),
        "stringify" => Some(spelled_fragment(src, (from + 3, src.pair(from + 2)))),
        _ => None,
    }
}

/// Spells `from..to` without synthetic grouping parentheses inserted around `macro_rules!` expression fragments.
pub(crate) fn spelled_fragment(src: &Source, (from, to): (u32, u32)) -> String {
    if from >= to {
        return String::new();
    }
    let mut skip = vec![0u8; (to - from) as usize];
    for (op_from, op_to, frag) in src.opaques_in(from, to) {
        if frag != Frag::Expr {
            continue;
        }
        let mut open = op_from;
        while open + 1 < op_to
            && src.kind(open) == TokenKind::Pound
            && src.kind(open + 1) == TokenKind::OpenBracket
        {
            open = src.pair(open + 1) + 1;
        }
        if open + 1 < op_to
            && src.kind(open) == TokenKind::OpenParen
            && src.pair(open) == op_to - 1
            && src.origin(open).is_none()
            && src.origin(op_to - 1).is_none()
            && !src.joint(open)
        {
            skip[(open - from) as usize] = 1;
            skip[(op_to - 1 - from) as usize] = 2;
        }
    }
    src.spelled_with((from, to), |at| skip[(at - from) as usize])
}

const MAX_RECURSION_LIMIT: usize = 4096;

fn crate_recursion_limit(krate: &super::Crate) -> usize {
    let Some(root) = krate.files.first() else {
        return crate::RECURSION_LIMIT;
    };
    let src = &root.source;
    for attr in &root.items.inner {
        if let super::Attr::Other(at) = *attr
            && src.text(at) == "recursion_limit"
            && src.text(at + 1) == "="
            && let Ok(n) = src.text(at + 2).trim_matches('"').parse::<usize>()
        {
            return n.min(MAX_RECURSION_LIMIT);
        }
    }
    crate::RECURSION_LIMIT
}
