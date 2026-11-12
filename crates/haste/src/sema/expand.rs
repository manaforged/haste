use anyhow::{Result, anyhow, bail};
use rustc_hash::FxHashMap as HashMap;
use rustc_lexer::TokenKind;

use crate::body::cursor::{RESERVED, is_name};
use crate::body::{self as syntax, Body, sig};
use crate::names::collect::{extern_crate_root, proc_kind};
use crate::names::resolve::macro_at;
use crate::names::{ItemRef, ModRef, plain};
use crate::outline::item::{Attr, ItemKind, Items, ProcKind};
use crate::outline::rules::{self, Rule};
use crate::outline::source::{Mark, Source, Tokens, opens};
use crate::outline::tree::{Expansion, File, Truth};
use crate::outline::{attr, derive, mbe, scan};

use super::Context;
use super::builtin::{self, Builtin};

/// A `macro_rules!` that a body defines, copied out so that its rules index its own tokens.
struct Local {
    name: String,
    source: Source,
    rules: Vec<Rule>,
}

/// The expansions of one body being written after the tokens of its file.
struct Writer {
    out: Tokens,
    calls: HashMap<u32, u32>,
    offset: u32,
    home: ModRef,
    locals: Vec<Local>,
    uses: Vec<(String, Source)>,
    globs: Vec<Source>,
    externs: Vec<(String, ModRef)>,
}

/// A token range being scanned for macro calls: its source, where it starts, how far its indices sit from those of the body, how deep in expansions it is, and the source and first token of the outermost macro call it came from.
#[derive(Clone, Copy)]
struct Run<'s> {
    src: &'s Source,
    from: u32,
    shift: u32,
    depth: u32,
    call: (&'s Source, u32),
}

impl<'s> Run<'s> {
    /// The outermost macro call for the output of the call that starts at token `start` of this run.
    fn call_at(self, start: u32) -> (&'s Source, u32) {
        if self.depth == 0 { (self.src, start) } else { self.call }
    }
}

/// The first token and the name token of the path of the macro call whose `!` is token `bang`.
fn call_path(src: &Source, bang: u32, from: u32) -> Option<(u32, u32)> {
    let name = bang.checked_sub(1).filter(|name| *name >= from)?;
    let kind = src.kind(name);
    if !is_name(kind) || (kind == TokenKind::Ident && RESERVED.contains(&src.text(name))) {
        return None;
    }
    let mut start = name;
    while start >= from + 3 && src.kind(start - 1) == TokenKind::Colon && src.kind(start - 2) == TokenKind::Colon && is_name(src.kind(start - 3)) {
        start -= 3;
    }
    if start >= from + 2 && src.kind(start - 1) == TokenKind::Colon && src.kind(start - 2) == TokenKind::Colon {
        start -= 2;
    } else if start > from && src.text(start) == "crate" && src.kind(start - 1) == TokenKind::Dollar {
        start -= 1;
    }
    Some((start, name))
}

/// The macro that the single name at `path` calls, resolved from the module of the `macro` item that wrote the name.
fn def_site_macro(cx: &mut Context<'_>, src: &Source, path: (u32, u32)) -> Option<ItemRef> {
    let Some(Mark::Home { krate, file, item }) = src.mark(path.0).filter(|_| path.0 == path.1) else {
        return None;
    };
    let module = cx.module_of(ItemRef { krate, file, item }).ok()?;
    macro_at(&cx.paths(module.krate), src, path, module)
}

/// The output of the `macro_rules!` or `macro` item `found` called with the group at `open` of `src`, or none when `found` is not one.
fn defined_expand(cx: &Context<'_>, found: ItemRef, src: &Source, open: u32) -> Option<Result<Source>> {
    let definition = cx.outline.crates[found.krate as usize].defined.get(&(found.file, found.item))?;
    let home = (cx.item(found).1.kind == ItemKind::Macro).then_some(Mark::Home { krate: found.krate, file: found.file, item: found.item });
    Some(mbe::expand(&definition.file.source, &definition.rules, src, (open, definition.krate as u32), home))
}

/// The output of the procedural macro `run` called from crate `krate` with the delimiter at `open`, once the macro expander ran it; until then the call asks for it.
fn proc_bang(cx: &mut Context<'_>, krate: u32, src: &Source, run: ItemRef, open: u32) -> Result<Option<Source>> {
    let input = crate::macros::bang_input(&cx.outline.deps[krate as usize], krate as usize, src, open);
    let key = (run, input);
    match cx.outline.procs.get(&key) {
        Some(Ok(text)) => Source::parse(src.path.clone(), text.clone()).map(Some),
        Some(Err(error)) => bail!("{}!: {error}", cx.name(run)),
        None => {
            cx.asks.procs.borrow_mut().push(key);
            Ok(None)
        }
    }
}

fn body_range(file: &File, item: u32) -> Option<(u32, u32)> {
    let entry = &file.items.items[item as usize];
    match (entry.kind, entry.name) {
        (ItemKind::Const | ItemKind::Static, Some(name)) => Some((sig::initializer(&file.source, name, entry.end)?, entry.end)),
        _ => entry.body.map(|open| (open, file.source.pair(open) + 1)),
    }
}

impl Writer {
    fn scan(&mut self, cx: &mut Context<'_>, run: Run<'_>, to: u32) -> Result<()> {
        let mut at = run.from;
        while at < to {
            at = self.step(cx, run, at)?;
        }
        Ok(())
    }

    fn step(&mut self, cx: &mut Context<'_>, run: Run<'_>, at: u32) -> Result<u32> {
        let src = run.src;
        if src.kind(at) == TokenKind::Pound
            && let Some(next) = self.module_file(cx, run, at)?
        {
            return Ok(next);
        }
        if src.kind(at) != TokenKind::Bang {
            return self.non_bang(cx, src, at);
        }
        let next = at + 1;
        if at > run.from && src.text(at - 1) == "macro_rules" && is_name(src.kind(next)) {
            if !opens(src.kind(next + 1)) {
                return Ok(next + 1);
            }
            self.define(src, next)?;
            return Ok(src.pair(next + 1) + 1);
        }
        let Some(path) = call_path(src, at, run.from).filter(|_| opens(src.kind(next))) else {
            return Ok(next);
        };
        self.call(cx, run, path, next)?;
        Ok(src.pair(next) + 1)
    }

    fn non_bang(&mut self, cx: &Context<'_>, src: &Source, at: u32) -> Result<u32> {
        if src.text(at) == "macro" && is_name(src.kind(at + 1)) && opens(src.kind(at + 2)) {
            return self.define_macro(src, at + 1);
        }
        if src.text(at) == "use"
            && let Some(next) = self.note_use(src, at)?
        {
            return Ok(next);
        }
        if src.text(at) == "extern"
            && src.text(at + 1) == "crate"
            && is_name(src.kind(at + 2))
            && let Some(next) = self.note_extern(cx, src, at)
        {
            return Ok(next);
        }
        Ok(at + 1)
    }

    fn call(&mut self, cx: &mut Context<'_>, run: Run<'_>, path: (u32, u32), open: u32) -> Result<()> {
        let Some(output) = self.expand_one(cx, run, path, open)? else {
            return Ok(());
        };
        let block = self.offset + self.out.len();
        self.calls.insert(run.shift + open, block);
        self.out.lexed("{");
        self.out.copy_range(&output, 0, output.len());
        self.out.space();
        self.out.lexed("}");
        self.scan(cx, Run { src: &output, from: 0, shift: block + 1, depth: run.depth + 1, call: run.call_at(path.0) }, output.len())
    }

    /// Writes the file that a `#[path = ".."] mod name;` starting at `at` names as the module's block, keyed by its `;`, and returns the token after it.
    fn module_file(&mut self, cx: &mut Context<'_>, run: Run<'_>, at: u32) -> Result<Option<u32>> {
        let src = run.src;
        let mut attrs = Vec::new();
        let mut word = attr::attrs(src, at, false, &mut attrs)?;
        let Some(path) = attrs.iter().find_map(|found| if let Attr::Path(path) = found { Some(path.clone()) } else { None }) else { return Ok(None) };
        if src.text(word) == "pub" {
            word += 1;
            if src.kind(word) == TokenKind::OpenParen {
                word = src.pair(word) + 1;
            }
        }
        if src.text(word) != "mod" || !is_name(src.kind(word + 1)) || src.kind(word + 2) != TokenKind::Semi {
            return Ok(None);
        }
        let output = Source::read(&run.call_at(at).0.beside(&path))?;
        let block = self.offset + self.out.len();
        self.calls.insert(run.shift + word + 2, block);
        self.out.lexed("{");
        self.out.copy_range(&output, 0, output.len());
        self.out.space();
        self.out.lexed("}");
        self.scan(cx, Run { src: &output, from: 0, shift: block + 1, depth: run.depth, call: (&output, 0) }, output.len())?;
        Ok(Some(word + 3))
    }

    fn note_use(&mut self, src: &Source, at: u32) -> Result<Option<u32>> {
        let mut end = at + 1;
        while src.kind(end) != TokenKind::Semi {
            if src.text(end) == "*" && src.kind(end + 1) == TokenKind::Semi && end >= at + 4 && src.kind(end - 1) == TokenKind::Colon && src.kind(end - 2) == TokenKind::Colon {
                let mut copy = Tokens::default();
                (at + 1..end).for_each(|token| copy.copy(src, token));
                self.globs.push(copy.finish(src.path.clone())?);
                return Ok(Some(end + 2));
            }
            if src.kind(end) == TokenKind::OpenBrace && end >= at + 3 && src.kind(end - 1) == TokenKind::Colon {
                return self.note_use_group(src, (at + 1, end));
            }
            if !is_name(src.kind(end)) && src.kind(end) != TokenKind::Colon {
                return Ok(None);
            }
            end += 1;
        }
        let (last, alias) = if end >= at + 4 && src.text(end - 2) == "as" { (end - 3, end - 1) } else { (end - 1, end - 1) };
        if last <= at || !is_name(src.kind(last)) {
            return Ok(None);
        }
        let mut copy = Tokens::default();
        (at + 1..=last).for_each(|token| copy.copy(src, token));
        self.uses.push((plain(src.text(alias)).to_owned(), copy.finish(src.path.clone())?));
        Ok(Some(end + 1))
    }

    /// Notes each `name` or `name as alias` of the one-level group at `open` of a `use prefix::{..};` whose prefix starts at `from`, and returns the token after the `;`.
    fn note_use_group(&mut self, src: &Source, (from, open): (u32, u32)) -> Result<Option<u32>> {
        let close = src.pair(open);
        if src.kind(close + 1) != TokenKind::Semi {
            return Ok(None);
        }
        let mut found = Vec::new();
        for entry in (open + 1..close).collect::<Vec<_>>().split(|&at| src.kind(at) == TokenKind::Comma) {
            match *entry {
                [] => {}
                [name] if is_name(src.kind(name)) => found.push((name, name)),
                [name, word, alias] if is_name(src.kind(name)) && src.text(word) == "as" && is_name(src.kind(alias)) => found.push((name, alias)),
                _ => return Ok(None),
            }
        }
        for (name, alias) in found {
            let mut copy = Tokens::default();
            (from..open).for_each(|token| copy.copy(src, token));
            copy.copy(src, name);
            self.uses.push((plain(src.text(alias)).to_owned(), copy.finish(src.path.clone())?));
        }
        Ok(Some(close + 2))
    }

    fn define(&mut self, src: &Source, name: u32) -> Result<()> {
        self.local(src, (name, src.pair(name + 1)), rules::parse)
    }

    fn define_macro(&mut self, src: &Source, name: u32) -> Result<u32> {
        let open = name + 1;
        let last = if src.kind(open) == TokenKind::OpenParen { src.pair(src.pair(open) + 1) } else { src.pair(open) };
        self.local(src, (name, last), rules::parse_macro)?;
        Ok(last + 1)
    }

    fn local(&mut self, src: &Source, (name, last): (u32, u32), parse: fn(&Source, u32) -> Result<Vec<Rule>>) -> Result<()> {
        let mut copy = Tokens::default();
        (name + 1..=last).for_each(|at| copy.copy(src, at));
        let source = copy.finish(src.path.clone())?;
        let rules = parse(&source, 0)?;
        self.locals.push(Local { name: plain(src.text(name)).to_owned(), source, rules });
        Ok(())
    }

    fn note_extern(&mut self, cx: &Context<'_>, src: &Source, at: u32) -> Option<u32> {
        let orig = plain(src.text(at + 2));
        let (alias, end) = match (src.text(at + 3), src.kind(at + 3)) {
            (_, TokenKind::Semi) => (orig, at + 4),
            ("as", _) if is_name(src.kind(at + 4)) && src.kind(at + 5) == TokenKind::Semi => (plain(src.text(at + 4)), at + 6),
            _ => return None,
        };
        if alias != "_"
            && let Some(root) = extern_crate_root(orig, self.home.krate, &cx.outline.deps[self.home.krate as usize])
        {
            self.externs.push((alias.to_owned(), root));
        }
        Some(end)
    }

    fn resolve_macro(&self, cx: &Context<'_>, src: &Source, path: (u32, u32)) -> Option<ItemRef> {
        if path.0 + 3 <= path.1 && is_name(src.kind(path.0)) && src.kind(path.0 + 1) == TokenKind::Colon {
            let first = plain(src.text(path.0));
            if let Some(&(_, root)) = self.externs.iter().rev().find(|(alias, _)| alias == first) {
                return macro_at(&cx.paths(root.krate), src, (path.0 + 3, path.1), root);
            }
        }
        macro_at(&cx.paths(self.home.krate), src, path, self.home)
    }

    fn resolve_glob(&self, cx: &Context<'_>, src: &Source, path: (u32, u32)) -> Result<Option<ItemRef>> {
        if path.0 != path.1 {
            return Ok(None);
        }
        for prefix in self.globs.iter().rev() {
            let mut full = Tokens::default();
            full.copy_range(prefix, 0, prefix.len());
            full.copy(src, path.1);
            let joined = full.finish(src.path.clone())?;
            if let Some(found) = self.resolve_macro(cx, &joined, (0, prefix.len())) {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    fn expand_one(&self, cx: &mut Context<'_>, run: Run<'_>, path: (u32, u32), open: u32) -> Result<Option<Source>> {
        if run.depth as usize >= crate::RECURSION_LIMIT {
            bail!("macro {} recursed past {} levels", run.src.text(path.1), crate::RECURSION_LIMIT);
        }
        let name = plain(run.src.text(path.1));
        if let Some(local) = self.locals.iter().rev().find(|local| path.0 == path.1 && local.name == name) {
            return mbe::expand(&local.source, &local.rules, run.src, (open, self.home.krate), None).map(Some);
        }
        let Some(found) = self.find_macro(cx, run, path, name)? else {
            return Ok(None);
        };
        if let Some(output) = defined_expand(cx, found, run.src, open) {
            return output.map(Some);
        }
        if let Some((_, ProcKind::Bang)) = found.outer().is_none().then(|| cx.item(found)).and_then(|(src, item)| proc_kind(src, item)) {
            return proc_bang(cx, self.home.krate, run.src, found, open);
        }
        let Some(builtin) = cx.builtin(found) else {
            return Ok(None);
        };
        let mut out = Tokens::default();
        let site = Site { builtin, krate: found.krate, start: path.0, open };
        if !self.builtin(cx, run, site, &mut out)? {
            return Ok(None);
        }
        out.finish(run.src.path.clone()).map(Some)
    }

    fn find_macro(&self, cx: &mut Context<'_>, run: Run<'_>, path: (u32, u32), name: &str) -> Result<Option<ItemRef>> {
        if let Some((_, source)) = self.uses.iter().rev().find(|(alias, _)| path.0 == path.1 && *alias == name) {
            return Ok(self.resolve_macro(cx, source, (0, source.len() - 1)));
        }
        match self.resolve_macro(cx, run.src, path).or_else(|| def_site_macro(cx, run.src, path)) {
            Some(found) => Ok(Some(found)),
            None => self.resolve_glob(cx, run.src, path),
        }
    }

    fn builtin(&self, cx: &mut Context<'_>, run: Run<'_>, site: Site, out: &mut Tokens) -> Result<bool> {
        let (src, open) = (run.src, site.open);
        let input = (open + 1, src.pair(open));
        match site.builtin {
            Builtin::Concat => return self.concat(cx, run, open, out),
            Builtin::Env | Builtin::OptionEnv => return self.env(cx, run, site, out),
            Builtin::IncludeStr | Builtin::IncludeBytes => return self.include(cx, run, site, out),
            Builtin::FormatArgs | Builtin::FormatArgsNl => {
                return self.format_args(cx, run, site, out);
            }
            Builtin::PatternType => return Ok(false),
            Builtin::Line | Builtin::Column | Builtin::File | Builtin::Stringify => position(run, site, input, out),
            Builtin::Cfg => self.cfg(cx, src, input, out)?,
            Builtin::ModulePath => out.lexed(&format!("{:?}", cx.outline.module_path(self.home))),
            Builtin::CfgSelect => return self.cfg_select(cx, run, open, out),
            Builtin::Assert => builtin::assert(src, &src.args(open), site.krate, out),
            Builtin::Asm => builtin::asm(src, &src.args(open), out)?,
            Builtin::Panic | Builtin::Unreachable => self.panic(cx, run, site, out),
        }
        Ok(true)
    }

    fn cfg(&self, cx: &Context<'_>, src: &Source, (from, to): (u32, u32), out: &mut Tokens) -> Result<()> {
        let predicate = (from..to).map(|at| src.text(at)).collect::<Vec<_>>().join(" ");
        let holds = cx.outline.crates[self.home.krate as usize].env.eval(&crate::graph::cfg::parse(&predicate)?) == Truth::Yes;
        out.lexed(if holds { "true" } else { "false" });
        Ok(())
    }

    fn cfg_select(&self, cx: &mut Context<'_>, run: Run<'_>, open: u32, out: &mut Tokens) -> Result<bool> {
        let env = &cx.outline.crates[self.home.krate as usize].env;
        let Some(arm) = crate::outline::mbe::select_arm(run.src, open, env)? else { bail!("no cfg_select! arm applies") };
        if !arm.braced {
            out.lexed("(");
        }
        out.copy_range(run.src, arm.from, arm.to);
        if !arm.braced {
            out.lexed(")");
        }
        Ok(true)
    }

    fn panic(&self, cx: &mut Context<'_>, run: Run<'_>, site: Site, out: &mut Tokens) {
        let edition = if cx.names.crates[self.home.krate as usize].edition.as_str() < "2021" { "2015" } else { "2021" };
        let name = if site.builtin == Builtin::Panic { "panic" } else { "unreachable" };
        builtin::crate_path(out, site.krate, &format!("::panic::{name}_{edition}!"));
        out.lexed("(");
        out.copy_range(run.src, site.open + 1, run.src.pair(site.open));
        out.lexed(")");
    }

    fn concat(&self, cx: &mut Context<'_>, run: Run<'_>, open: u32, out: &mut Tokens) -> Result<bool> {
        let mut text = String::new();
        for range in run.src.args(open) {
            let Some(piece) = self.literal(cx, run, range)? else {
                return Ok(false);
            };
            text.push_str(&piece);
        }
        out.lexed(&format!("{text:?}"));
        Ok(true)
    }

    fn env(&self, cx: &mut Context<'_>, run: Run<'_>, site: Site, out: &mut Tokens) -> Result<bool> {
        let Some(&name) = run.src.args(site.open).first() else { bail!("env! takes the name of a variable") };
        let Some(name) = self.literal(cx, run, name)? else {
            return Ok(false);
        };
        match (site.builtin, cx.outline.crates[self.home.krate as usize].env.var(&name)) {
            (Builtin::Env, Some(value)) => out.lexed(&format!("{value:?}")),
            (Builtin::Env, None) => {
                bail!("environment variable `{name}` is not defined at compile time")
            }
            (_, Some(value)) => {
                builtin::crate_path(out, site.krate, "::option::Option::Some(");
                out.lexed(&format!("{value:?})"));
            }
            (_, None) => builtin::crate_path(out, site.krate, "::option::Option::None::<&'static str>"),
        }
        Ok(true)
    }

    fn include(&self, cx: &mut Context<'_>, run: Run<'_>, site: Site, out: &mut Tokens) -> Result<bool> {
        let Some(&path) = run.src.args(site.open).first() else { bail!("include_str! and include_bytes! take a path") };
        let Some(path) = self.literal(cx, run, path)? else {
            return Ok(false);
        };
        let path = run.call_at(site.start).0.beside(&path);
        let bytes = std::fs::read(&path).map_err(|error| anyhow!("cannot read {}: {error}", path.display()))?;
        match site.builtin {
            Builtin::IncludeStr => out.lexed(&format!("{:?}", String::from_utf8(bytes).map_err(|error| anyhow!("{} is not UTF-8: {error}", path.display()))?)),
            _ => out.lexed(&builtin::byte_string(&bytes)),
        }
        Ok(true)
    }

    fn format_args(&self, cx: &mut Context<'_>, run: Run<'_>, site: Site, out: &mut Tokens) -> Result<bool> {
        let args = run.src.args(site.open);
        let Some((&template, rest)) = args.split_first() else {
            return Ok(false);
        };
        let start = template.0;
        let Some(mut template) = self.literal(cx, run, template)? else {
            return Ok(false);
        };
        if site.builtin == Builtin::FormatArgsNl {
            template.push('\n');
        }
        if rest.is_empty() && builtin::format_literal(&template, site.krate, out) {
            return Ok(true);
        }
        super::format::expand(&template, &super::format::Call::new(run.src, start, rest), site.krate, out)?;
        Ok(true)
    }

    fn literal(&self, cx: &mut Context<'_>, run: Run<'_>, (from, to): (u32, u32)) -> Result<Option<String>> {
        let src = run.src;
        if to > from + 1 && src.kind(from) == TokenKind::OpenParen && src.pair(from) == to - 1 {
            return self.literal(cx, run, (from + 1, to - 1));
        }
        if to == from + 1 {
            return src.literal_text(from);
        }
        if to == from + 2 && src.text(from) == "-" {
            return Ok(src.literal_text(from + 1)?.map(|text| format!("-{text}")));
        }
        let open = src.pair(to - 1);
        if !matches!(src.kind(to - 1), TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::CloseBrace) || open <= from || src.kind(open - 1) != TokenKind::Bang {
            return Ok(None);
        }
        let Some(path) = call_path(src, open - 1, from).filter(|path| path.0 == from) else {
            return Ok(None);
        };
        let call = run.call_at(path.0);
        let Some(output) = self.expand_one(cx, Run { depth: run.depth + 1, call, ..run }, path, open)? else {
            return Ok(None);
        };
        self.literal(cx, Run { src: &output, from: 0, shift: 0, depth: run.depth + 1, call }, (0, output.len()))
    }
}

/// A call of a builtin macro: which builtin, the crate that defines it, the first token of its path and its opening group.
#[derive(Clone, Copy)]
struct Site {
    builtin: Builtin,
    krate: u32,
    start: u32,
    open: u32,
}

impl<'a> Context<'a> {
    /// The tokens that the macro `item` writes in module `home` for the call whose group opens at `open` of `src`, expanded once per call and kept with `src`.
    pub fn macro_output(&mut self, home: ModRef, src: &'a Source, item: ItemRef, open: u32) -> Result<Option<&'a Source>> {
        let key = (home.krate, home.module, open);
        if let Some(found) = src.outputs.get(key)? {
            return Ok(Some(found));
        }
        if let Some(output) = defined_expand(self, item, src, open) {
            return src.outputs.keep(key, output?).map(Some);
        }
        let Some(builtin) = self.builtin(item) else {
            return Ok(None);
        };
        let writer = Writer { out: Tokens::default(), calls: HashMap::default(), offset: 0, home, locals: Vec::new(), uses: Vec::new(), globs: Vec::new(), externs: Vec::new() };
        let start = open.checked_sub(1).and_then(|bang| call_path(src, bang, 0)).map_or(open, |(start, _)| start);
        let run = Run { src, from: 0, shift: 0, depth: 0, call: (src, start) };
        let mut out = Tokens::default();
        if !writer.builtin(self, run, Site { builtin, krate: item.krate, start, open }, &mut out)? {
            return Ok(None);
        }
        src.outputs.keep(key, out.finish(src.path.clone())?).map(Some)
    }

    /// Parses the header of the function, constant or static `item`, with the macro expansions of any body that declares it.
    pub(super) fn parse_signature(&mut self, item: ItemRef) -> Result<sig::Signature> {
        let calls = item.declaring_body().and_then(|outer| self.expansion(outer)).map(|found| &found.calls);
        let config = self.body_config(item, calls);
        let (src, entry) = self.item(item);
        match (entry.kind, entry.name) {
            (ItemKind::Const | ItemKind::Static, Some(name)) => sig::const_signature(src, name, entry.end, config),
            _ => sig::signature(src, entry.start, entry.body.unwrap_or(entry.end), config),
        }
    }

    /// The expansion of the body of `item`, or of the body that declares it, once a pass read that body and found macro calls or items in it.
    pub fn expansion(&self, item: ItemRef) -> Option<&'a Expansion> {
        let item = item.body_owner();
        self.outline.crates[item.krate as usize].files[item.file as usize].expansion(item.item)
    }

    /// The source that the body of `item` is parsed from: its file, with the expansions of its macro calls when it has some.
    pub fn body_src(&self, item: ItemRef) -> &'a Source {
        self.expansion(item).map_or(&self.file(item).source, |found| &found.file.source)
    }

    /// The traits that the body of `root` declares, at any depth.
    pub fn body_declared_traits(&self, root: ItemRef) -> Vec<ItemRef> {
        let items = self.expansion(root).map_or(&[][..], |found| &found.file.items.items[..]);
        items.iter().enumerate().filter(|(_, entry)| entry.kind == ItemKind::Trait).map(|(at, _)| root.inner(at as u32)).collect()
    }

    fn body_config<'c>(&self, item: ItemRef, calls: Option<&'c HashMap<u32, u32>>) -> syntax::Config<'c>
    where
        'a: 'c,
    {
        syntax::Config { calls, env: Some(&self.outline.crates[item.krate as usize].env) }
    }

    /// Parses the body of `item`, expanding its macro calls and scanning the items it declares when no pass did yet.
    pub(super) fn parse_body(&mut self, item: ItemRef) -> Result<Body> {
        let file = self.file(item);
        if item.outer().is_some() || file.body_read(item.item) {
            let found = self.expansion(item);
            return super::body_scope::parse_in(self.body_src(item), self.body_config(item, found.map(|found| &found.calls)), self.item(item).1);
        }
        let Some((from, to)) = body_range(file, item.item) else {
            return super::body_scope::parse_in(&file.source, self.body_config(item, None), self.item(item).1);
        };
        let home = self.module_of(item)?;
        let before = self.asks.count();
        let mut writer = Writer { out: Tokens::default(), calls: HashMap::default(), offset: file.source.len(), home, locals: Vec::new(), uses: Vec::new(), globs: Vec::new(), externs: Vec::new() };
        writer.scan(self, Run { src: &file.source, from, shift: 0, depth: 0, call: (&file.source, from) }, to)?;
        if self.asks.count() > before {
            bail!("the macros that {} calls are not resolved yet", self.name(item));
        }
        let base = self.outline.crates[item.krate as usize].files[item.file as usize].clone();
        let source = std::mem::take(&mut writer.out).finish_over(base)?;
        let body = super::body_scope::parse_in(&source, self.body_config(item, Some(&writer.calls)), self.item(item).1)?;
        let (items, mut scopes, item_scopes) = super::body_scope::declared(&source, self.body_config(item, Some(&writer.calls)), &body)?;
        let (source, items, item_scopes) = self.body_derives(item, before, &mut writer, (source, items, &mut scopes, item_scopes))?;
        let calls = writer.calls;
        let found = (!calls.is_empty() || !items.items.is_empty()).then(|| Box::new(Expansion { file: File::new(source, items, true), calls, scopes: scopes.into(), item_scopes: item_scopes.into() }));
        self.file(item).set_expansion(item.item, found);
        Ok(body)
    }
}

impl Context<'_> {
    /// The body source `source` of `item` with the impls that the builtin and procedural derives of its structs, enums and unions write laid after it, and its `items` with those impls.
    fn body_derives(&mut self, item: ItemRef, before: usize, writer: &mut Writer, (source, mut items, scopes, mut item_scopes): (Source, Items, &mut Vec<Option<u32>>, Vec<u32>)) -> Result<(Source, Items, Vec<u32>)> {
        let derived = self.collect_derives(item.krate, writer, &source, &items, &item_scopes)?;
        if self.asks.count() > before {
            bail!("the macros that {} calls are not resolved yet", self.name(item));
        }
        let Some(base) = source.base.clone().filter(|_| !derived.is_empty()) else {
            return Ok((source, items, item_scopes));
        };
        writer.offset = source.offset;
        writer.out.copy_range(&source, source.offset, source.len());
        let (mut starts, mut chunks) = (Vec::new(), Vec::new());
        for (expanded, parent, scope) in derived {
            let chunk = writer.offset + writer.out.len();
            let mut at = 0;
            while at < expanded.len() {
                starts.push((chunk + at, parent, scope));
                at = scan::item_end(&expanded, at)?;
            }
            writer.out.copy_range(&expanded, 0, expanded.len());
            writer.out.space();
            chunks.push((chunk, expanded));
        }
        for (chunk, expanded) in &chunks {
            writer.scan(self, Run { src: expanded, from: 0, shift: *chunk, depth: 1, call: (expanded, 0) }, expanded.len())?;
        }
        if self.asks.count() > before {
            bail!("the macros that {} calls are not resolved yet", self.name(item));
        }
        let source = std::mem::take(&mut writer.out).finish_over(base)?;
        super::body_scope::scan_nested(&source, self.body_config(item, Some(&writer.calls)), starts, &mut items, scopes, &mut item_scopes)?;
        Ok((source, items, item_scopes))
    }

    fn collect_derives(&mut self, krate: u32, writer: &Writer, source: &Source, items: &Items, item_scopes: &[u32]) -> Result<Vec<(Source, Option<u32>, u32)>> {
        let mut derived = Vec::new();
        for (at, entry) in items.items.iter().enumerate() {
            if !matches!(entry.kind, ItemKind::Struct | ItemKind::Enum | ItemKind::Union) {
                continue;
            }
            let (builtin, foreign) = {
                let (env, deps) = (&self.outline.crates[krate as usize].env, &self.outline.deps[krate as usize][..]);
                derive::derived(source, entry, env, (krate as usize, deps))
            };
            if let Some(expanded) = builtin {
                derived.push((expanded?, entry.parent, item_scopes[at]));
            }
            for path in foreign {
                let run = Run { src: source, from: 0, shift: 0, depth: 0, call: (source, path.0) };
                let found = writer.find_macro(self, run, path, plain(source.text(path.1)))?;
                if let Some(expanded) = self.proc_derive(krate, found, source, entry)? {
                    derived.push((expanded, entry.parent, item_scopes[at]));
                }
            }
        }
        Ok(derived)
    }

    fn proc_derive(&mut self, krate: u32, found: Option<ItemRef>, source: &Source, entry: &crate::outline::item::Item) -> Result<Option<Source>> {
        let Some(run) = found.filter(|run| run.outer().is_none()) else {
            return Ok(None);
        };
        let (def_src, def_item) = self.item(run);
        let Some((_, ProcKind::Derive)) = proc_kind(def_src, def_item) else {
            return Ok(None);
        };
        let key = (run, crate::macros::derive_input(source, entry, &self.outline.crates[krate as usize].env));
        match self.outline.procs.get(&key) {
            Some(Ok(text)) => Source::parse(source.path.clone(), text.clone()).map(Some),
            Some(Err(error)) => bail!("{}: {error}", self.name(run)),
            None => {
                self.asks.procs.borrow_mut().push(key);
                Ok(None)
            }
        }
    }
}

/// The items that the item statement from `start` up to `end` declares in `items`, with the functions of an extern block it is.
pub fn stmt_items(items: &[crate::outline::item::Item], (start, end): (u32, u32)) -> impl Iterator<Item = usize> + '_ {
    let in_block = |entry: &crate::outline::item::Item| entry.parent.is_none_or(|parent| matches!(items[parent as usize].kind, ItemKind::Fn | ItemKind::Const | ItemKind::Static));
    let top = items.iter().position(|entry| in_block(entry) && entry.start >= start && entry.start < end);
    let block = top.filter(|&at| items[at].kind == ItemKind::ExternBlock);
    let children = items.iter().enumerate().filter(move |(_, entry)| block.is_some() && entry.parent == block.map(|at| at as u32)).map(|(at, _)| at);
    top.into_iter().chain(children)
}

fn position(run: Run<'_>, site: Site, input: (u32, u32), out: &mut Tokens) {
    let (src, at) = run.call_at(site.start);
    match site.builtin {
        Builtin::Line => out.lexed(&format!("{}u32", src.line(at))),
        Builtin::Column => out.lexed(&format!("{}u32", src.column(at))),
        Builtin::File => out.lexed(&format!("{:?}", src.path.display().to_string())),
        _ => out.lexed(&format!("{:?}", run.src.spelled(input))),
    }
}
