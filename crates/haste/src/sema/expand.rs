use anyhow::{Result, anyhow, bail};
use rustc_hash::FxHashMap as HashMap;
use rustc_lexer::TokenKind;

use crate::body::cursor::{RESERVED, is_name};
use crate::body::{self as syntax, Body, Stmt, sig};
use crate::names::collect::proc_kind;
use crate::names::resolve::macro_at;
use crate::names::{ItemRef, ModRef, plain};
use crate::outline::item::{ItemKind, Items, ProcKind};
use crate::outline::rules::{self, Rule};
use crate::outline::{derive, mbe, scan};
use crate::outline::source::{Mark, Source, Tokens, opens};
use crate::outline::tree::{Expansion, File, Truth};

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
    let Some(Mark::Home { krate, file, item }) = src.mark(path.0).filter(|_| path.0 == path.1) else { return None };
    let module = cx.module_of(ItemRef { krate, file, item }).ok()?;
    macro_at(&cx.paths(module.krate), src, path, module)
}

/// The output of the `macro_rules!` or `macro` item `found` called with the group at `open` of `src`, or none when `found` is not one.
fn defined_expand(cx: &Context<'_>, found: ItemRef, src: &Source, open: u32) -> Option<Result<Source>> {
    let definition = cx.outline.crates[found.krate as usize].defined.get(&(found.file, found.item))?;
    let home = (cx.item(found).1.kind == ItemKind::Macro).then_some(Mark::Home { krate: found.krate, file: found.file, item: found.item });
    Some(mbe::expand(&definition.file.source, &definition.rules, src, (open, definition.krate as u32), home))
}

/// The output of the procedural macro `run` called with the delimiter at `open`, once the macro expander ran it; until then the call asks for it.
fn proc_bang(cx: &mut Context<'_>, src: &Source, run: ItemRef, open: u32) -> Result<Option<Source>> {
    let close = src.pair(open);
    let input = if close > open + 1 { src.span(open + 1, close - 1).to_owned() } else { String::new() };
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
        if src.kind(at) != TokenKind::Bang {
            if src.text(at) == "macro" && is_name(src.kind(at + 1)) && opens(src.kind(at + 2)) {
                return self.define_macro(src, at + 1);
            }
            if src.text(at) == "use"
                && let Some(next) = self.note_use(src, at)?
            {
                return Ok(next);
            }
            return Ok(at + 1);
        }
        let next = at + 1;
        if at > run.from && src.text(at - 1) == "macro_rules" && is_name(src.kind(next)) {
            if !opens(src.kind(next + 1)) {
                return Ok(next + 1);
            }
            self.define(src, next)?;
            return Ok(src.pair(next + 1) + 1);
        }
        let Some(path) = call_path(src, at, run.from).filter(|_| opens(src.kind(next))) else { return Ok(next) };
        self.call(cx, run, path, next)?;
        Ok(src.pair(next) + 1)
    }

    fn call(&mut self, cx: &mut Context<'_>, run: Run<'_>, path: (u32, u32), open: u32) -> Result<()> {
        let Some(output) = self.expand_one(cx, run, path, open)? else { return Ok(()) };
        let block = self.offset + self.out.len();
        self.calls.insert(run.shift + open, block);
        self.out.lexed("{");
        (0..output.len()).for_each(|at| self.out.copy(&output, at));
        self.out.space();
        self.out.lexed("}");
        self.scan(cx, Run { src: &output, from: 0, shift: block + 1, depth: run.depth + 1, call: run.call_at(path.0) }, output.len())
    }

    fn note_use(&mut self, src: &Source, at: u32) -> Result<Option<u32>> {
        let mut end = at + 1;
        while src.kind(end) != TokenKind::Semi {
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

    fn expand_one(&self, cx: &mut Context<'_>, run: Run<'_>, path: (u32, u32), open: u32) -> Result<Option<Source>> {
        if run.depth as usize >= crate::RECURSION_LIMIT {
            bail!("macro {} recursed past {} levels", run.src.text(path.1), crate::RECURSION_LIMIT);
        }
        let name = plain(run.src.text(path.1));
        if let Some(local) = self.locals.iter().rev().find(|local| path.0 == path.1 && local.name == name) {
            return mbe::expand(&local.source, &local.rules, run.src, (open, self.home.krate), None).map(Some);
        }
        let imported = self.uses.iter().rev().find(|(alias, _)| path.0 == path.1 && *alias == name).map(|(_, source)| source);
        let found = match imported {
            Some(source) => macro_at(&cx.paths(self.home.krate), source, (0, source.len() - 1), self.home),
            None => macro_at(&cx.paths(self.home.krate), run.src, path, self.home).or_else(|| def_site_macro(cx, run.src, path)),
        };
        let Some(found) = found else { return Ok(None) };
        if let Some(output) = defined_expand(cx, found, run.src, open) {
            return output.map(Some);
        }
        if let Some((_, ProcKind::Bang)) = found.outer().is_none().then(|| cx.item(found)).and_then(|(src, item)| proc_kind(src, item)) {
            return proc_bang(cx, run.src, found, open);
        }
        let Some(builtin) = cx.builtin(found) else { return Ok(None) };
        let mut out = Tokens::default();
        let site = Site { builtin, krate: found.krate, start: path.0, open };
        if !self.builtin(cx, run, site, &mut out)? {
            return Ok(None);
        }
        out.finish(run.src.path.clone()).map(Some)
    }

    fn builtin(&self, cx: &mut Context<'_>, run: Run<'_>, site: Site, out: &mut Tokens) -> Result<bool> {
        let (src, open) = (run.src, site.open);
        let input = (open + 1, src.pair(open));
        match site.builtin {
            Builtin::Concat => return self.concat(cx, run, open, out),
            Builtin::Env | Builtin::OptionEnv => return self.env(cx, run, site, out),
            Builtin::IncludeStr | Builtin::IncludeBytes => return self.include(cx, run, site, out),
            Builtin::FormatArgs | Builtin::FormatArgsNl => return self.format_args(cx, run, site, out),
            Builtin::PatternType => return Ok(false),
            Builtin::Line | Builtin::Column | Builtin::File | Builtin::Stringify => position(run, site, input, out),
            Builtin::Cfg => self.cfg(cx, src, input, out)?,
            Builtin::ModulePath => out.lexed(&format!("{:?}", cx.outline.module_path(self.home))),
            Builtin::CfgSelect => return self.cfg_select(cx, run, open, out),
            Builtin::Assert => builtin::assert(src, &src.args(open), site.krate, out),
            Builtin::Asm => builtin::asm_barrier(src, &src.args(open), out)?,
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
        (arm.from..arm.to).for_each(|at| out.copy(run.src, at));
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
        (site.open + 1..run.src.pair(site.open)).for_each(|at| out.copy(run.src, at));
        out.lexed(")");
    }

    fn concat(&self, cx: &mut Context<'_>, run: Run<'_>, open: u32, out: &mut Tokens) -> Result<bool> {
        let mut text = String::new();
        for range in run.src.args(open) {
            let Some(piece) = self.literal(cx, run, range)? else { return Ok(false) };
            text.push_str(&piece);
        }
        out.lexed(&format!("{text:?}"));
        Ok(true)
    }

    fn env(&self, cx: &mut Context<'_>, run: Run<'_>, site: Site, out: &mut Tokens) -> Result<bool> {
        let Some(&name) = run.src.args(site.open).first() else { bail!("env! takes the name of a variable") };
        let Some(name) = self.literal(cx, run, name)? else { return Ok(false) };
        match (site.builtin, cx.outline.crates[self.home.krate as usize].env.var(&name)) {
            (Builtin::Env, Some(value)) => out.lexed(&format!("{value:?}")),
            (Builtin::Env, None) => bail!("environment variable `{name}` is not defined at compile time"),
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
        let Some(path) = self.literal(cx, run, path)? else { return Ok(false) };
        let path = run.call_at(site.start).0.beside(&path);
        let bytes = std::fs::read(&path).map_err(|error| anyhow!("cannot read {}: {error}", path.display()))?;
        match site.builtin {
            Builtin::IncludeStr => out.lexed(&format!("{:?}", String::from_utf8(bytes).map_err(|_| anyhow!("{} is not UTF-8", path.display()))?)),
            _ => out.lexed(&builtin::byte_string(&bytes)),
        }
        Ok(true)
    }

    fn format_args(&self, cx: &mut Context<'_>, run: Run<'_>, site: Site, out: &mut Tokens) -> Result<bool> {
        let args = run.src.args(site.open);
        let Some((&template, rest)) = args.split_first() else { return Ok(false) };
        let start = template.0;
        let Some(mut template) = self.literal(cx, run, template)? else { return Ok(false) };
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
        let Some(path) = call_path(src, open - 1, from).filter(|path| path.0 == from) else { return Ok(None) };
        let call = run.call_at(path.0);
        let Some(output) = self.expand_one(cx, Run { depth: run.depth + 1, call, ..run }, path, open)? else { return Ok(None) };
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
    /// The tokens that the macro `item` writes for the call whose group opens at `open` of `src`, expanded once per call and kept with `src`; none when `item` is not a `macro_rules!` or `macro`.
    pub fn macro_output(&self, src: &'a Source, item: ItemRef, open: u32) -> Result<Option<&'a Source>> {
        if let Some(found) = src.outputs.get(open)? {
            return Ok(Some(found));
        }
        let Some(output) = defined_expand(self, item, src, open) else { return Ok(None) };
        src.outputs.keep(open, output?).map(Some)
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
            return parse_in(self.body_src(item), self.body_config(item, found.map(|found| &found.calls)), self.item(item).1);
        }
        let Some((from, to)) = body_range(file, item.item) else { return parse_in(&file.source, self.body_config(item, None), self.item(item).1) };
        let home = self.module_of(item)?;
        let before = self.asks.count();
        let mut writer = Writer { out: Tokens::default(), calls: HashMap::default(), offset: file.source.len(), home, locals: Vec::new(), uses: Vec::new() };
        writer.scan(self, Run { src: &file.source, from, shift: 0, depth: 0, call: (&file.source, from) }, to)?;
        if self.asks.count() > before {
            bail!("the macros that {} calls are not resolved yet", self.name(item));
        }
        let (source, calls) = (writer.out.finish_over(self.outline.crates[item.krate as usize].files[item.file as usize].clone())?, writer.calls);
        let config = self.body_config(item, Some(&calls));
        let body = parse_in(&source, config, self.item(item).1)?;
        let items = declared(&source, config, &body)?;
        let (source, items) = self.body_derives(item.krate, (source, items), &calls)?;
        let found = (!calls.is_empty() || !items.items.is_empty()).then(|| Box::new(Expansion { file: File::new(source, items, true), calls }));
        file.set_expansion(item.item, found);
        Ok(body)
    }
}

impl Context<'_> {
    /// The body source `source` of crate `krate` with the impls that the builtin derives of its structs, enums and unions write laid after it, and its `items` with those impls.
    fn body_derives(&self, krate: u32, (source, items): (Source, Items), calls: &HashMap<u32, u32>) -> Result<(Source, Items)> {
        let (env, deps) = (&self.outline.crates[krate as usize].env, &self.outline.deps[krate as usize][..]);
        let adts = items.items.iter().filter(|entry| matches!(entry.kind, ItemKind::Struct | ItemKind::Enum | ItemKind::Union));
        let derived: Vec<(Source, Option<u32>)> = adts.filter_map(|entry| Some((derive::derived(&source, entry, env, (krate as usize, deps)).0?, entry.parent))).map(|(expanded, parent)| Ok((expanded?, parent))).collect::<Result<_>>()?;
        let Some(base) = source.base.clone().filter(|_| !derived.is_empty()) else { return Ok((source, items)) };
        let mut out = Tokens::default();
        (source.offset..source.len()).for_each(|at| out.copy(&source, at));
        let mut starts = Vec::new();
        for (expanded, parent) in derived {
            let mut at = 0;
            while at < expanded.len() {
                starts.push((source.offset + out.len() + at, parent));
                at = scan::item_end(&expanded, at)?;
            }
            (0..expanded.len()).for_each(|at| out.copy(&expanded, at));
            out.space();
        }
        let source = out.finish_over(base)?;
        let items = scan::body_items(&source, starts, (items, Some(calls)))?;
        Ok((source, items))
    }
}

fn item_starts(body: &Body) -> impl Iterator<Item = u32> + '_ {
    body.stmts.iter().filter_map(|stmt| if let Stmt::Item(start, _) = stmt { Some(*start) } else { None })
}

/// The items that `body` declares, and those that the bodies of its functions declare in turn.
fn declared(source: &Source, config: syntax::Config<'_>, body: &Body) -> Result<Items> {
    let mut items = Items::default();
    let mut starts: Vec<(u32, Option<u32>)> = item_starts(body).map(|at| (at, None)).collect();
    while !starts.is_empty() {
        let first = items.items.len();
        items = scan::body_items(source, std::mem::take(&mut starts), (items, config.calls))?;
        for (index, entry) in items.items.iter().enumerate().skip(first) {
            let has_body = match entry.kind {
                ItemKind::Fn => entry.body.is_some(),
                ItemKind::Const | ItemKind::Static => true,
                _ => false,
            };
            if !has_body {
                continue;
            }
            if let Ok(inner) = parse_in(source, config, entry) {
                starts.extend(item_starts(&inner).map(|at| (at, Some(index as u32))));
            }
        }
    }
    Ok(items)
}

/// The items that the item statement from `start` up to `end` declares in `items`, with the functions of an extern block it is.
pub fn stmt_items(items: &[crate::outline::item::Item], (start, end): (u32, u32)) -> impl Iterator<Item = usize> + '_ {
    let in_block = |entry: &crate::outline::item::Item| entry.parent.is_none_or(|parent| matches!(items[parent as usize].kind, ItemKind::Fn | ItemKind::Const | ItemKind::Static));
    let top = items.iter().position(|entry| in_block(entry) && entry.start >= start && entry.start < end);
    let block = top.filter(|&at| items[at].kind == ItemKind::ExternBlock);
    let children = items.iter().enumerate().filter(move |(_, entry)| block.is_some() && entry.parent == block.map(|at| at as u32)).map(|(at, _)| at);
    top.into_iter().chain(children)
}

fn parse_in(src: &Source, config: syntax::Config<'_>, entry: &crate::outline::item::Item) -> Result<Body> {
    match (entry.kind, entry.name) {
        (ItemKind::Const | ItemKind::Static, Some(name)) => {
            let from = sig::initializer(src, name, entry.end).ok_or_else(|| anyhow!("constant {} has no value", src.text(name)))?;
            syntax::parse_expr(src, from, entry.end - 1, config)
        }
        _ => syntax::parse(src, entry.body.ok_or_else(|| anyhow!("function {} has no body", src.text(entry.name.unwrap_or(entry.start))))?, config),
    }
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
