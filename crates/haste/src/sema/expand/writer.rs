use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;
use rustc_lexer::TokenKind;

use crate::body::cursor::{PATH_KEYWORDS, RESERVED, is_name};
use crate::names::collect::proc_kind;
use crate::names::resolve::macro_at;
use crate::names::{ItemRef, ModRef, plain};
use crate::outline::item::{Attr, ItemKind, ProcKind};
use crate::outline::rules::Rule;
use crate::outline::source::{Mark, Source, Tokens, opens};
use crate::outline::{attr, mbe};
use crate::sema::Context;

use super::builtins::Site;

pub(super) struct Local {
    pub(super) name: String,
    pub(super) source: Source,
    pub(super) rules: Vec<Rule>,
}

pub(super) struct Writer {
    pub(super) out: Tokens,
    pub(super) calls: HashMap<u32, u32>,
    pub(super) offset: u32,
    pub(super) home: ModRef,
    pub(super) locals: Vec<Local>,
    pub(super) uses: Vec<(String, Source)>,
    pub(super) globs: Vec<Source>,
    pub(super) externs: Vec<(String, ModRef)>,
}

#[derive(Clone, Copy)]
pub(super) struct Run<'s> {
    pub(super) src: &'s Source,
    pub(super) from: u32,
    pub(super) shift: u32,
    pub(super) depth: u32,
    pub(super) call: (&'s Source, u32),
}

impl<'s> Run<'s> {
    pub(super) fn call_at(self, start: u32) -> (&'s Source, u32) {
        if self.depth == 0 {
            (self.src, start)
        } else {
            self.call
        }
    }
}

pub(super) fn call_path(src: &Source, bang: u32, from: u32) -> Option<(u32, u32)> {
    let name = bang.checked_sub(1).filter(|name| *name >= from)?;
    let kind = src.kind(name);
    if !is_name(kind) || (kind == TokenKind::Ident && RESERVED.contains(&src.text(name))) {
        return None;
    }
    let mut start = name;
    while start >= from + 3
        && src.kind(start - 1) == TokenKind::Colon
        && src.kind(start - 2) == TokenKind::Colon
        && path_word(src, start - 3)
    {
        start -= 3;
    }
    if start >= from + 2
        && src.kind(start - 1) == TokenKind::Colon
        && src.kind(start - 2) == TokenKind::Colon
    {
        start -= 2;
    } else if start > from && src.text(start) == "crate" && src.kind(start - 1) == TokenKind::Dollar
    {
        start -= 1;
    }
    Some((start, name))
}

fn path_word(src: &Source, at: u32) -> bool {
    let kind = src.kind(at);
    is_name(kind)
        && (kind != TokenKind::Ident
            || !RESERVED.contains(&src.text(at))
            || PATH_KEYWORDS.contains(&src.text(at)))
}

pub(super) fn def_site_macro(
    cx: &mut Context<'_>,
    src: &Source,
    path: (u32, u32),
) -> Option<ItemRef> {
    let mark_at = if src.kind(path.0) == TokenKind::Colon && path.0 + 2 <= path.1 {
        path.0 + 2
    } else {
        path.0
    };
    let Some(Mark::Home { krate, file, item }) = src.mark(mark_at) else {
        return None;
    };
    let module = cx.module_of(ItemRef { krate, file, item }).ok()?;
    macro_at(&cx.paths(module.krate), src, path, module)
}

pub(super) fn defined_expand(
    cx: &Context<'_>,
    found: ItemRef,
    src: &Source,
    open: u32,
) -> Option<Result<Source>> {
    let definition = cx.outline.crates[found.krate as usize]
        .defined
        .get(&(found.file, found.item))?;
    let home = (cx.item(found).1.kind == ItemKind::Macro).then_some(Mark::Home {
        krate: found.krate,
        file: found.file,
        item: found.item,
    });
    Some(mbe::expand(
        &definition.file.source,
        &definition.rules,
        src,
        (open, definition.krate as u32),
        home,
    ))
}

fn proc_bang(
    cx: &mut Context<'_>,
    krate: u32,
    src: &Source,
    run: ItemRef,
    open: u32,
) -> Result<Option<Source>> {
    let input =
        crate::macros::bang_input(&cx.outline.deps[krate as usize], krate as usize, src, open);
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

impl Writer {
    pub(super) fn new(home: ModRef, offset: u32) -> Self {
        Writer {
            out: Tokens::default(),
            calls: HashMap::default(),
            offset,
            home,
            locals: Vec::new(),
            uses: Vec::new(),
            globs: Vec::new(),
            externs: Vec::new(),
        }
    }

    pub(super) fn scan(&mut self, cx: &mut Context<'_>, run: Run<'_>, to: u32) -> Result<()> {
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
        if src.kind(at) == TokenKind::Pound
            && let Some(next) = self.configured_out(cx, src, at)
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

    fn configured_out(&self, cx: &Context<'_>, src: &Source, at: u32) -> Option<u32> {
        let mut attrs = Vec::new();
        let Ok(word) = attr::attrs(src, at, false, &mut attrs) else {
            return None;
        };
        if word == at
            || cx.outline.crates[self.home.krate as usize]
                .env
                .enabled(&attrs)
        {
            return None;
        }
        if src.kind(word) == TokenKind::OpenBrace {
            return Some(src.pair(word) + 1);
        }
        let mut end = word;
        while is_name(src.kind(end)) || src.kind(end) == TokenKind::Colon {
            end += 1;
        }
        (end > word && src.kind(end) == TokenKind::Bang && opens(src.kind(end + 1)))
            .then(|| src.pair(end + 1) + 1)
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

    fn call(
        &mut self,
        cx: &mut Context<'_>,
        run: Run<'_>,
        path: (u32, u32),
        open: u32,
    ) -> Result<()> {
        let Some(output) = self.expand_one(cx, run, path, open)? else {
            return Ok(());
        };
        let block = self.offset + self.out.len();
        self.calls.insert(run.shift + open, block);
        self.out.lexed("{");
        self.out.copy_range(&output, 0, output.len());
        self.out.space();
        self.out.lexed("}");
        self.scan(
            cx,
            Run {
                src: &output,
                from: 0,
                shift: block + 1,
                depth: run.depth + 1,
                call: run.call_at(path.0),
            },
            output.len(),
        )
    }

    fn module_file(&mut self, cx: &mut Context<'_>, run: Run<'_>, at: u32) -> Result<Option<u32>> {
        let src = run.src;
        let mut attrs = Vec::new();
        let mut word = attr::attrs(src, at, false, &mut attrs)?;
        let Some(path) = attrs.iter().find_map(|found| {
            if let Attr::Path(path) = found {
                Some(path.clone())
            } else {
                None
            }
        }) else {
            return Ok(None);
        };
        if src.text(word) == "pub" {
            word += 1;
            if src.kind(word) == TokenKind::OpenParen {
                word = src.pair(word) + 1;
            }
        }
        if src.text(word) != "mod"
            || !is_name(src.kind(word + 1))
            || src.kind(word + 2) != TokenKind::Semi
        {
            return Ok(None);
        }
        let output = Source::read(&run.call_at(at).0.beside(&path))?;
        let block = self.offset + self.out.len();
        self.calls.insert(run.shift + word + 2, block);
        self.out.lexed("{");
        self.out.copy_range(&output, 0, output.len());
        self.out.space();
        self.out.lexed("}");
        self.scan(
            cx,
            Run {
                src: &output,
                from: 0,
                shift: block + 1,
                depth: run.depth,
                call: (&output, 0),
            },
            output.len(),
        )?;
        Ok(Some(word + 3))
    }

    pub(super) fn expand_one(
        &self,
        cx: &mut Context<'_>,
        run: Run<'_>,
        path: (u32, u32),
        open: u32,
    ) -> Result<Option<Source>> {
        if run.depth as usize >= crate::RECURSION_LIMIT {
            bail!(
                "macro {} recursed past {} levels",
                run.src.text(path.1),
                crate::RECURSION_LIMIT
            );
        }
        let name = plain(run.src.text(path.1));
        if let Some(local) = self
            .locals
            .iter()
            .rev()
            .find(|local| path.0 == path.1 && local.name == name)
        {
            return mbe::expand(
                &local.source,
                &local.rules,
                run.src,
                (open, self.home.krate),
                None,
            )
            .map(Some);
        }
        let Some(found) = self.find_macro(cx, run, path, name)? else {
            return Ok(None);
        };
        if let Some(output) = defined_expand(cx, found, run.src, open) {
            return output.map(Some);
        }
        if let Some((_, ProcKind::Bang)) = found
            .outer()
            .is_none()
            .then(|| cx.item(found))
            .and_then(|(src, item)| proc_kind(src, item))
        {
            return proc_bang(cx, self.home.krate, run.src, found, open);
        }
        let Some(builtin) = cx.builtin(found) else {
            return Ok(None);
        };
        let mut out = Tokens::default();
        let site = Site {
            builtin,
            krate: found.krate,
            start: path.0,
            open,
        };
        if !self.builtin(cx, run, site, &mut out)? {
            return Ok(None);
        }
        out.finish(run.src.path.clone()).map(Some)
    }
}
