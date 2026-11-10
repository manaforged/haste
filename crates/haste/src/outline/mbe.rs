use rustc_hash::FxHashMap as HashMap;

use anyhow::{Result, bail};
use rustc_lexer::TokenKind;

use super::fragment;
use super::rules::{Frag, Meta, Op, Piece, Rule};
use super::source::{Mark, Source, Tokens, opens};

/// What a metavariable matched: a token range of the input, grouped when it is a compound expression, or one binding per repetition.
#[derive(Clone, Debug)]
pub enum Bind {
    One(u32, u32, bool),
    Many(Vec<Bind>),
}

type Binds = HashMap<String, Bind>;

struct Matcher<'a> {
    def: &'a Source,
    input: &'a Source,
}

fn names(def: &Source, pieces: &[Piece], into: &mut Vec<String>) {
    for piece in pieces {
        match piece {
            Piece::Var(name, _) | Piece::Meta(Meta::Ignore(name) | Meta::Count(name)) => into.push(def.text(*name).to_owned()),
            Piece::Group(_, inner) | Piece::Repeat(inner, ..) => names(def, inner, into),
            Piece::Token(_) | Piece::Crate | Piece::CratePath | Piece::Meta(_) => {}
        }
    }
}

impl Matcher<'_> {
    fn seq(&self, pieces: &[Piece], mut at: u32, end: u32, binds: &mut Binds) -> Option<u32> {
        for piece in pieces {
            at = self.piece(piece, at, end, binds)?;
        }
        Some(at)
    }

    fn sep(&self, sep: (u32, u32), at: u32, end: u32) -> Option<u32> {
        let len = sep.1 - sep.0 + 1;
        let same = (0..len).all(|offset| at + offset < end && self.input.text(at + offset) == self.def.text(sep.0 + offset));
        same.then_some(at + len)
    }

    fn piece(&self, piece: &Piece, at: u32, end: u32, binds: &mut Binds) -> Option<u32> {
        match piece {
            Piece::Token(token) => (at < end && self.input.text(at) == self.def.text(*token)).then_some(at + 1),
            Piece::Group(open, inner) => {
                if at >= end || self.input.kind(at) != self.def.kind(*open) {
                    return None;
                }
                let close = self.input.pair(at);
                (self.seq(inner, at + 1, close, binds)? == close).then_some(close + 1)
            }
            Piece::Var(name, Some(frag)) => {
                let to = fragment::extent(self.input, *frag, at, end)?;
                let grouped = *frag == Frag::Expr && fragment::tree(self.input, at) != to;
                binds.insert(self.def.text(*name).to_owned(), Bind::One(at, to, grouped));
                Some(to)
            }
            Piece::Var(_, None) | Piece::Crate | Piece::CratePath | Piece::Meta(_) => None,
            Piece::Repeat(inner, sep, op) => self.repeat(inner, *sep, *op, (at, end), binds),
        }
    }

    fn repeat(&self, inner: &[Piece], sep: Option<(u32, u32)>, op: Op, (mut at, end): (u32, u32), binds: &mut Binds) -> Option<u32> {
        let mut rounds: Vec<Binds> = Vec::new();
        while !(op == Op::Maybe && rounds.len() == 1) {
            let start = match sep {
                Some(sep) if !rounds.is_empty() => match self.sep(sep, at, end) {
                    Some(next) => next,
                    None => break,
                },
                _ => at,
            };
            let mut local = Binds::default();
            match self.seq(inner, start, end, &mut local) {
                Some(next) if next > start => {
                    rounds.push(local);
                    at = next;
                }
                _ => break,
            }
        }
        if op == Op::Some && rounds.is_empty() {
            return None;
        }
        let mut found = Vec::new();
        names(self.def, inner, &mut found);
        for name in found {
            let each = rounds.iter_mut().map(|round| round.remove(&name).unwrap_or(Bind::Many(Vec::new()))).collect();
            binds.insert(name, Bind::Many(each));
        }
        Some(at)
    }
}

struct Writer<'a> {
    def: &'a Source,
    input: &'a Source,
    binds: Binds,
    lens: Vec<usize>,
    krate: u32,
    home: Option<Mark>,
    scope: u32,
    out: Tokens,
}

/// The next `macro_rules!` expansion's hygiene scope; zero stays for names the program spelled.
static SCOPES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

fn leaves(bind: &Bind) -> usize {
    match bind {
        Bind::One(..) => 1,
        Bind::Many(each) => each.iter().map(leaves).sum(),
    }
}

fn lookup<'b>(binds: &'b Binds, name: &str, rounds: &[usize]) -> Option<&'b Bind> {
    let mut bind = binds.get(name)?;
    for &round in rounds {
        match bind {
            Bind::Many(each) => bind = each.get(round)?,
            Bind::One(..) => break,
        }
    }
    Some(bind)
}

impl Writer<'_> {
    fn emit(&mut self, pieces: &[Piece], rounds: &mut Vec<usize>) -> Result<()> {
        for piece in pieces {
            match piece {
                Piece::Token(token) => {
                    let scope = Mark::Scope { id: self.scope, parent: self.def.scope(*token).0 };
                    let home = match self.def.kind(*token) {
                        TokenKind::Ident | TokenKind::RawIdent => Some(self.home.unwrap_or(scope)),
                        TokenKind::Literal { .. } => Some(scope),
                        _ => None,
                    };
                    self.out.copy_marked(self.def, *token, home);
                }
                Piece::Group(open, inner) => {
                    self.out.copy(self.def, *open);
                    self.emit(inner, rounds)?;
                    self.out.copy(self.def, self.def.pair(*open));
                }
                Piece::Crate => self.out.dollar_crate(self.krate),
                Piece::CratePath => {
                    self.out.dollar_crate(self.krate);
                    self.out.lexed("::");
                }
                Piece::Meta(meta) => self.meta(meta, rounds),
                Piece::Var(name, _) => self.var(*name, rounds)?,
                Piece::Repeat(inner, sep, _) => self.repeat(inner, *sep, rounds)?,
            }
        }
        Ok(())
    }

    fn var(&mut self, name: u32, rounds: &[usize]) -> Result<()> {
        match lookup(&self.binds, self.def.text(name), rounds) {
            Some(&Bind::One(from, to, grouped)) => {
                if grouped {
                    self.out.lexed("(");
                }
                (from..to).for_each(|at| self.out.copy(self.input, at));
                if grouped {
                    self.out.lexed(")");
                }
                self.out.space();
            }
            Some(Bind::Many(_)) => bail!("metavariable ${} is still repeating", self.def.text(name)),
            None => {
                self.out.copy(self.def, name - 1);
                self.out.copy(self.def, name);
            }
        }
        Ok(())
    }

    fn meta(&mut self, meta: &Meta, rounds: &[usize]) {
        let value = match meta {
            Meta::Ignore(_) => return,
            Meta::Index(depth) => rounds.len().checked_sub(depth + 1).map_or(0, |at| rounds[at]),
            Meta::Len(depth) => self.lens.len().checked_sub(depth + 1).map_or(0, |at| self.lens[at]),
            Meta::Count(name) => lookup(&self.binds, self.def.text(*name), rounds).map_or(0, leaves),
            Meta::Concat(parts) => {
                let text: String = parts.iter().map(|&(variable, token)| self.concat_part(variable, token, rounds)).collect();
                self.out.lexed(&text);
                return;
            }
        };
        self.out.lexed(&value.to_string());
    }

    fn concat_part(&self, variable: bool, token: u32, rounds: &[usize]) -> String {
        let text = self.def.text(token);
        let bound = if variable { lookup(&self.binds, text, rounds) } else { None };
        let part = match bound {
            Some(&Bind::One(from, ..)) => self.input.text(from),
            _ => text,
        };
        part.trim_start_matches("r#").trim_matches('"').to_owned()
    }

    fn count(&self, inner: &[Piece], rounds: &[usize]) -> Option<usize> {
        let mut found = Vec::new();
        names(self.def, inner, &mut found);
        found.iter().find_map(|name| match lookup(&self.binds, name, rounds) {
            Some(Bind::Many(each)) => Some(each.len()),
            _ => None,
        })
    }

    fn repeat(&mut self, inner: &[Piece], sep: Option<(u32, u32)>, rounds: &mut Vec<usize>) -> Result<()> {
        let Some(count) = self.count(inner, rounds) else { bail!("repetition in a transcriber uses no repeating metavariable") };
        for round in 0..count {
            if round > 0
                && let Some((first, last)) = sep
            {
                (first..=last).for_each(|at| self.out.copy(self.def, at));
            }
            rounds.push(round);
            self.lens.push(count);
            self.emit(inner, rounds)?;
            self.lens.pop();
            rounds.pop();
        }
        Ok(())
    }
}

/// Expands the call whose input is the group opening at `open` in `input` with the first rule that matches, written by a macro of crate `krate` whose own names carry the mark `home`.
pub fn expand(def: &Source, rules: &[Rule], input: &Source, (open, krate): (u32, u32), home: Option<Mark>) -> Result<Source> {
    let matcher = Matcher { def, input };
    let close = input.pair(open);
    for rule in rules {
        let mut binds = Binds::default();
        if matcher.seq(&rule.matcher, open + 1, close, &mut binds) == Some(close) {
            let mut writer = Writer { def, input, binds, lens: Vec::new(), krate, home, scope: SCOPES.fetch_add(1, std::sync::atomic::Ordering::Relaxed), out: Tokens::default() };
            writer.emit(&rule.body, &mut Vec::new())?;
            return writer.out.finish(input.path.clone());
        }
    }
    bail!("no rule of the macro matches the call at byte {} of {}", input.byte(open), input.path.display())
}

/// The tokens of the first `cfg_select!` arm that `env` may enable, or `None` when no arm applies.
pub fn select(src: &Source, open: u32, env: &super::tree::Env) -> Result<Option<Source>> {
    let Some(arm) = select_arm(src, open, env)? else { return Ok(None) };
    if !arm.braced {
        bail!("a cfg_select! arm of items needs braces at byte {} of {}", src.byte(arm.from), src.path.display());
    }
    let mut out = Tokens::default();
    (arm.from..arm.to).for_each(|token| out.copy(src, token));
    out.finish(src.path.clone()).map(Some)
}

/// The body of one `cfg_select!` arm: its tokens inside any braces, and whether braces held them.
pub struct Arm {
    pub from: u32,
    pub to: u32,
    pub braced: bool,
}

/// The first arm of the `cfg_select!` call at `open` whose predicate `env` does not rule out, where `_` always applies and an arm without braces runs to the next top-level comma.
pub fn select_arm(src: &Source, open: u32, env: &super::tree::Env) -> Result<Option<Arm>> {
    let close = src.pair(open);
    let mut at = open + 1;
    while at < close {
        let arrow = skip_until(src, (at, close), |at| src.kind(at) == TokenKind::Eq && src.kind(at + 1) == TokenKind::Gt && src.joint(at));
        if arrow >= close {
            bail!("malformed cfg_select! arm at byte {} of {}", src.byte(at), src.path.display());
        }
        let body = arrow + 2;
        let arm = if src.kind(body) == TokenKind::OpenBrace {
            Arm { from: body + 1, to: src.pair(body), braced: true }
        } else {
            Arm { from: body, to: skip_until(src, (body, close), |at| src.kind(at) == TokenKind::Comma), braced: false }
        };
        let predicate = (at..arrow).map(|token| src.text(token)).collect::<Vec<_>>().join(" ");
        if predicate == "_" || env.allows(&crate::graph::cfg::parse(&predicate)?) {
            return Ok(Some(arm));
        }
        at = if arm.braced { arm.to + 1 } else { arm.to };
        if src.kind(at) == TokenKind::Comma {
            at += 1;
        }
    }
    Ok(None)
}

fn skip_until(src: &Source, (mut at, close): (u32, u32), stop: impl Fn(u32) -> bool) -> u32 {
    while at < close && !stop(at) {
        at = if opens(src.kind(at)) { src.pair(at) + 1 } else { at + 1 };
    }
    at
}
