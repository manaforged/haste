use rustc_hash::FxHashMap as HashMap;

use anyhow::{Result, bail};
use rustc_lexer::TokenKind;

use super::fragment;
use super::rules::{Frag, Meta, Op, Piece, Rule};
use super::source::{Mark, Source, Tokens, opens};

/// What a metavariable matched: a token range of the input and its fragment kind, or one binding per repetition.
#[derive(Clone, Debug)]
pub enum Bind {
    One(u32, u32, Frag),
    Many(Vec<Bind>),
}

/// The bindings of one rule's metavariables, by the slot that the rule's parse gave each name.
type Slots = Vec<Option<Bind>>;

fn bind(slots: &mut Slots, slot: Option<u32>, found: Bind) {
    if let Some(slot) = slot {
        slots[slot as usize] = Some(found);
    }
}

struct Matcher<'a> {
    def: &'a Source,
    input: &'a Source,
    parsed: Option<HashMap<(u32, u32, Frag), Option<u32>>>,
}

fn accepts_opaque(want: Frag, held: Frag) -> bool {
    want == Frag::Tt
        || want == held
        || matches!(
            (held, want),
            (Frag::Expr, Frag::Stmt | Frag::Pat | Frag::PatParam)
                | (Frag::Pat, Frag::PatParam)
                | (Frag::PatParam, Frag::Pat)
                | (Frag::Ty, Frag::Path)
                | (Frag::Path, Frag::Ty | Frag::Expr | Frag::Pat | Frag::PatParam | Frag::Stmt | Frag::Meta)
                | (Frag::Block, Frag::Expr | Frag::Stmt)
                | (Frag::Item, Frag::Stmt)
        )
}

fn opaque_literal(src: &Source, mut from: u32, mut to: u32) -> bool {
    while to > from + 1 && src.kind(from) == TokenKind::OpenParen && src.pair(from) == to - 1 {
        (from, to) = (from + 1, to - 1);
    }
    fragment::extent(src, Frag::Literal, from, to) == Some(to)
}

impl Matcher<'_> {
    fn seq(&mut self, pieces: &[Piece], mut at: u32, end: u32, slots: &mut Slots) -> Option<u32> {
        for piece in pieces {
            at = self.piece(piece, at, end, slots)?;
        }
        Some(at)
    }

    fn sep(&self, sep: (u32, u32), at: u32, end: u32) -> Option<u32> {
        let len = sep.1 - sep.0 + 1;
        let same = (0..len).all(|offset| at + offset < end && same_token((self.input, at + offset, end), (self.def, sep.0 + offset)));
        same.then_some(at + len)
    }

    fn extent(&mut self, frag: Frag, at: u32, end: u32) -> Option<u32> {
        let input = self.input;
        match self.parsed.as_mut().filter(|_| !fragment::is_short(frag)) {
            Some(parsed) => *parsed.entry((at, end, frag)).or_insert_with(|| fragment::extent(input, frag, at, end)),
            None => fragment::extent(input, frag, at, end),
        }
    }

    fn opaque_piece(&mut self, piece: &Piece, (at, to, end): (u32, u32, u32), held: Frag, slots: &mut Slots) -> Option<u32> {
        if to > end {
            return None;
        }
        let (slot, from, reach, frag) = match piece {
            Piece::Repeat(inner, sep, op, names) => {
                return self.repeat((inner, names), (*sep, *op), (at, end), slots);
            }
            Piece::Var(_, Some(Frag::Tt), slot) => (slot, at, to, Frag::Tt),
            Piece::Var(_, Some(Frag::Vis), slot) if held != Frag::Vis => (slot, at, at, Frag::Vis),
            Piece::Var(_, Some(Frag::Literal), slot) if held == Frag::Expr && opaque_literal(self.input, at, to) => (slot, at, to, Frag::Literal),
            Piece::Var(_, Some(frag), slot) if accepts_opaque(*frag, held) => (slot, at, self.extent(*frag, at, end).map_or(to, |parsed| parsed.max(to)), *frag),
            _ => return None,
        };
        if reach > end {
            return None;
        }
        bind(slots, *slot, Bind::One(from, reach, frag));
        Some(reach)
    }

    fn piece(&mut self, piece: &Piece, at: u32, end: u32, slots: &mut Slots) -> Option<u32> {
        if let Some((to, held)) = self.input.opaque_at(at).filter(|_| at < end) {
            return self.opaque_piece(piece, (at, to, end), held, slots);
        }
        match piece {
            Piece::Token(token) => (at < end && same_token((self.input, at, end), (self.def, *token))).then_some(at + 1),
            Piece::Group(open, inner) => {
                if at >= end || self.input.kind(at) != self.def.kind(*open) {
                    return None;
                }
                let close = self.input.pair(at);
                (self.seq(inner, at + 1, close, slots)? == close).then_some(close + 1)
            }
            Piece::Var(_, Some(frag), slot) => {
                let to = self.extent(*frag, at, end)?;
                bind(slots, *slot, Bind::One(at, to, *frag));
                Some(to)
            }
            Piece::Var(_, None, _) | Piece::Crate | Piece::CratePath | Piece::Meta(_) => None,
            Piece::Repeat(inner, sep, op, names) => self.repeat((inner, names), (*sep, *op), (at, end), slots),
        }
    }

    fn repeat(&mut self, (inner, names): (&[Piece], &[u32]), (sep, op): (Option<(u32, u32)>, Op), (mut at, end): (u32, u32), slots: &mut Slots) -> Option<u32> {
        let mut each: Vec<Vec<Bind>> = names.iter().map(|_| Vec::new()).collect();
        let mut rounds = 0;
        while !(op == Op::Maybe && rounds == 1) {
            let start = match sep {
                Some(sep) if rounds > 0 => match self.sep(sep, at, end) {
                    Some(next) => next,
                    None => break,
                },
                _ => at,
            };
            match self.seq(inner, start, end, slots) {
                Some(next) if next > start => {
                    for (&slot, column) in names.iter().zip(&mut each) {
                        column.push(slots[slot as usize].take().unwrap_or(Bind::Many(Vec::new())));
                    }
                    rounds += 1;
                    at = next;
                }
                _ => break,
            }
        }
        if op == Op::Some && rounds == 0 {
            return None;
        }
        for (&slot, column) in names.iter().zip(each) {
            slots[slot as usize] = Some(Bind::Many(column));
        }
        Some(at)
    }
}

struct Writer<'a> {
    def: &'a Source,
    input: &'a Source,
    slots: Slots,
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

fn lookup<'b>(slots: &'b Slots, slot: Option<u32>, rounds: &[usize]) -> Option<&'b Bind> {
    let mut bind = slots.get(slot? as usize)?.as_ref()?;
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
                Piece::Var(name, _, slot) => self.var(*name, *slot, rounds)?,
                Piece::Repeat(inner, sep, _, names) => self.repeat((inner, names), *sep, rounds)?,
            }
        }
        Ok(())
    }

    fn var(&mut self, name: u32, slot: Option<u32>, rounds: &[usize]) -> Result<()> {
        match lookup(&self.slots, slot, rounds) {
            Some(&Bind::One(from, to, frag)) => {
                let grouped = frag == Frag::Expr && fragment::tree(self.input, from) != to;
                let start = self.out.len();
                if grouped {
                    self.out.lexed("(");
                }
                self.out.copy_range(self.input, from, to);
                if grouped {
                    self.out.lexed(")");
                }
                if !matches!(frag, Frag::Tt | Frag::Ident | Frag::Lifetime | Frag::Literal) {
                    self.out.opaque_from(start, frag);
                }
                if frag != Frag::Tt {
                    self.out.space();
                }
            }
            Some(Bind::Many(_)) => {
                bail!("metavariable ${} is still repeating", self.def.text(name))
            }
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
            Meta::Count(slot) => lookup(&self.slots, *slot, rounds).map_or(0, leaves),
            Meta::Concat(parts) => {
                let text: String = parts.iter().map(|&(slot, token)| self.concat_part(slot, token, rounds)).collect();
                self.out.lexed(&text);
                return;
            }
        };
        self.out.lexed(&value.to_string());
    }

    fn concat_part(&self, slot: Option<u32>, token: u32, rounds: &[usize]) -> String {
        let part = match lookup(&self.slots, slot, rounds) {
            Some(&Bind::One(from, ..)) => self.input.text(from),
            _ => self.def.text(token),
        };
        part.trim_start_matches("r#").trim_matches('"').to_owned()
    }

    fn count(&self, names: &[u32], rounds: &[usize]) -> Option<usize> {
        names.iter().find_map(|&slot| match lookup(&self.slots, Some(slot), rounds) {
            Some(Bind::Many(each)) => Some(each.len()),
            _ => None,
        })
    }

    fn repeat(&mut self, (inner, names): (&[Piece], &[u32]), sep: Option<(u32, u32)>, rounds: &mut Vec<usize>) -> Result<()> {
        let Some(count) = self.count(names, rounds) else { bail!("repetition in a transcriber uses no repeating metavariable") };
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
    let mut matcher = Matcher { def, input, parsed: (rules.len() > 1).then(HashMap::default) };
    let close = input.pair(open);
    let mut slots = Slots::new();
    for rule in rules {
        slots.clear();
        slots.resize_with(rule.slots, || None);
        if matcher.seq(&rule.matcher, open + 1, close, &mut slots) == Some(close) {
            let mut writer = Writer { def, input, slots, lens: Vec::new(), krate, home, scope: SCOPES.fetch_add(1, std::sync::atomic::Ordering::Relaxed), out: Tokens::default() };
            writer.emit(&rule.body, &mut Vec::new())?;
            return writer.out.finish(input.path.clone());
        }
    }
    bail!("no rule of the macro matches the call at byte {} of {}", input.byte(open), input.path.display())
}

/// The tokens of the first `cfg_select!` arm that `env` may enable, or `None` when no arm applies.
pub fn select(src: &Source, open: u32, env: &super::tree::Env) -> Result<Option<Source>> {
    let Some(arm) = select_arm(src, open, env)? else {
        return Ok(None);
    };
    if !arm.braced {
        bail!("a cfg_select! arm of items needs braces at byte {} of {}", src.byte(arm.from), src.path.display());
    }
    let mut out = Tokens::default();
    out.copy_range(src, arm.from, arm.to);
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
        let arm = if src.kind(body) == TokenKind::OpenBrace { Arm { from: body + 1, to: src.pair(body), braced: true } } else { Arm { from: body, to: skip_until(src, (body, close), |at| src.kind(at) == TokenKind::Comma), braced: false } };
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

fn same_token((input, at, end): (&Source, u32, u32), (def, token): (&Source, u32)) -> bool {
    input.text(at) == def.text(token) && (input.text(at) != ":" || part_of_colon2(input, at, end) == part_of_colon2(def, token, def.len()))
}

fn part_of_colon2(src: &Source, at: u32, end: u32) -> bool {
    (src.joint(at) && at + 1 < end && src.text(at + 1) == ":") || (at > 0 && src.joint(at - 1) && src.text(at - 1) == ":")
}
