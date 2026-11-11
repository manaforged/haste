use anyhow::{Result, bail};
use rustc_lexer::TokenKind;

use super::source::{Source, opens};

/// How often a repetition may match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Many,
    Some,
    Maybe,
}

/// A fragment specifier of a metavariable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Frag {
    Ident,
    Path,
    Expr,
    Ty,
    Pat,
    PatParam,
    Stmt,
    Block,
    Item,
    Meta,
    Tt,
    Literal,
    Lifetime,
    Vis,
}

const FRAGS: [(&str, Frag); 15] = [
    ("ident", Frag::Ident),
    ("path", Frag::Path),
    ("expr", Frag::Expr),
    ("expr_2021", Frag::Expr),
    ("ty", Frag::Ty),
    ("pat", Frag::Pat),
    ("pat_param", Frag::PatParam),
    ("stmt", Frag::Stmt),
    ("block", Frag::Block),
    ("item", Frag::Item),
    ("meta", Frag::Meta),
    ("tt", Frag::Tt),
    ("literal", Frag::Literal),
    ("lifetime", Frag::Lifetime),
    ("vis", Frag::Vis),
];

/// A metavariable expression in a transcriber: `${ignore($x)}`, `${index()}`, `${len()}`, `${count($x)}` or `${concat(..)}`, naming metavariables by their slot in the rule, or `None` when the matcher binds no such name.
#[derive(Clone, Debug)]
pub enum Meta {
    Ignore(Option<u32>),
    Index(usize),
    Len(usize),
    Count(Option<u32>),
    Concat(Vec<(Option<u32>, u32)>),
}

/// One element of a matcher or transcriber, as token positions in the definition's file; a metavariable also carries its slot in the rule, and a repetition the slots of every metavariable inside it.
#[derive(Clone, Debug)]
pub enum Piece {
    Token(u32),
    Group(u32, Vec<Piece>),
    Var(u32, Option<Frag>, Option<u32>),
    Crate,
    CratePath,
    Meta(Meta),
    Repeat(Vec<Piece>, Option<(u32, u32)>, Op, Box<[u32]>),
}

/// One `matcher => transcriber` arm, whose matcher binds `slots` distinct metavariable names.
#[derive(Clone, Debug)]
pub struct Rule {
    pub matcher: Vec<Piece>,
    pub body: Vec<Piece>,
    pub slots: usize,
}

fn is_name(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Ident | TokenKind::RawIdent)
}

fn op_of(kind: TokenKind) -> Option<Op> {
    match kind {
        TokenKind::Star => Some(Op::Many),
        TokenKind::Plus => Some(Op::Some),
        TokenKind::Question => Some(Op::Maybe),
        _ => None,
    }
}

struct Parser<'a> {
    src: &'a Source,
    matcher: bool,
    names: Vec<&'a str>,
}

fn slots(pieces: &[Piece], into: &mut Vec<u32>) {
    for piece in pieces {
        match piece {
            Piece::Var(_, _, slot) | Piece::Meta(Meta::Ignore(slot) | Meta::Count(slot)) => into.extend(*slot),
            Piece::Group(_, inner) => slots(inner, into),
            Piece::Repeat(.., names) => into.extend_from_slice(names),
            Piece::Token(_) | Piece::Crate | Piece::CratePath | Piece::Meta(_) => {}
        }
    }
}

impl<'a> Parser<'a> {
    fn slot(&mut self, name: u32) -> Option<u32> {
        let src = self.src;
        let text = src.text(name);
        if let Some(found) = self.names.iter().position(|known| *known == text) {
            return Some(found as u32);
        }
        if !self.matcher {
            return None;
        }
        self.names.push(text);
        Some(self.names.len() as u32 - 1)
    }

    fn pieces(&mut self, mut at: u32, to: u32) -> Result<Vec<Piece>> {
        let mut out = Vec::new();
        while at < to {
            let kind = self.src.kind(at);
            at = if kind == TokenKind::Dollar && at + 1 < to {
                self.dollar(at + 1, &mut out)?
            } else if opens(kind) {
                let close = self.src.pair(at);
                out.push(Piece::Group(at, self.pieces(at + 1, close)?));
                close + 1
            } else {
                out.push(Piece::Token(at));
                at + 1
            };
        }
        Ok(out)
    }

    fn dollar(&mut self, next: u32, out: &mut Vec<Piece>) -> Result<u32> {
        let kind = self.src.kind(next);
        if kind == TokenKind::OpenParen {
            let close = self.src.pair(next);
            let inner = self.pieces(next + 1, close)?;
            let (sep, op, after) = self.tail(close + 1)?;
            let mut names = Vec::new();
            slots(&inner, &mut names);
            out.push(Piece::Repeat(inner, sep, op, names.into_boxed_slice()));
            return Ok(after);
        }
        if kind == TokenKind::OpenBrace && !self.matcher {
            out.push(Piece::Meta(self.meta(next)?));
            return Ok(self.src.pair(next) + 1);
        }
        if !is_name(kind) {
            out.push(Piece::Token(next - 1));
            return Ok(next);
        }
        if !self.matcher && self.src.text(next) == "crate" {
            out.push(Piece::Crate);
            return Ok(next + 1);
        }
        if self.matcher && self.src.kind(next + 1) == TokenKind::Colon && is_name(self.src.kind(next + 2)) {
            let frag = frag(self.src.text(next + 2))?;
            out.push(Piece::Var(next, Some(frag), self.slot(next)));
            return Ok(next + 3);
        }
        out.push(Piece::Var(next, None, self.slot(next)));
        Ok(next + 1)
    }

    fn meta(&mut self, open: u32) -> Result<Meta> {
        let name = open + 1;
        let args = name + 1;
        let variable = (self.src.kind(args + 1) == TokenKind::Dollar).then_some(args + 2);
        let depth = (matches!(self.src.kind(args + 1), TokenKind::Literal { .. })).then(|| self.src.text(args + 1).parse().unwrap_or(0)).unwrap_or(0);
        let src = self.src;
        match (src.text(name), variable) {
            ("ignore", Some(variable)) => Ok(Meta::Ignore(self.slot(variable))),
            ("count", Some(variable)) => Ok(Meta::Count(self.slot(variable))),
            ("index", _) => Ok(Meta::Index(depth)),
            ("len", _) => Ok(Meta::Len(depth)),
            ("concat", _) => Ok(Meta::Concat(self.concat_parts(args))),
            (other, _) => bail!("{}: unknown metavariable expression {other}", self.src.path.display()),
        }
    }

    fn concat_parts(&mut self, open: u32) -> Vec<(Option<u32>, u32)> {
        let close = self.src.pair(open);
        let mut parts = Vec::new();
        let mut at = open + 1;
        while at < close {
            let variable = self.src.kind(at) == TokenKind::Dollar;
            let token = if variable { at + 1 } else { at };
            parts.push((if variable { self.slot(token) } else { None }, token));
            at = token + 1;
            if self.src.kind(at) == TokenKind::Comma {
                at += 1;
            }
        }
        parts
    }

    fn tail(&self, at: u32) -> Result<(Option<(u32, u32)>, Op, u32)> {
        if let Some(op) = op_of(self.src.kind(at)) {
            return Ok((None, op, at + 1));
        }
        let mut last = at;
        while self.src.joint(last) && op_of(self.src.kind(last + 1)).is_none() && last + 1 < self.src.len() {
            last += 1;
        }
        match op_of(self.src.kind(last + 1)) {
            Some(op) => Ok((Some((at, last)), op, last + 2)),
            None => bail!("{}: repetition without an operator at byte {}", self.src.path.display(), self.src.byte(at)),
        }
    }
}

fn frag(name: &str) -> Result<Frag> {
    match FRAGS.iter().find(|(text, _)| *text == name) {
        Some(&(_, frag)) => Ok(frag),
        None => bail!("unknown fragment specifier {name}"),
    }
}

/// The rules of a `#[macro_export(local_inner_macros)]` macro, whose transcribers call each macro named by one word as `$crate::name!`.
pub fn local_inner(src: &Source, rules: Vec<Rule>) -> Vec<Rule> {
    rules.into_iter().map(|rule| Rule { matcher: rule.matcher, body: qualified(src, rule.body), slots: rule.slots }).collect()
}

fn qualified(src: &Source, pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out = Vec::with_capacity(pieces.len());
    for (at, piece) in pieces.iter().enumerate() {
        let after_path = at > 0 && matches!(pieces[at - 1], Piece::Token(token) if src.text(token) == ":");
        let called = matches!(pieces.get(at + 1), Some(Piece::Token(bang)) if src.text(*bang) == "!");
        if let Piece::Token(name) = piece
            && called
            && !after_path
            && is_name(src.kind(*name))
            && src.text(*name) != "macro_rules"
        {
            out.push(Piece::CratePath);
        }
        out.push(match piece.clone() {
            Piece::Group(open, inner) => Piece::Group(open, qualified(src, inner)),
            Piece::Repeat(inner, sep, op, names) => Piece::Repeat(qualified(src, inner), sep, op, names),
            other => other,
        });
    }
    out
}

/// Parses a `macro` definition whose body opens at `open`: one `(args) { body }` rule or a braced list of rules.
pub fn parse_macro(src: &Source, open: u32) -> Result<Vec<Rule>> {
    if src.kind(open) != TokenKind::OpenParen {
        return parse(src, open);
    }
    Ok(vec![rule(src, open, src.pair(open) + 1)?])
}

/// The rule whose matcher opens at `matcher` and whose transcriber opens at `body`, with each metavariable name given its slot.
fn rule(src: &Source, matcher: u32, body: u32) -> Result<Rule> {
    let mut parser = Parser { src, matcher: true, names: Vec::new() };
    let matcher = parser.pieces(matcher + 1, src.pair(matcher))?;
    parser.matcher = false;
    let body = parser.pieces(body + 1, src.pair(body))?;
    Ok(Rule { matcher, body, slots: parser.names.len() })
}

/// Parses the rules of a `macro_rules!` body that opens at token `open`.
pub fn parse(src: &Source, open: u32) -> Result<Vec<Rule>> {
    let close = src.pair(open);
    let mut rules = Vec::new();
    let mut at = open + 1;
    while at < close {
        let arrow = src.pair(at) + 1;
        let body = arrow + 2;
        if !opens(src.kind(at)) || src.kind(arrow) != TokenKind::Eq || !opens(src.kind(body)) {
            bail!("{}: malformed macro rule at byte {}", src.path.display(), src.byte(at));
        }
        rules.push(rule(src, at, body)?);
        at = src.pair(body) + 1;
        if matches!(src.kind(at), TokenKind::Semi | TokenKind::Comma) {
            at += 1;
        }
    }
    Ok(rules)
}
