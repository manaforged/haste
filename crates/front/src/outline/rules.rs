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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// A metavariable expression in a transcriber: `${ignore($x)}`, `${index()}`, `${len()}` or `${count($x)}`.
#[derive(Clone, Debug)]
pub enum Meta {
    Ignore(u32),
    Index(usize),
    Len(usize),
    Count(u32),
    Concat(Vec<(bool, u32)>),
}

/// One element of a matcher or transcriber, as token positions in the definition's file.
#[derive(Clone, Debug)]
pub enum Piece {
    Token(u32),
    Group(u32, Vec<Piece>),
    Var(u32, Option<Frag>),
    Crate,
    CratePath,
    Meta(Meta),
    Repeat(Vec<Piece>, Option<(u32, u32)>, Op),
}

/// One `matcher => transcriber` arm.
#[derive(Clone, Debug)]
pub struct Rule {
    pub matcher: Vec<Piece>,
    pub body: Vec<Piece>,
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
}

impl Parser<'_> {
    fn pieces(&self, mut at: u32, to: u32) -> Result<Vec<Piece>> {
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

    fn dollar(&self, next: u32, out: &mut Vec<Piece>) -> Result<u32> {
        let kind = self.src.kind(next);
        if kind == TokenKind::OpenParen {
            let close = self.src.pair(next);
            let inner = self.pieces(next + 1, close)?;
            let (sep, op, after) = self.tail(close + 1)?;
            out.push(Piece::Repeat(inner, sep, op));
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
            out.push(Piece::Var(next, Some(frag(self.src.text(next + 2))?)));
            return Ok(next + 3);
        }
        out.push(Piece::Var(next, None));
        Ok(next + 1)
    }

    fn meta(&self, open: u32) -> Result<Meta> {
        let name = open + 1;
        let args = name + 1;
        let variable = (self.src.kind(args + 1) == TokenKind::Dollar).then_some(args + 2);
        let depth = (matches!(self.src.kind(args + 1), TokenKind::Literal { .. })).then(|| self.src.text(args + 1).parse().unwrap_or(0)).unwrap_or(0);
        match (self.src.text(name), variable) {
            ("ignore", Some(variable)) => Ok(Meta::Ignore(variable)),
            ("count", Some(variable)) => Ok(Meta::Count(variable)),
            ("index", _) => Ok(Meta::Index(depth)),
            ("len", _) => Ok(Meta::Len(depth)),
            ("concat", _) => Ok(Meta::Concat(self.concat_parts(args))),
            (other, _) => bail!("{}: unknown metavariable expression {other}", self.src.path.display()),
        }
    }

    fn concat_parts(&self, open: u32) -> Vec<(bool, u32)> {
        let close = self.src.pair(open);
        let mut parts = Vec::new();
        let mut at = open + 1;
        while at < close {
            let variable = self.src.kind(at) == TokenKind::Dollar;
            let token = if variable { at + 1 } else { at };
            parts.push((variable, token));
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

/// Parses a `macro` definition whose body opens at `open`: one `(args) { body }` rule or a braced list of rules.
/// The rules of a `#[macro_export(local_inner_macros)]` macro, whose transcribers call each macro named by one word as `$crate::name!`.
pub fn local_inner(src: &Source, rules: Vec<Rule>) -> Vec<Rule> {
    rules.into_iter().map(|rule| Rule { matcher: rule.matcher, body: qualified(src, rule.body) }).collect()
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
            Piece::Repeat(inner, sep, op) => Piece::Repeat(qualified(src, inner), sep, op),
            other => other,
        });
    }
    out
}

pub fn parse_macro(src: &Source, open: u32) -> Result<Vec<Rule>> {
    if src.kind(open) != TokenKind::OpenParen {
        return parse(src, open);
    }
    let body = src.pair(open) + 1;
    let matcher = Parser { src, matcher: true }.pieces(open + 1, src.pair(open))?;
    Ok(vec![Rule { matcher, body: Parser { src, matcher: false }.pieces(body + 1, src.pair(body))? }])
}

/// Parses the rules of a `macro_rules!` body that opens at token `open`.
pub fn parse(src: &Source, open: u32) -> Result<Vec<Rule>> {
    let close = src.pair(open);
    let (matchers, bodies) = (Parser { src, matcher: true }, Parser { src, matcher: false });
    let mut rules = Vec::new();
    let mut at = open + 1;
    while at < close {
        let arrow = src.pair(at) + 1;
        let body = arrow + 2;
        if !opens(src.kind(at)) || src.kind(arrow) != TokenKind::Eq || !opens(src.kind(body)) {
            bail!("{}: malformed macro rule at byte {}", src.path.display(), src.byte(at));
        }
        let matcher = matchers.pieces(at + 1, src.pair(at))?;
        rules.push(Rule { matcher, body: bodies.pieces(body + 1, src.pair(body))? });
        at = src.pair(body) + 1;
        if matches!(src.kind(at), TokenKind::Semi | TokenKind::Comma) {
            at += 1;
        }
    }
    Ok(rules)
}
