use std::collections::BTreeSet;

use anyhow::{Result, anyhow, bail};

/// A `cfg(...)` predicate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cfg {
    Name(String),
    KeyValue(String, String),
    All(Vec<Cfg>),
    Any(Vec<Cfg>),
    Not(Box<Cfg>),
}

/// The cfg names and key-value pairs that hold for one target. `internal` holds the ones the compiler sets for source but does not show Cargo.
#[derive(Clone, Debug, Default)]
pub struct Target {
    pub triple: String,
    pub names: BTreeSet<String>,
    pub pairs: BTreeSet<(String, String)>,
    pub internal: BTreeSet<(String, String)>,
}

impl Target {
    /// Whether the compiler sets the name `name` for source, including names Cargo does not see.
    pub fn has_name(&self, name: &str) -> bool {
        self.names.contains(name) || self.internal.contains(&(name.to_owned(), String::new()))
    }

    /// Whether the compiler sets `key = "value"` for source, including pairs Cargo does not see.
    pub fn has_pair(&self, key: &str, value: &str) -> bool {
        let pair = (key.to_owned(), value.to_owned());
        self.pairs.contains(&pair) || self.internal.contains(&pair)
    }

    /// The cfgs of `target`, from its catalog entry: the lines before `[internal]` are what Cargo sees.
    pub fn of(target: &haste_codegen::Target) -> Self {
        let mut cfg = Target { triple: target.triple.to_owned(), ..Target::default() };
        let mut internal = false;
        for line in target.cfg.lines().map(str::trim).filter(|line| !line.is_empty()) {
            if line == "[internal]" {
                internal = true;
                continue;
            }
            let pair = line.split_once('=').map(|(key, value)| (key.to_owned(), value.trim_matches('"').to_owned()));
            match (pair, internal) {
                (Some(pair), false) => _ = cfg.pairs.insert(pair),
                (Some(pair), true) => _ = cfg.internal.insert(pair),
                (None, false) => _ = cfg.names.insert(line.to_owned()),
                (None, true) => _ = cfg.internal.insert((line.to_owned(), String::new())),
            }
        }
        cfg
    }

    /// The cfgs of the machine this program runs on.
    pub fn host() -> Self {
        Target::of(haste_codegen::Target::host())
    }

    /// Whether `cfg` holds for this target.
    pub fn holds(&self, cfg: &Cfg) -> bool {
        match cfg {
            Cfg::Name(name) => self.names.contains(name),
            Cfg::KeyValue(key, value) => self.pairs.contains(&(key.clone(), value.clone())),
            Cfg::All(all) => all.iter().all(|cfg| self.holds(cfg)),
            Cfg::Any(any) => any.iter().any(|cfg| self.holds(cfg)),
            Cfg::Not(cfg) => !self.holds(cfg),
        }
    }

    /// Whether a `[target.KEY.dependencies]` table applies: KEY is a triple or `cfg(...)`.
    pub fn matches_key(&self, key: &str) -> Result<bool> {
        match key.strip_prefix("cfg(").and_then(|rest| rest.strip_suffix(')')) {
            Some(inner) => Ok(self.holds(&parse(inner)?)),
            None => Ok(key == self.triple),
        }
    }
}

#[derive(Debug, PartialEq)]
enum Lexeme {
    Ident(String),
    Text(String),
    Open,
    Close,
    Comma,
    Equals,
}

fn quoted(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Result<Lexeme> {
    let mut text = String::new();
    for c in chars.by_ref() {
        if c == '"' {
            return Ok(Lexeme::Text(text));
        }
        text.push(c);
    }
    bail!("unterminated string in cfg")
}

fn ident(first: char, chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Lexeme {
    let mut text = first.to_string();
    while let Some(c) = chars.next_if(|c| c.is_alphanumeric() || *c == '_') {
        text.push(c);
    }
    Lexeme::Ident(text)
}

fn lex(text: &str) -> Result<Vec<Lexeme>> {
    let mut chars = text.chars().peekable();
    let mut out = Vec::new();
    while let Some(c) = chars.next() {
        out.push(match c {
            '(' => Lexeme::Open,
            ')' => Lexeme::Close,
            ',' => Lexeme::Comma,
            '=' => Lexeme::Equals,
            '"' => quoted(&mut chars)?,
            c if c.is_whitespace() => continue,
            c if c.is_alphabetic() || c == '_' => ident(c, &mut chars),
            other => bail!("unexpected {other:?} in cfg"),
        });
    }
    Ok(out)
}

/// Parses the inside of `cfg(...)`.
pub fn parse(text: &str) -> Result<Cfg> {
    let lexemes = lex(text)?;
    let mut at = 0;
    let cfg = predicate(&lexemes, &mut at)?;
    if at != lexemes.len() {
        bail!("trailing input in cfg({text})");
    }
    Ok(cfg)
}

fn predicate(lexemes: &[Lexeme], at: &mut usize) -> Result<Cfg> {
    let Some(Lexeme::Ident(name)) = lexemes.get(*at) else { bail!("expected a cfg name") };
    *at += 1;
    match lexemes.get(*at) {
        Some(Lexeme::Equals) => {
            let Some(Lexeme::Text(value)) = lexemes.get(*at + 1) else { bail!("expected a string after {name} =") };
            *at += 2;
            Ok(Cfg::KeyValue(name.clone(), value.clone()))
        }
        Some(Lexeme::Open) => {
            *at += 1;
            let inner = list(lexemes, at)?;
            match name.as_str() {
                "all" => Ok(Cfg::All(inner)),
                "any" => Ok(Cfg::Any(inner)),
                "not" => inner.into_iter().next().map(|cfg| Cfg::Not(Box::new(cfg))).ok_or_else(|| anyhow!("not() is empty")),
                other => bail!("unknown cfg operator {other}"),
            }
        }
        _ => Ok(Cfg::Name(name.clone())),
    }
}

fn list(lexemes: &[Lexeme], at: &mut usize) -> Result<Vec<Cfg>> {
    let mut items = Vec::new();
    loop {
        if lexemes.get(*at) == Some(&Lexeme::Close) {
            *at += 1;
            return Ok(items);
        }
        items.push(predicate(lexemes, at)?);
        if lexemes.get(*at) == Some(&Lexeme::Comma) {
            *at += 1;
        }
    }
}
