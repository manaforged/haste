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

const X86_64_LINUX_GNU: &[(&str, &str)] = &[
    ("panic", "unwind"),
    ("target_abi", ""),
    ("target_arch", "x86_64"),
    ("target_endian", "little"),
    ("target_env", "gnu"),
    ("target_family", "unix"),
    ("target_feature", "fxsr"),
    ("target_feature", "sse"),
    ("target_feature", "sse2"),
    ("target_has_atomic", "16"),
    ("target_has_atomic", "32"),
    ("target_has_atomic", "64"),
    ("target_has_atomic", "8"),
    ("target_has_atomic", "ptr"),
    ("target_has_atomic_primitive_alignment", "16"),
    ("target_has_atomic_primitive_alignment", "32"),
    ("target_has_atomic_primitive_alignment", "64"),
    ("target_has_atomic_primitive_alignment", "8"),
    ("target_has_atomic_primitive_alignment", "ptr"),
    ("target_os", "linux"),
    ("target_pointer_width", "64"),
    ("target_vendor", "unknown"),
];

const X86_64_LINUX_GNU_INTERNAL: &[(&str, &str)] = &[
    ("fmt_debug", "full"),
    ("relocation_model", "pic"),
    ("target_feature", "x87"),
    ("target_has_atomic", ""),
    ("target_has_atomic_load_store", ""),
    ("target_has_atomic_load_store", "16"),
    ("target_has_atomic_load_store", "32"),
    ("target_has_atomic_load_store", "64"),
    ("target_has_atomic_load_store", "8"),
    ("target_has_atomic_load_store", "ptr"),
    ("target_has_reliable_f128", ""),
    ("target_has_reliable_f128_math", ""),
    ("target_has_reliable_f16", ""),
    ("target_has_reliable_f16_math", ""),
    ("target_has_threads", ""),
    ("target_object_format", "elf"),
    ("target_thread_local", ""),
];

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

    /// The x86-64 Linux GNU target: `rustc --print cfg` without `debug_assertions`, plus the internal cfgs of `rustc +nightly --print cfg -Zunstable-options`.
    pub fn x86_64_linux_gnu() -> Self {
        let pairs = X86_64_LINUX_GNU.iter().map(|(key, value)| ((*key).to_owned(), (*value).to_owned())).collect();
        let internal = X86_64_LINUX_GNU_INTERNAL.iter().map(|(key, value)| ((*key).to_owned(), (*value).to_owned())).collect();
        Target { triple: "x86_64-unknown-linux-gnu".to_owned(), names: ["unix".to_owned()].into(), pairs, internal }
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
