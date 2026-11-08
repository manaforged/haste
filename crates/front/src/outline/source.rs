use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rustc_lexer::{FrontmatterAllowed, TokenKind, tokenize};

/// One non-trivia token, with the index of its matching delimiter when it opens or closes a group.
#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub start: u32,
    pub len: u32,
    pub pair: u32,
    pub joint: bool,
}

/// Whether `kind` is a doc comment, and whether it documents the enclosing item.
pub fn doc(kind: TokenKind) -> Option<bool> {
    match kind {
        TokenKind::LineComment { doc_style: Some(style) } | TokenKind::BlockComment { doc_style: Some(style), .. } => {
            Some(style == rustc_lexer::DocStyle::Inner)
        }
        _ => None,
    }
}

/// A source file and its tokens.
#[derive(Debug)]
pub struct Source {
    pub path: PathBuf,
    pub text: String,
    pub tokens: Vec<Token>,
    pub dollars: Vec<(u32, u32)>,
}

/// Tokens being written by a macro expansion, copied from other sources without lexing them again.
#[derive(Default)]
pub struct Tokens {
    text: String,
    tokens: Vec<Token>,
    dollars: Vec<(u32, u32)>,
}

fn is_punct_kind(kind: TokenKind) -> bool {
    !matches!(kind, TokenKind::Ident | TokenKind::RawIdent | TokenKind::Literal { .. } | TokenKind::Lifetime { .. } | TokenKind::RawLifetime)
}

impl Tokens {
    fn append(&mut self, kind: TokenKind, text: &str, joint: bool) {
        let start = self.text.len() as u32;
        self.text.push_str(text);
        let joint = joint && is_punct_kind(kind);
        if !joint {
            self.text.push(' ');
        }
        self.tokens.push(Token { kind, start, len: text.len() as u32, pair: u32::MAX, joint });
    }

    /// Copies token `at` of `src`, with its spacing and its `$crate` origin.
    pub fn copy(&mut self, src: &Source, at: u32) {
        if let Some(krate) = src.dollar(at) {
            self.dollars.push((self.tokens.len() as u32, krate));
        }
        self.append(src.kind(at), src.text(at), src.joint(at));
    }

    /// Writes `$crate` for the crate `krate`.
    pub fn dollar_crate(&mut self, krate: u32) {
        self.dollars.push((self.tokens.len() as u32, krate));
        self.append(TokenKind::Dollar, "$", true);
        self.append(TokenKind::Ident, "crate", false);
    }

    /// Lexes `text`, a short piece a metavariable expression made, and writes its tokens.
    pub fn lexed(&mut self, text: &str) {
        let mut at = 0;
        for token in tokenize(text, FrontmatterAllowed::No) {
            let piece = &text[at..at + token.len as usize];
            at += token.len as usize;
            if !is_trivia(token.kind) {
                self.append(token.kind, piece, false);
            }
        }
    }

    /// Ends the last token's spacing, so it never joins the next one.
    pub fn space(&mut self) {
        if let Some(last) = self.tokens.last_mut().filter(|last| last.joint) {
            last.joint = false;
            self.text.push(' ');
        }
    }

    /// The written tokens as a source with the name `path`.
    pub fn finish(self, path: PathBuf) -> Result<Source> {
        let mut tokens = self.tokens;
        pair_up(&mut tokens, &path)?;
        Ok(Source { path, text: self.text, tokens, dollars: self.dollars })
    }
}

fn is_trivia(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Whitespace | TokenKind::LineComment { doc_style: None } | TokenKind::BlockComment { doc_style: None, .. })
}

fn opens(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::OpenParen | TokenKind::OpenBrace | TokenKind::OpenBracket)
}

fn closes(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::CloseParen | TokenKind::CloseBrace | TokenKind::CloseBracket)
}

fn tokens_of(text: &str, base: u32, into: &mut Vec<Token>) {
    let mut start = base;
    let mut joint = false;
    for token in tokenize(text, FrontmatterAllowed::No) {
        if is_trivia(token.kind) {
            joint = false;
        } else {
            if joint && let Some(last) = into.last_mut() {
                last.joint = true;
            }
            into.push(Token { kind: token.kind, start, len: token.len, pair: u32::MAX, joint: false });
            joint = true;
        }
        start += token.len;
    }
}

fn raw_hashes(body: &str) -> usize {
    let bytes = body.as_bytes();
    let runs = bytes.iter().enumerate().filter(|(_, byte)| **byte == b'"').map(|(at, _)| bytes[at + 1..].iter().take_while(|byte| **byte == b'#').count());
    runs.max().map_or(1, |most| most + 1)
}

fn doc_tokens(comment: &str, inner: bool, (tail, base): (&mut String, u32), into: &mut Vec<Token>) {
    let body = if comment.starts_with("/*") { &comment[3..comment.len().saturating_sub(2).max(3)] } else { &comment[3..] };
    let hashes = raw_hashes(body);
    let mut emit = |tail: &mut String, kind: TokenKind, pieces: &[&str]| {
        let start = base + tail.len() as u32;
        pieces.iter().for_each(|piece| tail.push_str(piece));
        into.push(Token { kind, start, len: base + tail.len() as u32 - start, pair: u32::MAX, joint: true });
    };
    emit(tail, TokenKind::Pound, &["#"]);
    if inner {
        emit(tail, TokenKind::Bang, &["!"]);
    }
    emit(tail, TokenKind::OpenBracket, &["["]);
    emit(tail, TokenKind::Ident, &["doc"]);
    emit(tail, TokenKind::Eq, &["="]);
    let fence = &"################################"[..hashes.min(32)];
    let literal = TokenKind::Literal { kind: rustc_lexer::LiteralKind::RawStr { n_hashes: Some(hashes as u8) }, suffix_start: (body.len() + 3 + 2 * hashes) as u32 };
    emit(tail, literal, &["r", fence, "\"", body, "\"", fence]);
    emit(tail, TokenKind::CloseBracket, &["]"]);
    if let Some(last) = into.last_mut() {
        last.joint = false;
    }
    tail.push('\n');
}

fn lex(text: &mut String) -> Vec<Token> {
    let bom = if text.starts_with('\u{feff}') { 3 } else { 0 };
    let skip = bom + rustc_lexer::strip_shebang(&text[bom..]).unwrap_or(0);
    let mut raw = Vec::with_capacity(text.len() / 5);
    tokens_of(&text[skip..], skip as u32, &mut raw);
    let docs: Vec<usize> = raw.iter().filter(|token| doc(token.kind).is_some()).map(|token| token.len as usize).collect();
    if docs.is_empty() {
        return raw;
    }
    let base = text.len() as u32 + 1;
    let mut tail = String::with_capacity(1 + docs.iter().map(|len| len + 16).sum::<usize>());
    tail.push('\n');
    let mut tokens = Vec::with_capacity(raw.len() + docs.len() * 7);
    for token in raw {
        match doc(token.kind) {
            Some(inner) => doc_tokens(&text[token.start as usize..(token.start + token.len) as usize], inner, (&mut tail, base - 1), &mut tokens),
            None => tokens.push(token),
        }
    }
    text.push_str(&tail);
    tokens
}

fn pair_up(tokens: &mut [Token], path: &Path) -> Result<()> {
    let mut open = Vec::new();
    for at in 0..tokens.len() {
        let kind = tokens[at].kind;
        if opens(kind) {
            open.push(at);
        } else if closes(kind) {
            let Some(start) = open.pop() else { bail!("{}: unmatched closing delimiter at byte {}", path.display(), tokens[at].start) };
            tokens[start].pair = at as u32;
            tokens[at].pair = start as u32;
        }
    }
    if let Some(start) = open.pop() {
        bail!("{}: unclosed delimiter at byte {}", path.display(), tokens[start].start);
    }
    Ok(())
}

impl Source {
    /// Reads and lexes the file at `path`.
    pub fn read(path: &Path) -> Result<Source> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Source::parse(path.to_owned(), text)
    }

    /// Lexes `text` as the contents of `path`.
    pub fn parse(path: PathBuf, text: String) -> Result<Source> {
        let mut text = text;
        let mut tokens = lex(&mut text);
        pair_up(&mut tokens, &path)?;
        Ok(Source { path, text, tokens, dollars: Vec::new() })
    }

    /// The text of token `at`.
    pub fn text(&self, at: u32) -> &str {
        let token = self.tokens[at as usize];
        &self.text[token.start as usize..(token.start + token.len) as usize]
    }

    /// The source text from the start of token `from` to the end of token `to`, inclusive.
    pub fn span(&self, from: u32, to: u32) -> &str {
        let (first, last) = (self.tokens[from as usize], self.tokens[to as usize]);
        &self.text[first.start as usize..(last.start + last.len) as usize]
    }

    /// The kind of token `at`, or `Eof` past the end.
    pub fn kind(&self, at: u32) -> TokenKind {
        self.tokens.get(at as usize).map_or(TokenKind::Eof, |token| token.kind)
    }

    /// The index of the delimiter that matches token `at`.
    pub fn pair(&self, at: u32) -> u32 {
        self.tokens[at as usize].pair
    }

    /// Whether token `at` is followed by the next token with no space between them.
    pub fn joint(&self, at: u32) -> bool {
        self.tokens[at as usize].joint
    }

    /// The crate that the `$crate` starting at token `at` names, when a macro wrote it.
    pub fn dollar(&self, at: u32) -> Option<u32> {
        self.dollars.binary_search_by_key(&at, |(token, _)| *token).ok().map(|found| self.dollars[found].1)
    }

    /// The number of tokens.
    pub fn len(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// Whether the file has no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}
