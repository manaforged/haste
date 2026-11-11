use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::tree::File;

use anyhow::{Context, Result, bail};
use rustc_hash::FxHashMap;
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

/// Where a token that a macro wrote came from: the `$` of a `$crate` naming a crate, or a name from the body of a `macro` item, which also resolves where that item is defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    Crate(u32),
    Home { krate: u32, file: u32, item: u32 },
    /// A name from the body of a `macro_rules!` expansion, which binds local variables only within that expansion and also finds those of the scope where the macro was defined.
    Scope { id: u32, parent: u32 },
}

/// A source file and its tokens, or tokens laid after those of a base file, numbered on from its last one.
#[derive(Debug)]
pub struct Source {
    pub path: PathBuf,
    pub text: String,
    pub tokens: Vec<Token>,
    pub marks: Vec<(u32, Mark)>,
    /// The token ranges, by index in this layer, that a macro wrote from an expression-like fragment, which later macros match only as a whole.
    pub opaque: Vec<(u32, u32)>,
    pub base: Option<Arc<File>>,
    pub offset: u32,
    /// The tokens that each macro call in type position writes, by the token that opens its group.
    pub outputs: Outputs,
}

/// Macro outputs kept for as long as the source that holds their calls.
#[derive(Debug, Default)]
pub struct Outputs(Mutex<FxHashMap<u32, Box<Source>>>);

impl Outputs {
    /// The output kept for the call whose group opens at `open`.
    pub fn get(&self, open: u32) -> Result<Option<&Source>> {
        let kept = self.0.lock().map_err(|_| anyhow::anyhow!("macro outputs poisoned"))?;
        // SAFETY: entries are never removed or replaced while self lives, and a box keeps its heap address when the map grows.
        Ok(kept.get(&open).map(|found| unsafe { &*std::ptr::from_ref::<Source>(found) }))
    }

    /// Keeps `output` for the call whose group opens at `open`, or the output another caller kept first.
    pub fn keep(&self, open: u32, output: Source) -> Result<&Source> {
        let mut kept = self.0.lock().map_err(|_| anyhow::anyhow!("macro outputs poisoned"))?;
        let found = kept.entry(open).or_insert_with(|| Box::new(output));
        // SAFETY: entries are never removed or replaced while self lives, and a box keeps its heap address when the map grows.
        Ok(unsafe { &*std::ptr::from_ref::<Source>(found) })
    }
}

/// Tokens being written by a macro expansion, copied from other sources without lexing them again.
#[derive(Default)]
pub struct Tokens {
    text: String,
    tokens: Vec<Token>,
    marks: Vec<(u32, Mark)>,
    opaque: Vec<(u32, u32)>,
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

    /// Copies token `at` of `src`, with its spacing and its origin mark.
    pub fn copy(&mut self, src: &Source, at: u32) {
        let (layer, local) = src.layer(at);
        let mark = layer.own_mark(local as u32);
        self.copy_marked(src, at, mark);
    }

    /// Copies token `at` of `src` with its spacing, marked as coming from `mark`.
    pub fn copy_marked(&mut self, src: &Source, at: u32, mark: Option<Mark>) {
        let (layer, local) = src.layer(at);
        let token = layer.tokens[local];
        if let Some(mark) = mark {
            self.marks.push((self.tokens.len() as u32, mark));
        }
        self.append(token.kind, &layer.text[token.start as usize..(token.start + token.len) as usize], token.joint);
    }

    /// Writes the name `name`, marked as coming from `mark`.
    pub fn ident(&mut self, name: &str, mark: Option<Mark>) {
        if let Some(mark) = mark {
            self.marks.push((self.tokens.len() as u32, mark));
        }
        self.append(TokenKind::Ident, name, false);
    }

    /// Writes `$crate` for the crate `krate`.
    pub fn dollar_crate(&mut self, krate: u32) {
        self.marks.push((self.tokens.len() as u32, Mark::Crate(krate)));
        self.append(TokenKind::Dollar, "$", true);
        self.append(TokenKind::Ident, "crate", false);
    }

    /// Lexes `text`, a short piece that a metavariable expression or a builtin macro made, and writes its tokens.
    pub fn lexed(&mut self, text: &str) {
        let mut at = 0;
        let mut tokens = tokenize(text, FrontmatterAllowed::No).peekable();
        while let Some(token) = tokens.next() {
            let piece = &text[at..at + token.len as usize];
            at += token.len as usize;
            if !is_trivia(token.kind) {
                let joint = tokens.peek().is_some_and(|next| !is_trivia(next.kind));
                self.append(token.kind, piece, joint);
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
        Ok(Source { path, text: self.text, tokens, marks: self.marks, opaque: self.opaque, base: None, offset: 0, outputs: Outputs::default() })
    }

    /// The written tokens laid after every token of `base`, so that the indices of `base` still name its tokens.
    pub fn finish_over(self, base: Arc<File>) -> Result<Source> {
        if base.source.base.is_some() {
            bail!("a layered source cannot be the base of another");
        }
        let mut source = self.finish(base.source.path.clone())?;
        source.offset = base.source.len();
        source.base = Some(base);
        Ok(source)
    }

    /// The number of tokens written so far.
    pub fn len(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// Marks the tokens written since `from` as one opaque fragment.
    pub fn opaque_from(&mut self, from: u32) {
        if from < self.len() {
            self.opaque.push((from, self.len()));
        }
    }
}

fn is_trivia(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Whitespace | TokenKind::LineComment { doc_style: None } | TokenKind::BlockComment { doc_style: None, .. })
}

/// Whether `kind` opens a delimited group: `(`, `[` or `{`.
pub fn opens(kind: TokenKind) -> bool {
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
    /// The file that `path` names from this source, as `include!` and `include_str!` read it: relative to this source's directory.
    pub fn beside(&self, path: &str) -> PathBuf {
        self.path.parent().unwrap_or(Path::new("")).join(path)
    }

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
        Ok(Source { path, text, tokens, marks: Vec::new(), opaque: Vec::new(), base: None, offset: 0, outputs: Outputs::default() })
    }

    #[inline]
    fn layer(&self, at: u32) -> (&Source, usize) {
        if at >= self.offset {
            return (self, (at - self.offset) as usize);
        }
        self.base_layer(at)
    }

    #[cold]
    fn base_layer(&self, at: u32) -> (&Source, usize) {
        match &self.base {
            Some(base) => (&base.source, at as usize),
            None => (self, at as usize),
        }
    }

    /// Whether a token from `from` up to `to` reads `word`, searching the raw text first when the tokens share one layer.
    pub fn holds_word(&self, (from, to): (u32, u32), word: &str) -> bool {
        if from >= to {
            return false;
        }
        let ((first, start), (last, end)) = (self.layer(from), self.layer(to - 1));
        if std::ptr::eq(first, last) {
            let span = &first.text[first.tokens[start].start as usize..(last.tokens[end].start + last.tokens[end].len) as usize];
            if !span.contains(word) {
                return false;
            }
        }
        (from..to).any(|at| self.text(at) == word)
    }

    /// The text of token `at`.
    pub fn text(&self, at: u32) -> &str {
        let (src, local) = self.layer(at);
        let token = src.tokens[local];
        &src.text[token.start as usize..(token.start + token.len) as usize]
    }

    /// The source text from the start of token `from` to the end of token `to`, inclusive, within the layer of `from`.
    pub fn span(&self, from: u32, to: u32) -> &str {
        let (src, first) = self.layer(from);
        let last = if src.offset == self.offset { (to - self.offset) as usize } else { (to.min(self.offset - 1)) as usize };
        let (first, last) = (src.tokens[first], src.tokens[last]);
        &src.text[first.start as usize..(last.start + last.len) as usize]
    }

    /// The kind of token `at`, or `Eof` past the end.
    pub fn kind(&self, at: u32) -> TokenKind {
        let (src, local) = self.layer(at);
        src.tokens.get(local).map_or(TokenKind::Eof, |token| token.kind)
    }

    /// The token ranges of the comma-separated arguments in the group that opens at `open`.
    pub fn args(&self, open: u32) -> Vec<(u32, u32)> {
        super::attr::split_commas(self, open + 1, self.pair(open))
    }

    /// The text a literal token or `true` and `false` adds to `concat!`, or none for any other token.
    pub fn literal_text(&self, at: u32) -> Result<Option<String>> {
        let text = self.text(at);
        Ok(match self.kind(at) {
            TokenKind::Literal { kind: rustc_lexer::LiteralKind::Str { .. } | rustc_lexer::LiteralKind::RawStr { .. }, .. } => Some(String::from_utf8(crate::body::literal::string_bytes(text)?)?),
            TokenKind::Literal { kind: rustc_lexer::LiteralKind::Char { .. }, .. } => Some(crate::body::literal::char_value(text)?.to_string()),
            TokenKind::Literal { kind: rustc_lexer::LiteralKind::Int { .. } | rustc_lexer::LiteralKind::Float { .. }, suffix_start } => Some(text[..suffix_start as usize].to_owned()),
            TokenKind::Ident if matches!(text, "true" | "false") => Some(text.to_owned()),
            _ => None,
        })
    }

    /// The index of the delimiter that matches token `at`.
    pub fn pair(&self, at: u32) -> u32 {
        let (src, local) = self.layer(at);
        src.tokens[local].pair + (at - local as u32)
    }

    /// Whether token `at` is followed by the next token with no space between them.
    /// The tokens from `from` up to `to` as `stringify!` spells them.
    pub fn spelled(&self, (from, to): (u32, u32)) -> String {
        let mut text = String::new();
        for at in from..to {
            text.push_str(self.text(at));
            if at + 1 < to && !self.joint(at) {
                text.push(' ');
            }
        }
        text
    }

    pub fn joint(&self, at: u32) -> bool {
        let (src, local) = self.layer(at);
        src.tokens[local].joint
    }

    /// The crate that the `$crate` starting at token `at` names, when a macro wrote it.
    pub fn dollar(&self, at: u32) -> Option<u32> {
        match self.mark(at) {
            Some(Mark::Crate(krate)) => Some(krate),
            _ => None,
        }
    }

    /// The hygiene scope of name `at` for local variables and the scope its macro was defined in: the `macro_rules!` expansion that wrote it, or zero for a name the program spelled.
    pub fn scope(&self, at: u32) -> (u32, u32) {
        match self.mark(at) {
            Some(Mark::Scope { id, parent }) => (id, parent),
            _ => (0, 0),
        }
    }

    /// The origin mark of token `at`, when a macro wrote it.
    pub fn mark(&self, at: u32) -> Option<Mark> {
        let (src, local) = self.layer(at);
        src.own_mark(local as u32)
    }

    /// The end of the opaque fragment that starts at token `at`, if one does.
    pub fn opaque_end(&self, at: u32) -> Option<u32> {
        let (src, local) = self.layer(at);
        let found = src.opaque.binary_search_by_key(&(local as u32), |&(start, _)| start).ok()?;
        Some(src.opaque[found].1 + (at - local as u32))
    }

    fn own_mark(&self, local: u32) -> Option<Mark> {
        self.marks.binary_search_by_key(&local, |(token, _)| *token).ok().map(|found| self.marks[found].1)
    }

    /// The byte offset of token `at` in the text of its layer.
    pub fn byte(&self, at: u32) -> u32 {
        let (src, local) = self.layer(at);
        src.tokens.get(local).map_or(src.text.len() as u32, |token| token.start)
    }

    /// The line of token `at` in the text of its layer, counting from one.
    pub fn line(&self, at: u32) -> usize {
        let (src, _) = self.layer(at);
        src.text[..self.byte(at) as usize].matches('\n').count() + 1
    }

    /// The column of token `at` in its line, counting characters from one.
    pub fn column(&self, at: u32) -> usize {
        let (src, _) = self.layer(at);
        let before = &src.text[..self.byte(at) as usize];
        before[before.rfind('\n').map_or(0, |newline| newline + 1)..].chars().count() + 1
    }

    /// The number of tokens, those of the base included.
    pub fn len(&self) -> u32 {
        self.offset + self.tokens.len() as u32
    }

    /// Whether the source has no tokens.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
