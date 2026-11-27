mod lex;
mod storage;
mod tokens;
pub(crate) use lex::pair_up as pair_up_tokens;
use lex::{lex, pair_up};
pub(crate) use storage::DiskSource;
pub use storage::{SourceText, SourceTokens};
pub use tokens::Tokens;

use super::{rules::Frag, tree::File};
use anyhow::{Context, Result};
use rustc_hash::FxHashMap;
use rustc_lexer::TokenKind;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

/// One non-trivia token, with the index of its matching delimiter when it opens or closes a group.
#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub start: u32,
    len: u32,
    pair: u32,
}

/// The bit of a token's length that marks it joint to the next token.
const JOINT: u32 = 1 << 31;

impl Token {
    fn new(kind: TokenKind, start: u32, len: u32, joint: bool) -> Token {
        let mut token = Token {
            kind,
            start,
            len,
            pair: u32::MAX,
        };
        token.set_joint(joint);
        token
    }

    /// The byte offset just past the token.
    fn end(self) -> u32 {
        self.start + (self.len & !JOINT)
    }

    /// Whether the next token follows with no space between them.
    #[inline]
    pub(crate) fn joint(self) -> bool {
        self.len & JOINT != 0
    }

    fn set_joint(&mut self, joint: bool) {
        self.len = (self.len & !JOINT) | (u32::from(joint) * JOINT);
    }

    pub(crate) fn from_raw(kind: TokenKind, start: u32, len: u32, pair: u32) -> Token {
        Token {
            kind,
            start,
            len,
            pair,
        }
    }

    pub(crate) fn raw(self) -> (u32, u32) {
        (self.len, self.pair)
    }
}

/// Whether `kind` is a doc comment, and whether it documents the enclosing item.
pub fn doc(kind: TokenKind) -> Option<bool> {
    match kind {
        TokenKind::LineComment {
            doc_style: Some(style),
        }
        | TokenKind::BlockComment {
            doc_style: Some(style),
            ..
        } => Some(style == rustc_lexer::DocStyle::Inner),
        _ => None,
    }
}

/// Where a token that a macro wrote came from: the `$` of a `$crate` naming a crate, or a name from the body of a `macro` item, which also resolves where that item is defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    Crate(u32),
    Home {
        krate: u32,
        file: u32,
        item: u32,
    },
    /// A name from the body of a `macro_rules!` expansion, which binds local variables only within that expansion and also finds those of the scope where the macro was defined.
    Scope {
        id: u32,
        parent: u32,
    },
}

/// A source file and its tokens, or tokens laid after those of a base file, numbered on from its last one.
#[derive(Debug)]
pub struct Source {
    pub path: PathBuf,
    pub text: SourceText,
    pub tokens: SourceTokens,
    pub marks: Vec<(u32, Mark)>,
    /// The token ranges, by index in this layer, that a macro wrote from an opaque fragment, with its kind.
    pub opaque: Vec<(u32, u32, Frag)>,
    pub base: Option<Arc<File>>,
    pub offset: u32,
    /// The tokens that each macro call in type position writes, by the token that opens its group.
    pub outputs: Outputs,
    /// Where the tokens of this layer lie in the file at `path` as read from disk.
    pub origins: Origins,
    pub(crate) line_starts: OnceLock<Vec<u32>>,
}

/// Where the tokens of a source lie in the file at its path as read from disk: every token where it is, or the listed tokens, by index, at the byte offsets of the tokens a macro copied them from.
#[derive(Debug)]
pub enum Origins {
    /// Every token lies where the source has it.
    Read,
    /// The listed tokens, by index, lie at the byte offsets of the tokens a macro copied them from; the rest lie nowhere on disk.
    Copied(Vec<(u32, u32)>),
}

/// Macro outputs kept for as long as the source that holds their calls, keyed by `(krate, module, open)`.
#[derive(Debug, Default)]
pub struct Outputs(Mutex<FxHashMap<(u32, u32, u32), Box<Source>>>);

impl Outputs {
    /// The output kept for the call `key`.
    pub fn get(&self, key: (u32, u32, u32)) -> Result<Option<&Source>> {
        let kept = crate::sync::lock(&self.0);
        // SAFETY: entries are never removed or replaced while self lives, and a box keeps its heap address when the map grows.
        Ok(kept
            .get(&key)
            .map(|found| unsafe { &*std::ptr::from_ref::<Source>(found) }))
    }

    /// Keeps `output` for the call `key`, or the output another caller kept first.
    pub fn keep(&self, key: (u32, u32, u32), output: Source) -> Result<&Source> {
        let mut kept = crate::sync::lock(&self.0);
        let found = kept.entry(key).or_insert_with(|| Box::new(output));
        // SAFETY: entries are never removed or replaced while self lives, and a box keeps its heap address when the map grows.
        Ok(unsafe { &*std::ptr::from_ref::<Source>(found) })
    }
}

/// The stable 32-bit key of a source file path, the high half of `Source::place`; never 0.
pub fn file_key(path: &Path) -> u32 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    path.hash(&mut hasher);
    (hasher.finish() as u32).max(1)
}

/// Whether `kind` opens a delimited group: `(`, `[` or `{`.
pub fn opens(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::OpenParen | TokenKind::OpenBrace | TokenKind::OpenBracket
    )
}

fn layer_opaques(
    entries: &[(u32, u32, Frag)],
    (low, high): (u32, u32),
    shift: u32,
    out: &mut Vec<(u32, u32, Frag)>,
) {
    let start = entries.partition_point(|&(s, _, _)| s < low);
    for &(s, e, frag) in &entries[start..] {
        if s >= high {
            break;
        }
        if e <= high {
            out.push((s + shift, e + shift, frag));
        }
    }
}

impl Source {
    /// The file that `path` names from this source, as `include!` and `include_str!` read it: relative to this source's directory.
    pub fn beside(&self, path: &str) -> PathBuf {
        self.path.parent().unwrap_or(Path::new("")).join(path)
    }

    /// Reads and lexes the file at `path`.
    pub fn read(path: &Path) -> Result<Source> {
        let read = || -> std::io::Result<String> {
            let mut file = std::fs::File::open(path)?;
            let len = file.metadata()?.len() as usize;
            let mut text = String::with_capacity(len + len / 8);
            std::io::Read::read_to_string(&mut file, &mut text)?;
            Ok(text)
        };
        let text = read().with_context(|| format!("reading {}", path.display()))?;
        let mut source = Source::parse(path.to_owned(), text)?;
        source.origins = Origins::Read;
        Ok(source)
    }

    /// Lexes `text` as the contents of `path`.
    pub fn parse(path: PathBuf, text: String) -> Result<Source> {
        let mut text = text;
        let mut tokens = lex(&mut text);
        pair_up(&mut tokens, &path)?;
        Ok(Source {
            path,
            text: text.into(),
            tokens: tokens.into(),
            marks: Vec::new(),
            opaque: Vec::new(),
            base: None,
            offset: 0,
            outputs: Outputs::default(),
            origins: Origins::Copied(Vec::new()),
            line_starts: OnceLock::new(),
        })
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
            let (Some((f_start, _)), Some((l_start, l_len))) =
                (first.tokens.raw_span(start), last.tokens.raw_span(end))
            else {
                return false;
            };
            let end_byte = (l_start + (l_len & !JOINT)) as usize;
            if (f_start as usize) <= end_byte
                && !first.text[f_start as usize..end_byte].contains(word)
            {
                return false;
            }
        }
        (from..to).any(|at| self.text(at) == word)
    }

    /// The text of token `at`.
    #[inline]
    pub fn text(&self, at: u32) -> &str {
        let (src, local) = self.layer(at);
        let Some((start, raw_len)) = src.tokens.raw_span(local) else {
            return "";
        };
        let end = start + (raw_len & !JOINT);
        &src.text[start as usize..end as usize]
    }

    /// The source text from the start of token `from` to the end of token `to`, inclusive, within the layer of `from`.
    pub fn span(&self, from: u32, to: u32) -> &str {
        if from > to {
            return "";
        }
        let (src, first) = self.layer(from);
        let (to_src, to_local) = self.layer(to);
        let last = if std::ptr::eq(src, to_src) {
            to_local
        } else {
            let clamped = to
                .min(self.offset.saturating_sub(1))
                .saturating_sub(src.offset) as usize;
            clamped.min(src.tokens.len().saturating_sub(1))
        };
        let (Some((f_start, _)), Some((l_start, l_len))) =
            (src.tokens.raw_span(first), src.tokens.raw_span(last))
        else {
            return "";
        };
        let (start, end) = (f_start as usize, (l_start + (l_len & !JOINT)) as usize);
        if start > end || end > src.text.len() {
            return "";
        }
        &src.text[start..end]
    }

    /// The kind of token `at`, or `Eof` past the end.
    #[inline]
    pub fn kind(&self, at: u32) -> TokenKind {
        let (src, local) = self.layer(at);
        src.tokens
            .at(local)
            .map_or(TokenKind::Eof, |token| token.kind)
    }

    #[inline]
    pub(crate) fn token_at(&self, at: u32) -> Option<Token> {
        let (src, local) = self.layer(at);
        src.tokens.at(local)
    }

    /// The token ranges of the comma-separated arguments in the group that opens at `open`.
    pub fn args(&self, open: u32) -> Vec<(u32, u32)> {
        super::attr::split_commas(self, open + 1, self.pair(open))
    }

    /// The text a literal token or `true` and `false` adds to `concat!`, or none for any other token.
    pub fn literal_text(&self, at: u32) -> Result<Option<String>> {
        let text = self.text(at);
        Ok(match self.kind(at) {
            TokenKind::Literal {
                kind: rustc_lexer::LiteralKind::Str { .. } | rustc_lexer::LiteralKind::RawStr { .. },
                ..
            } => Some(String::from_utf8(crate::body::literal::string_bytes(
                text,
            )?)?),
            TokenKind::Literal {
                kind: rustc_lexer::LiteralKind::Char { .. },
                ..
            } => Some(crate::body::literal::char_value(text)?.to_string()),
            TokenKind::Literal {
                kind: rustc_lexer::LiteralKind::Int { .. } | rustc_lexer::LiteralKind::Float { .. },
                suffix_start,
            } => Some(text[..suffix_start as usize].to_owned()),
            TokenKind::Ident if matches!(text, "true" | "false") => Some(text.to_owned()),
            _ => None,
        })
    }

    /// The index of the delimiter that matches token `at`.
    #[inline]
    pub fn pair(&self, at: u32) -> u32 {
        let (src, local) = self.layer(at);
        src.tokens
            .at(local)
            .map_or(at, |token| token.pair + (at - local as u32))
    }

    /// The tokens from `from` up to `to` as `stringify!` spells them.
    pub fn spelled(&self, range: (u32, u32)) -> String {
        self.spelled_with(range, |_| 0)
    }

    /// Spells `from..to` with `skip(at)` returning `0` to emit, `1` to drop an opening wrapper, or `2` to drop a closing wrapper while preserving trailing whitespace.
    pub(crate) fn spelled_with(
        &self,
        (from, to): (u32, u32),
        mut skip: impl FnMut(u32) -> u8,
    ) -> String {
        let mut text = String::new();
        let mut pending_space = false;
        for at in from..to {
            match skip(at) {
                1 => continue,
                2 => {
                    if !text.is_empty() && !self.joint(at) {
                        pending_space = true;
                    }
                    continue;
                }
                _ => {}
            }
            if pending_space {
                text.push(' ');
            }
            text.push_str(self.text(at));
            pending_space = !self.joint(at);
        }
        text
    }

    #[inline]
    pub fn joint(&self, at: u32) -> bool {
        let (src, local) = self.layer(at);
        src.tokens
            .raw_span(local)
            .is_some_and(|(_, raw_len)| raw_len & JOINT != 0)
    }

    /// The crate that the `$crate` starting at token `at` names, when a macro wrote it.
    #[inline]
    pub fn dollar(&self, at: u32) -> Option<u32> {
        match self.mark(at) {
            Some(Mark::Crate(krate)) => Some(krate),
            _ => None,
        }
    }

    /// The hygiene scope of name `at` for local variables and the scope its macro was defined in: the `macro_rules!` expansion that wrote it, or zero for a name the program spelled.
    #[inline]
    pub fn scope(&self, at: u32) -> (u32, u32) {
        match self.mark(at) {
            Some(Mark::Scope { id, parent }) => (id, parent),
            _ => (0, 0),
        }
    }

    /// The origin mark of token `at`, when a macro wrote it.
    #[inline]
    pub fn mark(&self, at: u32) -> Option<Mark> {
        if self.marks.is_empty() && self.base.is_none() {
            return None;
        }
        let (src, local) = self.layer(at);
        src.own_mark(local as u32)
    }

    /// The end and fragment kind of the opaque fragment that starts at token `at`, if one does.
    #[inline]
    pub fn opaque_at(&self, at: u32) -> Option<(u32, Frag)> {
        if self.opaque.is_empty() && self.base.is_none() {
            return None;
        }
        let (src, local) = self.layer(at);
        let found = src
            .opaque
            .binary_search_by_key(&(local as u32), |&(start, _, _)| start)
            .ok()?;
        let (_, end, frag) = src.opaque[found];
        Some((end + (at - local as u32), frag))
    }

    /// Each opaque fragment of `self` that lies within `from..to`.
    pub fn opaques_in(&self, from: u32, to: u32) -> Vec<(u32, u32, Frag)> {
        let mut found = Vec::new();
        if from >= to {
            return found;
        }
        if from < self.offset
            && let Some(base) = &self.base
        {
            layer_opaques(
                &base.source.opaque,
                (from, to.min(self.offset)),
                0,
                &mut found,
            );
        }
        if to > self.offset {
            layer_opaques(
                &self.opaque,
                (from.saturating_sub(self.offset), to - self.offset),
                self.offset,
                &mut found,
            );
        }
        found
    }

    fn own_mark(&self, local: u32) -> Option<Mark> {
        self.marks
            .binary_search_by_key(&local, |(token, _)| *token)
            .ok()
            .map(|found| self.marks[found].1)
    }

    /// The byte offset in the file at `path`, as read from disk, of token `at` or of the token a macro copied it from.
    pub fn origin(&self, at: u32) -> Option<u32> {
        let (src, local) = self.layer(at);
        match &src.origins {
            Origins::Read => src.tokens.raw_span(local).map(|(start, _)| start),
            Origins::Copied(copied) => copied
                .binary_search_by_key(&(local as u32), |&(token, _)| token)
                .ok()
                .map(|found| copied[found].1),
        }
    }

    /// The source place of token `at` for debug line tables: `file_key(path)` in the high 32 bits and the disk byte offset in the low 32, or 0 when the token lies nowhere on disk.
    pub fn place(&self, at: u32) -> u64 {
        self.origin(at).map_or(0, |byte| {
            u64::from(file_key(&self.path)) << 32 | u64::from(byte)
        })
    }

    /// The byte offset of token `at` in the text of its layer.
    pub fn byte(&self, at: u32) -> u32 {
        let (src, local) = self.layer(at);
        src.tokens
            .raw_span(local)
            .map_or(src.text.len() as u32, |(start, _)| start)
    }

    /// Byte offsets of every `\n` in this layer's text, computed once on first lookup.
    pub(crate) fn line_starts(&self) -> &[u32] {
        self.line_starts.get_or_init(|| {
            let mut starts = Vec::new();
            let bytes = self.text.as_bytes();
            let mut i = 0;
            while let Some(rel) = bytes[i..].iter().position(|&b| b == b'\n') {
                let nl = i + rel;
                starts.push(nl as u32);
                i = nl + 1;
            }
            starts
        })
    }

    /// One-based `(line, column)` of byte offset `byte` in this layer's text.
    pub(crate) fn line_col_at(&self, byte: usize) -> (usize, usize) {
        let mut byte = byte.min(self.text.len());
        while byte > 0 && !self.text.is_char_boundary(byte) {
            byte -= 1;
        }
        let newlines = self.line_starts();
        let line_idx = newlines.partition_point(|&nl| (nl as usize) < byte);
        let line_start = if line_idx == 0 {
            0
        } else {
            newlines[line_idx - 1] as usize + 1
        };
        let col = self.text[line_start..byte].chars().count() + 1;
        (line_idx + 1, col)
    }

    /// The line of token `at` in the text of its layer, counting from one.
    pub fn line(&self, at: u32) -> usize {
        let (src, _) = self.layer(at);
        src.line_col_at(self.byte(at) as usize).0
    }

    /// The column of token `at` in its line, counting characters from one.
    pub fn column(&self, at: u32) -> usize {
        let (src, _) = self.layer(at);
        src.line_col_at(self.byte(at) as usize).1
    }

    /// Resolves token range `start_tok..end_tok` (half-open) to `(layer, start_byte, end_byte, from_macro)`.
    pub(crate) fn token_byte_span(
        &self,
        start_tok: u32,
        end_tok: u32,
    ) -> Option<(&Source, u32, u32, bool)> {
        if self.is_empty() || start_tok >= self.len() {
            return None;
        }
        let max_tok = self.len().saturating_sub(1);
        let first_tok = start_tok;
        let last_tok = end_tok.saturating_sub(1).max(first_tok).min(max_tok);
        let (s_layer, s_local) = self.layer(first_tok);
        let (e_layer, e_local) = self.layer(last_tok);
        let (s_start, s_raw_len) = s_layer.tokens.raw_span(s_local)?;
        let s_len = s_raw_len & !JOINT;
        let e_span = e_layer.tokens.raw_span(e_local);
        if first_tok >= self.offset && matches!(s_layer.origins, Origins::Copied(_)) {
            let base = self.base.as_deref()?;
            let sb = self.origin(first_tok)?;
            let text_len = base.source.text.len() as u32;
            let sb = sb.min(text_len);
            let same_line_end = self.origin(last_tok).zip(e_span).and_then(|(eb, (_, el))| {
                let eb_clamped = eb.min(text_len);
                let end = eb.saturating_add(el & !JOINT).min(text_len);
                (eb >= sb
                    && !base.source.text.as_bytes()[sb as usize..eb_clamped as usize]
                        .contains(&b'\n'))
                .then_some(end)
            });
            let end_byte = same_line_end
                .unwrap_or_else(|| sb.saturating_add(s_len).min(text_len))
                .max(sb);
            return Some((&base.source, sb, end_byte, true));
        }
        if matches!(s_layer.origins, Origins::Copied(_)) {
            return None;
        }
        let text_len = s_layer.text.len() as u32;
        let end_byte = if std::ptr::eq(s_layer, e_layer)
            && let Some((e_start, e_raw_len)) = e_span
        {
            e_start.saturating_add(e_raw_len & !JOINT)
        } else if let Some(eb) = self.origin(last_tok)
            && let Some((_, el)) = e_span
            && eb >= s_start
            && !s_layer.text.as_bytes()[s_start.min(text_len) as usize..eb.min(text_len) as usize]
                .contains(&b'\n')
        {
            eb.saturating_add(el & !JOINT)
        } else {
            s_start.saturating_add(s_len)
        }
        .min(text_len);
        let start_byte = s_start.min(end_byte);
        Some((s_layer, start_byte, end_byte, false))
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
