use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::rules::Frag;
use super::tree::File;

use anyhow::{Context, Result, bail};
use rustc_hash::FxHashMap;
use rustc_lexer::{FrontmatterAllowed, TokenKind, tokenize};

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
        let mut token = Token { kind, start, len, pair: u32::MAX };
        token.set_joint(joint);
        token
    }

    /// The byte offset just past the token.
    fn end(self) -> u32 {
        self.start + (self.len & !JOINT)
    }

    /// Whether the next token follows with no space between them.
    fn joint(self) -> bool {
        self.len & JOINT != 0
    }

    fn set_joint(&mut self, joint: bool) {
        self.len = (self.len & !JOINT) | (u32::from(joint) * JOINT);
    }
}

/// Whether `kind` is a doc comment, and whether it documents the enclosing item.
pub fn doc(kind: TokenKind) -> Option<bool> {
    match kind {
        TokenKind::LineComment { doc_style: Some(style) } | TokenKind::BlockComment { doc_style: Some(style), .. } => Some(style == rustc_lexer::DocStyle::Inner),
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
    pub text: String,
    pub tokens: Vec<Token>,
    pub marks: Vec<(u32, Mark)>,
    /// The token ranges, by index in this layer, that a macro wrote from an opaque fragment, with its kind.
    pub opaque: Vec<(u32, u32, Frag)>,
    pub base: Option<Arc<File>>,
    pub offset: u32,
    /// The tokens that each macro call in type position writes, by the token that opens its group.
    pub outputs: Outputs,
    /// Where the tokens of this layer lie in the file at `path` as read from disk.
    pub origins: Origins,
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
        let kept = self.0.lock().map_err(|_| anyhow::anyhow!("macro outputs poisoned"))?;
        // SAFETY: entries are never removed or replaced while self lives, and a box keeps its heap address when the map grows.
        Ok(kept.get(&key).map(|found| unsafe { &*std::ptr::from_ref::<Source>(found) }))
    }

    /// Keeps `output` for the call `key`, or the output another caller kept first.
    pub fn keep(&self, key: (u32, u32, u32), output: Source) -> Result<&Source> {
        let mut kept = self.0.lock().map_err(|_| anyhow::anyhow!("macro outputs poisoned"))?;
        let found = kept.entry(key).or_insert_with(|| Box::new(output));
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
    opaque: Vec<(u32, u32, Frag)>,
    origins: Vec<(u32, u32)>,
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
        self.tokens.push(Token::new(kind, start, text.len() as u32, joint));
    }

    /// Copies token `at` of `src`, with its spacing and its origin mark.
    pub fn copy(&mut self, src: &Source, at: u32) {
        let (layer, local) = src.layer(at);
        self.copy_local(layer, local, layer.own_mark(local as u32));
    }

    /// Copies token `at` of `src` with its spacing, marked as coming from `mark`.
    pub fn copy_marked(&mut self, src: &Source, at: u32, mark: Option<Mark>) {
        let (layer, local) = src.layer(at);
        self.copy_local(layer, local, mark);
    }

    fn copy_local(&mut self, layer: &Source, local: usize, mark: Option<Mark>) {
        let token = layer.tokens[local];
        if let Some(mark) = mark {
            self.marks.push((self.tokens.len() as u32, mark));
        }
        self.append(token.kind, &layer.text[token.start as usize..token.end() as usize], token.joint());
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
        if let Some(last) = self.tokens.last_mut().filter(|last| last.joint()) {
            last.set_joint(false);
            self.text.push(' ');
        }
    }

    /// The written tokens as a source with the name `path`.
    pub fn finish(self, path: PathBuf) -> Result<Source> {
        let mut tokens = self.tokens;
        pair_up(&mut tokens, &path)?;
        Ok(Source { path, text: self.text, tokens, marks: self.marks, opaque: self.opaque, base: None, offset: 0, outputs: Outputs::default(), origins: Origins::Copied(self.origins) })
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

    /// Copies tokens `from..to` of `src`, keeping their spacing, origin marks, places on disk and opaque sub-ranges.
    pub fn copy_range(&mut self, src: &Source, from: u32, to: u32) {
        let base = self.len();
        for at in from..to {
            if let Some(origin) = src.origin(at) {
                self.origins.push((self.len(), origin));
            }
            self.copy(src, at);
        }
        for (op_from, op_to, frag) in src.opaques_in(from, to) {
            self.opaque.push((base + (op_from - from), base + (op_to - from), frag));
        }
    }

    /// Marks the tokens written since `from` as one opaque fragment of kind `frag`.
    pub fn opaque_from(&mut self, from: u32, frag: Frag) {
        let to = self.len();
        if from >= to {
            return;
        }
        let slot = self.opaque.partition_point(|&(start, _, _)| start < from);
        if self.opaque.get(slot).is_some_and(|&(start, _, _)| start == from) {
            self.opaque[slot] = (from, to, frag);
        } else {
            self.opaque.insert(slot, (from, to, frag));
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
        into.push(Token::new(kind, start, base + tail.len() as u32 - start, true));
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
        last.set_joint(false);
    }
    tail.push('\n');
}

/// What the lexer knows of one open delimited group.
#[derive(Clone, Copy, Default)]
struct Group {
    /// Every doc comment inside is kept: the group is macro input, or part of an item that a macro reads.
    keep: bool,
    /// The attributes read last at this level hand their item to a macro, so the groups of that item keep their doc comments.
    shown: bool,
    /// The last token at this level closed a brace group, so a name now starts another item.
    after_brace: bool,
}

/// The doc comments and attributes in front of one item, still being read.
#[derive(Clone, Copy)]
struct Run {
    from: usize,
    depth: usize,
    docs: bool,
    shown: bool,
    attr: usize,
}

/// The tokens of one file as they are lexed: doc comments become `#[doc = r"..."]` where a macro can read them, and are dropped where nothing reads them.
struct Lexer<'a> {
    text: &'a str,
    tail: String,
    tokens: Vec<Token>,
    groups: Vec<Group>,
    run: Option<Run>,
    after_doc: bool,
    spare: Vec<Token>,
}

impl<'a> Lexer<'a> {
    fn new(text: &'a str) -> Self {
        Lexer { text, tail: String::new(), tokens: Vec::with_capacity(text.len() / 5), groups: vec![Group::default()], run: None, after_doc: false, spare: Vec::new() }
    }

    fn group(&mut self) -> &mut Group {
        let last = self.groups.len() - 1;
        &mut self.groups[last]
    }

    /// The text of `token`, in the file or in the doc text written after it.
    fn piece(&self, token: &Token) -> &str {
        let (start, end, len) = (token.start as usize, token.end() as usize, self.text.len());
        if start < len { &self.text[start..end] } else { &self.tail[start - len..end - len] }
    }

    fn push(&mut self, token: Token, joint: bool) {
        if joint
            && !self.after_doc
            && let Some(last) = self.tokens.last_mut()
        {
            last.set_joint(true);
        }
        self.after_doc = doc(token.kind).is_some();
        if let Some(run) = self.run.filter(|run| !self.continues(run, token.kind)) {
            self.run = None;
            self.settle(run);
        }
        let keep = self.group().keep;
        match doc(token.kind) {
            Some(inner) if keep => self.expand(token, inner),
            Some(_) => {
                self.start_run().docs = true;
                self.tokens.push(token);
            }
            None => {
                if token.kind == TokenKind::Pound && !keep {
                    self.start_run();
                }
                self.tokens.push(token);
                self.track(token.kind);
            }
        }
    }

    fn start_run(&mut self) -> &mut Run {
        let (from, depth) = (self.tokens.len(), self.groups.len());
        self.run.get_or_insert(Run { from, depth, docs: false, shown: false, attr: 0 })
    }

    /// Whether a token of `kind` still belongs to the doc comments and attributes of `run`.
    fn continues(&self, run: &Run, kind: TokenKind) -> bool {
        if self.groups.len() > run.depth || doc(kind).is_some() {
            return true;
        }
        let last = self.tokens.last().map(|token| token.kind);
        match kind {
            TokenKind::Pound => true,
            TokenKind::Bang => last == Some(TokenKind::Pound),
            TokenKind::OpenBracket => matches!(last, Some(TokenKind::Pound | TokenKind::Bang)),
            _ => false,
        }
    }

    /// Ends `run`: its doc comments become attributes when an attribute of the run hands the item to a macro, and are dropped otherwise.
    fn settle(&mut self, run: Run) {
        self.group().shown = run.shown;
        if !run.docs {
            return;
        }
        if run.shown {
            let mut spare = std::mem::take(&mut self.spare);
            spare.extend(self.tokens.drain(run.from..));
            for token in spare.drain(..) {
                match doc(token.kind) {
                    Some(inner) => self.expand(token, inner),
                    None => self.tokens.push(token),
                }
            }
            self.spare = spare;
        } else {
            self.drop_docs(run.from);
        }
    }

    fn drop_docs(&mut self, from: usize) {
        let mut write = from;
        for read in from..self.tokens.len() {
            let token = self.tokens[read];
            if doc(token.kind).is_none() {
                self.tokens[write] = token;
                write += 1;
            } else if let Some(before) = write.checked_sub(1) {
                self.tokens[before].set_joint(false);
            }
        }
        self.tokens.truncate(write);
    }

    fn expand(&mut self, token: Token, inner: bool) {
        if self.tail.is_empty() {
            self.tail.push('\n');
        }
        let text = self.text;
        doc_tokens(&text[token.start as usize..token.end() as usize], inner, (&mut self.tail, text.len() as u32), &mut self.tokens);
    }

    fn track(&mut self, kind: TokenKind) {
        if opens(kind) {
            self.open();
        } else if closes(kind) {
            self.close(kind);
        } else {
            let word = matches!(kind, TokenKind::Ident | TokenKind::RawIdent);
            let named_macro = kind == TokenKind::Ident && self.tokens.last().is_some_and(|last| self.piece(last) == "macro");
            let group = self.group();
            if kind == TokenKind::Semi || (word && group.after_brace) {
                group.shown = false;
            }
            group.after_brace = false;
            group.shown |= named_macro;
        }
    }

    fn open(&mut self) {
        let before = |back: usize| self.tokens.len().checked_sub(back + 1).map(|at| self.tokens[at].kind);
        let call = before(1) == Some(TokenKind::Bang) || (matches!(before(1), Some(TokenKind::Ident | TokenKind::RawIdent)) && before(2) == Some(TokenKind::Bang));
        let depth = self.groups.len();
        let at = self.tokens.len() - 1;
        if let Some(run) = self.run.as_mut().filter(|run| run.depth == depth) {
            run.attr = at;
        }
        let parent = self.group();
        parent.after_brace = false;
        let keep = parent.keep || parent.shown || call;
        self.groups.push(Group { keep, ..Group::default() });
    }

    fn close(&mut self, kind: TokenKind) {
        if self.groups.len() > 1 {
            self.groups.pop();
        }
        self.group().after_brace = kind == TokenKind::CloseBrace;
        let (depth, end) = (self.groups.len(), self.tokens.len() - 1);
        if let Some(run) = self.run.filter(|run| run.depth == depth && run.attr < end) {
            let shown = self.reaches_macro(&self.tokens[run.attr + 1..end]);
            if let Some(run) = self.run.as_mut() {
                run.shown |= shown;
            }
        }
    }

    /// Whether the attribute with `attr` inside its brackets hands the item it stands on to a macro: a derive, also under `cfg_attr`, or any path that is not a builtin attribute.
    fn reaches_macro(&self, attr: &[Token]) -> bool {
        let word = |token: &Token| (token.kind == TokenKind::Ident).then(|| self.piece(token));
        let call = attr.get(1).is_some_and(|next| next.kind == TokenKind::OpenParen);
        match attr.first().and_then(word) {
            None => true,
            Some(name @ ("cfg" | "cfg_attr")) if call => name == "cfg_attr" && attr.iter().filter_map(word).any(|path| matches!(path, "derive" | "derive_const")),
            Some(name) => !super::build::inert(name),
        }
    }

    fn finish(mut self) -> (Vec<Token>, String) {
        if let Some(run) = self.run.take() {
            self.settle(run);
        }
        (self.tokens, self.tail)
    }
}

fn lex(text: &mut String) -> Vec<Token> {
    let bom = if text.starts_with('\u{feff}') { 3 } else { 0 };
    let skip = bom + rustc_lexer::strip_shebang(&text[bom..]).unwrap_or(0);
    let mut lexer = Lexer::new(text);
    let (mut start, mut joint) = (skip as u32, false);
    for token in tokenize(&text[skip..], FrontmatterAllowed::No) {
        if is_trivia(token.kind) {
            joint = false;
        } else {
            lexer.push(Token::new(token.kind, start, token.len, false), joint);
            joint = true;
        }
        start += token.len;
    }
    let (tokens, tail) = lexer.finish();
    text.reserve_exact(tail.len());
    text.push_str(&tail);
    tokens
}

fn pair_up(tokens: &mut [Token], path: &Path) -> Result<()> {
    let mut open = Vec::with_capacity(64);
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

fn layer_opaques(entries: &[(u32, u32, Frag)], (low, high): (u32, u32), shift: u32, out: &mut Vec<(u32, u32, Frag)>) {
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
        Ok(Source { path, text, tokens, marks: Vec::new(), opaque: Vec::new(), base: None, offset: 0, outputs: Outputs::default(), origins: Origins::Copied(Vec::new()) })
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
            let span = &first.text[first.tokens[start].start as usize..last.tokens[end].end() as usize];
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
        &src.text[token.start as usize..token.end() as usize]
    }

    /// The source text from the start of token `from` to the end of token `to`, inclusive, within the layer of `from`.
    pub fn span(&self, from: u32, to: u32) -> &str {
        let (src, first) = self.layer(from);
        let last = if src.offset == self.offset { (to - self.offset) as usize } else { (to.min(self.offset - 1)) as usize };
        let (first, last) = (src.tokens[first], src.tokens[last]);
        &src.text[first.start as usize..last.end() as usize]
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
        src.tokens[local].joint()
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

    /// The end and fragment kind of the opaque fragment that starts at token `at`, if one does.
    pub fn opaque_at(&self, at: u32) -> Option<(u32, Frag)> {
        let (src, local) = self.layer(at);
        let found = src.opaque.binary_search_by_key(&(local as u32), |&(start, _, _)| start).ok()?;
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
            layer_opaques(&base.source.opaque, (from, to.min(self.offset)), 0, &mut found);
        }
        if to > self.offset {
            layer_opaques(&self.opaque, (from.saturating_sub(self.offset), to - self.offset), self.offset, &mut found);
        }
        found
    }

    fn own_mark(&self, local: u32) -> Option<Mark> {
        self.marks.binary_search_by_key(&local, |(token, _)| *token).ok().map(|found| self.marks[found].1)
    }

    /// The byte offset in the file at `path`, as read from disk, of token `at` or of the token a macro copied it from.
    pub fn origin(&self, at: u32) -> Option<u32> {
        let (src, local) = self.layer(at);
        match &src.origins {
            Origins::Read => src.tokens.get(local).map(|token| token.start),
            Origins::Copied(copied) => copied.binary_search_by_key(&(local as u32), |&(token, _)| token).ok().map(|found| copied[found].1),
        }
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
