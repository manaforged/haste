use std::path::Path;

use anyhow::{Result, bail};
use rustc_lexer::{FrontmatterAllowed, TokenKind, tokenize};

use super::{Token, doc, opens};

pub(super) fn is_trivia(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Whitespace
            | TokenKind::LineComment { doc_style: None }
            | TokenKind::BlockComment {
                doc_style: None,
                ..
            }
    )
}

fn closes(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::CloseParen | TokenKind::CloseBrace | TokenKind::CloseBracket
    )
}

fn raw_hashes(body: &str) -> usize {
    let bytes = body.as_bytes();
    let runs = bytes
        .iter()
        .enumerate()
        .filter(|(_, byte)| **byte == b'"')
        .map(|(at, _)| {
            bytes[at + 1..]
                .iter()
                .take_while(|byte| **byte == b'#')
                .count()
        });
    runs.max().map_or(1, |most| most + 1)
}

fn doc_tokens(comment: &str, inner: bool, (tail, base): (&mut String, u32), into: &mut Vec<Token>) {
    let raw = if comment.starts_with("/*") {
        &comment[3..comment.len().saturating_sub(2).max(3)]
    } else {
        comment[3..].strip_suffix('\r').unwrap_or(&comment[3..])
    };
    let normalized;
    let body = if raw.contains('\r') {
        normalized = raw.replace("\r\n", "\n").replace('\r', "");
        normalized.as_str()
    } else {
        raw
    };
    let hashes = raw_hashes(body);
    let mut emit = |tail: &mut String, kind: TokenKind, pieces: &[&str]| {
        let start = base + tail.len() as u32;
        pieces.iter().for_each(|piece| tail.push_str(piece));
        into.push(Token::new(
            kind,
            start,
            base + tail.len() as u32 - start,
            true,
        ));
    };
    emit(tail, TokenKind::Pound, &["#"]);
    if inner {
        emit(tail, TokenKind::Bang, &["!"]);
    }
    emit(tail, TokenKind::OpenBracket, &["["]);
    emit(tail, TokenKind::Ident, &["doc"]);
    emit(tail, TokenKind::Eq, &["="]);
    let fence = &"################################"[..hashes.min(32)];
    let literal = TokenKind::Literal {
        kind: rustc_lexer::LiteralKind::RawStr {
            n_hashes: Some(hashes as u8),
        },
        suffix_start: (body.len() + 3 + 2 * hashes) as u32,
    };
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
        Lexer {
            text,
            tail: String::new(),
            tokens: Vec::with_capacity(text.len() / 5),
            groups: vec![Group::default()],
            run: None,
            after_doc: false,
            spare: Vec::new(),
        }
    }

    fn group(&mut self) -> &mut Group {
        let last = self.groups.len() - 1;
        &mut self.groups[last]
    }

    /// The text of `token`, in the file or in the doc text written after it.
    fn piece(&self, token: &Token) -> &str {
        let (start, end, len) = (token.start as usize, token.end() as usize, self.text.len());
        if start < len {
            &self.text[start..end]
        } else {
            &self.tail[start - len..end - len]
        }
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
        self.run.get_or_insert(Run {
            from,
            depth,
            docs: false,
            shown: false,
            attr: 0,
        })
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
        doc_tokens(
            &text[token.start as usize..token.end() as usize],
            inner,
            (&mut self.tail, text.len() as u32),
            &mut self.tokens,
        );
    }

    fn track(&mut self, kind: TokenKind) {
        if opens(kind) {
            self.open();
        } else if closes(kind) {
            self.close(kind);
        } else {
            let word = matches!(kind, TokenKind::Ident | TokenKind::RawIdent);
            let named_macro = kind == TokenKind::Ident
                && self
                    .tokens
                    .last()
                    .is_some_and(|last| self.piece(last) == "macro");
            let group = self.group();
            if kind == TokenKind::Semi || (word && group.after_brace) {
                group.shown = false;
            }
            group.after_brace = false;
            group.shown |= named_macro;
        }
    }

    fn open(&mut self) {
        let before = |back: usize| {
            self.tokens
                .len()
                .checked_sub(back + 1)
                .map(|at| self.tokens[at].kind)
        };
        let call = before(1) == Some(TokenKind::Bang)
            || (matches!(before(1), Some(TokenKind::Ident | TokenKind::RawIdent))
                && before(2) == Some(TokenKind::Bang));
        let depth = self.groups.len();
        let at = self.tokens.len() - 1;
        if let Some(run) = self.run.as_mut().filter(|run| run.depth == depth) {
            run.attr = at;
        }
        let parent = self.group();
        parent.after_brace = false;
        let keep = parent.keep || parent.shown || call;
        self.groups.push(Group {
            keep,
            ..Group::default()
        });
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
        let call = attr
            .get(1)
            .is_some_and(|next| next.kind == TokenKind::OpenParen);
        match attr.first().and_then(word) {
            None => true,
            Some(name @ ("cfg" | "cfg_attr")) if call => {
                name == "cfg_attr"
                    && attr
                        .iter()
                        .filter_map(word)
                        .any(|path| matches!(path, "derive" | "derive_const"))
            }
            Some(name) => !super::super::build::inert(name),
        }
    }

    fn finish(mut self) -> (Vec<Token>, String) {
        if let Some(run) = self.run.take() {
            self.settle(run);
        }
        (self.tokens, self.tail)
    }
}

pub(super) fn lex(text: &mut String) -> Vec<Token> {
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

pub(crate) fn pair_up(tokens: &mut [Token], path: &Path) -> Result<()> {
    let mut open = Vec::with_capacity(64);
    for at in 0..tokens.len() {
        let kind = tokens[at].kind;
        if opens(kind) {
            open.push(at);
        } else if closes(kind) {
            let Some(start) = open.pop() else {
                bail!(
                    "{}: unmatched closing delimiter at byte {}",
                    path.display(),
                    tokens[at].start
                )
            };
            tokens[start].pair = at as u32;
            tokens[at].pair = start as u32;
        }
    }
    if let Some(start) = open.pop() {
        bail!(
            "{}: unclosed delimiter at byte {}",
            path.display(),
            tokens[start].start
        );
    }
    Ok(())
}
