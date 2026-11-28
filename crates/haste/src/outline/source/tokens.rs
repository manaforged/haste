use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail};
use rustc_lexer::{FrontmatterAllowed, TokenKind, tokenize};

use super::super::{rules::Frag, tree::File};
use super::lex::{is_trivia, pair_up};
use super::{Mark, Origins, Outputs, Source, Token};

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
    !matches!(
        kind,
        TokenKind::Ident
            | TokenKind::RawIdent
            | TokenKind::Literal { .. }
            | TokenKind::Lifetime { .. }
            | TokenKind::RawLifetime
    )
}

impl Tokens {
    pub fn with_capacity(tokens: usize, text: usize) -> Self {
        Self {
            text: String::with_capacity(text),
            tokens: Vec::with_capacity(tokens),
            marks: Vec::new(),
            opaque: Vec::new(),
            origins: Vec::new(),
        }
    }

    fn append(&mut self, kind: TokenKind, text: &str, joint: bool) {
        let start = self.text.len() as u32;
        self.text.push_str(text);
        let joint = joint && is_punct_kind(kind);
        if !joint {
            self.text.push(' ');
        }
        self.tokens
            .push(Token::new(kind, start, text.len() as u32, joint));
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
        let token = layer.tokens.at(local).expect("token index in bounds");
        if let Some(mark) = mark {
            self.marks.push((self.tokens.len() as u32, mark));
        }
        self.append(
            token.kind,
            &layer.text[token.start as usize..token.end() as usize],
            token.joint(),
        );
    }

    #[inline]
    pub fn punct(&mut self, kind: TokenKind, text: &str, joint: bool) {
        self.append(kind, text, joint);
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
        self.marks
            .push((self.tokens.len() as u32, Mark::Crate(krate)));
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
        Ok(Source {
            path,
            text: self.text.into(),
            tokens: tokens.into(),
            marks: self.marks.into(),
            opaque: self.opaque,
            base: None,
            offset: 0,
            outputs: Outputs::default(),
            origins: Origins::Copied(self.origins),
            line_starts: OnceLock::new(),
            disk_len: None,
        })
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
        if from >= to {
            return;
        }
        let count = (to - from) as usize;
        self.tokens.reserve(count);
        let base = self.len();
        if from < src.offset
            && let Some(parent) = &src.base
        {
            self.copy_layer_range(&parent.source, from as usize, to.min(src.offset) as usize);
        }
        if to > src.offset {
            let lo = from.saturating_sub(src.offset) as usize;
            let hi = (to - src.offset) as usize;
            self.copy_layer_range(src, lo, hi);
        }
        for (op_from, op_to, frag) in src.opaques_in(from, to) {
            self.opaque
                .push((base + (op_from - from), base + (op_to - from), frag));
        }
    }

    fn copy_layer_range(&mut self, layer: &Source, lo: usize, hi: usize) {
        if lo >= hi {
            return;
        }
        let slice = &layer.tokens.as_slice()[lo..hi];
        let base_dst = self.tokens.len() as u32;
        if !layer.marks.is_empty() {
            let mut mark_idx = layer.marks.partition_point(|&(t, _)| (t as usize) < lo);
            while let Some(&(t, mark)) = layer.marks.get(mark_idx) {
                if t as usize >= hi {
                    break;
                }
                self.marks.push((base_dst + (t - lo as u32), mark));
                mark_idx += 1;
            }
        }
        match &layer.origins {
            Origins::Read => {
                self.origins.reserve(slice.len());
                for (offset, &token) in slice.iter().enumerate() {
                    self.origins.push((base_dst + offset as u32, token.start));
                }
            }
            Origins::Copied(v) if !v.is_empty() => {
                let mut orig_idx = v.partition_point(|&(t, _)| (t as usize) < lo);
                while let Some(&(t, pos)) = v.get(orig_idx) {
                    if t as usize >= hi {
                        break;
                    }
                    self.origins.push((base_dst + (t - lo as u32), pos));
                    orig_idx += 1;
                }
            }
            Origins::Copied(_) => {}
        }
        let layer_text = layer.text.as_str();
        let first_start = slice[0].start;
        let last = slice[slice.len() - 1];
        let last_end = last.end();
        let span = last_end.saturating_sub(first_start) as usize;
        let can_bulk = last_end >= first_start
            && (last_end as usize) <= layer_text.len()
            && (matches!(layer.origins, Origins::Copied(_)) || span <= slice.len() * 12 + 64)
            && slice
                .iter()
                .all(|t| t.start >= first_start && t.end() <= last_end);
        if can_bulk {
            let dst_text_base = self.text.len() as u32;
            let last_joint = last.joint() && is_punct_kind(last.kind);
            self.text.reserve(span + usize::from(!last_joint));
            self.text
                .push_str(&layer_text[first_start as usize..last_end as usize]);
            if !last_joint {
                self.text.push(' ');
            }
            self.tokens.extend(slice.iter().map(|&token| {
                let start = dst_text_base + (token.start - first_start);
                let joint = token.joint() && is_punct_kind(token.kind);
                Token::new(token.kind, start, token.end() - token.start, joint)
            }));
            return;
        }
        self.text.reserve(span.min(layer_text.len()) + slice.len());
        for &token in slice {
            self.append(
                token.kind,
                &layer_text[token.start as usize..token.end() as usize],
                token.joint(),
            );
        }
    }

    /// Marks the tokens written since `from` as one opaque fragment of kind `frag`.
    pub fn opaque_from(&mut self, from: u32, frag: Frag) {
        let to = self.len();
        if from >= to {
            return;
        }
        let slot = self.opaque.partition_point(|&(start, _, _)| start < from);
        if self
            .opaque
            .get(slot)
            .is_some_and(|&(start, _, _)| start == from)
        {
            self.opaque[slot] = (from, to, frag);
        } else {
            self.opaque.insert(slot, (from, to, frag));
        }
    }
}
