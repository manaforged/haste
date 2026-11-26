use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result};

use crate::substrate::Backing;
use crate::substrate::sysroot::tokens::decode_compact_tokens;

use super::Token;
use super::lex::{lex, pair_up};

/// Lazily reads and lexes an unmodified on-disk source file once, sharing both text and tokens.
pub(crate) struct DiskSource {
    path: PathBuf,
    text_len: usize,
    tok_count: usize,
    loaded: OnceLock<(Box<str>, Vec<Token>)>,
}

impl DiskSource {
    pub(crate) fn new(path: PathBuf, text_len: usize, tok_count: usize) -> Self {
        Self {
            path,
            text_len,
            tok_count,
            loaded: OnceLock::new(),
        }
    }

    fn load(&self) -> &(Box<str>, Vec<Token>) {
        self.loaded.get_or_init(|| {
            let mut text =
                std::fs::read_to_string(&self.path).expect("readable cached sysroot file");
            let mut tokens = lex(&mut text);
            pair_up(&mut tokens, &self.path).expect("balanced cached sysroot file");
            (text.into_boxed_str(), tokens)
        })
    }
}

enum TextRepr {
    Owned(Box<str>),
    Mapped {
        _backing: Arc<Backing>,
        ptr: *const u8,
        len: usize,
    },
    Disk(Arc<DiskSource>),
}

// SAFETY: `TextRepr::Mapped` holds an immutable UTF-8 slice kept alive by `_backing`; `Disk` uses synchronized `OnceLock`.
unsafe impl Send for TextRepr {}
// SAFETY: `TextRepr` exposes only immutable reads and synchronized `OnceLock` initialization.
unsafe impl Sync for TextRepr {}

/// Source text stored as an owned string, a zero-copy mmap slice, or a lazily read on-disk file.
pub struct SourceText(TextRepr);

impl SourceText {
    /// Wraps an owned `String` as source text.
    pub fn owned(text: String) -> Self {
        Self(TextRepr::Owned(text.into_boxed_str()))
    }

    /// Creates zero-copy mapped source text from trusted cache bytes `backing[start..end]` after checking bounds.
    pub fn mapped_trusted(backing: Arc<Backing>, start: usize, end: usize) -> Result<Self> {
        let slice = backing
            .as_slice()
            .get(start..end)
            .context("mapped source text out of bounds")?;
        let (ptr, len) = (slice.as_ptr(), slice.len());
        Ok(Self(TextRepr::Mapped {
            _backing: backing,
            ptr,
            len,
        }))
    }

    /// Creates source text backed lazily by an unmodified on-disk file.
    pub(crate) fn disk(source: Arc<DiskSource>) -> Self {
        Self(TextRepr::Disk(source))
    }

    /// Returns the source text as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        match &self.0 {
            TextRepr::Owned(boxed) => boxed,
            TextRepr::Mapped { ptr, len, .. } => {
                // SAFETY: `ptr..ptr + len` is a valid immutable UTF-8 slice kept alive by `_backing`.
                unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(*ptr, *len)) }
            }
            TextRepr::Disk(disk) => &disk.load().0,
        }
    }

    /// Returns the byte length of the source text without forcing an on-disk read.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.0 {
            TextRepr::Owned(boxed) => boxed.len(),
            TextRepr::Mapped { len, .. } => *len,
            TextRepr::Disk(disk) => disk.text_len,
        }
    }
}

impl Default for SourceText {
    fn default() -> Self {
        Self::owned(String::new())
    }
}

impl From<String> for SourceText {
    fn from(text: String) -> Self {
        Self::owned(text)
    }
}

impl Deref for SourceText {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for SourceText {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for SourceText {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for SourceText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl fmt::Debug for SourceText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl PartialEq for SourceText {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for SourceText {}

impl PartialEq<str> for SourceText {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for SourceText {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for SourceText {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Hash for SourceText {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

enum TokensRepr {
    Eager(Vec<Token>),
    Compact {
        _backing: Arc<Backing>,
        ptr: *const u8,
        byte_len: usize,
        count: usize,
        decoded: OnceLock<Vec<Token>>,
    },
    Disk(Arc<DiskSource>),
}

// SAFETY: `TokensRepr::Compact::ptr` points into immutable read-only bytes kept alive by `_backing`.
unsafe impl Send for TokensRepr {}
// SAFETY: `TokensRepr` exposes only immutable reads and synchronized `OnceLock` initialization.
unsafe impl Sync for TokensRepr {}

/// Source token stream stored eagerly, decoded lazily from compact cache bytes, or lexed on demand from disk.
pub struct SourceTokens(TokensRepr);

impl SourceTokens {
    /// Wraps an already-decoded vector of tokens.
    pub fn eager(tokens: Vec<Token>) -> Self {
        Self(TokensRepr::Eager(tokens))
    }

    /// Creates a lazy token stream backed by compact wire bytes `backing[start..end]`.
    pub fn compact(backing: Arc<Backing>, start: usize, end: usize, count: usize) -> Result<Self> {
        let slice = backing
            .as_slice()
            .get(start..end)
            .context("compact source tokens out of bounds")?;
        let (ptr, byte_len) = (slice.as_ptr(), slice.len());
        Ok(Self(TokensRepr::Compact {
            _backing: backing,
            ptr,
            byte_len,
            count,
            decoded: OnceLock::new(),
        }))
    }

    /// Creates a lazy token stream backed by an unmodified on-disk source file.
    pub(crate) fn disk(source: Arc<DiskSource>) -> Self {
        Self(TokensRepr::Disk(source))
    }

    /// Returns the number of tokens without forcing lazy decoding or disk reads.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.len(),
            TokensRepr::Compact { count, .. } => *count,
            TokensRepr::Disk(disk) => disk.tok_count,
        }
    }

    /// Returns token `index` in O(1) once decoded.
    #[inline(always)]
    pub fn at(&self, index: usize) -> Option<Token> {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.get(index).copied(),
            TokensRepr::Compact { .. } | TokensRepr::Disk(_) => self.as_slice().get(index).copied(),
        }
    }

    /// Returns `(start, raw_len)` of token `index`.
    #[inline]
    pub fn raw_span(&self, index: usize) -> Option<(u32, u32)> {
        self.at(index).map(|t| (t.start, t.raw().0))
    }

    /// Returns the decoded token slice, decoding or reading on first access if backed lazily.
    #[inline]
    pub fn as_slice(&self) -> &[Token] {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.as_slice(),
            TokensRepr::Compact {
                ptr,
                byte_len,
                count,
                decoded,
                ..
            } => decoded.get_or_init(|| {
                // SAFETY: `ptr` is valid for `byte_len` immutable bytes kept alive by `_backing`.
                let bytes = unsafe { std::slice::from_raw_parts(*ptr, *byte_len) };
                decode_compact_tokens(bytes, *count).expect("valid sysroot compact token stream")
            }),
            TokensRepr::Disk(disk) => &disk.load().1,
        }
    }
}

impl Default for SourceTokens {
    fn default() -> Self {
        Self::eager(Vec::new())
    }
}

impl From<Vec<Token>> for SourceTokens {
    fn from(tokens: Vec<Token>) -> Self {
        Self::eager(tokens)
    }
}

impl Deref for SourceTokens {
    type Target = [Token];

    #[inline]
    fn deref(&self) -> &[Token] {
        self.as_slice()
    }
}

impl AsRef<[Token]> for SourceTokens {
    #[inline]
    fn as_ref(&self) -> &[Token] {
        self.as_slice()
    }
}

impl fmt::Debug for SourceTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            TokensRepr::Eager(tokens) => fmt::Debug::fmt(tokens, f),
            TokensRepr::Compact { count, decoded, .. } => {
                if let Some(tokens) = decoded.get() {
                    fmt::Debug::fmt(tokens, f)
                } else {
                    f.debug_struct("CompactTokens")
                        .field("count", count)
                        .finish()
                }
            }
            TokensRepr::Disk(disk) => {
                if let Some((_, tokens)) = disk.loaded.get() {
                    fmt::Debug::fmt(tokens, f)
                } else {
                    f.debug_struct("DiskTokens")
                        .field("count", &disk.tok_count)
                        .finish()
                }
            }
        }
    }
}
