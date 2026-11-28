use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result};

use crate::substrate::Backing;
use crate::substrate::sysroot::tokens::{decode_compact_tokens, decode_marks_body};

use super::{Mark, Token};

/// Lazily reads an unmodified on-disk source file's text (plus optional lexed doc-comment tail) without re-lexing.
pub(crate) struct DiskSource {
    path: PathBuf,
    text_len: usize,
    tail: Box<str>,
    loaded: OnceLock<Box<str>>,
}

impl DiskSource {
    pub(crate) fn new(path: PathBuf, text_len: usize, tail: Box<str>) -> Self {
        Self {
            path,
            text_len,
            tail,
            loaded: OnceLock::new(),
        }
    }

    fn load(&self) -> &str {
        self.loaded.get_or_init(|| {
            let mut text =
                std::fs::read_to_string(&self.path).expect("readable cached sysroot file");
            if !self.tail.is_empty() {
                text.push_str(&self.tail);
            }
            text.into_boxed_str()
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
            TextRepr::Disk(disk) => disk.load(),
        }
    }

    pub fn duplicate(&self) -> Self {
        Self(match &self.0 {
            TextRepr::Owned(boxed) => TextRepr::Owned(boxed.clone()),
            TextRepr::Mapped { _backing, ptr, len } => TextRepr::Mapped {
                _backing: Arc::clone(_backing),
                ptr: *ptr,
                len: *len,
            },
            TextRepr::Disk(disk) => TextRepr::Disk(Arc::clone(disk)),
        })
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
}

// SAFETY: `TokensRepr::Compact::ptr` points into immutable read-only bytes kept alive by `_backing`.
unsafe impl Send for TokensRepr {}
// SAFETY: `TokensRepr` exposes only immutable reads and synchronized `OnceLock` initialization.
unsafe impl Sync for TokensRepr {}

/// Source token stream stored eagerly or decoded lazily from compact cache bytes.
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

    pub fn duplicate(&self) -> Self {
        Self(match &self.0 {
            TokensRepr::Eager(tokens) => TokensRepr::Eager(tokens.clone()),
            TokensRepr::Compact {
                _backing,
                ptr,
                byte_len,
                count,
                ..
            } => TokensRepr::Compact {
                _backing: Arc::clone(_backing),
                ptr: *ptr,
                byte_len: *byte_len,
                count: *count,
                decoded: OnceLock::new(),
            },
        })
    }

    /// Returns the number of tokens without forcing lazy decoding.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.len(),
            TokensRepr::Compact { count, .. } => *count,
        }
    }

    /// Returns token `index` in O(1) once decoded.
    #[inline(always)]
    pub fn at(&self, index: usize) -> Option<Token> {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.get(index).copied(),
            TokensRepr::Compact { .. } => self.as_slice().get(index).copied(),
        }
    }

    /// Returns `(start, raw_len)` of token `index`.
    #[inline]
    pub fn raw_span(&self, index: usize) -> Option<(u32, u32)> {
        self.at(index).map(|t| (t.start, t.raw().0))
    }

    /// Returns the decoded token slice, decoding on first access if backed lazily.
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
        }
    }
}

enum MarksRepr {
    Eager(Vec<(u32, Mark)>),
    Compact {
        _backing: Arc<Backing>,
        ptr: *const u8,
        byte_len: usize,
        count: usize,
        remap: Arc<[u32]>,
        scope_bit: u32,
        decoded: OnceLock<Vec<(u32, Mark)>>,
    },
}

// SAFETY: `MarksRepr::Compact::ptr` points into immutable read-only bytes kept alive by `_backing`.
unsafe impl Send for MarksRepr {}
// SAFETY: `MarksRepr` exposes only immutable reads and synchronized `OnceLock` initialization.
unsafe impl Sync for MarksRepr {}

pub struct SourceMarks(MarksRepr);

impl SourceMarks {
    pub fn eager(marks: Vec<(u32, Mark)>) -> Self {
        Self(MarksRepr::Eager(marks))
    }

    pub fn compact(
        backing: Arc<Backing>,
        start: usize,
        end: usize,
        count: usize,
        remap: Arc<[u32]>,
        scope_bit: u32,
    ) -> Result<Self> {
        let slice = backing
            .as_slice()
            .get(start..end)
            .context("compact source marks out of bounds")?;
        if count == 0 {
            if !slice.is_empty() {
                anyhow::bail!("trailing bytes in empty source marks");
            }
            return Ok(Self::eager(Vec::new()));
        }
        let (ptr, byte_len) = (slice.as_ptr(), slice.len());
        Ok(Self(MarksRepr::Compact {
            _backing: backing,
            ptr,
            byte_len,
            count,
            remap,
            scope_bit,
            decoded: OnceLock::new(),
        }))
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.0 {
            MarksRepr::Eager(marks) => marks.len(),
            MarksRepr::Compact { count, .. } => *count,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[(u32, Mark)] {
        match &self.0 {
            MarksRepr::Eager(marks) => marks.as_slice(),
            MarksRepr::Compact {
                ptr,
                byte_len,
                count,
                remap,
                scope_bit,
                decoded,
                ..
            } => decoded.get_or_init(|| {
                // SAFETY: `ptr` is valid for `byte_len` immutable bytes kept alive by `_backing`.
                let bytes = unsafe { std::slice::from_raw_parts(*ptr, *byte_len) };
                decode_marks_body(*count, bytes, remap, *scope_bit)
                    .expect("valid sysroot compact mark stream")
            }),
        }
    }
}

impl Default for SourceMarks {
    fn default() -> Self {
        Self::eager(Vec::new())
    }
}

impl From<Vec<(u32, Mark)>> for SourceMarks {
    fn from(marks: Vec<(u32, Mark)>) -> Self {
        Self::eager(marks)
    }
}

impl Deref for SourceMarks {
    type Target = [(u32, Mark)];

    #[inline]
    fn deref(&self) -> &[(u32, Mark)] {
        self.as_slice()
    }
}

impl AsRef<[(u32, Mark)]> for SourceMarks {
    #[inline]
    fn as_ref(&self) -> &[(u32, Mark)] {
        self.as_slice()
    }
}

impl PartialEq for SourceMarks {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for SourceMarks {}

impl fmt::Debug for SourceMarks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            MarksRepr::Eager(marks) => fmt::Debug::fmt(marks, f),
            MarksRepr::Compact { count, decoded, .. } => {
                if let Some(marks) = decoded.get() {
                    fmt::Debug::fmt(marks, f)
                } else {
                    f.debug_struct("CompactMarks")
                        .field("count", count)
                        .finish()
                }
            }
        }
    }
}
