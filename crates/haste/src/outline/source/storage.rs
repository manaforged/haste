use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result, bail};

use crate::substrate::Backing;
use crate::substrate::sysroot::tokens::{TOKEN_WIRE_LEN, decode_token_at, decode_tokens_slice, token_raw_span_at};

use super::Token;

#[allow(dead_code, reason = "keeps backing allocation alive for SourceText::ptr")]
enum TextOwner {
    Owned(Box<str>),
    Mapped(Arc<Backing>),
}

/// Source text stored either as an owned string or a zero-copy UTF-8 slice of a memory-mapped backing.
pub struct SourceText {
    ptr: *const u8,
    len: usize,
    _owner: TextOwner,
}

// SAFETY: `SourceText` holds an immutable UTF-8 slice backed by `Box<str>` or read-only `Arc<Backing>` for its entire lifetime.
unsafe impl Send for SourceText {}
// SAFETY: `SourceText` is immutable after construction and safe to share across threads.
unsafe impl Sync for SourceText {}

impl SourceText {
    /// Wraps an owned `String` as source text.
    pub fn owned(text: String) -> Self {
        let boxed = text.into_boxed_str();
        let (ptr, len) = (boxed.as_ptr(), boxed.len());
        Self { ptr, len, _owner: TextOwner::Owned(boxed) }
    }

    /// Creates zero-copy mapped source text from trusted cache bytes `backing[start..end]` after checking bounds.
    pub fn mapped_trusted(backing: Arc<Backing>, start: usize, end: usize) -> Result<Self> {
        let slice = backing.as_slice().get(start..end).context("mapped source text out of bounds")?;
        let (ptr, len) = (slice.as_ptr(), slice.len());
        Ok(Self { ptr, len, _owner: TextOwner::Mapped(backing) })
    }

    /// Returns the source text as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        // SAFETY: `ptr..ptr + len` is a valid immutable UTF-8 slice owned by `_owner` for the lifetime of `self`.
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.ptr, self.len)) }
    }

    /// Returns the byte length of the source text.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
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
    Lazy {
        _backing: Arc<Backing>,
        ptr: *const u8,
        byte_len: usize,
        count: usize,
        decoded: OnceLock<Vec<Token>>,
    },
}

// SAFETY: `TokensRepr::Lazy::ptr` points into immutable read-only bytes kept alive by `_backing`.
unsafe impl Send for TokensRepr {}
// SAFETY: `TokensRepr` exposes only immutable reads and synchronized `OnceLock` initialization.
unsafe impl Sync for TokensRepr {}

/// Source token stream stored either eagerly or decoded on demand from a memory-mapped backing.
pub struct SourceTokens(TokensRepr);

impl SourceTokens {
    /// Wraps an already-decoded vector of tokens.
    pub fn eager(tokens: Vec<Token>) -> Self {
        Self(TokensRepr::Eager(tokens))
    }

    /// Creates a lazy token stream backed by `backing[start..end]` after validating bounds and wire length.
    pub fn lazy(backing: Arc<Backing>, start: usize, end: usize, count: usize) -> Result<Self> {
        let expected = count.checked_mul(TOKEN_WIRE_LEN).context("token stream overflow")?;
        let slice = backing.as_slice().get(start..end).context("lazy source tokens out of bounds")?;
        if slice.len() != expected {
            bail!("lazy source tokens byte length mismatch");
        }
        let (ptr, byte_len) = (slice.as_ptr(), slice.len());
        Ok(Self(TokensRepr::Lazy { _backing: backing, ptr, byte_len, count, decoded: OnceLock::new() }))
    }

    /// Returns the number of tokens without forcing lazy decoding.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.len(),
            TokensRepr::Lazy { count, .. } => *count,
        }
    }

    /// Returns token `index` in O(1), decoding only that 14-byte record from mmap when lazy.
    #[inline]
    pub fn at(&self, index: usize) -> Option<Token> {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.get(index).copied(),
            TokensRepr::Lazy { ptr, byte_len, count, .. } => {
                if index >= *count {
                    return None;
                }
                // SAFETY: `ptr` is valid for `byte_len == count * TOKEN_WIRE_LEN` immutable bytes kept alive by `_backing`.
                let bytes = unsafe { std::slice::from_raw_parts(*ptr, *byte_len) };
                Some(decode_token_at(bytes, index * TOKEN_WIRE_LEN))
            }
        }
    }

    /// Returns `(start, raw_len)` of token `index` in O(1) without decoding its `TokenKind`.
    #[inline]
    pub fn raw_span(&self, index: usize) -> Option<(u32, u32)> {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.get(index).map(|t| (t.start, t.raw().0)),
            TokensRepr::Lazy { ptr, byte_len, count, .. } => {
                if index >= *count {
                    return None;
                }
                // SAFETY: `ptr` is valid for `byte_len == count * TOKEN_WIRE_LEN` immutable bytes kept alive by `_backing`.
                let bytes = unsafe { std::slice::from_raw_parts(*ptr, *byte_len) };
                Some(token_raw_span_at(bytes, index * TOKEN_WIRE_LEN))
            }
        }
    }

    /// Returns the decoded token slice, decoding on first access if backed lazily.
    #[inline]
    pub fn as_slice(&self) -> &[Token] {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.as_slice(),
            TokensRepr::Lazy { ptr, byte_len, count, decoded, .. } => decoded.get_or_init(|| {
                // SAFETY: `ptr` is valid for `byte_len == count * TOKEN_WIRE_LEN` immutable bytes kept alive by `_backing`.
                let bytes = unsafe { std::slice::from_raw_parts(*ptr, *byte_len) };
                decode_tokens_slice(bytes, *count).expect("valid sysroot token stream")
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
            TokensRepr::Lazy { count, decoded, .. } => {
                if let Some(tokens) = decoded.get() {
                    fmt::Debug::fmt(tokens, f)
                } else {
                    f.debug_struct("LazyTokens").field("count", count).finish()
                }
            }
        }
    }
}
