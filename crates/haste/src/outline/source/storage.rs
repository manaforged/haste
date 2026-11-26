use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use anyhow::{Context as _, Result, bail};

use crate::substrate::Backing;
use crate::substrate::sysroot::tokens::{TOKEN_WIRE_LEN, decode_tokens_slice};

use super::Token;

enum TextRepr {
    Owned(String),
    Mapped {
        backing: Arc<Backing>,
        start: usize,
        end: usize,
    },
}

/// Source text stored either as an owned `String` or a zero-copy UTF-8 slice of a memory-mapped backing.
pub struct SourceText(TextRepr);

impl SourceText {
    /// Wraps an owned `String` as source text.
    pub fn owned(text: String) -> Self {
        Self(TextRepr::Owned(text))
    }

    /// Creates zero-copy mapped source text from `backing[start..end]` after validating UTF-8 and bounds.
    pub fn mapped(backing: Arc<Backing>, start: usize, end: usize) -> Result<Self> {
        let slice = backing.as_slice().get(start..end).context("mapped source text out of bounds")?;
        std::str::from_utf8(slice).context("invalid utf-8 in mapped source text")?;
        Ok(Self(TextRepr::Mapped { backing, start, end }))
    }

    /// Returns the source text as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        match &self.0 {
            TextRepr::Owned(text) => text.as_str(),
            TextRepr::Mapped { backing, start, end } => {
                // SAFETY: `start..end` was bounds-checked and validated as UTF-8 in `SourceText::mapped`.
                unsafe { std::str::from_utf8_unchecked(&backing.as_slice()[*start..*end]) }
            }
        }
    }

    /// Returns the byte length of the source text.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.0 {
            TextRepr::Owned(text) => text.len(),
            TextRepr::Mapped { start, end, .. } => *end - *start,
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
    Lazy {
        backing: Arc<Backing>,
        start: usize,
        end: usize,
        count: usize,
        decoded: OnceLock<Vec<Token>>,
    },
}

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
        Ok(Self(TokensRepr::Lazy { backing, start, end, count, decoded: OnceLock::new() }))
    }

    /// Returns the number of tokens without forcing lazy decoding.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.len(),
            TokensRepr::Lazy { count, .. } => *count,
        }
    }

    /// Returns the decoded token slice, decoding on first access if backed lazily.
    #[inline]
    pub fn as_slice(&self) -> &[Token] {
        match &self.0 {
            TokensRepr::Eager(tokens) => tokens.as_slice(),
            TokensRepr::Lazy { backing, start, end, count, decoded } => decoded.get_or_init(|| {
                decode_tokens_slice(&backing.as_slice()[*start..*end], *count).expect("valid sysroot token stream")
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
