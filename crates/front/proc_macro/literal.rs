//! Literal tokens, kept as the source text that writes them.

use std::fmt;
use std::str::FromStr;

use super::{LexError, Span, TokenTree, lex};

/// A literal: a number, a character, a byte or a string of either, with its suffix.
#[derive(Clone)]
pub struct Literal {
    repr: String,
    span: Span,
}

macro_rules! suffixed {
    ($($name:ident $ty:ident)*) => {$(
        /// The literal of `n` with its type as a suffix.
        pub fn $name(n: $ty) -> Literal {
            Literal::of(format!("{n}{}", stringify!($ty)))
        }
    )*};
}

macro_rules! unsuffixed {
    ($($name:ident $ty:ident)*) => {$(
        /// The literal of `n` with no suffix.
        pub fn $name(n: $ty) -> Literal {
            Literal::of(n.to_string())
        }
    )*};
}

fn float(text: String) -> String {
    if text.contains(['.', 'e', 'E']) || text.contains("inf") || text.contains("NaN") { text } else { text + ".0" }
}

fn escaped(bytes: &[u8]) -> String {
    let mut body = String::new();
    for &byte in bytes {
        match byte {
            b'\'' => body.push('\''),
            _ => body.extend(std::ascii::escape_default(byte).map(char::from)),
        }
    }
    body
}

fn finite(n: f64) {
    assert!(n.is_finite(), "invalid float literal {n}");
}

impl Literal {
    pub(crate) fn of(repr: String) -> Literal {
        Literal { repr, span: Span::call_site() }
    }

    suffixed!(u8_suffixed u8 u16_suffixed u16 u32_suffixed u32 u64_suffixed u64 u128_suffixed u128 usize_suffixed usize i8_suffixed i8 i16_suffixed i16 i32_suffixed i32 i64_suffixed i64 i128_suffixed i128 isize_suffixed isize);
    unsuffixed!(u8_unsuffixed u8 u16_unsuffixed u16 u32_unsuffixed u32 u64_unsuffixed u64 u128_unsuffixed u128 usize_unsuffixed usize i8_unsuffixed i8 i16_unsuffixed i16 i32_unsuffixed i32 i64_unsuffixed i64 i128_unsuffixed i128 isize_unsuffixed isize);

    /// The literal of `n` with no suffix.
    pub fn f32_unsuffixed(n: f32) -> Literal {
        finite(f64::from(n));
        Literal::of(float(n.to_string()))
    }

    /// The literal of `n` with the suffix `f32`.
    pub fn f32_suffixed(n: f32) -> Literal {
        finite(f64::from(n));
        Literal::of(format!("{n}f32"))
    }

    /// The literal of `n` with no suffix.
    pub fn f64_unsuffixed(n: f64) -> Literal {
        finite(n);
        Literal::of(float(n.to_string()))
    }

    /// The literal of `n` with the suffix `f64`.
    pub fn f64_suffixed(n: f64) -> Literal {
        finite(n);
        Literal::of(format!("{n}f64"))
    }

    /// The string literal of `text`.
    pub fn string(text: &str) -> Literal {
        let mut repr = String::from("\"");
        for ch in text.chars() {
            match ch {
                '\'' => repr.push(ch),
                _ => repr.extend(ch.escape_debug()),
            }
        }
        repr.push('"');
        Literal::of(repr)
    }

    /// The character literal of `ch`.
    pub fn character(ch: char) -> Literal {
        let body: String = if ch == '"' { "\"".to_owned() } else { ch.escape_debug().collect() };
        Literal::of(format!("'{body}'"))
    }

    /// The byte character literal of `byte`.
    pub fn byte_character(byte: u8) -> Literal {
        let body: String = if byte == b'"' { "\"".to_owned() } else { std::ascii::escape_default(byte).map(char::from).collect() };
        Literal::of(format!("b'{body}'"))
    }

    /// The byte string literal of `bytes`.
    pub fn byte_string(bytes: &[u8]) -> Literal {
        Literal::of(format!("b\"{}\"", escaped(bytes)))
    }

    /// The C string literal of `text`.
    pub fn c_string(text: &std::ffi::CStr) -> Literal {
        Literal::of(format!("c\"{}\"", escaped(text.to_bytes())))
    }

    /// The span of the literal.
    pub fn span(&self) -> Span {
        self.span
    }

    /// Sets the span of the literal.
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }

    /// The span of the bytes `range` of the literal, which the image does not track.
    pub fn subspan<R: std::ops::RangeBounds<usize>>(&self, _range: R) -> Option<Span> {
        None
    }
}

impl FromStr for Literal {
    type Err = LexError;

    fn from_str(src: &str) -> Result<Literal, LexError> {
        let fail = || LexError { message: format!("not a literal: {src}") };
        let stream = lex::stream(src).map_err(|message| LexError { message })?;
        let mut trees = stream.into_iter();
        let (first, second, third) = (trees.next(), trees.next(), trees.next());
        match (first, second, third) {
            (Some(TokenTree::Literal(literal)), None, None) => Ok(literal),
            (Some(TokenTree::Punct(minus)), Some(TokenTree::Literal(literal)), None) if minus.as_char() == '-' && literal.repr.starts_with(|ch: char| ch.is_ascii_digit()) => Ok(Literal::of(format!("-{}", literal.repr))),
            _ => Err(fail()),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.repr)
    }
}

impl fmt::Debug for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Literal").field("lit", &self.repr).field("span", &self.span).finish()
    }
}
