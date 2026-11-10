//! The `proc_macro` crate that haste builds into its image: token trees held as plain data, text in and text out at the two entry points the host calls.

mod lex;
mod literal;

use std::fmt;
use std::str::FromStr;

pub use literal::Literal;

/// Whether the running code is a procedural macro, which is always so in the image.
pub fn is_available() -> bool {
    true
}

/// A sequence of token trees.
#[derive(Clone, Default)]
pub struct TokenStream {
    trees: Vec<TokenTree>,
}

/// One token or a delimited group of tokens.
#[derive(Clone)]
pub enum TokenTree {
    Group(Group),
    Ident(Ident),
    Punct(Punct),
    Literal(Literal),
}

/// A token stream between a pair of delimiters.
#[derive(Clone)]
pub struct Group {
    delimiter: Delimiter,
    stream: TokenStream,
    span: Span,
}

/// The delimiters of a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Delimiter {
    Parenthesis,
    Brace,
    Bracket,
    None,
}

/// A single punctuation character.
#[derive(Clone)]
pub struct Punct {
    ch: char,
    spacing: Spacing,
    span: Span,
}

/// Whether a punctuation character joins the next one into one operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Spacing {
    Joint,
    Alone,
}

/// An identifier or keyword.
#[derive(Clone)]
pub struct Ident {
    name: String,
    raw: bool,
    span: Span,
}

/// A region of source code; every span in the image is the call site.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    at: u32,
}

/// The error from parsing text that is not a valid token stream.
#[derive(Debug)]
pub struct LexError {
    message: String,
}

/// The iterator over a token stream's trees.
pub mod token_stream {
    /// An owning iterator over the token trees of a stream.
    #[derive(Clone)]
    pub struct IntoIter {
        pub(crate) trees: std::vec::IntoIter<super::TokenTree>,
    }

    impl Iterator for IntoIter {
        type Item = super::TokenTree;

        fn next(&mut self) -> Option<super::TokenTree> {
            self.trees.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.trees.size_hint()
        }
    }
}

impl TokenStream {
    /// An empty stream.
    pub fn new() -> TokenStream {
        TokenStream { trees: Vec::new() }
    }

    /// Whether the stream has no trees.
    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }
}

impl FromStr for TokenStream {
    type Err = LexError;

    fn from_str(src: &str) -> Result<TokenStream, LexError> {
        lex::stream(src).map_err(|message| LexError { message })
    }
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut joint = true;
        for tree in &self.trees {
            if !joint {
                f.write_str(" ")?;
            }
            joint = matches!(tree, TokenTree::Punct(punct) if punct.spacing == Spacing::Joint);
            fmt::Display::fmt(tree, f)?;
        }
        Ok(())
    }
}

impl fmt::Debug for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TokenStream ")?;
        f.debug_list().entries(&self.trees).finish()
    }
}

impl From<TokenTree> for TokenStream {
    fn from(tree: TokenTree) -> TokenStream {
        TokenStream { trees: vec![tree] }
    }
}

impl FromIterator<TokenTree> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenTree>>(trees: I) -> TokenStream {
        TokenStream { trees: trees.into_iter().collect() }
    }
}

impl FromIterator<TokenStream> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> TokenStream {
        TokenStream { trees: streams.into_iter().flat_map(|stream| stream.trees).collect() }
    }
}

impl Extend<TokenTree> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenTree>>(&mut self, trees: I) {
        self.trees.extend(trees);
    }
}

impl Extend<TokenStream> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, streams: I) {
        self.trees.extend(streams.into_iter().flat_map(|stream| stream.trees));
    }
}

impl IntoIterator for TokenStream {
    type Item = TokenTree;
    type IntoIter = token_stream::IntoIter;

    fn into_iter(self) -> token_stream::IntoIter {
        token_stream::IntoIter { trees: self.trees.into_iter() }
    }
}

impl TokenTree {
    /// The span of the tree.
    pub fn span(&self) -> Span {
        match self {
            TokenTree::Group(group) => group.span,
            TokenTree::Ident(ident) => ident.span,
            TokenTree::Punct(punct) => punct.span,
            TokenTree::Literal(literal) => literal.span(),
        }
    }

    /// Sets the span of the tree.
    pub fn set_span(&mut self, span: Span) {
        match self {
            TokenTree::Group(group) => group.span = span,
            TokenTree::Ident(ident) => ident.span = span,
            TokenTree::Punct(punct) => punct.span = span,
            TokenTree::Literal(literal) => literal.set_span(span),
        }
    }
}

impl fmt::Display for TokenTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenTree::Group(group) => fmt::Display::fmt(group, f),
            TokenTree::Ident(ident) => fmt::Display::fmt(ident, f),
            TokenTree::Punct(punct) => fmt::Display::fmt(punct, f),
            TokenTree::Literal(literal) => fmt::Display::fmt(literal, f),
        }
    }
}

impl fmt::Debug for TokenTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenTree::Group(group) => fmt::Debug::fmt(group, f),
            TokenTree::Ident(ident) => fmt::Debug::fmt(ident, f),
            TokenTree::Punct(punct) => fmt::Debug::fmt(punct, f),
            TokenTree::Literal(literal) => fmt::Debug::fmt(literal, f),
        }
    }
}

impl From<Group> for TokenTree {
    fn from(group: Group) -> TokenTree {
        TokenTree::Group(group)
    }
}

impl From<Ident> for TokenTree {
    fn from(ident: Ident) -> TokenTree {
        TokenTree::Ident(ident)
    }
}

impl From<Punct> for TokenTree {
    fn from(punct: Punct) -> TokenTree {
        TokenTree::Punct(punct)
    }
}

impl From<Literal> for TokenTree {
    fn from(literal: Literal) -> TokenTree {
        TokenTree::Literal(literal)
    }
}

impl Group {
    /// A group of `stream` between `delimiter`.
    pub fn new(delimiter: Delimiter, stream: TokenStream) -> Group {
        Group { delimiter, stream, span: Span::call_site() }
    }

    /// The delimiter of the group.
    pub fn delimiter(&self) -> Delimiter {
        self.delimiter
    }

    /// The tokens between the delimiters.
    pub fn stream(&self) -> TokenStream {
        self.stream.clone()
    }

    /// The span of the whole group.
    pub fn span(&self) -> Span {
        self.span
    }

    /// The span of the opening delimiter.
    pub fn span_open(&self) -> Span {
        self.span
    }

    /// The span of the closing delimiter.
    pub fn span_close(&self) -> Span {
        self.span
    }

    /// Sets the span of the group.
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (open, close) = match self.delimiter {
            Delimiter::Parenthesis => ("(", ")"),
            Delimiter::Brace => ("{ ", " }"),
            Delimiter::Bracket => ("[", "]"),
            Delimiter::None => ("", ""),
        };
        f.write_str(open)?;
        fmt::Display::fmt(&self.stream, f)?;
        f.write_str(close)
    }
}

impl fmt::Debug for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Group").field("delimiter", &self.delimiter).field("stream", &self.stream).field("span", &self.span).finish()
    }
}

impl Punct {
    /// The punctuation character `ch` with `spacing`.
    pub fn new(ch: char, spacing: Spacing) -> Punct {
        Punct { ch, spacing, span: Span::call_site() }
    }

    /// The character.
    pub fn as_char(&self) -> char {
        self.ch
    }

    /// Whether the character joins the next one.
    pub fn spacing(&self) -> Spacing {
        self.spacing
    }

    /// The span of the character.
    pub fn span(&self) -> Span {
        self.span
    }

    /// Sets the span of the character.
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

impl PartialEq<char> for Punct {
    fn eq(&self, other: &char) -> bool {
        self.ch == *other
    }
}

impl PartialEq<Punct> for char {
    fn eq(&self, other: &Punct) -> bool {
        *self == other.ch
    }
}

impl fmt::Display for Punct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.ch, f)
    }
}

impl fmt::Debug for Punct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Punct").field("ch", &self.ch).field("spacing", &self.spacing).field("span", &self.span).finish()
    }
}

impl Ident {
    /// The identifier `name` at `span`.
    pub fn new(name: &str, span: Span) -> Ident {
        match name.strip_prefix("r#") {
            Some(rest) => Ident { name: rest.to_owned(), raw: true, span },
            None => Ident { name: name.to_owned(), raw: false, span },
        }
    }

    /// The raw identifier `r#name` at `span`.
    pub fn new_raw(name: &str, span: Span) -> Ident {
        Ident { name: name.to_owned(), raw: true, span }
    }

    /// The span of the identifier.
    pub fn span(&self) -> Span {
        self.span
    }

    /// Sets the span of the identifier.
    pub fn set_span(&mut self, span: Span) {
        self.span = span;
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.raw {
            f.write_str("r#")?;
        }
        f.write_str(&self.name)
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ident").field("ident", &self.to_string()).field("span", &self.span).finish()
    }
}

impl Span {
    /// The span of the macro call.
    pub fn call_site() -> Span {
        Span { at: 0 }
    }

    /// The span that resolves locals at the macro definition and other names at the call.
    pub fn mixed_site() -> Span {
        Span { at: 0 }
    }

    /// The span of the macro definition.
    pub fn def_site() -> Span {
        Span { at: 0 }
    }

    /// This span with the name resolution of `other`.
    pub fn resolved_at(&self, other: Span) -> Span {
        other
    }

    /// `other` with the name resolution of this span.
    pub fn located_at(&self, _other: Span) -> Span {
        *self
    }

    /// The span that covers this span and `other`.
    pub fn join(&self, _other: Span) -> Option<Span> {
        Some(*self)
    }

    /// The source text of the span, which the image does not keep.
    pub fn source_text(&self) -> Option<String> {
        None
    }

    /// The empty span at the start of this span.
    pub fn start(&self) -> Span {
        *self
    }

    /// The empty span at the end of this span.
    pub fn end(&self) -> Span {
        *self
    }

    /// The one-based line of the span's start.
    pub fn line(&self) -> usize {
        1
    }

    /// The one-based column of the span's start.
    pub fn column(&self) -> usize {
        1
    }

    /// The path of the span's file for display.
    pub fn file(&self) -> String {
        String::new()
    }

    /// The path of the span's file on disk.
    pub fn local_file(&self) -> Option<std::path::PathBuf> {
        None
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{} bytes(0..0)", self.at)
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LexError {}

fn parsed(text: &[u8]) -> TokenStream {
    let text = std::str::from_utf8(text).unwrap_or_else(|error| panic!("macro input is not UTF-8: {error}"));
    TokenStream::from_str(text).unwrap_or_else(|error| panic!("macro input does not lex: {error}"))
}

fn hand_back(stream: TokenStream, out: *mut usize) {
    let text: &'static mut [u8] = stream.to_string().into_bytes().leak();
    // SAFETY: the host passes a pointer to two writable words that it reads after the call.
    unsafe {
        *out = text.as_ptr() as usize;
        *out.add(1) = text.len();
    }
}

/// Runs the derive or function-like macro `run` on the text at `input` and writes the address and length of its output text to `out`.
#[doc(hidden)]
pub fn __haste_expand1(run: fn(TokenStream) -> TokenStream, input: *const u8, len: usize, out: *mut usize) {
    // SAFETY: the host passes `len` readable bytes at `input`.
    let input = parsed(unsafe { std::slice::from_raw_parts(input, len) });
    hand_back(run(input), out);
}

/// Runs the attribute macro `run` on the attribute text and the item text, and writes the address and length of its output text to `out`.
#[doc(hidden)]
pub fn __haste_expand2(run: fn(TokenStream, TokenStream) -> TokenStream, attr: *const u8, attr_len: usize, item: *const u8, item_len: usize, out: *mut usize) {
    // SAFETY: the host passes the given number of readable bytes at each address.
    let (attr, item) = unsafe { (parsed(std::slice::from_raw_parts(attr, attr_len)), parsed(std::slice::from_raw_parts(item, item_len))) };
    hand_back(run(attr, item), out);
}
