//! The lexer behind `FromStr`: Rust source text to token trees, with doc comments turned into `doc` attributes.

use super::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

const OPERATORS: &str = "=<>!~+-*/%^&|@.,;:#$?";

struct Lexer {
    chars: Vec<char>,
    at: usize,
}

/// The token trees that `src` writes, or why it does not lex.
pub(crate) fn stream(src: &str) -> Result<TokenStream, String> {
    let mut lexer = Lexer { chars: src.chars().collect(), at: 0 };
    let mut open: Vec<(Delimiter, Vec<TokenTree>)> = Vec::new();
    let mut trees = Vec::new();
    while let Some(ch) = lexer.trivia(&mut trees) {
        match ch {
            '(' | '[' | '{' => {
                lexer.at += 1;
                open.push((delimiter(ch), std::mem::take(&mut trees)));
            }
            ')' | ']' | '}' => {
                lexer.at += 1;
                let Some((delim, outer)) = open.pop() else { return Err(format!("unexpected `{ch}`")) };
                if delim != delimiter(ch) {
                    return Err(format!("mismatched closing `{ch}`"));
                }
                let inner = std::mem::replace(&mut trees, outer);
                trees.push(TokenTree::Group(Group::new(delim, TokenStream { trees: inner })));
            }
            _ => lexer.token(ch, &mut trees)?,
        }
    }
    if !open.is_empty() {
        return Err("unclosed delimiter".to_owned());
    }
    Ok(TokenStream { trees })
}

fn delimiter(ch: char) -> Delimiter {
    match ch {
        '(' | ')' => Delimiter::Parenthesis,
        '[' | ']' => Delimiter::Bracket,
        _ => Delimiter::Brace,
    }
}

fn ident_start(ch: char) -> bool {
    ch == '_' || ch.is_alphabetic()
}

fn ident_continue(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

fn doc(text: String, inner: bool, trees: &mut Vec<TokenTree>) {
    trees.push(TokenTree::Punct(Punct::new('#', Spacing::Alone)));
    if inner {
        trees.push(TokenTree::Punct(Punct::new('!', Spacing::Alone)));
    }
    let body = vec![TokenTree::Ident(Ident::new("doc", Span::call_site())), TokenTree::Punct(Punct::new('=', Spacing::Alone)), TokenTree::Literal(Literal::string(&text))];
    trees.push(TokenTree::Group(Group::new(Delimiter::Bracket, TokenStream { trees: body })));
}

impl Lexer {
    fn peek(&self, ahead: usize) -> Option<char> {
        self.chars.get(self.at + ahead).copied()
    }

    fn text(&self, from: usize) -> String {
        self.chars[from..self.at].iter().collect()
    }

    fn eat_while(&mut self, keep: impl Fn(char) -> bool) {
        while self.peek(0).is_some_and(&keep) {
            self.at += 1;
        }
    }

    fn trivia(&mut self, trees: &mut Vec<TokenTree>) -> Option<char> {
        loop {
            let ch = self.peek(0)?;
            match (ch, self.peek(1)) {
                (ch, _) if ch.is_whitespace() => self.at += 1,
                ('/', Some('/')) => self.line_comment(trees),
                ('/', Some('*')) => self.block_comment(trees),
                _ => return Some(ch),
            }
        }
    }

    fn line_comment(&mut self, trees: &mut Vec<TokenTree>) {
        let kind = (self.peek(2), self.peek(3));
        self.at += 2;
        let from = self.at;
        self.eat_while(|ch| ch != '\n');
        match kind {
            (Some('/'), next) if next != Some('/') => doc(self.chars[from + 1..self.at].iter().collect(), false, trees),
            (Some('!'), _) => doc(self.chars[from + 1..self.at].iter().collect(), true, trees),
            _ => {}
        }
    }

    fn block_comment(&mut self, trees: &mut Vec<TokenTree>) {
        let kind = (self.peek(2), self.peek(3));
        self.at += 2;
        let from = self.at;
        let mut depth = 1;
        while depth > 0 && self.at < self.chars.len() {
            match (self.chars[self.at], self.peek(1)) {
                ('/', Some('*')) => (depth, self.at) = (depth + 1, self.at + 2),
                ('*', Some('/')) => (depth, self.at) = (depth - 1, self.at + 2),
                _ => self.at += 1,
            }
        }
        let body = || self.chars[from + 1..self.at.saturating_sub(2).max(from + 1)].iter().collect();
        match kind {
            (Some('*'), next) if next != Some('*') && next != Some('/') => doc(body(), false, trees),
            (Some('!'), _) => doc(body(), true, trees),
            _ => {}
        }
    }

    fn token(&mut self, ch: char, trees: &mut Vec<TokenTree>) -> Result<(), String> {
        let from = self.at;
        if let Some(literal) = self.prefixed(ch)? {
            trees.push(TokenTree::Literal(literal));
            return Ok(());
        }
        match ch {
            ch if ident_start(ch) => trees.push(TokenTree::Ident(self.ident())),
            '0'..='9' => trees.push(TokenTree::Literal(self.number())),
            '"' => trees.push(TokenTree::Literal(self.quoted(from, '"')?)),
            '\'' => self.quote(from, trees)?,
            ch if OPERATORS.contains(ch) => {
                self.at += 1;
                let spacing = if self.peek(0).is_some_and(|next| OPERATORS.contains(next)) { Spacing::Joint } else { Spacing::Alone };
                trees.push(TokenTree::Punct(Punct::new(ch, spacing)));
            }
            _ => return Err(format!("unknown start of token `{ch}`")),
        }
        Ok(())
    }

    fn ident(&mut self) -> Ident {
        let raw = self.peek(0) == Some('r') && self.peek(1) == Some('#') && self.peek(2).is_some_and(ident_start);
        if raw {
            self.at += 2;
        }
        let from = self.at;
        self.eat_while(ident_continue);
        let name = self.text(from);
        if raw { Ident::new_raw(&name, Span::call_site()) } else { Ident::new(&name, Span::call_site()) }
    }

    fn prefixed(&mut self, ch: char) -> Result<Option<Literal>, String> {
        let from = self.at;
        let skip = match (ch, self.peek(1), self.peek(2)) {
            ('b' | 'c', Some('r'), Some('"' | '#')) => 2,
            ('b', Some('\'' | '"'), _) | ('c', Some('"'), _) | ('r', Some('"'), _) => 1,
            ('r', Some('#'), Some('"' | '#')) => 1,
            _ => return Ok(None),
        };
        self.at += skip;
        let literal = match self.peek(0) {
            Some('#' | '"') if self.chars[self.at - 1] == 'r' => self.raw(from)?,
            Some('\'') => self.quoted(from, '\'')?,
            _ => self.quoted(from, '"')?,
        };
        Ok(Some(literal))
    }

    fn quoted(&mut self, from: usize, close: char) -> Result<Literal, String> {
        self.at += 1;
        loop {
            match self.peek(0) {
                None => return Err("unterminated literal".to_owned()),
                Some('\\') => self.at += 2,
                Some(ch) if ch == close => break,
                Some(_) => self.at += 1,
            }
        }
        self.at += 1;
        Ok(self.suffixed(from))
    }

    fn raw(&mut self, from: usize) -> Result<Literal, String> {
        let hashes = self.chars[self.at..].iter().take_while(|ch| **ch == '#').count();
        self.at += hashes;
        if self.peek(0) != Some('"') {
            return Err("malformed raw string".to_owned());
        }
        self.at += 1;
        loop {
            match self.peek(0) {
                None => return Err("unterminated raw string".to_owned()),
                Some('"') if (1..=hashes).all(|ahead| self.peek(ahead) == Some('#')) => break,
                Some(_) => self.at += 1,
            }
        }
        self.at += 1 + hashes;
        Ok(self.suffixed(from))
    }

    fn quote(&mut self, from: usize, trees: &mut Vec<TokenTree>) -> Result<(), String> {
        let lifetime = self.peek(1).is_some_and(ident_start) && self.peek(2) != Some('\'');
        if !lifetime {
            trees.push(TokenTree::Literal(self.quoted(from, '\'')?));
            return Ok(());
        }
        self.at += 1;
        trees.push(TokenTree::Punct(Punct::new('\'', Spacing::Joint)));
        trees.push(TokenTree::Ident(self.ident()));
        Ok(())
    }

    fn number(&mut self) -> Literal {
        let from = self.at;
        let radix = matches!((self.peek(0), self.peek(1)), (Some('0'), Some('x' | 'o' | 'b')));
        if radix {
            self.at += 2;
            self.eat_while(|ch| ch.is_ascii_hexdigit() || ch == '_');
            return self.suffixed(from);
        }
        self.eat_while(|ch| ch.is_ascii_digit() || ch == '_');
        self.decimal_tail();
        self.suffixed(from)
    }

    fn decimal_tail(&mut self) {
        let fraction = self.peek(0) == Some('.') && self.peek(1).is_none_or(|next| next != '.' && !ident_start(next));
        if fraction {
            self.at += 1;
            self.eat_while(|ch| ch.is_ascii_digit() || ch == '_');
        }
        let signed = matches!(self.peek(1), Some('+' | '-')) && self.peek(2).is_some_and(|ch| ch.is_ascii_digit());
        let exponent = matches!(self.peek(0), Some('e' | 'E')) && (signed || self.peek(1).is_some_and(|ch| ch.is_ascii_digit()));
        if exponent {
            self.at += if signed { 2 } else { 1 };
            self.eat_while(|ch| ch.is_ascii_digit() || ch == '_');
        }
    }

    fn suffixed(&mut self, from: usize) -> Literal {
        self.eat_while(ident_continue);
        Literal::of(self.text(from))
    }
}
