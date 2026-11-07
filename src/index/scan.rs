use rustc_lexer::{FrontmatterAllowed, TokenKind, tokenize};

use super::{Alias, Entry, Header, Kind, MacroFacts, MacroUse, Scanned};

mod header;

#[derive(Clone, Copy)]
struct Token {
    kind: TokenKind,
    start: u32,
    len: u32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Word {
    As,
    Async,
    Auto,
    Const,
    Default,
    Dyn,
    Enum,
    Extern,
    Fn,
    For,
    Gen,
    Impl,
    Macro,
    MacroRules,
    Mod,
    Pub,
    Safe,
    Static,
    Struct,
    Trait,
    Type,
    Union,
    Unsafe,
    Use,
    Where,
    Other,
}

fn word_of(text: &str) -> Word {
    match text {
        "async" => Word::Async,
        "auto" => Word::Auto,
        "const" => Word::Const,
        "default" => Word::Default,
        "enum" => Word::Enum,
        "extern" => Word::Extern,
        "fn" => Word::Fn,
        "gen" => Word::Gen,
        "as" => Word::As,
        "dyn" => Word::Dyn,
        "for" => Word::For,
        "where" => Word::Where,
        "impl" => Word::Impl,
        "macro" => Word::Macro,
        "macro_rules" => Word::MacroRules,
        "mod" => Word::Mod,
        "pub" => Word::Pub,
        "safe" => Word::Safe,
        "static" => Word::Static,
        "struct" => Word::Struct,
        "trait" => Word::Trait,
        "type" => Word::Type,
        "union" => Word::Union,
        "unsafe" => Word::Unsafe,
        "use" => Word::Use,
        _ => Word::Other,
    }
}

fn is_trivia(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Whitespace | TokenKind::LineComment { .. } | TokenKind::BlockComment { .. })
}

fn lex(text: &str) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(text.len() / 6);
    let mut start = 0;
    for token in tokenize(text, FrontmatterAllowed::No) {
        if !is_trivia(token.kind) {
            tokens.push(Token { kind: token.kind, start, len: token.len });
        }
        start += token.len;
    }
    tokens
}

fn closes(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::CloseParen | TokenKind::CloseBrace | TokenKind::CloseBracket)
}

fn opens(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::OpenParen | TokenKind::OpenBrace | TokenKind::OpenBracket)
}

fn is_name(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Ident | TokenKind::RawIdent)
}

struct Scanner<'a> {
    text: &'a str,
    tokens: Vec<Token>,
    at: usize,
    path: String,
    found: Vec<Entry>,
    headers: Vec<Header>,
    aliases: Vec<Alias>,
    macros: Vec<MacroFacts>,
    uses: Vec<MacroUse>,
}

pub fn scan_all(text: &str) -> Scanned {
    let mut scanner = Scanner {
        text,
        tokens: lex(text),
        at: 0,
        path: String::new(),
        found: Vec::new(),
        headers: Vec::new(),
        aliases: Vec::new(),
        macros: Vec::new(),
        uses: Vec::new(),
    };
    scanner.items();
    Scanned {
        entries: scanner.found,
        headers: scanner.headers,
        aliases: scanner.aliases,
        macros: scanner.macros,
        uses: scanner.uses,
    }
}

pub fn scan(text: &str) -> Vec<Entry> {
    scan_all(text).entries
}

impl Scanner<'_> {
    fn kind(&self, ahead: usize) -> TokenKind {
        self.tokens.get(self.at + ahead).map_or(TokenKind::Eof, |token| token.kind)
    }

    fn word(&self, ahead: usize) -> Word {
        if self.kind(ahead) == TokenKind::Ident { word_of(self.lexeme(ahead)) } else { Word::Other }
    }

    fn lexeme(&self, ahead: usize) -> &str {
        self.tokens.get(self.at + ahead).map_or("", |token| {
            &self.text[token.start as usize..(token.start + token.len) as usize]
        })
    }

    fn name(&self, ahead: usize) -> String {
        let text = self.lexeme(ahead);
        text.strip_prefix("r#").unwrap_or(text).to_owned()
    }

    fn skip_group(&mut self) {
        let mut depth = 0_usize;
        while self.at < self.tokens.len() {
            let kind = self.kind(0);
            self.at += 1;
            if opens(kind) {
                depth += 1;
            } else if closes(kind) {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return;
                }
            }
        }
    }

    fn skip_attributes(&mut self) {
        while self.kind(0) == TokenKind::Pound {
            let inner = usize::from(self.kind(1) == TokenKind::Bang);
            if self.kind(1 + inner) != TokenKind::OpenBracket {
                return;
            }
            self.at += 1 + inner;
            self.skip_group();
        }
    }

    fn skip_visibility(&mut self) {
        if self.word(0) == Word::Pub {
            self.at += 1;
            if self.kind(0) == TokenKind::OpenParen {
                self.skip_group();
            }
        }
    }

    fn skip_restriction(&mut self) {
        if self.word(0) != Word::Impl || self.kind(1) != TokenKind::OpenParen {
            return;
        }
        let before = self.at;
        self.at += 1;
        self.skip_group();
        if !self.modifier() && self.word(0) != Word::Trait {
            self.at = before;
        }
    }

    fn modifier(&self) -> bool {
        match self.word(0) {
            Word::Async | Word::Auto | Word::Default | Word::Gen | Word::Safe | Word::Unsafe => {
                !matches!(self.kind(1), TokenKind::Bang | TokenKind::Colon)
            }
            Word::Const => {
                matches!(self.word(1), Word::Fn | Word::Unsafe | Word::Extern | Word::Async | Word::Safe | Word::Gen | Word::Impl | Word::Trait)
            }
            Word::Extern => self.extern_function(),
            _ => false,
        }
    }

    fn extern_function(&self) -> bool {
        let next = usize::from(matches!(self.kind(1), TokenKind::Literal { .. })) + 1;
        matches!(self.word(next), Word::Fn | Word::Unsafe | Word::Safe | Word::Async)
    }

    fn skip_modifiers(&mut self) {
        while self.modifier() {
            self.at += if self.word(0) == Word::Extern && matches!(self.kind(1), TokenKind::Literal { .. }) { 2 } else { 1 };
        }
    }

    fn items(&mut self) {
        while self.at < self.tokens.len() && self.kind(0) != TokenKind::CloseBrace {
            let before = self.at;
            self.item();
            if self.at == before {
                self.at += 1;
            }
        }
    }

    fn emit(&mut self, kind: Kind, name: String) {
        self.found.push(Entry { path: self.path.clone(), kind, name });
    }

    fn named_to_body(&mut self, kind: Kind) {
        self.emit(kind, self.name(1));
        self.at += 2;
        self.to_body();
    }

    fn named_to_semicolon(&mut self, kind: Kind, name_at: usize) {
        let name = self.name(name_at);
        self.emit(kind, if name == "_" { String::new() } else { name });
        self.at += name_at + 1;
        self.to_semicolon();
    }

    fn item(&mut self) {
        self.skip_attributes();
        self.skip_visibility();
        self.skip_restriction();
        self.skip_modifiers();
        match self.word(0) {
            Word::Fn => self.named_to_body(Kind::Function),
            Word::Struct => self.named_to_body(Kind::Struct),
            Word::Enum => self.named_to_body(Kind::Enum),
            Word::Trait => self.named_to_body(Kind::Trait),
            Word::Macro if is_name(self.kind(1)) => self.named_to_body(Kind::Macro2),
            Word::Union if is_name(self.kind(1)) => self.named_to_body(Kind::Union),
            Word::Impl => {
                self.headers.push(self.header());
                self.emit(Kind::Impl, String::new());
                self.at += 1;
                self.to_body();
            }
            Word::Const if is_name(self.kind(1)) => self.named_to_semicolon(Kind::Const, 1),
            Word::Static => {
                let name_at = if self.lexeme(1) == "mut" { 2 } else { 1 };
                self.named_to_semicolon(Kind::Static, name_at);
            }
            Word::Type if is_name(self.kind(1)) => {
                self.aliases.extend(self.type_alias());
                self.named_to_semicolon(Kind::TypeAlias, 1);
            }
            Word::Use => {
                let renames = self.use_aliases();
                self.aliases.extend(renames);
                self.emit(Kind::Use, String::new());
                self.at += 1;
                self.to_semicolon();
            }
            Word::Mod if is_name(self.kind(1)) => self.module(),
            Word::Extern => self.external(),
            Word::MacroRules if self.kind(1) == TokenKind::Bang && is_name(self.kind(2)) => self.macro_rules(),
            _ => self.macro_call(),
        }
    }

    fn module(&mut self) {
        let name = self.name(1);
        self.emit(Kind::Mod, name.clone());
        self.at += 2;
        if self.kind(0) == TokenKind::Semi {
            self.at += 1;
            return;
        }
        let outer = self.path.len();
        self.path.push_str(&name);
        self.path.push_str("::");
        self.block();
        self.path.truncate(outer);
    }

    fn block(&mut self) {
        self.at += 1;
        self.items();
        self.at += 1;
    }

    fn external(&mut self) {
        if self.lexeme(1) == "crate" {
            self.named_to_semicolon(Kind::ExternCrate, 2);
            return;
        }
        self.emit(Kind::ExternBlock, String::new());
        self.at += if matches!(self.kind(1), TokenKind::Literal { .. }) { 2 } else { 1 };
        self.block();
    }

    fn macro_rules(&mut self) {
        self.emit(Kind::MacroRules, self.name(2));
        let facts = self.macro_facts(self.name(2), 3);
        self.macros.push(facts);
        self.at += 3;
        self.macro_body();
    }

    fn macro_body(&mut self) {
        let braced = self.kind(0) == TokenKind::OpenBrace;
        self.skip_group();
        if !braced && self.kind(0) == TokenKind::Semi {
            self.at += 1;
        }
    }

    fn macro_call(&mut self) {
        let mut ahead = 0;
        if self.kind(0) == TokenKind::Dollar {
            ahead = 1;
        }
        if self.kind(ahead) == TokenKind::Colon {
            ahead += 2;
        }
        while is_name(self.kind(ahead)) && self.kind(ahead + 1) == TokenKind::Colon {
            ahead += 3;
        }
        if is_name(self.kind(ahead)) && self.kind(ahead + 1) == TokenKind::Bang && opens(self.kind(ahead + 2)) {
            self.emit(Kind::MacroCall, self.name(ahead));
            let args = self.group_names(ahead + 2);
            self.uses.push(MacroUse { name: self.name(ahead), args });
            self.at += ahead + 2;
            self.macro_body();
        }
    }

    fn to_semicolon(&mut self) {
        while self.at < self.tokens.len() {
            let kind = self.kind(0);
            if kind == TokenKind::Semi {
                self.at += 1;
                return;
            }
            if closes(kind) {
                return;
            }
            if opens(kind) {
                self.skip_group();
            } else {
                self.at += 1;
            }
        }
    }

    fn to_body(&mut self) {
        let mut angles = 0_i32;
        while self.at < self.tokens.len() {
            let kind = self.kind(0);
            match kind {
                TokenKind::Semi => {
                    self.at += 1;
                    return;
                }
                TokenKind::OpenBrace if angles <= 0 => {
                    self.skip_group();
                    return;
                }
                TokenKind::CloseBrace | TokenKind::CloseParen | TokenKind::CloseBracket => return,
                TokenKind::Lt => angles += 1,
                TokenKind::Gt if self.at > 0 && !matches!(self.tokens[self.at - 1].kind, TokenKind::Minus | TokenKind::Eq) => {
                    angles -= 1;
                }
                _ => {}
            }
            if opens(kind) {
                self.skip_group();
            } else {
                self.at += 1;
            }
        }
    }
}
