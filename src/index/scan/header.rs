use rustc_lexer::TokenKind;

use super::{Scanner, Word, is_name};
use crate::index::{Alias, Head, Header, MacroFacts, MacroUse};

impl Scanner<'_> {
    fn param_name(&self, at: usize) -> Option<usize> {
        match self.kind(at) {
            TokenKind::Lifetime { .. } => None,
            _ if self.word(at) == Word::Const => Some(at + 1),
            kind if is_name(kind) => Some(at),
            _ => None,
        }
    }

    fn params(&self, at: &mut usize) -> Vec<String> {
        let mut found = Vec::new();
        if self.kind(*at) != TokenKind::Lt {
            return found;
        }
        let (mut angles, mut groups, mut expects) = (0_i32, 0_i32, true);
        while self.kind(*at) != TokenKind::Eof {
            let kind = self.kind(*at);
            if expects && angles == 1 && groups == 0 {
                found.extend(self.param_name(*at).map(|name| self.name(name)));
                expects = false;
            }
            match kind {
                TokenKind::Lt => angles += 1,
                TokenKind::Gt if !self.follows_arrow(*at) => angles -= 1,
                TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace => groups += 1,
                TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::CloseBrace => groups -= 1,
                TokenKind::Comma => expects = true,
                _ => {}
            }
            *at += 1;
            if angles == 0 {
                break;
            }
            if kind == TokenKind::Lt && angles == 1 {
                expects = true;
            }
        }
        found
    }

    fn follows_arrow(&self, at: usize) -> bool {
        at > 0 && matches!(self.kind(at - 1), TokenKind::Minus | TokenKind::Eq)
    }

    fn after_binder(&self, at: usize) -> usize {
        if self.word(at) == Word::For && self.kind(at + 1) == TokenKind::Lt {
            let mut next = at + 1;
            self.params(&mut next);
            return next;
        }
        at
    }

    fn type_end(&self, start: usize) -> usize {
        let (mut at, mut angles, mut groups) = (start, 0_i32, 0_i32);
        loop {
            let kind = self.kind(at);
            let outer = angles <= 0 && groups == 0;
            let word = self.word(at);
            let splits = word == Word::Where || (word == Word::For && self.kind(at + 1) != TokenKind::Lt);
            if kind == TokenKind::Eof || (outer && (splits || matches!(kind, TokenKind::OpenBrace | TokenKind::Semi))) {
                return at;
            }
            match kind {
                TokenKind::Lt => angles += 1,
                TokenKind::Gt if !self.follows_arrow(at) => angles -= 1,
                TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace => groups += 1,
                TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::CloseBrace => groups -= 1,
                _ => {}
            }
            at += 1;
        }
    }

    fn path_colons(&self, at: usize) -> bool {
        self.kind(at) == TokenKind::Colon && self.kind(at + 1) == TokenKind::Colon
    }

    fn last_name(&self, start: usize) -> (String, usize) {
        let mut at = start;
        let mut last = String::new();
        loop {
            if self.path_colons(at) {
                at += 2;
            } else if is_name(self.kind(at)) {
                last = self.name(at);
                at += 1;
                if !self.path_colons(at) {
                    return (last, at);
                }
            } else {
                return (last, at);
            }
        }
    }

    fn path_head(&self, at: usize, params: &[String]) -> Head {
        if is_name(self.kind(at)) && params.contains(&self.name(at)) {
            return if self.path_colons(at + 1) { Head::Unknown } else { Head::Param };
        }
        let (name, end) = self.last_name(at);
        if name.is_empty() || self.kind(end) == TokenKind::Bang {
            return Head::Unknown;
        }
        Head::Name(name)
    }

    fn group_has(&self, start: usize, wanted: TokenKind) -> bool {
        let (mut at, mut level) = (start, 0_i32);
        while self.kind(at) != TokenKind::Eof {
            match self.kind(at) {
                TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace => level += 1,
                TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::CloseBrace => {
                    level -= 1;
                    if level == 0 {
                        return false;
                    }
                }
                kind if kind == wanted && level == 1 => return true,
                _ => {}
            }
            at += 1;
        }
        false
    }

    pub(super) fn head_at(&self, start: usize, params: &[String]) -> Head {
        let at = self.after_binder(start);
        match self.kind(at) {
            TokenKind::And => Head::Ref,
            TokenKind::Star => Head::Ptr,
            TokenKind::Bang => Head::Never,
            TokenKind::Lt => Head::Unknown,
            TokenKind::OpenBracket if self.group_has(at, TokenKind::Semi) => Head::Array,
            TokenKind::OpenBracket => Head::Slice,
            TokenKind::OpenParen if self.kind(at + 1) == TokenKind::CloseParen || self.group_has(at, TokenKind::Comma) => {
                Head::Tuple
            }
            TokenKind::OpenParen => Head::Unknown,
            _ => match self.word(at) {
                Word::Dyn => Head::Dyn,
                Word::Fn | Word::Unsafe | Word::Extern => Head::Fn,
                _ => self.path_head(at, params),
            },
        }
    }

    pub(super) fn header(&self) -> Header {
        let mut at = 1;
        let params = self.params(&mut at);
        if self.word(at) == Word::Const {
            at += 1;
        }
        if self.kind(at) == TokenKind::Bang {
            at += 1;
        }
        let end = self.type_end(at);
        if self.word(end) == Word::For {
            let trait_at = self.after_binder(at);
            return Header { trait_name: self.last_name(trait_at).0, head: self.head_at(end + 1, &params) };
        }
        Header { trait_name: String::new(), head: self.head_at(at, &params) }
    }

    pub(super) fn type_alias(&self) -> Option<Alias> {
        let mut at = 2;
        let params = self.params(&mut at);
        (self.kind(at) == TokenKind::Eq).then(|| Alias { name: self.name(1), head: self.head_at(at + 1, &params) })
    }

    fn group_end(&self, start: usize) -> usize {
        let (mut at, mut level) = (start, 0_i32);
        while self.kind(at) != TokenKind::Eof {
            match self.kind(at) {
                TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace => level += 1,
                TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::CloseBrace => {
                    level -= 1;
                    if level == 0 {
                        return at;
                    }
                }
                _ => {}
            }
            at += 1;
        }
        at
    }

    pub(super) fn group_names(&self, start: usize) -> Vec<String> {
        let names = (start..self.group_end(start)).filter(|at| is_name(self.kind(*at)));
        names.map(|at| self.name(at)).collect()
    }

    pub(super) fn macro_facts(&self, name: String, start: usize) -> MacroFacts {
        let mut facts = MacroFacts { name, declares_alias: false, calls: Vec::new() };
        for at in start..self.group_end(start) {
            facts.declares_alias |= self.word(at) == Word::Type && self.kind(at + 1) == TokenKind::Dollar;
            if is_name(self.kind(at)) && self.kind(at + 1) == TokenKind::Bang {
                facts.calls.push(MacroUse { name: self.name(at), args: self.group_names(at + 2) });
            }
        }
        facts
    }

    fn renamed_path(&self, at: usize) -> String {
        if self.lexeme(at) != "self" {
            return self.name(at);
        }
        let (mut back, mut level) = (at, 0_i32);
        while back > 0 {
            back -= 1;
            match self.kind(back) {
                TokenKind::CloseBrace => level += 1,
                TokenKind::OpenBrace if level == 0 => return if back >= 3 { self.name(back - 3) } else { String::new() },
                TokenKind::OpenBrace => level -= 1,
                _ => {}
            }
        }
        String::new()
    }

    pub(super) fn use_aliases(&self) -> Vec<Alias> {
        let mut found = Vec::new();
        let mut at = 1;
        while !matches!(self.kind(at), TokenKind::Semi | TokenKind::Eof) {
            if self.word(at) == Word::As && is_name(self.kind(at + 1)) {
                let from = self.renamed_path(at - 1);
                found.push(Alias { name: self.name(at + 1), head: Head::Name(from) });
            }
            at += 1;
        }
        found
    }
}
