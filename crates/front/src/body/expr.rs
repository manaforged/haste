use anyhow::Result;
use rustc_lexer::TokenKind;

use super::cursor::Cursor;
use super::path::Style;
use super::{Arm, BinOp, BlockKind, Expr, ExprId, FieldInit, Param, UnOp};

const BINARY: [(&str, BinOp, u8); 18] = [
    ("||", BinOp::Or, 4),
    ("&&", BinOp::And, 5),
    ("==", BinOp::Eq, 6),
    ("!=", BinOp::Ne, 6),
    ("<=", BinOp::Le, 6),
    (">=", BinOp::Ge, 6),
    ("<<", BinOp::Shl, 10),
    (">>", BinOp::Shr, 10),
    ("<", BinOp::Lt, 6),
    (">", BinOp::Gt, 6),
    ("|", BinOp::BitOr, 7),
    ("^", BinOp::BitXor, 8),
    ("&", BinOp::BitAnd, 9),
    ("+", BinOp::Add, 11),
    ("-", BinOp::Sub, 11),
    ("*", BinOp::Mul, 12),
    ("/", BinOp::Div, 12),
    ("%", BinOp::Rem, 12),
];

const COMPOUND: [(&str, BinOp); 10] = [
    ("<<=", BinOp::Shl),
    (">>=", BinOp::Shr),
    ("+=", BinOp::Add),
    ("-=", BinOp::Sub),
    ("*=", BinOp::Mul),
    ("/=", BinOp::Div),
    ("%=", BinOp::Rem),
    ("^=", BinOp::BitXor),
    ("&=", BinOp::BitAnd),
    ("|=", BinOp::BitOr),
];

const CAST: u8 = 13;
const LET_SCRUTINEE: u8 = 6;

impl Cursor<'_> {
    /// Parses an expression, assignment included.
    pub fn expr(&mut self) -> Result<ExprId> {
        let lhs = self.range()?;
        self.assign_tail(lhs)
    }

    pub(super) fn assign_tail(&mut self, lhs: ExprId) -> Result<ExprId> {
        if self.punct("=") && !self.punct("==") && !self.punct("=>") {
            self.at += 1;
            let rhs = self.expr()?;
            return Ok(self.push_expr(Expr::Assign(lhs, rhs)));
        }
        if let Some((text, op)) = COMPOUND.iter().find(|(text, _)| self.punct(text)) {
            self.at += text.len() as u32;
            let rhs = self.expr()?;
            return Ok(self.push_expr(Expr::AssignOp(*op, lhs, rhs)));
        }
        Ok(lhs)
    }

    fn range(&mut self) -> Result<ExprId> {
        if self.punct("..") {
            return self.range_from(None);
        }
        let lhs = self.binary(0)?;
        if self.punct("..") { self.range_from(Some(lhs)) } else { Ok(lhs) }
    }

    fn range_from(&mut self, from: Option<ExprId>) -> Result<ExprId> {
        let inclusive = self.punct("..=");
        self.at += if inclusive { 3 } else { 2 };
        let to = if self.expr_start() { Some(self.binary(0)?) } else { None };
        Ok(self.push_expr(Expr::Range { from, to, inclusive }))
    }

    fn binop(&self) -> Option<(usize, BinOp, u8)> {
        if self.is_word("as") {
            return Some((0, BinOp::Add, CAST));
        }
        if COMPOUND.iter().any(|(text, _)| self.punct(text)) || self.punct("..") {
            return None;
        }
        BINARY.iter().find(|(text, _, _)| self.punct(text)).map(|(text, op, prec)| (text.len(), *op, *prec))
    }

    /// Parses binary operators that bind at least as tightly as `min`.
    pub fn binary(&mut self, min: u8) -> Result<ExprId> {
        let lhs = self.unary_operand()?;
        self.binary_tail(lhs, min)
    }

    pub(super) fn binary_tail(&mut self, mut lhs: ExprId, min: u8) -> Result<ExprId> {
        while let Some((len, op, prec)) = self.binop() {
            if prec < min {
                break;
            }
            if prec == CAST {
                self.at += 1;
                let ty = self.ty_no_bounds()?;
                lhs = self.push_expr(Expr::Cast(lhs, ty));
                continue;
            }
            self.at += len as u32;
            let rhs = self.binary(prec + 1)?;
            lhs = self.push_expr(Expr::Binary(op, lhs, rhs));
        }
        Ok(lhs)
    }

    /// Parses a unary expression and its postfix operators.
    pub fn unary_operand(&mut self) -> Result<ExprId> {
        self.skip_attrs()?;
        let op = if self.punct("-") {
            UnOp::Neg
        } else if self.punct("!") {
            UnOp::Not
        } else if self.punct("*") {
            UnOp::Deref
        } else if self.punct("&") {
            return self.ref_expr();
        } else {
            let primary = self.primary()?;
            return self.postfix(primary);
        };
        self.at += 1;
        let operand = self.unary_operand()?;
        Ok(self.push_expr(Expr::Unary(op, operand)))
    }

    fn ref_expr(&mut self) -> Result<ExprId> {
        let double = self.punct("&&");
        self.at += if double { 2 } else { 1 };
        let raw = self.is_word("raw") && (self.word_at(self.at + 1) == "const" || self.word_at(self.at + 1) == "mut");
        if raw {
            self.at += 1;
        }
        let mutable = self.eat_word("mut");
        if raw && !mutable {
            self.eat_word("const");
        }
        let expr = self.unary_operand()?;
        let inner = self.push_expr(Expr::Ref { mutable, raw, expr });
        Ok(if double { self.push_expr(Expr::Ref { mutable: false, raw: false, expr: inner }) } else { inner })
    }

    /// Applies postfix operators to `expr`: calls, methods, fields, indexing, `?` and `.await`.
    pub fn postfix(&mut self, mut expr: ExprId) -> Result<ExprId> {
        loop {
            expr = match self.kind() {
                TokenKind::Question => {
                    self.at += 1;
                    self.push_expr(Expr::Try(expr))
                }
                TokenKind::OpenParen => {
                    let open = self.group();
                    let args = self.comma_list(open, Cursor::arg)?;
                    self.push_expr(Expr::Call(expr, args))
                }
                TokenKind::OpenBracket => {
                    let open = self.group();
                    let index = self.inside(open, Cursor::expr)?;
                    self.push_expr(Expr::Index(expr, index))
                }
                TokenKind::Dot if !self.punct("..") => self.dot(expr)?,
                _ => return Ok(expr),
            };
        }
    }

    fn arg(&mut self) -> Result<u32> {
        let attrs = self.skip_attrs()?;
        let expr = self.expr()?;
        if let Some((from, to)) = attrs {
            self.body.attrs.push((super::Node::Expr(expr), from, to));
        }
        Ok(expr.0)
    }

    fn dot(&mut self, expr: ExprId) -> Result<ExprId> {
        self.at += 1;
        if let TokenKind::Literal { kind, .. } = self.kind() {
            let text = self.src.text(self.at);
            self.at += 1;
            let float = matches!(kind, rustc_lexer::LiteralKind::Float { .. });
            let mut out = expr;
            for part in text.split('.').filter(|part| !part.is_empty() || !float) {
                let index = part.parse::<u32>().or_else(|_| self.fail("expected a tuple index"))?;
                out = self.push_expr(Expr::TupleField(out, index));
            }
            return Ok(out);
        }
        if self.eat_word("await") {
            return Ok(self.push_expr(Expr::Await(expr)));
        }
        let name = self.name()?;
        let generics = if self.punct("::") && self.punct_at(self.at + 2, "<") {
            self.at += 2;
            Some(self.angle()?)
        } else {
            None
        };
        if self.kind() != TokenKind::OpenParen {
            return Ok(self.push_expr(Expr::Field(expr, name)));
        }
        let open = self.group();
        let params = self.comma_list(open, Cursor::arg)?;
        let args = generics.map(|args| match args {
            super::Args::Angle(list) => list,
            _ => super::List::default(),
        });
        Ok(self.push_expr(Expr::Method { receiver: expr, name, args, params }))
    }

    /// Whether the next token can start an expression.
    pub fn expr_start(&self) -> bool {
        match self.kind() {
            TokenKind::Literal { .. } | TokenKind::Ident | TokenKind::RawIdent | TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::Lifetime { .. } => true,
            TokenKind::OpenBrace => !self.no_struct,
            _ => ["-", "!", "*", "&", "|", "<", "::", "..", "$", "#"].iter().any(|text| self.punct(text)) && !self.punct("=>"),
        }
    }

    pub(super) fn primary(&mut self) -> Result<ExprId> {
        match self.kind() {
            TokenKind::Literal { .. } => self.literal(),
            TokenKind::OpenParen => self.paren_expr(),
            TokenKind::OpenBracket => self.array_expr(),
            TokenKind::OpenBrace => self.block_expr(None, BlockKind::Plain),
            TokenKind::Lifetime { .. } => self.labeled(),
            _ if self.punct("|") => self.closure(),
            _ if self.punct("<") || self.punct("::") => self.path_expr(),
            _ => self.keyword_expr(),
        }
    }

    fn labeled(&mut self) -> Result<ExprId> {
        let label = self.lifetime();
        self.expect(":")?;
        match self.word() {
            "loop" | "while" | "for" => self.looping(label),
            _ => self.block_expr(label, BlockKind::Plain),
        }
    }

    fn keyword_expr(&mut self) -> Result<ExprId> {
        if let Some(expr) = self.prefixed()? {
            return Ok(expr);
        }
        match self.word() {
            "true" | "false" => self.literal(),
            "if" => self.if_expr(),
            "match" => self.match_expr(),
            "loop" | "while" | "for" => self.looping(None),
            "let" => self.let_expr(),
            "return" | "yield" | "become" => self.jump_expr(),
            "break" => self.break_expr(),
            "continue" => {
                self.at += 1;
                let label = self.lifetime();
                Ok(self.push_expr(Expr::Continue(label)))
            }
            "_" => {
                self.at += 1;
                Ok(self.push_expr(Expr::Underscore))
            }
            _ => self.path_expr(),
        }
    }

    fn prefixed(&mut self) -> Result<Option<ExprId>> {
        let word = self.word();
        let closure_next = self.punct_at(self.at + 1, "|") || self.word_at(self.at + 1) == "move";
        if matches!(word, "move" | "static" | "const" | "async") && closure_next {
            return self.closure().map(Some);
        }
        if matches!(word, "async" | "gen" | "unsafe" | "const" | "try") && self.block_kind_ahead() {
            return self.kinded_block().map(Some);
        }
        if word == "builtin" && self.punct_at(self.at + 1, "#") {
            self.at += 2;
            self.name()?;
            let open = self.group();
            return Ok(Some(self.push_expr(Expr::FormatArgs(open))));
        }
        Ok(None)
    }

    fn block_kind_ahead(&self) -> bool {
        let next = self.at + 1;
        self.kind_at(next) == TokenKind::OpenBrace || (self.word_at(next) == "move" && self.kind_at(next + 1) == TokenKind::OpenBrace)
    }

    fn kinded_block(&mut self) -> Result<ExprId> {
        let kind = match self.word() {
            "async" => BlockKind::Async,
            "gen" => BlockKind::Gen,
            "unsafe" => BlockKind::Unsafe,
            "const" => BlockKind::Const,
            _ => BlockKind::Try,
        };
        self.at += 1;
        self.eat_word("move");
        self.block_expr(None, kind)
    }

    fn path_expr(&mut self) -> Result<ExprId> {
        let path = self.path(Style::Expr)?;
        if self.punct("!") && !self.punct("!=") && matches!(self.kind_at(self.at + 1), TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace) {
            self.at += 1;
            let open = self.group();
            return Ok(self.push_expr(Expr::Macro { path, open }));
        }
        if self.kind() == TokenKind::OpenBrace && !self.no_struct {
            return self.struct_expr(path);
        }
        Ok(self.push_expr(Expr::Path(path)))
    }

    fn struct_expr(&mut self, path: super::PathId) -> Result<ExprId> {
        let open = self.group();
        let mut base = None;
        let mut rest = false;
        let fields = self.inside(open, |cursor| {
            let mut fields = Vec::new();
            while !cursor.done() {
                let attrs = cursor.skip_attrs()?;
                if cursor.eat("..") {
                    rest = true;
                    base = if cursor.done() { None } else { Some(cursor.expr()?) };
                    break;
                }
                let field = cursor.field_init()?;
                if let Some((from, to)) = attrs {
                    cursor.body.attrs.push((super::Node::Field(field), from, to));
                }
                fields.push(field);
                if !cursor.eat(",") {
                    break;
                }
            }
            Ok(fields)
        })?;
        let fields = self.list(&fields);
        Ok(self.push_expr(Expr::Struct { path, fields, base, rest }))
    }

    fn field_init(&mut self) -> Result<u32> {
        let name = self.at;
        if !(matches!(self.kind(), TokenKind::Literal { .. }) || crate::body::cursor::is_name(self.kind())) {
            return self.fail("expected a field");
        }
        self.at += 1;
        let value = if self.eat(":") { self.expr()? } else { self.name_path(name) };
        self.body.fields.push(FieldInit { name, value });
        Ok(self.body.fields.len() as u32 - 1)
    }

    fn name_path(&mut self, name: u32) -> ExprId {
        self.body.segments.push(super::Segment { name, args: None });
        let segment = self.body.segments.len() as u32 - 1;
        let segments = self.list(&[segment]);
        let path = self.push_path(super::Path { qself: None, global: false, segments });
        self.push_expr(Expr::Path(path))
    }

    fn paren_expr(&mut self) -> Result<ExprId> {
        let open = self.group();
        let (items, trailing) = self.inside(open, |cursor| {
            let mut items = Vec::new();
            let mut trailing = false;
            while !cursor.done() {
                items.push(cursor.arg()?);
                trailing = cursor.eat(",");
                if !trailing {
                    break;
                }
            }
            Ok((items, trailing))
        })?;
        if items.len() == 1 && !trailing {
            return Ok(self.push_expr(Expr::Paren(ExprId(items[0]))));
        }
        let list = self.list(&items);
        Ok(self.push_expr(Expr::Tuple(list)))
    }

    fn array_expr(&mut self) -> Result<ExprId> {
        let open = self.group();
        let (items, len) = self.inside(open, |cursor| {
            let mut items = Vec::new();
            while !cursor.done() {
                items.push(cursor.arg()?);
                if items.len() == 1 && cursor.eat(";") {
                    return Ok((items, Some(cursor.expr()?)));
                }
                if !cursor.eat(",") {
                    break;
                }
            }
            Ok((items, None))
        })?;
        if let Some(len) = len {
            return Ok(self.push_expr(Expr::Repeat(ExprId(items[0]), len)));
        }
        let list = self.list(&items);
        Ok(self.push_expr(Expr::Array(list)))
    }

    fn cond(&mut self) -> Result<ExprId> {
        let no_struct = self.no_struct;
        self.no_struct = true;
        let cond = self.expr();
        self.no_struct = no_struct;
        cond
    }

    fn if_expr(&mut self) -> Result<ExprId> {
        self.at += 1;
        let cond = self.cond()?;
        let then = self.block_expr(None, BlockKind::Plain)?;
        let otherwise = if self.eat_word("else") {
            Some(if self.is_word("if") { self.if_expr()? } else { self.block_expr(None, BlockKind::Plain)? })
        } else {
            None
        };
        Ok(self.push_expr(Expr::If(cond, then, otherwise)))
    }

    fn let_expr(&mut self) -> Result<ExprId> {
        self.at += 1;
        let pat = self.pat()?;
        self.expect("=")?;
        let no_struct = self.no_struct;
        self.no_struct = true;
        let value = self.binary(LET_SCRUTINEE);
        self.no_struct = no_struct;
        Ok(self.push_expr(Expr::Let(pat, value?)))
    }

    fn looping(&mut self, label: Option<u32>) -> Result<ExprId> {
        let word = self.word();
        self.at += 1;
        match word {
            "loop" => {
                let body = self.block_expr(None, BlockKind::Plain)?;
                Ok(self.push_expr(Expr::Loop { label, body }))
            }
            "while" => {
                let cond = self.cond()?;
                let body = self.block_expr(None, BlockKind::Plain)?;
                Ok(self.push_expr(Expr::While { label, cond, body }))
            }
            _ => {
                let pat = self.pat()?;
                if !self.eat_word("in") {
                    return self.fail("expected `in`");
                }
                let iter = self.cond()?;
                let body = self.block_expr(None, BlockKind::Plain)?;
                Ok(self.push_expr(Expr::For { label, pat, iter, body }))
            }
        }
    }

    fn match_expr(&mut self) -> Result<ExprId> {
        self.at += 1;
        let scrutinee = self.cond()?;
        let open = self.expect_kind(TokenKind::OpenBrace)?;
        self.at = open;
        self.group();
        let arms = self.inside(open, |cursor| {
            let mut arms = Vec::new();
            while !cursor.done() {
                arms.push(cursor.arm()?);
            }
            Ok(arms)
        })?;
        let arms = self.list(&arms);
        Ok(self.push_expr(Expr::Match(scrutinee, arms)))
    }

    fn arm(&mut self) -> Result<u32> {
        let attrs = self.skip_attrs()?;
        let pat = self.pat()?;
        let guard = if self.eat_word("if") { Some(self.expr()?) } else { None };
        self.expect("=>")?;
        let (body, block_like) = self.stmt_expr()?;
        if !self.eat(",") && !block_like && !self.done() {
            return self.fail("expected `,` after a match arm");
        }
        self.body.arms.push(Arm { pat, guard, body });
        let index = self.body.arms.len() as u32 - 1;
        if let Some((from, to)) = attrs {
            self.body.attrs.push((super::Node::Arm(index), from, to));
        }
        Ok(index)
    }

    /// Whether the next expression is block-like: a block, `if`, `match`, a loop, or a kinded block.
    pub fn block_like_ahead(&self) -> bool {
        match self.kind() {
            TokenKind::OpenBrace => true,
            TokenKind::Lifetime { .. } => self.punct_at(self.at + 1, ":"),
            _ => match self.word() {
                "if" | "match" | "loop" | "while" | "for" => true,
                "unsafe" | "const" | "async" | "gen" | "try" => self.block_kind_ahead(),
                _ => false,
            },
        }
    }

    fn jump_expr(&mut self) -> Result<ExprId> {
        let word = self.word();
        self.at += 1;
        let value = if self.value_ahead() { Some(self.expr()?) } else { None };
        Ok(self.push_expr(match (word, value) {
            ("return", value) => Expr::Return(value),
            ("yield", value) => Expr::Yield(value),
            (_, Some(value)) => Expr::Become(value),
            _ => return self.fail("expected a call after `become`"),
        }))
    }

    fn value_ahead(&self) -> bool {
        !self.done() && self.expr_start() && !(self.kind() == TokenKind::OpenBrace && self.no_struct)
    }

    fn break_expr(&mut self) -> Result<ExprId> {
        self.at += 1;
        let label = if matches!(self.kind(), TokenKind::Lifetime { .. }) && !self.punct_at(self.at + 1, ":") { self.lifetime() } else { None };
        let value = if self.value_ahead() { Some(self.expr()?) } else { None };
        Ok(self.push_expr(Expr::Break(label, value)))
    }

    fn closure(&mut self) -> Result<ExprId> {
        let capture = self.closure_head();
        let params = if self.eat("||") { Vec::new() } else { self.closure_params()? };
        let params = self.list(&params);
        let ret = if self.eat("->") { Some(self.ty_no_bounds()?) } else { None };
        let body = if ret.is_some() { self.block_expr(None, BlockKind::Plain)? } else { self.expr()? };
        Ok(self.push_expr(Expr::Closure { params, ret, body, capture }))
    }

    fn closure_head(&mut self) -> bool {
        let mut capture = false;
        while matches!(self.word(), "move" | "static" | "async" | "const") {
            capture |= self.word() == "move";
            self.at += 1;
        }
        capture
    }

    fn closure_params(&mut self) -> Result<Vec<u32>> {
        self.expect("|")?;
        let mut params = Vec::new();
        while !self.punct("|") {
            self.skip_attrs()?;
            let pat = self.pat_no_alt()?;
            let ty = if self.eat(":") { Some(self.ty()?) } else { None };
            self.body.params.push(Param { pat, ty });
            params.push(self.body.params.len() as u32 - 1);
            if !self.eat(",") {
                break;
            }
        }
        self.expect("|")?;
        Ok(params)
    }
}
