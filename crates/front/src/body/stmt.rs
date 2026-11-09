use anyhow::Result;
use rustc_lexer::TokenKind;

use crate::outline::scan;

use super::cursor::{Cursor, is_name};
use super::{BlockKind, Expr, ExprId, Node, Stmt};

const QUALIFIED: [(&str, &[&str]); 3] = [("unsafe", &["fn", "impl", "trait", "extern", "auto"]), ("async", &["fn", "unsafe"]), ("safe", &["fn", "static"])];

const ITEM_WORDS: [&str; 12] = ["fn", "struct", "enum", "trait", "impl", "mod", "use", "type", "extern", "pub", "macro_rules", "macro"];

impl Cursor<'_> {
    /// Parses the block that opens at the current token.
    pub fn block_expr(&mut self, label: Option<u32>, kind: BlockKind) -> Result<ExprId> {
        if self.kind() != TokenKind::OpenBrace {
            return self.fail("expected a block");
        }
        let open = self.group();
        let (stmts, tail) = self.inside(open, Cursor::stmts)?;
        let stmts = self.list(&stmts);
        Ok(self.push_expr(Expr::Block { stmts, tail, label, kind }))
    }

    fn stmts(&mut self) -> Result<(Vec<u32>, Option<ExprId>)> {
        let mut stmts = Vec::new();
        while self.punct("#") && self.punct_at(self.at + 1, "!") {
            self.at += 2;
            self.group();
        }
        while !self.done() {
            if self.eat(";") {
                continue;
            }
            let start = self.at;
            let attrs = self.skip_attrs()?;
            let (stmt, tail) = self.stmt(start)?;
            if let Some(tail) = tail {
                return Ok(self.ended(stmts, tail, attrs));
            }
            if let Stmt::Expr(expr, semi) = stmt
                && !self.configured_out(attrs)
                && let Some((inner, inner_tail)) = self.spliced(expr)
            {
                stmts.extend(inner);
                if let Some(inner_tail) = inner_tail {
                    self.push_stmt(Stmt::Expr(inner_tail, semi), None, &mut stmts);
                }
                continue;
            }
            self.push_stmt(stmt, attrs, &mut stmts);
        }
        Ok((stmts, None))
    }

    /// The statements and tail of a block whose last expression is `tail` with attributes `attrs`: the last statement without a semicolon when `tail` is configured out or a macro that expands to no tail.
    fn ended(&self, mut stmts: Vec<u32>, tail: ExprId, attrs: Option<(u32, u32)>) -> (Vec<u32>, Option<ExprId>) {
        if self.configured_out(attrs) {
            return self.last_as_tail(stmts);
        }
        let Some((inner, inner_tail)) = self.spliced(tail) else { return (stmts, Some(tail)) };
        stmts.extend(inner);
        match inner_tail {
            Some(inner_tail) => (stmts, Some(inner_tail)),
            None => self.last_as_tail(stmts),
        }
    }

    /// The statements and tail of `expr` when it is the block a macro call in statement position expanded to, whose statements belong to the enclosing block as in rustc.
    fn spliced(&self, expr: ExprId) -> Option<(Vec<u32>, Option<ExprId>)> {
        if self.expansion != Some(expr) {
            return None;
        }
        match self.body.exprs[expr.0 as usize] {
            super::Expr::Block { stmts, tail, label: None, kind: super::BlockKind::Plain } => Some((self.body.list(stmts).to_vec(), tail)),
            _ => None,
        }
    }

    fn last_as_tail(&self, mut stmts: Vec<u32>) -> (Vec<u32>, Option<ExprId>) {
        match stmts.last().map(|&index| &self.body.stmts[index as usize]) {
            Some(&Stmt::Expr(expr, false)) => {
                stmts.pop();
                (stmts, Some(expr))
            }
            _ => (stmts, None),
        }
    }

    fn push_stmt(&mut self, stmt: Stmt, attrs: Option<(u32, u32)>, into: &mut Vec<u32>) {
        self.body.stmts.push(stmt);
        let index = self.body.stmts.len() as u32 - 1;
        if let Some((from, to)) = attrs {
            self.body.attrs.push((Node::Stmt(index), from, to));
        }
        if !self.configured_out(attrs) {
            into.push(index);
        }
    }

    fn stmt(&mut self, start: u32) -> Result<(Stmt, Option<ExprId>)> {
        if self.item_ahead() {
            let end = scan::item_end(self.src, start)?;
            self.at = end;
            return Ok((Stmt::Item(start, end), None));
        }
        if self.is_word("let") {
            return Ok((self.let_stmt()?, None));
        }
        let (expr, ends) = self.stmt_expr()?;
        if self.eat(";") {
            return Ok((self.stmt_of(expr, true), None));
        }
        if self.done() {
            return Ok((Stmt::Empty, Some(expr)));
        }
        if ends {
            return Ok((self.stmt_of(expr, false), None));
        }
        self.fail("expected `;`")
    }

    fn stmt_of(&self, expr: ExprId, semi: bool) -> Stmt {
        match self.body.exprs[expr.0 as usize] {
            Expr::Macro { path, open } => Stmt::Macro { path, open, semi },
            _ => Stmt::Expr(expr, semi),
        }
    }

    /// Parses an expression under the statement rule: a block-like expression ends there unless `.` or `?` follows. Returns it and whether it ended that way.
    pub fn stmt_expr(&mut self) -> Result<(ExprId, bool)> {
        let ends = self.block_like_ahead() || self.brace_macro_ahead();
        if !ends {
            return Ok((self.expr()?, false));
        }
        let expr = self.primary()?;
        if self.punct("?") || (self.punct(".") && !self.punct("..")) {
            return Ok((self.continue_expr(expr)?, false));
        }
        Ok((expr, true))
    }

    fn continue_expr(&mut self, expr: ExprId) -> Result<ExprId> {
        let expr = self.postfix(expr)?;
        let expr = self.binary_tail(expr, 0)?;
        self.assign_tail(expr)
    }

    fn brace_macro_ahead(&self) -> bool {
        let mut at = self.at;
        if self.punct_at(at, "$") {
            at += 1;
        }
        if self.punct_at(at, "::") {
            at += 2;
        }
        while is_name(self.kind_at(at)) && self.punct_at(at + 1, "::") {
            at += 3;
        }
        is_name(self.kind_at(at)) && self.punct_at(at + 1, "!") && self.kind_at(at + 2) == TokenKind::OpenBrace
    }

    fn item_ahead(&self) -> bool {
        let (word, next) = (self.word(), self.word_at(self.at + 1));
        if let Some((_, after)) = QUALIFIED.iter().find(|(qualifier, _)| *qualifier == word) {
            return after.contains(&next);
        }
        match word {
            "macro_rules" => self.punct_at(self.at + 1, "!"),
            "macro" => !next.is_empty(),
            "static" => next != "move" && !self.punct_at(self.at + 1, "|"),
            "const" => !matches!(next, "" | "move" | "async") || (next == "async" && self.word_at(self.at + 2) == "fn"),
            "union" | "auto" => !next.is_empty() && !self.punct_at(self.at + 1, "!"),
            word => ITEM_WORDS.contains(&word),
        }
    }

    fn let_stmt(&mut self) -> Result<Stmt> {
        self.at += 1;
        let pat = self.pat()?;
        let ty = if self.eat(":") { Some(self.ty()?) } else { None };
        let init = if self.eat("=") { Some(self.expr()?) } else { None };
        let otherwise = if self.eat_word("else") { Some(self.block_expr(None, BlockKind::Plain)?) } else { None };
        self.expect(";")?;
        Ok(Stmt::Let { pat, ty, init, otherwise })
    }
}
