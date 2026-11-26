use anyhow::Result;
use rustc_lexer::TokenKind;

use super::cursor::Cursor;
use super::{Arm, BlockKind, Expr, ExprId, Param};

const LET_SCRUTINEE: u8 = 6;

impl Cursor<'_> {
    fn cond(&mut self) -> Result<ExprId> {
        let no_struct = self.no_struct;
        self.no_struct = true;
        let cond = self.expr();
        self.no_struct = no_struct;
        cond
    }

    pub(super) fn if_expr(&mut self) -> Result<ExprId> {
        let start = self.at;
        self.at += 1;
        let cond = self.cond()?;
        let then = self.block_expr(None, BlockKind::Plain)?;
        let otherwise = if self.eat_word("else") {
            Some(if self.is_word("if") {
                self.if_expr()?
            } else {
                self.block_expr(None, BlockKind::Plain)?
            })
        } else {
            None
        };
        Ok(self.push_expr(start, Expr::If(cond, then, otherwise)))
    }

    pub(super) fn let_expr(&mut self) -> Result<ExprId> {
        let start = self.at;
        self.at += 1;
        let pat = self.pat()?;
        self.expect("=")?;
        let no_struct = self.no_struct;
        self.no_struct = true;
        let value = self.binary(LET_SCRUTINEE);
        self.no_struct = no_struct;
        Ok(self.push_expr(start, Expr::Let(pat, value?)))
    }

    pub(super) fn looping(&mut self, start: u32, label: Option<u32>) -> Result<ExprId> {
        let word = self.word();
        self.at += 1;
        match word {
            "loop" => {
                let body = self.block_expr(None, BlockKind::Plain)?;
                Ok(self.push_expr(start, Expr::Loop { label, body }))
            }
            "while" => {
                let cond = self.cond()?;
                let body = self.block_expr(None, BlockKind::Plain)?;
                Ok(self.push_expr(start, Expr::While { label, cond, body }))
            }
            _ => {
                let pat = self.pat()?;
                if !self.eat_word("in") {
                    return self.fail("expected `in`");
                }
                let iter = self.cond()?;
                let body = self.block_expr(None, BlockKind::Plain)?;
                Ok(self.push_expr(
                    start,
                    Expr::For {
                        label,
                        pat,
                        iter,
                        body,
                    },
                ))
            }
        }
    }

    pub(super) fn match_expr(&mut self) -> Result<ExprId> {
        let start = self.at;
        self.at += 1;
        let scrutinee = self.cond()?;
        let open = self.expect_kind(TokenKind::OpenBrace)?;
        self.at = open;
        self.group();
        let arms = self.inside(open, |cursor| {
            let mut arms = Vec::new();
            while !cursor.done() {
                arms.extend(cursor.arm()?);
            }
            Ok(arms)
        })?;
        let arms = self.list(&arms);
        Ok(self.push_expr(start, Expr::Match(scrutinee, arms)))
    }

    fn arm(&mut self) -> Result<Option<u32>> {
        let attrs = self.skip_attrs()?;
        let pat = self.pat()?;
        let guard = if self.eat_word("if") {
            Some(self.expr()?)
        } else {
            None
        };
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
        Ok((!self.configured_out(attrs)).then_some(index))
    }

    /// Whether the next expression is block-like: a block, `if`, `match`, a loop, or a kinded block.
    pub fn block_like_ahead(&self) -> bool {
        match self.kind() {
            TokenKind::OpenBrace => true,
            TokenKind::Lifetime { .. } => self.punct_at(self.at + 1, ":"),
            _ => match self.word() {
                "if" | "match" | "loop" | "while" | "for" => true,
                word => self.kinded_block_ahead(word),
            },
        }
    }

    pub(super) fn jump_expr(&mut self) -> Result<ExprId> {
        let start = self.at;
        let word = self.word();
        self.at += 1;
        let value = if self.value_ahead() {
            Some(self.expr()?)
        } else {
            None
        };
        Ok(self.push_expr(
            start,
            match (word, value) {
                ("return", value) => Expr::Return(value),
                ("yield", value) => Expr::Yield(value),
                (_, Some(value)) => Expr::Become(value),
                _ => return self.fail("expected a call after `become`"),
            },
        ))
    }

    fn value_ahead(&self) -> bool {
        !self.done()
            && self.expr_start()
            && !(self.kind() == TokenKind::OpenBrace && self.no_struct)
    }

    pub(super) fn break_expr(&mut self) -> Result<ExprId> {
        let start = self.at;
        self.at += 1;
        let label = if matches!(self.kind(), TokenKind::Lifetime { .. })
            && !self.punct_at(self.at + 1, ":")
        {
            self.lifetime()
        } else {
            None
        };
        let value = if self.value_ahead() {
            Some(self.expr()?)
        } else {
            None
        };
        Ok(self.push_expr(start, Expr::Break(label, value)))
    }

    pub(super) fn closure(&mut self) -> Result<ExprId> {
        let start = self.at;
        let (capture, asynchronous) = self.closure_head();
        let params = if self.eat("||") {
            Vec::new()
        } else {
            self.closure_params()?
        };
        let params = self.list(&params);
        let ret = if self.eat("->") {
            Some(self.ty_no_bounds()?)
        } else {
            None
        };
        let body = if ret.is_some() {
            self.block_expr(None, BlockKind::Plain)?
        } else {
            self.expr()?
        };
        let (ret, body) = match asynchronous {
            true => (
                None,
                self.push_expr(
                    start,
                    Expr::Block {
                        stmts: super::List::default(),
                        tail: Some(body),
                        label: None,
                        kind: BlockKind::Async,
                    },
                ),
            ),
            false => (ret, body),
        };
        Ok(self.push_expr(
            start,
            Expr::Closure {
                params,
                ret,
                body,
                capture,
            },
        ))
    }

    fn closure_head(&mut self) -> (bool, bool) {
        let (mut capture, mut asynchronous) = (false, false);
        while matches!(self.word(), "move" | "static" | "async" | "const") {
            capture |= self.word() == "move";
            asynchronous |= self.word() == "async";
            self.at += 1;
        }
        (capture, asynchronous)
    }

    fn closure_params(&mut self) -> Result<Vec<u32>> {
        self.expect("|")?;
        let mut params = Vec::new();
        while !self.punct("|") {
            self.skip_attrs()?;
            let pat = self.pat_no_alt()?;
            let ty = if self.eat(":") {
                Some(self.ty()?)
            } else {
                None
            };
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
