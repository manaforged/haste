use anyhow::{Result, bail};
use haste_codegen::{Block, Value};

use crate::body::{Expr, ExprId, List};

use super::super::place::Place;
use super::super::{Loop, Lower, Repr};
use super::const_bool;

impl Lower<'_, '_> {
    pub(in crate::lower) fn branch_value(&mut self, value: Option<Value>, join: Block) {
        if self.open {
            let args: Vec<Value> = value.into_iter().collect();
            let args = if self.b.block_params(join).is_empty() {
                Vec::new()
            } else {
                args
            };
            self.jump(join, &args);
            self.close();
        }
    }

    pub(in crate::lower) fn join_block(
        &mut self,
        expr: ExprId,
    ) -> Result<(Block, Option<Value>, Option<Place>)> {
        let join = self.b.create_block();
        let repr = self.repr_of(expr)?;
        let slot = self.join_slot(repr);
        let joined = Self::carrier(repr)
            .filter(|_| slot.is_none())
            .map(|ty| self.b.append_block_param(join, ty));
        Ok((join, joined, slot))
    }

    pub(in crate::lower) fn branch_into(
        &mut self,
        value: Option<Value>,
        (join, slot): (Block, Option<Place>),
    ) {
        match (slot, value) {
            (Some(place), Some(value)) if self.open => {
                self.write(place, value);
                self.branch_value(None, join);
            }
            _ => self.branch_value(value, join),
        }
    }

    pub(in crate::lower) fn joined(
        &mut self,
        joined: Option<Value>,
        slot: Option<Place>,
    ) -> Option<Value> {
        match slot {
            Some(place) => self.read(place),
            None => joined,
        }
    }

    pub(super) fn if_expr(
        &mut self,
        expr: ExprId,
        cond: ExprId,
        then: ExprId,
        otherwise: Option<ExprId>,
    ) -> Result<Option<Value>> {
        if let Some(taken) = const_bool(self.body, cond) {
            return if taken {
                self.expr(then)
            } else {
                match otherwise {
                    Some(otherwise) => self.expr(otherwise),
                    None => Ok(None),
                }
            };
        }
        let mark = self.owned.len();
        let Some(test) = self.expr(cond)? else {
            return Ok(None);
        };
        if !self.cond_has_let(cond) {
            self.end_statement(mark)?;
        }
        let cond_end = self.owned.len();
        let (then_block, else_block) = (self.b.create_block(), self.b.create_block());
        let (join, joined, slot) = match otherwise {
            Some(_) => self.join_block(expr)?,
            None => (self.b.create_block(), None, None),
        };
        self.brif(test, (then_block, &[]), (else_block, &[]));
        let cond_owned = self.owned[mark..cond_end].to_vec();
        self.enter(then_block);
        let value = self.expr(then)?;
        let value = self.snapshot_before_drops(mark, then, value)?;
        if self.open {
            self.end_scope(mark)?;
        } else {
            self.owned.truncate(mark);
        }
        self.branch_into(value, (join, slot));
        let else_mark = self.owned.len();
        self.owned.extend(cond_owned);
        self.enter(else_block);
        self.drop_to(else_mark)?;
        self.owned.truncate(else_mark);
        let value = match otherwise {
            Some(otherwise) => {
                let value = self.expr(otherwise)?;
                self.snapshot_before_drops(else_mark, otherwise, value)?
            }
            None => None,
        };
        if self.open {
            self.end_scope(else_mark)?;
        } else {
            self.owned.truncate(else_mark);
        }
        self.branch_into(value, (join, slot));
        self.enter(join);
        Ok(self.joined(joined, slot))
    }

    pub(super) fn while_expr(
        &mut self,
        cond: ExprId,
        body: ExprId,
        label: Option<u32>,
    ) -> Result<Option<Value>> {
        let (head, inner, exit) = (
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
        );
        self.jump(head, &[]);
        self.enter(head);
        let depth = self.owned.len();
        let test = self.expr(cond)?;
        if self.open && !self.cond_has_let(cond) {
            self.end_statement(depth)?;
        }
        if let Some(test) = test {
            let fail = self.b.create_block();
            self.brif(test, (inner, &[]), (fail, &[]));
            self.enter(fail);
            self.drop_to(depth)?;
            self.jump(exit, &[]);
        }
        self.enter(inner);
        self.loop_body((head, exit, depth), label, body)
    }

    fn cond_has_let(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Let(..) => true,
            Expr::Paren(inner) => self.cond_has_let(inner),
            Expr::Binary(crate::body::BinOp::And | crate::body::BinOp::Or, lhs, rhs) => {
                self.cond_has_let(lhs) || self.cond_has_let(rhs)
            }
            _ => false,
        }
    }

    pub(super) fn labeled_block(
        &mut self,
        expr: ExprId,
        (stmts, tail): (List, Option<ExprId>),
        label: u32,
    ) -> Result<Option<Value>> {
        let exit = self.b.create_block();
        let value = self.value_slot(expr)?;
        self.loops.push(Loop {
            head: exit,
            exit,
            value,
            depth: self.owned.len(),
            label: Some(label),
            block: true,
        });
        let output = self.block(stmts, tail);
        self.loops.pop();
        if let (Some(output), Some(place)) = (output?, value)
            && self.open
        {
            self.write(place, output);
        }
        self.branch_value(None, exit);
        self.enter(exit);
        Ok(value.and_then(|place| self.read(place)))
    }

    pub(in crate::lower) fn value_slot(&mut self, expr: ExprId) -> Result<Option<Place>> {
        let repr = self.repr_of(expr)?;
        if let Some(place) = self.join_slot(repr) {
            return Ok(Some(place));
        }
        let held = Self::carrier(repr).map_or(Repr::Zst, Repr::Scalar);
        Ok(self.slot(held).map(|slot| Place::slot(slot, held)))
    }

    pub(super) fn loop_expr(
        &mut self,
        expr: ExprId,
        body: ExprId,
        label: Option<u32>,
    ) -> Result<Option<Value>> {
        let (head, exit) = (self.b.create_block(), self.b.create_block());
        let value = self.value_slot(expr)?;
        self.jump(head, &[]);
        self.enter(head);
        let depth = self.owned.len();
        self.loops.push(Loop {
            head,
            exit,
            value,
            depth,
            label,
            block: false,
        });
        let result = self.expr(body);
        self.loops.pop();
        result?;
        self.branch_value(None, head);
        self.enter(exit);
        Ok(value.and_then(|place| self.read(place)))
    }

    fn jump_target(&self, label: Option<u32>) -> Option<Loop> {
        self.loops
            .iter()
            .rev()
            .find(|frame| crate::body::jumps_to(self.src, (frame.label, frame.block), label))
            .copied()
    }

    pub(super) fn break_expr(
        &mut self,
        label: Option<u32>,
        value: Option<ExprId>,
    ) -> Result<Option<Value>> {
        let Some(Loop {
            exit,
            value: slot,
            depth,
            ..
        }) = self.jump_target(label)
        else {
            bail!("`break` outside a loop")
        };
        let value = match value {
            Some(expr) => {
                let value = self.expr(expr)?;
                self.snapshot_before_drops(depth, expr, value)?
            }
            None => None,
        };
        if let (Some(value), Some(place)) = (value, slot) {
            self.write(place, value);
        }
        self.drop_to(depth)?;
        self.branch_value(None, exit);
        Ok(None)
    }

    pub(super) fn continue_expr(&mut self, label: Option<u32>) -> Result<Option<Value>> {
        let Some(Loop { head, depth, .. }) = self.jump_target(label) else {
            bail!("`continue` outside a loop")
        };
        self.drop_to(depth)?;
        self.branch_value(None, head);
        Ok(None)
    }

    pub(super) fn return_expr(&mut self, value: Option<ExprId>) -> Result<Option<Value>> {
        let value = match value {
            Some(value) => self.expr(value)?,
            None => None,
        };
        self.finish(value)?;
        Ok(None)
    }
}
