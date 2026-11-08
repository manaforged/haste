use hir_def::HasModule;
use hir_ty::mir::{AggregateKind, MirBody, Operand, Rvalue, Statement, StatementKind, TerminatorKind};
use hir_ty::next_solver::Ty;

use super::{Lower, Place};
use crate::backend::{Resolved, Target, body_of, resolve};

/// Limits on a callee body that is lowered into its caller.
struct Budget {
    size: usize,
    blocks: usize,
    depth: usize,
}

const BUDGET: Budget = Budget { size: 12, blocks: 4, depth: 3 };

fn lowered_statements(body: &MirBody<'_>) -> usize {
    let assigns = |statements: &[Statement]| {
        statements.iter().filter(|statement| matches!(statement.kind, StatementKind::Assign(..))).count()
    };
    body.basic_blocks.iter().map(|(_, block)| assigns(&block.statements) + 1).sum()
}

fn small(body: &MirBody<'_>) -> bool {
    body.basic_blocks.len() <= BUDGET.blocks && lowered_statements(body) <= BUDGET.size
}

fn lowerable(kind: &StatementKind) -> bool {
    let StatementKind::Assign(_, rvalue) = kind else {
        return true;
    };
    match rvalue {
        Rvalue::Aggregate(aggregate, _) => !matches!(aggregate, AggregateKind::Union(..)),
        Rvalue::Use(..)
        | Rvalue::Repeat(..)
        | Rvalue::CopyForDeref(..)
        | Rvalue::Ref(..)
        | Rvalue::Len(..)
        | Rvalue::Discriminant(..)
        | Rvalue::Cast(..)
        | Rvalue::CheckedBinaryOp(..)
        | Rvalue::UnaryOp(..) => true,
        _ => false,
    }
}

impl<'db> Lower<'_, '_, 'db> {
    /// Lowers a small callee body in place of a direct call, or returns false to keep the call.
    pub(super) fn inline(&mut self, callee: &Target<'db>, dest: Place<'db>, places: &[Place<'db>]) -> Result<bool, String> {
        let Some(body) = self.inline_body(callee, dest, places) else {
            return Ok(false);
        };
        let landing = self.landing();
        let (mark, notes) = (self.builder.mark(), crate::backend::note_count());
        let exit = self.builder.create_block();
        let ret_pointer = (self.layout(dest.ty.ty)?.size.bytes() != 0).then_some(dest.addr);
        let mut inlined = self.inlined.clone();
        inlined.push(self.body);
        let mut inner = Lower::new(self.db, self.env, body, &mut *self.builder, self.slots, self.call_conv, self.sig);
        inner.landing = landing;
        inner.exit = Some(exit);
        inner.ret_pointer = ret_pointer;
        inner.inlined = inlined;
        if inner.enter(places).is_err() {
            self.builder.rewind(mark);
            crate::backend::truncate_notes(notes);
            return Ok(false);
        }
        self.builder.switch_to_block(exit);
        Ok(true)
    }

    fn enter(mut self, places: &[Place<'db>]) -> Result<(), String> {
        self.prepare()?;
        let body = self.body;
        for (param, place) in body.param_locals.iter().zip(places) {
            let size = self.layout(body.locals[*param].ty.as_ref())?.size.bytes();
            if size != 0 {
                let to = self.local_addr(*param);
                self.copy(to, place.addr, size);
            }
            self.set_flag(*param, true);
        }
        self.lower_blocks()
    }

    /// Copies the return local out and leaves the body, by a return or, when inlined, a jump to the caller.
    pub(super) fn leave(&mut self) -> Result<(), String> {
        if let Some(pointer) = self.ret_pointer {
            let (ret, local) = self.body.locals.iter().next().ok_or("no return local")?;
            let size = self.layout(local.ty.as_ref())?.size.bytes();
            let from = self.local_addr(ret);
            self.copy(pointer, from, size);
        }
        match self.exit {
            Some(exit) => {
                self.builder.ins().jump(exit, &[]);
            }
            None => {
                self.builder.ins().return_(&[]);
            }
        }
        Ok(())
    }

    fn inline_body(&self, callee: &Target<'db>, dest: Place<'db>, places: &[Place<'db>]) -> Option<&'db MirBody<'db>> {
        let Target::Func(func, _) = callee else {
            return None;
        };
        if self.inlined.len() >= BUDGET.depth || func.module(self.db).krate(self.db).data(self.db).origin.is_local() {
            return None;
        }
        let body = body_of(self.db, self.env, callee).ok()?;
        let owns_drop = body.locals.iter().any(|(_, local)| self.needs_drop(local.ty.as_ref()));
        let eligible = !owns_drop && !self.recursive(body) && small(body) && self.fits(body, dest, places) && self.plain(body);
        eligible.then_some(body)
    }

    fn recursive(&self, body: &MirBody<'db>) -> bool {
        std::ptr::eq(self.body, body) || self.inlined.iter().any(|outer| std::ptr::eq(*outer, body))
    }

    fn sized(&self, ty: Ty<'db>) -> Option<u64> {
        self.layout(ty).ok().filter(|layout| !layout.is_unsized()).map(|layout| layout.size.bytes())
    }

    fn fits(&self, body: &MirBody<'db>, dest: Place<'db>, places: &[Place<'db>]) -> bool {
        let Some((_, ret)) = body.locals.iter().next() else {
            return false;
        };
        let same = |ty: Ty<'db>, site: Ty<'db>| self.sized(ty).is_some_and(|size| self.sized(site) == Some(size));
        let params = body.param_locals.iter().map(|param| body.locals[*param].ty.as_ref());
        body.param_locals.len() == places.len()
            && same(ret.ty.as_ref(), dest.ty.ty)
            && params.zip(places).all(|(ty, place)| same(ty, place.ty.ty))
    }

    fn plain(&self, body: &'db MirBody<'db>) -> bool {
        let site_cleanup = self.has_landing();
        body.basic_blocks.iter().all(|(_, block)| {
            block.statements.iter().all(|statement| lowerable(&statement.kind))
                && block.terminator.as_ref().is_some_and(|terminator| self.plain_exit(body, &terminator.kind, site_cleanup))
        })
    }

    fn plain_exit(&self, body: &'db MirBody<'db>, kind: &TerminatorKind, site_cleanup: bool) -> bool {
        match kind {
            TerminatorKind::Goto { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::UnwindResume
            | TerminatorKind::Abort
            | TerminatorKind::Assert { .. } => true,
            TerminatorKind::Call { func, cleanup, .. } => (cleanup.is_none() || !site_cleanup) && !self.raises(body, func),
            _ => false,
        }
    }

    fn raises(&self, body: &'db MirBody<'db>, func: &Operand) -> bool {
        matches!(
            resolve(self.db, self.env, body, func),
            Ok(Resolved { target: Target::Foreign(name), .. }) if name == crate::unwind::RAISE_SYMBOL
        )
    }
}
