use haste_codegen::{StackSlot, Type, Value};
use hir_def::{EnumVariantId, VariantId};
use hir_ty::mir::{LocalId, Operand, OperandKind, PlaceTy, ProjectionElem, Rvalue, StatementKind, StoredPlace, TerminatorKind};
use hir_ty::next_solver::{Ty, TyKind};
use rustc_type_ir::inherent::{GenericArgs as _, IntoKind};

use super::Lower;

/// One owned step from a local to a part of it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Step {
    Field(Option<EnumVariantId>, u32),
    Box,
    Item(u64),
}

/// A part of a local that some operand moves, with its run-time moved flag.
#[derive(Clone)]
pub(super) struct MovePath {
    steps: Vec<Step>,
    slot: StackSlot,
}

/// The move paths below one place during an open drop.
#[derive(Clone, Copy)]
pub(super) struct Cut<'a> {
    paths: &'a [MovePath],
    depth: usize,
    variant: Option<EnumVariantId>,
}

impl<'a> Cut<'a> {
    pub(super) fn none() -> Cut<'static> {
        Cut { paths: &[], depth: 0, variant: None }
    }

    pub(super) fn whole(paths: &'a [MovePath]) -> Self {
        Cut::at(paths, 0)
    }

    pub(super) fn at(paths: &'a [MovePath], depth: usize) -> Self {
        Cut { paths, depth, variant: None }
    }

    pub(super) fn is_empty(self) -> bool {
        self.paths.is_empty()
    }

    pub(super) fn variant(self, variant: EnumVariantId) -> Self {
        Cut { variant: Some(variant), ..self }
    }

    pub(super) fn field(self, index: u32) -> Vec<MovePath> {
        self.under(Step::Field(self.variant, index))
    }

    fn under(self, step: Step) -> Vec<MovePath> {
        self.paths.iter().filter(|path| path.steps.get(self.depth) == Some(&step)).cloned().collect()
    }

    pub(super) fn into<'b>(self, paths: &'b [MovePath]) -> Cut<'b> {
        Cut::at(paths, self.depth + 1)
    }
}

fn operands(rvalue: &Rvalue) -> Vec<&Operand> {
    match rvalue {
        Rvalue::Use(op) | Rvalue::Repeat(op, _) | Rvalue::Cast(_, op, _) | Rvalue::UnaryOp(_, op) => vec![op],
        Rvalue::CheckedBinaryOp(_, left, right) => vec![left, right],
        Rvalue::Aggregate(_, ops) => ops.iter().collect(),
        Rvalue::Ref(..)
        | Rvalue::Len(_)
        | Rvalue::Discriminant(_)
        | Rvalue::CopyForDeref(_)
        | Rvalue::ThreadLocalRef(_)
        | Rvalue::AddressOf(_)
        | Rvalue::BinaryOp(_)
        | Rvalue::NullaryOp(_) => Vec::new(),
    }
}

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn make_move_paths(&mut self) {
        let body = self.body;
        for (_, block) in body.basic_blocks.iter() {
            for statement in &block.statements {
                if let StatementKind::Assign(_, rvalue) = &statement.kind {
                    operands(rvalue).into_iter().for_each(|op| self.track_operand(op));
                }
            }
            if let Some(TerminatorKind::Call { func, args, .. }) = block.terminator.as_ref().map(|it| &it.kind) {
                self.track_operand(func);
                args.iter().for_each(|op| self.track_operand(op));
            }
        }
    }

    fn track_operand(&mut self, op: &Operand) {
        let (OperandKind::Copy(place) | OperandKind::Move(place)) = &op.kind else {
            return;
        };
        let local = place.as_ref().local;
        for steps in self.moved_paths(place).into_iter().flatten().filter(|steps| !steps.is_empty()) {
            let known = self.move_paths.entry(local).or_default().iter().any(|path| path.steps == steps);
            if known {
                continue;
            }
                        let slot = self.builder.create_stack_slot(1, 0);
            self.store_flag(slot, false);
            self.move_paths.entry(local).or_default().push(MovePath { steps, slot });
        }
    }

    fn store_flag(&mut self, slot: StackSlot, on: bool) {
        let value = self.builder.ins().iconst(Type::I8, i64::from(on));
        self.builder.ins().stack_store(value, slot, 0);
    }

    pub(super) fn owned_paths(&self, place: &StoredPlace) -> Option<Vec<Vec<Step>>> {
        Some(self.walk_owned(place)?.0)
    }

    fn moved_paths(&self, place: &StoredPlace) -> Option<Vec<Vec<Step>>> {
        if !self.flags.contains_key(&place.as_ref().local) {
            return None;
        }
        let (paths, ty) = self.walk_owned(place)?;
        self.needs_drop(ty).then_some(paths)
    }

    fn walk_owned(&self, place: &StoredPlace) -> Option<(Vec<Vec<Step>>, Ty<'db>)> {
        let stored = place.as_ref();
        let mut ty = PlaceTy::from_ty(self.body.locals[stored.local].ty.as_ref());
        let mut steps = Vec::new();
        let elements = stored.projection.as_slice();
        for (position, element) in elements.iter().enumerate() {
            match element {
                ProjectionElem::Deref if self.is_box(ty.ty) => steps.push(Step::Box),
                ProjectionElem::Field(field) => {
                    let variant = match ty.variant_id {
                        Some(VariantId::EnumVariantId(variant)) => Some(variant),
                        _ => None,
                    };
                    steps.push(Step::Field(variant, field.0 as u32));
                }
                ProjectionElem::ConstantIndex { offset, from_end } => {
                    let count = self.array_len(ty.ty)?;
                    steps.push(Step::Item(if *from_end { count - offset - 1 } else { *offset }));
                }
                ProjectionElem::Subslice { from, to } if position + 1 == elements.len() => {
                    let count = self.array_len(ty.ty)?;
                    let items = (*from..count - *to).map(|index| {
                        let mut path = steps.clone();
                        path.push(Step::Item(index));
                        path
                    });
                    let paths = items.collect();
                    return Some((paths, ty.projection_ty(&self.infcx, element, self.env.param_env).ty));
                }
                ProjectionElem::Downcast(_) => {}
                _ => return None,
            }
            ty = ty.projection_ty(&self.infcx, element, self.env.param_env);
        }
        Some((vec![steps], ty.ty))
    }

    fn array_len(&self, ty: Ty<'db>) -> Option<u64> {
        matches!(ty.kind(), TyKind::Array(..)).then(|| self.layout(ty).ok().map(|it| it.fields.count() as u64))?
    }

    pub(super) fn moved(&mut self, place: &StoredPlace) -> Result<(), String> {
        let local = place.as_ref().local;
        for steps in self.moved_paths(place).into_iter().flatten() {
            if steps.is_empty() {
                self.set_flag(local, false);
                continue;
            }
            let found = self.move_paths.get(&local).and_then(|paths| paths.iter().find(|path| path.steps == steps));
            let slot = found.ok_or("a move that the path scan did not see")?.slot;
            self.store_flag(slot, true);
        }
        Ok(())
    }

    pub(super) fn paths_under(&self, local: LocalId, steps: &[Step]) -> Vec<MovePath> {
        let paths = self.move_paths.get(&local).map(Vec::as_slice).unwrap_or_default();
        paths.iter().filter(|path| path.steps.starts_with(steps)).cloned().collect()
    }

    pub(super) fn clear_moved(&mut self, local: LocalId, steps: &[Step]) {
        for path in self.paths_under(local, steps) {
            self.store_flag(path.slot, false);
        }
    }

    pub(super) fn drop_open(
        &mut self,
        ty: Ty<'db>,
        addr: Value,
        meta: Option<Value>,
        cut: Cut<'_>,
    ) -> Result<(), String> {
        if cut.is_empty() {
            return self.call_glue(ty, addr, meta);
        }
        let own = cut.paths.iter().find(|path| path.steps.len() == cut.depth).map(|path| path.slot);
        let done = self.builder.create_block();
        if let Some(slot) = own {
            let flag = self.builder.ins().stack_load(Type::I8, slot, 0);
            let run = self.builder.create_block();
            self.builder.ins().brif(flag, done, &[], run, &[]);
            self.builder.switch_to_block(run);
        }
        let deeper: Vec<_> = cut.paths.iter().filter(|path| path.steps.len() > cut.depth).cloned().collect();
        let below = Cut { paths: &deeper, ..cut };
        if below.is_empty() {
            self.call_glue(ty, addr, meta)?;
        } else if self.is_box(ty) {
            self.drop_open_box(ty, addr, below)?;
        } else {
            self.drop_parts(ty, addr, meta, below)?;
        }
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(())
    }

    fn drop_open_box(&mut self, ty: Ty<'db>, addr: Value, cut: Cut<'_>) -> Result<(), String> {
        let TyKind::Adt(_, args) = ty.kind() else {
            return Err(format!("box of {ty:?}"));
        };
        let inner = args.as_slice().first().and_then(|arg| arg.ty()).ok_or("box without a type")?;
        let pointer = self.load(Type::I64, addr, 0);
        let paths = cut.under(Step::Box);
        self.drop_open(inner, pointer, None, cut.into(&paths))?;
        let holder = self.thin_holder(addr);
        self.drop_impl(ty, holder)?;
        self.drop_parts(ty, addr, None, Cut::none())
    }

    pub(super) fn drop_open_items(
        &mut self,
        item: Ty<'db>,
        addr: Value,
        count: u64,
        cut: Cut<'_>,
    ) -> Result<(), String> {
        if cut.is_empty() {
            let count = self.builder.ins().iconst(Type::I64, count as i64);
            return self.drop_items(item, addr, count);
        }
        let stride = self.layout(item)?.size.bytes() as i64;
        for index in 0..count {
            let at = self.builder.ins().iadd_imm(addr, index as i64 * stride);
            let paths = cut.under(Step::Item(index));
            self.drop_open(item, at, None, cut.into(&paths))?;
        }
        Ok(())
    }
}
