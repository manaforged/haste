use std::collections::{HashMap, HashSet};

use hir_def::{AdtId, VariantId};
use hir_ty::mir::{BorrowKind, LocalId, MirBody, Operand, OperandKind, Rvalue, StatementKind, StoredPlace, TerminatorKind};
use hir_ty::next_solver::{Ty, TyKind};
use rustc_type_ir::inherent::{GenericArgs as _, IntoKind};

use super::Lower;

const FREEZE_DEPTH: u32 = 16;

fn plain_local(place: &StoredPlace) -> Option<LocalId> {
    let place = place.as_ref();
    place.projection.as_slice().is_empty().then_some(place.local)
}

fn constant_operand(op: &Operand, promoted: &HashSet<LocalId>) -> bool {
    match &op.kind {
        OperandKind::Constant { .. } | OperandKind::Allocation { .. } => true,
        OperandKind::Copy(place) | OperandKind::Move(place) => plain_local(place).is_some_and(|it| promoted.contains(&it)),
        OperandKind::Static(_) => false,
    }
}

fn constant_rvalue(rvalue: &Rvalue, promoted: &HashSet<LocalId>) -> bool {
    match rvalue {
        Rvalue::Use(op) => constant_operand(op, promoted),
        Rvalue::Aggregate(_, ops) => ops.iter().all(|op| constant_operand(op, promoted)),
        Rvalue::Ref(BorrowKind::Shared, place) => plain_local(place).is_some_and(|it| promoted.contains(&it)),
        _ => false,
    }
}

struct Uses {
    writes: HashMap<LocalId, u32>,
    mutated: HashSet<LocalId>,
    borrowed: HashSet<LocalId>,
}

fn survey(body: &MirBody<'_>) -> Uses {
    let mut uses = Uses { writes: HashMap::new(), mutated: HashSet::new(), borrowed: HashSet::new() };
    for (_, block) in body.basic_blocks.iter() {
        for statement in &block.statements {
            let StatementKind::Assign(target, rvalue) = &statement.kind else {
                continue;
            };
            let root = target.as_ref();
            *uses.writes.entry(root.local).or_insert(0) += 1;
            if !root.projection.as_slice().is_empty() {
                uses.mutated.insert(root.local);
            }
            if let Rvalue::Ref(kind, place) = rvalue {
                let local = place.as_ref().local;
                if matches!(kind, BorrowKind::Shared) {
                    uses.borrowed.insert(local);
                } else {
                    uses.mutated.insert(local);
                }
            }
        }
        if let Some(terminator) = &block.terminator
            && let TerminatorKind::Call { destination, .. } = &terminator.kind
        {
            uses.mutated.insert(destination.as_ref().local);
        }
    }
    uses
}

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn promoted_locals(&self) -> HashSet<LocalId> {
        let body = self.body;
        let uses = survey(body);
        let mut promoted = HashSet::new();
        loop {
            let before = promoted.len();
            for (_, block) in body.basic_blocks.iter() {
                for statement in &block.statements {
                    let StatementKind::Assign(target, rvalue) = &statement.kind else {
                        continue;
                    };
                    let Some(local) = plain_local(target) else {
                        continue;
                    };
                    let single = uses.writes.get(&local) == Some(&1) && !uses.mutated.contains(&local);
                    let plain = !body.param_locals.contains(&local) && !promoted.contains(&local);
                    if single && plain && constant_rvalue(rvalue, &promoted) && self.promotable(body.locals[local].ty.as_ref()) {
                        promoted.insert(local);
                    }
                }
            }
            if promoted.len() == before {
                break;
            }
        }
        let needed: HashSet<LocalId> = promoted.iter().copied().filter(|local| uses.borrowed.contains(local)).collect();
        if needed.is_empty() { needed } else { promoted }
    }

    fn promotable(&self, ty: Ty<'db>) -> bool {
        !self.needs_drop(ty) && self.freeze(ty, FREEZE_DEPTH)
    }

    fn freeze(&self, ty: Ty<'db>, depth: u32) -> bool {
        if depth == 0 {
            return false;
        }
        match ty.kind() {
            TyKind::Adt(adt, args) => {
                let variants: Vec<VariantId> = match adt.def_id() {
                    AdtId::StructId(id) => {
                        if self.infcx.interner.lang_items().UnsafeCell == Some(id) {
                            return false;
                        }
                        vec![id.into()]
                    }
                    AdtId::UnionId(id) => vec![id.into()],
                    AdtId::EnumId(id) => id.enum_variants(self.db).variants.values().map(|it| it.0.into()).collect(),
                };
                variants.into_iter().all(|variant| {
                    self.fields_of(variant, args).into_iter().all(|field| self.freeze(field, depth - 1))
                })
            }
            TyKind::Tuple(items) => items.as_slice().iter().all(|item| self.freeze(*item, depth - 1)),
            TyKind::Array(item, _) | TyKind::Slice(item) | TyKind::Pat(item, _) => self.freeze(item, depth - 1),
            TyKind::Closure(_, args) => self.freeze(args.as_closure().tupled_upvars_ty(), depth - 1),
            TyKind::Dynamic(..) | TyKind::Alias(..) | TyKind::Param(_) => false,
            _ => true,
        }
    }
}
