use rustc_abi::FieldsShape;

use crate::body::{ExprId, List};
use crate::names::ItemRef;
use crate::sema::infer::Res;
use crate::sema::layout::{Layout, field_offset};
use crate::sema::ty::{Kind, Ty};

use super::super::const_val::{ConstEnv, Frame, PtrTarget, Val};
use super::super::literal::Scope;

impl<'l, 'a, 'r> ConstEnv<'l, 'a, 'r> {
    pub(in super::super) fn eval_tuple_val(
        &mut self,
        scope: Scope<'_>,
        list: List,
        ty: Ty,
        frame: &mut Frame,
    ) -> Option<Val> {
        let Kind::Tuple(_) = *self.lower.cx.types.kind(ty) else {
            return None;
        };
        let layout = self.lower.cx.layout(ty).ok()?;
        let size = self.ty_size(ty)?;
        let id = self.alloc_val(Val::zeroed(size))?;
        for (idx, &raw) in scope.body.list(list).iter().enumerate() {
            let val = self.eval_expr_val(scope, ExprId(raw), frame)?;
            let off = u64::try_from(field_offset(&layout, 0, idx as u32)).ok()?;
            self.write_bytes(PtrTarget::Obj { id, addend: off }, &val)?;
        }
        self.read_bytes(PtrTarget::Obj { id, addend: 0 }, size)
    }

    pub(in super::super) fn eval_array_val(
        &mut self,
        scope: Scope<'_>,
        list: List,
        ty: Ty,
        frame: &mut Frame,
    ) -> Option<Val> {
        let Kind::Array(elem, _) = *self.lower.cx.types.kind(ty) else {
            return None;
        };
        let stride = self.lower.cx.layout(elem).ok()?.size.bytes();
        let id = self.alloc_ty(ty)?;
        for (idx, &raw) in scope.body.list(list).iter().enumerate() {
            let val = self.eval_expr_val(scope, ExprId(raw), frame)?;
            self.write_bytes(
                PtrTarget::Obj {
                    id,
                    addend: stride.checked_mul(idx as u64)?,
                },
                &val,
            )?;
        }
        let size = self.ty_size(ty)?;
        self.read_bytes(PtrTarget::Obj { id, addend: 0 }, size)
    }

    pub(in super::super) fn eval_repeat_val(
        &mut self,
        scope: Scope<'_>,
        item: ExprId,
        ty: Ty,
        frame: &mut Frame,
    ) -> Option<Val> {
        let Kind::Array(elem, count_ty) = *self.lower.cx.types.kind(ty) else {
            return None;
        };
        let stride = self.lower.cx.layout(elem).ok()?.size.bytes();
        let count = self.lower.cx.types.len(count_ty)?;
        let val = self.eval_expr_val(scope, item, frame)?;
        let id = self.alloc_ty(ty)?;
        for idx in 0..count {
            self.write_bytes(
                PtrTarget::Obj {
                    id,
                    addend: stride.checked_mul(idx)?,
                },
                &val,
            )?;
        }
        let size = self.ty_size(ty)?;
        self.read_bytes(PtrTarget::Obj { id, addend: 0 }, size)
    }

    pub(in super::super) fn eval_struct_val(
        &mut self,
        scope: Scope<'_>,
        (expr, fields, base): (ExprId, List, Option<ExprId>),
        ty: Ty,
        frame: &mut Frame,
    ) -> Option<Val> {
        let variant = match scope.typed.res.get(&expr) {
            Some(&Res::Ctor(_, variant, _)) => variant,
            _ => 0,
        };
        let base_val = match base {
            Some(b) => Some(self.eval_expr_val(scope, b, frame)?),
            None => None,
        };
        let mut pairs = Vec::new();
        for &raw in scope.body.list(fields) {
            let value = scope.body.fields[raw as usize].value;
            pairs.push((
                *scope.typed.inits.get(&value)?,
                self.eval_expr_val(scope, value, frame)?,
            ));
        }
        self.construct_adt(ty, variant, pairs, base_val)
    }

    /// Constructs an ADT value from field `(index, value)` pairs and optional `base`.
    pub(in super::super) fn construct_adt(
        &mut self,
        ty: Ty,
        variant: u32,
        fields: Vec<(u32, Val)>,
        base: Option<Val>,
    ) -> Option<Val> {
        let Kind::Adt(item, _) = *self.lower.cx.types.kind(ty) else {
            return None;
        };
        let layout = self.lower.cx.layout(ty).ok()?;
        if matches!(layout.fields, FieldsShape::Union(_)) {
            return None;
        }
        let size = self.ty_size(ty)?;
        let id = self.alloc_val(base.unwrap_or_else(|| Val::zeroed(size)))?;
        for (idx, val) in fields {
            let off = u64::try_from(field_offset(&layout, variant, idx)).ok()?;
            self.write_bytes(PtrTarget::Obj { id, addend: off }, &val)?;
        }
        self.write_adt_tag((id, item, variant), &layout)?;
        self.read_bytes(PtrTarget::Obj { id, addend: 0 }, size)
    }

    fn write_adt_tag(
        &mut self,
        (id, item, variant): (u32, ItemRef, u32),
        layout: &Layout,
    ) -> Option<()> {
        if let Some((off, tag, bits)) =
            super::super::super::adt::tag_bits(self.lower.cx, item, layout, variant).ok()?
        {
            let tag_val = Val::scalar(tag.bytes() as usize, bits);
            self.write_bytes(
                PtrTarget::Obj {
                    id,
                    addend: u64::try_from(off).ok()?,
                },
                &tag_val,
            )?;
        }
        Some(())
    }
}
