use anyhow::Result;

use crate::body::{Expr, ExprId, List, Lit, Pat, PatId, UnOp};
use crate::diag::DiagnosticKind;
use crate::names::{ItemRef, Primitive};
use crate::outline::item::{Attr, ItemKind};
use crate::sema::ty::{Kind, Ty};

use super::super::{Infer, Res, parse_int, position, rest_split};

const MAX_BUDGET: u32 = 32_768;
pub(super) const MAX_ARRAY_EXPAND: u64 = 16;

#[derive(Clone, Debug)]
pub(super) enum MPat {
    Wild,
    Or(Vec<MPat>),
    Ctor(CtorKind, Vec<MPat>),
    IntRange(u128, u128),
    Slice {
        prefix: Vec<MPat>,
        rest: bool,
        suffix: Vec<MPat>,
    },
    Opaque,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum CtorKind {
    Bool(bool),
    Variant(ItemRef, u32),
    Tuple(usize),
    Struct(ItemRef),
    Array(usize),
}

impl Infer<'_, '_> {
    pub(in super::super) fn check_all_matches_exhaustive(&mut self) -> Result<()> {
        let matches = std::mem::take(&mut self.matches);
        for (scrutinee, scrutinee_ty, arms) in matches {
            self.check_match_exhaustive(scrutinee, scrutinee_ty, arms)?;
        }
        Ok(())
    }

    fn check_match_exhaustive(
        &mut self,
        scrutinee: ExprId,
        scrutinee_ty: Ty,
        arms: List,
    ) -> Result<()> {
        if self.cx.outline.crates[self.home.krate as usize].env.fixed {
            return Ok(());
        }
        let ty = self.peel_match_ref(scrutinee_ty);
        if matches!(
            *self.cx.types.kind(ty),
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) | Kind::Param(_) | Kind::Proj(_)
        ) {
            return Ok(());
        }
        let mut rows: Vec<Vec<MPat>> = Vec::new();
        for &at in self.body.list(arms) {
            let arm = self.body.arms[at as usize];
            if arm.guard.is_some() {
                continue;
            }
            rows.push(vec![self.lower_match_pat(arm.pat, ty)]);
        }
        let mut budget = MAX_BUDGET;
        if let Some(mut witness) = self.find_witness(&rows, &[ty], &mut budget) {
            let missing = witness.pop().unwrap_or_else(|| "_".to_owned());
            return Err(self.diag_expr(
                DiagnosticKind::Exhaustive,
                Some(scrutinee),
                format!("non-exhaustive patterns: `{missing}` not covered"),
                format!("pattern `{missing}` not covered"),
            ));
        }
        Ok(())
    }

    pub(super) fn peel_match_ref(&self, mut ty: Ty) -> Ty {
        loop {
            let resolved = self.resolve(ty);
            match *self.cx.types.kind(resolved) {
                Kind::Ref(_, inner) => ty = inner,
                _ => return resolved,
            }
        }
    }

    pub(super) fn is_uninhabited(&mut self, ty: Ty, depth: u32) -> bool {
        if depth > 8 {
            return false;
        }
        let ty = self.peel_match_ref(ty);
        match *self.cx.types.kind(ty) {
            Kind::Never => true,
            Kind::Tuple(items) => self
                .cx
                .types
                .substs(items)
                .to_vec()
                .into_iter()
                .any(|t| self.is_uninhabited(t, depth + 1)),
            Kind::Array(elem, len) => {
                self.cx.types.len(len).is_some_and(|n| n > 0)
                    && self.is_uninhabited(elem, depth + 1)
            }
            Kind::Adt(item, args) => {
                let kind = self.cx.item(item).1.kind;
                if kind == ItemKind::Enum {
                    let Ok(def) = self.cx.adt_def(item) else {
                        return false;
                    };
                    if def.variants.is_empty() {
                        return !self.is_foreign_non_exhaustive(item);
                    }
                    let slice = self.cx.types.substs(args).to_vec();
                    let Ok(fields) = self.adt_fields(item, &slice) else {
                        return false;
                    };
                    !self.is_foreign_non_exhaustive(item)
                        && fields
                            .iter()
                            .all(|vf| vf.iter().any(|&f| self.is_uninhabited(f, depth + 1)))
                } else if kind == ItemKind::Struct {
                    let slice = self.cx.types.substs(args).to_vec();
                    let Ok(fields) = self.adt_fields(item, &slice) else {
                        return false;
                    };
                    fields
                        .first()
                        .is_some_and(|sf| sf.iter().any(|&f| self.is_uninhabited(f, depth + 1)))
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub(super) fn is_foreign_non_exhaustive(&self, item: ItemRef) -> bool {
        if item.krate == self.home.krate {
            return false;
        }
        let (src, entry) = self.cx.item(item);
        entry.attrs.iter().any(|attr| match attr {
            Attr::Other(tok) => src.text(*tok) == "non_exhaustive",
            Attr::Path(p) => p == "non_exhaustive",
            _ => false,
        })
    }

    fn ctor_kind_for(&self, item: ItemRef, variant: u32) -> CtorKind {
        if self.cx.item(item).1.kind == ItemKind::Enum {
            CtorKind::Variant(item, variant)
        } else {
            CtorKind::Struct(item)
        }
    }

    fn lower_match_pat(&mut self, pat: PatId, ty: Ty) -> MPat {
        let ty = self.peel_match_ref(ty);
        match *self.body.pat(pat) {
            Pat::Wild | Pat::Rest | Pat::Never => MPat::Wild,
            Pat::Paren(inner) | Pat::Ref { pat: inner, .. } => self.lower_match_pat(inner, ty),
            Pat::Box(inner) | Pat::Deref(inner) => match self.cx.boxed(ty) {
                Some(inner_ty) => self.lower_match_pat(inner, inner_ty),
                None => MPat::Opaque,
            },
            Pat::Or(list) => {
                let alts: Vec<MPat> = self
                    .body
                    .list(list)
                    .to_vec()
                    .into_iter()
                    .map(|alt| self.lower_match_pat(PatId(alt), ty))
                    .collect();
                match alts.len() {
                    0 => MPat::Opaque,
                    1 => alts.into_iter().next().unwrap_or(MPat::Opaque),
                    _ => MPat::Or(alts),
                }
            }
            Pat::Bind { sub: Some(sub), .. } => self.lower_match_pat(sub, ty),
            Pat::Bind { sub: None, .. } => self.lower_bind_or_path_pat(pat, ty, true),
            Pat::Path(_) => self.lower_bind_or_path_pat(pat, ty, false),
            Pat::Lit(expr) => self.lower_lit_pat(expr, ty),
            Pat::Range {
                from,
                to,
                inclusive,
            } => self.lower_range_pat(from, to, inclusive, ty),
            Pat::Tuple(list) => self.lower_tuple_pat(list, ty),
            Pat::TupleStruct(_, list) => self.lower_tuple_struct_pat(pat, list),
            Pat::Struct { fields, .. } => self.lower_struct_pat(pat, fields),
            Pat::Slice(list) => self.lower_slice_pat(list, ty),
            Pat::Macro { .. } => MPat::Opaque,
        }
    }

    fn lower_bind_or_path_pat(&mut self, pat: PatId, ty: Ty, is_bind: bool) -> MPat {
        match self.out.pat_res.get(&pat).copied() {
            Some(Res::Ctor(item, variant, _)) => {
                MPat::Ctor(self.ctor_kind_for(item, variant), Vec::new())
            }
            Some(Res::Fn(item, _)) if self.cx.item(item).1.kind == ItemKind::Const => {
                self.eval_const_pat_item(item, ty).unwrap_or(MPat::Opaque)
            }
            None if is_bind => MPat::Wild,
            _ => MPat::Opaque,
        }
    }

    fn eval_const_pat_item(&mut self, item: ItemRef, ty: Ty) -> Option<MPat> {
        let (val, _) = self.const_value(item, &[]).ok()?;
        match *self.cx.types.kind(ty) {
            Kind::Prim(Primitive::Bool) => Some(MPat::Ctor(CtorKind::Bool(val != 0), Vec::new())),
            Kind::Prim(prim) if int_spec(prim).is_some() => {
                let key = encode_raw_int(val, prim)?;
                Some(MPat::IntRange(key, key))
            }
            _ => None,
        }
    }

    fn lower_lit_pat(&mut self, expr: ExprId, ty: Ty) -> MPat {
        match *self.body.expr(expr) {
            Expr::Lit(Lit::Bool(b), _) => MPat::Ctor(CtorKind::Bool(b), Vec::new()),
            _ => match self.eval_int_pat_key(expr, ty) {
                Some(k) => MPat::IntRange(k, k),
                None => MPat::Opaque,
            },
        }
    }

    fn lower_range_pat(
        &mut self,
        from: Option<ExprId>,
        to: Option<ExprId>,
        inclusive: bool,
        ty: Ty,
    ) -> MPat {
        let Kind::Prim(prim) = *self.cx.types.kind(ty) else {
            return MPat::Opaque;
        };
        let Some((min, max)) = int_key_domain(prim) else {
            return MPat::Opaque;
        };
        let lo = match from {
            Some(e) => match self.eval_int_pat_key(e, ty) {
                Some(v) => v,
                None => return MPat::Opaque,
            },
            None => min,
        };
        let hi = match to {
            Some(e) => match self.eval_int_pat_key(e, ty) {
                Some(v) => {
                    if inclusive {
                        v
                    } else if v == min {
                        return MPat::Opaque;
                    } else {
                        v - 1
                    }
                }
                None => return MPat::Opaque,
            },
            None => max,
        };
        if lo > hi {
            MPat::Opaque
        } else {
            MPat::IntRange(lo, hi)
        }
    }

    fn eval_int_pat_key(&mut self, expr: ExprId, ty: Ty) -> Option<u128> {
        let Kind::Prim(prim) = *self.cx.types.kind(ty) else {
            return None;
        };
        let (bits, signed) = int_spec(prim)?;
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.eval_int_pat_key(inner, ty),
            Expr::Unary(UnOp::Neg, inner) => {
                if !signed {
                    return None;
                }
                let raw = self.eval_lit_u128(inner)?;
                let neg = raw.wrapping_neg();
                encode_raw_int(neg, prim)
            }
            Expr::Lit(..) => {
                let raw = self.eval_lit_u128(expr)?;
                if signed && bits < 128 && raw >= (1u128 << (bits - 1)) {
                    return None;
                }
                encode_raw_int(raw, prim)
            }
            Expr::Path(_) => {
                let Res::Fn(item, _) = self.out.res.get(&expr).copied()? else {
                    return None;
                };
                if self.cx.item(item).1.kind != ItemKind::Const {
                    return None;
                }
                let (val, _) = self.const_value(item, &[]).ok()?;
                encode_raw_int(val, prim)
            }
            _ => None,
        }
    }

    fn eval_lit_u128(&self, expr: ExprId) -> Option<u128> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.eval_lit_u128(inner),
            Expr::Lit(Lit::Int, tok) => parse_int(self.src.text(tok)).ok().map(|(v, _)| v),
            Expr::Lit(Lit::Char, tok) => crate::body::literal::char_value(self.src.text(tok))
                .ok()
                .map(|c| u128::from(c as u32)),
            Expr::Lit(Lit::Byte, tok) => crate::body::literal::byte_value(self.src.text(tok))
                .ok()
                .map(u128::from),
            _ => None,
        }
    }

    fn lower_positional_pat(&mut self, list: List, field_tys: &[Ty], ctor: CtorKind) -> MPat {
        let Ok((pats, rest)) = rest_split(self.body, list) else {
            return MPat::Opaque;
        };
        let mut args = vec![MPat::Wild; field_tys.len()];
        for (at, &sub) in pats.iter().enumerate() {
            if let Some(idx) = position(at, pats.len(), rest, field_tys.len()) {
                args[idx] = self.lower_match_pat(sub, field_tys[idx]);
            }
        }
        MPat::Ctor(ctor, args)
    }

    fn lower_tuple_pat(&mut self, list: List, ty: Ty) -> MPat {
        let Kind::Tuple(items) = *self.cx.types.kind(ty) else {
            return MPat::Opaque;
        };
        let elems = self.cx.types.substs(items).to_vec();
        self.lower_positional_pat(list, &elems, CtorKind::Tuple(elems.len()))
    }

    fn lower_tuple_struct_pat(&mut self, pat: PatId, list: List) -> MPat {
        let Some(Res::Ctor(item, variant, args)) = self.out.pat_res.get(&pat).copied() else {
            return MPat::Opaque;
        };
        let slice = self.cx.types.substs(args).to_vec();
        let Ok(all_fields) = self.adt_fields(item, &slice) else {
            return MPat::Opaque;
        };
        let fields = all_fields[variant as usize].clone();
        let ctor = self.ctor_kind_for(item, variant);
        self.lower_positional_pat(list, &fields, ctor)
    }

    fn lower_struct_pat(&mut self, pat: PatId, fields_list: List) -> MPat {
        let Some(Res::Ctor(item, variant, args)) = self.out.pat_res.get(&pat).copied() else {
            return MPat::Opaque;
        };
        let slice = self.cx.types.substs(args).to_vec();
        let Ok(all_fields) = self.adt_fields(item, &slice) else {
            return MPat::Opaque;
        };
        let fields = all_fields[variant as usize].clone();
        let mut args = vec![MPat::Wild; fields.len()];
        for &at in self.body.list(fields_list).to_vec().iter() {
            let fp = self.body.field_pats[at as usize];
            if let Some(&idx) = self.out.field_pats.get(&fp.pat) {
                let idx = idx as usize;
                if idx < fields.len() {
                    args[idx] = self.lower_match_pat(fp.pat, fields[idx]);
                }
            }
        }
        MPat::Ctor(self.ctor_kind_for(item, variant), args)
    }

    fn lower_slice_pat(&mut self, list: List, ty: Ty) -> MPat {
        let Ok((pats, rest)) = rest_split(self.body, list) else {
            return MPat::Opaque;
        };
        match *self.cx.types.kind(ty) {
            Kind::Array(elem, len) => {
                let Some(n) = self.cx.types.len(len) else {
                    return MPat::Opaque;
                };
                if n > MAX_ARRAY_EXPAND {
                    let fixed = pats.len() - usize::from(rest.is_some());
                    let all_wild = pats.iter().enumerate().all(|(i, &p)| {
                        Some(i) == rest || matches!(self.lower_match_pat(p, elem), MPat::Wild)
                    });
                    return if all_wild
                        && (fixed as u64 == n || (rest.is_some() && fixed as u64 <= n))
                    {
                        MPat::Wild
                    } else {
                        MPat::Opaque
                    };
                }
                let n = n as usize;
                let mut args = vec![MPat::Wild; n];
                for (at, &sub) in pats.iter().enumerate() {
                    if let Some(idx) = position(at, pats.len(), rest, n) {
                        args[idx] = self.lower_match_pat(sub, elem);
                    }
                }
                MPat::Ctor(CtorKind::Array(n), args)
            }
            Kind::Slice(elem) => {
                let split = rest.unwrap_or(pats.len());
                let prefix = pats[..split]
                    .iter()
                    .map(|&p| self.lower_match_pat(p, elem))
                    .collect();
                let suffix = rest.map_or_else(Vec::new, |s| {
                    pats[s + 1..]
                        .iter()
                        .map(|&p| self.lower_match_pat(p, elem))
                        .collect()
                });
                MPat::Slice {
                    prefix,
                    rest: rest.is_some(),
                    suffix,
                }
            }
            _ => MPat::Opaque,
        }
    }
}

pub(super) fn int_spec(prim: Primitive) -> Option<(u32, bool)> {
    Some(match prim {
        Primitive::I8 => (8, true),
        Primitive::I16 => (16, true),
        Primitive::I32 => (32, true),
        Primitive::I64 | Primitive::Isize => (64, true),
        Primitive::I128 => (128, true),
        Primitive::U8 => (8, false),
        Primitive::U16 => (16, false),
        Primitive::U32 => (32, false),
        Primitive::U64 | Primitive::Usize => (64, false),
        Primitive::U128 => (128, false),
        Primitive::Char => (32, false),
        _ => return None,
    })
}

pub(super) fn int_key_domain(prim: Primitive) -> Option<(u128, u128)> {
    if prim == Primitive::Char {
        return Some((0, 0x10_FFFF));
    }
    let (bits, _) = int_spec(prim)?;
    let max = if bits == 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };
    Some((0, max))
}

fn encode_raw_int(raw: u128, prim: Primitive) -> Option<u128> {
    let (bits, signed) = int_spec(prim)?;
    let mask = if bits == 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };
    let val = raw & mask;
    if signed {
        Some(val ^ (1u128 << (bits - 1)))
    } else {
        Some(val)
    }
}

pub(super) fn decode_key_to_string(key: u128, prim: Primitive) -> String {
    if prim == Primitive::Char {
        let ch = char::from_u32(key as u32).unwrap_or('\0');
        return format!("{ch:?}");
    }
    let Some((bits, signed)) = int_spec(prim) else {
        return key.to_string();
    };
    if !signed {
        return key.to_string();
    }
    let raw = key ^ (1u128 << (bits - 1));
    let signed_val = if bits == 128 {
        raw as i128
    } else {
        let shift = 128 - bits;
        ((raw << shift) as i128) >> shift
    };
    signed_val.to_string()
}
