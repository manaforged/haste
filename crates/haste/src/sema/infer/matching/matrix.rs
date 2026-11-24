use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty};

use super::super::Infer;
use super::exhaustive::{
    CtorKind, MAX_ARRAY_EXPAND, MPat, decode_key_to_string, int_key_domain, int_spec,
};

impl Infer<'_, '_> {
    pub(super) fn find_witness(
        &mut self,
        rows: &[Vec<MPat>],
        tys: &[Ty],
        budget: &mut u32,
    ) -> Option<Vec<String>> {
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        let (&first_ty, rest_tys) = match tys.split_first() {
            Some(parts) => parts,
            None => return rows.is_empty().then(Vec::new),
        };
        let ty = self.peel_match_ref(first_ty);
        if self.is_uninhabited(ty, 0) {
            return None;
        }
        let expanded = expand_col0_or(rows);
        let rows = expanded.as_slice();
        match *self.cx.types.kind(ty) {
            Kind::Prim(Primitive::Bool) => self.witness_bool(rows, rest_tys, budget),
            Kind::Prim(prim) if int_spec(prim).is_some() => {
                self.witness_int(rows, prim, rest_tys, budget)
            }
            Kind::Tuple(items) => {
                let elems = self.cx.types.substs(items).to_vec();
                self.witness_product(
                    rows,
                    &CtorKind::Tuple(elems.len()),
                    &elems,
                    rest_tys,
                    budget,
                    |parts| format!("({})", parts.join(", ")),
                )
            }
            Kind::Array(elem, len) => {
                if let Some(n) = self.cx.types.len(len).filter(|&n| n <= MAX_ARRAY_EXPAND) {
                    let n = n as usize;
                    let elems = vec![elem; n];
                    self.witness_product(
                        rows,
                        &CtorKind::Array(n),
                        &elems,
                        rest_tys,
                        budget,
                        |parts| format!("[{}]", parts.join(", ")),
                    )
                } else {
                    self.witness_infinite(rows, rest_tys, budget)
                }
            }
            Kind::Slice(elem) => self.witness_slice(rows, elem, rest_tys, budget),
            Kind::Adt(item, args) => self.witness_adt(rows, item, args, rest_tys, budget),
            _ => self.witness_infinite(rows, rest_tys, budget),
        }
    }

    fn witness_bool(
        &mut self,
        rows: &[Vec<MPat>],
        rest_tys: &[Ty],
        budget: &mut u32,
    ) -> Option<Vec<String>> {
        for b in [false, true] {
            let spec = specialize_ctor(rows, &CtorKind::Bool(b), 0);
            if let Some(mut rest) = self.find_witness(&spec, rest_tys, budget) {
                rest.push(b.to_string());
                return Some(rest);
            }
        }
        None
    }

    fn witness_int(
        &mut self,
        rows: &[Vec<MPat>],
        prim: Primitive,
        rest_tys: &[Ty],
        budget: &mut u32,
    ) -> Option<Vec<String>> {
        let (min, max) = int_key_domain(prim)?;
        let domains: Vec<(u128, u128)> = if prim == Primitive::Char {
            vec![(0, 0xD7FF), (0xE000, 0x10_FFFF)]
        } else {
            vec![(min, max)]
        };
        let mut points: Vec<u128> = Vec::new();
        for &(d_lo, d_hi) in &domains {
            points.push(d_lo);
            for row in rows {
                if let MPat::IntRange(lo, hi) = row[0] {
                    let c_lo = lo.max(d_lo);
                    let c_hi = hi.min(d_hi);
                    if c_lo <= c_hi {
                        points.push(c_lo);
                        if c_hi < d_hi {
                            points.push(c_hi + 1);
                        }
                    }
                }
            }
        }
        points.sort_unstable();
        points.dedup();
        for &(d_lo, d_hi) in &domains {
            let domain_pts: Vec<u128> = points
                .iter()
                .copied()
                .filter(|&p| p >= d_lo && p <= d_hi)
                .collect();
            for i in 0..domain_pts.len() {
                let seg_lo = domain_pts[i];
                let seg_hi = domain_pts
                    .get(i + 1)
                    .map_or(d_hi, |&next| next - 1)
                    .min(d_hi);
                if seg_lo > seg_hi {
                    continue;
                }
                let spec: Vec<Vec<MPat>> = rows
                    .iter()
                    .filter(|row| match row[0] {
                        MPat::Wild => true,
                        MPat::IntRange(lo, hi) => lo <= seg_lo && seg_hi <= hi,
                        _ => false,
                    })
                    .map(|row| row[1..].to_vec())
                    .collect();
                if let Some(mut rest) = self.find_witness(&spec, rest_tys, budget) {
                    rest.push(format_int_witness(seg_lo, seg_hi, prim));
                    return Some(rest);
                }
            }
        }
        None
    }

    fn witness_product(
        &mut self,
        rows: &[Vec<MPat>],
        ctor: &CtorKind,
        field_tys: &[Ty],
        rest_tys: &[Ty],
        budget: &mut u32,
        format: impl FnOnce(&[String]) -> String,
    ) -> Option<Vec<String>> {
        let arity = field_tys.len();
        let spec = specialize_ctor(rows, ctor, arity);
        let mut sub_tys = Vec::with_capacity(arity + rest_tys.len());
        sub_tys.extend_from_slice(field_tys);
        sub_tys.extend_from_slice(rest_tys);
        let mut w = self.find_witness(&spec, &sub_tys, budget)?;
        let split = w.len() - arity;
        let mut parts: Vec<String> = w.drain(split..).collect();
        parts.reverse();
        w.push(format(&parts));
        Some(w)
    }

    fn witness_adt(
        &mut self,
        rows: &[Vec<MPat>],
        item: ItemRef,
        args: Substs,
        rest_tys: &[Ty],
        budget: &mut u32,
    ) -> Option<Vec<String>> {
        let kind = self.cx.item(item).1.kind;
        if kind == ItemKind::Union {
            return self.witness_infinite(rows, rest_tys, budget);
        }
        let slice = self.cx.types.substs(args).to_vec();
        let Ok(all_fields) = self.adt_fields(item, &slice) else {
            return None;
        };
        let Ok(def) = self.cx.adt_def(item) else {
            return None;
        };
        let adt_name = self.cx.name(item).to_owned();
        if kind == ItemKind::Struct {
            let fields = all_fields.first().cloned().unwrap_or_default();
            let shape = def.variants.first().map(|v| v.shape);
            return self.witness_product(
                rows,
                &CtorKind::Struct(item),
                &fields,
                rest_tys,
                budget,
                |parts| format_ctor_witness(&adt_name, shape, parts),
            );
        }
        let adt_src = &self.cx.file(item).source;
        let variants: Vec<(u32, String, crate::body::adt::Shape)> = def
            .variants
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, adt_src.text(v.name).to_owned(), v.shape))
            .collect();
        for (v_idx, v_name, shape) in variants {
            let v_fields = all_fields[v_idx as usize].clone();
            if v_fields.iter().any(|&f| self.is_uninhabited(f, 1)) {
                continue;
            }
            let full = format!("{adt_name}::{v_name}");
            if let Some(w) = self.witness_product(
                rows,
                &CtorKind::Variant(item, v_idx),
                &v_fields,
                rest_tys,
                budget,
                |parts| format_ctor_witness(&full, Some(shape), parts),
            ) {
                return Some(w);
            }
        }
        if self.is_foreign_non_exhaustive(item) {
            return self.witness_infinite(rows, rest_tys, budget);
        }
        None
    }

    fn witness_slice(
        &mut self,
        rows: &[Vec<MPat>],
        elem: Ty,
        rest_tys: &[Ty],
        budget: &mut u32,
    ) -> Option<Vec<String>> {
        let mut min_rest: Option<usize> = None;
        let mut max_prefix = 0usize;
        let mut max_suffix = 0usize;
        for row in rows {
            match &row[0] {
                MPat::Wild => min_rest = Some(0),
                MPat::Slice {
                    prefix,
                    rest: true,
                    suffix,
                } => {
                    let fixed = prefix.len() + suffix.len();
                    min_rest = Some(min_rest.map_or(fixed, |cur| cur.min(fixed)));
                    max_prefix = max_prefix.max(prefix.len());
                    max_suffix = max_suffix.max(suffix.len());
                }
                _ => {}
            }
        }
        let Some(k) = min_rest else {
            let def = specialize_default(rows);
            if let Some(mut rest) = self.find_witness(&def, rest_tys, budget) {
                rest.push("[..]".to_owned());
                return Some(rest);
            }
            return None;
        };
        let split_len = k
            .max(max_prefix + max_suffix)
            .min(MAX_ARRAY_EXPAND as usize);
        for len in 0..split_len {
            let spec = specialize_slice_len(rows, len);
            let mut sub_tys = vec![elem; len];
            sub_tys.extend_from_slice(rest_tys);
            if let Some(mut w) = self.find_witness(&spec, &sub_tys, budget) {
                let split = w.len() - len;
                let mut parts: Vec<String> = w.drain(split..).collect();
                parts.reverse();
                w.push(format!("[{}]", parts.join(", ")));
                return Some(w);
            }
        }
        let open_cols = max_prefix + max_suffix;
        let open_rows = if open_cols <= MAX_ARRAY_EXPAND as usize {
            specialize_slice_open(rows, max_prefix, max_suffix, open_cols)
        } else {
            specialize_default(rows)
        };
        let mut open_tys = if open_cols <= MAX_ARRAY_EXPAND as usize {
            vec![elem; open_cols]
        } else {
            Vec::new()
        };
        open_tys.extend_from_slice(rest_tys);
        if let Some(mut w) = self.find_witness(&open_rows, &open_tys, budget) {
            if open_cols <= MAX_ARRAY_EXPAND as usize {
                let split = w.len() - open_cols;
                let mut parts: Vec<String> = w.drain(split..).collect();
                parts.reverse();
                let pref = &parts[..max_prefix.min(parts.len())];
                let suff = &parts[max_prefix.min(parts.len())..];
                let mut joined = pref.to_vec();
                joined.push("..".to_owned());
                joined.extend_from_slice(suff);
                w.push(format!("[{}]", joined.join(", ")));
            } else {
                w.push("[..]".to_owned());
            }
            return Some(w);
        }
        None
    }

    fn witness_infinite(
        &mut self,
        rows: &[Vec<MPat>],
        rest_tys: &[Ty],
        budget: &mut u32,
    ) -> Option<Vec<String>> {
        let def = specialize_default(rows);
        if let Some(mut rest) = self.find_witness(&def, rest_tys, budget) {
            rest.push("_".to_owned());
            return Some(rest);
        }
        None
    }
}

fn expand_col0_or(rows: &[Vec<MPat>]) -> Vec<Vec<MPat>> {
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        push_expanded_or(&row[0], &row[1..], &mut out);
    }
    out
}

fn push_expanded_or(head: &MPat, tail: &[MPat], out: &mut Vec<Vec<MPat>>) {
    if let MPat::Or(alts) = head {
        for alt in alts {
            push_expanded_or(alt, tail, out);
        }
    } else {
        let mut next = Vec::with_capacity(1 + tail.len());
        next.push(head.clone());
        next.extend_from_slice(tail);
        out.push(next);
    }
}

fn specialize_ctor(rows: &[Vec<MPat>], ctor: &CtorKind, arity: usize) -> Vec<Vec<MPat>> {
    let mut out = Vec::new();
    for row in rows {
        match &row[0] {
            MPat::Wild => {
                let mut next = vec![MPat::Wild; arity];
                next.extend_from_slice(&row[1..]);
                out.push(next);
            }
            MPat::Ctor(k, args) if k == ctor && args.len() == arity => {
                let mut next = args.clone();
                next.extend_from_slice(&row[1..]);
                out.push(next);
            }
            _ => {}
        }
    }
    out
}

fn specialize_default(rows: &[Vec<MPat>]) -> Vec<Vec<MPat>> {
    rows.iter()
        .filter(|row| matches!(row[0], MPat::Wild))
        .map(|row| row[1..].to_vec())
        .collect()
}

fn specialize_slice_len(rows: &[Vec<MPat>], len: usize) -> Vec<Vec<MPat>> {
    let mut out = Vec::new();
    for row in rows {
        match &row[0] {
            MPat::Wild => {
                let mut next = vec![MPat::Wild; len];
                next.extend_from_slice(&row[1..]);
                out.push(next);
            }
            MPat::Slice {
                prefix,
                rest,
                suffix,
            } => {
                let fixed = prefix.len() + suffix.len();
                if (!rest && fixed == len) || (*rest && fixed <= len) {
                    let mut next = prefix.clone();
                    next.extend(std::iter::repeat_n(MPat::Wild, len - fixed));
                    next.extend(suffix.iter().cloned());
                    next.extend_from_slice(&row[1..]);
                    out.push(next);
                }
            }
            _ => {}
        }
    }
    out
}

fn specialize_slice_open(
    rows: &[Vec<MPat>],
    max_prefix: usize,
    max_suffix: usize,
    open_cols: usize,
) -> Vec<Vec<MPat>> {
    let mut out = Vec::new();
    for row in rows {
        match &row[0] {
            MPat::Wild => {
                let mut next = vec![MPat::Wild; open_cols];
                next.extend_from_slice(&row[1..]);
                out.push(next);
            }
            MPat::Slice {
                prefix,
                rest: true,
                suffix,
            } if prefix.len() + suffix.len() <= open_cols => {
                let mut next = prefix.clone();
                next.extend(std::iter::repeat_n(
                    MPat::Wild,
                    max_prefix.saturating_sub(prefix.len()),
                ));
                next.extend(std::iter::repeat_n(
                    MPat::Wild,
                    max_suffix.saturating_sub(suffix.len()),
                ));
                next.extend(suffix.iter().cloned());
                next.truncate(open_cols);
                next.extend_from_slice(&row[1..]);
                out.push(next);
            }
            _ => {}
        }
    }
    out
}

fn format_int_witness(lo: u128, hi: u128, prim: Primitive) -> String {
    let s_lo = decode_key_to_string(lo, prim);
    if lo == hi {
        s_lo
    } else {
        let s_hi = decode_key_to_string(hi, prim);
        format!("{s_lo}..={s_hi}")
    }
}

fn format_ctor_witness(
    name: &str,
    shape: Option<crate::body::adt::Shape>,
    parts: &[String],
) -> String {
    match shape {
        Some(crate::body::adt::Shape::Unit) | None if parts.is_empty() => name.to_owned(),
        Some(crate::body::adt::Shape::Named) => format!("{name} {{ .. }}"),
        _ if parts.is_empty() => name.to_owned(),
        _ => format!("{name}({})", parts.join(", ")),
    }
}
