use anyhow::{Result, bail};
use haste_codegen::{Block, IntCC, Type, Value};

use crate::body::{BlockKind, Body, Expr, ExprId, List, Lit, Pat, PatId, UnOp};
use crate::body::literal::{byte_value, char_value};
use crate::names::Primitive;
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::infer::{Res, Typed, parse_int, position, rest_split};
use crate::sema::ty::{Kind, Ty, int_info, mask};

use super::equal::bitwise_prim_bytes;
use super::ops::WIDE;
use super::place::{Base, Place};
use super::{Lower, Repr, host_is_x64_or_arm64, repr};

fn bitwise_elem(cx: &mut Context<'_>, ty: Ty) -> Option<(Primitive, usize)> {
    let Kind::Prim(prim) = *cx.types.kind(ty) else { return None };
    let bytes = bitwise_prim_bytes(prim).filter(|&b| b <= 4)?;
    (cx.layout(ty).ok()?.size.bytes() == bytes as u64).then_some((prim, bytes))
}

fn wide_match_window(vals: &[Option<u64>], elem_bytes: usize) -> Option<(usize, Type, u64, u64, bool)> {
    let elem_bits = elem_bytes * 8;
    let elem_mask = u64::MAX >> (64 - elem_bits);
    for (w_bytes, load_ty) in [(8, Type::I64), (4, Type::I32), (2, Type::I16)] {
        let k = w_bytes / elem_bytes;
        if k < 2 || vals.len() < k {
            continue;
        }
        let m = vals[..k].iter().filter(|v| v.is_some()).count();
        let ok = m == k || (elem_bytes == 1 && ((w_bytes == 4 && m >= 3) || (w_bytes == 8 && m >= 5)));
        if !ok {
            continue;
        }
        let (mut exp, mut mask_val) = (0u64, 0u64);
        for (j, val) in vals[..k].iter().enumerate() {
            if let Some(v) = val {
                let shift = j * elem_bits;
                exp |= (v & elem_mask) << shift;
                mask_val |= elem_mask << shift;
            }
        }
        return Some((k, load_ty, exp, mask_val, m < k));
    }
    None
}

fn eval_lit_expr(cx: &Context<'_>, (body, src, typed): (&Body, &Source, &Typed), expr: ExprId, prim: Primitive) -> Option<u64> {
    if typed.coercions.contains_key(&expr) {
        return None;
    }
    match *body.expr(expr) {
        Expr::Paren(inner) => eval_lit_expr(cx, (body, src, typed), inner, prim),
        Expr::Block { stmts, tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Unsafe | BlockKind::Const } if body.list(stmts).is_empty() => {
            eval_lit_expr(cx, (body, src, typed), tail, prim)
        }
        Expr::Unary(UnOp::Neg, inner) if !typed.operators.contains_key(&expr) => {
            let (bits, true) = int_info(prim)? else { return None };
            let v = eval_lit_expr(cx, (body, src, typed), inner, prim)?;
            Some((u128::from(v).wrapping_neg() & mask(bits)) as u64)
        }
        _ => eval_lit_atom(cx, (body, src, typed), expr, prim),
    }
}

fn eval_lit_atom(cx: &Context<'_>, (body, src, typed): (&Body, &Source, &Typed), expr: ExprId, prim: Primitive) -> Option<u64> {
    match *body.expr(expr) {
        Expr::Lit(Lit::Bool(v), _) if prim == Primitive::Bool => Some(u64::from(v)),
        Expr::Lit(Lit::Byte, at) if prim == Primitive::U8 => byte_value(src.text(at)).ok().map(u64::from),
        Expr::Lit(Lit::Char, at) if prim == Primitive::Char => char_value(src.text(at)).ok().map(|c| u64::from(u32::from(c))),
        Expr::Lit(Lit::Int, at) => int_info(prim).and_then(|(bits, _)| parse_int(src.text(at)).ok().map(|(v, _)| (v & mask(bits)) as u64)),
        Expr::Path(_) => {
            let Some(&Res::Const(val_ty)) = typed.res.get(&expr) else { return None };
            let Kind::Value(v) = *cx.types.kind(val_ty) else { return None };
            let (bits, _) = int_info(prim)?;
            Some((v & mask(bits)) as u64)
        }
        _ => None,
    }
}

impl Lower<'_, '_> {
    /// The items of slice pattern `pat` that match an element of the array or slice at `place`, each with that element's place; branches to `fail` first when a slice's length cannot match.
    pub(super) fn slice_parts(&mut self, (pat, list): (PatId, List), place: Option<Place>, fail: Option<Block>) -> Result<Vec<(PatId, Option<Place>)>> {
        let ty = self.pat_ty(pat)?;
        let element = match *self.cx.types.kind(ty) {
            Kind::Array(element, _) | Kind::Slice(element) => element,
            _ => bail!("slice pattern on a type that is not an array or slice"),
        };
        let (pats, rest) = rest_split(self.body, list)?;
        let count = pats.len();
        let Some((base, slice_pair, array_len)) = self.slice_target(place, ty, (count, rest.is_some()), fail)? else {
            return Ok(Vec::new());
        };
        let stride = self.cx.layout(element)?.size.bytes();
        let elem_repr = repr(self.cx, element)?;
        let mut tested = vec![false; count];
        let mut suffix_base = None;
        if let Some(fail_blk) = fail {
            suffix_base = self.coalesce_slice((&pats, rest), (element, stride), (base, slice_pair, array_len), (&mut tested, fail_blk))?;
        }
        self.collect_slice_parts((&pats, rest), (element, elem_repr, stride), (base, slice_pair, array_len, suffix_base), (&tested, fail))
    }

    fn slice_target(&mut self, place: Option<Place>, ty: Ty, (count, has_rest): (usize, bool), fail: Option<Block>) -> Result<Option<(Place, Option<(Value, Value)>, Option<usize>)>> {
        if let Kind::Array(_, len_ty) = *self.cx.types.kind(ty) {
            let Some(at) = place else { return Ok(None) };
            return Ok(Some((at, None, Some(self.len(len_ty)? as usize))));
        }
        let Some((data, len)) = self.data_len(place, ty)? else { return Ok(None) };
        let min_len = count - usize::from(has_rest);
        if let Some(fail) = fail
            && (!has_rest || min_len > 0)
        {
            let given = self.ins().iconst(Type::I64, min_len as i64);
            let cc = if has_rest { IntCC::UnsignedGreaterThanOrEqual } else { IntCC::Equal };
            let ok = self.ins().icmp(cc, len, given);
            self.pass(ok, fail);
        }
        Ok(Some((Place { base: Base::Addr(data), offset: 0, repr: Repr::Zst }, Some((data, len)), None)))
    }

    fn coalesce_slice(
        &mut self,
        (pats, rest): (&[PatId], Option<usize>),
        (element, stride): (Ty, u64),
        (base, slice_pair, array_len): (Place, Option<(Value, Value)>, Option<usize>),
        (tested, fail): (&mut [bool], Block),
    ) -> Result<Option<Place>> {
        let Some((prim, elem_bytes)) = bitwise_elem(self.cx, element) else { return Ok(None) };
        if !host_is_x64_or_arm64() || stride != elem_bytes as u64 {
            return Ok(None);
        }
        let pre_end = rest.unwrap_or(pats.len());
        self.coalesce_run(&pats[..pre_end], (prim, elem_bytes), base, (&mut tested[..pre_end], fail));
        let Some(split) = rest.filter(|&s| s + 1 < pats.len()) else { return Ok(None) };
        let suffix = &pats[split + 1..];
        let suffix_base = if let Some(total) = array_len {
            let start_off = ((total - suffix.len()) as u64 * stride) as i32;
            base.at(start_off, Repr::Zst)
        } else if let Some((data, len)) = slice_pair
            && suffix.iter().filter(|&&p| self.pat_scalar_bits(p, prim).is_some()).count() >= 2
        {
            let idx = self.ins().iadd_imm(len, -(suffix.len() as i64));
            self.element_place(data, idx, element)?
        } else {
            return Ok(None);
        };
        self.coalesce_run(suffix, (prim, elem_bytes), suffix_base, (&mut tested[split + 1..], fail));
        Ok(slice_pair.map(|_| suffix_base))
    }

    fn coalesce_run(&mut self, pats: &[PatId], (prim, elem_bytes): (Primitive, usize), base: Place, (tested, fail): (&mut [bool], Block)) {
        let vals: Vec<Option<u64>> = pats.iter().map(|&p| self.pat_scalar_bits(p, prim)).collect();
        if vals.iter().filter(|v| v.is_some()).count() < 2 {
            return;
        }
        let mut i = 0;
        while i < pats.len() {
            let Some((k, load_ty, exp, mask_val, masked)) = vals[i].and_then(|_| wide_match_window(&vals[i..], elem_bytes)) else {
                i += 1;
                continue;
            };
            let Some(mut word) = self.read(base.at((i * elem_bytes) as i32, Repr::Scalar(load_ty))) else {
                i += 1;
                continue;
            };
            if masked {
                let mask_c = self.ins().iconst(load_ty, mask_val as i64);
                word = self.ins().band(word, mask_c);
            }
            let ok = self.ins().icmp_imm(IntCC::Equal, word, exp as i64);
            self.pass(ok, fail);
            for j in 0..k {
                if vals[i + j].is_some() {
                    tested[i + j] = true;
                }
            }
            i += k;
        }
    }

    fn pat_scalar_bits(&mut self, mut pat: PatId, prim: Primitive) -> Option<u64> {
        loop {
            if self.typed.pat_derefs(pat) != 0 {
                return None;
            }
            match *self.body.pat(pat) {
                Pat::Paren(inner) | Pat::Bind { sub: Some(inner), .. } if !self.typed.pat_res.contains_key(&pat) => pat = inner,
                Pat::Lit(expr) if !self.typed.pat_res.contains_key(&pat) => {
                    return eval_lit_expr(self.cx, (self.body, self.src, self.typed), expr, prim);
                }
                Pat::Path(_) | Pat::Bind { sub: None, .. } => {
                    let Some(&Res::Fn(item, substs)) = self.typed.pat_res.get(&pat) else { return None };
                    if self.cx.item(item).1.kind != ItemKind::Const {
                        return None;
                    }
                    let (source, const_typed) = self.literal_source(item, substs).ok()??;
                    let const_typed = const_typed.scoped(&self.cx.types);
                    if *self.cx.types.kind(const_typed.ret_ty()) != Kind::Prim(prim) {
                        return None;
                    }
                    let const_body = self.cx.body(source).ok()?;
                    let (Some(root), const_src) = (const_body.root, self.cx.body_src(source)) else { return None };
                    return eval_lit_expr(self.cx, (&const_body, const_src, &const_typed), root, prim);
                }
                _ => return None,
            }
        }
    }

    fn skip_slice_elem(&self, sub: PatId, testing: bool) -> bool {
        if !testing {
            return matches!(*self.body.pat(sub), Pat::Wild | Pat::Lit(..) | Pat::Range { .. });
        }
        if self.typed.pat_derefs(sub) != 0 || self.typed.pat_res.contains_key(&sub) {
            return false;
        }
        match *self.body.pat(sub) {
            Pat::Wild | Pat::Bind { sub: None, .. } => true,
            Pat::Bind { sub: Some(inner), .. } => matches!(*self.body.pat(inner), Pat::Rest),
            _ => false,
        }
    }

    fn collect_slice_parts(
        &mut self,
        (pats, rest): (&[PatId], Option<usize>),
        (element, elem_repr, stride): (Ty, Repr, u64),
        (base, slice_pair, array_len, mut suffix_base): (Place, Option<(Value, Value)>, Option<usize>, Option<Place>),
        (tested, fail): (&[bool], Option<Block>),
    ) -> Result<Vec<(PatId, Option<Place>)>> {
        let count = pats.len();
        let mut found = Vec::new();
        for (at, &sub) in pats.iter().enumerate() {
            if tested[at] || self.skip_slice_elem(sub, fail.is_some()) {
                continue;
            }
            if position(at, count, rest, count).is_none() {
                if !matches!(self.body.pat(sub), Pat::Rest) {
                    let len = slice_pair.map(|(_, len)| len);
                    found.push((sub, Some(self.rest_place(sub, base, len, (at, count - 1))?)));
                }
                continue;
            }
            let elem_place = self.slice_elem_place((at, pats, rest), (element, elem_repr, stride), (base, slice_pair, array_len, &mut suffix_base), (tested, fail.is_some()))?;
            found.push((sub, Some(elem_place)));
        }
        Ok(found)
    }

    fn slice_elem_place(
        &mut self,
        (at, pats, rest): (usize, &[PatId], Option<usize>),
        (element, elem_repr, stride): (Ty, Repr, u64),
        (base, slice_pair, array_len, suffix_base): (Place, Option<(Value, Value)>, Option<usize>, &mut Option<Place>),
        (tested, testing): (&[bool], bool),
    ) -> Result<Place> {
        let count = pats.len();
        if let (Some(split), None) = (rest, array_len)
            && at > split
        {
            let Some((data, len)) = slice_pair else { bail!("slice pattern missing slice data") };
            let kept = pats[split + 1..].iter().enumerate().filter(|&(j, &p)| !tested[split + 1 + j] && !self.skip_slice_elem(p, testing)).count();
            if suffix_base.is_none() && kept >= 2 {
                let idx = self.ins().iadd_imm(len, -((count - 1 - split) as i64));
                *suffix_base = Some(self.element_place(data, idx, element)?);
            }
            return match *suffix_base {
                Some(sbase) => Ok(sbase.at(((at - split - 1) as u64 * stride) as i32, elem_repr)),
                None => {
                    let idx = self.ins().iadd_imm(len, at as i64 - count as i64);
                    self.element_place(data, idx, element)
                }
            };
        }
        if let (Some(split), Some(total)) = (rest, array_len)
            && at > split
        {
            return Ok(base.at(((total - (count - at)) as u64 * stride) as i32, elem_repr));
        }
        match i32::try_from(at as u64 * stride) {
            Ok(off) => Ok(base.at(off, elem_repr)),
            Err(_) => {
                let data = self.address(base);
                let idx = self.ins().iconst(Type::I64, at as i64);
                self.element_place(data, idx, element)
            }
        }
    }

    fn rest_place(&mut self, sub: PatId, base: Place, len: Option<Value>, (split, given): (usize, usize)) -> Result<Place> {
        let ty = self.pat_ty(sub)?;
        let (Kind::Array(element, _) | Kind::Slice(element)) = *self.cx.types.kind(ty) else { bail!("the rest of a slice pattern is not an array or slice") };
        let stride = self.cx.layout(element)?.size.bytes() as i64;
        if let Kind::Array(..) = self.cx.types.kind(ty) {
            return Ok(base.at((stride * split as i64) as i32, repr(self.cx, ty)?));
        }
        let data = self.address(base);
        let start = if split == 0 { data } else { self.ins().iadd_imm(data, stride * split as i64) };
        let len = len.unwrap_or_else(|| self.ins().iconst(Type::I64, 0));
        let left = if given == 0 { len } else { self.ins().iadd_imm(len, -(given as i64)) };
        Ok(Place { base: Base::Addr(self.wide(start, left)), offset: 0, repr: WIDE })
    }
}
