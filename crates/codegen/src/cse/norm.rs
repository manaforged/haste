use super::Table;
use super::key::{Key, Operand, canonical_icmp, key, operand, order_pair};
use crate::fold::extend;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Unary, Value};
use crate::merge::find;

impl Table {
    fn norm_unary(
        &mut self,
        u: Unary,
        x: Value,
        types: &[Type],
        result: Value,
        subst: &[Value],
    ) -> Option<Result<Key, Value>> {
        let xv = find(subst, x);
        if let Some((prev_u, inner)) = self.unary_src[xv.index()]
            && prev_u == u
        {
            return Some(Err(find(subst, inner)));
        }
        self.unary_src[result.index()] = Some((u, xv));
        Some(Ok(Key::Unary(u, operand(&self.consts, types, subst, xv))))
    }

    fn norm_add_sub(
        &mut self,
        op: &Op,
        bin: Binary,
        (x, y): (Value, Value),
        types: &[Type],
        ty: Type,
        result: Value,
        subst: &[Value],
    ) -> Option<Result<Key, Value>> {
        let ox = operand(&self.consts, types, subst, x);
        let oy = operand(&self.consts, types, subst, y);
        let val_imm = match (bin, ox, oy) {
            (Binary::Iadd, Operand::Val(base), Operand::Imm(_, imm))
            | (Binary::Iadd, Operand::Imm(_, imm), Operand::Val(base)) => Some((base, imm)),
            (Binary::Isub, Operand::Val(base), Operand::Imm(_, imm)) => {
                Some((base, imm.wrapping_neg()))
            }
            (Binary::Isub, Operand::Imm(_, 1), Operand::Val(v)) if self.bool_val[v.index()] => {
                let (a, b) = order_pair(Operand::Val(v), Operand::Imm(ty, 1));
                return Some(Ok(Key::Binary(Binary::Bxor, a, b)));
            }
            _ => None,
        };
        if let Some((base, delta)) = val_imm {
            let delta = extend(delta, ty, false);
            if delta == 0 {
                return Some(Err(base));
            }
            if ty == Type::I64
                && let Some((slot, base_off)) =
                    self.stack_addrs.get(base.index()).copied().flatten()
                && let Some(total) = i32::try_from(delta as i64)
                    .ok()
                    .and_then(|d| base_off.checked_add(d))
            {
                self.stack_addrs[result.index()] = Some((slot, total));
                return Some(Ok(Key::StackAddr(slot, total)));
            }
            let (root, base_off) = self.affine[base.index()];
            let root = find(subst, root);
            let total = extend(base_off.wrapping_add(delta), ty, false);
            self.affine[result.index()] = (root, total);
            if total == 0 {
                return Some(Err(root));
            }
            let (a, b) = order_pair(Operand::Val(root), Operand::Imm(ty, total));
            return Some(Ok(Key::Binary(Binary::Iadd, a, b)));
        }
        key(
            op,
            (&self.consts, &self.bool_neg, self.const_epoch),
            types,
            ty,
            subst,
        )
        .map(Ok)
    }

    fn norm_assoc_bin(
        &mut self,
        op: &Op,
        bin: Binary,
        (x, y): (Value, Value),
        types: &[Type],
        ty: Type,
        result: Value,
        subst: &[Value],
    ) -> Option<Result<Key, Value>> {
        let ox = operand(&self.consts, types, subst, x);
        let oy = operand(&self.consts, types, subst, y);
        let val_imm = match (ox, oy) {
            (Operand::Val(v), Operand::Imm(_, imm)) | (Operand::Imm(_, imm), Operand::Val(v)) => {
                Some((v, imm))
            }
            _ => None,
        };
        let Some((v, raw_imm)) = val_imm else {
            return key(
                op,
                (&self.consts, &self.bool_neg, self.const_epoch),
                types,
                ty,
                subst,
            )
            .map(Ok);
        };
        let imm = extend(raw_imm, ty, false);
        let all_ones = extend(-1, ty, false);
        let identity_val = match bin {
            Binary::Band => all_ones,
            Binary::Imul => 1,
            _ => 0,
        };
        let (root, cur_imm) = match self.assoc[v.index()] {
            Some((prev_bin, r, c)) if prev_bin == bin => (find(subst, r), c),
            _ => (v, identity_val),
        };
        let combined = match bin {
            Binary::Band => extend(cur_imm & imm, ty, false),
            Binary::Bor => extend(cur_imm | imm, ty, false),
            Binary::Bxor => extend(cur_imm ^ imm, ty, false),
            _ => extend(cur_imm.wrapping_mul(imm), ty, false),
        };
        if combined == identity_val
            || (bin == Binary::Band && self.bool_val[root.index()] && (combined & 1) == 1)
        {
            return Some(Err(root));
        }
        let known_const = if combined == 0 && matches!(bin, Binary::Band | Binary::Imul) {
            Some(0)
        } else if bin == Binary::Band && self.bool_val[root.index()] && (combined & 1) == 0 {
            Some(0)
        } else if bin == Binary::Bor && combined == all_ones {
            Some(all_ones)
        } else {
            None
        };
        if let Some(cval) = known_const
            && let Some(&existing) = self.known.get(&Key::Const(self.const_epoch, ty, cval))
        {
            return Some(Err(existing));
        }
        self.assoc[result.index()] = Some((bin, root, combined));
        let (a, b) = order_pair(Operand::Val(root), Operand::Imm(ty, combined));
        Some(Ok(Key::Binary(bin, a, b)))
    }

    fn norm_convert(
        &mut self,
        conv: Convert,
        x: Value,
        types: &[Type],
        ty: Type,
        result: Value,
        subst: &[Value],
    ) -> Option<Result<Key, Value>> {
        let xv = find(subst, x);
        let x_ty = types[xv.index()];
        if matches!(
            conv,
            Convert::Bitcast | Convert::Uextend | Convert::Sextend | Convert::Ireduce
        ) && ty == x_ty
        {
            return Some(Err(xv));
        }
        if let Some((prev_conv, inner, inner_ty)) = self.conv_src[xv.index()] {
            let inner = find(subst, inner);
            match (prev_conv, conv) {
                (Convert::Bitcast, Convert::Bitcast) if ty == inner_ty => return Some(Err(inner)),
                (Convert::Bitcast, Convert::Bitcast) => {
                    self.conv_src[result.index()] = Some((Convert::Bitcast, inner, inner_ty));
                    return Some(Ok(Key::Convert(
                        Convert::Bitcast,
                        ty,
                        operand(&self.consts, types, subst, inner),
                    )));
                }
                (Convert::Uextend, Convert::Uextend | Convert::Sextend) => {
                    self.conv_src[result.index()] = Some((Convert::Uextend, inner, inner_ty));
                    return Some(Ok(Key::Convert(
                        Convert::Uextend,
                        ty,
                        operand(&self.consts, types, subst, inner),
                    )));
                }
                (Convert::Sextend, Convert::Sextend) => {
                    self.conv_src[result.index()] = Some((Convert::Sextend, inner, inner_ty));
                    return Some(Ok(Key::Convert(
                        Convert::Sextend,
                        ty,
                        operand(&self.consts, types, subst, inner),
                    )));
                }
                (Convert::Ireduce, Convert::Ireduce) => {
                    self.conv_src[result.index()] = Some((Convert::Ireduce, inner, inner_ty));
                    return Some(Ok(Key::Convert(
                        Convert::Ireduce,
                        ty,
                        operand(&self.consts, types, subst, inner),
                    )));
                }
                (Convert::Uextend | Convert::Sextend, Convert::Ireduce) => {
                    if ty == inner_ty {
                        return Some(Err(inner));
                    }
                    let eff = if ty.bytes() < inner_ty.bytes() {
                        Convert::Ireduce
                    } else {
                        prev_conv
                    };
                    self.conv_src[result.index()] = Some((eff, inner, inner_ty));
                    return Some(Ok(Key::Convert(
                        eff,
                        ty,
                        operand(&self.consts, types, subst, inner),
                    )));
                }
                (Convert::Ireduce, Convert::Uextend)
                    if ty == inner_ty && matches!(x_ty, Type::I8 | Type::I16 | Type::I32) =>
                {
                    let mask = x_ty.unsigned_max() as i64;
                    self.assoc[result.index()] = Some((Binary::Band, inner, mask));
                    let (a, b) = order_pair(
                        operand(&self.consts, types, subst, inner),
                        Operand::Imm(ty, mask),
                    );
                    return Some(Ok(Key::Binary(Binary::Band, a, b)));
                }
                _ => {}
            }
        }
        self.conv_src[result.index()] = Some((conv, xv, x_ty));
        let eff_conv = if conv == Convert::Sextend && self.bool_val[xv.index()] {
            Convert::Uextend
        } else {
            conv
        };
        Some(Ok(Key::Convert(
            eff_conv,
            ty,
            operand(&self.consts, types, subst, xv),
        )))
    }

    fn norm_icmp(
        &self,
        cc: IntCC,
        x: Value,
        y: Value,
        types: &[Type],
        ty: Type,
        subst: &[Value],
    ) -> Option<Result<Key, Value>> {
        let x_ty = types[find(subst, x).index()];
        let (cc, a, b) = canonical_icmp(
            cc,
            operand(&self.consts, types, subst, x),
            operand(&self.consts, types, subst, y),
        );
        if a == b {
            let cval = i64::from(crate::fold::holds(cc, x_ty, 0, 0));
            if let Some(&existing) = self.known.get(&Key::Const(self.const_epoch, ty, cval)) {
                return Some(Err(existing));
            }
        }
        if let (Operand::Val(v), Operand::Imm(ity, imm)) = (a, b)
            && self.bool_val[v.index()]
        {
            if (cc == IntCC::NotEqual && imm == 0) || (cc == IntCC::Equal && imm == 1) {
                if ity == Type::I8 {
                    return Some(Err(v));
                }
                if let Some((Convert::Uextend, inner, Type::I8)) =
                    self.conv_src.get(v.index()).copied().flatten()
                {
                    return Some(Err(find(subst, inner)));
                }
            }
            if ity == Type::I8
                && ((cc == IntCC::Equal && imm == 0) || (cc == IntCC::NotEqual && imm == 1))
            {
                let (pa, pb) = order_pair(Operand::Val(v), Operand::Imm(Type::I8, 1));
                return Some(Ok(Key::Binary(Binary::Bxor, pa, pb)));
            }
        }
        Some(Ok(Key::Icmp(cc, x_ty, a, b)))
    }

    fn norm_select(
        &mut self,
        c: Value,
        x: Value,
        y: Value,
        types: &[Type],
        result: Value,
        subst: &[Value],
    ) -> Option<Result<Key, Value>> {
        let (cond, sx, sy, _) = self.simplify_select(c, x, y, types, result, subst);
        if sx == sy {
            return Some(Err(sx));
        }
        if !types[cond.index()].is_vector()
            && let Some(imm) = self.consts[cond.index()]
        {
            return Some(Err(if imm == 0 { sy } else { sx }));
        }
        Some(Ok(Key::Select(
            operand(&self.consts, types, subst, cond),
            operand(&self.consts, types, subst, sx),
            operand(&self.consts, types, subst, sy),
        )))
    }

    pub(super) fn normalized_key(
        &mut self,
        op: &Op,
        types: &[Type],
        ty: Type,
        result: Value,
        subst: &[Value],
    ) -> Option<Result<Key, Value>> {
        match *op {
            Op::StackAddr(slot, offset) => {
                self.stack_addrs[result.index()] = Some((slot, offset));
                Some(Ok(Key::StackAddr(slot, offset)))
            }
            Op::Unary(u @ (Unary::Ineg | Unary::Bnot | Unary::Bswap), x) => {
                self.norm_unary(u, x, types, result, subst)
            }
            Op::Binary(
                Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr,
                x,
                y,
            ) => {
                if let Operand::Imm(_, 0) = operand(&self.consts, types, subst, y) {
                    return Some(Err(find(subst, x)));
                }
                key(
                    op,
                    (&self.consts, &self.bool_neg, self.const_epoch),
                    types,
                    ty,
                    subst,
                )
                .map(Ok)
            }
            Op::Binary(bin @ (Binary::Iadd | Binary::Isub), x, y) if ty.is_small_int() => {
                self.norm_add_sub(op, bin, (x, y), types, ty, result, subst)
            }
            Op::Binary(bin @ (Binary::Band | Binary::Bor | Binary::Bxor | Binary::Imul), x, y)
                if ty.is_small_int() =>
            {
                self.norm_assoc_bin(op, bin, (x, y), types, ty, result, subst)
            }
            Op::Convert(conv, x) => self.norm_convert(conv, x, types, ty, result, subst),
            Op::Icmp(cc, x, y) => self.norm_icmp(cc, x, y, types, ty, subst),
            Op::Select(c, x, y) => self.norm_select(c, x, y, types, result, subst),
            _ => key(
                op,
                (&self.consts, &self.bool_neg, self.const_epoch),
                types,
                ty,
                subst,
            )
            .map(Ok),
        }
    }
}
