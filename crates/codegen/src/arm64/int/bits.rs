use super::super::asm::{Alu, Cond, Reg, Size, ZR};
use super::super::fuse::{Bfx, unwrap_elided_tst};
use super::{A, C, Emit, add_imm_parts, cond, encode_bitmask_imm, swap_cond};
use crate::alloc::{Location, Plan};
use crate::frame::Ext;
use crate::ir::{Binary, Convert, Function, IntCC, Op, Type, Value};

impl Emit<'_> {
    pub(super) fn emit_bfm(&mut self, bfx: Bfx, ty: Type, dst: Reg) {
        let size = if ty == Type::I64 { Size::X } else { Size::W };
        let reg_bits: u8 = if size == Size::X { 64 } else { 32 };
        let reg = self.bits(self.peel_uextend(bfx.src), A);
        if !bfx.insert && bfx.lsb == 0 && bfx.width == reg_bits {
            if size == Size::W {
                self.asm.extend(false, 4, dst, reg);
            } else {
                self.asm.mov(dst, reg);
            }
            return;
        }
        let (immr, imms) = if bfx.insert {
            ((reg_bits - bfx.lsb) % reg_bits, bfx.width - 1)
        } else {
            (bfx.lsb, bfx.lsb + bfx.width - 1)
        };
        let base = match (bfx.signed, size == Size::X) {
            (false, false) => 0x5300_0000,
            (true, false) => 0x1300_0000,
            (false, true) => 0xD340_0000,
            (true, true) => 0x9340_0000,
        };
        self.asm.emit_gp(dst, base | u32::from(immr) << 16 | u32::from(imms) << 10 | u32::from(reg.0) << 5 | u32::from(dst.0));
    }

    pub(in crate::arm64) fn emit_tst(&mut self, x: Value, y: Value) {
        let ty = self.ty(x);
        let bits = ty.bits();
        let size = if bits >= 64 { Size::X } else { Size::W };
        let ty_mask = if bits == 64 { u64::MAX } else { (1_u64 << bits) - 1 };
        let imm_pair = match (self.const_imm(y), self.const_imm(x)) {
            (Some(imm), _) => Some((x, (imm as u64) & ty_mask)),
            (None, Some(imm)) => Some((y, (imm as u64) & ty_mask)),
            (None, None) => None,
        };
        if let Some((raw_val, raw_imm)) = imm_pair
            && let Some((val, imm)) = unwrap_elided_tst(self.func, &self.plan, raw_val, raw_imm, &|v| self.defs[v.index()])
        {
            let left = self.bits(val, A);
            if let Some(bitmask) = encode_bitmask_imm(imm, size == Size::X) {
                self.asm.alu_bitmask(Alu::Ands, size, bitmask, ZR, left);
            } else {
                self.asm.mov_imm(C, imm as i64);
                self.asm.alu(Alu::Ands, size, ZR, left, C);
                self.clobber_reg(C);
            }
            return;
        }
        let (left, right) = if bits >= 32 { self.bits_pair(x, y) } else { self.src_pair(x, y, Ext::Zero) };
        self.asm.alu(Alu::Ands, size, ZR, left, right);
    }

    pub(super) fn cmp_high_bits(&mut self, cc: IntCC, (x, y): (Value, Value), size: Size) -> Option<Cond> {
        let (val, raw, c) = match (self.plan.const_of(y), self.plan.const_of(x)) {
            (Some(raw), _) => (x, raw, cond(cc)),
            (None, Some(raw)) => (y, raw, swap_cond(cond(cc))),
            (None, None) => return None,
        };
        if self.plan.elided[val.index()] {
            return None;
        }
        let ty = self.ty(val);
        let ty_mask = ty.unsigned_max();
        let imm = (raw as u64) & ty_mask;
        let (low, hit) = match c {
            Cond::Hi => (imm, Cond::Ne),
            Cond::Ls => (imm, Cond::Eq),
            Cond::Hs => (imm.wrapping_sub(1), Cond::Ne),
            Cond::Lo => (imm.wrapping_sub(1), Cond::Eq),
            _ => return None,
        };
        if low == 0 || low & low.wrapping_add(1) != 0 {
            return None;
        }
        let bitmask = encode_bitmask_imm(!low & ty_mask, size == Size::X)?;
        let reg = self.cmp_arg(val, A, ty.bytes(), false);
        self.asm.alu_bitmask(Alu::Ands, size, bitmask, ZR, reg);
        self.last_cmp = None;
        Some(hit)
    }

    pub(super) fn is_zext_sub_cmp(&self, lhs: Value, cc: IntCC, raw_imm: i64) -> bool {
        let ty = self.ty(lhs);
        if ty.bytes() >= 4 || cc.is_signed() {
            return false;
        }
        let Some(inst) = self.defs[lhs.index()] else { return false };
        let Op::Binary(Binary::Isub, base, c_val) = self.func.insts[inst.index()].op else { return false };
        let Some(raw_c) = self.plan.const_of(c_val) else { return false };
        let simm_c = crate::fold::extend(raw_c, ty, true);
        if simm_c <= 0 || !self.is_zext_reg(base) {
            return false;
        }
        let wrap_floor = (1_u64 << ty.bits()) - (simm_c as u64);
        let u_imm = crate::fold::extend(raw_imm, ty, false) as u64;
        if matches!(cc, IntCC::UnsignedLessThan | IntCC::UnsignedGreaterThanOrEqual) {
            u_imm <= wrap_floor
        } else {
            u_imm < wrap_floor
        }
    }

    fn sload_op(&self, val: Value) -> Option<Op> {
        match self.func.insts[self.defs[val.index()]?.index()].op {
            Op::Load(addr, off) => Some(Op::Load(addr, off)),
            Op::StackLoad(slot, off) => Some(Op::StackLoad(slot, off)),
            _ => None,
        }
    }

    pub(super) fn cmp_sload(&mut self, cc: IntCC, x: Value, y: Value) -> Option<Cond> {
        let ty = self.ty(x);
        if !cc.is_signed() || !matches!(ty, Type::I8 | Type::I16) {
            return None;
        }
        let (load, other, swapped) = [(x, y, false), (y, x, true)]
            .into_iter()
            .find(|&(l, o, _)| self.plan.elided[l.index()] && !self.plan.elided[o.index()])?;
        let op = self.sload_op(load)?;
        if let Some((sub, imm)) = self.plan.const_of(other).and_then(|v| add_imm_parts(true, crate::fold::extend(v, ty, true))) {
            self.load_signed_reg(A, ty, &op);
            self.asm.add_imm(sub, true, Size::W, ZR, A, imm);
        } else if matches!(self.plan.location(other), Location::Reg(_)) {
            self.load_signed_reg(A, ty, &op);
            let right = self.src(other, C, Ext::Sign);
            self.asm.alu(Alu::Subs, Size::W, ZR, A, right);
        } else {
            return None;
        }
        self.last_cmp = None;
        Some(if swapped { swap_cond(cond(cc)) } else { cond(cc) })
    }

    pub(super) fn emit_extend_conv(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        let dst_ty = self.ty(result);
        let dst = self.dst(result);
        if self.plan.elided[x.index()]
            && let Some(bfx) = super::super::fuse::conv_bfx_parts(self.func, &self.plan, (dst_ty, op), x, &|v| self.defs[v.index()])
        {
            self.emit_bfm(bfx, dst_ty, dst);
            self.put(result, dst);
            return Ok(());
        }
        if op == Convert::Sextend
            && dst_ty != Type::I128
            && self.plan.elided[x.index()]
            && let Some(load_op) = self.sload_op(x)
        {
            self.load_signed(result, self.ty(x), &load_op);
            return Ok(());
        }
        let signed = op == Convert::Sextend;
        self.get(dst, x, Ext::of(signed));
        self.put(result, dst);
        if dst_ty == Type::I128 && signed {
            self.asm.asr_imm(C, dst, 63);
            self.put_high(result, C);
            self.clobber_reg(C);
        } else if dst_ty == Type::I128 {
            self.put_high(result, ZR);
        }
        Ok(())
    }

    fn elided_node(&self, val: Value) -> Option<&Op> {
        (self.plan.elided[val.index()] && !self.plan.fused[val.index()])
            .then(|| self.defs[val.index()])
            .flatten()
            .map(|inst| &self.func.insts[inst.index()].op)
    }

    fn emit_ccmp_leaf(&mut self, prev: Cond, op: Binary, (cc, a, b): (IntCC, Value, Value)) -> Cond {
        let (guard, is_or) = match op {
            Binary::Band => (prev, false),
            _ => (prev.invert(), true),
        };
        let ty = self.ty(a);
        let (bytes, signed) = (ty.bytes(), cc.is_signed());
        let size = if bytes >= 8 { Size::X } else { Size::W };
        let imm_pick = self
            .plan
            .const_of(b)
            .and_then(|v| ccmp_imm_parts(ty, cc, v))
            .map(|p| (a, p, cond(cc)))
            .or_else(|| self.plan.const_of(a).and_then(|v| ccmp_imm_parts(ty, cc, v)).map(|p| (b, p, swap_cond(cond(cc)))));
        let out = imm_pick.map_or_else(|| cond(cc), |(_, _, c)| c);
        let nzcv = nzcv_for_false(if is_or { out.invert() } else { out });
        if let Some((lhs, (neg, imm5), _)) = imm_pick {
            let rn = self.cmp_arg(lhs, A, bytes, signed);
            self.asm.ccmp(neg, size, true, (guard, rn, imm5, nzcv));
        } else {
            let (left, right) = if bytes >= 4 { self.bits_pair(a, b) } else { self.pair_opt(a, b, signed) };
            self.asm.ccmp(false, size, false, (guard, left, right.0, nzcv));
        }
        out
    }

    pub(super) fn try_emit_ccmp(&mut self, op: Binary, (x, y): (Value, Value), result: Value) -> bool {
        let Some(chain) = self.plan.ccmp_chain(self.func, (op, x, y, ()), |(), v| self.elided_node(v).map(|node| (node, ()))) else {
            return false;
        };
        let Some((head, tails)) = chain.leaves.split_first() else { return false };
        let mut cur = self.compare(head.cc, head.a, head.b);
        for t in tails {
            cur = self.emit_ccmp_leaf(cur, t.op, (t.cc, t.a, t.b));
        }
        self.last_cmp = None;
        self.set_flag(result, cur);
        true
    }
}

const fn nzcv_for_false(c: Cond) -> u8 {
    match c {
        Cond::Ne | Cond::Gt => 4,
        Cond::Lo | Cond::Ls => 2,
        Cond::Pl => 8,
        Cond::Vc | Cond::Ge => 1,
        _ => 0,
    }
}

fn ccmp_imm_parts(ty: Type, cc: IntCC, raw: i64) -> Option<(bool, u8)> {
    let signed = cc.is_signed();
    if signed || matches!(cc, IntCC::Equal | IntCC::NotEqual) {
        let simm = crate::fold::extend(raw, ty, true);
        if (0..=31).contains(&simm) {
            return Some((false, simm as u8));
        }
        if (-31..=-1).contains(&simm) && (signed || ty.bytes() >= 4) {
            return Some((true, (-simm) as u8));
        }
        return None;
    }
    let uimm = (raw as u64) & ty.unsigned_max();
    (uimm <= 31).then_some((false, uimm as u8))
}

fn cmp_imm_fits(ty: Type, cc: IntCC, raw: i64) -> bool {
    let fits = |signed: bool| add_imm_parts(true, crate::fold::extend(raw, ty, signed)).is_some();
    fits(cc.is_signed()) || (matches!(cc, IntCC::Equal | IntCC::NotEqual) && fits(true))
}

const MAX_CCMP_LEAVES: usize = 8;

#[derive(Clone, Copy)]
pub(in crate::arm64) struct CcmpLeaf<C> {
    pub(in crate::arm64) op: Binary,
    pub(in crate::arm64) cc: IntCC,
    pub(in crate::arm64) a: Value,
    pub(in crate::arm64) b: Value,
    pub(in crate::arm64) val: Value,
    pub(in crate::arm64) at: C,
}

pub(in crate::arm64) struct CcmpChain<C> {
    pub(in crate::arm64) leaves: Vec<CcmpLeaf<C>>,
    pub(in crate::arm64) joins: Vec<Value>,
}

impl Plan<Reg> {
    pub(in crate::arm64) fn plain_pair(&self, a: Value, b: Value) -> bool {
        !self.elided[a.index()] && !self.elided[b.index()]
    }

    fn ccmp_leaf<C>(&self, func: &Function, op: Binary, val: Value, (node, at): (&Op, C)) -> Option<CcmpLeaf<C>> {
        let &Op::Icmp(cc, a, b) = node else { return None };
        func.ty(a).is_small_int().then_some(CcmpLeaf { op, cc, a, b, val, at })
    }

    fn ccmp_head_gain<C>(&self, func: &Function, leaf: &CcmpLeaf<C>) -> bool {
        let ty = func.ty(leaf.a);
        let consts = [leaf.b, leaf.a].map(|v| self.const_of(v));
        let tail_imm = consts.iter().flatten().any(|&raw| ccmp_imm_parts(ty, leaf.cc, raw).is_some());
        !tail_imm && consts.iter().flatten().any(|&raw| cmp_imm_fits(ty, leaf.cc, raw))
    }

    fn ccmp_head_first<C: Copy>(&self, func: &Function, mut chain: CcmpChain<C>) -> CcmpChain<C> {
        let first = chain.leaves[0];
        if !self.plain_pair(first.a, first.b) || self.ccmp_head_gain(func, &first) {
            return chain;
        }
        if let Some(k) = chain.leaves[1..].iter().take_while(|l| l.op == first.op).position(|l| self.ccmp_head_gain(func, l)) {
            chain.leaves.swap(0, k + 1);
        }
        chain
    }

    pub(in crate::arm64) fn ccmp_chain<'f, C: Copy>(
        &self,
        func: &'f Function,
        (mut op, mut x, mut y, mut at): (Binary, Value, Value, C),
        node: impl Fn(C, Value) -> Option<(&'f Op, C)>,
    ) -> Option<CcmpChain<C>> {
        let (mut leaves, mut joins) = (Vec::new(), Vec::new());
        while leaves.len() + 1 < MAX_CCMP_LEAVES {
            let (nx, ny) = (node(at, x)?, node(at, y)?);
            let (lx, ly) = (self.ccmp_leaf(func, op, x, nx), self.ccmp_leaf(func, op, y, ny));
            let plain = |l: &Option<CcmpLeaf<C>>| l.as_ref().is_some_and(|l| self.plain_pair(l.a, l.b));
            if let (Some(hx), Some(hy)) = (lx, ly) {
                let pair = if plain(&ly) { [hy, hx] } else if plain(&lx) { [hx, hy] } else { return None };
                leaves.extend(pair);
                leaves.reverse();
                return Some(self.ccmp_head_first(func, CcmpChain { leaves, joins }));
            }
            let (tail, next_val, next) = match (lx, ly) {
                (_, Some(t)) if plain(&ly) => (t, x, nx),
                (Some(t), _) if plain(&lx) => (t, y, ny),
                _ => return None,
            };
            let (&Op::Binary(next_op @ (Binary::Band | Binary::Bor), nx2, ny2), next_at) = next else { return None };
            leaves.push(tail);
            joins.push(next_val);
            (op, x, y, at) = (next_op, nx2, ny2, next_at);
        }
        None
    }
}
