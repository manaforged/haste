use super::Emit;
use super::asm::{Vl, Xmm};
use super::float::{X0, X1, X2};
use crate::ir::{Binary, Convert, Lane, Op, Value};

/// `pmuludq`: the unsigned 64-bit product of the low 32 bits of each 64-bit lane.
pub(super) const PMULUDQ: &[u8] = &[0x0F, 0xF4];
const PADDQ: &[u8] = &[0x0F, 0xD4];
const PSHUFD: u8 = 0x70;
pub(super) const SRL: u8 = 2;
pub(super) const SRA: u8 = 4;
pub(super) const SLL: u8 = 6;
const HALF: u8 = 32;
const EVEN_DWORDS: u8 = 0xE8;
const FACT_DEPTH: u8 = 2;

impl Emit<'_> {
    /// Shifts every lane of `src` by `imm` into `dst`, where `ext` selects the shift.
    pub(super) fn pshift_into(&mut self, ext: u8, lane_bytes: u32, dst: Xmm, src: Xmm, imm: u8) {
        if self.features.avx {
            self.asm
                .vex_pshift_imm(Vl::X, ext, lane_bytes, dst, src, imm);
            return;
        }
        if dst != src {
            self.asm.movaps(dst, src);
        }
        self.asm.pshift_imm(ext, lane_bytes, dst, imm);
    }

    pub(super) fn lives(&self, dst: Xmm, vals: &[Value]) -> bool {
        vals.iter().any(|&v| self.xmm_of(v) == Some(dst))
    }

    pub(super) fn fins<const N: usize>(
        &mut self,
        dst: Xmm,
        tied: bool,
        vals: [Value; N],
        rest: &[Value],
    ) -> [Xmm; N] {
        debug_assert!(
            !tied || rest.is_empty(),
            "a tied operand overwrites dst before rest is read"
        );
        let mut free = (!tied && !self.lives(dst, &vals) && !self.lives(dst, rest)).then_some(dst);
        let mut spare = [X1, X0, X2].into_iter().filter(|&xmm| xmm != dst);
        let mut regs = [dst; N];
        for i in usize::from(tied)..N {
            let v = vals[i];
            regs[i] = if let Some(first) = vals[..i].iter().position(|&u| u == v) {
                regs[first]
            } else if let Some(home) = self.xmm_of(v).filter(|&xmm| !tied || xmm != dst) {
                home
            } else {
                let Some(to) = free.take().or_else(|| spare.next()) else {
                    unreachable!()
                };
                self.fget(to, v);
                to
            };
        }
        if tied {
            self.fget(dst, vals[0]);
        }
        regs
    }

    fn splat_const(&self, v: Value) -> Option<i64> {
        let (_, count) = self.ty(v).vec_shape()?;
        let scalar = self.splat_source(v, count)?;
        let inst = self.defs.get(scalar.index()).copied().flatten()?;
        match self.func.insts[inst.index()].op {
            Op::Const(imm) => Some(imm),
            _ => None,
        }
    }

    fn high_zero(&self, v: Value, depth: u8) -> bool {
        let Some(inst) = self.defs.get(v.index()).copied().flatten() else {
            return false;
        };
        match &self.func.insts[inst.index()].op {
            Op::Convert(Convert::Uextend, _) => true,
            Op::Binary(Binary::Band, a, b) => [*a, *b].into_iter().any(|side| {
                self.splat_const(side)
                    .is_some_and(|imm| imm as u64 <= u64::from(u32::MAX))
                    || (depth > 0 && self.high_zero(side, depth - 1))
            }),
            Op::Binary(Binary::Ushr, _, amount) => self
                .splat_const(*amount)
                .is_some_and(|imm| imm & 63 >= i64::from(HALF)),
            _ => false,
        }
    }

    /// Multiplies the 64-bit lanes of `x` and `y` into `dst`, wrapping.
    pub(super) fn imul_quads(&mut self, (x, y): (Value, Value), dst: Xmm) {
        if self.high_zero(x, FACT_DEPTH) && self.high_zero(y, FACT_DEPTH) {
            return self.vex_packed_regs(&[0x66], PMULUDQ, (x, y), dst, true);
        }
        let a = self.fin(x, X0);
        let b = self.fin(y, X1);
        self.pshift_into(SRL, 8, X2, a, HALF);
        self.sse_or_vex(&[0x66], PMULUDQ, (X2, X2, b));
        self.pshift_into(SRL, 8, X1, b, HALF);
        self.sse_or_vex(&[0x66], PMULUDQ, (X1, X1, a));
        self.sse_or_vex(&[0x66], PADDQ, (X2, X2, X1));
        self.pshift_into(SLL, 8, X2, X2, HALF);
        self.clobber_xmms(&[X1, X2]);
        let b = self.fin(y, X1);
        self.sse_or_vex(&[0x66], PMULUDQ, (X0, a, b));
        self.sse_or_vex(&[0x66], PADDQ, (dst, X2, X0));
        self.clobber_xmms(&[X0, X1]);
    }

    /// Truncates each `from` lane of `x` to its low half, packed into the low eight bytes of `dst`.
    pub(super) fn narrow(&mut self, from: Lane, x: Value, dst: Xmm) {
        let src = self.fin(x, dst);
        if from == Lane::I64 {
            if self.features.avx {
                self.asm
                    .vex_xmm_imm(0b00001, 0b01, PSHUFD, (dst, Xmm(0)), src, EVEN_DWORDS);
            } else {
                self.asm
                    .sse_imm(&[0x66], &[0x0F, PSHUFD], dst.0, src, EVEN_DWORDS);
            }
            return;
        }
        let half = (from.bits() / 2) as u8;
        self.pshift_into(SLL, from.bytes(), dst, src, half);
        self.pshift_into(SRA, from.bytes(), dst, dst, half);
        let pack = if from == Lane::I16 { 0x63 } else { 0x6B };
        self.sse_or_vex(&[0x66], &[0x0F, pack], (dst, dst, dst));
    }
}
