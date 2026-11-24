use super::super::fuse::shrn_shift;
use super::super::{A, C, Emit, V0};
use super::VecSpec;
use crate::ir::{Binary, Convert, Op, Type, Value};

impl Emit<'_> {
    pub(in crate::arm64) fn vec_convert(
        &mut self,
        op: Convert,
        x: Value,
        result: Value,
    ) -> Result<(), String> {
        let (from, to) = (self.ty(x), self.ty(result));
        if op == Convert::Bitcast {
            return self.vec_bitcast(from, to, x, result);
        }
        let fspec = VecSpec::of(from)?;
        let tspec = VecSpec::of(to)?;
        let dst = self.fdst(result);
        if op == Convert::Ireduce
            && self.plan.elided[x.index()]
            && let Some(inst) = self.defs[x.index()]
            && let Op::Binary(Binary::Ushr | Binary::Sshr, val, amt) =
                self.func.insts[inst.index()].op
            && let Some(imm) = self.splat_const_lane(amt)
            && let Some(shift) = shrn_shift(from, to, imm)
        {
            let src = self.fin(val, V0);
            let enc_imm = 2 * tspec.lane.bits() - shift;
            self.v_two(0x0F00_8400 | (enc_imm << 16), dst, src);
            self.fput(result, dst);
            return Ok(());
        }
        let src = self.fin(x, V0);
        match op {
            Convert::Fpromote => self.v_two(0x0E61_7800, dst, src),
            Convert::Fdemote => self.v_two(0x0E61_6800, dst, src),
            Convert::Sextend | Convert::Uextend => {
                let u = u32::from(op == Convert::Uextend);
                self.v_two(
                    0x0F00_A400 | (u << 29) | ((1 << fspec.size) << 19),
                    dst,
                    src,
                );
            }
            Convert::Ireduce => self.v_two(0x0E21_2800 | (tspec.size << 22), dst, src),
            Convert::NarrowSat { signed } => {
                let base = if signed { 0x0E21_4800 } else { 0x2E21_2800 };
                self.v_two(base | (tspec.size << 22), dst, src);
            }
            Convert::FromSint | Convert::FromUint => {
                let u = u32::from(op == Convert::FromUint);
                if fspec.size == tspec.size {
                    let base = if tspec.is_1d() {
                        0x5E61_D800 | (u << 29)
                    } else {
                        0x0E21_D800 | (tspec.q << 30) | (u << 29) | (tspec.fsz() << 22)
                    };
                    self.v_two(base, dst, src);
                } else if fspec.size < tspec.size {
                    self.v_two(
                        0x0F00_A400 | (u << 29) | ((1 << fspec.size) << 19),
                        dst,
                        src,
                    );
                    self.v_two(
                        0x0E21_D800 | (tspec.q << 30) | (u << 29) | (tspec.fsz() << 22),
                        dst,
                        dst,
                    );
                } else {
                    self.v_two(
                        0x0E21_D800 | (fspec.q << 30) | (u << 29) | (1 << 22),
                        dst,
                        src,
                    );
                    self.v_two(0x0E61_6800, dst, dst);
                }
            }
            Convert::ToSintSat | Convert::ToUintSat => {
                let u = u32::from(op == Convert::ToUintSat);
                let base = if fspec.is_1d() {
                    0x5EE1_B800 | (u << 29)
                } else {
                    0x0EA1_B800 | (fspec.q << 30) | (u << 29) | (fspec.fsz() << 22)
                };
                self.v_two(base, dst, src);
            }
            other => return Err(format!("{other:?} from {from:?} to {to:?}")),
        }
        self.fput(result, dst);
        Ok(())
    }

    fn vec_bitcast(&mut self, from: Type, to: Type, x: Value, result: Value) -> Result<(), String> {
        if from.bytes() != to.bytes() {
            return Err(format!("bitcast from {from:?} to {to:?}"));
        }
        if to.is_vector() {
            let dst = self.fdst(result);
            match from {
                Type::Vec { .. } | Type::F64 => {
                    let src = self.fin(x, dst);
                    self.asm.vmov(dst, src);
                }
                Type::I64 => self.emit_vec_from_i64(dst, x),
                Type::I128 => {
                    self.raw(A, x);
                    self.high(C, x);
                    self.asm.fmov_in(true, dst, A);
                    self.ins_gpr(dst, 3, 1, C);
                }
                _ => return Err(format!("bitcast from {from:?} to {to:?}")),
            }
            self.fput(result, dst);
            return Ok(());
        }
        let src = self.fin(x, V0);
        match to {
            Type::F64 => {
                let dst = self.fdst(result);
                self.asm.fmov_reg(true, dst, src);
                self.fput(result, dst);
            }
            Type::I64 => {
                let dst = self.dst(result);
                self.asm.fmov_out(true, dst, src);
                self.clobber_reg(dst);
                self.put(result, dst);
            }
            Type::I128 => {
                self.asm.fmov_out(true, A, src);
                self.clobber_reg(A);
                self.put(result, A);
                self.asm.umov_d1(A, src);
                self.clobber_reg(A);
                self.put_high(result, A);
            }
            _ => return Err(format!("bitcast from {from:?} to {to:?}")),
        }
        Ok(())
    }
}
