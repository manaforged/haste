use anyhow::Result;
use haste_codegen::{Lane, Type, Value};

use crate::sema::intrinsic::Simd;

use super::super::super::ops::Num;
use super::super::super::{Lower, host_is_x64_or_arm64};
use super::super::{Lanes, int_for_bytes, vec_type};

impl Lower<'_, '_> {
    pub(in super::super) fn vec_cast(
        &mut self,
        (from, from_vec): (&Lanes, Type),
        to: &Lanes,
        x: Value,
    ) -> Result<Option<Value>> {
        let Some(to_vec) = vec_type(to) else {
            return Ok(None);
        };
        if from.count != to.count {
            return Ok(None);
        }
        let v = self.ins().load(from_vec, x, 0);
        let out = match (from.num, to.num) {
            (Num::Int { ty: a, signed }, Num::Int { ty: b, .. }) => {
                if a == b {
                    v
                } else if a.bytes() < b.bytes() {
                    if signed {
                        self.ins().sextend(to_vec, v)
                    } else {
                        self.ins().uextend(to_vec, v)
                    }
                } else {
                    self.ins().ireduce(to_vec, v)
                }
            }
            (Num::Float(a), Num::Float(b)) => {
                if a == b {
                    v
                } else if a == Type::F32 {
                    self.ins().fpromote(to_vec, v)
                } else {
                    self.ins().fdemote(to_vec, v)
                }
            }
            (Num::Int { ty: a, signed }, Num::Float(b))
                if a.bytes() == b.bytes() || a.bytes() * 2 == b.bytes() =>
            {
                if signed {
                    self.ins().fcvt_from_sint(to_vec, v)
                } else {
                    self.ins().fcvt_from_uint(to_vec, v)
                }
            }
            (Num::Float(a), Num::Int { ty: b, signed }) if a.bytes() == b.bytes() => {
                if signed {
                    self.ins().fcvt_to_sint_sat(to_vec, v)
                } else {
                    self.ins().fcvt_to_uint_sat(to_vec, v)
                }
            }
            _ => return Ok(None),
        };
        Ok(Some(self.store_vec(out, to_vec.bytes())))
    }

    pub(super) fn eval_cast(
        &mut self,
        (from, to): (Lanes, Lanes),
        x: Value,
    ) -> Result<Option<Value>> {
        if from.count != to.count
            || from.count <= 0
            || !(from.count as u32).is_power_of_two()
            || !host_is_x64_or_arm64()
        {
            return Ok(None);
        }
        if let Some(vec_ty) = vec_type(&from)
            && let Some(res) = self.vec_cast((&from, vec_ty), &to, x)?
        {
            return Ok(Some(res));
        }
        if let Some(mid) = cast_step_lanes(from, to) {
            let Some(m_val) = self.eval_core_half(Simd::Cast, (&from, &mid), &[x])? else {
                return Ok(None);
            };
            return self.eval_core_half(Simd::Cast, (&mid, &to), &[m_val]);
        }
        if from.bytes() <= 16 && to.bytes() <= 16 && (from.bytes() < 8 || to.bytes() < 8) {
            return self.sub8_cast((from, to), x);
        }
        Ok(None)
    }

    fn sub8_cast(&mut self, (from, to): (Lanes, Lanes), x: Value) -> Result<Option<Value>> {
        match (from.num, to.num) {
            (Num::Int { .. }, Num::Int { .. }) if from.size == to.size => {
                let (_, addr) = self.lanes_slot(to);
                self.copy_bytes(addr, 0, x, 0, from.bytes());
                Ok(Some(addr))
            }
            (Num::Int { signed, .. }, Num::Int { .. }) if to.size == from.size * 2 => {
                let Some((wf, wf_ty)) = from.widened() else {
                    return Ok(None);
                };
                let w = self.widen_sub8(from.bytes(), wf_ty, x);
                let Some(lane) = Lane::from_type(to.machine) else {
                    return Ok(None);
                };
                let wide_to = Type::Vec {
                    lane,
                    count: wf.count as u8,
                };
                let out = if signed {
                    self.ins().sextend(wide_to, w)
                } else {
                    self.ins().uextend(wide_to, w)
                };
                let tmp = self.store_vec(out, wide_to.bytes());
                let (_, addr) = self.lanes_slot(to);
                self.copy_bytes(addr, 0, tmp, 0, to.bytes());
                Ok(Some(addr))
            }
            (Num::Int { .. }, Num::Int { .. }) if from.size == to.size * 2 => {
                let Some(fl) = Lane::from_type(from.machine) else {
                    return Ok(None);
                };
                let Some(tl) = Lane::from_type(to.machine) else {
                    return Ok(None);
                };
                let wf_ty = Type::Vec {
                    lane: fl,
                    count: (16 / from.size) as u8,
                };
                let wt_ty = Type::Vec {
                    lane: tl,
                    count: (8 / to.size) as u8,
                };
                let slot16 = self.b.create_stack_slot(16, 4);
                let addr16 = self.ins().stack_addr(Type::I64, slot16, 0);
                let z = self.ins().iconst(Type::I64, 0);
                self.ins().stack_store(z, slot16, 0);
                self.ins().stack_store(z, slot16, 8);
                self.copy_bytes(addr16, 0, x, 0, from.bytes());
                let wf = self.ins().stack_load(wf_ty, slot16, 0);
                let out = self.ins().ireduce(wt_ty, wf);
                let tmp = self.store_vec(out, 8);
                let (_, addr) = self.lanes_slot(to);
                self.copy_bytes(addr, 0, tmp, 0, to.bytes());
                Ok(Some(addr))
            }
            _ => Ok(None),
        }
    }
}

fn cast_step_lanes(from: Lanes, to: Lanes) -> Option<Lanes> {
    match (from.num, to.num) {
        (Num::Int { signed, .. }, Num::Int { .. }) if to.size > from.size * 2 => {
            let size = from.size * 2;
            let ty = int_for_bytes(size)?;
            Some(Lanes {
                num: Num::Int { ty, signed },
                machine: ty,
                size,
                count: from.count,
            })
        }
        (Num::Int { signed, .. }, Num::Int { .. }) if from.size > to.size * 2 => {
            let size = from.size / 2;
            let ty = int_for_bytes(size)?;
            Some(Lanes {
                num: Num::Int { ty, signed },
                machine: ty,
                size,
                count: from.count,
            })
        }
        (Num::Int { ty: a, signed }, Num::Float(b))
            if a.bytes() < b.bytes() && (from.bytes() < 8 || b.bytes() > a.bytes() * 2) =>
        {
            let size = from.size * 2;
            let ty = int_for_bytes(size)?;
            Some(Lanes {
                num: Num::Int { ty, signed },
                machine: ty,
                size,
                count: from.count,
            })
        }
        _ => None,
    }
}
