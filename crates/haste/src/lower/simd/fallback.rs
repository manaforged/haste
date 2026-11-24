mod cast;
mod shuffle;
mod sub8;

use anyhow::Result;
use haste_codegen::{IntCC, Lane, Type, Value};

use crate::body::BinOp;
use crate::sema::intrinsic::Simd;

use super::super::ops::Num;
use super::super::stencil::SymbolSource;
use super::super::{Lower, host_is_x64_or_arm64};
use super::{Lanes, int_for_bytes, vec_for_bytes, vec_type};

impl Lower<'_, '_> {
    /// Emits an SSA basic-block loop across lanes `[start..count)`, carrying `init` across iterations.
    pub(super) fn emit_lane_loop(
        &mut self,
        start: i64,
        count: i32,
        init: &[Value],
        mut step: impl FnMut(&mut Self, Value, &[Value]) -> Result<Vec<Value>>,
    ) -> Result<Vec<Value>> {
        if !self.open || count <= start as i32 {
            return Ok(init.to_vec());
        }
        let head = self.b.create_block();
        let body = self.b.create_block();
        let exit = self.b.create_block();
        let idx_param = self.b.append_block_param(head, Type::I64);
        let tys: Vec<Type> = init.iter().map(|&v| self.b.func.ty(v)).collect();
        let acc_params: Vec<Value> = tys
            .iter()
            .map(|&ty| self.b.append_block_param(head, ty))
            .collect();
        let exit_params: Vec<Value> = tys
            .iter()
            .map(|&ty| self.b.append_block_param(exit, ty))
            .collect();
        let mut entry_args = Vec::with_capacity(1 + init.len());
        entry_args.push(self.ins().iconst(Type::I64, start));
        entry_args.extend_from_slice(init);
        self.jump(head, &entry_args);
        self.enter(head);
        let limit = self.ins().iconst(Type::I64, i64::from(count));
        let more = self.ins().icmp(IntCC::UnsignedLessThan, idx_param, limit);
        self.brif(more, (body, &[]), (exit, &acc_params));
        self.enter(body);
        let next_accs = step(self, idx_param, &acc_params)?;
        let next_idx = self.ins().iadd_imm(idx_param, 1);
        let mut back_args = Vec::with_capacity(1 + next_accs.len());
        back_args.push(next_idx);
        back_args.extend_from_slice(&next_accs);
        self.jump(head, &back_args);
        self.enter(exit);
        Ok(exit_params)
    }

    /// Copies `bytes` from `src + src_off` to `dst + dst_off` using recursive power-of-two vector/scalar moves.
    pub(super) fn copy_bytes(
        &mut self,
        dst: Value,
        dst_off: i32,
        src: Value,
        src_off: i32,
        bytes: i32,
    ) {
        if bytes <= 0 {
            return;
        }
        if bytes > 16 {
            let half = if (bytes as u32).is_power_of_two() {
                bytes / 2
            } else {
                16
            };
            self.copy_bytes(dst, dst_off, src, src_off, half);
            self.copy_bytes(dst, dst_off + half, src, src_off + half, bytes - half);
            return;
        }
        if let Some(vt) = vec_for_bytes(bytes).filter(|_| host_is_x64_or_arm64()) {
            let v = self.ins().load(vt, src, src_off);
            self.ins().store(v, dst, dst_off);
            return;
        }
        if let Some(it) = int_for_bytes(bytes) {
            let v = self.ins().load(it, src, src_off);
            self.ins().store(v, dst, dst_off);
            return;
        }
        let chunk = if bytes >= 8 {
            8
        } else if bytes >= 4 {
            4
        } else {
            2
        };
        self.copy_bytes(dst, dst_off, src, src_off, chunk);
        self.copy_bytes(dst, dst_off + chunk, src, src_off + chunk, bytes - chunk);
    }

    fn join_fallback_halves(
        &mut self,
        lanes: Lanes,
        half_bytes: i32,
        lo: Value,
        hi: Value,
    ) -> Value {
        let (_, addr) = self.lanes_slot(lanes);
        self.copy_bytes(addr, 0, lo, 0, half_bytes);
        self.copy_bytes(addr, half_bytes, hi, 0, half_bytes);
        addr
    }

    pub(super) fn wide_lane_intrinsic(
        &mut self,
        op: Simd,
        lanes: &Lanes,
        args: &[Value],
    ) -> Result<Option<Value>> {
        if lanes.bytes() <= 16
            || lanes.count <= 0
            || !(lanes.count as u32).is_power_of_two()
            || !host_is_x64_or_arm64()
        {
            return Ok(None);
        }
        if matches!(op, Simd::MaskedLoad | Simd::MaskedStore) {
            return Ok(None);
        }
        let half = lanes.half();
        let hb = i64::from(half.bytes());
        match (op, args) {
            (Simd::Unary(_) | Simd::Fma, _) => {
                let hi_args: Vec<Value> =
                    args.iter().map(|&p| self.ins().iadd_imm(p, hb)).collect();
                let Some(lo) = self.eval_lane_half(op, &half, args)? else {
                    return Ok(None);
                };
                let Some(hi) = self.eval_lane_half(op, &half, &hi_args)? else {
                    return Ok(None);
                };
                Ok(Some(self.join_fallback_halves(
                    *lanes,
                    half.bytes(),
                    lo,
                    hi,
                )))
            }
            (Simd::Pair(_), &[x, y]) => {
                let (hx, hy) = (self.ins().iadd_imm(x, hb), self.ins().iadd_imm(y, hb));
                let Some(lo) = self.eval_lane_half(op, &half, &[x, y])? else {
                    return Ok(None);
                };
                let Some(hi) = self.eval_lane_half(op, &half, &[hx, hy])? else {
                    return Ok(None);
                };
                Ok(Some(self.join_fallback_halves(
                    *lanes,
                    half.bytes(),
                    lo,
                    hi,
                )))
            }
            (Simd::Sum { mul }, &[x, ref start @ ..]) => {
                let hx = self.ins().iadd_imm(x, hb);
                let init = start.first().copied();
                let res = if let Some(s) = init {
                    let Some(acc) = self.eval_lane_half(Simd::Sum { mul }, &half, &[x, s])? else {
                        return Ok(None);
                    };
                    let Some(total) = self.eval_lane_half(Simd::Sum { mul }, &half, &[hx, acc])?
                    else {
                        return Ok(None);
                    };
                    total
                } else {
                    let Some(lo) = self.eval_lane_half(Simd::Sum { mul }, &half, &[x])? else {
                        return Ok(None);
                    };
                    let Some(hi) = self.eval_lane_half(Simd::Sum { mul }, &half, &[hx])? else {
                        return Ok(None);
                    };
                    self.lane_op(
                        if mul { BinOp::Mul } else { BinOp::Add },
                        lanes.num,
                        (lo, hi),
                        lanes.machine,
                    )?
                };
                Ok(Some(res))
            }
            _ => Ok(None),
        }
    }

    fn eval_lane_half(&mut self, op: Simd, half: &Lanes, args: &[Value]) -> Result<Option<Value>> {
        if let Some(v) = self.vec_lane_intrinsic(op, half, args)? {
            return Ok(Some(v));
        }
        self.wide_lane_intrinsic(op, half, args)
    }

    pub(super) fn wide_simd(
        &mut self,
        op: Simd,
        (from, to): (&Lanes, &Lanes),
        args: &[Value],
    ) -> Result<Option<Value>> {
        if (from.bytes() > 16 || to.bytes() > 16)
            && let Some(res) = self.wide_split_simd(op, (from, to), args)?
        {
            return Ok(Some(res));
        }
        if op == Simd::Cast
            && let &[x] = args
            && let Some(res) = self.eval_cast((*from, *to), x)?
        {
            return Ok(Some(res));
        }
        if from.bytes() < 8 && to.bytes() < 8 {
            return self.sub8_simd(op, (from, to), args);
        }
        Ok(None)
    }

    fn wide_split_simd(
        &mut self,
        op: Simd,
        (from, to): (&Lanes, &Lanes),
        args: &[Value],
    ) -> Result<Option<Value>> {
        if from.count <= 0 || !(from.count as u32).is_power_of_two() || !host_is_x64_or_arm64() {
            return Ok(None);
        }
        let (hf, ht) = (from.half(), to.half());
        let hfb = i64::from(hf.bytes());
        match (op, args) {
            (Simd::Splat, &[scalar]) => {
                let Some(lo) = self.eval_core_half(Simd::Splat, (&hf, &hf), &[scalar])? else {
                    return Ok(None);
                };
                Ok(Some(self.join_fallback_halves(*from, hf.bytes(), lo, lo)))
            }
            (Simd::Sqrt, &[x]) => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half(Simd::Sqrt, (&hf, &hf), &[x])? else {
                    return Ok(None);
                };
                let Some(hi) = self.eval_core_half(Simd::Sqrt, (&hf, &hf), &[hx])? else {
                    return Ok(None);
                };
                Ok(Some(self.join_fallback_halves(*from, hf.bytes(), lo, hi)))
            }
            (Simd::Lanewise(_), &[x, y]) if from.count == to.count => {
                let (hx, hy) = (self.ins().iadd_imm(x, hfb), self.ins().iadd_imm(y, hfb));
                let Some(lo) = self.eval_core_half(op, (&hf, &ht), &[x, y])? else {
                    return Ok(None);
                };
                let Some(hi) = self.eval_core_half(op, (&hf, &ht), &[hx, hy])? else {
                    return Ok(None);
                };
                Ok(Some(self.join_fallback_halves(*to, ht.bytes(), lo, hi)))
            }
            (Simd::Cast, &[x])
                if from.count == to.count
                    && ((hf.bytes() >= 8 && ht.bytes() >= 8) || hf.bytes() == ht.bytes()) =>
            {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half(Simd::Cast, (&hf, &ht), &[x])? else {
                    return Ok(None);
                };
                let Some(hi) = self.eval_core_half(Simd::Cast, (&hf, &ht), &[hx])? else {
                    return Ok(None);
                };
                Ok(Some(self.join_fallback_halves(*to, ht.bytes(), lo, hi)))
            }
            (Simd::Select, &[mask, yes, no]) if from.count == to.count => {
                let (hm, hy, hn) = (
                    self.ins().iadd_imm(mask, hfb),
                    self.ins().iadd_imm(yes, i64::from(ht.bytes())),
                    self.ins().iadd_imm(no, i64::from(ht.bytes())),
                );
                let Some(lo) = self.eval_core_half(Simd::Select, (&hf, &ht), &[mask, yes, no])?
                else {
                    return Ok(None);
                };
                let Some(hi) = self.eval_core_half(Simd::Select, (&hf, &ht), &[hm, hy, hn])? else {
                    return Ok(None);
                };
                Ok(Some(self.join_fallback_halves(*to, ht.bytes(), lo, hi)))
            }
            (Simd::Reduce { all }, &[x]) => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half(op, (&hf, &hf), &[x])? else {
                    return Ok(None);
                };
                let Some(hi) = self.eval_core_half(op, (&hf, &hf), &[hx])? else {
                    return Ok(None);
                };
                Ok(Some(if all {
                    self.ins().band(lo, hi)
                } else {
                    self.ins().bor(lo, hi)
                }))
            }
            (Simd::Fold(bin), &[x]) => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half(op, (&hf, &hf), &[x])? else {
                    return Ok(None);
                };
                let Some(hi) = self.eval_core_half(op, (&hf, &hf), &[hx])? else {
                    return Ok(None);
                };
                self.apply(bin, from.num, (lo, hi), from.num).map(Some)
            }
            (Simd::Extreme { max }, &[x]) => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half(op, (&hf, &hf), &[x])? else {
                    return Ok(None);
                };
                let Some(hi) = self.eval_core_half(op, (&hf, &hf), &[hx])? else {
                    return Ok(None);
                };
                let better = if max { BinOp::Gt } else { BinOp::Lt };
                let wins = self.apply(better, from.num, (lo, hi), from.num)?;
                Ok(Some(self.ins().select(wins, lo, hi)))
            }
            (Simd::Bitmask, &[x]) if from.count <= 64 => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half_u64(op, &hf, &[x])? else {
                    return Ok(None);
                };
                let Some(hi) = self.eval_core_half_u64(op, &hf, &[hx])? else {
                    return Ok(None);
                };
                let sh = self.ins().iconst(Type::I64, i64::from(hf.count));
                let hi_shifted = self.ins().ishl(hi, sh);
                let combined = self.ins().bor(lo, hi_shifted);
                Ok(Some(self.resize(combined, (Type::I64, false), to.machine)))
            }
            _ => Ok(None),
        }
    }

    fn eval_core_half(
        &mut self,
        op: Simd,
        (from, to): (&Lanes, &Lanes),
        args: &[Value],
    ) -> Result<Option<Value>> {
        if let Some(v) = self.vec_simd_lanes(None, op, (*from, *to), &[], args)? {
            return Ok(Some(v));
        }
        self.wide_simd(op, (from, to), args)
    }

    fn eval_core_half_u64(
        &mut self,
        op: Simd,
        from: &Lanes,
        args: &[Value],
    ) -> Result<Option<Value>> {
        let out_u64 = Lanes {
            num: Num::Int {
                ty: Type::I64,
                signed: false,
            },
            machine: Type::I64,
            size: 8,
            count: 1,
        };
        if let Some(v) = self.vec_simd_lanes(None, op, (*from, out_u64), &[], args)? {
            return Ok(Some(v));
        }
        self.wide_simd(op, (from, &out_u64), args)
    }

    pub(super) fn select_bitmask_vec(
        &mut self,
        to: &Lanes,
        bits: Value,
        (yes, no): (Value, Value),
    ) -> Result<Option<Value>> {
        if to.count <= 0
            || !(to.count as u32).is_power_of_two()
            || to.count > 64
            || !host_is_x64_or_arm64()
        {
            return Ok(None);
        }
        if to.bytes() > 16 {
            let ht = to.half();
            let hb = i64::from(ht.bytes());
            let sh = self.ins().iconst(Type::I64, i64::from(ht.count));
            let hi_bits = self.ins().ushr(bits, sh);
            let (hy, hn) = (self.ins().iadd_imm(yes, hb), self.ins().iadd_imm(no, hb));
            let Some(lo) = self.select_bitmask_vec(&ht, bits, (yes, no))? else {
                return Ok(None);
            };
            let Some(hi) = self.select_bitmask_vec(&ht, hi_bits, (hy, hn))? else {
                return Ok(None);
            };
            return Ok(Some(self.join_fallback_halves(*to, ht.bytes(), lo, hi)));
        }
        if to.bytes() < 8 {
            let Some((wt, wt_ty)) = to.widened() else {
                return Ok(None);
            };
            let (uy, un) = (
                self.widen_sub8(to.bytes(), wt_ty, yes),
                self.widen_sub8(to.bytes(), wt_ty, no),
            );
            let (wy, wn) = (self.store_vec(uy, 8), self.store_vec(un, 8));
            let Some(w_res) = self.select_bitmask_vec(&wt, bits, (wy, wn))? else {
                return Ok(None);
            };
            return Ok(Some(self.narrow_sub8(to.bytes(), wt_ty, w_res)));
        }
        let Some(out_vec) = vec_type(to) else {
            return Ok(None);
        };
        let Some((lane, count)) = out_vec.vec_shape() else {
            return Ok(None);
        };
        let mask_lane = lane.int_mask();
        let mask_vec = Type::Vec {
            lane: mask_lane,
            count,
        };
        let expanded = if mask_vec
            == (Type::Vec {
                lane: Lane::I8,
                count: 16,
            }) {
            let pair = self.ins().ireduce(Type::I16, bits);
            let wide = self.ins().splat(
                Type::Vec {
                    lane: Lane::I16,
                    count: 8,
                },
                pair,
            );
            let bytes = self.bitcast_vec(mask_vec, wide);
            self.ins().shuffle(
                bytes,
                bytes,
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            )
        } else {
            let lane_scalar = self.resize(bits, (Type::I64, false), mask_lane.to_type());
            self.ins().splat(mask_vec, lane_scalar)
        };
        let lane_bits = self.lane_bitmask_const(mask_vec);
        let masked = self.ins().band(expanded, lane_bits);
        let zero = self.ins().iconst(mask_vec, 0);
        let cond = self.ins().icmp(IntCC::NotEqual, masked, zero);
        let (a, b) = (
            self.ins().load(out_vec, yes, 0),
            self.ins().load(out_vec, no, 0),
        );
        let chosen = self.ins().select(cond, a, b);
        Ok(Some(self.store_vec(chosen, out_vec.bytes())))
    }

    fn lane_bitmask_const(&mut self, mask_vec: Type) -> Value {
        let Some((lane, count)) = mask_vec.vec_shape() else {
            return self.ins().iconst(mask_vec, 0);
        };
        let lane_bytes = lane.bytes() as usize;
        let total_bytes = lane_bytes * usize::from(count);
        let mut raw = vec![0u8; total_bytes];
        raw.chunks_exact_mut(lane_bytes)
            .enumerate()
            .for_each(|(i, chunk)| {
                let bit_index = if total_bytes == 16 && lane == Lane::I8 {
                    i & 7
                } else {
                    i
                };
                let bit = 1u64 << bit_index;
                chunk.copy_from_slice(&bit.to_le_bytes()[..lane_bytes]);
            });
        let id = self.slots.rodata(&raw, &[]);
        let addr = self.emit_symbol(SymbolSource::Rodata(id));
        self.ins().load(mask_vec, addr, 0)
    }
}
