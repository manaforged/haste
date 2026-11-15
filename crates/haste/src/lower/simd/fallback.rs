use anyhow::Result;
use haste_codegen::{IntCC, Lane, ReduceOp, Type, Value};

use crate::body::BinOp;
use crate::sema::intrinsic::Simd;

use super::super::ops::Num;
use super::super::reloc::SiteOrigin;
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
        let acc_params: Vec<Value> = tys.iter().map(|&ty| self.b.append_block_param(head, ty)).collect();
        let exit_params: Vec<Value> = tys.iter().map(|&ty| self.b.append_block_param(exit, ty)).collect();
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
    pub(super) fn copy_bytes(&mut self, dst: Value, dst_off: i32, src: Value, src_off: i32, bytes: i32) {
        if bytes <= 0 {
            return;
        }
        if bytes > 16 {
            let half = if (bytes as u32).is_power_of_two() { bytes / 2 } else { 16 };
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
        let chunk = if bytes >= 8 { 8 } else if bytes >= 4 { 4 } else { 2 };
        self.copy_bytes(dst, dst_off, src, src_off, chunk);
        self.copy_bytes(dst, dst_off + chunk, src, src_off + chunk, bytes - chunk);
    }

    pub(super) fn wide_lane_intrinsic(&mut self, op: Simd, lanes: &Lanes, args: &[Value]) -> Result<Option<Value>> {
        if lanes.bytes() <= 16 || lanes.count <= 0 || !(lanes.count as u32).is_power_of_two() || !host_is_x64_or_arm64() {
            return Ok(None);
        }
        if matches!(op, Simd::MaskedLoad | Simd::MaskedStore) {
            return Ok(None);
        }
        let half = lanes.half();
        let hb = i64::from(half.bytes());
        match (op, args) {
            (Simd::Unary(_) | Simd::Fma, _) => {
                let (_, addr) = self.lanes_slot(*lanes);
                let hi_args: Vec<Value> = args.iter().map(|&p| self.ins().iadd_imm(p, hb)).collect();
                let Some(lo) = self.eval_lane_half(op, &half, args)? else { return Ok(None) };
                let Some(hi) = self.eval_lane_half(op, &half, &hi_args)? else { return Ok(None) };
                self.copy_bytes(addr, 0, lo, 0, half.bytes());
                self.copy_bytes(addr, half.bytes(), hi, 0, half.bytes());
                Ok(Some(addr))
            }
            (Simd::Pair(_), &[x, y]) => {
                let (_, addr) = self.lanes_slot(*lanes);
                let (hx, hy) = (self.ins().iadd_imm(x, hb), self.ins().iadd_imm(y, hb));
                let Some(lo) = self.eval_lane_half(op, &half, &[x, y])? else { return Ok(None) };
                let Some(hi) = self.eval_lane_half(op, &half, &[hx, hy])? else { return Ok(None) };
                self.copy_bytes(addr, 0, lo, 0, half.bytes());
                self.copy_bytes(addr, half.bytes(), hi, 0, half.bytes());
                Ok(Some(addr))
            }
            (Simd::Sum { mul }, &[x, ref start @ ..]) => {
                let hx = self.ins().iadd_imm(x, hb);
                let init = start.first().copied();
                let res = if let Some(s) = init {
                    let Some(acc) = self.eval_lane_half(Simd::Sum { mul }, &half, &[x, s])? else { return Ok(None) };
                    let Some(total) = self.eval_lane_half(Simd::Sum { mul }, &half, &[hx, acc])? else { return Ok(None) };
                    total
                } else {
                    let Some(lo) = self.eval_lane_half(Simd::Sum { mul }, &half, &[x])? else { return Ok(None) };
                    let Some(hi) = self.eval_lane_half(Simd::Sum { mul }, &half, &[hx])? else { return Ok(None) };
                    self.lane_op(if mul { BinOp::Mul } else { BinOp::Add }, lanes.num, (lo, hi), lanes.machine)?
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

    pub(super) fn sub8_lane_intrinsic(&mut self, op: Simd, lanes: &Lanes, args: &[Value]) -> Result<Option<Value>> {
        let Some((wlanes, wvec)) = lanes.widened() else { return Ok(None) };
        match (op, args) {
            (Simd::Sum { mul }, &[x, ref start @ ..]) => {
                if lanes.count == 1 {
                    let v = self.ins().load(lanes.machine, x, 0);
                    let bin = if mul { BinOp::Mul } else { BinOp::Add };
                    let out = match start.first().copied() {
                        Some(init) => self.lane_op(bin, lanes.num, (init, v), lanes.machine)?,
                        None => v,
                    };
                    return Ok(Some(out));
                }
                let mut v = self.widen_sub8(lanes.bytes(), wvec, x);
                if mul {
                    let one = self.lane_identity_one(lanes.num)?;
                    v = self.fill_upper_lanes(wvec, lanes.count as usize, v, one);
                }
                self.vec_sum(&wlanes, wvec, v, (start.first().copied(), mul)).map(Some)
            }
            (Simd::Unary(_) | Simd::Pair(_) | Simd::Fma, _) => {
                let widened: Vec<Value> = args.iter().map(|&p| {
                    let w = self.widen_sub8(lanes.bytes(), wvec, p);
                    self.store_vec(w, 8)
                }).collect();
                let Some(wide_res) = self.vec_lane_intrinsic(op, &wlanes, &widened)? else { return Ok(None) };
                Ok(Some(self.narrow_sub8(lanes.bytes(), wvec, wide_res)))
            }
            _ => Ok(None),
        }
    }

    pub(super) fn wide_simd(&mut self, op: Simd, (from, to): (&Lanes, &Lanes), args: &[Value]) -> Result<Option<Value>> {
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

    pub(super) fn wide_lane_op(&mut self, origin: &SiteOrigin, op: Simd, (from, to): (&Lanes, &Lanes), args: &[Value]) -> Result<Option<Value>> {
        match (op, args) {
            (Simd::Extract, &[x, _]) => {
                let Some(limit) = u8::try_from(from.count).ok() else { return Ok(None) };
                let Some(idx) = self.const_arg_u8(origin, 1, limit) else { return Ok(None) };
                if from.bytes() > 16 && (from.count as u32).is_power_of_two() && host_is_x64_or_arm64() {
                    let Some(lane) = Lane::from_type(from.machine) else { return Ok(None) };
                    let chunk_lanes = 16 / from.size;
                    let chunk_vec = Type::Vec { lane, count: chunk_lanes as u8 };
                    let chunk_off = (i32::from(idx) / chunk_lanes) * 16;
                    let sub_idx = (i32::from(idx) % chunk_lanes) as u8;
                    let v = self.ins().load(chunk_vec, x, chunk_off);
                    return Ok(Some(self.ins().extract(v, sub_idx)));
                }
                Ok(Some(self.ins().load(from.machine, x, i32::from(idx) * from.size)))
            }
            (Simd::Insert, &[x, _, val]) => {
                let Some(limit) = u8::try_from(from.count).ok() else { return Ok(None) };
                let Some(idx) = self.const_arg_u8(origin, 1, limit) else { return Ok(None) };
                let (slot, addr) = self.lanes_slot(*from);
                if from.bytes() > 16 && (from.count as u32).is_power_of_two() && host_is_x64_or_arm64() {
                    let Some(lane) = Lane::from_type(from.machine) else { return Ok(None) };
                    let chunk_lanes = 16 / from.size;
                    let chunk_vec = Type::Vec { lane, count: chunk_lanes as u8 };
                    let chunk_off = (i32::from(idx) / chunk_lanes) * 16;
                    let sub_idx = (i32::from(idx) % chunk_lanes) as u8;
                    self.copy_bytes(addr, 0, x, 0, chunk_off);
                    let chunk = self.ins().load(chunk_vec, x, chunk_off);
                    let updated = self.ins().insert(chunk, val, sub_idx);
                    self.ins().stack_store(updated, slot, chunk_off);
                    self.copy_bytes(addr, chunk_off + 16, x, chunk_off + 16, from.bytes() - (chunk_off + 16));
                    return Ok(Some(addr));
                }
                self.copy_bytes(addr, 0, x, 0, from.bytes());
                self.ins().stack_store(val, slot, i32::from(idx) * from.size);
                Ok(Some(addr))
            }
            (Simd::Shuffle, &[x, y, _]) => self.wide_shuffle(origin, (*from, *to), (x, y)),
            _ => Ok(None),
        }
    }

    fn wide_shuffle(&mut self, origin: &SiteOrigin, (from, to): (Lanes, Lanes), (x, y): (Value, Value)) -> Result<Option<Value>> {
        if from.count <= 0
            || to.count <= 0
            || !(from.count as u32).is_power_of_two()
            || !(to.count as u32).is_power_of_two()
            || from.size != to.size
            || to.count > 64
            || !host_is_x64_or_arm64()
        {
            return Ok(None);
        }
        let Some(mut mask) = self.shuffle_const_mask_wide(origin, from.count as usize, to.count as usize) else { return Ok(None) };
        let (in_lanes, in_x, in_y) = if from.bytes() < 8 {
            let Some((wf, wf_ty)) = from.widened() else { return Ok(None) };
            let wx_val = self.widen_sub8(from.bytes(), wf_ty, x);
            let wy_val = self.widen_sub8(from.bytes(), wf_ty, y);
            let wx = self.store_vec(wx_val, 8);
            let wy = self.store_vec(wy_val, 8);
            for slot in &mut mask[..to.count as usize] {
                if *slot >= from.count as u8 {
                    *slot = wf.count as u8 + (*slot - from.count as u8);
                }
            }
            (wf, wx, wy)
        } else {
            (from, x, y)
        };
        let (out_lanes, narrow_bytes) = if to.bytes() < 8 {
            let Some((wt, _)) = to.widened() else { return Ok(None) };
            (wt, Some(to.bytes()))
        } else {
            (to, None)
        };
        let chunk_bytes = if in_lanes.bytes() >= 16 && out_lanes.bytes() >= 16 { 16 } else { 8 };
        let chunk_lanes = (chunk_bytes / out_lanes.size) as usize;
        let Some(chunk_lane) = Lane::from_type(out_lanes.machine) else { return Ok(None) };
        let chunk_vec = Type::Vec { lane: chunk_lane, count: chunk_lanes as u8 };
        let int_vec = Type::Vec { lane: chunk_lane.int_mask(), count: chunk_lanes as u8 };
        let in_chunks = (in_lanes.bytes() / chunk_bytes) as usize;
        let out_chunks = (out_lanes.bytes() / chunk_bytes) as usize;
        let (slot, addr) = self.lanes_slot(out_lanes);
        for oc in 0..out_chunks {
            let sub = &mask[oc * chunk_lanes..(oc + 1) * chunk_lanes];
            let chunk = self.shuffle_chunk((chunk_vec, int_vec), (chunk_bytes, chunk_lanes), in_chunks, (in_x, in_y), sub);
            self.ins().stack_store(chunk, slot, oc as i32 * chunk_bytes);
        }
        if let Some(bytes) = narrow_bytes {
            let (_, out_addr) = self.lanes_slot(to);
            self.copy_bytes(out_addr, 0, addr, 0, bytes);
            return Ok(Some(out_addr));
        }
        Ok(Some(addr))
    }

    fn shuffle_chunk(&mut self, (chunk_vec, int_vec): (Type, Type), (chunk_bytes, chunk_lanes): (i32, usize), in_chunks: usize, (in_x, in_y): (Value, Value), sub: &[u8]) -> Value {
        let mut cids = Vec::with_capacity(4);
        let mut items = [(0usize, 0u8); 16];
        for (j, &m) in sub.iter().enumerate() {
            let cid = usize::from(m) / chunk_lanes;
            let lane = (usize::from(m) % chunk_lanes) as u8;
            items[j] = (cid, lane);
            if !cids.contains(&cid) {
                cids.push(cid);
            }
        }
        let load_chunk = |lower: &mut Self, cid: usize| {
            let ptr = if cid < in_chunks { in_x } else { in_y };
            let off = ((cid % in_chunks) as i32) * chunk_bytes;
            lower.ins().load(chunk_vec, ptr, off)
        };
        if cids.len() == 1 {
            let v = load_chunk(self, cids[0]);
            if (0..chunk_lanes).all(|j| items[j].1 == j as u8) {
                return v;
            }
            let mut m = [0u8; 16];
            for j in 0..chunk_lanes {
                m[j] = items[j].1;
            }
            return self.ins().shuffle(v, v, m);
        }
        if cids.len() == 2 {
            let (v0, v1) = (load_chunk(self, cids[0]), load_chunk(self, cids[1]));
            let mut m = [0u8; 16];
            for j in 0..chunk_lanes {
                m[j] = if items[j].0 == cids[0] { items[j].1 } else { chunk_lanes as u8 + items[j].1 };
            }
            return self.ins().shuffle(v0, v1, m);
        }
        let zero = (2 * chunk_lanes) as u8;
        let mut acc = None;
        for pair in cids.chunks(2) {
            let ca = pair[0];
            let opt_cb = pair.get(1).copied();
            let va = load_chunk(self, ca);
            let vb = opt_cb.map_or(va, |cb| load_chunk(self, cb));
            let mut m = [zero; 16];
            for j in 0..chunk_lanes {
                if items[j].0 == ca {
                    m[j] = items[j].1;
                } else if Some(items[j].0) == opt_cb {
                    m[j] = chunk_lanes as u8 + items[j].1;
                }
            }
            let shuf = self.ins().shuffle(va, vb, m);
            let part = self.bitcast_vec(int_vec, shuf);
            acc = Some(match acc {
                Some(prev) => self.ins().bor(prev, part),
                None => part,
            });
        }
        let res = match acc {
            Some(v) => v,
            None => self.ins().iconst(int_vec, 0),
        };
        self.bitcast_vec(chunk_vec, res)
    }

    pub(super) fn vec_cast(&mut self, (from, from_vec): (&Lanes, Type), to: &Lanes, x: Value) -> Result<Option<Value>> {
        let Some(to_vec) = vec_type(to) else { return Ok(None) };
        if from.count != to.count {
            return Ok(None);
        }
        let v = self.ins().load(from_vec, x, 0);
        let out = match (from.num, to.num) {
            (Num::Int { ty: a, signed }, Num::Int { ty: b, .. }) => {
                if a == b {
                    v
                } else if a.bytes() < b.bytes() {
                    if signed { self.ins().sextend(to_vec, v) } else { self.ins().uextend(to_vec, v) }
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
            (Num::Int { ty: a, signed }, Num::Float(b)) if a.bytes() == b.bytes() || a.bytes() * 2 == b.bytes() => {
                if signed { self.ins().fcvt_from_sint(to_vec, v) } else { self.ins().fcvt_from_uint(to_vec, v) }
            }
            (Num::Float(a), Num::Int { ty: b, signed }) if a.bytes() == b.bytes() => {
                if signed { self.ins().fcvt_to_sint_sat(to_vec, v) } else { self.ins().fcvt_to_uint_sat(to_vec, v) }
            }
            _ => return Ok(None),
        };
        Ok(Some(self.store_vec(out, to_vec.bytes())))
    }

    fn eval_cast(&mut self, (from, to): (Lanes, Lanes), x: Value) -> Result<Option<Value>> {
        if from.count != to.count || from.count <= 0 || !(from.count as u32).is_power_of_two() || !host_is_x64_or_arm64() {
            return Ok(None);
        }
        if let Some(vec_ty) = vec_type(&from)
            && let Some(res) = self.vec_cast((&from, vec_ty), &to, x)?
        {
            return Ok(Some(res));
        }
        if let Some(mid) = cast_step_lanes(from, to) {
            let Some(m_val) = self.eval_core_half(Simd::Cast, (&from, &mid), &[x])? else { return Ok(None) };
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
                let Some((wf, wf_ty)) = from.widened() else { return Ok(None) };
                let w = self.widen_sub8(from.bytes(), wf_ty, x);
                let Some(lane) = Lane::from_type(to.machine) else { return Ok(None) };
                let wide_to = Type::Vec { lane, count: wf.count as u8 };
                let out = if signed { self.ins().sextend(wide_to, w) } else { self.ins().uextend(wide_to, w) };
                let tmp = self.store_vec(out, wide_to.bytes());
                let (_, addr) = self.lanes_slot(to);
                self.copy_bytes(addr, 0, tmp, 0, to.bytes());
                Ok(Some(addr))
            }
            (Num::Int { .. }, Num::Int { .. }) if from.size == to.size * 2 => {
                let Some(fl) = Lane::from_type(from.machine) else { return Ok(None) };
                let Some(tl) = Lane::from_type(to.machine) else { return Ok(None) };
                let wf_ty = Type::Vec { lane: fl, count: (16 / from.size) as u8 };
                let wt_ty = Type::Vec { lane: tl, count: (8 / to.size) as u8 };
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

    fn wide_split_simd(&mut self, op: Simd, (from, to): (&Lanes, &Lanes), args: &[Value]) -> Result<Option<Value>> {
        if from.count <= 0 || !(from.count as u32).is_power_of_two() || !host_is_x64_or_arm64() {
            return Ok(None);
        }
        let (hf, ht) = (from.half(), to.half());
        let hfb = i64::from(hf.bytes());
        match (op, args) {
            (Simd::Splat, &[scalar]) => {
                let Some(lo) = self.eval_core_half(Simd::Splat, (&hf, &hf), &[scalar])? else { return Ok(None) };
                let (_, addr) = self.lanes_slot(*from);
                self.copy_bytes(addr, 0, lo, 0, hf.bytes());
                self.copy_bytes(addr, hf.bytes(), lo, 0, hf.bytes());
                Ok(Some(addr))
            }
            (Simd::Sqrt, &[x]) => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half(Simd::Sqrt, (&hf, &hf), &[x])? else { return Ok(None) };
                let Some(hi) = self.eval_core_half(Simd::Sqrt, (&hf, &hf), &[hx])? else { return Ok(None) };
                let (_, addr) = self.lanes_slot(*from);
                self.copy_bytes(addr, 0, lo, 0, hf.bytes());
                self.copy_bytes(addr, hf.bytes(), hi, 0, hf.bytes());
                Ok(Some(addr))
            }
            (Simd::Lanewise(_), &[x, y]) if from.count == to.count => {
                let (hx, hy) = (self.ins().iadd_imm(x, hfb), self.ins().iadd_imm(y, hfb));
                let Some(lo) = self.eval_core_half(op, (&hf, &ht), &[x, y])? else { return Ok(None) };
                let Some(hi) = self.eval_core_half(op, (&hf, &ht), &[hx, hy])? else { return Ok(None) };
                let (_, addr) = self.lanes_slot(*to);
                self.copy_bytes(addr, 0, lo, 0, ht.bytes());
                self.copy_bytes(addr, ht.bytes(), hi, 0, ht.bytes());
                Ok(Some(addr))
            }
            (Simd::Cast, &[x]) if from.count == to.count && (hf.bytes() >= 8 && ht.bytes() >= 8 || hf.bytes() == ht.bytes()) => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half(Simd::Cast, (&hf, &ht), &[x])? else { return Ok(None) };
                let Some(hi) = self.eval_core_half(Simd::Cast, (&hf, &ht), &[hx])? else { return Ok(None) };
                let (_, addr) = self.lanes_slot(*to);
                self.copy_bytes(addr, 0, lo, 0, ht.bytes());
                self.copy_bytes(addr, ht.bytes(), hi, 0, ht.bytes());
                Ok(Some(addr))
            }
            (Simd::Select, &[mask, yes, no]) if from.count == to.count => {
                let (hm, hy, hn) = (self.ins().iadd_imm(mask, hfb), self.ins().iadd_imm(yes, i64::from(ht.bytes())), self.ins().iadd_imm(no, i64::from(ht.bytes())));
                let Some(lo) = self.eval_core_half(Simd::Select, (&hf, &ht), &[mask, yes, no])? else { return Ok(None) };
                let Some(hi) = self.eval_core_half(Simd::Select, (&hf, &ht), &[hm, hy, hn])? else { return Ok(None) };
                let (_, addr) = self.lanes_slot(*to);
                self.copy_bytes(addr, 0, lo, 0, ht.bytes());
                self.copy_bytes(addr, ht.bytes(), hi, 0, ht.bytes());
                Ok(Some(addr))
            }
            (Simd::Reduce { all }, &[x]) => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half(op, (&hf, &hf), &[x])? else { return Ok(None) };
                let Some(hi) = self.eval_core_half(op, (&hf, &hf), &[hx])? else { return Ok(None) };
                Ok(Some(if all { self.ins().band(lo, hi) } else { self.ins().bor(lo, hi) }))
            }
            (Simd::Fold(bin), &[x]) => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half(op, (&hf, &hf), &[x])? else { return Ok(None) };
                let Some(hi) = self.eval_core_half(op, (&hf, &hf), &[hx])? else { return Ok(None) };
                self.apply(bin, from.num, (lo, hi), from.num).map(Some)
            }
            (Simd::Extreme { max }, &[x]) => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half(op, (&hf, &hf), &[x])? else { return Ok(None) };
                let Some(hi) = self.eval_core_half(op, (&hf, &hf), &[hx])? else { return Ok(None) };
                let better = if max { BinOp::Gt } else { BinOp::Lt };
                let wins = self.apply(better, from.num, (lo, hi), from.num)?;
                Ok(Some(self.ins().select(wins, lo, hi)))
            }
            (Simd::Bitmask, &[x]) if from.count <= 64 => {
                let hx = self.ins().iadd_imm(x, hfb);
                let Some(lo) = self.eval_core_half_u64(op, &hf, &[x])? else { return Ok(None) };
                let Some(hi) = self.eval_core_half_u64(op, &hf, &[hx])? else { return Ok(None) };
                let sh = self.ins().iconst(Type::I64, i64::from(hf.count));
                let hi_shifted = self.ins().ishl(hi, sh);
                let combined = self.ins().bor(lo, hi_shifted);
                Ok(Some(self.resize(combined, (Type::I64, false), to.machine)))
            }
            _ => Ok(None),
        }
    }

    fn eval_core_half(&mut self, op: Simd, (from, to): (&Lanes, &Lanes), args: &[Value]) -> Result<Option<Value>> {
        if let Some(v) = self.vec_simd_lanes(None, op, (*from, *to), &[], args)? {
            return Ok(Some(v));
        }
        self.wide_simd(op, (from, to), args)
    }

    fn eval_core_half_u64(&mut self, op: Simd, from: &Lanes, args: &[Value]) -> Result<Option<Value>> {
        let out_u64 = Lanes { num: Num::Int { ty: Type::I64, signed: false }, machine: Type::I64, size: 8, count: 1 };
        if let Some(v) = self.vec_simd_lanes(None, op, (*from, out_u64), &[], args)? {
            return Ok(Some(v));
        }
        self.wide_simd(op, (from, &out_u64), args)
    }

    fn sub8_simd(&mut self, op: Simd, (from, to): (&Lanes, &Lanes), args: &[Value]) -> Result<Option<Value>> {
        let Some((wf, wf_ty)) = from.widened() else { return Ok(None) };
        let Some((wt, _)) = to.widened() else { return Ok(None) };
        match (op, args) {
            (Simd::Splat, &[scalar]) => {
                let Some(w_res) = self.vec_simd_lanes(None, Simd::Splat, (wf, wf), &[], &[scalar])? else { return Ok(None) };
                Ok(Some(self.narrow_sub8(from.bytes(), wf_ty, w_res)))
            }
            (Simd::Bitmask, &[x]) => {
                let w = self.widen_sub8(from.bytes(), wf_ty, x);
                let mask = self.ins().reduce(ReduceOp::Movemask, w);
                let mask64 = self.resize(mask, (Type::I32, false), Type::I64);
                let keep = self.ins().iconst(Type::I64, (1i64 << from.count) - 1);
                let trimmed = self.ins().band(mask64, keep);
                Ok(Some(self.resize(trimmed, (Type::I64, false), to.machine)))
            }
            (Simd::Sqrt, &[x]) => {
                let wide_x = self.widen_sub8(from.bytes(), wf_ty, x);
                let w = self.store_vec(wide_x, 8);
                let Some(w_res) = self.vec_simd_lanes(None, Simd::Sqrt, (wf, wf), &[], &[w])? else { return Ok(None) };
                Ok(Some(self.narrow_sub8(from.bytes(), wf_ty, w_res)))
            }
            (Simd::Reduce { all }, &[x]) => {
                let mut v = self.widen_sub8(from.bytes(), wf_ty, x);
                if all {
                    let first = self.ins().extract(v, 0);
                    v = self.fill_upper_lanes(wf_ty, from.count as usize, v, first);
                }
                let zero = self.ins().iconst(wf_ty, 0);
                let mask = self.ins().icmp(IntCC::NotEqual, v, zero);
                let red = self.ins().reduce(if all { ReduceOp::And } else { ReduceOp::Or }, mask);
                Ok(Some(self.ins().icmp_imm(IntCC::NotEqual, red, 0)))
            }
            (Simd::Fold(bin), &[x]) if wf_ty.lane().is_some_and(Lane::is_int) => {
                let red = match bin {
                    BinOp::BitAnd => ReduceOp::And,
                    BinOp::BitOr => ReduceOp::Or,
                    BinOp::BitXor => ReduceOp::Xor,
                    BinOp::Add => ReduceOp::Add,
                    _ => return Ok(None),
                };
                let mut v = self.widen_sub8(from.bytes(), wf_ty, x);
                if bin == BinOp::BitAnd {
                    let first = self.ins().extract(v, 0);
                    v = self.fill_upper_lanes(wf_ty, from.count as usize, v, first);
                }
                Ok(Some(self.ins().reduce(red, v)))
            }
            (Simd::Extreme { max }, &[x]) => {
                let v = self.widen_sub8(from.bytes(), wf_ty, x);
                let first = self.ins().extract(v, 0);
                let filled = self.fill_upper_lanes(wf_ty, from.count as usize, v, first);
                let red = match (from.num, max) {
                    (Num::Float(_), true) => ReduceOp::Fmax,
                    (Num::Float(_), false) => ReduceOp::Fmin,
                    (Num::Int { signed: true, .. }, true) => ReduceOp::Smax,
                    (Num::Int { signed: true, .. }, false) => ReduceOp::Smin,
                    (_, true) => ReduceOp::Umax,
                    (_, false) => ReduceOp::Umin,
                };
                Ok(Some(self.ins().reduce(red, filled)))
            }
            (Simd::Lanewise(_), &[x, y]) if from.size == to.size && from.count == to.count => {
                let (ux, uy) = (self.widen_sub8(from.bytes(), wf_ty, x), self.widen_sub8(from.bytes(), wf_ty, y));
                let (wx, wy) = (self.store_vec(ux, 8), self.store_vec(uy, 8));
                let Some(w_res) = self.vec_simd_lanes(None, op, (wf, wt), &[], &[wx, wy])? else { return Ok(None) };
                Ok(Some(self.narrow_sub8(from.bytes(), wf_ty, w_res)))
            }
            (Simd::Select, &[mask, yes, no]) if from.size == to.size && from.count == to.count => {
                let Some((_, wt_ty)) = to.widened() else { return Ok(None) };
                let (um, uy, un) = (
                    self.widen_sub8(from.bytes(), wf_ty, mask),
                    self.widen_sub8(to.bytes(), wt_ty, yes),
                    self.widen_sub8(to.bytes(), wt_ty, no),
                );
                let (wm, wy, wn) = (self.store_vec(um, 8), self.store_vec(uy, 8), self.store_vec(un, 8));
                let Some(w_res) = self.vec_simd_lanes(None, Simd::Select, (wf, wt), &[], &[wm, wy, wn])? else { return Ok(None) };
                Ok(Some(self.narrow_sub8(to.bytes(), wt_ty, w_res)))
            }
            _ => Ok(None),
        }
    }

    pub(super) fn widen_sub8(&mut self, bytes: i32, wvec: Type, ptr: Value) -> Value {
        let Some(small_ty) = int_for_bytes(bytes) else {
            return self.ins().load(wvec, ptr, 0);
        };
        let raw = self.ins().load(small_ty, ptr, 0);
        let wide = self.ins().uextend(Type::I64, raw);
        let slot = self.b.create_stack_slot(8, 3);
        self.ins().stack_store(wide, slot, 0);
        self.ins().stack_load(wvec, slot, 0)
    }

    pub(super) fn narrow_sub8(&mut self, bytes: i32, wvec: Type, w_addr: Value) -> Value {
        let w = self.ins().load(wvec, w_addr, 0);
        let slot = self.b.create_stack_slot(8, 3);
        self.ins().stack_store(w, slot, 0);
        let word = self.ins().stack_load(Type::I64, slot, 0);
        let Some(small_ty) = int_for_bytes(bytes) else {
            return w_addr;
        };
        let reduced = self.ins().ireduce(small_ty, word);
        let out_slot = self.b.create_stack_slot(bytes as u32, bytes.trailing_zeros() as u8);
        self.ins().stack_store(reduced, out_slot, 0);
        self.ins().stack_addr(Type::I64, out_slot, 0)
    }

    fn fill_upper_lanes(&mut self, wvec: Type, keep: usize, vec: Value, fill_scalar: Value) -> Value {
        let wcount = wvec.vec_shape().map_or(1, |(_, c)| usize::from(c));
        let fill_vec = self.ins().splat(wvec, fill_scalar);
        let mask = std::array::from_fn(|i| {
            if i >= wcount {
                0
            } else if i < keep {
                i as u8
            } else {
                wcount as u8
            }
        });
        self.ins().shuffle(vec, fill_vec, mask)
    }

    fn lane_identity_one(&mut self, num: Num) -> Result<Value> {
        match num {
            Num::Float(ty) => self.float_const(ty, 1.0),
            Num::Int { ty, .. } => Ok(self.ins().iconst(ty, 1)),
            Num::Bool => Ok(self.ins().iconst(Type::I8, 1)),
        }
    }

    pub(super) fn select_bitmask_vec(&mut self, to: &Lanes, bits: Value, (yes, no): (Value, Value)) -> Result<Option<Value>> {
        if to.count <= 0 || !(to.count as u32).is_power_of_two() || to.count > 64 || !host_is_x64_or_arm64() {
            return Ok(None);
        }
        if to.bytes() > 16 {
            let ht = to.half();
            let hb = i64::from(ht.bytes());
            let sh = self.ins().iconst(Type::I64, i64::from(ht.count));
            let hi_bits = self.ins().ushr(bits, sh);
            let (hy, hn) = (self.ins().iadd_imm(yes, hb), self.ins().iadd_imm(no, hb));
            let Some(lo) = self.select_bitmask_vec(&ht, bits, (yes, no))? else { return Ok(None) };
            let Some(hi) = self.select_bitmask_vec(&ht, hi_bits, (hy, hn))? else { return Ok(None) };
            let (_, addr) = self.lanes_slot(*to);
            self.copy_bytes(addr, 0, lo, 0, ht.bytes());
            self.copy_bytes(addr, ht.bytes(), hi, 0, ht.bytes());
            return Ok(Some(addr));
        }
        if to.bytes() < 8 {
            let Some((wt, wt_ty)) = to.widened() else { return Ok(None) };
            let (uy, un) = (self.widen_sub8(to.bytes(), wt_ty, yes), self.widen_sub8(to.bytes(), wt_ty, no));
            let (wy, wn) = (self.store_vec(uy, 8), self.store_vec(un, 8));
            let Some(w_res) = self.select_bitmask_vec(&wt, bits, (wy, wn))? else { return Ok(None) };
            return Ok(Some(self.narrow_sub8(to.bytes(), wt_ty, w_res)));
        }
        let Some(out_vec) = vec_type(to) else { return Ok(None) };
        let Some((lane, count)) = out_vec.vec_shape() else { return Ok(None) };
        let mask_lane = lane.int_mask();
        let mask_vec = Type::Vec { lane: mask_lane, count };
        let expanded = if mask_vec == (Type::Vec { lane: Lane::I8, count: 16 }) {
            let pair = self.ins().ireduce(Type::I16, bits);
            let wide = self.ins().splat(Type::Vec { lane: Lane::I16, count: 8 }, pair);
            let bytes = self.bitcast_vec(mask_vec, wide);
            self.ins().shuffle(bytes, bytes, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        } else {
            let lane_scalar = self.resize(bits, (Type::I64, false), mask_lane.to_type());
            self.ins().splat(mask_vec, lane_scalar)
        };
        let lane_bits = self.lane_bitmask_const(mask_vec);
        let masked = self.ins().band(expanded, lane_bits);
        let zero = self.ins().iconst(mask_vec, 0);
        let cond = self.ins().icmp(IntCC::NotEqual, masked, zero);
        let (a, b) = (self.ins().load(out_vec, yes, 0), self.ins().load(out_vec, no, 0));
        let chosen = self.ins().select(cond, a, b);
        Ok(Some(self.store_vec(chosen, out_vec.bytes())))
    }

    fn lane_bitmask_const(&mut self, mask_vec: Type) -> Value {
        let Some((lane, count)) = mask_vec.vec_shape() else { return self.ins().iconst(mask_vec, 0) };
        let lane_bytes = lane.bytes() as usize;
        let total_bytes = lane_bytes * usize::from(count);
        let mut raw = vec![0u8; total_bytes];
        raw.chunks_exact_mut(lane_bytes).enumerate().for_each(|(i, chunk)| {
            let bit_index = if total_bytes == 16 && lane == Lane::I8 { i & 7 } else { i };
            let bit = 1u64 << bit_index;
            chunk.copy_from_slice(&bit.to_le_bytes()[..lane_bytes]);
        });
        let id = self.slots.rodata(&raw, &[]);
        let addr = self.emit_symbol(SymbolSource::Rodata(id));
        self.ins().load(mask_vec, addr, 0)
    }
}

fn cast_step_lanes(from: Lanes, to: Lanes) -> Option<Lanes> {
    match (from.num, to.num) {
        (Num::Int { signed, .. }, Num::Int { .. }) if to.size > from.size * 2 => {
            let size = from.size * 2;
            let ty = int_for_bytes(size)?;
            Some(Lanes { num: Num::Int { ty, signed }, machine: ty, size, count: from.count })
        }
        (Num::Int { signed, .. }, Num::Int { .. }) if from.size > to.size * 2 => {
            let size = from.size / 2;
            let ty = int_for_bytes(size)?;
            Some(Lanes { num: Num::Int { ty, signed }, machine: ty, size, count: from.count })
        }
        (Num::Int { ty: a, signed }, Num::Float(b)) if a.bytes() < b.bytes() && (from.bytes() < 8 || b.bytes() > a.bytes() * 2) => {
            let size = from.size * 2;
            let ty = int_for_bytes(size)?;
            Some(Lanes { num: Num::Int { ty, signed }, machine: ty, size, count: from.count })
        }
        _ => None,
    }
}
