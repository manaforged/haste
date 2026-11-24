use anyhow::Result;
use haste_codegen::{Lane, Type, Value};

use crate::sema::intrinsic::Simd;

use super::super::super::reloc::SiteOrigin;
use super::super::super::{Lower, host_is_x64_or_arm64};
use super::super::Lanes;

impl Lower<'_, '_> {
    pub(in super::super) fn wide_lane_op(
        &mut self,
        origin: &SiteOrigin,
        op: Simd,
        (from, to): (&Lanes, &Lanes),
        args: &[Value],
    ) -> Result<Option<Value>> {
        match (op, args) {
            (Simd::Extract, &[x, _]) => {
                let Some(limit) = u8::try_from(from.count).ok() else {
                    return Ok(None);
                };
                let Some(idx) = self.const_arg_u8(origin, 1, limit) else {
                    return Ok(None);
                };
                if from.bytes() > 16
                    && (from.count as u32).is_power_of_two()
                    && host_is_x64_or_arm64()
                {
                    let Some(lane) = Lane::from_type(from.machine) else {
                        return Ok(None);
                    };
                    let chunk_lanes = 16 / from.size;
                    let chunk_vec = Type::Vec {
                        lane,
                        count: chunk_lanes as u8,
                    };
                    let chunk_off = (i32::from(idx) / chunk_lanes) * 16;
                    let sub_idx = (i32::from(idx) % chunk_lanes) as u8;
                    let v = self.ins().load(chunk_vec, x, chunk_off);
                    return Ok(Some(self.ins().extract(v, sub_idx)));
                }
                Ok(Some(self.ins().load(
                    from.machine,
                    x,
                    i32::from(idx) * from.size,
                )))
            }
            (Simd::Insert, &[x, _, val]) => {
                let Some(limit) = u8::try_from(from.count).ok() else {
                    return Ok(None);
                };
                let Some(idx) = self.const_arg_u8(origin, 1, limit) else {
                    return Ok(None);
                };
                let (slot, addr) = self.lanes_slot(*from);
                if from.bytes() > 16
                    && (from.count as u32).is_power_of_two()
                    && host_is_x64_or_arm64()
                {
                    let Some(lane) = Lane::from_type(from.machine) else {
                        return Ok(None);
                    };
                    let chunk_lanes = 16 / from.size;
                    let chunk_vec = Type::Vec {
                        lane,
                        count: chunk_lanes as u8,
                    };
                    let chunk_off = (i32::from(idx) / chunk_lanes) * 16;
                    let sub_idx = (i32::from(idx) % chunk_lanes) as u8;
                    self.copy_bytes(addr, 0, x, 0, chunk_off);
                    let chunk = self.ins().load(chunk_vec, x, chunk_off);
                    let updated = self.ins().insert(chunk, val, sub_idx);
                    self.ins().stack_store(updated, slot, chunk_off);
                    self.copy_bytes(
                        addr,
                        chunk_off + 16,
                        x,
                        chunk_off + 16,
                        from.bytes() - (chunk_off + 16),
                    );
                    return Ok(Some(addr));
                }
                self.copy_bytes(addr, 0, x, 0, from.bytes());
                self.ins()
                    .stack_store(val, slot, i32::from(idx) * from.size);
                Ok(Some(addr))
            }
            (Simd::Shuffle, &[x, y, _]) => self.wide_shuffle(origin, (*from, *to), (x, y)),
            _ => Ok(None),
        }
    }

    fn wide_shuffle(
        &mut self,
        origin: &SiteOrigin,
        (from, to): (Lanes, Lanes),
        (x, y): (Value, Value),
    ) -> Result<Option<Value>> {
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
        let Some(mut mask) =
            self.shuffle_const_mask_wide(origin, from.count as usize, to.count as usize)
        else {
            return Ok(None);
        };
        let (in_lanes, in_x, in_y) = if from.bytes() < 8 {
            let Some((wf, wf_ty)) = from.widened() else {
                return Ok(None);
            };
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
            let Some((wt, _)) = to.widened() else {
                return Ok(None);
            };
            (wt, Some(to.bytes()))
        } else {
            (to, None)
        };
        let chunk_bytes = if in_lanes.bytes() >= 16 && out_lanes.bytes() >= 16 {
            16
        } else {
            8
        };
        let chunk_lanes = (chunk_bytes / out_lanes.size) as usize;
        let Some(chunk_lane) = Lane::from_type(out_lanes.machine) else {
            return Ok(None);
        };
        let chunk_vec = Type::Vec {
            lane: chunk_lane,
            count: chunk_lanes as u8,
        };
        let int_vec = Type::Vec {
            lane: chunk_lane.int_mask(),
            count: chunk_lanes as u8,
        };
        let in_chunks = (in_lanes.bytes() / chunk_bytes) as usize;
        let out_chunks = (out_lanes.bytes() / chunk_bytes) as usize;
        let (slot, addr) = self.lanes_slot(out_lanes);
        for oc in 0..out_chunks {
            let sub = &mask[oc * chunk_lanes..(oc + 1) * chunk_lanes];
            let chunk = self.shuffle_chunk(
                (chunk_vec, int_vec),
                (chunk_bytes, chunk_lanes),
                in_chunks,
                (in_x, in_y),
                sub,
            );
            self.ins().stack_store(chunk, slot, oc as i32 * chunk_bytes);
        }
        if let Some(bytes) = narrow_bytes {
            let (_, out_addr) = self.lanes_slot(to);
            self.copy_bytes(out_addr, 0, addr, 0, bytes);
            return Ok(Some(out_addr));
        }
        Ok(Some(addr))
    }

    fn shuffle_chunk(
        &mut self,
        (chunk_vec, int_vec): (Type, Type),
        (chunk_bytes, chunk_lanes): (i32, usize),
        in_chunks: usize,
        (in_x, in_y): (Value, Value),
        sub: &[u8],
    ) -> Value {
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
                m[j] = if items[j].0 == cids[0] {
                    items[j].1
                } else {
                    chunk_lanes as u8 + items[j].1
                };
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
}
