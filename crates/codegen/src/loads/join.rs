use super::dom::{clobbers_loc, loc_immutable};
use super::{Assembled, State, assemble_const, emit_op};
use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Function, Inst, Lane, Op, Type, Value};
use crate::mem::{KnownLoads, MemBase, MemLoc, MemRoot, may_alias};
use crate::merge::find;

const MAX_JOIN_PHIS_PER_BLOCK: usize = 8;

pub(super) enum BlockClobber {
    Loc(MemLoc),
    CopyLoc(MemLoc),
    Op(Inst),
}

pub(super) struct JoinCtx<'a> {
    pub cfg: &'a Cfg,
    pub known_out: &'a [KnownLoads],
    pub def_blocks: &'a mut Vec<Option<Block>>,
    pub preds: Option<Vec<Block>>,
    pub added_phis: usize,
}

pub(super) fn int_of(ty: Type) -> Option<Type> {
    match ty {
        Type::F32 => Some(Type::I32),
        Type::F64 => Some(Type::I64),
        t if t.is_small_int() => Some(t),
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub(super) enum SubwordPlan {
    Int {
        kty: Type,
        kval: Value,
        target_int: Type,
        k_int: Type,
        shift: i64,
        need_reduce: bool,
        bitcast_out: bool,
    },
    VecBitcast {
        kval: Value,
    },
    VecExtract {
        kty: Type,
        kval: Value,
        lane_vec: Type,
        lane_idx: u8,
        out_vec: Option<Type>,
    },
}

impl SubwordPlan {
    pub(super) fn kval(self) -> Value {
        match self {
            Self::Int { kval, .. } | Self::VecBitcast { kval } | Self::VecExtract { kval, .. } => kval,
        }
    }

    pub(super) fn new(loc_offset: i64, ty: Type, koff: i64, kty: Type, kval: Value) -> Option<Self> {
        if koff == loc_offset && kty.bytes() == ty.bytes() && ty != kty && (ty.is_vector() || kty.is_vector()) {
            let ok = (ty.is_vector() && kty.is_vector())
                || (ty.bytes() == 8 && (matches!(ty, Type::I64 | Type::F64) || matches!(kty, Type::I64 | Type::F64)));
            return ok.then_some(Self::VecBitcast { kval });
        }
        if kty.is_vector() {
            let ty_end = loc_offset.checked_add(i64::from(ty.bytes()))?;
            let k_end = koff.checked_add(i64::from(kty.bytes()))?;
            if loc_offset < koff || ty_end > k_end {
                return None;
            }
            if let Some(lane) = Lane::from_type(ty) {
                let lbytes = i64::from(lane.bytes());
                if (loc_offset - koff) % lbytes != 0 {
                    return None;
                }
                let lane_vec = Type::Vec { lane, count: (kty.bytes() / lane.bytes()) as u8 };
                let lane_idx = ((loc_offset - koff) / lbytes) as u8;
                return Some(Self::VecExtract { kty, kval, lane_vec, lane_idx, out_vec: None });
            }
            if ty.is_vector() && ty.bytes() == 8 && kty.bytes() == 16 && (loc_offset - koff) % 8 == 0 {
                let lane_vec = Type::Vec { lane: Lane::I64, count: 2 };
                let lane_idx = ((loc_offset - koff) / 8) as u8;
                return Some(Self::VecExtract { kty, kval, lane_vec, lane_idx, out_vec: Some(ty) });
            }
            return None;
        }
        let target_int = int_of(ty)?;
        let k_int = int_of(kty)?;
        let shift = (loc_offset - koff) * 8;
        let need_shift = shift > 0;
        let need_reduce = ty.bytes() < kty.bytes();
        let bitcast_out = ty.is_float();
        if !(kty.is_float() || need_shift || need_reduce || bitcast_out) {
            return None;
        }
        Some(Self::Int { kty, kval, target_int, k_int, shift, need_reduce, bitcast_out })
    }

    pub(super) fn build_op(self, mut emit: impl FnMut(Type, Op) -> Value) -> Op {
        match self {
            Self::VecBitcast { kval } => Op::Convert(Convert::Bitcast, kval),
            Self::VecExtract { kty, kval, lane_vec, lane_idx, out_vec } => {
                let src = if kty == lane_vec { kval } else { emit(lane_vec, Op::Convert(Convert::Bitcast, kval)) };
                let extract = Op::Extract(src, lane_idx);
                if out_vec.is_some() {
                    let raw = emit(Type::I64, extract);
                    Op::Convert(Convert::Bitcast, raw)
                } else {
                    extract
                }
            }
            Self::Int { kty, kval, target_int, k_int, shift, need_reduce, bitcast_out } => {
                let mut cur = kval;
                let need_shift = shift > 0;
                if kty.is_float() {
                    let op = Op::Convert(Convert::Bitcast, cur);
                    if !need_shift && !need_reduce && !bitcast_out {
                        return op;
                    }
                    cur = emit(k_int, op);
                }
                if need_shift {
                    let sh = emit(k_int, Op::Const(shift));
                    let op = Op::Binary(Binary::Ushr, cur, sh);
                    if !need_reduce && !bitcast_out {
                        return op;
                    }
                    cur = emit(k_int, op);
                }
                if need_reduce {
                    let op = Op::Convert(Convert::Ireduce, cur);
                    if !bitcast_out {
                        return op;
                    }
                    cur = emit(target_int, op);
                }
                Op::Convert(Convert::Bitcast, cur)
            }
        }
    }
}

pub(super) fn find_subword_source(known: &KnownLoads, loc: MemLoc, ty: Type) -> Option<(i64, Type, Value)> {
    if !(int_of(ty).is_some() || ty.is_vector()) {
        return None;
    }
    let cell_end = loc.offset.checked_add(i64::from(ty.bytes()))?;
    known
        .entries_for(loc.base)
        .iter()
        .filter(|&&(off, kty, kval)| {
            off <= loc.offset
                && off.checked_add(i64::from(kty.bytes())).is_some_and(|end| cell_end <= end)
                && SubwordPlan::new(loc.offset, ty, off, kty, kval).is_some()
        })
        .min_by_key(|&&(off, kty, _)| (kty.bytes(), !kty.is_small_int(), loc.offset - off))
        .copied()
}

pub(super) fn rodata_const(func: &Function, loc: MemLoc, ty: Type) -> Option<i64> {
    let MemBase::Global(sym) = loc.base else { return None };
    if !(ty.is_small_int() || matches!(ty, Type::F32 | Type::F64)) {
        return None;
    }
    let start = usize::try_from(loc.offset).ok()?;
    let size = usize::try_from(ty.bytes()).ok()?;
    let (_, bytes) = func.rodata_bytes.iter().find(|&&(s, _)| s == sym)?;
    let mut raw = [0_u8; 8];
    raw[..size].copy_from_slice(bytes.get(start..start.checked_add(size)?)?);
    let bits = u64::from_le_bytes(raw) as i64;
    Some(if ty.is_small_int() { crate::fold::extend(bits, ty, true) } else { bits })
}

pub(super) fn const_bits_eq(ty: Type, ca: i64, cb: i64) -> bool {
    match ty {
        t if t.is_small_int() => (ca as u64 & t.unsigned_max()) == (cb as u64 & t.unsigned_max()),
        Type::F32 => (ca as u32) == (cb as u32),
        Type::F64 => ca == cb,
        _ => false,
    }
}

pub(super) fn const_eq_for_ty(consts: &[Option<i64>], ty: Type, a: Value, b: Value) -> bool {
    consts.get(a.index()).copied().flatten().zip(consts.get(b.index()).copied().flatten()).is_some_and(|(ca, cb)| const_bits_eq(ty, ca, cb))
}

pub(super) fn for_edge_args(op: &Op, target: Block, mut f: impl FnMut(&[Value])) -> bool {
    match op {
        Op::Jump(b, args) if *b == target => f(args),
        Op::Brif(_, t, ta, e, ea) => {
            if *t == target {
                f(ta);
            }
            if *e == target {
                f(ea);
            }
        }
        _ if op.targets().any(|t| t == target) => return false,
        _ => {}
    }
    true
}

pub(super) fn eligible_join_preds(func: &Function, cfg: &Cfg, visited: &[bool], block: Block) -> Option<Vec<Block>> {
    if Some(block) == func.entry {
        return None;
    }
    let param_len = func.blocks[block.index()].params.len();
    let mut unique = Vec::new();
    for &pred in cfg.preds_of(block) {
        if !cfg.reachable[pred.index()] || !visited[pred.index()] {
            return None;
        }
        if !unique.contains(&pred) {
            let (mut seen, mut ok) = (false, true);
            if !func.last(pred).is_some_and(|op| for_edge_args(op, block, |a| { seen = true; ok &= a.len() == param_len; }))
                || !seen
                || !ok
            {
                return None;
            }
            unique.push(pred);
        }
    }
    (2..=8).contains(&unique.len()).then_some(unique)
}

pub(super) fn for_edge_args_mut(func: &mut Function, pred: Block, target: Block, mut f: impl FnMut(&mut Vec<Value>)) {
    let Some(&last) = func.blocks[pred.index()].insts.last() else { return };
    match &mut func.insts[last.index()].op {
        Op::Jump(b, args) if *b == target => f(args),
        Op::Brif(_, t, ta, e, ea) => {
            if *t == target {
                f(ta);
            }
            if *e == target {
                f(ea);
            }
        }
        _ => {}
    }
}

#[derive(Clone, Copy)]
enum LocPlan {
    Exact(Value),
    Const(i64),
    Subword(SubwordPlan),
}

fn plan_loc(func: &Function, state: &State<'_>, known: &KnownLoads, loc: MemLoc, ty: Type) -> Option<LocPlan> {
    if let Some(v) = known.get(loc.base, loc.offset, ty) {
        return Some(LocPlan::Exact(v));
    }
    if let Some(imm) = rodata_const(func, loc, ty) {
        return Some(LocPlan::Const(imm));
    }
    if let Some(Assembled::Int(imm) | Assembled::FloatBits(imm)) = assemble_const(known, loc, ty, state.consts) {
        return Some(LocPlan::Const(imm));
    }
    let (koff, kty, kval) = find_subword_source(known, loc, ty)?;
    let kval = find(state.subst, kval);
    SubwordPlan::new(loc.offset, ty, koff, kty, kval).map(LocPlan::Subword)
}

fn materialize_plan(
    func: &mut Function,
    state: &mut State<'_>,
    ty: Type,
    plan: LocPlan,
    mut emit: impl FnMut(&mut Function, &mut State<'_>, Type, Op) -> Value,
) -> Value {
    match plan {
        LocPlan::Exact(v) => v,
        LocPlan::Const(imm) => emit(func, state, ty, Op::Const(imm)),
        LocPlan::Subword(sub) => {
            let final_op = sub.build_op(|op_ty, op| emit(func, state, op_ty, op));
            emit(func, state, ty, final_op)
        }
    }
}

fn emit_pred_inst(func: &mut Function, state: &mut State<'_>, def_blocks: &mut Vec<Option<Block>>, pred: Block, ty: Type, op: Op) -> Value {
    let val = func.value(ty);
    state.subst.push(val);
    state.consts.push(match op {
        Op::Const(imm) => Some(imm),
        _ => None,
    });
    def_blocks.push(Some(pred));
    let inst = func.add(op, Some(val));
    let insts = &mut func.blocks[pred.index()].insts;
    let pos = insts.len().saturating_sub(1);
    insts.insert(pos, inst);
    val
}

pub(super) fn resolve_select_load(
    func: &mut Function,
    state: &mut State<'_>,
    kept: &mut Vec<Option<Inst>>,
    known: &KnownLoads,
    ty: Type,
    (loc_a, loc_b): (MemLoc, MemLoc),
) -> Option<(Value, Value)> {
    let plan_a = plan_loc(func, state, known, loc_a, ty)?;
    let plan_b = plan_loc(func, state, known, loc_b, ty)?;
    let mut emit = |f: &mut Function, s: &mut State<'_>, op_ty: Type, op: Op| emit_op(f, s, kept, op_ty, op);
    let val_a = materialize_plan(func, state, ty, plan_a, &mut emit);
    let val_b = materialize_plan(func, state, ty, plan_b, &mut emit);
    Some((val_a, val_b))
}

fn join_clobbered(func: &Function, prov: &crate::mem::Provenance, loc: MemLoc, clobbers_seen: &[BlockClobber]) -> bool {
    let immut = loc_immutable(func, loc);
    clobbers_seen.iter().any(|c| match *c {
        BlockClobber::Loc(w) => {
            if immut {
                matches!(loc.root, MemRoot::ReadonlyParam(_)) && w.root == loc.root && may_alias(w, loc, prov)
            } else {
                may_alias(w, loc, prov)
            }
        }
        BlockClobber::CopyLoc(w) => {
            if immut {
                matches!(loc.root, MemRoot::ReadonlyParam(_)) && w.root == loc.root
            } else {
                w.base == loc.base || (w.root != MemRoot::Unknown && w.root == loc.root) || may_alias(w, loc, prov)
            }
        }
        BlockClobber::Op(c_inst) => clobbers_loc(func, &func.insts[c_inst.index()].op, loc, prov),
    })
}

pub(super) fn try_join_phi_forward(
    func: &mut Function,
    (block, inst, loc, ty, result): (Block, Inst, MemLoc, Type, Value),
    state: &mut State<'_>,
    known: &mut KnownLoads,
    join: &mut JoinCtx<'_>,
    clobbers_seen: &[BlockClobber],
) -> Option<Option<Value>> {
    let preds = join.preds.as_ref()?;
    if matches!(loc.base, MemBase::Addr(b) if !join.def_blocks.get(b.index()).copied().flatten().is_some_and(|d| d != block && join.cfg.dominates(d, block)))
        || join_clobbered(func, state.prov, loc, clobbers_seen)
    {
        return None;
    }
    let mut plans: Vec<(Block, LocPlan)> = Vec::with_capacity(preds.len());
    for &p in preds {
        let plan = match plan_loc(func, state, &join.known_out[p.index()], loc, ty)? {
            LocPlan::Exact(raw) => {
                let v = find(state.subst, raw);
                if !join.def_blocks.get(v.index()).copied().flatten().is_some_and(|d| join.cfg.dominates(d, p)) {
                    return None;
                }
                LocPlan::Exact(v)
            }
            LocPlan::Subword(sub) => {
                if !join.def_blocks.get(sub.kval().index()).copied().flatten().is_some_and(|d| join.cfg.dominates(d, p)) {
                    return None;
                }
                LocPlan::Subword(sub)
            }
            c @ LocPlan::Const(_) => c,
        };
        plans.push((p, plan));
    }
    if let (_, LocPlan::Exact(v0)) = plans[0]
        && plans.iter().all(|&(_, pl)| matches!(pl, LocPlan::Exact(v) if v == v0))
    {
        known.remember(loc, ty, v0);
        return Some(Some(v0));
    }
    let const_of = |pl: LocPlan| -> Option<i64> {
        match pl {
            LocPlan::Const(imm) => Some(imm),
            LocPlan::Exact(v) => state.consts.get(v.index()).copied().flatten(),
            LocPlan::Subword(_) => None,
        }
    };
    if let Some(imm0) = const_of(plans[0].1)
        && plans.iter().all(|&(_, pl)| const_of(pl).is_some_and(|c| const_bits_eq(ty, imm0, c)))
    {
        func.insts[inst.index()].op = Op::Const(imm0);
        state.consts[result.index()] = Some(imm0);
        known.remember(loc, ty, result);
        return Some(None);
    }
    if join.added_phis >= MAX_JOIN_PHIS_PER_BLOCK {
        return None;
    }
    let mut vals: Vec<(Block, Value)> = Vec::with_capacity(plans.len());
    let def_blocks = &mut *join.def_blocks;
    for (p, plan) in plans {
        let mut emit = |f: &mut Function, s: &mut State<'_>, op_ty: Type, op: Op| emit_pred_inst(f, s, def_blocks, p, op_ty, op);
        let val = materialize_plan(func, state, ty, plan, &mut emit);
        vals.push((p, val));
    }
    let phi = func.value(ty);
    func.blocks[block.index()].params.push(phi);
    state.subst.push(phi);
    state.consts.push(None);
    def_blocks.resize(func.types.len(), Some(block));
    join.added_phis += 1;
    vals.into_iter().for_each(|(p, v)| for_edge_args_mut(func, p, block, |a| a.push(v)));
    known.remember(loc, ty, phi);
    Some(Some(phi))
}
