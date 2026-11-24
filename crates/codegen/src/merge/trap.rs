use rustc_hash::FxHashMap;

use crate::ir::{
    Binary, Block, Convert, FloatCC, Function, Inst, IntCC, Op, StackSlot, Type, Unary, Value,
};

const MAX_TRAP_BLOCK_INSTS: usize = 24;

#[derive(Clone, Copy)]
enum LeafKind {
    Trap = 0,
    EmptyReturn = 1,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ValKey {
    Local(u16, Type),
    Entry(Value),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum StepKey {
    Const(i64),
    SlotAddr(u32),
    Symbol(u32),
    Runtime(u8),
    Unary(Unary, ValKey),
    Binary(Binary, ValKey, ValKey),
    Convert(Convert, ValKey),
    Icmp(IntCC, ValKey, ValKey),
    Fcmp(FloatCC, ValKey, ValKey),
    Select(ValKey, ValKey, ValKey),
    Load(ValKey, i32),
    PrivAddr(u16, i32),
    PrivLoad(u16, i32),
    PrivStore(ValKey, u16, i32),
    PrivPtrStore(ValKey, ValKey, i32),
    Call {
        sig_idx: usize,
        callee: ValKey,
        args: Vec<ValKey>,
    },
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct TrapBlockKey {
    slots: Vec<(u32, u8)>,
    steps: Vec<(Option<Type>, u8, StepKey)>,
}

fn leaf_kind(func: &Function, block: usize) -> Option<LeafKind> {
    let data = &func.blocks[block];
    if !data.params.is_empty() {
        return None;
    }
    let [inst] = data.insts[..] else { return None };
    match &func.insts[inst.index()].op {
        Op::Trap => Some(LeafKind::Trap),
        Op::Return(vals) if vals.is_empty() => Some(LeafKind::EmptyReturn),
        _ => None,
    }
}

fn redirect_leaf_target(
    target: &mut Block,
    empty_args: bool,
    dup_canon: &[Option<Block>],
    kept_dup: &mut [bool],
) -> bool {
    let idx = target.index();
    let Some(canon) = dup_canon[idx] else {
        return false;
    };
    if empty_args {
        *target = canon;
        return true;
    }
    kept_dup[idx] = true;
    false
}

fn redirect_leaf_op(op: &mut Op, dup_canon: &[Option<Block>], kept_dup: &mut [bool]) -> bool {
    match op {
        Op::Jump(target, args) => {
            redirect_leaf_target(target, args.is_empty(), dup_canon, kept_dup)
        }
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            let a = redirect_leaf_target(then, then_args.is_empty(), dup_canon, kept_dup);
            let b = redirect_leaf_target(otherwise, else_args.is_empty(), dup_canon, kept_dup);
            a || b
        }
        Op::BrTable(_, default, targets) => {
            let mut did = redirect_leaf_target(default, true, dup_canon, kept_dup);
            for target in targets {
                did |= redirect_leaf_target(target, true, dup_canon, kept_dup);
            }
            did
        }
        Op::TryCall { normal, .. } => redirect_leaf_target(normal, true, dup_canon, kept_dup),
        _ => false,
    }
}

fn apply_redirects(func: &mut Function, dup_canon: Vec<Option<Block>>) -> bool {
    let mut changed = false;
    let mut kept_dup = vec![false; func.blocks.len()];
    for inst in &mut func.insts {
        changed |= redirect_leaf_op(&mut inst.op, &dup_canon, &mut kept_dup);
    }
    for (block, canon) in dup_canon.into_iter().enumerate() {
        if canon.is_some() && !kept_dup[block] {
            for inst in std::mem::take(&mut func.blocks[block].insts) {
                func.insts[inst.index()].op = Op::Trap;
                func.insts[inst.index()].result = None;
            }
            changed = true;
        }
    }
    changed
}

pub(super) fn merge_leaves(func: &mut Function) -> bool {
    let entry = func.entry.map(Block::index);
    let mut canonical = [None; 2];
    let mut dup_canon = vec![None; func.blocks.len()];
    let mut has_dup = false;
    for block in 0..func.blocks.len() {
        if Some(block) != entry
            && let Some(kind) = leaf_kind(func, block)
        {
            let slot = &mut canonical[kind as usize];
            match *slot {
                None => *slot = Some(Block(block as u32)),
                Some(canon) => {
                    dup_canon[block] = Some(canon);
                    has_dup = true;
                }
            }
        }
    }
    has_dup && apply_redirects(func, dup_canon)
}

fn note_slot_block(owner: &mut Option<usize>, shared: &mut bool, b: usize) {
    if b == usize::MAX || matches!(*owner, Some(prev) if prev != b) {
        *shared = true;
    } else if owner.is_none() {
        *owner = Some(b);
    }
}

fn private_slots_and_uses(func: &Function) -> (Vec<u32>, Vec<Option<usize>>) {
    let mut all_uses = vec![0u32; func.types.len()];
    let mut slot_owner = vec![None; func.slots.len()];
    let mut slot_shared = vec![false; func.slots.len()];
    let mut in_block = vec![usize::MAX; func.insts.len()];
    for (b, blk) in func.blocks.iter().enumerate() {
        blk.insts
            .iter()
            .for_each(|&inst| in_block[inst.index()] = b);
    }
    for (idx, data) in func.insts.iter().enumerate() {
        data.op.for_operands(|v| all_uses[v.index()] += 1);
        if let Op::StackAddr(s, _) | Op::StackLoad(s, _) | Op::StackStore(_, s, _) = data.op {
            note_slot_block(
                &mut slot_owner[s.index()],
                &mut slot_shared[s.index()],
                in_block[idx],
            );
        }
    }
    for (s, shared) in slot_shared.into_iter().enumerate() {
        if shared {
            slot_owner[s] = None;
        }
    }
    (all_uses, slot_owner)
}

fn entry_defined_values(func: &Function) -> Vec<bool> {
    let mut entry_vals = vec![false; func.types.len()];
    let Some(entry) = func.entry else {
        return entry_vals;
    };
    let e = entry.index();
    func.blocks[e]
        .params
        .iter()
        .for_each(|&p| entry_vals[p.index()] = true);
    for &inst in &func.blocks[e].insts {
        if let Some(res) = func.insts[inst.index()].result {
            entry_vals[res.index()] = true;
        }
    }
    entry_vals
}

struct BlockEnv<'a> {
    func: &'a Function,
    block: usize,
    entry_vals: &'a [bool],
    slot_owner: &'a [Option<usize>],
    sig_canon: &'a [usize],
    locals: Vec<(Value, ValKey, bool)>,
    slots: Vec<(StackSlot, (u32, u8))>,
}

impl BlockEnv<'_> {
    fn val(&self, v: Value) -> Option<ValKey> {
        if let Some(&(_, key, _)) = self.locals.iter().find(|&&(lv, _, _)| lv == v) {
            return Some(key);
        }
        self.entry_vals[v.index()].then_some(ValKey::Entry(v))
    }

    fn is_priv_ptr(&self, v: Value) -> bool {
        self.locals
            .iter()
            .find(|&&(lv, _, _)| lv == v)
            .is_some_and(|&(_, _, p)| p)
    }

    fn pair(&self, x: Value, y: Value) -> Option<(ValKey, ValKey)> {
        Some((self.val(x)?, self.val(y)?))
    }

    fn slot(&mut self, s: StackSlot) -> Option<u16> {
        if self.slot_owner[s.index()] != Some(self.block) {
            return None;
        }
        if let Some(pos) = self.slots.iter().position(|&(prev, _)| prev == s) {
            return Some(pos as u16);
        }
        let idx = u16::try_from(self.slots.len()).ok()?;
        let data = &self.func.slots[s.index()];
        self.slots.push((s, (data.size, data.align_shift)));
        Some(idx)
    }

    fn step_pure(&self, op: &Op) -> Option<(StepKey, bool)> {
        match *op {
            Op::Const(imm) => Some((StepKey::Const(imm), false)),
            Op::SlotAddr(sym) => Some((StepKey::SlotAddr(sym), false)),
            Op::Symbol(sym) => Some((StepKey::Symbol(sym), false)),
            Op::Runtime(rt) => Some((StepKey::Runtime(rt as u8), false)),
            Op::Unary(u, x) => self.val(x).map(|kx| (StepKey::Unary(u, kx), false)),
            Op::Convert(c, x) => self.val(x).map(|kx| (StepKey::Convert(c, kx), false)),
            Op::Binary(b, x, y) if op.is_speculable() => {
                let p = b == Binary::Iadd && (self.is_priv_ptr(x) || self.is_priv_ptr(y));
                self.pair(x, y)
                    .map(|(kx, ky)| (StepKey::Binary(b, kx, ky), p))
            }
            _ => None,
        }
    }

    fn step_cmp_or_mem(&mut self, op: &Op) -> Option<(StepKey, bool)> {
        match *op {
            Op::Icmp(cc, x, y) => self
                .pair(x, y)
                .map(|(kx, ky)| (StepKey::Icmp(cc, kx, ky), false)),
            Op::Fcmp(cc, x, y) => self
                .pair(x, y)
                .map(|(kx, ky)| (StepKey::Fcmp(cc, kx, ky), false)),
            Op::Select(c, x, y) => self
                .val(c)
                .zip(self.pair(x, y))
                .map(|(kc, (kx, ky))| (StepKey::Select(kc, kx, ky), false)),
            Op::Load(addr, off) => self.val(addr).map(|ka| (StepKey::Load(ka, off), false)),
            Op::StackAddr(s, off) => self.slot(s).map(|ks| (StepKey::PrivAddr(ks, off), true)),
            Op::StackLoad(s, off) => self.slot(s).map(|ks| (StepKey::PrivLoad(ks, off), false)),
            Op::StackStore(v, s, off) => self.val(v).and_then(|kv| {
                self.slot(s)
                    .map(|ks| (StepKey::PrivStore(kv, ks, off), false))
            }),
            Op::Store(v, addr, off) if self.is_priv_ptr(addr) => self
                .pair(v, addr)
                .map(|(kv, ka)| (StepKey::PrivPtrStore(kv, ka, off), false)),
            _ => None,
        }
    }

    fn step_op(&mut self, op: &Op) -> Option<(StepKey, bool)> {
        self.step_pure(op).or_else(|| self.step_cmp_or_mem(op))
    }
}

fn trap_call_candidate(
    func: &Function,
    block: usize,
    entry: Option<usize>,
) -> Option<(&[Inst], Inst)> {
    if Some(block) == entry || !func.blocks[block].params.is_empty() {
        return None;
    }
    let insts = &func.blocks[block].insts;
    if !(2..=MAX_TRAP_BLOCK_INSTS).contains(&insts.len()) {
        return None;
    }
    let (&trap_inst, prefix) = insts.split_last()?;
    let (&call_inst, setup) = prefix.split_last()?;
    let trap_ok = matches!(func.insts[trap_inst.index()].op, Op::Trap)
        && func.insts[trap_inst.index()].result.is_none();
    let call_ok = matches!(&func.insts[call_inst.index()].op, Op::Call { results, .. } if results.is_empty())
        && func.insts[call_inst.index()].result.is_none();
    (trap_ok && call_ok).then_some((setup, call_inst))
}

fn locals_self_contained(
    func: &Function,
    setup: &[Inst],
    call_inst: Inst,
    locals: &[(Value, ValKey, bool)],
    all_uses: &[u32],
) -> bool {
    locals.iter().all(|&(lv, _, _)| {
        let mut inside = 0u32;
        for &inst in setup.iter().chain(std::iter::once(&call_inst)) {
            func.insts[inst.index()]
                .op
                .for_operands(|v| inside += u32::from(v == lv));
        }
        all_uses[lv.index()] == inside
    })
}

fn build_trap_key(
    func: &Function,
    block: usize,
    (setup, call_inst): (&[Inst], Inst),
    (all_uses, entry_vals, slot_owner, sig_canon): (&[u32], &[bool], &[Option<usize>], &[usize]),
) -> Option<TrapBlockKey> {
    let mut env = BlockEnv {
        func,
        block,
        entry_vals,
        slot_owner,
        sig_canon,
        locals: Vec::with_capacity(setup.len()),
        slots: Vec::new(),
    };
    let mut steps = Vec::with_capacity(setup.len() + 1);
    for &inst in setup {
        let data = &func.insts[inst.index()];
        let (step, is_priv_ptr) = env.step_op(&data.op)?;
        let (res_ty, flags) = match data.result {
            Some(res) => {
                let ty = func.ty(res);
                let idx = u16::try_from(env.locals.len()).ok()?;
                env.locals.push((res, ValKey::Local(idx, ty), is_priv_ptr));
                (
                    Some(ty),
                    func.value_flags.get(res.index()).copied().unwrap_or(0),
                )
            }
            None => (None, 0),
        };
        steps.push((res_ty, flags, step));
    }
    let Op::Call {
        sig,
        callee,
        ref args,
        ..
    } = func.insts[call_inst.index()].op
    else {
        return None;
    };
    let callee_k = env.val(callee)?;
    let args_k: Vec<ValKey> = args.iter().map(|&a| env.val(a)).collect::<Option<_>>()?;
    steps.push((
        None,
        0,
        StepKey::Call {
            sig_idx: env.sig_canon[sig.index()],
            callee: callee_k,
            args: args_k,
        },
    ));
    locals_self_contained(func, setup, call_inst, &env.locals, all_uses).then(|| TrapBlockKey {
        slots: env.slots.into_iter().map(|(_, meta)| meta).collect(),
        steps,
    })
}

pub(super) fn merge_trap_calls(func: &mut Function) -> bool {
    if func.coroutine.is_some() {
        return false;
    }
    let entry = func.entry.map(Block::index);
    let cands: Vec<usize> = (0..func.blocks.len())
        .filter(|&b| trap_call_candidate(func, b, entry).is_some())
        .collect();
    if cands.len() < 2 {
        return false;
    }
    let mut sig_map = FxHashMap::default();
    let sig_canon: Vec<usize> = func
        .sigs
        .iter()
        .enumerate()
        .map(|(i, s)| *sig_map.entry(s).or_insert(i))
        .collect();
    let (all_uses, slot_owner) = private_slots_and_uses(func);
    let entry_vals = entry_defined_values(func);
    let mut canonical: FxHashMap<TrapBlockKey, Block> = FxHashMap::default();
    let mut dup_canon = vec![None; func.blocks.len()];
    let mut has_dup = false;
    for b in cands {
        let Some(pair) = trap_call_candidate(func, b, entry) else {
            continue;
        };
        let Some(key) = build_trap_key(
            func,
            b,
            pair,
            (&all_uses, &entry_vals, &slot_owner, &sig_canon),
        ) else {
            continue;
        };
        match canonical.get(&key) {
            None => drop(canonical.insert(key, Block(b as u32))),
            Some(&canon) => {
                dup_canon[b] = Some(canon);
                has_dup = true;
            }
        }
    }
    has_dup && apply_redirects(func, dup_canon)
}
