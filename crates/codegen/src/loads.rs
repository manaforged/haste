//! Reuses what memory holds along straight-line code: a load of the bytes an earlier load or store on the same path read or wrote, with no write of unknown reach between them, takes that value.

use crate::ir::{Binary, Block, Function, Inst, Op, StackSlot, Type, Value};
use crate::merge::{find, identity, pred_counts, rename};

/// The most cells one path remembers, so a long block costs linear time.
const CELLS: usize = 64;

/// Where remembered bytes start: a frame slot, or an address held in a value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Base {
    Slot(StackSlot),
    Addr(Value),
}

/// Bytes of memory read or written as one value of `ty`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Cell {
    base: Base,
    offset: i64,
    ty: Type,
}

impl Cell {
    fn overlaps(&self, other: &Cell) -> bool {
        self.offset < other.offset + i64::from(other.ty.bytes()) && other.offset < self.offset + i64::from(self.ty.bytes())
    }

    /// Whether a write of `self` may change the bytes of `other`: frame slots never overlap each other, an unescaped slot never aliases an address.
    fn clobbers(&self, other: &Cell, unescaped: &[bool]) -> bool {
        match (self.base, other.base) {
            (Base::Slot(a), Base::Slot(b)) => a == b && self.overlaps(other),
            (Base::Addr(a), Base::Addr(b)) if a == b => self.overlaps(other),
            (Base::Addr(_), Base::Slot(b)) | (Base::Slot(b), Base::Addr(_)) => !unescaped[b.index()],
            (Base::Addr(_), Base::Addr(_)) => true,
        }
    }
}

/// What memory holds along one path, by cell.
#[derive(Clone, Default)]
struct Known(Vec<(Cell, Value)>);

impl Known {
    fn get(&self, cell: &Cell) -> Option<Value> {
        self.0.iter().find(|(known, _)| known == cell).map(|(_, value)| *value)
    }

    fn remember(&mut self, cell: Cell, value: Value) {
        if self.0.len() < CELLS {
            self.0.push((cell, value));
        }
    }

    fn assemble_const(&self, cell: &Cell, consts: &[Option<i64>]) -> Option<i64> {
        let Base::Slot(slot) = cell.base else { return None };
        let width = i64::from(cell.ty.bytes());
        if !cell.ty.is_small_int() || !matches!(width, 2 | 4 | 8) {
            return None;
        }
        let mut assembled = 0_u64;
        for byte in 0..width {
            let at = cell.offset + byte;
            let (kcell, kval) = self.0.iter().find(|(c, _)| {
                c.base == Base::Slot(slot) && c.ty.is_small_int() && c.offset <= at && at < c.offset + i64::from(c.ty.bytes())
            })?;
            let imm = consts[kval.index()]?;
            let shift = 8 * (at - kcell.offset);
            let piece = ((imm as u64) >> shift) & 0xFF;
            assembled |= piece << (8 * byte);
        }
        Some(assembled as i64)
    }

    fn write(&mut self, cell: Cell, value: Value, unescaped: &[bool]) {
        self.0.retain(|(known, _)| !cell.clobbers(known, unescaped));
        self.remember(cell, value);
    }

    fn clobber_slot(&mut self, slot: StackSlot, offset: i64, bytes: i64, unescaped: &[bool]) {
        let end = offset + bytes;
        let slot_escaped = !unescaped[slot.index()];
        self.0.retain(|(known, _)| match known.base {
            Base::Slot(s) if s == slot => !(known.offset < end && offset < known.offset + i64::from(known.ty.bytes())),
            Base::Addr(_) if slot_escaped => false,
            _ => true,
        });
    }
}

/// What one instruction does to memory.
enum Effect {
    None,
    Read(Cell, Value),
    Write(Cell, Value),
    ClobberSlot { slot: StackSlot, offset: i64, bytes: i64 },
    Unknown,
}

/// Whether `op` may write memory it does not name by a plain address, or orders accesses around it.
pub(crate) fn clobbers(op: &Op) -> bool {
    matches!(op, Op::AtomicLoad(_) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence | Op::Copy(..) | Op::Move(..) | Op::Call { .. } | Op::TryCall { .. })
}

struct State<'a> {
    slots: &'a [Option<(StackSlot, i32)>],
    affine: &'a [(Value, i64)],
    unescaped: &'a [bool],
    call_readonly: &'a [bool],
    selects: &'a [Option<(Value, Value, Value)>],
    consts: &'a mut [Option<i64>],
    subst: &'a mut [Value],
}

fn addr_cell(slots: &[Option<(StackSlot, i32)>], affine: &[(Value, i64)], subst: &[Value], addr: Value, offset: i32, ty: Type) -> Cell {
    match slots[addr.index()] {
        Some((slot, base)) => Cell { base: Base::Slot(slot), offset: i64::from(base) + i64::from(offset), ty },
        None => {
            let resolved = find(subst, addr);
            let (root, base_off) = affine[resolved.index()];
            match base_off.checked_add(i64::from(offset)) {
                Some(total) => Cell { base: Base::Addr(find(subst, root)), offset: total, ty },
                None => Cell { base: Base::Addr(resolved), offset: i64::from(offset), ty },
            }
        }
    }
}

fn effect(func: &Function, state: &State<'_>, inst: Inst) -> Effect {
    let data = &func.insts[inst.index()];
    let slot = |slot: StackSlot, offset: i32, ty: Type| Cell { base: Base::Slot(slot), offset: i64::from(offset), ty };
    match (&data.op, data.result) {
        (&Op::Load(addr, offset), Some(result)) => Effect::Read(addr_cell(state.slots, state.affine, state.subst, addr, offset, func.ty(result)), result),
        (&Op::StackLoad(from, offset), Some(result)) => Effect::Read(slot(from, offset, func.ty(result)), result),
        (&Op::Store(value, addr, offset), _) => Effect::Write(addr_cell(state.slots, state.affine, state.subst, addr, offset, func.ty(value)), find(state.subst, value)),
        (&Op::StackStore(value, to, offset), _) => Effect::Write(slot(to, offset, func.ty(value)), find(state.subst, value)),
        (&Op::Move(to, _, bytes), _) => match state.slots[to.index()] {
            Some((slot, base)) => Effect::ClobberSlot { slot, offset: i64::from(base), bytes: i64::from(bytes) },
            None => Effect::Unknown,
        },
        (op, _) if clobbers(op) => Effect::Unknown,
        _ => Effect::None,
    }
}

fn flush_reads(pending: &mut Vec<(StackSlot, i64, i64, usize)>, slot: StackSlot, offset: i64, bytes: i64) {
    let end = offset + bytes;
    pending.retain(|&(s, off, len, _)| s != slot || !(off < end && offset < off + len));
}

/// Drops from `block` each load whose bytes `known` holds, recording its replacement in `subst`, eliminates dead overwritten stores to unescaped slots, and leaves in `known` what memory holds at the block's end.
fn scan(func: &mut Function, block: Block, state: &mut State<'_>, known: &mut Known) -> bool {
    let insts = std::mem::take(&mut func.blocks[block.index()].insts);
    let mut kept: Vec<Option<Inst>> = Vec::with_capacity(insts.len());
    let mut pending_stores: Vec<(StackSlot, i64, i64, usize)> = Vec::new();
    let mut renamed = false;
    for inst in insts {
        match func.insts[inst.index()].op {
            Op::Move(_, from, bytes) => {
                if let Some((slot, base)) = state.slots[from.index()] {
                    flush_reads(&mut pending_stores, slot, i64::from(base), i64::from(bytes));
                }
            }
            Op::TryCall { .. } => pending_stores.clear(),
            _ => {}
        }
        match effect(func, state, inst) {
            Effect::Read(cell, result) => match known.get(&cell) {
                Some(value) => {
                    state.subst[result.index()] = value;
                    renamed = true;
                    continue;
                }
                None => {
                    if let Some(imm) = known.assemble_const(&cell, state.consts) {
                        func.insts[inst.index()].op = Op::Const(imm);
                        state.consts[result.index()] = Some(imm);
                        known.remember(cell, result);
                    } else if let Op::Load(addr, offset) = func.insts[inst.index()].op
                        && let Some((c, ptr_a, ptr_b)) = state.selects[find(state.subst, addr).index()]
                        && let (Some(val_a), Some(val_b)) = (
                            known.get(&addr_cell(state.slots, state.affine, state.subst, ptr_a, offset, cell.ty)),
                            known.get(&addr_cell(state.slots, state.affine, state.subst, ptr_b, offset, cell.ty)),
                        )
                    {
                        func.insts[inst.index()].op = Op::Select(find(state.subst, c), val_a, val_b);
                        known.remember(cell, result);
                    } else {
                        known.remember(cell, result);
                        if let Base::Slot(s) = cell.base {
                            flush_reads(&mut pending_stores, s, cell.offset, i64::from(cell.ty.bytes()));
                        }
                    }
                }
            },
            Effect::Write(cell, value) => {
                known.write(cell, value, state.unescaped);
                if let Base::Slot(s) = cell.base
                    && state.unescaped[s.index()]
                {
                    let bytes = i64::from(cell.ty.bytes());
                    let end = cell.offset + bytes;
                    pending_stores.retain(|&(ps, off, len, kept_idx)| {
                        if ps == s && off >= cell.offset && off + len <= end {
                            if let Some(dead_store) = kept[kept_idx].take() {
                                func.insts[dead_store.index()].op = Op::Trap;
                            }
                            false
                        } else {
                            true
                        }
                    });
                    pending_stores.push((s, cell.offset, bytes, kept.len()));
                }
            }
            Effect::ClobberSlot { slot, offset, bytes } => known.clobber_slot(slot, offset, bytes, state.unescaped),
            Effect::Unknown => {
                let is_call = matches!(func.insts[inst.index()].op, Op::Call { .. } | Op::TryCall { .. });
                known.0.retain(|(c, _)| matches!(c.base, Base::Slot(s) if state.unescaped[s.index()] || (is_call && state.call_readonly[s.index()])));
            }
            Effect::None => {}
        }
        kept.push(Some(inst));
    }
    func.blocks[block.index()].insts = kept.into_iter().flatten().collect();
    renamed
}

/// The frame slot and offset each value holds the address of, when an instruction computes it as one.
pub(crate) fn slot_addresses(func: &Function) -> Vec<Option<(StackSlot, i32)>> {
    let mut consts = vec![None; func.types.len()];
    let mut slots = vec![None; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        match (&data.op, data.result) {
            (&Op::Const(imm), Some(result)) => consts[result.index()] = Some(imm),
            (&Op::StackAddr(slot, offset), Some(result)) => slots[result.index()] = Some((slot, offset)),
            _ => {}
        }
    }
    loop {
        let mut changed = false;
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            let data = &func.insts[inst.index()];
            let Some(result) = data.result else { continue };
            if slots[result.index()].is_some() {
                continue;
            }
            if let Op::Binary(Binary::Iadd, a, b) = data.op {
                let derived = [(a, b), (b, a)].into_iter().find_map(|(ptr, off)| {
                    let (slot, base) = slots[ptr.index()]?;
                    let c = i32::try_from(consts[off.index()]?).ok()?;
                    Some((slot, base.checked_add(c)?))
                });
                if let Some(val) = derived {
                    slots[result.index()] = Some(val);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    slots
}

fn slot_use_counts(func: &Function, slots: &[Option<(StackSlot, i32)>]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let count = func.types.len();
    let (mut total, mut safe, mut readonly_calls) = (vec![0_u32; count], vec![0_u32; count], vec![0_u32; count]);
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        data.op.for_operands(|value| total[value.index()] += 1);
        match &data.op {
            Op::Load(addr, _) => safe[addr.index()] += 1,
            Op::Store(val, addr, _) if val != addr => safe[addr.index()] += 1,
            Op::Move(to, from, _) => {
                safe[to.index()] += 1;
                safe[from.index()] += 1;
            }
            Op::Binary(Binary::Iadd, a, b) if data.result.is_some_and(|r| slots[r.index()].is_some()) => {
                if slots[a.index()].is_some() {
                    safe[a.index()] += 1;
                } else if slots[b.index()].is_some() {
                    safe[b.index()] += 1;
                }
            }
            Op::Call { sig, args, .. } | Op::TryCall { sig, args, .. } => {
                let mask = func.sigs[sig.index()].readonly_params;
                for (i, arg) in args.iter().enumerate() {
                    if i < 64 && ((mask >> i) & 1) != 0 {
                        readonly_calls[arg.index()] += 1;
                    }
                }
            }
            _ => {}
        }
    }
    (total, safe, readonly_calls)
}

/// For each stack slot, `(unescaped, call_readonly)`: never escaped at all, or passed outside direct memory ops only to `readonly_params` calls.
pub(crate) fn classify_slots(func: &Function, slots: &[Option<(StackSlot, i32)>]) -> (Vec<bool>, Vec<bool>) {
    let (total, safe, readonly_calls) = slot_use_counts(func, slots);
    let mut unescaped = vec![true; func.slots.len()];
    let mut call_readonly = vec![true; func.slots.len()];
    for (v, slot) in slots.iter().enumerate() {
        let Some((s, _)) = *slot else { continue };
        if safe[v] != total[v] {
            unescaped[s.index()] = false;
        }
        if safe[v] + readonly_calls[v] != total[v] {
            call_readonly[s.index()] = false;
        }
    }
    (unescaped, call_readonly)
}

fn affine_table(func: &Function) -> Vec<(Value, i64)> {
    let count = func.types.len();
    let mut consts = vec![None; count];
    let mut out: Vec<(Value, i64)> = (0..count as u32).map(|i| (Value(i), 0)).collect();
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        if let (&Op::Const(imm), Some(res)) = (&data.op, data.result) {
            consts[res.index()] = Some(imm);
        }
    }
    for _ in 0..8 {
        let mut changed = false;
        for inst in func.blocks.iter().flat_map(|b| &b.insts) {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if func.ty(res) != Type::I64 {
                continue;
            }
            let cand = match data.op {
                Op::Binary(Binary::Iadd, a, b) => {
                    if let Some(c) = consts[b.index()] {
                        (out[a.index()].0, out[a.index()].1.wrapping_add(c))
                    } else if let Some(c) = consts[a.index()] {
                        (out[b.index()].0, out[b.index()].1.wrapping_add(c))
                    } else {
                        continue;
                    }
                }
                Op::Binary(Binary::Isub, a, b) => {
                    let Some(c) = consts[b.index()] else { continue };
                    (out[a.index()].0, out[a.index()].1.wrapping_sub(c))
                }
                _ => continue,
            };
            if cand != out[res.index()] {
                out[res.index()] = cand;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    out
}

fn eliminate_affine_block_params(func: &mut Function) {
    let Some(entry) = func.entry else { return };
    let affine = affine_table(func);
    let mut subst = identity(func.types.len());
    let mut any_changed = false;
    for b_idx in 0..func.blocks.len() {
        let target = Block(b_idx as u32);
        if target == entry || func.blocks[b_idx].params.len() < 2 {
            continue;
        }
        let mut incoming: Vec<Vec<Value>> = Vec::new();
        let mut non_branch_pred = false;
        for pred in &func.blocks {
            let Some(&last) = pred.insts.last() else { continue };
            let op = &func.insts[last.index()].op;
            match op {
                Op::Jump(b, args) if *b == target => incoming.push(args.clone()),
                Op::Brif(_, t, t_args, e, e_args) => {
                    if *t == target {
                        incoming.push(t_args.clone());
                    }
                    if *e == target {
                        incoming.push(e_args.clone());
                    }
                }
                _ if op.targets().any(|t| t == target) => {
                    non_branch_pred = true;
                    break;
                }
                _ => {}
            }
        }
        let params = func.blocks[b_idx].params.clone();
        if non_branch_pred || incoming.is_empty() || incoming.iter().any(|args| args.len() != params.len()) {
            continue;
        }
        let mut eliminated = vec![false; params.len()];
        let mut prepended = Vec::new();
        for i in 1..params.len() {
            if func.ty(params[i]) != Type::I64 {
                continue;
            }
            for j in 0..i {
                if eliminated[j] || func.ty(params[j]) != Type::I64 {
                    continue;
                }
                let (r0_i, o0_i) = affine[incoming[0][i].index()];
                let (r0_j, o0_j) = affine[incoming[0][j].index()];
                if r0_i != r0_j {
                    continue;
                }
                let delta = o0_i.wrapping_sub(o0_j);
                let all_match = incoming.iter().skip(1).all(|args| {
                    let (ri, oi) = affine[args[i].index()];
                    let (rj, oj) = affine[args[j].index()];
                    ri == rj && oi.wrapping_sub(oj) == delta
                });
                if !all_match {
                    continue;
                }
                eliminated[i] = true;
                if delta == 0 {
                    subst[params[i].index()] = params[j];
                } else {
                    let c_val = func.value(Type::I64);
                    let c_inst = func.add(Op::Const(delta), Some(c_val));
                    let derived = func.value(Type::I64);
                    let d_inst = func.add(Op::Binary(Binary::Iadd, params[j], c_val), Some(derived));
                    subst.push(c_val);
                    subst.push(derived);
                    subst[params[i].index()] = derived;
                    prepended.push(c_inst);
                    prepended.push(d_inst);
                }
                break;
            }
        }
        if !eliminated.iter().any(|&e| e) {
            continue;
        }
        any_changed = true;
        if !prepended.is_empty() {
            prepended.append(&mut func.blocks[b_idx].insts);
            func.blocks[b_idx].insts = prepended;
        }
        let mut idx = 0;
        func.blocks[b_idx].params.retain(|_| {
            let keep = !eliminated[idx];
            idx += 1;
            keep
        });
        for pred_idx in 0..func.blocks.len() {
            let Some(&last) = func.blocks[pred_idx].insts.last() else { continue };
            let retain_args = |args: &mut Vec<Value>| {
                let mut k = 0;
                args.retain(|_| {
                    let keep = !eliminated[k];
                    k += 1;
                    keep
                });
            };
            match &mut func.insts[last.index()].op {
                Op::Jump(b, args) if *b == target => retain_args(args),
                Op::Brif(_, t, t_args, e, e_args) => {
                    if *t == target {
                        retain_args(t_args);
                    }
                    if *e == target {
                        retain_args(e_args);
                    }
                }
                _ => {}
            }
        }
    }
    if any_changed {
        rename(func, &subst);
    }
}

fn fits_mem_imm(ty: Type, off: i64) -> bool {
    let fits_one = |align: i64, o: i64| (-256..=255).contains(&o) || (o >= 0 && o % align == 0 && o <= align * 4095);
    if ty == Type::I128 {
        return off.checked_add(8).is_some_and(|hi| fits_one(8, off) && fits_one(8, hi));
    }
    let align = i64::from(ty.bytes().max(1));
    fits_one(align, off)
}

fn fold_affine_mem(func: &mut Function, slots: &[Option<(StackSlot, i32)>], affine: &[(Value, i64)]) {
    let mut is_stack_addr = vec![false; func.types.len()];
    for data in &func.insts {
        if let (Op::StackAddr(..), Some(res)) = (&data.op, data.result) {
            is_stack_addr[res.index()] = true;
        }
    }
    for block in &func.blocks {
        for &inst in &block.insts {
            let res_ty = func.insts[inst.index()].result.map(|r| func.ty(r));
            let val_ty = match func.insts[inst.index()].op {
                Op::Load(..) => res_ty,
                Op::Store(val, ..) => Some(func.ty(val)),
                _ => None,
            };
            let Some(ty) = val_ty else { continue };
            let (Op::Load(addr, off) | Op::Store(_, addr, off)) = &mut func.insts[inst.index()].op else { continue };
            if slots[addr.index()].is_some() {
                continue;
            }
            let (root, base_off) = affine[addr.index()];
            if root != *addr
                && func.types[root.index()] == Type::I64
                && !is_stack_addr[root.index()]
                && let Some(new_off) = i64::from(*off).checked_add(base_off)
                && fits_mem_imm(ty, new_off)
            {
                *addr = root;
                *off = new_off as i32;
            }
        }
    }
}

fn sink_terminator_defs(func: &mut Function) {
    let uses = func.use_counts();
    let mut fits_used = vec![false; func.types.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        if let Op::Fits(_, _, x, y) = func.insts[inst.index()].op {
            fits_used[x.index()] = true;
            fits_used[y.index()] = true;
        }
    }
    let mut term_uses = vec![0_u32; func.types.len()];
    for b in 0..func.blocks.len() {
        if func.blocks[b].insts.len() < 2 {
            continue;
        }
        let Some(&last) = func.blocks[b].insts.last() else { continue };
        match &func.insts[last.index()].op {
            Op::Jump(_, args) => args.iter().for_each(|v| term_uses[v.index()] += 1),
            Op::Brif(_, _, t_args, _, e_args) => {
                t_args.iter().for_each(|v| term_uses[v.index()] += 1);
                e_args.iter().for_each(|v| term_uses[v.index()] += 1);
            }
            _ => continue,
        }
        let insts = std::mem::take(&mut func.blocks[b].insts);
        let len = insts.len();
        let mut kept = Vec::with_capacity(len);
        let mut sunk = Vec::new();
        for &inst in &insts[..len - 1] {
            let data = &func.insts[inst.index()];
            let pure_alu = matches!(data.op, Op::Binary(op, ..) if !matches!(op, Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem))
                || matches!(data.op, Op::Unary(..) | Op::Convert(..) | Op::Icmp(..) | Op::Select(..));
            if let Some(res) = data.result
                && pure_alu
                && !fits_used[res.index()]
                && term_uses[res.index()] > 0
                && uses[res.index()] == term_uses[res.index()]
            {
                sunk.push(inst);
            } else {
                kept.push(inst);
            }
        }
        kept.extend(sunk);
        kept.push(last);
        func.blocks[b].insts = kept;
        match &func.insts[last.index()].op {
            Op::Jump(_, args) => args.iter().for_each(|v| term_uses[v.index()] = 0),
            Op::Brif(_, _, t_args, _, e_args) => {
                t_args.iter().for_each(|v| term_uses[v.index()] = 0);
                e_args.iter().for_each(|v| term_uses[v.index()] = 0);
            }
            _ => {}
        }
    }
}

/// Replaces each load of bytes that an earlier access on every path to it read or wrote with no write of unknown reach between: through each block and on into a successor that only it enters.
pub(crate) fn run(func: &mut Function) {
    let Some(entry) = func.entry else { return };
    if func.coroutine.is_some() {
        return;
    }
    eliminate_affine_block_params(func);
    let preds = pred_counts(func);
    let slots = slot_addresses(func);
    let affine = affine_table(func);
    fold_affine_mem(func, &slots, &affine);
    sink_terminator_defs(func);
    let (unescaped, call_readonly) = classify_slots(func, &slots);
    let mut consts = vec![None; func.types.len()];
    let mut selects = vec![None; func.types.len()];
    for data in &func.insts {
        match (&data.op, data.result) {
            (&Op::Const(imm), Some(result)) => consts[result.index()] = Some(imm),
            (&Op::Select(c, a, b), Some(result)) => selects[result.index()] = Some((c, a, b)),
            _ => {}
        }
    }
    let mut subst = identity(func.types.len());
    let mut state = State { slots: &slots, affine: &affine, unescaped: &unescaped, call_readonly: &call_readonly, selects: &selects, consts: &mut consts, subst: &mut subst };
    let mut dropped = false;
    let mut seen = vec![false; func.blocks.len()];
    let mut stack = vec![(entry, Known::default())];
    while let Some((block, mut known)) = stack.pop() {
        if std::mem::replace(&mut seen[block.index()], true) {
            continue;
        }
        dropped |= scan(func, block, &mut state, &mut known);
        let ends_in_call = func.last(block).is_some_and(|op| matches!(op, Op::TryCall { .. }));
        for next in func.successors(block) {
            let inherits = preds[next.index()] == 1 && next != entry && !ends_in_call;
            stack.push((next, if inherits { known.clone() } else { Known::default() }));
        }
    }
    if dropped {
        rename(func, state.subst);
    }
}
