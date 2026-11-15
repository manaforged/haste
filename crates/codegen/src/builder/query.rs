use crate::builder::FunctionBuilder;
use crate::ir::{Binary, Convert, Function, Inst, Op, StackSlot, Value};

fn def_op(func: &Function, mut value: Value) -> Option<(Value, &Op)> {
    for _ in 0..4 {
        let data = func.insts.iter().rev().take(128).find(|d| d.result == Some(value))?;
        if let Op::Convert(Convert::Bitcast, inner) = data.op {
            value = inner;
            continue;
        }
        return Some((value, &data.op));
    }
    None
}

fn const_int_at(func: &Function, value: Value, depth: u8) -> Option<i64> {
    if depth > 4 {
        return None;
    }
    let (_, op) = def_op(func, value)?;
    match *op {
        Op::Const(imm) => Some(imm),
        Op::Binary(Binary::Imul, a, b) => const_int_at(func, a, depth + 1)?.checked_mul(const_int_at(func, b, depth + 1)?),
        _ => None,
    }
}

fn rodata_fat_len(func: &Function, sym: u32) -> Option<u64> {
    let (_, bytes) = func.rodata_bytes.iter().rev().find(|&&(s, _)| s == sym)?;
    if bytes.len() != 16 {
        return None;
    }
    let raw: [u8; 8] = bytes.get(8..16)?.try_into().ok()?;
    Some(u64::from_le_bytes(raw))
}

enum SlotStep {
    Hit(Option<u64>),
    Stop,
    Continue,
}

fn check_slot_inst(func: &Function, ptr: Value, slot: StackSlot, target_off: i32, inst: Inst) -> SlotStep {
    match func.insts[inst.index()].op {
        Op::StackStore(val, s, off) if s == slot => {
            if off == target_off {
                return SlotStep::Hit(const_int_at(func, val, 0).and_then(|n| u64::try_from(n).ok()));
            }
            let end = i64::from(off) + i64::from(func.ty(val).bytes());
            let target = i64::from(target_off);
            if i64::from(off) < target + 8 && end > target { SlotStep::Stop } else { SlotStep::Continue }
        }
        Op::Store(val, addr, off) if addr == ptr => {
            if off == 8 {
                return SlotStep::Hit(const_int_at(func, val, 0).and_then(|n| u64::try_from(n).ok()));
            }
            let end = i64::from(off) + i64::from(func.ty(val).bytes());
            if i64::from(off) < 16 && end > 8 { SlotStep::Stop } else { SlotStep::Continue }
        }
        Op::Store(..) | Op::Call { .. } | Op::TryCall { .. } | Op::Move(..) | Op::Copy(..) => SlotStep::Stop,
        _ => SlotStep::Continue,
    }
}

fn slot_store_const_u64(builder: &FunctionBuilder<'_>, ptr: Value, slot: StackSlot, target_off: i32) -> Option<u64> {
    let block = builder.current?;
    for &inst in builder.func.blocks[block.index()].insts.iter().rev().take(128) {
        match check_slot_inst(builder.func, ptr, slot, target_off, inst) {
            SlotStep::Hit(len) => return len,
            SlotStep::Stop => return None,
            SlotStep::Continue => {}
        }
    }
    None
}

impl FunctionBuilder<'_> {
    /// The integer constant value of `value` when defined directly in recent instructions.
    pub fn const_int(&self, value: Value) -> Option<i64> {
        const_int_at(self.func, value, 0)
    }

    /// Whether `value` was defined as an immediate constant in the current function.
    pub fn is_const(&self, value: Value) -> bool {
        self.const_int(value).is_some()
    }

    /// The statically known element count of a fat-pointer address `ptr` when backed by a recent slot write or 16-byte `.rodata` blob.
    pub fn wide_const_len(&self, ptr: Value) -> Option<u64> {
        let (base, op) = def_op(self.func, ptr)?;
        match *op {
            Op::Symbol(sym) => rodata_fat_len(self.func, sym),
            Op::StackAddr(slot, base_off) => {
                let target_off = base_off.checked_add(8)?;
                slot_store_const_u64(self, base, slot, target_off)
            }
            _ => None,
        }
    }
}
