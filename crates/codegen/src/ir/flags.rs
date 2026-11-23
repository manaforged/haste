use super::{Function, Value};

const UNIQUE_PTR: u8 = 1;
const UNPREDICTABLE: u8 = 2;
const IMAGE_ADDR: u8 = 4;
const THREAD_CELL: u8 = 8;
const FRAME_PTR: u8 = 16;

impl Function {
    /// Marks `value` as a pointer that a `Box` or `Unique` owner holds, to a heap allocation apart from the owner's own.
    pub fn mark_unique_ptr(&mut self, value: Value) {
        self.set_flags(value, UNIQUE_PTR);
    }

    /// Marks the `Op::Select` that defines `value` as unpredictable, so no pass turns it into a branch.
    pub fn mark_unpredictable(&mut self, value: Value) {
        self.set_flags(value, UNPREDICTABLE);
    }

    /// Marks `value` as the address of data that the output image holds, so code may reach it relative to the program counter.
    pub fn mark_image_addr(&mut self, value: Value) {
        self.set_flags(value, IMAGE_ADDR);
    }

    /// Marks `value` as the runtime's thread-cell lookup, `(slot, size) -> cell`, whose fast path a target may inline.
    pub fn mark_thread_cell(&mut self, value: Value) {
        self.set_flags(value, THREAD_CELL);
    }

    /// Marks `value` as the start of this coroutine's own frame, whose first `frame_private` bytes no other code can address.
    pub(crate) fn mark_frame_ptr(&mut self, value: Value) {
        self.set_flags(value, FRAME_PTR);
    }

    /// Whether `value` is the start of this coroutine's own frame.
    pub(crate) fn is_frame_ptr(&self, value: Value) -> bool {
        self.flags(value) & FRAME_PTR != 0
    }

    /// Whether `value` is the runtime's thread-cell lookup.
    pub(crate) fn is_thread_cell(&self, value: Value) -> bool {
        self.flags(value) & THREAD_CELL != 0
    }

    /// Mask of values defined by `Op::Const(0)`, empty when no thread-cell lookup is present.
    pub(crate) fn thread_cell_zeros(&self) -> Vec<bool> {
        if !self.value_flags.iter().any(|&bits| bits & THREAD_CELL != 0) {
            return Vec::new();
        }
        let mut zeros = vec![false; self.types.len()];
        for inst in &self.insts {
            if let (super::Op::Const(0), Some(res)) = (&inst.op, inst.result) {
                zeros[res.index()] = true;
            }
        }
        zeros
    }

    /// Result value of `op` when it is a zero-initialized `ThreadCell` lookup with no user-visible memory effects.
    pub(crate) fn pure_thread_cell_result(&self, op: &super::Op, is_zero: impl Fn(Value) -> bool) -> Option<Value> {
        match op {
            super::Op::Call { callee, args, results, .. }
                if self.is_thread_cell(*callee) && args.get(2).is_some_and(|&v| is_zero(v)) => results.first().copied(),
            _ => None,
        }
    }

    /// Result value of `op` when it looks up the internal `#[track_caller]` cell (`slot == 0`, `init == 0`).
    pub(crate) fn caller_loc_result(&self, op: &super::Op, is_zero: impl Fn(Value) -> bool) -> Option<Value> {
        match op {
            super::Op::Call { callee, args, results, .. }
                if self.is_thread_cell(*callee)
                    && args.first().is_some_and(|&v| is_zero(v))
                    && args.get(2).is_some_and(|&v| is_zero(v)) => results.first().copied(),
            _ => None,
        }
    }

    /// Whether `value` is the address of data that the output image holds.
    pub(crate) fn is_image_addr(&self, value: Value) -> bool {
        self.flags(value) & IMAGE_ADDR != 0
    }

    /// Drops every image-address mark, for targets that cannot reach data relative to the program counter.
    pub(crate) fn forget_image_addrs(&mut self) {
        self.value_flags.iter_mut().for_each(|bits| *bits &= !IMAGE_ADDR);
    }

    /// Whether `value` is a pointer that a `Box` or `Unique` owner holds, to a heap allocation apart from the owner's own.
    pub(crate) fn is_unique_ptr(&self, value: Value) -> bool {
        self.flags(value) & UNIQUE_PTR != 0
    }

    /// Whether `value` is defined by a select that must stay branch-free.
    pub(crate) fn is_unpredictable(&self, value: Value) -> bool {
        self.flags(value) & UNPREDICTABLE != 0
    }

    /// Drops every unpredictable mark, for targets where a branch beats a conditional move.
    pub(crate) fn forget_unpredictable(&mut self) {
        self.value_flags.iter_mut().for_each(|bits| *bits &= !UNPREDICTABLE);
    }

    /// Whether any value carries the unpredictable mark.
    pub(crate) fn has_unpredictable(&self) -> bool {
        self.value_flags.iter().any(|&bits| bits & UNPREDICTABLE != 0)
    }

    /// The marks on `value`.
    pub(crate) fn flags(&self, value: Value) -> u8 {
        self.value_flags.get(value.index()).copied().unwrap_or(0)
    }

    /// Adds the marks `bits` to `value`.
    pub(crate) fn set_flags(&mut self, value: Value, bits: u8) {
        if bits == 0 {
            return;
        }
        if value.index() >= self.value_flags.len() {
            self.value_flags.resize(value.index() + 1, 0);
        }
        self.value_flags[value.index()] |= bits;
    }

    /// A new value with the type and marks of `from`, for a copy of the instruction that defines it.
    pub(crate) fn value_like(&mut self, from: Value) -> Value {
        let value = self.value(self.ty(from));
        self.set_flags(value, self.flags(from));
        value
    }

    /// New values for every value of `callee`, with their marks, indexed as in `callee`.
    pub(crate) fn import_values(&mut self, callee: &Function) -> Vec<Value> {
        let values: Vec<Value> = callee.types.iter().map(|&ty| self.value(ty)).collect();
        for (index, &value) in values.iter().enumerate() {
            self.set_flags(value, callee.flags(Value(index as u32)) & !FRAME_PTR);
        }
        values
    }
}
