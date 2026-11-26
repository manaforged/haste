use crate::ir::{
    AbiParam, ArgumentPurpose, Block, Function, Op, ShuffleMask, Signature, Type, Value,
};

use super::{CONVERTS, FORMAT, SCALARS, tag_of};

pub(super) fn encode(func: &Function) -> Option<Vec<u8>> {
    if func.coroutine.is_some() {
        return None;
    }
    let mut w = Writer(Vec::with_capacity(64 + func.insts.len() * 8));
    w.byte(FORMAT);
    w.signature(&func.signature);
    w.slice(&func.types, |w, &ty| w.ty(ty));
    w.raw(&func.value_flags);
    w.slice(&func.slots, |w, slot| {
        w.u32(slot.size);
        w.byte(slot.align_shift);
    });
    w.slice(&func.sigs, Writer::signature);
    w.body(func);
    w.metadata(func);
    Some(w.0)
}

struct Writer(Vec<u8>);

impl Writer {
    fn byte(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u64(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.0.push(value as u8 | 0x80);
            value >>= 7;
        }
        self.0.push(value as u8);
    }

    fn u32(&mut self, value: u32) {
        self.u64(u64::from(value));
    }

    fn len(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn i64(&mut self, value: i64) {
        self.u64(((value << 1) ^ (value >> 63)) as u64);
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.len(bytes.len());
        self.0.extend_from_slice(bytes);
    }

    fn slice<T>(&mut self, items: &[T], mut write: impl FnMut(&mut Self, &T)) {
        self.len(items.len());
        for item in items {
            write(self, item);
        }
    }

    fn value(&mut self, value: Value) {
        self.u32(value.0);
    }

    fn values(&mut self, values: &[Value]) {
        self.slice(values, |w, &value| w.value(value));
    }

    fn block(&mut self, block: Block) {
        self.u32(block.0);
    }

    fn ty(&mut self, ty: Type) {
        match ty {
            Type::Vec { lane, count } => {
                self.byte(SCALARS.len() as u8);
                self.byte(lane as u8);
                self.byte(count);
            }
            scalar => self.byte(tag_of(&SCALARS, &scalar)),
        }
    }

    fn param(&mut self, param: &AbiParam) {
        self.ty(param.value_type);
        match param.purpose {
            ArgumentPurpose::Normal => self.byte(0),
            ArgumentPurpose::StructReturn => self.byte(1),
            ArgumentPurpose::StructArgument(bytes) => {
                self.byte(2);
                self.u32(bytes);
            }
            ArgumentPurpose::Dict => self.byte(3),
            ArgumentPurpose::Variadic => self.byte(4),
        }
        self.byte(u8::from(param.readonly) | u8::from(param.noalias) << 1);
        self.u32(param.deref_bytes);
    }

    fn signature(&mut self, sig: &Signature) {
        self.slice(&sig.params, Self::param);
        self.slice(&sig.returns, Self::param);
        self.byte(sig.call_conv as u8);
    }

    fn body(&mut self, func: &Function) {
        self.len(func.blocks.len());
        self.len(func.insts.len());
        for data in &func.insts {
            self.op(&data.op);
            self.u32(data.result.map_or(0, |value| value.0 + 1));
            self.u64(data.loc);
        }
        for block in &func.blocks {
            self.values(&block.params);
            self.slice(&block.insts, |w, inst| w.u32(inst.0));
        }
        self.u32(func.entry.map_or(0, |block| block.0 + 1));
    }

    fn metadata(&mut self, func: &Function) {
        self.slice(&func.self_slots, |w, &sym| w.u32(sym));
        self.slice(&func.vtable_slots, |w, &(table, offset, slot, low)| {
            w.u32(table);
            w.i64(i64::from(offset));
            w.u32(slot);
            w.byte(u8::from(low));
        });
        self.slice(&func.rodata_bytes, |w, (sym, bytes)| {
            w.u32(*sym);
            w.raw(bytes);
        });
        self.byte(u8::from(func.keep_fp));
        self.slice(&func.unwind_drops, |w, u| {
            w.u32(u.local.0);
            w.i64(i64::from(u.local_off));
            w.u32(u.flag.0);
            w.u32(u.drop_sym);
            w.byte(u8::from(u.indirect));
        });
        self.u32(func.frame_private);
        self.u32(func.cfg_rev);
    }

    fn values3(&mut self, tag: u8, (a, b, c): (Value, Value, Value)) {
        self.byte(tag);
        self.value(a);
        self.value(b);
        self.value(c);
    }

    fn pair(&mut self, tag: u8, kind: u8, (x, y): (Value, Value)) {
        self.byte(tag);
        self.byte(kind);
        self.value(x);
        self.value(y);
    }

    fn mem_load(&mut self, tag: u8, x: Value, offset: i32) {
        self.byte(tag);
        self.value(x);
        self.i64(i64::from(offset));
    }

    fn mem_store(&mut self, tag: u8, x: Value, y: Value, offset: i32) {
        self.pair(tag, 0, (x, y));
        self.i64(i64::from(offset));
    }

    fn stack_ref(&mut self, tag: u8, slot: u32, offset: i32) {
        self.byte(tag);
        self.u32(slot);
        self.i64(i64::from(offset));
    }

    fn op(&mut self, op: &Op) {
        match op {
            Op::Const(_)
            | Op::SlotAddr(_)
            | Op::Symbol(_)
            | Op::Runtime(_)
            | Op::Unary(..)
            | Op::Binary(..)
            | Op::Fma(..)
            | Op::Convert(..)
            | Op::Icmp(..)
            | Op::Fits(..)
            | Op::Fcmp(..)
            | Op::Shuffle(..)
            | Op::Extract(..)
            | Op::Insert(..)
            | Op::Reduce(..)
            | Op::TargetOp(..)
            | Op::Select(..) => self.op_value(op),
            Op::Load(..)
            | Op::Store(..)
            | Op::VolatileLoad(..)
            | Op::VolatileStore(..)
            | Op::StackAddr(..)
            | Op::StackLoad(..)
            | Op::StackStore(..)
            | Op::FramePointer
            | Op::Copy(..)
            | Op::Move(..) => self.op_memory(op),
            Op::AtomicLoad(..)
            | Op::AtomicStore(..)
            | Op::AtomicRmw(..)
            | Op::AtomicCas(..)
            | Op::Fence(..) => self.op_atomic(op),
            Op::Call { .. }
            | Op::TryCall { .. }
            | Op::Jump(..)
            | Op::Brif(..)
            | Op::BrTable(..)
            | Op::Return(..)
            | Op::Trap
            | Op::Trapz(..)
            | Op::Assume(..) => self.op_control(op),
        }
    }

    fn op_value(&mut self, op: &Op) {
        match op {
            Op::Const(imm) => {
                self.byte(0);
                self.i64(*imm);
            }
            Op::SlotAddr(sym) => {
                self.byte(1);
                self.u32(*sym);
            }
            Op::Symbol(sym) => {
                self.byte(2);
                self.u32(*sym);
            }
            Op::Runtime(sym) => {
                self.byte(3);
                self.byte(*sym as u8);
            }
            Op::Unary(unary, x) => {
                self.byte(4);
                self.byte(*unary as u8);
                self.value(*x);
            }
            Op::Binary(binary, x, y) => self.pair(5, *binary as u8, (*x, *y)),
            Op::Fma(a, b, c) => self.values3(6, (*a, *b, *c)),
            Op::Convert(convert, x) => {
                self.byte(7);
                self.byte(tag_of(&CONVERTS, convert));
                self.value(*x);
            }
            Op::Icmp(cc, x, y) => self.pair(8, *cc as u8, (*x, *y)),
            Op::Fits(binary, signed, x, y) => {
                self.pair(9, *binary as u8, (*x, *y));
                self.byte(u8::from(*signed));
            }
            Op::Fcmp(cc, x, y) => self.pair(10, *cc as u8, (*x, *y)),
            Op::Shuffle(a, b, mask) => {
                self.byte(11);
                self.value(*a);
                self.value(*b);
                match mask {
                    ShuffleMask::Const(lanes) => {
                        self.byte(0);
                        self.0.extend_from_slice(lanes);
                    }
                    ShuffleMask::Dyn(index) => {
                        self.byte(1);
                        self.value(*index);
                    }
                }
            }
            Op::Extract(x, lane) => {
                self.byte(12);
                self.value(*x);
                self.byte(*lane);
            }
            Op::Insert(x, y, lane) => self.pair(13, *lane, (*x, *y)),
            Op::Reduce(reduce, x) => {
                self.byte(14);
                self.byte(*reduce as u8);
                self.value(*x);
            }
            Op::TargetOp(id, args) => {
                self.byte(15);
                self.byte(*id as u8);
                self.values(args.as_slice());
            }
            Op::Select(c, x, y) => self.values3(16, (*c, *x, *y)),
            _ => {}
        }
    }

    fn op_memory(&mut self, op: &Op) {
        match op {
            Op::Load(x, offset) => self.mem_load(17, *x, *offset),
            Op::Store(x, y, offset) => self.mem_store(18, *x, *y, *offset),
            Op::VolatileLoad(x, offset) => self.mem_load(39, *x, *offset),
            Op::VolatileStore(x, y, offset) => self.mem_store(40, *x, *y, *offset),
            Op::StackAddr(slot, offset) => self.stack_ref(19, slot.0, *offset),
            Op::StackLoad(slot, offset) => self.stack_ref(20, slot.0, *offset),
            Op::StackStore(x, slot, offset) => {
                self.byte(21);
                self.value(*x);
                self.u32(slot.0);
                self.i64(i64::from(*offset));
            }
            Op::FramePointer => self.byte(27),
            Op::Copy(a, b, c) => self.values3(28, (*a, *b, *c)),
            Op::Move(x, y, bytes) => {
                self.pair(29, 0, (*x, *y));
                self.u32(*bytes);
            }
            _ => {}
        }
    }

    fn op_atomic(&mut self, op: &Op) {
        match op {
            Op::AtomicLoad(x, ord) => {
                self.byte(22);
                self.byte(*ord as u8);
                self.value(*x);
            }
            Op::AtomicStore(x, y, ord) => self.pair(23, *ord as u8, (*x, *y)),
            Op::AtomicRmw(rmw, ord, x, y) => {
                self.pair(24, *rmw as u8, (*x, *y));
                self.byte(*ord as u8);
            }
            Op::AtomicCas(ord, a, b, c) => {
                self.values3(25, (*a, *b, *c));
                self.byte(*ord as u8);
            }
            Op::Fence(ord) => {
                self.byte(26);
                self.byte(*ord as u8);
            }
            _ => {}
        }
    }

    fn op_control(&mut self, op: &Op) {
        match op {
            Op::Call {
                sig,
                callee,
                args,
                results,
            } => {
                self.byte(30);
                self.u32(sig.0);
                self.value(*callee);
                self.values(args);
                self.values(results);
            }
            Op::TryCall {
                sig,
                callee,
                args,
                normal,
                handler,
            } => {
                self.byte(31);
                self.u32(sig.0);
                self.value(*callee);
                self.values(args);
                self.block(*normal);
                self.block(*handler);
            }
            Op::Jump(block, args) => {
                self.byte(32);
                self.block(*block);
                self.values(args);
            }
            Op::Brif(c, then, then_args, otherwise, otherwise_args) => {
                self.byte(33);
                self.value(*c);
                self.block(*then);
                self.values(then_args);
                self.block(*otherwise);
                self.values(otherwise_args);
            }
            Op::BrTable(index, default, targets) => {
                self.byte(34);
                self.value(*index);
                self.block(*default);
                self.slice(targets, |w, &target| w.block(target));
            }
            Op::Return(values) => {
                self.byte(35);
                self.values(values);
            }
            Op::Trap => self.byte(36),
            Op::Trapz(x) => {
                self.byte(37);
                self.value(*x);
            }
            Op::Assume(x) => {
                self.byte(38);
                self.value(*x);
            }
            _ => {}
        }
    }
}
