use crate::UnwindDrop;
use crate::ir::{
    AbiParam, ArgumentPurpose, AtomicOrd, Block, BlockData, Function, Inst, InstData, Op,
    ShuffleMask, SigRef, Signature, SlotData, StackSlot, TinyArgs, Type, Value,
};
use crate::reloc::RuntimeSymbol;

use super::{
    BINARY, CONVERTS, CONVS, FLOAT_CC, FORMAT, INT_CC, LANES, REDUCE, RMW, SCALARS, TARGET_OPS,
    UNARY, pick,
};

pub(super) fn decode(bytes: &[u8]) -> Option<Function> {
    let mut r = Reader {
        bytes,
        at: 0,
        values: 0,
        blocks: 0,
        insts: 0,
        slots: 0,
        sigs: 0,
    };
    if r.byte()? != FORMAT {
        return None;
    }
    let mut func = Function::new(r.signature()?);
    func.types = r.read_vec(Reader::ty)?;
    func.value_flags = r.raw()?.to_vec();
    func.slots = r.read_vec(Reader::slot_data)?;
    func.sigs = r.read_vec(Reader::signature)?;
    r.decode_body(&mut func)?;
    r.decode_metadata(&mut func)?;
    (r.at == bytes.len()).then_some(func)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    values: u32,
    blocks: u32,
    insts: u32,
    slots: u32,
    sigs: u32,
}

impl<'a> Reader<'a> {
    fn byte(&mut self) -> Option<u8> {
        let value = *self.bytes.get(self.at)?;
        self.at += 1;
        Some(value)
    }

    fn flag(&mut self) -> Option<bool> {
        match self.byte()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    fn u64(&mut self) -> Option<u64> {
        let mut value = 0_u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Some(value);
            }
        }
        None
    }

    fn u32(&mut self) -> Option<u32> {
        u32::try_from(self.u64()?).ok()
    }

    fn len(&mut self) -> Option<usize> {
        let len = self.u32()? as usize;
        (len <= self.bytes.len() - self.at).then_some(len)
    }

    fn i64(&mut self) -> Option<i64> {
        let raw = self.u64()?;
        Some((raw >> 1) as i64 ^ -((raw & 1) as i64))
    }

    fn i32(&mut self) -> Option<i32> {
        i32::try_from(self.i64()?).ok()
    }

    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let bytes = self.bytes.get(self.at..self.at.checked_add(len)?)?;
        self.at += len;
        Some(bytes)
    }

    fn raw(&mut self) -> Option<&'a [u8]> {
        let len = self.len()?;
        self.take(len)
    }

    fn read_vec<T>(&mut self, mut read: impl FnMut(&mut Self) -> Option<T>) -> Option<Vec<T>> {
        let len = self.len()?;
        (0..len).map(|_| read(self)).collect()
    }

    fn value(&mut self) -> Option<Value> {
        let at = self.u32()?;
        (at < self.values).then_some(Value(at))
    }

    fn opt_value(&mut self) -> Option<Option<Value>> {
        match self.u32()? {
            0 => Some(None),
            at => (at - 1 < self.values).then_some(Some(Value(at - 1))),
        }
    }

    fn values(&mut self) -> Option<Vec<Value>> {
        self.read_vec(Self::value)
    }

    fn check_block(&self, at: u32) -> Option<Block> {
        (at < self.blocks).then_some(Block(at))
    }

    fn block(&mut self) -> Option<Block> {
        let at = self.u32()?;
        self.check_block(at)
    }

    fn inst(&mut self) -> Option<Inst> {
        let at = self.u32()?;
        (at < self.insts).then_some(Inst(at))
    }

    fn slot(&mut self) -> Option<StackSlot> {
        let at = self.u32()?;
        (at < self.slots).then_some(StackSlot(at))
    }

    fn sig(&mut self) -> Option<SigRef> {
        let at = self.u32()?;
        (at < self.sigs).then_some(SigRef(at))
    }

    fn atomic_ord(&mut self) -> Option<AtomicOrd> {
        AtomicOrd::from_u8(self.byte()?)
    }

    fn slot_data(&mut self) -> Option<SlotData> {
        let size = self.u32()?;
        let flags = self.byte()?;
        Some(SlotData {
            size,
            align_shift: flags & 0x7f,
            poll_local: (flags & 0x80) != 0,
        })
    }

    fn inst_data(&mut self) -> Option<InstData> {
        Some(InstData {
            op: self.op()?,
            result: self.opt_value()?,
            loc: self.u64()?,
        })
    }

    fn block_data(&mut self) -> Option<BlockData> {
        let params = self.values()?;
        let insts = self.read_vec(Self::inst)?;
        Some(BlockData { params, insts })
    }

    fn vtable_slot(&mut self) -> Option<(u32, i32, u32, bool)> {
        Some((self.u32()?, self.i32()?, self.u32()?, self.flag()?))
    }

    fn rodata_entry(&mut self) -> Option<(u32, Vec<u8>)> {
        Some((self.u32()?, self.raw()?.to_vec()))
    }

    fn unwind_drop(&mut self) -> Option<UnwindDrop> {
        Some(UnwindDrop {
            local: self.slot()?,
            local_off: self.i32()?,
            flag: self.slot()?,
            drop_sym: self.u32()?,
            arg: crate::UnwindArg::from_u8(self.byte()?)?,
        })
    }

    fn decode_body(&mut self, func: &mut Function) -> Option<()> {
        let blocks = self.len()?;
        let insts = self.len()?;
        (self.values, self.blocks, self.insts, self.slots, self.sigs) = (
            func.types.len() as u32,
            blocks as u32,
            insts as u32,
            func.slots.len() as u32,
            func.sigs.len() as u32,
        );
        func.insts = (0..insts)
            .map(|_| self.inst_data())
            .collect::<Option<_>>()?;
        func.blocks = (0..blocks)
            .map(|_| self.block_data())
            .collect::<Option<_>>()?;
        func.entry = match self.u32()? {
            0 => None,
            at => Some(self.check_block(at - 1)?),
        };
        Some(())
    }

    fn decode_metadata(&mut self, func: &mut Function) -> Option<()> {
        func.self_slots = self.read_vec(Self::u32)?;
        func.vtable_slots = self.read_vec(Self::vtable_slot)?;
        func.rodata_bytes = self.read_vec(Self::rodata_entry)?;
        func.keep_fp = self.flag()?;
        func.abort_on_unwind = self.flag()?;
        func.save_all_regs = self.flag()?;
        func.unwind_drops = self.read_vec(Self::unwind_drop)?;
        func.frame_private = self.u32()?;
        func.cfg_rev = self.u32()?;
        Some(())
    }

    fn ty(&mut self) -> Option<Type> {
        let tag = self.byte()?;
        if usize::from(tag) == SCALARS.len() {
            return Some(Type::Vec {
                lane: pick(LANES, self.byte()?)?,
                count: self.byte()?,
            });
        }
        pick(&SCALARS, tag)
    }

    fn param(&mut self) -> Option<AbiParam> {
        let value_type = self.ty()?;
        let purpose = match self.byte()? {
            0 => ArgumentPurpose::Normal,
            1 => ArgumentPurpose::StructReturn,
            2 => ArgumentPurpose::StructArgument(self.u32()?),
            3 => ArgumentPurpose::Dict,
            4 => ArgumentPurpose::Variadic,
            _ => return None,
        };
        let marks = self.byte()?;
        if marks > 3 {
            return None;
        }
        Some(AbiParam {
            value_type,
            purpose,
            readonly: marks & 1 != 0,
            noalias: marks & 2 != 0,
            deref_bytes: self.u32()?,
        })
    }

    fn signature(&mut self) -> Option<Signature> {
        let params = self.read_vec(Self::param)?;
        let returns = self.read_vec(Self::param)?;
        Some(Signature {
            params,
            returns,
            call_conv: pick(CONVS, self.byte()?)?,
        })
    }

    fn unary_op<T: Copy>(&mut self, table: &[T]) -> Option<(T, Value)> {
        Some((pick(table, self.byte()?)?, self.value()?))
    }

    fn pair<T: Copy>(&mut self, table: &[T]) -> Option<(T, Value, Value)> {
        Some((pick(table, self.byte()?)?, self.value()?, self.value()?))
    }

    fn values3(&mut self) -> Option<(Value, Value, Value)> {
        Some((self.value()?, self.value()?, self.value()?))
    }

    fn plain_pair(&mut self) -> Option<(Value, Value)> {
        let (_, x, y) = self.pair(&[0_u8])?;
        Some((x, y))
    }

    fn op(&mut self) -> Option<Op> {
        match self.byte()? {
            tag @ 0..=10 => self.op_scalar(tag),
            tag @ 11..=16 => self.op_vector(tag),
            tag @ (17..=21 | 27..=29 | 39 | 40) => self.op_memory(tag),
            tag @ 22..=26 => self.op_atomic(tag),
            tag @ 30..=38 => self.op_control(tag),
            _ => None,
        }
    }

    fn op_scalar(&mut self, tag: u8) -> Option<Op> {
        Some(match tag {
            0 => Op::Const(self.i64()?),
            1 => Op::SlotAddr(self.u32()?),
            2 => Op::Symbol(self.u32()?),
            3 => Op::Runtime(RuntimeSymbol::from_raw(self.byte()?)?),
            4 => {
                let (unary, x) = self.unary_op(UNARY)?;
                Op::Unary(unary, x)
            }
            5 => {
                let (binary, x, y) = self.pair(BINARY)?;
                Op::Binary(binary, x, y)
            }
            6 => {
                let (a, b, c) = self.values3()?;
                Op::Fma(a, b, c)
            }
            7 => {
                let (convert, x) = self.unary_op(&CONVERTS)?;
                Op::Convert(convert, x)
            }
            8 => {
                let (cc, x, y) = self.pair(INT_CC)?;
                Op::Icmp(cc, x, y)
            }
            9 => {
                let (binary, x, y) = self.pair(BINARY)?;
                Op::Fits(binary, self.flag()?, x, y)
            }
            10 => {
                let (cc, x, y) = self.pair(FLOAT_CC)?;
                Op::Fcmp(cc, x, y)
            }
            _ => return None,
        })
    }

    fn op_shuffle(&mut self) -> Option<Op> {
        let (a, b) = (self.value()?, self.value()?);
        let mask = match self.byte()? {
            0 => ShuffleMask::Const(self.take(16)?.try_into().ok()?),
            1 => ShuffleMask::Dyn(self.value()?),
            _ => return None,
        };
        Some(Op::Shuffle(a, b, mask))
    }

    fn op_vector(&mut self, tag: u8) -> Option<Op> {
        Some(match tag {
            11 => self.op_shuffle()?,
            12 => Op::Extract(self.value()?, self.byte()?),
            13 => {
                let lane = self.byte()?;
                Op::Insert(self.value()?, self.value()?, lane)
            }
            14 => {
                let (reduce, x) = self.unary_op(REDUCE)?;
                Op::Reduce(reduce, x)
            }
            15 => Op::TargetOp(
                pick(TARGET_OPS, self.byte()?)?,
                TinyArgs::from_slice(&self.values()?)?,
            ),
            16 => {
                let (c, x, y) = self.values3()?;
                Op::Select(c, x, y)
            }
            _ => return None,
        })
    }

    fn op_memory(&mut self, tag: u8) -> Option<Op> {
        Some(match tag {
            17 => Op::Load(self.value()?, self.i32()?),
            18 => {
                let (x, y) = self.plain_pair()?;
                Op::Store(x, y, self.i32()?)
            }
            19 => Op::StackAddr(self.slot()?, self.i32()?),
            20 => Op::StackLoad(self.slot()?, self.i32()?),
            21 => Op::StackStore(self.value()?, self.slot()?, self.i32()?),
            27 => Op::FramePointer,
            28 => {
                let (a, b, c) = self.values3()?;
                Op::Copy(a, b, c)
            }
            29 => {
                let (x, y) = self.plain_pair()?;
                Op::Move(x, y, self.u32()?)
            }
            39 => Op::VolatileLoad(self.value()?, self.i32()?),
            40 => {
                let (x, y) = self.plain_pair()?;
                Op::VolatileStore(x, y, self.i32()?)
            }
            _ => return None,
        })
    }

    fn op_atomic(&mut self, tag: u8) -> Option<Op> {
        Some(match tag {
            22 => {
                let ord = self.atomic_ord()?;
                Op::AtomicLoad(self.value()?, ord)
            }
            23 => {
                let ord = self.atomic_ord()?;
                let (x, y) = (self.value()?, self.value()?);
                Op::AtomicStore(x, y, ord)
            }
            24 => {
                let (rmw, x, y) = self.pair(RMW)?;
                let ord = self.atomic_ord()?;
                Op::AtomicRmw(rmw, ord, x, y)
            }
            25 => {
                let (a, b, c) = self.values3()?;
                let ord = self.atomic_ord()?;
                Op::AtomicCas(ord, a, b, c)
            }
            26 => Op::Fence(self.atomic_ord()?),
            _ => return None,
        })
    }

    fn op_control(&mut self, tag: u8) -> Option<Op> {
        Some(match tag {
            30 => Op::Call {
                sig: self.sig()?,
                callee: self.value()?,
                args: self.values()?,
                results: self.values()?,
            },
            31 => Op::TryCall {
                sig: self.sig()?,
                callee: self.value()?,
                args: self.values()?,
                normal: self.block()?,
                handler: self.block()?,
            },
            32 => Op::Jump(self.block()?, self.values()?),
            33 => Op::Brif(
                self.value()?,
                self.block()?,
                self.values()?,
                self.block()?,
                self.values()?,
            ),
            34 => Op::BrTable(self.value()?, self.block()?, self.read_vec(Self::block)?),
            35 => Op::Return(self.values()?),
            36 => Op::Trap,
            37 => Op::Trapz(self.value()?),
            38 => Op::Assume(self.value()?),
            _ => return None,
        })
    }
}
