use crate::ir::{
    AbiParam, ArgumentPurpose, AtomicRmwOp, Binary, Block, BlockData, CallConv, Convert, FloatCC, Function, Inst, InstData, IntCC, Lane, Op, ReduceOp,
    ShuffleMask, SigRef, Signature, SlotData, StackSlot, TargetOpId, TinyArgs, Type, Unary, Value,
};
use crate::reloc::RuntimeSymbol;

const FORMAT: u8 = 1;

macro_rules! table {
    ($name:ident: $ty:ident = [$($variant:ident),* $(,)?]) => {
        const $name: &[$ty] = &[$($ty::$variant),*];
        const _: () = {
            let mut at = 0;
            while at < $name.len() {
                assert!($name[at] as usize == at);
                at += 1;
            }
        };
    };
}

table!(UNARY: Unary = [Ineg, Bnot, Bswap, Bitrev, Clz, Ctz, Popcnt, Fneg, Fabs, Sqrt, Floor, Ceil, Trunc]);
table!(BINARY: Binary = [
    Iadd, Isub, Imul, Sdiv, Udiv, Srem, Urem, Umulhi, Smulhi, Smin, Smax, Umin, Umax, Band, Bor, Bxor, Ishl, Ushr, Sshr, Rotl, Rotr, Iconcat, Fadd, Fsub,
    Fmul, Fdiv, Fmin, Fmax, Fcopysign, SaddSat, UaddSat, SsubSat, UsubSat, AvgRound,
]);
table!(INT_CC: IntCC = [
    Equal, NotEqual, SignedLessThan, SignedGreaterThanOrEqual, SignedGreaterThan, SignedLessThanOrEqual, UnsignedLessThan, UnsignedGreaterThanOrEqual,
    UnsignedGreaterThan, UnsignedLessThanOrEqual,
]);
table!(FLOAT_CC: FloatCC = [Equal, NotEqual, LessThan, LessThanOrEqual, GreaterThan, GreaterThanOrEqual]);
table!(RMW: AtomicRmwOp = [Add, Sub, And, Nand, Or, Xor, Xchg, Umin, Umax, Smin, Smax]);
table!(REDUCE: ReduceOp = [Add, Smin, Smax, Umin, Umax, Fmin, Fmax, And, Or, Xor, Movemask]);
table!(TARGET_OPS: TargetOpId = [
    Arm64Sha256H, Arm64Sha256H2, Arm64Sha256Su0, Arm64Sha256Su1, Arm64AesE, Arm64AesD, Arm64AesMc, Arm64AesImc, X64Sha1NextE, X64Sha1Msg1, X64Sha1Msg2,
    X64Sha256Rnds2, X64Sha256Msg1, X64Sha256Msg2, X64AesEnc, X64AesEncLast, X64AesDec, X64AesDecLast, X64AesImc, X64Low128, X64High128, X64Concat128,
    X64Pshufb256, X64Pcmpeqb256, X64Pcmpgtb256, X64Pmovmskb256, X64ClmulLowLow, X64ClmulHighLow, X64ClmulLowHigh, X64ClmulHighHigh, X64AddCarry, X64AddCarryOut, X64SubBorrow, X64SubBorrowOut,
    X64DivWide, X64RemWide, X64Psadbw256, X64Pmaddubsw256, X64Pmaddwd256, X64Psadbw,
]);
table!(LANES: Lane = [I8, I16, I32, I64, F32, F64]);
table!(CONVS: CallConv = [Internal, Native]);

const CONVERTS: [Convert; 12] = [
    Convert::Uextend,
    Convert::Sextend,
    Convert::Ireduce,
    Convert::Fpromote,
    Convert::Fdemote,
    Convert::FromSint,
    Convert::FromUint,
    Convert::ToSintSat,
    Convert::ToUintSat,
    Convert::Bitcast,
    Convert::NarrowSat { signed: false },
    Convert::NarrowSat { signed: true },
];

const SCALARS: [Type; 7] = [Type::I8, Type::I16, Type::I32, Type::I64, Type::I128, Type::F32, Type::F64];

fn pick<T: Copy>(table: &[T], tag: u8) -> Option<T> {
    table.get(usize::from(tag)).copied()
}

fn tag_of<T: PartialEq>(table: &[T], value: &T) -> u8 {
    table.iter().position(|entry| entry == value).map_or(u8::MAX, |at| at as u8)
}

impl Function {
    /// Encodes this function into bytes that `Function::decode` turns back into an equal function, or `None` for a coroutine.
    pub fn encode(&self) -> Option<Vec<u8>> {
        if self.coroutine.is_some() {
            return None;
        }
        let mut w = Writer(Vec::with_capacity(64 + self.insts.len() * 8));
        w.byte(FORMAT);
        w.signature(&self.signature);
        w.len(self.types.len());
        self.types.iter().for_each(|&ty| w.ty(ty));
        w.raw(&self.value_flags);
        w.len(self.slots.len());
        for slot in &self.slots {
            w.u32(slot.size);
            w.byte(slot.align_shift);
        }
        w.len(self.sigs.len());
        self.sigs.iter().for_each(|sig| w.signature(sig));
        w.len(self.blocks.len());
        w.len(self.insts.len());
        for data in &self.insts {
            w.op(&data.op);
            w.u32(data.result.map_or(0, |value| value.0 + 1));
        }
        for block in &self.blocks {
            w.values(&block.params);
            w.len(block.insts.len());
            block.insts.iter().for_each(|inst| w.u32(inst.0));
        }
        w.u32(self.entry.map_or(0, |block| block.0 + 1));
        w.len(self.self_slots.len());
        self.self_slots.iter().for_each(|&sym| w.u32(sym));
        w.len(self.vtable_slots.len());
        for &(table, offset, slot, low) in &self.vtable_slots {
            w.u32(table);
            w.i64(i64::from(offset));
            w.u32(slot);
            w.byte(u8::from(low));
        }
        w.len(self.rodata_bytes.len());
        for (sym, bytes) in &self.rodata_bytes {
            w.u32(*sym);
            w.raw(bytes);
        }
        w.u32(self.cfg_rev);
        Some(w.0)
    }

    /// Decodes a function that `Function::encode` wrote, or `None` when `bytes` do not hold one.
    pub fn decode(bytes: &[u8]) -> Option<Function> {
        let mut r = Reader { bytes, at: 0, values: 0, blocks: 0, insts: 0, slots: 0, sigs: 0 };
        if r.byte()? != FORMAT {
            return None;
        }
        let mut func = Function::new(r.signature()?);
        let types = r.len()?;
        func.types = (0..types).map(|_| r.ty()).collect::<Option<_>>()?;
        func.value_flags = r.raw()?.to_vec();
        let slots = r.len()?;
        func.slots = (0..slots).map(|_| Some(SlotData { size: r.u32()?, align_shift: r.byte()? })).collect::<Option<_>>()?;
        let sigs = r.len()?;
        func.sigs = (0..sigs).map(|_| r.signature()).collect::<Option<_>>()?;
        let blocks = r.len()?;
        let insts = r.len()?;
        (r.values, r.blocks, r.insts, r.slots, r.sigs) = (types as u32, blocks as u32, insts as u32, slots as u32, sigs as u32);
        func.insts = (0..insts).map(|_| Some(InstData { op: r.op()?, result: r.opt_value()? })).collect::<Option<_>>()?;
        func.blocks = (0..blocks).map(|_| r.block_data()).collect::<Option<_>>()?;
        func.entry = match r.u32()? {
            0 => None,
            at => Some(r.check_block(at - 1)?),
        };
        let self_slots = r.len()?;
        func.self_slots = (0..self_slots).map(|_| r.u32()).collect::<Option<_>>()?;
        let vtable_slots = r.len()?;
        func.vtable_slots = (0..vtable_slots).map(|_| Some((r.u32()?, i32::try_from(r.i64()?).ok()?, r.u32()?, r.flag()?))).collect::<Option<_>>()?;
        let rodata = r.len()?;
        func.rodata_bytes = (0..rodata).map(|_| Some((r.u32()?, r.raw()?.to_vec()))).collect::<Option<_>>()?;
        func.cfg_rev = r.u32()?;
        (r.at == bytes.len()).then_some(func)
    }
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

    fn value(&mut self, value: Value) {
        self.u32(value.0);
    }

    fn values(&mut self, values: &[Value]) {
        self.len(values.len());
        values.iter().for_each(|&value| self.value(value));
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
        self.len(sig.params.len());
        sig.params.iter().for_each(|param| self.param(param));
        self.len(sig.returns.len());
        sig.returns.iter().for_each(|param| self.param(param));
        self.byte(sig.call_conv as u8);
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

    fn op(&mut self, op: &Op) {
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
            Op::Insert(x, y, lane) => {
                self.pair(13, *lane, (*x, *y));
            }
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
            Op::Load(x, offset) => {
                self.byte(17);
                self.value(*x);
                self.i64(i64::from(*offset));
            }
            Op::Store(x, y, offset) => {
                self.pair(18, 0, (*x, *y));
                self.i64(i64::from(*offset));
            }
            Op::StackAddr(slot, offset) | Op::StackLoad(slot, offset) => {
                self.byte(if matches!(op, Op::StackAddr(..)) { 19 } else { 20 });
                self.u32(slot.0);
                self.i64(i64::from(*offset));
            }
            Op::StackStore(x, slot, offset) => {
                self.byte(21);
                self.value(*x);
                self.u32(slot.0);
                self.i64(i64::from(*offset));
            }
            Op::AtomicLoad(x) => {
                self.byte(22);
                self.value(*x);
            }
            Op::AtomicStore(x, y) => self.pair(23, 0, (*x, *y)),
            Op::AtomicRmw(rmw, x, y) => self.pair(24, *rmw as u8, (*x, *y)),
            Op::AtomicCas(a, b, c) => self.values3(25, (*a, *b, *c)),
            Op::Fence => self.byte(26),
            Op::FramePointer => self.byte(27),
            Op::Copy(a, b, c) => self.values3(28, (*a, *b, *c)),
            Op::Move(x, y, bytes) => {
                self.pair(29, 0, (*x, *y));
                self.u32(*bytes);
            }
            Op::Call { sig, callee, args, results } => {
                self.byte(30);
                self.u32(sig.0);
                self.value(*callee);
                self.values(args);
                self.values(results);
            }
            Op::TryCall { sig, callee, args, normal, handler } => {
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
                self.len(targets.len());
                targets.iter().for_each(|&target| self.block(target));
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
        }
    }
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
        let len = self.len()?;
        (0..len).map(|_| self.value()).collect()
    }

    fn check_block(&self, at: u32) -> Option<Block> {
        (at < self.blocks).then_some(Block(at))
    }

    fn block(&mut self) -> Option<Block> {
        let at = self.u32()?;
        self.check_block(at)
    }

    fn slot(&mut self) -> Option<StackSlot> {
        let at = self.u32()?;
        (at < self.slots).then_some(StackSlot(at))
    }

    fn sig(&mut self) -> Option<SigRef> {
        let at = self.u32()?;
        (at < self.sigs).then_some(SigRef(at))
    }

    fn block_data(&mut self) -> Option<BlockData> {
        let params = self.values()?;
        let len = self.len()?;
        let insts = (0..len)
            .map(|_| {
                let at = self.u32()?;
                (at < self.insts).then_some(Inst(at))
            })
            .collect::<Option<_>>()?;
        Some(BlockData { params, insts })
    }

    fn ty(&mut self) -> Option<Type> {
        let tag = self.byte()?;
        if usize::from(tag) == SCALARS.len() {
            return Some(Type::Vec { lane: pick(LANES, self.byte()?)?, count: self.byte()? });
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
        Some(AbiParam { value_type, purpose, readonly: marks & 1 != 0, noalias: marks & 2 != 0, deref_bytes: self.u32()? })
    }

    fn signature(&mut self) -> Option<Signature> {
        let params = self.len()?;
        let params = (0..params).map(|_| self.param()).collect::<Option<_>>()?;
        let returns = self.len()?;
        let returns = (0..returns).map(|_| self.param()).collect::<Option<_>>()?;
        Some(Signature { params, returns, call_conv: pick(CONVS, self.byte()?)? })
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
        Some(match self.byte()? {
            0 => Op::Const(self.i64()?),
            1 => Op::SlotAddr(self.u32()?),
            2 => Op::Symbol(self.u32()?),
            3 => Op::Runtime(RuntimeSymbol::from_raw(self.byte()?)?),
            4 => Op::Unary(pick(UNARY, self.byte()?)?, self.value()?),
            5 => {
                let (binary, x, y) = self.pair(BINARY)?;
                Op::Binary(binary, x, y)
            }
            6 => {
                let (a, b, c) = self.values3()?;
                Op::Fma(a, b, c)
            }
            7 => Op::Convert(pick(&CONVERTS, self.byte()?)?, self.value()?),
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
            11 => {
                let (a, b) = (self.value()?, self.value()?);
                let mask = match self.byte()? {
                    0 => ShuffleMask::Const(self.take(16)?.try_into().ok()?),
                    1 => ShuffleMask::Dyn(self.value()?),
                    _ => return None,
                };
                Op::Shuffle(a, b, mask)
            }
            12 => Op::Extract(self.value()?, self.byte()?),
            13 => {
                let lane = self.byte()?;
                Op::Insert(self.value()?, self.value()?, lane)
            }
            14 => Op::Reduce(pick(REDUCE, self.byte()?)?, self.value()?),
            15 => Op::TargetOp(pick(TARGET_OPS, self.byte()?)?, TinyArgs::from_slice(&self.values()?)?),
            16 => {
                let (c, x, y) = self.values3()?;
                Op::Select(c, x, y)
            }
            17 => Op::Load(self.value()?, self.i32()?),
            18 => {
                let (x, y) = self.plain_pair()?;
                Op::Store(x, y, self.i32()?)
            }
            19 => Op::StackAddr(self.slot()?, self.i32()?),
            20 => Op::StackLoad(self.slot()?, self.i32()?),
            21 => Op::StackStore(self.value()?, self.slot()?, self.i32()?),
            22 => Op::AtomicLoad(self.value()?),
            23 => {
                let (x, y) = self.plain_pair()?;
                Op::AtomicStore(x, y)
            }
            24 => {
                let (rmw, x, y) = self.pair(RMW)?;
                Op::AtomicRmw(rmw, x, y)
            }
            25 => {
                let (a, b, c) = self.values3()?;
                Op::AtomicCas(a, b, c)
            }
            26 => Op::Fence,
            27 => Op::FramePointer,
            28 => {
                let (a, b, c) = self.values3()?;
                Op::Copy(a, b, c)
            }
            29 => {
                let (x, y) = self.plain_pair()?;
                Op::Move(x, y, self.u32()?)
            }
            30 => Op::Call { sig: self.sig()?, callee: self.value()?, args: self.values()?, results: self.values()? },
            31 => Op::TryCall { sig: self.sig()?, callee: self.value()?, args: self.values()?, normal: self.block()?, handler: self.block()? },
            32 => Op::Jump(self.block()?, self.values()?),
            33 => Op::Brif(self.value()?, self.block()?, self.values()?, self.block()?, self.values()?),
            34 => {
                let (index, default) = (self.value()?, self.block()?);
                let len = self.len()?;
                Op::BrTable(index, default, (0..len).map(|_| self.block()).collect::<Option<_>>()?)
            }
            35 => Op::Return(self.values()?),
            36 => Op::Trap,
            37 => Op::Trapz(self.value()?),
            38 => Op::Assume(self.value()?),
            _ => return None,
        })
    }
}
