use crate::coroutine::Coroutine;
use crate::ir::{AbiParam, ArgumentPurpose, Block, Function, Inst, Op, SigRef, Signature, StackSlot, Type, Value};
use crate::reloc::FootprintHasher;

const TAG_SIG: u64 = 1;
const TAG_BLOCK: u64 = 2;
const TAG_PARAM: u64 = 3;
const TAG_ENTRY: u64 = 4;
const TAG_SLOT: u64 = 5;
const TAG_CALL: u64 = 6;
const TAG_TRY_CALL: u64 = 7;
const TAG_JUMP: u64 = 8;
const TAG_BRIF: u64 = 9;
const TAG_RETURN: u64 = 10;
const TAG_INLINE: u64 = 11;
const TAG_COROUTINE: u64 = 12;

#[derive(Clone)]
pub(crate) struct Probe {
    pub hasher: FootprintHasher,
    pub slots: u32,
    pub sig_returns: Vec<(u8, [Type; 4])>,
    pub last_results: Vec<Value>,
}

impl Probe {
    pub fn new(sig: &Signature) -> Self {
        let mut probe = Self { hasher: FootprintHasher::new(), slots: 0, sig_returns: Vec::new(), last_results: Vec::new() };
        probe.hash_sig(sig);
        probe
    }

    fn hash_param(&mut self, param: AbiParam) {
        let (kind, extra) = match param.purpose {
            ArgumentPurpose::Normal => (0_u64, 0_u64),
            ArgumentPurpose::StructReturn => (1, 0),
            ArgumentPurpose::StructArgument(bytes) => (2, u64::from(bytes)),
            ArgumentPurpose::Dict => (3, 0),
        };
        self.hasher.word((u64::from(param.value_type as u8) << 40) | (kind << 32) | extra);
    }

    fn hash_sig(&mut self, sig: &Signature) {
        self.hasher.word((TAG_SIG << 48) | (u64::from(sig.call_conv as u8) << 40) | ((sig.params.len() as u64) << 16) | (sig.returns.len() as u64));
        for &param in &sig.params {
            self.hash_param(param);
        }
        for &ret in &sig.returns {
            self.hash_param(ret);
        }
    }

    pub fn block(&mut self, block: Block) {
        self.hasher.word((TAG_BLOCK << 48) | u64::from(block.0));
    }

    pub fn block_param(&mut self, block: Block, ty: Type, val: Value) {
        self.hasher.word((TAG_PARAM << 48) | (u64::from(ty as u8) << 40) | (u64::from(block.0) << 20) | u64::from(val.0));
    }

    pub fn entry(&mut self, block: Block) {
        self.hasher.word((TAG_ENTRY << 48) | u64::from(block.0));
    }

    pub fn slot(&mut self, size: u32, align_shift: u8) -> StackSlot {
        let id = StackSlot(self.slots);
        self.slots += 1;
        self.hasher.word((TAG_SLOT << 48) | (u64::from(align_shift) << 32) | u64::from(size));
        id
    }

    pub fn sig(&mut self, sig: &Signature) -> SigRef {
        self.hash_sig(sig);
        let mut rets = [Type::I8; 4];
        let count = sig.returns.len().min(4);
        for (dst, src) in rets.iter_mut().zip(&sig.returns[..count]) {
            *dst = src.value_type;
        }
        let id = SigRef(self.sig_returns.len() as u32);
        self.sig_returns.push((count as u8, rets));
        id
    }

    fn values(&mut self, vals: &[Value]) {
        self.hasher.word(vals.len() as u64);
        for &val in vals {
            self.hasher.word(u64::from(val.0));
        }
    }

    pub fn op(&mut self, block: Block, op: &Op, ty: Option<Type>) {
        let ty_tag = ty.map_or(0xFF_u64, |t| u64::from(t as u8));
        self.hasher.word((u64::from(block.0) << 16) | ty_tag);
        if !self.hash_value_op(op) {
            self.hash_mem_op(op);
        }
    }

    fn hash_value_op(&mut self, op: &Op) -> bool {
        match *op {
            Op::Const(imm) => {
                self.hasher.word(100);
                self.hasher.word(imm as u64);
            }
            Op::SlotAddr(sym) => self.hasher.word((101 << 32) | u64::from(sym)),
            Op::Symbol(sym) => self.hasher.word((102 << 32) | u64::from(sym)),
            Op::Runtime(rt) => self.hasher.word((103 << 32) | u64::from(rt as u8)),
            Op::Unary(kind, x) => self.hasher.word((104 << 48) | (u64::from(kind as u8) << 32) | u64::from(x.0)),
            Op::Binary(kind, x, y) => {
                self.hasher.word((105 << 48) | u64::from(kind as u8));
                self.hasher.word((u64::from(x.0) << 32) | u64::from(y.0));
            }
            Op::Convert(kind, x) => self.hasher.word((106 << 48) | (u64::from(kind as u8) << 32) | u64::from(x.0)),
            Op::Icmp(cc, x, y) => {
                self.hasher.word((107 << 48) | u64::from(cc as u8));
                self.hasher.word((u64::from(x.0) << 32) | u64::from(y.0));
            }
            Op::Fits(kind, signed, x, y) => {
                self.hasher.word((108 << 48) | (u64::from(signed) << 40) | u64::from(kind as u8));
                self.hasher.word((u64::from(x.0) << 32) | u64::from(y.0));
            }
            Op::Fcmp(cc, x, y) => {
                self.hasher.word((109 << 48) | u64::from(cc as u8));
                self.hasher.word((u64::from(x.0) << 32) | u64::from(y.0));
            }
            Op::Select(c, x, y) => {
                self.hasher.word((110 << 32) | u64::from(c.0));
                self.hasher.word((u64::from(x.0) << 32) | u64::from(y.0));
            }
            Op::ByteCmp(cc, x, y) => {
                self.hasher.word((111 << 48) | u64::from(cc as u8));
                self.hasher.word((u64::from(x.0) << 32) | u64::from(y.0));
            }
            Op::Splat(x) => self.hasher.word((112 << 32) | u64::from(x.0)),
            Op::Movemask(x) => self.hasher.word((113 << 32) | u64::from(x.0)),
            _ => return false,
        }
        true
    }

    fn hash_mem_op(&mut self, op: &Op) {
        match *op {
            Op::Load(addr, off) => self.hasher.word((120 << 56) | (u64::from(addr.0) << 32) | u64::from(off as u32)),
            Op::Store(val, addr, off) => {
                self.hasher.word((121 << 32) | u64::from(val.0));
                self.hasher.word((u64::from(addr.0) << 32) | u64::from(off as u32));
            }
            Op::StackAddr(slot, off) => self.hasher.word((122 << 56) | (u64::from(slot.0) << 32) | u64::from(off as u32)),
            Op::StackLoad(slot, off) => self.hasher.word((123 << 56) | (u64::from(slot.0) << 32) | u64::from(off as u32)),
            Op::StackStore(val, slot, off) => {
                self.hasher.word((124 << 32) | u64::from(val.0));
                self.hasher.word((u64::from(slot.0) << 32) | u64::from(off as u32));
            }
            Op::AtomicLoad(addr) => self.hasher.word((125 << 32) | u64::from(addr.0)),
            Op::AtomicStore(val, addr) => self.hasher.word((126 << 56) | (u64::from(val.0) << 28) | u64::from(addr.0)),
            Op::AtomicRmw(rmw, addr, val) => {
                self.hasher.word((127 << 48) | u64::from(rmw as u8));
                self.hasher.word((u64::from(addr.0) << 32) | u64::from(val.0));
            }
            Op::AtomicCas(addr, exp, rep) => {
                self.hasher.word((128 << 32) | u64::from(addr.0));
                self.hasher.word((u64::from(exp.0) << 32) | u64::from(rep.0));
            }
            Op::Copy(to, from, bytes) => {
                self.hasher.word((129 << 32) | u64::from(to.0));
                self.hasher.word((u64::from(from.0) << 32) | u64::from(bytes.0));
            }
            Op::Move(to, from, bytes) => {
                self.hasher.word((134 << 32) | u64::from(to.0));
                self.hasher.word((u64::from(from.0) << 32) | u64::from(bytes));
            }
            Op::Fence => self.hasher.word(130),
            Op::FramePointer => self.hasher.word(131),
            Op::Trap => self.hasher.word(132),
            Op::Trapz(cond) => self.hasher.word((133 << 32) | u64::from(cond.0)),
            _ => {}
        }
    }

    pub fn call(&mut self, func: &mut Function, block: Block, sig: SigRef, callee: Value, args: &[Value]) -> Inst {
        let (count, rets) = self.sig_returns[sig.index()];
        self.last_results.clear();
        for &ty in &rets[..usize::from(count)] {
            self.last_results.push(func.value(ty));
        }
        self.hasher.word((TAG_CALL << 48) | (u64::from(block.0) << 24) | u64::from(sig.0));
        self.hasher.word(u64::from(callee.0));
        self.values(args);
        Inst(u32::MAX)
    }

    pub fn try_call(&mut self, block: Block, sig: SigRef, (callee, args): (Value, &[Value]), (normal, handler): (Block, Block)) -> Inst {
        self.hasher.word((TAG_TRY_CALL << 48) | (u64::from(block.0) << 24) | u64::from(sig.0));
        self.hasher.word((u64::from(callee.0) << 32) | (u64::from(normal.0) << 16) | u64::from(handler.0));
        self.values(args);
        Inst(u32::MAX)
    }

    pub fn jump(&mut self, block: Block, target: Block, args: &[Value]) -> Inst {
        self.hasher.word((TAG_JUMP << 48) | (u64::from(block.0) << 24) | u64::from(target.0));
        self.values(args);
        Inst(u32::MAX)
    }

    pub fn brif(&mut self, block: Block, cond: Value, (then, then_args): (Block, &[Value]), (otherwise, else_args): (Block, &[Value])) -> Inst {
        self.hasher.word((TAG_BRIF << 48) | (u64::from(block.0) << 24) | u64::from(cond.0));
        self.hasher.word((u64::from(then.0) << 32) | u64::from(otherwise.0));
        self.values(then_args);
        self.values(else_args);
        Inst(u32::MAX)
    }

    pub fn return_(&mut self, block: Block, values: &[Value]) -> Inst {
        self.hasher.word((TAG_RETURN << 48) | u64::from(block.0));
        self.values(values);
        Inst(u32::MAX)
    }

    pub fn inline(&mut self, block: Block, callee_item: u128, callee_footprint: u128, args: &[Value], slots: u32, sigs: usize) {
        self.slots += slots;
        self.sig_returns.resize(self.sig_returns.len() + sigs, (0, [Type::I8; 4]));
        self.hasher.word((TAG_INLINE << 48) | u64::from(block.0));
        self.hasher.u128(callee_item);
        self.hasher.u128(callee_footprint);
        self.values(args);
    }

    pub fn finish(mut self, coroutine: Option<&Coroutine>) -> u128 {
        if let Some(spec) = coroutine {
            self.hasher.word((TAG_COROUTINE << 48) | (spec.object as u64));
            self.hasher.word(((spec.header.0 as u32 as u64) << 32) | (spec.header.1 as u32 as u64));
            self.hasher.word(spec.resumes.len() as u64);
            for &b in &spec.resumes {
                self.hasher.word(u64::from(b.0));
            }
            self.hasher.word(spec.done.len() as u64);
            for &b in &spec.done {
                self.hasher.word(u64::from(b.0));
            }
        }
        self.hasher.finish()
    }
}
