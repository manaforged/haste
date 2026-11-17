use super::count::{Counted, counted, head_defs, outside_preds};
use super::{pop_terminator, remap_other_blocks, remap_val};
use crate::fill::BYTES;
use crate::fold::extend;
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Value};
use crate::reloc::RuntimeSymbol;

const MAX_BODY: usize = 24;

struct Fill {
    value: Value,
    byte: Option<i64>,
    addr_idx: usize,
    off: i32,
    width: i64,
}

/// Instructions appended in order to `out`, each with a fresh result in `func`.
pub(super) struct Emit<'f> {
    pub(super) func: &'f mut Function,
    pub(super) out: Vec<Inst>,
}

impl Emit<'_> {
    /// Appends `op` with a fresh result of type `ty`.
    pub(super) fn op(&mut self, op: Op, ty: Type) -> Value {
        let value = self.func.value(ty);
        let inst = self.func.add(op, Some(value));
        self.out.push(inst);
        value
    }

    /// Appends the constant `imm` of type `ty`.
    pub(super) fn konst(&mut self, imm: i64, ty: Type) -> Value {
        self.op(Op::Const(imm), ty)
    }

    fn resize(&mut self, value: Value, ty: Type) -> Value {
        let from = self.func.ty(value);
        match from.bits().cmp(&ty.bits()) {
            std::cmp::Ordering::Equal => value,
            std::cmp::Ordering::Less => self.op(Op::Convert(Convert::Uextend, value), ty),
            std::cmp::Ordering::Greater => self.op(Op::Convert(Convert::Ireduce, value), ty),
        }
    }
}

fn fill_store(func: &Function, body: &[Inst], params: &[Value], defs: &[bool], global_consts: &[Option<i64>]) -> Option<Fill> {
    let mut effects = body.iter().copied().filter(|&i| !func.insts[i.index()].op.is_speculable());
    let store = effects.next()?;
    if effects.next().is_some() {
        return None;
    }
    let Op::Store(value, addr, off) = func.insts[store.index()].op else { return None };
    let ty = func.ty(value);
    let addr_idx = params.iter().position(|&p| p == addr)?;
    if !ty.is_small_int() || func.ty(addr) != Type::I64 {
        return None;
    }
    let byte = match global_consts[value.index()] {
        Some(imm) if extend(imm, ty, false) == extend((imm & 0xFF).wrapping_mul(BYTES), ty, false) => Some(imm & 0xFF),
        None if ty == Type::I8 && !defs[value.index()] => None,
        _ => return None,
    };
    Some(Fill { value, byte, addr_idx, off, width: i64::from(ty.bytes()) })
}

/// The constant step of each loop parameter, zero when it passes through unchanged, or `None` when it does not step by a constant.
pub(super) fn steps(func: &Function, body: &[Inst], params: &[Value], back_args: &[Value], global_consts: &[Option<i64>]) -> Vec<Option<i64>> {
    let step = |param: Value, next: Value| -> Option<i64> {
        if next == param {
            return Some(0);
        }
        let inst = body.iter().copied().find(|&i| func.insts[i.index()].result == Some(next))?;
        let Op::Binary(Binary::Iadd, x, y) = func.insts[inst.index()].op else { return None };
        let by = if x == param { global_consts[y.index()] } else if y == param { global_consts[x.index()] } else { None };
        by.filter(|_| func.ty(param).is_small_int())
    };
    params.iter().zip(back_args).map(|(&param, &next)| step(param, next)).collect()
}

/// For each value of loop `b`, whether another block or the exit edge reads it, when every such read is a parameter or a back-edge value.
pub(super) fn outside_uses(func: &Function, b: usize, defs: &[bool], (params, back_args): (&[Value], &[Value]), exit_args: &[Value]) -> Option<Vec<bool>> {
    let mut wanted = vec![false; func.types.len()];
    let mut fine = true;
    let mut see = |v: Value| {
        if defs[v.index()] {
            fine &= params.contains(&v) || back_args.contains(&v);
            wanted[v.index()] = true;
        }
    };
    exit_args.iter().for_each(|&v| see(v));
    for (_, block) in func.blocks.iter().enumerate().filter(|&(o, _)| o != b) {
        for &inst in &block.insts {
            func.insts[inst.index()].op.for_operands(&mut see);
        }
    }
    fine.then_some(wanted)
}

fn trips(e: &mut Emit<'_>, count: &Counted, start: Value) -> Value {
    let ty = count.iv_ty;
    let one = e.konst(1, ty);
    let base = if count.bias == 0 {
        start
    } else {
        let bias = e.konst(count.bias, ty);
        e.op(Op::Binary(Binary::Iadd, start, bias), ty)
    };
    let first = e.op(Op::Binary(Binary::Iadd, base, one), ty);
    let enter = e.op(Op::Icmp(IntCC::UnsignedLessThan, first, count.limit), Type::I8);
    let span = e.op(Op::Binary(Binary::Isub, count.limit, base), ty);
    e.op(Op::Select(enter, span, one), ty)
}

fn call_memset(e: &mut Emit<'_>, fill: &Fill, params: &[Value], trips: Value) {
    let count = e.resize(trips, Type::I64);
    let bytes = if fill.width == 1 {
        count
    } else {
        let width = e.konst(fill.width, Type::I64);
        e.op(Op::Binary(Binary::Imul, count, width), Type::I64)
    };
    let base = params[fill.addr_idx];
    let to = if fill.off == 0 {
        base
    } else {
        let off = e.konst(i64::from(fill.off), Type::I64);
        e.op(Op::Binary(Binary::Iadd, base, off), Type::I64)
    };
    let byte = match fill.byte {
        Some(byte) => e.konst(byte, Type::I64),
        None => e.resize(fill.value, Type::I64),
    };
    let callee = e.op(Op::Runtime(RuntimeSymbol::Memset), Type::I64);
    let args = vec![to, byte, bytes];
    let sig = crate::native::signature(e.func, &args, None);
    let call = e.func.add(Op::Call { sig, callee, args, results: Vec::new() }, None);
    e.out.push(call);
}

fn finals(e: &mut Emit<'_>, (params, back_args): (&[Value], &[Value]), steps: &[i64], wanted: &[bool], trips: Value) -> Vec<(Value, Value)> {
    let mut map = Vec::new();
    for ((&param, &next), &by) in params.iter().zip(back_args).zip(steps) {
        if by == 0 || !(wanted[param.index()] || wanted[next.index()]) {
            continue;
        }
        let ty = e.func.ty(param);
        let count = e.resize(trips, ty);
        let scaled = if by == 1 {
            count
        } else {
            let by = e.konst(by, ty);
            e.op(Op::Binary(Binary::Imul, count, by), ty)
        };
        let after = e.op(Op::Binary(Binary::Iadd, param, scaled), ty);
        let by = e.konst(by, ty);
        let last = e.op(Op::Binary(Binary::Isub, after, by), ty);
        map.push((next, after));
        map.push((param, last));
    }
    map
}

/// Replaces the one-block loop `b` that stores one repeated byte over a counted run of addresses by one `memset` call and the closed form of each stepped value.
pub(super) fn run(
    func: &mut Function,
    b: usize,
    exit_test: (Value, bool),
    back_args: &[Value],
    (exit_blk, exit_args): (Block, &[Value]),
    global_consts: &[Option<i64>],
) -> bool {
    let insts = &func.blocks[b].insts;
    if insts.len() > MAX_BODY + 1 || outside_preds(func, b).len() != 1 {
        return false;
    }
    let body = insts[..insts.len() - 1].to_vec();
    let params = func.blocks[b].params.clone();
    if back_args.len() != params.len() {
        return false;
    }
    let defs = head_defs(func, b, &body);
    let Some(count) = counted(func, b, &body, exit_test, back_args, &defs, global_consts) else { return false };
    if count.cmp_cc != IntCC::UnsignedLessThan || count.is_ne || global_consts[count.stride.index()] != Some(1) {
        return false;
    }
    let Some(steps) = steps(func, &body, &params, back_args, global_consts).into_iter().collect::<Option<Vec<i64>>>() else { return false };
    let Some(fill) = fill_store(func, &body, &params, &defs, global_consts) else { return false };
    if steps[fill.addr_idx] != fill.width {
        return false;
    }
    let Some(wanted) = outside_uses(func, b, &defs, (&params, back_args), exit_args) else { return false };
    let mut e = Emit { func, out: Vec::new() };
    let trips = trips(&mut e, &count, params[count.iv_idx]);
    call_memset(&mut e, &fill, &params, trips);
    let map = finals(&mut e, (&params, back_args), &steps, &wanted, trips);
    let Emit { func, mut out } = e;
    let exit_args = exit_args.iter().map(|&v| remap_val(&map, v)).collect();
    out.push(func.add(Op::Jump(exit_blk, exit_args), None));
    pop_terminator(func, b);
    func.blocks[b].insts = out;
    remap_other_blocks(func, &[b], &map);
    true
}
