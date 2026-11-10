use crate::ir::{AbiParam, CallConv, Function, Inst, InstData, Op, SigRef, Signature, Type, Value};

/// The most chunks a copy of a constant size is emitted inline with, one scratch register each.
const CHUNKS: usize = 4;

/// The widths and offsets of the loads that copy `bytes` bytes inline, widest first, when there are at most `CHUNKS` of them.
pub(crate) fn chunks(bytes: i64) -> Option<Vec<(u32, u32)>> {
    let mut left = u32::try_from(bytes).ok().filter(|bytes| *bytes <= 8 * CHUNKS as u32)?;
    let mut chunks = Vec::new();
    for width in [8, 4, 2, 1] {
        while left >= width {
            chunks.push((width, bytes as u32 - left));
            left -= width;
        }
    }
    (chunks.len() <= CHUNKS).then_some(chunks)
}

/// Turns each copy whose size is not a constant that `chunks` covers into a call to the C library's `memmove`.
pub(crate) fn lower(func: &mut Function) {
    let mut consts = vec![None; func.types.len()];
    for data in &func.insts {
        if let (Op::Const(imm), Some(result)) = (&data.op, data.result) {
            consts[result.index()] = Some(*imm);
        }
    }
    let mut sig = None;
    for block in 0..func.blocks.len() {
        let mut at = 0;
        while at < func.blocks[block].insts.len() {
            let inst = func.blocks[block].insts[at];
            if let Op::Copy(to, from, bytes) = func.insts[inst.index()].op
                && consts[bytes.index()].and_then(chunks).is_none()
            {
                let sig = *sig.get_or_insert_with(|| memmove_signature(func));
                let callee = Value(func.types.len() as u32);
                func.types.push(Type::I64);
                func.insts.push(InstData { op: Op::Const(libc::memmove as *const () as usize as i64), result: Some(callee) });
                func.blocks[block].insts.insert(at, Inst(func.insts.len() as u32 - 1));
                func.insts[inst.index()].op = Op::Call { sig, callee, args: vec![to, from, bytes], results: Vec::new() };
                at += 1;
            }
            at += 1;
        }
    }
}

fn memmove_signature(func: &mut Function) -> SigRef {
    let mut signature = Signature::new(CallConv::Native);
    signature.params = vec![AbiParam::new(Type::I64); 3];
    func.sigs.push(signature);
    SigRef(func.sigs.len() as u32 - 1)
}
