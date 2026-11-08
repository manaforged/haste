use crate::ir::{Binary, Block, Function, Op, Type, Value};

fn types(func: &Function, values: &[Value]) -> Vec<Type> {
    values.iter().map(|value| func.ty(*value)).collect()
}

fn edge(func: &Function, block: Block, args: &[Value]) -> Result<(), String> {
    let params = types(func, &func.blocks[block.index()].params);
    if params != types(func, args) {
        return Err(format!("block{} takes {params:?}, given {:?}", block.0, types(func, args)));
    }
    Ok(())
}

fn call(func: &Function, sig: usize, args: &[Value]) -> Result<(), String> {
    let wanted: Vec<Type> = func.sigs[sig].params.iter().map(|param| param.value_type).collect();
    if wanted != types(func, args) {
        return Err(format!("a call takes {wanted:?}, given {:?}", types(func, args)));
    }
    Ok(())
}

fn binary(func: &Function, op: Binary, x: Value, y: Value) -> Result<(), String> {
    let mixed = matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr | Binary::Iconcat);
    if !mixed && func.ty(x) != func.ty(y) {
        return Err(format!("{op:?} of {:?} and {:?}", func.ty(x), func.ty(y)));
    }
    Ok(())
}

fn condition(func: &Function, value: Value) -> Result<(), String> {
    if func.ty(value) == Type::I128 {
        return Err("a branch, select or trap tests a 128-bit condition".to_owned());
    }
    Ok(())
}

fn shape(func: &Function) -> Result<(), String> {
    let mut entered = vec![false; func.blocks.len()];
    for inst in &func.insts {
        for target in inst.op.targets() {
            entered[target.index()] = true;
        }
    }
    for (index, block) in func.blocks.iter().enumerate() {
        let Some((last, body)) = block.insts.split_last() else {
            if entered[index] {
                return Err(format!("block{index} is a branch target and holds no instructions"));
            }
            continue;
        };
        if !func.insts[last.index()].op.is_terminator() || body.iter().any(|inst| func.insts[inst.index()].op.is_terminator()) {
            return Err(format!("block{index} does not end with exactly one terminator"));
        }
    }
    Ok(())
}

fn op(func: &Function, op: &Op) -> Result<(), String> {
    match op {
        Op::Brif(cond, ..) | Op::Select(cond, ..) | Op::Trapz(cond) if func.ty(*cond) == Type::I128 => condition(func, *cond),
        Op::Binary(kind, x, y) => binary(func, *kind, *x, *y),
        Op::Icmp(_, x, y) | Op::Fcmp(_, x, y) | Op::Select(_, x, y) if func.ty(*x) != func.ty(*y) => {
            Err(format!("{op:?} mixes {:?} and {:?}", func.ty(*x), func.ty(*y)))
        }
        Op::Jump(block, args) => edge(func, *block, args),
        Op::Brif(_, then, then_args, otherwise, else_args) => edge(func, *then, then_args).and(edge(func, *otherwise, else_args)),
        Op::Call { sig, args, .. } => call(func, sig.index(), args),
        Op::TryCall { sig, args, normal, handler, .. } => try_call(func, sig.index(), args, *normal, *handler),
        Op::Return(values) => returned(func, values),
        _ => Ok(()),
    }
}

fn try_call(func: &Function, sig: usize, args: &[Value], normal: Block, handler: Block) -> Result<(), String> {
    call(func, sig, args)?;
    edge(func, normal, &[])?;
    match types(func, &func.blocks[handler.index()].params)[..] {
        [Type::I64] if func.sigs[sig].returns.is_empty() => Ok(()),
        _ => Err("a try_call returns nothing and its handler takes only the exception".to_owned()),
    }
}

fn returned(func: &Function, values: &[Value]) -> Result<(), String> {
    let wanted: Vec<Type> = func.signature.returns.iter().map(|param| param.value_type).collect();
    if wanted != types(func, values) {
        return Err(format!("return of {:?} from a function returning {wanted:?}", types(func, values)));
    }
    Ok(())
}

/// Checks that every instruction agrees with the types of its operands, blocks and signatures, and every block ends with one terminator.
pub(crate) fn verify(func: &Function) -> Result<(), String> {
    let entry = func.entry.ok_or("the function has no entry block")?;
    let wanted: Vec<Type> = func.signature.params.iter().map(|param| param.value_type).collect();
    if types(func, &func.blocks[entry.index()].params) != wanted {
        return Err("the entry block parameters do not match the signature".to_owned());
    }
    for inst in &func.insts {
        op(func, &inst.op)?;
    }
    shape(func)
}
