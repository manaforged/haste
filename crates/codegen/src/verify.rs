use crate::ir::{AbiParam, Binary, Block, Function, Op, Type, Value};

fn value_types<'a>(func: &'a Function, values: &'a [Value]) -> impl Iterator<Item = Type> + 'a {
    values.iter().map(|value| func.ty(*value))
}

fn abi_types(params: &[AbiParam]) -> impl Iterator<Item = Type> + '_ {
    params.iter().map(|param| param.value_type)
}

fn types(func: &Function, values: &[Value]) -> Vec<Type> {
    value_types(func, values).collect()
}

/// Whether `values` have exactly the types of `params`.
fn typed(func: &Function, values: &[Value], params: &[AbiParam]) -> bool {
    value_types(func, values).eq(abi_types(params))
}

fn edge(func: &Function, block: Block, args: &[Value]) -> Result<(), String> {
    let params = &func.blocks[block.index()].params;
    if !value_types(func, params).eq(value_types(func, args)) {
        return Err(format!("block{} takes {:?}, given {:?}", block.0, types(func, params), types(func, args)));
    }
    Ok(())
}

fn call(func: &Function, sig: usize, args: &[Value]) -> Result<(), String> {
    let params = &func.sigs[sig].params;
    if !typed(func, args, params) {
        return Err(format!("a call takes {:?}, given {:?}", abi_types(params).collect::<Vec<_>>(), types(func, args)));
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
        Op::Icmp(_, x, y) | Op::Fits(_, _, x, y) | Op::Fcmp(_, x, y) | Op::Select(_, x, y) if func.ty(*x) != func.ty(*y) => {
            Err(format!("{op:?} mixes {:?} and {:?}", func.ty(*x), func.ty(*y)))
        }
        Op::Jump(block, args) => edge(func, *block, args),
        Op::Brif(_, then, then_args, otherwise, else_args) => edge(func, *then, then_args).and(edge(func, *otherwise, else_args)),
        Op::BrTable(index, default, targets) => {
            if !func.ty(*index).is_small_int() {
                return Err(format!("a br_table index must be an integer, got {:?}", func.ty(*index)));
            }
            edge(func, *default, &[])?;
            targets.iter().try_for_each(|target| edge(func, *target, &[]))
        }
        Op::Call { sig, args, .. } => call(func, sig.index(), args),
        Op::TryCall { sig, args, normal, handler, .. } => try_call(func, sig.index(), args, *normal, *handler),
        Op::Return(values) => returned(func, values),
        _ => Ok(()),
    }
}

fn try_call(func: &Function, sig: usize, args: &[Value], normal: Block, handler: Block) -> Result<(), String> {
    call(func, sig, args)?;
    edge(func, normal, &[])?;
    match func.blocks[handler.index()].params[..] {
        [exception] if func.ty(exception) == Type::I64 && func.sigs[sig].returns.is_empty() => Ok(()),
        _ => Err("a try_call returns nothing and its handler takes only the exception".to_owned()),
    }
}

fn returned(func: &Function, values: &[Value]) -> Result<(), String> {
    let returns = &func.signature.returns;
    if !typed(func, values, returns) {
        return Err(format!("return of {:?} from a function returning {:?}", types(func, values), abi_types(returns).collect::<Vec<_>>()));
    }
    Ok(())
}

/// Checks that every instruction agrees with the types of its operands, blocks and signatures, and every block ends with one terminator.
pub(crate) fn verify(func: &Function) -> Result<(), String> {
    let entry = func.entry.ok_or("the function has no entry block")?;
    if !typed(func, &func.blocks[entry.index()].params, &func.signature.params) {
        return Err("the entry block parameters do not match the signature".to_owned());
    }
    for inst in &func.insts {
        if matches!(inst.op, Op::SlotAddr(_) | Op::Symbol(_) | Op::Runtime(_)) && inst.result.map(|res| func.ty(res)) != Some(Type::I64) {
            return Err("a symbol relocation instruction must define an I64 value".to_owned());
        }
        op(func, &inst.op)?;
    }
    shape(func)
}
