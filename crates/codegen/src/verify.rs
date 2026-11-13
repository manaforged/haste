use crate::ir::{AbiParam, Binary, Block, Function, Lane, Op, ReduceOp, ShuffleMask, TargetOpId, Type, Unary, Value};

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

fn is_float_or_vec(ty: Type) -> bool {
    ty.is_float() || ty.lane().is_some_and(Lane::is_float)
}

fn is_int_or_vec(ty: Type) -> bool {
    ty.is_small_int() || ty == Type::I128 || ty.lane().is_some_and(Lane::is_int)
}

fn unary(func: &Function, op: Unary, x: Value, result: Option<Value>) -> Result<(), String> {
    let ty = func.ty(x);
    if result.map(|r| func.ty(r)) != Some(ty) {
        return Err(format!("{op:?} result type mismatch for {ty:?}"));
    }
    let ok = match op {
        Unary::Fneg | Unary::Fabs | Unary::Sqrt | Unary::Floor | Unary::Ceil | Unary::Trunc => is_float_or_vec(ty),
        _ => is_int_or_vec(ty),
    };
    if !ok {
        return Err(format!("{op:?} does not accept {ty:?}"));
    }
    Ok(())
}

fn binary(func: &Function, op: Binary, x: Value, y: Value, result: Option<Value>) -> Result<(), String> {
    let (tx, ty) = (func.ty(x), func.ty(y));
    let shift = matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr);
    if op == Binary::Iconcat {
        if tx != Type::I64 || ty != Type::I64 || result.map(|r| func.ty(r)) != Some(Type::I128) {
            return Err(format!("Iconcat of {tx:?} and {ty:?}"));
        }
        return Ok(());
    }
    if !(tx == ty || (shift && (tx.is_small_int() || tx == Type::I128 || tx.is_vector()) && ty.is_small_int())) {
        return Err(format!("{op:?} of {tx:?} and {ty:?}"));
    }
    if result.map(|r| func.ty(r)) != Some(tx) {
        return Err(format!("{op:?} result type mismatch for {tx:?}"));
    }
    let ok = match op {
        Binary::Fadd | Binary::Fsub | Binary::Fmul | Binary::Fdiv | Binary::Fmin | Binary::Fmax | Binary::Fcopysign => is_float_or_vec(tx),
        Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => is_int_or_vec(tx),
        _ => true,
    };
    if !ok {
        return Err(format!("{op:?} does not accept {tx:?}"));
    }
    Ok(())
}

fn cmp_result_ty(ty: Type) -> Type {
    match ty {
        Type::Vec { lane, count } => Type::Vec { lane: lane.int_mask(), count },
        _ => Type::I8,
    }
}

fn select_check(func: &Function, cond: Value, x: Value, y: Value, result: Option<Value>) -> Result<(), String> {
    let (tc, tx, ty) = (func.ty(cond), func.ty(x), func.ty(y));
    if tx != ty || result.map(|r| func.ty(r)) != Some(tx) {
        return Err(format!("Select mixes {tx:?} and {ty:?}"));
    }
    if let Some((c_lane, c_count)) = tc.vec_shape() {
        let valid = tx.vec_shape().is_some_and(|(v_lane, v_count)| c_count == v_count && c_lane == v_lane.int_mask());
        if !valid {
            return Err(format!("vector Select condition {tc:?} does not match {tx:?}"));
        }
        return Ok(());
    }
    condition(func, cond)
}

fn target_arity(id: TargetOpId) -> usize {
    match id {
        TargetOpId::Arm64AesMc | TargetOpId::Arm64AesImc | TargetOpId::X64AesImc => 1,
        TargetOpId::Arm64Sha256H | TargetOpId::Arm64Sha256H2 | TargetOpId::Arm64Sha256Su1 | TargetOpId::X64Sha256Rnds2 => 3,
        _ => 2,
    }
}

fn vector_op(func: &Function, op: &Op, result: Option<Value>) -> Result<(), String> {
    match *op {
        Op::Shuffle(a, b, mask) => {
            let (ta, tb) = (func.ty(a), func.ty(b));
            let Some((_, count)) = ta.vec_shape() else { return Err(format!("Shuffle requires vector operands, got {ta:?}")) };
            if tb != ta || result.map(|r| func.ty(r)) != Some(ta) {
                return Err(format!("Shuffle type mismatch: {ta:?}, {tb:?}"));
            }
            match mask {
                ShuffleMask::Const(idxs) => {
                    let n = usize::from(count);
                    if idxs[..n].iter().any(|&i| i >= 2 * count) || idxs[n..].iter().any(|&i| i != 0) {
                        return Err(format!("Shuffle mask out of range for {ta:?}"));
                    }
                }
                ShuffleMask::Dyn(idx) => {
                    let ti = func.ty(idx);
                    if !matches!(ti, Type::Vec { lane: Lane::I8, count: c } if u32::from(c) == ta.bytes()) {
                        return Err(format!("Shuffle dynamic index {ti:?} does not match {ta:?}"));
                    }
                }
            }
        }
        Op::Extract(vec, lane_idx) => {
            let Some((lane, count)) = func.ty(vec).vec_shape() else {
                return Err(format!("Extract requires vector operand, got {:?}", func.ty(vec)));
            };
            if lane_idx >= count || result.map(|r| func.ty(r)) != Some(lane.to_type()) {
                return Err(format!("invalid Extract lane {lane_idx} on {:?}", func.ty(vec)));
            }
        }
        Op::Insert(vec, scalar, lane_idx) => {
            let tv = func.ty(vec);
            let Some((lane, count)) = tv.vec_shape() else {
                return Err(format!("Insert requires vector operand, got {tv:?}"));
            };
            if lane_idx >= count || func.ty(scalar) != lane.to_type() || result.map(|r| func.ty(r)) != Some(tv) {
                return Err(format!("invalid Insert lane {lane_idx} scalar {:?} on {tv:?}", func.ty(scalar)));
            }
        }
        Op::Reduce(kind, vec) => {
            let Some((lane, _)) = func.ty(vec).vec_shape() else {
                return Err(format!("Reduce requires vector operand, got {:?}", func.ty(vec)));
            };
            let (ok_lane, want) = match kind {
                ReduceOp::Movemask => (lane.is_int(), Type::I32),
                ReduceOp::Fmin | ReduceOp::Fmax => (lane.is_float(), lane.to_type()),
                _ => (lane.is_int(), lane.to_type()),
            };
            if !ok_lane || result.map(|r| func.ty(r)) != Some(want) {
                return Err(format!("invalid Reduce({kind:?}) on {:?}", func.ty(vec)));
            }
        }
        Op::TargetOp(id, args) => {
            if args.as_slice().len() != target_arity(id) || result.is_none() {
                return Err(format!("invalid TargetOp({id:?}) arity or missing result"));
            }
        }
        _ => {}
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

fn op(func: &Function, op: &Op, result: Option<Value>) -> Result<(), String> {
    match op {
        Op::Brif(cond, ..) | Op::Trapz(cond) if func.ty(*cond) == Type::I128 => condition(func, *cond),
        Op::Select(cond, x, y) => select_check(func, *cond, *x, *y, result),
        Op::Unary(kind, x) => unary(func, *kind, *x, result),
        Op::Binary(kind, x, y) => binary(func, *kind, *x, *y, result),
        Op::Icmp(_, x, y) | Op::Fcmp(_, x, y) => {
            let tx = func.ty(*x);
            if tx != func.ty(*y) || result.map(|r| func.ty(r)) != Some(cmp_result_ty(tx)) {
                return Err(format!("{op:?} compare type mismatch for {tx:?} and {:?}", func.ty(*y)));
            }
            let ok = if matches!(op, Op::Fcmp(..)) { is_float_or_vec(tx) } else { is_int_or_vec(tx) };
            if !ok {
                return Err(format!("{op:?} does not accept {tx:?}"));
            }
            Ok(())
        }
        Op::Fits(_, _, x, y) if func.ty(*x) != func.ty(*y) => {
            Err(format!("{op:?} mixes {:?} and {:?}", func.ty(*x), func.ty(*y)))
        }
        Op::Shuffle(..) | Op::Extract(..) | Op::Insert(..) | Op::Reduce(..) | Op::TargetOp(..) => vector_op(func, op, result),
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
    for &ty in &func.types {
        if ty.is_vector() && !ty.is_valid_vec() {
            return Err(format!("invalid vector type {ty:?}"));
        }
    }
    for inst in &func.insts {
        if matches!(inst.op, Op::SlotAddr(_) | Op::Symbol(_) | Op::Runtime(_)) && inst.result.map(|res| func.ty(res)) != Some(Type::I64) {
            return Err("a symbol relocation instruction must define an I64 value".to_owned());
        }
        op(func, &inst.op, inst.result)?;
    }
    shape(func)
}
