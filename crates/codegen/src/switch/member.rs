use crate::fold::extend;
use crate::ir::{Binary, Block, Function, Inst, IntCC, Op, Type, Value};

use super::emit::push_node;
use super::{Edge, Info, Run, same_edge};

const MAX_VALUES: u64 = 4;
const CCMP_IMM_MAX: u64 = 31;
const CMP_IMM_MAX: u64 = 4095;

fn run_cost(ty: Type, r: &Run) -> u64 {
    if r.lo == r.hi || r.lo == 0 || r.hi == ty.unsigned_max() { 2 } else { 3 }
}

/// The values of `runs` when all of them reach one edge and an Or of equality compares beats one test per run.
pub(super) fn member_values(func: &Function, info: &Info, ty: Type, (runs, covered): (&[Run], u64)) -> Option<Vec<u64>> {
    let (first, rest) = runs.split_first()?;
    if rest.is_empty() || covered > MAX_VALUES || rest.iter().any(|r| !same_edge(func, info, &first.hit, &r.hit)) {
        return None;
    }
    let values: Vec<u64> = runs.iter().flat_map(|r| r.lo..=r.hi).collect();
    if values.iter().any(|&x| x > CMP_IMM_MAX) || values.iter().filter(|&&x| x > CCMP_IMM_MAX).count() > 1 {
        return None;
    }
    let per_run: u64 = runs.iter().map(|r| run_cost(ty, r)).sum();
    (values.len() as u64 + 1 < per_run).then_some(values)
}

fn emit_eq(func: &mut Function, (v, ty): (Value, Type), x: u64, insts: &mut Vec<Inst>) -> Value {
    let (c, eq) = (func.value(ty), func.value(Type::I8));
    insts.push(func.add(Op::Const(extend(x as i64, ty, true)), Some(c)));
    insts.push(func.add(Op::Icmp(IntCC::Equal, v, c), Some(eq)));
    eq
}

/// Replaces the branch at `at` with one branch on `v` equal to `head` or any of `rest`.
pub(super) fn emit_member(func: &mut Function, at: (usize, Inst), (v, ty): (Value, Type), (head, rest): (u64, &[u64]), (hit, otherwise): (&Edge, Block)) -> Block {
    let mut insts = Vec::with_capacity(3 * rest.len() + 3);
    let mut any = emit_eq(func, (v, ty), head, &mut insts);
    for &x in rest {
        let (eq, or) = (emit_eq(func, (v, ty), x, &mut insts), func.value(Type::I8));
        insts.push(func.add(Op::Binary(Binary::Bor, any, eq), Some(or)));
        any = or;
    }
    insts.push(func.add(Op::Brif(any, hit.block, hit.args.clone(), otherwise, Vec::new()), None));
    push_node(func, Some(at), insts)
}
