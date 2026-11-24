use crate::alloc::Location;
use crate::ir::{Function, Value};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeScratch {
    Int(usize),
    Float(usize),
    Stack(usize),
}

#[derive(Clone, Copy)]
pub(crate) enum Step {
    Move(Value, Value),
    Save(Value, EdgeScratch),
    Restore(EdgeScratch, Value),
}

#[derive(Clone, Copy)]
enum Source {
    Value(Value),
    Scratch(EdgeScratch),
}

pub(crate) fn same_home<R: Copy + PartialEq>(
    locations: &[Location<R>],
    values: &[i32],
    a: Value,
    b: Value,
) -> bool {
    a == b
        || matches!((locations[a.index()], locations[b.index()]), (Location::Reg(x), Location::Reg(y)) if x == y)
        || matches!((locations[a.index()], locations[b.index()]), (Location::Float(x), Location::Float(y)) if x == y)
        || (matches!(locations[a.index()], Location::Slot)
            && matches!(locations[b.index()], Location::Slot)
            && values[a.index()] == values[b.index()])
}

fn edge_kind<R>(func: &Function, locations: &[Location<R>], arg: Value, param: Value) -> usize {
    if matches!(locations[arg.index()], Location::Reg(_))
        && matches!(locations[param.index()], Location::Reg(_))
    {
        return 0;
    }
    if matches!(locations[arg.index()], Location::Float(_))
        && matches!(locations[param.index()], Location::Float(_))
    {
        return 1;
    }
    let (a_ty, p_ty) = (func.ty(arg), func.ty(param));
    if a_ty.bytes() <= 8
        && p_ty.bytes() <= 8
        && !a_ty.in_float_register()
        && !p_ty.in_float_register()
        && !matches!(locations[arg.index()], Location::Float(_))
        && !matches!(locations[param.index()], Location::Float(_))
    {
        0
    } else if a_ty.in_float_register() && p_ty.in_float_register() {
        1
    } else {
        2
    }
}

fn alloc_scratch(busy: &mut Vec<bool>) -> usize {
    match busy.iter().position(|&used| !used) {
        Some(idx) => {
            busy[idx] = true;
            idx
        }
        None => {
            busy.push(true);
            busy.len() - 1
        }
    }
}

fn reads_home<R: Copy + PartialEq>(
    locations: &[Location<R>],
    values: &[i32],
    src: Value,
    dst: Value,
) -> bool {
    same_home(locations, values, src, dst)
        || matches!((locations[src.index()], locations[dst.index()]), (Location::Offset(x, _), Location::Reg(y)) if x == y)
}

pub(crate) fn edge_steps<R: Copy + PartialEq>(
    func: &Function,
    locations: &[Location<R>],
    values: &[i32],
    args: &[Value],
    params: &[Value],
) -> Vec<Step> {
    let mut pending: Vec<(Source, Value)> = args
        .iter()
        .zip(params)
        .filter(|(arg, param)| !same_home(locations, values, **arg, **param))
        .map(|(arg, param)| (Source::Value(*arg), *param))
        .collect();
    let mut steps = Vec::with_capacity(pending.len());
    let mut busy = [Vec::new(), Vec::new(), Vec::new()];
    while !pending.is_empty() {
        let read = |at: usize| {
            pending.iter().enumerate().any(|(other, (source, _))| {
                other != at
                    && matches!(source, Source::Value(arg) if reads_home(locations, values, *arg, pending[at].1))
            })
        };
        if let Some(at) = (0..pending.len()).find(|at| !read(*at)) {
            steps.push(match pending.remove(at) {
                (Source::Value(arg), param) => Step::Move(arg, param),
                (Source::Scratch(scratch), param) => {
                    match scratch {
                        EdgeScratch::Int(i) => busy[0][i] = false,
                        EdgeScratch::Float(i) => busy[1][i] = false,
                        EdgeScratch::Stack(i) => busy[2][i] = false,
                    }
                    Step::Restore(scratch, param)
                }
            });
            continue;
        }
        let Some((at, arg)) = pending.iter().enumerate().find_map(|(at, (source, _))| {
            if let Source::Value(arg) = source {
                Some((at, *arg))
            } else {
                None
            }
        }) else {
            break;
        };
        let kind = edge_kind(func, locations, arg, pending[at].1);
        let scratch = match kind {
            0 => match alloc_scratch(&mut busy[0]) {
                i if i < 2 => EdgeScratch::Int(i),
                i => {
                    busy[0][i] = false;
                    EdgeScratch::Stack(alloc_scratch(&mut busy[2]))
                }
            },
            1 => match alloc_scratch(&mut busy[1]) {
                i if i < 2 => EdgeScratch::Float(i),
                i => {
                    busy[1][i] = false;
                    EdgeScratch::Stack(alloc_scratch(&mut busy[2]))
                }
            },
            _ => EdgeScratch::Stack(alloc_scratch(&mut busy[2])),
        };
        steps.push(Step::Save(arg, scratch));
        pending[at].0 = Source::Scratch(scratch);
    }
    steps
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ArgStep<T> {
    Move(T, T),
    Save(T),
    Restore(T),
    Load(Value, T),
}

#[derive(Clone, Copy)]
enum RegSrc<T> {
    Reg(T),
    Saved,
}

pub(crate) fn reg_steps<T: Copy + PartialEq>(
    pairs: impl IntoIterator<Item = (T, T)>,
) -> Vec<ArgStep<T>> {
    let mut reg_moves: Vec<(RegSrc<T>, T)> = pairs
        .into_iter()
        .filter(|(src, dst)| src != dst)
        .map(|(src, dst)| (RegSrc::Reg(src), dst))
        .collect();
    let mut steps = Vec::with_capacity(reg_moves.len() + 1);
    while !reg_moves.is_empty() {
        let blocked = |at: usize| {
            reg_moves.iter().enumerate().any(|(other, &(s, _))| {
                other != at && matches!(s, RegSrc::Reg(r) if r == reg_moves[at].1)
            })
        };
        if let Some(at) = (0..reg_moves.len()).find(|&at| !blocked(at)) {
            steps.push(match reg_moves.remove(at) {
                (RegSrc::Reg(src), dst) => ArgStep::Move(src, dst),
                (RegSrc::Saved, dst) => ArgStep::Restore(dst),
            });
            continue;
        }
        let Some((at, src)) = reg_moves.iter().enumerate().find_map(|(at, &(s, _))| {
            if let RegSrc::Reg(r) = s {
                Some((at, r))
            } else {
                None
            }
        }) else {
            break;
        };
        steps.push(ArgStep::Save(src));
        reg_moves[at].0 = RegSrc::Saved;
    }
    steps
}

pub(crate) fn arg_steps<T: Copy + PartialEq>(
    targets: &[(Value, T)],
    current: impl Fn(Value) -> Option<T>,
) -> Vec<ArgStep<T>> {
    let mut pairs = Vec::new();
    let mut loads = Vec::new();
    for &(val, dst) in targets {
        match current(val) {
            Some(src) => pairs.push((src, dst)),
            None => loads.push((val, dst)),
        }
    }
    let mut steps = reg_steps(pairs);
    steps.extend(loads.into_iter().map(|(val, dst)| ArgStep::Load(val, dst)));
    steps
}
