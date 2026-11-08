use rustc_hash::FxHashMap as HashMap;

use crate::names::ItemRef;

use super::Context;
use super::infer::{Typed, infer, register};
use super::ty::{Kind, Ty};

/// One function at concrete type arguments.
pub type Instance = (ItemRef, Vec<Ty>);

/// Every function reached from some roots, each with its inferred body or the reason it has none.
#[derive(Default)]
pub struct Reached {
    pub list: Vec<(Instance, Result<Typed, String>)>,
    pub index: HashMap<Instance, usize>,
    /// The bodies whose types have no layout yet, which lowering would fail on.
    pub unlaid: Vec<String>,
}

impl Reached {
    /// `root` and every function it reaches, through the bodies that type-checked.
    pub fn closure(&self, root: usize) -> Vec<usize> {
        let mut seen = vec![false; self.list.len()];
        let (mut stack, mut found) = (vec![root], Vec::new());
        while let Some(at) = stack.pop() {
            if std::mem::replace(&mut seen[at], true) {
                continue;
            }
            found.push(at);
            if let Ok(typed) = &self.list[at].1 {
                stack.extend(self.callees(typed));
            }
        }
        found
    }

    /// The indexes of the functions that `typed` calls.
    pub fn callees<'r>(&'r self, typed: &'r Typed) -> impl Iterator<Item = usize> + 'r {
        typed.calls().filter_map(|(item, substs)| self.index.get(&(item, substs.to_vec())).copied())
    }
}

/// Infers `roots` and every function their bodies call, at the type arguments the calls give.
pub fn reach(cx: &mut Context<'_>, roots: &[Instance]) -> Reached {
    let mut reached = Reached::default();
    let mut queue: Vec<Instance> = roots.to_vec();
    let mut next = 0;
    while next < queue.len() {
        let instance = queue[next].clone();
        next += 1;
        if reached.index.contains_key(&instance) {
            continue;
        }
        let typed = typed_once(cx, &instance, &mut reached.unlaid);
        if let Ok(typed) = &typed {
            let calls = typed.calls().filter(|(item, _)| cx.intrinsic(*item).is_none()).map(|(item, substs)| (item, substs.to_vec()));
            queue.extend(calls);
        }
        reached.index.insert(instance.clone(), reached.list.len());
        reached.list.push((instance, typed));
    }
    reached
}

/// The inferred body of `instance`: the one an earlier pass kept, or a fresh inference that is kept when it type-checked, laid out, and asked for nothing.
fn typed_once(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>) -> Result<Typed, String> {
    if let Some(typed) = cx.finals.get(instance).cloned() {
        register(cx, (instance.0, &instance.1), &typed);
        return Ok(typed);
    }
    let asked = cx.asks.count();
    let typed = infer(cx, instance.0, &instance.1).map_err(|error| format!("{}: {error:#}", cx.name(instance.0)))?;
    let laid = lay_out(cx, &typed).map_err(|error| unlaid.push(format!("{}: {error:#}", cx.name(instance.0))));
    if laid.is_ok() && cx.asks.count() == asked {
        cx.finals.insert(instance.clone(), typed.clone());
    }
    Ok(typed)
}

/// Computes the layout of each ADT, tuple and array type that lowering `typed` reads, so the names those layouts need are asked for while inference can still retry.
fn lay_out(cx: &mut Context<'_>, typed: &Typed) -> anyhow::Result<()> {
    let intrinsic_args: Vec<Ty> = typed.calls().filter(|(item, _)| cx.intrinsic(*item).is_some()).flat_map(|(_, substs)| substs.iter().copied()).collect();
    let locals = typed.locals.iter().map(|local| local.ty);
    for ty in typed.exprs.iter().copied().chain(locals).chain(intrinsic_args) {
        if ty != Ty(u32::MAX) && matches!(cx.types.kind(ty), Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..)) {
            cx.layout(ty)?;
        }
    }
    Ok(())
}
