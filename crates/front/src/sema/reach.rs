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
    pub list: Vec<(Instance, Result<std::rc::Rc<Typed>, String>)>,
    pub index: HashMap<Instance, usize>,
    /// The bodies whose types have no layout yet, which lowering would fail on.
    pub unlaid: Vec<String>,
    calls: Vec<Vec<usize>>,
    untyped: rustc_hash::FxHashSet<usize>,
}

impl Reached {
    /// `roots` and every function they reach through the bodies that type-checked, each once and after every function it reaches.
    pub fn closure(&self, roots: &[usize]) -> Vec<usize> {
        let mut seen = vec![false; self.list.len()];
        let (mut stack, mut found) = (roots.iter().rev().map(|&root| (root, false)).collect::<Vec<_>>(), Vec::new());
        while let Some((at, done)) = stack.pop() {
            if done {
                found.push(at);
                continue;
            }
            if std::mem::replace(&mut seen[at], true) {
                continue;
            }
            stack.push((at, true));
            stack.extend(self.calls[at].iter().map(|&callee| (callee, false)));
        }
        found
    }

    /// The first function in `closure` that did not type-check or, by `broken`, did not compile, with why.
    pub fn failure<'r>(&'r self, closure: &[usize], broken: &'r HashMap<usize, String>) -> Option<(usize, &'r String)> {
        closure.iter().find_map(|&at| Some((at, self.list[at].1.as_ref().err().or_else(|| broken.get(&at))?)))
    }

    /// The index of `instance`, and whether it is new: a new one is listed without its body until `infer_declared` infers it.
    pub fn declare(&mut self, instance: Instance) -> (usize, bool) {
        if let Some(&at) = self.index.get(&instance) {
            return (at, false);
        }
        let at = self.list.len();
        self.index.insert(instance.clone(), at);
        self.list.push((instance, Err(String::new())));
        self.untyped.insert(at);
        (at, true)
    }

    /// Takes back the body of declared function `index`, so the next `infer_declared` infers it again.
    pub fn forget(&mut self, index: usize) {
        self.list[index].1 = Err(String::new());
        self.untyped.insert(index);
    }

    fn link(&mut self, indices: &[usize]) {
        self.calls.resize(self.list.len(), Vec::new());
        for &at in indices {
            let callees = self.list[at].1.as_ref().map(|typed| self.callees(typed).collect()).unwrap_or_default();
            self.calls[at] = callees;
        }
    }

    fn callees<'r>(&'r self, typed: &'r Typed) -> impl Iterator<Item = usize> + 'r {
        typed.calls().filter_map(|(item, substs)| self.index.get(&(item, substs.to_vec())).copied())
    }
}

/// Infers `roots` and every function their bodies call, at the type arguments the calls give.
pub fn reach(cx: &mut Context<'_>, roots: &[Instance]) -> Reached {
    let mut reached = Reached::default();
    let found = drain(cx, &mut reached, roots.to_vec());
    reached.link(&found);
    reached
}

/// Infers again the reached functions `stale` that an edit did not replace, whose source changed, and every function they now call that was not reached yet, and returns the indexes of both.
pub fn reach_again(cx: &mut Context<'_>, reached: &mut Reached, stale: &[usize]) -> Vec<usize> {
    let mut queue = Vec::new();
    let stale: Vec<usize> = stale.iter().copied().filter(|index| reached.index.get(&reached.list[*index].0) == Some(index)).collect();
    for &index in &stale {
        let instance = reached.list[index].0.clone();
        let typed = if cx.undeclared(&instance) { Err("no longer declared".to_owned()) } else { typed_once(cx, &instance, &mut reached.unlaid) };
        queue.extend(calls(cx, &typed));
        reached.list[index].1 = typed;
    }
    let mut found = stale.to_vec();
    found.extend(drain(cx, reached, queue));
    reached.link(&found);
    found
}

/// Drops the reached functions `dead`, whose items an edit replaced, then infers `roots` and every function they call that is not reached, and returns the indexes of those.
pub fn reach_anew(cx: &mut Context<'_>, reached: &mut Reached, dead: &[usize], roots: &[Instance]) -> Vec<usize> {
    for &index in dead {
        reached.index.remove(&reached.list[index].0);
        reached.list[index].1 = Err("replaced by an edit".to_owned());
    }
    reached.link(dead);
    let found = drain(cx, reached, roots.to_vec());
    reached.link(&found);
    found
}

/// Infers declared function `index` when it has no body yet, and returns whether it did.
pub fn infer_declared(cx: &mut Context<'_>, reached: &mut Reached, index: usize) -> bool {
    if !reached.untyped.remove(&index) {
        return false;
    }
    let instance = reached.list[index].0.clone();
    reached.list[index].1 = typed_once(cx, &instance, &mut reached.unlaid);
    true
}

/// The functions that the inferred body `typed` calls, other than intrinsics.
pub fn calls<'t>(cx: &'t Context<'_>, typed: &'t Result<std::rc::Rc<Typed>, String>) -> impl Iterator<Item = Instance> + 't {
    typed.iter().flat_map(|typed| typed.calls()).filter(|(item, _)| cx.intrinsic(*item).is_none()).map(|(item, substs)| (item, substs.to_vec()))
}

fn drain(cx: &mut Context<'_>, reached: &mut Reached, mut queue: Vec<Instance>) -> Vec<usize> {
    let mut added = Vec::new();
    let mut next = 0;
    while next < queue.len() {
        let instance = queue[next].clone();
        next += 1;
        if reached.index.contains_key(&instance) {
            continue;
        }
        let typed = typed_once(cx, &instance, &mut reached.unlaid);
        queue.extend(calls(cx, &typed));
        added.push(reached.list.len());
        reached.index.insert(instance.clone(), reached.list.len());
        reached.list.push((instance, typed));
    }
    added
}

/// The inferred body of `instance`: the one an earlier pass kept, or a fresh inference that is kept when it type-checked, laid out, and asked for nothing.
fn typed_once(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>) -> Result<std::rc::Rc<Typed>, String> {
    if let Some(typed) = cx.store.finals.get(instance).cloned() {
        register(cx, (instance.0, &instance.1), &typed);
        return Ok(typed);
    }
    let asked = cx.asks.count();
    let typed = std::rc::Rc::new(infer(cx, instance.0, &instance.1).map_err(|error| format!("{}: {error:#}", cx.name(instance.0)))?);
    let laid = lay_out(cx, &typed).map_err(|error| unlaid.push(format!("{}: {error:#}", cx.name(instance.0))));
    if laid.is_ok() && cx.asks.count() == asked {
        cx.store.finals.insert(instance.clone(), typed.clone());
    }
    Ok(typed)
}

/// Computes the layout of each ADT, tuple and array type that lowering `typed` reads, so the names those layouts need are asked for while inference can still retry.
fn lay_out(cx: &mut Context<'_>, typed: &Typed) -> anyhow::Result<()> {
    let intrinsic_args: Vec<Ty> = typed.calls().filter(|(item, _)| cx.intrinsic(*item).is_some()).flat_map(|(_, substs)| substs.iter().copied()).collect();
    let locals = typed.locals.iter().map(|local| local.ty);
    for ty in typed.exprs.iter().copied().chain(locals).chain(intrinsic_args) {
        if ty != Ty::NONE && matches!(cx.types.kind(ty), Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..)) {
            cx.layout(ty)?;
        }
    }
    Ok(())
}

/// The two roots a program run calls: std's `start` lang item at the return type of `main`, and `main`.
pub fn main_roots(cx: &mut Context<'_>, main: ItemRef) -> anyhow::Result<[Instance; 2]> {
    let start = cx.lang_item("start").ok_or_else(|| anyhow::anyhow!("no `start` lang item"))?;
    let ret = super::infer::return_type(cx, main)?;
    Ok([(start, vec![ret]), (main, Vec::new())])
}
