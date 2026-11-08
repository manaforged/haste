use rustc_hash::FxHashMap as HashMap;

use crate::names::ItemRef;

use super::Context;
use super::infer::{Res, Typed, infer};
use super::ty::Ty;

/// One function at concrete type arguments.
pub type Instance = (ItemRef, Vec<Ty>);

/// Every function reached from some roots, each with its inferred body or the reason it has none.
#[derive(Default)]
pub struct Reached {
    pub list: Vec<(Instance, Result<Typed, String>)>,
    pub index: HashMap<Instance, usize>,
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
        typed.res.values().filter_map(|res| match res {
            Res::Fn(item, substs) => self.index.get(&(*item, substs.clone())).copied(),
            Res::Local(_) => None,
        })
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
        let typed = infer(cx, instance.0, &instance.1).map_err(|error| format!("{}: {error:#}", cx.name(instance.0)));
        if let Ok(typed) = &typed {
            let calls = typed.res.values().filter_map(|res| match res {
                Res::Fn(item, substs) => Some((*item, substs.clone())),
                Res::Local(_) => None,
            });
            queue.extend(calls);
        }
        reached.index.insert(instance.clone(), reached.list.len());
        reached.list.push((instance, typed));
    }
    reached
}
