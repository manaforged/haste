//! A numbering of reached functions that does not depend on the order parallel reach found them in.

use std::collections::VecDeque;

use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{CallSites, recount_call_sites};
use crate::sema::summary::codec::type_id_key;

use super::{Instance, Reached, ReachedEntry};

impl Reached {
    /// Renumbers every function breadth-first from `roots` along the call graph, then the rest by item and argument types, so indices are the same on every run.
    pub fn canonicalize(&mut self, cx: &Context<'_>, roots: &[Instance]) {
        let len = self.list.len();
        self.calls.resize(len, Vec::new());
        self.indirect.resize(len, Vec::new());
        self.pinned.resize(len, false);
        let mut seen = vec![false; len];
        let mut queue = VecDeque::new();
        for root in roots {
            if let Some(&at) = self.index.get(root)
                && !std::mem::replace(&mut seen[at], true)
            {
                queue.push_back(at);
            }
        }
        let mut order = Vec::with_capacity(len);
        while let Some(at) = queue.pop_front() {
            order.push(at);
            for &callee in &self.calls[at] {
                if !std::mem::replace(&mut seen[callee], true) {
                    queue.push_back(callee);
                }
            }
        }
        if order.len() < len {
            let mut rest: Vec<usize> = (0..len).filter(|&at| !seen[at]).collect();
            rest.sort_by_cached_key(|&at| instance_key(cx, self.list[at].0));
            order.extend(rest);
        }
        let mut new_of = vec![0; len];
        for (new, &old) in order.iter().enumerate() {
            new_of[old] = new;
        }
        permute_by_order(&mut self.list, &order);
        permute_copy(&mut self.states, &order);
        permute_copy(&mut self.pinned, &order);
        permute_vecs(&mut self.calls, &order);
        permute_vecs(&mut self.indirect, &order);
        for edges in self.calls.iter_mut().chain(self.indirect.iter_mut()) {
            edges.iter_mut().for_each(|at| *at = new_of[*at]);
        }
        self.index.values_mut().for_each(|at| *at = new_of[*at]);
        self.crate_errors
            .iter_mut()
            .for_each(|at| *at = new_of[*at]);
        self.call_sites = CallSites::default();
        let states = &self.states;
        recount_call_sites(
            len,
            |at| states[at] == ReachedEntry::Settled,
            &self.calls,
            &self.indirect,
            &self.pinned,
            &mut self.call_sites,
        );
    }
}

fn instance_key(cx: &Context<'_>, (item, substs): Instance) -> (ItemRef, Vec<u64>) {
    (
        item,
        cx.types
            .substs(substs)
            .iter()
            .map(|&ty| type_id_key(cx.outline, &cx.types, ty))
            .collect(),
    )
}

fn permute_by_order<T>(items: &mut Vec<T>, order: &[usize]) {
    let mut slots: Vec<Option<T>> = items.drain(..).map(Some).collect();
    for &old in order {
        if let Some(item) = slots[old].take() {
            items.push(item);
        }
    }
}

fn permute_copy<T: Copy>(items: &mut Vec<T>, order: &[usize]) {
    let reordered: Vec<T> = order.iter().map(|&old| items[old]).collect();
    *items = reordered;
}

fn permute_vecs<T>(items: &mut Vec<Vec<T>>, order: &[usize]) {
    let mut old_items = std::mem::take(items);
    items.reserve(order.len());
    for &old in order {
        items.push(std::mem::take(&mut old_items[old]));
    }
}
