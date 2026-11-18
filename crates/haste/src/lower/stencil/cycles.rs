use std::sync::RwLock;

use rustc_hash::FxHashMap;

use crate::sema::Context;
use crate::sema::infer::Shape;
use crate::sema::reach::Reached;
use crate::sync::{read_lock, write_lock};

use super::Bodies;
use super::super::plan::closure_calls_uncached;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CycleState {
    Acyclic,
    Cyclic { complete: bool },
}

impl CycleState {
    fn is_complete(self) -> bool {
        matches!(self, Self::Acyclic | Self::Cyclic { complete: true })
    }
}

#[derive(Clone, Copy)]
struct NodeInfo {
    index: u32,
    low: u32,
    on_stack: bool,
    self_loop: bool,
    complete: bool,
}

struct TarjanRun {
    nodes: FxHashMap<usize, NodeInfo>,
    stack: Vec<usize>,
    clock: u32,
}

impl TarjanRun {
    fn enter(&mut self, u: usize, (edges, self_loop, complete): (Vec<usize>, bool, bool)) -> (usize, usize, Vec<usize>) {
        self.clock += 1;
        let info = NodeInfo { index: self.clock, low: self.clock, on_stack: true, self_loop, complete };
        self.nodes.insert(u, info);
        self.stack.push(u);
        (u, 0, edges)
    }

    fn finish(&mut self, u: usize, parent: Option<usize>, cyclic: &mut FxHashMap<usize, CycleState>) {
        let u_info = self.nodes[&u];
        if u_info.low == u_info.index {
            self.pop_scc(u, cyclic);
        }
        if let Some(p) = parent {
            let u_now = self.nodes[&u];
            if let Some(p_info) = self.nodes.get_mut(&p) {
                p_info.low = p_info.low.min(u_now.low);
                p_info.complete &= u_now.complete;
            }
        }
    }

    fn pop_scc(&mut self, u: usize, cyclic: &mut FxHashMap<usize, CycleState>) {
        let mut scc = Vec::new();
        let mut self_loop = false;
        let mut complete = true;
        while let Some(w) = self.stack.pop() {
            if let Some(info) = self.nodes.get_mut(&w) {
                info.on_stack = false;
                self_loop |= info.self_loop;
                complete &= info.complete;
            }
            scc.push(w);
            if w == u {
                break;
            }
        }
        let is_cyc = scc.len() > 1 || self_loop;
        let state = match (is_cyc, complete) {
            (true, complete) => Some(CycleState::Cyclic { complete }),
            (false, true) => Some(CycleState::Acyclic),
            (false, false) => None,
        };
        for &w in &scc {
            if let Some(st) = state {
                cyclic.insert(w, st);
            }
            if !complete && let Some(info) = self.nodes.get_mut(&w) {
                info.complete = false;
            }
        }
    }
}

impl Bodies {
    pub(super) fn is_cyclic(&self, cx: &mut Context<'_>, reached: &Reached, root: usize) -> bool {
        if let Some(&st) = read_lock(&self.cyclic).get(&root) {
            return st != CycleState::Acyclic;
        }
        if reached.is_untyped(root) {
            return false;
        }
        let mut view = CycleView { shared: &self.cyclic, found: FxHashMap::default() };
        if super::super::may_inline(cx, reached, root) {
            tarjan_from(cx, reached, &mut view, root);
        } else {
            view.found.insert(root, CycleState::Acyclic);
        }
        let mut shared = write_lock(&self.cyclic);
        for (node, state) in view.found {
            match shared.get(&node).copied() {
                Some(CycleState::Acyclic) | Some(CycleState::Cyclic { complete: true }) => {}
                _ => {
                    shared.insert(node, state);
                }
            }
        }
        shared.get(&root).is_some_and(|&st| st != CycleState::Acyclic)
    }
}

struct CycleView<'b> {
    shared: &'b RwLock<FxHashMap<usize, CycleState>>,
    found: FxHashMap<usize, CycleState>,
}

impl CycleView<'_> {
    fn is_done(&self, v: usize) -> bool {
        let done = |st: &CycleState| st.is_complete();
        self.found.get(&v).is_some_and(done) || read_lock(self.shared).get(&v).is_some_and(done)
    }
}

fn tarjan_from(cx: &mut Context<'_>, reached: &Reached, cyclic: &mut CycleView<'_>, root: usize) {
    let mut run = TarjanRun { nodes: FxHashMap::default(), stack: Vec::new(), clock: 0 };
    let mut frames = vec![run.enter(root, node_edges(cx, reached, cyclic, root))];
    while let Some(top) = frames.last_mut() {
        let u = top.0;
        if top.1 < top.2.len() {
            let v = top.2[top.1];
            top.1 += 1;
            step_edge(cx, reached, cyclic, (&mut run, &mut frames), (u, v));
        } else if let Some((done, _, _)) = frames.pop() {
            run.finish(done, frames.last().map(|f| f.0), &mut cyclic.found);
        }
    }
}

type Frames = Vec<(usize, usize, Vec<usize>)>;

fn step_edge(cx: &mut Context<'_>, reached: &Reached, cyclic: &mut CycleView<'_>, (run, frames): (&mut TarjanRun, &mut Frames), (u, v): (usize, usize)) {
    if cyclic.is_done(v) {
        return;
    }
    if let Some(&v_info) = run.nodes.get(&v) {
        if let Some(u_info) = run.nodes.get_mut(&u) {
            if v_info.on_stack {
                u_info.low = u_info.low.min(v_info.index);
            } else {
                u_info.complete &= v_info.complete;
            }
        }
        return;
    }
    frames.push(run.enter(v, node_edges(cx, reached, cyclic, v)));
}

fn node_edges(cx: &mut Context<'_>, reached: &Reached, cyclic: &mut CycleView<'_>, u: usize) -> (Vec<usize>, bool, bool) {
    let mut info = NodeInfo { index: 0, low: 0, on_stack: false, self_loop: false, complete: true };
    let mut edges = Vec::new();
    let closure = matches!(&reached.list[u].1, Ok(typed) if matches!(typed.shape, Shape::Closure(_)));
    if let Some(slice) = reached.direct_calls(u).filter(|_| !closure) {
        for &v in slice {
            scan_callee(cx, reached, cyclic, (u, v), (&mut info, &mut edges));
        }
        return (edges, info.self_loop, info.complete);
    }
    let Ok(typed) = &reached.list[u].1 else { return (edges, false, true) };
    let calls = closure.then(|| closure_calls_uncached(cx, typed)).flatten().unwrap_or_else(|| typed.calls().collect());
    for (item, substs) in calls {
        if cx.intrinsic(item).is_some() {
            continue;
        }
        match reached.index.get(&(item, substs)) {
            Some(&v) => scan_callee(cx, reached, cyclic, (u, v), (&mut info, &mut edges)),
            None => info.complete = false,
        }
    }
    (edges, info.self_loop, info.complete)
}

fn scan_callee(cx: &mut Context<'_>, reached: &Reached, cyclic: &mut CycleView<'_>, (u, v): (usize, usize), (info, edges): (&mut NodeInfo, &mut Vec<usize>)) {
    if v == u {
        info.self_loop |= !matches!(&reached.list[u].1, Ok(typed) if matches!(typed.shape, Shape::Closure(_)));
        return;
    }
    if own_glue(cx, reached, (u, v)) {
        return;
    }
    if cyclic.is_done(v) {
        return;
    }
    if reached.is_untyped(v) {
        info.complete = false;
        return;
    }
    if !super::super::may_inline(cx, reached, v) {
        cyclic.found.insert(v, CycleState::Acyclic);
        return;
    }
    edges.push(v);
}

/// Whether `v` is glue of closure `u`'s own type, which `u` can name but never reach while it runs.
fn own_glue(cx: &Context<'_>, reached: &Reached, (u, v): (usize, usize)) -> bool {
    let closure = matches!(&reached.list[u].1, Ok(typed) if matches!(typed.shape, Shape::Closure(_)));
    let glue = matches!(&reached.list[v].1, Ok(typed) if matches!(typed.shape, Shape::Glue(_)));
    let own = |index: usize| cx.types.substs(reached.list[index].0.1).first().copied();
    closure && glue && own(u).is_some() && own(u) == own(v)
}
