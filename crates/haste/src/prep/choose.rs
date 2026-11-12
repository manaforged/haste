use std::collections::BTreeSet;

use anyhow::{Error, Result};
use rustc_hash::FxHashMap as HashMap;

use crate::sema::Context;
use crate::sema::reach::Reached;

use super::name::{Paths, Printer};
use super::select::{self, Crates, Node};
use super::shim::Wrapper;

/// The most functions that a call tree Haste keeps may hold, counting each call: past it, Haste's partial inlining costs more than one call into rustc's code.
const KEPT_FUNCTIONS: usize = 16;

/// Where deciding whether Haste keeps a function stands: not started, under way, or decided, with the size of the tree it keeps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark {
    New,
    Open,
    Kept(Option<usize>),
}

/// What choosing the prepped functions of a build knows so far: each function's node and whether its whole call tree is preppable, which functions Haste keeps, and the wrappers made or the errors met.
struct Chooser<'c, 'p> {
    crates: &'c Crates,
    reached: &'c Reached,
    printer: Printer<'p>,
    nodes: Vec<Node>,
    whole: Vec<bool>,
    marks: Vec<Mark>,
    made: HashMap<usize, Option<Wrapper>>,
    errors: HashMap<usize, Error>,
}

/// The outermost wrappable functions on the call paths from `entries` whose whole call tree rustc may compile and that Haste does not keep, and the crates their wrappers name.
pub(super) fn choose(cx: &mut Context<'_>, (crates, paths): (&Crates, &Paths), reached: &Reached, entries: &[usize]) -> Result<(Vec<Wrapper>, BTreeSet<u32>)> {
    let nodes = select::nodes(cx, crates, reached);
    let whole = select::whole(reached, &nodes);
    let marks = vec![Mark::New; reached.list.len()];
    let mut chooser = Chooser { crates, reached, printer: Printer::new(paths), nodes, whole, marks, made: HashMap::default(), errors: HashMap::default() };
    let (mut picked, mut failure) = (Vec::new(), None);
    select::visit(reached, entries, |at| {
        if !chooser.whole[at] || chooser.kept(cx, at) {
            return false;
        }
        let made = chooser.wrapper(cx, at);
        if let Some(error) = chooser.errors.remove(&at) {
            failure.get_or_insert(error);
        }
        made.then(|| picked.push(at)).is_some()
    });
    if let Some(error) = failure {
        return Err(error);
    }
    let wrappers = picked.into_iter().filter_map(|at| chooser.made.remove(&at).flatten()).collect();
    Ok((wrappers, chooser.printer.krates))
}

impl Chooser<'_, '_> {
    /// Whether Haste keeps function `at`: it is small, and so is its call tree, up to boundaries and calls aside from the work.
    fn kept(&mut self, cx: &mut Context<'_>, at: usize) -> bool {
        self.tree(cx, at).is_some()
    }

    /// The count of functions in the call tree Haste keeps from `at`, or none when Haste does not keep it.
    fn tree(&mut self, cx: &mut Context<'_>, at: usize) -> Option<usize> {
        match self.marks[at] {
            Mark::Kept(size) => return size,
            Mark::Open => return None,
            Mark::New => {}
        }
        self.marks[at] = Mark::Open;
        let reached = self.reached;
        let mut size = select::small(cx, reached, at).then_some(1);
        for &callee in reached.direct_calls(at).unwrap_or_default() {
            if size.is_none() || self.nodes[callee] == Node::Boundary || select::aside(cx, reached, callee) {
                continue;
            }
            size = size.zip(self.tree(cx, callee)).map(|(size, more)| size + more).filter(|&size| size <= KEPT_FUNCTIONS);
        }
        self.marks[at] = Mark::Kept(size);
        size
    }

    /// Whether function `at` has a wrapper, making it the first time it is asked.
    fn wrapper(&mut self, cx: &mut Context<'_>, at: usize) -> bool {
        if !self.made.contains_key(&at) && !self.errors.contains_key(&at) {
            let made = if select::wrappable(cx, self.reached, at) { super::candidate(cx, (&mut self.printer, self.crates), self.reached, at) } else { Ok(None) };
            match made {
                Ok(wrapper) => _ = self.made.insert(at, wrapper),
                Err(error) => _ = self.errors.insert(at, error),
            }
        }
        self.made.get(&at).is_some_and(Option::is_some)
    }
}
