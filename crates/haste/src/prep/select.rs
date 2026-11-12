use std::hash::BuildHasher as _;

use anyhow::Result;
use rustc_abi::{TagEncoding, Variants};
use rustc_hash::FxHashSet as HashSet;

use crate::graph::cfg::Cfg;
use crate::graph::{Side, Workspace};
use crate::names::ItemRef;
use crate::outline::attr::split_commas;
use crate::outline::item::{Attr, ItemKind};
use crate::outline::tree::{PROFILE, Truth};
use crate::outline::{Outline, Role};
use crate::sema::Context;
use crate::sema::infer::{Shape, Typed, field_types};
use crate::sema::reach::Reached;
use crate::sema::traits::PANIC_HANDLER;
use crate::sema::ty::{Kind, Ty};

const PANIC_LANG: &str = "panic";
const BEGIN_PANIC: &str = "begin_panic";
const UNSAFE_CELL: &str = "unsafe_cell";
const MUTABLE: &str = "mut";
const REPR: &str = "repr";
const SIMD: &str = "simd";

/// The most source expressions a body may have for selection to count it small, fixed apart from Haste's inline budgets.
const SMALL_EXPRS: usize = 20;

/// What one reached function is to prep: code rustc may compile in its place, code whose callees rustc handles itself, or code Haste must compile.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Node {
    No,
    Yes,
    Boundary,
}

/// Why a type walk looks at a type: it crosses a wrapper, or it is the type of a static that prepped code reads.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Purpose {
    Crossing,
    Frozen,
}

/// The facts selection reads about the crates of a build: whether rustc may compile a crate's code, whether the crate is local, and whether it is a standard crate.
pub(super) struct Crates {
    preppable: Vec<bool>,
    local: Vec<bool>,
    sysroot: Vec<bool>,
}

impl Crates {
    /// The facts of every crate in `outline`.
    pub(super) fn new(outline: &Outline, workspace: &Workspace) -> Crates {
        let local: Vec<bool> = (0..outline.roots.len()).map(|krate| outline.package(workspace, krate).local()).collect();
        let preppable = (0..outline.roots.len())
            .map(|krate| {
                let root = &outline.roots[krate];
                let target = workspace.graph.units[root.unit].side == Side::Target && root.role == Role::Lib;
                target && !local[krate] && release(outline, krate)
            })
            .collect();
        let sysroot = (0..outline.roots.len()).map(|krate| outline.package(workspace, krate).sysroot).collect();
        Crates { preppable, local, sysroot }
    }
}

/// Whether crate `krate` compiles under release profile cfgs: no debug assertions, overflow checks or UB checks.
fn release(outline: &Outline, krate: usize) -> bool {
    let env = &outline.crates[krate].env;
    PROFILE.iter().all(|name| env.eval(&Cfg::Name((*name).to_owned())) == Truth::No)
}

/// What each reached function is to prep.
pub(super) fn nodes(cx: &mut Context<'_>, crates: &Crates, reached: &Reached) -> Vec<Node> {
    let handler = cx.lang_item(PANIC_HANDLER);
    (0..reached.list.len()).map(|at| node(cx, crates, (reached, at), handler)).collect()
}

/// The reached functions whose whole call tree rustc may compile: each is preppable and so is every callee, up to the callees that end in a panic or a foreign function.
pub(super) fn whole(reached: &Reached, nodes: &[Node]) -> Vec<bool> {
    let mut whole: Vec<bool> = nodes.iter().map(|&node| node != Node::No).collect();
    let callers = callers(reached);
    let mut stack: Vec<usize> = (0..whole.len()).filter(|&at| !whole[at]).collect();
    while let Some(at) = stack.pop() {
        for &caller in &callers[at] {
            if whole[caller] && nodes[caller] == Node::Yes {
                whole[caller] = false;
                stack.push(caller);
            }
        }
    }
    whole
}

fn node(cx: &mut Context<'_>, crates: &Crates, (reached, at): (&Reached, usize), handler: Option<ItemRef>) -> Node {
    let ((item, _), typed) = &reached.list[at];
    let Ok(typed) = typed else { return Node::No };
    if !crates.preppable[item.krate as usize] {
        return Node::No;
    }
    let lang = cx.lang(*item);
    if cx.foreign(*item) || Some(*item) == handler || lang.is_some_and(|name| name.starts_with(PANIC_LANG) || name == BEGIN_PANIC) {
        return Node::Boundary;
    }
    if crate::lower::is_static(cx, *item) && !frozen_static(cx, crates, *item, typed) {
        return Node::No;
    }
    Node::Yes
}

fn frozen_static(cx: &mut Context<'_>, crates: &Crates, item: ItemRef, typed: &Typed) -> bool {
    let (source, entry) = cx.item(item);
    let mutable = entry.name.is_some_and(|name| name > 0 && source.text(name - 1) == MUTABLE);
    let Some(ty) = typed.ret else { return false };
    !mutable && !cx.thread_local(item) && walk(cx, crates, &[ty], Purpose::Frozen).is_some()
}

/// For each reached function, the reached functions that call it.
pub(super) fn callers(reached: &Reached) -> Vec<Vec<usize>> {
    let mut callers = vec![Vec::new(); reached.list.len()];
    for at in 0..reached.list.len() {
        for &callee in reached.direct_calls(at).unwrap_or_default() {
            callers[callee].push(at);
        }
    }
    callers
}

/// Visits the reached functions from `entries` along calls, stopping at each function `stop` takes; returns those visited and not stopped, then those stopped.
pub(super) fn visit(reached: &Reached, entries: &[usize], mut stop: impl FnMut(usize) -> bool) -> (Vec<usize>, Vec<usize>) {
    let mut seen = vec![false; reached.list.len()];
    let (mut compiled, mut stopped, mut stack) = (Vec::new(), Vec::new(), entries.to_vec());
    while let Some(at) = stack.pop() {
        if std::mem::replace(&mut seen[at], true) {
            continue;
        }
        if stop(at) {
            stopped.push(at);
            continue;
        }
        compiled.push(at);
        stack.extend(reached.direct_calls(at).unwrap_or_default().iter().rev());
    }
    (compiled, stopped)
}

/// Whether reached function `at` is aside from a caller's work: drop glue, a constructor or a clone shim, or a function that never returns.
pub(super) fn aside(cx: &Context<'_>, reached: &Reached, at: usize) -> bool {
    let Ok(typed) = &reached.list[at].1 else { return false };
    matches!(typed.shape, Shape::Glue(_) | Shape::Ctor(_) | Shape::Clone(_)) || typed.ret.is_some_and(|ret| matches!(cx.types.kind(ret), Kind::Never))
}

/// Whether reached function `at` is small: a shim, or a body or closure of at most `SMALL_EXPRS` source expressions.
pub(super) fn small(cx: &mut Context<'_>, reached: &Reached, at: usize) -> bool {
    let ((item, _), Ok(typed)) = &reached.list[at] else { return false };
    let source = match &typed.shape {
        Shape::Body => *item,
        Shape::Closure(closure) => closure.parent,
        _ => return true,
    };
    cx.body(source).is_ok_and(|body| body.exprs.len() <= SMALL_EXPRS)
}

/// Whether reached function `at` is a plain body that the shim can wrap.
pub(super) fn wrappable(cx: &Context<'_>, reached: &Reached, at: usize) -> bool {
    let (item, _) = reached.list[at].0;
    matches!(&reached.list[at].1, Ok(typed) if matches!(typed.shape, Shape::Body)) && cx.item(item).1.kind == ItemKind::Fn
}

/// The hash of the layouts of `types` and of every type their values hold or point to, or none when one of them cannot cross between rustc's code and Haste's, or when they mix a niche-encoded enum with a type from outside the standard crates, whose niche Haste may place apart from rustc.
pub(super) fn walk(cx: &mut Context<'_>, crates: &Crates, types: &[Ty], purpose: Purpose) -> Option<u64> {
    let mut seen = HashSet::default();
    let (mut stack, mut hashes, mut niche, mut outer) = (types.to_vec(), Vec::new(), false, false);
    while let Some(ty) = stack.pop() {
        if !seen.insert(ty) {
            continue;
        }
        match inner(cx, crates, ty, purpose) {
            Ok(Some(inner)) => stack.extend(inner),
            _ => return None,
        }
        let Ok(layout) = cx.layout(ty) else { return None };
        niche |= matches!(layout.variants, Variants::Multiple { tag_encoding: TagEncoding::Niche { .. }, .. });
        outer |= matches!(*cx.types.kind(ty), Kind::Adt(item, _) if !crates.sysroot[item.krate as usize]);
        hashes.push(rustc_hash::FxBuildHasher.hash_one(&*layout));
    }
    (!(niche && outer)).then(|| rustc_hash::FxBuildHasher.hash_one(&hashes))
}

fn inner(cx: &mut Context<'_>, crates: &Crates, ty: Ty, purpose: Purpose) -> Result<Option<Vec<Ty>>> {
    Ok(Some(match cx.types.kind(ty).clone() {
        Kind::Prim(_) | Kind::Never => Vec::new(),
        Kind::Ref(_, inner) | Kind::Ptr(_, inner) | Kind::Slice(inner) | Kind::Pat(inner, ..) | Kind::Array(inner, _) => vec![inner],
        Kind::Tuple(items) => cx.types.substs(items).to_vec(),
        Kind::Adt(item, args) => {
            let args = cx.types.substs(args).to_vec();
            return adt(cx, crates, (item, args), purpose);
        }
        _ if purpose == Purpose::Frozen => Vec::new(),
        _ => return Ok(None),
    }))
}

fn adt(cx: &mut Context<'_>, crates: &Crates, (item, args): (ItemRef, Vec<Ty>), purpose: Purpose) -> Result<Option<Vec<Ty>>> {
    if crates.local[item.krate as usize] || simd(cx, item) || (purpose == Purpose::Frozen && cx.lang(item) == Some(UNSAFE_CELL)) {
        return Ok(None);
    }
    if cx.item(item).1.kind == ItemKind::Enum && !contiguous(&cx.discriminants(item)?) {
        return Ok(None);
    }
    Ok(Some(field_types(cx, item, &args)?.iter().flatten().copied().collect()))
}

/// Whether `item` is `#[repr(simd)]`, a vector whose layout and passing Haste does not compute as rustc does.
fn simd(cx: &Context<'_>, item: ItemRef) -> bool {
    let (src, entry) = cx.item(item);
    let words = |at: u32| split_commas(src, at + 2, src.pair(at + 1)).into_iter().any(|(from, _)| src.text(from) == SIMD);
    entry.attrs.iter().any(|attr| matches!(*attr, Attr::Other(at) if src.text(at) == REPR && src.kind(at + 1) == rustc_lexer::TokenKind::OpenParen && words(at)))
}

/// Whether `values` hold every integer between their least and greatest, the discriminants whose tag range every layout version computes alike.
fn contiguous(values: &[i128]) -> bool {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).all(|pair| pair[1].checked_sub(pair[0]) == Some(1))
}
