use rustc_hash::FxHashMap;

use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, Stmt, UnOp};
use crate::host::HostSymbol;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{LazyTables, Res, Shape, SiteKey, Typed};
use crate::sema::intrinsic::Intrinsic;
use crate::sema::ty::{Kind, Ty};

use super::Reached;

/// Whether calling host symbol `host` may initiate or propagate a panic unwind.
pub(crate) fn host_symbol_may_unwind(host: &HostSymbol) -> bool {
    matches!(
        host,
        HostSymbol::StartPanic
            | HostSymbol::CatchUnwind
            | HostSymbol::FillOnce
            | HostSymbol::ThreadCell
            | HostSymbol::Dynamic { .. }
    )
}

fn foreign_host_symbol(cx: &Context<'_>, item: ItemRef) -> Option<HostSymbol> {
    let lib = cx.link_lib(item);
    cx.symbol_name(item)
        .as_deref()
        .and_then(|name| HostSymbol::resolve_for_lib(name, lib.as_deref(), true))
}

fn foreign_may_unwind(cx: &Context<'_>, item: ItemRef) -> bool {
    let Some(host) = foreign_host_symbol(cx, item) else {
        return true;
    };
    match host {
        HostSymbol::Dynamic { .. } => cx.extern_block_may_unwind(item),
        _ => host_symbol_may_unwind(&host),
    }
}

/// Whether foreign item `item` calls into dynamic native code whose ABI permits unwinding.
pub(crate) fn foreign_dynamic_may_unwind(cx: &Context<'_>, item: ItemRef) -> bool {
    cx.extern_block_may_unwind(item)
        && matches!(
            foreign_host_symbol(cx, item),
            None | Some(HostSymbol::Dynamic { .. })
        )
}

fn intrinsic_may_unwind(intr: Intrinsic) -> bool {
    matches!(intr, Intrinsic::ConstEvalSelect | Intrinsic::CatchUnwind)
}

enum BodyScan {
    Unwinds,
    IndirectCalls {
        calls: Vec<ExprId>,
        no_normal_drops: bool,
    },
}

fn expr_avoids_normal_drop(body: &Body, expr: &Expr) -> bool {
    match expr {
        Expr::Lit(..)
        | Expr::Path(_)
        | Expr::Call(..)
        | Expr::Tuple(_)
        | Expr::Array(_)
        | Expr::Paren(_)
        | Expr::Cast(..) => true,
        Expr::Unary(op, _) => !matches!(op, UnOp::Deref),
        Expr::Binary(op, ..) => !matches!(
            op,
            BinOp::And
                | BinOp::Or
                | BinOp::Eq
                | BinOp::Ne
                | BinOp::Lt
                | BinOp::Le
                | BinOp::Gt
                | BinOp::Ge
        ),
        Expr::Struct { rest: false, .. } => true,
        Expr::Block {
            stmts,
            tail: Some(_),
            kind: BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe | BlockKind::Const,
            ..
        } => body
            .list(*stmts)
            .iter()
            .all(|&s| matches!(body.stmts[s as usize], Stmt::Empty | Stmt::Item(..))),
        _ => false,
    }
}

fn all_params_moved(
    cx: &Context<'_>,
    body: &Body,
    typed: &Typed,
    call_callees: &mut Vec<bool>,
) -> bool {
    if typed.params.is_empty() {
        return true;
    }
    if !body.fn_params.is_empty() || typed.params.len() > 8 {
        return false;
    }
    call_callees.clear();
    call_callees.resize(body.exprs.len(), false);
    for expr in &body.exprs {
        if let Expr::Call(callee, _) = *expr {
            call_callees[callee.0 as usize] = true;
        }
    }
    let mut moved = 0u16;
    for (idx, expr) in body.exprs.iter().enumerate() {
        if !matches!(expr, Expr::Path(_)) || call_callees[idx] {
            continue;
        }
        if let Some(Res::Local(local)) =
            typed.site_res(&cx.types, SiteKey::ExprRes(ExprId(idx as u32)))
            && let Some(pos) = typed.params.iter().position(|&p| p == local)
        {
            moved |= 1u16 << pos;
        }
    }
    moved == (1u16 << typed.params.len()) - 1
}

fn scan_body_ops(
    cx: &mut Context<'_>,
    source: ItemRef,
    overflow_checks: bool,
    cache: &mut FxHashMap<(ItemRef, bool), BodyScan>,
) -> bool {
    let entry = cache.entry((source, overflow_checks)).or_insert_with(|| {
        let Ok(body) = cx.body(source) else {
            return BodyScan::Unwinds;
        };
        let mut calls = Vec::new();
        let mut no_normal_drops = body.root.is_some();
        for (idx, expr) in body.exprs.iter().enumerate() {
            match expr {
                Expr::FormatArgs(_) | Expr::Index(..) | Expr::Macro { .. } => {
                    return BodyScan::Unwinds;
                }
                Expr::Binary(op, ..) | Expr::AssignOp(op, ..)
                    if matches!(op, BinOp::Div | BinOp::Rem)
                        || (overflow_checks
                            && matches!(
                                op,
                                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl | BinOp::Shr
                            )) =>
                {
                    return BodyScan::Unwinds;
                }
                Expr::Unary(UnOp::Neg, _) if overflow_checks => return BodyScan::Unwinds,
                Expr::Call(..) => calls.push(ExprId(idx as u32)),
                _ => {}
            }
            no_normal_drops &= expr_avoids_normal_drop(&body, expr);
        }
        BodyScan::IndirectCalls {
            calls,
            no_normal_drops,
        }
    });
    matches!(entry, BodyScan::Unwinds)
}

pub(super) struct UnwindScratch {
    body_cache: FxHashMap<(ItemRef, bool), BodyScan>,
    item_cache: FxHashMap<ItemRef, ItemUnwind>,
    call_callees: Vec<bool>,
}

impl UnwindScratch {
    pub(super) fn new() -> Self {
        Self {
            body_cache: FxHashMap::default(),
            item_cache: FxHashMap::default(),
            call_callees: Vec::new(),
        }
    }
}

fn has_indirect_call(
    cx: &mut Context<'_>,
    source: ItemRef,
    overflow_checks: bool,
    typed: &Typed,
    scratch: &mut UnwindScratch,
) -> (bool, bool) {
    if scan_body_ops(cx, source, overflow_checks, &mut scratch.body_cache) {
        return (true, false);
    }
    let Some(BodyScan::IndirectCalls {
        calls,
        no_normal_drops,
    }) = scratch.body_cache.get(&(source, overflow_checks))
    else {
        return (true, false);
    };
    let no_normal_drops = *no_normal_drops;
    if calls.is_empty() && (!no_normal_drops || typed.params.is_empty()) {
        return (false, no_normal_drops);
    }
    let Ok(body) = cx.body(source) else {
        return (true, false);
    };
    for &expr in calls {
        let Expr::Call(callee, _) = body.exprs[expr.0 as usize] else {
            continue;
        };
        if typed.expr_ty(&cx.types, expr) == Ty::NONE
            && typed.expr_ty(&cx.types, callee) == Ty::NONE
        {
            continue;
        }
        let direct = matches!(
            typed.site_res(&cx.types, SiteKey::ExprRes(expr)),
            Some(Res::Fn(..) | Res::Ctor(..))
        ) || matches!(
            typed.site_res(&cx.types, SiteKey::ExprRes(callee)),
            Some(Res::Fn(..) | Res::Ctor(..))
        );
        if !direct {
            return (true, false);
        }
    }
    let skip_drops = no_normal_drops
        && matches!(typed.shape, Shape::Body)
        && all_params_moved(cx, &body, typed, &mut scratch.call_callees);
    (false, skip_drops)
}

#[derive(Clone, Copy)]
enum ItemUnwind {
    Normal,
    Intrinsic(bool),
    Foreign(bool),
}

fn item_unwind(
    cx: &Context<'_>,
    item: ItemRef,
    cache: &mut FxHashMap<ItemRef, ItemUnwind>,
) -> ItemUnwind {
    *cache.entry(item).or_insert_with(|| {
        if let Some(intr) = cx.intrinsic(item) {
            ItemUnwind::Intrinsic(intrinsic_may_unwind(intr))
        } else if cx.foreign(item) {
            ItemUnwind::Foreign(foreign_may_unwind(cx, item))
        } else {
            ItemUnwind::Normal
        }
    })
}

pub(super) fn direct_may_unwind(
    cx: &mut Context<'_>,
    reached: &Reached,
    index: usize,
    scratch: &mut UnwindScratch,
) -> (bool, bool) {
    let Some(((item, substs), res)) = reached.list.get(index) else {
        return (true, false);
    };
    let item = *item;
    let Ok(typed) = res else {
        return match item_unwind(cx, item, &mut scratch.item_cache) {
            ItemUnwind::Intrinsic(u) | ItemUnwind::Foreign(u) => (u, false),
            ItemUnwind::Normal => (true, false),
        };
    };
    if let LazyTables::ReachMemo(memo) = &typed.tables {
        return (memo.direct_unwinds, memo.skip_drops);
    }
    match item_unwind(cx, item, &mut scratch.item_cache) {
        ItemUnwind::Intrinsic(u) | ItemUnwind::Foreign(u) => return (u, false),
        ItemUnwind::Normal => {}
    }
    let mut skip_drops = false;
    match &typed.shape {
        Shape::Virtual(_) | Shape::Asm(_) | Shape::Anon(_) => return (true, false),
        Shape::Untuple(Res::Fn(..)) | Shape::Clone(_) | Shape::Ctor(_) | Shape::GlobalAlloc(_) => {}
        Shape::Untuple(_) => return (true, false),
        Shape::Glue(_) => {
            if let Some(&ty) = cx.types.substs(*substs).first()
                && matches!(cx.types.kind(ty), Kind::Dyn(..))
            {
                return (true, false);
            }
        }
        Shape::Body | Shape::Closure(_) => {
            if matches!(typed.tables, LazyTables::CheckOnly(_)) {
                return (true, false);
            }
            let source = match &typed.shape {
                Shape::Closure(cb) => cb.parent,
                _ => item,
            };
            let overflow_checks = cx.overflow_checks(source.krate);
            let (indirect, no_normal_drops) =
                has_indirect_call(cx, source, overflow_checks, typed, scratch);
            if indirect {
                return (true, false);
            }
            skip_drops = matches!(typed.shape, Shape::Body) && no_normal_drops;
        }
    }
    let callees = if skip_drops {
        typed.calls_without_drops()
    } else {
        typed.calls()
    };
    for (callee, csubsts) in callees {
        match item_unwind(cx, callee, &mut scratch.item_cache) {
            ItemUnwind::Intrinsic(true) | ItemUnwind::Foreign(true) => return (true, false),
            ItemUnwind::Intrinsic(false) | ItemUnwind::Foreign(false) => continue,
            ItemUnwind::Normal => {}
        }
        if !reached.index.contains_key(&(callee, csubsts)) {
            return (true, false);
        }
    }
    (false, skip_drops)
}

fn scan_unwind_slice(
    cx: &mut Context<'_>,
    reached: &Reached,
    range: std::ops::Range<usize>,
    n: usize,
) -> (Vec<u32>, Vec<(u32, u32)>) {
    let mut scratch = UnwindScratch::new();
    let mut direct_unwinds = Vec::new();
    let mut edges = Vec::new();
    for u in range {
        let (direct, skip_drops) = direct_may_unwind(cx, reached, u, &mut scratch);
        if direct {
            direct_unwinds.push(u as u32);
            continue;
        }
        if skip_drops {
            if let Some((_, Ok(typed))) = reached.list.get(u) {
                for (callee, csubsts) in typed.calls_without_drops() {
                    if let Some(&v) = reached.index.get(&(callee, csubsts))
                        && v < n
                    {
                        edges.push((v as u32, u as u32));
                    }
                }
            }
        } else if let Some(callees) = reached.calls.get(u).filter(|c| !c.is_empty()) {
            for &v in callees {
                if v < n {
                    edges.push((v as u32, u as u32));
                }
            }
        } else if let Some((_, Ok(typed))) = reached.list.get(u) {
            for v in typed.collect_link_edges(&reached.index).0 {
                if v < n {
                    edges.push((v as u32, u as u32));
                }
            }
        }
    }
    (direct_unwinds, edges)
}

pub(super) fn compute_may_unwind(cx: &mut Context<'_>, reached: &Reached) -> Vec<bool> {
    let n = reached.list.len();
    let workers = crate::jobs().min(n / 128);
    let parts: Vec<(Vec<u32>, Vec<(u32, u32)>)> = if workers <= 1 {
        vec![scan_unwind_slice(cx, reached, 0..n, n)]
    } else {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let worker_scan = |worker: &mut Context<'_>| {
            let mut direct_all = Vec::new();
            let mut edges_all = Vec::new();
            loop {
                let start = next.fetch_add(128, std::sync::atomic::Ordering::Relaxed);
                if start >= n {
                    return (direct_all, edges_all);
                }
                let end = (start + 128).min(n);
                let (d, e) = scan_unwind_slice(worker, reached, start..end, n);
                direct_all.extend(d);
                edges_all.extend(e);
            }
        };
        let forks: Vec<Context<'_>> = (1..workers).map(|_| cx.fork()).collect();
        crate::sync::scope(|scope| {
            let handles: Vec<_> = forks
                .into_iter()
                .map(|mut fork| scope.spawn(move || worker_scan(&mut fork)))
                .collect();
            let mut out = Vec::with_capacity(workers);
            out.push(worker_scan(cx));
            for h in handles {
                if let Ok(res) = h.join() {
                    out.push(res);
                }
            }
            out
        })
    };
    let mut may_unwind = vec![false; n];
    let mut work = Vec::new();
    let mut head = vec![0u32; n + 1];
    for (directs, edges) in &parts {
        for &u in directs {
            may_unwind[u as usize] = true;
            work.push(u);
        }
        for &(v, _) in edges {
            head[v as usize + 1] += 1;
        }
    }
    for i in 1..=n {
        head[i] += head[i - 1];
    }
    let mut cursor = head[..n].to_vec();
    let mut flat_rev = vec![0u32; head[n] as usize];
    for (_, edges) in parts {
        for (v, u) in edges {
            let at = cursor[v as usize] as usize;
            cursor[v as usize] = (at + 1) as u32;
            flat_rev[at] = u;
        }
    }
    while let Some(v) = work.pop() {
        let start = head[v as usize] as usize;
        let end = head[v as usize + 1] as usize;
        for &u in &flat_rev[start..end] {
            if !may_unwind[u as usize] {
                may_unwind[u as usize] = true;
                work.push(u);
            }
        }
    }
    may_unwind
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::names::Names;
    use crate::outline::Outline;
    use crate::sema::Cache;
    use crate::sema::substs::Substs;
    use std::sync::Arc;

    #[test]
    fn thread_cell_host_symbol_may_unwind() {
        assert!(host_symbol_may_unwind(&HostSymbol::ThreadCell));
    }

    #[test]
    fn parallel_may_unwind_propagates_across_csr_graph() {
        let mut src = String::new();
        for i in 0..300 {
            src.push_str(&format!("fn f{i}() {{}}\n"));
        }
        let parsed = crate::outline::source::Source::parse("test.rs".into(), src).expect("parse");
        let scanned = crate::outline::scan::scan(&parsed).expect("scan");
        let mut outline = Outline::file(
            std::path::Path::new("test.rs"),
            crate::graph::cfg::Target::default(),
        );
        outline.crates[0]
            .files
            .push(Arc::new(crate::outline::tree::File::new(
                parsed, scanned, false,
            )));
        let names = Names::default();
        let store = Cache::default();
        let empty_typed = Arc::new(Typed::default());
        let mut reached = Reached::default();
        for i in 0..300 {
            let inst = (
                ItemRef {
                    krate: 0,
                    file: 0,
                    item: i,
                },
                Substs::EMPTY,
            );
            let res = if i == 299 {
                Err("err".to_owned())
            } else {
                Ok(Arc::clone(&empty_typed))
            };
            reached.list.push((inst, res));
            reached.index.insert(inst, i as usize);
            reached.calls.push(if i < 299 {
                vec![(i + 1) as usize]
            } else {
                Vec::new()
            });
        }
        let mut cx = Context::new(&outline, &names, store);
        let unwinds = compute_may_unwind(&mut cx, &reached);
        assert_eq!(unwinds.len(), 300);
        assert!(unwinds.into_iter().all(|u| u));
    }
}
