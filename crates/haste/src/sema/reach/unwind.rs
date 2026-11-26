use rustc_hash::FxHashMap;

use crate::body::{BinOp, Expr, ExprId, UnOp};
use crate::host::HostSymbol;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{LazyTables, Res, Shape, SiteKey, Typed};
use crate::sema::intrinsic::Intrinsic;
use crate::sema::ty::Kind;

use super::Reached;

/// Whether calling host symbol `host` may initiate or propagate a panic unwind.
pub(crate) fn host_symbol_may_unwind(host: &HostSymbol) -> bool {
    matches!(host, HostSymbol::StartPanic | HostSymbol::CatchUnwind | HostSymbol::FillOnce | HostSymbol::Dynamic(_))
}

fn foreign_may_unwind(cx: &Context<'_>, item: ItemRef) -> bool {
    cx.symbol_name(item)
        .as_deref()
        .and_then(|name| HostSymbol::resolve_for(name, true))
        .is_none_or(|host| host_symbol_may_unwind(&host))
}

fn intrinsic_may_unwind(intr: Intrinsic) -> bool {
    matches!(intr, Intrinsic::ConstEvalSelect | Intrinsic::CatchUnwind)
}

enum BodyScan {
    Unwinds,
    IndirectCalls(Vec<ExprId>),
}

fn scan_body_ops(cx: &mut Context<'_>, source: ItemRef, overflow_checks: bool, cache: &mut FxHashMap<(ItemRef, bool), BodyScan>) -> bool {
    let entry = cache.entry((source, overflow_checks)).or_insert_with(|| {
        let Ok(body) = cx.body(source) else {
            return BodyScan::Unwinds;
        };
        let mut calls = Vec::new();
        for (idx, expr) in body.exprs.iter().enumerate() {
            match expr {
                Expr::FormatArgs(_) | Expr::Index(..) | Expr::Macro { .. } => return BodyScan::Unwinds,
                Expr::Binary(op, ..) | Expr::AssignOp(op, ..)
                    if matches!(op, BinOp::Div | BinOp::Rem)
                        || (overflow_checks && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl | BinOp::Shr)) =>
                {
                    return BodyScan::Unwinds;
                }
                Expr::Unary(UnOp::Neg, _) if overflow_checks => return BodyScan::Unwinds,
                Expr::Call(..) => calls.push(ExprId(idx as u32)),
                _ => {}
            }
        }
        BodyScan::IndirectCalls(calls)
    });
    matches!(entry, BodyScan::Unwinds)
}

fn has_indirect_call(cx: &mut Context<'_>, source: ItemRef, overflow_checks: bool, typed: &Typed, cache: &mut FxHashMap<(ItemRef, bool), BodyScan>) -> bool {
    if scan_body_ops(cx, source, overflow_checks, cache) {
        return true;
    }
    let Some(BodyScan::IndirectCalls(calls)) = cache.get(&(source, overflow_checks)) else {
        return true;
    };
    if calls.is_empty() {
        return false;
    }
    let Ok(body) = cx.body(source) else {
        return true;
    };
    for &expr in calls {
        let Expr::Call(callee, _) = body.exprs[expr.0 as usize] else { continue };
        let direct = matches!(typed.site_res(&cx.types, SiteKey::ExprRes(expr)), Some(Res::Fn(..) | Res::Ctor(..)))
            || matches!(typed.site_res(&cx.types, SiteKey::ExprRes(callee)), Some(Res::Fn(..) | Res::Ctor(..)));
        if !direct {
            return true;
        }
    }
    false
}

#[derive(Clone, Copy)]
enum ItemUnwind {
    Normal,
    Intrinsic(bool),
    Foreign(bool),
}

fn item_unwind(cx: &Context<'_>, item: ItemRef, cache: &mut FxHashMap<ItemRef, ItemUnwind>) -> ItemUnwind {
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

fn direct_may_unwind(
    cx: &mut Context<'_>,
    reached: &Reached,
    index: usize,
    body_cache: &mut FxHashMap<(ItemRef, bool), BodyScan>,
    item_cache: &mut FxHashMap<ItemRef, ItemUnwind>,
) -> bool {
    let Some(((item, substs), res)) = reached.list.get(index) else {
        return true;
    };
    let item = *item;
    match item_unwind(cx, item, item_cache) {
        ItemUnwind::Intrinsic(u) | ItemUnwind::Foreign(u) => return u,
        ItemUnwind::Normal => {}
    }
    let Ok(typed) = res else {
        return true;
    };
    match &typed.shape {
        Shape::Virtual(_) | Shape::Asm(_) | Shape::Anon(_) => return true,
        Shape::Untuple(Res::Fn(..)) | Shape::Clone(_) | Shape::Ctor(_) | Shape::GlobalAlloc(_) => {}
        Shape::Untuple(_) => return true,
        Shape::Glue(_) => {
            if let Some(&ty) = cx.types.substs(*substs).first()
                && matches!(cx.types.kind(ty), Kind::Dyn(..))
            {
                return true;
            }
        }
        Shape::Body | Shape::Closure(_) => {
            if matches!(typed.tables, LazyTables::CheckOnly(_)) {
                return true;
            }
            let source = match &typed.shape {
                Shape::Closure(cb) => cb.parent,
                _ => item,
            };
            let overflow_checks = cx.overflow_checks(source.krate);
            if has_indirect_call(cx, source, overflow_checks, typed, body_cache) {
                return true;
            }
        }
    }
    for (callee, csubsts) in typed.calls() {
        match item_unwind(cx, callee, item_cache) {
            ItemUnwind::Intrinsic(true) | ItemUnwind::Foreign(true) => return true,
            ItemUnwind::Intrinsic(false) | ItemUnwind::Foreign(false) => continue,
            ItemUnwind::Normal => {}
        }
        if !reached.index.contains_key(&(callee, csubsts)) {
            return true;
        }
    }
    false
}

pub(super) fn compute_may_unwind(cx: &mut Context<'_>, reached: &Reached) -> Vec<bool> {
    let n = reached.list.len();
    let mut may_unwind = vec![false; n];
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut body_cache = FxHashMap::default();
    let mut item_cache = FxHashMap::default();
    let mut work = Vec::new();
    for u in 0..n {
        if let Some(callees) = reached.calls.get(u).filter(|c| !c.is_empty()) {
            for &v in callees {
                if v < n {
                    rev[v].push(u);
                }
            }
        } else if let Some((_, Ok(typed))) = reached.list.get(u) {
            for v in typed.collect_link_edges(&reached.index).0 {
                if v < n {
                    rev[v].push(u);
                }
            }
        }
        if direct_may_unwind(cx, reached, u, &mut body_cache, &mut item_cache) {
            may_unwind[u] = true;
            work.push(u);
        }
    }
    while let Some(v) = work.pop() {
        for &u in &rev[v] {
            if !may_unwind[u] {
                may_unwind[u] = true;
                work.push(u);
            }
        }
    }
    may_unwind
}
