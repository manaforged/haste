use rustc_hash::FxHashMap;

use crate::body::{BinOp, BlockKind, Body, Expr, ExprId, Stmt, UnOp};
use crate::host::HostSymbol;
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
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
            | HostSymbol::Dynamic(_)
    )
}

fn extern_abi_may_unwind(cx: &Context<'_>, item: ItemRef) -> bool {
    let Some(parent) = cx.parent_item(item) else {
        return true;
    };
    let (src, entry) = cx.item(parent);
    if entry.kind != ItemKind::ExternBlock {
        return true;
    }
    let Some(open) = entry.body else {
        return true;
    };
    for tok in entry.start..open {
        let text = src.text(tok);
        if text.starts_with('"') && (text.contains("-unwind") || text == "\"Rust\"") {
            return true;
        }
    }
    false
}

fn foreign_may_unwind(cx: &Context<'_>, item: ItemRef) -> bool {
    let Some(host) = cx
        .symbol_name(item)
        .as_deref()
        .and_then(|name| HostSymbol::resolve_for(name, true))
    else {
        return true;
    };
    match host {
        HostSymbol::Dynamic(_) => extern_abi_may_unwind(cx, item),
        _ => host_symbol_may_unwind(&host),
    }
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

fn all_params_moved(cx: &Context<'_>, body: &Body, typed: &Typed) -> bool {
    if typed.params.is_empty() {
        return true;
    }
    if !body.fn_params.is_empty() || typed.params.len() > 8 {
        return false;
    }
    let mut call_callees = vec![false; body.exprs.len()];
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

fn has_indirect_call(
    cx: &mut Context<'_>,
    source: ItemRef,
    overflow_checks: bool,
    typed: &Typed,
    cache: &mut FxHashMap<(ItemRef, bool), BodyScan>,
) -> (bool, bool) {
    if scan_body_ops(cx, source, overflow_checks, cache) {
        return (true, false);
    }
    let Some(BodyScan::IndirectCalls {
        calls,
        no_normal_drops,
    }) = cache.get(&(source, overflow_checks))
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
    let skip_drops =
        no_normal_drops && matches!(typed.shape, Shape::Body) && all_params_moved(cx, &body, typed);
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

fn direct_may_unwind(
    cx: &mut Context<'_>,
    reached: &Reached,
    index: usize,
    body_cache: &mut FxHashMap<(ItemRef, bool), BodyScan>,
    item_cache: &mut FxHashMap<ItemRef, ItemUnwind>,
) -> (bool, bool) {
    let Some(((item, substs), res)) = reached.list.get(index) else {
        return (true, false);
    };
    let item = *item;
    match item_unwind(cx, item, item_cache) {
        ItemUnwind::Intrinsic(u) | ItemUnwind::Foreign(u) => return (u, false),
        ItemUnwind::Normal => {}
    }
    let Ok(typed) = res else {
        return (true, false);
    };
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
                has_indirect_call(cx, source, overflow_checks, typed, body_cache);
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
        match item_unwind(cx, callee, item_cache) {
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

pub(super) fn compute_may_unwind(cx: &mut Context<'_>, reached: &Reached) -> Vec<bool> {
    let n = reached.list.len();
    let mut may_unwind = vec![false; n];
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut body_cache = FxHashMap::default();
    let mut item_cache = FxHashMap::default();
    let mut work = Vec::new();
    for u in 0..n {
        let (direct, skip_drops) =
            direct_may_unwind(cx, reached, u, &mut body_cache, &mut item_cache);
        if direct {
            may_unwind[u] = true;
            work.push(u);
            continue;
        }
        if skip_drops {
            if let Some((_, Ok(typed))) = reached.list.get(u) {
                for (callee, csubsts) in typed.calls_without_drops() {
                    if let Some(&v) = reached.index.get(&(callee, csubsts))
                        && v < n
                    {
                        rev[v].push(u);
                    }
                }
            }
        } else if let Some(callees) = reached.calls.get(u).filter(|c| !c.is_empty()) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_cell_host_symbol_may_unwind() {
        assert!(host_symbol_may_unwind(&HostSymbol::ThreadCell));
    }
}
