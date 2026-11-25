use crate::body::{Expr, ExprId};
use crate::lower::is_freeze_ty;
use crate::sema::ty::Ty;

use super::super::{Checker, State};
use super::{BorrowSource, dedup_sources};

#[derive(Clone, Debug)]
pub(in super::super) enum TupleSources {
    Leaf {
        sources: Vec<BorrowSource>,
        inner_ref: Option<Vec<BorrowSource>>,
    },
    Tuple(Vec<TupleSources>),
}

impl TupleSources {
    pub(in super::super) fn merge(&mut self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Leaf {
                    sources: a,
                    inner_ref: ar,
                },
                Self::Leaf {
                    sources: b,
                    inner_ref: br,
                },
            ) => {
                a.extend(b.iter().cloned());
                *a = dedup_sources(std::mem::take(a));
                *ar = match (ar.take(), br) {
                    (Some(mut x), Some(y)) => {
                        x.extend(y.iter().cloned());
                        Some(dedup_sources(x))
                    }
                    _ => None,
                };
                true
            }
            (Self::Tuple(a), Self::Tuple(b)) if a.len() == b.len() => {
                a.iter_mut().zip(b).all(|(x, y)| x.merge(y))
            }
            _ => false,
        }
    }

    pub(in super::super) fn flatten_into(&self, out: &mut Vec<BorrowSource>) {
        match self {
            Self::Leaf { sources, .. } => out.extend(sources.iter().cloned()),
            Self::Tuple(elems) => {
                for elem in elems {
                    elem.flatten_into(out);
                }
            }
        }
    }

    pub(in super::super) fn sources_for(&self, outer_hrtb: bool) -> Vec<BorrowSource> {
        if outer_hrtb
            && let Self::Leaf {
                inner_ref: Some(inner),
                ..
            } = self
        {
            return inner.clone();
        }
        let mut flat = Vec::new();
        self.flatten_into(&mut flat);
        flat
    }
}

impl Checker<'_, '_> {
    pub(in super::super) fn expr_tuple_sources(
        &mut self,
        expr: ExprId,
        state: &State,
        use_let_extend: bool,
    ) -> Option<TupleSources> {
        let cur = self.unwrap_parens(expr);
        let ty = self.expr_ty(cur);
        if ty == Ty::NONE || !is_freeze_ty(self.cx, ty, 0) {
            return None;
        }
        match *self.body.expr(cur) {
            Expr::Tuple(list) => {
                let items: Vec<u32> = self.body.list(list).to_vec();
                let elems = items
                    .into_iter()
                    .map(|id| {
                        let elem = ExprId(id);
                        self.expr_tuple_sources(elem, state, use_let_extend)
                            .unwrap_or_else(|| {
                                let sources = if use_let_extend {
                                    self.expr_borrow_sources_let(elem, state)
                                } else {
                                    self.expr_borrow_sources(elem, state)
                                };
                                let inner_ref = match *self.body.expr(self.unwrap_parens(elem)) {
                                    Expr::Ref {
                                        raw: false,
                                        expr: inner,
                                        ..
                                    } if is_freeze_ty(self.cx, self.expr_ty(inner), 0) => {
                                        Some(self.expr_borrow_sources(inner, state))
                                    }
                                    _ => None,
                                };
                                TupleSources::Leaf { sources, inner_ref }
                            })
                    })
                    .collect();
                Some(TupleSources::Tuple(elems))
            }
            Expr::Path(_) => self
                .expr_plain_local(cur)
                .and_then(|l| state.tuple_elem_sources.get(&l).cloned()),
            _ => None,
        }
    }
}
