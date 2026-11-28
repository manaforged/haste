use rustc_hash::FxHashSet as HashSet;

use crate::cli::test_items;
use crate::graph;
use crate::image::ImageMode;
use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::sema::reach::{
    ReloadScope, check_poly_items, dirty_poly_items, reach_again, reach_anew_in, reach_anew_scoped,
    reload_invalidate, split_dirty_roots,
};
use crate::sema::substs::Substs;
use crate::sema::{self, Edited};

use super::super::test_args::{map_tests, select_test_roots};
use super::{Batch, BatchKind, batch_labels, concrete_roots, recompile, test_indices};

#[cfg(feature = "diff-harness")]
use super::item_roots;

impl Batch {
    fn needs_member_recheck(&self) -> bool {
        !self.member_errors.is_empty() || matches!(self.kind, BatchKind::Tests | BatchKind::MainAot)
    }

    fn finish_incremental(&mut self, mut cx: sema::Context<'_>, fresh: &[usize]) -> bool {
        recompile(
            &mut self.image,
            &mut self.broken,
            (&mut cx, &self.reached),
            fresh,
            &self.roots,
            self.kind.mode() == ImageMode::Check,
        );
        self.store = Some(cx.release().0);
        true
    }

    pub(in crate::build) fn again(
        &mut self,
        workspace: &graph::Workspace,
        compilation: &Compilation,
        stale: &[(u32, u32)],
    ) -> bool {
        if self.needs_member_recheck() {
            return false;
        }
        let Some(mut store) = self.store.take() else {
            return false;
        };
        let touched = store.invalidate(
            &compilation.outline,
            self.reached.list.iter().map(|(instance, _)| instance),
            Edited::Files(stale),
        );
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        if matches!(self.kind, BatchKind::LibCheck | BatchKind::MainCheck) {
            let in_stale = |item: ItemRef| {
                let owner = item.body_owner();
                stale.contains(&(owner.krate, owner.file))
            };
            self.poly_failures.retain(|(item, _)| !in_stale(*item));
            let edited: Vec<ItemRef> = self
                .kind
                .pick(workspace, &compilation.outline)
                .into_iter()
                .filter(|&item| in_stale(item))
                .collect();
            self.poly_failures
                .extend(check_poly_items(&mut cx, &edited));
        }
        let fresh = reach_again(&mut cx, &mut self.reached, &touched);
        cx.resolved() && self.finish_incremental(cx, &fresh)
    }

    pub(in crate::build) fn reload(
        &mut self,
        workspace: &graph::Workspace,
        compilation: &Compilation,
        crates: &[u32],
        scoped: Option<&ReloadScope>,
    ) -> bool {
        if self.needs_member_recheck() {
            return false;
        }
        let Some(mut store) = self.store.take() else {
            return false;
        };
        let ((touched, in_place), evicted_summaries) = match scoped {
            Some(scope) => {
                let (dead, in_place, evicted) =
                    reload_invalidate(&mut store, &compilation.outline, &self.reached, scope);
                ((dead, in_place), evicted)
            }
            None => (
                (
                    store.invalidate(
                        &compilation.outline,
                        self.reached.list.iter().map(|(instance, _)| instance),
                        Edited::Crates(crates),
                    ),
                    Vec::new(),
                ),
                HashSet::default(),
            ),
        };
        let mut cases = self.kind.pick(workspace, &compilation.outline);
        let mut preserved_roots: Option<Vec<usize>> = None;
        let roots;
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        let mut test_starts = Vec::new();
        match self.kind {
            #[cfg(feature = "diff-harness")]
            BatchKind::Cases => {
                cases.retain(|&item| sema::infer::is_case_sig(&mut cx, item));
                roots = item_roots(&cases);
            }
            BatchKind::Tests => {
                let (valid, starts, test_roots) = select_test_roots(&mut cx, &cases, &[]);
                (cases, test_starts, roots) = (valid, starts, test_roots);
            }
            BatchKind::LibCheck | BatchKind::MainCheck => {
                let edited = match scoped {
                    Some(scope) => {
                        dirty_poly_items(&cases, &mut self.poly_failures, scope, &evicted_summaries)
                    }
                    None => {
                        self.poly_failures
                            .retain(|(i, _)| !crates.contains(&i.krate));
                        cases
                            .iter()
                            .copied()
                            .filter(|i| crates.contains(&i.krate))
                            .collect()
                    }
                };
                self.poly_failures
                    .extend(check_poly_items(&mut cx, &edited));
                if scoped.is_some() {
                    let (kept, dirty) =
                        split_dirty_roots(&mut cx, &cases, &self.reached, &self.roots, &touched);
                    (preserved_roots, roots) = (Some(kept), dirty);
                } else {
                    roots = concrete_roots(&mut cx, &cases);
                }
            }
            BatchKind::MainAot => {
                let Some(&main) = cases.first() else {
                    return false;
                };
                let Ok(pair) = sema::reach::main_roots(&mut cx, main) else {
                    return false;
                };
                cases.clear();
                roots = pair.to_vec();
            }
        }
        let fresh = match scoped {
            Some(scope) => reach_anew_scoped(
                &mut cx,
                &mut self.reached,
                (&touched, &in_place),
                &roots,
                scope,
            ),
            None => reach_anew_in(&mut cx, &mut self.reached, &touched, &roots, Some(crates)),
        };
        if !cx.resolved() {
            return false;
        }
        self.labels = batch_labels(&mut cx, self.kind, &cases);
        self.roots = if self.kind == BatchKind::Tests {
            cases
                .iter()
                .map(|&item| self.reached.index[&(item, Substs::EMPTY)])
                .collect()
        } else {
            let mut next = preserved_roots.unwrap_or_default();
            next.extend(roots.iter().map(|root| self.reached.index[root]));
            next
        };
        if self.kind == BatchKind::Tests {
            let starts = test_indices(&test_starts, &self.reached);
            self.tests = map_tests(
                &self.roots,
                &self.reached,
                &starts,
                test_items(workspace, &compilation.outline),
            );
        }
        touched
            .iter()
            .chain(&in_place)
            .for_each(|idx| drop(self.broken.remove(idx)));
        self.finish_incremental(cx, &fresh)
    }
}
