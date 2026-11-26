use anyhow::{Context, Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

#[cfg(feature = "diff-harness")]
use crate::cli::case_items;
use crate::cli::{TestItem, test_items};
use crate::graph;
use crate::image::Image;
use crate::macros::Compilation;
use crate::names::ItemRef;
use crate::outline;
use crate::sema::reach::{Instance, Reached, reach_again, reach_anew};
use crate::sema::{self, Cache, Edited};

use super::substrate::{
    SubstrateSlot, cache_with_substrate, export_pre_summaries, save_batch_substrate,
};
use super::test_args::{check_fn_items, map_tests, select_test_roots, test_label};
use super::{main_item, require_main};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BatchKind {
    #[cfg(feature = "diff-harness")]
    Cases,
    Tests,
    LibCheck,
    MainCheck,
    MainAot,
}

impl BatchKind {
    fn mode(self) -> crate::image::ImageMode {
        match self {
            #[cfg(feature = "diff-harness")]
            BatchKind::Cases => crate::image::ImageMode::Jit,
            BatchKind::Tests => crate::image::ImageMode::Jit,
            BatchKind::LibCheck | BatchKind::MainCheck => crate::image::ImageMode::Check,
            BatchKind::MainAot => crate::image::ImageMode::Aot,
        }
    }

    fn pick(self, workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<ItemRef> {
        match self {
            #[cfg(feature = "diff-harness")]
            BatchKind::Cases => case_items(workspace, outline),
            BatchKind::Tests => test_items(workspace, outline)
                .into_iter()
                .map(|t| t.item)
                .collect(),
            BatchKind::LibCheck => check_fn_items(workspace, outline, None),
            BatchKind::MainCheck => {
                let sel = crate::selection();
                let target = (sel.package.is_some() || sel.bin.is_some() || sel.example.is_some())
                    .then(|| main_item(workspace, outline).map(|m| m.krate as usize))
                    .flatten();
                check_fn_items(workspace, outline, target)
            }
            BatchKind::MainAot => main_items(workspace, outline),
        }
    }
}

pub(super) struct Batch {
    pub(super) kind: BatchKind,
    pub(super) store: Option<Cache>,
    pub(super) labels: Vec<String>,
    pub(super) roots: Vec<usize>,
    pub(super) tests: Vec<(TestItem, Option<usize>)>,
    pub(super) reached: Reached,
    pub(super) image: Image,
    pub(super) broken: HashMap<usize, String>,
    pub(super) poly_failures: Vec<(ItemRef, String)>,
    pub(super) member_errors: Vec<String>,
    pub(super) reach_span: sema::census::Span,
    pub(super) codegen_span: sema::census::Span,
}

/// Type-checks and borrow-checks every concrete and polymorphic function in the target member crate(s) without emitting code.
pub(super) fn check_member_items(
    workspace: &graph::Workspace,
    compilation: &mut Compilation,
    cache: Cache,
    target: Option<usize>,
) -> Result<(Cache, Vec<String>)> {
    super::load_members(workspace, compilation)?;
    let items = check_fn_items(workspace, &compilation.outline, target);
    let roots: Vec<Instance> = items
        .iter()
        .copied()
        .map(|item| (item, sema::substs::Substs::EMPTY))
        .collect();
    let (store, diagnostics) = crate::macros::reach_resolved_with_cache(
        compilation,
        cache,
        |cx| {
            roots
                .iter()
                .copied()
                .filter(|&(item, _)| sema::infer::is_concrete_fn(cx, item))
                .collect()
        },
        |mut cx, reached| {
            let poly = check_poly_items(&mut cx, &items);
            let mut out = Vec::new();
            let mut seen = HashSet::default();
            let mut record = |item: ItemRef, error: &str| {
                let msg = Batch::format_entry(Some(cx.outline), item, None, error);
                if seen.insert(msg.clone()) {
                    out.push(msg);
                }
            };
            for ((item, _), res) in &reached.list {
                if let Err(err) = res {
                    record(*item, err);
                }
            }
            for &(item, ref err) in &poly {
                record(item, err);
            }
            Ok((cx.release().0, out))
        },
    )?;
    Ok((store, diagnostics))
}

fn batch_roots(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
    kind: BatchKind,
) -> (Vec<ItemRef>, Vec<Instance>) {
    let items = kind.pick(workspace, outline);
    let roots = items
        .iter()
        .map(|item| (*item, sema::substs::Substs::EMPTY))
        .collect();
    (items, roots)
}

fn batch_labels(cx: &mut sema::Context<'_>, kind: BatchKind, items: &[ItemRef]) -> Vec<String> {
    items
        .iter()
        .map(|&item| {
            if kind == BatchKind::Tests {
                test_label(cx, item)
            } else {
                cx.name(item).to_owned()
            }
        })
        .collect()
}

fn main_items(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<ItemRef> {
    main_item(workspace, outline).into_iter().collect()
}

fn check_poly_items(cx: &mut sema::Context<'_>, items: &[ItemRef]) -> Vec<(ItemRef, String)> {
    let mut out = Vec::new();
    for &item in items {
        if !sema::infer::is_concrete_fn(cx, item)
            && let Err(err) = sema::summary::check_poly_item(cx, item)
        {
            out.push((item, err));
        }
    }
    out
}

fn recompile(
    image: &mut Image,
    broken: &mut HashMap<usize, String>,
    (cx, reached): (&mut sema::Context<'_>, &Reached),
    fresh: &[usize],
    roots: &[usize],
) {
    let mut expanded = fresh.to_vec();
    if crate::profile() == outline::tree::Profile::Release
        && !fresh.is_empty()
        && fresh.len() < reached.list.len()
    {
        let mut dirty = vec![false; reached.list.len()];
        for &at in fresh {
            dirty[at] = true;
        }
        for _ in 0..3 {
            let added: Vec<usize> = (0..reached.list.len())
                .filter(|&u| {
                    !dirty[u]
                        && reached
                            .direct_calls(u)
                            .is_some_and(|callees| callees.iter().any(|&c| dirty[c]))
                })
                .collect();
            if added.is_empty() {
                break;
            }
            for u in added {
                dirty[u] = true;
                expanded.push(u);
            }
        }
    }
    expanded.iter().for_each(|at| drop(broken.remove(at)));
    broken.extend(image.compile(cx, reached, &expanded));
    if let &[start, main] = roots {
        let mut tried: HashSet<usize> = expanded.into_iter().collect();
        loop {
            let missing: Vec<usize> = image
                .missing_aot_targets([start, main], reached)
                .into_iter()
                .filter(|idx| !broken.contains_key(idx) && tried.insert(*idx))
                .collect();
            if missing.is_empty() {
                break;
            }
            broken.extend(image.compile(cx, reached, &missing));
        }
    }
}

fn finish_batch(
    (kind, main): (BatchKind, Option<ItemRef>),
    (cases, mut reached): (&[ItemRef], Reached),
    drop_finals: bool,
    substrate: Option<SubstrateSlot>,
    (mut cx, reach_clock): (sema::Context<'_>, sema::census::Clock),
) -> Result<Batch> {
    let mode = kind.mode();
    let reach_span = reach_clock.elapsed();
    let codegen_clock = sema::census::Clock::now();
    let pre_builder = export_pre_summaries(&mut cx, drop_finals, substrate.as_ref());
    let mut test_starts: Vec<Option<sema::reach::Instance>> = Vec::new();
    let valid: Vec<ItemRef> = match kind {
        #[cfg(feature = "diff-harness")]
        BatchKind::Cases => cases
            .iter()
            .copied()
            .filter(|&item| sema::infer::is_case_sig(&mut cx, item))
            .collect(),
        BatchKind::Tests => {
            let (valid, starts, _) = select_test_roots(&mut cx, cases);
            test_starts = starts;
            valid
        }
        BatchKind::LibCheck | BatchKind::MainCheck => cases
            .iter()
            .copied()
            .filter(|&item| sema::infer::is_concrete_fn(&mut cx, item))
            .collect(),
        BatchKind::MainAot => Vec::new(),
    };
    let poly_failures = if matches!(kind, BatchKind::LibCheck | BatchKind::MainCheck) {
        check_poly_items(&mut cx, cases)
    } else {
        Vec::new()
    };
    let mut root_instances: Vec<sema::reach::Instance> = match main {
        Some(main) => sema::reach::main_roots(&mut cx, main)?.to_vec(),
        None => valid
            .iter()
            .map(|item| (*item, sema::substs::Substs::EMPTY))
            .collect(),
    };
    let extra_starts: Vec<sema::reach::Instance> = test_starts.iter().flatten().copied().collect();
    let primary_len = root_instances.len();
    root_instances.extend(extra_starts);
    reached.canonicalize(&cx, &root_instances);
    let roots: Vec<usize> = root_instances[..primary_len]
        .iter()
        .map(|inst| reached.index[inst])
        .collect();
    let start_indices: Vec<Option<usize>> = test_starts
        .iter()
        .map(|opt| opt.and_then(|inst| reached.index.get(&inst).copied()))
        .collect();
    if drop_finals {
        cx.compact_for_oneshot_codegen();
    }
    let (mut image, broken) = if mode == crate::image::ImageMode::Aot && roots.len() == 2 {
        Image::build_aot(&mut cx, &reached, [roots[0], roots[1]])?
    } else {
        Image::build(&mut cx, &reached, mode)?
    };
    let codegen_span = codegen_clock.elapsed();
    let labels = batch_labels(&mut cx, kind, &valid);
    let outline = cx.outline;
    let mut store = cx.release().0;
    save_batch_substrate(
        (outline, &mut store),
        &mut image,
        (drop_finals, kind),
        substrate,
        pre_builder,
    );
    let tests = start_indices
        .into_iter()
        .map(|start| {
            (
                TestItem {
                    item: ItemRef {
                        krate: 0,
                        file: 0,
                        item: 0,
                    },
                    ignored: false,
                    should_panic: false,
                    expected: None,
                },
                start,
            )
        })
        .collect();
    Ok(Batch {
        kind,
        store: Some(store),
        labels,
        roots,
        tests,
        reached,
        image,
        broken,
        poly_failures,
        member_errors: Vec::new(),
        reach_span,
        codegen_span,
    })
}

impl Batch {
    pub(super) fn new(
        workspace: &graph::Workspace,
        compilation: &mut Compilation,
        kind: BatchKind,
        drop_finals: bool,
    ) -> Result<Batch> {
        let main = (kind == BatchKind::MainAot)
            .then(|| {
                let sel = crate::selection();
                if sel.bin.is_some() || sel.example.is_some() {
                    require_main(workspace, &compilation.outline)
                } else {
                    main_item(workspace, &compilation.outline).context(
                        "no binary or example target to emit (pass --bin NAME or --example NAME)",
                    )
                }
            })
            .transpose()?;
        Self::rooted(workspace, compilation, (kind, main), drop_finals)
    }

    pub(super) fn rooted(
        workspace: &graph::Workspace,
        compilation: &mut Compilation,
        (kind, main): (BatchKind, Option<ItemRef>),
        drop_finals: bool,
    ) -> Result<Batch> {
        let mode = kind.mode();
        let reach_clock = sema::census::Clock::now();
        let (cases, root_insts) = if main.is_some() {
            (Vec::new(), Vec::new())
        } else {
            batch_roots(workspace, &compilation.outline, kind)
        };
        let (mut substrate, mut cache) =
            cache_with_substrate(workspace, mode != crate::image::ImageMode::Check);
        if matches!(kind, BatchKind::Tests | BatchKind::MainAot) {
            let target = match kind {
                BatchKind::MainAot => main.map(|m| m.krate as usize),
                _ => None,
            };
            let (checked_cache, member_errors) =
                check_member_items(workspace, compilation, cache, target)?;
            cache = checked_cache;
            if !member_errors.is_empty() {
                let reached = Reached::default();
                let image = Image::create(&reached, crate::image::ImageMode::Check, (0, 0, 1))?;
                return Ok(Batch {
                    kind,
                    store: Some(cache),
                    labels: Vec::new(),
                    roots: Vec::new(),
                    tests: Vec::new(),
                    reached,
                    image,
                    broken: HashMap::default(),
                    poly_failures: Vec::new(),
                    member_errors,
                    reach_span: reach_clock.elapsed(),
                    codegen_span: sema::census::Span::default(),
                });
            }
        }
        let batch = crate::macros::reach_resolved_with_cache(
            compilation,
            cache,
            |cx| match (kind, main) {
                (BatchKind::MainAot, Some(main_ref)) => sema::reach::main_roots(cx, main_ref)
                    .map_or_else(
                        |_| vec![(main_ref, sema::substs::Substs::EMPTY)],
                        Into::into,
                    ),
                #[cfg(feature = "diff-harness")]
                (BatchKind::Cases, _) => root_insts
                    .iter()
                    .copied()
                    .filter(|&(item, _)| sema::infer::is_case_sig(cx, item))
                    .collect(),
                (BatchKind::Tests, _) => select_test_roots(cx, &cases).2,
                (BatchKind::LibCheck | BatchKind::MainCheck, _) => root_insts
                    .iter()
                    .copied()
                    .filter(|&(item, _)| sema::infer::is_concrete_fn(cx, item))
                    .collect(),
                _ => root_insts.clone(),
            },
            |cx, reached| {
                finish_batch(
                    (kind, main),
                    (&cases, reached),
                    drop_finals,
                    substrate.take(),
                    (cx, reach_clock),
                )
            },
        )?;
        crate::substrate::save_fixed_files(
            workspace,
            crate::host_target().triple,
            &compilation.outline,
        );
        Ok(batch)
    }

    pub(super) fn write_aot(
        &mut self,
        compilation: &Compilation,
        path: &std::path::Path,
    ) -> Result<()> {
        if !self.member_errors.is_empty() {
            bail!("{}", self.member_errors.join("\n"));
        }
        if !self.broken.is_empty() {
            bail!("{} functions failed to compile", self.broken.len());
        }
        let reached_crates: HashSet<usize> = self
            .reached
            .list
            .iter()
            .map(|((item, _), _)| item.krate as usize)
            .collect();
        for krate_idx in reached_crates {
            let krate = &compilation.outline.crates[krate_idx];
            if let Some(script) = &krate.env.script
                && let Some((_, lib)) = script
                    .link_libs()
                    .find(|(kind, _)| *kind == crate::graph::script::LinkKind::Static)
            {
                let name = &compilation.outline.roots[krate_idx].name;
                bail!(
                    "standalone AOT (-o) does not yet link static C archive `{lib}` from crate `{name}`; use `haste run` or `cargo build`"
                );
            }
        }
        let store = self
            .store
            .take()
            .context("the binary's inference store is in use")?;
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        let exe = self.image.to_aot(
            &crate::host_target(),
            (self.roots[0], self.roots[1]),
            &mut cx,
            &self.reached,
        );
        self.store = Some(cx.release().0);
        exe?.write(path)
            .with_context(|| format!("writing {}", path.display()))
    }

    fn needs_member_recheck(&self) -> bool {
        !self.member_errors.is_empty() || matches!(self.kind, BatchKind::Tests | BatchKind::MainAot)
    }

    pub(super) fn again(&mut self, compilation: &Compilation, stale: &[(u32, u32)]) -> bool {
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
        let fresh = reach_again(&mut cx, &mut self.reached, &touched);
        if !cx.resolved() {
            return false;
        }
        recompile(
            &mut self.image,
            &mut self.broken,
            (&mut cx, &self.reached),
            &fresh,
            &self.roots,
        );
        self.store = Some(cx.release().0);
        true
    }

    pub(super) fn reload(
        &mut self,
        workspace: &graph::Workspace,
        compilation: &Compilation,
        crates: &[u32],
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
            Edited::Crates(crates),
        );
        let (mut cases, _) = batch_roots(workspace, &compilation.outline, self.kind);
        let roots;
        let mut cx = sema::Context::new(&compilation.outline, &compilation.names, store);
        let mut test_starts = Vec::new();
        match self.kind {
            #[cfg(feature = "diff-harness")]
            BatchKind::Cases => {
                cases.retain(|&item| sema::infer::is_case_sig(&mut cx, item));
                roots = cases
                    .iter()
                    .map(|&item| (item, sema::substs::Substs::EMPTY))
                    .collect();
            }
            BatchKind::Tests => {
                let (valid, starts, test_roots) = select_test_roots(&mut cx, &cases);
                cases = valid;
                test_starts = starts;
                roots = test_roots;
            }
            BatchKind::LibCheck | BatchKind::MainCheck => {
                self.poly_failures = check_poly_items(&mut cx, &cases);
                cases.retain(|&item| sema::infer::is_concrete_fn(&mut cx, item));
                roots = cases
                    .iter()
                    .map(|&item| (item, sema::substs::Substs::EMPTY))
                    .collect();
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
        let fresh = reach_anew(&mut cx, &mut self.reached, &touched, &roots);
        if !cx.resolved() {
            return false;
        }
        self.labels = batch_labels(&mut cx, self.kind, &cases);
        self.roots = match self.kind {
            BatchKind::Tests => cases
                .iter()
                .map(|&item| self.reached.index[&(item, sema::substs::Substs::EMPTY)])
                .collect(),
            _ => roots.iter().map(|root| self.reached.index[root]).collect(),
        };
        if self.kind == BatchKind::Tests {
            let starts: Vec<Option<usize>> = test_starts
                .into_iter()
                .map(|opt| opt.and_then(|s| self.reached.index.get(&s).copied()))
                .collect();
            self.tests = map_tests(
                &self.roots,
                &self.reached,
                &starts,
                test_items(workspace, &compilation.outline),
            );
        }
        recompile(
            &mut self.image,
            &mut self.broken,
            (&mut cx, &self.reached),
            &fresh,
            &self.roots,
        );
        self.store = Some(cx.release().0);
        true
    }
}
