use std::time::{Duration, Instant};

use anyhow::Result;

use crate::cli::{Failures, TestItem, error_kind, print_failures, print_test_report};
use crate::names::ItemRef;
use crate::outline;
use crate::sema;

use super::batch::Batch;
#[cfg(feature = "diff-harness")]
use super::batch::BatchKind;
use super::test_args::parse_test_args;

#[cfg(feature = "diff-harness")]
include!("../../../../tests/case_inputs.rs");

impl Batch {
    fn sort(&self, outline: Option<&outline::Outline>) -> (Failures, Vec<(usize, &str)>) {
        let mut failures = Failures::new();
        let mut runnable = Vec::new();
        for (&root, name) in self.roots.iter().zip(&self.labels) {
            let closure = self.reached.closure(&[root]);
            let found = closure.iter().rev().find_map(|&at| {
                self.reached.list[at]
                    .1
                    .as_ref()
                    .err()
                    .or_else(|| self.broken.get(&at))
                    .map(|err| (at, err))
            });
            match found {
                Some((at, error)) => {
                    let loc = outline
                        .map(|o| item_loc(o, self.reached.list[at].0.0))
                        .unwrap_or_default();
                    let msg = if error.starts_with(name.as_str()) {
                        format!("{loc}{error}")
                    } else {
                        format!("{loc}{name}: {error}")
                    };
                    failures.entry(error_kind(error)).or_default().push(msg);
                }
                None => runnable.push((root, name.as_str())),
            }
        }
        for &(item, ref error) in &self.poly_failures {
            let loc = outline.map(|o| item_loc(o, item)).unwrap_or_default();
            failures
                .entry(error_kind(error))
                .or_default()
                .push(format!("{loc}{error}"));
        }
        (failures, runnable)
    }

    pub(super) fn failures(&self, cx: &sema::Context<'_>) -> Failures {
        let mut failures = Failures::new();
        for at in self.reached.closure(&self.roots) {
            if let Some(error) = self.reached.list[at]
                .1
                .as_ref()
                .err()
                .or_else(|| self.broken.get(&at))
            {
                let (item, substs) = self.reached.list[at].0;
                let (src, entry) = cx.item(item);
                let substs: Vec<String> = cx
                    .types
                    .substs(substs)
                    .iter()
                    .map(|&ty| cx.type_text(ty, &|ty| ty))
                    .collect();
                failures.entry(error_kind(error)).or_default().push(format!(
                    "{}:{} {error} [{}]",
                    src.path.display(),
                    src.line(entry.start),
                    substs.join(", ")
                ));
            }
        }
        failures
    }

    pub(super) fn check(&self, outline: &outline::Outline, built: Duration) -> i32 {
        let (failures, _) = self.sort(Some(outline));
        print_failures(&failures);
        eprintln!(
            "{} functions reached, {} failed, built in {:.1} ms",
            self.reached.list.len(),
            failures.values().map(Vec::len).sum::<usize>(),
            built.as_secs_f64() * 1e3
        );
        i32::from(!failures.is_empty())
    }

    pub(super) fn report(
        &self,
        outline: &outline::Outline,
        started: Instant,
        _built: Duration,
        filters: &[String],
    ) -> Result<i32> {
        #[cfg(feature = "diff-harness")]
        if self.kind != BatchKind::Tests {
            return self.report_cases(outline, started, _built);
        }
        self.report_tests(outline, started, filters)
    }

    #[cfg(feature = "diff-harness")]
    fn report_cases(
        &self,
        outline: &outline::Outline,
        started: Instant,
        built: Duration,
    ) -> Result<i32> {
        let (mut failures, runnable) = self.sort(Some(outline));
        let statuses = self.image.run_cases(&runnable, &CASE_INPUTS)?;
        for ((_, name), status) in runnable
            .iter()
            .zip(statuses)
            .filter(|(_, status)| *status != 0)
        {
            failures
                .entry(format!("crashed with status {status}"))
                .or_default()
                .push((*name).to_owned());
        }
        print_failures(&failures);
        let ms = |at: Duration| at.as_secs_f64() * 1e3;
        eprintln!(
            "cases {} found, {} ran, {} functions reached, built in {:.1} ms, total {:.1} ms",
            self.labels.len(),
            runnable.len(),
            self.reached.list.len(),
            ms(built),
            ms(started.elapsed())
        );
        Ok(if failures.is_empty() { 0 } else { 101 })
    }

    fn report_tests(
        &self,
        outline: &outline::Outline,
        started: Instant,
        filters: &[String],
    ) -> Result<i32> {
        let sel = crate::selection();
        let args = parse_test_args(filters);
        let ignored_only = sel.ignored_only || args.ignored_only;
        let include_ignored = sel.include_ignored || args.include_ignored;
        let matches = |name: &str, pat: &str| {
            if args.exact {
                name == pat
            } else {
                name.contains(pat)
            }
        };
        let mut to_run = Vec::new();
        let mut errs: Vec<String> = Vec::new();
        let mut raw: Vec<(
            &str,
            &TestItem,
            Option<Result<(usize, Option<usize>), usize>>,
        )> = Vec::new();
        for ((&root, name), (meta, start)) in self.roots.iter().zip(&self.labels).zip(&self.tests) {
            if (!args.filters.is_empty() && !args.filters.iter().any(|f| matches(name, f)))
                || args.skips.iter().any(|s| matches(name, s))
            {
                continue;
            }
            if ignored_only && !meta.ignored {
                continue;
            }
            let skip = meta.ignored && !ignored_only && !include_ignored;
            let check_roots: Vec<usize> =
                (*start).into_iter().chain(std::iter::once(root)).collect();
            let outcome = if skip {
                None
            } else if let Some((at, err)) = self
                .reached
                .closure(&check_roots)
                .iter()
                .rev()
                .find_map(|&at| {
                    self.reached.list[at]
                        .1
                        .as_ref()
                        .err()
                        .or_else(|| self.broken.get(&at))
                        .map(|e| (at, e))
                })
            {
                let idx = errs.len();
                errs.push(format!(
                    "{}{err}",
                    item_loc(outline, self.reached.list[at].0.0)
                ));
                Some(Err(idx))
            } else {
                if meta.expected.is_none() {
                    to_run.push((root, *start, name.as_str()));
                }
                Some(Ok((root, *start)))
            };
            raw.push((name.as_str(), meta, outcome));
        }
        let mut ran = self.image.run_tests(&to_run)?.into_iter();
        let mut entries: Vec<(&str, &TestItem, Option<Result<(i32, Option<String>), &str>>)> =
            Vec::with_capacity(raw.len());
        for (name, meta, outcome) in raw {
            let resolved = match outcome {
                None => None,
                Some(Err(idx)) => Some(Err(errs[idx].as_str())),
                Some(Ok((root, start))) if meta.expected.is_some() => {
                    let (status, stderr) = self.image.run_test_captured(root, start, name)?;
                    Some(Ok((status, Some(stderr))))
                }
                Some(Ok((root, start))) => match ran.next() {
                    Some(status) if status != 0 && !meta.should_panic => {
                        let (cap_status, stderr) =
                            self.image.run_test_captured(root, start, name)?;
                        Some(Ok((
                            if cap_status != 0 { cap_status } else { status },
                            Some(stderr),
                        )))
                    }
                    other => other.map(|s| Ok((s, None))),
                },
            };
            entries.push((name, meta, resolved));
        }
        Ok(print_test_report(&entries, started.elapsed()))
    }
}

fn item_loc(outline: &outline::Outline, item: ItemRef) -> String {
    let outer = item.body_owner();
    let file = &outline.crates[outer.krate as usize].files[outer.file as usize];
    let Some(entry) = file.items.items.get(outer.item as usize) else {
        return format!("{} ", file.source.path.display());
    };
    let line = file.source.line(entry.start);
    format!("{}:{line} ", file.source.path.display())
}
