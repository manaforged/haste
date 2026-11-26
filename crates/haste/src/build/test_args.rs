use rustc_hash::FxHashMap as HashMap;

use crate::cli::TestItem;
use crate::graph::Workspace;
use crate::names::ItemRef;
use crate::outline::{self, Outline, item::ItemKind};
use crate::sema::{
    self,
    reach::{Instance, Reached},
};

pub(super) struct TestArgs<'a> {
    pub(super) filters: Vec<&'a str>,
    pub(super) skips: Vec<&'a str>,
    pub(super) exact: bool,
    pub(super) list: bool,
    pub(super) nocapture: bool,
    pub(super) show_output: bool,
    pub(super) ignored_only: bool,
    pub(super) include_ignored: bool,
}

pub(super) fn parse_test_args(args: &[String]) -> TestArgs<'_> {
    let mut parsed = TestArgs {
        filters: Vec::new(),
        skips: Vec::new(),
        exact: false,
        list: false,
        nocapture: false,
        show_output: false,
        ignored_only: false,
        include_ignored: false,
    };
    let mut iter = args.iter().map(String::as_str);
    while let Some(arg) = iter.next() {
        if arg == "--" {
            continue;
        } else if arg == "--exact" {
            parsed.exact = true;
        } else if arg == "--list" {
            parsed.list = true;
        } else if arg == "--nocapture" {
            parsed.nocapture = true;
        } else if arg == "--show-output" {
            parsed.show_output = true;
        } else if arg == "--ignored" {
            parsed.ignored_only = true;
        } else if arg == "--include-ignored" {
            parsed.include_ignored = true;
        } else if let Some(val) = arg.strip_prefix("--skip=") {
            parsed.skips.push(val);
        } else if arg == "--skip" {
            if let Some(val) = iter.next() {
                parsed.skips.push(val);
            }
        } else if matches!(
            arg,
            "--test-threads" | "--color" | "--format" | "--logfile" | "--shuffle-seed" | "-Z"
        ) {
            let _ = iter.next();
        } else if !arg.starts_with('-') {
            parsed.filters.push(arg);
        }
    }
    parsed
}

impl TestArgs<'_> {
    pub(super) fn matches_name(&self, name: &str) -> bool {
        let matches = |pat: &str| {
            if self.exact {
                name == pat
            } else {
                name.contains(pat)
            }
        };
        (self.filters.is_empty() || self.filters.iter().any(|f| matches(f)))
            && !self.skips.iter().any(|s| matches(s))
    }
}

pub(super) fn select_test_roots(
    cx: &mut sema::Context<'_>,
    items: &[ItemRef],
    filters: &[String],
) -> (Vec<ItemRef>, Vec<Option<Instance>>, Vec<Instance>) {
    let args = parse_test_args(filters);
    let has_filter = !args.filters.is_empty() || !args.skips.is_empty();
    let mut valid = Vec::new();
    let mut starts = Vec::new();
    let mut roots = Vec::new();
    for &item in items {
        if has_filter && !args.matches_name(&test_label(cx, item)) {
            continue;
        }
        let Some(unit) = sema::infer::test_sig(cx, item) else {
            continue;
        };
        let start = if unit {
            None
        } else {
            sema::reach::main_roots(cx, item).ok().map(|[s, _]| s)
        };
        if let Some(s) = start {
            roots.push(s);
        }
        roots.push((item, sema::substs::Substs::EMPTY));
        starts.push(start);
        valid.push(item);
    }
    (valid, starts, roots)
}

pub(super) fn test_label(cx: &mut sema::Context<'_>, item: ItemRef) -> String {
    let fn_name = cx.name(item).to_owned();
    let Ok(mod_ref) = cx.module_of(item) else {
        return fn_name;
    };
    let modules = &cx.outline.crates[mod_ref.krate as usize].modules;
    let mut parts = Vec::new();
    let mut current = Some(mod_ref.module as usize);
    while let Some(module) = current.filter(|&m| m != 0) {
        parts.push(modules[module].name.as_str());
        current = modules[module].parent;
    }
    if parts.is_empty() {
        return fn_name;
    }
    parts.reverse();
    parts.push(&fn_name);
    parts.join("::")
}

pub(super) fn map_tests(
    roots: &[usize],
    reached: &Reached,
    starts: &[Option<usize>],
    items: Vec<TestItem>,
) -> Vec<(TestItem, Option<usize>)> {
    let by_item: HashMap<ItemRef, TestItem> = items.into_iter().map(|t| (t.item, t)).collect();
    roots
        .iter()
        .zip(starts)
        .map(|(&idx, &start)| (by_item[&reached.list[idx].0.0].clone(), start))
        .collect()
}

pub(super) fn check_fn_items(
    workspace: &Workspace,
    outline: &Outline,
    target: Option<usize>,
) -> Vec<ItemRef> {
    let keep_krate = |k: usize| {
        target.is_none() || target == Some(k) || outline.roots[k].role == outline::Role::Lib
    };
    let mut out: Vec<ItemRef> = crate::cli::lib_fn_items(workspace, outline)
        .into_iter()
        .filter(|item| keep_krate(item.krate as usize))
        .collect();
    for krate in outline.members(workspace) {
        if !keep_krate(krate) {
            continue;
        }
        let tree = &outline.crates[krate];
        for (file_at, file) in tree.files.iter().enumerate() {
            let items = &file.items.items;
            for (at, item) in items.iter().enumerate() {
                if item.kind != ItemKind::Fn || item.body.is_none() {
                    continue;
                }
                if !item.parent.is_some_and(|p| {
                    matches!(items[p as usize].kind, ItemKind::Impl | ItemKind::Trait)
                }) {
                    continue;
                }
                let mut ok = true;
                let mut cur = Some(at);
                while let Some(idx) = cur {
                    let cur_item = &items[idx];
                    if !tree.env.enabled(&cur_item.attrs)
                        || (idx != at
                            && !matches!(
                                cur_item.kind,
                                ItemKind::Mod | ItemKind::Impl | ItemKind::Trait
                            ))
                    {
                        ok = false;
                        break;
                    }
                    cur = cur_item.parent.map(|p| p as usize);
                }
                if ok {
                    out.push(ItemRef {
                        krate: krate as u32,
                        file: file_at as u32,
                        item: at as u32,
                    });
                }
            }
        }
    }
    out
}
