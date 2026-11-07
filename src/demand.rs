use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use hir_def::nameres::crate_def_map;
use hir_def::{FunctionId, HasModule, Lookup};
use hir_expand::proc_macro::ProcMacroKind;
use hir_expand::{HirFileId, MacroDefKind};
use hir_ty::db::HirDatabase;
use ide_db::RootDatabase;

use crate::patch;

const TOP: usize = 40;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Counter {
    DefMaps,
    Expansions,
    Parses,
    ItemTrees,
}

impl Counter {
    fn of(name: &str) -> Option<Self> {
        match name {
            "crate_local_def_map" => Some(Self::DefMaps),
            "MacroCallId::parse_macro_expansion_" => Some(Self::Expansions),
            "EditionedFileId::parse_" => Some(Self::Parses),
            "file_item_tree_query" => Some(Self::ItemTrees),
            _ => None,
        }
    }
}

#[derive(Default)]
struct Needed {
    crates: HashSet<base_db::Crate>,
    files: HashSet<HirFileId>,
    expansions: HashSet<HirFileId>,
}

impl Needed {
    fn add(&mut self, db: &dyn HirDatabase, function: FunctionId) {
        self.crates.insert(function.krate(db));
        let file = function.lookup(db).id.file_id;
        self.files.insert(file);
        if file.is_macro() {
            self.expansions.insert(file);
        }
    }

    fn of(&self, counter: Counter) -> usize {
        match counter {
            Counter::DefMaps => self.crates.len(),
            Counter::Expansions => self.expansions.len(),
            Counter::Parses => self.files.len() - self.expansions.len(),
            Counter::ItemTrees => self.files.len(),
        }
    }
}

pub fn run(db: RootDatabase, entry: &str) -> Result<()> {
    patch::start(db, false, false)?;
    let started = Instant::now();
    let (table, slot) = patch::launch(entry)?;
    let value = crate::backend::call(table, slot, 20);
    println!("first call: total_us={} entry(20)={value}", started.elapsed().as_micros());
    patch::with_engine(report);
    Ok(())
}

fn report(db: &dyn HirDatabase, backend: &mut crate::backend::Backend) {
    let mut needed = Needed::default();
    let functions = backend.functions();
    for function in &functions {
        needed.add(db, *function);
    }
    println!("reached functions={} crates_total={}", functions.len(), base_db::all_crates(db).len());
    queries(db, &needed);
    impls(db);
    origins(db);
    callers();
}

fn queries(db: &dyn HirDatabase, needed: &Needed) {
    let mut totals: HashMap<String, (u64, u64)> = HashMap::new();
    for (key, runs) in base_db::demand::runs() {
        let name = db.ingredient_debug_name(key.ingredient_index()).into_owned();
        let total = totals.entry(name).or_default();
        total.0 += runs;
        total.1 += 1;
    }
    let mut rows: Vec<_> = totals.into_iter().collect();
    rows.sort_by_key(|row| std::cmp::Reverse(row.1.0));
    for (name, (runs, keys)) in rows.iter().take(TOP) {
        println!("query {name} runs={runs} keys={keys}");
    }
    for (name, (runs, keys)) in &rows {
        if let Some(counter) = Counter::of(name) {
            let need = needed.of(counter);
            let share = need as f64 * 100.0 / (*keys).max(1) as f64;
            println!("counter {counter:?} runs={runs} keys={keys} needed_at_least={need} share={share:.1}%");
        }
    }
}

fn impls(db: &dyn HirDatabase) {
    let used = base_db::demand::used_impls();
    let crates: HashSet<base_db::Crate> = used.iter().copied().collect();
    let held = |krate: &base_db::Crate| -> usize {
        crate_def_map(db, *krate).modules().map(|(_, module)| module.scope.impls().len()).sum()
    };
    let total: usize = crates.iter().map(held).sum();
    println!("impls used={} used_crates={} impls_in_used_crates={total}", used.len(), crates.len());
    let mut names: Vec<_> = crates.iter().map(|krate| crate_name(db, *krate)).collect();
    names.sort();
    println!("impl_crates {}", names.join(" "));
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum Origin {
    Source,
    MacroRules,
    BuiltinDerive,
    ProcDerive,
    ProcAttr,
    ProcBang,
    Builtin,
}

fn origin(db: &dyn HirDatabase, file: HirFileId) -> Origin {
    let Some(call) = file.macro_file() else {
        return Origin::Source;
    };
    match call.loc(db).def.kind {
        MacroDefKind::Declarative(..) => Origin::MacroRules,
        MacroDefKind::BuiltInDerive(..) => Origin::BuiltinDerive,
        MacroDefKind::ProcMacro(_, _, ProcMacroKind::CustomDerive) => Origin::ProcDerive,
        MacroDefKind::ProcMacro(_, _, ProcMacroKind::Attr) => Origin::ProcAttr,
        MacroDefKind::ProcMacro(_, _, ProcMacroKind::Bang) => Origin::ProcBang,
        _ => Origin::Builtin,
    }
}

fn origins(db: &dyn HirDatabase) {
    let mut impls: HashMap<Origin, usize> = HashMap::new();
    let mut crates: HashMap<Origin, HashSet<base_db::Crate>> = HashMap::new();
    for krate in base_db::all_crates(db).iter().copied() {
        for (_, module) in crate_def_map(db, krate).modules() {
            for id in module.scope.impls() {
                let found = origin(db, id.lookup(db).id.file_id);
                *impls.entry(found).or_default() += 1;
                crates.entry(found).or_default().insert(krate);
            }
        }
    }
    let mut rows: Vec<_> = impls.into_iter().collect();
    rows.sort();
    for (found, count) in rows {
        println!("impl_origin {found:?} impls={count} crates={}", crates[&found].len());
    }
}

fn crate_name(db: &dyn HirDatabase, krate: base_db::Crate) -> String {
    krate.extra_data(db).display_name.as_ref().map_or_else(|| "?".to_owned(), ToString::to_string)
}

fn callers() {
    let mut rows = base_db::demand::nodes();
    rows.sort_by_key(|row| std::cmp::Reverse(row.1));
    let total: u64 = rows.iter().map(|row| row.1).sum();
    println!("to_node total={total} callers={}", rows.len());
    for (caller, count) in rows.iter().take(TOP) {
        println!("to_node {count} {}:{}", caller.file(), caller.line());
    }
}
