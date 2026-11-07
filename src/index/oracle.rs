use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use base_db::{EditionedFileId, SourceDatabase, relevant_crates};
use hir_def::item_tree::{TreeKind, file_item_tree};
use hir_ty::attach_db;
use hir_ty::db::HirDatabase;
use vfs::Vfs;

use super::{Entry, Kind, scan};

const SHOWN: usize = 40;

fn kind_of(kind: TreeKind) -> Kind {
    match kind {
        TreeKind::Const => Kind::Const,
        TreeKind::Enum => Kind::Enum,
        TreeKind::ExternBlock => Kind::ExternBlock,
        TreeKind::ExternCrate => Kind::ExternCrate,
        TreeKind::Function => Kind::Function,
        TreeKind::Impl => Kind::Impl,
        TreeKind::Macro2 => Kind::Macro2,
        TreeKind::MacroCall => Kind::MacroCall,
        TreeKind::MacroRules => Kind::MacroRules,
        TreeKind::Mod => Kind::Mod,
        TreeKind::Static => Kind::Static,
        TreeKind::Struct => Kind::Struct,
        TreeKind::Trait => Kind::Trait,
        TreeKind::TypeAlias => Kind::TypeAlias,
        TreeKind::Union => Kind::Union,
        TreeKind::Use => Kind::Use,
    }
}

fn counted(entries: Vec<Entry>) -> BTreeMap<Entry, i64> {
    let mut counts = BTreeMap::new();
    for entry in entries {
        *counts.entry(entry).or_insert(0) += 1;
    }
    counts
}

fn differences(ours: Vec<Entry>, theirs: Vec<Entry>) -> Vec<(Entry, i64)> {
    let mut counts = counted(ours);
    for (entry, count) in counted(theirs) {
        *counts.entry(entry).or_insert(0) -= count;
    }
    counts.into_iter().filter(|(_, count)| *count != 0).collect()
}

#[derive(Default)]
struct Tally {
    files: usize,
    bytes: usize,
    items: usize,
    wrong_files: usize,
    wrong_items: usize,
    ours: Duration,
    theirs: Duration,
}

pub fn run(db: &dyn HirDatabase, vfs: &Vfs) {
    let mut tally = Tally::default();
    attach_db(db, || {
        for (file, path) in vfs.iter() {
            if !path.name_and_extension().is_some_and(|(_, extension)| extension == Some("rs")) {
                continue;
            }
            let Some(krate) = relevant_crates(db, file).first().copied() else { continue };
            let text = db.file_text(file).text(db);
            let started = Instant::now();
            let ours = scan(text);
            tally.ours += started.elapsed();
            let id = EditionedFileId::new(db, file, krate.data(db).edition);
            let started = Instant::now();
            let tree = file_item_tree(db, id.into(), krate);
            tally.theirs += started.elapsed();
            let theirs = tree
                .inventory()
                .into_iter()
                .map(|item| Entry { path: item.path, kind: kind_of(item.kind), name: item.name })
                .collect();
            tally.files += 1;
            tally.bytes += text.len();
            tally.items += ours.len();
            let wrong = differences(ours, theirs);
            if !wrong.is_empty() {
                if tally.wrong_files < SHOWN {
                    println!("{path}: {wrong:?}");
                }
                tally.wrong_files += 1;
                tally.wrong_items += wrong.len();
            }
        }
    });
    println!(
        "index files={} bytes={} items={} wrong_files={} wrong_items={} ours_ms={} theirs_ms={}",
        tally.files,
        tally.bytes,
        tally.items,
        tally.wrong_files,
        tally.wrong_items,
        tally.ours.as_millis(),
        tally.theirs.as_millis(),
    );
}
