use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use base_db::salsa::{Cancelled, Database};
use base_db::{Crate, EditionedFileId, SourceDatabase};
use hir_def::item_tree::file_item_tree;
use ide_db::RootDatabase;
use vfs::FileId;

fn closure(db: &RootDatabase, roots: &[Crate]) -> Vec<Crate> {
    let mut seen = BTreeSet::new();
    let mut stack = roots.to_vec();
    while let Some(krate) = stack.pop() {
        if krate.data(db).target_kind.is_proc_macro() || !seen.insert(krate) {
            continue;
        }
        stack.extend(krate.data(db).dependencies.iter().map(|dep| dep.crate_id));
    }
    seen.into_iter().collect()
}

fn files_of(db: &RootDatabase, krate: Crate, roots: &BTreeSet<FileId>) -> Vec<(FileId, Crate)> {
    let root = krate.data(db).root_file_id;
    let source_root = db.source_root(db.file_source_root(root).source_root_id(db)).source_root(db);
    let Some(dir) = source_root.path_for_file(&root).and_then(|path| path.parent()) else {
        return vec![(root, krate)];
    };
    let module = |path: &vfs::VfsPath| path.starts_with(&dir) && path.name_and_extension().is_some_and(|(_, ext)| ext == Some("rs"));
    let inside = |file: &FileId| (*file == root || !roots.contains(file)) && source_root.path_for_file(file).is_some_and(module);
    source_root.iter().filter(inside).map(|file| (file, krate)).collect()
}

/// Parses every file of the crates `roots` reach, then resolves their names in dependency order, on `workers` threads.
pub fn warm(db: &RootDatabase, roots: &[Crate], workers: usize) {
    let crates = closure(db, roots);
    let roots: BTreeSet<FileId> = base_db::all_crates(db).iter().map(|krate| krate.data(db).root_file_id).collect();
    let files: Vec<(FileId, Crate)> = crates.iter().flat_map(|krate| files_of(db, *krate, &roots)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let db = db.clone();
            let (files, next) = (&files, &next);
            scope.spawn(move || {
                while let Some((file, krate)) = files.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let parsed = Cancelled::catch(|| {
                        let edition = krate.data(&db).edition;
                        file_item_tree(&db, EditionedFileId::new(&db, *file, edition).into(), *krate);
                    });
                    if parsed.is_err() {
                        return;
                    }
                }
            });
        }
    });
    ide_db::prime_caches::prime_def_maps(db, &crates, workers);
}

/// Warm work on a database snapshot in the background, stopped before the database changes or the process forks.
pub struct Warmer(Option<std::thread::JoinHandle<()>>);

impl Warmer {
    /// Starts warming the crates `roots` reach on a snapshot of the engine's database, so their proc-macro hosts compile while scripts run.
    pub fn start(roots: Vec<Crate>, workers: usize) -> Self {
        let db = crate::patch::snapshot();
        let spawned = std::thread::Builder::new().name("warm".to_owned()).spawn(move || {
            let _ = Cancelled::catch(|| warm(&db, &roots, workers));
        });
        Warmer(spawned.ok())
    }

    /// Cancels the work in flight, keeping what finished, and waits for every warm thread to end.
    pub fn stop(mut self) {
        crate::patch::with_database(|db| db.trigger_cancellation());
        if let Some(thread) = self.0.take() {
            let _ = thread.join();
        }
    }
}
