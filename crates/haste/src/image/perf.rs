//! The code map that `--perf-map` asks for: one line per function this process places code for, so a profile of the program can name the samples that land in that code.

use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::sema::Context;
use crate::sema::reach::Reached;

/// The directory under Haste's home that holds code maps, one per compiling process by its id.
const DIR: &str = "perf";
const EXTENSION: &str = "map";

static WANTED: AtomicBool = AtomicBool::new(false);
static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

/// Where code came from: Haste's own backend, an adapter to rustc's code, or rustc's code itself.
#[derive(Clone, Copy, Debug)]
pub enum Origin {
    Haste,
    Adapter,
    Rustc,
}

/// Makes this process write a code map.
pub fn enable() {
    WANTED.store(true, Ordering::Relaxed);
}

/// Notes that reached function `index` runs the code at `address`, which came from `origin`, with why prepped code does not run it when that was decided.
pub(super) fn note(cx: &mut Context<'_>, reached: &Reached, (index, address): (usize, usize), origin: Origin) {
    if !WANTED.load(Ordering::Relaxed) {
        return;
    }
    let Some(file) = FILE.get_or_init(open) else { return };
    let ((item, substs), _) = reached.list[index];
    let args: Vec<String> = cx.types.substs(substs).to_vec().into_iter().map(|ty| cx.type_text(ty, &|ty| ty)).collect();
    let module = cx.module_of(item).map_or_else(|_| cx.outline.roots[item.krate as usize].name.clone(), |at| cx.outline.module_path(at));
    let why = crate::prep::why(reached, index).unwrap_or_default();
    let line = format!("{address:x}\t{index}\t{origin:?}\t{why}\t{module}::{}<{}>\n", cx.name(item), args.join(","));
    if let Ok(mut file) = file.lock()
        && let Err(error) = file.write_all(line.as_bytes())
    {
        eprintln!("code map: {error}");
    }
}

fn open() -> Option<Mutex<std::fs::File>> {
    let dir = crate::graph::store::home().join(DIR);
    let path = dir.join(format!("{}.{EXTENSION}", std::process::id()));
    match std::fs::create_dir_all(&dir).and_then(|()| std::fs::File::create(&path)) {
        Ok(file) => Some(Mutex::new(file)),
        Err(error) => {
            eprintln!("code map {}: {error}", path.display());
            None
        }
    }
}
