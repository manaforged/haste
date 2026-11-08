use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};

use anyhow::Result;
use hir_expand::proc_macro::ProcMacroKind;
use proc_macro_api::legacy_protocol::msg::{Request, Response};
use serde_json::Value;

use crate::script::Scripts;

const FILE: &str = "haste-generated/expansions";
const MACROS_FILE: &str = "haste-generated/macros";

/// The macros one proc-macro crate declares, as name and kind.
pub type Declared = Vec<(String, ProcMacroKind)>;

struct Replay {
    mode: Scripts,
    root: PathBuf,
    saved: HashMap<u64, String>,
    recorded: Mutex<HashMap<u64, String>>,
    macros: HashMap<String, Declared>,
    declared: Mutex<BTreeMap<String, Declared>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

static REPLAY: OnceLock<Replay> = OnceLock::new();

fn parse(text: &str) -> HashMap<u64, String> {
    let entries = text.lines().filter_map(|line| line.split_once(' '));
    entries.filter_map(|(key, reply)| Some((u64::from_str_radix(key, 16).ok()?, reply.to_owned()))).collect()
}

const KINDS: [(ProcMacroKind, &str); 3] = [(ProcMacroKind::CustomDerive, "derive"), (ProcMacroKind::Bang, "bang"), (ProcMacroKind::Attr, "attr")];

fn kind_text(kind: ProcMacroKind) -> &'static str {
    KINDS.iter().find(|(known, _)| *known == kind).map_or("derive", |(_, text)| text)
}

fn parse_macros(text: &str) -> HashMap<String, Declared> {
    let mut macros: HashMap<String, Declared> = HashMap::new();
    for line in text.lines() {
        let mut fields = line.split('\t');
        let (Some(krate), Some(kind), Some(name)) = (fields.next(), fields.next(), fields.next()) else {
            continue;
        };
        let entry = macros.entry(krate.to_owned()).or_default();
        if let Some((kind, _)) = KINDS.iter().find(|(_, text)| *text == kind) {
            entry.push((name.to_owned(), *kind));
        }
    }
    macros
}

fn read(root: &Path, file: &str) -> String {
    std::fs::read_to_string(root.join(file)).unwrap_or_default()
}

/// Opens the saved expansions of the project at `root` for replay or for recording, as `mode` says.
pub fn open(root: &Path, mode: Scripts) {
    let (saved, macros) = match mode {
        Scripts::Saved => (parse(&read(root, FILE)), parse_macros(&read(root, MACROS_FILE))),
        Scripts::Live | Scripts::Generate => (HashMap::new(), HashMap::new()),
    };
    let replay = Replay {
        mode,
        root: root.to_owned(),
        saved,
        recorded: Mutex::default(),
        macros,
        declared: Mutex::default(),
        hits: AtomicUsize::new(0),
        misses: AtomicUsize::new(0),
    };
    let _ = REPLAY.set(replay);
}

/// The saved macros of the proc-macro crate labelled `krate`, when replaying.
pub(super) fn declared(krate: &str) -> Option<Declared> {
    REPLAY.get().filter(|replay| replay.mode == Scripts::Saved)?.macros.get(krate).cloned()
}

/// Records the macros of the proc-macro crate labelled `krate` while generating.
pub(super) fn keep_declared(krate: &str, macros: &Declared) {
    if let Some(replay) = REPLAY.get().filter(|replay| replay.mode == Scripts::Generate) {
        replay.declared.lock().unwrap_or_else(PoisonError::into_inner).insert(krate.to_owned(), macros.clone());
    }
}

const SPAN_WORDS: usize = 5;
const FILE_WORD: usize = 0;
const CONTEXT_WORD: usize = 4;
const TABLE: &str = "span_data_table";
const SCRATCH_MARK: &str = "${SCRATCH}";

/// One request in a form that is the same in every process: interned file and context ids become ordinals.
pub(super) struct Canon {
    key: u64,
    files: Vec<u64>,
    contexts: Vec<u64>,
}

fn table_of(value: &mut Value) -> Option<&mut Vec<Value>> {
    match value {
        Value::Object(map) => {
            if map.contains_key(TABLE) {
                return map.get_mut(TABLE)?.as_array_mut();
            }
            map.values_mut().find_map(table_of)
        }
        _ => None,
    }
}

fn to_ordinal(word: &mut Value, seen: &mut Vec<u64>, grow: bool) -> Option<()> {
    let id = word.as_u64()?;
    let position = match seen.iter().position(|known| *known == id) {
        Some(position) => position,
        None if grow => {
            seen.push(id);
            seen.len() - 1
        }
        None => return None,
    };
    *word = Value::from(position);
    Some(())
}

fn to_ordinals(value: &mut Value, files: &mut Vec<u64>, contexts: &mut Vec<u64>, grow: bool) -> Option<()> {
    for span in table_of(value)?.chunks_mut(SPAN_WORDS) {
        to_ordinal(span.get_mut(FILE_WORD)?, files, grow)?;
        to_ordinal(span.get_mut(CONTEXT_WORD)?, contexts, grow)?;
    }
    Some(())
}

fn from_ordinals(value: &mut Value, canon: &Canon) -> Option<()> {
    for span in table_of(value)?.chunks_mut(SPAN_WORDS) {
        let file = canon.files.get(usize::try_from(span.get(FILE_WORD)?.as_u64()?).ok()?)?;
        let context = canon.contexts.get(usize::try_from(span.get(CONTEXT_WORD)?.as_u64()?).ok()?)?;
        span[FILE_WORD] = Value::from(*file);
        span[CONTEXT_WORD] = Value::from(*context);
    }
    Some(())
}

/// The canonical form of one expansion request, keyed by a hash of its text.
pub(super) fn canon(request: &Request) -> Option<Canon> {
    let mut value = serde_json::to_value(request).ok()?;
    let (mut files, mut contexts) = (Vec::new(), Vec::new());
    to_ordinals(&mut value, &mut files, &mut contexts, true)?;
    let text = value.to_string().replace(project_model::script_scratch().as_str(), SCRATCH_MARK);
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    Some(Canon { key: hasher.finish(), files, contexts })
}

/// The saved reply to `canon`'s request, with its spans moved to this process's ids.
pub(super) fn find(canon: &Canon) -> Option<Response> {
    let replay = REPLAY.get().filter(|replay| replay.mode == Scripts::Saved)?;
    let found = replay.saved.get(&canon.key).and_then(|text| {
        let mut value: Value = serde_json::from_str(text).ok()?;
        from_ordinals(&mut value, canon)?;
        serde_json::from_value(value).ok()
    });
    let counter = if found.is_some() { &replay.hits } else { &replay.misses };
    counter.fetch_add(1, Ordering::Relaxed);
    found
}

/// Records the reply to `canon`'s request while generating, when every reply span comes from the request.
pub(super) fn keep(canon: &Canon, reply: &Response) {
    let Some(replay) = REPLAY.get().filter(|replay| replay.mode == Scripts::Generate) else {
        return;
    };
    let Ok(mut value) = serde_json::to_value(reply) else {
        return;
    };
    let (mut files, mut contexts) = (canon.files.clone(), canon.contexts.clone());
    if to_ordinals(&mut value, &mut files, &mut contexts, false).is_some() {
        replay.recorded.lock().unwrap_or_else(PoisonError::into_inner).insert(canon.key, value.to_string());
    }
}

/// Writes every recorded expansion, replacing the saved file.
pub fn save() -> Result<usize> {
    let Some(replay) = REPLAY.get() else {
        return Ok(0);
    };
    let recorded = replay.recorded.lock().unwrap_or_else(PoisonError::into_inner);
    let mut keys: Vec<&u64> = recorded.keys().collect();
    keys.sort();
    let text: String = keys.into_iter().map(|key| format!("{key:016x} {}\n", recorded[key])).collect();
    let declared = replay.declared.lock().unwrap_or_else(PoisonError::into_inner);
    let lines = declared.iter().flat_map(|(krate, macros)| macros.iter().map(move |(name, kind)| format!("{krate}\t{}\t{name}\n", kind_text(*kind))));
    let path = replay.root.join(FILE);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)?;
    std::fs::write(replay.root.join(MACROS_FILE), lines.collect::<String>())?;
    Ok(recorded.len())
}

/// How many expansions were replayed and how many ran live.
pub fn counts() -> (usize, usize) {
    REPLAY.get().map_or((0, 0), |replay| (replay.hits.load(Ordering::Relaxed), replay.misses.load(Ordering::Relaxed)))
}
