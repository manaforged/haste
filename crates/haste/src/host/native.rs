//! The native libraries that build scripts link: their static archives linked whole into one shared library, kept in the store under a key of the archives, and loaded with the dynamic libraries they name, so foreign declarations find their symbols, hidden ones through a table the library exports.

use std::ffi::{CStr, CString, c_void};
use std::hash::BuildHasher as _;
use std::io::Read as _;
use std::os::unix::ffi::OsStrExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::RwLock;

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::graph::script::{LinkKind, tool};
use crate::graph::store;
use crate::outline::Outline;

const NAMES: &str = "haste_native_names";
const ADDRESSES: &str = "haste_native_addresses";
const COUNT: &str = "haste_native_count";
const ARCHIVE_MAGIC: &[u8; 8] = b"!<arch>\n";
const MEMBER_HEADER: usize = 60;
const INDEX_NAMES: [(&[u8], usize); 2] = [(b"/ ", 4), (b"/SYM64/", 8)];

static LOADED: RwLock<Loaded> = RwLock::new(Loaded { libraries: Vec::new(), symbols: None });

/// What this process loaded: each library by archive path or name, and the name and address of each symbol of a loaded archive.
struct Loaded {
    libraries: Vec<String>,
    symbols: Option<HashMap<Box<str>, usize>>,
}

/// The libraries that build scripts ask to link: static archives, dynamic libraries by kind and name, and the directories to search.
#[derive(Default, Hash)]
struct Libraries {
    archives: Vec<PathBuf>,
    dynamic: Vec<(LinkKind, String)>,
    search: Vec<PathBuf>,
}

impl Libraries {
    /// The path of each archive and the name of each dynamic library.
    fn names(&self) -> Vec<String> {
        self.archives.iter().map(|archive| archive.display().to_string()).chain(self.dynamic.iter().map(|(_, name)| name.clone())).collect()
    }
}

/// Loads the libraries that the build scripts of the crates in `outline` link and that this process has not loaded yet.
pub fn load(outline: &Outline) -> Result<()> {
    let mut libraries = libraries(outline)?;
    let mut loaded = LOADED.write().unwrap_or_else(std::sync::PoisonError::into_inner);
    libraries.archives.retain(|archive| !loaded.libraries.contains(&archive.display().to_string()));
    libraries.dynamic.retain(|(_, name)| !loaded.libraries.contains(name));
    if libraries.archives.is_empty() && libraries.dynamic.is_empty() {
        return Ok(());
    }
    let symbols = open_libraries(&libraries).with_context(|| format!("loading the native libraries {}", libraries.names().join(", ")))?;
    loaded.symbols.get_or_insert_default().extend(symbols);
    loaded.libraries.extend(libraries.names());
    Ok(())
}

/// The address of `name` in a loaded static archive.
pub fn symbol(name: &str) -> Option<usize> {
    LOADED.read().ok()?.symbols.as_ref()?.get(name).copied()
}

/// What the build scripts that ran for the crates of `outline` link, each archive found in their search directories.
fn libraries(outline: &Outline) -> Result<Libraries> {
    let mut wanted = Vec::new();
    let mut libraries = Libraries::default();
    for script in outline.crates.iter().filter_map(|krate| krate.env.script.as_ref()) {
        libraries.search.extend(script.search_dirs().map(PathBuf::from));
        wanted.extend(script.link_libs().map(|(kind, name)| (kind, name.to_owned())));
    }
    for (kind, name) in wanted {
        match kind {
            LinkKind::Static => libraries.archives.push(archive(&libraries.search, &name)?),
            LinkKind::Dynamic | LinkKind::Framework => libraries.dynamic.push((kind, name)),
        }
    }
    libraries.archives.sort();
    libraries.archives.dedup();
    libraries.dynamic.sort();
    libraries.dynamic.dedup();
    Ok(libraries)
}

fn archive(search: &[PathBuf], name: &str) -> Result<PathBuf> {
    let file = format!("lib{name}.a");
    search.iter().map(|dir| dir.join(&file)).find(|path| path.is_file()).with_context(|| format!("no {file} in the build scripts' search directories"))
}

/// Links `libraries` into the store unless it holds them, opens the result, and reads its table of static symbols.
fn open_libraries(libraries: &Libraries) -> Result<HashMap<Box<str>, usize>> {
    let mut stamps = Vec::new();
    libraries.archives.iter().for_each(|archive| store::walk(archive, &mut stamps));
    let key = rustc_hash::FxBuildHasher.hash_one((stamps, libraries, store::program()));
    let path = store::home().join("native").join(format!("{key:016x}.so"));
    if !path.is_file() {
        link(libraries, &path)?;
    }
    let handle = open(&path, libc::RTLD_LAZY)?;
    table(handle)
}

fn link(libraries: &Libraries, path: &Path) -> Result<()> {
    let dir = path.parent().unwrap_or(path);
    std::fs::create_dir_all(dir)?;
    let mut names = Vec::new();
    for archive in &libraries.archives {
        names.extend(index(archive).with_context(|| format!("reading the symbol index of {}", archive.display()))?);
    }
    names.retain(|name| is_identifier(name));
    names.sort();
    names.dedup();
    let source = path.with_extension("c");
    store::write_new(&source, &table_source(&names))?;
    let fresh = store::fresh(path);
    let mut command = Command::new(tool("CC"));
    command.args(["-shared", "-fPIC", "-fno-builtin", "-w", "-o"]).arg(&fresh).arg(&source);
    command.args(libraries.search.iter().map(|dir| format!("-L{}", dir.display())));
    command.args(whole(&libraries.archives)).args(needed(&libraries.dynamic));
    let ran = command.stdin(Stdio::null()).output().with_context(|| format!("running {command:?}"))?;
    if !ran.status.success() {
        bail!("linking native libraries failed: {}", String::from_utf8_lossy(&ran.stderr));
    }
    std::fs::rename(&fresh, path)?;
    Ok(())
}

/// The linker arguments that take every member of each of `archives`, used or not, since compiled code asks for symbols only as it runs.
fn whole(archives: &[PathBuf]) -> Vec<String> {
    if cfg!(target_vendor = "apple") {
        let mut args: Vec<String> = archives.iter().map(|archive| format!("-Wl,-force_load,{}", archive.display())).collect();
        args.extend(["-undefined".to_owned(), "dynamic_lookup".to_owned()]);
        return args;
    }
    let mut args = vec!["-Wl,--whole-archive".to_owned()];
    args.extend(archives.iter().map(|archive| archive.display().to_string()));
    args.push("-Wl,--no-whole-archive".to_owned());
    args
}

/// The linker arguments that name each of `dynamic`, on ELF kept as needed though the linked code references none of their symbols, so loading the result loads them.
fn needed(dynamic: &[(LinkKind, String)]) -> Vec<String> {
    let mut args = Vec::new();
    for (kind, name) in dynamic {
        match kind {
            LinkKind::Framework => args.extend(["-framework".to_owned(), name.clone()]),
            LinkKind::Static | LinkKind::Dynamic => args.push(format!("-l{name}")),
        }
    }
    if cfg!(target_vendor = "apple") || args.is_empty() {
        return args;
    }
    args.insert(0, "-Wl,--push-state,--no-as-needed".to_owned());
    args.push("-Wl,--pop-state".to_owned());
    args
}

fn is_identifier(name: &str) -> bool {
    name.bytes().next().is_some_and(|first| !first.is_ascii_digit()) && name.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// A C source whose exported arrays hold the name and address of each of `names`, which may be hidden from the dynamic linker.
fn table_source(names: &[String]) -> String {
    let mut source: String = names.iter().map(|name| format!("extern char {name}[];\n")).collect();
    source.push_str(&format!("const char *const {NAMES}[] = {{\n"));
    source.extend(names.iter().map(|name| format!("\"{name}\",\n")));
    source.push_str(&format!("0}};\nvoid *const {ADDRESSES}[] = {{\n"));
    source.extend(names.iter().map(|name| format!("{name},\n")));
    source.push_str(&format!("0}};\nconst unsigned long {COUNT} = {};\n", names.len()));
    source
}

/// The names in the symbol index of the System V archive at `path`, or none for an archive without one.
fn index(path: &Path) -> Result<Vec<String>> {
    let mut file = std::fs::File::open(path)?;
    let mut head = [0u8; ARCHIVE_MAGIC.len() + MEMBER_HEADER];
    file.read_exact(&mut head)?;
    let (magic, header) = head.split_at(ARCHIVE_MAGIC.len());
    if magic != ARCHIVE_MAGIC {
        bail!("not an archive");
    }
    let Some(&(_, width)) = INDEX_NAMES.iter().find(|(name, _)| header.starts_with(name)) else { return Ok(Vec::new()) };
    let size: usize = std::str::from_utf8(&header[48..58])?.trim().parse()?;
    let mut body = vec![0u8; size];
    file.read_exact(&mut body)?;
    let count = body.get(..width).map(|bytes| bytes.iter().fold(0usize, |count, byte| count << 8 | usize::from(*byte))).context("a short symbol index")?;
    let strings = body.get(width * (count + 1)..).context("a short symbol index")?;
    Ok(strings.split(|byte| *byte == 0).filter(|name| !name.is_empty()).take(count).map(|name| String::from_utf8_lossy(name).into_owned()).collect())
}

/// Opens the library at `path` with its symbols global and its references bound by `mode`, `RTLD_LAZY` or `RTLD_NOW`, so it serves every foreign symbol it exports, and returns its handle.
pub fn open(path: &Path, mode: libc::c_int) -> Result<*mut c_void> {
    let text = CString::new(path.as_os_str().as_bytes())?;
    // SAFETY: text is a NUL-terminated string that lives across the call; haste opens only libraries it built, whose initializers are those a cargo build of the same code would run.
    let handle = unsafe { libc::dlopen(text.as_ptr(), mode | libc::RTLD_GLOBAL) };
    if !handle.is_null() {
        return Ok(handle);
    }
    // SAFETY: dlerror takes no arguments and returns null or a message that stays valid until the next dynamic linker call on this thread.
    let reason = unsafe { libc::dlerror() };
    // SAFETY: reason is a live NUL-terminated string when it is not null, and it is copied before any other dynamic linker call.
    let reason = if reason.is_null() { String::new() } else { unsafe { CStr::from_ptr(reason) }.to_string_lossy().into_owned() };
    bail!("loading {}: {reason}", path.display())
}

fn find(handle: *mut c_void, name: &str) -> Result<*const c_void> {
    let text = CString::new(name)?;
    // SAFETY: handle came from dlopen and stays open for the process; text is NUL-terminated and lives across the call.
    let address = unsafe { libc::dlsym(handle, text.as_ptr()) };
    if address.is_null() {
        bail!("the native library exports no {name}");
    }
    Ok(address.cast_const())
}

/// The name and address of each static symbol in the table of the library `handle`.
fn table(handle: *mut c_void) -> Result<HashMap<Box<str>, usize>> {
    let (names, addresses, count) = (find(handle, NAMES)?.cast::<*const libc::c_char>(), find(handle, ADDRESSES)?.cast::<usize>(), find(handle, COUNT)?.cast::<libc::c_ulong>());
    // SAFETY: the table source defines count as an unsigned long, initialized before dlopen returned.
    let count = usize::try_from(unsafe { *count })?;
    let mut symbols = HashMap::default();
    for at in 0..count {
        // SAFETY: both arrays hold count entries past which at does not reach, and each name is a NUL-terminated literal in the library, which stays loaded.
        let (name, address) = unsafe { (CStr::from_ptr(*names.add(at)), *addresses.add(at)) };
        symbols.insert(name.to_string_lossy().into(), address);
    }
    Ok(symbols)
}
