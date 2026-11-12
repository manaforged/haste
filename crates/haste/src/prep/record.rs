use std::path::Path;

use anyhow::{Context as _, Result};
use serde::Deserialize;

/// The manifest that the prep driver writes beside its library: the library's own facts, one record per function instance it exports, each trait object and function pointer type that rustc's code makes, calls or hands to the shared standard library, and each static that holds state.
#[derive(Deserialize)]
pub(super) struct Library {
    #[serde(default)]
    pub library: Facts,
    #[serde(default)]
    pub record: Vec<Record>,
    #[serde(default)]
    pub object: Vec<Object>,
    #[serde(default)]
    pub pointer: Vec<Pointer>,
    #[serde(default, rename = "static")]
    pub statics: Vec<Static>,
}

/// A static that holds state, which the library defines once: its key, symbol, the layouts its value holds or points to, and whether each thread has its own.
#[derive(Deserialize)]
pub(super) struct Static {
    #[serde(rename = "static")]
    pub key: String,
    pub symbol: String,
    pub layouts: Option<Vec<String>>,
    pub thread_local: bool,
}

/// What the driver found about the library as a whole: the symbols of the shared standard library's argument initializer, of its `stdout`, and of `flush` on the `Stdout` that returns, and whether rustc's code touches a trait object or function pointer type that has no key.
#[derive(Deserialize, Default)]
pub(super) struct Facts {
    pub args_init: Option<String>,
    pub stdout: Option<String>,
    pub stdout_flush: Option<String>,
    #[serde(default)]
    pub unkeyed: bool,
}

/// A trait object type that rustc's code touches: its key, each method slot of its vtable as rustc lays it out, and each trait whose methods its vtable holds.
#[derive(Deserialize)]
pub(super) struct Object {
    pub object: String,
    #[serde(default)]
    pub slots: Vec<Slot>,
    #[serde(default)]
    pub supers: Vec<Super>,
}

/// One method slot of a vtable: its index, the key of the trait method it holds, the convention of a call through it, and the layouts that call crosses.
#[derive(Deserialize)]
pub(super) struct Slot {
    pub at: usize,
    pub method: Option<String>,
    pub abi: Abi,
    pub layouts: Option<Vec<String>>,
}

/// A trait whose methods a vtable holds: its key, and the slot that holds a pointer to its own vtable, or none when its vtable starts this one.
#[derive(Deserialize)]
pub(super) struct Super {
    #[serde(rename = "trait")]
    pub of: Option<String>,
    pub vptr: Option<usize>,
}

/// A function pointer type that rustc's code touches: its key, the convention of a call through it, and the layouts that call crosses.
#[derive(Deserialize)]
pub(super) struct Pointer {
    pub pointer: String,
    pub abi: Abi,
    pub layouts: Option<Vec<String>>,
}

/// One function instance in the library: its symbol, the key that names it, the stable id of the crate that defines it, where it and the functions rustc inlined into it are defined, its definition path, closure identity, type arguments, calling convention, the layouts it crosses with, what it calls, the statics holding state it touches, the keys of the function pointer types it calls through (none when one has no key), whether it reaches standard library state each std keeps apart, and whether rustc's release build kept this copy.
#[derive(Deserialize)]
pub(super) struct Record {
    pub symbol: String,
    pub key: Option<String>,
    pub crate_id: String,
    pub at: At,
    pub kind: InstanceKind,
    pub path: String,
    pub closure: Option<ClosureId>,
    pub args: Vec<String>,
    pub abi: Abi,
    pub layouts: Option<Vec<String>>,
    pub inlined: Vec<At>,
    pub uses: Vec<String>,
    #[serde(default)]
    pub statics: Vec<String>,
    pub calls: Option<Vec<String>>,
    pub held: bool,
    pub release_copy: bool,
}

/// Where a definition is: its crate, and its file and the byte offset of its name unless a macro expansion wrote it.
#[derive(Deserialize)]
pub(super) struct At {
    #[serde(rename = "crate")]
    pub krate: String,
    pub file: Option<String>,
    pub lo: Option<u32>,
}

/// What rustc generated an instance's code from: a function's own body, a closure's by-value call shim, a drop glue, or another shim.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum InstanceKind {
    Item,
    #[serde(rename = "Shim::ClosureOnce")]
    ClosureOnce,
    #[serde(rename = "Shim::DropGlue")]
    DropGlue,
    #[serde(other)]
    Other,
}

/// What a closure record says of its closure: the call trait it implements at most.
#[derive(Deserialize)]
pub(super) struct ClosureId {
    pub kind: Option<ClosureKind>,
}

/// The call trait a closure implements at most.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ClosureKind {
    Fn,
    FnMut,
    FnOnce,
}

/// The calling convention rustc lowered an instance to.
#[derive(Deserialize, PartialEq, Eq)]
pub(super) struct Abi {
    pub conv: Conv,
    pub variadic: bool,
    pub ret: Arg,
    pub args: Vec<Arg>,
}

/// The convention of an instance: the Rust one, or another that Haste never binds.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Conv {
    Rust,
    #[serde(other)]
    Other,
}

/// One argument or the return: the key of its type, its size, alignment and passing mode.
#[derive(Deserialize, PartialEq, Eq)]
pub(super) struct Arg {
    pub key: Option<String>,
    pub size: u64,
    pub align: u64,
    pub pass: Pass,
}

/// How rustc passes one value.
#[derive(Deserialize, PartialEq, Eq, Debug)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub(super) enum Pass {
    Ignore,
    Direct { regs: Vec<Reg> },
    Pair { regs: Vec<Reg> },
    Cast {},
    Indirect {
        on_stack: bool,
        #[serde(rename = "unsized")]
        dynamic: bool,
    },
}

/// One register a value or a part of it travels in, by class and byte offset in the value.
#[derive(Deserialize, PartialEq, Eq, Debug)]
pub(super) struct Reg {
    pub class: Class,
    pub offset: u64,
}

/// The class of one register part: an integer of a width and signedness, a float, a pointer, a vector, or a class Haste has no machine type for.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub(super) enum Class {
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    Ptr,
    Vector,
    #[serde(other)]
    Other,
}

impl Library {
    /// The manifest at `path`.
    pub(super) fn read(path: &Path) -> Result<Library> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}
