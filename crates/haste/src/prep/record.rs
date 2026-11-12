use std::path::Path;

use anyhow::{Context as _, Result};
use serde::Deserialize;

/// The manifest that the prep driver writes beside its library: the library's own facts and one record per function instance it exports.
#[derive(Deserialize)]
pub(super) struct Library {
    #[serde(default)]
    pub library: Facts,
    #[serde(default)]
    pub record: Vec<Record>,
}

/// What the driver found about the library as a whole: the symbols of the shared standard library's argument initializer, of its `stdout`, and of `flush` on the `Stdout` that returns.
#[derive(Deserialize, Default)]
pub(super) struct Facts {
    pub args_init: Option<String>,
    pub stdout: Option<String>,
    pub stdout_flush: Option<String>,
}

/// One function instance in the library: its symbol, definition path, closure identity, type arguments, calling convention, the layouts it crosses with, what rustc inlined into it, what it calls, whether it touches mutable state outside the standard crates, and whether rustc's release build kept this copy.
#[derive(Deserialize)]
pub(super) struct Record {
    pub symbol: String,
    pub kind: InstanceKind,
    pub path: String,
    pub closure: Option<ClosureId>,
    pub args: Vec<String>,
    pub abi: Abi,
    pub layouts: Option<Vec<String>>,
    pub inlined: Vec<String>,
    pub uses: Vec<String>,
    pub state: bool,
    pub release_copy: bool,
}

/// What rustc generated an instance's code from: a function's own body, a closure's by-value call shim, or another shim.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum InstanceKind {
    Item,
    #[serde(rename = "Shim::ClosureOnce")]
    ClosureOnce,
    #[serde(other)]
    Other,
}

/// Where a closure is: its typeck root, its call trait, and its place among the root's closures in the order their expressions end, with their count.
#[derive(Deserialize)]
pub(super) struct ClosureId {
    pub root: String,
    pub kind: Option<ClosureKind>,
    pub order: Option<usize>,
    pub count: Option<usize>,
}

/// The call trait a closure implements at most.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ClosureKind {
    Fn,
    FnMut,
    FnOnce,
}

/// The calling convention rustc lowered an instance to.
#[derive(Deserialize)]
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

/// One argument or the return: its size, alignment and passing mode.
#[derive(Deserialize)]
pub(super) struct Arg {
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
