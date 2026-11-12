use std::collections::BTreeSet;

use anyhow::{Result, bail};
use haste_codegen::{ArgumentPurpose, Type};

use crate::graph::Workspace;
use crate::graph::manifest::Kind as DepKind;
use crate::lower::{Parts, Repr, repr, returned, signature_of, unwrapped};
use crate::sema::Context;
use crate::sema::ty::Ty;

use super::name::{Callee, Printer, alias};

/// The source every shim starts with: the pair a two-register result returns in, and the helpers that move values between registers and Rust types.
pub(super) const PRELUDE: &str = "#![allow(warnings)]
extern crate alloc;

#[repr(C)]
pub struct Pair<A, B>(A, B);

#[inline(always)]
unsafe fn take<S, T>(raw: S) -> T {
    unsafe { ::core::mem::transmute_copy(&raw) }
}

#[inline(always)]
fn part<T, S>(value: &T, offset: usize, size: usize) -> ::core::mem::MaybeUninit<S> {
    let mut out = ::core::mem::MaybeUninit::<S>::zeroed();
    unsafe { ::core::ptr::copy_nonoverlapping((value as *const T as *const u8).add(offset), out.as_mut_ptr() as *mut u8, size) };
    out
}

#[inline(always)]
fn give<T, S>(value: T) -> ::core::mem::MaybeUninit<S> {
    let value = ::core::mem::ManuallyDrop::new(value);
    part(&*value, 0, ::core::mem::size_of::<T>())
}

#[inline(always)]
fn pair<T, A, B>(value: T, at: [usize; 3]) -> Pair<::core::mem::MaybeUninit<A>, ::core::mem::MaybeUninit<B>> {
    let value = ::core::mem::ManuallyDrop::new(value);
    Pair(part(&*value, 0, at[1]), part(&*value, at[0], at[2]))
}
";

const UNINIT: &str = "::core::mem::MaybeUninit";

const CARGO_TOML: &str = "[package]
name = \"{name}\"
version = \"0.0.0\"
edition = \"2024\"

[lib]
crate-type = [\"cdylib\"]

[workspace]

[dependencies]
";

const NAME_HOLE: &str = "{name}";
const WRAPPER_LINES: usize = 4;

/// How one argument crosses a wrapper: in one register, as the address of its bytes, or not at all.
#[derive(Clone, Copy)]
enum Pass {
    Reg(Type),
    Addr,
    Skip,
}

/// How a wrapper's result crosses back: not at all, in one register, in two registers with the second's offset, or through a leading result pointer.
#[derive(Clone, Copy)]
enum Give {
    Unit,
    Reg(Type),
    Pair(Type, Type, i32),
    Out,
}

/// One prepped instance as the shim exports it: the reached function, its manifest text, the hash of its crossing layouts, and its call.
pub(super) struct Wrapper {
    pub index: usize,
    pub key: String,
    pub layout: u64,
    callee: String,
    params: Vec<(Pass, String)>,
    ret: (Give, Option<String>),
}

impl Wrapper {
    /// The wrapper of reached function `index` with `callee` and `layout`, passing `params` and returning `ret` as Haste's own calls do, or none when the shim cannot spell a type.
    pub(super) fn new(cx: &mut Context<'_>, printer: &mut Printer<'_>, (index, callee, layout): (usize, Callee, u64), (params, ret): (&[Ty], Option<Ty>)) -> Result<Option<Wrapper>> {
        let mut passed = Vec::with_capacity(params.len());
        for &ty in params {
            let Some(text) = printer.ty(cx, ty) else { return Ok(None) };
            passed.push((pass(cx, ty)?, text));
        }
        let given = match ret {
            Some(ty) => match printer.ty(cx, ty) {
                Some(text) => (give(cx, ty)?, Some(text)),
                None => return Ok(None),
            },
            None => (Give::Unit, None),
        };
        let wrapper = Wrapper { index, key: callee.key, layout, callee: callee.path, params: passed, ret: given };
        wrapper.agree(cx, params, ret)?;
        Ok(Some(wrapper))
    }

    fn agree(&self, cx: &mut Context<'_>, params: &[Ty], ret: Option<Ty>) -> Result<()> {
        let signature = signature_of(cx, ret, params.iter().copied())?;
        let mut expected: Vec<(Type, bool)> = Vec::new();
        if matches!(self.ret.0, Give::Out) {
            expected.push((Type::I64, true));
        }
        expected.extend(self.params.iter().filter_map(|(pass, _)| match pass {
            Pass::Reg(machine) => Some((*machine, false)),
            Pass::Addr => Some((Type::I64, false)),
            Pass::Skip => None,
        }));
        let returns: Vec<Type> = match self.ret.0 {
            Give::Reg(machine) => vec![machine],
            Give::Pair(first, second, _) => vec![first, second],
            Give::Unit | Give::Out => Vec::new(),
        };
        let found: Vec<(Type, bool)> = signature.params.iter().map(|param| (param.value_type, param.purpose == ArgumentPurpose::StructReturn)).collect();
        let found_returns: Vec<Type> = signature.returns.iter().map(|param| param.value_type).collect();
        if found != expected || found_returns != returns {
            bail!("the wrapper of {} passes {expected:?} -> {returns:?}, Haste's call passes {found:?} -> {found_returns:?}", self.key);
        }
        Ok(())
    }

    /// The source lines of this wrapper exported as `symbol`.
    fn render(&self, symbol: &str) -> [String; WRAPPER_LINES] {
        let mut inputs = Vec::new();
        if matches!(self.ret.0, Give::Out) {
            inputs.push("out: *mut u8".to_owned());
        }
        let mut args = Vec::new();
        for (at, (pass, ty)) in self.params.iter().enumerate() {
            match pass {
                Pass::Reg(machine) => inputs.push(format!("a{at}: {}", register(*machine))),
                Pass::Addr => inputs.push(format!("a{at}: *const u8")),
                Pass::Skip => {}
            }
            args.push(match pass {
                Pass::Reg(_) => format!("take::<_, {ty}>(a{at})"),
                Pass::Addr => format!("::core::ptr::read(a{at}.cast::<{ty}>())"),
                Pass::Skip => format!("::core::mem::zeroed::<{ty}>()"),
            });
        }
        let call = format!("{}({})", self.callee, args.join(", "));
        let (arrow, body) = self.result(&call);
        ["#[unsafe(no_mangle)]".to_owned(), format!("pub unsafe extern \"C\" fn {symbol}({}){arrow} {{", inputs.join(", ")), format!("    unsafe {{ {body} }}"), "}".to_owned()]
    }

    fn result(&self, call: &str) -> (String, String) {
        let ty = self.ret.1.as_deref().unwrap_or("()");
        match self.ret.0 {
            Give::Unit => (String::new(), format!("::core::mem::forget::<{ty}>({call})")),
            Give::Reg(machine) => (format!(" -> {UNINIT}<{}>", register(machine)), format!("give::<{ty}, {}>({call})", register(machine))),
            Give::Pair(first, second, offset) => {
                let sizes = format!("[{offset}, {}, {}]", first.bytes(), second.bytes());
                let body = format!("pair::<{ty}, {}, {}>({call}, {sizes})", slot(first), slot(second));
                (format!(" -> Pair<{UNINIT}<{}>, {UNINIT}<{}>>", slot(first), slot(second)), body)
            }
            Give::Out => (String::new(), format!("::core::ptr::write(out.cast::<{ty}>(), {call})")),
        }
    }
}

fn pass(cx: &mut Context<'_>, ty: Ty) -> Result<Pass> {
    Ok(match repr(cx, ty)? {
        Repr::Scalar(machine) => Pass::Reg(machine),
        Repr::Zst => Pass::Skip,
        Repr::Mem { .. } => unwrapped(cx, ty)?.map_or(Pass::Addr, Pass::Reg),
    })
}

fn give(cx: &mut Context<'_>, ty: Ty) -> Result<Give> {
    Ok(match repr(cx, ty)? {
        Repr::Scalar(machine) => Give::Reg(machine),
        Repr::Zst => Give::Unit,
        Repr::Mem { .. } => match returned(cx, ty)? {
            Some(Parts::One(machine)) => Give::Reg(machine),
            Some(Parts::Pair(first, second, offset)) => Give::Pair(first, second, offset),
            Some(Parts::Bytes) | None => Give::Out,
        },
    })
}

/// The Rust type that carries `machine` in one register of the C convention.
fn register(machine: Type) -> &'static str {
    match machine {
        Type::I8 | Type::I16 | Type::I32 | Type::I64 => "u64",
        Type::I128 => "u128",
        Type::F32 => "f32",
        Type::F64 => "f64",
    }
}

/// The Rust type of one eightbyte of a returned pair that the C convention returns `machine` in, alone in its register.
fn slot(machine: Type) -> &'static str {
    match machine {
        Type::F32 => "[f32; 2]",
        other => register(other),
    }
}

/// A shim crate: its name, its wrappers with the ones rustc rejected marked, and the dependency lines of its manifest.
pub(super) struct Shim {
    pub name: String,
    pub wrappers: Vec<Wrapper>,
    pub dropped: Vec<bool>,
    deps: String,
}

impl Shim {
    /// The shim named after `key` that exports `wrappers` and depends on the packages of the crates `krates` name.
    pub(super) fn new(cx: &Context<'_>, workspace: &Workspace, key: u64, wrappers: Vec<Wrapper>, krates: &BTreeSet<u32>) -> Shim {
        let dropped = vec![false; wrappers.len()];
        Shim { name: format!("haste_prep_{key:016x}"), wrappers, dropped, deps: deps(cx, workspace, krates) }
    }

    /// The exported symbol of wrapper `at`.
    pub(super) fn symbol(&self, at: usize) -> String {
        format!("{}_{at}", self.name)
    }

    /// The text of the shim's `Cargo.toml`.
    pub(super) fn manifest(&self) -> String {
        format!("{}{}", CARGO_TOML.replace(NAME_HOLE, &self.name), self.deps)
    }

    /// The text of the shim's `src/lib.rs` with every wrapper not dropped, and the wrapper that each line holds.
    pub(super) fn source(&self) -> (String, Vec<Option<usize>>) {
        let mut lines: Vec<String> = PRELUDE.lines().map(str::to_owned).collect();
        let mut owners = vec![None; lines.len()];
        for (at, wrapper) in self.wrappers.iter().enumerate().filter(|(at, _)| !self.dropped[*at]) {
            lines.extend(wrapper.render(&self.symbol(at)));
            owners.extend([Some(at); WRAPPER_LINES]);
        }
        (lines.join("\n") + "\n", owners)
    }
}

fn deps(cx: &Context<'_>, workspace: &Workspace, krates: &BTreeSet<u32>) -> String {
    let units = &workspace.graph.units;
    let outer = |unit: usize| !workspace.packages[units[unit].package].sysroot;
    let mut stack: Vec<usize> = krates.iter().map(|&krate| cx.outline.roots[krate as usize].unit).filter(|&unit| outer(unit)).collect();
    let mut seen = BTreeSet::new();
    while let Some(unit) = stack.pop() {
        if seen.insert(unit) {
            stack.extend(units[unit].deps.iter().filter(|(_, dep, kind)| *kind == DepKind::Normal && outer(*dep)).map(|(_, dep, _)| *dep));
        }
    }
    seen.into_iter().map(|unit| dep_line(workspace, unit)).collect()
}

fn dep_line(workspace: &Workspace, unit: usize) -> String {
    let entry = &workspace.graph.units[unit];
    let locked = &workspace.packages[entry.package].locked;
    let features: Vec<String> = entry.features.iter().map(|feature| format!("{feature:?}")).collect();
    let source = match locked.git() {
        Some((url, commit)) => format!("git = {url:?}, rev = {commit:?}"),
        None => format!("version = \"={}\"", locked.version),
    };
    format!("{} = {{ package = {:?}, {source}, default-features = false, features = [{}] }}\n", alias(unit), locked.name, features.join(", "))
}
