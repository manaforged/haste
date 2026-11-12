use crate::body::cursor::punct_in;
use crate::names::ItemRef;
use crate::outline::item::Attr;
use crate::outline::source::{Source, Tokens};
use anyhow::{Result, bail};

use super::Context;

/// A macro whose expansion the compiler writes instead of a definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin {
    Concat,
    Stringify,
    Line,
    Column,
    File,
    Cfg,
    Assert,
    FormatArgs,
    FormatArgsNl,
    Panic,
    Unreachable,
    PatternType,
    CfgSelect,
    Env,
    OptionEnv,
    Asm,
    ModulePath,
    IncludeStr,
    IncludeBytes,
}

const BUILTINS: [(&str, Builtin); 20] = [
    ("concat", Builtin::Concat),
    ("stringify", Builtin::Stringify),
    ("line", Builtin::Line),
    ("column", Builtin::Column),
    ("file", Builtin::File),
    ("cfg", Builtin::Cfg),
    ("assert", Builtin::Assert),
    ("format_args", Builtin::FormatArgs),
    ("const_format_args", Builtin::FormatArgs),
    ("format_args_nl", Builtin::FormatArgsNl),
    ("panic", Builtin::Panic),
    ("unreachable", Builtin::Unreachable),
    ("pattern_type", Builtin::PatternType),
    ("cfg_select", Builtin::CfgSelect),
    ("env", Builtin::Env),
    ("option_env", Builtin::OptionEnv),
    ("asm", Builtin::Asm),
    ("module_path", Builtin::ModulePath),
    ("include_str", Builtin::IncludeStr),
    ("include_bytes", Builtin::IncludeBytes),
];

impl Builtin {
    /// The builtin that the `#[rustc_builtin_macro]` definition named `name` stands for.
    pub fn named(name: &str) -> Option<Builtin> {
        BUILTINS.iter().find(|(known, _)| *known == name).map(|(_, builtin)| *builtin)
    }
}

impl Context<'_> {
    /// The builtin that the macro `item` is, when its definition is `#[rustc_builtin_macro]`.
    pub fn builtin(&self, item: ItemRef) -> Option<Builtin> {
        let (src, entry) = self.item(item);
        let marked = entry.attrs.iter().any(|attr| matches!(attr, Attr::Other(at) if src.text(*at) == "rustc_builtin_macro"));
        Builtin::named(src.text(entry.name?)).filter(|_| marked)
    }
}

/// The byte string literal of `bytes`.
pub fn byte_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() + 3);
    out.push_str("b\"");
    for &byte in bytes {
        match byte {
            b'"' | b'\\' => out.extend(['\\', byte as char]),
            b' '..=b'~' => out.push(byte as char),
            _ => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out.push('"');
    out
}

/// Writes `$crate` of crate `krate` and then `rest`.
pub fn crate_path(out: &mut Tokens, krate: u32, rest: &str) {
    out.dollar_crate(krate);
    out.lexed(rest);
}

/// The expansion of `assert!` with the arguments `args` of the call in `src`, for a macro of crate `krate`.
pub fn assert(src: &Source, args: &[(u32, u32)], krate: u32, out: &mut Tokens) {
    let Some(&(from, to)) = args.first() else {
        return;
    };
    out.lexed("if ! (");
    out.copy_range(src, from, to);
    out.lexed(") {");
    match args.get(1) {
        None => {
            crate_path(out, krate, "::panicking::panic");
            out.lexed(&format!("({:?})", format!("assertion failed: {}", src.spelled((from, to)))));
        }
        Some(&(rest, _)) => {
            crate_path(out, krate, "::panic::panic_2021!");
            out.lexed("(");
            out.copy_range(src, rest, args.last().map_or(rest, |last| last.1));
            out.lexed(")");
        }
    }
    out.lexed("}");
}

/// The registers that carry a Linux system call's number and its six arguments on x86-64, in the order the host's system call entry takes them.
const SYSCALL_REGISTERS: [&str; 7] = ["rax", "rdi", "rsi", "rdx", "r10", "r8", "r9"];

/// The registers that the `syscall` instruction overwrites.
const SYSCALL_CLOBBERS: [&str; 2] = ["rcx", "r11"];

/// The expansion of an `asm!`: an optimization barrier when its templates hold no instruction, and a call of the host's system call entry when they hold one `syscall`.
pub fn asm(src: &Source, args: &[(u32, u32)], out: &mut Tokens) -> Result<()> {
    let mut instructions = String::new();
    for &(from, _) in args.iter().filter(|(from, to)| *to == from + 1) {
        if let Some(template) = src.literal_text(from)? {
            instructions.push_str(&strip_comments(&template));
            instructions.push('\n');
        }
    }
    match instructions.trim() {
        "" => asm_barrier(src, args, out),
        "syscall" => asm_syscall(src, args, out),
        _ => bail!("inline assembly with instructions other than one `syscall` is not supported"),
    }
}

/// The expansion of an `asm!` whose templates hold only comments and placeholders and whose operands only read or keep their places: an optimization barrier with no effect.
fn asm_barrier(src: &Source, args: &[(u32, u32)], out: &mut Tokens) -> Result<()> {
    for &(from, to) in args {
        match src.literal_text(from)? {
            Some(_) if to == from + 1 => {}
            _ if matches!(src.text(from), "in" | "inout") && !(from..to).any(|at| punct_in(src, (at, to), "=>")) => {}
            _ if src.text(from) == "options" => {}
            _ => bail!("inline assembly operand `{}` is not supported", src.spelled((from, to))),
        }
    }
    out.lexed("()");
    Ok(())
}

/// One register operand of an `asm!`: its register, the expression it reads and the place it writes.
struct Operand {
    register: String,
    input: Option<(u32, u32)>,
    output: Option<(u32, u32)>,
}

/// The operand `in`, `out`, `lateout`, `inout` or `inlateout` that spans `from` to `to`, a `_` place writing nowhere.
fn operand(src: &Source, (from, to): (u32, u32)) -> Result<Operand> {
    let open = from + 1;
    let register = src.literal_text(open + 1)?.filter(|_| src.text(open) == "(").ok_or_else(|| anyhow::anyhow!("inline assembly operand `{}` names no register", src.spelled((from, to))))?;
    let rest = (src.pair(open) + 1, to);
    let mut arrow = None;
    let mut at = rest.0;
    while at < rest.1 && arrow.is_none() {
        arrow = punct_in(src, (at, rest.1), "=>").then_some(at);
        at = if matches!(src.text(at), "(" | "[" | "{") { src.pair(at) + 1 } else { at + 1 };
    }
    let place = |(from, to): (u32, u32)| (src.text(from) != "_" || to != from + 1).then_some((from, to));
    let (input, output) = match (src.text(from), arrow) {
        ("in", None) => (Some(rest), None),
        ("out" | "lateout", None) => (None, place(rest)),
        ("inout" | "inlateout", None) => (Some(rest), Some(rest)),
        ("inout" | "inlateout", Some(arrow)) => (Some((rest.0, arrow)), place((arrow + 2, rest.1))),
        _ => bail!("inline assembly operand `{}` is not supported", src.spelled((from, to))),
    };
    Ok(Operand { register, input, output })
}

/// The expansion of an `asm!` that runs one `syscall` with its operands in the x86-64 Linux system call registers: a call of the host's system call entry, whose result the `rax` output takes.
fn asm_syscall(src: &Source, args: &[(u32, u32)], out: &mut Tokens) -> Result<()> {
    let mut inputs: [Option<(u32, u32)>; 7] = [None; 7];
    let (mut output, mut diverges) = (None, false);
    for &(from, to) in args.iter().filter(|(from, to)| *to != from + 1) {
        if src.text(from) == "options" {
            diverges = (from..to).any(|at| src.text(at) == "noreturn");
            continue;
        }
        let operand = operand(src, (from, to))?;
        let slot = SYSCALL_REGISTERS.iter().position(|register| *register == operand.register);
        match (slot, operand.output) {
            (Some(0), written) => output = written,
            (_, Some(_)) => bail!("a `syscall` writes only `rax`, not `{}`", operand.register),
            (None, None) if SYSCALL_CLOBBERS.contains(&operand.register.as_str()) && operand.input.is_none() => {}
            (None, None) => bail!("a `syscall` reads no `{}`", operand.register),
            (Some(_), None) => {}
        }
        if let Some(slot) = slot {
            inputs[slot] = operand.input;
        }
    }
    let entry = crate::host::HostSymbol::Syscall.link_name();
    out.lexed(&format!("{{ extern \"C\" {{ fn {entry}(words: &[usize; 7]) -> usize; }} let __haste_words: [usize; 7] = ["));
    for input in inputs {
        match input {
            Some((from, to)) => {
                out.lexed("::core::mem::transmute::<_, usize>(");
                out.copy_range(src, from, to);
                out.lexed("),");
            }
            None => out.lexed("0,"),
        }
    }
    out.lexed(&format!("]; let __haste_ret = {entry}(&__haste_words);"));
    if let Some((from, to)) = output {
        out.copy_range(src, from, to);
        out.lexed(" = ::core::mem::transmute(__haste_ret);");
    }
    out.lexed(if diverges { "::core::hint::unreachable_unchecked() }" } else { "}" });
    Ok(())
}

fn strip_comments(template: &str) -> String {
    let mut rest = template;
    let mut kept = String::new();
    while let Some(start) = rest.find(['/', '{', '#']) {
        kept.push_str(&rest[..start]);
        let tail = &rest[start..];
        let close = if tail.starts_with("/*") {
            tail.find("*/").map(|end| start + end + 2)
        } else if tail.starts_with("//") || tail.starts_with('#') {
            Some(tail.find('\n').map_or(rest.len(), |end| start + end + 1))
        } else if tail.starts_with('{') {
            tail.find('}').map(|end| start + end + 1)
        } else {
            Some(start + 1)
        };
        let Some(close) = close else {
            return kept + tail;
        };
        rest = &rest[close..];
    }
    kept + rest
}

/// The expansion of `format_args!` whose whole template is `template` with no placeholder, or none when it has one.
pub fn format_literal(template: &str, krate: u32, out: &mut Tokens) -> bool {
    let plain = template.replace("{{", "").replace("}}", "");
    if plain.contains(['{', '}']) {
        return false;
    }
    crate_path(out, krate, "::fmt::Arguments::from_str");
    out.lexed(&format!("({:?})", template.replace("{{", "{").replace("}}", "}")));
    true
}
