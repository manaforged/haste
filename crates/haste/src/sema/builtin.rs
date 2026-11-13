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
    Include,
    IncludeStr,
    IncludeBytes,
}

const BUILTINS: [(&str, Builtin); 21] = [
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
    ("include", Builtin::Include),
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

/// One instruction an `asm!` template holds, which its expansion writes as Rust.
#[derive(Clone, Copy)]
enum Instruction<'a> {
    Barrier,
    Syscall,
    LockNot(u32),
    Divide(u32),
    Crc32 { bits: u32, poly: u32, crc: &'a str, data: &'a str },
}

/// The expansion of an `asm!` whose `templates` the call spells before its `operands`: an optimization barrier when they hold no instruction, a call of the host's system call entry for one `syscall`, an atomic invert for one `lock not` of memory, and a wide division for one `div`.
pub fn asm(src: &Source, templates: &[String], operands: &[(u32, u32)], out: &mut Tokens) -> Result<()> {
    let text = templates.iter().map(|template| strip_comments(template)).collect::<Vec<_>>().join("\n");
    match instruction(&text) {
        Some(Instruction::Barrier) => asm_barrier(src, operands, out),
        Some(Instruction::Syscall) => asm_syscall(src, operands, out),
        Some(Instruction::LockNot(bits)) => asm_lock_not(src, operands, bits, out),
        Some(Instruction::Divide(bits)) => asm_divide(src, operands, bits, out),
        Some(Instruction::Crc32 { bits, poly, crc, data }) => asm_crc32(src, operands, (bits, poly, crc, data), out),
        None => bail!("inline assembly `{}` is not supported", text.trim()),
    }
}

/// The instruction that the comment-free template `text` holds.
fn instruction(text: &str) -> Option<Instruction<'_>> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let placeholder = |word: &str| word.starts_with('{') && word.ends_with('}');
    match words.as_slice() {
        all if all.iter().all(|word| placeholder(word)) => Some(Instruction::Barrier),
        ["syscall"] => Some(Instruction::Syscall),
        ["lock", "not", width, "ptr", "[{0}]"] => Some(Instruction::LockNot(memory_bits(width)?)),
        ["div", "{0:r}"] => Some(Instruction::Divide(64)),
        ["div", "{0:e}"] => Some(Instruction::Divide(32)),
        [op, dst, src1, src2] => crc32_instruction(op, dst, src1, src2),
        _ => None,
    }
}

fn crc32_instruction<'a>(op: &str, dst: &'a str, src1: &'a str, src2: &'a str) -> Option<Instruction<'a>> {
    let (bits, poly) = match op {
        "crc32b" => (8, 0xEDB8_8320),
        "crc32h" => (16, 0xEDB8_8320),
        "crc32w" => (32, 0xEDB8_8320),
        "crc32x" => (64, 0xEDB8_8320),
        "crc32cb" => (8, 0x82F6_3B78),
        "crc32ch" => (16, 0x82F6_3B78),
        "crc32cw" => (32, 0x82F6_3B78),
        "crc32cx" => (64, 0x82F6_3B78),
        _ => return None,
    };
    let strip = |tok: &'a str| -> Option<&'a str> {
        let inner = tok.trim_end_matches(',').strip_prefix('{')?.strip_suffix('}')?;
        Some(inner.split(':').next().unwrap_or(inner))
    };
    let (d, s1, s2) = (strip(dst)?, strip(src1)?, strip(src2)?);
    (d == s1).then_some(Instruction::Crc32 { bits, poly, crc: d, data: s2 })
}

/// The bits of the memory operand size `width`.
fn memory_bits(width: &str) -> Option<u32> {
    match width {
        "qword" => Some(64),
        "dword" => Some(32),
        _ => None,
    }
}

/// The expansion of an `asm!` whose templates hold only comments and placeholders and whose operands only read or keep their places: an optimization barrier with no effect.
fn asm_barrier(src: &Source, args: &[(u32, u32)], out: &mut Tokens) -> Result<()> {
    for &(from, to) in args {
        let reads = matches!(src.text(from), "in" | "inout") && !(from..to).any(|at| punct_in(src, (at, to), "=>"));
        if !reads && src.text(from) != "options" {
            bail!("inline assembly operand `{}` is not supported", src.spelled((from, to)));
        }
    }
    out.lexed("()");
    Ok(())
}

/// The general register `name` names without its size prefix, as `ax` for `rax` and `eax`.
fn base_register(name: &str) -> &str {
    name.strip_prefix(['r', 'e']).filter(|rest| GENERAL_REGISTERS.contains(rest)).unwrap_or(name)
}

/// The x86 general registers that a size prefix widens.
const GENERAL_REGISTERS: [&str; 8] = ["ax", "bx", "cx", "dx", "si", "di", "bp", "sp"];

/// The expansion of an `asm!` that inverts the `bits`-wide memory its one `in(reg)` operand points at, under a `lock` prefix: a sequentially consistent atomic exclusive or with all ones.
fn asm_lock_not(src: &Source, args: &[(u32, u32)], bits: u32, out: &mut Tokens) -> Result<()> {
    let mut address = None;
    for &(from, to) in args.iter().filter(|(from, _)| src.text(*from) != "options") {
        let operand = operand(src, (from, to))?;
        match (operand.input, operand.output, address) {
            (Some(input), None, None) => address = Some(input),
            _ => bail!("a `lock not` takes one address operand, not `{}`", src.spelled((from, to))),
        }
    }
    let Some((from, to)) = address else { bail!("a `lock not` names no address") };
    out.lexed("{ let __haste_cell = (");
    out.copy_range(src, from, to);
    out.lexed(&format!(") as *const ::core::sync::atomic::AtomicU{bits}; (*__haste_cell).fetch_xor(!0, ::core::sync::atomic::Ordering::SeqCst); }}"));
    Ok(())
}

fn asm_crc32(src: &Source, args: &[(u32, u32)], (bits, poly, crc_name, data_name): (u32, u32, &str, &str), out: &mut Tokens) -> Result<()> {
    let mut named: Vec<(Option<&str>, Operand)> = Vec::new();
    for &(from, to) in args.iter().filter(|(from, _)| src.text(*from) != "options") {
        if from + 2 <= to && src.text(from + 1) == "=" {
            named.push((Some(src.text(from)), operand(src, (from + 2, to))?));
        } else {
            named.push((None, operand(src, (from, to))?));
        }
    }
    let find = |key: &str| -> Option<&Operand> {
        if let Ok(pos) = key.parse::<usize>() {
            return named.get(pos).map(|(_, op)| op);
        }
        named.iter().find(|(n, _)| *n == Some(key)).map(|(_, op)| op)
    };
    let crc_op = find(crc_name).ok_or_else(|| anyhow::anyhow!("crc32 missing operand `{crc_name}`"))?;
    let data_op = find(data_name).ok_or_else(|| anyhow::anyhow!("crc32 missing operand `{data_name}`"))?;
    let crc_in = crc_op.input.ok_or_else(|| anyhow::anyhow!("crc32 `{crc_name}` has no input"))?;
    let data_in = data_op.input.ok_or_else(|| anyhow::anyhow!("crc32 `{data_name}` has no input"))?;
    out.lexed("{ let mut __haste_crc: u32 = (");
    out.copy_range(src, crc_in.0, crc_in.1);
    out.lexed(") as u32; let mut __haste_data: u64 = (");
    out.copy_range(src, data_in.0, data_in.1);
    out.lexed(&format!(
        ") as u64; let mut __haste_i: u32 = 0; while __haste_i < {bits} {{ let __haste_lsb = (__haste_crc ^ (__haste_data as u32)) & 1; __haste_crc = (__haste_crc >> 1) ^ (0u32.wrapping_sub(__haste_lsb) & {poly}); __haste_data >>= 1; __haste_i += 1; }} "
    ));
    if let Some((out_from, out_to)) = crc_op.output {
        out.copy_range(src, out_from, out_to);
        out.lexed(" = __haste_crc as _; ");
    }
    out.lexed("}");
    Ok(())
}

/// The expansion of an `asm!` that runs one `bits`-wide `div`: the dividend `dx:ax` over the register operand, the quotient to `ax` and the remainder to `dx`.
fn asm_divide(src: &Source, args: &[(u32, u32)], bits: u32, out: &mut Tokens) -> Result<()> {
    let (mut divisor, mut high, mut low) = (None, None, None);
    for &(from, to) in args.iter().filter(|(from, _)| src.text(*from) != "options") {
        let operand = operand(src, (from, to))?;
        let slot = match base_register(&operand.register) {
            "reg" => &mut divisor,
            "dx" => &mut high,
            "ax" => &mut low,
            other => bail!("a `div` uses no `{other}`"),
        };
        *slot = Some(operand);
    }
    let (Some(divisor), Some(high), Some(low)) = (divisor, high, low) else { bail!("a `div` needs a divisor and `dx` and `ax` operands") };
    out.lexed("{");
    for (name, operand) in [("__haste_high", &high), ("__haste_low", &low), ("__haste_by", &divisor)] {
        let Some((from, to)) = operand.input else { bail!("a `div` reads its `{}` operand", operand.register) };
        out.lexed(&format!("let {name} = ("));
        out.copy_range(src, from, to);
        out.lexed(&format!(") as u{};", bits * 2));
    }
    out.lexed(&format!("let __haste_wide = (__haste_high << {bits}) | __haste_low;"));
    for (operand, op) in [(&low, "/"), (&high, "%")] {
        if let Some((from, to)) = operand.output {
            out.copy_range(src, from, to);
            out.lexed(&format!(" = (__haste_wide {op} __haste_by) as u{bits};"));
        }
    }
    out.lexed("}");
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
    let named = src.literal_text(open + 1)?.unwrap_or_else(|| src.text(open + 1).to_owned());
    let register = Some(named).filter(|_| src.text(open) == "(").ok_or_else(|| anyhow::anyhow!("inline assembly operand `{}` names no register", src.spelled((from, to))))?;
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
    for &(from, to) in args {
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
    while let Some(start) = rest.find(['/', '#']) {
        kept.push_str(&rest[..start]);
        let tail = &rest[start..];
        let close = if tail.starts_with("/*") {
            tail.find("*/").map(|end| start + end + 2)
        } else if tail.starts_with("//") || tail.starts_with('#') {
            Some(tail.find('\n').map_or(rest.len(), |end| start + end + 1))
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
