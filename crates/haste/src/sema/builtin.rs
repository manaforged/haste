use anyhow::{Result, bail};
use crate::names::ItemRef;
use crate::outline::item::Attr;
use crate::outline::source::{Source, Tokens};

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
    let Some(&(from, to)) = args.first() else { return };
    out.lexed("if ! (");
    (from..to).for_each(|at| out.copy(src, at));
    out.lexed(") {");
    match args.get(1) {
        None => {
            crate_path(out, krate, "::panicking::panic");
            out.lexed(&format!("({:?})", format!("assertion failed: {}", src.spelled((from, to)))));
        }
        Some(&(rest, _)) => {
            crate_path(out, krate, "::panic::panic_2021!");
            out.lexed("(");
            (rest..args.last().map_or(rest, |last| last.1)).for_each(|at| out.copy(src, at));
            out.lexed(")");
        }
    }
    out.lexed("}");
}

/// The expansion of an `asm!` whose templates hold only comments and placeholders and whose operands only read or keep their places: an optimization barrier with no effect.
pub fn asm_barrier(src: &Source, args: &[(u32, u32)], out: &mut Tokens) -> Result<()> {
    for &(from, to) in args {
        match src.literal_text(from)? {
            Some(template) if to == from + 1 => {
                if !strip_comments(&template).chars().all(|c| c.is_whitespace()) {
                    bail!("inline assembly with instructions is not supported");
                }
            }
            _ if matches!(src.text(from), "in" | "inout") && !(from..to).any(|at| src.text(at) == "=>") => {}
            _ if src.text(from) == "options" => {}
            _ => bail!("inline assembly operand `{}` is not supported", src.spelled((from, to))),
        }
    }
    out.lexed("()");
    Ok(())
}

fn strip_comments(template: &str) -> String {
    let mut rest = template;
    let mut kept = String::new();
    while let Some(start) = rest.find(['/', '{']) {
        kept.push_str(&rest[..start]);
        let close = if rest[start..].starts_with("/*") { rest[start..].find("*/").map(|end| start + end + 2) } else if rest[start..].starts_with('{') { rest[start..].find('}').map(|end| start + end + 1) } else { None };
        let Some(close) = close else { return kept + &rest[start..] };
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
