//! An assembler for the text of module-level `global_asm!` blocks: the directives `.text`, `.p2align`, `.globl` and labels, and each target's instruction subset.

use crate::Code;
use crate::reloc::StencilReloc;
use crate::target::{Arch, Target};

/// Machine code assembled from `global_asm!` text, with the byte offset of each label it defines.
#[derive(Clone, Debug)]
pub struct Assembly {
    pub code: Code,
    pub labels: Vec<(String, u32)>,
}

/// Resolves a `{name}` or `{index}` operand to its relocation symbol index.
pub(crate) type Symbols<'s> = dyn Fn(&str) -> Result<u32, String> + 's;

/// One target's instruction encoder for assembly text.
pub(crate) trait Encoder {
    /// The byte offset the next instruction lands at.
    fn pos(&self) -> u32;
    /// Emits one no-op padding instruction.
    fn pad(&mut self);
    /// Encodes one instruction from its mnemonic and its operands split at top-level commas.
    fn instruction(
        &mut self,
        mnemonic: &str,
        operands: &[&str],
        symbol: &Symbols<'_>,
    ) -> Result<(), String>;
    /// The machine code and its relocations.
    fn finish(self) -> Result<(Vec<u8>, Vec<StencilReloc>), String>;
}

/// Assembles `text` for `target`, where the placeholder `{name}` or `{index}` names an entry of `symbols` and relocates against that entry's position.
pub fn assemble(text: &str, target: &Target, symbols: &[&str]) -> Result<Assembly, String> {
    match target.arch {
        Arch::X64 => drive(
            text,
            symbols,
            crate::x64::text::Text::new(target.direct_calls),
        ),
        Arch::Arm64 => drive(
            text,
            symbols,
            crate::arm64::text::Text::new(target.direct_calls),
        ),
    }
}

fn drive(text: &str, symbols: &[&str], mut encoder: impl Encoder) -> Result<Assembly, String> {
    let symbol = |operand: &str| placeholder(operand, symbols);
    let mut labels = Vec::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        statement(line, &mut encoder, &mut labels, &symbol)
            .map_err(|error| format!("global_asm `{line}`: {error}"))?;
    }
    let (bytes, relocs) = encoder.finish()?;
    Ok(Assembly {
        code: Code::from_parts(bytes, Vec::new(), relocs, Vec::new()),
        labels,
    })
}

fn statement(
    line: &str,
    encoder: &mut impl Encoder,
    labels: &mut Vec<(String, u32)>,
    symbol: &Symbols<'_>,
) -> Result<(), String> {
    if let Some(name) = line.strip_suffix(':') {
        let name = symbol_name(name)?;
        if labels.iter().any(|(seen, _)| seen == name) {
            return Err(format!("label `{name}` is defined twice"));
        }
        labels.push((name.to_owned(), encoder.pos()));
        return Ok(());
    }
    let (mnemonic, rest) = line
        .split_once(char::is_whitespace)
        .map_or((line, ""), |(mnemonic, rest)| (mnemonic, rest.trim()));
    match mnemonic {
        ".text" if rest.is_empty() => Ok(()),
        ".p2align" => align(encoder, rest),
        ".globl" | ".global" => symbol_name(rest).map(|_| ()),
        _ if mnemonic.starts_with('.') => Err(format!("unsupported directive `{mnemonic}`")),
        _ => encoder.instruction(mnemonic, &operands(rest), symbol),
    }
}

fn align(encoder: &mut impl Encoder, power: &str) -> Result<(), String> {
    let unit = power
        .parse::<u32>()
        .ok()
        .filter(|&power| power <= 12)
        .map(|power| 1u32 << power)
        .ok_or_else(|| format!("`.p2align {power}` takes one power of two up to 12"))?;
    while encoder.pos() % unit != 0 {
        encoder.pad();
    }
    Ok(())
}

fn symbol_name(name: &str) -> Result<&str, String> {
    let valid = name
        .chars()
        .next()
        .is_some_and(|first| !first.is_ascii_digit())
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '$'));
    if valid {
        Ok(name)
    } else {
        Err(format!("`{name}` is not a symbol name"))
    }
}

fn operands(text: &str) -> Vec<&str> {
    let (mut parts, mut depth, mut start) = (Vec::new(), 0i32, 0);
    for (at, ch) in text.char_indices() {
        match ch {
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(text[start..at].trim());
                start = at + 1;
            }
            _ => {}
        }
    }
    if !text.is_empty() {
        parts.push(text[start..].trim());
    }
    parts
}

fn placeholder(operand: &str, symbols: &[&str]) -> Result<u32, String> {
    let name = operand
        .strip_prefix('{')
        .and_then(|rest| rest.strip_suffix('}'))
        .ok_or_else(|| format!("`{operand}` is not a `{{name}}` operand"))?;
    let index = match name.parse::<usize>() {
        Ok(index) => (index < symbols.len()).then_some(index),
        Err(_) => symbols.iter().position(|symbol| *symbol == name),
    };
    index
        .map(|index| index as u32)
        .ok_or_else(|| format!("no `sym` operand is named `{name}`"))
}

/// The integer `text`, decimal or `0x` hexadecimal, after an optional `#` and sign.
pub(crate) fn integer(text: &str) -> Result<i64, String> {
    let bare = text.strip_prefix('#').unwrap_or(text).trim();
    let (negative, digits) = match bare.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, bare.strip_prefix('+').unwrap_or(bare)),
    };
    let magnitude = match digits.strip_prefix("0x") {
        Some(hex) => i64::from_str_radix(hex, 16),
        None => digits.parse::<i64>(),
    };
    let magnitude = magnitude.map_err(|_| format!("`{text}` is not an integer"))?;
    Ok(if negative { -magnitude } else { magnitude })
}
