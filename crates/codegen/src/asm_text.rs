//! An assembler for the text of module-level `global_asm!` blocks: the directives `.text`, `.p2align`, `.globl` and labels, and each target's instruction subset.

use rustc_hash::FxHashMap as HashMap;

use crate::Code;
use crate::reloc::StencilReloc;
use crate::target::{Arch, Target};

/// Tracks local numeric labels (`0:`, `1:`, `2:`, ...) for forward (`Nf`) and backward (`Nb`) branches.
pub(crate) struct LocalLabels<L> {
    bound: HashMap<u32, L>,
    pending: HashMap<u32, L>,
}

impl<L> Default for LocalLabels<L> {
    fn default() -> Self {
        Self {
            bound: HashMap::default(),
            pending: HashMap::default(),
        }
    }
}

impl<L: Copy> LocalLabels<L> {
    pub(crate) fn target(
        &mut self,
        id: u32,
        forward: bool,
        new_label: impl FnOnce() -> L,
    ) -> Result<L, String> {
        if forward {
            Ok(*self.pending.entry(id).or_insert_with(new_label))
        } else {
            self.bound
                .get(&id)
                .copied()
                .ok_or_else(|| format!("local label `{id}b` is not defined"))
        }
    }

    pub(crate) fn define(&mut self, id: u32, new_label: impl FnOnce() -> L) -> L {
        let label = self.pending.remove(&id).unwrap_or_else(new_label);
        self.bound.insert(id, label);
        label
    }

    pub(crate) fn finish(self) -> Result<(), String> {
        if let Some(&id) = self.pending.keys().next() {
            return Err(format!("local label `{id}f` is never defined"));
        }
        Ok(())
    }
}

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
    /// Defines a local numeric label (`0:`, `1:`, `2:`, ...) at the current offset.
    fn define_local(&mut self, id: u32);
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
        if !name.is_empty() && name.chars().all(|ch| ch.is_ascii_digit()) {
            let id = name
                .parse::<u32>()
                .map_err(|_| format!("local label `{name}` is out of range"))?;
            encoder.define_local(id);
            return Ok(());
        }
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

/// Parses a local numeric branch target like `2f` (forward) or `2b` (backward).
pub(crate) fn local_target(text: &str) -> Option<(u32, bool)> {
    let (digits, forward) = if let Some(digits) = text.strip_suffix('f') {
        (digits, true)
    } else if let Some(digits) = text.strip_suffix('b') {
        (digits, false)
    } else {
        return None;
    };
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u32>().ok().map(|id| (id, forward))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm64_assembles_fp_pairs_zero_store_and_local_branch() {
        let target = Target::find("aarch64-apple-darwin").expect("arm64 target");
        let words = |asm: &str| -> Vec<u32> {
            assemble(asm, target, &["inner"])
                .expect("arm64 global_asm must assemble")
                .code
                .bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        assert_eq!(words("stp d8, d9, [sp, #112]\n"), vec![0x6D07_27E8]);
        assert_eq!(words("ldp d8, d9, [sp, #112]\n"), vec![0x6D47_27E8]);
        assert_eq!(words("str xzr, [sp]\n"), vec![0xF900_03FF]);
        assert_eq!(words("ldr x9, [sp, #8]\n"), vec![0xF940_07E9]);
        let loop_words = words("1:\ncbz x9, 1b\ncbz x9, 2f\n2:\nret\n");
        assert_eq!(loop_words[0], 0xB400_0009);
        assert_eq!(loop_words[1], 0xB400_0029);
        assert!(assemble("str x0, [sp, #40000]\n", target, &[]).is_err());
        assert!(assemble("str x0, [sp, #300]\n", target, &[]).is_err());
    }

    #[test]
    fn x64_assembles_qword_zero_store_test_and_local_branch() {
        let target = Target::find("x86_64-unknown-linux-gnu").expect("x64 target");
        let bytes = |asm: &str| -> Vec<u8> {
            assemble(asm, target, &["inner"])
                .expect("x64 global_asm must assemble")
                .code
                .bytes
        };
        assert_eq!(
            bytes("mov qword ptr [rsp], 0\n"),
            vec![0x48, 0xC7, 0x04, 0x24, 0, 0, 0, 0]
        );
        assert_eq!(bytes("test r11, r11\n"), vec![0x4D, 0x85, 0xDB]);
        let out = assemble(
            ".globl tramp\ntramp:\n1:\ntest r11, r11\njz 2f\njmp 1b\n2:\nret\n",
            target,
            &[],
        )
        .expect("x64 local branches must assemble");
        assert_eq!(out.labels, vec![("tramp".to_owned(), 0)]);
    }

    #[test]
    fn rejects_unresolved_local_labels_on_both_arches() {
        let arm = Target::find("aarch64-apple-darwin").expect("arm64 target");
        assert!(assemble("cbz x9, 2f\n", arm, &[]).is_err());
        assert!(assemble("cbz x9, 2b\n", arm, &[]).is_err());
        let x64 = Target::find("x86_64-unknown-linux-gnu").expect("x64 target");
        assert!(assemble("jz 2f\n", x64, &[]).is_err());
        assert!(assemble("jz 2b\n", x64, &[]).is_err());
    }
}
