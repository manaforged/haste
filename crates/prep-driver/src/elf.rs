use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::error::{Error, Result};

const SYMTAB: u32 = 2;
const DYNSYM: u32 = 11;
const SYMBOL_SIZE: usize = 24;
const SECTION_SIZE: usize = 64;
const HIDDEN: u8 = 2;
const VISIBILITY: u8 = 3;
const GLOBAL: u8 = 1;
const WEAK: u8 = 2;
const UNDEFINED: usize = 0;

/// Which symbol table of an ELF64 file to read: the full one of an object, or the dynamic one of a shared library.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Table {
    Static,
    Dynamic,
}

/// Which defined symbols to list: all of them, or only the global and weak ones.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    All,
    Extern,
}

/// One entry of a symbol table: where it lies in the file, its name, binding byte, visibility byte and section index.
struct Symbol {
    at: usize,
    name: String,
    info: u8,
    other: u8,
    section: usize,
}

/// The names of the symbols that the ELF64 file at `path` defines in its `table`, global and weak ones only for `Scope::Extern`.
pub fn defined(path: &Path, table: Table, scope: Scope) -> Result<HashSet<String>> {
    let bytes = fs::read(path)?;
    let kind = match table {
        Table::Static => SYMTAB,
        Table::Dynamic => DYNSYM,
    };
    let kept = |symbol: &Symbol| symbol.section != UNDEFINED && !symbol.name.is_empty() && (scope == Scope::All || matches!(symbol.info >> 4, GLOBAL | WEAK));
    Ok(symbols(&bytes, kind)?.into_iter().filter(kept).map(|symbol| symbol.name).collect())
}

/// Makes every global symbol that the ELF64 object at `path` defines weak, so the release and exported copies of an instance may both be linked and the first one kept, and gives default visibility to every hidden global or weak symbol that `names` lists, so a shared library exports the instances that rustc declared hidden across codegen units; returns how many symbols changed.
pub fn rebind(path: &Path, names: &HashSet<String>) -> Result<usize> {
    let mut bytes = fs::read(path)?;
    let mut changed = 0;
    for symbol in symbols(&bytes, SYMTAB)? {
        let bound = matches!(symbol.info >> 4, GLOBAL | WEAK);
        if bound && symbol.other & VISIBILITY == HIDDEN && names.contains(&symbol.name) {
            bytes[symbol.at + 5] &= !VISIBILITY;
            changed += 1;
        }
        if symbol.info >> 4 == GLOBAL && symbol.section != UNDEFINED {
            bytes[symbol.at + 4] = WEAK << 4 | (symbol.info & 0xf);
            changed += 1;
        }
    }
    if changed > 0 {
        fs::write(path, &bytes)?;
    }
    Ok(changed)
}

/// Every entry of each symbol table of section type `kind` in the ELF64 file `bytes`.
fn symbols(bytes: &[u8], kind: u32) -> Result<Vec<Symbol>> {
    let mut found = Vec::new();
    for section in sections(bytes)? {
        if word(bytes, section + 4)? as u32 != kind {
            continue;
        }
        let (start, size) = (word(bytes, section + 0x18)? as usize, word(bytes, section + 0x20)? as usize);
        let strings = section_offset(bytes, word(bytes, section + 0x28)? as u32 as usize)?;
        for at in (start..start + size).step_by(SYMBOL_SIZE) {
            let info = *bytes.get(at + 4).ok_or_else(truncated)?;
            let other = *bytes.get(at + 5).ok_or_else(truncated)?;
            let name = strings + word(bytes, at)? as u32 as usize;
            let end = bytes.get(name..).and_then(|tail| tail.iter().position(|&byte| byte == 0)).ok_or_else(truncated)?;
            let name = String::from_utf8_lossy(&bytes[name..name + end]).into_owned();
            found.push(Symbol { at, name, info, other, section: half(bytes, at + 6)? });
        }
    }
    Ok(found)
}

fn sections(bytes: &[u8]) -> Result<impl Iterator<Item = usize>> {
    if bytes.get(..5) != Some(b"\x7fELF\x02") {
        return Err(Error("not an ELF64 object".to_string()));
    }
    let offset = word(bytes, 0x28)? as usize;
    let count = half(bytes, 0x3c)?;
    Ok((0..count).map(move |index| offset + index * SECTION_SIZE))
}

fn section_offset(bytes: &[u8], index: usize) -> Result<usize> {
    let header = word(bytes, 0x28)? as usize + index * SECTION_SIZE;
    Ok(word(bytes, header + 0x18)? as usize)
}

fn word(bytes: &[u8], at: usize) -> Result<u64> {
    let slice = bytes.get(at..at + 8).ok_or_else(truncated)?;
    let mut array = [0; 8];
    array.copy_from_slice(slice);
    Ok(u64::from_le_bytes(array))
}

fn half(bytes: &[u8], at: usize) -> Result<usize> {
    let slice = bytes.get(at..at + 2).ok_or_else(truncated)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]) as usize)
}

fn truncated() -> Error {
    Error("a truncated ELF object".to_string())
}
