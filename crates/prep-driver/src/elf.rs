use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::error::{Error, Result};

const SYMTAB: u32 = 2;
const SYMBOL_SIZE: usize = 24;
const SECTION_SIZE: usize = 64;
const HIDDEN: u8 = 2;
const VISIBILITY: u8 = 3;
const GLOBAL: u8 = 1;
const WEAK: u8 = 2;

/// Makes every global symbol that the ELF64 object at `path` defines weak, so the release and exported copies of an instance may both be linked and the first one kept, and gives default visibility to every hidden global or weak symbol that `names` lists, so a shared library exports the instances that rustc declared hidden across codegen units; returns how many symbols changed.
pub fn rebind(path: &Path, names: &HashSet<String>) -> Result<usize> {
    let mut bytes = fs::read(path)?;
    let mut changed = 0;
    let headers: Vec<usize> = sections(&bytes)?.collect();
    for section in headers {
        if word(&bytes, section + 4)? as u32 != SYMTAB {
            continue;
        }
        let (start, size) = (word(&bytes, section + 0x18)? as usize, word(&bytes, section + 0x20)? as usize);
        let strings = section_offset(&bytes, word(&bytes, section + 0x28)? as u32 as usize)?;
        for symbol in (start..start + size).step_by(SYMBOL_SIZE) {
            if exported_hidden(&bytes, symbol, strings, names)? {
                bytes[symbol + 5] &= !VISIBILITY;
                changed += 1;
            }
            if defined_global(&bytes, symbol)? {
                bytes[symbol + 4] = WEAK << 4 | (bytes[symbol + 4] & 0xf);
                changed += 1;
            }
        }
    }
    if changed > 0 {
        fs::write(path, &bytes)?;
    }
    Ok(changed)
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

fn defined_global(bytes: &[u8], symbol: usize) -> Result<bool> {
    let info = *bytes.get(symbol + 4).ok_or_else(truncated)?;
    Ok(info >> 4 == GLOBAL && half(bytes, symbol + 6)? != 0)
}

fn exported_hidden(bytes: &[u8], symbol: usize, strings: usize, names: &HashSet<String>) -> Result<bool> {
    let info = *bytes.get(symbol + 4).ok_or_else(truncated)?;
    let other = *bytes.get(symbol + 5).ok_or_else(truncated)?;
    let bound = matches!(info >> 4, GLOBAL | WEAK);
    if !bound || other & VISIBILITY != HIDDEN {
        return Ok(false);
    }
    let name = strings + word(bytes, symbol)? as u32 as usize;
    let end = bytes.get(name..).and_then(|tail| tail.iter().position(|&byte| byte == 0)).ok_or_else(truncated)?;
    Ok(std::str::from_utf8(&bytes[name..name + end]).is_ok_and(|text| names.contains(text)))
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
