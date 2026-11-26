use std::path::Path;

use anyhow::{Context as _, Result};
use rustc_hash::FxHashSet as HashSet;

use super::{ARCHIVE_MAGIC, MEMBER_HEADER};

const ELF_MAGIC: &[u8; 4] = b"\x7fELF";
const ELF64: u8 = 2;
const LITTLE: u8 = 1;
const SHT_SYMTAB: u32 = 2;
const STT_TLS: u8 = 6;
const SHN_UNDEF: u16 = 0;
const SYM_SIZE: usize = 24;

/// The thread-local symbols that the 64-bit little-endian ELF members of the archive at `path` define, which a table of data addresses cannot name.
pub(super) fn thread_locals(path: &Path) -> Result<HashSet<String>> {
    let bytes = std::fs::read(path)?;
    let mut found = HashSet::default();
    let mut at = ARCHIVE_MAGIC.len();
    while at + MEMBER_HEADER <= bytes.len() {
        let header = &bytes[at..at + MEMBER_HEADER];
        let size: usize = std::str::from_utf8(&header[48..58])?
            .trim()
            .parse()
            .with_context(|| format!("a member size at byte {at}"))?;
        let body = bytes
            .get(at + MEMBER_HEADER..at + MEMBER_HEADER + size)
            .context("a short archive member")?;
        member_thread_locals(body, &mut found);
        at += MEMBER_HEADER + size + size % 2;
    }
    Ok(found)
}

fn member_thread_locals(body: &[u8], found: &mut HashSet<String>) {
    if body.len() < 64 || &body[..4] != ELF_MAGIC || body[4] != ELF64 || body[5] != LITTLE {
        return;
    }
    let (shoff, shentsize, shnum) = (
        u64_at(body, 0x28) as usize,
        u16_at(body, 0x3a) as usize,
        u16_at(body, 0x3c) as usize,
    );
    let section =
        |index: usize| body.get(shoff + index * shentsize..shoff + (index + 1) * shentsize);
    for index in 0..shnum {
        let Some(head) = section(index) else { return };
        if u32_at(head, 4) != SHT_SYMTAB {
            continue;
        }
        let Some(strings) = section(u32_at(head, 0x28) as usize) else {
            return;
        };
        let strtab = body
            .get(u64_at(strings, 0x18) as usize..)
            .unwrap_or_default();
        let (offset, size) = (u64_at(head, 0x18) as usize, u64_at(head, 0x20) as usize);
        let Some(symbols) = body.get(offset..offset + size) else {
            return;
        };
        for symbol in symbols.chunks_exact(SYM_SIZE) {
            if symbol[4] & 0xf == STT_TLS && u16_at(symbol, 6) != SHN_UNDEF {
                let name = strtab.get(u32_at(symbol, 0) as usize..).unwrap_or_default();
                let end = name
                    .iter()
                    .position(|&byte| byte == 0)
                    .unwrap_or(name.len());
                found.insert(String::from_utf8_lossy(&name[..end]).into_owned());
            }
        }
    }
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    bytes
        .get(at..at + 2)
        .map_or(0, |b| u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    bytes
        .get(at..at + 4)
        .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    bytes
        .get(at..at + 8)
        .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap_or_default()))
}
