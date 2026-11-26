use std::ops::Range;

use anyhow::{Context as _, Result, anyhow, bail};
use haste_codegen::{Code, CodeUnwindDrop, RelocKind, Site, StencilReloc};

const SITE_WIRE_LEN: usize = 12;
const RELOC_WIRE_LEN: usize = 12;
const LOOP_WIRE_LEN: usize = 8;
const UNWIND_WIRE_LEN: usize = 16;

const RELOC_KINDS: [RelocKind; 9] = [
    RelocKind::Abs32,
    RelocKind::Abs64,
    RelocKind::Arm64Movw64,
    RelocKind::PcRel32,
    RelocKind::Arm64Call26,
    RelocKind::Arm64AdrpAdd,
    RelocKind::X64RipRel32,
    RelocKind::ThreadOff32,
    RelocKind::Arm64TlvCell,
];

/// Zero-copy view of a pre-compiled `Effort::Full` stencil in a substrate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilSlice<'a> {
    pub(super) code: &'a [u8],
    pub(super) sites: &'a [u8],
    pub(super) relocs: &'a [u8],
    pub(super) plan: &'a [u8],
    pub(super) loops: &'a [u8],
    pub(super) unwind_drops: &'a [u8],
    pub(super) saved_regs: u32,
}

impl<'a> StencilSlice<'a> {
    /// Encoded machine code bytes.
    pub fn code(&self) -> &'a [u8] {
        self.code
    }

    /// Canonical bitmask of callee-saved registers saved in this stencil's prologue.
    pub fn saved_regs(&self) -> u32 {
        self.saved_regs
    }

    /// Serialized portable relocation plan bytes, empty when absent.
    pub fn plan(&self) -> &'a [u8] {
        self.plan
    }

    /// Decodes the unwind site table into `Site` records.
    pub fn decode_sites(&self) -> Result<Vec<Site>> {
        let sites = self.sites;
        decode_table(sites, SITE_WIRE_LEN, "site", |at| {
            Ok(Site {
                ret: read_u32(sites, at),
                handler: read_u32(sites, at + 4),
                frame_offset: read_u32(sites, at + 8),
            })
        })
    }

    /// Decodes the relocation descriptor table into `StencilReloc` records.
    pub fn decode_relocs(&self) -> Result<Vec<StencilReloc>> {
        let relocs = self.relocs;
        decode_table(relocs, RELOC_WIRE_LEN, "reloc", |at| {
            Ok(StencilReloc {
                offset: read_u32(relocs, at),
                symbol: read_u32(relocs, at + 4),
                kind: decode_reloc_kind(relocs[at + 8])?,
            })
        })
    }

    fn decode_loops(&self) -> Result<Vec<Range<u32>>> {
        let loops = self.loops;
        decode_table(loops, LOOP_WIRE_LEN, "loop", |at| {
            Ok(read_u32(loops, at)..read_u32(loops, at + 4))
        })
    }

    /// Decodes the frame-table unwind drop records.
    pub fn decode_unwind_drops(&self) -> Result<Vec<CodeUnwindDrop>> {
        let drops = self.unwind_drops;
        decode_table(drops, UNWIND_WIRE_LEN, "unwind_drop", |at| {
            Ok(CodeUnwindDrop {
                local_off: read_u32(drops, at) as i32,
                flag_off: read_u32(drops, at + 4) as i32,
                drop_sym: read_u32(drops, at + 8),
                indirect: read_u32(drops, at + 12) != 0,
            })
        })
    }

    /// Decodes this slice into a `Code` stencil buffer.
    pub fn to_code(&self) -> Result<Code> {
        Ok(Code::from_parts(
            self.code.to_vec(),
            self.decode_sites()?,
            self.decode_relocs()?,
            self.decode_loops()?,
        )
        .with_unwind_drops(self.decode_unwind_drops()?)
        .with_saved_regs(self.saved_regs))
    }
}

fn decode_table<T>(
    bytes: &[u8],
    width: usize,
    name: &str,
    entry: impl FnMut(usize) -> Result<T>,
) -> Result<Vec<T>> {
    if bytes.len() % width != 0 {
        bail!("misaligned {name} table");
    }
    (0..bytes.len()).step_by(width).map(entry).collect()
}

/// Patches a frozen `StencilSlice` against `symbols` into a fresh byte buffer, or fails if any relocation is runtime-only.
pub fn patch_stencil(slice: &StencilSlice<'_>, symbols: &[u64]) -> Result<Vec<u8>> {
    let relocs = slice.decode_relocs()?;
    let mut buf = slice.code().to_vec();
    for reloc in relocs {
        let Some(sym_idx) = reloc.symbol_index() else {
            bail!("frozen stencil has runtime relocation");
        };
        let value = *symbols
            .get(sym_idx)
            .context("frozen stencil symbol index out of bounds")?;
        let off = reloc.offset as usize;
        let end = buf.len();
        haste_codegen::patch_reloc(
            reloc.kind,
            value,
            buf.get_mut(off..end)
                .context("frozen stencil reloc offset out of bounds")?,
        )
        .map_err(|err| anyhow!("{err}"))?;
    }
    Ok(buf)
}

/// Encodes `sites` into compact fixed-width little-endian bytes.
pub fn encode_sites(sites: &[Site], out: &mut Vec<u8>) {
    out.reserve(sites.len() * SITE_WIRE_LEN);
    for site in sites {
        out.extend_from_slice(&site.ret.to_le_bytes());
        out.extend_from_slice(&site.handler.to_le_bytes());
        out.extend_from_slice(&site.frame_offset.to_le_bytes());
    }
}

/// Encodes `relocs` into compact fixed-width little-endian bytes.
pub fn encode_relocs(relocs: &[StencilReloc], out: &mut Vec<u8>) {
    out.reserve(relocs.len() * RELOC_WIRE_LEN);
    for reloc in relocs {
        out.extend_from_slice(&reloc.offset.to_le_bytes());
        out.extend_from_slice(&reloc.symbol.to_le_bytes());
        out.push(encode_reloc_kind(reloc.kind));
        out.extend_from_slice(&[0u8; 3]);
    }
}

pub(super) fn encode_loops(loops: &[Range<u32>], out: &mut Vec<u8>) {
    out.reserve(loops.len() * LOOP_WIRE_LEN);
    for range in loops {
        out.extend_from_slice(&range.start.to_le_bytes());
        out.extend_from_slice(&range.end.to_le_bytes());
    }
}

pub(super) fn encode_unwind_drops(drops: &[CodeUnwindDrop], out: &mut Vec<u8>) {
    out.reserve(drops.len() * UNWIND_WIRE_LEN);
    for d in drops {
        out.extend_from_slice(&(d.local_off as u32).to_le_bytes());
        out.extend_from_slice(&(d.flag_off as u32).to_le_bytes());
        out.extend_from_slice(&d.drop_sym.to_le_bytes());
        out.extend_from_slice(&u32::from(d.indirect).to_le_bytes());
    }
}

#[inline]
pub(crate) fn read_u32(data: &[u8], at: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[at..at + 4]);
    u32::from_le_bytes(bytes)
}

#[inline]
pub(crate) fn read_u64(data: &[u8], at: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[at..at + 8]);
    u64::from_le_bytes(bytes)
}

const fn encode_reloc_kind(kind: RelocKind) -> u8 {
    match kind {
        RelocKind::Abs32 => 0,
        RelocKind::Abs64 => 1,
        RelocKind::Arm64Movw64 => 2,
        RelocKind::PcRel32 => 3,
        RelocKind::Arm64Call26 => 4,
        RelocKind::Arm64AdrpAdd => 5,
        RelocKind::X64RipRel32 => 6,
        RelocKind::ThreadOff32 => 7,
        RelocKind::Arm64TlvCell => 8,
    }
}

const _: () = {
    let mut i = 0;
    while i < RELOC_KINDS.len() {
        assert!(encode_reloc_kind(RELOC_KINDS[i]) as usize == i);
        i += 1;
    }
};

fn decode_reloc_kind(raw: u8) -> Result<RelocKind> {
    RELOC_KINDS
        .get(raw as usize)
        .copied()
        .ok_or_else(|| anyhow!("unknown relocation kind"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reloc_kind_round_trips_every_variant() {
        for &kind in &RELOC_KINDS {
            let encoded = encode_reloc_kind(kind);
            assert_eq!(decode_reloc_kind(encoded).expect("valid kind"), kind);
        }
        assert!(decode_reloc_kind(RELOC_KINDS.len() as u8).is_err());
    }
}
