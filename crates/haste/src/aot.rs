use std::ops::Range;
use std::path::Path;

use anyhow::{Result, bail};
use haste_codegen::{Arch, Format, Os, RelocKind, RuntimeSymbol, Target};

use crate::host::HostSymbol;

mod dwarf;
mod elf;
mod macho;
mod patch;
mod pe;
mod plan;
mod shims;
#[cfg(test)]
mod tests;
mod tls;
mod unwind;
mod wide;

pub(super) use patch::{PatchCtx, Veneers, patch_code, plan_veneers};
pub(super) use plan::{Plan, resolve_sym};
pub(crate) use unwind::UNWIND_FN_REC_BYTES;

pub(super) const ELF_BASE: u64 = 0x0040_0000;
pub(super) const MACH_BASE: u64 = 0x1_0000_0000;
pub(super) const MACH_PAGE: u64 = 0x4000;
const LOOP_PAGE: u32 = 0x1000;
const LOOP_SHIFT: u32 = 256;

/// A relocation target inside a standalone AOT executable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(crate) enum Symbol {
    Text(u32),
    Rodata(u32),
    Data(u32),
    Bss(u32),
    Host(HostSymbol),
    Runtime(RuntimeSymbol),
    Word(u64),
}

#[derive(Clone, Debug)]
pub(super) struct CodeReloc {
    pub(super) offset: u32,
    pub(super) kind: RelocKind,
    pub(super) symbol: Symbol,
}

#[derive(Clone, Debug)]
pub(super) struct PtrReloc {
    pub(super) in_rodata: bool,
    pub(super) offset: u32,
    pub(super) symbol: Symbol,
}

/// One named function at a `.text` offset in an AOT executable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExeSymbol {
    pub(crate) offset: u32,
    pub(crate) name: String,
    pub(crate) exported: bool,
}

/// A standalone native executable assembled directly from compiled image sections without intermediate object files.
#[derive(Clone, Debug)]
pub(crate) struct Executable {
    pub(super) target: Target,
    pub(super) text: Vec<u8>,
    pub(super) rodata: Vec<u8>,
    pub(super) data: Vec<u8>,
    pub(super) bss: u32,
    pub(super) entry: Option<(u32, u32)>,
    pub(super) relocs: Vec<CodeReloc>,
    pub(super) pointers: Vec<PtrReloc>,
    pub(super) symbols: Vec<ExeSymbol>,
    pub(super) fn_starts: Vec<u32>,
    pub(super) thread_slots: u32,
    pub(super) unwind_table_off: Option<u32>,
    pub(super) lines: Vec<(u32, u64)>,
    pub(super) sources: Vec<std::path::PathBuf>,
    pub(crate) extra_libs: Vec<String>,
}

#[derive(Clone, Copy, Default)]
pub(super) struct SectionAddrs {
    pub(super) text: u64,
    pub(super) rodata: u64,
    pub(super) data: u64,
    pub(super) bss: u64,
    pub(super) dso_handle: u64,
    pub(super) thread_key: u64,
    pub(super) tlv_desc: u64,
    pub(super) unwind_tls: u64,
    pub(super) thread_tls: u64,
    pub(super) frame_tls: u64,
    pub(super) unwind_table: u64,
    pub(super) unwind_walk: u64,
}

impl Executable {
    /// Creates an empty executable image targeting `target`.
    #[must_use]
    pub(crate) fn new(target: &Target) -> Self {
        Self {
            target: *target,
            text: Vec::new(),
            rodata: Vec::new(),
            data: Vec::new(),
            bss: 0,
            entry: None,
            relocs: Vec::new(),
            pointers: Vec::new(),
            symbols: Vec::new(),
            fn_starts: Vec::new(),
            thread_slots: 0,
            unwind_table_off: None,
            lines: Vec::new(),
            sources: Vec::new(),
            extra_libs: Vec::new(),
        }
    }

    /// Names the functions at `.text` offsets in the executable's symbol table.
    pub(crate) fn symbols(&mut self, mut named: Vec<ExeSymbol>) {
        named.sort_unstable_by(|a, b| {
            a.offset
                .cmp(&b.offset)
                .then_with(|| a.exported.cmp(&b.exported))
                .then_with(|| a.name.cmp(&b.name))
        });
        self.symbols = named;
    }

    pub(super) fn split_symbols(&self) -> (Vec<&ExeSymbol>, Vec<&ExeSymbol>) {
        let locals: Vec<&ExeSymbol> = self.symbols.iter().filter(|sym| !sym.exported).collect();
        let mut exports: Vec<&ExeSymbol> = self.symbols.iter().filter(|sym| sym.exported).collect();
        exports.sort_unstable_by(|a, b| a.name.cmp(&b.name).then_with(|| a.offset.cmp(&b.offset)));
        exports.dedup_by(|a, b| a.name == b.name);
        (locals, exports)
    }

    pub(super) fn sorted_fn_starts(&self) -> Vec<u32> {
        let mut starts = self.fn_starts.clone();
        starts.extend(self.symbols.iter().map(|sym| sym.offset));
        starts.sort_unstable();
        starts.dedup();
        starts
    }

    pub(super) fn fn_size(&self, starts: &[u32], offset: u32) -> u32 {
        let next = starts.partition_point(|&start| start <= offset);
        let end = starts.get(next).copied().unwrap_or(self.text.len() as u32);
        end.saturating_sub(offset)
    }

    pub(super) fn push_prefixed_names(
        syms: &[&ExeSymbol],
        prefix: &str,
        strtab: &mut Vec<u8>,
    ) -> Vec<u32> {
        syms.iter()
            .map(|sym| push_name(strtab, &[prefix, &sym.name]))
            .collect()
    }

    /// Appends 16-byte-aligned machine code to `.text` and returns its section offset.
    pub(crate) fn text(&mut self, bytes: &[u8]) -> u32 {
        let off = append_aligned(&mut self.text, bytes, 16);
        self.fn_starts.push(off);
        off
    }

    /// Appends a function to `.text` at `align` bytes, shifted until the fewest of its `loops` cross a 4 KiB address, and returns its section offset.
    pub(crate) fn function(
        &mut self,
        bytes: &[u8],
        align: u32,
        loops: &[Range<u32>],
        lines: &[(u32, u64)],
    ) -> u32 {
        let base = (self.text.len() as u32).next_multiple_of(align);
        let split = |at: u32| {
            loops
                .iter()
                .filter(|each| {
                    each.end > each.start
                        && (at + each.start) / LOOP_PAGE != (at + each.end - 1) / LOOP_PAGE
                })
                .count()
        };
        let at = (0..=LOOP_SHIFT / align)
            .map(|step| base + step * align)
            .min_by_key(|&at| split(at))
            .unwrap_or(base);
        self.text.resize(at as usize, 0);
        let placed = append_aligned(&mut self.text, bytes, align);
        self.lines.extend(
            lines
                .iter()
                .map(|&(offset, place)| (placed + offset, place)),
        );
        self.fn_starts.push(placed);
        placed
    }

    /// Names the source files whose `Source::place` keys the debug line table resolves.
    pub(crate) fn sources(&mut self, paths: Vec<std::path::PathBuf>) {
        self.sources = paths;
    }

    /// Appends `align`-byte-aligned constant bytes to `.rodata` and returns its section offset.
    pub(crate) fn rodata(&mut self, bytes: &[u8], align: u32) -> u32 {
        append_aligned(&mut self.rodata, bytes, align.max(1))
    }

    /// Appends `align`-byte-aligned writable bytes to `.data` and returns its section offset.
    pub(crate) fn data(&mut self, bytes: &[u8], align: u32) -> u32 {
        append_aligned(&mut self.data, bytes, align.max(1))
    }

    /// Reserves `size` zero-initialized bytes in `.bss` and returns the reservation's section offset.
    pub(crate) fn bss(&mut self, size: u32, align: u32) -> u32 {
        let off = self.bss.next_multiple_of(align.max(1));
        self.bss = off.saturating_add(size);
        off
    }

    /// Sets the `.text` offsets of `std::rt::lang_start` and the program's `main`.
    pub(crate) fn entry(&mut self, lang_start: u32, user_main: u32) {
        self.entry = Some((lang_start, user_main));
    }

    /// Records a machine-code relocation at `text_offset` targeting `symbol`.
    pub(crate) fn patch(&mut self, text_offset: u32, kind: RelocKind, symbol: Symbol) {
        self.relocs.push(CodeReloc {
            offset: text_offset,
            kind,
            symbol,
        });
    }

    /// Records an 8-byte pointer relocation in `.rodata` or `.data` at `offset` targeting `symbol`.
    pub(crate) fn pointer(&mut self, in_rodata: bool, offset: u32, symbol: Symbol) {
        self.pointers.push(PtrReloc {
            in_rodata,
            offset,
            symbol,
        });
    }

    /// Encodes the complete standalone executable file bytes.
    #[must_use = "the encoded executable bytes should be written or inspected"]
    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        let Some((lang_start, user_main)) = self.entry else {
            bail!("executable has no entry point");
        };
        match (self.target.format, self.target.arch) {
            (Format::Elf, Arch::X64 | Arch::Arm64) => elf::emit(self, (lang_start, user_main)),
            (Format::MachO, Arch::X64 | Arch::Arm64) => macho::emit(self, (lang_start, user_main)),
            (Format::Pe, Arch::X64 | Arch::Arm64) => pe::emit(self, (lang_start, user_main)),
        }
    }

    /// Encodes and writes the standalone executable to `path`, marking it executable on Unix.
    pub(crate) fn write(&self, path: &Path) -> Result<()> {
        let bytes = self.encode()?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mut perms = std::fs::metadata(path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(path, perms)?;
        }
        if self.target.os == Os::Darwin && cfg!(target_os = "macos") {
            let signed = std::process::Command::new("codesign")
                .args(["--force", "--sign", "-"])
                .arg(path)
                .output()?;
            if !signed.status.success() {
                bail!(
                    "codesign failed: {}",
                    std::str::from_utf8(&signed.stderr)
                        .unwrap_or("<non-utf8 stderr>")
                        .trim()
                );
            }
        }
        Ok(())
    }
}

fn append_aligned(buf: &mut Vec<u8>, bytes: &[u8], align: u32) -> u32 {
    let off = buf.len().next_multiple_of(align as usize);
    buf.resize(off, 0);
    buf.extend_from_slice(bytes);
    off as u32
}

/// Appends the concatenation of `parts` and a NUL to `strtab` and returns where it starts.
pub(super) fn push_name(strtab: &mut Vec<u8>, parts: &[&str]) -> u32 {
    let at = strtab.len() as u32;
    parts
        .iter()
        .for_each(|part| strtab.extend_from_slice(part.as_bytes()));
    strtab.push(0);
    at
}

pub(super) struct Buf(pub(super) Vec<u8>);

impl Buf {
    pub(super) fn new() -> Self {
        Self(Vec::new())
    }
    pub(super) fn pos(&self) -> usize {
        self.0.len()
    }
    pub(super) fn align(&mut self, align: usize) {
        self.0.resize(self.0.len().next_multiple_of(align), 0);
    }
    pub(super) fn bytes(&mut self, data: &[u8]) {
        self.0.extend_from_slice(data);
    }
    pub(super) fn words(&mut self, data: &[u32]) {
        data.iter().for_each(|&w| self.u32(w));
    }
    pub(super) fn u64s(&mut self, data: &[u64]) {
        data.iter().for_each(|&w| self.u64(w));
    }
    pub(super) fn u8(&mut self, val: u8) {
        self.0.push(val);
    }
    pub(super) fn u16(&mut self, val: u16) {
        self.0.extend(val.to_le_bytes());
    }
    pub(super) fn u32(&mut self, val: u32) {
        self.0.extend(val.to_le_bytes());
    }
    pub(super) fn u64(&mut self, val: u64) {
        self.0.extend(val.to_le_bytes());
    }
    pub(super) fn i64(&mut self, val: i64) {
        self.0.extend(val.to_le_bytes());
    }
    pub(super) fn patch_u32(&mut self, at: usize, val: u32) {
        self.0[at..at + 4].copy_from_slice(&val.to_le_bytes());
    }
    pub(super) fn patch_u64(&mut self, at: usize, val: u64) {
        self.0[at..at + 8].copy_from_slice(&val.to_le_bytes());
    }

    pub(super) fn fixed_name(&mut self, name: &str) {
        let mut fixed = [0u8; 16];
        let len = name.len().min(16);
        fixed[..len].copy_from_slice(&name.as_bytes()[..len]);
        self.bytes(&fixed);
    }

    pub(super) fn uleb(&mut self, mut val: u64) {
        loop {
            let mut byte = (val & 0x7F) as u8;
            val >>= 7;
            if val != 0 {
                byte |= 0x80;
            }
            self.0.push(byte);
            if val == 0 {
                break;
            }
        }
    }
}
