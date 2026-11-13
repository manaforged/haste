use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use haste_codegen::{Arch, Format, Os, RelocKind, RuntimeSymbol, Target, patch_reloc};
use rustc_hash::FxHashMap;

use crate::host::{HostSymbol, MathSymbol};

mod elf;
mod macho;
mod shims;
mod wide;

use shims::Shim;
use wide::WideShim;

pub(super) const ELF_BASE: u64 = 0x0040_0000;
pub(super) const MACH_BASE: u64 = 0x1_0000_0000;
pub(super) const MACH_PAGE: u64 = 0x4000;

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
}

#[derive(Clone, Copy)]
pub(in crate::aot) enum Binding<'a> {
    Static(u64),
    DsoHandle,
    Shim(Shim),
    Import(&'a str, Option<&'static str>),
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
        }
    }

    /// Appends 16-byte-aligned machine code to `.text` and returns its section offset.
    pub(crate) fn text(&mut self, bytes: &[u8]) -> u32 {
        append_aligned(&mut self.text, bytes, 16)
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
        self.relocs.push(CodeReloc { offset: text_offset, kind, symbol });
    }

    /// Records an 8-byte pointer relocation in `.rodata` or `.data` at `offset` targeting `symbol`.
    pub(crate) fn pointer(&mut self, in_rodata: bool, offset: u32, symbol: Symbol) {
        self.pointers.push(PtrReloc { in_rodata, offset, symbol });
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
            (format, arch) => bail!("AOT emission does not support {format:?} on {arch:?}"),
        }
    }

    /// Encodes and writes the standalone executable to `path`, marking it executable on Unix.
    pub(crate) fn write(&self, path: &Path) -> Result<()> {
        let bytes = self.encode()?;
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
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
            let signed = std::process::Command::new("codesign").args(["--force", "--sign", "-"]).arg(path).output()?;
            if !signed.status.success() {
                bail!("codesign failed: {}", std::str::from_utf8(&signed.stderr).unwrap_or("<non-utf8 stderr>").trim());
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

pub(super) struct Buf(pub(super) Vec<u8>);

impl Buf {
    pub(super) fn new() -> Self { Self(Vec::new()) }
    pub(super) fn pos(&self) -> usize { self.0.len() }
    pub(super) fn align(&mut self, align: usize) { self.0.resize(self.0.len().next_multiple_of(align), 0); }
    pub(super) fn bytes(&mut self, data: &[u8]) { self.0.extend_from_slice(data); }
    pub(super) fn words(&mut self, data: &[u32]) { data.iter().for_each(|&w| self.u32(w)); }
    pub(super) fn u64s(&mut self, data: &[u64]) { data.iter().for_each(|&w| self.u64(w)); }
    pub(super) fn u8(&mut self, val: u8) { self.0.push(val); }
    pub(super) fn u16(&mut self, val: u16) { self.0.extend(val.to_le_bytes()); }
    pub(super) fn u32(&mut self, val: u32) { self.0.extend(val.to_le_bytes()); }
    pub(super) fn u64(&mut self, val: u64) { self.0.extend(val.to_le_bytes()); }
    pub(super) fn i64(&mut self, val: i64) { self.0.extend(val.to_le_bytes()); }
    pub(super) fn patch_u32(&mut self, at: usize, val: u32) { self.0[at..at + 4].copy_from_slice(&val.to_le_bytes()); }
    pub(super) fn patch_u64(&mut self, at: usize, val: u64) { self.0[at..at + 8].copy_from_slice(&val.to_le_bytes()); }

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

fn classify_math(math: MathSymbol) -> Result<&'static str> {
    use MathSymbol::*;
    const TABLE: &[(MathSymbol, &str)] = &[
        (Sinf32, "sinf"), (Sinf64, "sin"), (Cosf32, "cosf"), (Cosf64, "cos"),
        (Expf32, "expf"), (Expf64, "exp"), (Exp2f32, "exp2f"), (Exp2f64, "exp2"),
        (Logf32, "logf"), (Logf64, "log"), (Log2f32, "log2f"), (Log2f64, "log2"),
        (Log10f32, "log10f"), (Log10f64, "log10"), (Powf32, "powf"), (Powf64, "pow"),
        (Fmaf32, "fmaf"), (Fmuladdf32, "fmaf"), (Fmaf64, "fma"), (Fmuladdf64, "fma"),
        (Maxnumf32, "fmaxf"), (Maxnumf64, "fmax"), (Minnumf32, "fminf"), (Minnumf64, "fmin"),
        (RoundTiesEvenF32, "rintf"), (RoundTiesEvenF64, "rint"), (Fmodf32, "fmodf"), (Fmodf64, "fmod"),
    ];
    TABLE.iter().find(|&&(k, _)| k == math).map(|&(_, n)| n).with_context(|| format!("{} requires a self-contained powi shim", math.link_name()))
}

fn dynamic_lib(name: &str) -> Option<&'static str> {
    if name.starts_with("_Unwind_") {
        return Some("libgcc_s.so.1");
    }
    const LIBM: &[&str] = &[
        "acos", "acosf", "acosh", "acoshf", "asin", "asinf", "asinh", "asinhf", "atan", "atanf", "atan2", "atan2f", "atanh", "atanhf",
        "cbrt", "cbrtf", "cosh", "coshf", "erf", "erff", "erfc", "erfcf", "expm1", "expm1f", "fdim", "fdimf", "hypot", "hypotf",
        "ldexp", "ldexpf", "lgamma_r", "lgammaf_r", "log1p", "log1pf", "nextafter", "nextafterf", "sinh", "sinhf", "tan", "tanf", "tanh", "tanhf", "tgamma", "tgammaf",
    ];
    LIBM.contains(&name).then_some("libm.so.6")
}

fn classify_host(host: &HostSymbol) -> Result<Binding<'_>> {
    use HostSymbol::*;
    Ok(match host {
        Noop => Binding::Shim(Shim::Noop),
        Barrier => Binding::Shim(Shim::Barrier),
        Cpuid => Binding::Shim(Shim::Cpuid),
        ExtendedState => Binding::Shim(Shim::ExtendedState),
        Syscall => Binding::Shim(Shim::Syscall),
        Alloc => Binding::Shim(Shim::Alloc),
        AllocZeroed => Binding::Shim(Shim::AllocZeroed),
        Dealloc => Binding::Shim(Shim::Dealloc),
        Realloc => Binding::Shim(Shim::Realloc),
        FillOnce => Binding::Shim(Shim::FillOnce),
        ThreadCell => Binding::Shim(Shim::ThreadCell),
        DsoHandle => Binding::DsoHandle,
        Abort => Binding::Import("abort", None),
        Math(MathSymbol::Powif32) => Binding::Shim(Shim::Wide(WideShim::Powi { f64: false })),
        Math(MathSymbol::Powif64) => Binding::Shim(Shim::Wide(WideShim::Powi { f64: true })),
        Math(math) => Binding::Import(classify_math(*math)?, Some("libm.so.6")),
        Dynamic(name) => Binding::Import(name, dynamic_lib(name)),
        X86(sym) => bail!("AOT emission requires a compiled shim for {}", sym.link_name()),
    })
}

fn classify_runtime(rt: RuntimeSymbol) -> Result<Binding<'static>> {
    use RuntimeSymbol::*;
    Ok(match rt {
        Memcpy => Binding::Import("memcpy", None),
        Memmove => Binding::Import("memmove", None),
        Memset => Binding::Import("memset", None),
        Memcmp => Binding::Import("memcmp", None),
        Malloc => Binding::Import("malloc", None),
        Free => Binding::Import("free", None),
        other => Binding::Shim(Shim::Wide(WideShim::Runtime(other))),
    })
}

#[derive(Clone, Copy, Default)]
pub(super) struct SectionAddrs {
    pub(super) text: u64,
    pub(super) rodata: u64,
    pub(super) data: u64,
    pub(super) bss: u64,
    pub(super) dso_handle: u64,
    pub(super) thread_key: u64,
}

fn classify<'a>(sym: &'a Symbol, addrs: SectionAddrs) -> Result<Binding<'a>> {
    Ok(match sym {
        Symbol::Text(off) => Binding::Static(addrs.text + u64::from(*off)),
        Symbol::Rodata(off) => Binding::Static(addrs.rodata + u64::from(*off)),
        Symbol::Data(off) => Binding::Static(addrs.data + u64::from(*off)),
        Symbol::Bss(off) => Binding::Static(addrs.bss + u64::from(*off)),
        Symbol::Word(val) => Binding::Static(*val),
        Symbol::Host(host) => classify_host(host)?,
        Symbol::Runtime(rt) => classify_runtime(*rt)?,
    })
}

pub(super) struct Plan<'a> {
    pub(in crate::aot) shims: Vec<Shim>,
    shim_index: FxHashMap<Shim, usize>,
    pub(super) imports: Vec<&'a str>,
    import_index: FxHashMap<&'a str, usize>,
    pub(super) libs: BTreeSet<&'a str>,
}

impl<'a> Plan<'a> {
    pub(super) fn collect(exe: &'a Executable, extra_imports: &[&'static str]) -> Result<Self> {
        let mut plan = Self {
            shims: Vec::new(),
            shim_index: FxHashMap::default(),
            imports: Vec::new(),
            import_index: FxHashMap::default(),
            libs: BTreeSet::new(),
        };
        extra_imports.iter().for_each(|&imp| plan.add_import(imp, None));
        let zero = SectionAddrs::default();
        for sym in exe.relocs.iter().map(|r| &r.symbol).chain(exe.pointers.iter().map(|p| &p.symbol)) {
            plan.visit(classify(sym, zero)?);
        }
        Ok(plan)
    }

    fn visit(&mut self, binding: Binding<'a>) {
        match binding {
            Binding::Shim(shim) if !self.shim_index.contains_key(&shim) => {
                self.shim_index.insert(shim, self.shims.len());
                self.shims.push(shim);
                shims::deps(shim).iter().for_each(|&dep| self.add_import(dep, None));
            }
            Binding::Import(name, lib) => self.add_import(name, lib),
            _ => {}
        }
    }

    fn add_import(&mut self, name: &'a str, lib: Option<&'static str>) {
        if let Some(lib) = lib {
            self.libs.insert(lib);
        }
        if !self.import_index.contains_key(name) {
            self.import_index.insert(name, self.imports.len());
            self.imports.push(name);
        }
    }

    pub(super) fn import_idx(&self, name: &str) -> Result<usize> {
        self.import_index.get(name).copied().with_context(|| format!("missing import {name}"))
    }

    pub(in crate::aot) fn shim_idx(&self, shim: Shim) -> Result<usize> {
        self.shim_index.get(&shim).copied().context("missing runtime shim")
    }

    pub(in crate::aot) fn has_shim(&self, shim: Shim) -> bool {
        self.shim_index.contains_key(&shim)
    }

    pub(super) fn bss_layout(&self, bss: u32) -> (u32, u32) {
        if !self.has_shim(Shim::ThreadCell) {
            return (bss, bss);
        }
        let key_off = bss.next_multiple_of(16);
        (key_off, key_off + 16)
    }
}

fn resolve_binding(sym: &Symbol, addrs: SectionAddrs, base_va: u64, shim_offs: &[usize], plan: &Plan<'_>, import_va: &dyn Fn(&str) -> Result<u64>) -> Result<(bool, u64)> {
    Ok(match classify(sym, addrs)? {
        Binding::Static(val) => (false, val),
        Binding::DsoHandle => (false, addrs.dso_handle),
        Binding::Shim(shim) => (false, base_va + shim_offs[plan.shim_idx(shim)?] as u64),
        Binding::Import(name, _) => (true, import_va(name)?),
    })
}

pub(super) fn resolve_sym(sym: &Symbol, addrs: SectionAddrs, base_va: u64, shim_offs: &[usize], plan: &Plan<'_>, stub_va: &dyn Fn(&str) -> Result<u64>) -> Result<u64> {
    resolve_binding(sym, addrs, base_va, shim_offs, plan, stub_va).map(|(_, va)| va)
}

pub(super) struct PatchCtx<'a> {
    pub(super) text_base: usize,
    pub(super) text_va: u64,
    pub(super) pie_arm64: bool,
    pub(super) addrs: SectionAddrs,
    pub(super) base_va: u64,
    pub(super) shim_offs: &'a [usize],
    pub(super) plan: &'a Plan<'a>,
    pub(super) stub_va: &'a dyn Fn(&str) -> Result<u64>,
    pub(super) got_va: &'a dyn Fn(&str) -> Result<u64>,
}

pub(super) fn patch_code(relocs: &[CodeReloc], ctx: &PatchCtx<'_>, out: &mut [u8]) -> Result<()> {
    for reloc in relocs {
        let at = ctx.text_base + reloc.offset as usize;
        if ctx.pie_arm64 && reloc.kind == RelocKind::Arm64Movw64 && !matches!(reloc.symbol, Symbol::Word(_)) {
            let pc_va = ctx.text_va + u64::from(reloc.offset);
            let (load_got, target_va) = resolve_binding(&reloc.symbol, ctx.addrs, ctx.base_va, ctx.shim_offs, ctx.plan, ctx.got_va)?;
            shims::patch_arm64_pie(&mut out[at..at + 16], pc_va, load_got, target_va)?;
            continue;
        }
        let value = resolve_sym(&reloc.symbol, ctx.addrs, ctx.base_va, ctx.shim_offs, ctx.plan, ctx.stub_va)?;
        match reloc.kind {
            RelocKind::Arm64Call26 => {
                let pc_va = ctx.text_va + u64::from(reloc.offset);
                let delta = value as i64 - pc_va as i64;
                if delta & 3 != 0 || !(-0x800_0000..0x800_0000).contains(&delta) {
                    bail!("Arm64Call26 target {value:#x} out of range from {pc_va:#x}");
                }
                let word = 0x9400_0000 | (((delta >> 2) as u32) & 0x03FF_FFFF);
                out[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
            RelocKind::PcRel32 => {
                let next_ip = ctx.text_va + u64::from(reloc.offset) + 4;
                let delta = i32::try_from(value as i64 - next_ip as i64).context("PcRel32 target out of range")?;
                out[at..at + 4].copy_from_slice(&delta.to_le_bytes());
            }
            kind => patch_reloc(kind, value, &mut out[at..])?,
        }
    }
    Ok(())
}
