use std::collections::BTreeSet;
use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use haste_codegen::{Arch, Format, Os, RelocKind, RuntimeSymbol, Target, patch_reloc};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::host::{HostSymbol, MathSymbol};

mod dwarf;
mod elf;
mod macho;
mod shims;
mod wide;

use shims::Shim;
use wide::WideShim;

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
    pub(super) symbols: Vec<(u32, String)>,
    pub(super) thread_slots: u32,
    pub(super) lines: Vec<(u32, u64)>,
    pub(super) sources: Vec<std::path::PathBuf>,
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
            symbols: Vec::new(),
            thread_slots: 0,
            lines: Vec::new(),
            sources: Vec::new(),
        }
    }

    /// Names the function at each `.text` offset in the executable's symbol table.
    pub(crate) fn symbols(&mut self, mut named: Vec<(u32, String)>) {
        named.sort_unstable_by_key(|&(offset, _)| offset);
        self.symbols = named;
    }

    fn symbol_names(&self, prefix: &str, strtab: &mut Vec<u8>) -> Vec<u32> {
        self.symbols.iter().map(|(_, name)| push_name(strtab, &[prefix, name])).collect()
    }

    /// Appends 16-byte-aligned machine code to `.text` and returns its section offset.
    pub(crate) fn text(&mut self, bytes: &[u8]) -> u32 {
        append_aligned(&mut self.text, bytes, 16)
    }

    /// Appends a function to `.text` at `align` bytes, shifted until the fewest of its `loops` cross a 4 KiB address, and returns its section offset.
    pub(crate) fn function(&mut self, bytes: &[u8], align: u32, loops: &[Range<u32>], lines: &[(u32, u64)]) -> u32 {
        let base = (self.text.len() as u32).next_multiple_of(align);
        let split = |at: u32| loops.iter().filter(|each| each.end > each.start && (at + each.start) / LOOP_PAGE != (at + each.end - 1) / LOOP_PAGE).count();
        let at = (0..=LOOP_SHIFT / align).map(|step| base + step * align).min_by_key(|&at| split(at)).unwrap_or(base);
        self.text.resize(at as usize, 0);
        let placed = append_aligned(&mut self.text, bytes, align);
        self.lines.extend(lines.iter().map(|&(offset, place)| (placed + offset, place)));
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

/// The libm function that each math intrinsic calls.
const MATH: &[(MathSymbol, &str)] = {
    use MathSymbol::*;
    &[
        (Sinf32, "sinf"), (Sinf64, "sin"), (Cosf32, "cosf"), (Cosf64, "cos"),
        (Expf32, "expf"), (Expf64, "exp"), (Exp2f32, "exp2f"), (Exp2f64, "exp2"),
        (Logf32, "logf"), (Logf64, "log"), (Log2f32, "log2f"), (Log2f64, "log2"),
        (Log10f32, "log10f"), (Log10f64, "log10"), (Powf32, "powf"), (Powf64, "pow"),
        (Fmaf32, "fmaf"), (Fmuladdf32, "fmaf"), (Fmaf64, "fma"), (Fmuladdf64, "fma"),
        (RoundTiesEvenF32, "rintf"), (RoundTiesEvenF64, "rint"), (Fmodf32, "fmodf"), (Fmodf64, "fmod"),
    ]
};

fn classify_math(math: MathSymbol) -> Result<&'static str> {
    MATH.iter().find(|&&(k, _)| k == math).map(|&(_, n)| n).with_context(|| format!("{} requires a self-contained powi shim", math.link_name()))
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
    (LIBM.contains(&name) || MATH.iter().any(|&(_, math)| math == name)).then_some("libm.so.6")
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
        Malloc => Binding::Shim(Shim::FrameAlloc),
        Free => Binding::Shim(Shim::FrameFree),
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
    /// How far below the thread pointer the executable's static TLS block starts, on x86-64 ELF.
    pub(super) thread_tls: u64,
    /// How far below the thread pointer the coroutine frame cache starts, on x86-64 ELF.
    pub(super) frame_tls: u64,
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

fn collect_dynamic_chunk<'a>(syms: impl Iterator<Item = &'a Symbol>) -> Result<Vec<Binding<'a>>> {
    let mut seen_host = FxHashSet::default();
    let mut seen_rt = FxHashSet::default();
    let mut found = Vec::new();
    for sym in syms {
        match sym {
            Symbol::Host(host) if seen_host.insert(host) => found.push(classify_host(host)?),
            Symbol::Runtime(rt) if seen_rt.insert(*rt) => found.push(classify_runtime(*rt)?),
            _ => {}
        }
    }
    Ok(found)
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
        let total = exe.relocs.len() + exe.pointers.len();
        let workers = crate::jobs().min(total / 2048);
        if workers <= 1 {
            for binding in collect_dynamic_chunk(exe.relocs.iter().map(|r| &r.symbol).chain(exe.pointers.iter().map(|p| &p.symbol)))? {
                plan.visit(binding);
            }
            return Ok(plan);
        }
        let reloc_chunk = exe.relocs.len().div_ceil(workers).max(1);
        let ptr_chunk = exe.pointers.len().div_ceil(workers).max(1);
        let (reloc_batches, ptr_batches): (Result<Vec<_>>, Result<Vec<_>>) = std::thread::scope(|scope| {
            let reloc_handles: Vec<_> = exe
                .relocs
                .chunks(reloc_chunk)
                .map(|chunk| scope.spawn(move || collect_dynamic_chunk(chunk.iter().map(|r| &r.symbol))))
                .collect();
            let ptr_handles: Vec<_> = exe
                .pointers
                .chunks(ptr_chunk)
                .map(|chunk| scope.spawn(move || collect_dynamic_chunk(chunk.iter().map(|p| &p.symbol))))
                .collect();
            let r = reloc_handles
                .into_iter()
                .map(|h| h.join().map_err(|panic| anyhow::anyhow!("AOT plan worker panicked: {}", crate::sync::panic_text(&*panic)))?)
                .collect();
            let p = ptr_handles
                .into_iter()
                .map(|h| h.join().map_err(|panic| anyhow::anyhow!("AOT plan worker panicked: {}", crate::sync::panic_text(&*panic)))?)
                .collect();
            (r, p)
        });
        for batch in reloc_batches?.into_iter().chain(ptr_batches?) {
            for binding in batch {
                plan.visit(binding);
            }
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

/// Appends the concatenation of `parts` and a NUL to `strtab` and returns where it starts.
pub(super) fn push_name(strtab: &mut Vec<u8>, parts: &[&str]) -> u32 {
    let at = strtab.len() as u32;
    parts.iter().for_each(|part| strtab.extend_from_slice(part.as_bytes()));
    strtab.push(0);
    at
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
    pub(super) veneers: &'a Veneers,
}

#[derive(Default)]
pub(super) struct Veneers {
    pub(super) pre_len: usize,
    pub(super) post_len: usize,
    pre: Vec<(Symbol, u64)>,
    post: Vec<(Symbol, u64)>,
    pre_vas: FxHashMap<Symbol, u64>,
    post_vas: FxHashMap<Symbol, u64>,
}

impl Veneers {
    fn resolve(&self, pc_va: u64, value: u64, sym: &Symbol) -> Option<u64> {
        if in_bl_range(pc_va, value) {
            return Some(value);
        }
        self.pre_vas
            .get(sym)
            .copied()
            .filter(|&va| in_bl_range(pc_va, va))
            .or_else(|| self.post_vas.get(sym).copied().filter(|&va| in_bl_range(pc_va, va)))
    }

    pub(super) fn emit_pre(&self, resolve: &dyn Fn(&Symbol) -> Result<u64>, out: &mut Buf) -> Result<()> {
        emit_veneer_list(&self.pre, self.pre_len, resolve, out)
    }

    pub(super) fn emit_post(&self, resolve: &dyn Fn(&Symbol) -> Result<u64>, out: &mut Buf) -> Result<()> {
        emit_veneer_list(&self.post, self.post_len, resolve, out)
    }
}

fn in_bl_range(pc_va: u64, target_va: u64) -> bool {
    let delta = target_va as i64 - pc_va as i64;
    delta & 3 == 0 && (-0x800_0000..0x800_0000).contains(&delta)
}

fn emit_veneer_list(list: &[(Symbol, u64)], total_len: usize, resolve: &dyn Fn(&Symbol) -> Result<u64>, out: &mut Buf) -> Result<()> {
    let start = out.pos();
    for (sym, va) in list {
        shims::emit_arm64_veneer(*va, resolve(sym)?, out);
    }
    out.0.resize(start + total_len, 0);
    Ok(())
}

pub(super) fn plan_veneers(exe: &Executable, plan: &Plan<'_>, base_va: u64, head_off: usize, shim_offs: &[usize], calc_stubs_off: impl Fn(usize, usize) -> usize) -> Result<Veneers> {
    let lead = |veneers: usize| (head_off + veneers * 12).next_multiple_of(LOOP_PAGE as usize) - head_off;
    if exe.target.arch != Arch::Arm64 || (head_off + exe.text.len() + plan.imports.len() * 12) < 0x700_0000 {
        return Ok(Veneers { pre_len: lead(0), ..Veneers::default() });
    }
    let post_rel = exe.text.len().next_multiple_of(16);
    let mut pre_syms = Vec::new();
    let mut post_syms = Vec::new();
    let mut pre_seen = FxHashSet::default();
    let mut post_seen = FxHashSet::default();
    loop {
        let pre_len = lead(pre_syms.len());
        let post_len = (post_syms.len() * 12).next_multiple_of(16);
        let text_off = head_off + pre_len;
        let stubs_off = calc_stubs_off(pre_len, post_len);
        let addrs = SectionAddrs { text: base_va + text_off as u64, ..SectionAddrs::default() };
        let stub_va = |name: &str| Ok(base_va + (stubs_off + plan.import_idx(name)? * 12) as u64);
        let mut added = false;
        for reloc in &exe.relocs {
            if reloc.kind != RelocKind::Arm64Call26 {
                continue;
            }
            let pc_va = addrs.text + u64::from(reloc.offset);
            let target_va = resolve_sym(&reloc.symbol, addrs, base_va, shim_offs, plan, &stub_va)?;
            if in_bl_range(pc_va, target_va) {
                continue;
            }
            if reloc.offset as usize >= 0x700_0000 {
                if post_seen.insert(reloc.symbol.clone()) {
                    post_syms.push(reloc.symbol.clone());
                    added = true;
                }
            } else if pre_seen.insert(reloc.symbol.clone()) {
                pre_syms.push(reloc.symbol.clone());
                added = true;
            }
        }
        if !added {
            let pre_va = base_va + head_off as u64;
            let post_va = base_va + (text_off + post_rel) as u64;
            let pre: Vec<_> = pre_syms.into_iter().enumerate().map(|(i, s)| (s, pre_va + (i * 12) as u64)).collect();
            let post: Vec<_> = post_syms.into_iter().enumerate().map(|(i, s)| (s, post_va + (i * 12) as u64)).collect();
            let pre_vas = pre.iter().map(|(s, va)| (s.clone(), *va)).collect();
            let post_vas = post.iter().map(|(s, va)| (s.clone(), *va)).collect();
            return Ok(Veneers { pre_len, post_len, pre, post, pre_vas, post_vas });
        }
    }
}

fn reloc_span(kind: RelocKind, offset: u32) -> (usize, usize) {
    let off = offset as usize;
    match kind {
        RelocKind::X64RipRel32 => (off.saturating_sub(3), off + 4),
        RelocKind::Abs32 | RelocKind::PcRel32 | RelocKind::Arm64Call26 | RelocKind::ThreadOff32 => (off, off + 4),
        RelocKind::Abs64 | RelocKind::Arm64AdrpAdd => (off, off + 8),
        RelocKind::Arm64Movw64 => (off, off + 16),
    }
}

fn resolve_fast(sym: &Symbol, addrs: SectionAddrs, base_va: u64, shim_offs: &[usize], plan: &Plan<'_>, vas: &[u64]) -> Result<(bool, u64)> {
    Ok(match classify(sym, addrs)? {
        Binding::Static(val) => (false, val),
        Binding::DsoHandle => (false, addrs.dso_handle),
        Binding::Shim(shim) => (false, base_va + shim_offs[plan.shim_idx(shim)?] as u64),
        Binding::Import(name, _) => (true, vas[plan.import_idx(name)?]),
    })
}

fn patch_slice(relocs: &[&CodeReloc], slice_base: usize, text_va: u64, pie_arm64: bool, addrs: SectionAddrs, base_va: u64, shim_offs: &[usize], plan: &Plan<'_>, stub_vas: &[u64], got_vas: &[u64], veneers: &Veneers, out: &mut [u8]) -> Result<()> {
    for &reloc in relocs {
        let at = (reloc.offset as usize) - slice_base;
        if reloc.kind == RelocKind::Arm64AdrpAdd {
            let pc_va = text_va + u64::from(reloc.offset);
            let (load_got, target_va) = resolve_fast(&reloc.symbol, addrs, base_va, shim_offs, plan, got_vas)?;
            shims::patch_arm64_adrp_add(&mut out[at..at + 8], pc_va, load_got, target_va, matches!(reloc.symbol, Symbol::Word(_)))?;
            continue;
        }
        if pie_arm64 && reloc.kind == RelocKind::Arm64Movw64 && !matches!(reloc.symbol, Symbol::Word(_)) {
            let pc_va = text_va + u64::from(reloc.offset);
            let (load_got, target_va) = resolve_fast(&reloc.symbol, addrs, base_va, shim_offs, plan, got_vas)?;
            shims::patch_arm64_pie(&mut out[at..at + 16], pc_va, load_got, target_va)?;
            continue;
        }
        if reloc.kind == RelocKind::ThreadOff32 {
            ensure!(addrs.frame_tls != 0, "ThreadOff32 without a thread-local frame cache");
            let addend = i64::from(i32::from_le_bytes([out[at], out[at + 1], out[at + 2], out[at + 3]]));
            let off = i32::try_from(addend - addrs.frame_tls as i64).context("ThreadOff32 offset out of range")?;
            out[at..at + 4].copy_from_slice(&off.to_le_bytes());
            continue;
        }
        if reloc.kind == RelocKind::X64RipRel32 {
            if let Symbol::Word(val) = reloc.symbol {
                let val32 = u32::try_from(val).context("X64RipRel32 word exceeds u32")?;
                let reg = ((out[at - 3] & 4) << 1) | ((out[at - 1] >> 3) & 7);
                let mut bytes = [0x40 | ((reg >> 3) & 1), 0xB8 | (reg & 7), 0, 0, 0, 0, 0x90];
                bytes[2..6].copy_from_slice(&val32.to_le_bytes());
                out[at - 3..at + 4].copy_from_slice(&bytes);
                continue;
            }
            let (load_got, value) = resolve_fast(&reloc.symbol, addrs, base_va, shim_offs, plan, got_vas)?;
            if load_got {
                out[at - 2] = 0x8B;
            }
            let next_ip = text_va + u64::from(reloc.offset) + 4;
            let delta = i32::try_from(value as i64 - next_ip as i64).context("X64RipRel32 target out of range")?;
            out[at..at + 4].copy_from_slice(&delta.to_le_bytes());
            continue;
        }
        if reloc.kind == RelocKind::Abs64 && at >= 2 && out[at - 2] & 0xF8 == 0x48 && out[at - 1] & 0xF8 == 0xB8 {
            let (load_got, slot) = resolve_fast(&reloc.symbol, addrs, base_va, shim_offs, plan, got_vas)?;
            if load_got {
                let next_ip = text_va + u64::from(reloc.offset) + 5;
                let delta = i32::try_from(slot as i64 - next_ip as i64).context("Abs64 import GOT slot out of range")?;
                let reg = out[at - 1] & 7;
                out[at - 2] = 0x48 | ((out[at - 2] & 1) << 2);
                out[at - 1] = 0x8B;
                out[at] = 0x05 | (reg << 3);
                out[at + 1..at + 5].copy_from_slice(&delta.to_le_bytes());
                out[at + 5..at + 8].copy_from_slice(&[0x0F, 0x1F, 0x00]);
                continue;
            }
        }
        let (_, value) = resolve_fast(&reloc.symbol, addrs, base_va, shim_offs, plan, stub_vas)?;
        match reloc.kind {
            RelocKind::Arm64Call26 => {
                let pc_va = text_va + u64::from(reloc.offset);
                let target = veneers.resolve(pc_va, value, &reloc.symbol).with_context(|| format!("Arm64Call26 target {value:#x} out of range from {pc_va:#x}"))?;
                let delta = target as i64 - pc_va as i64;
                let word = 0x9400_0000 | (((delta >> 2) as u32) & 0x03FF_FFFF);
                out[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
            RelocKind::PcRel32 => {
                let next_ip = text_va + u64::from(reloc.offset) + 4;
                let addend = i64::from(i32::from_le_bytes([out[at], out[at + 1], out[at + 2], out[at + 3]]));
                let delta = i32::try_from(value as i64 + addend - next_ip as i64).context("PcRel32 target out of range")?;
                out[at..at + 4].copy_from_slice(&delta.to_le_bytes());
            }
            kind => patch_reloc(kind, value, &mut out[at..])?,
        }
    }
    Ok(())
}

pub(super) fn patch_code(relocs: &[CodeReloc], ctx: &PatchCtx<'_>, out: &mut [u8]) -> Result<()> {
    let mut stub_vas = Vec::with_capacity(ctx.plan.imports.len());
    let mut got_vas = Vec::with_capacity(ctx.plan.imports.len());
    for &name in &ctx.plan.imports {
        stub_vas.push((ctx.stub_va)(name)?);
        got_vas.push((ctx.got_va)(name)?);
    }
    let refs: Vec<&CodeReloc> = relocs.iter().collect();
    let workers = crate::jobs().min(refs.len() / 1024);
    if workers <= 1 {
        return patch_slice(&refs, 0, ctx.text_va, ctx.pie_arm64, ctx.addrs, ctx.base_va, ctx.shim_offs, ctx.plan, &stub_vas, &got_vas, ctx.veneers, &mut out[ctx.text_base..]);
    }
    let mut sorted = refs;
    sorted.sort_unstable_by_key(|r| r.offset);
    let step = sorted.len().div_ceil(workers).max(1);
    let mut bounds = Vec::with_capacity(workers + 1);
    let mut pos = 0;
    while pos < sorted.len() {
        let target_end = (pos + step).min(sorted.len());
        let mut max_end = 0_usize;
        for reloc in &sorted[pos..target_end] {
            max_end = max_end.max(reloc_span(reloc.kind, reloc.offset).1);
        }
        let mut end = target_end;
        while end < sorted.len() && reloc_span(sorted[end].kind, sorted[end].offset).0 < max_end {
            max_end = max_end.max(reloc_span(sorted[end].kind, sorted[end].offset).1);
            end += 1;
        }
        let slice_start = if pos == 0 { 0 } else { reloc_span(sorted[pos].kind, sorted[pos].offset).0 };
        bounds.push((pos, end, slice_start));
        pos = end;
    }
    let mut slices = Vec::with_capacity(bounds.len());
    let mut tail = &mut out[ctx.text_base..];
    let mut prev_start = 0_usize;
    for (idx, &(c_start, c_end, _)) in bounds.iter().enumerate() {
        let slice_len = if idx + 1 == bounds.len() { tail.len() } else { bounds[idx + 1].2 - prev_start };
        let (head, rest) = tail.split_at_mut(slice_len);
        slices.push((&sorted[c_start..c_end], prev_start, head));
        prev_start += slice_len;
        tail = rest;
    }
    let (text_va, pie_arm64, addrs, base_va, shim_offs, plan, veneers) = (ctx.text_va, ctx.pie_arm64, ctx.addrs, ctx.base_va, ctx.shim_offs, ctx.plan, ctx.veneers);
    let (stub_vas_ref, got_vas_ref) = (stub_vas.as_slice(), got_vas.as_slice());
    let results: Vec<Result<()>> = std::thread::scope(|scope| {
        let handles: Vec<_> = slices
            .into_iter()
            .map(|(chunk, base, buf)| scope.spawn(move || patch_slice(chunk, base, text_va, pie_arm64, addrs, base_va, shim_offs, plan, stub_vas_ref, got_vas_ref, veneers, buf)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().map_err(|panic| anyhow::anyhow!("AOT patch worker panicked: {}", crate::sync::panic_text(&*panic)))?)
            .collect()
    });
    for res in results {
        res?;
    }
    Ok(())
}
