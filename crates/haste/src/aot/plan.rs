use std::collections::BTreeSet;

use anyhow::{Context, Result, bail};
use haste_codegen::RuntimeSymbol;
use rustc_hash::{FxHashMap, FxHashSet};

use super::shims::{self, Shim};
use super::wide::WideShim;
use super::{Executable, SectionAddrs, Symbol};
use crate::host::{HostSymbol, MathSymbol};

#[derive(Clone, Copy)]
enum Binding<'a> {
    Static(u64),
    DsoHandle,
    Shim(Shim),
    Import(&'a str, Option<&'static str>),
}

const MATH: &[(MathSymbol, &str)] = {
    use MathSymbol::*;
    &[
        (Sinf32, "sinf"),
        (Sinf64, "sin"),
        (Cosf32, "cosf"),
        (Cosf64, "cos"),
        (Expf32, "expf"),
        (Expf64, "exp"),
        (Exp2f32, "exp2f"),
        (Exp2f64, "exp2"),
        (Logf32, "logf"),
        (Logf64, "log"),
        (Log2f32, "log2f"),
        (Log2f64, "log2"),
        (Log10f32, "log10f"),
        (Log10f64, "log10"),
        (Powf32, "powf"),
        (Powf64, "pow"),
        (Fmaf32, "fmaf"),
        (Fmuladdf32, "fmaf"),
        (Fmaf64, "fma"),
        (Fmuladdf64, "fma"),
        (RoundTiesEvenF32, "rintf"),
        (RoundTiesEvenF64, "rint"),
        (Fmodf32, "fmodf"),
        (Fmodf64, "fmod"),
    ]
};

fn classify_math(math: MathSymbol) -> Result<&'static str> {
    MATH.iter()
        .find(|&&(k, _)| k == math)
        .map(|&(_, n)| n)
        .with_context(|| format!("{} requires a self-contained powi shim", math.link_name()))
}

fn dynamic_lib(name: &str) -> Option<&'static str> {
    if name.starts_with("_Unwind_") {
        return Some("libgcc_s.so.1");
    }
    const LIBM: &[&str] = &[
        "acos",
        "acosf",
        "acosh",
        "acoshf",
        "asin",
        "asinf",
        "asinh",
        "asinhf",
        "atan",
        "atanf",
        "atan2",
        "atan2f",
        "atanh",
        "atanhf",
        "cbrt",
        "cbrtf",
        "cosh",
        "coshf",
        "erf",
        "erff",
        "erfc",
        "erfcf",
        "expm1",
        "expm1f",
        "fdim",
        "fdimf",
        "hypot",
        "hypotf",
        "ldexp",
        "ldexpf",
        "lgamma_r",
        "lgammaf_r",
        "log1p",
        "log1pf",
        "nextafter",
        "nextafterf",
        "sinh",
        "sinhf",
        "tan",
        "tanf",
        "tanh",
        "tanhf",
        "tgamma",
        "tgammaf",
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
        StartPanic => Binding::Shim(Shim::StartPanic),
        PanicCleanup => Binding::Shim(Shim::PanicCleanup),
        DropPush => Binding::Shim(Shim::DropPush),
        DropPop => Binding::Shim(Shim::DropPop),
        CatchUnwind => Binding::Shim(Shim::CatchUnwind),
        Math(MathSymbol::Powif32) => Binding::Shim(Shim::Wide(WideShim::Powi { f64: false })),
        Math(MathSymbol::Powif64) => Binding::Shim(Shim::Wide(WideShim::Powi { f64: true })),
        Math(math) => Binding::Import(classify_math(*math)?, Some("libm.so.6")),
        Dynamic(name) => Binding::Import(name, dynamic_lib(name)),
        X86(sym) => bail!(
            "AOT emission requires a compiled shim for {}",
            sym.link_name()
        ),
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

pub(crate) struct Plan<'a> {
    pub(in crate::aot) shims: Vec<Shim>,
    shim_index: FxHashMap<Shim, usize>,
    pub(crate) imports: Vec<&'a str>,
    import_index: FxHashMap<&'a str, usize>,
    pub(crate) libs: BTreeSet<&'a str>,
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
    pub(crate) fn collect(exe: &'a Executable, extra_imports: &[&'static str]) -> Result<Self> {
        let mut plan = Self {
            shims: Vec::new(),
            shim_index: FxHashMap::default(),
            imports: Vec::new(),
            import_index: FxHashMap::default(),
            libs: BTreeSet::new(),
        };
        extra_imports
            .iter()
            .for_each(|&imp| plan.add_import(imp, None));
        let total = exe.relocs.len() + exe.pointers.len();
        let workers = crate::jobs().min(total / 2048);
        if workers <= 1 {
            for binding in collect_dynamic_chunk(
                exe.relocs
                    .iter()
                    .map(|r| &r.symbol)
                    .chain(exe.pointers.iter().map(|p| &p.symbol)),
            )? {
                plan.visit(binding);
            }
            plan.add_unwind_walk();
            return Ok(plan);
        }
        let reloc_chunk = exe.relocs.len().div_ceil(workers).max(1);
        let ptr_chunk = exe.pointers.len().div_ceil(workers).max(1);
        let join_err = |panic: Box<dyn std::any::Any + Send>| {
            anyhow::anyhow!(
                "AOT plan worker panicked: {}",
                crate::sync::panic_text(&*panic)
            )
        };
        let (reloc_batches, ptr_batches): (Result<Vec<_>>, Result<Vec<_>>) =
            crate::sync::scope(|scope| {
                let reloc_handles: Vec<_> = exe
                    .relocs
                    .chunks(reloc_chunk)
                    .map(|chunk| {
                        scope.spawn(move || collect_dynamic_chunk(chunk.iter().map(|r| &r.symbol)))
                    })
                    .collect();
                let ptr_handles: Vec<_> = exe
                    .pointers
                    .chunks(ptr_chunk)
                    .map(|chunk| {
                        scope.spawn(move || collect_dynamic_chunk(chunk.iter().map(|p| &p.symbol)))
                    })
                    .collect();
                let r = reloc_handles
                    .into_iter()
                    .map(|h| h.join().map_err(join_err)?)
                    .collect();
                let p = ptr_handles
                    .into_iter()
                    .map(|h| h.join().map_err(join_err)?)
                    .collect();
                (r, p)
            });
        for batch in reloc_batches?.into_iter().chain(ptr_batches?) {
            for binding in batch {
                plan.visit(binding);
            }
        }
        plan.add_unwind_walk();
        Ok(plan)
    }

    fn add_unwind_walk(&mut self) {
        if self.has_shim(Shim::StartPanic) {
            self.visit(Binding::Shim(Shim::UnwindWalk));
        }
    }

    pub(crate) fn unwind_addrs(
        &self,
        exe: &Executable,
        rodata_va: u64,
        base_va: u64,
        shim_offs: &[usize],
    ) -> (u64, u64) {
        let walk = self
            .shim_idx(Shim::UnwindWalk)
            .map_or(0, |idx| base_va + shim_offs[idx] as u64);
        let table = exe
            .unwind_table_off
            .map_or(0, |off| rodata_va + u64::from(off));
        (table, walk)
    }

    fn visit(&mut self, binding: Binding<'a>) {
        match binding {
            Binding::Shim(shim) if !self.shim_index.contains_key(&shim) => {
                self.shim_index.insert(shim, self.shims.len());
                self.shims.push(shim);
                shims::deps(shim)
                    .iter()
                    .for_each(|&dep| self.add_import(dep, None));
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

    pub(crate) fn import_idx(&self, name: &str) -> Result<usize> {
        self.import_index
            .get(name)
            .copied()
            .with_context(|| format!("missing import {name}"))
    }

    pub(in crate::aot) fn shim_idx(&self, shim: Shim) -> Result<usize> {
        self.shim_index
            .get(&shim)
            .copied()
            .context("missing runtime shim")
    }

    pub(in crate::aot) fn has_shim(&self, shim: Shim) -> bool {
        self.shim_index.contains_key(&shim)
    }

    pub(in crate::aot) fn has_unwind(&self) -> bool {
        self.has_shim(Shim::CatchUnwind)
            || self.has_shim(Shim::StartPanic)
            || self.has_shim(Shim::DropPush)
            || self.has_shim(Shim::DropPop)
    }

    pub(in crate::aot) fn has_frame_cache(&self) -> bool {
        self.has_shim(Shim::FrameAlloc) || self.has_shim(Shim::FrameFree)
    }

    pub(crate) fn bss_layout(&self, bss: u32) -> (u32, u32) {
        if !(self.has_shim(Shim::ThreadCell) || self.has_unwind()) {
            return (bss, bss);
        }
        let key_off = bss.next_multiple_of(16);
        (key_off, key_off + 32)
    }

    pub(super) fn resolve_with(
        &self,
        sym: &Symbol,
        addrs: SectionAddrs,
        (base_va, shim_offs): (u64, &[usize]),
        import_va: impl FnOnce(&str) -> Result<u64>,
    ) -> Result<(bool, u64)> {
        Ok(match classify(sym, addrs)? {
            Binding::Static(val) => (false, val),
            Binding::DsoHandle => (false, addrs.dso_handle),
            Binding::Shim(shim) => (false, base_va + shim_offs[self.shim_idx(shim)?] as u64),
            Binding::Import(name, _) => (true, import_va(name)?),
        })
    }
}

pub(crate) fn resolve_sym(
    sym: &Symbol,
    addrs: SectionAddrs,
    base_va: u64,
    shim_offs: &[usize],
    plan: &Plan<'_>,
    stub_va: &dyn Fn(&str) -> Result<u64>,
) -> Result<u64> {
    plan.resolve_with(sym, addrs, (base_va, shim_offs), stub_va)
        .map(|(_, va)| va)
}
