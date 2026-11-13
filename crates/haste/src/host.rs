//! The host functions that foreign declarations link to: the allocator shims through the host's allocator, the spin-loop hints and `vzeroupper` as no-ops, the AArch64 barriers as one full barrier, `cpuid` and `xgetbv` as the host CPU's own answers less the extensions the code generator cannot encode, the DSO handle of the host program, the allocation error handler and the panic runtime as an abort, and every other name through the dynamic linker.

mod cpu;
mod math;
pub mod native;
mod syscall;
mod x86;

pub use math::{MathSymbol, symbol as intrinsic};
pub use x86::X86Symbol;

use std::alloc::Layout;
use std::ffi::CString;

/// A typed host or runtime helper symbol referenced by lowered code or foreign declarations.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum HostSymbol {
    Alloc,
    AllocZeroed,
    Dealloc,
    Realloc,
    Noop,
    Barrier,
    Cpuid,
    ExtendedState,
    DsoHandle,
    Abort,
    FillOnce,
    ThreadCell,
    Syscall,
    Math(MathSymbol),
    X86(X86Symbol),
    Dynamic(Box<str>),
}

const DIRECT: &[HostSymbol] = &[
    HostSymbol::Alloc,
    HostSymbol::AllocZeroed,
    HostSymbol::Dealloc,
    HostSymbol::Realloc,
    HostSymbol::DsoHandle,
    HostSymbol::Syscall,
];

impl HostSymbol {
    /// Classifies `name` into a typed host symbol when the host links to it.
    pub fn resolve(name: &str) -> Option<Self> {
        if let Some(found) = DIRECT.iter().find(|sym| sym.link_name() == name) {
            return Some(found.clone());
        }
        let built_in = match name {
            "__rust_no_alloc_shim_is_unstable_v2" | "llvm.x86.sse2.pause" | "llvm.x86.avx.vzeroupper" | "llvm.aarch64.hint" => Self::Noop,
            "llvm.aarch64.isb" | "llvm.aarch64.dmb" | "llvm.aarch64.dsb" => Self::Barrier,
            "llvm.x86.xgetbv" => Self::ExtendedState,
            "__rust_alloc_error_handler" | "__rust_panic_cleanup" | "__rust_start_panic" => Self::Abort,
            _ => {
                if let Some(math) = MathSymbol::from_name(name) {
                    return Some(Self::Math(math));
                }
                return dynamic(name).map(|_| Self::Dynamic(name.into()));
            }
        };
        Some(built_in)
    }

    /// The external linker symbol name for this host helper.
    pub fn link_name(&self) -> &str {
        match self {
            Self::Alloc => "__rust_alloc",
            Self::AllocZeroed => "__rust_alloc_zeroed",
            Self::Dealloc => "__rust_dealloc",
            Self::Realloc => "__rust_realloc",
            Self::Noop => "__haste_noop",
            Self::Barrier => "__haste_barrier",
            Self::Cpuid => "__haste_cpuid",
            Self::ExtendedState => "__haste_xgetbv",
            Self::DsoHandle => "__dso_handle",
            Self::Abort => "abort",
            Self::FillOnce => "__haste_fill_once",
            Self::ThreadCell => "__haste_thread_cell",
            Self::Syscall => "__haste_syscall",
            Self::Math(math) => math.link_name(),
            Self::X86(x86) => x86.link_name(),
            Self::Dynamic(name) => name,
        }
    }

    /// The address of this host symbol in the running process, when available.
    pub fn host_addr(&self) -> Option<usize> {
        Some(match self {
            Self::Alloc => alloc as *const () as usize,
            Self::AllocZeroed => alloc_zeroed as *const () as usize,
            Self::Dealloc => dealloc as *const () as usize,
            Self::Realloc => realloc as *const () as usize,
            Self::Noop => nothing as *const () as usize,
            Self::Barrier => barrier as *const () as usize,
            Self::Cpuid => cpuid as *const () as usize,
            Self::ExtendedState => extended_state as *const () as usize,
            Self::DsoHandle => &DSO_HANDLE as *const u8 as usize,
            Self::Abort => abort as *const () as usize,
            Self::FillOnce => crate::image::fill_once as *const () as usize,
            Self::ThreadCell => crate::image::thread_cell as *const () as usize,
            Self::Syscall => syscall::run as *const () as usize,
            Self::Math(math) => math.host_addr(),
            Self::X86(x86) => return x86.host_addr(),
            Self::Dynamic(_) => return dynamic(self.link_name()),
        })
    }
}

/// The object whose address stands for the host program where the linker's hidden `__dso_handle` would, which the C runtime reads as the main program.
static DSO_HANDLE: u8 = 0;

fn dynamic(name: &str) -> Option<usize> {
    if let Some(address) = native::symbol(name) {
        return Some(address);
    }
    let name = CString::new(name).ok()?;
    // SAFETY: name is a NUL-terminated string that lives across the call.
    let address = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
    (!address.is_null()).then_some(address as usize)
}

/// Writes the four registers that `cpuid` leaf `leaf`, subleaf `sub_leaf` gives compiled code to `out`.
extern "C" fn cpuid(out: *mut [u32; 4], leaf: u32, sub_leaf: u32) {
    // SAFETY: compiled code passes the address of a 16-byte result slot in its frame.
    unsafe { out.write_unaligned(cpu::registers(leaf, sub_leaf)) }
}

/// The extended control register `register` that compiled code sees, or zero when the OS has not enabled `xgetbv`.
extern "C" fn extended_state(register: u32) -> u64 {
    cpu::control(register)
}

extern "C" fn alloc(size: usize, align: usize) -> *mut u8 {
    // SAFETY: compiled code calls the shim with the nonzero size and power-of-two alignment of a valid layout, as `GlobalAlloc::alloc` requires.
    unsafe { std::alloc::alloc(Layout::from_size_align_unchecked(size, align)) }
}

extern "C" fn alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    // SAFETY: as for `alloc`.
    unsafe { std::alloc::alloc_zeroed(Layout::from_size_align_unchecked(size, align)) }
}

extern "C" fn dealloc(pointer: *mut u8, size: usize, align: usize) {
    // SAFETY: compiled code frees a block that the shims allocated with this layout.
    unsafe { std::alloc::dealloc(pointer, Layout::from_size_align_unchecked(size, align)) }
}

extern "C" fn realloc(pointer: *mut u8, size: usize, align: usize, new_size: usize) -> *mut u8 {
    // SAFETY: compiled code grows or shrinks a block that the shims allocated with this layout.
    unsafe { std::alloc::realloc(pointer, Layout::from_size_align_unchecked(size, align), new_size) }
}

extern "C" fn nothing() {}

/// A full system barrier and pipeline flush, at least as strong as any `dmb`, `dsb` or `isb` option that compiled code asks for.
extern "C" fn barrier(_option: u32) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: the barrier instructions read and write no memory and leave the stack alone.
    unsafe {
        std::arch::asm!("dsb sy", "isb", options(nostack, preserves_flags))
    }
}

extern "C" fn abort() -> ! {
    std::process::abort()
}
