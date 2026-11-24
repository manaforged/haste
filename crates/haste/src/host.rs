//! The host functions that foreign declarations link to: the allocator shims through the host's allocator, the spin-loop hints and `vzeroupper` as no-ops, the AArch64 barriers as one full barrier, `cpuid` and `xgetbv` as the host CPU's own answers less the extensions the code generator cannot encode, the DSO handle of the host program, the allocation error handler and the panic runtime as an abort, and every other name through the dynamic linker.

pub(crate) mod cpu;
mod math;
pub mod native;
mod syscall;
pub(crate) mod unwind;
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
    CatchUnwind,
    StartPanic,
    PanicCleanup,
    DropPush,
    DropPop,
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
    /// Classifies `name` into a typed host symbol when the host links to it (or as a dynamic import when cross-compiling with `aot` true).
    pub fn resolve_for(name: &str, aot: bool) -> Option<Self> {
        if let Some(found) = DIRECT.iter().find(|sym| sym.link_name() == name) {
            return Some(found.clone());
        }
        let built_in = match name {
            "__rust_no_alloc_shim_is_unstable_v2"
            | "llvm.x86.sse2.pause"
            | "llvm.x86.avx.vzeroupper"
            | "llvm.aarch64.hint" => Self::Noop,
            "llvm.aarch64.isb" | "llvm.aarch64.dmb" | "llvm.aarch64.dsb" => Self::Barrier,
            "llvm.x86.xgetbv" => Self::ExtendedState,
            "__rust_alloc_error_handler" => Self::Abort,
            "__rust_start_panic" => Self::StartPanic,
            "__rust_panic_cleanup" => Self::PanicCleanup,
            _ => {
                if let Some(math) = MathSymbol::from_name(name) {
                    return Some(Self::Math(math));
                }
                if aot && crate::compile_target().triple != haste_codegen::Target::host().triple {
                    return Some(Self::Dynamic(name.into()));
                }
                return dynamic(name).map(|_| Self::Dynamic(name.into()));
            }
        };
        Some(built_in)
    }

    /// Classifies `name` into a typed host symbol when the host links to it.
    pub fn resolve(name: &str) -> Option<Self> {
        Self::resolve_for(name, false)
    }

    /// Whether this host symbol allocates fresh unaliased heap memory (`__rust_alloc` / `__rust_alloc_zeroed`).
    pub fn is_fresh_alloc(&self) -> bool {
        matches!(self, Self::Alloc | Self::AllocZeroed)
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
            Self::CatchUnwind => "__haste_catch_unwind",
            Self::StartPanic => "__rust_start_panic",
            Self::PanicCleanup => "__rust_panic_cleanup",
            Self::DropPush => "__haste_drop_push",
            Self::DropPop => "__haste_drop_pop",
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
            Self::CatchUnwind => unwind::catch_unwind as *const () as usize,
            Self::StartPanic => unwind::start_panic_addr(),
            Self::PanicCleanup => unwind::panic_cleanup as *const () as usize,
            Self::DropPush => unwind::drop_push as *const () as usize,
            Self::DropPop => unwind::drop_pop as *const () as usize,
            Self::Math(math) => math.host_addr(),
            Self::X86(x86) => return x86.host_addr(),
            Self::Dynamic(_) => return dynamic(self.link_name()),
        })
    }
}

/// The object whose address stands for the host program where the linker's hidden `__dso_handle` would, which the C runtime reads as the main program.
static DSO_HANDLE: u8 = 0;

fn dynamic(name: &str) -> Option<usize> {
    #[cfg(target_vendor = "apple")]
    match name {
        "_NSGetArgc" => return Some(apple_args::get_argc as *const () as usize),
        "_NSGetArgv" => return Some(apple_args::get_argv as *const () as usize),
        _ => {}
    }
    if let Some(address) = native::symbol(name) {
        return Some(address);
    }
    let name = CString::new(name).ok()?;
    #[cfg(unix)]
    {
        // SAFETY: name is a NUL-terminated string that lives across the call.
        let address = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
        (!address.is_null()).then_some(address as usize)
    }
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetModuleHandleA(module_name: *const i8) -> *mut core::ffi::c_void;
            fn GetProcAddress(
                module: *mut core::ffi::c_void,
                proc_name: *const i8,
            ) -> *const core::ffi::c_void;
        }
        for module_name in [
            std::ptr::null(),
            c"ucrtbase.dll".as_ptr(),
            c"kernel32.dll".as_ptr(),
            c"msvcrt.dll".as_ptr(),
        ] {
            // SAFETY: module_name is null or a static NUL-terminated DLL name; name is a live NUL-terminated CString.
            let module = unsafe { GetModuleHandleA(module_name) };
            if module.is_null() {
                continue;
            }
            // SAFETY: module is a live loaded module handle and name is NUL-terminated.
            let address = unsafe { GetProcAddress(module, name.as_ptr()) };
            if !address.is_null() {
                return Some(address as usize);
            }
        }
        None
    }
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
    unsafe {
        std::alloc::realloc(
            pointer,
            Layout::from_size_align_unchecked(size, align),
            new_size,
        )
    }
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

#[cfg(target_vendor = "apple")]
pub(crate) mod apple_args {
    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, Ordering};

    static ACTIVE: AtomicBool = AtomicBool::new(false);
    static ARGC: AtomicI32 = AtomicI32::new(0);
    static ARGV: AtomicPtr<*mut libc::c_char> = AtomicPtr::new(std::ptr::null_mut());

    unsafe extern "C" {
        fn _NSGetArgc() -> *mut libc::c_int;
        fn _NSGetArgv() -> *mut *mut *mut libc::c_char;
    }

    pub(super) extern "C" fn get_argc() -> *mut libc::c_int {
        if ACTIVE.load(Ordering::Acquire) {
            ARGC.as_ptr()
        } else {
            // SAFETY: _NSGetArgc returns libdyld's static argc pointer.
            unsafe { _NSGetArgc() }
        }
    }

    pub(super) extern "C" fn get_argv() -> *mut *mut *mut libc::c_char {
        if ACTIVE.load(Ordering::Acquire) {
            ARGV.as_ptr()
        } else {
            // SAFETY: _NSGetArgv returns libdyld's static argv pointer.
            unsafe { _NSGetArgv() }
        }
    }

    /// Scoped override of `_NSGetArgc` / `_NSGetArgv` for JIT-executed programs on Apple hosts.
    pub(crate) struct Guard {
        prev_active: bool,
        prev_argc: libc::c_int,
        prev_argv: *mut *mut libc::c_char,
    }

    impl Guard {
        pub(crate) fn install(argc: libc::c_int, argv: *mut *mut libc::c_char) -> Self {
            let prev_active = ACTIVE.load(Ordering::Acquire);
            let prev_argc = ARGC.swap(argc, Ordering::AcqRel);
            let prev_argv = ARGV.swap(argv, Ordering::AcqRel);
            ACTIVE.store(true, Ordering::Release);
            Self {
                prev_active,
                prev_argc,
                prev_argv,
            }
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            if !self.prev_active {
                ACTIVE.store(false, Ordering::Release);
            }
            ARGC.store(self.prev_argc, Ordering::Release);
            ARGV.store(self.prev_argv, Ordering::Release);
            ACTIVE.store(self.prev_active, Ordering::Release);
        }
    }

    #[cfg(test)]
    mod tests {
        use std::ffi::{CStr, CString};

        use super::*;

        #[test]
        fn guard_overrides_and_restores_apple_argv() {
            // SAFETY: get_argc returns libdyld's live static argc pointer.
            let orig_argc = unsafe { *get_argc() };
            let owned = [
                CString::new("prog").expect("cstring"),
                CString::new("--flag").expect("cstring"),
            ];
            let mut ptrs: Vec<*mut libc::c_char> = owned
                .iter()
                .map(|s| s.as_ptr().cast_mut())
                .chain(std::iter::once(std::ptr::null_mut()))
                .collect();
            {
                let _guard = Guard::install(owned.len() as libc::c_int, ptrs.as_mut_ptr());
                // SAFETY: guard points to live ARGC/ARGV backing `owned` for this scope.
                let (argc, argv) = unsafe { (*get_argc(), *get_argv()) };
                assert_eq!(argc, 2);
                // SAFETY: argv[0] and argv[1] are live NUL-terminated strings in `owned`.
                let first = unsafe { CStr::from_ptr(*argv) }.to_str().expect("utf8");
                let second = unsafe { CStr::from_ptr(*argv.add(1)) }
                    .to_str()
                    .expect("utf8");
                assert_eq!((first, second), ("prog", "--flag"));
            }
            // SAFETY: guard dropped, restoring libdyld's static argc pointer.
            assert_eq!(unsafe { *get_argc() }, orig_argc);
        }
    }
}
