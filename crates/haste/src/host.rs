//! The host functions that foreign declarations link to: the allocator shims through the host's allocator, the spin-loop hint as a no-op, `xgetbv` as no extended state enabled, the DSO handle of the host program, the allocation error handler and the panic runtime as an abort, and every other name through the dynamic linker.

mod math;

pub use math::symbol as intrinsic;

use std::alloc::Layout;
use std::ffi::CString;

/// The address of the host function that the foreign function `name` links to, when the host has one.
pub fn symbol(name: &str) -> Option<usize> {
    let shim = match name {
        "__rust_alloc" => alloc as *const () as usize,
        "__rust_alloc_zeroed" => alloc_zeroed as *const () as usize,
        "__rust_dealloc" => dealloc as *const () as usize,
        "__rust_realloc" => realloc as *const () as usize,
        "__rust_no_alloc_shim_is_unstable_v2" | "llvm.x86.sse2.pause" => nothing as *const () as usize,
        "llvm.x86.xgetbv" => no_extended_state as *const () as usize,
        "__dso_handle" => &DSO_HANDLE as *const u8 as usize,
        "__rust_alloc_error_handler" | "__rust_panic_cleanup" | "__rust_start_panic" => abort as *const () as usize,
        _ => return math::symbol(name).or_else(|| dynamic(name)),
    };
    Some(shim)
}

/// The object whose address stands for the host program where the linker's hidden `__dso_handle` would, which the C runtime reads as the main program.
static DSO_HANDLE: u8 = 0;

fn dynamic(name: &str) -> Option<usize> {
    let name = CString::new(name).ok()?;
    // SAFETY: name is a NUL-terminated string that lives across the call.
    let address = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
    (!address.is_null()).then_some(address as usize)
}

extern "C" fn no_extended_state(_register: u32) -> u64 {
    0
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

extern "C" fn abort() -> ! {
    std::process::abort()
}
