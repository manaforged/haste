use crate::{Code, Error, ErrorKind};

const CHUNK: usize = 64 << 20;
const ALIGN: usize = 16;

/// A bump pointer over `size` bytes: what is handed out ends at `used`.
#[derive(Clone, Copy)]
struct Bump {
    used: usize,
    size: usize,
}

impl Bump {
    fn new(size: usize) -> Bump {
        Bump { used: 0, size }
    }

    /// The offset of `len` bytes aligned to 16, when they fit.
    fn take(&mut self, len: usize) -> Option<usize> {
        let at = self.used.next_multiple_of(ALIGN);
        (at + len <= self.size).then(|| {
            self.used = at + len;
            at
        })
    }

    /// Keeps only part `part` of `parts` of the free space.
    fn split(&mut self, parts: usize, part: usize) {
        let share = ((self.size - self.used) / parts.max(1)) & !(ALIGN - 1);
        let from = self.used.next_multiple_of(ALIGN) + share * part;
        (self.used, self.size) = (from, from + share);
    }

    /// Hands out nothing more.
    fn close(&mut self) {
        self.used = self.size;
    }
}

/// One mapping of code: written through `write` and run through `exec`, two views of the same pages.
struct Chunk {
    write: usize,
    exec: usize,
    len: usize,
    bump: Bump,
    owner: i32,
}

/// Memory that holds generated code, packed densely: written through one mapping and run through a second, executable mapping of the same pages.
#[derive(Default)]
pub struct Memory {
    chunks: Vec<Chunk>,
    sealed: bool,
}

fn page() -> usize {
    // SAFETY: sysconf only reads a system constant.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

fn this_process() -> i32 {
    // SAFETY: getpid has no preconditions.
    unsafe { libc::getpid() }
}

fn memory_error(message: String) -> Error {
    Error::new(ErrorKind::Memory, message)
}

impl Memory {
    fn reserve_at(&mut self, len: usize) -> Result<(usize, usize), Error> {
        let owner = this_process();
        let taken = self.chunks.last_mut().filter(|chunk| chunk.owner == owner).and_then(|chunk| Some((chunk.bump.take(len)?, chunk.write, chunk.exec)));
        match taken {
            Some((at, write, exec)) => Ok((write + at, exec + at)),
            None if self.sealed => Err(memory_error("the code window of this worker is full".to_owned())),
            None => {
                self.chunks.push(map(len.max(CHUNK).next_multiple_of(page()), owner)?);
                self.reserve_at(len)
            }
        }
    }

    /// Copies `code` into code memory and returns the address it runs at.
    pub fn write(&mut self, code: &[u8]) -> Result<usize, Error> {
        let (write, exec) = self.reserve_at(code.len())?;
        // SAFETY: write..write + len is inside this chunk's writable mapping, past every byte written before, and code does not overlap it.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), write as *mut u8, code.len()) };
        flush(exec, code.len());
        Ok(exec)
    }

    /// Copies `code` into code memory, patches its relocations with `symbols`, and returns its executable address.
    pub fn write_patched(&mut self, code: &Code, symbols: &[u64]) -> Result<usize, Error> {
        let len = code.bytes().len();
        let (write, exec) = self.reserve_at(len)?;
        // SAFETY: write..write + len is a fresh unaliased slice of this chunk's writable mapping.
        let out = unsafe { std::slice::from_raw_parts_mut(write as *mut u8, len) };
        code.patch(symbols, out);
        flush(exec, len);
        Ok(exec)
    }

    /// Reserves up to `max_len` bytes, builds bytes knowing the executable address, and writes them.
    pub fn write_with(&mut self, max_len: usize, build: impl FnOnce(usize) -> Result<Vec<u8>, Error>) -> Result<usize, Error> {
        let (write, exec) = self.reserve_at(max_len)?;
        let bytes = build(exec)?;
        let len = bytes.len().min(max_len);
        // SAFETY: write..write + max_len is inside this chunk's writable mapping and bytes does not overlap it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), write as *mut u8, len) };
        flush(exec, len);
        Ok(exec)
    }

    /// Maps a chunk of `size` bytes now, so that processes forked after this call write into the same pages.
    pub fn reserve(&mut self, size: usize) -> Result<(), Error> {
        self.chunks.push(map(size.next_multiple_of(page()), this_process())?);
        Ok(())
    }

    /// The addresses that the code of the last chunk runs at.
    pub fn span(&self) -> Option<std::ops::Range<usize>> {
        self.chunks.last().map(|chunk| chunk.exec..chunk.exec + chunk.bump.size)
    }

    /// Stops writing into the last chunk, whose free space forked workers filled, so that the next write maps a new chunk.
    pub fn close(&mut self) {
        if let Some(chunk) = self.chunks.last_mut() {
            chunk.bump.close();
        }
    }

    /// Gives this forked process part `part` of `parts` of the free space of the last chunk, and no other chunk.
    pub fn split(&mut self, parts: usize, part: usize) {
        self.sealed = true;
        if let Some(chunk) = self.chunks.last_mut() {
            chunk.bump.split(parts, part);
            chunk.owner = this_process();
        }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        for chunk in &self.chunks {
            unmap(chunk.write, chunk.len);
            unmap(chunk.exec, chunk.len);
        }
    }
}

/// How the pages of an `Arena` are mapped.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sharing {
    Shared,
    Private,
}

/// Read-write memory at the same addresses in the processes forked after it is made, handed out by a bump pointer; shared memory holds what workers write for the parent, and private memory what each run of a program writes for itself.
pub struct Arena {
    base: usize,
    bump: Bump,
    split: bool,
    sharing: Sharing,
    maps: Vec<(usize, usize)>,
}

impl Arena {
    /// Maps `size` bytes of shared memory, which the kernel backs only where they are written.
    pub fn shared(size: usize) -> Result<Arena, Error> {
        Ok(Arena::over(map_data(size, Sharing::Shared)?, size, Sharing::Shared))
    }

    /// Maps `size` bytes that each forked process holds its own copy of, which the kernel backs only where they are written.
    pub fn private(size: usize) -> Result<Arena, Error> {
        Ok(Arena::over(map_data(size, Sharing::Private)?, size, Sharing::Private))
    }

    /// Maps `size` bytes of shared memory below 2 GiB when an address there is free, so code reaches it with a 32-bit absolute address.
    pub fn low(size: usize) -> Result<Arena, Error> {
        let base = LOW_HINTS.into_iter().find_map(|hint| map_low(size, hint));
        Ok(Arena::over(base.map_or_else(|| map_data(size, Sharing::Shared), Ok)?, size, Sharing::Shared))
    }

    /// Whether the entire initial mapping of this arena lies below 2 GiB.
    pub fn is_low(&self) -> bool {
        self.base > 0 && self.base.saturating_add(self.bump.size) <= (1 << 31)
    }

    fn over(base: usize, size: usize, sharing: Sharing) -> Arena {
        Arena { base, bump: Bump::new(size), split: false, sharing, maps: vec![(base, size)] }
    }

    /// The address of `len` zeroed bytes aligned to 16, in a new mapping when this one is full and no worker shares it.
    pub fn alloc(&mut self, len: usize) -> Result<usize, Error> {
        if let Some(at) = self.bump.take(len) {
            return Ok(self.base + at);
        }
        if self.split {
            return Err(memory_error(format!("shared memory is full: {len} bytes past {} of {}", self.bump.used, self.bump.size)));
        }
        self.remap(self.bump.size.max(len.next_multiple_of(page())))?;
        self.alloc(len)
    }

    /// Gives this forked process part `part` of `parts` of the free space.
    pub fn split(&mut self, parts: usize, part: usize) {
        self.bump.split(parts, part);
        self.split = true;
    }

    /// Stops handing out the free space that forked workers filled, so that the next allocation maps new memory.
    pub fn close(&mut self) {
        self.bump.close();
    }

    /// Maps fresh space of the same size for workers about to fork, keeping the old mapping alive for what it holds.
    pub fn renew(&mut self) -> Result<(), Error> {
        self.remap(self.bump.size)?;
        self.split = false;
        Ok(())
    }

    fn remap(&mut self, size: usize) -> Result<(), Error> {
        self.base = map_data(size, self.sharing)?;
        self.bump = Bump::new(size);
        self.maps.push((self.base, size));
        Ok(())
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        self.maps.iter().for_each(|&(base, size)| unmap(base, size));
    }
}

/// Addresses `Arena::low` tries in turn, each well above the null page and the start of a non-relocated program.
const LOW_HINTS: [usize; 6] = [0x1000_0000, 0x2000_0000, 0x3000_0000, 0x4000_0000, 0x5000_0000, 0x6000_0000];

fn unmap(base: usize, size: usize) {
    // SAFETY: base and size name a mapping this process made and still owns, and its owner is being dropped, so nothing reads it after this.
    unsafe { libc::munmap(base as *mut libc::c_void, size) };
}

/// Asks the kernel to fail rather than move the mapping when the hint is taken, where it can.
#[cfg(target_os = "linux")]
const NO_REPLACE: i32 = libc::MAP_FIXED_NOREPLACE;
#[cfg(not(target_os = "linux"))]
const NO_REPLACE: i32 = 0;

fn map_low(size: usize, hint: usize) -> Option<usize> {
    let flags = libc::MAP_SHARED | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | NO_REPLACE;
    // SAFETY: without MAP_FIXED the kernel never replaces an existing mapping, so the new mapping touches no existing memory.
    let base = unsafe { libc::mmap(hint as *mut libc::c_void, size, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0) };
    if base as usize == hint && hint + size <= 1 << 31 {
        return Some(hint);
    }
    if base != libc::MAP_FAILED {
        unmap(base as usize, size);
    }
    None
}

fn map_data(size: usize, sharing: Sharing) -> Result<usize, Error> {
    let flags = if sharing == Sharing::Private { libc::MAP_PRIVATE } else { libc::MAP_SHARED } | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
    // SAFETY: a new anonymous mapping touches no existing memory.
    let base = unsafe { libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0) };
    if base == libc::MAP_FAILED {
        return Err(memory_error(format!("mmap: {}", std::io::Error::last_os_error())));
    }
    Ok(base as usize)
}

/// Maps a chunk of `size` bytes as two views of the same pages, each shared with forked processes.
fn map(size: usize, owner: i32) -> Result<Chunk, Error> {
    let (write, exec) = views(size)?;
    Ok(Chunk { write, exec, len: size, bump: Bump::new(size), owner })
}

#[cfg(target_os = "linux")]
fn views(size: usize) -> Result<(usize, usize), Error> {
    // SAFETY: the name is a static NUL-terminated string; memfd_create makes an anonymous file with no path.
    let fd = unsafe { libc::memfd_create(c"haste-code".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(memory_error(format!("memfd_create: {}", std::io::Error::last_os_error())));
    }
    let mapped = file_views(fd, size);
    // SAFETY: fd is the descriptor opened above; the mappings keep the object alive after it closes.
    unsafe { libc::close(fd) };
    mapped
}

#[cfg(target_os = "linux")]
fn file_views(fd: i32, size: usize) -> Result<(usize, usize), Error> {
    // SAFETY: fd is an open shared memory object that this process owns.
    if unsafe { libc::ftruncate(fd, size as libc::off_t) } != 0 {
        return Err(memory_error(format!("ftruncate: {}", std::io::Error::last_os_error())));
    }
    let view = |prot| {
        // SAFETY: a new shared mapping of the whole object touches no existing memory.
        let base = unsafe { libc::mmap(std::ptr::null_mut(), size, prot, libc::MAP_SHARED, fd, 0) };
        if base == libc::MAP_FAILED { Err(memory_error(format!("mmap: {}", std::io::Error::last_os_error()))) } else { Ok(base as usize) }
    };
    Ok((view(libc::PROT_READ | libc::PROT_WRITE)?, view(libc::PROT_READ | libc::PROT_EXEC)?))
}

/// Darwin runs no code from a shared file mapping, so the executable view is a second mapping of anonymous memory, remapped in the kernel; both views stay shared across `fork`.
#[cfg(target_vendor = "apple")]
fn views(size: usize) -> Result<(usize, usize), Error> {
    use mach::*;
    let check = |what: &str, code: i32| if code == 0 { Ok(()) } else { Err(memory_error(format!("{what}: kernel error {code}"))) };
    let (mut write, mut exec, mut current, mut max) = (0_u64, 0_u64, 0, 0);
    // SAFETY: each call takes this task's own port and the addresses it hands back; the remap aliases only the region just allocated.
    unsafe {
        let task = mach_task_self_;
        check("mach_vm_allocate", mach_vm_allocate(task, &mut write, size as u64, VM_FLAGS_ANYWHERE))?;
        check("mach_vm_inherit", mach_vm_inherit(task, write, size as u64, VM_INHERIT_SHARE))?;
        check("mach_vm_remap", mach_vm_remap(task, &mut exec, size as u64, 0, VM_FLAGS_ANYWHERE, task, write, 0, &mut current, &mut max, VM_INHERIT_SHARE))?;
        check("mach_vm_protect", mach_vm_protect(task, exec, size as u64, 0, VM_PROT_READ | VM_PROT_EXECUTE))?;
    }
    Ok((write as usize, exec as usize))
}

#[cfg(target_vendor = "apple")]
mod mach {
    pub const VM_FLAGS_ANYWHERE: i32 = 1;
    pub const VM_INHERIT_SHARE: u32 = 0;
    pub const VM_PROT_READ: i32 = 1;
    pub const VM_PROT_EXECUTE: i32 = 4;
    unsafe extern "C" {
        pub static mach_task_self_: u32;
        pub fn mach_vm_allocate(task: u32, address: *mut u64, size: u64, flags: i32) -> i32;
        pub fn mach_vm_inherit(task: u32, address: u64, size: u64, inherit: u32) -> i32;
        pub fn mach_vm_protect(task: u32, address: u64, size: u64, set_maximum: u32, protection: i32) -> i32;
        #[allow(clippy::too_many_arguments, reason = "the declaration mirrors the eleven-argument Mach call")]
        pub fn mach_vm_remap(target: u32, address: *mut u64, size: u64, mask: u64, flags: i32, source: u32, source_address: u64, copy: u32, current: *mut i32, max: *mut i32, inherit: u32) -> i32;
        pub fn sys_icache_invalidate(start: *mut std::ffi::c_void, len: usize);
    }
}

/// Makes `len` bytes of code just written at `exec` visible to instruction fetch, which arm64 does not keep coherent with stores.
fn flush(exec: usize, len: usize) {
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    // SAFETY: exec..exec + len is code memory this process maps.
    unsafe {
        mach::sys_icache_invalidate(exec as *mut std::ffi::c_void, len)
    };
    #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
    {
        unsafe extern "C" {
            fn __clear_cache(start: *mut u8, end: *mut u8);
        }
        // SAFETY: exec..exec + len is code memory this process maps.
        unsafe { __clear_cache(exec as *mut u8, (exec + len) as *mut u8) };
    }
    let _ = (exec, len);
}
