use crate::{Error, ErrorKind};

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
    /// Copies `code` into code memory and returns the address it runs at.
    pub fn write(&mut self, code: &[u8]) -> Result<usize, Error> {
        let owner = this_process();
        let taken = self.chunks.last_mut().filter(|chunk| chunk.owner == owner).and_then(|chunk| Some((chunk.bump.take(code.len())?, chunk.write, chunk.exec)));
        let (at, write, exec) = match taken {
            Some(found) => found,
            None if self.sealed => return Err(memory_error("the code window of this worker is full".to_owned())),
            None => {
                self.chunks.push(map(code.len().max(CHUNK).next_multiple_of(page()), owner)?);
                return self.write(code);
            }
        };
        // SAFETY: at + len is inside this chunk's writable mapping, past every byte written before, and code does not overlap it.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), (write + at) as *mut u8, code.len()) };
        Ok(exec + at)
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
        let base = LOW_HINTS.map(|hint| map_low(size, hint)).into_iter().flatten().next();
        Ok(Arena::over(base.map_or_else(|| map_data(size, Sharing::Shared), Ok)?, size, Sharing::Shared))
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

fn map_low(size: usize, hint: usize) -> Option<usize> {
    let flags = libc::MAP_SHARED | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED_NOREPLACE;
    // SAFETY: MAP_FIXED_NOREPLACE fails rather than replace an existing mapping, so the new mapping touches no existing memory.
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

fn map(size: usize, owner: i32) -> Result<Chunk, Error> {
    // SAFETY: the name is a static NUL-terminated string; memfd_create makes an anonymous file with no path.
    let fd = unsafe { libc::memfd_create(c"haste-code".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(memory_error(format!("memfd_create: {}", std::io::Error::last_os_error())));
    }
    let mapped = views(fd, size);
    // SAFETY: fd is the descriptor opened above; the mappings keep the object alive after it closes.
    unsafe { libc::close(fd) };
    let (write, exec) = mapped?;
    Ok(Chunk { write, exec, len: size, bump: Bump::new(size), owner })
}

fn views(fd: i32, size: usize) -> Result<(usize, usize), Error> {
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
