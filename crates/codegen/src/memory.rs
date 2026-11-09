
const CHUNK: usize = 64 << 20;
const ALIGN: usize = 16;

struct Chunk {
    write: usize,
    exec: usize,
    size: usize,
    used: usize,
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

impl Memory {
    /// Copies `code` into code memory and returns the address it runs at.
    pub fn write(&mut self, code: &[u8]) -> Result<usize, String> {
        let owner = this_process();
        let fits = |chunk: &Chunk| chunk.owner == owner && chunk.used.next_multiple_of(ALIGN) + code.len() <= chunk.size;
        if !self.chunks.last().is_some_and(fits) {
            if self.sealed {
                return Err("the code window of this worker is full".to_owned());
            }
            self.chunks.push(map(code.len().max(CHUNK).next_multiple_of(page()), owner)?);
        }
        let chunk = self.chunks.last_mut().ok_or("no code memory")?;
        let at = chunk.used.next_multiple_of(ALIGN);
        // SAFETY: at + len is inside this chunk's writable mapping, past every byte written before, and code does not overlap it.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), (chunk.write + at) as *mut u8, code.len()) };
        chunk.used = at + code.len();
        Ok(chunk.exec + at)
    }

    /// Maps a chunk of `size` bytes now, so that processes forked after this call write into the same pages.
    pub fn reserve(&mut self, size: usize) -> Result<(), String> {
        self.chunks.push(map(size.next_multiple_of(page()), this_process())?);
        Ok(())
    }

    /// Stops writing into the last chunk, whose free space forked workers filled, so that the next write maps a new chunk.
    pub fn close(&mut self) {
        if let Some(chunk) = self.chunks.last_mut() {
            chunk.used = chunk.size;
        }
    }

    /// Gives this forked process part `part` of `parts` of the free space of the last chunk, and no other chunk.
    pub fn split(&mut self, parts: usize, part: usize) {
        self.sealed = true;
        if let Some(chunk) = self.chunks.last_mut() {
            let (from, to) = window((chunk.used, chunk.size), parts, part);
            (chunk.used, chunk.size, chunk.owner) = (from, to, this_process());
        }
    }
}

fn window((used, size): (usize, usize), parts: usize, part: usize) -> (usize, usize) {
    let share = ((size - used) / parts.max(1)) & !(ALIGN - 1);
    let from = used.next_multiple_of(ALIGN) + share * part;
    (from, from + share)
}

/// Read-write memory at the same addresses in the processes forked after it is made, handed out by a bump pointer; shared memory holds what workers write for the parent, and private memory what each run of a program writes for itself.
pub struct Shared {
    base: usize,
    used: usize,
    size: usize,
    split: bool,
    private: bool,
}

impl Shared {
    /// Maps `size` bytes of shared memory, which the kernel backs only where they are written.
    pub fn new(size: usize) -> Result<Shared, String> {
        Ok(Shared { base: map_data(size, false)?, used: 0, size, split: false, private: false })
    }

    /// Maps `size` bytes that each forked process holds its own copy of, which the kernel backs only where they are written.
    pub fn private(size: usize) -> Result<Shared, String> {
        Ok(Shared { base: map_data(size, true)?, used: 0, size, split: false, private: true })
    }

    /// The address of `len` zeroed bytes aligned to 16, in a new mapping when this one is full and no worker shares it.
    pub fn alloc(&mut self, len: usize) -> Result<usize, String> {
        let at = self.used.next_multiple_of(ALIGN);
        if at + len <= self.size {
            self.used = at + len;
            return Ok(self.base + at);
        }
        if self.split {
            return Err(format!("shared memory is full: {len} bytes past {at} of {}", self.size));
        }
        let size = self.size.max(len.next_multiple_of(page()));
        (self.base, self.used, self.size) = (map_data(size, self.private)?, 0, size);
        self.alloc(len)
    }

    /// Gives this forked process part `part` of `parts` of the free space.
    pub fn split(&mut self, parts: usize, part: usize) {
        (self.used, self.size) = window((self.used, self.size), parts, part);
        self.split = true;
    }

    /// Stops handing out the free space that forked workers filled, so that the next allocation maps new memory.
    pub fn close(&mut self) {
        self.used = self.size;
    }

    /// Maps fresh space of the same size for workers about to fork, keeping the old mapping alive for what it holds.
    pub fn renew(&mut self) -> Result<(), String> {
        (self.base, self.used, self.split) = (map_data(self.size, self.private)?, 0, false);
        Ok(())
    }
}

fn map_data(size: usize, private: bool) -> Result<usize, String> {
    let flags = if private { libc::MAP_PRIVATE } else { libc::MAP_SHARED } | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
    // SAFETY: a new anonymous mapping touches no existing memory.
    let base = unsafe { libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0) };
    if base == libc::MAP_FAILED {
        return Err(format!("mmap: {}", std::io::Error::last_os_error()));
    }
    Ok(base as usize)
}

fn map(size: usize, owner: i32) -> Result<Chunk, String> {
    // SAFETY: the name is a static NUL-terminated string; memfd_create makes an anonymous file with no path.
    let fd = unsafe { libc::memfd_create(c"haste-code".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(format!("memfd_create: {}", std::io::Error::last_os_error()));
    }
    let mapped = views(fd, size);
    // SAFETY: fd is the descriptor opened above; the mappings keep the object alive after it closes.
    unsafe { libc::close(fd) };
    let (write, exec) = mapped?;
    Ok(Chunk { write, exec, size, used: 0, owner })
}

fn views(fd: i32, size: usize) -> Result<(usize, usize), String> {
    // SAFETY: fd is an open shared memory object that this process owns.
    if unsafe { libc::ftruncate(fd, size as libc::off_t) } != 0 {
        return Err(format!("ftruncate: {}", std::io::Error::last_os_error()));
    }
    let view = |prot| {
        // SAFETY: a new shared mapping of the whole object touches no existing memory.
        let base = unsafe { libc::mmap(std::ptr::null_mut(), size, prot, libc::MAP_SHARED, fd, 0) };
        if base == libc::MAP_FAILED { Err(format!("mmap: {}", std::io::Error::last_os_error())) } else { Ok(base as usize) }
    };
    Ok((view(libc::PROT_READ | libc::PROT_WRITE)?, view(libc::PROT_READ | libc::PROT_EXEC)?))
}
