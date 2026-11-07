const CHUNK: usize = 64 << 20;
const ALIGN: usize = 16;

struct Chunk {
    base: usize,
    size: usize,
    used: usize,
    sealed: usize,
}

/// Memory that holds generated code: written first, then made executable by `publish`.
#[derive(Default)]
pub struct Memory {
    chunks: Vec<Chunk>,
}

fn page() -> usize {
    // SAFETY: sysconf only reads a system constant.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

impl Memory {
    /// Copies `code` into writable memory and returns its address, which runs after the next `publish`.
    pub fn write(&mut self, code: &[u8]) -> Result<usize, String> {
        let fits = |chunk: &Chunk| chunk.used.next_multiple_of(ALIGN) + code.len() <= chunk.size;
        if !self.chunks.last().is_some_and(fits) {
            self.chunks.push(map(code.len().max(CHUNK).next_multiple_of(page()))?);
        }
        let chunk = self.chunks.last_mut().ok_or("no code memory")?;
        let at = chunk.used.next_multiple_of(ALIGN);
        // SAFETY: at + len is inside this chunk's writable mapping, past every published byte, and code does not overlap it.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), (chunk.base + at) as *mut u8, code.len()) };
        chunk.used = at + code.len();
        Ok(chunk.base + at)
    }

    /// Makes every written byte executable; later writes start on a fresh page.
    pub fn publish(&mut self) -> Result<(), String> {
        let page = page();
        for chunk in &mut self.chunks {
            let end = chunk.used.next_multiple_of(page).min(chunk.size);
            if end > chunk.sealed {
                // SAFETY: the range is whole pages of this chunk's mapping, which only this engine uses.
                let failed = unsafe {
                    libc::mprotect((chunk.base + chunk.sealed) as *mut libc::c_void, end - chunk.sealed, libc::PROT_READ | libc::PROT_EXEC)
                };
                if failed != 0 {
                    return Err(format!("mprotect: {}", std::io::Error::last_os_error()));
                }
                chunk.sealed = end;
            }
            chunk.used = chunk.used.max(end);
        }
        Ok(())
    }
}

fn map(size: usize) -> Result<Chunk, String> {
    // SAFETY: an anonymous private mapping touches no existing memory.
    let base = unsafe {
        libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0)
    };
    if base == libc::MAP_FAILED {
        return Err(format!("mmap: {}", std::io::Error::last_os_error()));
    }
    Ok(Chunk { base: base as usize, size, used: 0, sealed: 0 })
}
