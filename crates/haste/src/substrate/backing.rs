#[cfg(unix)]
use std::ffi::OsString;
#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::io;
#[cfg(unix)]
use std::os::fd::AsRawFd as _;
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt as _, OsStringExt as _};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::ptr::NonNull;

#[cfg(unix)]
use anyhow::{Context as _, Result, anyhow, bail};
#[cfg(not(unix))]
use anyhow::{Context as _, Result, bail};

#[cfg(unix)]
pub(crate) struct Backing {
    ptr: NonNull<u8>,
    len: usize,
}

#[cfg(unix)]
// SAFETY: `Backing` points to a read-only immutable mapping owned exclusively by `Backing`.
unsafe impl Send for Backing {}
#[cfg(unix)]
// SAFETY: `Backing` is immutable for its entire lifetime and safe to read concurrently.
unsafe impl Sync for Backing {}

#[cfg(unix)]
impl Drop for Backing {
    fn drop(&mut self) {
        // SAFETY: `ptr` and `len` were returned by a successful `libc::mmap` call in `Backing::open`.
        unsafe { if libc::munmap(self.ptr.as_ptr().cast(), self.len) != 0 {} }
    }
}

#[cfg(unix)]
impl Backing {
    pub(crate) fn open(path: &Path, min_len: usize) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let meta = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?;
        let len_u64 = meta.len();
        if len_u64 < min_len as u64 {
            bail!("backing file smaller than header");
        }
        let len = usize::try_from(len_u64).context("backing file exceeds address space")?;
        // SAFETY: `file` is open for reading and `len > 0`; `MAP_SHARED` creates a read-only mapping owned by `Backing`.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(io::Error::last_os_error())
                .with_context(|| format!("mmap {}", path.display()));
        }
        let ptr =
            NonNull::new(raw.cast::<u8>()).ok_or_else(|| anyhow!("mmap returned null pointer"))?;
        Ok(Backing { ptr, len })
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` bytes for the lifetime of `&self` and never mutated.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

#[cfg(not(unix))]
pub(crate) struct Backing(Box<[u8]>);

#[cfg(not(unix))]
impl Backing {
    pub(crate) fn open(path: &Path, min_len: usize) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        if bytes.len() < min_len {
            bail!("backing file smaller than header");
        }
        Ok(Backing(bytes.into_boxed_slice()))
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

pub(crate) fn path_to_bytes(path: &Path) -> Option<Vec<u8>> {
    #[cfg(unix)]
    {
        Some(path.as_os_str().as_bytes().to_vec())
    }
    #[cfg(not(unix))]
    {
        path.to_str().map(|s| s.as_bytes().to_vec())
    }
}

pub(crate) fn bytes_to_path(bytes: &[u8]) -> Option<PathBuf> {
    #[cfg(unix)]
    {
        Some(PathBuf::from(OsString::from_vec(bytes.to_vec())))
    }
    #[cfg(not(unix))]
    {
        std::str::from_utf8(bytes).ok().map(PathBuf::from)
    }
}
