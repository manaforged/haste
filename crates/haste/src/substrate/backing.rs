#[cfg(unix)]
use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Read as _, Write as _};
#[cfg(unix)]
use std::os::fd::AsRawFd as _;
#[cfg(unix)]
use std::os::unix::ffi::{OsStrExt as _, OsStringExt as _};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::ptr::NonNull;

#[cfg(unix)]
use anyhow::anyhow;
use anyhow::{Context as _, Result, bail};

const COMPRESSED_MAGIC: [u8; 8] = *b"HSTZ0002";
const COMPRESSED_HEADER: usize = 16;

/// Atomically writes `chunks` compressed with fast LZ4 (`HSTZ0002` framing) to `path`.
pub(crate) fn write_compressed(path: &Path, chunks: &[&[u8]]) -> Result<()> {
    let raw_len: usize = chunks.iter().map(|c| c.len()).sum();
    let compressed = if chunks.len() == 1 {
        lz4_flex::block::compress(chunks[0])
    } else {
        let mut combined = Vec::with_capacity(raw_len);
        for chunk in chunks {
            combined.extend_from_slice(chunk);
        }
        lz4_flex::block::compress(&combined)
    };
    crate::graph::store::write_new_with(path, |writer| {
        writer.write_all(&COMPRESSED_MAGIC)?;
        writer.write_all(&(raw_len as u64).to_le_bytes())?;
        writer.write_all(&compressed)
    })
    .with_context(|| format!("writing {}", path.display()))
}

fn try_open_compressed(file: &mut File, len_u64: u64, min_len: usize) -> Result<Option<Box<[u8]>>> {
    if len_u64 < COMPRESSED_HEADER as u64 {
        return Ok(None);
    }
    let mut hdr = [0u8; COMPRESSED_HEADER];
    file.read_exact(&mut hdr)?;
    if hdr[..8] != COMPRESSED_MAGIC {
        return Ok(None);
    }
    let raw_len = usize::try_from(u64::from_le_bytes([
        hdr[8], hdr[9], hdr[10], hdr[11], hdr[12], hdr[13], hdr[14], hdr[15],
    ]))
    .context("compressed backing size exceeds address space")?;
    if raw_len < min_len {
        bail!("backing file smaller than header");
    }
    let payload_len = usize::try_from(len_u64 - COMPRESSED_HEADER as u64)
        .context("compressed payload exceeds address space")?;
    let mut payload = Vec::with_capacity(payload_len);
    file.read_to_end(&mut payload)?;
    let decoded = lz4_flex::block::decompress(&payload, raw_len)
        .map_err(|err| anyhow::anyhow!("decompress backing failed: {err}"))?;
    if decoded.len() != raw_len {
        bail!("decompressed backing length mismatch");
    }
    Ok(Some(decoded.into_boxed_slice()))
}

#[cfg(unix)]
pub(crate) enum Backing {
    Mmap { ptr: NonNull<u8>, len: usize },
    Owned(Box<[u8]>),
}

#[cfg(unix)]
// SAFETY: `Backing` holds either a read-only immutable mapping or an owned immutable byte slice.
unsafe impl Send for Backing {}
#[cfg(unix)]
// SAFETY: `Backing` is immutable for its entire lifetime and safe to read concurrently.
unsafe impl Sync for Backing {}

#[cfg(unix)]
impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Mmap { ptr, len } = *self {
            // SAFETY: `ptr` and `len` were returned by a successful `libc::mmap` call in `Backing::open`.
            unsafe { if libc::munmap(ptr.as_ptr().cast(), len) != 0 {} }
        }
    }
}

#[cfg(unix)]
impl Backing {
    pub(crate) fn open(path: &Path, min_len: usize) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let meta = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?;
        let len_u64 = meta.len();
        if let Some(decoded) = try_open_compressed(&mut file, len_u64, min_len)? {
            return Ok(Backing::Owned(decoded));
        }
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
        Ok(Backing::Mmap { ptr, len })
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Backing::Mmap { ptr, len } => {
                // SAFETY: `ptr` is valid for `len` bytes for the lifetime of `&self` and never mutated.
                unsafe { std::slice::from_raw_parts(ptr.as_ptr(), *len) }
            }
            Backing::Owned(bytes) => bytes,
        }
    }
}

#[cfg(not(unix))]
pub(crate) struct Backing(Box<[u8]>);

#[cfg(not(unix))]
impl Backing {
    pub(crate) fn open(path: &Path, min_len: usize) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let meta = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?;
        let len_u64 = meta.len();
        if let Some(decoded) = try_open_compressed(&mut file, len_u64, min_len)? {
            return Ok(Backing(decoded));
        }
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
