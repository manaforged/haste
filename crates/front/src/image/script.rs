//! Running build scripts: each script's `main` in its own forked child, in the package directory with Cargo's variables set, with its standard output read back.

use std::ffi::CString;
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use super::{Image, wait};

/// One build script to run: its reached `lang_start` and `main`, every function they reach, the variables to set, the directory to run in, and the file its standard error goes to.
pub struct Script {
    pub entry: (usize, usize),
    pub closure: Vec<usize>,
    pub vars: Vec<(String, String)>,
    pub dir: PathBuf,
    pub log: PathBuf,
}

struct Running {
    index: usize,
    pid: libc::pid_t,
    out: std::fs::File,
    text: Vec<u8>,
}

impl Image {
    /// Runs the scripts, at most `jobs` at once, and gives each one's exit status and standard output.
    pub fn run_scripts(&self, scripts: &[Script], jobs: usize) -> Result<Vec<(i32, Vec<u8>)>> {
        let mut results: Vec<Option<(i32, Vec<u8>)>> = scripts.iter().map(|_| None).collect();
        let mut running: Vec<Running> = Vec::new();
        let mut next = 0;
        while next < scripts.len() || !running.is_empty() {
            while next < scripts.len() && running.len() < jobs.max(1) {
                running.push(self.start_script(next, &scripts[next])?);
                next += 1;
            }
            for done in drain(&mut running)? {
                results[done.index] = Some((wait(done.pid), done.text));
            }
        }
        Ok(results.into_iter().map(|result| result.unwrap_or((-1, Vec::new()))).collect())
    }

    fn start_script(&self, index: usize, script: &Script) -> Result<Running> {
        let vars: Vec<(CString, CString)> = script.vars.iter().map(|(key, value)| Ok((CString::new(key.as_str())?, CString::new(value.as_str())?))).collect::<Result<_>>()?;
        let dir = c_path(&script.dir)?;
        let log = std::fs::File::create(&script.log)?;
        let mut ends = [0; 2];
        // SAFETY: ends is a live two-element array for the new descriptors.
        if unsafe { libc::pipe(ends.as_mut_ptr()) } != 0 {
            bail!("pipe failed: {}", std::io::Error::last_os_error());
        }
        let setup = || {
            // SAFETY: in the child, ends[0] is the read end it does not use, ends[1] and the log become its standard output and error, and every string is a live C string made before the fork.
            unsafe {
                libc::close(ends[0]);
                libc::dup2(ends[1], 1);
                libc::dup2(log.as_raw_fd(), 2);
                libc::chdir(dir.as_ptr());
                vars.iter().for_each(|(key, value)| _ = libc::setenv(key.as_ptr(), value.as_ptr(), 1));
            }
        };
        let pid = self.fork_main(script.entry, &script.closure, &["build-script-build".to_owned()], setup);
        // SAFETY: ends[1] is the write end that only the child uses, and ends[0] is the read end this process owns from here on.
        let out = unsafe {
            libc::close(ends[1]);
            std::fs::File::from_raw_fd(ends[0])
        };
        Ok(Running { index, pid: pid?, out, text: Vec::new() })
    }
}

fn c_path(path: &Path) -> Result<CString> {
    Ok(CString::new(path.as_os_str().as_encoded_bytes())?)
}

/// Reads what the running scripts wrote until at least one ends, and takes out the ones that ended.
fn drain(running: &mut Vec<Running>) -> Result<Vec<Running>> {
    let mut polls: Vec<libc::pollfd> = running.iter().map(|run| libc::pollfd { fd: run.out.as_raw_fd(), events: libc::POLLIN, revents: 0 }).collect();
    // SAFETY: polls is a live array of pollfd with its length.
    if unsafe { libc::poll(polls.as_mut_ptr(), polls.len() as libc::nfds_t, -1) } < 0 {
        bail!("poll failed: {}", std::io::Error::last_os_error());
    }
    let mut ended = Vec::new();
    for (at, poll) in polls.iter().enumerate() {
        if poll.revents == 0 {
            continue;
        }
        let mut chunk = [0u8; 1 << 16];
        let read = running[at].out.read(&mut chunk)?;
        running[at].text.extend_from_slice(&chunk[..read]);
        if read == 0 {
            ended.push(at);
        }
    }
    Ok(ended.into_iter().rev().map(|at| running.swap_remove(at)).collect())
}
