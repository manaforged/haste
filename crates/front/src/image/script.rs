//! Running build scripts: each script's `main` in a forked child, in the package directory with Cargo's variables set, with its standard output on the child's output pipe.

use std::ffi::CString;
use std::os::fd::AsRawFd;
use std::path::PathBuf;

use super::Image;

/// One build script to run: its declared `lang_start` and `main`, the variables to set, the directory to run in, and the file its standard error goes to.
pub struct Script {
    pub entry: (usize, usize),
    pub vars: Vec<(String, String)>,
    pub dir: PathBuf,
    pub log: PathBuf,
}

impl Image {
    /// Runs `script` in this forked child with `out` as its standard output, and returns its exit status.
    pub fn enter_script(&self, script: &Script, out: std::fs::File) -> i32 {
        let vars: Result<Vec<(CString, CString)>, std::ffi::NulError> = script.vars.iter().map(|(key, value)| Ok((CString::new(key.as_str())?, CString::new(value.as_str())?))).collect();
        let (Ok(vars), Ok(dir), Ok(log)) = (vars, CString::new(script.dir.as_os_str().as_encoded_bytes()), std::fs::File::create(&script.log)) else { return 101 };
        // SAFETY: this is the forked child, which owns out and log; every string is a live C string.
        unsafe {
            libc::dup2(out.as_raw_fd(), 1);
            libc::dup2(log.as_raw_fd(), 2);
            libc::chdir(dir.as_ptr());
            vars.iter().for_each(|(key, value)| _ = libc::setenv(key.as_ptr(), value.as_ptr(), 1));
        }
        drop((out, log));
        self.enter_main(script.entry, &["build-script-build".to_owned()])
    }
}
