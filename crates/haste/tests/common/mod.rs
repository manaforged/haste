use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

pub struct FixtureDir(PathBuf);

impl FixtureDir {
    pub fn new(prefix: &str, name: &str) -> Self {
        let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "haste-{prefix}-{name}-{}-{seq}",
            std::process::id()
        ));
        drop(std::fs::remove_dir_all(&dir));
        std::fs::create_dir_all(dir.join("src")).expect("create fixture src");
        Self(dir)
    }

    pub fn write_crate(prefix: &str, name: &str, pkg: &str, main_rs: &str) -> Self {
        let dir = Self::new(prefix, name);
        std::fs::write(
            dir.path().join("Cargo.toml"),
            format!("[package]\nname = \"{pkg}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n"),
        )
        .expect("write Cargo.toml");
        std::fs::write(
            dir.path().join("Cargo.lock"),
            format!("version = 4\n\n[[package]]\nname = \"{pkg}\"\nversion = \"0.1.0\"\n"),
        )
        .expect("write Cargo.lock");
        std::fs::write(dir.path().join("src/main.rs"), main_rs).expect("write main.rs");
        dir
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}
