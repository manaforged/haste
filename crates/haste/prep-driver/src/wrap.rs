use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface;
use rustc_middle::ty::TyCtxt;

use crate::args::{self, Opt};
use crate::{export, record};

const BUG_URL: &str = "https://github.com/rust-lang/rust/issues/new";

/// The environment variable that names the directory prep builds write to; without it the driver only compiles.

/// The subdirectory of a unit that holds the objects of the build that keeps rustc's own linkage, so its code matches a release build.
pub const RELEASE: &str = "release";

/// Where the prep build of one crate writes its objects and manifest, and the crate's package version.
pub struct Unit {
    pub dir: PathBuf,
    pub version: String,
}

struct Plain;

impl Callbacks for Plain {}

/// The release prep compile: rustc's own partitioning, with each static exported so both builds of the crate share one.
struct Released;

impl Callbacks for Released {
    fn config(&mut self, config: &mut interface::Config) {
        config.override_queries = Some(export::override_statics);
    }
}

struct Prep {
    unit: Unit,
}

impl Callbacks for Prep {
    fn config(&mut self, config: &mut interface::Config) {
        config.override_queries = Some(export::override_queries);
    }

    fn after_analysis<'tcx>(&mut self, _: &interface::Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        if let Err(error) = record::write(tcx, &self.unit) {
            tcx.dcx().fatal(format!("prep manifest: {error}"));
        }
        Compilation::Continue
    }
}

/// The name of the prep compile that exports every function instance.
const EXPORTED: &str = "exported";

/// The environment variable that tells a driver process to run one prep compile of its crate and nothing else.
const PHASE: &str = "PREP_PHASE";

/// One prep compile of a crate: as released, into the `release` subdirectory, or with every function instance exported.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Release,
    Exported,
}

impl Phase {
    const ALL: [Phase; 2] = [Phase::Release, Phase::Exported];

    fn name(self) -> &'static str {
        match self {
            Phase::Release => RELEASE,
            Phase::Exported => EXPORTED,
        }
    }
}

/// Compiles as rustc would with the arguments cargo passed, and, when a prep directory is set and the crate is prepped, compiles it twice more in child processes at the same time: once as released into the `release` subdirectory and once with every function instance exported.
pub fn main(args: Vec<String>) -> ExitCode {
    rustc_driver::install_ice_hook(BUG_URL, |_| ());
    let args = compiler_args(args);
    let Some((opts, unit)) = plan(&args) else {
        return rustc_driver::catch_with_exit_code(|| rustc_driver::run_compiler(&args, &mut Plain));
    };
    let phase = std::env::var(PHASE).ok().and_then(|name| Phase::ALL.into_iter().find(|phase| phase.name() == name));
    match phase {
        Some(phase) => compile(&args, &opts, unit, phase),
        None => prep(&args, unit),
    }
}

/// Compiles the crate as cargo asked while both prep compiles run beside it, then keeps only the prep outputs.
fn prep(args: &[String], unit: Unit) -> ExitCode {
    let release = unit.dir.join(RELEASE);
    for dir in [&unit.dir, &release] {
        if let Err(error) = fresh(dir) {
            eprintln!("prep: {}: {error}", dir.display());
            return ExitCode::FAILURE;
        }
    }
    let children: Vec<_> = Phase::ALL.iter().map(|phase| spawn(args, *phase)).collect();
    let code = rustc_driver::catch_with_exit_code(|| rustc_driver::run_compiler(args, &mut Plain));
    let mut failed = false;
    for child in children {
        let status = child.and_then(|mut child| child.wait());
        failed |= !status.as_ref().is_ok_and(std::process::ExitStatus::success);
        if let Err(error) = status {
            eprintln!("prep: {error}");
        }
    }
    if code != ExitCode::SUCCESS || failed {
        return ExitCode::FAILURE;
    }
    for dir in [&unit.dir, &release] {
        if let Err(error) = keep_outputs(dir) {
            eprintln!("prep: {}: {error}", dir.display());
            return ExitCode::FAILURE;
        }
    }
    code
}

/// Starts this driver again on `args` to run the prep compile `phase` alone.
fn spawn(args: &[String], phase: Phase) -> std::io::Result<std::process::Child> {
    std::process::Command::new(std::env::current_exe()?).args(args).env(PHASE, phase.name()).stdin(std::process::Stdio::null()).spawn()
}

/// Runs the prep compile `phase` of the crate that `opts` builds.
fn compile(args: &[String], opts: &[Opt], unit: Unit, phase: Phase) -> ExitCode {
    let program = args.first().map(String::as_str).unwrap_or("rustc");
    match phase {
        Phase::Release => {
            let released = args::prep(program, opts, &unit.dir.join(RELEASE), phase.name());
            rustc_driver::catch_with_exit_code(|| rustc_driver::run_compiler(&released, &mut Released))
        }
        Phase::Exported => {
            let exported = args::prep(program, opts, &unit.dir, phase.name());
            let mut prep = Prep { unit };
            rustc_driver::catch_with_exit_code(|| rustc_driver::run_compiler(&exported, &mut prep))
        }
    }
}

fn compiler_args(mut args: Vec<String>) -> Vec<String> {
    let wrapped = args.get(1).is_some_and(|arg| Path::new(arg).file_stem().is_some_and(|stem| stem == "rustc"));
    if wrapped {
        args.remove(0);
    }
    args
}

fn plan(args: &[String]) -> Option<(Vec<Opt>, Unit)> {
    let root = std::env::var_os(crate::files::PREP_DIR)?;
    let opts = args::parse(args.get(1..)?);
    let name = args::last(&opts, "--crate-name")?;
    if !prepped(&opts, name) {
        return None;
    }
    let extra = args::codegen(&opts, "extra-filename").unwrap_or_default();
    let dir = PathBuf::from(root).join(format!("{name}{extra}"));
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    Some((opts, Unit { dir, version }))
}

fn prepped(opts: &[Opt], name: &str) -> bool {
    let printing = opts.iter().any(|opt| opt.flag == "--print" || opt.flag == "-vV" || opt.flag == "-V");
    let proc_macro = opts.iter().any(|opt| opt.flag == "--crate-type" && opt.value.as_deref() == Some("proc-macro"));
    let input = opts.iter().any(|opt| opt.value.is_none() && !opt.flag.starts_with('-'));
    input && !printing && !proc_macro && !name.starts_with("build_script_")
}

fn keep_outputs(dir: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path.file_name().and_then(|name| name.to_str()).unwrap_or_default();
        if name.ends_with(".o") || name == record::MANIFEST || name == record::LINK || name == RELEASE {
            continue;
        }
        if path.is_dir() { std::fs::remove_dir_all(&path)? } else { std::fs::remove_file(&path)? }
    }
    Ok(())
}

fn fresh(dir: &Path) -> std::io::Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)
}
