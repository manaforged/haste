use std::path::Path;

const VALUED: &[&str] = &[
    "--cfg",
    "--check-cfg",
    "--crate-type",
    "--crate-name",
    "--edition",
    "--emit",
    "--print",
    "--out-dir",
    "--explain",
    "--target",
    "--force-warn",
    "--cap-lints",
    "--error-format",
    "--json",
    "--color",
    "--diagnostic-width",
    "--remap-path-prefix",
    "--extern",
    "--sysroot",
    "--env-set",
    "-o",
    "-L",
    "-l",
    "-C",
    "-Z",
    "-A",
    "-W",
    "-D",
    "-F",
];

const SHORT: &[&str] = &["-o", "-L", "-l", "-C", "-Z", "-A", "-W", "-D", "-F"];

const DROPPED: &[&str] = &["--emit", "--out-dir", "-o", "--json", "--cap-lints"];

const INCREMENTAL: &str = "incremental";

const DROPPED_CODEGEN: &[&str] = &[INCREMENTAL, "panic"];

const DEFAULT_CODEGEN_UNITS: &str = "codegen-units=16";

const EMIT: &str = "--emit=obj";

/// One command-line option of rustc: its flag and value, or a lone positional argument with no flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opt {
    pub flag: String,
    pub value: Option<String>,
}

/// The options of a rustc command line, after the program name.
pub fn parse(args: &[String]) -> Vec<Opt> {
    let mut opts = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        if let Some((flag, value)) = arg.split_once('=').filter(|(flag, _)| flag.starts_with("--")) {
            opts.push(Opt { flag: flag.to_string(), value: Some(value.to_string()) });
        } else if VALUED.contains(&arg.as_str()) {
            opts.push(Opt { flag: arg.clone(), value: rest.next().cloned() });
        } else if let Some(short) = SHORT.iter().find(|short| arg.starts_with(**short) && arg.len() > short.len()) {
            opts.push(Opt { flag: short.to_string(), value: Some(arg[short.len()..].to_string()) });
        } else {
            opts.push(Opt { flag: arg.clone(), value: None });
        }
    }
    opts
}

/// The value of the last `flag` option.
pub fn last<'o>(opts: &'o [Opt], flag: &str) -> Option<&'o str> {
    opts.iter().rev().find(|opt| opt.flag == flag).and_then(|opt| opt.value.as_deref())
}

/// The value of the last `-C key=value` option for `key`.
pub fn codegen<'o>(opts: &'o [Opt], key: &str) -> Option<&'o str> {
    opts.iter()
        .rev()
        .filter(|opt| opt.flag == "-C")
        .filter_map(|opt| opt.value.as_deref()?.split_once('='))
        .find(|(name, _)| *name == key)
        .map(|(_, value)| value)
}

/// The command line of the prep build `phase` of the crate that `opts` builds: objects into `dir`, aborting on panic, without lints or cargo's artifact messages, with rustc's non-incremental codegen-unit count stated so that emitting objects does not force one unit, and incremental in a directory of its own under cargo's when cargo builds the crate incrementally.
pub fn prep(program: &str, opts: &[Opt], dir: &Path, phase: &str) -> Vec<String> {
    let mut line = vec![program.to_string()];
    for opt in opts.iter().filter(|opt| !dropped(opt)) {
        line.push(opt.flag.clone());
        line.extend(opt.value.clone());
    }
    line.push(EMIT.to_string());
    line.extend(["--out-dir".to_string(), dir.display().to_string()]);
    line.extend(["-C", "panic=abort", "--cap-lints", "allow"].map(String::from));
    if codegen(opts, "codegen-units").is_none() {
        line.extend(["-C", DEFAULT_CODEGEN_UNITS].map(String::from));
    }
    if let Some(cache) = codegen(opts, INCREMENTAL) {
        line.extend(["-C".to_string(), format!("{INCREMENTAL}={}", Path::new(cache).join(phase).display())]);
    }
    line
}

fn dropped(opt: &Opt) -> bool {
    if DROPPED.contains(&opt.flag.as_str()) {
        return true;
    }
    let key = opt.value.as_deref().and_then(|value| value.split('=').next());
    opt.flag == "-C" && key.is_some_and(|key| DROPPED_CODEGEN.contains(&key))
}
