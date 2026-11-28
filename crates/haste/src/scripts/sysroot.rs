use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;

use crate::graph::script::Output;
use crate::graph::{Workspace, store};

const STD_SUPPORTED_OS: &[&str] = &[
    "linux",
    "android",
    "netbsd",
    "dragonfly",
    "openbsd",
    "freebsd",
    "solaris",
    "illumos",
    "macos",
    "ios",
    "tvos",
    "watchos",
    "visionos",
    "windows",
    "fuchsia",
    "motor",
    "hermit",
    "trusty",
    "l4re",
    "redox",
    "haiku",
    "vxworks",
    "espidf",
    "vita",
    "aix",
    "nto",
    "qnx",
    "xous",
    "hurd",
    "uefi",
    "teeos",
    "zkvm",
    "rtems",
    "nuttx",
    "cygwin",
    "vexos",
];

const STD_KNOWN_CFGS: &[&str] = &[
    "rustc-cfg=netbsd10",
    "rustc-cfg=restricted_std",
    "rustc-cfg=backtrace_in_libstd",
    "rustc-cfg=vxworks_lt_25_09",
];

const LIBC_KNOWN_CFGS: &[&str] = &[
    "emscripten_old_stat_abi",
    "espidf_picolibc",
    "espidf_time32",
    "freebsd10",
    "freebsd11",
    "freebsd12",
    "freebsd13",
    "freebsd14",
    "freebsd15",
    "gnu_file_offset_bits64",
    "gnu_time_bits64",
    "libc_deny_warnings",
    "linux_time_bits64",
    "musl_v1_2_3",
    "musl32_time64",
    "musl_redir_time64",
    "vxworks_lt_25_09",
    "libc_pauthtest",
];

const LIBC_FALLBACK_VARS: &[&str] = &[
    "LIBC_CI",
    "LIBC_BUILD_VERBOSE",
    "RUST_LIBC_UNSTABLE_FREEBSD_VERSION",
    "CARGO_CFG_LIBC_UNSTABLE_FREEBSD_VERSION",
    "RUST_LIBC_UNSTABLE_MUSL_V1_2_3",
    "RUST_LIBC_UNSTABLE_LINUX_TIME_BITS64",
    "RUST_LIBC_UNSTABLE_GNU_TIME_BITS",
    "RUST_LIBC_UNSTABLE_GNU_FILE_OFFSET_BITS",
    "CARGO_CFG_LIBC_UNSTABLE_GNU_TIME_BITS",
    "CARGO_CFG_LIBC_UNSTABLE_GNU_FILE_OFFSET_BITS",
    "WIND_RELEASE_ID",
];

const LIBC_WATCHED_ENVS: &[&str] = &[
    "LIBC_BUILD_VERBOSE",
    "RUST_LIBC_UNSTABLE_FREEBSD_VERSION",
    "RUST_LIBC_UNSTABLE_MUSL_V1_2_3",
    "RUST_LIBC_UNSTABLE_LINUX_TIME_BITS64",
    "RUST_LIBC_UNSTABLE_GNU_FILE_OFFSET_BITS",
    "RUST_LIBC_UNSTABLE_GNU_TIME_BITS",
];

const CB_KNOWN_SET_CFGS: &[&str] = &[
    "thumb",
    "thumb_1",
    "thumb1_only",
    "x86_no_sse2",
    "assert_no_panic",
    "intrinsics_enabled",
    "optimizations_enabled",
    "f16_enabled",
    "f128_enabled",
    "mem_unaligned",
    "kernel_user_helpers",
];

pub(super) fn settle_unit(
    workspace: &Workspace,
    unit: usize,
    vars: &[(String, String)],
) -> Option<Output> {
    let entry = &workspace.graph.units[unit];
    let pkg = &workspace.packages[entry.package];
    if !pkg.sysroot {
        return None;
    }
    let build_rel = pkg.manifest.build.as_deref()?;
    let build_path = pkg.manifest.dir.join(build_rel);
    let text = match pkg.locked.name.as_str() {
        "std" => eval_std(&build_path, vars)?,
        "libc" => eval_libc(&build_path, vars)?,
        "compiler_builtins" => {
            eval_compiler_builtins(&build_path, &pkg.manifest.dir, &entry.features, vars)?
        }
        _ => return None,
    };
    let output = Output::parse(&text);
    if !output.errors.is_empty() {
        return None;
    }
    let dir = store::dir(workspace, unit);
    if std::fs::create_dir_all(dir.join(super::OUT)).is_ok() {
        let _ = store::keep(workspace, unit, vars, &text);
    }
    Some(output)
}

fn lookup<'a>(vars: &'a [(String, String)], key: &str) -> Option<&'a str> {
    vars.iter()
        .rev()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

fn env_var(vars: &[(String, String)], key: &str) -> Option<String> {
    lookup(vars, key)
        .map(str::to_owned)
        .or_else(|| crate::process_env_var(key))
}

fn env_flag(vars: &[(String, String)], key: &str) -> bool {
    matches!(env_var(vars, key).as_deref(), Some(val) if val != "0")
}

fn has_unknown_error_or_link(source: &str, allow_openbsd_link: bool) -> bool {
    source.lines().any(|line| {
        if line.contains("cargo:error=") || line.contains("cargo::error=") {
            return true;
        }
        if line.contains("rustc-link-") {
            if allow_openbsd_link
                && (line.contains("rustc-link-search=native=/usr/lib")
                    || line.contains("rustc-link-lib=compiler_rt"))
            {
                return false;
            }
            return true;
        }
        false
    })
}

fn eval_std(build_path: &Path, vars: &[(String, String)]) -> Option<String> {
    let target_arch = lookup(vars, "CARGO_CFG_TARGET_ARCH")?;
    let target_os = lookup(vars, "CARGO_CFG_TARGET_OS")?;
    let target_vendor = lookup(vars, "CARGO_CFG_TARGET_VENDOR")?;
    let target_env = lookup(vars, "CARGO_CFG_TARGET_ENV")?;
    if target_os == "vxworks" {
        return None;
    }
    let source = std::fs::read_to_string(build_path).ok()?;
    if !std_source_known(&source) {
        return None;
    }
    let mut out = String::from("cargo:rerun-if-changed=build.rs\n");
    if target_os == "netbsd" && env_var(vars, "RUSTC_STD_NETBSD10").is_some() {
        out.push_str("cargo:rustc-cfg=netbsd10\n");
    }
    let supported = STD_SUPPORTED_OS.contains(&target_os)
        || (target_vendor == "fortanix" && target_env == "sgx")
        || matches!(target_arch, "wasm32" | "wasm64")
        || target_os.starts_with("solid")
        || (target_vendor == "nintendo" && target_env == "newlib")
        || env_var(vars, "RUSTC_BOOTSTRAP_SYNTHETIC_TARGET").is_some();
    if !supported {
        out.push_str("cargo:rustc-cfg=restricted_std\n");
    }
    out.push_str("cargo:rustc-cfg=backtrace_in_libstd\n");
    let _ = writeln!(out, "cargo:rustc-env=STD_ENV_ARCH={target_arch}");
    Some(out)
}

fn std_source_known(source: &str) -> bool {
    if has_unknown_error_or_link(source, false) {
        return false;
    }
    source.lines().all(|line| {
        if line.contains("rustc-cfg=") && !STD_KNOWN_CFGS.iter().any(|k| line.contains(k)) {
            return false;
        }
        if line.contains("rustc-env=") && !line.contains("rustc-env=STD_ENV_ARCH=") {
            return false;
        }
        true
    })
}

fn eval_libc(build_path: &Path, vars: &[(String, String)]) -> Option<String> {
    let target_os = lookup(vars, "CARGO_CFG_TARGET_OS").unwrap_or_default();
    if matches!(target_os, "emscripten" | "vxworks") {
        return None;
    }
    if LIBC_FALLBACK_VARS
        .iter()
        .any(|key| env_var(vars, key).is_some())
    {
        return None;
    }
    let source = std::fs::read_to_string(build_path).ok()?;
    if !libc_source_known(&source) {
        return None;
    }
    let target_env = lookup(vars, "CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_ptr_width = lookup(vars, "CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default();
    let target_arch = lookup(vars, "CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_abi = lookup(vars, "CARGO_CFG_TARGET_ABI").unwrap_or_default();

    let mut out = String::from("cargo:rerun-if-changed=build.rs\n");
    for var in LIBC_WATCHED_ENVS {
        if source.contains(&format!("rerun-if-env-changed={var}")) {
            let _ = writeln!(out, "cargo:rerun-if-env-changed={var}");
        }
    }
    if target_abi == "pauthtest" {
        out.push_str("cargo:rustc-cfg=libc_pauthtest\n");
    }
    out.push_str("cargo:rustc-cfg=freebsd12\n");
    let musl = target_env == "musl" || target_env == "ohos";
    let musl_v1_2_3 = env_flag(vars, "CARGO_CFG_LIBC_UNSTABLE_MUSL_V1_2_3")
        || matches!(target_arch, "loongarch64" | "hexagon")
        || target_env == "ohos"
        || target_abi == "pauthtest";
    if musl && musl_v1_2_3 {
        out.push_str("cargo:rustc-cfg=musl_v1_2_3\n");
        if target_ptr_width == "32" {
            out.push_str("cargo:rustc-cfg=musl32_time64\n");
            out.push_str("cargo:rustc-cfg=linux_time_bits64\n");
        }
        if matches!(target_arch, "arm" | "mips" | "powerpc" | "x86") {
            out.push_str("cargo:rustc-cfg=musl_redir_time64\n");
        }
    }
    if target_env == "uclibc" && env_flag(vars, "CARGO_CFG_LIBC_UNSTABLE_UCLIBC_TIME64") {
        out.push_str("cargo:rustc-cfg=linux_time_bits64\n");
    }
    Some(out)
}

fn libc_source_known(source: &str) -> bool {
    if has_unknown_error_or_link(source, false) || !source.contains("12 => set_cfg(\"freebsd12\")")
    {
        return false;
    }
    source.lines().all(|line| {
        if line.contains("rustc-env=") {
            return false;
        }
        if line.contains("rustc-cfg=") && !line.contains("cargo:rustc-cfg={cfg}") {
            return false;
        }
        if let Some((_, rest)) = line.split_once("set_cfg(\"") {
            let Some((name, _)) = rest.split_once('"') else {
                return false;
            };
            if !LIBC_KNOWN_CFGS.contains(&name) {
                return false;
            }
        }
        true
    })
}

fn eval_compiler_builtins(
    build_path: &Path,
    manifest_dir: &Path,
    features: &BTreeSet<String>,
    vars: &[(String, String)],
) -> Option<String> {
    if features.contains("c") || env_flag(vars, "LIBM_BUILD_VERBOSE") {
        return None;
    }
    let target = lookup(vars, "TARGET")?;
    let opt_level = lookup(vars, "OPT_LEVEL")?;
    let target_arch = lookup(vars, "CARGO_CFG_TARGET_ARCH")?;
    let target_os = lookup(vars, "CARGO_CFG_TARGET_OS")?;
    let first_part = target.split('-').next()?;
    let configure_path = manifest_dir.join("../libm/configure.rs");
    let build_src = std::fs::read_to_string(build_path).ok()?;
    let configure_src = std::fs::read_to_string(&configure_path).ok()?;
    if !cb_source_known(&build_src, &configure_src) {
        return None;
    }

    let mut out = String::from("cargo:rerun-if-env-changed=LIBM_BUILD_VERBOSE\n");
    if manifest_dir
        .file_name()
        .is_some_and(|n| n == "builtins-shim")
    {
        out.push_str("cargo::rerun-if-changed=../compiler-builtins/build.rs\n");
    } else {
        out.push_str("cargo::rerun-if-changed=build.rs\n");
    }
    out.push_str("cargo::rerun-if-changed=../libm/configure.rs\n");

    let target_features = lookup(vars, "CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let thumb = first_part.starts_with("thumb");
    let x86_no_sse2 = target_arch == "x86" && !target_features.split(',').any(|f| f == "sse2");
    let opt = !matches!(opt_level, "0" | "1");

    push_cfg(&mut out, "thumb", thumb);
    if configure_src.contains("set_cfg(\"thumb1_only\"") {
        let thumb1_only = matches!(
            first_part,
            "thumbv4t" | "thumbv5te" | "thumbv6" | "thumbv6m" | "thumbv8m.base"
        );
        push_cfg(&mut out, "thumb1_only", thumb1_only);
    } else if configure_src.contains("set_cfg(\"thumb_1\"") {
        let thumb_1 = matches!(first_part, "thumbv6m" | "thumbv8m.base");
        push_cfg(&mut out, "thumb_1", thumb_1);
    } else {
        return None;
    }
    push_cfg(&mut out, "x86_no_sse2", x86_no_sse2);
    push_cfg(
        &mut out,
        "assert_no_panic",
        env_flag(vars, "ENSURE_NO_PANIC"),
    );
    push_cfg(&mut out, "intrinsics_enabled", true);
    push_cfg(&mut out, "optimizations_enabled", opt);
    push_cfg(
        &mut out,
        "f16_enabled",
        lookup(vars, "CARGO_CFG_TARGET_HAS_RELIABLE_F16").is_some(),
    );
    push_cfg(
        &mut out,
        "f128_enabled",
        lookup(vars, "CARGO_CFG_TARGET_HAS_RELIABLE_F128").is_some(),
    );

    let _ = writeln!(
        out,
        "cargo:compiler-rt={}",
        manifest_dir.join("compiler-rt").display()
    );
    if (target.contains("wasm") && !target.contains("wasi"))
        || (target.contains("sgx") && target.contains("fortanix"))
        || target.contains("-none")
        || target.contains("nvptx")
        || target.contains("uefi")
        || target.contains("xous")
    {
        out.push_str("cargo:rustc-cfg=feature=\"mem\"\n");
    }
    let mem_unaligned = target_arch.contains("x86_64")
        || target_arch.contains("x86")
        || target_arch.contains("aarch64")
        || target_arch.contains("bpf");
    push_cfg(&mut out, "mem_unaligned", mem_unaligned);
    let kernel_user_helpers =
        matches!(first_part, "armv4t" | "armv5te") || target == "arm-linux-androideabi";
    push_cfg(&mut out, "kernel_user_helpers", kernel_user_helpers);
    if target_os == "openbsd" {
        out.push_str("cargo:rustc-link-search=native=/usr/lib\n");
        out.push_str("cargo:rustc-link-lib=compiler_rt\n");
    }
    Some(out)
}

fn push_cfg(out: &mut String, name: &str, enabled: bool) {
    if enabled {
        let _ = writeln!(out, "cargo:rustc-cfg={name}");
    }
}

fn cb_source_known(build_src: &str, configure_src: &str) -> bool {
    let active_build = build_src
        .split_once("#[cfg(feature = \"c\")]")
        .map_or(build_src, |(head, _)| head);
    if has_unknown_error_or_link(active_build, true)
        || has_unknown_error_or_link(configure_src, false)
    {
        return false;
    }
    let check_lines = |src: &str| {
        src.lines().all(|line| {
            if line.contains("rustc-env=") && !line.contains("CFG_") {
                return false;
            }
            if line.contains("rustc-cfg=")
                && !line.contains("cargo:rustc-cfg={name}")
                && !line.contains(r#"cargo:rustc-cfg=feature=\"mem\""#)
            {
                return false;
            }
            if let Some((_, rest)) = line.split_once("set_cfg(\"") {
                let Some((name, _)) = rest.split_once('"') else {
                    return false;
                };
                if !CB_KNOWN_SET_CFGS.contains(&name) {
                    return false;
                }
            }
            true
        })
    };
    check_lines(active_build) && check_lines(configure_src)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_std_cfg_falls_back() {
        assert!(!std_source_known(
            "println!(\"cargo:rustc-cfg=future_unknown_std_cfg\");\n"
        ));
        assert!(!std_source_known(
            "println!(\"cargo:rustc-env=NEW_STD_VAR=1\");\n"
        ));
    }

    #[test]
    fn unknown_libc_cfg_falls_back() {
        assert!(!libc_source_known(
            "12 => set_cfg(\"freebsd12\"),\nset_cfg(\"future_libc_cfg\");\n"
        ));
    }

    #[test]
    fn unknown_cb_cfg_falls_back() {
        assert!(!cb_source_known("set_cfg(\"future_cb_cfg\", true);\n", ""));
    }

    #[test]
    fn real_sysroot_scripts_settle_synthetically() {
        let Ok(library) = crate::graph::library() else {
            return;
        };
        let _guard = crate::sync::lock(&crate::substrate::TEST_HOME_LOCK);
        let prev_home = std::env::var_os("HASTE_HOME");
        let tmp = std::env::temp_dir().join(format!("haste-sysroot-test-{}", std::process::id()));
        drop(std::fs::remove_dir_all(&tmp));
        std::fs::create_dir_all(&tmp).expect("mkdir");
        // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
        unsafe { std::env::set_var("HASTE_HOME", &tmp) };
        let host = crate::graph::cfg::Target::of(&crate::host_target());
        let targets = crate::graph::Targets {
            target: host.clone(),
            host,
        };
        let workspace = crate::graph::load_sysroot(&library, &targets).expect("load sysroot");
        let mut settled = Vec::new();
        for (unit, entry) in workspace.graph.units.iter().enumerate() {
            let pkg = &workspace.packages[entry.package];
            if pkg.manifest.build.is_none() {
                continue;
            }
            let vars = crate::graph::script::script_vars(&workspace, &targets, unit, &[]);
            let out = settle_unit(&workspace, unit, &vars)
                .unwrap_or_else(|| panic!("expected synthetic settlement for {}", pkg.locked.name));
            settled.push((pkg.locked.name.clone(), out));
        }
        // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HASTE_HOME", v),
                None => std::env::remove_var("HASTE_HOME"),
            }
        }
        drop(std::fs::remove_dir_all(&tmp));
        assert!(settled.iter().any(|(n, _)| n == "std"));
        assert!(settled.iter().any(|(n, _)| n == "libc"));
        assert!(settled.iter().any(|(n, _)| n == "compiler_builtins"));
    }
}
