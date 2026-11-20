use std::process::Command;

fn main() {
    println!("cargo:rustc-env=PREP_TARGET={}", std::env::var("TARGET").unwrap_or_default());
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = Command::new(rustc).args(["--print", "sysroot"]).output();
    if let Ok(output) = output {
        let sysroot = String::from_utf8_lossy(&output.stdout);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", sysroot.trim());
        println!("cargo:rustc-env=PREP_SYSROOT={}", sysroot.trim());
    }
}
