use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = Command::new(rustc).args(["--print", "sysroot"]).output();
    if let Ok(output) = output {
        let sysroot = String::from_utf8_lossy(&output.stdout);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", sysroot.trim());
    }
}
