use std::io::Write;

fn down(n: u64) -> u64 {
    let pad = [n; 64];
    if pad[n as usize % 64] == u64::MAX { 0 } else { down(n + 1) + pad[3] }
}

fn main() {
    println!("ready");
    if std::io::stdout().flush().is_err() {
        std::process::exit(3);
    }
    println!("{}", down(0));
}
