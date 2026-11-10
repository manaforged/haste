fn main() {
    let table: toml::Table = "x = -3".parse().unwrap();
    println!("{}", table.len());
}
