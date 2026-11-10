fn main() {
    let table: toml::Table = "x = true".parse().unwrap();
    println!("{}", table.len());
}
