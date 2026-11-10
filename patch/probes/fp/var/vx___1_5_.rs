fn main() {
    let table: toml::Table = "x = 1.5".parse().unwrap();
    println!("{}", table.len());
}
