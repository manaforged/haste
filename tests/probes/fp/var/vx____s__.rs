fn main() {
    let table: toml::Table = "x = \"s\"".parse().unwrap();
    println!("{}", table.len());
}
