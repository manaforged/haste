fn main() {
    let table: toml::Table = "x = [1]".parse().unwrap();
    println!("{}", table.len());
}
