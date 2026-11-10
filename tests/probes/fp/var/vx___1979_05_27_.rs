fn main() {
    let table: toml::Table = "x = 1979-05-27".parse().unwrap();
    println!("{}", table.len());
}
