fn main() {
    let value: serde_json::Value = toml::from_str("name = \"haste\"\ncount = 3\n").unwrap();
    println!("{value}");
}
