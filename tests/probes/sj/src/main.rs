use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Record {
    id: u64,
    name: String,
    tags: Vec<String>,
    scores: Vec<i32>,
    active: bool,
}

fn main() {
    let raw = r#"{"id":42,"name":"haste","tags":["jit","aot","ssa"],"scores":[10,20,30],"active":true}"#;
    let mut rec: Record = serde_json::from_str(raw).expect("parse json");
    rec.id += 1;
    rec.tags.push("rust".to_owned());
    let total: i32 = rec.scores.iter().sum();
    let encoded = serde_json::to_string(&rec).expect("serialize json");
    let reparsed: Record = serde_json::from_str(&encoded).expect("reparse json");
    assert_eq!(rec, reparsed);
    println!("{total} {encoded}");
}
