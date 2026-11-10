struct Shown<'a> {
    r#enum: &'a str,
    r#type: usize,
}

fn show(r#enum: &str, prefix: bool) -> String {
    let r#enum = if prefix { format!("E::{}", r#enum) } else { r#enum.to_string() };
    let r#type = r#enum.len();
    let shown = Shown { r#enum: &r#enum, r#type };
    let Shown { r#enum: name, r#type: len } = shown;
    let text = format!("{name} {len} {}", shown.r#type);
    let pair = Pair { r#enum };
    format!("{text} {}", pair.r#enum)
}

fn main() {
    println!("{} | {}", show("A", true), show("B", false));
}

struct Pair {
    r#enum: String,
}
