macro_rules! inner {
    ($err:expr) => {{
        let e = String::from("inner");
        format!("{} {}", $err, e)
    }};
}

macro_rules! outer {
    () => {{
        let e = String::from("outer");
        inner!(e)
    }};
}

fn main() {
    println!("{}", outer!());
}
