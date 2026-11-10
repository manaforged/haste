use pm_derive::{Total, doubled, literals, make_const};

#[derive(Total)]
struct Point {
    x: i64,
    y: i32,
    #[skip]
    z: u8,
}

#[doubled]
fn base(x: i64) -> i64 {
    x + 1
}

#[doubled(5)]
fn other(x: i64) -> i64 {
    x - 4
}

make_const!(SEVEN, 7);

pub fn case_derive_total(input: i64) -> i64 {
    let point = Point { x: input, y: 3, z: 9 };
    point.total() * 10 + Point::field_count() as i64
}

pub fn case_attr_doubled(input: i64) -> i64 {
    base_doubled(input) * 100 + other_doubled(input)
}

pub fn case_bang_const(input: i64) -> i64 {
    SEVEN + input + literals!()
}

fn main() {
    for input in [-5, 0, 1, 3, 20] {
        println!("case_derive_total {input} {}", case_derive_total(input));
        println!("case_attr_doubled {input} {}", case_attr_doubled(input));
        println!("case_bang_const {input} {}", case_bang_const(input));
    }
}
