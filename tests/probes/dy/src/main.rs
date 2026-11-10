use std::fmt;

trait Expected {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result;
}

impl fmt::Display for dyn Expected + '_ {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        Expected::fmt(self, formatter)
    }
}

struct Number;

impl Expected for Number {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a number")
    }
}

fn describe(exp: &dyn Expected) -> String {
    format!("expected {}", exp)
}

fn main() {
    println!("{}", describe(&Number));
}
