struct Lit(String);

macro_rules! suffixed {
    ($($name:ident => $kind:ident,)*) => ($(
        pub fn $name(n: $kind) -> Lit {
            Lit(format!("{}{}", n.to_string(), stringify!($kind)))
        }
    )*)
}

impl Lit {
    suffixed! {
        i64_suffixed => i64,
        u8_suffixed => u8,
    }
}

fn main() {
    println!("{} {}", Lit::i64_suffixed(40).0, Lit::u8_suffixed(2).0);
}
