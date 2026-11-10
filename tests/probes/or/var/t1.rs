mod token {
    #[derive(Debug, Default)]
    pub struct Comma;
    #[derive(Debug, Default)]
    pub struct Eq(pub u8);
}

macro_rules! Tok {
    [,] => { crate::token::Comma };
    [=] => { crate::token::Eq };
}

#[derive(Debug, Default)]
struct Pair {
    comma: Tok![,],
    eq: Tok![=],
}

fn make(n: u8) -> Tok![=] {
    token::Eq(n)
}

fn main() {
    let pair = Pair { comma: token::Comma, eq: make(7) };
    let other: Vec<Tok![,]> = vec![Default::default()];
    println!("{pair:?} {other:?}");
}
