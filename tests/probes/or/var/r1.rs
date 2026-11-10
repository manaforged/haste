#[derive(Debug)]
enum Mode<E> {
    Back(E),
    Inc(u8),
    Cut(E),
}

fn pick(e: Vec<String>, o: Vec<String>) -> Vec<String> {
    drop(e);
    o
}

fn or(a: Mode<Vec<String>>, b: Mode<Vec<String>>) -> Mode<Vec<String>> {
    match (a, b) {
        (Mode::Back(e), Mode::Back(o)) => Mode::Back(pick(e, o)),
        (Mode::Inc(e), _) | (_, Mode::Inc(e)) => Mode::Inc(e),
        (Mode::Cut(e), _) | (_, Mode::Cut(e)) => Mode::Cut(e),
    }
}

fn main() {
    let a = Mode::Back(vec![String::from("x")]);
    let b = Mode::Back(vec![String::from("y")]);
    println!("{:?}", or(a, b));
    println!("{:?}", or(Mode::Cut(vec![String::from("c")]), Mode::Back(vec![String::from("d")])));
    println!("{:?}", or(Mode::Back(vec![String::from("e")]), Mode::Cut(vec![String::from("f")])));
    println!("{:?}", or(Mode::Inc(1), Mode::Cut(vec![String::from("g")])));
}
