enum Style {
    Unit,
    Newtype,
    Tuple(usize),
}

fn names(style: &Style) -> usize {
    let fields: &[_] = &match style {
        Style::Unit => {
            if std::env::args().count() > 99 {
                vec![]
            } else {
                unreachable!()
            }
        }
        Style::Newtype => vec![String::from("__field0")],
        Style::Tuple(len) => (0..*len).map(|i| format!("__field{i}")).collect(),
    };
    println!("{fields:?}");
    fields.len()
}

fn main() {
    let total = names(&Style::Newtype) + names(&Style::Tuple(3));
    let text: &str = &String::from("owned");
    let bytes: &[u8] = &vec![1, 2, 3];
    println!("{total} {text} {bytes:?}");
}
