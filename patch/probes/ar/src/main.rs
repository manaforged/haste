fn split<'s, T>(s: &'s T) -> Vec<&'s str>
where
    T: AsRef<str>,
{
    let s = s.as_ref();
    if s.is_empty() {
        return Vec::new();
    }
    s.split(' ').collect()
}

fn convert<T: AsRef<str>>(s: T) -> String {
    let words = split(&s);
    words.join("_")
}

fn main() {
    let owned = String::from("a b c");
    println!("{} {} {}", convert(&owned), convert("x y"), convert(owned.clone()));
}
