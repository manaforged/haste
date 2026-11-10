struct Attr<T> {
    values: Vec<T>,
}

impl<T> Attr<T> {
    fn none() -> Self {
        Attr { values: Vec::new() }
    }

    fn insert(&mut self, value: T) {
        self.values.push(value);
    }

    fn at_most_one(mut self) -> Option<T> {
        self.values.pop()
    }

    fn get(self) -> Vec<T> {
        self.values
    }
}

fn split<T, F, R>(input: &str, f: F) -> Result<(Attr<T>, Attr<T>), String>
where
    T: Clone,
    F: Fn(&str) -> Result<R, String>,
    R: Into<Option<T>>,
{
    let (mut ser, mut de) = (Attr::none(), Attr::none());
    for part in input.split(',') {
        if let Some(value) = f(part)?.into() {
            ser.insert(value.clone());
            de.insert(value);
        }
    }
    Ok((ser, de))
}

fn word(part: &str) -> Result<Option<String>, String> {
    Ok((!part.is_empty()).then(|| part.to_uppercase()))
}

fn renames(input: &str) -> Result<(Option<String>, Option<String>), String> {
    let (ser, de) = split(input, word)?;
    Ok((ser.at_most_one(), de.at_most_one()))
}

fn many(input: &str) -> Result<(Option<String>, Vec<String>), String> {
    let (ser, de) = split(input, word)?;
    Ok((ser.at_most_one(), de.get()))
}

fn main() {
    println!("{:?}", renames("a,,b"));
    println!("{:?}", many("x,y"));
}
