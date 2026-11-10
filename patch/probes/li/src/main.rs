fn depth_of(text: &str) -> (u8, u8) {
    #[derive(Debug, Default, Clone, Copy)]
    struct Depth {
        structure: u8,
        array: u8,
    }

    impl Depth {
        fn inc_structure(mut self) -> Self {
            self.structure += 1;
            self
        }
    }

    let mut depth = Depth::default();
    for byte in text.bytes() {
        match byte {
            b'(' => depth = depth.inc_structure(),
            b'a' => depth.array += 1,
            _ => {}
        }
    }
    println!("{depth:?}");
    (depth.structure, depth.array)
}

mod fold {
    pub trait Fold {
        fn fold_name(&mut self, name: String) -> String {
            name
        }

        fn fold_path(&mut self, path: Vec<String>) -> Vec<String> {
            let mut out = Vec::new();
            for name in path {
                out.push(self.fold_name(name));
            }
            out
        }
    }
}

fn rename(from: &str, path: &[&str], to: &str) -> Vec<String> {
    use fold::Fold;
    struct Replace<'a>(&'a str, String);

    impl Fold for Replace<'_> {
        fn fold_name(&mut self, name: String) -> String {
            if name == self.0 { self.1.clone() } else { name }
        }
    }
    Replace(from, to.to_string()).fold_path(path.iter().map(|part| part.to_string()).collect())
}

fn main() {
    println!("{:?}", rename("a", &["x", "a", "a"], "b"));
    println!("{:?}", depth_of("(a(s)a)"));
}
