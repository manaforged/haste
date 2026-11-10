struct T {
    data: [u8; if Self::ENABLE { 4 } else { 0 }],
}

impl T {
    const ENABLE: bool = cfg!(not(opt_level = "s"));
    const WIDE: usize = 3;

    fn new() -> Self {
        let mut data = [0u8; if Self::ENABLE { 4 } else { 0 }];
        let mut at = 0;
        while at < data.len() && Self::ENABLE {
            data[at] = at as u8 + Self::WIDE as u8;
            at += 1;
        }
        T { data }
    }
}

fn main() {
    let t = T::new();
    println!("{:?}", t.data);
    const OFF: bool = false;
    let mut empty = [0u8; if OFF { 4 } else { 0 }];
    let mut at = 0;
    while at < 4 && OFF {
        empty[at] = 1;
        at += 1;
    }
    println!("{:?}", empty);
}
