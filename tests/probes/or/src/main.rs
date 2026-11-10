use std::cell::Cell;

thread_local! {
    static DROPS: Cell<u32> = const { Cell::new(0) };
}

struct Noisy(&'static str);

impl Drop for Noisy {
    fn drop(&mut self) {
        DROPS.with(|drops| drops.set(drops.get() + 1));
        println!("drop {}", self.0);
    }
}

fn pick(left: Result<u8, Noisy>, right: Result<(), Noisy>) -> Result<u8, Noisy> {
    match (left, right) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(err), _) | (_, Err(err)) => Err(err),
    }
}

fn guarded(pair: (Option<Noisy>, Option<Noisy>), allow: bool) -> &'static str {
    match pair {
        (Some(kept), _) | (None, Some(kept)) if allow => kept.0,
        _ => "none",
    }
}

fn main() {
    for (name, left, right) in [
        ("both ok", Ok(1), Ok(())),
        ("left err", Err(Noisy("left")), Ok(())),
        ("right err", Ok(2), Err(Noisy("right"))),
        ("both err", Err(Noisy("first")), Err(Noisy("second"))),
    ] {
        let result = pick(left, right);
        println!("{name}: {}", match &result { Ok(value) => format!("ok {value}"), Err(err) => format!("err {}", err.0) });
    }
    println!("guard {}", guarded((Some(Noisy("a")), None), true));
    println!("guard {}", guarded((None, Some(Noisy("b"))), true));
    println!("guard {}", guarded((Some(Noisy("c")), Some(Noisy("d"))), false));
    println!("drops {}", DROPS.with(Cell::get));
}
