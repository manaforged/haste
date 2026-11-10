use std::cell::Cell;
use std::sync::{Arc, Mutex, OnceLock};

static NAMES: Mutex<Vec<u32>> = Mutex::new(Vec::new());
static ONCE: OnceLock<u64> = OnceLock::new();
thread_local! { static LOCAL: Cell<u32> = const { Cell::new(1) }; }

struct Init {
    handle: Arc<u32>,
    start: Box<dyn FnOnce() -> u32 + Send>,
}

impl Init {
    fn take(self: Box<Self>) -> Box<dyn FnOnce() -> u32 + Send> {
        println!("handle {}", self.handle);
        self.start
    }
}

fn main() {
    let t = std::thread::spawn(|| 7u32);
    println!("one {}", t.join().unwrap());
    let handles: Vec<_> = (0..8u32).map(|i| std::thread::spawn(move || {
        NAMES.lock().unwrap().push(i);
        *ONCE.get_or_init(|| 42)
    })).collect();
    let sum: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let mut names = NAMES.lock().unwrap().clone();
    names.sort();
    println!("statics {sum} {names:?}");
    LOCAL.with(|c| c.set(5));
    let other = std::thread::spawn(|| LOCAL.with(|c| c.get())).join().unwrap();
    println!("local {} {other}", LOCAL.with(|c| c.get()));
    std::thread::scope(|s| {
        s.spawn(|| println!("scoped"));
    });
    let n = 3u32;
    let start = Box::new(Init { handle: Arc::new(9), start: Box::new(move || n + 2) }).take();
    println!("moved {}", start());
}
