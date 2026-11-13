use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

struct YieldOnce(bool, i64);

impl Future for YieldOnce {
    type Output = i64;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<i64> {
        if self.0 {
            Poll::Ready(self.1)
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

struct Loud<'a>(&'a std::cell::Cell<i64>, i64);

impl Drop for Loud<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() * 10 + self.1);
    }
}

async fn add(a: i64, b: i64) -> i64 {
    let x = YieldOnce(false, 7).await;
    a + b + x
}

async fn twice(n: i64) -> i64 {
    let s = String::from("hi");
    let a = add(n, 1).await;
    let b = add(a, 2).await;
    b + s.len() as i64
}

fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = Box::pin(fut);
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    loop {
        if let Poll::Ready(value) = fut.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

fn case_ready(n: i64) -> i64 {
    block_on(async move { n + 1 })
}

fn case_one(n: i64) -> i64 {
    block_on(async move { YieldOnce(false, n).await })
}

fn case_two(n: i64) -> i64 {
    block_on(async move { YieldOnce(false, n).await + YieldOnce(false, 1).await })
}

fn case_after(n: i64) -> i64 {
    let v = vec![1, 2, 3];
    block_on(async move { YieldOnce(false, n).await + v[2] })
}

fn case_after_len(n: i64) -> i64 {
    let v = vec![1, 2, 3];
    block_on(async move { v.len() as i64 + YieldOnce(false, n).await })
}

fn case_vec_ready(n: i64) -> i64 {
    let v = vec![1, 2, 3];
    block_on(async move { v.len() as i64 + n })
}

fn case_vec_forget(n: i64) -> i64 {
    let v = vec![1, 2, 3];
    let fut = async move { v.len() as i64 + n };
    std::mem::forget(fut);
    n
}

fn case_mod_sync(n: i64) -> i64 {
    mod util {
        pub(super) type Mask = u8;
    }
    let m: util::Mask = Default::default();
    n + m as i64
}

fn case_mod_async(n: i64) -> i64 {
    block_on(async move {
        mod util {
            pub(super) type Mask = u8;
        }
        let m: util::Mask = Default::default();
        n + m as i64
    })
}

async fn masked(n: i64) -> i64 {
    mod util {
        pub(super) type Mask = u8;
    }
    let m: util::Mask = Default::default();
    YieldOnce(false, n).await + m as i64
}

macro_rules! masked_by_macro {
    ($n:expr) => {{
        mod __util {
            pub(super) type Mask = u8;
        }
        let m: __util::Mask = Default::default();
        $n + m as i64
    }};
}

async fn masked_macro(n: i64) -> i64 {
    YieldOnce(false, n).await + masked_by_macro!(1)
}

fn case_mod_fn(n: i64) -> i64 {
    block_on(masked(n))
}

fn case_mod_macro(n: i64) -> i64 {
    block_on(masked_macro(n))
}

fn case_nested(n: i64) -> i64 {
    block_on(async move { add(n, 1).await })
}

fn case_loop(n: i64) -> i64 {
    block_on(async move {
        let mut t = n;
        for step in 0..3 {
            t += YieldOnce(false, step).await;
        }
        t
    })
}

fn case_async(n: i64) -> i64 {
    block_on(twice(n))
}

fn case_async_block(n: i64) -> i64 {
    let v = vec![1, 2, 3];
    block_on(async move { YieldOnce(false, n).await + v.len() as i64 })
}

fn case_async_borrow(n: i64) -> i64 {
    let mut total = n;
    block_on(async {
        for step in 0..3 {
            total += YieldOnce(false, step).await;
        }
    });
    total
}

fn case_async_drop(n: i64) -> i64 {
    let log = std::cell::Cell::new(0);
    {
        let fut = async {
            let _a = Loud(&log, 1);
            YieldOnce(false, 0).await;
            let _b = Loud(&log, 2);
            n
        };
        let mut fut = Box::pin(fut);
        let mut cx = Context::from_waker(Waker::noop());
        let _ = fut.as_mut().poll(&mut cx);
    }
    log.get() * 100 + n
}

fn case_async_done_drop(n: i64) -> i64 {
    let log = std::cell::Cell::new(0);
    let got = block_on(async {
        let _a = Loud(&log, 3);
        YieldOnce(false, n).await
    });
    log.get() * 100 + got
}

include!("../../case_inputs.rs");

fn main() {
    let cases: [(&str, fn(i64) -> i64); 18] = [
        ("case_mod_fn", case_mod_fn),
        ("case_mod_macro", case_mod_macro),
        ("case_mod_sync", case_mod_sync),
        ("case_mod_async", case_mod_async),
        ("case_vec_ready", case_vec_ready),
        ("case_vec_forget", case_vec_forget),
        ("case_two", case_two),
        ("case_after", case_after),
        ("case_after_len", case_after_len),
        ("case_nested", case_nested),
        ("case_loop", case_loop),
        ("case_ready", case_ready),
        ("case_one", case_one),
        ("case_async", case_async),
        ("case_async_block", case_async_block),
        ("case_async_borrow", case_async_borrow),
        ("case_async_drop", case_async_drop),
        ("case_async_done_drop", case_async_done_drop),
    ];
    for (name, case) in cases {
        for n in CASE_INPUTS {
            println!("{name} {n} {}", case(n));
        }
    }
}
