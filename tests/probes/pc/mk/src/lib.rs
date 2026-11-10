use proc_macro::TokenStream;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

#[proc_macro]
pub fn mk(_input: TokenStream) -> TokenStream {
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("const gen_{n}: usize = {n};").parse().unwrap()
}
