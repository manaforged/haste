#![feature(rustc_private)]
//! A rustc driver that builds a crate as cargo asks, then builds it again with every function instance exported, and links those objects into one prep library with a manifest.

extern crate rustc_abi;
extern crate rustc_attr_ir;
extern crate rustc_attr_parsing;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;

mod abi;
mod args;
mod elf;
mod error;
mod export;
mod link;
mod record;
mod uses;
mod wrap;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("link") {
        return link::main(&args[2..]);
    }
    wrap::main(args)
}
