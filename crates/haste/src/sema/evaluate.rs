//! Constant items and anonymous constants whose syntax the folder cannot evaluate, evaluated by running their bodies through the runner that the layer owning code memory installs.

use std::sync::OnceLock;

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use super::Context;
use super::reach::Instance;

/// What runs constant items: compiles the body of one and each function it calls, runs it, and gives its value.
pub trait Evaluate {
    /// The value of the integer constant `instance`, `bits` wide, from running its body.
    fn evaluate(&mut self, cx: &mut Context<'_>, instance: &Instance, bits: u32) -> Result<u128>;
}

/// Makes a runner of constant items.
pub type Make = fn() -> Result<Box<dyn Evaluate>>;

static MAKE: OnceLock<Make> = OnceLock::new();

/// Installs `make` as the maker of runners of constant items, once, before any pass runs.
pub fn install(make: Make) {
    _ = MAKE.set(make);
}

/// What one pass ran: the value of each constant item, `None` while it runs, and the runner that no evaluation holds.
#[derive(Default)]
pub struct Evaluated {
    values: HashMap<Instance, Option<u128>>,
    runner: Option<Box<dyn Evaluate>>,
}

/// The value of the integer constant `instance`, `bits` wide, from running its body once per pass; a constant whose evaluation needs its own value is an error.
pub fn run(cx: &mut Context<'_>, instance: &Instance, bits: u32) -> Result<u128> {
    match cx.evaluated.values.get(instance) {
        Some(&Some(value)) => return Ok(value),
        Some(None) => bail!("the constant {} depends on itself", cx.name(instance.0)),
        None => {}
    }
    let mut runner = match (cx.evaluated.runner.take(), MAKE.get()) {
        (Some(runner), _) => runner,
        (None, Some(make)) => make()?,
        (None, None) => bail!("no runner of constants is installed"),
    };
    cx.evaluated.values.insert(instance.clone(), None);
    let found = runner.evaluate(cx, instance, bits);
    cx.evaluated.runner = Some(runner);
    match &found {
        Ok(value) => cx.evaluated.values.insert(instance.clone(), Some(*value)),
        Err(_) => cx.evaluated.values.remove(instance),
    };
    found
}
