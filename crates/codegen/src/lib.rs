//! Haste's code generator: an SSA function builder, a machine encoder per target, and executable memory.

mod builder;
mod coroutine;
mod ir;
mod memory;
mod merge;
mod promote;
mod verify;
#[cfg(target_arch = "x86_64")]
mod x64;

pub use builder::{FunctionBuilder, Ins, Mark};
pub use ir::{AbiParam, ArgumentPurpose, AtomicRmwOp, Block, CallConv, FloatCC, Function, Inst, IntCC, SigRef, Signature, StackSlot, Type, Value};
pub use coroutine::{Coroutine, FIRST_SUSPEND, RETURNED, UNRESUMED};
pub use memory::{Memory, Shared};

/// A call site that has an exception handler, as offsets into its function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Site {
    pub ret: u32,
    pub handler: u32,
    pub frame_offset: u32,
}

/// Machine code for one function and its exception call sites.
pub struct Compiled {
    pub code: Vec<u8>,
    pub sites: Vec<Site>,
}

/// Promotes `func`'s frame slots to values and folds its straight-line blocks, as `compile` does before encoding.
pub fn simplify(func: &mut Function) {
    promote::run(func);
    merge::run(func);
}

/// Encodes `func` for the host machine.
pub fn compile(mut func: Function) -> Result<Compiled, String> {
    coroutine::split(&mut func);
    verify::verify(&func)?;
    promote::run(&mut func);
    merge::run(&mut func);
    verify::verify(&func).map_err(|error| format!("after slot promotion: {error}"))?;
    #[cfg(target_arch = "x86_64")]
    return x64::compile(&func);
    #[cfg(not(target_arch = "x86_64"))]
    return Err("no encoder for this target yet".to_owned());
}
