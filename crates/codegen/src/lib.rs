//! Haste's code generator: an SSA function builder, a machine encoder per target, and executable memory.

mod builder;
mod ir;
mod memory;
mod promote;
mod verify;
#[cfg(target_arch = "x86_64")]
mod x64;

pub use builder::{FunctionBuilder, Ins, Mark};
pub use ir::{AbiParam, ArgumentPurpose, AtomicRmwOp, Block, CallConv, FloatCC, Function, Inst, IntCC, SigRef, Signature, StackSlot, Type, Value};
pub use memory::Memory;

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

/// Encodes `func` for the host machine.
pub fn compile(func: &Function) -> Result<Compiled, String> {
    verify::verify(func)?;
    let mut func = func.clone();
    promote::run(&mut func);
    verify::verify(&func).map_err(|error| format!("after slot promotion: {error}"))?;
    #[cfg(target_arch = "x86_64")]
    return x64::compile(&func);
    #[cfg(not(target_arch = "x86_64"))]
    return Err("no encoder for this target yet".to_owned());
}
