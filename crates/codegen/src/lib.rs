//! Haste's code generator: an SSA function builder, a machine encoder per target, and executable memory.

mod alloc;
mod builder;
mod coroutine;
mod copy;
mod error;
mod ir;
mod memory;
mod fold;
mod frame;
mod live;
mod merge;
mod native;
mod promote;
mod target;
mod verify;
mod arm64;
mod x64;

pub use builder::{FunctionBuilder, Ins, Mark};
pub use ir::{AbiParam, ArgumentPurpose, AtomicRmwOp, Block, CallConv, FloatCC, Function, Inst, IntCC, SigRef, Signature, StackSlot, Type, Value};
pub use coroutine::{Coroutine, FIRST_SUSPEND, RETURNED, UNRESUMED};
pub use error::{Error, ErrorKind};
pub use memory::{Arena, Memory};
pub use target::{Arch, CallAbi, Format, Os, TARGETS, Target};

/// A call site that has an exception handler, as offsets into its function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Site {
    pub ret: u32,
    pub handler: u32,
    pub frame_offset: u32,
}

/// Machine code for one function and its exception call sites.
#[derive(Clone, Debug)]
pub struct Code {
    bytes: Vec<u8>,
    sites: Vec<Site>,
}

impl Code {
    /// The machine code.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The call sites that have an exception handler.
    pub fn sites(&self) -> &[Site] {
        &self.sites
    }
}

/// Promotes `func`'s frame slots to values, folds constants and branches on them, merges straight-line blocks as `compile` does before encoding, and drops the dead entries of a body kept for copying.
pub fn simplify(func: &mut Function) {
    optimize(func);
    fold::compact(func);
}

fn optimize(func: &mut Function) {
    promote::run(func);
    fold::run(func);
    merge::run(func);
}

/// Encodes `func` for `target`.
pub fn compile(mut func: Function, target: &Target) -> Result<Code, Error> {
    coroutine::split(&mut func);
    verify::verify(&func).map_err(|message| Error::new(ErrorKind::Verify, message))?;
    optimize(&mut func);
    native::lower(&mut func);
    verify::verify(&func).map_err(|message| Error::new(ErrorKind::Verify, format!("after simplification: {message}")))?;
    let code = match target.arch {
        Arch::X64 => x64::compile(&func),
        Arch::Arm64 => arm64::compile(&func, target),
        other => Err(format!("no encoder for {other:?} yet")),
    };
    code.map_err(|message| Error::new(ErrorKind::Unsupported, message))
}
