//! Haste's code generator: an SSA function builder, a machine encoder per target, and executable memory.

mod alloc;
mod builder;
mod color;
mod coroutine;
mod copy;
mod cse;
mod dom;
mod error;
mod ir;
mod memory;
mod fold;
mod fill;
mod frame;
mod licm;
mod live;
mod loads;
mod merge;
mod native;
mod probe;
mod promote;
mod reloc;
mod returns;
mod split;
mod target;
mod verify;
/// Defines an encoder's `value_op`, which emits a value operation through the encoder's method for its kind.
macro_rules! value_op {
    () => {
        fn value_op(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
            match op {
                Op::Unary(op, x) => self.unary(*op, *x, result()?),
                Op::Binary(op, x, y) => self.binary(*op, *x, *y, result()?),
                Op::Convert(op, x) => self.convert(*op, *x, result()?),
                Op::Icmp(cc, x, y) => self.icmp(*cc, *x, *y, result()?),
                Op::Fits(op, signed, x, y) => self.fits(*op, *signed, (*x, *y), result()?),
                Op::Fcmp(cc, x, y) => {
                    self.fcmp(*cc, *x, *y, result()?);
                    Ok(())
                }
                _ => self.memory_op(op, result),
            }
        }
    };
}

mod arm64;
mod x64;

pub use builder::{FunctionBuilder, Ins, Mark};
pub use ir::{AbiParam, ArgumentPurpose, AtomicRmwOp, Block, ByteCC, CallConv, FloatCC, Function, Inst, IntCC, SigRef, Signature, StackSlot, Type, Value, VectorOp};
pub use coroutine::{Coroutine, FIRST_SUSPEND, RETURNED, UNRESUMED};
pub use error::{Error, ErrorKind};
pub use memory::{Arena, Memory};
pub use reloc::{DictTierUp, FootprintHasher, RelocKind, RuntimeSymbol, StencilReloc};
pub use target::{Arch, CallAbi, Effort, Format, Os, TARGETS, Target};

/// A call site that has an exception handler, as offsets into its function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Site {
    pub ret: u32,
    pub handler: u32,
    pub frame_offset: u32,
}

/// Machine code for one function, its exception call sites, and its stencil relocations.
#[derive(Clone, Debug)]
pub struct Code {
    bytes: Vec<u8>,
    sites: Vec<Site>,
    relocs: Vec<StencilReloc>,
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

    /// The relocations inside `bytes`.
    pub fn relocs(&self) -> &[StencilReloc] {
        &self.relocs
    }

    /// Copies `bytes` into `out` and patches every relocation against `symbols`.
    pub fn patch(&self, symbols: &[u64], out: &mut [u8]) {
        out[..self.bytes.len()].copy_from_slice(&self.bytes);
        reloc::patch_bytes(&self.relocs, symbols, &mut out[..self.bytes.len()]);
    }
}

/// Encodes a call-table entry stub that loads `dict_addr` into the dictionary register and jumps to `body_exec`.
pub fn dict_entry_stub(arch: Arch, stub_exec: usize, body_exec: usize, dict_addr: usize, tier_up: Option<DictTierUp>) -> Result<Vec<u8>, Error> {
    let bytes = match arch {
        Arch::X64 => x64::dict_stub(stub_exec, body_exec, dict_addr, tier_up),
        Arch::Arm64 => arm64::dict_stub(stub_exec, body_exec, dict_addr, tier_up),
        other => Err(format!("no encoder for {other:?} yet")),
    };
    bytes.map_err(|message| Error::new(ErrorKind::Unsupported, message))
}

/// Promotes `func`'s frame slots to values, folds constants and branches on them, merges straight-line blocks, reuses values a dominating instruction computed and returns in place of jumps to a return, as `compile` does before encoding, and drops the dead entries of a body kept for copying.
pub fn simplify(func: &mut Function, effort: Effort) {
    optimize(func, promote::Moves::Guided, effort);
    fold::compact(func);
}

fn optimize(func: &mut Function, moves: promote::Moves, effort: Effort) {
    fill::run(func);
    fold::run(func);
    promote::run(func, if effort == Effort::Full { moves } else { promote::Moves::Words });
    fold::run(func);
    merge::run(func);
    cse::run(func);
    if effort == Effort::Full {
        loads::run(func);
        licm::run(func);
    }
    returns::run(func);
}

/// Encodes `func` for `target`.
pub fn compile(mut func: Function, target: &Target) -> Result<Code, Error> {
    coroutine::split(&mut func);
    verify::verify(&func).map_err(|message| Error::new(ErrorKind::Verify, message))?;
    optimize(&mut func, promote::Moves::Split, target.effort);
    if target.arch != Arch::X64 {
        copy::expand(&mut func, |_, _, bytes| Some(copy::words(bytes)));
    }
    native::lower(&mut func);
    verify::verify(&func).map_err(|message| Error::new(ErrorKind::Verify, format!("after simplification: {message}")))?;
    let code = match target.arch {
        Arch::X64 => x64::compile(&func, target.effort),
        Arch::Arm64 => arm64::compile(&func, target),
        other => Err(format!("no encoder for {other:?} yet")),
    };
    code.map_err(|message| Error::new(ErrorKind::Unsupported, message))
}
