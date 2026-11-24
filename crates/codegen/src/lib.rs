//! Haste's code generator: an SSA function builder, a machine encoder per target, and executable memory.

mod algebra;
mod alloc;
mod asm_text;
mod builder;
pub(crate) mod cfg;
mod color;
mod copy;
mod coroutine;
mod cse;
mod drop_elide;
mod emit;
mod error;
mod facts;
mod fill;
mod fold;
mod frame;
mod ifconv;
mod ir;
mod licm;
mod live;
mod loads;
mod loopargs;
mod lsr;
pub(crate) mod mem;
mod memory;
mod merge;
mod native;
mod overalign;
mod pairs;
mod park;
mod promote;
mod reloc;
mod returns;
mod rotate;
mod schedule;
mod slp;
mod spills;
mod split;
mod switch;
mod tailrec;
mod target;
mod thread;
pub mod trace;
mod unroll;
mod verify;
mod wide;
mod wire;
/// Defines an encoder's `value_op`, which emits a value operation through the encoder's method for its kind.
macro_rules! value_op {
    () => {
        fn value_op(
            &mut self,
            op: &Op,
            result: impl Fn() -> Result<Value, String>,
        ) -> Result<(), String> {
            match op {
                Op::Unary(op, x) => self.unary(*op, *x, result()?),
                Op::Binary(op, x, y) => self.binary(*op, *x, *y, result()?),
                Op::Convert(op, x) => self.convert(*op, *x, result()?),
                Op::Icmp(cc, x, y) => self.icmp(*cc, *x, *y, result()?),
                Op::Fits(op, signed, x, y) => self.fits(*op, *signed, (*x, *y), result()?),
                Op::Fma(a, b, c) => self.fma(*a, *b, *c, result()?),
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

pub use asm_text::{Assembly, assemble};
pub use builder::{FunctionBuilder, Ins, Mark};
pub use coroutine::Coroutine;
pub use drop_elide::UnwindDrop;
pub use error::{Error, ErrorKind};
pub use ir::{
    AbiParam, ArgumentPurpose, AtomicOrd, AtomicRmwOp, Block, CallConv, FloatCC, Function, Inst,
    IntCC, Lane, ReduceOp, ShuffleMask, SigRef, Signature, StackSlot, TargetOpId, TinyArgs, Type,
    Value,
};
pub use memory::{Arena, Memory};
pub use reloc::{
    DictTierUp, FRAME_CACHE_DEPTH, FRAME_CACHE_LIMIT, RelocKind, RuntimeSymbol, StencilReloc,
    patch_reloc,
};
pub use target::{Arch, CallAbi, Effort, Format, Os, TARGETS, Target};
pub use trace::{enabled as trace_enabled, missed_report, with_label as with_trace_label};

use std::ops::Range;

/// A call site that has an exception handler, as offsets into its function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Site {
    pub ret: u32,
    pub handler: u32,
    pub frame_offset: u32,
}

/// One frame-relative owned value dropped during panic unwinding when its 1-byte active flag is non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeUnwindDrop {
    pub local_off: i32,
    pub flag_off: i32,
    pub drop_sym: u32,
    pub indirect: bool,
}

/// Machine code for one function, its exception call sites, its stencil relocations, and its small loops.
#[must_use]
#[derive(Clone, Debug)]
pub struct Code {
    bytes: Vec<u8>,
    sites: Vec<Site>,
    relocs: Vec<StencilReloc>,
    loops: Vec<Range<u32>>,
    lines: Vec<(u32, u64)>,
    unwind_drops: Vec<CodeUnwindDrop>,
    saved_regs: u32,
}

impl Code {
    /// Builds a `Code` buffer from pre-encoded machine code, unwind sites, relocations, and small loop ranges.
    pub fn from_parts(
        bytes: Vec<u8>,
        sites: Vec<Site>,
        relocs: Vec<StencilReloc>,
        loops: Vec<Range<u32>>,
    ) -> Self {
        Self {
            bytes,
            sites,
            relocs,
            loops,
            lines: Vec::new(),
            unwind_drops: Vec::new(),
            saved_regs: 0,
        }
    }

    /// Attaches frame-relative unwind drop descriptors to this `Code` buffer.
    pub fn with_unwind_drops(mut self, unwind_drops: Vec<CodeUnwindDrop>) -> Self {
        self.unwind_drops = unwind_drops;
        self
    }

    /// Attaches the canonical bitmask of callee-saved registers saved in this function's prologue.
    pub fn with_saved_regs(mut self, saved_regs: u32) -> Self {
        self.saved_regs = saved_regs;
        self
    }

    /// Canonical bitmask of callee-saved registers saved in this function's prologue.
    pub fn saved_regs(&self) -> u32 {
        self.saved_regs
    }

    /// Frame-relative unwind drop descriptors emitted for this function.
    pub fn unwind_drops(&self) -> &[CodeUnwindDrop] {
        &self.unwind_drops
    }

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

    /// The byte ranges of the small innermost loops, which run slower when they cross a page boundary.
    pub fn loops(&self) -> &[Range<u32>] {
        &self.loops
    }

    /// The offsets in `bytes` where a new source place starts, with that place as `FunctionBuilder::set_loc` took it.
    pub fn lines(&self) -> &[(u32, u64)] {
        &self.lines
    }

    /// Copies `bytes` into `out` and patches every relocation against `symbols`.
    pub fn patch(&self, symbols: &[u64], out: &mut [u8]) -> Result<(), Error> {
        if out.len() < self.bytes.len() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!(
                    "patch buffer too small: {} < {}",
                    out.len(),
                    self.bytes.len()
                ),
            ));
        }
        out[..self.bytes.len()].copy_from_slice(&self.bytes);
        reloc::patch_bytes(&self.relocs, symbols, &mut out[..self.bytes.len()])
    }
}

/// Whether environment variable `name` is set to a non-empty value other than `"0"`.
pub(crate) fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| !v.is_empty() && v != "0")
}

impl Target {
    /// Encodes a call-table entry stub that loads `dict_addr` into the dictionary register and jumps to `body_exec`.
    pub fn dict_entry_stub(
        &self,
        stub_exec: usize,
        body_exec: usize,
        dict_addr: usize,
        tier_up: Option<DictTierUp>,
    ) -> Result<Vec<u8>, Error> {
        let bytes = match self.arch {
            Arch::X64 => x64::dict_stub(stub_exec, body_exec, dict_addr, tier_up),
            Arch::Arm64 => arm64::dict_stub(stub_exec, body_exec, dict_addr, tier_up),
        };
        bytes.map_err(|message| Error::new(ErrorKind::Unsupported, message))
    }

    /// Returns the `(dict_offset, body_offset, kind)` relocation offsets inside an unlinked `dict_entry_stub(0, 1 << 40, 0, None)`.
    pub fn dict_stub_relocs(&self) -> Result<[(u32, RelocKind); 2], Error> {
        match self.arch {
            Arch::X64 => Ok(x64::DICT_STUB_RELOCS),
            Arch::Arm64 => Ok(arm64::DICT_STUB_RELOCS),
        }
    }
}

/// Promotes `func`'s frame slots to values, folds constants and branches on them, merges straight-line blocks, reuses values a dominating instruction computed and returns in place of jumps to a return, as `compile` does before encoding, and drops the dead entries of a body kept for copying.
pub fn simplify(func: &mut Function, target: &Target) {
    keep_honored_marks(func, target);
    schedule::run(func, promote::Moves::Guided, target);
    fold::compact(func);
}

fn keep_honored_marks(func: &mut Function, target: &Target) {
    if target.arch == Arch::X64 {
        func.forget_unpredictable();
    }
}

/// Splits `func` into its state machine when it is a coroutine's resume function, leaving a plain function a caller can inline.
pub fn split_coroutine(func: &mut Function) {
    copy::build_in_place(func);
    coroutine::split(func);
}

/// Encodes `func` for `target`.
pub fn compile(mut func: Function, target: &Target) -> Result<Code, Error> {
    let was_coro = func.is_coroutine();
    split_coroutine(&mut func);
    if was_coro {
        if target.effort == Effort::Full {
            func.elide_dead_unwind_drops(|_| true);
        }
        func.zero_init_unwind_flags();
    }
    keep_honored_marks(&mut func, target);
    if !(target.direct_calls && target.arch == Arch::X64) {
        func.forget_image_addrs();
    }
    let tracing = trace::enabled();
    if tracing {
        trace::on_compile_start(&func);
    }
    verify::verify(&func, target).map_err(|message| {
        if verify::verify_passes_enabled() {
            eprintln!("[haste-verify] failed at entry: {message}\n{func}");
        }
        Error::new(ErrorKind::Verify, message)
    })?;
    schedule::run(&mut func, promote::Moves::Split, target);
    if tracing {
        trace::on_compile_end(&func);
    }
    native::lower(&mut func);
    overalign::lower(&mut func);
    verify::verify(&func, target).map_err(|message| {
        Error::new(
            ErrorKind::Verify,
            format!("after simplification: {message}"),
        )
    })?;
    let code = match target.arch {
        Arch::X64 => x64::compile(&func, target),
        Arch::Arm64 => arm64::compile(&func, target),
    };
    code.map_err(|message| Error::new(ErrorKind::Unsupported, message))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unpredictable_fn() -> (Function, Value) {
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(Type::I64));
        sig.returns.push(AbiParam::new(Type::I64));
        let mut func = Function::new(sig);
        let mut b = FunctionBuilder::new(&mut func);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        let p = b.block_params(entry)[0];
        b.switch_to_block(entry);
        let zero = b.ins().iconst(Type::I64, 0);
        let cond = b.ins().icmp(IntCC::Equal, p, zero);
        b.func.mark_unpredictable(cond);
        let one = b.ins().iconst(Type::I64, 1);
        let res = b.ins().select(cond, one, zero);
        b.ins().return_(&[res]);
        (func, cond)
    }

    #[test]
    fn simplify_respects_explicit_target_arch() {
        let x64 = Target::find("x86_64-unknown-linux-gnu").expect("x64 target");
        let arm64 = Target::find("aarch64-apple-darwin").expect("arm64 target");
        let (mut func_x64, cond_x64) = unpredictable_fn();
        let (mut func_arm64, cond_arm64) = unpredictable_fn();
        simplify(&mut func_x64, x64);
        simplify(&mut func_arm64, arm64);
        assert!(!func_x64.is_unpredictable(cond_x64));
        assert!(func_arm64.is_unpredictable(cond_arm64));
    }
}
