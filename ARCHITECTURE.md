# Haste Architecture

Haste is a clean-room Rust compiler, JIT engine, and standalone AOT compiler with no LLVM dependency. Its thesis is that Rust does not have to choose between sub-second incremental builds and `-O3` runtime throughput:

1. **Fast builds come from skipping unused work and caching settled sysroot state.** Instead of compiling every dependency crate end-to-end into bitcode and object files, Haste resolves workspace metadata, expands macros, and runs wave-based lazy reachability from binary or test entry points—type-checking and lowering only reached monomorphizations. Settled standard-library outlines, macros, and name scopes live in a persistent disk cache (`HSTSYS01` in `substrate/sysroot.rs`), eliminating 896-file stdlib re-parsing on one-shot CLI builds.
2. **Release throughput comes from a dedicated SSA IR, demand-driven pass scheduler, and native machine-code backends.** Debug and JIT builds run a compact `QUICK_RULES` pass set (`schedule.rs`), while `--release` runs the full `FULL_RULES` fixed-point pipeline—scalar promotion, range/relational bounds-check elimination, load/store coalescing, LICM, unrolling, vectorization, live-range splitting at loop headers, and target-native instruction fusion—before emitting machine code and executable binaries directly without an external linker.

## Workspace Layout

- [`crates/haste/src/`](crates/haste/src/) — build graph, parallel outline/macro frontend, persistent substrate cache, type inference, borrow checker, SSA lowering, JIT runtime, `serve` daemon, and standalone AOT executable emitters.
- [`crates/codegen/src/`](crates/codegen/src/) — target-independent SSA IR, demand-driven optimization pipeline, register allocator, and native `x64` / `arm64` encoders.

---

## 1. Whole-Workspace Lazy Reachability & Persistent Sysroot Cache

1. **Build graph & `build.rs` execution ([`graph/`](crates/haste/src/graph/))**
   `manifest.rs`, `lock.rs`, `features.rs`, `cfg.rs`, `script.rs`, and `store.rs` parse `Cargo.toml` / `Cargo.lock`, resolve target `cfg` predicates and feature unification, and execute `build.rs` scripts, caching outputs in `$HASTE_HOME` (`~/.cache/haste`).

2. **Parallel outline, macro expansion & name resolution ([`outline/`](crates/haste/src/outline/), [`macros.rs`](crates/haste/src/macros.rs), [`names/`](crates/haste/src/names/))**
   - `outline/parallel.rs` and `outline/scan.rs` lex and scan crate sources in parallel, pruning disabled `cfg` subtrees (`outline/attr.rs`, `outline/tree.rs`) without building full ASTs for unreached bodies.
   - `outline/mbe.rs` and `macros.rs` expand `macro_rules!` and procedural macros; `names/{collect,resolve,path,want}.rs` build module/item scopes on demand.

3. **Persistent sysroot & workspace substrate ([`substrate/`](crates/haste/src/substrate/))**
   - [`substrate/sysroot.rs`](crates/haste/src/substrate/sysroot.rs) (`names.rs`, `tokens.rs`, `wire.rs`) serializes settled sysroot crate outlines, macro definitions, and resolved name scopes (`HSTSYS01`), skipping 896-file stdlib re-parsing on one-shot invocations.
   - [`substrate/noop.rs`](crates/haste/src/substrate/noop.rs) performs a fast timestamp/digest check to skip unchanged workspaces (~25 ms no-op builds).
   - [`serve.rs`](crates/haste/src/serve.rs) keeps the workspace loaded in memory and answers incremental `build`, `test`, and `run` requests, rebuilding only items affected by edited files.

4. **Wave-based lazy reachability ([`sema/reach.rs`](crates/haste/src/sema/reach.rs))**
   - `reach/{wave,filter,order}.rs` start from `main`, `#[test]`, or `case_*` roots and discover callees in parallel waves.
   - Each reached monomorphization is keyed by `Instance = (ItemRef, Substs)` (16 bytes) and tracked through `Declared` → `Enqueued` → `Blocked` → `Settled`. Unreached functions in dependencies and the standard library are never type-checked or lowered.

---

## 2. Type Inference, Borrow Checking & Stencil/Dictionary Lowering

1. **Typed HIR & cross-crate summaries ([`sema/`](crates/haste/src/sema/))**
   - `sema/infer/` runs bidirectional type inference, method/autoderef resolution, closure capture inference, and pattern typing, backed by trait resolution (`sema/traits.rs`), layout computation (`sema/layout.rs`), and `const fn` evaluation (`sema/evaluate.rs`).
   - Cross-crate items serialize compact frozen summaries (`sema/summary/{codec,frozen,inst,thaw}.rs`) so callers instantiate generic signatures and layouts without re-running full crate inference.

2. **Borrow checker ([`sema/borrow.rs`](crates/haste/src/sema/borrow.rs))**
   Local workspace bodies are checked for initialization, use-after-move (across branches and loops), binding mutability, disjoint struct-field loans (`FieldKey::Named` / `FieldKey::Tuple`), NLL-style shared/mutable aliasing conflicts, and dangling local reference returns.

3. **SSA lowering, stencils & JIT tier-up ([`lower/`](crates/haste/src/lower/), [`image/`](crates/haste/src/image/))**
   - `lower/` translates typed bodies into `haste_codegen` SSA IR, handling `async`/`await` coroutines (`lower/coroutine.rs`), explicit `Drop` elaboration (`lower/drop.rs`), and portable/target SIMD (`lower/simd/`).
   - **Dictionary/stencil sharing ([`lower/stencil.rs`](crates/haste/src/lower/stencil.rs)):** Generic bodies above `HASTE_DICT_MIN_BYTES` (default `512`) with fanout ≥ `HASTE_DICT_MIN_FANOUT` (default `8`) can lower once as a dictionary-passing stencil keyed by a 128-bit structural `FootprintId` (`stencil/{splice,persist,cycles}.rs`).
   - **JIT tier-up ([`image/stencil_jit.rs`](crates/haste/src/image/stencil_jit.rs)):** In JIT mode, hot dictionary instances specialize to concrete monomorphized machine code after `HASTE_JIT_TIER_UP` (default `64`) calls.

---

## 3. Direct SSA IR & Optimization Pipeline ([`crates/codegen/src/`](crates/codegen/src/))

[`lib.rs`](crates/codegen/src/lib.rs) splits coroutines (`coroutine.rs`), verifies IR invariants (`verify.rs`), runs the demand-driven optimizer (`schedule.rs`), lowers wide/runtime helpers (`native.rs`, `overalign.rs`), and invokes the target encoder.

### Demand-Driven Pass Scheduler (`schedule.rs`)

`schedule::run` tracks dirty `Domain` flags (`CFG`, `MEM`, `VALUES`) so passes re-run only when an earlier pass invalidates a domain they read:

- **Debug / JIT (`Effort::Quick`):** `Fill` → `Promote` → `Fold` → `Merge` → `Cse` → `Returns`.
- **Release (`Effort::Full`):** Adds the full loop, memory, and control-flow optimization suite:
  - **Scalar & algebraic simplification:** `fold.rs` (`fold/{exits,range,unused,vector}.rs`), `algebra.rs` (`algebra/{bits,cond,icmp,ord,ovf,select,sum,vec}.rs`), `cse.rs`, and `merge.rs` (short-circuit compare folding).
  - **Memory & scalar promotion:** `promote.rs` (`promote/{box_fwd,cold_copy,join_slots,vec_slot}.rs`), `promote_subword.rs`, and `loads.rs` (`loads/{elide,coalesce,dom,join,pack,regroup,remat,sink}.rs`) guided by escape and frame alias analysis (`mem/{escape,frame,loop_mem}.rs`).
  - **Range & relational fact propagation ([`facts/`](crates/codegen/src/facts/)):** `facts/{backprop,exits,live,loops,rel,transfer}.rs` track integer intervals, excluded constants (`MAX_NOT=32`), and symbolic inequalities (`MAX_FACTS=64`, `MAX_RELS=64`) so `fold/range.rs` eliminates redundant slice bounds and overflow traps.
  - **Loop & control-flow transforms:** `rotate.rs`, `licm.rs` (`licm/reassoc.rs`), `lsr.rs` (`lsr/lftr.rs`), `unroll.rs` (`unroll/region.rs`), `ifconv.rs`, `thread.rs` (jump threading), `tailrec.rs`, `slp.rs` (superword-level parallelism), and `switch.rs` (`switch/{bitmask,lut,member,select,tree}.rs`).

### Register Allocation & Live-Range Shaping

- **Loop-header live-range splitting ([`loopargs.rs`](crates/codegen/src/loopargs.rs)):** Passes loop-invariant values read inside a loop as explicit block parameters at the loop header so their inner-loop live range can stay in a register independent of outer-loop call pressure.
- **Allocator & parking ([`alloc/`](crates/codegen/src/alloc/), [`park/`](crates/codegen/src/park/)):** Weighted register allocation (`park/{choose,place,recolor}.rs`, `alloc/{demote,hints,hoist,walk}.rs`, `color.rs`) balances callee-saved registers against park slots across calls, performs copy coalescing, and resolves back-edge parallel copies/slot swaps cleanly.
- **IR serialization ([`wire.rs`](crates/codegen/src/wire.rs)):** Binary `Function::encode` / `Function::decode` persists compiled stencils and substrate artifacts without re-lowering.

---

## 4. Native Backends & Zero-Linker Executable Emitters

1. **Machine-code encoders**
   - **x86-64 ([`crates/codegen/src/x64/`](crates/codegen/src/x64/)):** Encodes SSE2 baseline and runtime-detected AVX/AVX2 (`features.rs`, `ymm.rs`, `vector*.rs`), 128-bit/wide arithmetic (`wide.rs`), and memory/addressing/product fusion (`location/fuse.rs`).
   - **ARM64 ([`crates/codegen/src/arm64/`](crates/codegen/src/arm64/)):** Encodes AArch64 scalar, NEON (`vector*.rs`), and target fusion (`fuse/{bfi,bits,ccmp,flag,mem,shift,sign,vconst}.rs`) including bitfield insert, shifted operands, and `ccmp` chains.

2. **Zero-linker AOT executables ([`crates/haste/src/aot/`](crates/haste/src/aot/))**
   `aot.rs` assembles `.text`, `.rodata`, `.data`, and `.bss` directly into runnable binaries—with optional DWARF `.debug_line` tables (`aot/dwarf.rs`) and runtime shims (`aot/shims.rs`)—for all six supported targets (`--target TRIPLE`) without invoking an external linker:
   - **ELF** ([`aot/elf.rs`](crates/haste/src/aot/elf.rs)): `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`
   - **Mach-O** ([`aot/macho.rs`](crates/haste/src/aot/macho.rs)): `aarch64-apple-darwin`, `x86_64-apple-darwin`
   - **PE32+** ([`aot/pe.rs`](crates/haste/src/aot/pe.rs)): `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`
