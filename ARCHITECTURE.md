# haste architecture

haste is a standalone Rust compiler, JIT runner, and ahead-of-time executable builder with no LLVM dependency and no `target/` directory. Edit-loop speed comes from skipping unreached external work, memory-mapping cached artifacts, and recompiling only changed functions in a resident daemon. Release throughput comes from a dedicated block-parameter SSA IR, a domain-tracked pass scheduler, linear-scan register allocation with call-site parking and loop-header live-range splitting, and native x86-64 and ARM64 encoders coupled to direct ELF, Mach-O, and PE32+ emitters.

## Workspace layout

- [`crates/haste/`](crates/haste/): build graph, parallel outline and macro frontend, memory-mapped substrate cache, type/trait solver, borrow checker, SSA lowering, JIT runtime, `serve` daemon, and standalone AOT executable emitters.
- [`crates/codegen/`](crates/codegen/): target-independent SSA IR, fixed-point optimization scheduler, register allocator, dual-mapped executable memory arenas, and native `x64` / `arm64` encoders.
- [`crates/haste/proc_macro/`](crates/haste/proc_macro/): in-image `proc_macro` crate used to execute procedural macros inside forked worker processes.
- [`crates/haste/prep-driver/`](crates/haste/prep-driver/): x86-64 Linux `rustc_private` driver used by `haste prep` and `--prepared`.

## Compilation pipeline

### 1. Build graph and build scripts

[`crates/haste/src/graph/`](crates/haste/src/graph/) (`manifest.rs`, `lock.rs`, `features.rs`, `cfg.rs`, `script.rs`, `store.rs`) parses `Cargo.toml` and `Cargo.lock` and merges the workspace with the active toolchain's `rust-src` standard library. It resolves target `cfg` predicates, unifies features, and runs `build.rs` scripts through [`crates/haste/src/scripts.rs`](crates/haste/src/scripts.rs). When a workspace lacks `Cargo.lock`, `graph.rs` runs `cargo generate-lockfile`. When locked registry or git crates are missing locally, it runs `cargo fetch`. Settled `build.rs` outputs live under `$HASTE_HOME` (`~/.cache/haste`).

### 2. Parallel outline, macro expansion, and name resolution

[`crates/haste/src/outline/`](crates/haste/src/outline/) (`parallel.rs`, `scan.rs`, `attr.rs`, `tree.rs`) scans crate source trees in parallel using `ra-ap-rustc_lexer` and prunes disabled `cfg` subtrees before parsing function bodies. Symbols are interned once in a lock-free lookup cache backed by a sharded `RwLock` `Symbols` table ([`crates/haste/src/names.rs`](crates/haste/src/names.rs)). Declarative macros (`outline/mbe.rs`) and procedural macros ([`crates/haste/src/macros.rs`](crates/haste/src/macros.rs), [`crates/haste/src/image/expand.rs`](crates/haste/src/image/expand.rs)) expand in forked worker children backed by `proc_macro`, while [`crates/haste/src/names/`](crates/haste/src/names/) resolves module and item scopes on demand.

### 3. Persistent sysroot and three-tier substrate cache

On the first run for a toolchain, `macros.rs` indexes `rust-src` and writes a binary sysroot cache (`HSTSYS01` in [`crates/haste/src/substrate/sysroot.rs`](crates/haste/src/substrate/sysroot.rs)). Later CLI runs map that cache instead of re-parsing `core`, `alloc`, and `std`.

Compiled artifacts live in three chained `HSTSUB01` memory-mapped substrate files ([`crates/haste/src/substrate.rs`](crates/haste/src/substrate.rs), [`crates/haste/src/substrate/keys.rs`](crates/haste/src/substrate/keys.rs)):

1. **Toolchain layer** (`toolchain/<hash>.substrate`): compiled standard library items and summaries keyed by toolchain and target fingerprint.
2. **Dependency layer** (`deps/<hash>.substrate`): compiled registry and git dependency items keyed by lockfile dependency graph and active features.
3. **Workspace layer** (`workspace/<hash>.substrate`): compiled workspace items for fast cold starts.

[`crates/haste/src/substrate/noop.rs`](crates/haste/src/substrate/noop.rs) records watched source, manifest, and environment stamps so unchanged `haste build` invocations exit in under 3 ms. With `--daemon` or `haste serve`, [`crates/haste/src/serve.rs`](crates/haste/src/serve.rs) keeps the workspace, type cache, and JIT image resident over a Unix socket and incrementally updates only changed files or crates via `Build::refresh` ([`crates/haste/src/build.rs`](crates/haste/src/build.rs)).

### 4. Member checking, lazy reachability, type inference, and borrow checking

`build`, `run`, `test`, and `build -o` check every concrete and polymorphic function in the selected member crate(s) before running or emitting code (`check_member_items` in [`crates/haste/src/build/batch.rs`](crates/haste/src/build/batch.rs), `check_poly_item` in [`crates/haste/src/sema/summary.rs`](crates/haste/src/sema/summary.rs)). Across external dependencies and the standard library, wave-based reachability ([`crates/haste/src/sema/reach.rs`](crates/haste/src/sema/reach.rs)) starts from `main`, `#[test]`, or `case_*` roots and parses bodies into dense arenas ([`crates/haste/src/body.rs`](crates/haste/src/body.rs)) only for reached `Instance = (ItemRef, Substs)` pairs.

Semantic analysis ([`crates/haste/src/sema/`](crates/haste/src/sema/)) covers:

- Bidirectional type inference, method/autoderef resolution, closure captures, and layout calculation (`infer.rs`, `traits.rs`, `impls.rs`, `layout.rs`).
- Structural `Send` / `Sync` auto traits, concrete and generic trait bounds, match exhaustiveness (`sema/infer/matching/exhaustive.rs`), module/item/field privacy, and checked `const` / `static` evaluation (`evaluate.rs`).
- Ownership, initialization, use-after-move across branches and loops, binding mutability, disjoint named/tuple field loans, NLL shared/mutable aliasing, non-`'static` borrow escapes, and named-lifetime return subtyping (`borrow.rs`).
- Multi-label spanned diagnostic rendering (`diag.rs`) and compact frozen cross-crate polymorphic summaries (`summary/`).

### 5. SSA lowering, generic stencils, and JIT tier-up

[`crates/haste/src/lower/`](crates/haste/src/lower/) lowers typed bodies into `haste-codegen` SSA IR (`FunctionBuilder`), elaborating `async`/`await` state machines (`coroutine.rs`), drop scopes (`drop.rs`), and portable or target-specific SIMD (`simd.rs`).

[`crates/haste/src/lower/stencil.rs`](crates/haste/src/lower/stencil.rs) computes a 128-bit structural `FootprintId` over each generic body's lowered layout, ABI, inlining, and unwind footprint. Generic bodies larger than `HASTE_DICT_MIN_BYTES` (default `512`) with at least `HASTE_DICT_MIN_FANOUT` (default `8`) instantiations can share one dictionary-passing template across monomorphizations instead of re-compiling each specialization. In JIT mode, [`crates/haste/src/image/stencil_jit.rs`](crates/haste/src/image/stencil_jit.rs) tracks call counts and specializes hot dictionary functions into concrete machine code once calls pass `HASTE_JIT_TIER_UP` (default `64`).

### 6. SSA IR, optimization scheduler, and register allocation

[`crates/codegen/src/lib.rs`](crates/codegen/src/lib.rs) splits coroutines (`coroutine.rs`), verifies SSA and CFG dominance invariants (`verify.rs`), runs the demand-driven optimizer (`schedule.rs`), lowers wide integer and runtime helpers (`native.rs`, `overalign.rs`), and invokes the target backend.

[`crates/codegen/src/schedule.rs`](crates/codegen/src/schedule.rs) tracks 3-bit `Domain` (`CFG | MEM | VALUES`) dirty flags so passes re-run only when an earlier pass dirties a domain they read:

- **Debug / JIT (`Effort::Quick`):** runs `Promote`, `Fold`, and `Merge` fixed-point rules followed by `Returns` cleanup (after `Fill`, `Fold`, `Merge` initialization).
- **Release (`Effort::Full`):** adds `Loads`, `Cse`, `Thread`, `IfConv`, `Rotate`, `Unroll`, `Licm`, and `Lsr` to the fixed-point loop, finishing with `Switch`, `Returns`, and `TailRec`. Key components:
  - Scalar and algebraic simplification: `fold.rs`, `algebra.rs`, `cse.rs`, and `merge.rs`.
  - Stack promotion and memory forwarding: `promote.rs`, `promote_subword.rs`, and `loads.rs` guided by escape and frame alias analysis (`mem/`).
  - Range and relational fact propagation: `facts/` tracks integer intervals, excluded constants (`MAX_NOT = 32`), and symbolic inequalities (`MAX_FACTS = 64`, `MAX_RELS = 64`) so `fold/range.rs` removes redundant bounds and overflow traps.
  - Loop and control-flow passes: `rotate.rs`, `licm.rs`, `lsr.rs`, `unroll.rs`, `ifconv.rs`, `thread.rs`, `tailrec.rs`, `slp.rs`, and `switch.rs`.
  - Unwind drop elision: `drop_elide.rs`.

Before register allocation, [`crates/codegen/src/loopargs.rs`](crates/codegen/src/loopargs.rs) lifts loop-invariant reads into explicit loop-header block parameters so inner-loop values keep registers independently of outer call pressure. [`crates/codegen/src/alloc.rs`](crates/codegen/src/alloc.rs) and [`crates/codegen/src/park.rs`](crates/codegen/src/park.rs) perform weighted linear-scan allocation, balance callee-saved registers against targeted spill/reload parking across calls, coalesce copies, and resolve edge copies. [`crates/codegen/src/frame.rs`](crates/codegen/src/frame.rs) shrink-wraps prologues and epilogues. [`crates/codegen/src/wire.rs`](crates/codegen/src/wire.rs) serializes and deserializes compiled IR for substrate storage.

### 7. Execution backends and native executable emitters

- **JIT and hot-reload ([`crates/haste/src/image/`](crates/haste/src/image/), [`crates/codegen/src/memory.rs`](crates/codegen/src/memory.rs)):** JIT code writes into dual-mapped writable/executable memory arenas (`mach_vm_remap` on macOS, dual `mmap` on Unix). Functions compile on first call inside forked child processes (`image/lazy.rs`), and `haste run --watch` (`image/watch.rs`) uses `kqueue` or `inotify` to patch modified function bodies into the live process.
- **Native CPU encoders ([`crates/codegen/src/x64/`](crates/codegen/src/x64/), [`crates/codegen/src/arm64/`](crates/codegen/src/arm64/)):** Emit x86-64 (baseline SSE2 or runtime CPUID-detected LZCNT/BMI/POPCNT/SSE4.1/AVX/AVX2/FMA) and ARM64 (AArch64 scalar, NEON, and bitfield/shifted/conditional-compare fusion) machine code directly.
- **Zero-linker AOT emitters ([`crates/haste/src/aot/`](crates/haste/src/aot/), [`crates/haste/src/image/aot/`](crates/haste/src/image/aot/)):** [`image/aot/icf.rs`](crates/haste/src/image/aot/icf.rs) and [`image/aot/vdce.rs`](crates/haste/src/image/aot/vdce.rs) run identical code folding and virtual dead-code elimination. [`aot.rs`](crates/haste/src/aot.rs) then emits standalone ELF (`aot/elf.rs`), Mach-O with `__thread_vars` TLV descriptors and ad-hoc `codesign` on macOS (`aot/macho.rs`, `aot/tls.rs`), and PE32+ (`aot/pe.rs`) binaries with native unwind tables (`aot/unwind.rs`), runtime shims (`aot/shims.rs`), and optional DWARF `.debug_line` tables (`aot/dwarf.rs`).
- **Prepared mode ([`crates/haste/src/prep.rs`](crates/haste/src/prep.rs)):** On x86-64 Linux, `haste prep` invokes `prep-driver` with `RUSTC_BOOTSTRAP=1` to precompile dependency instances for `haste --release --prepared run`.

## Internal debugging and tuning variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HASTE_NO_SUBSTRATE` | Unset (`0`) | Set to `1` to bypass cached sysroot, dependency, and workspace substrates. |
| `HASTE_DICT_MIN_BYTES` | `512` | Minimum lowered generic body size in bytes before dictionary lowering applies. |
| `HASTE_DICT_MIN_FANOUT` | `8` | Minimum monomorphization fanout before sharing a dictionary body. |
| `HASTE_JIT_TIER_UP` | `64` | Call threshold before a hot dictionary function specializes in JIT (`0` disables tier-up). |
| `HASTE_VERIFY_STENCIL` | Unset (`0`) | Set to `1` to cross-check stencil instantiations against concrete lowering. |
| `HASTE_VERIFY_PASSES` | Unset (`0`) | Set to `1` to run the SSA and dominance verifier after every mutating codegen pass. |
| `HASTE_TRACE` | Unset (`0`) | Function substring filter (`1` matches all) for printing codegen pass summaries. |
| `HASTE_TRACE_IR` | Unset (`0`) | Set to `1` with `HASTE_TRACE` to dump final IR at end of compile, or `passes` to dump full IR after each changing pass. |
| `HASTE_MISSED` | Unset (`0`) | Set to `1` to print aggregated codegen pass timings and stencil cache counters. |
| `HASTE_SERVE_IDLE_SECS` | `120` | Idle shutdown timeout in seconds for spawned `--daemon` workers. |

## References and design influences

- **Iterative dominance (`crates/codegen/src/cfg.rs`):** Cooper, Harvey, and Kennedy (2001), *A Simple, Fast Dominance Algorithm*.
- **Block-parameter SSA and on-demand promotion (`crates/codegen/src/ir.rs`, `promote.rs`):** Block-parameter SSA (Kelsey 1995; Appel 1998) with demand-driven SSA construction in the spirit of Braun et al. (2013), *Simple and Efficient Construction of Static Single Assignment Form* (CC 2013).
- **Linear-scan register allocation and out-of-SSA copy resolution (`crates/codegen/src/park.rs`, `alloc.rs`, `frame.rs`):** Poletto and Sarkar (1999), *Linear Scan Register Allocation* (TOPLAS); Wimmer and Franz (2010), *Linear Scan Register Allocation on SSA Form* (CGO 2010); Boissinot et al. (2009), *Revisiting Out-of-SSA Translation for Correctness, Code Quality, and Efficiency* (CGO 2009).
- **Shrink-wrapping (`crates/codegen/src/frame.rs`):** Fred C. Chow (1988), *Minimizing Register Usage Penalty at Procedure Calls* (PLDI 1988).
- **Division by invariant integers via multiplication (`crates/codegen/src/algebra.rs`):** Granlund and Montgomery (1994), *Division by Invariant Integers using Multiplication* (PLDI 1994).
- **Superword-level parallelism (`crates/codegen/src/slp.rs`):** Larsen and Amarasinghe (2000), *Exploiting Superword Level Parallelism with Multimedia Instruction Sets* (PLDI 2000).
- **Dictionary passing and stencil instantiation (`crates/haste/src/lower/stencil.rs`, `image/stencil_jit.rs`):** Hall et al. (1996), *Type Classes in Haskell* (TOPLAS); Xu and Kjolstad (2021), *Copy-and-Patch Compilation* (OOPSLA 2021).
- **Identical Code Folding (`crates/haste/src/image/aot/icf.rs`):** Tallam et al. (2010), *Safe ICF: Pointer Safe and Unwinding Aware Identical Code Folding in the Gold Linker*.
- **Non-Lexical Lifetimes (`crates/haste/src/sema/borrow.rs`):** Rust RFC 2094 (*Non-Lexical Lifetimes*).
- **Match exhaustiveness (`crates/haste/src/sema/infer/matching/exhaustive.rs`):** Luc Maranget (2007), *Warnings for Pattern Matching* (JFP 2007).
