# Security Policy

## Supported versions

`haste` is currently experimental (`version = "0.0.0"`, no tagged release yet). Security fixes are applied to the `main` branch on a best-effort basis.

## Security model

- Like `cargo`, `haste` (both the CLI and library methods such as `Compiler::check`, `test`, and `emit`) executes crate `build.rs` scripts and procedural macros unsandboxed with the invoking user's privileges. Only build trusted workspaces directly; untrusted or customer-submitted repositories require isolated, resource-limited workers without service credentials.
- `haste serve` listens on a per-user Unix domain socket (abstract namespace on Linux, temp file on macOS) and verifies the peer UID on every connection.
- Dependency lookup matches `Cargo.lock` `source` URLs against `.cargo/config.toml` `[source.*]` / `replace-with` entries. When `Cargo.lock` records a checksum, `haste` verifies `.cargo-checksum.json`'s `"package"` hash fail-closed against `Cargo.lock` when `.cargo-checksum.json` is present (if the field is missing or differs, the crate is rejected; individual unpacked files on disk are not re-hashed), and falls back to Cargo's `.cargo-ok` marker when an extracted registry crate directory omits `.cargo-checksum.json`.
- When locked registry or git crates are missing locally, `haste` runs `cargo fetch --locked` automatically unless `CARGO_NET_OFFLINE=true` or `[net] offline = true` is set, or dependencies are vendored via `cargo vendor` / `.cargo/config.toml`.
- Teams piloting `haste` can pin an exact revision (`cargo install --git https://github.com/manaforged/haste --rev <SHA> --locked haste`), verify it with `haste --version`, and roll back by reinstalling a known-good `--rev` or `--tag`.

## Reporting a Vulnerability

Please report security vulnerabilities privately via [GitHub Security Advisories](https://github.com/manaforged/haste/security/advisories/new) rather than opening a public issue.

Include:

- The `haste --version` output, OS, and CPU architecture.
- The exact command and a minimal reproduction (manifest, source file, and `Cargo.lock` if relevant).
- Expected vs observed behavior (for example, memory safety issues in JIT/AOT codegen, build-script isolation, or daemon IPC).
