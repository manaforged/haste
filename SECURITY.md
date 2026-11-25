# Security Policy

## Supported Versions

`haste` is currently experimental (`version = "0.0.0"`, no tagged release yet). Security fixes are applied to the `main` branch.

## Reporting a Vulnerability

Please report security vulnerabilities privately via [GitHub Security Advisories](https://github.com/manaforged/haste/security/advisories/new) rather than opening a public issue.

Include:

- The `haste --version` output, OS, and CPU architecture.
- The exact command and a minimal reproduction (manifest, source file, and `Cargo.lock` if relevant).
- Expected vs observed behavior (for example, memory safety issues in JIT/AOT codegen, build-script isolation, or daemon IPC).
