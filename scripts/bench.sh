#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cargo.sh

build_haste </dev/null >/dev/null 2>&1 || { echo "haste does not build"; exit 2; }
haste_bin="${HASTE_BIN:-${CARGO_TARGET_DIR:-target}/release/haste}"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

now_ms() {
    python3 -c 'import time; print(int(time.perf_counter() * 1000))'
}

median_ms() {
    local rounds=$1; shift
    local times=()
    for ((i = 0; i < rounds; i++)); do
        local t0 t1
        t0=$(now_ms)
        "$@" >/dev/null 2>&1
        t1=$(now_ms)
        times+=($((t1 - t0)))
    done
    printf '%s\n' "${times[@]}" | sort -n | sed -n "$(((rounds + 1) / 2))p"
}

echo "Host: $(uname -s) $(uname -m) | $(rustc --version)"
echo

# 1. Copy examples/hello and tests/diff into isolated scratch dirs for fair cold/edit/noop timing
cp -R examples/hello "$tmp/hello"
cp -R tests/diff "$tmp/diff"
cp tests/case_inputs.rs "$tmp/case_inputs.rs"

haste_home="$tmp/haste_home"
cargo_target="$tmp/cargo_target"

# Warm up haste stdlib substrate once (just like a user after first run)
HASTE_HOME="$haste_home" "$haste_bin" build "$tmp/hello" >/dev/null 2>&1

# Benchmark 1: examples/hello run (warm, median of 5)
cargo build --quiet --locked --manifest-path "$tmp/hello/Cargo.toml" --target-dir "$cargo_target"
h_hello=$(HASTE_HOME="$haste_home" median_ms 5 "$haste_bin" run "$tmp/hello")
c_hello=$(median_ms 5 cargo run --quiet --locked --manifest-path "$tmp/hello/Cargo.toml" --target-dir "$cargo_target")

# Benchmark 2: tests/diff cold build (debug)
t0=$(now_ms); HASTE_HOME="$haste_home" "$haste_bin" build "$tmp/diff" >/dev/null 2>&1; t1=$(now_ms)
h_diff_cold=$((t1 - t0))
t0=$(now_ms); cargo build --quiet --locked --manifest-path "$tmp/diff/Cargo.toml" --target-dir "$cargo_target" >/dev/null 2>&1; t1=$(now_ms)
c_diff_cold=$((t1 - t0))

# Benchmark 3: tests/diff unchanged rebuild (no-op, median of 5)
h_diff_noop=$(HASTE_HOME="$haste_home" median_ms 5 "$haste_bin" build "$tmp/diff")
c_diff_noop=$(median_ms 5 cargo build --quiet --locked --manifest-path "$tmp/diff/Cargo.toml" --target-dir "$cargo_target")

# Benchmark 4: tests/diff single-line edit rebuild (median of 3)
h_edits=()
c_edits=()
for i in 1 2 3; do
    printf '\n// edit %d\n' "$i" >> "$tmp/diff/src/main.rs"
    t0=$(now_ms); HASTE_HOME="$haste_home" "$haste_bin" build "$tmp/diff" >/dev/null 2>&1; t1=$(now_ms)
    h_edits+=($((t1 - t0)))
    t0=$(now_ms); cargo build --quiet --locked --manifest-path "$tmp/diff/Cargo.toml" --target-dir "$cargo_target" >/dev/null 2>&1; t1=$(now_ms)
    c_edits+=($((t1 - t0)))
done
h_diff_edit=$(printf '%s\n' "${h_edits[@]}" | sort -n | sed -n '2p')
c_diff_edit=$(printf '%s\n' "${c_edits[@]}" | sort -n | sed -n '2p')

# Benchmark 5: tests/diff --release standalone AOT build & runtime
t0=$(now_ms); HASTE_HOME="$haste_home" "$haste_bin" --release build -o "$tmp/diff_haste" "$tmp/diff" >/dev/null 2>&1; t1=$(now_ms)
h_aot_build=$((t1 - t0))
t0=$(now_ms); cargo build --quiet --locked --release --manifest-path "$tmp/diff/Cargo.toml" --target-dir "$cargo_target" >/dev/null 2>&1; t1=$(now_ms)
c_aot_build=$((t1 - t0))

# Verify identical output between haste AOT and rustc -O3
"$tmp/diff_haste" > "$tmp/h.out" 2>/dev/null
"$cargo_target/release/diff" > "$tmp/c.out" 2>/dev/null
diff -u "$tmp/c.out" "$tmp/h.out" >/dev/null

h_aot_run=$(median_ms 5 "$tmp/diff_haste")
c_aot_run=$(median_ms 5 "$cargo_target/release/diff")

echo "| Benchmark | haste | cargo / rustc |"
echo "| --- | ---: | ---: |"
echo "| \`examples/hello\` warm run | ${h_hello} ms | ${c_hello} ms |"
echo "| \`tests/diff\` (124 fns) cold debug build | ${h_diff_cold} ms | ${c_diff_cold} ms |"
echo "| \`tests/diff\` single-line edit rebuild | ${h_diff_edit} ms | ${c_diff_edit} ms |"
echo "| \`tests/diff\` unchanged no-op build | ${h_diff_noop} ms | ${c_diff_noop} ms |"
echo "| \`tests/diff\` \`--release\` AOT build | ${h_aot_build} ms | ${c_aot_build} ms |"
echo "| \`tests/diff\` \`--release\` binary runtime (identical output) | ${h_aot_run} ms | ${c_aot_run} ms |"
