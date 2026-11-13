#!/bin/bash
set -u
cd "$(dirname "$0")/.."
source scripts/cargo.sh
build_haste </dev/null >/dev/null 2>&1 || { echo "haste does not build"; exit 2; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
with_timeout() {
    perl -e 'alarm shift; exec @ARGV' 60 "$@"
}
for build_target in examples/hello tests/diff tests/probes/cl; do
    build_out=$(run_haste build "$build_target" </dev/null 2>&1); build_status=$?
    if [ "$build_status" -ne 0 ]; then
        echo "build $build_target DIFF: status $build_status: $(printf '%s\n' "$build_out" | tail -2 | cut -c1-160)"
        exit 1
    fi
done
check_reject() {
    local label=$1 body=$2 dir="$tmp/reject_$1" c_status b_status r_status
    mkdir -p "$dir/src"
    printf '[package]\nname = "reject_%s"\nversion = "0.1.0"\nedition = "2021"\n' "$label" >"$dir/Cargo.toml"
    cp examples/hello/Cargo.lock "$dir/Cargo.lock"
    printf '%s\n' "$body" >"$dir/src/main.rs"
    run_cargo check --quiet --locked --manifest-path "$dir/Cargo.toml" </dev/null >/dev/null 2>&1; c_status=$?
    run_haste build "$dir" </dev/null >/dev/null 2>&1; b_status=$?
    run_haste run "$dir" </dev/null >/dev/null 2>&1; r_status=$?
    if [ "$c_status" -eq 0 ] || [ "$b_status" -eq 0 ] || [ "$r_status" -eq 0 ]; then
        echo "reject $label DIFF: cargo=$c_status build=$b_status run=$r_status"
        return 1
    fi
    return 0
}
check_reject unsatisfied_bound 'trait Marker {} fn needs<T: Marker>(_: T) {} fn main() { needs(1u8); }' || exit 1
check_reject unconstrained_var 'fn main() { let _v = Vec::new(); }' || exit 1
mkdir -p "$tmp/malformed/src"
printf '[package]\nname = "malformed"\nversion = "0.1.0"\nedition = "2021"\n' >"$tmp/malformed/Cargo.toml"
cp examples/hello/Cargo.lock "$tmp/malformed/Cargo.lock"
printf 'macro x' >"$tmp/malformed/src/main.rs"
run_haste debug items "$tmp/malformed" </dev/null >/dev/null 2>"$tmp/malformed.err"; mal_status=$?
if [ "$mal_status" -ge 128 ] || grep -q 'panicked at' "$tmp/malformed.err"; then
    echo "malformed macro DIFF: status $mal_status: $(tail -2 "$tmp/malformed.err")"
    exit 1
fi
dirs=(tests/probes/*/)
total=${#dirs[@]}
jobs=$(cpu_count)
[ "$jobs" -gt "$total" ] && jobs=$total
[ "$jobs" -lt 1 ] && jobs=1
probe_one() {
    local dir=$1 out=$2 name expected expected_status found found_status aot_out aot_status
    name=$(basename "$dir")
    expected=$(run_manifest "$dir" </dev/null 2>/dev/null); expected_status=$?
    if [ -n "${HASTE_BIN:-}" ]; then
        found=$(with_timeout "$HASTE_BIN" run "$dir" </dev/null 2>/dev/null); found_status=$?
    else
        found=$(run_haste run "$dir" </dev/null 2>/dev/null); found_status=$?
    fi
    if [ "$name" = "so" ]; then
        if [ "$expected" = "ready" ] && [ "$found" = "ready" ] && [ "$expected_status" -ne 0 ] && [ "$found_status" -ge 128 ]; then
            echo "$name same" >"$out"
            return 0
        fi
        echo "$name DIFF: status $found_status (expected=$found), cargo $expected_status ($expected)" >"$out"
        return 1
    fi
    if [ "$expected_status" -ne 0 ] || [ "$found_status" -ne 0 ] || [ "$expected" != "$found" ]; then
        echo "$name DIFF: status $found_status, cargo $expected_status: $(printf '%s\n' "$found" | tail -2 | cut -c1-160)" >"$out"
        return 1
    fi
    if ! run_haste --release build -o "$tmp/aot_$name" "$dir" </dev/null >/dev/null 2>"$tmp/aot_$name.err"; then
        echo "$name AOT BUILD DIFF: $(tail -2 "$tmp/aot_$name.err" | cut -c1-160)" >"$out"
        return 1
    fi
    aot_out=$(with_timeout "$tmp/aot_$name" </dev/null 2>/dev/null); aot_status=$?
    if [ "$aot_status" -ne 0 ] || [ "$aot_out" != "$expected" ]; then
        echo "$name AOT RUN DIFF: status $aot_status: $(printf '%s\n' "$aot_out" | tail -2 | cut -c1-160)" >"$out"
        return 1
    fi
    echo "$name same" >"$out"
}
for ((w = 0; w < jobs; w++)); do
    (
        for ((i = 0; i < total; i++)); do
            mkdir "$tmp/claim.$i" 2>/dev/null || continue
            probe_one "${dirs[$i]}" "$tmp/out.$i" || : >"$tmp/failed"
        done
    ) &
done
wait
for ((i = 0; i < total; i++)); do
    cat "$tmp/out.$i"
done
[ ! -e "$tmp/failed" ]
