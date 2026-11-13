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
write_scratch_crate() {
    local dir=$1 pkg=$2 body=$3
    mkdir -p "$dir/src"
    printf '[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2021"\n' "$pkg" >"$dir/Cargo.toml"
    printf 'version = 4\n\n[[package]]\nname = "%s"\nversion = "0.1.0"\n' "$pkg" >"$dir/Cargo.lock"
    printf '%s\n' "$body" >"$dir/src/main.rs"
}
check_build_target() {
    local build_target=$1 out_msg=$2 build_out build_status
    build_out=$(run_haste build "$build_target" </dev/null 2>&1); build_status=$?
    if [ "$build_status" -ne 0 ] || ! printf '%s\n' "$build_out" | grep -Eq '^[1-9][0-9]* functions reached, 0 failed'; then
        echo "build $build_target DIFF: status $build_status: $(printf '%s\n' "$build_out" | tail -2 | cut -c1-160)" >"$out_msg"
        return 1
    fi
}
check_accept_ok() {
    local out_msg=$1 accept_out accept_b=0 accept_c=0 p_b p_c
    write_scratch_crate "$tmp/accept_ok" "accept_ok" 'fn main() {}'
    run_haste build "$tmp/accept_ok" </dev/null >"$tmp/accept_ok.out" 2>&1 &
    p_b=$!
    run_cargo check --quiet --locked --manifest-path "$tmp/accept_ok/Cargo.toml" </dev/null >/dev/null 2>&1 &
    p_c=$!
    wait "$p_b" || accept_b=$?
    wait "$p_c" || accept_c=$?
    accept_out=$(cat "$tmp/accept_ok.out")
    if [ "$accept_c" -ne 0 ] || [ "$accept_b" -ne 0 ] || ! printf '%s\n' "$accept_out" | grep -Eq '^[1-9][0-9]* functions reached, 0 failed'; then
        echo "accept_ok DIFF: cargo=$accept_c build=$accept_b: $(printf '%s\n' "$accept_out" | tail -2 | cut -c1-160)" >"$out_msg"
        return 1
    fi
}
check_reject() {
    local label=$1 body=$2 out_msg=$3 dir="$tmp/reject_$1" pkg="reject_$1" c_status=0 h_status=0 p_c p_h
    write_scratch_crate "$dir" "$pkg" "$body"
    run_cargo check --quiet --locked --manifest-path "$dir/Cargo.toml" </dev/null >/dev/null 2>&1 &
    p_c=$!
    (
        local b_status=0 r_status=0
        run_haste build "$dir" </dev/null >/dev/null 2>&1 || b_status=$?
        run_haste run "$dir" </dev/null >/dev/null 2>&1 || r_status=$?
        printf '%s %s\n' "$b_status" "$r_status" >"$dir/haste.status"
        [ "$b_status" -ne 0 ] && [ "$r_status" -ne 0 ]
    ) &
    p_h=$!
    wait "$p_c" || c_status=$?
    wait "$p_h" || h_status=$?
    if [ "$c_status" -eq 0 ] || [ "$h_status" -ne 0 ]; then
        local hr
        hr=$(cat "$dir/haste.status" 2>/dev/null || echo "? ?")
        echo "reject $label DIFF: cargo=$c_status build/run=($hr)" >"$out_msg"
        return 1
    fi
}
check_malformed() {
    local out_msg=$1 mal_status=0
    write_scratch_crate "$tmp/malformed" "malformed" 'macro x'
    run_haste debug items "$tmp/malformed" </dev/null >/dev/null 2>"$tmp/malformed.err" || mal_status=$?
    if [ "$mal_status" -ge 128 ] || grep -q 'panicked at' "$tmp/malformed.err"; then
        echo "malformed macro DIFF: status $mal_status: $(tail -2 "$tmp/malformed.err")" >"$out_msg"
        return 1
    fi
}
pre_pids=()
pre_msgs=()
pre_launch() {
    local msg="$tmp/pre.${#pre_pids[@]}.msg"
    "$@" "$msg" &
    pre_pids+=($!)
    pre_msgs+=("$msg")
}
for build_target in examples/hello tests/diff tests/probes/cl; do
    pre_launch check_build_target "$build_target"
done
pre_launch check_accept_ok
pre_launch check_reject unsatisfied_bound 'trait Marker {} fn needs<T: Marker>(_: T) {} fn main() { needs(1u8); }'
pre_launch check_reject unconstrained_var 'fn main() { let _v = Vec::new(); }'
pre_launch check_malformed
pre_fail=0
for idx in "${!pre_pids[@]}"; do
    if ! wait "${pre_pids[$idx]}"; then
        [ -s "${pre_msgs[$idx]}" ] && cat "${pre_msgs[$idx]}"
        pre_fail=1
    fi
done
[ "$pre_fail" -ne 0 ] && exit 1
dirs=(tests/probes/*/)
total=${#dirs[@]}
jobs=$(cpu_count)
[ "$jobs" -gt "$total" ] && jobs=$total
[ "$jobs" -lt 1 ] && jobs=1
probe_one() {
    local dir=$1 out=$2 name expected expected_status=0 found found_status=0 aot_out aot_status=0 p_exp
    name=$(basename "$dir")
    run_manifest "$dir" </dev/null >"$out.exp" 2>/dev/null &
    p_exp=$!
    if [ -n "${HASTE_BIN:-}" ]; then
        found=$(with_timeout "$HASTE_BIN" run "$dir" </dev/null 2>/dev/null) || found_status=$?
    else
        found=$(run_haste run "$dir" </dev/null 2>/dev/null) || found_status=$?
    fi
    if [ "$name" = "so" ]; then
        wait "$p_exp" || expected_status=$?
        expected=$(cat "$out.exp")
        if [ "$expected" = "ready" ] && [ "$found" = "ready" ] && [ "$expected_status" -ne 0 ] && [ "$found_status" -ge 128 ]; then
            echo "$name same" >"$out"
            return 0
        fi
        echo "$name DIFF: status $found_status (expected=$found), cargo $expected_status ($expected)" >"$out"
        return 1
    fi
    if [ "$found_status" -ne 0 ]; then
        wait "$p_exp" || expected_status=$?
        echo "$name DIFF: status $found_status, cargo $expected_status: $(printf '%s\n' "$found" | tail -2 | cut -c1-160)" >"$out"
        return 1
    fi
    if ! run_haste --release build -o "$tmp/aot_$name" "$dir" </dev/null >/dev/null 2>"$tmp/aot_$name.err"; then
        wait "$p_exp" || expected_status=$?
        echo "$name AOT BUILD DIFF: $(tail -2 "$tmp/aot_$name.err" | cut -c1-160)" >"$out"
        return 1
    fi
    aot_out=$(with_timeout "$tmp/aot_$name" </dev/null 2>/dev/null) || aot_status=$?
    wait "$p_exp" || expected_status=$?
    expected=$(cat "$out.exp")
    if [ "$expected_status" -ne 0 ] || [ "$expected" != "$found" ]; then
        echo "$name DIFF: status $found_status, cargo $expected_status: $(printf '%s\n' "$found" | tail -2 | cut -c1-160)" >"$out"
        return 1
    fi
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
