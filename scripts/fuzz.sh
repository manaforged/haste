#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cargo.sh

start="${1:-0}"
end="${2:-40}"
jobs="${FUZZ_JOBS:-4}"
shopt -s nullglob
if [ "$start" -ge "$end" ]; then
    echo "fuzz agree=0 of 0 (seeds $start..$end, jit+aot)"
    exit 0
fi
work="${CARGO_TARGET_DIR:-target}/fuzz.$$"
mkdir -p "$work"
trap 'rm -rf "$work"' EXIT

build_haste </dev/null >"$work/build.err" 2>&1 || {
    cat "$work/build.err"
    echo "engine did not build"
    exit 2
}
haste_bin="${HASTE_BIN:-${CARGO_TARGET_DIR:-target}/release/haste}"
if [ ! -x "$haste_bin" ]; then
    echo "haste binary not found at $haste_bin"
    exit 2
fi

python3 scripts/fuzz_gen.py "$start" "$end" "$work/src"
mkdir -p "$work/out"

run_with_timeout() {
    perl -e 'alarm shift; exec @ARGV' "$@"
}

for f in "$work/src"/*.rs; do
    n=$(basename "$f" .rs)
    c="$work/out/$n"
    mkdir -p "$c/src"
    cp "$f" "$c/src/main.rs"
    printf '[package]\nname = "p_%s"\nversion = "0.1.0"\nedition = "2021"\n' "$n" >"$c/Cargo.toml"
    printf 'version = 4\n\n[[package]]\nname = "p_%s"\nversion = "0.1.0"\n' "$n" >"$c/Cargo.lock"
    (
        set +e
        rustc --edition 2021 -O -o "$c/bin_rustc" "$f" >"$c/rustc.err" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "$n | rustc=FAIL($rc)" >"$c/fail"
            exit 0
        fi
        run_with_timeout 30 "$c/bin_rustc" >"$c/rustc.out" 2>"$c/rustc.run.err"
        rr=$?
        rustc --edition 2021 -o "$c/bin_rustc_dbg" "$f" >"$c/rustc_dbg.err" 2>&1
        rdc=$?
        if [ "$rdc" -eq 0 ]; then
            run_with_timeout 30 "$c/bin_rustc_dbg" >"$c/rustc_dbg.out" 2>"$c/rustc_dbg.run.err"
            rdr=$?
        else
            rdr=1
        fi
        run_with_timeout 60 "$haste_bin" run "$c" >"$c/jit.out" 2>"$c/jit.err" </dev/null
        jr=$?
        run_with_timeout 60 "$haste_bin" --release build -o "$c/bin_aot" "$c" >"$c/aot.build" 2>&1 </dev/null
        ab=$?
        if [ "$ab" -eq 0 ]; then
            run_with_timeout 30 "$c/bin_aot" >"$c/aot.out" 2>"$c/aot.run.err"
            ar=$?
        else
            ar=1
        fi
        js=same
        as=same
        [ "$rdc" -eq 0 ] && [ "$rdr" -eq 0 ] && [ "$jr" -eq 0 ] && cmp -s "$c/rustc_dbg.out" "$c/jit.out" || js=DIFF
        [ "$rr" -eq 0 ] && [ "$ab" -eq 0 ] && [ "$ar" -eq 0 ] && cmp -s "$c/rustc.out" "$c/aot.out" || as=DIFF
        if [ "$js" != same ] || [ "$as" != same ]; then
            echo "$n | jit=rc$jr $js | aot=build$ab/rc$ar $as" >"$c/fail"
        fi
    ) &
    while [ "$(jobs -r | wc -l)" -ge "$jobs" ]; do
        sleep 0.1
    done
done
wait

total=$((end - start))
fails=$(find "$work/out" -name fail | sort)
if [ -n "$fails" ]; then
    echo "FUZZ MISMATCH:"
    while IFS= read -r fp; do
        cat "$fp"
    done <<< "$fails"
    exit 1
fi
echo "fuzz agree=$total of $total (seeds $start..$end, jit+aot)"
