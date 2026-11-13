#!/bin/bash
set -u
cd "$(dirname "$0")/.."
source scripts/cargo.sh
dir="${1:-tests/diff}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
run_manifest "$dir" </dev/null >"$tmp/rustc.out" 2>"$tmp/rustc.err" &
pid_rustc=$!
build_haste </dev/null >"$tmp/build.err" 2>&1 || { cat "$tmp/build.err"; echo "engine did not build"; exit 2; }
run_haste test "$dir" </dev/null >"$tmp/engine" 2>&1 &
pid_engine=$!
(env -u HASTE_VERIFY_STENCIL bash -c 'source scripts/cargo.sh && run_haste --release build -o "$1" "$2" && "$1"' _ "$tmp/aot_bin" "$dir") </dev/null >"$tmp/aot.out" 2>"$tmp/aot.err" &
pid_aot=$!
wait "$pid_rustc"; status_rustc=$?
wait "$pid_engine"
wait "$pid_aot"; status_aot=$?
rustc_side=$(grep '^case_' "$tmp/rustc.out" | sort)
engine_all=$(cat "$tmp/engine"); finished='^cases '
engine_side=$(printf '%s\n' "$engine_all" | grep '^case_' | sort)
aot_side=$(grep '^case_' "$tmp/aot.out" | sort)
printf '%s\n' "$engine_all" | grep -E "$finished" | cut -c1-200
[ "$status_rustc" -eq 0 ] && [ -n "$rustc_side" ] || { echo "no rustc output (status $status_rustc)"; cat "$tmp/rustc.err" "$tmp/rustc.out" | grep -E '^error|panicked' -A4 | head -20 | cut -c1-160; exit 2; }
[ -n "$engine_all" ] || { echo "engine did not start"; exit 2; }
printf '%s\n' "$engine_all" | grep -q "$finished" || { printf '%s\n' "$engine_all" | grep '^stub ' | tail -4 | cut -c1-220; echo "died at: $(printf '%s\n' "$engine_all" | grep '^running ' | tail -1)"; exit 1; }
check_side() {
    local label=$1 side=$2 wrong missing
    wrong=$(comm -13 <(printf '%s\n' "$rustc_side") <(printf '%s\n' "$side"))
    if [ -n "$wrong" ]; then
        echo "MISMATCH ($label):"
        printf '%s\n' "$wrong" | while read -r name input value; do
            expected=$(printf '%s\n' "$rustc_side" | grep "^$name $input " | cut -d' ' -f3)
            echo "$name($input) $label=$value rustc=$expected"
        done
        exit 1
    fi
    missing=$(comm -23 <(printf '%s\n' "$rustc_side") <(printf '%s\n' "$side") | cut -d' ' -f1 | sort -u)
    if [ -n "$missing" ]; then
        echo "MISSING ($label): no result for $(printf '%s\n' "$missing" | tr '\n' ' ')"
        exit 1
    fi
}
check_side jit "$engine_side"
if [ "$status_aot" -ne 0 ]; then
    echo "AOT FAILED (status $status_aot):"
    cat "$tmp/aot.err" | tail -20 | cut -c1-200
    exit 1
fi
check_side aot "$aot_side"
echo "agree=$(comm -12 <(printf '%s\n' "$rustc_side") <(printf '%s\n' "$engine_side") | wc -l | tr -d ' ') of $(printf '%s\n' "$rustc_side" | wc -l | tr -d ' ') (jit+aot)"
