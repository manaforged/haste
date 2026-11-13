#!/bin/bash
set -u
cd "$(dirname "$0")/.."
source scripts/cargo.sh
dir="${1:-tests/diff}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
run_manifest "$dir" </dev/null >"$tmp/rustc.out" 2>"$tmp/rustc.err" &
pid_rustc=$!
(build_haste && run_haste test "$dir") </dev/null >"$tmp/engine" 2>&1 &
pid_engine=$!
wait "$pid_rustc"
wait "$pid_engine"
rustc_side=$(grep '^case_' "$tmp/rustc.out" | sort)
engine_all=$(cat "$tmp/engine"); finished='^cases '
engine_side=$(printf '%s\n' "$engine_all" | grep '^case_' | sort)
printf '%s\n' "$engine_all" | grep -E "$finished" | cut -c1-200
[ -n "$rustc_side" ] || { echo "no rustc output"; cat "$tmp/rustc.err" "$tmp/rustc.out" | grep -E '^error|panicked' -A4 | head -20 | cut -c1-160; exit 2; }
[ -n "$engine_all" ] || { echo "engine did not build or start"; exit 2; }
printf '%s\n' "$engine_all" | grep -q "$finished" || { printf '%s\n' "$engine_all" | grep '^stub ' | tail -4 | cut -c1-220; echo "died at: $(printf '%s\n' "$engine_all" | grep '^running ' | tail -1)"; exit 1; }
wrong=$(comm -13 <(printf '%s\n' "$rustc_side") <(printf '%s\n' "$engine_side"))
if [ -n "$wrong" ]; then
    echo "MISMATCH:"
    printf '%s\n' "$wrong" | while read -r name input value; do
        expected=$(printf '%s\n' "$rustc_side" | grep "^$name $input " | cut -d' ' -f3)
        echo "$name($input) engine=$value rustc=$expected"
    done
    exit 1
fi
missing=$(comm -23 <(printf '%s\n' "$rustc_side") <(printf '%s\n' "$engine_side") | cut -d' ' -f1 | sort -u)
if [ -n "$missing" ]; then
    echo "MISSING: no result for $(printf '%s\n' "$missing" | tr '\n' ' ')"
    printf '%s\n' "$engine_all" | grep -E '^ +[0-9]+ |^       ' | cut -c1-200
    exit 1
fi
echo "agree=$(comm -12 <(printf '%s\n' "$rustc_side") <(printf '%s\n' "$engine_side") | wc -l | tr -d ' ') of $(printf '%s\n' "$rustc_side" | wc -l | tr -d ' ')"
