#!/bin/bash
set -u
cd "$(dirname "$0")/.."
dir="${1:-patch/diff}"
rustc_side=$(cargo run --quiet --manifest-path $dir/Cargo.toml </dev/null 2>/dev/null | grep '^case_' | sort)
engine_all=$(cargo run --quiet --release -- check $dir </dev/null 2>/dev/null)
engine_side=$(printf '%s\n' "$engine_all" | grep '^case_' | sort)
printf '%s\n' "$engine_all" | grep '^summary' | cut -c1-200
[ -n "$rustc_side" ] || { echo "no rustc output"; cargo run --quiet --manifest-path $dir/Cargo.toml </dev/null 2>&1 | grep -E '^error|panicked' -A4 | head -20 | cut -c1-160; exit 2; }
[ -n "$engine_all" ] || { echo "engine did not build or start"; exit 2; }
printf '%s\n' "$engine_all" | grep -q '^summary ' || { printf '%s\n' "$engine_all" | grep '^stub ' | tail -4 | cut -c1-220; echo "died at: $(printf '%s\n' "$engine_all" | grep '^running ' | tail -1)"; exit 1; }
wrong=$(comm -13 <(printf '%s\n' "$rustc_side") <(printf '%s\n' "$engine_side"))
if [ -n "$wrong" ]; then
    echo "MISMATCH:"
    printf '%s\n' "$wrong" | while read -r name input value; do
        expected=$(printf '%s\n' "$rustc_side" | grep "^$name $input " | cut -d' ' -f3)
        echo "$name($input) engine=$value rustc=$expected"
    done
    exit 1
fi
echo "agree=$(comm -12 <(printf '%s\n' "$rustc_side") <(printf '%s\n' "$engine_side") | wc -l | tr -d ' ') of $(printf '%s\n' "$rustc_side" | wc -l | tr -d ' ')"
