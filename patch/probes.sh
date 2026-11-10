#!/bin/bash
set -u
cd "$(dirname "$0")/.."
source patch/cargo.sh
run_cargo build --quiet --release -p haste </dev/null >/dev/null 2>&1 || { echo "haste does not build"; exit 2; }
failed=0
for dir in patch/probes/*/; do
    name=$(basename "$dir")
    expected=$(run_cargo run --quiet --manifest-path "$dir/Cargo.toml" </dev/null 2>/dev/null); expected_status=$?
    found=$(run_cargo run --quiet --release -p haste -- run "$dir" </dev/null 2>/dev/null); found_status=$?
    if [ "$expected" = "$found" ] && [ $((expected_status == 0)) = $((found_status == 0)) ]; then
        echo "$name same"
    else
        echo "$name DIFF: status $found_status, cargo $expected_status: $(printf '%s\n' "$found" | tail -2 | cut -c1-160)"
        failed=1
    fi
done
exit $failed
