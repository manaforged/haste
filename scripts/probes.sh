#!/bin/bash
set -u
cd "$(dirname "$0")/.."
source scripts/cargo.sh
build_haste </dev/null >/dev/null 2>&1 || { echo "haste does not build"; exit 2; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
dirs=(tests/probes/*/)
total=${#dirs[@]}
jobs=$(cpu_count)
[ "$jobs" -gt "$total" ] && jobs=$total
[ "$jobs" -lt 1 ] && jobs=1
probe_one() {
    local dir=$1 out=$2 name expected expected_status found found_status
    name=$(basename "$dir")
    expected=$(run_manifest "$dir" </dev/null 2>/dev/null); expected_status=$?
    found=$(run_haste run "$dir" </dev/null 2>/dev/null); found_status=$?
    if [ "$expected" = "$found" ] && [ $((expected_status == 0)) = $((found_status == 0)) ]; then
        echo "$name same" >"$out"
    else
        echo "$name DIFF: status $found_status, cargo $expected_status: $(printf '%s\n' "$found" | tail -2 | cut -c1-160)" >"$out"
        return 1
    fi
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
