#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LANE_NAME="${1:-$(basename "$ROOT")}"
LOG_FILE="${HASTE_CLIMB_LOG:-$HOME/build/haste-climb-log.jsonl}"
OS="$(uname -s)"

if [ "$OS" = "Linux" ]; then
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/build/target/dev/$LANE_NAME}"
elif [ "$OS" = "Darwin" ]; then
  unset CARGO_TARGET_DIR || true
fi

HASTE_HOME="$(mktemp -d)"
export HASTE_HOME
unset HASTE_BIN || true
trap 'rm -rf "$HASTE_HOME"' EXIT

t0=$(date +%s)

echo "[gate:$LANE_NAME] 1/7 format & file-length ceilings (<800 lines)"
cargo fmt --all --check
OVER=$(find crates -name '*.rs' -exec wc -l {} + | awk '$2 != "total" && $1 >= 800 {print $1, $2}')
if [ -n "$OVER" ]; then
  echo "FAIL: files >= 800 lines:"
  echo "$OVER"
  exit 1
fi

if [ -f "$HOME/.agents/library/complexity/scripts/cc.py" ]; then
  echo "[gate:$LANE_NAME] 2/7 cognitive complexity (CC <= 50)"
  CC_OUT=$(python3 "$HOME/.agents/library/complexity/scripts/cc.py" --lang rust --max 50 crates)
  echo "$CC_OUT" | tail -n 3
  echo "$CC_OUT" | grep -q '51+: 0' || { echo "FAIL: CC > 50"; exit 1; }
fi

echo "[gate:$LANE_NAME] 3/7 workspace --lib (-D warnings)"
RUSTFLAGS="-D warnings" cargo test --workspace --lib --quiet

echo "[gate:$LANE_NAME] 4/7 compiler_api & soundness (sequential, --test-threads=2)"
cargo test --test compiler_api --quiet -- --test-threads=2
cargo test --test soundness --quiet -- --test-threads=2

echo "[gate:$LANE_NAME] 5/7 differential suites (diff, async, proc-macro)"
bash scripts/check.sh tests/diff
bash scripts/check.sh tests/async
bash scripts/check.sh tests/proc-macro

echo "[gate:$LANE_NAME] 6/7 probes & fuzz (0..40)"
bash scripts/probes.sh
bash scripts/fuzz.sh 0 40

echo "[gate:$LANE_NAME] 7/7 release build (-D warnings)"
RUSTFLAGS="-D warnings" cargo build --release --bin haste --quiet

t1=$(date +%s)
mkdir -p "$(dirname "$LOG_FILE")"
printf '{"ts":"%s","host":"%s","os":"%s","lane":"%s","status":"PASS","elapsed_s":%d}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "$OS" "$LANE_NAME" "$((t1 - t0))" >> "$LOG_FILE"
echo "[gate:$LANE_NAME] ALL GATES PASS in $((t1 - t0))s (logged to $LOG_FILE)"
