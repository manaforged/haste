command=${CARGO:-$(cat "${XDG_CONFIG_HOME:-$HOME/.config}/haste/cargo" 2>/dev/null)}
cargo_bin=cargo
if [[ $command == cargo || ( $command != *[[:space:]]* && $command == */cargo && -x $command ) ]]; then
    cargo_bin=$command
    command=
fi
cpu_count() {
    nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4
}
run_cargo() {
    if [ -n "$command" ]; then $command "$@"; return; fi
    local vars=()
    while [[ ${1:-} == [A-Z_]*=* ]]; do vars+=("$1"); shift; done
    env ${vars[@]+"${vars[@]}"} "$cargo_bin" "$@"
}
run_locked() {
    local lock_file=$1; shift
    if [ -f "$lock_file" ]; then
        run_cargo "$1" --quiet --locked "${@:2}"
    else
        run_cargo "$1" --quiet "${@:2}"
    fi
}
run_manifest() {
    run_locked "$1/Cargo.lock" run --manifest-path "$1/Cargo.toml"
}
build_haste() {
    if [ -n "${HASTE_BIN:-}" ]; then [ -x "$HASTE_BIN" ]; return; fi
    local bin="${CARGO_TARGET_DIR:-target}/release/haste" prev=
    [ -z "$command" ] && [ -x "$bin" ] && prev="$bin.prev.$$" && { cp -p "$bin" "$prev" 2>/dev/null || prev=; }
    if ! run_locked Cargo.lock build --release -p haste; then
        [ -n "$prev" ] && rm -f "$prev"
        return 1
    fi
    if [ -n "$prev" ]; then
        if cmp -s "$bin" "$prev"; then mv -f "$prev" "$bin"; else rm -f "$prev"; fi
    fi
    if [ -x "$bin" ]; then export HASTE_BIN="$bin"; fi
}
run_haste() {
    if [ -n "${HASTE_BIN:-}" ]; then
        "$HASTE_BIN" "$@"
    elif [ -z "$command" ]; then
        "${CARGO_TARGET_DIR:-target}/release/haste" "$@"
    else
        run_locked Cargo.lock run --release -p haste -- "$@"
    fi
}
