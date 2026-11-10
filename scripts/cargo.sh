# Sourced by the gate scripts: `run_cargo [VAR=value ...] ARGS` runs cargo, or the command named in $CARGO or in the user's haste/cargo config file, which takes the same leading variables.
command=${CARGO:-$(cat "${XDG_CONFIG_HOME:-$HOME/.config}/haste/cargo" 2>/dev/null)}
run_cargo() {
    if [ -n "$command" ]; then
        $command "$@"
        return
    fi
    local vars=()
    while [[ ${1:-} == [A-Z_]*=* ]]; do
        vars+=("$1")
        shift
    done
    env "${vars[@]}" cargo "$@"
}
