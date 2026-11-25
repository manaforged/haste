#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export HASTE_BIN="${HASTE_BIN:-${HASTE:-}}"
set +e
source scripts/cargo.sh
set -e

usage() {
    cat >&2 <<'EOF'
Usage: scripts/rustc-ui.sh [TESTS_UI_DIR] [--borrowck | --run-pass | --all] [--limit N] [--aot] [-j N]

Test haste against official rust-lang/rust tests/ui/ test files.
If TESTS_UI_DIR is omitted, uses $RUSTC_TESTS_DIR.

To fetch tests/ui/ sparsely without cloning the full rust-lang/rust history:
  git clone --depth=1 --filter=blob:none --sparse https://github.com/rust-lang/rust.git
  cd rust && git sparse-checkout set tests/ui

Options:
  --borrowck    Test negative borrow-checker tests (tests/ui/borrowck/)
  --run-pass    Test runnable pass tests
  --all         Run both --borrowck and --run-pass (default)
  --limit N     Test at most N candidate files per suite
  --aot         Also verify AOT compilation & execution (haste --release build -o) for --run-pass
  -j, --jobs N  Number of parallel workers (default: CPU count)
  -h, --help    Show this help
EOF
}

tests_dir="${RUSTC_TESTS_DIR:-}"
mode="all"
limit=0
run_aot=0
jobs=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --borrowck)
            mode="borrowck"
            shift
            ;;
        --run-pass)
            mode="run-pass"
            shift
            ;;
        --all)
            mode="all"
            shift
            ;;
        --aot)
            run_aot=1
            shift
            ;;
        --limit)
            [ "$#" -ge 2 ] || { echo "error: --limit requires a value" >&2; exit 2; }
            limit="$2"
            shift 2
            ;;
        --limit=*)
            limit="${1#*=}"
            shift
            ;;
        -j|--jobs)
            [ "$#" -ge 2 ] || { echo "error: $1 requires a value" >&2; exit 2; }
            jobs="$2"
            shift 2
            ;;
        --jobs=*)
            jobs="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "error: unknown option: $1" >&2
            usage
            exit 2
            ;;
        *)
            tests_dir="$1"
            shift
            ;;
    esac
done

case "$limit" in
    ''|*[!0-9]*)
        echo "error: --limit must be a non-negative integer: $limit" >&2
        exit 2
        ;;
esac

case "$jobs" in
    ''|*[!0-9]*)
        echo "error: --jobs must be a non-negative integer: $jobs" >&2
        exit 2
        ;;
esac

if [ -z "$tests_dir" ] || [ ! -d "$tests_dir" ]; then
    echo "error: valid tests/ui directory not provided (got: '${tests_dir:-<empty>}')" >&2
    usage
    exit 2
fi

if ! command -v rustc >/dev/null 2>&1; then
    echo "error: rustc not found in PATH" >&2
    exit 2
fi

tests_dir=$(cd "$tests_dir" && pwd)

if [ "$jobs" -eq 0 ]; then
    jobs=$(cpu_count)
fi
[ "$jobs" -lt 1 ] && jobs=1

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

build_haste </dev/null >"$tmp/build.err" 2>&1 || {
    cat "$tmp/build.err" >&2
    echo "error: haste did not build" >&2
    exit 2
}
HASTE_BIN="${HASTE_BIN:-${CARGO_TARGET_DIR:-target}/release/haste}"

with_timeout() {
    local secs=$1
    shift
    perl -e 'alarm shift; exec @ARGV or die "exec failed: $!\n"' "$secs" "$@"
}

should_skip_common() {
    local file=$1
    if grep -Eq '#!\[feature\(|//@ (aux-build:|compile-flags:|edition:2015|only-|ignore-|revisions:)' "$file"; then
        return 0
    fi
    if grep -E '^[[:space:]]*(pub[[:space:]]+)?extern[[:space:]]+crate[[:space:]]+' "$file" | grep -Evq 'extern[[:space:]]+crate[[:space:]]+(std|core|alloc)[[:space:]]*;'; then
        return 0
    fi
    return 1
}

write_crate() {
    local dir=$1 pkg=$2 src_file=$3 crate_type=$4
    mkdir -p "$dir/src"
    printf '[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2021"\n' "$pkg" >"$dir/Cargo.toml"
    printf 'version = 4\n\n[[package]]\nname = "%s"\nversion = "0.1.0"\n' "$pkg" >"$dir/Cargo.lock"
    if [ "$crate_type" = "bin" ]; then
        cp "$src_file" "$dir/src/main.rs"
    else
        cp "$src_file" "$dir/src/lib.rs"
    fi
}

is_crash() {
    local status=$1 log_file=$2
    if [ "$status" -eq 0 ]; then
        return 1
    fi
    if [ "$status" -ge 128 ]; then
        return 0
    fi
    if [ -s "$log_file" ] && grep -Eq 'panicked at|fatal runtime error|SIGSEGV|Segmentation fault' "$log_file"; then
        return 0
    fi
    return 1
}

rel_path() {
    local file=$1
    printf '%s\n' "${file#"$tests_dir"/}"
}

stdout_matches() {
    local actual=$1 expected=$2 work=$3
    if cmp -s "$actual" "$expected"; then
        return 0
    fi
    perl -e '
        my ($act, $exp, $work, $haste) = @ARGV;
        (my $haste_base = $haste) =~ s{.*/}{};
        (my $haste_nodot = $haste) =~ s{^\./}{};
        my @bins = grep { length } ("$work/rustc_bin", "$work/aot_bin", "$work/haste_bin", $haste, "./$haste_nodot", $haste_nodot, $haste_base);
        my $re = join("|", map { quotemeta } sort { length($b) <=> length($a) } @bins);
        my $norm = sub {
            local $/;
            open my $fh, "<", $_[0] or return "";
            my $s = <$fh>;
            $s =~ s/$re/<BIN>/g if length $re;
            return $s;
        };
        exit($norm->($act) eq $norm->($exp) ? 0 : 1);
    ' "$actual" "$expected" "$work" "$HASTE_BIN"
}

run_borrowck_one() {
    local file=$1 work=$2 res_file=$3
    local rel crate_type="lib" r_status=0 h_status=0
    rel=$(rel_path "$file")
    mkdir -p "$work/rustc_out"

    if grep -Eq '^[[:space:]]*(pub[[:space:]]+)?(async[[:space:]]+)?(unsafe[[:space:]]+)?fn[[:space:]]+main[[:space:]]*\(' "$file"; then
        crate_type="bin"
    fi

    with_timeout 30 rustc --edition=2021 --crate-type="$crate_type" --out-dir "$work/rustc_out" "$file" </dev/null >/dev/null 2>"$work/rustc.err" || r_status=$?
    if [ "$r_status" -eq 0 ]; then
        printf 'SKIP\t%s\trustc accepted under edition 2021\n' "$rel" >"$res_file"
        return 0
    fi

    write_crate "$work/crate" "ui_test" "$file" "$crate_type"
    with_timeout 30 "$HASTE_BIN" build "$work/crate" </dev/null >"$work/haste.out" 2>"$work/haste.err" || h_status=$?
    cat "$work/haste.out" "$work/haste.err" >"$work/haste.combined"

    if is_crash "$h_status" "$work/haste.combined"; then
        printf 'CRASH\t%s\texit %s\n' "$rel" "$h_status" >"$res_file"
    elif [ "$h_status" -ne 0 ]; then
        if grep -q 'error\[borrow\]:' "$work/haste.combined"; then
            printf 'PASS\t%s\n' "$rel" >"$res_file"
        else
            local reason
            reason=$(grep -Em1 '^(error|Error)' "$work/haste.combined" || true)
            reason=$(printf '%s' "$reason" | tr '\t' ' ' | cut -c1-120)
            printf 'OTHER\t%s\t%s\n' "$rel" "${reason:-non-borrow error}" >"$res_file"
        fi
    else
        printf 'FAIL\t%s\taccepted\n' "$rel" >"$res_file"
    fi
}

run_pass_one() {
    local file=$1 work=$2 res_file=$3
    local rel r_build=0 r_status=0 h_status=0 aot_b=0 aot_r=0 build_only=0
    rel=$(rel_path "$file")
    mkdir -p "$work/rustc_out"

    if grep -Eq '//@ (check-pass|build-pass)' "$file" && ! grep -Eq '//@ run-pass' "$file"; then
        build_only=1
    fi

    with_timeout 30 rustc --edition=2021 --out-dir "$work/rustc_out" -o "$work/rustc_bin" "$file" </dev/null >/dev/null 2>"$work/rustc.err" || r_build=$?
    if [ "$r_build" -ne 0 ]; then
        printf 'SKIP\t%s\trustc failed under edition 2021\n' "$rel" >"$res_file"
        return 0
    fi

    write_crate "$work/crate" "ui_test" "$file" "bin"
    if [ "$build_only" -eq 1 ]; then
        with_timeout 30 "$HASTE_BIN" build "$work/crate" </dev/null >"$work/haste.out" 2>"$work/haste.err" || h_status=$?
        if is_crash "$h_status" "$work/haste.err"; then
            printf 'CRASH\t%s\tbuild crash (exit %s)\n' "$rel" "$h_status" >"$res_file"
            return 0
        fi
        if [ "$h_status" -ne 0 ]; then
            printf 'FAIL\t%s\tbuild failed (exit %s)\n' "$rel" "$h_status" >"$res_file"
            return 0
        fi
    else
        with_timeout 30 "$work/rustc_bin" </dev/null >"$work/rustc.out" 2>"$work/rustc.run_err" || r_status=$?
        if [ "$r_status" -ne 0 ] && grep -Eq '//@ run-pass' "$file"; then
            printf 'SKIP\t%s\trustc run failed on //@ run-pass (exit %s)\n' "$rel" "$r_status" >"$res_file"
            return 0
        fi

        with_timeout 30 "$HASTE_BIN" run "$work/crate" </dev/null >"$work/haste.out" 2>"$work/haste.err" || h_status=$?
        if is_crash "$h_status" "$work/haste.err"; then
            printf 'CRASH\t%s\tjit crash (exit %s)\n' "$rel" "$h_status" >"$res_file"
            return 0
        fi
        if [ "$h_status" -ne "$r_status" ]; then
            printf 'FAIL\t%s\tjit exit %s vs rustc %s\n' "$rel" "$h_status" "$r_status" >"$res_file"
            return 0
        fi
        if ! stdout_matches "$work/haste.out" "$work/rustc.out" "$work"; then
            local r2=0
            with_timeout 30 "$work/rustc_bin" </dev/null >"$work/rustc2.out" 2>/dev/null || r2=$?
            if [ "$r2" -ne 0 ] || cmp -s "$work/rustc.out" "$work/rustc2.out"; then
                printf 'FAIL\t%s\tjit stdout mismatch\n' "$rel" >"$res_file"
                return 0
            fi
        fi
    fi

    if [ "$run_aot" -eq 1 ]; then
        with_timeout 30 "$HASTE_BIN" --release build -o "$work/aot_bin" "$work/crate" </dev/null >/dev/null 2>"$work/aot.err" || aot_b=$?
        if is_crash "$aot_b" "$work/aot.err"; then
            printf 'CRASH\t%s\taot build crash (exit %s)\n' "$rel" "$aot_b" >"$res_file"
            return 0
        fi
        if [ "$aot_b" -ne 0 ]; then
            printf 'FAIL\t%s\taot build failed (exit %s)\n' "$rel" "$aot_b" >"$res_file"
            return 0
        fi
        if [ "$build_only" -eq 0 ]; then
            with_timeout 30 "$work/aot_bin" </dev/null >"$work/aot.out" 2>"$work/aot.run_err" || aot_r=$?
            if [ "$aot_r" -ne "$r_status" ]; then
                printf 'FAIL\t%s\taot exit %s vs rustc %s\n' "$rel" "$aot_r" "$r_status" >"$res_file"
                return 0
            fi
            if ! stdout_matches "$work/aot.out" "$work/rustc.out" "$work"; then
                local r2=0
                with_timeout 30 "$work/rustc_bin" </dev/null >"$work/rustc2.out" 2>/dev/null || r2=$?
                if [ "$r2" -ne 0 ] || cmp -s "$work/rustc.out" "$work/rustc2.out"; then
                    printf 'FAIL\t%s\taot stdout mismatch\n' "$rel" >"$res_file"
                    return 0
                fi
            fi
        fi
    fi

    printf 'PASS\t%s\n' "$rel" >"$res_file"
}

run_suite() {
    local suite=$1
    local search_root="$tests_dir"
    local candidates=()
    local static_skipped=0
    local file

    if [ "$suite" = "borrowck" ] && [ -d "$tests_dir/borrowck" ]; then
        search_root="$tests_dir/borrowck"
    fi

    while IFS= read -r file; do
        if should_skip_common "$file"; then
            static_skipped=$((static_skipped + 1))
            continue
        fi
        if [ "$suite" = "borrowck" ]; then
            if ! grep -Eq '//~[^[:space:]]*[[:space:]]+ERROR' "$file"; then
                static_skipped=$((static_skipped + 1))
                continue
            fi
        else
            if grep -Eq '//~[^[:space:]]*[[:space:]]+ERROR|//@ (check-fail|build-fail|run-fail|compile-fail)' "$file"; then
                static_skipped=$((static_skipped + 1))
                continue
            fi
            if ! grep -Eq '//@ run-pass|(^|[^[:alnum:]_])fn[[:space:]]+main[[:space:]]*\(' "$file"; then
                static_skipped=$((static_skipped + 1))
                continue
            fi
        fi
        candidates+=("$file")
        if [ "$limit" -gt 0 ] && [ "${#candidates[@]}" -ge "$limit" ]; then
            break
        fi
    done < <(find "$search_root" -type f -name '*.rs' ! -path '*/auxiliary/*' | sort)

    local suite_tmp="$tmp/$suite"
    mkdir -p "$suite_tmp/work" "$suite_tmp/res"

    local total="${#candidates[@]}"
    local worker_jobs="$jobs"
    [ "$worker_jobs" -gt "$total" ] && worker_jobs="$total"
    [ "$worker_jobs" -lt 1 ] && worker_jobs=1

    if [ "$total" -gt 0 ]; then
        local w i
        for ((w = 0; w < worker_jobs; w++)); do
            (
                for ((i = 0; i < total; i++)); do
                    mkdir "$suite_tmp/claim.$i" 2>/dev/null || continue
                    if [ "$suite" = "borrowck" ]; then
                        run_borrowck_one "${candidates[$i]}" "$suite_tmp/work/$i" "$suite_tmp/res/$i.res"
                    else
                        run_pass_one "${candidates[$i]}" "$suite_tmp/work/$i" "$suite_tmp/res/$i.res"
                    fi
                done
            ) &
        done
        wait
    fi

    local tested=0 skipped="$static_skipped" passed=0 other_err=0 failed=0 crashed=0
    local kind name reason
    for ((i = 0; i < total; i++)); do
        IFS=$'\t' read -r kind name reason <"$suite_tmp/res/$i.res"
        case "$kind" in
            SKIP)
                skipped=$((skipped + 1))
                ;;
            PASS)
                tested=$((tested + 1))
                passed=$((passed + 1))
                ;;
            OTHER)
                tested=$((tested + 1))
                other_err=$((other_err + 1))
                printf '%s\t%s\tother-err: %s\n' "$suite" "$name" "$reason" >>"$tmp/failures.tsv"
                ;;
            FAIL)
                tested=$((tested + 1))
                failed=$((failed + 1))
                printf '%s\t%s\t%s\n' "$suite" "$name" "$reason" >>"$tmp/failures.tsv"
                ;;
            CRASH)
                tested=$((tested + 1))
                crashed=$((crashed + 1))
                printf '%s\t%s\t%s (crash)\n' "$suite" "$name" "$reason" >>"$tmp/failures.tsv"
                ;;
        esac
    done

    local rate
    rate=$(awk -v p="$passed" -v t="$tested" 'BEGIN { if (t > 0) printf "%.1f%%", (p * 100.0) / t; else printf "0.0%" }')
    printf '| %s | %d | %d | %d | %d | %d | %d | %s |\n' "$suite" "$tested" "$skipped" "$passed" "$other_err" "$failed" "$crashed" "$rate" >>"$tmp/summary.rows"

    total_tested=$((total_tested + tested))
    total_skipped=$((total_skipped + skipped))
    total_passed=$((total_passed + passed))
    total_other=$((total_other + other_err))
    total_failed=$((total_failed + failed))
    total_crashed=$((total_crashed + crashed))
}

: >"$tmp/failures.tsv"
: >"$tmp/summary.rows"
total_tested=0
total_skipped=0
total_passed=0
total_other=0
total_failed=0
total_crashed=0

case "$mode" in
    borrowck)
        run_suite "borrowck"
        ;;
    run-pass)
        run_suite "run-pass"
        ;;
    all)
        run_suite "borrowck"
        run_suite "run-pass"
        ;;
esac

printf '| Suite | Tested | Skipped | Passed | OtherErr | Failed | Crashed | Pass Rate |\n'
printf '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
cat "$tmp/summary.rows"
if [ "$mode" = "all" ]; then
    total_rate=$(awk -v p="$total_passed" -v t="$total_tested" 'BEGIN { if (t > 0) printf "%.1f%%", (p * 100.0) / t; else printf "0.0%" }')
    printf '| **total** | **%d** | **%d** | **%d** | **%d** | **%d** | **%d** | **%s** |\n' "$total_tested" "$total_skipped" "$total_passed" "$total_other" "$total_failed" "$total_crashed" "$total_rate"
fi

if [ -n "${HASTE_UI_TSV:-}" ] && [ -f "$tmp/failures.tsv" ]; then
    cp "$tmp/failures.tsv" "$HASTE_UI_TSV"
fi

if [ -s "$tmp/failures.tsv" ]; then
    printf '\n### Top 10 Failing Tests\n\n'
    head -n 10 "$tmp/failures.tsv" | while IFS=$'\t' read -r f_suite f_name f_reason; do
        printf -- '- `[%s]` `%s` — %s\n' "$f_suite" "$f_name" "$f_reason"
    done
fi

[ "$((total_other + total_failed + total_crashed))" -eq 0 ]
