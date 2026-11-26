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
  --run-pass    Test pass tests (run-pass, check-pass, build-pass)
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

discover_suite() {
    local suite=$1 search_root=$2 limit=$3 suite_tmp=$4
    find "$search_root" -type f -name '*.rs' ! -path '*/auxiliary/*' | sort | perl -e '
        use strict;
        use warnings;
        my ($suite, $limit, $cand_path, $counts_path) = @ARGV;
        open my $cand_fh, ">", $cand_path or die "open $cand_path: $!";
        my ($total, $s_feat, $s_aux, $s_flags, $s_rev, $s_only, $s_other, $s_limit, $cands) = (0) x 9;
        while (my $file = <STDIN>) {
            chomp $file;
            next unless length $file;
            $total++;
            open my $fh, "<", $file or do {
                $s_other++;
                next;
            };
            my ($has_feat, $has_aux, $has_flags, $has_rev, $has_only, $has_ed15, $has_extern) = (0) x 7;
            my ($has_err_anno, $has_fail_dir, $has_run_or_main) = (0) x 3;
            my ($lineno, $prev_target) = (0, undef);
            my (@expected_lines, %seen_line);
            while (my $line = <$fh>) {
                $lineno++;
                $has_feat = 1 if $line =~ /#!\[feature\(/;
                $has_aux = 1 if $line =~ m{//\@\s*(?:aux-build|aux-crate|aux-bin):};
                $has_flags = 1 if $line =~ m{//\@\s*compile-flags:};
                $has_rev = 1 if $line =~ m{//\@\s*revisions:};
                $has_only = 1 if $line =~ m{//\@\s*(?:only-|ignore-)};
                $has_ed15 = 1 if $line =~ m{//\@\s*edition:2015\b};
                if ($line =~ /^\s*(?:pub\s+)?extern\s+crate\s+/ && $line !~ /extern\s+crate\s+(?:std|core|alloc)\s*;/) {
                    $has_extern = 1;
                }
                $has_err_anno = 1 if $line =~ m{//~(?:\[[^\]]*\])?(?:\^+|[vV]+|\||\?)?\s*ERROR\b};
                $has_fail_dir = 1 if $line =~ m{//\@\s*(?:check-fail|build-fail|run-fail|compile-fail)\b};
                if ($line =~ m{//\@\s*run-pass\b} || $line =~ /(?:^|[^A-Za-z0-9_])fn\s+main\s*\(/) {
                    $has_run_or_main = 1;
                }
                if ($suite eq "borrowck") {
                    if ($line =~ m{//~(?:\[[^\]]*\])?\?}) {
                        $prev_target = undef;
                    } elsif ($line =~ m{//~(?:\[[^\]]*\])?(\^+|[vV]+|\|)?\s*(ERROR|NOTE|HELP|WARN(?:ING)?)\b}) {
                        my ($marker, $kind) = (defined($1) ? $1 : "", $2);
                        my $target;
                        if ($marker eq "|") {
                            $target = defined($prev_target) ? $prev_target : $lineno;
                        } elsif ($marker =~ /^\^+$/) {
                            $target = $lineno - length($marker);
                            $target = 1 if $target < 1;
                        } elsif ($marker =~ /^[vV]+$/) {
                            $target = $lineno + length($marker);
                        } else {
                            $target = $lineno;
                        }
                        $prev_target = $target;
                        if ($kind eq "ERROR" && !$seen_line{$target}++) {
                            push @expected_lines, $target;
                        }
                    }
                }
            }
            close $fh;
            if ($has_feat) { $s_feat++; next; }
            if ($has_aux) { $s_aux++; next; }
            if ($has_flags) { $s_flags++; next; }
            if ($has_rev) { $s_rev++; next; }
            if ($has_only) { $s_only++; next; }
            if ($has_ed15 || $has_extern) { $s_other++; next; }
            if ($suite eq "borrowck") {
                if (!$has_err_anno) { $s_other++; next; }
            } else {
                if ($has_err_anno || $has_fail_dir || !$has_run_or_main) { $s_other++; next; }
            }
            if ($limit > 0 && $cands >= $limit) {
                $s_limit++;
                next;
            }
            $cands++;
            print $cand_fh $file, "\t", join(",", @expected_lines), "\n";
        }
        close $cand_fh;
        open my $cnt_fh, ">", $counts_path or die "open $counts_path: $!";
        print $cnt_fh join("\t", $total, $s_feat, $s_aux, $s_flags, $s_rev, $s_only, $s_other, $s_limit, $cands), "\n";
        close $cnt_fh;
    ' "$suite" "$limit" "$suite_tmp/candidates.tsv" "$suite_tmp/counts.tsv"
}

borrowck_line_matches() {
    local expected_csv=$1 combined_file=$2
    if [ -z "$expected_csv" ]; then
        return 1
    fi
    perl -e '
        use strict;
        use warnings;
        my ($expected_csv, $combined_path) = @ARGV;
        my %want = map { $_ => 1 } grep { length } split /,/, $expected_csv;
        open my $fh, "<", $combined_path or exit 1;
        my $in_borrow = 0;
        while (my $line = <$fh>) {
            if ($line =~ /^error\[borrow\]:/) {
                $in_borrow = 1;
            } elsif ($line =~ /^(?:error|warning|note)(?:\[[^\]]*\])?:/) {
                $in_borrow = 0;
            } elsif ($in_borrow && $line =~ m{^\s*-->\s+(.+):(\d+):(\d+)\s*$}) {
                my ($path, $lineno) = ($1, $2);
                if ($path =~ m{(?:^|[\\/])src[\\/](?:main|lib)\.rs$} && $want{$lineno}) {
                    exit 0;
                }
            }
        }
        exit 1;
    ' "$expected_csv" "$combined_file"
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

verify_stdout() {
    local actual=$1 work=$2 label=$3 pass_kind=$4 rel=$5 res_file=$6
    if stdout_matches "$actual" "$work/rustc.out" "$work"; then
        return 0
    fi
    local r2=0
    with_timeout 30 "$work/rustc_bin" </dev/null >"$work/rustc2.out" 2>/dev/null || r2=$?
    if [ "$r2" -ne 0 ] || cmp -s "$work/rustc.out" "$work/rustc2.out"; then
        printf 'FAIL\t%s\t%s\t%s stdout mismatch\n' "$pass_kind" "$rel" "$label" >"$res_file"
        return 1
    fi
    printf 'SKIP\t%s\t%s\tnondeterministic stdout\n' "$pass_kind" "$rel" >"$res_file"
    return 1
}

run_borrowck_one() {
    local file=$1 work=$2 res_file=$3 expected_lines=${4:-}
    local rel crate_type="lib" r_status=0 h_status=0 line_ok=0
    rel=$(rel_path "$file")
    mkdir -p "$work/rustc_out"

    if grep -Eq '^[[:space:]]*(pub[[:space:]]+)?(async[[:space:]]+)?(unsafe[[:space:]]+)?fn[[:space:]]+main[[:space:]]*\(' "$file"; then
        crate_type="bin"
    fi

    with_timeout 30 rustc --edition=2021 --crate-type="$crate_type" --out-dir "$work/rustc_out" "$file" </dev/null >/dev/null 2>"$work/rustc.err" || r_status=$?
    if [ "$r_status" -eq 0 ]; then
        printf 'SKIP\tborrowck\t%s\trustc accepted under edition 2021\n' "$rel" >"$res_file"
        return 0
    fi

    write_crate "$work/crate" "ui_test" "$file" "$crate_type"
    with_timeout 30 "$HASTE_BIN" build "$work/crate" </dev/null >"$work/haste.out" 2>"$work/haste.err" || h_status=$?
    cat "$work/haste.out" "$work/haste.err" >"$work/haste.combined"

    if is_crash "$h_status" "$work/haste.combined"; then
        printf 'CRASH\tborrowck\t%s\texit %s\n' "$rel" "$h_status" >"$res_file"
    elif [ "$h_status" -ne 0 ]; then
        if grep -q 'error\[borrow\]:' "$work/haste.combined"; then
            if borrowck_line_matches "$expected_lines" "$work/haste.combined"; then
                line_ok=1
            fi
            printf 'PASS\tborrowck\t%s\t%d\n' "$rel" "$line_ok" >"$res_file"
        else
            local reason
            reason=$(grep -Em1 '^(error|Error)' "$work/haste.combined" || true)
            reason=$(printf '%s' "$reason" | tr '\t' ' ' | cut -c1-120)
            printf 'OTHER\tborrowck\t%s\t%s\n' "$rel" "${reason:-non-borrow error}" >"$res_file"
        fi
    else
        printf 'FAIL\tborrowck\t%s\taccepted\n' "$rel" >"$res_file"
    fi
}

run_pass_one() {
    local file=$1 work=$2 res_file=$3
    local rel r_build=0 r_status=0 h_status=0 aot_b=0 aot_r=0 build_only=0 pass_kind="run-pass"
    rel=$(rel_path "$file")
    mkdir -p "$work/rustc_out"

    if grep -Eq '//@[[:space:]]*check-pass' "$file" && ! grep -Eq '//@[[:space:]]*run-pass' "$file"; then
        build_only=1
        pass_kind="check-pass"
    elif grep -Eq '//@[[:space:]]*build-pass' "$file" && ! grep -Eq '//@[[:space:]]*run-pass' "$file"; then
        build_only=1
        pass_kind="build-pass"
    fi

    with_timeout 30 rustc --edition=2021 --out-dir "$work/rustc_out" -o "$work/rustc_bin" "$file" </dev/null >/dev/null 2>"$work/rustc.err" || r_build=$?
    if [ "$r_build" -ne 0 ]; then
        printf 'SKIP\t%s\t%s\trustc failed under edition 2021\n' "$pass_kind" "$rel" >"$res_file"
        return 0
    fi

    write_crate "$work/crate" "ui_test" "$file" "bin"
    if [ "$build_only" -eq 1 ]; then
        with_timeout 30 "$HASTE_BIN" build "$work/crate" </dev/null >"$work/haste.out" 2>"$work/haste.err" || h_status=$?
        if is_crash "$h_status" "$work/haste.err"; then
            printf 'CRASH\t%s\t%s\tbuild crash (exit %s)\n' "$pass_kind" "$rel" "$h_status" >"$res_file"
            return 0
        fi
        if [ "$h_status" -ne 0 ]; then
            printf 'FAIL\t%s\t%s\tbuild failed (exit %s)\n' "$pass_kind" "$rel" "$h_status" >"$res_file"
            return 0
        fi
    else
        with_timeout 30 "$work/rustc_bin" </dev/null >"$work/rustc.out" 2>"$work/rustc.run_err" || r_status=$?
        if [ "$r_status" -ne 0 ] && grep -Eq '//@[[:space:]]*run-pass' "$file"; then
            printf 'SKIP\t%s\t%s\trustc run failed on //@ run-pass (exit %s)\n' "$pass_kind" "$rel" "$r_status" >"$res_file"
            return 0
        fi

        with_timeout 30 "$HASTE_BIN" run "$work/crate" </dev/null >"$work/haste.out" 2>"$work/haste.err" || h_status=$?
        if is_crash "$h_status" "$work/haste.err"; then
            printf 'CRASH\t%s\t%s\tjit crash (exit %s)\n' "$pass_kind" "$rel" "$h_status" >"$res_file"
            return 0
        fi
        if [ "$h_status" -ne "$r_status" ]; then
            printf 'FAIL\t%s\t%s\tjit exit %s vs rustc %s\n' "$pass_kind" "$rel" "$h_status" "$r_status" >"$res_file"
            return 0
        fi
        verify_stdout "$work/haste.out" "$work" "jit" "$pass_kind" "$rel" "$res_file" || return 0
    fi

    if [ "$run_aot" -eq 1 ]; then
        with_timeout 30 "$HASTE_BIN" --release build -o "$work/aot_bin" "$work/crate" </dev/null >/dev/null 2>"$work/aot.err" || aot_b=$?
        if is_crash "$aot_b" "$work/aot.err"; then
            printf 'CRASH\t%s\t%s\taot build crash (exit %s)\n' "$pass_kind" "$rel" "$aot_b" >"$res_file"
            return 0
        fi
        if [ "$aot_b" -ne 0 ]; then
            printf 'FAIL\t%s\t%s\taot build failed (exit %s)\n' "$pass_kind" "$rel" "$aot_b" >"$res_file"
            return 0
        fi
        if [ "$build_only" -eq 0 ]; then
            with_timeout 30 "$work/aot_bin" </dev/null >"$work/aot.out" 2>"$work/aot.run_err" || aot_r=$?
            if [ "$aot_r" -ne "$r_status" ]; then
                printf 'FAIL\t%s\t%s\taot exit %s vs rustc %s\n' "$pass_kind" "$rel" "$aot_r" "$r_status" >"$res_file"
                return 0
            fi
            verify_stdout "$work/aot.out" "$work" "aot" "$pass_kind" "$rel" "$res_file" || return 0
        fi
    fi

    printf 'PASS\t%s\t%s\t%s\n' "$pass_kind" "$rel" "$pass_kind" >"$res_file"
}

pct() {
    awk -v p="$1" -v t="$2" 'BEGIN { if (t > 0) printf "%.1f%%", (p * 100.0) / t; else printf "0.0%" }'
}

run_suite() {
    local suite=$1
    local search_root="$tests_dir"
    local candidates=()
    local cand_lines=()
    local file expected_lines

    if [ "$suite" = "borrowck" ] && [ -d "$tests_dir/borrowck" ]; then
        search_root="$tests_dir/borrowck"
    fi

    local suite_tmp="$tmp/$suite"
    mkdir -p "$suite_tmp/work" "$suite_tmp/res"

    discover_suite "$suite" "$search_root" "$limit" "$suite_tmp"

    local discovered_total=0 s_feat=0 s_aux=0 s_flags=0 s_rev=0 s_only=0 s_other=0 s_limit=0 cand_count=0
    IFS=$'\t' read -r discovered_total s_feat s_aux s_flags s_rev s_only s_other s_limit cand_count <"$suite_tmp/counts.tsv"

    while IFS=$'\t' read -r file expected_lines; do
        [ -n "$file" ] || continue
        candidates+=("$file")
        cand_lines+=("${expected_lines:-}")
    done <"$suite_tmp/candidates.tsv"

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
                        run_borrowck_one "${candidates[$i]}" "$suite_tmp/work/$i" "$suite_tmp/res/$i.res" "${cand_lines[$i]:-}"
                    else
                        run_pass_one "${candidates[$i]}" "$suite_tmp/work/$i" "$suite_tmp/res/$i.res"
                    fi
                done
            ) &
        done
        wait
    fi

    local ran=0 passed=0 line_matched=0 located_ran=0 other_err=0 failed=0 crashed=0
    local rp_ran=0 rp_passed=0 rp_failed=0 rp_crashed=0
    local cb_ran=0 cb_passed=0 cb_failed=0 cb_crashed=0
    local kind subkind name reason
    for ((i = 0; i < total; i++)); do
        IFS=$'\t' read -r kind subkind name reason <"$suite_tmp/res/$i.res"
        if [ "$kind" = "SKIP" ]; then
            s_other=$((s_other + 1))
            continue
        fi
        ran=$((ran + 1))
        if [ -n "${cand_lines[$i]:-}" ]; then
            located_ran=$((located_ran + 1))
        fi
        if [ "$suite" != "borrowck" ]; then
            if [ "$subkind" = "run-pass" ]; then
                rp_ran=$((rp_ran + 1))
            else
                cb_ran=$((cb_ran + 1))
            fi
        fi
        case "$kind" in
            PASS)
                passed=$((passed + 1))
                if [ "$suite" = "borrowck" ]; then
                    if [ "${reason:-0}" = "1" ]; then
                        line_matched=$((line_matched + 1))
                    fi
                elif [ "$subkind" = "run-pass" ]; then
                    rp_passed=$((rp_passed + 1))
                else
                    cb_passed=$((cb_passed + 1))
                fi
                ;;
            OTHER)
                other_err=$((other_err + 1))
                printf '%s\t%s\tother-err: %s\n' "$subkind" "$name" "$reason" >>"$tmp/failures.tsv"
                ;;
            FAIL)
                failed=$((failed + 1))
                if [ "$suite" != "borrowck" ]; then
                    if [ "$subkind" = "run-pass" ]; then
                        rp_failed=$((rp_failed + 1))
                    else
                        cb_failed=$((cb_failed + 1))
                    fi
                fi
                printf '%s\t%s\t%s\n' "$subkind" "$name" "$reason" >>"$tmp/failures.tsv"
                ;;
            CRASH)
                crashed=$((crashed + 1))
                if [ "$suite" != "borrowck" ]; then
                    if [ "$subkind" = "run-pass" ]; then
                        rp_crashed=$((rp_crashed + 1))
                    else
                        cb_crashed=$((cb_crashed + 1))
                    fi
                fi
                printf '%s\t%s\t%s (crash)\n' "$subkind" "$name" "$reason" >>"$tmp/failures.tsv"
                ;;
        esac
    done

    local suite_skipped=$((s_feat + s_aux + s_flags + s_rev + s_only + s_other + s_limit))
    local rate
    rate=$(pct "$passed" "$ran")

    local skip_detail="feature=$s_feat aux-build=$s_aux compile-flags=$s_flags revisions=$s_rev only/ignore=$s_only other=$s_other"
    if [ "$s_limit" -gt 0 ]; then
        skip_detail="$skip_detail limit=$s_limit"
    fi
    if [ "$suite" = "borrowck" ]; then
        printf '| %s | %d | %d | %d | %d | %d | %d | %d | %s |\n' \
            "$suite" "$discovered_total" "$suite_skipped" "$ran" "$passed" "$other_err" "$failed" "$crashed" "$rate" >>"$tmp/summary.rows"
        printf -- '- `%s`: total=%d skipped=%d (%s) ran=%d passed=%d other-err=%d failed=%d crashed=%d\n' \
            "$suite" "$discovered_total" "$suite_skipped" "$skip_detail" "$ran" "$passed" "$other_err" "$failed" "$crashed" >>"$tmp/breakdown.lines"
        borrow_ran=$ran
        borrow_diag_match=$passed
        borrow_line_match=$line_matched
        borrow_located=$located_ran
    else
        printf '| pass (all) | %d | %d | %d | %d | %d | %d | %d | %s |\n' \
            "$discovered_total" "$suite_skipped" "$ran" "$passed" "$other_err" "$failed" "$crashed" "$rate" >>"$tmp/summary.rows"
        printf '|   run-pass | - | - | %d | %d | 0 | %d | %d | %s |\n' \
            "$rp_ran" "$rp_passed" "$rp_failed" "$rp_crashed" "$(pct "$rp_passed" "$rp_ran")" >>"$tmp/summary.rows"
        printf '|   check/build-pass | - | - | %d | %d | 0 | %d | %d | %s |\n' \
            "$cb_ran" "$cb_passed" "$cb_failed" "$cb_crashed" "$(pct "$cb_passed" "$cb_ran")" >>"$tmp/summary.rows"
        printf -- '- `pass`: total=%d skipped=%d (%s) ran=%d passed=%d failed=%d crashed=%d (run-pass: ran=%d passed=%d failed=%d crashed=%d; check/build-pass: ran=%d passed=%d failed=%d crashed=%d)\n' \
            "$discovered_total" "$suite_skipped" "$skip_detail" "$ran" "$passed" "$failed" "$crashed" \
            "$rp_ran" "$rp_passed" "$rp_failed" "$rp_crashed" \
            "$cb_ran" "$cb_passed" "$cb_failed" "$cb_crashed" >>"$tmp/breakdown.lines"
        pass_rp_ran=$rp_ran
        pass_rp_passed=$rp_passed
        pass_cb_ran=$cb_ran
        pass_cb_passed=$cb_passed
    fi

    last_discovered_total="$discovered_total"
    total_ran=$((total_ran + ran))
    total_passed=$((total_passed + passed))
    total_other=$((total_other + other_err))
    total_failed=$((total_failed + failed))
    total_crashed=$((total_crashed + crashed))
}

: >"$tmp/failures.tsv"
: >"$tmp/summary.rows"
: >"$tmp/breakdown.lines"
last_discovered_total=0
total_ran=0
total_passed=0
total_other=0
total_failed=0
total_crashed=0
borrow_ran=0
borrow_diag_match=0
borrow_line_match=0
borrow_located=0
pass_rp_ran=0
pass_rp_passed=0
pass_cb_ran=0
pass_cb_passed=0

case "$mode" in
    borrowck)
        run_suite "borrowck"
        ;;
    run-pass)
        run_suite "pass"
        ;;
    all)
        run_suite "borrowck"
        run_suite "pass"
        ;;
esac

printf '| Suite | Total | Skipped | Ran | Passed | OtherErr | Failed | Crashed | Pass Rate |\n'
printf '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
cat "$tmp/summary.rows"
if [ "$mode" = "all" ]; then
    unique_total="$last_discovered_total"
    unique_skipped=$((unique_total - total_ran))
    total_rate=$(pct "$total_passed" "$total_ran")
    printf '| **total** | **%d** | **%d** | **%d** | **%d** | **%d** | **%d** | **%d** | **%s** |\n' \
        "$unique_total" "$unique_skipped" "$total_ran" "$total_passed" "$total_other" "$total_failed" "$total_crashed" "$total_rate"
fi

printf '\n### Suite Breakdown\n\n'
cat "$tmp/breakdown.lines"

if [ "$mode" = "borrowck" ] || [ "$mode" = "all" ]; then
    diag_rate=$(pct "$borrow_diag_match" "$borrow_ran")
    line_rate=$(pct "$borrow_line_match" "$borrow_located")
    borrow_unlocated=$((borrow_ran - borrow_located))
    if [ "$borrow_unlocated" -gt 0 ]; then
        printf '\nBorrowck accuracy: diag-match=%d/%d (%s) line-match=%d/%d (%s) unlocated=%d\n' \
            "$borrow_diag_match" "$borrow_ran" "$diag_rate" \
            "$borrow_line_match" "$borrow_located" "$line_rate" \
            "$borrow_unlocated"
    else
        printf '\nBorrowck accuracy: diag-match=%d/%d (%s) line-match=%d/%d (%s)\n' \
            "$borrow_diag_match" "$borrow_ran" "$diag_rate" \
            "$borrow_line_match" "$borrow_located" "$line_rate"
    fi
fi

if [ "$mode" = "run-pass" ] || [ "$mode" = "all" ]; then
    pass_mode="jit"
    [ "$run_aot" -eq 1 ] && pass_mode="jit+aot"
    rp_rate=$(pct "$pass_rp_passed" "$pass_rp_ran")
    cb_rate=$(pct "$pass_cb_passed" "$pass_cb_ran")
    printf '\nPass accuracy (%s): run-pass=%d/%d (%s) check/build-pass=%d/%d (%s)\n' \
        "$pass_mode" \
        "$pass_rp_passed" "$pass_rp_ran" "$rp_rate" \
        "$pass_cb_passed" "$pass_cb_ran" "$cb_rate"
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
