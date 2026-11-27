#!/bin/bash
set -u
cd "$(dirname "$0")/.."
source scripts/cargo.sh
build_haste </dev/null >/dev/null 2>&1 || { echo "haste does not build"; exit 2; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
with_timeout() {
    perl -e 'alarm shift; exec @ARGV' 60 "$@"
}
write_scratch_crate() {
    local dir=$1 pkg=$2 body=$3
    mkdir -p "$dir/src"
    printf '[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2021"\n' "$pkg" >"$dir/Cargo.toml"
    printf 'version = 4\n\n[[package]]\nname = "%s"\nversion = "0.1.0"\n' "$pkg" >"$dir/Cargo.lock"
    printf '%s\n' "$body" >"$dir/src/main.rs"
}
check_build_target() {
    local build_target=$1 out_msg=$2 build_out build_status
    build_out=$(run_haste build "$build_target" </dev/null 2>&1); build_status=$?
    if [ "$build_status" -ne 0 ] || ! printf '%s\n' "$build_out" | grep -Eq '^[1-9][0-9]* functions reached, 0 failed'; then
        echo "build $build_target DIFF: status $build_status: $(printf '%s\n' "$build_out" | tail -2 | cut -c1-160)" >"$out_msg"
        return 1
    fi
}
check_accept_ok() {
    local out_msg=$1 accept_out accept_b=0 accept_c=0 p_b p_c
    write_scratch_crate "$tmp/accept_ok" "accept_ok" 'fn main() {}'
    run_haste build "$tmp/accept_ok" </dev/null >"$tmp/accept_ok.out" 2>&1 &
    p_b=$!
    run_cargo check --quiet --locked --manifest-path "$tmp/accept_ok/Cargo.toml" </dev/null >/dev/null 2>&1 &
    p_c=$!
    wait "$p_b" || accept_b=$?
    wait "$p_c" || accept_c=$?
    accept_out=$(cat "$tmp/accept_ok.out")
    if [ "$accept_c" -ne 0 ] || [ "$accept_b" -ne 0 ] || ! printf '%s\n' "$accept_out" | grep -Eq '^[1-9][0-9]* functions reached, 0 failed'; then
        echo "accept_ok DIFF: cargo=$accept_c build=$accept_b: $(printf '%s\n' "$accept_out" | tail -2 | cut -c1-160)" >"$out_msg"
        return 1
    fi
}
reject_diag_ok() {
    local status=$1 err_file=$2
    [ "$status" -ne 0 ] && [ "$status" -lt 128 ] || return 1
    if grep -Eq 'panicked at|SIGSEGV|SIGABRT|SIGBUS' "$err_file"; then
        return 1
    fi
    grep -qi 'error' "$err_file"
}
check_reject() {
    local label=$1 body=$2 out_msg=$3 dir="$tmp/reject_$1" pkg="reject_$1" c_status=0 h_status=0 p_c p_h
    write_scratch_crate "$dir" "$pkg" "$body"
    run_cargo check --quiet --locked --manifest-path "$dir/Cargo.toml" </dev/null >/dev/null 2>&1 &
    p_c=$!
    (
        local b_status=0 r_status=0
        run_haste build "$dir" </dev/null >/dev/null 2>"$dir/build.err" || b_status=$?
        run_haste run "$dir" </dev/null >/dev/null 2>"$dir/run.err" || r_status=$?
        printf '%s %s\n' "$b_status" "$r_status" >"$dir/haste.status"
        reject_diag_ok "$b_status" "$dir/build.err" && reject_diag_ok "$r_status" "$dir/run.err"
    ) &
    p_h=$!
    wait "$p_c" || c_status=$?
    wait "$p_h" || h_status=$?
    if [ "$c_status" -eq 0 ] || [ "$h_status" -ne 0 ]; then
        local hr b_tail r_tail
        hr=$(cat "$dir/haste.status" 2>/dev/null || echo "? ?")
        b_tail=$(tail -2 "$dir/build.err" 2>/dev/null | tr '\n' ' ' | cut -c1-120)
        r_tail=$(tail -2 "$dir/run.err" 2>/dev/null | tr '\n' ' ' | cut -c1-120)
        echo "reject $label DIFF: cargo=$c_status build/run=($hr) build_err=($b_tail) run_err=($r_tail)" >"$out_msg"
        return 1
    fi
}
check_malformed() {
    local out_msg=$1 mal_status=0
    write_scratch_crate "$tmp/malformed" "malformed" 'macro x'
    run_haste debug items "$tmp/malformed" </dev/null >/dev/null 2>"$tmp/malformed.err" || mal_status=$?
    if [ "$mal_status" -ge 128 ] || grep -Eq 'panicked at|SIGSEGV|SIGABRT|SIGBUS' "$tmp/malformed.err"; then
        echo "malformed macro DIFF: status $mal_status: $(tail -2 "$tmp/malformed.err")" >"$out_msg"
        return 1
    fi
}
check_catch_unwind() {
    local out_msg=$1 dir="$tmp/unwind_ok" c_out h_out a_out c_status=0 h_status=0 a_status=0 p_c p_h p_a
    write_scratch_crate "$dir" "unwind_ok" 'use std::sync::atomic::{AtomicI64, Ordering};
static LOG: AtomicI64 = AtomicI64::new(0);
struct Guard(i64);
impl Drop for Guard {
    fn drop(&mut self) {
        let prev = LOG.load(Ordering::Relaxed);
        LOG.store(prev * 10 + self.0, Ordering::Relaxed);
    }
}
#[inline(never)]
fn nounwind_use(g: Guard) -> i64 {
    let v = g.0;
    drop(g);
    v
}
#[inline(never)]
fn maybe_panic(n: i64) {
    if n == 2 { panic!("loop_boom"); }
}
fn main() {
    let ok = std::panic::catch_unwind(|| 21 + 21).unwrap();
    let msg = std::panic::catch_unwind(|| -> i32 {
        let _kept = Guard(1);
        let moved = Guard(2);
        let _ = nounwind_use(moved);
        let _inner = Guard(3);
        panic!("boom");
    }).unwrap_err();
    let s = msg.downcast_ref::<&str>().copied().unwrap_or("");
    let _ = std::panic::catch_unwind(|| {
        for i in 0..4 {
            let _g = Guard(4 + i);
            maybe_panic(i);
        }
    });
    let t = std::thread::spawn(|| panic!("tboom")).join().is_err();
    println!("{ok} {s} {t} {}", LOG.load(Ordering::Relaxed));
}'
    run_manifest "$dir" </dev/null >"$dir/c.out" 2>/dev/null &
    p_c=$!
    run_haste run "$dir" </dev/null >"$dir/h.out" 2>/dev/null &
    p_h=$!
    (run_haste --release build -o "$dir/unwind_aot" "$dir" </dev/null >/dev/null 2>&1 && "$dir/unwind_aot" </dev/null >"$dir/a.out" 2>/dev/null) &
    p_a=$!
    wait "$p_c" || c_status=$?
    wait "$p_h" || h_status=$?
    wait "$p_a" || a_status=$?
    c_out=$(cat "$dir/c.out" 2>/dev/null || true)
    h_out=$(cat "$dir/h.out" 2>/dev/null || true)
    a_out=$(cat "$dir/a.out" 2>/dev/null || true)
    if [ "$c_status" -ne 0 ] || [ "$h_status" -ne 0 ] || [ "$a_status" -ne 0 ] || [ "$c_out" != "$h_out" ] || [ "$c_out" != "$a_out" ]; then
        echo "catch_unwind DIFF: cargo=($c_status,$c_out) jit=($h_status,$h_out) aot=($a_status,$a_out)" >"$out_msg"
        return 1
    fi
}
check_unit_tests() {
    local out_msg=$1 dir="$tmp/unit_tests" c_status=0 h_status=0 p_c p_h
    mkdir -p "$dir/src"
    printf '[package]\nname = "unit_tests"\nversion = "0.1.0"\nedition = "2021"\n' >"$dir/Cargo.toml"
    printf 'version = 4\n\n[[package]]\nname = "unit_tests"\nversion = "0.1.0"\n' >"$dir/Cargo.lock"
    cat >"$dir/src/lib.rs" <<'EOF'
#[cfg(test)]
mod tests {
    #[test]
    fn unit_ok() { assert_eq!(2 + 2, 4); }
    #[test]
    fn res_ok() -> Result<(), &'static str> { Ok(()) }
    #[test]
    #[should_panic(expected = "boom")]
    fn panic_ok() { panic!("got boom"); }
    #[test]
    #[ignore]
    fn skip_me() { panic!("should not run"); }
}
EOF
    run_cargo test --quiet --locked --manifest-path "$dir/Cargo.toml" </dev/null >/dev/null 2>&1 &
    p_c=$!
    run_haste test "$dir" </dev/null >"$dir/haste.out" 2>&1 &
    p_h=$!
    wait "$p_c" || c_status=$?
    wait "$p_h" || h_status=$?
    if [ "$c_status" -ne 0 ] || [ "$h_status" -ne 0 ] || ! grep -q 'test result: ok. 3 passed; 0 failed; 1 ignored;' "$dir/haste.out"; then
        echo "unit_tests DIFF: cargo=$c_status haste=$h_status: $(tail -3 "$dir/haste.out")" >"$out_msg"
        return 1
    fi
}
probe_one() {
    local dir=$1 out=$2 name expected expected_status=0 found found_status=0 aot_out aot_build_status=0 aot_status=0 p_exp p_jit p_aot=
    name=$(basename "$dir")
    run_manifest "$dir" </dev/null >"$out.exp" 2>/dev/null &
    p_exp=$!
    if [ -n "${HASTE_BIN:-}" ]; then
        with_timeout "$HASTE_BIN" run "$dir" </dev/null >"$out.jit" 2>/dev/null &
    else
        run_haste run "$dir" </dev/null >"$out.jit" 2>/dev/null &
    fi
    p_jit=$!
    if [ "$name" != "so" ]; then
        (
            if ! run_haste --release build -o "$tmp/aot_$name" "$dir" </dev/null >/dev/null 2>"$tmp/aot_$name.err"; then
                exit 2
            fi
            with_timeout "$tmp/aot_$name" </dev/null >"$out.aot" 2>/dev/null || exit 3
        ) &
        p_aot=$!
    fi
    wait "$p_jit" || found_status=$?
    found=$(cat "$out.jit" 2>/dev/null || true)
    if [ "$name" = "so" ]; then
        wait "$p_exp" || expected_status=$?
        expected=$(cat "$out.exp")
        if [ "$expected" = "ready" ] && [ "$found" = "ready" ] && [ "$expected_status" -ne 0 ] && [ "$found_status" -ge 128 ]; then
            echo "$name same" >"$out"
            return 0
        fi
        echo "$name DIFF: status $found_status (expected=$found), cargo $expected_status ($expected)" >"$out"
        return 1
    fi
    wait "$p_aot" || aot_build_status=$?
    wait "$p_exp" || expected_status=$?
    expected=$(cat "$out.exp")
    if [ "$found_status" -ne 0 ]; then
        echo "$name DIFF: status $found_status, cargo $expected_status: $(printf '%s\n' "$found" | tail -2 | cut -c1-160)" >"$out"
        return 1
    fi
    if [ "$aot_build_status" -eq 2 ]; then
        echo "$name AOT BUILD DIFF: $(tail -2 "$tmp/aot_$name.err" | cut -c1-160)" >"$out"
        return 1
    fi
    if [ "$expected_status" -ne 0 ] || [ "$expected" != "$found" ]; then
        echo "$name DIFF: status $found_status, cargo $expected_status: $(printf '%s\n' "$found" | tail -2 | cut -c1-160)" >"$out"
        return 1
    fi
    if [ "$aot_build_status" -ne 0 ]; then
        aot_status=$aot_build_status
        echo "$name AOT RUN DIFF: status $aot_status" >"$out"
        return 1
    fi
    aot_out=$(cat "$out.aot" 2>/dev/null || true)
    if [ "$aot_out" != "$expected" ]; then
        echo "$name AOT RUN DIFF: status 0: $(printf '%s\n' "$aot_out" | tail -2 | cut -c1-160)" >"$out"
        return 1
    fi
    echo "$name same" >"$out"
}
dirs=()
while IFS= read -r d; do
    dirs+=("$d")
done < <(
    for d in tests/probes/*/; do
        lock_sz=$(wc -c <"${d}Cargo.lock" 2>/dev/null | tr -d ' ' || echo 0)
        src_sz=$(wc -c <"${d}src/main.rs" 2>/dev/null | tr -d ' ' || echo 0)
        printf '%012d %012d %s\n' "$lock_sz" "$src_sz" "$d"
    done | sort -r | awk '{print $3}'
)
probe_total=${#dirs[@]}
reject_labels=(
    unsatisfied_bound unconstrained_var use_after_move loop_move assign_immut
    mut_ref_immut conflict_mut_imm mutate_borrowed return_local_ref
    return_byval_deref invalid_arc_unsize
)
reject_bodies=(
    'trait Marker {} fn needs<T: Marker>(_: T) {} fn main() { needs(1u8); }'
    'fn main() { let _v = Vec::new(); }'
    'fn main() { let s = String::from("hi"); drop(s); println!("{s}"); }'
    'fn main() { let s = String::from("hi"); for _ in 0..2 { drop(s); } }'
    'fn main() { let x = 1; x = 2; println!("{x}"); }'
    'fn main() { let x = 1; let r = &mut x; *r = 2; }'
    'fn main() { let mut x = 1; let r1 = &x; let r2 = &mut x; *r2 = 2; println!("{r1}"); }'
    'fn main() { let mut x = 1; let r = &x; x = 2; println!("{r}"); }'
    'fn bad() -> &'\''static i32 { let x = 1; &x } fn main() { let _ = bad(); }'
    'fn bad<'\''a>(p: (String, &'\''a str)) -> &'\''a str { &*p.0 } fn main() { let _ = bad((String::new(), "")); }'
    'use std::sync::Arc; fn main() { let a = [1, 2, 3]; let x: Arc<&[i32; 3]> = Arc::new(&a); let _y: Arc<&[i32]> = x; }'
)
build_targets=(examples/hello tests/diff tests/probes/cl)
# Unified queue ordered heaviest-first so slow probes and pre-checks share one bounded worker pool.
task_kinds=()
task_args=()
for ((i = 0; i < probe_total; i++)); do
    task_kinds+=(probe); task_args+=("$i")
done
task_kinds+=(catch_unwind unit_tests accept_ok malformed)
task_args+=(0 0 0 0)
for ((i = 0; i < ${#build_targets[@]}; i++)); do
    task_kinds+=(build_target); task_args+=("$i")
done
for ((i = 0; i < ${#reject_labels[@]}; i++)); do
    task_kinds+=(reject); task_args+=("$i")
done
task_total=${#task_kinds[@]}
run_task() {
    local idx=$1 kind=${task_kinds[$1]} arg=${task_args[$1]} msg="$tmp/pre.$1.msg"
    case "$kind" in
        probe) probe_one "${dirs[$arg]}" "$tmp/out.$arg" ;;
        catch_unwind) check_catch_unwind "$msg" ;;
        unit_tests) check_unit_tests "$msg" ;;
        accept_ok) check_accept_ok "$msg" ;;
        malformed) check_malformed "$msg" ;;
        build_target) check_build_target "${build_targets[$arg]}" "$msg" ;;
        reject) check_reject "${reject_labels[$arg]}" "${reject_bodies[$arg]}" "$msg" ;;
    esac
}
jobs=$(cpu_count)
[ "$jobs" -gt "$task_total" ] && jobs=$task_total
[ "$jobs" -lt 1 ] && jobs=1
for ((w = 0; w < jobs; w++)); do
    (
        for ((i = 0; i < task_total; i++)); do
            mkdir "$tmp/claim.$i" 2>/dev/null || continue
            run_task "$i" || : >"$tmp/failed"
        done
    ) &
done
wait
for ((i = 0; i < task_total; i++)); do
    [ -s "$tmp/pre.$i.msg" ] && cat "$tmp/pre.$i.msg"
done
for ((i = 0; i < probe_total; i++)); do
    cat "$tmp/out.$i"
done | sort
[ ! -e "$tmp/failed" ]
