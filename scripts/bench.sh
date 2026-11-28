#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cargo.sh

suite="${HASTE_BENCH_SUITE:-all}"
bench_long="${HASTE_BENCH_LONG:-0}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        all|compute|build|self) suite="$1"; shift ;;
        --long) bench_long=1; shift ;;
        -h|--help)
            echo "usage: $0 [all|compute|build|self] [--long]"
            exit 0
            ;;
        *)
            echo "usage: $0 [all|compute|build|self] [--long]" >&2
            exit 2
            ;;
    esac
done

build_haste </dev/null >/dev/null 2>&1 || { echo "haste does not build"; exit 2; }
haste_bin="${HASTE_BIN:-${CARGO_TARGET_DIR:-target}/release/haste}"
cache_dir="${CARGO_TARGET_DIR:-target}/bench-cache"

tmp=$(mktemp -d)
cargo_tmp="${CARGO_TARGET_DIR:-target}/bench-tmp-$$"
mkdir -p "$cargo_tmp"
trap 'rm -rf "$tmp" "$cargo_tmp"' EXIT

echo "Host: $(uname -s) $(uname -m) | $(rustc --version)"
echo

if [ "$suite" = "all" ] || [ "$suite" = "build" ]; then
    cp -R examples/hello "$cargo_tmp/hello"
    mkdir -p "$cargo_tmp/diff"
    cp tests/diff/Cargo.toml tests/diff/Cargo.lock "$cargo_tmp/diff/"
    cp -R tests/diff/src "$cargo_tmp/diff/src"
    cp tests/case_inputs.rs "$cargo_tmp/case_inputs.rs"
fi

if [ "$suite" = "all" ] || [ "$suite" = "compute" ]; then
    mkdir -p "$tmp/empty/src"
    cat > "$tmp/empty/Cargo.toml" <<'EOF'
[package]
name = "empty"
version = "0.1.0"
edition = "2021"
EOF
    cat > "$tmp/empty/Cargo.lock" <<'EOF'
version = 4

[[package]]
name = "empty"
version = "0.1.0"
EOF
    cat > "$tmp/empty/src/main.rs" <<'EOF'
fn main() {}
EOF
    mkdir -p "$tmp/compute/src"
    cat > "$tmp/compute/Cargo.toml" <<'EOF'
[package]
name = "compute"
version = "0.1.0"
edition = "2021"
EOF
    cat > "$tmp/compute/Cargo.lock" <<'EOF'
version = 4

[[package]]
name = "compute"
version = "0.1.0"
EOF
    cat > "$tmp/compute/src/main.rs" <<'EOF'
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        let j = i * 4;
        w[i] = u32::from_be_bytes([block[j], block[j + 1], block[j + 2], block[j + 3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g; g = f; f = e; e = d.wrapping_add(t1);
        d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a); state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c); state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e); state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g); state[7] = state[7].wrapping_add(h);
}

fn bench_sha256(mult: usize) -> u64 {
    let mut state = [0x6a09e667u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut block = [0u8; 64];
    for i in 0..64 { block[i] = (i as u8).wrapping_mul(37).wrapping_add(13); }
    for round in 0..(2_000_000usize * mult) {
        block[0] = round as u8;
        block[1] = (round >> 8) as u8;
        block[2] = (round >> 16) as u8;
        block[3] = (round >> 24) as u8;
        sha256_compress(&mut state, &block);
    }
    u64::from(state[0]) << 32 | u64::from(state[7])
}

fn bench_sieve(mult: usize) -> u64 {
    const N: usize = 4_000_000;
    let mut is_prime = vec![1u8; N + 1];
    let mut total = 0u64;
    for round in 0..(20 * mult) {
        let n = std::hint::black_box(N - (round % 1000) * 1021);
        is_prime[..=n].fill(1);
        is_prime[0] = 0;
        is_prime[1] = 0;
        let mut p = 2;
        while p * p <= n {
            if is_prime[p] != 0 {
                let mut m = p * p;
                while m <= n {
                    is_prime[m] = 0;
                    m += p;
                }
            }
            p += 1;
        }
        let mut sum = 0u64;
        for (i, &flag) in is_prime[..=n].iter().enumerate() {
            if flag != 0 { sum = sum.wrapping_add(i as u64); }
        }
        total = total.wrapping_add(sum);
    }
    total
}

fn quicksort(a: &mut [u64]) {
    if a.len() <= 16 {
        for i in 1..a.len() {
            let x = a[i];
            let mut j = i;
            while j > 0 && a[j - 1] > x {
                a[j] = a[j - 1];
                j -= 1;
            }
            a[j] = x;
        }
        return;
    }
    let mid = a.len() / 2;
    let pivot = a[mid];
    let (mut i, mut j) = (0usize, a.len() - 1);
    loop {
        while a[i] < pivot { i += 1; }
        while a[j] > pivot { j -= 1; }
        if i >= j { break; }
        a.swap(i, j);
        i += 1;
        j -= 1;
    }
    quicksort(&mut a[..=j]);
    quicksort(&mut a[j + 1..]);
}

fn bench_sort(mult: usize) -> u64 {
    let mut data = Vec::with_capacity(500_000);
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut acc = 0u64;
    for _ in 0..(8 * mult) {
        data.clear();
        for _ in 0..500_000 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            data.push(s);
        }
        quicksort(&mut data);
        for i in (0..data.len()).step_by(97) {
            if let Ok(pos) = data.binary_search(&data[i]) {
                acc = acc.wrapping_add(pos as u64 ^ data[pos]);
            }
        }
    }
    acc
}

fn bench_scan(mult: usize) -> u64 {
    let sample = br#"{"id":12345,"name":"alpha_beta","tags":["rust","compiler","jit","aot"],"active":true,"score":987654}"#;
    let mut buf = Vec::with_capacity(sample.len() * 2000);
    for _ in 0..2000 { buf.extend_from_slice(sample); }
    let mut total = 0u64;
    for round in 0..(1200 * mult) {
        buf[6 + (round % 5)] = b'0' + ((round % 9) as u8) + 1;
        let _ = std::hint::black_box(buf.as_ptr());
        let mut i = 0;
        while i < buf.len() {
            match buf[i] {
                b'{' | b'}' | b'[' | b']' | b':' | b',' => { total += 1; i += 1; }
                b'"' => {
                    i += 1;
                    let start = i;
                    while i < buf.len() && buf[i] != b'"' {
                        if buf[i] == b'\\' { i += 2; } else { i += 1; }
                    }
                    total += (i - start) as u64;
                    i += 1;
                }
                b'0'..=b'9' => {
                    let mut v = 0u64;
                    while i < buf.len() && buf[i].is_ascii_digit() {
                        v = v.wrapping_mul(10).wrapping_add((buf[i] - b'0') as u64);
                        i += 1;
                    }
                    total = total.wrapping_add(v);
                }
                _ => { i += 1; }
            }
        }
    }
    total
}

fn weighted_reduce<I, F>(iter: I, mut f: F) -> f64
where
    I: Iterator<Item = (usize, f64)>,
    F: FnMut(usize, f64) -> f64,
{
    iter.fold(0.0f64, |acc, (idx, v)| acc + f(idx, v))
}

fn bench_fp_iter(mult: usize) -> u64 {
    const N: usize = 65_536;
    let mut xs = Vec::with_capacity(N);
    let mut ys = Vec::with_capacity(N);
    for i in 0..N {
        let t = (i as f64 + 1.0) * 0.000125;
        xs.push(t * 1.125 - 0.5);
        ys.push(1.0 / (1.0 + t * t));
    }
    let mut total = 0.0f64;
    for round in 0..(4096usize * mult) {
        let scale = 1.0 + (round as f64) * 0.015625;
        let zipped = xs
            .iter()
            .zip(ys.iter())
            .enumerate()
            .filter(|(i, _)| (i & 3) != 3)
            .map(|(i, (&a, &b))| (i, (a * b + 0.25) / (1.0 + (a * a))));
        let part = weighted_reduce(zipped, |idx, val| {
            let w = if (idx & 1) == 0 { scale } else { -0.5 * scale };
            val * w
        });
        total += part;
        let slot = (round * 13) % N;
        xs[slot] = xs[slot] * 0.999_938_964_843_75 + 0.000_244_140_625;
    }
    total.to_bits()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let arg = args.next().unwrap_or_else(|| "all".to_owned());
    let mult: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .filter(|&m| m >= 1)
        .unwrap_or(1);
    let out = match arg.as_str() {
        "sha256" => bench_sha256(mult),
        "sieve" => bench_sieve(mult),
        "sort" => bench_sort(mult),
        "scan" => bench_scan(mult),
        "fp_iter" => bench_fp_iter(mult),
        _ => bench_sha256(mult) ^ bench_sieve(mult) ^ bench_sort(mult) ^ bench_scan(mult) ^ bench_fp_iter(mult),
    };
    println!("{out}");
}
EOF
fi

python3 - "$tmp" "$haste_bin" "$suite" "$cache_dir" "$cargo_tmp" "$bench_long" <<'PYEOF'
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

tmp, haste_bin, suite, cache_dir, cargo_tmp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
bench_long = sys.argv[6] == "1"
mult = 4 if bench_long else 1
run_cold = os.environ.get("HASTE_BENCH_COLD", "1") != "0"
rounds = max(1, int(os.environ.get("HASTE_BENCH_ROUNDS", "5")))
warm_haste_env = dict(os.environ)
cargo_env = {k: v for k, v in os.environ.items() if k not in ("RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER")}
cargo_env["CARGO_BUILD_RUSTC_WRAPPER"] = ""
cargo_env["CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"] = ""

def isolated_haste_env(tag):
    if not run_cold:
        return warm_haste_env
    return {**os.environ, "HASTE_HOME": os.path.join(tmp, f"haste_home_{tag}")}

cargo_bin = "cargo"
try:
    resolved = subprocess.run(["rustup", "which", "cargo"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if resolved.returncode == 0 and resolved.stdout.strip():
        cargo_bin = resolved.stdout.strip()
except OSError:
    pass

def cargo_cmd(sub, target_dir, manifest=None):
    cmd = [cargo_bin, sub, "--quiet", "--locked"]
    if manifest is not None:
        cmd.extend(["--manifest-path", manifest])
    cmd.extend(["--target-dir", target_dir])
    return cmd

def median_of(vals):
    s = sorted(vals)
    return s[len(s) // 2]

def run_timed(cmd, env=None):
    t0 = time.perf_counter()
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return (time.perf_counter() - t0) * 1000.0

def median_ms(n, cmd, env=None):
    return median_of(run_timed(cmd, env=env) for _ in range(n))

def median_ms_warmed(n, cmd, env=None):
    run_timed(cmd, env=env)
    return median_ms(n, cmd, env=env)

def median_ms_checked(n, cmd, env=None):
    warm = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
    out = warm.stdout.strip()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
        times.append((time.perf_counter() - t0) * 1000.0)
        val = res.stdout.strip()
        if val != out:
            print(f"NONDETERMINISTIC output for {cmd}: {out!r} vs {val!r}", file=sys.stderr)
            sys.exit(1)
    return median_of(times), out

def fmt_ms(ms):
    if ms >= 1000.0:
        return f"{ms / 1000.0:.1f} s"
    if ms >= 100.0:
        return f"{round(ms)} ms"
    return f"{ms:.1f} ms"

def fmt_net_raw(net_ms, raw_ms):
    return f"{fmt_ms(net_ms)} (raw {fmt_ms(raw_ms)})"

def print_build_table(title, rows, has_prev):
    if has_prev:
        print()
    print(title)
    print()
    print("| Benchmark | haste | cargo | Speedup |")
    print("| --- | ---: | ---: | ---: |")
    for label, h_ms, c_ms in rows:
        speedup = f"{c_ms / max(h_ms, 1e-6):.1f}x"
        print(f"| {label} | {fmt_ms(h_ms)} | {fmt_ms(c_ms)} | {speedup} |")

def write_edit(path, orig_src, needle, value):
    new_src = orig_src.replace(needle, f"(x + {value})", 1)
    assert new_src != orig_src, f"failed to mutate {path}"
    path.write_text(new_src)

sections = 0

if suite in ("all", "build"):
    hello_dir = os.path.join(cargo_tmp, "hello")
    diff_dir = os.path.join(cargo_tmp, "diff")
    diff_manifest = os.path.join(diff_dir, "Cargo.toml")
    diff_main = Path(diff_dir) / "src" / "main.rs"
    case_count = sum(
        len(re.findall(r'"case_[a-zA-Z0-9_]+"', p.read_text()))
        for p in (Path(diff_dir) / "src").glob("*.rs")
    )
    cargo_check_target = os.path.join(cargo_tmp, "cargo_check_target")
    cargo_build_target = os.path.join(cargo_tmp, "cargo_build_target")
    haste_out_bin = os.path.join(cargo_tmp, "diff_haste_out")
    check_haste_env = isolated_haste_env("check")
    emit_haste_env = isolated_haste_env("emit")
    daemon_haste_env = isolated_haste_env("daemon")
    haste_check_cmd = [haste_bin, "build", diff_dir]
    haste_daemon_cmd = [haste_bin, "build", "--daemon", diff_dir]
    haste_emit_cmd = [haste_bin, "build", "-o", haste_out_bin, diff_dir]
    cargo_check_cmd = cargo_cmd("check", cargo_check_target, diff_manifest)
    cargo_build_cmd = cargo_cmd("build", cargo_build_target, diff_manifest)

    h_check_empty = None
    h_emit_empty = None
    if run_cold:
        empty_check_env = isolated_haste_env("empty_check")
        empty_emit_env = isolated_haste_env("empty_emit")
        h_check_empty = run_timed(haste_check_cmd, env=empty_check_env)
        h_emit_empty = run_timed(haste_emit_cmd, env=empty_emit_env)
        for env in (check_haste_env, emit_haste_env, daemon_haste_env):
            subprocess.run([haste_bin, "build", hello_dir], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    h_check_cold = run_timed(haste_check_cmd, env=check_haste_env)
    c_check_cold = run_timed(cargo_check_cmd, env=cargo_env)
    h_emit_cold = run_timed(haste_emit_cmd, env=emit_haste_env)
    c_build_cold = run_timed(cargo_build_cmd, env=cargo_env)

    h_check_noop = median_ms(5, haste_check_cmd, env=check_haste_env)
    c_check_noop = median_ms(5, cargo_check_cmd, env=cargo_env)
    h_emit_noop = median_ms(5, haste_emit_cmd, env=emit_haste_env)
    c_build_noop = median_ms(5, cargo_build_cmd, env=cargo_env)

    orig_src = diff_main.read_text()
    needle = "(x + 3)"
    assert needle in orig_src, f"expected {needle!r} in {diff_main}"
    seed = int(time.time()) % 100_000
    h_check_edits, c_check_edits = [], []
    h_emit_edits, c_build_edits = [], []
    h_daemon_edits, c_daemon_check_edits = [], []
    cargo_out_bin = os.path.join(cargo_build_target, "debug", "diff" + (".exe" if os.name == "nt" else ""))
    prev_emit_out = subprocess.check_output([haste_out_bin], stderr=subprocess.DEVNULL)
    for i in range(1, rounds + 1):
        write_edit(diff_main, orig_src, needle, seed + i * 10)
        if i % 2 == 1:
            h_check_edits.append(run_timed(haste_check_cmd, env=check_haste_env))
            c_check_edits.append(run_timed(cargo_check_cmd, env=cargo_env))
            h_emit_edits.append(run_timed(haste_emit_cmd, env=emit_haste_env))
            c_build_edits.append(run_timed(cargo_build_cmd, env=cargo_env))
        else:
            c_check_edits.append(run_timed(cargo_check_cmd, env=cargo_env))
            h_check_edits.append(run_timed(haste_check_cmd, env=check_haste_env))
            c_build_edits.append(run_timed(cargo_build_cmd, env=cargo_env))
            h_emit_edits.append(run_timed(haste_emit_cmd, env=emit_haste_env))
        h_out = subprocess.check_output([haste_out_bin], stderr=subprocess.DEVNULL)
        c_out = subprocess.check_output([cargo_out_bin], stderr=subprocess.DEVNULL)
        assert h_out == c_out, f"edit {i}: haste output differed from cargo output"
        assert h_out != prev_emit_out, f"edit {i}: rebuilt executable output did not change"
        prev_emit_out = h_out
    if os.name == "posix":
        def stop_daemon():
            subprocess.run([haste_bin, "serve", "stop", diff_dir], env=daemon_haste_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        def on_signal(signum, _frame):
            stop_daemon()
            sys.exit(128 + signum)
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        try:
            subprocess.run(haste_daemon_cmd, env=daemon_haste_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            for _ in range(100):
                st = subprocess.run([haste_bin, "serve", "status", diff_dir], env=daemon_haste_env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                if "pid=" in st.stdout:
                    break
                time.sleep(0.1)
            else:
                sys.exit("haste serve daemon did not become ready")
            for i in range(rounds + 1, 2 * rounds + 1):
                write_edit(diff_main, orig_src, needle, seed + i * 10)
                if i % 2 == 1:
                    h_daemon_edits.append(run_timed(haste_daemon_cmd, env=daemon_haste_env))
                    c_daemon_check_edits.append(run_timed(cargo_check_cmd, env=cargo_env))
                else:
                    c_daemon_check_edits.append(run_timed(cargo_check_cmd, env=cargo_env))
                    h_daemon_edits.append(run_timed(haste_daemon_cmd, env=daemon_haste_env))
        finally:
            stop_daemon()
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)
    h_check_edit = median_of(h_check_edits)
    c_check_edit = median_of(c_check_edits)
    h_emit_edit = median_of(h_emit_edits)
    c_build_edit = median_of(c_build_edits)

    rows = [
        ("Single-line edit (`haste build` vs `cargo check`)", h_check_edit, c_check_edit),
        ("Single-line edit (`haste build -o` vs `cargo build`)", h_emit_edit, c_build_edit),
    ]
    if h_daemon_edits:
        rows.insert(0, ("Single-line edit (`haste build --daemon` vs `cargo check`)", median_of(h_daemon_edits), median_of(c_daemon_check_edits)))
    rows.extend([
        ("Unchanged no-op (`haste build` vs `cargo check`)", h_check_noop, c_check_noop),
        ("Unchanged no-op (`haste build -o` vs `cargo build`)", h_emit_noop, c_build_noop),
    ])
    if run_cold:
        rows.extend([
            ("Empty HASTE_HOME (`haste build` vs `cargo check`)", h_check_empty, c_check_cold),
            ("Empty HASTE_HOME (`haste build -o` vs `cargo build`)", h_emit_empty, c_build_cold),
            ("First project build, stdlib indexed (`haste build` vs `cargo check`)", h_check_cold, c_check_cold),
            ("First project build, stdlib indexed (`haste build -o` vs `cargo build`)", h_emit_cold, c_build_cold),
        ])
    else:
        rows.extend([
            ("Warm cache (`haste build` vs `cargo check`)", h_check_cold, c_check_cold),
            ("Warm cache (`haste build -o` vs `cargo build`)", h_emit_cold, c_build_cold),
        ])
    print_build_table(f"### Build Speed (`tests/diff`, {case_count} cases)", rows, sections > 0)
    sections += 1

if suite in ("all", "self"):
    haste_self_cmd = [haste_bin, "build", "."]
    if run_cold:
        self_haste_env = isolated_haste_env("self")
        self_cargo_check_target = os.path.join(cargo_tmp, "self_cargo_check_target")
        self_cargo_build_target = os.path.join(cargo_tmp, "self_cargo_build_target")
        cargo_self_check_cmd = cargo_cmd("check", self_cargo_check_target)
        cargo_self_build_cmd = cargo_cmd("build", self_cargo_build_target)
        h_self_cold = run_timed(haste_self_cmd, env=self_haste_env)
        c_self_check_cold = run_timed(cargo_self_check_cmd, env=cargo_env)
        c_self_build_cold = run_timed(cargo_self_build_cmd, env=cargo_env)
    else:
        self_haste_env = warm_haste_env
        self_cargo_check_target = os.path.join(cache_dir, "self_cargo_check")
        self_cargo_build_target = os.path.join(cache_dir, "self_cargo")
        os.makedirs(self_cargo_check_target, exist_ok=True)
        os.makedirs(self_cargo_build_target, exist_ok=True)
        cargo_self_check_cmd = cargo_cmd("check", self_cargo_check_target)
        cargo_self_build_cmd = cargo_cmd("build", self_cargo_build_target)
        procs = [
            subprocess.Popen(haste_self_cmd, env=self_haste_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
            subprocess.Popen(cargo_self_check_cmd, env=cargo_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
            subprocess.Popen(cargo_self_build_cmd, env=cargo_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        ]
        if any(p.wait() != 0 for p in procs):
            sys.exit(1)
        h_self_cold = None
        c_self_check_cold = None
        c_self_build_cold = None
    h_self_noop = median_ms(5, haste_self_cmd, env=self_haste_env)
    c_self_check_noop = median_ms(5, cargo_self_check_cmd, env=cargo_env)
    c_self_build_noop = median_ms(5, cargo_self_build_cmd, env=cargo_env)

    self_rows = []
    if h_self_cold is not None:
        self_rows.append(("Cold debug (`haste build` vs `cargo check`)", h_self_cold, c_self_check_cold))
        self_rows.append(("Cold debug (`haste build` vs `cargo build`)", h_self_cold, c_self_build_cold))
    self_rows.append(("Unchanged no-op (`haste build` vs `cargo check`)", h_self_noop, c_self_check_noop))
    self_rows.append(("Unchanged no-op (`haste build` vs `cargo build`)", h_self_noop, c_self_build_noop))
    print_build_table("### Self-Host Build Speed (`.`)", self_rows, sections > 0)
    sections += 1

if suite in ("all", "compute"):
    empty_dir = os.path.join(tmp, "empty")
    empty_haste = os.path.join(tmp, "empty_haste")
    empty_rustc = os.path.join(tmp, "empty_rustc")
    empty_rustc_matched = os.path.join(tmp, "empty_rustc_matched")
    empty_main = os.path.join(empty_dir, "src", "main.rs")
    compute_dir = os.path.join(tmp, "compute")
    compute_haste = os.path.join(tmp, "compute_haste")
    compute_rustc = os.path.join(tmp, "compute_rustc")
    compute_rustc_matched = os.path.join(tmp, "compute_rustc_matched")
    src_main = os.path.join(compute_dir, "src", "main.rs")
    procs = [
        subprocess.Popen([haste_bin, "--release", "build", "-o", empty_haste, empty_dir], env=warm_haste_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen(["rustc", "-C", "opt-level=3", "--edition=2021", empty_main, "-o", empty_rustc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen(["rustc", "-C", "opt-level=3", "-C", "target-cpu=native", "-C", "panic=abort", "--edition=2021", empty_main, "-o", empty_rustc_matched], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([haste_bin, "--release", "build", "-o", compute_haste, compute_dir], env=warm_haste_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen(["rustc", "-C", "opt-level=3", "--edition=2021", src_main, "-o", compute_rustc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen(["rustc", "-C", "opt-level=3", "-C", "target-cpu=native", "-C", "panic=abort", "--edition=2021", src_main, "-o", compute_rustc_matched], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    ]
    if any(p.wait() != 0 for p in procs):
        sys.exit(1)

    baseline_rounds = max(rounds, 7)
    h_base = median_ms_warmed(baseline_rounds, [empty_haste])
    c_base = median_ms_warmed(baseline_rounds, [empty_rustc])
    m_base = median_ms_warmed(baseline_rounds, [empty_rustc_matched])

    if sections:
        print()
    scale_label = f"long ({mult}x)" if bench_long else "default"
    print(f"### AOT Codegen Runtime (`haste --release` vs `rustc`, median of {rounds}, warmed, scale={scale_label})")
    print()
    print(
        f"Empty-main startup baseline (median of {baseline_rounds}, warmed): "
        f"`haste --release` = {fmt_ms(h_base)} | `rustc -O3` = {fmt_ms(c_base)} | `rustc -O3 native+abort` = {fmt_ms(m_base)}"
    )
    print()
    print("| Workload | haste --release (net / raw) | rustc -O3 (net / raw) | vs -O3 (haste/rustc) | rustc -O3 native+abort (net / raw) | vs matched (haste/rustc) | Output |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for work in ("sha256", "sieve", "sort", "scan", "fp_iter"):
        cmds = [
            ("h", [compute_haste, work, str(mult)]),
            ("c", [compute_rustc, work, str(mult)]),
            ("m", [compute_rustc_matched, work, str(mult)]),
        ]
        outs = {}
        times = {"h": [], "c": [], "m": []}
        for key, cmd in cmds:
            outs[key] = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True).stdout.strip()
        if outs["h"] != outs["c"] or outs["h"] != outs["m"]:
            print(f"MISMATCH on {work}: haste={outs['h']} rustc={outs['c']} matched={outs['m']}", file=sys.stderr)
            sys.exit(1)
        for r in range(rounds):
            order = cmds[r % 3:] + cmds[:r % 3]
            for key, cmd in order:
                t0 = time.perf_counter()
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
                times[key].append((time.perf_counter() - t0) * 1000.0)
                val = res.stdout.strip()
                if val != outs[key]:
                    print(f"NONDETERMINISTIC output for {cmd}: {outs[key]!r} vs {val!r}", file=sys.stderr)
                    sys.exit(1)
        h_raw, c_raw, m_raw = median_of(times["h"]), median_of(times["c"]), median_of(times["m"])
        h_out = outs["h"]
        h_net = max(0.0, h_raw - h_base)
        c_net = max(0.0, c_raw - c_base)
        m_net = max(0.0, m_raw - m_base)
        r_default = f"{h_net / max(c_net, 1e-3):.2f}x"
        r_matched = f"{h_net / max(m_net, 1e-3):.2f}x"
        print(
            f"| `{work}` | {fmt_net_raw(h_net, h_raw)} | {fmt_net_raw(c_net, c_raw)} | {r_default} | "
            f"{fmt_net_raw(m_net, m_raw)} | {r_matched} | exact match (`{h_out}`) |"
        )
PYEOF
