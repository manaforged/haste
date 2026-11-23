#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cargo.sh

suite="${1:-${HASTE_BENCH_SUITE:-all}}"
case "$suite" in
    all|compute|build|self) ;;
    *) echo "usage: $0 [all|compute|build|self]" >&2; exit 2 ;;
esac

build_haste </dev/null >/dev/null 2>&1 || { echo "haste does not build"; exit 2; }
haste_bin="${HASTE_BIN:-${CARGO_TARGET_DIR:-target}/release/haste}"
cache_dir="${CARGO_TARGET_DIR:-target}/bench-cache"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "Host: $(uname -s) $(uname -m) | $(rustc --version)"
echo

if [ "$suite" = "all" ] || [ "$suite" = "build" ]; then
    cp -R examples/hello "$tmp/hello"
    cp -R tests/diff "$tmp/diff"
    cp tests/case_inputs.rs "$tmp/case_inputs.rs"
fi

if [ "$suite" = "all" ] || [ "$suite" = "compute" ]; then
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

fn bench_sha256() -> u64 {
    let mut state = [0x6a09e667u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut block = [0u8; 64];
    for i in 0..64 { block[i] = (i as u8).wrapping_mul(37).wrapping_add(13); }
    for round in 0..250_000u32 {
        block[0] = round as u8;
        block[1] = (round >> 8) as u8;
        sha256_compress(&mut state, &block);
    }
    u64::from(state[0]) << 32 | u64::from(state[7])
}

fn bench_sieve() -> u64 {
    const N: usize = 4_000_000;
    let mut is_prime = vec![1u8; N + 1];
    is_prime[0] = 0;
    is_prime[1] = 0;
    let mut p = 2;
    while p * p <= N {
        if is_prime[p] != 0 {
            let mut m = p * p;
            while m <= N {
                is_prime[m] = 0;
                m += p;
            }
        }
        p += 1;
    }
    let mut sum = 0u64;
    for (i, &flag) in is_prime.iter().enumerate() {
        if flag != 0 { sum = sum.wrapping_add(i as u64); }
    }
    sum
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

fn bench_sort() -> u64 {
    let mut data = Vec::with_capacity(500_000);
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..500_000 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        data.push(s);
    }
    quicksort(&mut data);
    let mut acc = 0u64;
    for i in (0..data.len()).step_by(97) {
        if let Ok(pos) = data.binary_search(&data[i]) {
            acc = acc.wrapping_add(pos as u64 ^ data[pos]);
        }
    }
    acc
}

fn bench_scan() -> u64 {
    let sample = br#"{"id":12345,"name":"alpha_beta","tags":["rust","compiler","jit","aot"],"active":true,"score":987654}"#;
    let mut buf = Vec::with_capacity(sample.len() * 2000);
    for _ in 0..2000 { buf.extend_from_slice(sample); }
    let mut total = 0u64;
    for _ in 0..40 {
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

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| "all".to_owned());
    let out = match arg.as_str() {
        "sha256" => bench_sha256(),
        "sieve" => bench_sieve(),
        "sort" => bench_sort(),
        "scan" => bench_scan(),
        _ => bench_sha256() ^ bench_sieve() ^ bench_sort() ^ bench_scan(),
    };
    println!("{out}");
}
EOF
fi

python3 - "$tmp" "$haste_bin" "$suite" "$cache_dir" <<'PYEOF'
import os
import subprocess
import sys
import time
from pathlib import Path

tmp, haste_bin, suite, cache_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
run_cold = os.environ.get("HASTE_BENCH_COLD", "1") != "0"
rounds = max(1, int(os.environ.get("HASTE_BENCH_ROUNDS", "5")))
cold_haste_env = {**os.environ, "HASTE_HOME": os.path.join(tmp, "haste_home")}
warm_haste_env = dict(os.environ)
cargo_env = {k: v for k, v in os.environ.items() if k not in ("RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")}

def run_timed(cmd, env=None):
    t0 = time.perf_counter()
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return (time.perf_counter() - t0) * 1000.0

def median_ms(n, cmd, env=None):
    times = sorted(run_timed(cmd, env=env) for _ in range(n))
    return times[len(times) // 2]

def median_ms_checked(n, cmd, env=None):
    times = []
    out = None
    for _ in range(n):
        t0 = time.perf_counter()
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
        times.append((time.perf_counter() - t0) * 1000.0)
        val = res.stdout.strip()
        if out is None:
            out = val
        elif val != out:
            print(f"NONDETERMINISTIC output for {cmd}: {out!r} vs {val!r}", file=sys.stderr)
            sys.exit(1)
    times.sort()
    return times[len(times) // 2], out

def fmt_ms(ms):
    if ms >= 1000.0:
        return f"{ms / 1000.0:.1f} s"
    if ms >= 100.0:
        return f"{round(ms)} ms"
    return f"{ms:.1f} ms"

sections = 0

if suite in ("all", "build"):
    hello_dir = os.path.join(tmp, "hello")
    diff_dir = os.path.join(tmp, "diff")
    diff_manifest = os.path.join(diff_dir, "Cargo.toml")
    diff_main = Path(diff_dir) / "src" / "main.rs"
    cargo_target = os.path.join(tmp, "cargo_target")
    haste_env = cold_haste_env if run_cold else warm_haste_env
    haste_build_cmd = [haste_bin, "build", diff_dir]
    cargo_build_cmd = ["cargo", "build", "--quiet", "--locked", "--manifest-path", diff_manifest, "--target-dir", cargo_target]

    if run_cold:
        subprocess.run([haste_bin, "build", hello_dir], env=haste_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    h_diff_cold = run_timed(haste_build_cmd, env=haste_env)
    c_diff_cold = run_timed(cargo_build_cmd, env=cargo_env)
    h_diff_noop = median_ms(5, haste_build_cmd, env=haste_env)
    c_diff_noop = median_ms(5, cargo_build_cmd, env=cargo_env)

    orig_src = diff_main.read_text()
    needle = "(x + 3)"
    assert needle in orig_src, f"expected {needle!r} in {diff_main}"
    seed = int(time.time()) % 100_000
    prev_src = orig_src
    h_edits = []
    c_edits = []
    for i in range(1, 4):
        new_src = orig_src.replace(needle, f"(x + {seed + i * 10})", 1)
        assert new_src != prev_src
        diff_main.write_text(new_src)
        prev_src = new_src
        h_edits.append(run_timed(haste_build_cmd, env=haste_env))
        c_edits.append(run_timed(cargo_build_cmd, env=cargo_env))
    h_diff_edit = sorted(h_edits)[1]
    c_diff_edit = sorted(c_edits)[1]

    print("### Build Speed (`tests/diff`, 129 fns)")
    print()
    print("| Benchmark | haste | cargo | Speedup |")
    print("| --- | ---: | ---: | ---: |")
    for label, h_ms, c_ms in (
        ("Single-line code edit rebuild", h_diff_edit, c_diff_edit),
        ("Unchanged no-op build", h_diff_noop, c_diff_noop),
        ("Cold debug build", h_diff_cold, c_diff_cold),
    ):
        speedup = f"{c_ms / max(h_ms, 1e-6):.1f}x"
        print(f"| {label} | {fmt_ms(h_ms)} | {fmt_ms(c_ms)} | {speedup} |")
    sections += 1

if suite in ("all", "self"):
    haste_self_cmd = [haste_bin, "build", "."]
    if run_cold:
        self_haste_env = cold_haste_env
        self_cargo_target = os.path.join(tmp, "self_cargo_target")
        cargo_self_cmd = ["cargo", "build", "--quiet", "--locked", "--target-dir", self_cargo_target]
        h_self_cold = run_timed(haste_self_cmd, env=self_haste_env)
        c_self_cold = run_timed(cargo_self_cmd, env=cargo_env)
    else:
        self_haste_env = warm_haste_env
        self_cargo_target = os.path.join(cache_dir, "self_cargo")
        os.makedirs(self_cargo_target, exist_ok=True)
        cargo_self_cmd = ["cargo", "build", "--quiet", "--locked", "--target-dir", self_cargo_target]
        p_h = subprocess.Popen(haste_self_cmd, env=self_haste_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p_c = subprocess.Popen(cargo_self_cmd, env=cargo_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p_h.wait() != 0 or p_c.wait() != 0:
            sys.exit(1)
        h_self_cold = None
        c_self_cold = None
    h_self_noop = median_ms(5, haste_self_cmd, env=self_haste_env)
    c_self_noop = median_ms(5, cargo_self_cmd, env=cargo_env)

    if sections:
        print()
    print("### Self-Host Build Speed (`.`)")
    print()
    print("| Benchmark | haste | cargo | Speedup |")
    print("| --- | ---: | ---: | ---: |")
    self_rows = []
    if h_self_cold is not None:
        self_rows.append(("Cold debug build", h_self_cold, c_self_cold))
    self_rows.append(("Unchanged no-op build", h_self_noop, c_self_noop))
    for label, h_ms, c_ms in self_rows:
        speedup = f"{c_ms / max(h_ms, 1e-6):.1f}x"
        print(f"| {label} | {fmt_ms(h_ms)} | {fmt_ms(c_ms)} | {speedup} |")
    sections += 1

if suite in ("all", "compute"):
    compute_dir = os.path.join(tmp, "compute")
    compute_haste = os.path.join(tmp, "compute_haste")
    compute_rustc = os.path.join(tmp, "compute_rustc")
    p_h = subprocess.Popen([haste_bin, "--release", "build", "-o", compute_haste, compute_dir], env=warm_haste_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p_r = subprocess.Popen(["rustc", "-C", "opt-level=3", "--edition=2021", os.path.join(compute_dir, "src", "main.rs"), "-o", compute_rustc], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if p_h.wait() != 0 or p_r.wait() != 0:
        sys.exit(1)

    if sections:
        print()
    print(f"### AOT Codegen Runtime (`haste --release` vs `rustc -C opt-level=3`, median of {rounds})")
    print()
    print("| Workload | haste --release | rustc -O3 | Ratio | Output |")
    print("| --- | ---: | ---: | ---: | --- |")
    for work in ("sha256", "sieve", "sort", "scan"):
        h_ms, h_out = median_ms_checked(rounds, [compute_haste, work])
        c_ms, c_out = median_ms_checked(rounds, [compute_rustc, work])
        if h_out != c_out:
            print(f"MISMATCH on {work}: haste={h_out} rustc={c_out}", file=sys.stderr)
            sys.exit(1)
        ratio = f"{h_ms / max(c_ms, 1e-6):.2f}x"
        print(f"| `{work}` | {fmt_ms(h_ms)} | {fmt_ms(c_ms)} | {ratio} | exact match (`{h_out}`) |")
PYEOF
