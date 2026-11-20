#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cargo.sh

build_haste </dev/null >/dev/null 2>&1 || { echo "haste does not build"; exit 2; }
haste_bin="${HASTE_BIN:-${CARGO_TARGET_DIR:-target}/release/haste}"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

now_ms() {
    python3 -c 'import time; print(int(time.perf_counter() * 1000))'
}

median_ms() {
    local rounds=$1; shift
    local times=()
    for ((i = 0; i < rounds; i++)); do
        local t0 t1
        t0=$(now_ms)
        "$@" >/dev/null 2>&1
        t1=$(now_ms)
        times+=($((t1 - t0)))
    done
    printf '%s\n' "${times[@]}" | sort -n | sed -n "$(((rounds + 1) / 2))p"
}

echo "Host: $(uname -s) $(uname -m) | $(rustc --version)"
echo

# 1. Build-speed benchmarks on examples/hello and tests/diff
cp -R examples/hello "$tmp/hello"
cp -R tests/diff "$tmp/diff"
cp tests/case_inputs.rs "$tmp/case_inputs.rs"

haste_home="$tmp/haste_home"
cargo_target="$tmp/cargo_target"

# Warm haste's stdlib substrate once (matches normal developer steady state)
HASTE_HOME="$haste_home" "$haste_bin" build "$tmp/hello" >/dev/null 2>&1

# Cold debug build (tests/diff, 124 fns / ~9,300 reached std+user fns)
t0=$(now_ms); HASTE_HOME="$haste_home" "$haste_bin" build "$tmp/diff" >/dev/null 2>&1; t1=$(now_ms)
h_diff_cold=$((t1 - t0))
t0=$(now_ms); cargo build --quiet --locked --manifest-path "$tmp/diff/Cargo.toml" --target-dir "$cargo_target" >/dev/null 2>&1; t1=$(now_ms)
c_diff_cold=$((t1 - t0))

# Unchanged no-op build (median of 5)
h_diff_noop=$(HASTE_HOME="$haste_home" median_ms 5 "$haste_bin" build "$tmp/diff")
c_diff_noop=$(median_ms 5 cargo build --quiet --locked --manifest-path "$tmp/diff/Cargo.toml" --target-dir "$cargo_target")

# Single-line edit rebuild (median of 3)
h_edits=()
c_edits=()
for i in 1 2 3; do
    printf '\n// edit %d\n' "$i" >> "$tmp/diff/src/main.rs"
    t0=$(now_ms); HASTE_HOME="$haste_home" "$haste_bin" build "$tmp/diff" >/dev/null 2>&1; t1=$(now_ms)
    h_edits+=($((t1 - t0)))
    t0=$(now_ms); cargo build --quiet --locked --manifest-path "$tmp/diff/Cargo.toml" --target-dir "$cargo_target" >/dev/null 2>&1; t1=$(now_ms)
    c_edits+=($((t1 - t0)))
done
h_diff_edit=$(printf '%s\n' "${h_edits[@]}" | sort -n | sed -n '2p')
c_diff_edit=$(printf '%s\n' "${c_edits[@]}" | sort -n | sed -n '2p')

# 2. CPU-bound AOT codegen benchmarks: haste --release vs rustc -C opt-level=3
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

HASTE_HOME="$haste_home" "$haste_bin" --release build -o "$tmp/compute_haste" "$tmp/compute" >/dev/null 2>&1
rustc -C opt-level=3 --edition=2021 "$tmp/compute/src/main.rs" -o "$tmp/compute_rustc"

echo "### Build Speed (\`tests/diff\`, 124 fns)"
echo
echo "| Benchmark | haste | cargo |"
echo "| --- | ---: | ---: |"
echo "| Single-line edit rebuild | ${h_diff_edit} ms | ${c_diff_edit} ms |"
echo "| Unchanged no-op build | ${h_diff_noop} ms | ${c_diff_noop} ms |"
echo "| Cold debug build | ${h_diff_cold} ms | ${c_diff_cold} ms |"
echo
echo "### AOT Codegen Runtime (\`haste --release\` vs \`rustc -C opt-level=3\`, median of 5)"
echo
echo "| Workload | haste --release | rustc -O3 | Ratio | Output |"
echo "| --- | ---: | ---: | ---: | --- |"

for work in sha256 sieve sort scan; do
    h_out=$("$tmp/compute_haste" "$work")
    c_out=$("$tmp/compute_rustc" "$work")
    if [ "$h_out" != "$c_out" ]; then
        echo "MISMATCH on $work: haste=$h_out rustc=$c_out" >&2
        exit 1
    fi
    h_ms=$(median_ms 5 "$tmp/compute_haste" "$work")
    c_ms=$(median_ms 5 "$tmp/compute_rustc" "$work")
    ratio=$(python3 -c "print(f'{$h_ms / max($c_ms, 1):.2f}x')")
    echo "| \`$work\` | ${h_ms} ms | ${c_ms} ms | ${ratio} | exact match (\`$h_out\`) |"
done
