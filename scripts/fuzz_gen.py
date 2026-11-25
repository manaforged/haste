#!/usr/bin/env python3
"""Deterministic random integer-program generator for differential testing against rustc."""

import os
import random
import sys

INTS = [
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "i128",
    "u128",
    "usize",
    "isize",
]
BITS = {
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "u8": 8,
    "u16": 16,
    "u32": 32,
    "u64": 64,
    "i128": 128,
    "u128": 128,
    "usize": 64,
    "isize": 64,
}


def is_signed(ty: str) -> bool:
    return ty[0] == "i"


def lit(rng: random.Random, ty: str) -> str:
    bits = BITS[ty]
    if is_signed(ty):
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        choices = [
            lo,
            lo + 1,
            -1,
            0,
            1,
            2,
            hi,
            hi - 1,
            -(1 << (bits // 2)),
            (1 << (bits // 2)) - 1,
            rng.randint(lo, hi),
            rng.randint(-200, 200),
        ]
    else:
        hi = (1 << bits) - 1
        choices = [
            0,
            1,
            2,
            hi,
            hi - 1,
            1 << (bits - 1),
            (1 << (bits // 2)) - 1,
            1 << (bits // 2),
            rng.randint(0, hi),
            rng.randint(0, 300) & hi,
        ]
    val = rng.choice(choices)
    if val < 0:
        if val == -(1 << (bits - 1)):
            return f"{ty}::MIN"
        return f"({val}{ty})"
    return f"{val}{ty}"


def expr(rng: random.Random, ty: str, depth: int, vars_list: list[tuple[str, str]]) -> str:
    same = [name for name, vt in vars_list if vt == ty]
    if depth <= 0 or rng.random() < 0.12:
        if same and rng.random() < 0.7:
            return rng.choice(same)
        if vars_list and rng.random() < 0.5:
            name, _ = rng.choice(vars_list)
            return f"({name} as {ty})"
        return lit(rng, ty)
    pick = rng.random()
    sub = lambda target=ty, d=depth - 1: expr(rng, target, d, vars_list)
    if pick < 0.30:
        op = rng.choice(
            [
                "wrapping_add",
                "wrapping_sub",
                "wrapping_mul",
                "min",
                "max",
                "saturating_add",
                "saturating_sub",
                "saturating_mul",
            ]
        )
        return f"{sub()}.{op}({sub()})"
    if pick < 0.42:
        op = rng.choice(["&", "|", "^"])
        return f"({sub()} {op} {sub()})"
    if pick < 0.52:
        op = rng.choice(["wrapping_shl", "wrapping_shr", "rotate_left", "rotate_right"])
        return f'{sub()}.{op}({sub("u32")})'
    if pick < 0.60:
        op = rng.choice(["checked_div", "checked_rem", "checked_add", "checked_mul", "checked_sub"])
        return f"{sub()}.{op}({sub()}).unwrap_or({lit(rng, ty)})"
    if pick < 0.70:
        other = rng.choice(INTS)
        return f"({sub(other)} as {ty})"
    if pick < 0.80:
        cmp_ty = rng.choice(INTS)
        cmp_op = rng.choice(["<", "<=", ">", ">=", "==", "!="])
        return f"(if {sub(cmp_ty)} {cmp_op} {sub(cmp_ty)} {{ {sub()} }} else {{ {sub()} }})"
    if pick < 0.85:
        other = rng.choice(INTS)
        fn_name = rng.choice(["count_ones", "leading_zeros", "trailing_zeros"])
        return f"({sub(other)}.{fn_name}() as {ty})"
    if pick < 0.90:
        op = rng.choice(["overflowing_add", "overflowing_sub", "overflowing_mul"])
        return f"{{ let (v, o) = {sub()}.{op}({sub()}); if o {{ !v }} else {{ v }} }}"
    if pick < 0.95:
        flt = rng.choice(["f64", "f32"])
        o1 = rng.choice(INTS)
        o2 = rng.choice(INTS)
        fop = rng.choice(["+", "-", "*", "/"])
        return f"((({sub(o1)} as {flt}) {fop} ({sub(o2)} as {flt})) as {ty})"
    if is_signed(ty):
        op = rng.choice(["wrapping_abs", "wrapping_neg", "signum"])
        return f"{sub()}.{op}()"
    op = rng.choice(["swap_bytes", "reverse_bits", "wrapping_neg"])
    return f"{sub()}.{op}()"


def fold_var(name: str, ty: str, shift: int) -> str:
    if ty in ("i128", "u128"):
        base = f"(({name} as u64) ^ (({name} >> 64) as u64))"
    else:
        base = f"({name} as u64)"
    return f"{base}.rotate_left({shift % 64})"


def make_func(rng: random.Random, idx: int) -> tuple[list[tuple[str, str]], str]:
    params = [(f"p{j}", rng.choice(INTS)) for j in range(3)]
    vars_list = list(params)
    body: list[str] = []
    for step in range(rng.randint(3, 6)):
        ty = rng.choice(INTS)
        name = f"v{step}"
        pick = rng.random()
        if pick < 0.6:
            body.append(f"    let {name}: {ty} = {expr(rng, ty, 3, vars_list)};")
        elif pick < 0.85:
            body.append(f"    let mut {name}: {ty} = {expr(rng, ty, 2, vars_list)};")
            body.append(f'    for i in 0..({expr(rng, "u32", 2, vars_list)} & 31) {{')
            loop_vars = vars_list + [(name, ty), ("i", "u32")]
            body.append(f"        {name} = {expr(rng, ty, 3, loop_vars)};")
            if rng.random() < 0.4:
                body.append(f"        if {expr(rng, ty, 1, loop_vars)} == {lit(rng, ty)} {{ break; }}")
            body.append("    }")
        else:
            length = rng.choice([3, 4, 8, 16])
            elems = ", ".join(expr(rng, ty, 2, vars_list) for _ in range(length))
            body.append(f"    let arr{step}: [{ty}; {length}] = [{elems}];")
            body.append(
                f'    let mut {name}: {ty} = arr{step}[({expr(rng, "usize", 2, vars_list)}) % {length}];'
            )
            body.append(
                f"    for k in 0..{length} {{ {name} = {name}.wrapping_mul(31).wrapping_add(arr{step}[k]); }}"
            )
        vars_list.append((name, ty))
    fold = " ^ ".join(fold_var(name, ty, j * 7) for j, (name, ty) in enumerate(vars_list))
    attr = "#[inline(never)]\n" if rng.random() < 0.5 else ""
    sig = ", ".join(f"{name}: {ty}" for name, ty in params)
    src = f"{attr}fn f{idx}({sig}) -> u64 {{\n" + "\n".join(body) + f"\n    {fold}\n}}\n"
    return params, src


def preamble() -> str:
    return """#![allow(unused, unreachable_code, arithmetic_overflow, overflowing_literals, unconditional_panic, clippy::all)]
use std::hint::black_box;
use std::mem::size_of;
use std::num::{NonZeroI32, NonZeroU16, NonZeroU32, NonZeroU64};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering as AtomOrd};
use std::sync::Mutex;

#[inline(never)]
fn safe_f64_bits(v: f64) -> u64 {
    if v.is_nan() { 0x7ff8_0000_0000_0001u64 } else { v.to_bits() }
}

#[inline(never)]
fn safe_f32_bits(v: f32) -> u32 {
    if v.is_nan() { 0x7fc0_0001u32 } else { v.to_bits() }
}

#[inline(never)]
fn nan_safe_cmp_f64(a: f64, b: f64) -> u64 {
    if a.is_nan() && b.is_nan() {
        0
    } else if a.is_nan() {
        1
    } else if b.is_nan() {
        2
    } else if a < b {
        3
    } else if a > b {
        4
    } else if a.to_bits() == b.to_bits() {
        5
    } else {
        6
    }
}

#[inline(never)]
fn nan_safe_cmp_f32(a: f32, b: f32) -> u64 {
    if a.is_nan() && b.is_nan() {
        0
    } else if a.is_nan() {
        1
    } else if b.is_nan() {
        2
    } else if a < b {
        3
    } else if a > b {
        4
    } else if a.to_bits() == b.to_bits() {
        5
    } else {
        6
    }
}

#[inline(never)]
fn slice_hash(s: &[u64]) -> u64 {
    let mut h = s.len() as u64;
    for (i, &v) in s.iter().enumerate() {
        h = h.rotate_left(5).wrapping_add(v ^ (i as u64));
    }
    h
}
"""


def gen_adt_section(rng: random.Random) -> str:
    w0, w1, w2, w3, w4 = [rng.randrange(3, 99) | 1 for _ in range(5)]
    rec_fields = [
        ("tag", "u8"),
        ("small", "i16"),
        ("wide", "i128"),
        ("fp", "f64"),
        ("flag", "bool"),
    ]
    rng.shuffle(rec_fields)
    rec_decl = "\n".join(f"    {n}: {t}," for n, t in rec_fields)
    return f"""#[derive(Clone, Copy)]
struct FuzzRec {{
{rec_decl}
}}

#[derive(Clone, Copy)]
struct NicheRec {{
    nz: NonZeroU64,
    tag: u8,
    flag: bool,
    val: i32,
}}

#[derive(Clone, Copy)]
enum FuzzEnum {{
    Empty,
    Byte(u8),
    Pair(i32, u64),
    Niche(NonZeroU32),
    Named {{ a: i128, b: f32, ok: bool }},
}}

#[derive(Clone, Copy)]
enum NicheEnum {{
    Idle,
    Active(NonZeroU64),
    Both {{ small: NonZeroU16, flag: bool, k: i32 }},
}}

#[inline(never)]
fn fuzz_rec_hash(r: FuzzRec) -> u64 {{
    let w = (r.wide as u64) ^ ((r.wide >> 64) as u64);
    (u64::from(r.tag).wrapping_mul({w0}))
        ^ ((r.small as i64 as u64).rotate_left(9))
        ^ w
        ^ safe_f64_bits(r.fp)
        ^ (u64::from(r.flag) * {w1})
}}

#[inline(never)]
fn eval_adt(seed: u64) -> u64 {{
    let mut acc = (size_of::<Option<NonZeroU64>>() as u64)
        .wrapping_mul(31)
        .wrapping_add(size_of::<Option<NicheRec>>() as u64)
        .wrapping_mul(31)
        .wrapping_add(size_of::<Option<NicheEnum>>() as u64)
        .wrapping_mul(31)
        .wrapping_add(size_of::<Result<NonZeroU16, u8>>() as u64);
    let mut s = seed;
    for round in 0..16u64 {{
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let opt_nz64 = NonZeroU64::new(if (s & 3) == 0 {{ 0 }} else {{ s | 1 }});
        let opt_nzi32 = NonZeroI32::new(if (s & 7) == 0 {{ 0 }} else {{ ((s >> 8) as i32) | 1 }});
        let opt_bool: Option<bool> = match (s >> 4) & 3 {{
            0 => None,
            1 => Some(false),
            _ => Some(true),
        }};
        let opt_nested: Option<Option<bool>> = match (s >> 6) & 3 {{
            0 => None,
            1 => Some(None),
            2 => Some(Some(false)),
            _ => Some(Some(true)),
        }};
        let ev = match round % 5 {{
            0 => FuzzEnum::Empty,
            1 => FuzzEnum::Byte(s as u8),
            2 => FuzzEnum::Pair((s >> 16) as i32, s.rotate_left(13)),
            3 => FuzzEnum::Niche(NonZeroU32::new(((s >> 24) as u32) | 1).unwrap()),
            _ => FuzzEnum::Named {{
                a: ((s as i128).wrapping_shl(63)) ^ (round as i128),
                b: if (s & 8) != 0 {{ -0.0f32 }} else {{ 1f32 }},
                ok: (s & 16) != 0,
            }},
        }};
        let opt_ev = if (s & 32) == 0 {{ None }} else {{ Some(ev) }};
        let ne = match (s >> 7) % 3 {{
            0 => NicheEnum::Idle,
            1 => NicheEnum::Active(NonZeroU64::new(s | 1).unwrap()),
            _ => NicheEnum::Both {{
                small: NonZeroU16::new(((s as u16) | 1)).unwrap(),
                flag: (s & 1) != 0,
                k: (s >> 12) as i32,
            }},
        }};
        let opt_ne = if (s & 64) == 0 {{ None }} else {{ Some(ne) }};
        let opt_nr = if (s & 128) == 0 {{
            None
        }} else {{
            Some(NicheRec {{
                nz: NonZeroU64::new(s | 1).unwrap(),
                tag: s as u8,
                flag: (s & 2) != 0,
                val: (s >> 20) as i32,
            }})
        }};
        let res_u16: Result<NonZeroU16, u8> = match NonZeroU16::new((s >> 3) as u16) {{
            Some(v) => Ok(v),
            None => Err(s as u8),
        }};
        let v_ev = match opt_ev {{
            None | Some(FuzzEnum::Empty) => {w2}u64,
            Some(FuzzEnum::Byte(b)) => u64::from(b).wrapping_add({w3}),
            Some(FuzzEnum::Pair(x, y)) if x < 0 => (x as i64 as u64) ^ y,
            Some(FuzzEnum::Pair(x, y)) => (x as i64 as u64).wrapping_add(y),
            Some(FuzzEnum::Niche(n)) => u64::from(n.get()).rotate_left(11),
            Some(FuzzEnum::Named {{ a, b, ok }}) => {{
                ((a as u64) ^ ((a >> 64) as u64)) ^ u64::from(safe_f32_bits(b)) ^ u64::from(ok)
            }}
        }};
        let v_ne = match opt_ne {{
            None => 11,
            Some(NicheEnum::Idle) => 22,
            Some(NicheEnum::Active(v)) => v.get(),
            Some(NicheEnum::Both {{ small, flag, k }}) => {{
                u64::from(small.get()).wrapping_mul(if flag {{ 3 }} else {{ 5 }}) ^ (k as i64 as u64)
            }}
        }};
        let v_nr = match opt_nr {{
            Some(nr) => nr.nz.get() ^ u64::from(nr.tag) ^ u64::from(nr.flag) ^ (nr.val as i64 as u64),
            None => {w4}u64,
        }};
        let v_opts = opt_nz64.map_or(7, |v| v.get())
            ^ opt_nzi32.map_or(9, |v| v.get() as i64 as u64)
            ^ match opt_bool {{ None => 1, Some(false) => 2, Some(true) => 3 }}
            ^ match opt_nested {{ None => 5, Some(None) => 13, Some(Some(false)) => 17, Some(Some(true)) => 19 }}
            ^ match res_u16 {{ Ok(v) => u64::from(v.get()), Err(e) => u64::from(e) << 16 }};
        acc = acc.rotate_left(7) ^ v_ev ^ v_ne ^ v_nr ^ v_opts;
    }}
    acc
}}
"""


def gen_slice_closure_section(rng: random.Random) -> str:
    arr_len = rng.choice([7, 9, 11, 13])
    rot = rng.randint(1, 29)
    return f"""#[inline(never)]
fn apply_ref<F: Fn(u64) -> u64>(f: &F, x: u64) -> u64 {{
    f(x) ^ f(x.rotate_left({rot}))
}}

#[inline(never)]
fn apply_mut<F: FnMut(u64) -> u64>(f: &mut F, n: usize, init: u64) -> u64 {{
    let mut acc = init;
    for i in 0..n {{
        acc = f(acc.wrapping_add(i as u64));
    }}
    acc
}}

#[inline(never)]
fn apply_dyn_ref(f: &dyn Fn(u64) -> u64, x: u64) -> u64 {{
    f(x)
}}

#[inline(never)]
fn apply_dyn_mut(f: &mut dyn FnMut(u64) -> u64, x: u64) -> u64 {{
    f(x)
}}

#[inline(never)]
fn slice_ops(src: &[u64], dst: &mut [u64], seed: u64) -> u64 {{
    if src.is_empty() || dst.is_empty() {{
        return seed;
    }}
    let a = (seed as usize) % src.len();
    let b = ((seed >> 8) as usize) % src.len();
    let (lo, hi) = if a <= b {{ (a, b + 1) }} else {{ (b, a + 1) }};
    let sub = &src[lo..hi];
    let mut acc = sub[0];
    for w in sub.windows(2) {{
        acc = acc.rotate_left(3).wrapping_add(w[0] ^ w[1]);
    }}
    for ch in sub.chunks(3) {{
        acc ^= ch.iter().rev().fold(0u64, |x, &y| x.wrapping_mul(31).wrapping_add(y));
    }}
    let got = sub.get(((seed >> 16) as usize) % (sub.len() + 2)).copied().unwrap_or(seed);
    acc ^= got;
    let mid = ((seed >> 20) as usize) % (dst.len() + 1);
    let (left, right) = dst.split_at_mut(mid);
    for (i, slot) in left.iter_mut().enumerate() {{
        *slot = slot.wrapping_add(sub[i % sub.len()]);
    }}
    for (i, slot) in right.iter_mut().enumerate() {{
        *slot = slot.rotate_left(5) ^ sub[(i + 1) % sub.len()];
    }}
    if dst.len() >= 2 {{
        let i0 = (seed as usize) % dst.len();
        let i1 = ((seed >> 5) as usize) % dst.len();
        dst.swap(i0, i1);
    }}
    acc ^ slice_hash(dst)
}}

#[inline(never)]
fn eval_slices_closures(seed: u64) -> u64 {{
    let mut arr = [0u64; {arr_len}];
    let mut s = seed;
    for slot in &mut arr {{
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *slot = s;
    }}
    let mut vec = arr.to_vec();
    let base_sum = slice_ops(&arr, &mut vec, s);
    let bias = s ^ 0x9e37_79b9_7f4a_7c15u64;
    let ref_a = |x: u64| -> u64 {{
        let idx = (x as usize) % arr.len();
        arr[idx].wrapping_add(bias) ^ x.rotate_left(7)
    }};
    let ref_b = |x: u64| -> u64 {{
        let idx = ((x >> 3) as usize) % vec.len();
        vec[idx].rotate_left(11) ^ ref_a(x ^ 17)
    }};
    let mut acc = base_sum
        ^ apply_ref(&ref_a, s)
        ^ apply_ref(&ref_b, s.rotate_left(9))
        ^ apply_dyn_ref(&ref_a, s ^ 0x55aa)
        ^ apply_dyn_ref(&ref_b, s ^ 0xaa55);
    {{
        let mut state = acc;
        let mut mut_c = |x: u64| -> u64 {{
            let slice_mut: &mut [u64] = &mut vec[..];
            let pos = (x as usize) % slice_mut.len();
            slice_mut[pos] = slice_mut[pos].wrapping_mul(33).wrapping_add(state ^ x);
            state = state.rotate_left(5) ^ slice_mut[pos];
            state
        }};
        acc ^= apply_mut(&mut mut_c, 6, s);
        acc ^= apply_dyn_mut(&mut mut_c, s ^ 0x1234);
        acc ^= state;
    }}
    acc ^ slice_hash(&vec)
}}
"""


def _spill_arg_expr(rng: random.Random, ty: str, idx: int, loop_mode: bool) -> str:
    if ty == "bool":
        return f"black_box(((s >> {idx % 32}) & 1) != 0)" if loop_mode else f"black_box({rng.choice(['true', 'false'])})"
    if ty == "f32":
        if loop_mode:
            return f"black_box((((s.rotate_left({idx * 5 % 64}) & 0xffff) as f32) * 0.125) - 1f32)"
        return f"black_box({rng.choice(['-0.0f32', '1f32', 'f32::MIN_POSITIVE', 'f32::INFINITY', 'f32::NEG_INFINITY'])})"
    if ty == "f64":
        if loop_mode:
            return f"black_box((((s.rotate_left({idx * 7 % 64}) & 0xffff_ffff) as f64) * 0.0625) + 100f64)"
        return f"black_box({rng.choice(['-0.0f64', '100f64', 'f64::from_bits(1)', 'f64::INFINITY', 'f64::NEG_INFINITY'])})"
    if ty in ("i128", "u128"):
        if loop_mode:
            return f"black_box(((s.rotate_left({idx * 9 % 64}) as {ty}).wrapping_shl(65)) ^ {lit(rng, ty)})"
        return f"black_box({lit(rng, ty)})"
    if ty == "FuzzRec":
        return (
            "black_box(FuzzRec { tag: s as u8, small: (s >> 8) as i16, "
            "wide: (s as i128).wrapping_shl(64), fp: -0.0f64, flag: (s & 1) != 0 })"
        )
    if ty == "NonZeroU64":
        return "black_box(NonZeroU64::new(s | 1).unwrap())"
    if ty == "&[u64]":
        return "black_box(&buf[..])"
    if loop_mode:
        return f"black_box((s.rotate_left({idx * 11 % 64}) as {ty}))"
    return f"black_box({lit(rng, ty)})"


def _spill_fold_term(name: str, ty: str) -> str:
    if ty == "bool":
        return f"u64::from({name})"
    if ty == "f32":
        return f"u64::from(safe_f32_bits({name}))"
    if ty == "f64":
        return f"safe_f64_bits({name})"
    if ty in ("i128", "u128"):
        return f"(({name} as u64) ^ (({name} >> 64) as u64))"
    if ty == "FuzzRec":
        return f"fuzz_rec_hash({name})"
    if ty == "NonZeroU64":
        return f"{name}.get()"
    if ty == "&[u64]":
        return f"slice_hash({name})"
    return f"({name} as u64)"


def gen_stack_spill_section(rng: random.Random) -> str:
    int_pool = ["u8", "i16", "u32", "i64", "u128", "i8", "u16", "i32", "i128", "usize", "isize", "bool"]
    int_params = [rng.choice(int_pool) for _ in range(rng.randint(12, 15))]
    flt_params = [rng.choice(["f32", "f64"]) for _ in range(rng.randint(12, 14))]
    mix_pool = ["i64", "f64", "u8", "f32", "u128", "bool", "i16", "f64", "i32", "f32", "NonZeroU64", "FuzzRec", "i8", "&[u64]"]
    mix_params = list(mix_pool)
    rng.shuffle(mix_params)

    def make_spill_fn(fn_name: str, ptypes: list[str], ret_tuple: bool = False) -> str:
        sig = ", ".join(f"a{i}: {t}" for i, t in enumerate(ptypes))
        lines = ["    let mut acc = 0x517c_c1b7_2722_0a95u64;"]
        for i, t in enumerate(ptypes):
            w = rng.randrange(3, 1000) | 1
            term = _spill_fold_term(f"a{i}", t)
            lines.append(f"    acc = acc.rotate_left({(i * 5 + 1) % 63 + 1}).wrapping_add(({term}).wrapping_mul({w}u64));")
        if ret_tuple:
            f64_names = [f"a{i}" for i, t in enumerate(ptypes) if t == "f64"]
            f32_names = [f"a{i}" for i, t in enumerate(ptypes) if t == "f32"]
            fsum = " + ".join(f64_names[:2] + [f"({n} as f64)" for n in f32_names[:2]]) or "100f64"
            lines.append(f"    (acc, {fsum})")
            ret = "(u64, f64)"
        else:
            lines.append("    acc")
            ret = "u64"
        return f"#[inline(never)]\nfn {fn_name}({sig}) -> {ret} {{\n" + "\n".join(lines) + "\n}\n"

    fn_ints = make_spill_fn("spill_ints", int_params)
    fn_flts = make_spill_fn("spill_floats", flt_params)
    fn_mix = make_spill_fn("spill_mixed", mix_params, ret_tuple=True)

    loop_ints = ", ".join(_spill_arg_expr(rng, t, i, True) for i, t in enumerate(int_params))
    edge_ints = ", ".join(_spill_arg_expr(rng, t, i, False) for i, t in enumerate(int_params))
    loop_flts = ", ".join(_spill_arg_expr(rng, t, i, True) for i, t in enumerate(flt_params))
    edge_flts = ", ".join(_spill_arg_expr(rng, t, i, False) for i, t in enumerate(flt_params))
    loop_mix = ", ".join(_spill_arg_expr(rng, t, i, True) for i, t in enumerate(mix_params))
    edge_mix = ", ".join(_spill_arg_expr(rng, t, i, False) for i, t in enumerate(mix_params))

    return f"""{fn_ints}
{fn_flts}
{fn_mix}
#[inline(never)]
fn eval_stack_spills(seed: u64) -> u64 {{
    let buf = [seed, seed ^ 1, seed.rotate_left(13), seed.wrapping_mul(31)];
    let mut s = seed;
    let mut acc = 0u64;
    for _ in 0..8 {{
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        acc ^= spill_ints({loop_ints});
        acc = acc.rotate_left(3) ^ spill_floats({loop_flts});
        let (mu, mf) = spill_mixed({loop_mix});
        acc = acc.wrapping_add(mu ^ safe_f64_bits(mf));
    }}
    let (eu, ef) = spill_mixed({edge_mix});
    acc ^ spill_ints({edge_ints}) ^ spill_floats({edge_flts}) ^ eu ^ safe_f64_bits(ef)
}}
"""


def gen_i128_section(rng: random.Random) -> str:
    salt = rng.randrange(1, 1 << 63) | 1
    return f"""#[inline(never)]
fn eval_wide128(seed: u64) -> u64 {{
    let mut acc = 0u64;
    let s_hi = (seed as i128).wrapping_shl(64) | (seed.rotate_left(19) as i128);
    let ivals: [i128; 10] = [
        i128::MIN,
        i128::MIN + 1,
        -(1i128 << 64),
        -(1i128 << 63),
        -1i128,
        0i128,
        1i128,
        (1i128 << 64) | (seed as i128),
        s_hi,
        i128::MAX,
    ];
    let uvals: [u128; 8] = [
        0u128,
        1u128,
        u64::MAX as u128,
        1u128 << 64,
        (1u128 << 64) | u128::from(seed),
        1u128 << 127,
        u128::MAX - 1,
        u128::MAX,
    ];
    let shifts: [u32; 12] = [0, 1, 7, 31, 63, 64, 65, 95, 127, 128, 129, 255];
    for (idx, &iv) in ivals.iter().enumerate() {{
        let x = black_box(iv);
        let y = black_box(ivals[(idx + 3) % ivals.len()]);
        let sh = black_box(shifts[idx % shifts.len()]);
        let a = x.wrapping_add(y);
        let b = x.wrapping_sub(y);
        let c = x.wrapping_mul(y);
        let (oa, f1) = x.overflowing_add(y);
        let (os, f2) = x.overflowing_sub(y);
        let (om, f3) = x.overflowing_mul(3);
        let sat = x.saturating_add(y) ^ x.saturating_sub(y) ^ x.saturating_mul(-2);
        let div = x.checked_div(y).unwrap_or(x);
        let rem = x.checked_rem(y).unwrap_or(y);
        let sl = x.wrapping_shl(sh);
        let sr = x.wrapping_shr(sh);
        let csl = x.checked_shl(sh).unwrap_or(-1);
        let csr = x.checked_shr(sh).unwrap_or(-1);
        let rot = x.rotate_left(sh) ^ x.rotate_right(sh);
        let bits = u64::from(x.count_ones())
            | (u64::from(x.leading_zeros()) << 8)
            | (u64::from(x.trailing_zeros()) << 16);
        let cmp_mask = u64::from(x < y)
            | (u64::from(x <= y) << 1)
            | (u64::from(x > y) << 2)
            | (u64::from(x >= y) << 3)
            | (u64::from(x == y) << 4)
            | (u64::from(x != y) << 5)
            | (u64::from(f1) << 6)
            | (u64::from(f2) << 7)
            | (u64::from(f3) << 8);
        let folded = a ^ b ^ c ^ oa ^ os ^ om ^ sat ^ div ^ rem ^ sl ^ sr ^ csl ^ csr ^ rot ^ x.swap_bytes() ^ x.reverse_bits();
        let mn_mx = x.min(y) ^ x.max(y);
        acc = acc.rotate_left(5) ^ ((folded as u64) ^ ((folded >> 64) as u64)) ^ ((mn_mx as u64) ^ ((mn_mx >> 64) as u64)) ^ bits ^ cmp_mask;
    }}
    for (idx, &uv) in uvals.iter().enumerate() {{
        let u = black_box(uv);
        let v = black_box(uvals[(idx + 5) % uvals.len()]);
        let sh = black_box(shifts[(idx + 4) % shifts.len()]);
        let same_lo = black_box((1u128 << 64) | (seed as u128));
        let same_lo_2 = black_box((2u128 << 64) | (seed as u128));
        let cmp_hi = u64::from(same_lo < same_lo_2) | (u64::from(same_lo == same_lo_2) << 1) | (u64::from(u < v) << 2) | (u64::from(u >= v) << 3);
        let uw = u.wrapping_shl(sh) ^ u.wrapping_shr(sh) ^ u.wrapping_mul(v) ^ u.checked_div(v).unwrap_or({salt}u128) ^ u.checked_rem(v).unwrap_or(0);
        acc = acc.rotate_left(7) ^ ((uw as u64) ^ ((uw >> 64) as u64)) ^ cmp_hi;
    }}
    acc
}}
"""


def gen_float_section(rng: random.Random) -> str:
    extra = rng.choice([0.25, 0.5, 1.5, -0.125])
    return f"""#[inline(never)]
fn eval_float_edges(seed: u64) -> u64 {{
    let mut acc = (-0.0f64).to_bits()
        ^ 0.0f64.to_bits()
        ^ u64::from((1f32).to_bits())
        ^ (100f64).to_bits()
        ^ u64::from((-0.0f32).to_bits())
        ^ (1.0f64.copysign(-0.0f64)).to_bits();
    let f64s: [f64; 11] = [
        -0.0f64,
        0.0f64,
        1f32 as f64,
        100f64,
        -100f64,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        f64::MIN,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let f32s: [f32; 11] = [
        -0.0f32,
        0.0f32,
        1f32,
        -1f32,
        100f64 as f32,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MIN,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let bias = ((seed & 0xff) as f64) + {extra};
    for (i, &raw) in f64s.iter().enumerate() {{
        let v = black_box(raw);
        let w = black_box(f64s[(i + 3) % f64s.len()]);
        let ops = [
            v.abs(),
            -v,
            v.copysign(-1.0),
            v.floor(),
            v.ceil(),
            v.trunc(),
            v + bias,
            v - w,
            v * 1.5f64,
            v / w,
        ];
        for &r in &ops {{
            acc = acc.rotate_left(3) ^ safe_f64_bits(r);
        }}
        let casts = u64::from(safe_f32_bits(v as f32))
            ^ ((v as i64) as u64)
            ^ (v as u64)
            ^ ((v as i128) as u64)
            ^ ((v as u128) as u64)
            ^ (((v as i128) >> 64) as u64);
        let flags = u64::from(v == 0.0)
            | (u64::from(v.is_sign_negative()) << 1)
            | (u64::from(v.is_infinite()) << 2)
            | (u64::from(v.is_nan()) << 3);
        acc ^= casts ^ flags ^ nan_safe_cmp_f64(v, w) ^ nan_safe_cmp_f64(v / w, f64::NAN);
    }}
    for (i, &raw32) in f32s.iter().enumerate() {{
        let v = black_box(raw32);
        let w = black_box(f32s[(i + 5) % f32s.len()]);
        let ops = [v.abs(), -v, v.copysign(-1f32), v.floor(), v.ceil(), v.trunc(), v + 1f32, v / w];
        for &r in &ops {{
            acc = acc.rotate_left(5) ^ u64::from(safe_f32_bits(r));
        }}
        let casts = safe_f64_bits(v as f64) ^ ((v as i32) as i64 as u64) ^ u64::from(v as u32);
        acc ^= casts ^ nan_safe_cmp_f32(v, w) ^ nan_safe_cmp_f32(0.0f32 / 0.0f32, f32::NAN);
    }}
    let canon: [f64; 9] = [
        f64::from_bits(0xfff8_0000_0000_0001),
        f64::NEG_INFINITY,
        -100f64,
        -0.0f64,
        0.0f64,
        1f32 as f64,
        100f64,
        f64::INFINITY,
        f64::from_bits(0x7ff8_0000_0000_0001),
    ];
    for i in 0..canon.len() {{
        let a = black_box(canon[i]);
        let b = black_box(canon[(i + 1) % canon.len()]);
        let ord_tag = match a.total_cmp(&b) {{
            std::cmp::Ordering::Less => 1u64,
            std::cmp::Ordering::Equal => 2,
            std::cmp::Ordering::Greater => 3,
        }};
        acc = acc.rotate_left(2).wrapping_add(ord_tag ^ a.to_bits());
    }}
    acc
}}
"""


def gen_drop_unwind_section(rng: random.Random) -> str:
    base_id = rng.randrange(1, 50) * 10
    p_mask = rng.randrange(1, 16)
    return f"""struct DropLeaf<'a> {{
    id: u32,
    log: &'a Mutex<Vec<u32>>,
    hash: &'a AtomicU64,
}}

impl Drop for DropLeaf<'_> {{
    fn drop(&mut self) {{
        let prev = self.hash.load(AtomOrd::Relaxed);
        self.hash.store(
            prev.wrapping_mul(131).wrapping_add(u64::from(self.id) + 1),
            AtomOrd::Relaxed,
        );
        if let Ok(mut vec) = self.log.lock() {{
            vec.push(self.id);
        }}
    }}
}}

struct DropPair<'a>(DropLeaf<'a>, DropLeaf<'a>);

struct DropBox<'a> {{
    tag: u32,
    head: DropLeaf<'a>,
    arr: [DropLeaf<'a>; 2],
    pair: (DropLeaf<'a>, Option<DropLeaf<'a>>),
    log: &'a Mutex<Vec<u32>>,
    hash: &'a AtomicU64,
}}

impl Drop for DropBox<'_> {{
    fn drop(&mut self) {{
        let prev = self.hash.load(AtomOrd::Relaxed);
        self.hash.store(
            prev.wrapping_mul(131).wrapping_add(u64::from(self.tag) + 1),
            AtomOrd::Relaxed,
        );
        if let Ok(mut vec) = self.log.lock() {{
            vec.push(self.tag);
        }}
    }}
}}

#[inline(never)]
fn mk_drop<'a>(id: u32, log: &'a Mutex<Vec<u32>>, hash: &'a AtomicU64) -> DropLeaf<'a> {{
    DropLeaf {{ id, log, hash }}
}}

#[inline(never)]
fn eval_drop_unwind(seed: u64) -> (u64, u64, Vec<u32>) {{
    let log = Mutex::new(Vec::<u32>::new());
    let hash = AtomicU64::new(seed);
    let mask = black_box({p_mask}u32 ^ ((seed as u32) & 7));
    let mut caught = 0u64;

    let r0 = catch_unwind(AssertUnwindSafe(|| {{
        let mut x = mk_drop({base_id + 1}, &log, &hash);
        x = mk_drop({base_id + 2}, &log, &hash);
        let _nested = (
            mk_drop({base_id + 3}, &log, &hash),
            [mk_drop({base_id + 4}, &log, &hash), mk_drop({base_id + 5}, &log, &hash)],
            DropBox {{
                tag: {base_id + 6},
                head: mk_drop({base_id + 7}, &log, &hash),
                arr: [mk_drop({base_id + 8}, &log, &hash), mk_drop({base_id + 9}, &log, &hash)],
                pair: (mk_drop({base_id + 10}, &log, &hash), Some(mk_drop({base_id + 11}, &log, &hash))),
                log: &log,
                hash: &hash,
            }},
        );
        drop(x);
    }}));
    caught |= u64::from(r0.is_err());

    for phase in 0..2u32 {{
        let r1 = catch_unwind(AssertUnwindSafe(|| {{
            let _first = mk_drop({base_id + 20}, &log, &hash);
            let _tup = (
                DropPair(mk_drop({base_id + 21}, &log, &hash), mk_drop({base_id + 22}, &log, &hash)),
                [mk_drop({base_id + 23}, &log, &hash), mk_drop({base_id + 24}, &log, &hash), mk_drop({base_id + 25}, &log, &hash)],
            );
            if phase == 1 || (mask & 1) != 0 {{
                panic!("fuzz_unwind_1");
            }}
            let _tail = mk_drop({base_id + 26}, &log, &hash);
        }}));
        caught |= u64::from(r1.is_err()) << (1 + phase * 2);

        let r2 = catch_unwind(AssertUnwindSafe(|| {{
            let _outer = DropBox {{
                tag: {base_id + 30},
                head: mk_drop({base_id + 31}, &log, &hash),
                arr: [mk_drop({base_id + 32}, &log, &hash), mk_drop({base_id + 33}, &log, &hash)],
                pair: (
                    mk_drop({base_id + 34}, &log, &hash),
                    if ((mask ^ phase) & 2) != 0 {{ Some(mk_drop({base_id + 35}, &log, &hash)) }} else {{ None }},
                ),
                log: &log,
                hash: &hash,
            }};
            let inner = catch_unwind(AssertUnwindSafe(|| {{
                let _v = vec![mk_drop({base_id + 36}, &log, &hash), mk_drop({base_id + 37}, &log, &hash)];
                if phase == 1 || (mask & 4) != 0 {{
                    panic!("fuzz_unwind_inner");
                }}
            }}));
            if inner.is_err() || (mask & 8) != 0 {{
                panic!("fuzz_unwind_outer");
            }}
        }}));
        caught |= u64::from(r2.is_err()) << (2 + phase * 2);
    }}

    let seq = log.lock().map(|g| g.clone()).unwrap_or_default();
    (caught, hash.load(AtomOrd::Relaxed), seq)
}}
"""


def make_program(seed: int, func_count: int = 12) -> str:
    rng = random.Random(seed)
    out = [preamble()]
    calls = []
    for idx in range(func_count):
        params, src = make_func(rng, idx)
        out.append(src)
        args = ", ".join(
            f"black_box(s.wrapping_mul({rng.randrange(1, 1 << 62) | 1}u64).rotate_left({rng.randrange(64)}) as {ty})"
            for _, ty in params
        )
        edge = ", ".join(f"black_box({lit(rng, ty)})" for _, ty in params)
        calls.append(
            f"    {{ let mut h = 0u64; let mut s = {rng.randrange(1 << 63)}u64; "
            f"for _ in 0..32 {{ s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); "
            f"h = h.rotate_left(5) ^ f{idx}({args}); }} "
            f'h ^= f{idx}({edge}); println!("f{idx} {{:016x}}", h); }}'
        )

    out.append(gen_adt_section(rng))
    out.append(gen_slice_closure_section(rng))
    out.append(gen_stack_spill_section(rng))
    out.append(gen_i128_section(rng))
    out.append(gen_float_section(rng))
    out.append(gen_drop_unwind_section(rng))

    s_init = rng.randrange(1, 1 << 63) | 1
    calls.append(f'    println!("adt {{:016x}}", eval_adt(black_box({s_init}u64)));')
    calls.append(f'    println!("slice_closure {{:016x}}", eval_slices_closures(black_box({s_init ^ 0x13579}u64)));')
    calls.append(f'    println!("stack_spill {{:016x}}", eval_stack_spills(black_box({s_init ^ 0x2468a}u64)));')
    calls.append(f'    println!("wide128 {{:016x}}", eval_wide128(black_box({s_init ^ 0xabcdef}u64)));')
    calls.append(f'    println!("float_edges {{:016x}}", eval_float_edges(black_box({s_init ^ 0x7777}u64)));')
    calls.append(
        f"    {{ let prev_hook = std::panic::take_hook(); std::panic::set_hook(Box::new(|_| {{}})); "
        f"let (caught, dh, seq) = eval_drop_unwind(black_box({s_init}u64)); "
        f"std::panic::set_hook(prev_hook); "
        f'println!("drop_unwind {{caught:x}} {{dh:016x}} {{seq:?}}"); }}'
    )

    out.append("fn main() {\n" + "\n".join(calls) + "\n}")
    return "\n".join(out) + "\n"


def main() -> None:
    if len(sys.argv) == 2:
        if sys.argv[1] in ("-h", "--help"):
            print("usage: fuzz_gen.py SEED | START END OUT_DIR")
            return
        print(make_program(int(sys.argv[1])), end="")
        return
    if len(sys.argv) != 4:
        sys.exit("usage: fuzz_gen.py SEED | START END OUT_DIR")
    start, end, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    for seed in range(start, end):
        path = os.path.join(out_dir, f"z{seed:03d}.rs")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(make_program(seed))


if __name__ == "__main__":
    main()
