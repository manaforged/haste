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


def make_func(rng: random.Random, idx: int) -> tuple[list[tuple[str, str]], str]:
    params = [(f"p{j}", rng.choice(INTS)) for j in range(3)]
    vars_list = list(params)
    body: list[str] = []
    for step in range(rng.randint(3, 7)):
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
    fold = " ^ ".join(
        f"({name} as u64).rotate_left({j * 7 % 64})" for j, (name, _) in enumerate(vars_list)
    )
    attr = "#[inline(never)]\n" if rng.random() < 0.5 else ""
    sig = ", ".join(f"{name}: {ty}" for name, ty in params)
    src = f"{attr}fn f{idx}({sig}) -> u64 {{\n" + "\n".join(body) + f"\n    {fold}\n}}\n"
    return params, src


def make_program(seed: int, func_count: int = 20) -> str:
    rng = random.Random(seed)
    out = [
        "#![allow(unused, arithmetic_overflow, overflowing_literals, unconditional_panic, clippy::all)]",
        "use std::hint::black_box;",
    ]
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
            f"for _ in 0..64 {{ s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); "
            f"h = h.rotate_left(5) ^ f{idx}({args}); }} "
            f'h ^= f{idx}({edge}); println!("f{idx} {{:016x}}", h); }}'
        )
    out.append("fn main() {\n" + "\n".join(calls) + "\n}")
    return "\n".join(out) + "\n"


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("usage: fuzz_gen.py START END OUT_DIR")
    start, end, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    for seed in range(start, end):
        path = os.path.join(out_dir, f"z{seed:03d}.rs")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(make_program(seed))


if __name__ == "__main__":
    main()
