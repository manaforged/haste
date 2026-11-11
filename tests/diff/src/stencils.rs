use std::any::{TypeId, type_name};
use std::fmt;
use std::num::{NonZeroI32, NonZeroU64};

#[derive(Clone, Copy)]
struct MixTriple<T> {
    a: u8,
    mid: T,
    b: u16,
}

trait TagOp: Copy + 'static {
    fn raw(self) -> i64;
    fn wrap(v: i64) -> Self;
    fn step(self, k: i64) -> i64;
}

macro_rules! def_tag {
    ($name:ident, $bias:expr) => {
        #[derive(Clone, Copy)]
        struct $name(i64);

        impl TagOp for $name {
            fn raw(self) -> i64 {
                self.0
            }

            fn wrap(v: i64) -> Self {
                Self(v)
            }

            fn step(self, k: i64) -> i64 {
                let a = self.0.wrapping_mul(3).wrapping_add($bias);
                let b = k.wrapping_mul(5).wrapping_sub($bias);
                let c = (a ^ b).wrapping_add(self.0 & 7);
                let d = c.rotate_left(3).wrapping_sub(k);
                let e = d.wrapping_mul(11).wrapping_add($bias * 13);
                (e ^ (e >> 17)).wrapping_add(self.0).wrapping_sub($bias)
            }
        }
    };
}

def_tag!(Tag0, 3);
def_tag!(Tag1, 5);
def_tag!(Tag2, 7);
def_tag!(Tag3, 11);
def_tag!(Tag4, 13);
def_tag!(Tag5, 17);
def_tag!(Tag6, 19);
def_tag!(Tag7, 23);
def_tag!(Tag8, 29);
def_tag!(Tag9, 31);

fn opt_score<T>(opt: Option<T>, fallback: i64, read: fn(T) -> i64) -> i64 {
    match opt.map(read) {
        Some(v) => v.wrapping_mul(3).wrapping_add(1),
        None => fallback.wrapping_sub(9),
    }
}

pub fn case_stencil_niche(n: i64) -> i64 {
    let nz64 = NonZeroU64::new((n.unsigned_abs() + 1) * 7);
    let plain64 = if n >= 0 { Some(n as u64 + 10) } else { None };
    let nz32 = NonZeroI32::new(if n == 0 { 5 } else { n as i32 });
    let plain32 = if n == 0 { None } else { Some(n as i32 * 2) };
    let ri = n + 4;
    let ru = (n.unsigned_abs() + 9) as u64;
    let opt_ri: Option<&i64> = if n >= 0 { Some(&ri) } else { None };
    let opt_ru: Option<&u64> = if n >= 0 { Some(&ru) } else { None };
    let a = opt_score(nz64, n, |v| v.get() as i64);
    let b = opt_score(plain64, n, |v| v as i64);
    let c = opt_score(nz32, n, |v| v.get() as i64);
    let d = opt_score(plain32, n, |v| v as i64);
    let e = opt_score(opt_ri, n, |v| *v);
    let f = opt_score(opt_ru, n, |v| *v as i64);
    a.wrapping_add(b).wrapping_add(c).wrapping_add(d).wrapping_add(e).wrapping_add(f)
}

fn mix_eval<T: Copy>(m: MixTriple<T>, t: (u8, T, u16), f: fn(T) -> i64) -> i64 {
    let left = (m.a as i64) * 100 + f(m.mid) * 10 + (m.b as i64);
    let right = (t.0 as i64) * 1000 + f(t.1) * 3 + (t.2 as i64);
    left.wrapping_sub(right)
}

pub fn case_stencil_reorder(n: i64) -> i64 {
    let u = n.unsigned_abs();
    let r1 = n + 11;
    let r2 = u + 22;
    let a = mix_eval(MixTriple { a: 2, mid: (u & 0xff) as u8, b: 300 }, (5, 7_u8, 9), |x| x as i64);
    let b = mix_eval(MixTriple { a: 3, mid: (u & 0xffff) as u16, b: 400 }, (6, 11_u16, 13), |x| x as i64);
    let c = mix_eval(MixTriple { a: 4, mid: u as u32, b: 500 }, (7, 15_u32, 17), |x| x as i64);
    let d = mix_eval(MixTriple { a: 5, mid: u, b: 600 }, (8, 19_u64, 21), |x| x as i64);
    let e = mix_eval(MixTriple { a: 6, mid: (u as u32, 9_u32), b: 700 }, (9, (3_u32, 4_u32), 23), |p| (p.0 + p.1) as i64);
    let f = mix_eval(MixTriple { a: 7, mid: &r1, b: 800 }, (1, &r1, 25), |p| *p);
    let g = mix_eval(MixTriple { a: 8, mid: &r2, b: 900 }, (2, &r2, 27), |p| *p as i64);
    a.wrapping_add(b).wrapping_add(c).wrapping_add(d).wrapping_add(e).wrapping_add(f).wrapping_add(g)
}

struct UnitIter(usize);

impl Iterator for UnitIter {
    type Item = ();
    fn next(&mut self) -> Option<()> {
        if self.0 == 0 {
            None
        } else {
            self.0 -= 1;
            Some(())
        }
    }
}

struct QuadIter {
    cur: i64,
    rem: usize,
}

impl Iterator for QuadIter {
    type Item = [i64; 4];
    fn next(&mut self) -> Option<[i64; 4]> {
        if self.rem == 0 {
            None
        } else {
            self.rem -= 1;
            let v = self.cur;
            self.cur += 1;
            Some([v, v * 2, v - 3, v ^ 5])
        }
    }
}

fn drain_score<I: Iterator>(mut it: I, weight: fn(I::Item) -> i64) -> i64 {
    let mut acc = 0_i64;
    while let Some(item) = it.next() {
        acc = acc.wrapping_mul(5).wrapping_add(weight(item)).wrapping_add(1);
    }
    acc
}

pub fn case_stencil_assoc_abi(n: i64) -> i64 {
    let count = (n.unsigned_abs() as usize % 5) + 1;
    let zst = drain_score(UnitIter(count), |()| 13);
    let scalar = drain_score([n, n + 2, n - 1].into_iter(), |x| x);
    let pair = drain_score([(n, 1_i64), (n + 3, 4_i64)].into_iter(), |(a, b)| a * b);
    let sret = drain_score(QuadIter { cur: n, rem: count }, |q| q[0] + q[1] + q[2] + q[3]);
    zst.wrapping_add(scalar).wrapping_add(pair).wrapping_add(sret)
}

fn heavy_eval<T: TagOp>(v: T, n: i64) -> i64 {
    let mut acc = T::step(v, n);
    acc = acc.wrapping_add(T::step(T::wrap(acc), n.wrapping_add(1)));
    acc ^= T::step(T::wrap(acc), n.wrapping_add(2));
    acc = acc.wrapping_add(T::step(T::wrap(acc), n.wrapping_add(3)));
    acc = acc.wrapping_sub(T::step(T::wrap(acc), n.wrapping_add(4)));
    acc = acc.wrapping_add(T::step(T::wrap(acc), n.wrapping_add(5)));
    acc ^= T::step(T::wrap(acc), n.wrapping_add(6));
    acc = acc.wrapping_add(T::step(T::wrap(acc), n.wrapping_add(7)));
    acc = acc.wrapping_sub(T::step(T::wrap(acc), n.wrapping_add(8)));
    acc = acc.wrapping_add(T::step(T::wrap(acc), n.wrapping_add(9)));
    acc ^= T::step(T::wrap(acc), n.wrapping_add(10));
    acc = acc.wrapping_add(T::step(T::wrap(acc), n.wrapping_add(11)));
    let same = (TypeId::of::<T>() == TypeId::of::<Tag0>()) as i64;
    acc.wrapping_add(same).wrapping_add(v.raw())
}

fn inline_chain<T: Copy>(mut slot: T, next: T, ok: bool, proj: fn(T) -> i64) -> i64 {
    let old = std::mem::replace(&mut slot, next);
    let res: Result<T, i64> = if ok { Ok(slot) } else { Err(-7) };
    let mapped = Some(old).and_then(|v| Some(proj(v) + 3)).unwrap_or(0);
    res.map_or(19, proj).wrapping_add(mapped)
}

pub fn case_stencil_inline_split(n: i64) -> i64 {
    let a = n + 3;
    let b = n + 9;
    let ua = n.unsigned_abs() + 2;
    let ub = n.unsigned_abs() + 8;
    let p1 = inline_chain(&a, &b, n >= 0, |r| *r);
    let p2 = inline_chain(&ua, &ub, n != 0, |r| *r as i64);
    let p3 = inline_chain(NonZeroU64::new(ua), NonZeroU64::new(ub), n > 0, |o| o.map_or(0, |v| v.get() as i64));
    let p4 = inline_chain(Some(ua), None, n <= 0, |o| o.unwrap_or(5) as i64);
    let h = heavy_eval(Tag0(n), n)
        .wrapping_add(heavy_eval(Tag1(n), n))
        .wrapping_add(heavy_eval(Tag2(n), n))
        .wrapping_add(heavy_eval(Tag3(n), n))
        .wrapping_add(heavy_eval(Tag4(n), n))
        .wrapping_add(heavy_eval(Tag5(n), n))
        .wrapping_add(heavy_eval(Tag6(n), n))
        .wrapping_add(heavy_eval(Tag7(n), n))
        .wrapping_add(heavy_eval(Tag8(n), n))
        .wrapping_add(heavy_eval(Tag9(n), n));
    p1.wrapping_add(p2).wrapping_add(p3).wrapping_add(p4).wrapping_add(h)
}

struct FmtBox(i64);

impl fmt::Display for FmtBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "box:{}", self.0)
    }
}

fn render_sig<T: ToString + ?Sized>(val: &T) -> i64 {
    let text = val.to_string();
    let bytes = text.as_bytes();
    let first = bytes.first().copied().unwrap_or(0) as i64;
    let last = bytes.last().copied().unwrap_or(0) as i64;
    (bytes.len() as i64) * 257 + first * 17 + last
}

pub fn case_stencil_specialize(n: i64) -> i64 {
    let owned = format!("s:{n}");
    let u = n.unsigned_abs();
    let a = render_sig("hello-stencil");
    let b = render_sig(&owned);
    let c = render_sig(&n);
    let d = render_sig(&u);
    let e = render_sig(&FmtBox(n * 3));
    a.wrapping_add(b).wrapping_add(c).wrapping_add(d).wrapping_add(e)
}

fn info_sig<A: 'static, B: 'static>() -> i64 {
    let same = (TypeId::of::<A>() == TypeId::of::<B>()) as i64;
    let na = type_name::<A>().as_bytes();
    let nb = type_name::<B>().as_bytes();
    let ha = (na.len() as i64) * 31 + na.first().copied().unwrap_or(0) as i64 + na.last().copied().unwrap_or(0) as i64;
    let hb = (nb.len() as i64) * 17 + nb.first().copied().unwrap_or(0) as i64 + nb.last().copied().unwrap_or(0) as i64;
    same * 10_000 + ha - hb
}

pub fn case_stencil_type_info(n: i64) -> i64 {
    let a = info_sig::<i8, i8>();
    let b = info_sig::<i8, u8>();
    let c = info_sig::<u64, usize>();
    let d = info_sig::<&i64, &mut u64>();
    let e = info_sig::<(i32, i64), &(u8, u16, &[u32], &str)>();
    a.wrapping_add(b).wrapping_add(c).wrapping_add(d).wrapping_add(e).wrapping_add(n)
}

fn probe_frame<T: TagOp>(v: T, n: i64) -> i64 {
    let mut buf = [0_u8; 8192];
    let seed = T::step(v, n);
    buf[0] = (seed & 0xff) as u8;
    buf[4095] = ((seed >> 8) & 0xff) as u8;
    buf[4096] = ((seed >> 16) & 0xff) as u8;
    buf[8191] = ((seed >> 24) & 0xff) as u8;
    let mid = T::step(T::wrap(seed.wrapping_add(buf[0] as i64)), buf[8191] as i64);
    buf[1024] = (mid & 0xff) as u8;
    buf[7000] = ((mid >> 5) & 0xff) as u8;
    let end = T::step(T::wrap(mid), (buf[1024] as i64) + (buf[7000] as i64));
    end.wrapping_add(buf[0] as i64)
        .wrapping_add(buf[4095] as i64)
        .wrapping_add(buf[4096] as i64)
        .wrapping_add(buf[8191] as i64)
}

pub fn case_stencil_stack_probe(n: i64) -> i64 {
    probe_frame(Tag0(n), n)
        .wrapping_add(probe_frame(Tag1(n), n))
        .wrapping_add(probe_frame(Tag2(n), n))
        .wrapping_add(probe_frame(Tag3(n), n))
        .wrapping_add(probe_frame(Tag4(n), n))
        .wrapping_add(probe_frame(Tag5(n), n))
        .wrapping_add(probe_frame(Tag6(n), n))
        .wrapping_add(probe_frame(Tag7(n), n))
        .wrapping_add(probe_frame(Tag8(n), n))
        .wrapping_add(probe_frame(Tag9(n), n))
}

fn wide_mix<T: TagOp>(v: T, n: i64) -> i64 {
    let s1 = T::step(v, n) as i128;
    let wide = (s1 << 65) | ((n.unsigned_abs() as i128) + 17);
    let div = ((T::step(v, n + 3).abs() % 97) + 3) as i128;
    let q = wide / div;
    let r = wide % div;
    let u = (q.unsigned_abs() >> 19) ^ (r.unsigned_abs() << 7);
    let back = T::step(T::wrap((u & 0xffff_ffff) as i64), (q & 0xff) as i64);
    ((q ^ r) as i64).wrapping_add(back)
}

pub fn case_stencil_i128(n: i64) -> i64 {
    wide_mix(Tag0(n), n)
        .wrapping_add(wide_mix(Tag1(n), n))
        .wrapping_add(wide_mix(Tag2(n), n))
        .wrapping_add(wide_mix(Tag3(n), n))
        .wrapping_add(wide_mix(Tag4(n), n))
        .wrapping_add(wide_mix(Tag5(n), n))
        .wrapping_add(wide_mix(Tag6(n), n))
        .wrapping_add(wide_mix(Tag7(n), n))
        .wrapping_add(wide_mix(Tag8(n), n))
        .wrapping_add(wide_mix(Tag9(n), n))
}

trait BaseScore {
    fn base(&self, n: i64) -> i64;
}

trait FullScore: BaseScore {
    fn full(&self, n: i64) -> i64;
}

impl<T: TagOp> BaseScore for T {
    fn base(&self, n: i64) -> i64 {
        T::step(*self, n)
    }
}

impl<T: TagOp> FullScore for T {
    fn full(&self, n: i64) -> i64 {
        heavy_eval(*self, n)
    }
}

fn call_reified<T: Copy>(f: fn(T, i64) -> i64, v: T, n: i64) -> i64 {
    f(v, n)
}

fn call_dyn(obj: &dyn FullScore, n: i64) -> i64 {
    let up: &dyn BaseScore = obj;
    obj.full(n).wrapping_add(up.base(n + 1))
}

pub fn case_stencil_reify_vtable(n: i64) -> i64 {
    let reified = call_reified(heavy_eval::<Tag0>, Tag0(n), n)
        .wrapping_add(call_reified(heavy_eval::<Tag1>, Tag1(n), n))
        .wrapping_add(call_reified(heavy_eval::<Tag2>, Tag2(n), n))
        .wrapping_add(call_reified(heavy_eval::<Tag3>, Tag3(n), n))
        .wrapping_add(call_reified(heavy_eval::<Tag4>, Tag4(n), n))
        .wrapping_add(call_reified(heavy_eval::<Tag5>, Tag5(n), n))
        .wrapping_add(call_reified(heavy_eval::<Tag6>, Tag6(n), n))
        .wrapping_add(call_reified(heavy_eval::<Tag7>, Tag7(n), n))
        .wrapping_add(call_reified(heavy_eval::<Tag8>, Tag8(n), n))
        .wrapping_add(call_reified(heavy_eval::<Tag9>, Tag9(n), n));
    let objs: [&dyn FullScore; 4] = [&Tag0(n), &Tag3(n), &Tag6(n), &Tag9(n)];
    let mut virt = 0_i64;
    for obj in objs {
        virt = virt.wrapping_add(call_dyn(obj, n));
    }
    reified.wrapping_add(virt)
}

pub const CASES: &[(&str, fn(i64) -> i64)] = &[
    ("case_stencil_niche", case_stencil_niche),
    ("case_stencil_reorder", case_stencil_reorder),
    ("case_stencil_assoc_abi", case_stencil_assoc_abi),
    ("case_stencil_inline_split", case_stencil_inline_split),
    ("case_stencil_specialize", case_stencil_specialize),
    ("case_stencil_type_info", case_stencil_type_info),
    ("case_stencil_stack_probe", case_stencil_stack_probe),
    ("case_stencil_i128", case_stencil_i128),
    ("case_stencil_reify_vtable", case_stencil_reify_vtable),
];
