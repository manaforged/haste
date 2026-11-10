//! The float intrinsics that have no inline lowering, as host functions that rustc compiled from the same std methods.

macro_rules! math {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty = $body:expr;)*) => {
        $(extern "C" fn $name($($arg: $ty),*) -> $ret { $body })*

        /// The host function for the float intrinsic `name`, when it is one.
        pub fn symbol(name: &str) -> Option<usize> {
            match name {
                $(stringify!($name) => Some($name as extern "C" fn($($ty),*) -> $ret as usize),)*
                _ => None,
            }
        }
    };
}

math! {
    sinf32(x: f32) -> f32 = x.sin();
    sinf64(x: f64) -> f64 = x.sin();
    cosf32(x: f32) -> f32 = x.cos();
    cosf64(x: f64) -> f64 = x.cos();
    expf32(x: f32) -> f32 = x.exp();
    expf64(x: f64) -> f64 = x.exp();
    exp2f32(x: f32) -> f32 = x.exp2();
    exp2f64(x: f64) -> f64 = x.exp2();
    logf32(x: f32) -> f32 = x.ln();
    logf64(x: f64) -> f64 = x.ln();
    log2f32(x: f32) -> f32 = x.log2();
    log2f64(x: f64) -> f64 = x.log2();
    log10f32(x: f32) -> f32 = x.log10();
    log10f64(x: f64) -> f64 = x.log10();
    powf32(x: f32, y: f32) -> f32 = x.powf(y);
    powf64(x: f64, y: f64) -> f64 = x.powf(y);
    powif32(x: f32, n: i32) -> f32 = x.powi(n);
    powif64(x: f64, n: i32) -> f64 = x.powi(n);
    fmaf32(x: f32, y: f32, z: f32) -> f32 = x.mul_add(y, z);
    fmaf64(x: f64, y: f64, z: f64) -> f64 = x.mul_add(y, z);
    fmuladdf32(x: f32, y: f32, z: f32) -> f32 = x.mul_add(y, z);
    fmuladdf64(x: f64, y: f64, z: f64) -> f64 = x.mul_add(y, z);
    maxnumf32(x: f32, y: f32) -> f32 = x.max(y);
    maxnumf64(x: f64, y: f64) -> f64 = x.max(y);
    minnumf32(x: f32, y: f32) -> f32 = x.min(y);
    minnumf64(x: f64, y: f64) -> f64 = x.min(y);
    round_ties_even_f32(x: f32) -> f32 = x.round_ties_even();
    round_ties_even_f64(x: f64) -> f64 = x.round_ties_even();
}
