//! The float intrinsics that have no inline lowering, as host functions that rustc compiled from the same std methods.

macro_rules! math {
    ($($variant:ident = $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty = $body:expr;)*) => {
        /// A host float math helper symbol.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        #[repr(u8)]
        pub enum MathSymbol {
            $($variant,)*
        }

        $(extern "C" fn $name($($arg: $ty),*) -> $ret { $body })*

        impl MathSymbol {
            const ALL: &[Self] = &[$(Self::$variant),*];

            /// Looks up a math helper symbol by intrinsic name.
            pub fn from_name(name: &str) -> Option<Self> {
                Self::ALL.iter().copied().find(|sym| sym.link_name() == name)
            }

            /// The C link symbol name for this math helper.
            pub const fn link_name(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($name),)*
                }
            }

            /// The address of this math helper in the current process.
            pub fn host_addr(self) -> usize {
                match self {
                    $(Self::$variant => $name as extern "C" fn($($ty),*) -> $ret as usize,)*
                }
            }
        }

        /// The host function for the float intrinsic `name`, when it is one.
        pub fn symbol(name: &str) -> Option<usize> {
            MathSymbol::from_name(name).map(MathSymbol::host_addr)
        }
    };
}

math! {
    Sinf32 = sinf32(x: f32) -> f32 = x.sin();
    Sinf64 = sinf64(x: f64) -> f64 = x.sin();
    Cosf32 = cosf32(x: f32) -> f32 = x.cos();
    Cosf64 = cosf64(x: f64) -> f64 = x.cos();
    Expf32 = expf32(x: f32) -> f32 = x.exp();
    Expf64 = expf64(x: f64) -> f64 = x.exp();
    Exp2f32 = exp2f32(x: f32) -> f32 = x.exp2();
    Exp2f64 = exp2f64(x: f64) -> f64 = x.exp2();
    Logf32 = logf32(x: f32) -> f32 = x.ln();
    Logf64 = logf64(x: f64) -> f64 = x.ln();
    Log2f32 = log2f32(x: f32) -> f32 = x.log2();
    Log2f64 = log2f64(x: f64) -> f64 = x.log2();
    Log10f32 = log10f32(x: f32) -> f32 = x.log10();
    Log10f64 = log10f64(x: f64) -> f64 = x.log10();
    Powf32 = powf32(x: f32, y: f32) -> f32 = x.powf(y);
    Powf64 = powf64(x: f64, y: f64) -> f64 = x.powf(y);
    Powif32 = powif32(x: f32, n: i32) -> f32 = x.powi(n);
    Powif64 = powif64(x: f64, n: i32) -> f64 = x.powi(n);
    Fmaf32 = fmaf32(x: f32, y: f32, z: f32) -> f32 = x.mul_add(y, z);
    Fmaf64 = fmaf64(x: f64, y: f64, z: f64) -> f64 = x.mul_add(y, z);
    Fmuladdf32 = fmuladdf32(x: f32, y: f32, z: f32) -> f32 = x.mul_add(y, z);
    Fmuladdf64 = fmuladdf64(x: f64, y: f64, z: f64) -> f64 = x.mul_add(y, z);
    Maxnumf32 = maxnumf32(x: f32, y: f32) -> f32 = x.max(y);
    Maxnumf64 = maxnumf64(x: f64, y: f64) -> f64 = x.max(y);
    Minnumf32 = minnumf32(x: f32, y: f32) -> f32 = x.min(y);
    Minnumf64 = minnumf64(x: f64, y: f64) -> f64 = x.min(y);
    RoundTiesEvenF32 = round_ties_even_f32(x: f32) -> f32 = x.round_ties_even();
    RoundTiesEvenF64 = round_ties_even_f64(x: f64) -> f64 = x.round_ties_even();
}
