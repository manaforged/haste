use crate::ir::{
    AtomicRmwOp, Binary, CallConv, Convert, FloatCC, Function, IntCC, Lane, ReduceOp, TargetOpId,
    Type, Unary,
};

mod decode;
mod encode;

const FORMAT: u8 = 7;

macro_rules! table {
    ($name:ident: $ty:ident = [$($variant:ident),* $(,)?]) => {
        const $name: &[$ty] = &[$($ty::$variant),*];
        const _: () = {
            let mut at = 0;
            while at < $name.len() {
                assert!($name[at] as usize == at);
                at += 1;
            }
        };
    };
}

table!(UNARY: Unary = [
    Ineg, Bnot, Bswap, Bitrev, Clz, Ctz, Popcnt, Fneg, Fabs, Sqrt, Floor, Ceil, Trunc,
]);
table!(BINARY: Binary = [
    Iadd, Isub, Imul, Sdiv, Udiv, Srem, Urem, Umulhi, Smulhi, Smin, Smax, Umin, Umax, Band, Bor,
    Bxor, Ishl, Ushr, Sshr, Rotl, Rotr, Iconcat, Fadd, Fsub, Fmul, Fdiv, Fmin, Fmax, Fcopysign,
    SaddSat, UaddSat, SsubSat, UsubSat, AvgRound,
]);
table!(INT_CC: IntCC = [
    Equal,
    NotEqual,
    SignedLessThan,
    SignedGreaterThanOrEqual,
    SignedGreaterThan,
    SignedLessThanOrEqual,
    UnsignedLessThan,
    UnsignedGreaterThanOrEqual,
    UnsignedGreaterThan,
    UnsignedLessThanOrEqual,
]);
table!(FLOAT_CC: FloatCC = [
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
]);
table!(RMW: AtomicRmwOp = [Add, Sub, And, Nand, Or, Xor, Xchg, Umin, Umax, Smin, Smax]);
table!(REDUCE: ReduceOp = [
    Add, Smin, Smax, Umin, Umax, Fmin, Fmax, And, Or, Xor, Movemask,
]);
table!(TARGET_OPS: TargetOpId = [
    Arm64Sha256H,
    Arm64Sha256H2,
    Arm64Sha256Su0,
    Arm64Sha256Su1,
    Arm64AesE,
    Arm64AesD,
    Arm64AesMc,
    Arm64AesImc,
    X64Sha1NextE,
    X64Sha1Msg1,
    X64Sha1Msg2,
    X64Sha256Rnds2,
    X64Sha256Msg1,
    X64Sha256Msg2,
    X64AesEnc,
    X64AesEncLast,
    X64AesDec,
    X64AesDecLast,
    X64AesImc,
    X64Low128,
    X64High128,
    X64Concat128,
    X64Pshufb256,
    X64Pcmpeqb256,
    X64Pcmpgtb256,
    X64Pmovmskb256,
    X64ClmulLowLow,
    X64ClmulHighLow,
    X64ClmulLowHigh,
    X64ClmulHighHigh,
    X64AddCarry,
    X64AddCarryOut,
    X64SubBorrow,
    X64SubBorrowOut,
    X64DivWide,
    X64RemWide,
    X64Psadbw256,
    X64Pmaddubsw256,
    X64Pmaddwd256,
    X64Psadbw,
    Arm64Sqdmulh,
]);
table!(LANES: Lane = [I8, I16, I32, I64, F32, F64]);
table!(CONVS: CallConv = [Internal, Native]);

const CONVERTS: [Convert; 12] = [
    Convert::Uextend,
    Convert::Sextend,
    Convert::Ireduce,
    Convert::Fpromote,
    Convert::Fdemote,
    Convert::FromSint,
    Convert::FromUint,
    Convert::ToSintSat,
    Convert::ToUintSat,
    Convert::Bitcast,
    Convert::NarrowSat { signed: false },
    Convert::NarrowSat { signed: true },
];

const SCALARS: [Type; 7] = [
    Type::I8,
    Type::I16,
    Type::I32,
    Type::I64,
    Type::I128,
    Type::F32,
    Type::F64,
];

fn pick<T: Copy>(table: &[T], tag: u8) -> Option<T> {
    table.get(usize::from(tag)).copied()
}

fn tag_of<T: PartialEq>(table: &[T], value: &T) -> u8 {
    table
        .iter()
        .position(|entry| entry == value)
        .map_or(u8::MAX, |at| at as u8)
}

impl Function {
    /// Encodes this function into bytes that `Function::decode` turns back into an equal function, or `None` for a coroutine.
    pub fn encode(&self) -> Option<Vec<u8>> {
        encode::encode(self)
    }

    /// Decodes a function that `Function::encode` wrote, or `None` when `bytes` do not hold one.
    pub fn decode(bytes: &[u8]) -> Option<Function> {
        decode::decode(bytes)
    }
}
