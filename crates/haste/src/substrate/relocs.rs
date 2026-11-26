mod decode;
mod encode;

pub use decode::decode_reloc_plan;
pub use encode::encode_reloc_plan;

const PLAN_MAGIC: [u8; 4] = *b"HRP2";
