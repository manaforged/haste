//! Inline template outcomes, stored in the instance table under a tagged item hash with the rest of the key in the payload.

use anyhow::Result;

use super::{Substrate, SubstrateBuilder};

const TAG: u64 = 0x1417_E5ED_CA11_EE5A;
const BODY_TAG: u64 = 0xB0D1_E5ED_CA11_EE5A;
const WIRE_LEN: usize = 16;
const BODY_HEAD: usize = 13;
const LOW: u8 = 1;
const PRESENT: u8 = 2;
const RETURNS: u8 = 4;
const SOLE: u8 = 8;

/// Whether an inline template returns, its inline height and its instruction count, or `None` when the callee has no template.
pub type InlineShape = Option<(bool, u8, u16)>;

/// The identity of one inline template: its mode-keyed item hash, footprint, instruction limit, entry placement and whether it has one call site.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InlineKey {
    pub item: u64,
    pub footprint: u128,
    pub limit: u32,
    pub low: bool,
    pub sole: bool,
}

impl InlineKey {
    fn index(&self, tag: u64) -> (u64, u64) {
        (self.item ^ tag, self.footprint as u64)
    }

    fn body_head(&self) -> [u8; BODY_HEAD] {
        let mut head = [0; BODY_HEAD];
        head.copy_from_slice(&self.encode(None)[..BODY_HEAD]);
        head
    }

    fn encode(&self, shape: InlineShape) -> [u8; WIRE_LEN] {
        let (returns, height, size) = shape.unwrap_or_default();
        let mut out = [0u8; WIRE_LEN];
        out[..8].copy_from_slice(&((self.footprint >> 64) as u64).to_le_bytes());
        out[8..12].copy_from_slice(&self.limit.to_le_bytes());
        out[12] = LOW * u8::from(self.low) | SOLE * u8::from(self.sole) | PRESENT * u8::from(shape.is_some()) | RETURNS * u8::from(returns);
        out[13] = height;
        out[14..].copy_from_slice(&size.to_le_bytes());
        out
    }

    fn decode(&self, bytes: &[u8]) -> Option<InlineShape> {
        if bytes.len() != WIRE_LEN {
            return None;
        }
        let shape = (bytes[12] & PRESENT != 0).then(|| (bytes[12] & RETURNS != 0, bytes[13], u16::from_le_bytes([bytes[14], bytes[15]])));
        (self.encode(shape).as_slice() == bytes).then_some(shape)
    }
}

impl Substrate {
    /// Looks up the recorded outcome of inline template `key` across every layer, or `None` when absent or recorded with two different outcomes.
    pub fn inline_meta(&self, key: &InlineKey) -> Option<InlineShape> {
        let (item, footprint) = key.index(TAG);
        let mut found = None;
        let mut layer = Some(self);
        while let Some(substrate) = layer {
            for shape in substrate.local_instance_candidates(item, footprint).filter_map(|bytes| key.decode(bytes)) {
                if found.is_some_and(|prior| prior != shape) {
                    return None;
                }
                found = Some(shape);
            }
            layer = substrate.parent.as_deref();
        }
        found
    }

    /// Looks up the encoded body of inline template `key` in the nearest layer that holds one.
    pub fn inline_body(&self, key: &InlineKey) -> Option<&[u8]> {
        let (item, footprint) = key.index(BODY_TAG);
        let head = key.body_head();
        let mut layer = Some(self);
        while let Some(substrate) = layer {
            if let Some(body) = substrate.local_instance_candidates(item, footprint).find_map(|bytes| bytes.strip_prefix(head.as_slice())) {
                return Some(body);
            }
            layer = substrate.parent.as_deref();
        }
        None
    }
}

impl SubstrateBuilder {
    /// Appends the outcome of inline template `key`.
    pub fn add_inline_meta(&mut self, key: &InlineKey, shape: InlineShape) -> Result<()> {
        let (item, footprint) = key.index(TAG);
        self.add_instance(item, footprint, &key.encode(shape))
    }

    /// Appends the encoded body of inline template `key`.
    pub fn add_inline_body(&mut self, key: &InlineKey, body: &[u8]) -> Result<()> {
        let (item, footprint) = key.index(BODY_TAG);
        let mut bytes = Vec::with_capacity(BODY_HEAD + body.len());
        bytes.extend_from_slice(&key.body_head());
        bytes.extend_from_slice(body);
        self.add_instance(item, footprint, &bytes)
    }
}
