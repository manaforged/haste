use crate::arm64::fuse::width_mask;

const ADD_IMM_LIMIT: u64 = 1 << 24;
const ADD_IMM_LO_MASK: u64 = 0xFFF;
const ADD_IMM_HI_MASK: u64 = 0xFFF_000;

/// Splits `imm` into `(effective_sub, single_imm)` for one `add`/`sub` immediate instruction.
pub(crate) fn add_imm_parts(sub: bool, imm: i64) -> Option<(bool, u32)> {
    let abs = imm.unsigned_abs();
    (abs <= ADD_IMM_LO_MASK || ((abs & ADD_IMM_LO_MASK) == 0 && abs < ADD_IMM_LIMIT))
        .then(|| (if imm < 0 { !sub } else { sub }, abs as u32))
}

/// Splits `imm` into `(effective_sub, hi, lo)` for a two-instruction 24-bit `add`/`sub` immediate sequence.
pub(crate) fn add_imm_split_parts(sub: bool, imm: i64) -> Option<(bool, u32, u32)> {
    let abs = imm.unsigned_abs();
    let (hi, lo) = (
        (abs & ADD_IMM_HI_MASK) as u32,
        (abs & ADD_IMM_LO_MASK) as u32,
    );
    (abs < ADD_IMM_LIMIT && hi != 0 && lo != 0).then(|| (if imm < 0 { !sub } else { sub }, hi, lo))
}

pub(crate) fn encode_bitmask_imm(val: u64, is_64: bool) -> Option<(u32, u32, u32)> {
    let w = if is_64 {
        val
    } else {
        let low = (val as u32) as u64;
        low | (low << 32)
    };
    if w == 0 || w == u64::MAX {
        return None;
    }
    let mut esize = 64u32;
    while esize > 2 && w == w.rotate_right(esize / 2) {
        esize /= 2;
    }
    let mask = width_mask(esize);
    let elem = w & mask;
    let ones = elem.count_ones();
    if ones == 0 || ones == esize {
        return None;
    }
    let tz = elem.trailing_zeros();
    let immr = if tz > 0 {
        (esize - tz) & (esize - 1)
    } else {
        ones - (!elem & mask).trailing_zeros()
    };
    let (n, imms) = (
        u32::from(esize == 64),
        ((!(esize - 1) << 1) | (ones - 1)) & 0x3F,
    );
    let pat = (1u64 << ones) - 1;
    let mut decoded = if esize == 64 {
        pat.rotate_right(immr)
    } else {
        ((pat >> immr) | (pat << (esize - immr))) & mask
    };
    let mut shift = esize;
    while shift < 64 {
        decoded |= decoded << shift;
        shift *= 2;
    }
    (decoded == w).then_some((n, immr, imms))
}
