/// The most chunks a copy of a constant size is emitted inline with, one scratch register each.
const CHUNKS: usize = 4;

/// The widths and offsets of the loads that copy `bytes` bytes inline, widest first, when there are at most `CHUNKS` of them.
pub(crate) fn chunks(bytes: i64) -> Option<Vec<(u32, u32)>> {
    let mut left = u32::try_from(bytes).ok().filter(|bytes| *bytes <= 8 * CHUNKS as u32)?;
    let mut chunks = Vec::new();
    for width in [8, 4, 2, 1] {
        while left >= width {
            chunks.push((width, bytes as u32 - left));
            left -= width;
        }
    }
    (chunks.len() <= CHUNKS).then_some(chunks)
}
