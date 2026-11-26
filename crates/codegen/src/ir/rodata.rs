use super::{Function, Type};

impl Function {
    /// The `size` known read-only bytes of symbol `sym` at byte `start`.
    pub(crate) fn rodata_slice(&self, sym: u32, start: i64, size: usize) -> Option<&[u8]> {
        let start = usize::try_from(start).ok()?;
        let (_, bytes) = self.rodata_bytes.iter().find(|&&(s, _)| s == sym)?;
        bytes.get(start..start.checked_add(size)?)
    }

    /// The scalar of type `ty` that the known read-only bytes of symbol `sym` hold at byte `start`: sign-extended for an integer, raw bits for a float.
    pub(crate) fn rodata_read(&self, sym: u32, start: i64, ty: Type) -> Option<i64> {
        if !(ty.is_small_int() || matches!(ty, Type::F32 | Type::F64)) {
            return None;
        }
        let size = usize::try_from(ty.bytes()).ok()?;
        let mut raw = [0_u8; 8];
        raw.get_mut(..size)?
            .copy_from_slice(self.rodata_slice(sym, start, size)?);
        let bits = u64::from_le_bytes(raw) as i64;
        Some(if ty.is_small_int() {
            crate::fold::extend(bits, ty, true)
        } else {
            bits
        })
    }
}
