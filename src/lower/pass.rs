use haste_codegen::{Type, Value};

use super::{Lower, Place};

impl<'db> Lower<'_, '_, 'db> {
    pub(super) fn pass(&mut self, places: &[Place<'db>], skip: usize, values: &mut Vec<Value>, by_value: Option<u64>) -> Result<u64, String> {
        let Some(mut by_value) = by_value else {
            for place in places {
                if self.layout(place.ty.ty)?.size.bytes() != 0 {
                    values.push(place.addr);
                }
            }
            return Ok(0);
        };
        for place in places {
            let layout = self.layout(place.ty.ty)?;
            if layout.size.bytes() == 0 {
                continue;
            }
            let index = values.len() - skip;
            if crate::backend::passes_by_value(&layout) && index < 64 {
                by_value |= 1 << index;
                values.push(self.load(Type::I64, place.addr, 0));
            } else {
                values.push(place.addr);
            }
        }
        Ok(by_value)
    }
}
