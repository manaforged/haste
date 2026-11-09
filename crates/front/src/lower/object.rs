use anyhow::{Result, bail};
use haste_codegen::{Type, Value};

use crate::sema::infer::{Res, Vtable};

use super::{Lower, Word};

impl Lower<'_, '_> {
    /// The address of the vtable `vtable`: the drop glue, the size, the alignment, each method's code, then the address of each supertrait vtable.
    pub(super) fn vtable(&mut self, vtable: &Vtable) -> Result<i64> {
        let layout = self.cx.layout(vtable.ty)?;
        let mut words = vec![self.code_word(vtable.drop.as_ref())?, Word::Int(layout.size.bytes()), Word::Int(layout.align.abi.bytes())];
        for method in &vtable.methods {
            words.push(self.code_word(method.as_ref())?);
        }
        for inner in &vtable.supers {
            words.push(Word::Int(self.vtable(inner)? as u64));
        }
        Ok(self.slots.table(&words))
    }

    /// The wide pointer `wide` to a trait object as a pointer to the supertrait object whose vtable slot `slot` of its vtable holds.
    pub(super) fn upcast(&mut self, wide: Value, slot: u32) -> Value {
        let data = self.ins().load(Type::I64, wide, 0);
        let table = self.ins().load(Type::I64, wide, 8);
        let inner = self.ins().load(Type::I64, table, slot as i32 * 8);
        self.wide(data, inner)
    }

    fn code_word(&self, res: Option<&Res>) -> Result<Word> {
        match res {
            None => Ok(Word::Int(0)),
            Some(Res::Fn(item, substs)) => self.reached_index(*item, substs).map(Word::Code),
            Some(_) => bail!("a vtable entry that is not a function"),
        }
    }

    /// Lowers a trait object method: calls slot `slot` of the receiver's vtable with the receiver's data pointer and the other arguments.
    pub(super) fn virtual_call(&mut self, slot: u32) -> Result<()> {
        let Some(&receiver) = self.typed.params.first() else { bail!("a trait object method with no receiver") };
        let Some(wide) = self.locals[receiver as usize] else {
            self.ins().trap();
            self.close();
            return Ok(());
        };
        let pointer = self.address(wide);
        let data = self.ins().load(Type::I64, pointer, 0);
        let vtable = self.ins().load(Type::I64, pointer, 8);
        let code = self.ins().load(Type::I64, vtable, slot as i32 * 8);
        let mut values: Vec<Value> = self.sret.into_iter().chain([data]).collect();
        for &local in &self.typed.params[1..] {
            if let Some(value) = self.locals[local as usize].and_then(|place| self.read(place)) {
                let ty = self.typed.locals[local as usize].ty;
                values.push(self.passed(ty, value)?);
            }
        }
        let signature = super::signature(self.cx, self.typed)?;
        let returns = !signature.returns.is_empty();
        let sig = self.b.import_signature(signature);
        let call = self.ins().call_indirect(sig, code, &values);
        let results: Vec<Value> = if returns { vec![self.b.inst_results(call)[0]] } else { Vec::new() };
        self.ins().return_(&results);
        self.close();
        Ok(())
    }
}
