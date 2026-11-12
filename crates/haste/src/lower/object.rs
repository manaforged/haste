use anyhow::{Result, bail};
use haste_codegen::{Type, Value};

use crate::body::ExprId;
use crate::prep::Vtable as Laid;
use crate::sema::Context;
use crate::sema::infer::{ALIGN_SLOT, Res, Vtable};
use crate::sema::reach::Reached;
use crate::sema::ty::{Kind, Ty};

use super::reloc::{RelocStep, reached_index_of};
use super::stencil::{SymbolSource, VtableEntry, VtableId};
use super::{Lower, Slots, Word};

pub(super) fn build_vtable_id(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, vtable: &Vtable) -> Result<VtableId> {
    let layout = cx.layout(vtable.ty)?;
    let mut entries = vec![
        VtableEntry::Word(code_word(cx, reached, slots, vtable.drop.as_ref())?),
        VtableEntry::Word(Word::Int(layout.size.bytes())),
        VtableEntry::Word(Word::Int(layout.align.bytes())),
    ];
    if let Some(laid) = crate::prep::vtable(cx, vtable.object) {
        laid_entries(cx, reached, slots, vtable, &laid, &mut entries)?;
        return Ok(slots.vtable_id(&entries));
    }
    for method in &vtable.methods {
        entries.push(VtableEntry::Word(code_word(cx, reached, slots, method.as_ref())?));
    }
    for inner in &vtable.supers {
        entries.push(VtableEntry::Super(build_vtable_id(cx, reached, slots, inner)?));
    }
    Ok(slots.vtable_id(&entries))
}

fn laid_entries(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, vtable: &Vtable, laid: &Laid, entries: &mut Vec<VtableEntry>) -> Result<()> {
    let Kind::Dyn(of, ..) = *cx.types.kind(vtable.object) else { bail!("a vtable for a type that is not a trait object") };
    let (methods, supers) = (cx.vtable_methods(of)?, cx.vtable_supers(of)?);
    let ends = laid.slots.iter().map(|slot| slot.at + 1).chain(laid.supers.iter().filter_map(|(_, at)| at.map(|at| at + 1)));
    entries.resize(ends.max().unwrap_or(entries.len()).max(entries.len()), VtableEntry::Word(Word::Int(0)));
    for slot in &laid.slots {
        let found = methods.iter().position(|method| *method == Some(slot.method)).and_then(|at| vtable.methods.get(at)).and_then(Option::as_ref);
        let Some(&Res::Fn(item, substs)) = found else { bail!("a vtable with no function for a method rustc's vtable holds") };
        let index = reached_index_of(cx, reached, slots, (item, substs))?;
        entries[slot.at] = VtableEntry::Word(Word::Int(crate::prep::entry(cx, slots, index, slot)?));
    }
    for &(inner, at) in &laid.supers {
        let Some(at) = at else { continue };
        let Some(found) = supers.iter().position(|&of| of == inner).and_then(|place| vtable.supers.get(place)) else { bail!("a vtable with no supertrait vtable rustc's vtable holds") };
        entries[at] = VtableEntry::Super(build_vtable_id(cx, reached, slots, found)?);
    }
    Ok(())
}

fn code_word(cx: &Context<'_>, reached: &Reached, slots: &dyn Slots, res: Option<&Res>) -> Result<Word> {
    match res {
        None => Ok(Word::Int(0)),
        Some(&Res::Fn(item, substs)) => reached_index_of(cx, reached, slots, (item, substs)).map(Word::Code),
        Some(_) => bail!("a vtable entry that is not a function"),
    }
}

impl Lower<'_, '_> {
    pub(super) fn vtable(&mut self, expr: ExprId, vtable: &Vtable) -> Result<Value> {
        let id = build_vtable_id(self.cx, self.reached, self.slots, vtable)?;
        Ok(self.emit_symbol_with(SymbolSource::Vtable(id), RelocStep::VtableAt(expr)))
    }

    /// The wide pointer `wide` to a trait object of type `object` as a pointer to the supertrait object whose vtable slot `slot` of Haste's vtable holds: in rustc's layout, the same vtable when it starts with the supertrait's, else the one its slot holds.
    pub(super) fn upcast(&mut self, wide: Value, slot: u32, object: Ty) -> Result<Value> {
        let data = self.ins().load(Type::I64, wide, 0);
        let table = self.ins().load(Type::I64, wide, 8);
        let Some((laid, of)) = self.laid(object) else {
            let inner = self.ins().load(Type::I64, table, slot as i32 * 8);
            return Ok(self.wide(data, inner));
        };
        let at = (slot - ALIGN_SLOT - 1) as usize - self.cx.vtable_methods(of)?.len();
        let Some(&inner) = self.cx.vtable_supers(of)?.get(at) else { bail!("an upcast to no supertrait") };
        let Some(&(_, vptr)) = laid.supers.iter().find(|(of, _)| *of == inner) else { bail!("an upcast to a supertrait rustc's vtable lacks") };
        let inner = match vptr {
            Some(at) => self.ins().load(Type::I64, table, at as i32 * 8),
            None => table,
        };
        Ok(self.wide(data, inner))
    }

    /// rustc's layout of the vtables of the trait object that `ty` points to and the object's trait, when every vtable of it is laid out so.
    fn laid(&mut self, ty: Ty) -> Option<(std::sync::Arc<Laid>, crate::names::ItemRef)> {
        let object = self.object_in(ty, 4)?;
        let Kind::Dyn(of, ..) = *self.cx.types.kind(object) else { return None };
        crate::prep::vtable(self.cx, object).map(|laid| (laid, of))
    }

    /// The trait object type that pointer type `ty` points to, looking at most `depth` types deep.
    fn object_in(&self, ty: Ty, depth: usize) -> Option<Ty> {
        if matches!(self.cx.types.kind(ty), Kind::Dyn(..)) {
            return Some(ty);
        }
        let kind = self.cx.types.kind(ty).clone();
        let children: Vec<Ty> = kind.children(&self.cx.types).collect();
        children.into_iter().find_map(|child| depth.checked_sub(1).and_then(|depth| self.object_in(child, depth)))
    }

    /// Lowers a trait object method: calls slot `slot` of the receiver's vtable with the receiver's data pointer and the other arguments, in rustc's convention and slot when every vtable of the object is laid out as rustc lays it out.
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
        let mut values: Vec<Value> = self.sret.into_iter().chain([data]).collect();
        for &local in &self.typed.params[1..] {
            if let Some(value) = self.locals[local as usize].and_then(|place| self.read(place)) {
                let ty = self.typed.locals[local as usize].ty;
                self.push_passed(ty, value, &mut values)?;
            }
        }
        let (item, substs) = self.reached.list[self.index].0;
        let object = self.cx.types.substs(substs).first().copied();
        let laid = object.and_then(|object| crate::prep::vtable(self.cx, object));
        let results = match laid.as_ref().map(|laid| laid.slots.iter().find(|method| method.method == item)) {
            Some(None) => bail!("a call of a method that rustc's vtable lacks"),
            Some(Some(method)) => {
                let code = self.ins().load(Type::I64, vtable, method.at as i32 * 8);
                crate::prep::call_native(self.cx, &mut self.b, ((&method.params, method.ret), &method.carries), (code, &values))?
            }
            None => {
                let code = self.ins().load(Type::I64, vtable, slot as i32 * 8);
                let thin = self.cx.types.prim(crate::names::Primitive::Usize);
                let rest: Vec<Ty> = self.typed.params[1..].iter().map(|&local| self.typed.locals[local as usize].ty).collect();
                let signature = super::signature_of(self.cx, self.typed.ret, std::iter::once(thin).chain(rest))?;
                let sig = self.b.import_signature(signature);
                let call = self.ins().call_indirect(sig, code, &values);
                self.b.inst_results(call).to_vec()
            }
        };
        self.ins().return_(&results);
        self.close();
        Ok(())
    }
}
