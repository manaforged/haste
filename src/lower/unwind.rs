use std::cmp::Reverse;

use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, BlockCall, ExceptionTableData, ExceptionTableItem, InstBuilder, SigRef, Signature, StackSlotData,
    StackSlotKind, Value, types,
};
use hir_ty::mir::PlaceTy;

use super::{Lower, Place};
use crate::backend::{INTERNAL_CONV, Sig, pointer_signature};

const TRY: Sig = Sig { params: 1, ret: false };
const CATCH: Sig = Sig { params: 2, ret: false };

impl<'db> Lower<'_, '_, 'db> {
    fn landing(&mut self) -> Option<Block> {
        if self.cleaning || self.flags.is_empty() {
            return None;
        }
        if self.landing.is_none() {
            let block = self.builder.create_block();
            self.builder.append_block_param(block, types::I64);
            self.landing = Some(block);
        }
        self.landing
    }

    fn try_call(&mut self, signature: SigRef, code: Value, values: &[Value], handler: Block) {
        let next = self.builder.create_block();
        let pool = &mut self.builder.func.dfg.value_lists;
        let normal = BlockCall::new(next, [], pool);
        let caught = BlockCall::new(handler, [BlockArg::TryCallExn(0)], pool);
        let table = ExceptionTableData::new(signature, normal, [ExceptionTableItem::Default(caught)]);
        let table = self.builder.func.dfg.exception_tables.push(table);
        self.builder.ins().try_call_indirect(code, values, table);
        self.builder.switch_to_block(next);
    }

    /// Calls generated code, sending a panic to this function's cleanup when it has one.
    pub(super) fn invoke(&mut self, signature: SigRef, code: Value, values: &[Value]) {
        match self.landing() {
            Some(landing) => self.try_call(signature, code, values, landing),
            None => {
                self.builder.ins().call_indirect(signature, code, values);
            }
        }
    }

    pub(super) fn raise(&mut self, exception: Value) {
        let frame = self.builder.ins().get_frame_pointer(types::I64);
        let mut signature = Signature::new(self.call_conv);
        signature.params.extend([AbiParam::new(types::I64); 2]);
        let signature = self.builder.import_signature(signature);
        let code = self.builder.ins().iconst(types::I64, crate::unwind::raise as *const () as usize as i64);
        self.builder.ins().call_indirect(signature, code, &[frame, exception]);
        self.trap();
    }

    /// Drops every live local in reverse order and keeps unwinding.
    pub(super) fn fill_landing(&mut self) -> Result<(), String> {
        let Some(block) = self.landing else { return Ok(()) };
        self.cleaning = true;
        self.builder.switch_to_block(block);
        let exception = self.builder.block_params(block)[0];
        let mut locals: Vec<_> = self.flags.keys().copied().collect();
        locals.sort_by_key(|local| Reverse(local.into_raw().into_u32()));
        for local in locals {
            let found = Place { addr: self.local_addr(local), ty: PlaceTy::from_ty(self.body.locals[local].ty.as_ref()), meta: None };
            self.drop_flagged(local, found)?;
        }
        self.raise(exception);
        Ok(())
    }

    pub(super) fn catch_unwind(&mut self, dest: Place<'db>, args: &[Place<'db>]) -> Result<(), String> {
        let [try_fn, data, catch_fn] = args else { return Err("catch_unwind takes three arguments".to_owned()) };
        let try_cell = self.load(types::I64, try_fn.addr, 0);
        let catch_cell = self.load(types::I64, catch_fn.addr, 0);
        let mut signature = Signature::new(INTERNAL_CONV);
        pointer_signature(&mut signature, TRY);
        let signature = self.builder.import_signature(signature);
        let code = self.load(types::I64, try_cell, 0);
        let (caught, done) = (self.builder.create_block(), self.builder.create_block());
        self.builder.append_block_param(caught, types::I64);
        self.try_call(signature, code, &[data.addr], caught);
        let zero = self.builder.ins().iconst(types::I32, 0);
        self.store(zero, dest.addr, 0);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(caught);
        let exception = self.builder.block_params(caught)[0];
        let holder = self.builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
        let holder = self.builder.ins().stack_addr(types::I64, holder, 0);
        self.store(exception, holder, 0);
        self.call_cell(catch_cell, CATCH, &[data.addr, holder]);
        let one = self.builder.ins().iconst(types::I32, 1);
        self.store(one, dest.addr, 0);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(())
    }
}
