use std::sync::atomic::{AtomicUsize, Ordering};

use cranelift_codegen::ir::{
    AbiParam, BlockArg, BlockCall, ExceptionTableData, ExceptionTableItem, InstBuilder, Signature, TrapCode, types,
};
use cranelift_codegen::{LabelOrOffset, MachExceptionHandler};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{Linkage, Module};

use hir_ty::mir::MirBody;
use hir_ty::next_solver::{Ty, TyKind};
use rustc_type_ir::IntTy;
use rustc_type_ir::inherent::IntoKind;

use super::{Backend, INTERNAL_CONV, Sig, pointer_signature};

fn handler_offset(target: LabelOrOffset) -> Option<u32> {
    target.is_offset().then(|| target.as_offset())
}

static ENTRY: AtomicUsize = AtomicUsize::new(0);

type EntryCode = extern "C" fn(usize, *mut u64, *const i64);

/// Calls generated code at `code` with a return pointer and an argument pointer from the host.
pub fn enter(code: usize, ret: *mut u64, argument: *const i64) {
    let entry = ENTRY.load(Ordering::Acquire);
    // SAFETY: ENTRY holds the finalized entry thunk, which takes the code address and both pointers in the host convention.
    let entry: EntryCode = unsafe { std::mem::transmute(entry) };
    entry(code, ret, argument);
}

pub fn call(table: &[AtomicUsize], slot: usize, value: i64) -> i64 {
    let code = table[slot].load(Ordering::Acquire);
    let mut result = 0_u64;
    enter(code, &mut result, &value);
    result as i64
}

pub(super) fn handler_sites(code: &cranelift_codegen::CompiledCode) -> Vec<crate::fault::Site> {
    let found = code.buffer.call_sites().filter_map(|site| {
        let handler = site.exception_handlers.iter().find_map(|handler| match handler {
            MachExceptionHandler::Default(target) | MachExceptionHandler::Tag(_, target) => handler_offset(*target),
            _ => None,
        })?;
        Some(crate::fault::Site { ret: site.ret_addr, handler, frame_offset: site.frame_offset? })
    });
    found.collect()
}

impl Backend {
    pub(super) fn compile_entry(&mut self) -> Result<(), String> {
        let mut signature = self.module.make_signature();
        signature.params.extend([AbiParam::new(types::I64); 3]);
        let id = self.module.declare_function("entry", Linkage::Local, &signature).map_err(|e| e.to_string())?;
        self.ctx.func.signature = signature;
        let config = self.module.target_config();
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        let params = builder.block_params(block).to_vec();
        let mut target = Signature::new(INTERNAL_CONV);
        pointer_signature(&mut target, Sig { params: 1, ret: true });
        let target = builder.import_signature(target);
        let (done, caught) = (builder.create_block(), builder.create_block());
        builder.append_block_param(caught, types::I64);
        let normal = BlockCall::new(done, [], &mut builder.func.dfg.value_lists);
        let handler = BlockCall::new(caught, [BlockArg::TryCallExn(0)], &mut builder.func.dfg.value_lists);
        let table = ExceptionTableData::new(target, normal, [ExceptionTableItem::Default(handler)]);
        let table = builder.func.dfg.exception_tables.push(table);
        builder.ins().try_call_indirect(params[0], &params[1..], table);
        builder.switch_to_block(done);
        builder.ins().return_(&[]);
        builder.switch_to_block(caught);
        let exception = builder.block_params(caught)[0];
        let mut host = Signature::new(config.default_call_conv);
        host.params.push(AbiParam::new(types::I64));
        let host = builder.import_signature(host);
        let code = builder.ins().iconst(types::I64, crate::unwind::uncaught as *const () as usize as i64);
        builder.ins().call_indirect(host, code, &[exception]);
        builder.ins().trap(TrapCode::unwrap_user(2));
        builder.seal_all_blocks();
        builder.finalize(config);
        self.define_named(id, "entry")?;
        self.publish_stubs()?;
        ENTRY.store(self.module.get_finalized_function(id) as usize, Ordering::Release);
        Ok(())
    }
}

pub(super) fn takes_and_returns_i64(body: &MirBody<'_>) -> bool {
    let long = |ty: Ty<'_>| matches!(ty.kind(), TyKind::Int(IntTy::I64));
    let ret = body.locals.iter().next().is_some_and(|(_, local)| long(local.ty.as_ref()));
    let [param] = body.param_locals[..] else {
        return false;
    };
    ret && long(body.locals[param].ty.as_ref())
}
