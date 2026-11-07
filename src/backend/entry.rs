use std::sync::atomic::{AtomicUsize, Ordering};

use haste_codegen::{AbiParam, CallConv, Function, FunctionBuilder, Signature, Type};

use hir_def::FunctionId;
use hir_expand::name::Name;
use hir_ty::db::HirDatabase;
use hir_ty::mir::MirBody;
use hir_ty::next_solver::{DbInterner, GenericArgs, Ty, TyKind};
use rustc_type_ir::IntTy;
use rustc_type_ir::inherent::IntoKind;

use super::{Backend, INTERNAL_CONV, Sig, Target, body_of, env_of, pointer_signature};

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

impl Backend {
    /// Gives the slot of `<T as Termination>::report` for the type `main` returns, or `None` when it returns `()`.
    pub fn report<'db>(&mut self, db: &'db dyn HirDatabase, main: FunctionId) -> Result<Option<usize>, String> {
        let env = env_of(db, main);
        let body = body_of(db, env, &Target::Func(main, GenericArgs::empty()))?;
        let ret = body.locals.iter().next().ok_or("main has no return local")?.1.ty.as_ref();
        if matches!(ret.kind(), TyKind::Tuple(items) if items.is_empty()) {
            return Ok(None);
        }
        let termination = DbInterner::new_with(db, env.krate).lang_items().Termination.ok_or("no Termination trait")?;
        let report = termination.trait_items(db).method_by_name(&Name::new_root("report")).ok_or("no Termination::report")?;
        let (imp, args) = db.lookup_impl_method(env, report, GenericArgs::new_from_slice(&[ret.into()]));
        let imp = imp.left().ok_or("Termination::report resolved to a derive")?;
        let (slot, _) = self.instance(db, imp, args, main)?;
        Ok(Some(slot))
    }

    pub(super) fn compile_entry(&mut self) -> Result<(), String> {
        let mut signature = Signature::new(CallConv::Native);
        signature.params.extend([AbiParam::new(Type::I64); 3]);
        let mut func = Function::new(signature);
        let mut builder = FunctionBuilder::new(&mut func);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        let params = builder.block_params(block).to_vec();
        let mut target = Signature::new(INTERNAL_CONV);
        pointer_signature(&mut target, Sig { params: 1, ret: true });
        let target = builder.import_signature(target);
        let (done, caught) = (builder.create_block(), builder.create_block());
        let exception = builder.append_block_param(caught, Type::I64);
        builder.ins().try_call_indirect(target, params[0], &params[1..], done, caught);
        builder.switch_to_block(done);
        builder.ins().return_(&[]);
        builder.switch_to_block(caught);
        let mut host = Signature::new(CallConv::Native);
        host.params.push(AbiParam::new(Type::I64));
        let host = builder.import_signature(host);
        let code = builder.ins().iconst(Type::I64, crate::unwind::uncaught as *const () as usize as i64);
        builder.ins().call_indirect(host, code, &[exception]);
        builder.ins().trap();
        let entry = self.define(&func, "entry")?;
        self.publish_stubs()?;
        ENTRY.store(entry, Ordering::Release);
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
