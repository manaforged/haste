use hir_def::attrs::AttrFlags;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use cranelift_codegen::Context;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlagsData, Signature, TrapCode, types};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use either::Either;
use hir_def::builtin_derive::BuiltinDeriveImplMethod as Method;
use hir_def::signatures::FunctionSignature;
use hir_def::{CallableDefId, FunctionId, HasModule, ItemContainerId, VariantId};
use hir_ty::ParamEnvAndCrate;
use hir_ty::db::{HirDatabase, InternedClosureId};
use hir_ty::method_resolution::is_dyn_method;
use hir_ty::mir::{MirBody, Operand, OperandKind, PlaceTy};
use hir_ty::next_solver::infer::DbInternerInferExt;
use hir_ty::next_solver::{DbInterner, GenericArgs, Ty, TyKind, TypingMode};
use rustc_type_ir::inherent::{GenericArgs as _, IntoKind};

use hir_ty::mir::MirLowerError;

use crate::lower::Lower;
use crate::lower::native::NativeSig;

mod compiler;
mod entry;
mod slots;
mod specialize;

pub use compiler::{compile_slot, watch_forks};
pub use slots::{Key, Sig, Slots};
use slots::TABLE_SLOTS;

const PANIC_HANDLER: &str = "panic_impl";
const STD_CRATE: &str = "std";
const STD_PANIC_HANDLER: &str = "panic_handler";
const RUST_SYMBOLS: &[(&str, &str)] = &[
    ("__rust_start_panic", "panic_unwind"),
    ("__rust_panic_cleanup", "panic_unwind"),
    ("__rust_drop_panic", STD_CRATE),
    ("__rust_foreign_exception", STD_CRATE),
];
const CLONE: &str = "clone";
const DROP_GLUE: &str = "drop_glue";
const CONST_EVAL_SELECT: &str = "const_eval_select";
const STD_SYSCALLS: [&str; 7] = ["copy_file_range", "getrandom", "gettid", "preadv", "pwritev", "splice", "statx"];
const STD_ALIASES: [(&str, &str); 1] = [("min_stack_size", "__pthread_get_minstack")];

pub enum Target<'db> {
    Func(FunctionId, GenericArgs<'db>),
    Closure(InternedClosureId<'db>, GenericArgs<'db>),
    Ctor(VariantId),
    Intrinsic(String, GenericArgs<'db>),
    Foreign(String),
    Abort,
    Glue(Ty<'db>),
    Dyn(FunctionId, Ty<'db>),
    Pointer,
    Derive(Method, FunctionId, GenericArgs<'db>),
}

impl Target<'_> {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Target::Func(..) => "function",
            Target::Closure(..) => "closure",
            Target::Ctor(_) => "constructor",
            Target::Intrinsic(..) => "intrinsic",
            Target::Foreign(_) => "foreign function",
            Target::Abort => "abort",
            Target::Glue(_) => "drop glue",
            Target::Dyn(..) => "dyn method",
            Target::Pointer => "fn pointer",
            Target::Derive(..) => "derive",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Direct,
    FnTrait,
    Select,
}

pub struct Resolved<'db> {
    pub target: Target<'db>,
    pub shape: Shape,
}

enum Work<'db> {
    Body,
    Glue(Ty<'db>),
    Thunk(Target<'db>, bool),
    Derive(Method, FunctionId, GenericArgs<'db>),
    Shim(Ty<'db>),
    Native(Target<'db>, NativeSig),
}

pub struct Backend {
    module: JITModule,
    ctx: Context,
    builder: FunctionBuilderContext,
    slots: Slots,
    digests: HashMap<usize, u64>,
    pub stubs: Vec<(String, String)>,
    order: Vec<usize>,
    pub compiled_total: usize,
    unmapped: Vec<(FuncId, usize, String, Vec<crate::fault::Site>)>,
    cursor: usize,
    serial: usize,
    dump: bool,
    pub verbose: bool,
}

/// The calling convention of generated functions, which lets a try_call handler assume no preserved registers.
pub const INTERNAL_CONV: CallConv = CallConv::Tail;

const STUB_NAME: &str = "stub";

pub use entry::{call, enter};
use entry::handler_sites;


impl Backend {
    pub fn new(dump: bool, verbose: bool) -> Result<Self> {
        let flags = [("enable_llvm_abi_extensions", "true"), ("preserve_frame_pointers", "true")];
        let builder = JITBuilder::with_flags(&flags, default_libcall_names()).map_err(|e| anyhow!("{e}"))?;
        let module = JITModule::new(builder);
        let table = (0..TABLE_SLOTS).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
        let mut backend = Self {
            ctx: module.make_context(),
            module,
            builder: FunctionBuilderContext::new(),
            slots: Slots {
                table: Box::leak(table.into_boxed_slice()),
                by_key: HashMap::new(),
                info: Vec::new(),
                pending: Vec::new(),
                native_words: Vec::new(),
                statics: HashMap::new(),
                lazy_statics: HashMap::new(),
                thread_locals: HashMap::new(),
                vtables: HashMap::new(),
                notes: Vec::new(),
            },
            digests: HashMap::new(),
            stubs: Vec::new(),
            order: Vec::new(),
            compiled_total: 0,
            unmapped: Vec::new(),
            cursor: 0,
            serial: 0,
            dump,
            verbose,
        };
        backend.compile_entry().map_err(|e| anyhow!("{e}"))?;
        Ok(backend)
    }

    pub fn functions(&self) -> Vec<FunctionId> {
        self.slots.functions()
    }

    pub fn table(&self) -> &'static [AtomicUsize] {
        self.slots.table
    }

    pub fn entry<'db>(&mut self, db: &'db dyn HirDatabase, func: FunctionId) -> Result<(usize, bool), String> {
        let (slot, body) = self.instance(db, func, GenericArgs::empty(), func)?;
        Ok((slot, entry::takes_and_returns_i64(body)))
    }

    /// Gives the table slot of `func` with `args`, resolving impls as `scope` sees them, called through `enter`.
    pub fn instance<'db>(
        &mut self,
        db: &'db dyn HirDatabase,
        func: FunctionId,
        args: GenericArgs<'db>,
        scope: FunctionId,
    ) -> Result<(usize, &'db MirBody<'db>), String> {
        let env = env_of(db, scope);
        let target = Target::Func(func, args);
        let body = body_of(db, env, &target)?;
        let sig = signature_of(db, env, body)?;
        let key = key_of(&target).ok_or("entry is not compiled")?;
        let slot = self.slots.get(key, sig, env.store())?;
        self.publish_stubs()?;
        Ok((slot, body))
    }

    pub fn replay<'db>(&mut self, db: &'db dyn HirDatabase, order: &[usize]) {
        for slot in order.iter().copied() {
            if slot < self.slots.info.len() && !self.digests.contains_key(&slot) {
                self.compile(db, slot);
            }
        }
        self.order.clear();
    }

    pub fn take_order(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.order)
    }

    pub fn compile<'db>(&mut self, db: &'db dyn HirDatabase, slot: usize) {
        self.order.push(slot);
        let info = &self.slots.info[slot];
        let (key, sig, env) = (info.key.clone(), info.sig, info.env.clone());
        let env = env.as_ref(db);
        let name = key_name(db, &key);
        let lowered = self.compile_key(db, env, &name, &key, sig);
        for note in std::mem::take(&mut self.slots.notes) {
            if self.verbose {
                println!("stub {name}: {note}");
            }
            self.stubs.push((name.clone(), note));
        }
        match lowered {
            Ok(id) => {
                self.digests.insert(slot, target_digest(db, env, &target_of(&key)));
                self.finish(slot, id);
            }
            Err(reason) => {
                if self.verbose {
                    println!("stub {name}: {reason}");
                }
                let message = format!("{name}: {reason}");
                self.stubs.push((name, reason));
                self.digests.insert(slot, 0);
                if let Ok(id) = self.trap_function(sig, native_of(&key), &message) {
                    self.finish(slot, id);
                }
            }
        }
    }

    fn compile_key<'db>(
        &mut self,
        db: &'db dyn HirDatabase,
        env: ParamEnvAndCrate<'db>,
        name: &str,
        key: &Key,
        sig: Sig,
    ) -> Result<FuncId, String> {
        match key {
            Key::Func(..) | Key::Closure(..) => {
                let target = target_of(key);
                let body = body_of(db, env, &target)?;
                let found = signature_of(db, env, body)?;
                if found.params != sig.params || !return_fits(db, env, body, found, sig)? {
                    let params: Vec<String> = body.locals.iter().map(|(_, it)| format!("{:?}", it.ty.as_ref())).take(body.param_locals.len() + 1).collect();
                    return Err(format!(
                        "call site and callee disagree on the signature: site {}+{} callee {}+{} ret,params {params:?}",
                        sig.params, sig.ret, found.params, found.ret
                    ));
                }
                self.lower(db, env, name, body, sig, Work::Body)
            }
            Key::Glue(ty) => {
                let entry = self.slots.info.first().ok_or("no entry")?.key.clone();
                let body = body_of(db, env, &target_of(&entry))?;
                self.lower(db, env, name, body, sig, Work::Glue(ty.as_ref()))
            }
            Key::Const(id, args) => {
                let body = db
                    .monomorphized_mir_body((*id).into(), args.clone(), env.store())
                    .map_err(|error| format!("constant: {error:?}"))?;
                self.lower(db, env, name, body, sig, Work::Body)
            }
            Key::Static(id) => {
                let body = db
                    .monomorphized_mir_body((*id).into(), GenericArgs::empty().store(), env.store())
                    .map_err(|error| format!("static: {error:?}"))?;
                self.lower(db, env, name, body, sig, Work::Body)
            }
            Key::Derive(method, func, args) => {
                let entry = self.slots.info.first().ok_or("no entry")?.key.clone();
                let body = body_of(db, env, &target_of(&entry))?;
                self.lower(db, env, name, body, sig, Work::Derive(*method, *func, args.as_ref()))
            }
            Key::Shim(ty) => {
                let entry = self.slots.info.first().ok_or("no entry")?.key.clone();
                let body = body_of(db, env, &target_of(&entry))?;
                self.lower(db, env, name, body, sig, Work::Shim(ty.as_ref()))
            }
            Key::Native(inner, native) => {
                let target = target_of(inner);
                let body = body_of(db, env, &target)?;
                self.lower(db, env, name, body, sig, Work::Native(target, native.clone()))
            }
            Key::Thunk(inner, data) => {
                let target = target_of(inner);
                let body = body_of(db, env, &target)?;
                self.lower(db, env, name, body, sig, Work::Thunk(target, *data))
            }
        }
    }

    pub fn next_uncompiled(&mut self) -> Option<usize> {
        while self.cursor < self.slots.info.len() {
            let slot = self.cursor;
            self.cursor += 1;
            if !self.digests.contains_key(&slot) {
                return Some(slot);
            }
        }
        None
    }

    pub fn mark_failed(&mut self, slot: usize) {
        self.digests.insert(slot, 0);
    }

    pub fn refresh<'db>(&mut self, db: &'db dyn HirDatabase) -> Result<Vec<String>, String> {
        let mut changed = Vec::new();
        for slot in self.digests.keys().copied().collect::<Vec<_>>() {
            let info = &self.slots.info[slot];
            let (key, env) = (info.key.clone(), info.env.clone());
            if !matches!(key, Key::Func(..) | Key::Closure(..)) {
                continue;
            }
            let env = env.as_ref(db);
            let target = target_of(&key);
            let digest = target_digest(db, env, &target);
            if self.digests.get(&slot) == Some(&digest) {
                continue;
            }
            let body = body_of(db, env, &target).map_err(|why| format!("{}: {why}", name_of(db, &target)))?;
            changed.push((slot, name_of(db, &target), body, env, digest));
        }
        let mut names = Vec::with_capacity(changed.len());
        let mut done = Vec::with_capacity(changed.len());
        for (slot, name, body, env, digest) in changed {
            let sig = self.slots.info[slot].sig;
            if signature_of(db, env, body)? != sig {
                return Err(format!("{name}: signature changed"));
            }
            let id = self.lower(db, env, &name, body, sig, Work::Body).map_err(|why| format!("{name}: {why}"))?;
            done.push((slot, id, digest));
            names.push(name);
        }
        self.publish_stubs()?;
        for (slot, id, digest) in done {
            self.digests.insert(slot, digest);
            let code = self.module.get_finalized_function(id);
            self.slots.table[slot].store(code as usize, Ordering::Release);
        }
        Ok(names)
    }

    fn finish(&mut self, slot: usize, id: FuncId) {
        self.compiled_total += 1;
        if let Err(reason) = self.publish_stubs() {
            self.stubs.push(("table".to_owned(), reason));
        }
        let code = self.module.get_finalized_function(id);
        self.slots.table[slot].store(code as usize, Ordering::Release);
    }

    fn publish_stubs(&mut self) -> Result<(), String> {
        let pending = std::mem::take(&mut self.slots.pending);
        let mut made = Vec::with_capacity(pending.len());
        for slot in pending {
            made.push((slot, self.compile_stub(slot)?));
        }
        self.module.finalize_definitions().map_err(|e| e.to_string())?;
        for (id, size, name, sites) in std::mem::take(&mut self.unmapped) {
            crate::fault::remember(self.module.get_finalized_function(id) as usize, size, name, sites);
        }
        for (slot, id) in made {
            let code = self.module.get_finalized_function(id);
            self.slots.table[slot].store(code as usize, Ordering::Release);
        }
        for (word, slot) in std::mem::take(&mut self.slots.native_words) {
            let code = self.slots.table[slot].load(Ordering::Acquire);
            // SAFETY: word is a function pointer field inside constant data this engine leaked, and nothing runs that data until publish returns.
            unsafe { (word as *mut usize).write_unaligned(code) };
        }
        Ok(())
    }

    fn declare(&mut self, sig: Sig, native: Option<&NativeSig>) -> Result<FuncId, String> {
        let name = format!("f{}", self.serial);
        self.serial += 1;
        let mut signature = self.module.make_signature();
        match native {
            Some(native) => native.signature(&mut signature),
            None => {
                signature.call_conv = INTERNAL_CONV;
                pointer_signature(&mut signature, sig);
            }
        }
        let id = self.module.declare_function(&name, Linkage::Local, &signature).map_err(|e| e.to_string())?;
        self.ctx.func.signature = signature;
        Ok(id)
    }

    fn define(&mut self, id: FuncId) -> Result<FuncId, String> {
        self.define_named(id, STUB_NAME)
    }

    fn define_named(&mut self, id: FuncId, name: &str) -> Result<FuncId, String> {
        let defined = self.module.define_function(id, &mut self.ctx).map_err(|e| format!("{e:?}"));
        if defined.is_ok() {
            let size = self.ctx.compiled_code().map_or(0, |code| code.code_info().total_size as usize);
            let sites = self.ctx.compiled_code().map(handler_sites).unwrap_or_default();
            self.unmapped.push((id, size, name.to_owned(), sites));
        }
        self.module.clear_context(&mut self.ctx);
        defined.map(|()| id)
    }

    fn compile_stub(&mut self, slot: usize) -> Result<FuncId, String> {
        let sig = self.slots.info[slot].sig;
        let native = native_of(&self.slots.info[slot].key).cloned();
        let id = self.declare(sig, native.as_ref())?;
        let config = self.module.target_config();
        let address = self.slots.address(slot);
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);
        let params = builder.block_params(block).to_vec();
        let mut host = Signature::new(config.default_call_conv);
        host.params.push(AbiParam::new(types::I64));
        let host = builder.import_signature(host);
        let code = builder.ins().iconst(types::I64, compile_slot as *const () as usize as i64);
        let number = builder.ins().iconst(types::I64, slot as i64);
        builder.ins().call_indirect(host, code, &[number]);
        let mut target = Signature::new(config.default_call_conv);
        match &native {
            Some(native) => native.signature(&mut target),
            None => {
                target.call_conv = INTERNAL_CONV;
                pointer_signature(&mut target, sig);
            }
        }
        let target = builder.import_signature(target);
        let address = builder.ins().iconst(types::I64, address);
        let code = builder.ins().load(types::I64, MemFlagsData::trusted(), address, 0);
        let call = builder.ins().call_indirect(target, code, &params);
        let results = builder.inst_results(call).to_vec();
        builder.ins().return_(&results);
        builder.finalize(config);
        self.define(id)
    }

    fn trap_function(&mut self, sig: Sig, native: Option<&NativeSig>, reason: &str) -> Result<FuncId, String> {
        let id = self.declare(sig, native)?;
        let config = self.module.target_config();
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);
        let (text, size) = leak_text(reason);
        let mut host = Signature::new(config.default_call_conv);
        host.params.extend([AbiParam::new(types::I64); 2]);
        let host = builder.import_signature(host);
        let code = builder.ins().iconst(types::I64, report_trap as *const () as usize as i64);
        let text = builder.ins().iconst(types::I64, text);
        let size = builder.ins().iconst(types::I64, size);
        builder.ins().call_indirect(host, code, &[text, size]);
        builder.ins().trap(TrapCode::unwrap_user(2));
        builder.finalize(config);
        self.define(id)
    }

    fn lower<'db>(
        &mut self,
        db: &'db dyn HirDatabase,
        env: ParamEnvAndCrate<'db>,
        name: &str,
        body: &'db MirBody<'db>,
        sig: Sig,
        work: Work<'db>,
    ) -> Result<FuncId, String> {
        let native = if let Work::Native(_, native) = &work { Some(native.clone()) } else { None };
        let id = self.declare(sig, native.as_ref())?;
        let config = self.module.target_config();
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder);
        let lower = Lower::new(db, env, body, &mut builder, &mut self.slots, config.default_call_conv, sig);
        let result = match work {
            Work::Body => lower.run(),
            Work::Glue(ty) => lower.glue(ty),
            Work::Thunk(target, data) => lower.thunk(target, data),
            Work::Shim(ty) => lower.shim(ty),
            Work::Native(target, native) => lower.native(target, &native),
            Work::Derive(method, func, args) => lower.derive(method, func, args),
        };
        match result {
            Ok(()) => {
                builder.finalize(config);
                if self.dump {
                    println!("=== {name}\n{}", self.ctx.func.display());
                }
                self.define_named(id, name)
            }
            Err(reason) => {
                drop(builder);
                self.builder = FunctionBuilderContext::new();
                self.module.clear_context(&mut self.ctx);
                Err(reason)
            }
        }
    }
}


pub fn leak_text(text: &str) -> (i64, i64) {
    let text: &'static str = Box::leak(text.to_owned().into_boxed_str());
    (text.as_ptr() as i64, text.len() as i64)
}

pub extern "C" fn report_trap(text: *const u8, size: usize) {
    // SAFETY: text and size come from leak_text, which leaks the string for the process lifetime.
    let text = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(text, size)) };
    eprintln!("trap: {text}");
}

fn native_of(key: &Key) -> Option<&NativeSig> {
    if let Key::Native(_, native) = key { Some(native) } else { None }
}

fn env_of(db: &dyn HirDatabase, func: FunctionId) -> ParamEnvAndCrate<'_> {
    ParamEnvAndCrate { param_env: db.trait_environment(func.into()), krate: func.module(db).krate(db) }
}

fn target_of<'db>(key: &Key) -> Target<'db> {
    match key {
        Key::Func(func, args) => Target::Func(*func, args.as_ref()),
        Key::Closure(closure, args) => Target::Closure(*closure, args.as_ref()),
        Key::Glue(ty) => Target::Glue(ty.as_ref()),
        Key::Thunk(inner, _) | Key::Native(inner, _) => target_of(inner),
        Key::Derive(method, func, args) => Target::Derive(*method, *func, args.as_ref()),
        Key::Const(..) | Key::Shim(_) | Key::Static(_) => Target::Abort,
    }
}

fn key_name(db: &dyn HirDatabase, key: &Key) -> String {
    match key {
        Key::Glue(ty) => format!("drop glue {:?}", ty.as_ref()),
        Key::Const(..) => "constant".to_owned(),
        Key::Static(_) => "static".to_owned(),
        Key::Shim(ty) => format!("shim {:?}", ty.as_ref()),
        Key::Derive(method, _, args) => format!("derive {} {:?}", method.name().as_str(), args.as_ref().as_slice().first()),
        Key::Thunk(inner, _) => format!("thunk {}", key_name(db, inner)),
        Key::Native(inner, _) => format!("native {}", key_name(db, inner)),
        Key::Func(func, args) if !args.as_ref().as_slice().is_empty() => {
            format!("{} {:?}", FunctionSignature::of(db, *func).name.as_str(), args.as_ref().as_slice())
        }
        _ => name_of(db, &target_of(key)),
    }
}

pub fn key_of(target: &Target<'_>) -> Option<Key> {
    match target {
        Target::Func(func, args) => Some(Key::Func(*func, args.store())),
        Target::Closure(closure, args) => {
            // SAFETY: an interned closure id is a plain index valid while the database lives, and the engine owns both for the process lifetime.
            let closure = unsafe { std::mem::transmute::<InternedClosureId<'_>, InternedClosureId<'static>>(*closure) };
            Some(Key::Closure(closure, args.store()))
        }
        Target::Glue(ty) => Some(Key::Glue(ty.store())),
        Target::Derive(method, func, args) => Some(Key::Derive(*method, *func, args.store())),
        Target::Ctor(_) | Target::Intrinsic(..) | Target::Foreign(_) | Target::Abort | Target::Dyn(..) | Target::Pointer => None,
    }
}

pub fn pointer_signature(signature: &mut Signature, sig: Sig) {
    let count = sig.params + usize::from(sig.ret);
    signature.params.extend((0..count).map(|_| AbiParam::new(types::I64)));
}

fn name_of(db: &dyn HirDatabase, target: &Target<'_>) -> String {
    match target {
        Target::Func(func, _) => FunctionSignature::of(db, *func).name.as_str().to_owned(),
        _ => "closure".to_owned(),
    }
}

pub fn body_of<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    target: &Target<'db>,
) -> Result<&'db MirBody<'db>, String> {
    let body = match target {
        Target::Func(func, args) => match db.monomorphized_mir_body((*func).into(), args.store(), env.store()) {
            Err(MirLowerError::HasErrors) => match specialize::parent(db, env, *func, *args) {
                Some((func, args)) => db.monomorphized_mir_body(func.into(), args.store(), env.store()),
                None => Err(MirLowerError::HasErrors),
            },
            other => other,
        },
        Target::Closure(closure, args) => db.monomorphized_mir_body_for_closure(*closure, args.store(), env.store()),
        _ => return Err("not a compiled target".to_owned()),
    };
    body.map_err(|error| format!("{error:?}"))
}

fn target_digest<'db>(db: &'db dyn HirDatabase, env: ParamEnvAndCrate<'db>, target: &Target<'db>) -> u64 {
    let found = match target {
        Target::Func(func, args) => *hir_ty::mir::monomorphized_mir_digest(db, (*func).into(), args.store(), env.store()),
        Target::Closure(closure, args) => *hir_ty::mir::monomorphized_closure_digest(db, *closure, args.store(), env.store()),
        _ => return 1,
    };
    if found != 0 {
        return found;
    }
    body_of(db, env, target).map_or(0, hir_ty::mir::mir_body_digest)
}

fn return_fits<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    body: &MirBody<'db>,
    found: Sig,
    site: Sig,
) -> Result<bool, String> {
    if found.ret == site.ret || found.ret {
        return Ok(true);
    }
    let (_, ret) = body.locals.iter().next().ok_or("body has no return local")?;
    Ok(crate::lower::layout_of(db, env, ret.ty.as_ref())?.is_uninhabited())
}

fn signature_of<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    body: &MirBody<'db>,
) -> Result<Sig, String> {
    let sized = |ty: Ty<'db>| crate::lower::layout_of(db, env, ty).map(|layout| layout.size.bytes() != 0);
    let (_, ret) = body.locals.iter().next().ok_or("body has no return local")?;
    let mut params = 0;
    for param in &body.param_locals {
        params += usize::from(sized(body.locals[*param].ty.as_ref())?);
    }
    Ok(Sig { params, ret: sized(ret.ty.as_ref())? })
}

fn crate_function(db: &dyn HirDatabase, crate_name: &str, name: &str) -> Option<FunctionId> {
    let krate = base_db::all_crates(db).iter().copied().find(|krate| {
        krate.extra_data(db).display_name.as_ref().is_some_and(|it| it.to_string() == crate_name)
    })?;
    crate::free_functions_of(db, krate)
        .into_iter()
        .find(|func| FunctionSignature::of(db, *func).name.as_str() == name && FunctionSignature::of(db, *func).has_body())
}

fn std_panic_handler(db: &dyn HirDatabase) -> Option<FunctionId> {
    crate_function(db, STD_CRATE, STD_PANIC_HANDLER)
}

fn rust_symbol(db: &dyn HirDatabase, symbol: &str) -> Option<FunctionId> {
    let (_, crate_name) = RUST_SYMBOLS.iter().find(|(name, _)| *name == symbol)?;
    crate_function(db, crate_name, symbol)
}

pub fn operand_ty<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    body: &MirBody<'db>,
    operand: &Operand,
) -> Ty<'db> {
    match &operand.kind {
        OperandKind::Copy(place) | OperandKind::Move(place) => {
            let place = place.as_ref();
            let infcx = DbInterner::new_with(db, env.krate).infer_ctxt().build(TypingMode::PostAnalysis);
            let start = PlaceTy::from_ty(body.locals[place.local].ty.as_ref());
            start.multi_projection_ty(&infcx, env.param_env, place.projection.as_slice()).ty
        }
        OperandKind::Constant { ty, .. } => ty.as_ref(),
        OperandKind::Allocation { allocation } => allocation.as_ref().ty,
        OperandKind::Static(id) => hir_ty::InferenceResult::of(db, hir_def::DefWithBodyId::from(*id))
            .expr_ty(hir_def::expr_store::Body::of(db, (*id).into()).root_expr()),
    }
}

pub fn resolve<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    body: &MirBody<'db>,
    callee: &Operand,
) -> Result<Resolved<'db>, String> {
    let ty = operand_ty(db, env, body, callee);
    match ty.kind() {
        TyKind::FnDef(def, args) => resolve_def(db, env, def.0, args),
        TyKind::FnPtr(..) => Ok(Resolved { target: Target::Pointer, shape: Shape::Direct }),
        _ => Err(format!("indirect call through {ty:?}")),
    }
}

pub fn resolve_def<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    def: CallableDefId,
    args: GenericArgs<'db>,
) -> Result<Resolved<'db>, String> {
    let func = match def {
        CallableDefId::FunctionId(func) => func,
        CallableDefId::StructId(id) => return Ok(Resolved { target: Target::Ctor(id.into()), shape: Shape::Direct }),
        CallableDefId::EnumVariantId(id) => return Ok(Resolved { target: Target::Ctor(id.into()), shape: Shape::Direct }),
    };
    let interner = DbInterner::new_with(db, env.krate);
    if is_dyn_method(interner, env.param_env, func, args).is_some() {
        let self_ty = args.as_slice().first().and_then(|arg| arg.ty()).ok_or("dyn call without a self type")?;
        return Ok(Resolved { target: Target::Dyn(func, self_ty), shape: Shape::Direct });
    }
    if !func.module(db).krate(db).data(db).origin.is_local() && FunctionSignature::of(db, func).name.as_str() == DROP_GLUE {
        let ty = args.as_slice().first().and_then(|arg| arg.ty()).ok_or("drop glue without a type")?;
        return Ok(Resolved { target: Target::Glue(ty), shape: Shape::Direct });
    }
    let original = args;
    let (imp, args) = db.lookup_impl_method(env, func, args);
    let imp = match imp {
        Either::Left(imp) => imp,
        Either::Right((_, method)) => {
            return Ok(Resolved { target: Target::Derive(method, func, original), shape: Shape::Direct });
        }
    };
    let name = FunctionSignature::of(db, imp).name.as_str().to_owned();
    if let Some(order) = crate::lower::intrinsic::pointer_order_name(&name)
        && matches!(original.as_slice().first().and_then(|arg| arg.ty()).map(|ty| ty.kind()), Some(TyKind::RawPtr(..)))
    {
        return Ok(Resolved { target: Target::Intrinsic(order.to_owned(), original), shape: Shape::Direct });
    }
    if let Some((_, symbol)) = STD_ALIASES.iter().find(|(alias, _)| *alias == name)
        && imp.module(db).krate(db).data(db).origin.is_lang()
        && crate::lower::intrinsic::symbol(symbol).is_some()
    {
        return Ok(Resolved { target: Target::Foreign((*symbol).to_owned()), shape: Shape::Direct });
    }
    if STD_SYSCALLS.contains(&name.as_str()) && imp.module(db).block(db).is_some() && !imp.module(db).krate(db).data(db).origin.is_local() {
        return Ok(Resolved { target: Target::Foreign(name), shape: Shape::Direct });
    }
    if name == CONST_EVAL_SELECT {
        let chosen = args.as_slice().get(2).and_then(|arg| arg.ty()).ok_or("const_eval_select arguments")?;
        let TyKind::FnDef(def, inner) = chosen.kind() else {
            return Err(format!("const_eval_select of {chosen:?}"));
        };
        let resolved = resolve_def(db, env, def.0, inner)?;
        return Ok(Resolved { target: resolved.target, shape: Shape::Select });
    }
    if crate::lower::intrinsic::is_core_override(&name) && !imp.module(db).krate(db).data(db).origin.is_local() {
        return Ok(Resolved { target: Target::Intrinsic(name, args), shape: Shape::Direct });
    }
    let fallback = FunctionSignature::of(db, imp).has_body() && !crate::lower::intrinsic::is_known(&name);
    if FunctionSignature::is_intrinsic(db, imp) && !fallback {
        return Ok(Resolved { target: Target::Intrinsic(name, args), shape: Shape::Direct });
    }
    if FunctionSignature::of(db, imp).has_body() {
        return Ok(Resolved { target: Target::Func(imp, args), shape: Shape::Direct });
    }
    if !matches!(imp.loc(db).container, ItemContainerId::TraitId(_)) {
        if name == PANIC_HANDLER {
            let target = match std_panic_handler(db) {
                Some(handler) => Target::Func(handler, GenericArgs::empty()),
                None => Target::Abort,
            };
            return Ok(Resolved { target, shape: Shape::Direct });
        }
        let symbol = AttrFlags::link_name(db, imp.into()).map_or(name.clone(), |symbol| symbol.as_str().to_owned());
        if let Some(defined) = rust_symbol(db, &symbol) {
            return Ok(Resolved { target: Target::Func(defined, GenericArgs::empty()), shape: Shape::Direct });
        }
        return Ok(Resolved { target: Target::Foreign(symbol), shape: Shape::Direct });
    }
    if let Some((func, args)) = specialize::specialized(db, env, imp, args) {
        return Ok(Resolved { target: Target::Func(func, args), shape: Shape::Direct });
    }
    let mut callee = args.as_slice().first().and_then(|arg| arg.ty());
    while let Some(TyKind::Ref(_, inner, _)) = callee.map(|ty| ty.kind()) {
        callee = Some(inner);
    }
    if name == CLONE && matches!(callee.map(|ty| ty.kind()), Some(TyKind::FnDef(..) | TyKind::Closure(..))) {
        return Ok(Resolved { target: Target::Derive(Method::clone, func, original), shape: Shape::Direct });
    }
    match callee.map(|ty| ty.kind()) {
        Some(TyKind::FnDef(def, args)) => {
            let inner = resolve_def(db, env, def.0, args)?;
            Ok(Resolved { target: inner.target, shape: Shape::FnTrait })
        }
        Some(TyKind::Closure(closure, args)) => {
            let parent = GenericArgs::new_from_slice(args.as_closure().parent_args());
            Ok(Resolved { target: Target::Closure(closure.0, parent), shape: Shape::FnTrait })
        }
        Some(TyKind::FnPtr(..)) if name != CLONE => Ok(Resolved { target: Target::Pointer, shape: Shape::FnTrait }),
        Some(TyKind::Tuple(_) | TyKind::FnPtr(..)) if name == CLONE => {
            Ok(Resolved { target: Target::Derive(Method::clone, func, original), shape: Shape::Direct })
        }
        other => Err(format!("no body: {name} on {other:?} args {args:?}")),
    }
}
