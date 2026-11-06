use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use cranelift_codegen::Context;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlagsData, Signature, TrapCode, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use either::Either;
use hir_def::builtin_derive::BuiltinDeriveImplMethod as Method;
use hir_def::signatures::FunctionSignature;
use hir_def::{CallableDefId, FunctionId, HasModule, ItemContainerId, StaticId, VariantId};
use hir_ty::traits::StoredParamEnvAndCrate;
use hir_ty::{ParamEnvAndCrate, attach_db};
use hir_ty::db::{HirDatabase, InternedClosureId};
use hir_ty::method_resolution::is_dyn_method;
use hir_ty::mir::{MirBody, Operand, OperandKind, PlaceTy};
use hir_ty::next_solver::infer::DbInternerInferExt;
use hir_ty::next_solver::{DbInterner, GenericArgs, StoredGenericArgs, StoredTy, Ty, TyKind, TypingMode};
use rustc_type_ir::IntTy;
use rustc_type_ir::inherent::{GenericArgs as _, IntoKind};

use hir_ty::mir::MirLowerError;

use crate::lower::Lower;

mod specialize;

const PANIC_HANDLER: &str = "panic_impl";
const STD_CRATE: &str = "std";
const STD_PANIC_HANDLER: &str = "panic_handler";
const CLONE: &str = "clone";
const DROP_GLUE: &str = "drop_glue";
const CONST_EVAL_SELECT: &str = "const_eval_select";
const TABLE_SLOTS: usize = 1 << 20;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Func(FunctionId, StoredGenericArgs),
    Closure(InternedClosureId<'static>, StoredGenericArgs),
    Glue(StoredTy),
    Thunk(Box<Key>, bool),
    Derive(Method, FunctionId, StoredGenericArgs),
    Const(hir_def::ConstId, StoredGenericArgs),
    Shim(StoredTy),
}

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
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sig {
    pub params: usize,
    pub ret: bool,
}

struct SlotInfo {
    key: Key,
    sig: Sig,
    env: StoredParamEnvAndCrate,
}

pub struct Slots {
    table: &'static [AtomicUsize],
    by_key: HashMap<(Key, Sig), usize>,
    info: Vec<SlotInfo>,
    pending: Vec<usize>,
    pub statics: HashMap<StaticId, usize>,
    pub vtables: HashMap<(StoredTy, StoredTy), i64>,
    pub notes: Vec<String>,
}

impl Slots {
    pub fn get(&mut self, key: Key, sig: Sig, env: StoredParamEnvAndCrate) -> Result<usize, String> {
        if let Some(slot) = self.by_key.get(&(key.clone(), sig)) {
            return Ok(*slot);
        }
        let slot = self.info.len();
        if slot == TABLE_SLOTS {
            return Err("call table is full".to_owned());
        }
        self.info.push(SlotInfo { key: key.clone(), sig, env });
        self.by_key.insert((key, sig), slot);
        self.pending.push(slot);
        Ok(slot)
    }

    pub fn address(&self, slot: usize) -> i64 {
        &self.table[slot] as *const AtomicUsize as i64
    }
}

pub struct Backend {
    module: JITModule,
    ctx: Context,
    builder: FunctionBuilderContext,
    slots: Slots,
    digests: HashMap<usize, u64>,
    pub stubs: Vec<(String, String)>,
    pub compiled_total: usize,
    unmapped: Vec<(FuncId, usize, String)>,
    cursor: usize,
    serial: usize,
    dump: bool,
    pub verbose: bool,
}

pub fn call(table: &[AtomicUsize], slot: usize, value: i64) -> i64 {
    let code = table[slot].load(Ordering::Acquire);
    // SAFETY: slot holds finalized code taking (return pointer, argument pointer) for (i64) -> i64, checked by entry, and code memory is never freed.
    let code: extern "C" fn(*mut i64, *const i64) = unsafe { std::mem::transmute(code) };
    let mut result = 0;
    code(&mut result, &value);
    result
}

impl Backend {
    pub fn new(dump: bool, verbose: bool) -> Result<Self> {
        let builder = JITBuilder::new(default_libcall_names()).map_err(|e| anyhow!("{e}"))?;
        let module = JITModule::new(builder);
        let table = (0..TABLE_SLOTS).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
        Ok(Self {
            ctx: module.make_context(),
            module,
            builder: FunctionBuilderContext::new(),
            slots: Slots {
                table: Box::leak(table.into_boxed_slice()),
                by_key: HashMap::new(),
                info: Vec::new(),
                pending: Vec::new(),
                statics: HashMap::new(),
                vtables: HashMap::new(),
                notes: Vec::new(),
            },
            digests: HashMap::new(),
            stubs: Vec::new(),
            compiled_total: 0,
            unmapped: Vec::new(),
            cursor: 0,
            serial: 0,
            dump,
            verbose,
        })
    }

    pub fn table(&self) -> &'static [AtomicUsize] {
        self.slots.table
    }

    pub fn entry<'db>(&mut self, db: &'db dyn HirDatabase, func: FunctionId) -> Result<(usize, bool), String> {
        let env = env_of(db, func);
        let target = Target::Func(func, GenericArgs::empty());
        let body = body_of(db, env, &target)?;
        let sig = signature_of(db, env, body)?;
        let key = key_of(&target).ok_or("entry is not compiled")?;
        let slot = self.slots.get(key, sig, env.store())?;
        self.publish_stubs()?;
        Ok((slot, takes_and_returns_i64(body)))
    }

    pub fn compile<'db>(&mut self, db: &'db dyn HirDatabase, slot: usize) {
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
            Ok((id, digest)) => {
                self.digests.insert(slot, digest);
                self.finish(slot, id);
            }
            Err(reason) => {
                if self.verbose {
                    println!("stub {name}: {reason}");
                }
                let message = format!("{name}: {reason}");
                self.stubs.push((name, reason));
                self.digests.insert(slot, 0);
                if let Ok(id) = self.trap_function(sig, &message) {
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
    ) -> Result<(FuncId, u64), String> {
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
            let body = body_of(db, env, &target).map_err(|why| format!("{}: {why}", name_of(db, &target)))?;
            if self.digests.get(&slot) != Some(&digest(body)) {
                changed.push((slot, name_of(db, &target), body, env));
            }
        }
        let mut names = Vec::with_capacity(changed.len());
        let mut done = Vec::with_capacity(changed.len());
        for (slot, name, body, env) in changed {
            let sig = self.slots.info[slot].sig;
            if signature_of(db, env, body)? != sig {
                return Err(format!("{name}: signature changed"));
            }
            let (id, digest) = self.lower(db, env, &name, body, sig, Work::Body).map_err(|why| format!("{name}: {why}"))?;
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
        for (id, size, name) in std::mem::take(&mut self.unmapped) {
            crate::fault::remember(self.module.get_finalized_function(id) as usize, size, name);
        }
        for (slot, id) in made {
            let code = self.module.get_finalized_function(id);
            self.slots.table[slot].store(code as usize, Ordering::Release);
        }
        Ok(())
    }

    fn declare(&mut self, sig: Sig) -> Result<FuncId, String> {
        let name = format!("f{}", self.serial);
        self.serial += 1;
        let mut signature = self.module.make_signature();
        pointer_signature(&mut signature, sig);
        let id = self.module.declare_function(&name, Linkage::Local, &signature).map_err(|e| e.to_string())?;
        self.ctx.func.signature = signature;
        Ok(id)
    }

    fn define(&mut self, id: FuncId) -> Result<FuncId, String> {
        let defined = self.module.define_function(id, &mut self.ctx).map_err(|e| format!("{e:?}"));
        self.module.clear_context(&mut self.ctx);
        defined.map(|()| id)
    }

    fn define_named(&mut self, id: FuncId, name: &str) -> Result<FuncId, String> {
        self.module.define_function(id, &mut self.ctx).map_err(|e| format!("{e:?}"))?;
        let size = self.ctx.compiled_code().map_or(0, |code| code.code_info().total_size as usize);
        self.unmapped.push((id, size, name.to_owned()));
        self.module.clear_context(&mut self.ctx);
        Ok(id)
    }

    fn compile_stub(&mut self, slot: usize) -> Result<FuncId, String> {
        let sig = self.slots.info[slot].sig;
        let id = self.declare(sig)?;
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
        pointer_signature(&mut target, sig);
        let target = builder.import_signature(target);
        let address = builder.ins().iconst(types::I64, address);
        let code = builder.ins().load(types::I64, MemFlagsData::trusted(), address, 0);
        builder.ins().call_indirect(target, code, &params);
        builder.ins().return_(&[]);
        builder.finalize(config);
        self.define(id)
    }

    fn trap_function(&mut self, sig: Sig, reason: &str) -> Result<FuncId, String> {
        let id = self.declare(sig)?;
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
    ) -> Result<(FuncId, u64), String> {
        let id = self.declare(sig)?;
        let config = self.module.target_config();
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder);
        let lower = Lower::new(db, env, body, &mut builder, &mut self.slots, config.default_call_conv, sig);
        let result = match work {
            Work::Body => lower.run(),
            Work::Glue(ty) => lower.glue(ty),
            Work::Thunk(target, data) => lower.thunk(target, data),
            Work::Shim(ty) => lower.shim(ty),
            Work::Derive(method, func, args) => lower.derive(method, func, args),
        };
        match result {
            Ok(()) => {
                builder.finalize(config);
                if self.dump {
                    println!("=== {name}\n{}", self.ctx.func.display());
                }
                let id = self.define_named(id, name)?;
                Ok((id, digest(body)))
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

pub extern "C" fn compile_slot(slot: u64) {
    let mut guard = crate::patch::ENGINE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let engine = guard.as_mut().expect("engine is running");
    let (db, backend) = engine.parts();
    let db: &dyn HirDatabase = db;
    if !backend.digests.contains_key(&(slot as usize)) {
        attach_db(db, || backend.compile(db, slot as usize));
    }
}

fn env_of(db: &dyn HirDatabase, func: FunctionId) -> ParamEnvAndCrate<'_> {
    ParamEnvAndCrate { param_env: db.trait_environment(func.into()), krate: func.module(db).krate(db) }
}

fn target_of<'db>(key: &Key) -> Target<'db> {
    match key {
        Key::Func(func, args) => Target::Func(*func, args.as_ref()),
        Key::Closure(closure, args) => Target::Closure(*closure, args.as_ref()),
        Key::Glue(ty) => Target::Glue(ty.as_ref()),
        Key::Thunk(inner, _) => target_of(inner),
        Key::Derive(method, func, args) => Target::Derive(*method, *func, args.as_ref()),
        Key::Const(..) | Key::Shim(_) => Target::Abort,
    }
}

fn key_name(db: &dyn HirDatabase, key: &Key) -> String {
    match key {
        Key::Glue(ty) => format!("drop glue {:?}", ty.as_ref()),
        Key::Const(..) => "constant".to_owned(),
        Key::Shim(ty) => format!("shim {:?}", ty.as_ref()),
        Key::Derive(method, _, args) => format!("derive {} {:?}", method.name().as_str(), args.as_ref().as_slice().first()),
        Key::Thunk(inner, _) => format!("thunk {}", key_name(db, inner)),
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

fn takes_and_returns_i64(body: &MirBody<'_>) -> bool {
    let long = |ty: Ty<'_>| matches!(ty.kind(), TyKind::Int(IntTy::I64));
    let ret = body.locals.iter().next().is_some_and(|(_, local)| long(local.ty.as_ref()));
    let [param] = body.param_locals[..] else {
        return false;
    };
    ret && long(body.locals[param].ty.as_ref())
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

fn digest(body: &MirBody<'_>) -> u64 {
    let mut hasher = DefaultHasher::new();
    format!("{body:?}").hash(&mut hasher);
    hasher.finish()
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

fn std_panic_handler(db: &dyn HirDatabase) -> Option<FunctionId> {
    let krate = base_db::all_crates(db).iter().copied().find(|krate| {
        krate.extra_data(db).display_name.as_ref().is_some_and(|name| name.to_string() == STD_CRATE)
    })?;
    crate::free_functions_of(db, krate)
        .into_iter()
        .find(|func| FunctionSignature::of(db, *func).name.as_str() == STD_PANIC_HANDLER)
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
        return Ok(Resolved { target: Target::Foreign(name), shape: Shape::Direct });
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
