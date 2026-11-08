use hir_def::attrs::AttrFlags;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};

use anyhow::{Result, anyhow};
use haste_codegen::{AbiParam, CallConv, Memory, Signature, Type};
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

use crate::lower::native::NativeSig;

mod build;
mod compiler;
mod entry;
mod slots;
mod specialize;

pub use compiler::{compile_slot, quiesce, watch_forks};
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

struct State {
    memory: Memory,
    digests: HashMap<usize, u64>,
    claimed: std::collections::HashSet<usize>,
    stubs: Vec<(String, String)>,
    order: Vec<usize>,
    compiled_total: usize,
    unmapped: Vec<(usize, usize, String, Vec<crate::fault::Site>)>,
    cursor: usize,
}

/// The JIT: the slot table, code memory and what is compiled, shared by every thread that compiles.
pub struct Backend {
    slots: Mutex<Slots>,
    state: Mutex<State>,
    done: Condvar,
    publish: Mutex<()>,
    table: &'static [AtomicUsize],
    dump: bool,
    pub verbose: bool,
}

/// The calling convention of generated functions, which lets a try_call handler assume no preserved registers.
pub const INTERNAL_CONV: CallConv = CallConv::Internal;

const STUB_NAME: &str = "stub";

extern "C" fn missing_code() -> ! {
    eprintln!("haste: called a function whose code and trap both failed to build; the stub list names it");
    std::process::abort()
}

pub use entry::{call, enter};

std::thread_local! {
    static NOTES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Records why the function being compiled on this thread got a stub in place of some call.
pub fn note(text: String) {
    NOTES.with_borrow_mut(|notes| notes.push(text));
}

/// How many notes this thread has recorded, to drop the ones an abandoned inline attempt made.
pub fn note_count() -> usize {
    NOTES.with_borrow(Vec::len)
}

/// Drops this thread's notes after the first `count`.
pub fn truncate_notes(count: usize) {
    NOTES.with_borrow_mut(|notes| notes.truncate(count));
}

fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Backend {
    pub fn new(dump: bool, verbose: bool) -> Result<Self> {
        let table: &'static [AtomicUsize] = Box::leak((0..TABLE_SLOTS).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>().into_boxed_slice());
        let slots = Slots {
            table,
            by_key: HashMap::new(),
            info: Vec::new(),
            pending: Vec::new(),
            native_words: Vec::new(),
            statics: HashMap::new(),
            lazy_statics: HashMap::new(),
            thread_locals: HashMap::new(),
            vtables: HashMap::new(),
        };
        let state = State {
            memory: Memory::default(),
            digests: HashMap::new(),
            claimed: std::collections::HashSet::new(),
            stubs: Vec::new(),
            order: Vec::new(),
            compiled_total: 0,
            unmapped: Vec::new(),
            cursor: 0,
        };
        let backend = Self {
            slots: Mutex::new(slots),
            state: Mutex::new(state),
            done: Condvar::new(),
            publish: Mutex::new(()),
            table,
            dump,
            verbose,
        };
        backend.compile_entry().map_err(|e| anyhow!("{e}"))?;
        Ok(backend)
    }

    fn slots(&self) -> MutexGuard<'_, Slots> {
        locked(&self.slots)
    }

    fn state(&self) -> MutexGuard<'_, State> {
        locked(&self.state)
    }

    pub fn functions(&self) -> Vec<FunctionId> {
        self.slots().functions()
    }

    pub fn table(&self) -> &'static [AtomicUsize] {
        self.table
    }

    /// How many slots exist; the slots one compile adds are the callees it found.
    pub fn slot_count(&self) -> usize {
        self.slots().info.len()
    }

    /// Whether `slot` has code or a trap.
    pub fn compiled(&self, slot: usize) -> bool {
        self.state().digests.contains_key(&slot)
    }

    pub fn compiled_total(&self) -> usize {
        self.state().compiled_total
    }

    pub fn take_stubs(&self) -> Vec<(String, String)> {
        std::mem::take(&mut self.state().stubs)
    }

    pub fn entry<'db>(&self, db: &'db dyn HirDatabase, func: FunctionId) -> Result<(usize, bool), String> {
        let (slot, body) = self.instance(db, func, GenericArgs::empty(), func)?;
        Ok((slot, entry::takes_and_returns_i64(body)))
    }

    /// Gives the table slot of `func` with `args`, resolving impls as `scope` sees them, called through `enter`.
    pub fn instance<'db>(
        &self,
        db: &'db dyn HirDatabase,
        func: FunctionId,
        args: GenericArgs<'db>,
        scope: FunctionId,
    ) -> Result<(usize, &'db MirBody<'db>), String> {
        let env = env_of(db, scope);
        let target = Target::Func(func, args);
        let body = body_of(db, env, &target)?;
        let sig = Sig { by_value: 0, ..signature_of(db, env, body)? };
        let key = key_of(&target).ok_or("entry is not compiled")?;
        let slot = self.slots().get(key, sig, env.store())?;
        self.publish_stubs()?;
        Ok((slot, body))
    }

    pub fn replay<'db>(&self, db: &'db dyn HirDatabase, order: &[usize]) {
        for slot in order.iter().copied() {
            if slot < self.slot_count() {
                self.compile(db, slot);
            }
        }
        self.state().order.clear();
    }

    pub fn take_order(&self) -> Vec<usize> {
        std::mem::take(&mut self.state().order)
    }

    pub fn next_uncompiled(&self) -> Option<usize> {
        let count = self.slot_count();
        let mut state = self.state();
        while state.cursor < count {
            let slot = state.cursor;
            state.cursor += 1;
            if !state.digests.contains_key(&slot) {
                return Some(slot);
            }
        }
        None
    }

    pub fn mark_failed(&self, slot: usize) {
        self.state().digests.insert(slot, 0);
        self.table[slot].store(missing_code as *const () as usize, Ordering::Release);
    }

    pub fn refresh<'db>(&self, db: &'db dyn HirDatabase) -> Result<Vec<String>, String> {
        let mut changed = Vec::new();
        let digests: Vec<(usize, u64)> = self.state().digests.iter().map(|(slot, digest)| (*slot, *digest)).collect();
        for (slot, old) in digests {
            let (key, env) = {
                let slots = self.slots();
                (slots.info[slot].key.clone(), slots.info[slot].env.clone())
            };
            if !matches!(key, Key::Func(..) | Key::Closure(..)) {
                continue;
            }
            let env = env.as_ref(db);
            let target = target_of(&key);
            let digest = target_digest(db, env, &target);
            if old == digest {
                continue;
            }
            let body = body_of(db, env, &target).map_err(|why| format!("{}: {why}", name_of(db, &target)))?;
            changed.push((slot, name_of(db, &target), body, env, digest));
        }
        let mut names = Vec::with_capacity(changed.len());
        let mut done = Vec::with_capacity(changed.len());
        for (slot, name, body, env, digest) in changed {
            let sig = self.slots().info[slot].sig;
            let found = signature_of(db, env, body)?;
            if found.params != sig.params || found.ret != sig.ret || sig.by_value & !found.by_value != 0 {
                return Err(format!("{name}: signature changed"));
            }
            let id = self.lower(db, env, &name, body, sig, Work::Body).map_err(|why| format!("{name}: {why}"))?;
            done.push((slot, id, digest));
            names.push(name);
        }
        self.publish_stubs()?;
        for (slot, id, digest) in done {
            self.state().digests.insert(slot, digest);
            self.table[slot].store(id, Ordering::Release);
        }
        Ok(names)
    }

}

fn signature_for(sig: Sig, native: Option<&NativeSig>) -> Signature {
    let mut signature = Signature::new(CallConv::Native);
    match native {
        Some(native) => native.signature(&mut signature),
        None => {
            signature.call_conv = INTERNAL_CONV;
            pointer_signature(&mut signature, sig);
        }
    }
    signature
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

pub(super) fn env_of(db: &dyn HirDatabase, func: FunctionId) -> ParamEnvAndCrate<'_> {
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
    signature.params.extend((0..count).map(|_| AbiParam::new(Type::I64)));
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

fn site_fits<'db>(db: &'db dyn HirDatabase, env: ParamEnvAndCrate<'db>, body: &MirBody<'db>, sig: Sig) -> Result<(), String> {
    let found = signature_of(db, env, body)?;
    if found.params == sig.params && sig.by_value & !found.by_value == 0 && return_fits(db, env, body, found, sig)? {
        return Ok(());
    }
    let params: Vec<String> = body.locals.iter().map(|(_, it)| format!("{:?}", it.ty.as_ref())).take(body.param_locals.len() + 1).collect();
    Err(format!(
        "call site and callee disagree on the signature: site {}+{} {:#x} callee {}+{} {:#x} ret,params {params:?}",
        sig.params, sig.ret, sig.by_value, found.params, found.ret, found.by_value
    ))
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
    let mut sig = Sig::new(0, sized(ret.ty.as_ref())?);
    for param in &body.param_locals {
        let layout = crate::lower::layout_of(db, env, body.locals[*param].ty.as_ref())?;
        if layout.size.bytes() == 0 {
            continue;
        }
        if passes_by_value(&layout) && sig.params < 64 {
            sig.by_value |= 1 << sig.params;
        }
        sig.params += 1;
    }
    Ok(sig)
}

/// Whether a parameter of this layout travels as its 8-byte integer or pointer value instead of its address.
pub fn passes_by_value(layout: &hir_ty::layout::Layout) -> bool {
    let rustc_abi::BackendRepr::Scalar(scalar) = layout.backend_repr else {
        return false;
    };
    layout.size.bytes() == 8 && matches!(scalar.primitive(), rustc_abi::Primitive::Int(..) | rustc_abi::Primitive::Pointer(_))
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
