use std::sync::atomic::{AtomicUsize, Ordering};

use haste_codegen::{AbiParam, CallConv, Function, FunctionBuilder, Signature, Type};
use hir_ty::ParamEnvAndCrate;
use hir_ty::db::HirDatabase;
use hir_ty::mir::MirBody;
use hir_ty::next_solver::GenericArgs;

use super::{
    Backend, Key, NOTES, STUB_NAME, Sig, Work, body_of, compile_slot, key_name, leak_text, locked, missing_code, native_of, report_trap,
    signature_for, site_fits, target_digest, target_of,
};
use crate::lower::Lower;
use crate::lower::native::NativeSig;

pub(super) struct Claim<'a> {
    backend: &'a Backend,
    slot: usize,
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        locked(&self.backend.state).claimed.remove(&self.slot);
        self.backend.done.notify_all();
    }
}

impl Backend {
    pub(super) fn claim(&self, slot: usize, wait: bool) -> Option<Claim<'_>> {
        let mut state = self.state();
        loop {
            if state.digests.contains_key(&slot) {
                return None;
            }
            if state.claimed.insert(slot) {
                return Some(Claim { backend: self, slot });
            }
            if !wait {
                return None;
            }
            state = self.done.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Compiles `slot` unless it has code, waiting when another thread is compiling it.
    pub fn compile<'db>(&self, db: &'db dyn HirDatabase, slot: usize) {
        if let Some(claim) = self.claim(slot, true) {
            self.compile_claimed(db, claim);
        }
    }

    /// Compiles `slot` ahead of its first call, unless it has code or another thread has it.
    pub fn compile_ahead<'db>(&self, db: &'db dyn HirDatabase, slot: usize) {
        if let Some(claim) = self.claim(slot, false) {
            self.compile_claimed(db, claim);
        }
    }

    pub(super) fn compile_claimed<'db>(&self, db: &'db dyn HirDatabase, claim: Claim<'_>) {
        let slot = claim.slot;
        let (key, sig, env) = {
            let slots = self.slots();
            let info = &slots.info[slot];
            (info.key.clone(), info.sig, info.env.clone())
        };
        let env = env.as_ref(db);
        let name = key_name(db, &key);
        let lowered = self.compile_key(db, env, &name, &key, sig);
        let notes = NOTES.with_borrow_mut(std::mem::take);
        let (id, digest) = match lowered {
            Ok(id) => (id, target_digest(db, env, &target_of(&key))),
            Err(reason) => (self.trap(&name, reason, sig, &key), 0),
        };
        {
            let mut state = self.state();
            state.order.push(slot);
            for note in notes {
                if self.verbose {
                    println!("stub {name}: {note}");
                }
                state.stubs.push((name.clone(), note));
            }
        }
        self.finish(slot, id);
        self.state().digests.insert(slot, digest);
        drop(claim);
    }

    pub(super) fn trap(&self, name: &str, reason: String, sig: Sig, key: &Key) -> usize {
        if self.verbose {
            println!("stub {name}: {reason}");
        }
        let message = format!("{name}: {reason}");
        self.state().stubs.push((name.to_owned(), reason));
        self.trap_function(sig, native_of(key), &message).unwrap_or_else(|why| {
            self.state().stubs.push(("trap".to_owned(), why));
            missing_code as *const () as usize
        })
    }

    pub(super) fn compile_key<'db>(
        &self,
        db: &'db dyn HirDatabase,
        env: ParamEnvAndCrate<'db>,
        name: &str,
        key: &Key,
        sig: Sig,
    ) -> Result<usize, String> {
        let entry = || self.slots().info.first().map(|info| info.key.clone()).ok_or("no entry");
        match key {
            Key::Func(..) | Key::Closure(..) => {
                let target = target_of(key);
                let body = body_of(db, env, &target)?;
                site_fits(db, env, body, sig)?;
                self.lower(db, env, name, body, sig, Work::Body)
            }
            Key::Glue(ty) => {
                let body = body_of(db, env, &target_of(&entry()?))?;
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
                let body = body_of(db, env, &target_of(&entry()?))?;
                self.lower(db, env, name, body, sig, Work::Derive(*method, *func, args.as_ref()))
            }
            Key::Shim(ty) => {
                let body = body_of(db, env, &target_of(&entry()?))?;
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

    pub(super) fn finish(&self, slot: usize, id: usize) {
        self.state().compiled_total += 1;
        let id = match self.publish_stubs() {
            Ok(()) => id,
            Err(reason) => {
                self.state().stubs.push(("table".to_owned(), reason));
                missing_code as *const () as usize
            }
        };
        self.table[slot].store(id, Ordering::Release);
    }

    pub(super) fn publish_stubs(&self) -> Result<(), String> {
        let _publishing = locked(&self.publish);
        let pending = std::mem::take(&mut self.slots().pending);
        let mut made = Vec::with_capacity(pending.len());
        for (done, slot) in pending.iter().copied().enumerate() {
            match self.compile_stub(slot) {
                Ok(code) => made.push((slot, code)),
                Err(reason) => {
                    self.slots().pending.extend_from_slice(&pending[done..]);
                    return Err(reason);
                }
            }
        }
        for (code, size, name, sites) in std::mem::take(&mut self.state().unmapped) {
            crate::fault::remember(code, size, name, sites);
        }
        for (slot, code) in made {
            self.table[slot].store(code, Ordering::Release);
        }
        for (word, slot) in std::mem::take(&mut self.slots().native_words) {
            let code = self.table[slot].load(Ordering::Acquire);
            // SAFETY: word is a function pointer field inside constant data this engine leaked, and nothing runs that data until publish returns.
            unsafe { (word as *mut usize).write_unaligned(code) };
        }
        Ok(())
    }

    pub(super) fn define(&self, func: &Function, name: &str) -> Result<usize, String> {
        let compiled = haste_codegen::compile(func)?;
        let mut state = self.state();
        let code = state.memory.write(&compiled.code)?;
        state.unmapped.push((code, compiled.code.len(), name.to_owned(), compiled.sites));
        Ok(code)
    }

    pub(super) fn compile_stub(&self, slot: usize) -> Result<usize, String> {
        let (sig, native) = {
            let slots = self.slots();
            (slots.info[slot].sig, native_of(&slots.info[slot].key).cloned())
        };
        let signature = signature_for(sig, native.as_ref());
        let address = self.table_address(slot);
        let mut func = Function::new(signature.clone());
        let mut builder = FunctionBuilder::new(&mut func);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        let params = builder.block_params(block).to_vec();
        let mut host = Signature::new(CallConv::Native);
        host.params.push(AbiParam::new(Type::I64));
        let host = builder.import_signature(host);
        let code = builder.ins().iconst(Type::I64, compile_slot as *const () as usize as i64);
        let number = builder.ins().iconst(Type::I64, slot as i64);
        builder.ins().call_indirect(host, code, &[number]);
        let target = builder.import_signature(signature);
        let address = builder.ins().iconst(Type::I64, address);
        let code = builder.ins().load(Type::I64, address, 0);
        let call = builder.ins().call_indirect(target, code, &params);
        let results = builder.inst_results(call).to_vec();
        builder.ins().return_(&results);
        self.define(&func, STUB_NAME)
    }

    pub(super) fn table_address(&self, slot: usize) -> i64 {
        &self.table[slot] as *const AtomicUsize as i64
    }

    pub(super) fn trap_function(&self, sig: Sig, native: Option<&NativeSig>, reason: &str) -> Result<usize, String> {
        let mut func = Function::new(signature_for(sig, native));
        let mut builder = FunctionBuilder::new(&mut func);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        let (text, size) = leak_text(reason);
        let mut host = Signature::new(CallConv::Native);
        host.params.extend([AbiParam::new(Type::I64); 2]);
        let host = builder.import_signature(host);
        let code = builder.ins().iconst(Type::I64, report_trap as *const () as usize as i64);
        let text = builder.ins().iconst(Type::I64, text);
        let size = builder.ins().iconst(Type::I64, size);
        builder.ins().call_indirect(host, code, &[text, size]);
        builder.ins().trap();
        self.define(&func, STUB_NAME)
    }

    pub(super) fn lower<'db>(
        &self,
        db: &'db dyn HirDatabase,
        env: ParamEnvAndCrate<'db>,
        name: &str,
        body: &'db MirBody<'db>,
        sig: Sig,
        work: Work<'db>,
    ) -> Result<usize, String> {
        let native = if let Work::Native(_, native) = &work { Some(native.clone()) } else { None };
        let mut func = Function::new(signature_for(sig, native.as_ref()));
        let mut builder = FunctionBuilder::new(&mut func);
        let lower = Lower::new(db, env, body, &mut builder, &self.slots, CallConv::Native, sig);
        match work {
            Work::Body => lower.run(),
            Work::Glue(ty) => lower.glue(ty),
            Work::Thunk(target, data) => lower.thunk(target, data),
            Work::Shim(ty) => lower.shim(ty),
            Work::Native(target, native) => lower.native(target, &native),
            Work::Derive(method, func, args) => lower.derive(method, func, args),
        }?;
        if self.dump {
            println!("=== {name}\n{func}");
        }
        self.define(&func, name)
    }
}
