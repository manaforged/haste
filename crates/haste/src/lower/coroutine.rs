//! Lowering of coroutines: the object an `async` block or body makes, and its `Future::poll`, which suspends at each `.await` and which `haste_codegen` splits into a state machine.

use anyhow::{Result, bail};
use haste_codegen::{Block, Coroutine, IntCC, StackSlot, Type, Value};

use crate::body::{Expr, ExprId};
use crate::sema::infer::{Awaited, Res};
use crate::sema::ty::{Kind, Ty};

use super::moves::Root;
use super::place::{Base, Place};
use super::{Lower, Repr, repr};

/// The resume function being lowered: where it keeps the result pointer of the current call, the blocks it resumes at, and the blocks that end it.
pub(super) struct Resume {
    sret: Option<StackSlot>,
    header: (i32, i32),
    resumes: Vec<Block>,
    done: Vec<Block>,
}

impl Lower<'_, '_> {
    /// Lowers the body of a coroutine's `Future::poll` from its start, binding its captures in the coroutine object.
    pub(super) fn resume_root(&mut self, body: ExprId) -> Result<Option<Value>> {
        let sret = self.sret.map(|value| {
            let slot = self.b.create_stack_slot(8, 3);
            self.ins().stack_store(value, slot, 0);
            slot
        });
        let local = self.typed.params[0];
        let Kind::Adt(_, args) = *self.cx.types.kind(self.typed.locals[local as usize].ty) else { bail!("a resume function with no pinned coroutine") };
        let first = self.cx.types.substs(args).first().copied();
        let Some(&Kind::Ref(_, coroutine)) = first.map(|pointer| self.cx.types.kind(pointer)) else { bail!("a resume function whose pin holds no reference") };
        let header = self.header_of(coroutine)?;
        self.resume = Some(Resume { sret, header, resumes: Vec::new(), done: Vec::new() });
        self.bind_captures()?;
        let Expr::Block { stmts, tail, .. } = *self.body.expr(body) else { bail!("an `async` body that is not a block") };
        self.block(stmts, tail)
    }

    pub(super) fn finish_resume(&mut self) -> Result<()> {
        let Some(resume) = self.resume.take() else { return Ok(()) };
        let object = usize::from(resume.sret.is_some());
        self.b.func.coroutine = Some(Coroutine { object, header: resume.header, resumes: resume.resumes, done: resume.done });
        Ok(())
    }

    /// The address of the coroutine object, read from the `Pin<&mut Self>` parameter of the current call.
    fn object(&mut self) -> Result<Value> {
        let local = self.typed.params[0];
        let Some(pin) = self.locals[local as usize] else { bail!("a resume function with no pinned coroutine") };
        let pin_ty = self.typed.locals[local as usize].ty;
        let pointer = self.part_place(pin, pin_ty, (0, 0))?;
        self.read(pointer).ok_or_else(|| anyhow::anyhow!("a pinned coroutine with no address"))
    }

    fn set_state(&mut self, state: i64) -> Result<()> {
        let object = self.object()?;
        let Some(at) = self.resume.as_ref().map(|resume| resume.header.0) else { bail!("a coroutine state outside a resume function") };
        let value = self.ins().iconst(Type::I64, state);
        self.ins().store(value, object, at);
        Ok(())
    }

    /// Ends the coroutine with `value`: returns `Poll::Ready(value)` after dropping what the body owns, and frees the frame.
    pub(super) fn resume_finish(&mut self, value: Option<Value>) -> Result<()> {
        let ready = self.poll_value("Ready", value)?;
        self.drop_to(0)?;
        self.set_state(Coroutine::RETURNED)?;
        self.return_poll(ready, true)
    }

    fn poll_value(&mut self, variant: &str, payload: Option<Value>) -> Result<Option<Value>> {
        let Some(poll) = self.typed.ret else { bail!("a resume function with no result") };
        let index = self.variant_index(poll, variant)?;
        self.construct(poll, index, payload.map(|value| (0, Some(value))).into_iter().collect(), None)
    }

    fn variant_index(&mut self, ty: Ty, name: &str) -> Result<u32> {
        let Kind::Adt(item, _) = *self.cx.types.kind(ty) else { bail!("`Poll` is not an enum") };
        let def = self.cx.adt_def(item)?;
        let src = self.cx.item(item).0;
        let found = def.variants.iter().find(|variant| crate::names::plain(src.text(variant.name)) == name);
        found.map(|variant| variant.index).ok_or_else(|| anyhow::anyhow!("`Poll` has no `{name}`"))
    }

    fn return_poll(&mut self, value: Option<Value>, done: bool) -> Result<()> {
        if !self.open {
            return Ok(());
        }
        let block = self.b.create_block();
        self.jump(block, &[]);
        self.enter(block);
        let Some(poll) = self.typed.ret else { bail!("a resume function with no result") };
        let sret = self.resume.as_ref().and_then(|resume| resume.sret);
        match (sret, value, repr(self.cx, poll)?) {
            (Some(slot), Some(value), whole @ Repr::Mem { .. }) => {
                let out = self.ins().stack_load(Type::I64, slot, 0);
                self.copy(out, value, whole);
                self.ins().return_(&[]);
            }
            (None, Some(value), Repr::Scalar(_) | Repr::Mem { .. }) => {
                let values = self.returned_values(poll, value)?;
                self.ins().return_(&values);
            }
            (_, None, Repr::Zst) => {
                self.ins().return_(&[]);
            }
            _ => bail!("a `Poll` result that does not fit the resume signature"),
        }
        self.close();
        if done && let Some(resume) = &mut self.resume {
            resume.done.push(block);
        }
        Ok(())
    }

    /// Lowers `inner.await`: makes the future, then polls it, suspending with `Poll::Pending` until it is ready, and gives its output.
    pub(super) fn await_expr(&mut self, expr: ExprId, inner: ExprId) -> Result<Option<Value>> {
        let Some(awaited) = self.typed.awaits.get(&expr).cloned() else { bail!("an `.await` with no inferred future") };
        if self.resume.is_none() {
            bail!("an `.await` outside an `async` body");
        }
        let value = self.expr(inner)?;
        let Res::Fn(into, into_substs) = awaited.into_future else { bail!("an `.await` with no `into_future`") };
        let future = self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::AwaitInto(expr)), into, into_substs, value.into_iter().collect())?;
        let held = repr(self.cx, awaited.future)?;
        let place = self.slot(held).map(|slot| Place::slot(slot, held));
        if let (Some(place), Some(future)) = (place, future) {
            self.write(place, future);
        }
        self.own(Root::Held(expr), awaited.future, place, true)?;
        let head = self.b.create_block();
        self.jump(head, &[]);
        self.enter(head);
        let polled = self.poll_once(expr, &awaited, place)?;
        let ready = self.variant_index(awaited.polled, "Ready")?;
        let Some(is_ready) = self.is_variant(polled, awaited.polled, ready)? else { bail!("`Poll` has one variant") };
        let (done, pending) = (self.b.create_block(), self.b.create_block());
        self.brif(is_ready, (done, &[]), (pending, &[]));
        self.enter(pending);
        self.suspend(head)?;
        self.enter(done);
        let Some(polled) = polled else {
            return Ok(None);
        };
        let output = self.variant_field(polled, awaited.polled, (ready, 0))?;
        Ok(self.read(output))
    }

    fn poll_once(&mut self, expr: ExprId, awaited: &Awaited, future: Option<Place>) -> Result<Option<Place>> {
        let Res::Fn(poll, substs) = awaited.poll else { bail!("an `.await` with no `poll`") };
        let address = match future {
            Some(place) => self.address(place),
            None => self.ins().iconst(Type::I64, 1),
        };
        let pin = self.pin_of(address)?;
        let Some(context) = self.locals[self.typed.params[1] as usize].and_then(|place| self.read(place)) else { bail!("a resume function with no task context") };
        let polled = self.invoke(super::reloc::SiteOrigin::Table(crate::sema::infer::SiteKey::AwaitPoll(expr)), poll, substs, vec![pin, context])?;
        let shape = repr(self.cx, awaited.polled)?;
        Ok(match (shape, polled) {
            (Repr::Mem { .. }, Some(address)) => Some(Place { base: Base::Addr(address), offset: 0, repr: shape }),
            (Repr::Scalar(_), Some(value)) => {
                let slot = self.slot(shape).expect("a scalar has a slot");
                self.ins().stack_store(value, slot, 0);
                Some(Place::slot(slot, shape))
            }
            _ => None,
        })
    }

    /// The `Pin<&mut _>` argument for the pointer `address`: the address of a slot that holds it.
    fn pin_of(&mut self, address: Value) -> Result<Value> {
        let slot = self.b.create_stack_slot(8, 3);
        self.ins().stack_store(address, slot, 0);
        Ok(self.ins().stack_addr(Type::I64, slot, 0))
    }

    /// Suspends: records the resume point, returns `Poll::Pending`, and on resume takes the new call's arguments, then drops everything and ends when the call drops the coroutine, or polls again at `head`.
    fn suspend(&mut self, head: Block) -> Result<()> {
        let Some(count) = self.resume.as_ref().map(|resume| resume.resumes.len()) else { bail!("a suspension outside a resume function") };
        self.set_state(Coroutine::FIRST_SUSPEND + count as i64)?;
        let pending = self.poll_value("Pending", None)?;
        self.return_poll(pending, false)?;
        let block = self.b.create_block();
        let types: Vec<Type> = self.b.func.signature.params.iter().map(|param| param.value_type).collect();
        let params: Vec<Value> = types.into_iter().map(|ty| self.b.append_block_param(block, ty)).collect();
        if let Some(resume) = &mut self.resume {
            resume.resumes.push(block);
        }
        self.b.switch_to_block(block);
        self.open = true;
        let context = self.take_call(&params)?;
        let dropped = self.ins().icmp_imm(IntCC::Equal, context, 0);
        let (dropping, again) = (self.b.create_block(), self.b.create_block());
        self.brif(dropped, (dropping, &[]), (again, &[]));
        self.enter(dropping);
        self.drop_to(0)?;
        self.set_state(Coroutine::RETURNED)?;
        let pending = self.poll_value("Pending", None)?;
        self.return_poll(pending, true)?;
        self.enter(again);
        self.jump(head, &[]);
        self.close();
        Ok(())
    }

    fn take_call(&mut self, params: &[Value]) -> Result<Value> {
        let mut params = params.iter().copied();
        if let Some(slot) = self.resume.as_ref().and_then(|resume| resume.sret) {
            let Some(out) = params.next() else { bail!("a resume block with no result pointer") };
            self.ins().stack_store(out, slot, 0);
        }
        let mut last = None;
        for at in 0..2 {
            let local = self.typed.params[at];
            let Some(place) = self.locals[local as usize] else { bail!("a resume block with missing arguments") };
            last = self.store_passed(local, place, &mut params)?;
        }
        last.ok_or_else(|| anyhow::anyhow!("a resume block with no task context"))
    }

    /// The drop glue of a coroutine at `target`: the captures when it never ran, a resume in drop mode when it is suspended, nothing when it ended.
    pub(super) fn coroutine_glue(&mut self, ty: Ty, target: Value, resume: Res) -> Result<()> {
        let Res::Fn(poll, substs) = resume else { bail!("a coroutine whose glue has no `poll`") };
        let header = self.header_of(ty)?;
        let state = self.ins().load(Type::I64, target, header.0);
        let fresh = self.ins().icmp_imm(IntCC::Equal, state, Coroutine::UNRESUMED);
        let (unresumed, check, suspended, out) = (self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block());
        self.brif(fresh, (unresumed, &[]), (check, &[]));
        self.enter(unresumed);
        let (inner, offset) = self.captures_of(ty)?;
        let place = Place { base: Base::Addr(target), offset, repr: repr(self.cx, inner)? };
        self.drop_fields(inner, place, super::reloc::TyOrigin::Subst(0))?;
        self.jump(out, &[]);
        self.enter(check);
        let waiting = self.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, state, Coroutine::FIRST_SUSPEND);
        self.brif(waiting, (suspended, &[]), (out, &[]));
        self.enter(suspended);
        let pin = self.pin_of(target)?;
        let none = self.ins().iconst(Type::I64, 0);
        self.invoke(super::reloc::SiteOrigin::ShapeGlueResume, poll, substs, vec![pin, none])?;
        self.jump(out, &[]);
        self.enter(out);
        Ok(())
    }

    /// The offsets of the state word and of the heap frame pointer in the coroutine object of type `ty`.
    fn header_of(&mut self, ty: Ty) -> Result<(i32, i32)> {
        let Some(env) = self.cx.closure_env(ty) else { bail!("a coroutine whose defining body was not inferred") };
        Ok((self.field_offset(env, 0)?, self.field_offset(env, 1)?))
    }

    /// The tuple of a coroutine's captures and its offset in the coroutine object, after the state word and the frame pointer.
    pub(super) fn captures_of(&mut self, ty: Ty) -> Result<(Ty, i32)> {
        let Some(env) = self.cx.closure_env(ty) else { bail!("a coroutine whose defining body was not inferred") };
        let Kind::Tuple(items) = *self.cx.types.kind(env) else { bail!("a coroutine whose object is not a tuple") };
        let Some(&inner) = self.cx.types.substs(items).get(2) else { bail!("a coroutine object with no captures") };
        Ok((inner, self.field_offset(env, 2)?))
    }
}
