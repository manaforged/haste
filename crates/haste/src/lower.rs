mod abi;
mod adt;
mod array;
mod atomic;
mod budget;
mod closure;
mod coroutine;
mod drop;
mod equal;
mod expr;
mod format;
mod intrinsic;
mod looping;
mod matching;
mod memory;
mod moves;
mod object;
mod ops;
mod place;
mod plan;
mod promote;
pub(crate) mod reloc;
mod simd;
mod slice_pat;
mod slots;
mod stencil;
mod stmt;

pub(super) use abi::is_freeze_ty;
pub(crate) use abi::purpose_tag;
pub use abi::{signature, signature_of};
pub use expr::{can_inline_body, may_inline};
pub(crate) use object::VTABLE_HEADER;
pub use plan::eval_footprint;
pub use reloc::{RelocPlan, eval_relocs};
pub use slots::{Slots, Word};
pub use stencil::{
    Bodies, FootprintId, RelocVec, RodataId, RodataReloc, RodataTarget, StencilOutput,
    SymbolSource, VtableEntry, VtableId, lower_dict, lower_stencil, warm_inline,
};

use std::sync::Arc;

use anyhow::{Result, bail};
use haste_codegen::{
    Arch, ArgumentPurpose, Block, FunctionBuilder, Ins, Signature, StackSlot, Type, Value,
};

pub(crate) fn arch_is_x64_or_arm64(arch: Arch) -> bool {
    matches!(arch, Arch::X64 | Arch::Arm64)
}

pub(crate) fn host_is_x64_or_arm64() -> bool {
    arch_is_x64_or_arm64(crate::compile_target().arch)
}

use crate::body::{Body, ExprId};
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::infer::{Shape, Typed};
use crate::sema::reach::Reached;
use crate::sema::ty::{Kind, Ty};

/// How a value of one type travels: not at all, in one register, or as the address of its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repr {
    Zst,
    Scalar(Type),
    Mem { size: u32, align: u8, parts: Parts },
}

/// The scalars that a memory value is made of, each a machine type at a byte offset, which a typed copy moves one by one; `Bytes` when its layout names none exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parts {
    Bytes,
    One(Type),
    Pair(Type, Type, i32),
    Words([(Type, i32); 4], u8),
}

impl Parts {
    /// Each scalar's machine type and byte offset.
    pub fn scalars(self) -> impl Iterator<Item = (Type, i32)> {
        let list = match self {
            Parts::Bytes => [None; 4],
            Parts::One(ty) => [Some((ty, 0)), None, None, None],
            Parts::Pair(first, second, offset) => {
                [Some((first, 0)), Some((second, offset)), None, None]
            }
            Parts::Words(words, len) => {
                let mut out = [None; 4];
                for (slot, &word) in out.iter_mut().zip(&words[..usize::from(len)]) {
                    *slot = Some(word);
                }
                out
            }
        };
        list.into_iter().flatten()
    }
}

/// The register type that holds a scalar value of `ty`, or none when `ty` is not a scalar.
pub fn machine(cx: &mut Context<'_>, ty: Ty) -> Result<Option<Type>> {
    use Primitive::*;
    Ok(Some(match cx.types.kind(ty).clone() {
        Kind::Prim(Bool | I8 | U8) => Type::I8,
        Kind::Prim(I16 | U16) => Type::I16,
        Kind::Prim(I32 | U32 | Char) => Type::I32,
        Kind::Prim(I64 | U64 | Isize | Usize) => Type::I64,
        Kind::Prim(I128 | U128) => Type::I128,
        Kind::Prim(F32) => Type::F32,
        Kind::Prim(F64) => Type::F64,
        Kind::Pat(base, ..) => return machine(cx, base),
        Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) if !cx.is_unsized(pointee)? => Type::I64,
        Kind::FnPtr(..) => Type::I64,
        _ => return Ok(None),
    }))
}

/// How values of `ty` travel.
pub fn repr(cx: &mut Context<'_>, ty: Ty) -> Result<Repr> {
    if let Some(machine) = machine(cx, ty)? {
        return Ok(Repr::Scalar(machine));
    }
    if matches!(cx.types.kind(ty), Kind::FnDef(..)) {
        return Ok(Repr::Zst);
    }
    let layout = cx.layout(ty)?;
    let size = layout.size.bytes() as u32;
    Ok(if size == 0 {
        Repr::Zst
    } else {
        Repr::Mem {
            size,
            align: layout.align.bytes().trailing_zeros() as u8,
            parts: abi::copied_parts(&layout),
        }
    })
}

fn is_repr_c(cx: &Context<'_>, ty: Ty) -> bool {
    matches!(
        cx.types.kind(ty),
        Kind::Adt(def, _) if cx.repr(*def).flags.contains(rustc_abi::ReprFlags::IS_C)
    )
}

/// The scalars that a memory value of `ty` crosses an internal call in, one register each, or none when it travels by address.
pub fn passed_parts(cx: &mut Context<'_>, ty: Ty) -> Result<Option<Parts>> {
    Ok(match repr(cx, ty)? {
        Repr::Mem {
            parts: parts @ Parts::One(_),
            ..
        } => Some(parts),
        Repr::Mem {
            parts: parts @ Parts::Pair(first, second, _),
            ..
        } if first != Type::I128 && second != Type::I128 => Some(parts),
        Repr::Mem {
            parts: parts @ Parts::Words(..),
            ..
        } if !is_repr_c(cx, ty) => Some(parts),
        Repr::Mem {
            size,
            parts: Parts::Bytes | Parts::Pair(..),
            ..
        } if !is_repr_c(cx, ty) => match abi::packed_word_parts(size) {
            Parts::Bytes => None,
            parts => Some(parts),
        },
        _ => None,
    })
}

fn indirect(signature: &Signature) -> bool {
    signature
        .params
        .first()
        .is_some_and(|param| param.purpose == ArgumentPurpose::StructReturn)
}

#[derive(Clone, Copy)]
struct Loop {
    head: Block,
    exit: Block,
    value: Option<place::Place>,
    depth: usize,
    label: Option<u32>,
    block: bool,
}

#[derive(Default)]
pub(super) struct Inline {
    pub stack: Vec<usize>,
    pub local: rustc_hash::FxHashMap<(usize, usize), (Option<Arc<stencil::InlinedCallee>>, usize)>,
    pub local_meta: rustc_hash::FxHashMap<(usize, usize), (Option<stencil::InlineMeta>, usize)>,
    pub max_callee_height: u8,
    pub max_callee_size: u16,
    pub host: budget::Copies,
    pub warming: bool,
    pub aborted: bool,
}

impl Inline {
    pub(super) fn pop(&mut self, depth: usize) {
        self.stack.pop();
        self.local
            .retain(|_, (_, cyc)| *cyc == usize::MAX || *cyc < depth);
        self.local_meta
            .retain(|_, (_, cyc)| *cyc == usize::MAX || *cyc < depth);
    }
}

/// The lowering of one body.
pub struct Lower<'l, 'a> {
    cx: &'l mut Context<'a>,
    reached: &'l Reached,
    slots: &'l dyn Slots,
    inline: &'l mut Inline,
    body: &'l Body,
    src: &'a Source,
    typed: &'l Typed,
    b: FunctionBuilder<'l>,
    locals: Vec<Option<place::Place>>,
    loops: Vec<Loop>,
    try_exits: Vec<(ExprId, Loop)>,
    open: bool,
    dead: Option<Block>,
    sret: Option<Value>,
    dict_param: Option<Value>,
    mode: stencil::LowerMode,
    relocs: &'l mut RelocVec,
    reloc_plan: &'l mut Vec<reloc::RelocStep>,
    promoted_seq: u32,
    cycle_root: usize,
    owned: Vec<drop::Owned>,
    flags: Vec<StackSlot>,
    moved: rustc_hash::FxHashMap<moves::Root, Vec<moves::MovePath>>,
    extended: rustc_hash::FxHashSet<ExprId>,
    pending: Vec<drop::Pending>,
    index: usize,
    resume: Option<coroutine::Resume>,
    overflow_checks: bool,
    child_rodata: Vec<(RodataId, Option<Vec<u8>>)>,
    caller_loc_guard: Option<(Value, Value)>,
    prologue_block: Option<Block>,
    cached_caller_loc: Option<Value>,
    blob_statics: rustc_hash::FxHashSet<(ItemRef, crate::sema::substs::Substs)>,
}

/// Whether `item` is a static, whose reached instance writes its value into the static's cell.
pub fn is_static(cx: &Context<'_>, item: ItemRef) -> bool {
    cx.item(item).1.kind == ItemKind::Static
}

impl<'l, 'a> Lower<'l, 'a> {
    fn new(
        cx: &'l mut Context<'a>,
        (reached, slots, inline): (&'l Reached, &'l dyn Slots, &'l mut Inline),
        (index, source, body, typed): (usize, ItemRef, &'l Body, &'l Typed),
        (b, mode): (FunctionBuilder<'l>, stencil::LowerMode),
        (relocs, reloc_plan): (&'l mut RelocVec, &'l mut Vec<reloc::RelocStep>),
    ) -> Self {
        let src = cx.body_src(source);
        let overflow_checks = cx.overflow_checks(source.krate);
        Self {
            cx,
            reached,
            slots,
            inline,
            body,
            src,
            typed,
            b,
            locals: vec![None; typed.locals.len()],
            loops: Vec::new(),
            try_exits: Vec::new(),
            open: true,
            dead: None,
            sret: None,
            dict_param: None,
            mode,
            relocs,
            reloc_plan,
            promoted_seq: 0,
            cycle_root: usize::MAX,
            owned: Vec::new(),
            flags: Vec::new(),
            moved: Default::default(),
            extended: Default::default(),
            pending: Vec::new(),
            index,
            resume: None,
            overflow_checks,
            child_rodata: Vec::new(),
            caller_loc_guard: None,
            prologue_block: None,
            cached_caller_loc: None,
            blob_statics: rustc_hash::FxHashSet::default(),
        }
    }

    fn begin_entry(&mut self) -> Result<(Block, Block, Vec<Value>)> {
        let entry = self.b.create_block();
        self.b.append_block_params_for_function_params(entry);
        let entry_params = self.b.block_params(entry).to_vec();
        if self.mode == stencil::LowerMode::Dict {
            self.dict_param = entry_params.last().copied();
        }
        self.b.switch_to_block(entry);
        let (prologue, start) = (self.b.create_block(), self.b.create_block());
        self.prologue_block = Some(prologue);
        self.jump(prologue, &[]);
        self.b.switch_to_block(start);
        self.moved = self.moved_paths()?;
        self.declare_params(entry)?;
        Ok((prologue, start, entry_params))
    }

    fn finish_prologue(&mut self, prologue: Block, start: Block) {
        self.enter(prologue);
        for slot in std::mem::take(&mut self.flags) {
            let zero = self.ins().iconst(Type::I8, 0);
            self.ins().stack_store(zero, slot, 0);
        }
        self.jump(start, &[]);
        self.close();
        self.seal();
    }

    fn run(&mut self) -> Result<()> {
        let (prologue, start, _) = self.begin_entry()?;
        let filling = is_static(self.cx, self.reached.list[self.index].0.0);
        if !filling
            && self.inline.stack.is_empty()
            && self.cx.track_caller(self.reached.list[self.index].0.0)
        {
            self.enter_tracked_caller_loc()?;
        }
        let value = self.root_value()?;
        match filling {
            true => self.fill_static(value)?,
            false => self.finish(value)?,
        }
        self.finish_prologue(prologue, start);
        self.finish_resume()
    }

    fn declare_params(&mut self, entry: Block) -> Result<()> {
        let mut params = self.b.block_params(entry).to_vec().into_iter();
        self.sret = if indirect(&self.b.func.signature) {
            params.next()
        } else {
            None
        };
        for &local in &self.typed.params {
            self.declare_passed(local, &mut params)?;
            if let Some(&pat) = self.typed.unpacks.get(&local) {
                self.bind_matched(
                    pat,
                    self.locals[local as usize],
                    Some((moves::Root::Local(local), Vec::new())),
                )?;
                self.commit_moves()?;
            }
        }
        Ok(())
    }

    fn root_value(&mut self) -> Result<Option<Value>> {
        match &self.typed.shape {
            Shape::Glue(_) => self.glue_target().map(|_| None),
            Shape::Virtual(slot) => self.virtual_call(*slot).map(|_| None),
            Shape::Untuple(res) => self.untuple_call(res.clone()),
            &Shape::Ctor(variant) => self.ctor_body(variant),
            Shape::Clone(calls) => self.clone_body(calls.clone()),
            Shape::Asm(_) => bail!("global_asm function is assembled directly"),
            Shape::GlobalAlloc(shim) => self.global_alloc_body(shim.clone()),
            Shape::Anon(anon) => self.expr(anon.root),
            Shape::Closure(closure) if closure.closure.coroutine => {
                self.resume_root(closure.closure.body)
            }
            Shape::Closure(closure) => {
                let root = closure.closure.body;
                self.bind_captures()?;
                self.expr(root)
            }
            Shape::Body => {
                let Some(root) = self.body.root else {
                    bail!("body has no root")
                };
                match self
                    .typed
                    .closures
                    .get(&root)
                    .is_some_and(|closure| closure.coroutine)
                {
                    true => self.closure(root),
                    false => self.expr(root),
                }
            }
        }
    }

    fn glue_target(&mut self) -> Result<()> {
        let local = self.typed.params[0];
        let Kind::Ref(_, ty) = *self.cx.types.kind(self.typed.locals[local as usize].ty) else {
            bail!("drop glue takes a reference");
        };
        let target = self.locals[local as usize].and_then(|place| self.read(place));
        self.glue_body(ty, target)
    }

    /// Returns `value` from the function, through the result pointer when the result lives in memory.
    pub fn finish(&mut self, value: Option<Value>) -> Result<()> {
        if !self.open {
            return Ok(());
        }
        if self.resume.is_some() {
            return self.resume_finish(value);
        }
        let ret = self.typed.ret_ty();
        let returns = !self.b.func.signature.returns.is_empty();
        let values = match (self.sret, value) {
            (Some(out), Some(value)) => {
                let whole @ Repr::Mem { .. } = repr(self.cx, ret)? else {
                    bail!("result pointer for a non-memory result");
                };
                self.copy(out, value, whole);
                Some(Vec::new())
            }
            (None, Some(value)) if returns => Some(self.returned_values(ret, value)?),
            (None, _) if !returns => Some(Vec::new()),
            _ => None,
        };
        self.drop_to(0)?;
        if let Some((slot, prev)) = self.caller_loc_guard {
            self.ins().store(prev, slot, 0);
        }
        match values {
            Some(values) => self.ins().return_(&values),
            None => self.ins().trap(),
        };
        self.close();
        Ok(())
    }

    fn local_repr(&mut self, local: u32) -> Result<Repr> {
        repr(self.cx, self.typed.locals[local as usize].ty)
    }

    fn ins(&mut self) -> Ins<'_> {
        if !self.open {
            let dead = *self.dead.get_or_insert_with(|| self.b.create_block());
            self.b.switch_to_block(dead);
        }
        self.b.ins()
    }

    fn close(&mut self) {
        self.open = false;
        self.dead = None;
    }

    fn seal(&mut self) {
        if let Some(dead) = self.dead.take() {
            self.b.switch_to_block(dead);
            self.b.ins().trap();
        }
    }

    fn enter(&mut self, block: Block) {
        self.seal();
        self.b.switch_to_block(block);
        self.open = self.b.has_branch_to(block);
    }

    fn jump(&mut self, block: Block, args: &[Value]) {
        if self.open {
            self.b.ins().jump(block, args);
        }
    }

    fn brif(
        &mut self,
        test: Value,
        (then, then_args): (Block, &[Value]),
        (otherwise, else_args): (Block, &[Value]),
    ) {
        if self.open {
            self.b
                .ins()
                .brif(test, then, then_args, otherwise, else_args);
        }
    }

    fn declare_passed(
        &mut self,
        local: u32,
        params: &mut impl Iterator<Item = Value>,
    ) -> Result<()> {
        self.declare(local, None)?;
        if let Some(place) = self.locals[local as usize] {
            self.store_passed(local, place, params)?;
        }
        Ok(())
    }

    pub(super) fn store_passed(
        &mut self,
        local: u32,
        place: place::Place,
        params: &mut impl Iterator<Item = Value>,
    ) -> Result<Option<Value>> {
        let ty = self.typed.locals[local as usize].ty;
        let size = match place.repr {
            Repr::Mem { size, .. } => size,
            _ => 0,
        };
        let mut last = None;
        match passed_parts(self.cx, ty)? {
            Some(parts) => {
                for (machine, offset) in parts.scalars() {
                    if let Some(value) = params.next() {
                        self.store_packed_part(place, offset, machine, value, size);
                        last = Some(value);
                    }
                }
            }
            None => {
                if let Some(value) = params.next() {
                    self.write(place, value);
                    last = Some(value);
                }
            }
        }
        Ok(last)
    }

    pub(super) fn push_passed(&mut self, ty: Ty, value: Value, out: &mut Vec<Value>) -> Result<()> {
        let Repr::Mem { size, .. } = repr(self.cx, ty)? else {
            out.push(value);
            return Ok(());
        };
        match passed_parts(self.cx, ty)? {
            Some(parts) => out.extend(
                parts
                    .scalars()
                    .map(|(machine, offset)| self.load_packed_part(value, offset, machine, size)),
            ),
            None => out.push(value),
        }
        Ok(())
    }

    pub(super) fn returned_values(&mut self, ty: Ty, value: Value) -> Result<Vec<Value>> {
        let mut out = Vec::new();
        self.push_passed(ty, value, &mut out)?;
        Ok(out)
    }

    /// Gives `local` a fresh frame slot, writes `value` into it when given, and marks `local` initialized.
    pub fn declare(&mut self, local: u32, value: Option<Value>) -> Result<()> {
        self.declare_as(local, value, true)
    }

    fn declare_as(&mut self, local: u32, value: Option<Value>, init: bool) -> Result<()> {
        let repr = self.local_repr(local)?;
        let slot = self.slot(repr);
        if let Some(slot) = slot {
            self.locals[local as usize] = Some(place::Place::slot(slot, repr));
            if let Some(value) = value {
                self.write(place::Place::slot(slot, repr), value);
            }
        }
        if matches!(
            self.typed.shape,
            Shape::Body | Shape::Closure(_) | Shape::Anon(_)
        ) {
            let place = slot.map(|slot| place::Place::slot(slot, repr));
            self.own(
                moves::Root::Local(local),
                self.typed.locals[local as usize].ty,
                place,
                init,
            )?;
        }
        Ok(())
    }

    /// A fresh frame slot for a value of `repr`, or none for a zero-sized one.
    pub fn slot(&mut self, repr: Repr) -> Option<StackSlot> {
        match repr {
            Repr::Zst => None,
            Repr::Scalar(ty) => Some(
                self.b
                    .create_stack_slot(ty.bytes(), ty.bytes().trailing_zeros() as u8),
            ),
            Repr::Mem { size, align, .. } => Some(self.b.create_stack_slot(size, align)),
        }
    }

    fn ty_of(&self, expr: ExprId) -> Ty {
        self.typed.exprs[expr.0 as usize]
    }

    fn len(&self, len: Ty) -> Result<i64> {
        match self.cx.types.len(len) {
            Some(len) => Ok(len as i64),
            None => bail!("an array whose length is not known"),
        }
    }

    fn repr_of(&mut self, expr: ExprId) -> Result<Repr> {
        repr(self.cx, self.ty_of(expr))
    }
}
