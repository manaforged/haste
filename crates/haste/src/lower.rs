//! Lowering of inferred function bodies at concrete types to `haste-codegen` SSA.

mod abi;
mod adt;
mod array;
mod atomic;
mod closure;
mod coroutine;
mod drop;
mod equal;
mod expr;
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
mod reloc;
mod simd;
mod slice_pat;
mod stencil;

pub use expr::{can_inline_body, may_inline};
pub(crate) use abi::purpose_tag;
pub use plan::eval_footprint;
pub use reloc::{RelocPlan, eval_relocs};
pub use stencil::{Bodies, FootprintId, RelocVec, RodataId, RodataReloc, RodataTarget, StencilOutput, SymbolSource, VtableEntry, VtableId, lower_dict, lower_stencil};

use std::sync::Arc;

use anyhow::{Result, bail};
use haste_codegen::{AbiParam, ArgumentPurpose, Block, CallConv, FunctionBuilder, Ins, Signature, StackSlot, Type, Value};

use crate::body::{Body, Expr, ExprId, List, Pat, PatId, Stmt};
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::infer::{Shape, Typed};
use crate::sema::reach::Reached;
use crate::sema::ty::{Kind, Ty};

/// Where lowered code finds what it refers to: call table entries and constant data.
pub trait Slots {
    /// The address of the table entry that holds the code address of reached function `index`.
    fn entry(&self, index: usize) -> i64;
    /// The address of a 16-byte-aligned copy of `bytes` that lives as long as the program.
    fn data(&self, bytes: &[u8]) -> i64;
    /// The address of `len` zeroed, 16-byte-aligned bytes that each run of the program writes for itself.
    fn zeroed(&self, len: usize) -> i64;
    /// The address of a table of `words` that lives as long as the program, each `Word::Code` holding a code address once every function is compiled.
    fn table(&self, words: &[Word]) -> i64;
    /// The address of the zeroed, 16-byte-aligned cell of `size` bytes and one more that holds the value of the static whose initializer is reached function `index`, then whether it is filled.
    fn cell(&self, index: usize, size: u32) -> i64;
    /// Notes that lowering used `instance`, which was not reached, so a caller that reaches functions on demand can reach it and lower again.
    fn want(&self, _instance: crate::sema::reach::Instance) {}
    /// The count of unreached function instances recorded so far.
    fn wanted_len(&self) -> usize {
        0
    }
    /// The bodies that calls copy inline.
    fn bodies(&self) -> &Bodies;
    /// Whether call table entries live in the low 2 GiB address range.
    fn low_entries(&self) -> bool {
        false
    }
    /// Interns a read-only blob with optional intra-rodata relocations and returns its symbol id.
    fn rodata(&self, bytes: &[u8], _relocs: &[RodataReloc]) -> RodataId {
        let _ = self.data(bytes);
        RodataId(0)
    }
    /// Interns a vtable layout and returns its symbol id.
    fn vtable_id(&self, _entries: &[VtableEntry]) -> VtableId {
        VtableId(0)
    }
    /// Resolves a symbolic relocation entry to its concrete 64-bit value in this image.
    fn resolve_symbol(&self, sym: &SymbolSource) -> u64 {
        match *sym {
            SymbolSource::Slot(index) => self.entry(index) as u64,
            SymbolSource::StaticCell { index, size } => self.cell(index, size) as u64,
            SymbolSource::PromotedCell { size, .. } => self.zeroed(size as usize) as u64,
            SymbolSource::Host(ref host) => host.host_addr().unwrap_or(0) as u64,
            SymbolSource::Word(word) => word,
            SymbolSource::ThreadSlot(index) => index as u64,
            SymbolSource::Vtable(_) | SymbolSource::Rodata(_) => 0,
        }
    }
}

/// One machine word of a constant table: a number, or the code address of reached function `index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Word {
    Int(u64),
    Code(usize),
}

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
}

impl Parts {
    /// Each scalar's machine type and byte offset.
    pub fn scalars(self) -> impl Iterator<Item = (Type, i32)> {
        let list = match self {
            Parts::Bytes => [None, None],
            Parts::One(ty) => [Some((ty, 0)), None],
            Parts::Pair(first, second, offset) => [Some((first, 0)), Some((second, offset))],
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
    Ok(if size == 0 { Repr::Zst } else { Repr::Mem { size, align: layout.align.bytes().trailing_zeros() as u8, parts: abi::copied_parts(&layout) } })
}

/// The scalars that a memory value of `ty` crosses an internal call in, one register each, or none when it travels by address.
pub fn passed_parts(cx: &mut Context<'_>, ty: Ty) -> Result<Option<Parts>> {
    Ok(match repr(cx, ty)? {
        Repr::Mem { parts: parts @ Parts::One(_), .. } => Some(parts),
        Repr::Mem { parts: parts @ Parts::Pair(first, second, _), .. } if first != Type::I128 && second != Type::I128 => Some(parts),
        _ => None,
    })
}

/// The native signature of a reached function: memory values travel by address unless `passed_parts` splits them into one or two scalar registers, and a memory result returns in those registers or through a first `StructReturn` pointer.
pub fn signature(cx: &mut Context<'_>, typed: &Typed) -> Result<Signature> {
    signature_of(cx, typed.ret, typed.params.iter().map(|&local| typed.locals[local as usize].ty))
}

/// The native signature of a function that returns `ret` and takes `params`.
pub fn signature_of(cx: &mut Context<'_>, ret: Option<Ty>, params: impl IntoIterator<Item = Ty>) -> Result<Signature> {
    let mut signature = Signature::new(CallConv::Native);
    if let Some(ret) = ret {
        match repr(cx, ret)? {
            Repr::Scalar(ty) => signature.returns.push(AbiParam::new(ty)),
            Repr::Mem { .. } => match passed_parts(cx, ret)? {
                Some(parts) => signature.returns.extend(parts.scalars().map(|(machine, _)| AbiParam::new(machine))),
                None => signature.params.push(AbiParam::special(Type::I64, ArgumentPurpose::StructReturn)),
            },
            Repr::Zst => {}
        }
    }
    for param in params {
        match repr(cx, param)? {
            Repr::Scalar(ty) => signature.params.push(AbiParam::new(ty)),
            Repr::Mem { .. } => match passed_parts(cx, param)? {
                Some(parts) => signature.params.extend(parts.scalars().map(|(machine, _)| AbiParam::new(machine))),
                None => signature.params.push(AbiParam::new(Type::I64)),
            },
            Repr::Zst => {}
        }
    }
    Ok(signature)
}

/// Whether `signature` returns its result through a first `StructReturn` pointer.
fn indirect(signature: &Signature) -> bool {
    signature.params.first().is_some_and(|param| param.purpose == ArgumentPurpose::StructReturn)
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

/// The inline state of one top-level lowering: the reached indexes being lowered, and the callees whose inline decision rests on an open cycle, each with the stack depth that cycle reaches.
#[derive(Default)]
pub(super) struct Inline {
    pub stack: Vec<usize>,
    pub local: rustc_hash::FxHashMap<usize, (Option<Arc<stencil::InlinedCallee>>, usize)>,
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
}

/// Whether `item` is a static, whose reached instance writes its value into the static's cell.
pub fn is_static(cx: &Context<'_>, item: ItemRef) -> bool {
    cx.item(item).1.kind == ItemKind::Static
}

impl Lower<'_, '_> {
    fn run(&mut self) -> Result<()> {
        let entry = self.b.create_block();
        self.b.append_block_params_for_function_params(entry);
        if self.mode == stencil::LowerMode::Dict {
            self.dict_param = self.b.block_params(entry).last().copied();
        }
        self.b.switch_to_block(entry);
        let (prologue, start) = (self.b.create_block(), self.b.create_block());
        self.jump(prologue, &[]);
        self.b.switch_to_block(start);
        self.moved = self.moved_paths()?;
        self.declare_params(entry)?;
        let filling = is_static(self.cx, self.reached.list[self.index].0.0);
        let value = self.root_value()?;
        match filling {
            true => self.fill_static(value)?,
            false => self.finish(value)?,
        }
        self.enter(prologue);
        for slot in std::mem::take(&mut self.flags) {
            let zero = self.ins().iconst(Type::I8, 0);
            self.ins().stack_store(zero, slot, 0);
        }
        self.jump(start, &[]);
        self.close();
        self.seal();
        self.finish_resume()
    }

    fn declare_params(&mut self, entry: Block) -> Result<()> {
        let mut params = self.b.block_params(entry).to_vec().into_iter();
        self.sret = if indirect(&self.b.func.signature) { params.next() } else { None };
        for &local in &self.typed.params {
            self.declare_passed(local, &mut params)?;
            if let Some(&pat) = self.typed.unpacks.get(&local) {
                self.bind_matched(pat, self.locals[local as usize], Some((moves::Root::Local(local), Vec::new())))?;
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
            Shape::Anon(anon) => self.expr(anon.root),
            Shape::Closure(closure) if closure.closure.coroutine => self.resume_root(closure.closure.body),
            Shape::Closure(closure) => {
                let root = closure.closure.body;
                self.bind_captures()?;
                self.expr(root)
            }
            Shape::Body => {
                let Some(root) = self.body.root else { bail!("body has no root") };
                match self.typed.closures.get(&root).is_some_and(|closure| closure.coroutine) {
                    true => self.closure(root),
                    false => self.expr(root),
                }
            }
        }
    }

    fn glue_target(&mut self) -> Result<()> {
        let local = self.typed.params[0];
        let Kind::Ref(_, ty) = *self.cx.types.kind(self.typed.locals[local as usize].ty) else { bail!("drop glue takes a reference") };
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
                let whole @ Repr::Mem { .. } = repr(self.cx, ret)? else { bail!("result pointer for a non-memory result") };
                self.copy(out, value, whole);
                Some(Vec::new())
            }
            (None, Some(value)) if returns => Some(self.returned_values(ret, value)?),
            (None, _) if !returns => Some(Vec::new()),
            _ => None,
        };
        self.drop_to(0)?;
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

    /// The instruction cursor, in a block that nothing enters when the current one already ended.
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

    /// Moves the cursor to `block`, which code runs into only when some branch goes there.
    fn enter(&mut self, block: Block) {
        self.seal();
        self.b.switch_to_block(block);
        self.open = self.b.has_branch_to(block);
    }

    /// Goes to `block` with `args`, when the cursor is in code that runs.
    fn jump(&mut self, block: Block, args: &[Value]) {
        if self.open {
            self.b.ins().jump(block, args);
        }
    }

    /// Goes to `then` when `test` is nonzero and to `otherwise` when not, when the cursor is in code that runs.
    fn brif(&mut self, test: Value, (then, then_args): (Block, &[Value]), (otherwise, else_args): (Block, &[Value])) {
        if self.open {
            self.b.ins().brif(test, then, then_args, otherwise, else_args);
        }
    }

    fn declare_passed(&mut self, local: u32, params: &mut impl Iterator<Item = Value>) -> Result<()> {
        self.declare(local, None)?;
        if let Some(place) = self.locals[local as usize] {
            self.store_passed(local, place, params)?;
        }
        Ok(())
    }

    /// Stores into `place` the scalars or address that parameter `local` received from `params`, returning the last consumed `Value`.
    pub(super) fn store_passed(&mut self, local: u32, place: place::Place, params: &mut impl Iterator<Item = Value>) -> Result<Option<Value>> {
        let ty = self.typed.locals[local as usize].ty;
        let mut last = None;
        match passed_parts(self.cx, ty)? {
            Some(parts) => {
                for (machine, offset) in parts.scalars() {
                    if let Some(value) = params.next() {
                        self.write(place.at(offset, Repr::Scalar(machine)), value);
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

    /// Appends `value`, a value of `ty`, as it crosses a call: scalars loaded from memory when `passed_parts` names them.
    pub(super) fn push_passed(&mut self, ty: Ty, value: Value, out: &mut Vec<Value>) -> Result<()> {
        match passed_parts(self.cx, ty)? {
            Some(parts) => out.extend(parts.scalars().map(|(machine, offset)| self.ins().load(machine, value, offset))),
            None => out.push(value),
        }
        Ok(())
    }

    /// `value`, a result of `ty`, as the registers it returns in: the scalars loaded from its memory when `passed_parts` names them.
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
        if matches!(self.typed.shape, Shape::Body | Shape::Closure(_) | Shape::Anon(_)) {
            let place = slot.map(|slot| place::Place::slot(slot, repr));
            self.own(moves::Root::Local(local), self.typed.locals[local as usize].ty, place, init)?;
        }
        Ok(())
    }

    /// A fresh frame slot for a value of `repr`, or none for a zero-sized one.
    pub fn slot(&mut self, repr: Repr) -> Option<StackSlot> {
        match repr {
            Repr::Zst => None,
            Repr::Scalar(ty) => Some(self.b.create_stack_slot(ty.bytes(), ty.bytes().trailing_zeros() as u8)),
            Repr::Mem { size, align, .. } => Some(self.b.create_stack_slot(size, align)),
        }
    }

    fn ty_of(&self, expr: ExprId) -> Ty {
        self.typed.exprs[expr.0 as usize]
    }

    /// The length that the array length `len` holds, which inference has made known.
    fn len(&self, len: Ty) -> Result<i64> {
        match self.cx.types.len(len) {
            Some(len) => Ok(len as i64),
            None => bail!("an array whose length is not known"),
        }
    }

    fn repr_of(&mut self, expr: ExprId) -> Result<Repr> {
        repr(self.cx, self.ty_of(expr))
    }

    fn block(&mut self, stmts: List, tail: Option<ExprId>) -> Result<Option<Value>> {
        let mark = self.owned.len();
        for &stmt in self.body.list(stmts) {
            self.stmt(self.body.stmts[stmt as usize])?;
        }
        let value = match tail {
            Some(tail) => self.expr(tail)?,
            None => None,
        };
        self.end_scope(mark)?;
        Ok(value)
    }

    /// Lowers the output of a macro call in expression position: its statements, then its tail, whose temporaries the enclosing statement drops.
    fn expansion(&mut self, stmts: List, tail: Option<ExprId>) -> Result<Option<Value>> {
        for &stmt in self.body.list(stmts) {
            self.stmt(self.body.stmts[stmt as usize])?;
        }
        tail.map_or(Ok(None), |tail| self.expr(tail))
    }

    fn stmt(&mut self, stmt: Stmt) -> Result<()> {
        let mark = self.owned.len();
        match stmt {
            Stmt::Let { pat, init, otherwise: None, .. } => self.let_stmt(pat, init)?,
            Stmt::Let { pat, init: Some(init), otherwise: Some(otherwise), .. } => self.let_else(pat, init, otherwise)?,
            Stmt::Expr(expr, _) => self.discard(expr)?,
            Stmt::Item(..) | Stmt::Empty => {}
            other => bail!("cannot lower statement {other:?} yet"),
        }
        self.end_statement(mark)?;
        if let Stmt::Let { extend: true, .. } = stmt {
            self.extend_owned(mark);
        }
        Ok(())
    }

    fn discard(&mut self, expr: ExprId) -> Result<()> {
        let value = self.expr(expr)?;
        if !self.open {
            return Ok(());
        }
        let ty = self.ty_of(expr);
        let repr = repr(self.cx, ty)?;
        let place = value.map(|value| place::Place { base: place::Base::Addr(value), offset: 0, repr });
        match (repr, place) {
            (Repr::Scalar(_), _) => Ok(()),
            (_, place) => self.glue(ty, place, reloc::TyOrigin::Expr(expr)),
        }
    }

    fn let_stmt(&mut self, pat: PatId, init: Option<ExprId>) -> Result<()> {
        let Some(init) = init else {
            return self.declare_uninit(pat);
        };
        self.extend(init);
        let unboxed = self.moves_whole_box(pat, init)?;
        let place = if unboxed { self.temp(init)? } else { self.place_or_temp(init)? };
        if !self.open {
            return Ok(());
        }
        let path = if unboxed { Some((moves::Root::Temp(init), Vec::new())) } else { self.scrutinee_path(init) };
        self.bind_matched(pat, place, path)?;
        self.commit_moves()
    }

    fn declare_uninit(&mut self, pat: PatId) -> Result<()> {
        match *self.body.pat(pat) {
            Pat::Bind { sub: None, .. } => match self.typed.binds.get(&pat) {
                Some(&local) => self.declare_as(local, None, false),
                None => bail!("an uninitialized binding with no local"),
            },
            Pat::Wild => Ok(()),
            Pat::Paren(inner) => self.declare_uninit(inner),
            Pat::Tuple(items) => self.body.list(items).to_vec().into_iter().try_for_each(|item| self.declare_uninit(PatId(item))),
            _ => bail!("uninitialized `let` with a pattern"),
        }
    }

    fn extend(&mut self, expr: ExprId) {
        match *self.body.expr(expr) {
            Expr::Ref { expr: inner, .. } => {
                self.extended.insert(inner);
                self.extend(inner);
            }
            Expr::Paren(inner) | Expr::Cast(inner, _) => self.extend(inner),
            Expr::Block { tail: Some(tail), .. } => self.extend(tail),
            Expr::Tuple(list) | Expr::Array(list) => self.body.list(list).to_vec().into_iter().for_each(|item| self.extend(ExprId(item))),
            Expr::Struct { fields, .. } => self.body.list(fields).to_vec().into_iter().for_each(|at| self.extend(self.body.fields[at as usize].value)),
            _ => {}
        }
    }
}
