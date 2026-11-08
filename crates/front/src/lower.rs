//! Lowering of inferred function bodies at concrete types to `haste-codegen` SSA.

mod abi;
mod adt;
mod array;
mod closure;
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
mod promote;
mod slice_pat;

use anyhow::{Result, bail};
use haste_codegen::{AbiParam, ArgumentPurpose, Block, CallConv, Function, FunctionBuilder, Ins, Signature, StackSlot, Type, Value};

use crate::body::{Body, Expr, ExprId, List, Pat, PatId, Stmt};
use crate::names::Primitive;
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::infer::{Shape, Typed};
use crate::sema::reach::Reached;
use crate::sema::ty::{Kind, Ty, Types};

/// Where lowered code finds what it refers to: call table entries and constant data.
pub trait Slots {
    /// The address of the table entry that holds the code address of reached function `index`.
    fn entry(&self, index: usize) -> i64;
    /// The address of a 16-byte-aligned copy of `bytes` that lives as long as the program.
    fn data(&self, bytes: &[u8]) -> i64;
    /// The address of a table of `words` that lives as long as the program, each `Word::Code` holding a code address once every function is compiled.
    fn table(&self, words: &[Word]) -> i64;
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
    Mem { size: u32, align: u8 },
}

/// The register type that holds a scalar value of `ty`, or none when `ty` is not a scalar.
pub fn machine(types: &Types, ty: Ty) -> Option<Type> {
    use Primitive::*;
    Some(match types.kind(ty) {
        Kind::Prim(Bool | I8 | U8) => Type::I8,
        Kind::Prim(I16 | U16) => Type::I16,
        Kind::Prim(I32 | U32 | Char) => Type::I32,
        Kind::Prim(I64 | U64 | Isize | Usize) => Type::I64,
        Kind::Prim(I128 | U128) => Type::I128,
        Kind::Prim(F32) => Type::F32,
        Kind::Prim(F64) => Type::F64,
        Kind::Pat(base, ..) => return machine(types, *base),
        Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) if !types.kind(*pointee).is_unsized() => Type::I64,
        Kind::FnPtr(..) => Type::I64,
        _ => return None,
    })
}

/// How values of `ty` travel.
pub fn repr(cx: &mut Context<'_>, ty: Ty) -> Result<Repr> {
    if let Some(machine) = machine(&cx.types, ty) {
        return Ok(Repr::Scalar(machine));
    }
    if matches!(cx.types.kind(ty), Kind::FnDef(..)) {
        return Ok(Repr::Zst);
    }
    let layout = cx.layout(ty)?;
    let size = layout.size.bytes() as u32;
    Ok(if size == 0 { Repr::Zst } else { Repr::Mem { size, align: layout.align.abi.bytes().trailing_zeros() as u8 } })
}

/// The native signature of a reached function: memory values travel by address, and a memory result through a first `StructReturn` pointer.
pub fn signature(cx: &mut Context<'_>, typed: &Typed) -> Result<Signature> {
    signature_of(cx, typed.ret, typed.params.iter().map(|&local| typed.locals[local as usize].ty))
}

/// The native signature of a function that returns `ret` and takes `params`.
pub fn signature_of(cx: &mut Context<'_>, ret: Option<Ty>, params: impl IntoIterator<Item = Ty>) -> Result<Signature> {
    let mut signature = Signature::new(CallConv::Native);
    if let Some(ret) = ret {
        match repr(cx, ret)? {
            Repr::Scalar(ty) => signature.returns.push(AbiParam::new(ty)),
            Repr::Mem { .. } => signature.params.push(AbiParam::special(Type::I64, ArgumentPurpose::StructReturn)),
            Repr::Zst => {}
        }
    }
    for param in params {
        match repr(cx, param)? {
            Repr::Scalar(ty) => signature.params.push(AbiParam::new(ty)),
            Repr::Mem { .. } => signature.params.push(AbiParam::new(Type::I64)),
            Repr::Zst => {}
        }
    }
    Ok(signature)
}

#[derive(Clone, Copy)]
struct Loop {
    head: Block,
    exit: Block,
    value: Option<(StackSlot, Type)>,
    depth: usize,
}

/// The lowering of one body.
pub struct Lower<'l, 'a> {
    cx: &'l mut Context<'a>,
    reached: &'l Reached,
    slots: &'l dyn Slots,
    body: &'l Body,
    src: &'a Source,
    typed: &'l Typed,
    b: FunctionBuilder<'l>,
    locals: Vec<Option<place::Place>>,
    loops: Vec<Loop>,
    open: bool,
    dead: Option<Block>,
    sret: Option<Value>,
    owned: Vec<drop::Owned>,
    flags: Vec<StackSlot>,
    moved: rustc_hash::FxHashMap<moves::Root, Vec<moves::MovePath>>,
    extended: rustc_hash::FxHashSet<ExprId>,
    pending: Vec<(moves::Root, moves::MovePath)>,
}

/// Lowers reached function `index` to SSA, with calls through `slots`.
pub fn lower(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize) -> Result<Function> {
    let ((item, _), typed) = &reached.list[index];
    let Ok(typed) = typed else { bail!("{} did not type-check", cx.name(*item)) };
    let source = match &typed.shape {
        Shape::Closure(closure) => closure.parent,
        _ => *item,
    };
    let body = match typed.shape {
        Shape::Body | Shape::Closure(_) => cx.body(source)?,
        Shape::Glue(_) | Shape::Virtual(_) => std::rc::Rc::default(),
    };
    let src = cx.body_src(source);
    let mut func = Function::new(signature(cx, typed)?);
    let b = FunctionBuilder::new(&mut func);
    let mut lower = Lower {
        cx,
        reached,
        slots,
        body: &body,
        src,
        typed,
        b,
        locals: vec![None; typed.locals.len()],
        loops: Vec::new(),
        open: true,
        dead: None,
        sret: None,
        owned: Vec::new(),
        flags: Vec::new(),
        moved: Default::default(),
        extended: Default::default(),
        pending: Vec::new(),
    };
    lower.run()?;
    Ok(func)
}

impl Lower<'_, '_> {
    fn run(&mut self) -> Result<()> {
        let entry = self.b.create_block();
        self.b.append_block_params_for_function_params(entry);
        self.b.switch_to_block(entry);
        let (prologue, start) = (self.b.create_block(), self.b.create_block());
        self.jump(prologue, &[]);
        self.b.switch_to_block(start);
        self.moved = self.moved_paths()?;
        let mut params = self.b.block_params(entry).to_vec().into_iter();
        let sret = self.b.func.signature.params.first().is_some_and(|param| param.purpose == ArgumentPurpose::StructReturn);
        self.sret = if sret { params.next() } else { None };
        for &local in &self.typed.params {
            let value = match self.local_repr(local)? {
                Repr::Zst => None,
                _ => params.next(),
            };
            self.declare(local, value)?;
        }
        let value = self.root_value()?;
        self.finish(value)?;
        self.enter(prologue);
        for slot in std::mem::take(&mut self.flags) {
            let zero = self.ins().iconst(Type::I8, 0);
            self.ins().stack_store(zero, slot, 0);
        }
        self.jump(start, &[]);
        self.close();
        self.seal();
        Ok(())
    }

    fn root_value(&mut self) -> Result<Option<Value>> {
        match &self.typed.shape {
            Shape::Glue(_) => self.glue_target().map(|_| None),
            Shape::Virtual(slot) => self.virtual_call(*slot).map(|_| None),
            Shape::Closure(closure) => {
                let root = closure.closure.body;
                self.bind_captures()?;
                self.expr(root)
            }
            Shape::Body => {
                let Some(root) = self.body.root else { bail!("body has no root") };
                self.expr(root)
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
        let ret = self.typed.ret.unwrap_or(Ty(u32::MAX));
        if let (Some(out), Some(value)) = (self.sret, value) {
            let Repr::Mem { size, .. } = repr(self.cx, ret)? else { bail!("result pointer for a non-memory result") };
            self.copy(out, value, size);
        }
        self.drop_to(0)?;
        match (self.sret, value) {
            (Some(_), Some(_)) => {
                self.ins().return_(&[]);
            }
            (None, Some(value)) if !self.b.func.signature.returns.is_empty() => {
                self.ins().return_(&[value]);
            }
            (_, _) if self.b.func.signature.returns.is_empty() && self.sret.is_none() => {
                self.ins().return_(&[]);
            }
            _ => {
                self.ins().trap();
            }
        }
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

    /// Gives the initialized `local` its own frame slot, holding a copy of `value` when there is one, and owns what it holds.
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
        let place = slot.map(|slot| place::Place::slot(slot, repr));
        self.own(moves::Root::Local(local), self.typed.locals[local as usize].ty, place, init)
    }

    /// A fresh frame slot for a value of `repr`, or none for a zero-sized one.
    pub fn slot(&mut self, repr: Repr) -> Option<StackSlot> {
        match repr {
            Repr::Zst => None,
            Repr::Scalar(ty) => Some(self.b.create_stack_slot(ty.bytes(), ty.bytes().trailing_zeros() as u8)),
            Repr::Mem { size, align } => Some(self.b.create_stack_slot(size, align)),
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

    fn stmt(&mut self, stmt: Stmt) -> Result<()> {
        let mark = self.owned.len();
        match stmt {
            Stmt::Let { pat, init, otherwise: None, .. } => self.let_stmt(pat, init)?,
            Stmt::Let { pat, init: Some(init), otherwise: Some(otherwise), .. } => self.let_else(pat, init, otherwise)?,
            Stmt::Expr(expr, _) => self.discard(expr)?,
            Stmt::Item(..) | Stmt::Empty => {}
            other => bail!("cannot lower statement {other:?} yet"),
        }
        self.end_statement(mark)
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
            (_, place) => self.glue(ty, place),
        }
    }

    fn let_stmt(&mut self, pat: PatId, init: Option<ExprId>) -> Result<()> {
        let Some(init) = init else {
            return match (self.body.pat(pat), self.typed.binds.get(&pat)) {
                (Pat::Bind { .. }, Some(&local)) => self.declare_as(local, None, false),
                _ => bail!("uninitialized `let` with a pattern"),
            };
        };
        self.extend(init);
        let place = self.place_or_temp(init)?;
        if !self.open {
            return Ok(());
        }
        self.bind_matched(pat, place, self.scrutinee_path(init))?;
        self.commit_moves()
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
