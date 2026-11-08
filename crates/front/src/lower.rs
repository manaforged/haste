//! Lowering of inferred function bodies at concrete types to `haste-codegen` SSA.

mod adt;
mod array;
mod expr;
mod intrinsic;
mod looping;
mod matching;
mod memory;
mod ops;
mod place;
mod promote;

use anyhow::{Result, bail};
use haste_codegen::{AbiParam, ArgumentPurpose, Block, CallConv, Function, FunctionBuilder, Ins, Signature, StackSlot, Type, Value};

use crate::body::{Body, ExprId, List, Pat, PatId, Stmt};
use crate::names::Primitive;
use crate::outline::source::Source;
use crate::sema::Context;
use crate::sema::infer::Typed;
use crate::sema::reach::Reached;
use crate::sema::ty::{Kind, Ty, Types};

/// Where lowered code finds what it refers to: call table entries and constant data.
pub trait Slots {
    /// The address of the table entry that holds the code address of reached function `index`.
    fn entry(&self, index: usize) -> i64;
    /// The address of a 16-byte-aligned copy of `bytes` that lives as long as the program.
    fn data(&self, bytes: &[u8]) -> i64;
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
        Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) if !matches!(types.kind(*pointee), Kind::Slice(_) | Kind::Prim(Str)) => Type::I64,
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
    let mut signature = Signature::new(CallConv::Native);
    if let Some(ret) = typed.ret {
        match repr(cx, ret)? {
            Repr::Scalar(ty) => signature.returns.push(AbiParam::new(ty)),
            Repr::Mem { .. } => signature.params.push(AbiParam::special(Type::I64, ArgumentPurpose::StructReturn)),
            Repr::Zst => {}
        }
    }
    for &local in &typed.params {
        match repr(cx, typed.locals[local as usize].ty)? {
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
    locals: Vec<Option<(StackSlot, Repr)>>,
    loops: Vec<Loop>,
    open: bool,
    sret: Option<Value>,
}

/// Lowers reached function `index` to SSA, with calls through `slots`.
pub fn lower(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize) -> Result<Function> {
    let ((item, _), typed) = &reached.list[index];
    let Ok(typed) = typed else { bail!("{} did not type-check", cx.name(*item)) };
    for ty in typed.exprs.iter().chain(typed.locals.iter().map(|local| &local.ty)).filter(|ty| ty.0 != u32::MAX).copied().collect::<rustc_hash::FxHashSet<_>>() {
        if crate::sema::infer::needs_drop(cx, ty)? {
            bail!("{} holds a value that needs drop glue, which is not lowered yet", cx.name(*item));
        }
    }
    let body = cx.body(*item)?;
    let src = cx.body_src(*item);
    let mut func = Function::new(signature(cx, typed)?);
    let b = FunctionBuilder::new(&mut func);
    let mut lower = Lower { cx, reached, slots, body: &body, src, typed, b, locals: vec![None; typed.locals.len()], loops: Vec::new(), open: true, sret: None };
    lower.run()?;
    Ok(func)
}

impl Lower<'_, '_> {
    fn run(&mut self) -> Result<()> {
        let entry = self.b.create_block();
        self.b.append_block_params_for_function_params(entry);
        self.b.switch_to_block(entry);
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
        let Some(root) = self.body.root else { bail!("body has no root") };
        let value = self.expr(root)?;
        self.finish(value)
    }

    /// Returns `value` from the function, through the result pointer when the result lives in memory.
    pub fn finish(&mut self, value: Option<Value>) -> Result<()> {
        if !self.open {
            return Ok(());
        }
        let ret = self.typed.ret.unwrap_or(Ty(u32::MAX));
        match (self.sret, value) {
            (Some(out), Some(value)) => {
                let Repr::Mem { size, .. } = repr(self.cx, ret)? else { bail!("result pointer for a non-memory result") };
                self.copy(out, value, size);
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

    /// The instruction cursor, in a fresh block when the current one already ended.
    fn ins(&mut self) -> Ins<'_> {
        if !self.open {
            let dead = self.b.create_block();
            self.b.switch_to_block(dead);
            self.open = true;
        }
        self.b.ins()
    }

    fn close(&mut self) {
        self.open = false;
    }

    /// Gives `local` its own frame slot, holding a copy of `value` when there is one.
    pub fn declare(&mut self, local: u32, value: Option<Value>) -> Result<()> {
        let repr = self.local_repr(local)?;
        let Some(slot) = self.slot(repr) else { return Ok(()) };
        self.locals[local as usize] = Some((slot, repr));
        if let Some(value) = value {
            self.write(place::Place::slot(slot, repr), value);
        }
        Ok(())
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

    fn repr_of(&mut self, expr: ExprId) -> Result<Repr> {
        repr(self.cx, self.ty_of(expr))
    }

    fn block(&mut self, stmts: List, tail: Option<ExprId>) -> Result<Option<Value>> {
        for &stmt in self.body.list(stmts) {
            self.stmt(self.body.stmts[stmt as usize])?;
        }
        match tail {
            Some(tail) => self.expr(tail),
            None => Ok(None),
        }
    }

    fn stmt(&mut self, stmt: Stmt) -> Result<()> {
        match stmt {
            Stmt::Let { pat, init, otherwise: None, .. } => self.let_stmt(pat, init),
            Stmt::Let { pat, init: Some(init), otherwise: Some(otherwise), .. } => self.let_else(pat, init, otherwise),
            Stmt::Expr(expr, _) => self.expr(expr).map(drop),
            Stmt::Item(..) | Stmt::Empty => Ok(()),
            other => bail!("cannot lower statement {other:?} yet"),
        }
    }

    fn let_stmt(&mut self, pat: PatId, init: Option<ExprId>) -> Result<()> {
        let Some(init) = init else {
            return match (self.body.pat(pat), self.typed.binds.get(&pat)) {
                (Pat::Bind { .. }, Some(&local)) => self.declare(local, None),
                _ => bail!("uninitialized `let` with a pattern"),
            };
        };
        let place = self.place_or_temp(init)?;
        if !self.open {
            return Ok(());
        }
        self.bind_matched(pat, place)
    }
}
