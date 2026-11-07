use std::collections::HashMap;

use haste_codegen::{AbiParam, Block, CallConv, FunctionBuilder, IntCC, Signature, StackSlot, Type, Value};
use hir_def::signatures::FunctionSignature;
use hir_ty::ParamEnvAndCrate;
use hir_ty::db::HirDatabase;
use hir_ty::layout::Layout;
use hir_ty::mir::{
    AggregateKind, BasicBlockId, BinOp, CastKind, LocalId, MirBody, Operand, OperandKind, PlaceTy, ProjectionElem,
    Rvalue, StatementKind, StoredPlace, TerminatorKind, UnOp,
};
use hir_ty::next_solver::infer::{DbInternerInferExt, InferCtxt};
use hir_ty::next_solver::{DbInterner, Ty, TyKind, TypingMode};
use triomphe::Arc;
use rustc_type_ir::inherent::{GenericArgs as _, IntoKind};
use rustc_type_ir::{FloatTy, IntTy, UintTy};

use crate::backend::{Resolved, Shape, Sig, Slots, Target, key_of, pointer_signature, resolve};

mod arith;
mod c_abi;
mod compare;
mod constant;
pub mod derive;
mod drop;
mod inline;
mod moves;
pub mod intrinsic;
pub mod native;
mod pointer;
mod promote;
mod unwind;
mod variant;

#[derive(Clone, Copy)]
struct Scalar {
    ty: Type,
    signed: bool,
    boolean: bool,
}

#[derive(Clone, Copy)]
struct Place<'db> {
    addr: Value,
    ty: PlaceTy<'db>,
    meta: Option<Value>,
}

pub(crate) fn pointee(ty: Ty<'_>) -> Result<Ty<'_>, String> {
    match ty.kind() {
        TyKind::Ref(_, inner, _) | TyKind::RawPtr(inner, _) => Ok(inner),
        _ => Err(format!("deref of {ty:?}")),
    }
}

pub fn layout_of<'db>(db: &'db dyn HirDatabase, env: ParamEnvAndCrate<'db>, ty: Ty<'db>) -> Result<Arc<Layout>, String> {
    db.layout_of_ty(ty.store(), env.store()).map_err(|error| format!("layout of {ty:?}: {error:?}"))
}

pub struct Lower<'a, 'f, 'db> {
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    body: &'db MirBody<'db>,
    builder: &'a mut FunctionBuilder<'f>,
    infcx: InferCtxt<'db>,
    blocks: HashMap<BasicBlockId, Block>,
    frames: HashMap<LocalId, Option<StackSlot>>,
    flags: HashMap<LocalId, StackSlot>,
    move_paths: HashMap<LocalId, Vec<moves::MovePath>>,
    statics: HashMap<LocalId, i64>,
    slots: &'a mut Slots,
    call_conv: CallConv,
    sig: Sig,
    ret_pointer: Option<Value>,
    landing: Option<Block>,
    cleaning: bool,
    exit: Option<Block>,
    inlined: Vec<&'db MirBody<'db>>,
}

impl<'a, 'f, 'db> Lower<'a, 'f, 'db> {
    pub fn new(
        db: &'db dyn HirDatabase,
        env: ParamEnvAndCrate<'db>,
        body: &'db MirBody<'db>,
        builder: &'a mut FunctionBuilder<'f>,
        slots: &'a mut Slots,
        call_conv: CallConv,
        sig: Sig,
    ) -> Self {
        let infcx = DbInterner::new_with(db, env.krate).infer_ctxt().build(TypingMode::PostAnalysis);
        Self {
            db,
            env,
            body,
            builder,
            infcx,
            blocks: HashMap::new(),
            frames: HashMap::new(),
            flags: HashMap::new(),
            move_paths: HashMap::new(),
            statics: HashMap::new(),
            slots,
            call_conv,
            sig,
            ret_pointer: None,
            landing: None,
            cleaning: false,
            exit: None,
            inlined: Vec::new(),
        }
    }

    pub fn run(mut self) -> Result<(), String> {
        let entry = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry);
        self.builder.switch_to_block(entry);
        self.prepare()?;
        let mut values = self.builder.block_params(entry).to_vec().into_iter();
        if self.sig.ret {
            self.ret_pointer = values.next();
        }
        for param in &self.body.param_locals {
            let ty = self.body.locals[*param].ty.as_ref();
            let size = self.layout(ty)?.size.bytes();
            if size != 0 {
                let from = values.next().ok_or("parameter count")?;
                let to = self.local_addr(*param);
                self.copy(to, from, size);
            }
            self.set_flag(*param, true);
        }
        self.lower_blocks()?;
        self.fill_landing()?;
        Ok(())
    }

    fn prepare(&mut self) -> Result<(), String> {
        for (id, _) in self.body.basic_blocks.iter() {
            let block = self.builder.create_block();
            self.blocks.insert(id, block);
        }
        let promoted = self.promoted_locals();
        for (id, local) in self.body.locals.iter() {
            let layout = self.layout(local.ty.as_ref())?;
            let size = layout.size.bytes();
            if size != 0 && promoted.contains(&id) {
                let storage = Box::leak(vec![0_u64; (size as usize).div_ceil(8)].into_boxed_slice());
                self.statics.insert(id, storage.as_ptr() as i64);
                self.frames.insert(id, None);
                continue;
            }
            let slot = (size != 0).then(|| {
                let align = layout.align.abi.bytes().trailing_zeros() as u8;
                                self.builder.create_stack_slot(size as u32, align)
            });
            self.frames.insert(id, slot);
        }
        self.make_flags()
    }

    fn lower_blocks(&mut self) -> Result<(), String> {
        self.builder.ins().jump(self.blocks[&self.body.start_block], &[]);
        let body = self.body;
        for (id, block) in body.basic_blocks.iter() {
            self.builder.switch_to_block(self.blocks[&id]);
            for statement in &block.statements {
                if let StatementKind::Assign(place, rvalue) = &statement.kind {
                    self.assign(place, rvalue)?;
                }
            }
            match &block.terminator {
                Some(terminator) => self.terminator(&terminator.kind)?,
                None => self.trap(),
            }
        }
        Ok(())
    }

    fn layout(&self, ty: Ty<'db>) -> Result<Arc<Layout>, String> {
        layout_of(self.db, self.env, ty)
    }

    fn trap(&mut self) {
        self.builder.ins().trap();
    }

    fn load(&mut self, ty: Type, addr: Value, offset: i32) -> Value {
        self.builder.ins().load(ty, addr, offset)
    }

    fn store(&mut self, value: Value, addr: Value, offset: i32) {
        self.builder.ins().store(value, addr, offset);
    }

    fn copy(&mut self, to: Value, from: Value, size: u64) {
        let mut offset = 0_u64;
        for (ty, width) in [(Type::I64, 8), (Type::I32, 4), (Type::I16, 2), (Type::I8, 1)] {
            while size - offset >= width {
                let value = self.load(ty, from, offset as i32);
                self.store(value, to, offset as i32);
                offset += width;
            }
        }
    }

    fn local_addr(&mut self, local: LocalId) -> Value {
        if let Some(storage) = self.statics.get(&local).copied() {
            return self.builder.ins().iconst(Type::I64, storage);
        }
        match self.frames[&local] {
            Some(slot) => self.builder.ins().stack_addr(Type::I64, slot, 0),
            None => self.builder.ins().iconst(Type::I64, 16),
        }
    }

    fn place(&mut self, stored: &StoredPlace) -> Result<Place<'db>, String> {
        let stored = stored.as_ref();
        let mut place = Place {
            addr: self.local_addr(stored.local),
            ty: PlaceTy::from_ty(self.body.locals[stored.local].ty.as_ref()),
            meta: None,
        };
        for element in stored.projection.as_slice() {
            let previous = place.ty;
            place.ty = previous.projection_ty(&self.infcx, element, self.env.param_env);
            match element {
                ProjectionElem::Deref => {
                    place.meta = None;
                    if self.layout(place.ty.ty)?.is_unsized() {
                        place.meta = Some(self.load(Type::I64, place.addr, 8));
                    }
                    place.addr = self.load(Type::I64, place.addr, 0);
                }
                ProjectionElem::Field(field) => {
                    let layout = self.layout(previous.ty)?;
                    let offset = self.variant_fields(&layout, previous.variant_id)?.fields.offset(field.0 as usize).bytes();
                    place.addr = self.field_address(place.addr, offset, place.ty.ty, place.meta)?;
                    if !self.layout(place.ty.ty)?.is_unsized() {
                        place.meta = None;
                    }
                }
                ProjectionElem::Index(index) => {
                    let size = self.layout(place.ty.ty)?.size.bytes();
                    let index_addr = self.local_addr(*index);
                    let index = self.load(Type::I64, index_addr, 0);
                    let offset = self.builder.ins().imul_imm(index, size as i64);
                    place.addr = self.builder.ins().iadd(place.addr, offset);
                    place.meta = None;
                }
                ProjectionElem::ConstantIndex { offset, from_end: false } => {
                    let size = self.layout(place.ty.ty)?.size.bytes();
                    place.addr = self.builder.ins().iadd_imm(place.addr, (offset * size) as i64);
                    place.meta = None;
                }
                ProjectionElem::ConstantIndex { offset, from_end: true } => {
                    let size = self.layout(place.ty.ty)?.size.bytes() as i64;
                    let length = self.length_of(previous.ty, place.meta)?;
                    let index = self.builder.ins().iadd_imm(length, -(*offset as i64) - 1);
                    let delta = self.builder.ins().imul_imm(index, size);
                    place.addr = self.builder.ins().iadd(place.addr, delta);
                    place.meta = None;
                }
                ProjectionElem::Subslice { from, to } => {
                    let (TyKind::Array(item, _) | TyKind::Slice(item)) = previous.ty.kind() else {
                        return Err(format!("subslice of {:?}", previous.ty));
                    };
                    let size = self.layout(item)?.size.bytes() as i64;
                    if let Some(meta) = place.meta {
                        place.meta = Some(self.builder.ins().iadd_imm(meta, -((*from + *to) as i64)));
                    }
                    place.addr = self.builder.ins().iadd_imm(place.addr, *from as i64 * size);
                }
                ProjectionElem::Downcast(_) => {}
                other => return Err(format!("projection {other:?}")),
            }
        }
        Ok(place)
    }

    fn length_of(&mut self, ty: Ty<'db>, meta: Option<Value>) -> Result<Value, String> {
        match meta {
            Some(meta) => Ok(meta),
            None => {
                let count = self.layout(ty)?.fields.count();
                Ok(self.builder.ins().iconst(Type::I64, count as i64))
            }
        }
    }

    fn operand(&mut self, op: &Operand) -> Result<Place<'db>, String> {
        match &op.kind {
            OperandKind::Copy(place) | OperandKind::Move(place) => {
                self.moved(place)?;
                self.place(place)
            }
            OperandKind::Allocation { allocation } => self.allocation(allocation.as_ref()),
            OperandKind::Constant { konst, ty } => self.constant(konst.as_ref(), ty.as_ref()),
            OperandKind::Static(id) => self.static_cell(*id),
        }
    }

    fn scalar(&self, ty: Ty<'db>) -> Result<Scalar, String> {
        let plain = |ty, signed| Ok(Scalar { ty, signed, boolean: false });
        match ty.kind() {
            TyKind::Bool => Ok(Scalar { ty: Type::I8, signed: false, boolean: true }),
            TyKind::Char => plain(Type::I32, false),
            TyKind::Int(IntTy::I8) => plain(Type::I8, true),
            TyKind::Int(IntTy::I16) => plain(Type::I16, true),
            TyKind::Int(IntTy::I32) => plain(Type::I32, true),
            TyKind::Int(IntTy::I64 | IntTy::Isize) => plain(Type::I64, true),
            TyKind::Int(IntTy::I128) => plain(Type::I128, true),
            TyKind::Uint(UintTy::U128) => plain(Type::I128, false),
            TyKind::Uint(UintTy::U8) => plain(Type::I8, false),
            TyKind::Uint(UintTy::U16) => plain(Type::I16, false),
            TyKind::Uint(UintTy::U32) => plain(Type::I32, false),
            TyKind::Uint(UintTy::U64 | UintTy::Usize) => plain(Type::I64, false),
            TyKind::Float(FloatTy::F32) => plain(Type::F32, false),
            TyKind::Float(FloatTy::F64) => plain(Type::F64, false),
            TyKind::Ref(..) | TyKind::RawPtr(..) | TyKind::FnPtr(..) if self.layout(ty)?.size.bytes() == 8 => {
                plain(Type::I64, false)
            }
            _ => self.repr_scalar(ty),
        }
    }

    fn repr_scalar(&self, ty: Ty<'db>) -> Result<Scalar, String> {
        let layout = self.layout(ty)?;
        let rustc_abi::BackendRepr::Scalar(scalar) = layout.backend_repr else {
            return Err(format!("not a scalar: {ty:?}"));
        };
        match scalar.primitive() {
            rustc_abi::Primitive::Int(integer, signed) => {
                let width = match integer.size().bytes() {
                    1 => Type::I8,
                    2 => Type::I16,
                    4 => Type::I32,
                    8 => Type::I64,
                    _ => Type::I128,
                };
                Ok(Scalar { ty: width, signed, boolean: false })
            }
            rustc_abi::Primitive::Pointer(_) => Ok(Scalar { ty: Type::I64, signed: false, boolean: false }),
            rustc_abi::Primitive::Float(rustc_abi::Float::F32) => Ok(Scalar { ty: Type::F32, signed: false, boolean: false }),
            rustc_abi::Primitive::Float(rustc_abi::Float::F64) => Ok(Scalar { ty: Type::F64, signed: false, boolean: false }),
            other => Err(format!("scalar {other:?}")),
        }
    }

    fn read(&mut self, op: &Operand) -> Result<(Value, Scalar), String> {
        let place = self.operand(op)?;
        let scalar = self.scalar(place.ty.ty)?;
        Ok((self.load(scalar.ty, place.addr, 0), scalar))
    }

    fn assign(&mut self, target: &StoredPlace, rvalue: &Rvalue) -> Result<(), String> {
        let dest = self.place(target)?;
        let layout = self.layout(dest.ty.ty)?;
        if layout.size.bytes() == 0 && !matches!(rvalue, Rvalue::CheckedBinaryOp(..) | Rvalue::Cast(..)) {
            match rvalue {
                Rvalue::Use(op) => {
                    self.operand(op)?;
                }
                Rvalue::Aggregate(_, ops) => {
                    self.operands(ops)?;
                }
                _ => {}
            }
            self.initialized(target);
            return Ok(());
        }
        match rvalue {
            Rvalue::Use(op) => {
                let from = self.operand(op)?;
                self.replace_old(target, dest)?;
                self.copy(dest.addr, from.addr, layout.size.bytes());
            }
            Rvalue::Repeat(op, _) => {
                let from = self.operand(op)?;
                let count = self.builder.ins().iconst(Type::I64, layout.fields.count() as i64);
                let stride = self.layout(from.ty.ty)?.size.bytes();
                self.for_each_item(dest.addr, count, stride as i64, |lower, at| {
                    lower.copy(at, from.addr, stride);
                    Ok(())
                })?;
            }
            Rvalue::CopyForDeref(place) => {
                let from = self.place(place)?;
                self.copy(dest.addr, from.addr, layout.size.bytes());
                return Ok(());
            }
            Rvalue::Ref(_, borrowed) => {
                let place = self.place(borrowed)?;
                if place.ty.ty == dest.ty.ty {
                    self.copy(dest.addr, place.addr, layout.size.bytes());
                    self.initialized(target);
                    return Ok(());
                }
                self.store(place.addr, dest.addr, 0);
                if let Some(meta) = place.meta {
                    self.store(meta, dest.addr, 8);
                }
            }
            Rvalue::Len(place) => {
                let place = self.place(place)?;
                let len = self.length_of(place.ty.ty, place.meta)?;
                self.store(len, dest.addr, 0);
            }
            Rvalue::Discriminant(place) => {
                let place = self.place(place)?;
                self.discriminant(dest, place)?;
            }
            Rvalue::Aggregate(kind, ops) => self.aggregate(dest, &layout, kind, ops)?,
            Rvalue::Cast(kind, op, _) => self.cast(dest, kind, op, layout.size.bytes())?,
            Rvalue::CheckedBinaryOp(op, left, right) => {
                let value = self.binary_rvalue(op, left, right)?;
                self.store(value, dest.addr, 0);
            }
            Rvalue::UnaryOp(op, value) => {
                let (value, scalar) = self.read(value)?;
                let value = match op {
                    UnOp::Neg if scalar.ty.is_float() => self.builder.ins().fneg(value),
                    UnOp::Neg => self.builder.ins().ineg(value),
                    UnOp::Not if scalar.boolean => self.builder.ins().icmp_imm_u(IntCC::Equal, value, 0),
                    UnOp::Not => self.builder.ins().bnot(value),
                };
                self.store(value, dest.addr, 0);
            }
            other => return Err(format!("rvalue {other:?}")),
        }
        self.initialized(target);
        Ok(())
    }

    fn aggregate(
        &mut self,
        dest: Place<'db>,
        layout: &Layout,
        kind: &AggregateKind,
        ops: &[Operand],
    ) -> Result<(), String> {
        match kind {
            AggregateKind::Adt(variant, _) => {
                let mut fields = Vec::with_capacity(ops.len());
                for op in ops {
                    fields.push(self.operand(op)?);
                }
                self.write_variant(dest, *variant, &fields)
            }
            AggregateKind::Union(..) => Err("union value".to_owned()),
            AggregateKind::Array(_) => {
                let mut offset = 0;
                for op in ops {
                    let from = self.operand(op)?;
                    let size = self.layout(from.ty.ty)?.size.bytes();
                    let to = self.builder.ins().iadd_imm(dest.addr, offset as i64);
                    self.copy(to, from.addr, size);
                    offset += size;
                }
                Ok(())
            }
            AggregateKind::Tuple(_) | AggregateKind::Closure(_) => {
                for (index, op) in ops.iter().enumerate() {
                    let from = self.operand(op)?;
                    let size = self.layout(from.ty.ty)?.size.bytes();
                    let offset = layout.fields.offset(index).bytes();
                    let to = self.builder.ins().iadd_imm(dest.addr, offset as i64);
                    self.copy(to, from.addr, size);
                }
                Ok(())
            }
        }
    }

    fn terminator(&mut self, kind: &TerminatorKind) -> Result<(), String> {
        match kind {
            TerminatorKind::Goto { target }
            | TerminatorKind::FalseEdge { real_target: target, .. }
            | TerminatorKind::FalseUnwind { real_target: target, .. } => {
                self.builder.ins().jump(self.blocks[target], &[]);
            }
            TerminatorKind::Drop { place, target, .. } => {
                self.drop_local(place)?;
                self.builder.ins().jump(self.blocks[target], &[]);
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                let (value, scalar) = self.read(discr)?;
                for (case, target) in targets.iter() {
                    let case = self.int_const(scalar.ty, case as i128);
                    let hit = self.builder.ins().icmp(IntCC::Equal, value, case);
                    let next = self.builder.create_block();
                    self.builder.ins().brif(hit, self.blocks[&target], &[], next, &[]);
                    self.builder.switch_to_block(next);
                }
                self.builder.ins().jump(self.blocks[&targets.otherwise()], &[]);
            }
            TerminatorKind::Return => self.leave()?,
            TerminatorKind::Unreachable | TerminatorKind::UnwindResume | TerminatorKind::Abort => self.trap(),
            TerminatorKind::Assert { cond, expected, target, .. } => {
                let (value, _) = self.read(cond)?;
                let fail = self.builder.create_block();
                let good = self.blocks[target];
                if *expected {
                    self.builder.ins().brif(value, good, &[], fail, &[]);
                } else {
                    self.builder.ins().brif(value, fail, &[], good, &[]);
                }
                self.builder.switch_to_block(fail);
                self.trap();
            }
            TerminatorKind::Call { func, args, destination, target, .. } => {
                self.call(func, args, destination, *target)?;
            }
            other => return Err(format!("terminator {other:?}")),
        }
        Ok(())
    }

    fn call(
        &mut self,
        func: &Operand,
        args: &[Operand],
        destination: &StoredPlace,
        target: Option<BasicBlockId>,
    ) -> Result<(), String> {
        let Resolved { target: callee, shape } = match resolve(self.db, self.env, self.body, func) {
            Ok(resolved) => resolved,
            Err(reason) => return Ok(self.site_trap(reason, target)),
        };
        let dest = self.place(destination)?;
        if let Target::Dyn(method, self_ty) = callee {
            if shape == Shape::FnTrait {
                let (_, places) = self.untuple(&callee, args)?;
                self.dyn_item_call(method, self_ty, dest, &places)?;
            } else {
                self.dyn_call(method, self_ty, dest, args)?;
            }
            self.initialized(destination);
            self.end_call(target);
            return Ok(());
        }
        if matches!(callee, Target::Pointer) && shape == Shape::Direct {
            let cell = self.indirect_callee(func)?;
            let places = self.operands(args)?;
            if self.is_native_fn(crate::backend::operand_ty(self.db, self.env, self.body, func)) {
                self.native_call(cell, dest, &places)?;
            } else {
                self.call_pointer(cell, dest, &places)?;
            }
            self.initialized(destination);
            self.end_call(target);
            return Ok(());
        }
        let (receiver, places) = match shape {
            Shape::Direct => (None, self.direct_places(&callee, args)?),
            Shape::FnTrait => self.untuple(&callee, args)?,
            Shape::Select => (None, self.spread(args.first().ok_or("const_eval_select arguments")?)?),
        };
        match &callee {
            Target::Ctor(variant) => self.write_variant(dest, *variant, &places)?,
            Target::Intrinsic(name, generics) => self.intrinsic(name, *generics, dest, &places)?,
            Target::Foreign(name) if name == crate::unwind::RAISE_SYMBOL => {
                let exception = self.load(Type::I64, places.first().ok_or("raise without an exception")?.addr, 0);
                self.raise(exception);
                let after = self.builder.create_block();
                self.builder.switch_to_block(after);
            }
            Target::Foreign(name) if intrinsic::symbol(name).is_none() => {
                return Ok(self.site_trap(format!("no symbol: {name}"), target));
            }
            Target::Foreign(name) => self.foreign(name, dest, &places)?,
            Target::Abort => self.trap_and_continue(),
            Target::Glue(ty) if !self.needs_drop(*ty) => {}
            Target::Pointer => {
                let cell = self.load(Type::I64, receiver.ok_or("fn pointer without a receiver")?, 0);
                self.call_pointer(cell, dest, &places)?;
            }
            Target::Dyn(..) => return Err("dyn call reached the direct path".to_owned()),
            Target::Func(..) | Target::Closure(..) | Target::Glue(_) | Target::Derive(..) => {
                if shape != Shape::Direct || !self.inline(&callee, dest, &places)? {
                    self.direct(&callee, dest, receiver, &places)?;
                }
            }
        }
        self.initialized(destination);
        self.end_call(target);
        Ok(())
    }

    fn site_trap(&mut self, reason: String, target: Option<BasicBlockId>) {
        self.report_trap(&reason);
        self.slots.notes.push(reason);
        self.trap_and_continue();
        self.end_call(target);
    }

    fn report_trap(&mut self, reason: &str) {
        let (text, size) = crate::backend::leak_text(reason);
        let mut signature = Signature::new(self.call_conv);
        signature.params.extend([AbiParam::new(Type::I64); 2]);
        let signature = self.builder.import_signature(signature);
        let code = self.builder.ins().iconst(Type::I64, crate::backend::report_trap as *const () as usize as i64);
        let text = self.builder.ins().iconst(Type::I64, text);
        let size = self.builder.ins().iconst(Type::I64, size);
        self.builder.ins().call_indirect(signature, code, &[text, size]);
    }

    fn end_call(&mut self, target: Option<BasicBlockId>) {
        match target {
            Some(target) => {
                self.builder.ins().jump(self.blocks[&target], &[]);
            }
            None => self.trap(),
        }
    }

    fn spread(&mut self, tuple: &Operand) -> Result<Vec<Place<'db>>, String> {
        let tuple = self.operand(tuple)?;
        let TyKind::Tuple(items) = tuple.ty.ty.kind() else {
            return Err(format!("expected a tuple, got {:?}", tuple.ty.ty));
        };
        let layout = self.layout(tuple.ty.ty)?;
        let mut places = Vec::with_capacity(items.as_slice().len());
        for (index, item) in items.as_slice().iter().enumerate() {
            let offset = layout.fields.offset(index).bytes();
            let addr = self.builder.ins().iadd_imm(tuple.addr, offset as i64);
            places.push(Place { addr, ty: PlaceTy::from_ty(*item), meta: None });
        }
        Ok(places)
    }

    fn direct_places(&mut self, callee: &Target<'db>, args: &[Operand]) -> Result<Vec<Place<'db>>, String> {
        let places = self.operands(args)?;
        let Target::Func(func, _) = callee else {
            return Ok(places);
        };
        if FunctionSignature::of(self.db, *func).abi != rustc_abi::ExternAbi::RustCall {
            return Ok(places);
        }
        let params = self.params_of(callee)?.ok_or("rust-call without a body")?;
        let tupled = places.len() == params.len() && places.iter().zip(&params).all(|(place, param)| place.ty.ty == *param);
        let (Some(tuple), Some((receiver, rest))) = (params.last().filter(|_| !tupled), places.split_first()) else {
            return Ok(places);
        };
        let packed = self.pack(*tuple, rest)?;
        Ok(vec![*receiver, packed])
    }

    fn pack(&mut self, tuple: Ty<'db>, items: &[Place<'db>]) -> Result<Place<'db>, String> {
        let layout = self.layout(tuple)?;
        let align = layout.align.abi.bytes().trailing_zeros() as u8;
                let slot = self.builder.create_stack_slot(layout.size.bytes() as u32, align);
        let addr = self.builder.ins().stack_addr(Type::I64, slot, 0);
        for (index, item) in items.iter().enumerate() {
            let size = self.layout(item.ty.ty)?.size.bytes();
            let to = self.builder.ins().iadd_imm(addr, layout.fields.offset(index).bytes() as i64);
            self.copy(to, item.addr, size);
        }
        Ok(Place { addr, ty: PlaceTy::from_ty(tuple), meta: None })
    }

    fn operands(&mut self, args: &[Operand]) -> Result<Vec<Place<'db>>, String> {
        args.iter().map(|arg| self.operand(arg)).collect()
    }

    fn params_of(&self, callee: &Target<'db>) -> Result<Option<Vec<Ty<'db>>>, String> {
        let (body, skip) = match callee {
            Target::Func(..) => (crate::backend::body_of(self.db, self.env, callee)?, 0),
            Target::Closure(closure, args) => {
                let body = self.db.monomorphized_mir_body_for_closure(*closure, args.store(), self.env.store());
                (body.map_err(|error| format!("{error:?}"))?, 1)
            }
            _ => return Ok(None),
        };
        Ok(Some(body.param_locals[skip..].iter().map(|local| body.locals[*local].ty.as_ref()).collect()))
    }

    fn untuple(
        &mut self,
        callee: &Target<'db>,
        args: &[Operand],
    ) -> Result<(Option<Value>, Vec<Place<'db>>), String> {
        let (receiver, rest) = args.split_first().ok_or("fn trait call without a receiver")?;
        let receiver = self.operand(receiver)?;
        let (mut ty, mut addr) = (receiver.ty.ty, receiver.addr);
        while let TyKind::Ref(_, inner, _) = ty.kind() {
            addr = self.load(Type::I64, addr, 0);
            ty = inner;
        }
        let places = self.operands(rest)?;
        let params = match self.params_of(callee)? {
            Some(params) => Some(params),
            None => ty.callable_sig(self.infcx.interner).map(|sig| rustc_type_ir::Binder::skip_binder(sig).inputs().iter().copied().collect()),
        };
        let [only] = places[..] else {
            return Ok((Some(addr), places));
        };
        if !is_tupled(only.ty.ty, params.as_deref()) {
            return Ok((Some(addr), places));
        }
        Ok((Some(addr), self.spread(&rest[0])?))
    }

    fn scalar_call(&mut self, code: usize, ty: Type, values: &[Value]) -> Value {
        let mut signature = Signature::new(self.call_conv);
        signature.params.extend(values.iter().map(|_| AbiParam::new(ty)));
        signature.returns.push(AbiParam::new(ty));
        let signature = self.builder.import_signature(signature);
        let code = self.builder.ins().iconst(Type::I64, code as i64);
        let call = self.builder.ins().call_indirect(signature, code, values);
        self.builder.inst_results(call)[0]
    }

    fn foreign(&mut self, name: &str, dest: Place<'db>, places: &[Place<'db>]) -> Result<(), String> {
        let code = intrinsic::symbol(name).ok_or_else(|| format!("no symbol: {name}"))?;
        let code = self.builder.ins().iconst(Type::I64, code as i64);
        self.native_call(code, dest, places)
    }

    fn direct(
        &mut self,
        callee: &Target<'db>,
        dest: Place<'db>,
        receiver: Option<Value>,
        places: &[Place<'db>],
    ) -> Result<(), String> {
        let key = key_of(callee).ok_or("not a compiled target")?;
        let mut values = Vec::with_capacity(places.len() + 2);
        let ret = self.layout(dest.ty.ty)?.size.bytes() != 0;
        if ret {
            values.push(dest.addr);
        }
        if let Target::Closure(closure, args) = callee {
            let body = self
                .db
                .monomorphized_mir_body_for_closure(*closure, args.store(), self.env.store())
                .map_err(|error| format!("closure: {error:?}"))?;
            let param = body.locals[body.param_locals[0]].ty.as_ref();
            let data = receiver.ok_or("closure without a receiver")?;
            if matches!(param.kind(), TyKind::Ref(..)) {
                let slot = self.builder.create_stack_slot(8, 3);
                let holder = self.builder.ins().stack_addr(Type::I64, slot, 0);
                self.store(data, holder, 0);
                values.push(holder);
            } else if self.layout(param)?.size.bytes() != 0 {
                values.push(data);
            }
        }
        for place in places {
            if self.layout(place.ty.ty)?.size.bytes() != 0 {
                values.push(place.addr);
            }
        }
        let sig = Sig { params: values.len() - usize::from(ret), ret };
        let slot = self.slots.get(key, sig, self.env.store())?;
        let mut signature = Signature::new(crate::backend::INTERNAL_CONV);
        pointer_signature(&mut signature, sig);
        let signature = self.builder.import_signature(signature);
        let address = self.builder.ins().iconst(Type::I64, self.slots.address(slot));
        let code = self.load(Type::I64, address, 0);
        self.invoke(signature, code, &values);
        Ok(())
    }
}

fn single_depth(ty: Ty<'_>) -> usize {
    match ty.kind() {
        TyKind::Tuple(items) if items.as_slice().len() == 1 => 1 + single_depth(items.as_slice()[0]),
        _ => 0,
    }
}

fn is_tupled<'db>(only: Ty<'db>, params: Option<&[Ty<'db>]>) -> bool {
    if !matches!(only.kind(), TyKind::Tuple(_)) {
        return false;
    }
    match params {
        Some([param]) => single_depth(only) == single_depth(*param) + 1,
        _ => true,
    }
}
