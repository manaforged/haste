use hir_def::attrs::AttrFlags;
use haste_codegen::{Block, IntCC, Type, Value};
use hir_def::builtin_derive::BuiltinDeriveImplMethod as Method;
use hir_def::{AdtId, CallableDefId, EnumVariantId, FunctionId, ImplId, VariantId};
use hir_ty::db::HirDatabase;
use hir_ty::method_resolution::InherentImpls;
use hir_ty::mir::PlaceTy;
use hir_ty::next_solver::{GenericArg, GenericArgs, SolverDefId, Ty, TyKind};
use rustc_type_ir::IntTy;
use rustc_type_ir::fast_reject::SimplifiedType;
use rustc_type_ir::inherent::{GenericArgs as _, IntoKind};

use super::{Lower, Place};
use crate::backend::{Sig, Target, resolve_def};

const FIELDS_STRUCT: &str = "debug_struct_fields_finish";
const FIELDS_TUPLE: &str = "debug_tuple_fields_finish";
const WRITE_TEXT: &str = "write_str";

struct Part<'db> {
    variant: Option<EnumVariantId>,
    name: String,
    named: bool,
    fields: Vec<(Ty<'db>, u64, String)>,
}

pub fn derive_sig<'db>(lower_size: impl Fn(Ty<'db>) -> Result<u64, String>, method: Method, self_ty: Ty<'db>) -> Result<Sig, String> {
    Ok(match method {
        Method::clone => Sig { params: 1, ret: lower_size(self_ty)? != 0 },
        Method::default => Sig { params: 0, ret: lower_size(self_ty)? != 0 },
        Method::hash => Sig { params: 2, ret: false },
        Method::fmt | Method::cmp | Method::partial_cmp | Method::eq => Sig { params: 2, ret: true },
    })
}

fn impl_named(db: &dyn HirDatabase, imp: ImplId, name: &str) -> Option<FunctionId> {
    imp.impl_items(db).items.iter().find_map(|(found, item)| match item {
        hir_def::AssocItemId::FunctionId(func) if found.as_str() == name => Some(*func),
        _ => None,
    })
}

const DEFAULT_ATTR: &str = "default";

impl<'db> Lower<'_, '_, 'db> {
    fn parts(&self, ty: Ty<'db>) -> Result<Vec<Part<'db>>, String> {
        let layout = self.layout(ty)?;
        match ty.kind() {
            TyKind::Adt(adt, args) => match adt.def_id() {
                AdtId::StructId(id) => {
                    let name = hir_def::signatures::StructSignature::of(self.db, id).name.as_str().to_owned();
                    let offsets = |index: usize| layout.fields.offset(index).bytes();
                    Ok(vec![self.part(id.into(), None, name, args, &offsets)])
                }
                AdtId::UnionId(_) => Ok(vec![Part { variant: None, name: String::new(), named: false, fields: Vec::new() }]),
                AdtId::EnumId(id) => {
                    let mut parts = Vec::new();
                    for (name, value) in &id.enum_variants(self.db).variants {
                        let shape = self.variant_fields(&layout, Some(VariantId::EnumVariantId(value.0)))?;
                        let offsets = |index: usize| shape.fields.offset(index).bytes();
                        parts.push(self.part(value.0.into(), Some(value.0), name.as_str().to_owned(), args, &offsets));
                    }
                    Ok(parts)
                }
            },
            TyKind::Tuple(items) => {
                let fields = items.as_slice().iter().enumerate();
                let fields = fields.map(|(i, item)| (*item, layout.fields.offset(i).bytes(), i.to_string())).collect();
                Ok(vec![Part { variant: None, name: String::new(), named: false, fields }])
            }
            TyKind::Closure(_, args) => self.parts(args.as_closure().tupled_upvars_ty()),
            _ => Ok(vec![Part { variant: None, name: String::new(), named: false, fields: Vec::new() }]),
        }
    }

    fn part(
        &self,
        variant: VariantId,
        id: Option<EnumVariantId>,
        name: String,
        args: GenericArgs<'db>,
        offsets: &dyn Fn(usize) -> u64,
    ) -> Part<'db> {
        let data = variant.fields(self.db);
        let named = matches!(data.shape, hir_def::item_tree::FieldsShape::Record);
        let names: Vec<String> = data.fields().iter().map(|(_, field)| field.name.as_str().to_owned()).collect();
        let types = self.fields_of(variant, args);
        let fields = types.into_iter().zip(names).enumerate().map(|(i, (ty, name))| (ty, offsets(i), name)).collect();
        Part { variant: id, name, named, fields }
    }

    fn tag_of(&mut self, ty: Ty<'db>, addr: Value) -> Result<Value, String> {
        let wide = Ty::new_int(self.infcx.interner, IntTy::Isize);
        let slot = self.builder.create_stack_slot(8, 3);
        let at = self.builder.ins().stack_addr(Type::I64, slot, 0);
        self.discriminant(
            Place { addr: at, ty: PlaceTy::from_ty(wide), meta: None },
            Place { addr, ty: PlaceTy::from_ty(ty), meta: None },
        )?;
        Ok(self.load(Type::I64, at, 0))
    }

    fn each_part(
        &mut self,
        ty: Ty<'db>,
        addr: Value,
        mut work: impl FnMut(&mut Self, &Part<'db>) -> Result<(), String>,
    ) -> Result<(), String> {
        let parts = self.parts(ty)?;
        let enumed = parts.iter().any(|part| part.variant.is_some());
        if !enumed {
            for part in &parts {
                work(self, part)?;
            }
            return Ok(());
        }
        let tag = self.tag_of(ty, addr)?;
        let done = self.builder.create_block();
        for part in &parts {
            let variant = part.variant.ok_or("enum part without a variant")?;
            let value = self.db.const_eval_discriminant(variant).map_err(|error| format!("discriminant: {error:?}"))?;
            let hit = self.builder.ins().icmp_imm_u(IntCC::Equal, tag, value as i64);
            let (run, next) = (self.builder.create_block(), self.builder.create_block());
            self.builder.ins().brif(hit, run, &[], next, &[]);
            self.builder.switch_to_block(run);
            work(self, part)?;
            self.builder.ins().jump(done, &[]);
            self.builder.switch_to_block(next);
        }
        self.trap();
        self.builder.switch_to_block(done);
        Ok(())
    }

    fn call_method(
        &mut self,
        func: FunctionId,
        args: Vec<GenericArg<'db>>,
        dest: Place<'db>,
        places: &[Place<'db>],
    ) -> Result<(), String> {
        let args = GenericArgs::new_from_iter(args);
        let resolved = resolve_def(self.db, self.env, CallableDefId::FunctionId(func), args)?;
        match &resolved.target {
            Target::Func(..) | Target::Closure(..) | Target::Derive(..) => self.direct(&resolved.target, dest, None, places),
            Target::Intrinsic(name, generics) => self.intrinsic(name, *generics, dest, places),
            other => Err(format!("derive call on {}", other.kind_name())),
        }
    }

    fn temp(&mut self, ty: Ty<'db>) -> Result<Place<'db>, String> {
        let layout = self.layout(ty)?;
        let align = layout.align.abi.bytes().trailing_zeros() as u8;
                let slot = self.builder.create_stack_slot(layout.size.bytes().max(1) as u32, align);
        let addr = self.builder.ins().stack_addr(Type::I64, slot, 0);
        Ok(Place { addr, ty: PlaceTy::from_ty(ty), meta: None })
    }

    fn ref_place(&mut self, ty: Ty<'db>, at: Value) -> Place<'db> {
        let region = hir_ty::next_solver::Region::new_erased(self.infcx.interner);
        let ty = Ty::new_imm_ref(self.infcx.interner, region, ty);
        let addr = self.thin_holder(at);
        Place { addr, ty: PlaceTy::from_ty(ty), meta: None }
    }

    pub fn derive(mut self, method: Method, func: FunctionId, args: GenericArgs<'db>) -> Result<(), String> {
        let entry = self.builder.create_block();
        self.builder.append_block_params_for_function_params(entry);
        self.builder.switch_to_block(entry);
        let params = self.builder.block_params(entry).to_vec();
        let self_ty = args.as_slice().first().and_then(|arg| arg.ty()).ok_or("derive without a self type")?;
        let self_ty = self.normalized(self_ty);
        let sig = self.sig;
        let (ret, rest) = if sig.ret { (Some(params[0]), &params[1..]) } else { (None, &params[..]) };
        let ret = ret.unwrap_or_else(|| self.builder.ins().iconst(Type::I64, 16));
        match method {
            Method::clone => self.derive_clone(func, self_ty, ret, rest[0])?,
            Method::default => self.derive_default(func, self_ty, ret)?,
            Method::eq => self.derive_compare(Compare::Eq, func, self_ty, ret, rest)?,
            Method::cmp | Method::partial_cmp => self.derive_compare(Compare::Order, func, self_ty, ret, rest)?,
            Method::hash => self.derive_hash(func, args, self_ty, rest)?,
            Method::fmt => self.derive_debug(self_ty, ret, rest)?,
        }
        self.builder.ins().return_(&[]);
        Ok(())
    }

    fn derive_clone(&mut self, func: FunctionId, ty: Ty<'db>, ret: Value, holder: Value) -> Result<(), String> {
        let size = self.layout(ty)?.size.bytes();
        let from = self.load(Type::I64, holder, 0);
        self.copy(ret, from, size);
        self.each_part(ty, from, |lower, part| {
            for (field, offset, _) in &part.fields {
                let source = lower.builder.ins().iadd_imm(from, *offset as i64);
                let to = lower.builder.ins().iadd_imm(ret, *offset as i64);
                let dest = Place { addr: to, ty: PlaceTy::from_ty(*field), meta: None };
                let reference = lower.ref_place(*field, source);
                lower.call_method(func, vec![(*field).into()], dest, &[reference])?;
            }
            Ok(())
        })
    }

    fn default_variant(&self, ty: Ty<'db>) -> Option<hir_def::EnumVariantId> {
        let TyKind::Adt(adt, _) = ty.kind() else {
            return None;
        };
        let hir_def::AdtId::EnumId(id) = adt.def_id() else {
            return None;
        };
        id.enum_variants(self.db).variants.iter().map(|(_, variant)| variant.0).find(|variant| AttrFlags::has_attr(self.db, (*variant).into(), DEFAULT_ATTR))
    }

    fn derive_default(&mut self, func: FunctionId, ty: Ty<'db>, ret: Value) -> Result<(), String> {
        let parts = self.parts(ty)?;
        if parts.iter().any(|part| part.variant.is_some()) {
            let variant = self.default_variant(ty).ok_or("enum without a default variant")?;
            let dest = Place { addr: ret, ty: PlaceTy::from_ty(ty), meta: None };
            return self.write_variant(dest, variant.into(), &[]);
        }
        for part in &parts {
            for (field, offset, _) in &part.fields {
                let to = self.builder.ins().iadd_imm(ret, *offset as i64);
                let dest = Place { addr: to, ty: PlaceTy::from_ty(*field), meta: None };
                self.call_method(func, vec![(*field).into()], dest, &[])?;
            }
        }
        Ok(())
    }

    fn derive_compare(&mut self, kind: Compare, func: FunctionId, ty: Ty<'db>, ret: Value, rest: &[Value]) -> Result<(), String> {
        let left = self.load(Type::I64, rest[0], 0);
        let right = self.load(Type::I64, rest[1], 0);
        let done = self.builder.create_block();
        let parts = self.parts(ty)?;
        if parts.iter().any(|part| part.variant.is_some()) {
            let (a, b) = (self.tag_of(ty, left)?, self.tag_of(ty, right)?);
            self.compare_tags(kind, a, b, ret, done);
        }
        let result_ty = self.compare_result(func)?;
        self.each_part(ty, left, |lower, part| {
            for (field, offset, _) in &part.fields {
                let at_left = lower.builder.ins().iadd_imm(left, *offset as i64);
                let at_right = lower.builder.ins().iadd_imm(right, *offset as i64);
                let first = lower.ref_place(*field, at_left);
                let second = lower.ref_place(*field, at_right);
                let temp = lower.temp(result_ty)?;
                let args = match kind {
                    Compare::Eq | Compare::Order if is_two_param(lower.db, func) => vec![(*field).into(), (*field).into()],
                    _ => vec![(*field).into()],
                };
                lower.call_method(func, args, temp, &[first, second])?;
                let value = lower.load(Type::I8, temp.addr, 0);
                lower.store(value, ret, 0);
                let stop = match kind {
                    Compare::Eq => lower.builder.ins().icmp_imm_u(IntCC::Equal, value, 0),
                    Compare::Order => lower.builder.ins().icmp_imm_u(IntCC::NotEqual, value, 0),
                };
                let next = lower.builder.create_block();
                lower.builder.ins().brif(stop, done, &[], next, &[]);
                lower.builder.switch_to_block(next);
            }
            Ok(())
        })?;
        let same = self.builder.ins().iconst(Type::I8, kind.same_value());
        self.store(same, ret, 0);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(done);
        Ok(())
    }

    fn compare_result(&self, func: FunctionId) -> Result<Ty<'db>, String> {
        let signature = self.db.callable_item_signature(func.into());
        Ok(signature.skip_binder().skip_binder().output())
    }

    fn compare_tags(&mut self, kind: Compare, a: Value, b: Value, ret: Value, done: Block) {
        let same = self.builder.ins().icmp(IntCC::Equal, a, b);
        let (differ, next) = (self.builder.create_block(), self.builder.create_block());
        self.builder.ins().brif(same, next, &[], differ, &[]);
        self.builder.switch_to_block(differ);
        let value = match kind {
            Compare::Eq => self.builder.ins().iconst(Type::I8, 0),
            Compare::Order => {
                let up = self.builder.ins().icmp(IntCC::SignedGreaterThan, a, b);
                let down = self.builder.ins().icmp(IntCC::SignedLessThan, a, b);
                self.builder.ins().isub(up, down)
            }
        };
        self.store(value, ret, 0);
        self.builder.ins().jump(done, &[]);
        self.builder.switch_to_block(next);
    }

    fn derive_hash(&mut self, func: FunctionId, args: GenericArgs<'db>, ty: Ty<'db>, rest: &[Value]) -> Result<(), String> {
        let from = self.load(Type::I64, rest[0], 0);
        let state = rest[1];
        let hasher = args.as_slice().get(1).copied().ok_or("hash without a hasher")?;
        let state_ty = self.state_ty(hasher)?;
        let nothing = self.temp(<Ty<'db> as rustc_type_ir::inherent::Ty<_>>::new_tup(self.infcx.interner, &[]))?;
        let parts = self.parts(ty)?;
        if parts.iter().any(|part| part.variant.is_some()) {
            let tag = self.tag_of(ty, from)?;
            let wide = Ty::new_int(self.infcx.interner, IntTy::Isize);
            let cell = self.temp(wide)?;
            self.store(tag, cell.addr, 0);
            let first = self.ref_place(wide, cell.addr);
            let second = Place { addr: state, ty: PlaceTy::from_ty(state_ty), meta: None };
            self.call_method(func, vec![wide.into(), hasher], nothing, &[first, second])?;
        }
        self.each_part(ty, from, |lower, part| {
            for (field, offset, _) in &part.fields {
                let at = lower.builder.ins().iadd_imm(from, *offset as i64);
                let first = lower.ref_place(*field, at);
                let second = Place { addr: state, ty: PlaceTy::from_ty(state_ty), meta: None };
                lower.call_method(func, vec![(*field).into(), hasher], nothing, &[first, second])?;
            }
            Ok(())
        })
    }

    fn state_ty(&self, hasher: GenericArg<'db>) -> Result<Ty<'db>, String> {
        let hasher = hasher.ty().ok_or("hasher is not a type")?;
        let region = hir_ty::next_solver::Region::new_erased(self.infcx.interner);
        Ok(Ty::new_imm_ref(self.infcx.interner, region, hasher))
    }

    fn formatter_fn(&self, formatter: Ty<'db>, name: &str) -> Result<FunctionId, String> {
        let TyKind::Adt(adt, _) = formatter.kind() else {
            return Err("formatter is not a struct".to_owned());
        };
        let def = adt.def_id();
        let krate = hir_def::HasModule::module(&def, self.db).krate(self.db);
        let key = SimplifiedType::Adt(SolverDefId::from(def));
        let impls = InherentImpls::for_crate(self.db, krate);
        impls
            .for_self_ty(&key)
            .iter()
            .find_map(|imp| impl_named(self.db, *imp, name))
            .ok_or_else(|| format!("no Formatter::{name}"))
    }

    fn derive_debug(&mut self, ty: Ty<'db>, ret: Value, rest: &[Value]) -> Result<(), String> {
        let from = self.load(Type::I64, rest[0], 0);
        let formatter_holder = rest[1];
        let debug_fmt = self.infcx.interner.lang_items().Debug.ok_or("no Debug trait")?;
        let fmt = debug_fmt.trait_items(self.db).items.iter().find_map(|(_, item)| match item {
            hir_def::AssocItemId::FunctionId(func) => Some(*func),
            _ => None,
        });
        let fmt = fmt.ok_or("Debug without fmt")?;
        let signature = self.db.callable_item_signature(fmt.into());
        let inputs = signature.skip_binder().skip_binder().inputs();
        let formatter_ref = inputs.iter().nth(1).ok_or("fmt without a formatter")?;
        let formatter = super::pointee(*formatter_ref)?;
        let result_ty = signature.skip_binder().skip_binder().output();
        let dest = Place { addr: ret, ty: PlaceTy::from_ty(result_ty), meta: None };
        self.each_part(ty, from, |lower, part| lower.debug_part(part, from, formatter, formatter_holder, dest))
    }

    fn debug_part(&mut self, part: &Part<'db>, from: Value, formatter: Ty<'db>, holder: Value, dest: Place<'db>) -> Result<(), String> {
        let TyKind::Adt(_, formatter_args) = formatter.kind() else {
            return Err("formatter is not a struct".to_owned());
        };
        let name = self.text_place(&part.name);
        let receiver = Place { addr: holder, ty: PlaceTy::from_ty(self.state_ty(formatter.into())?), meta: None };
        if part.fields.is_empty() {
            let func = self.formatter_fn(formatter, WRITE_TEXT)?;
            let target = Target::Func(func, formatter_args);
            return self.direct(&target, dest, None, &[receiver, name]);
        }
        let helper = if part.named { FIELDS_STRUCT } else { FIELDS_TUPLE };
        let func = self.formatter_fn(formatter, helper)?;
        let target = Target::Func(func, formatter_args);
        let body = crate::backend::body_of(self.db, self.env, &target)?;
        let last = *body.param_locals.last().ok_or("fields helper without parameters")?;
        let values_ty = body.locals[last].ty.as_ref();
        let TyKind::Slice(item_ty) = super::pointee(values_ty)?.kind() else {
            return Err("fields helper without a slice".to_owned());
        };
        let dyn_ty = super::pointee(item_ty)?;
        let values = self.debug_values(part, from, dyn_ty)?;
        let values = Place { addr: values, ty: PlaceTy::from_ty(values_ty), meta: None };
        let mut places = vec![receiver, name];
        if part.named {
            let names_local = body.param_locals[body.param_locals.len() - 2];
            let names_ty = body.locals[names_local].ty.as_ref();
            let names = self.names_slice(part, names_ty);
            places.push(names);
        }
        places.push(values);
        self.direct(&target, dest, None, &places)
    }

    fn text_place(&mut self, text: &str) -> Place<'db> {
        let text: &'static str = Box::leak(text.to_owned().into_boxed_str());
        let cell: &'static [usize; 2] = Box::leak(Box::new([text.as_ptr() as usize, text.len()]));
        let addr = self.builder.ins().iconst(Type::I64, cell.as_ptr() as i64);
        let region = hir_ty::next_solver::Region::new_erased(self.infcx.interner);
        let ty = Ty::new_imm_ref(self.infcx.interner, region, Ty::new(self.infcx.interner, TyKind::Str));
        Place { addr, ty: PlaceTy::from_ty(ty), meta: None }
    }

    fn names_slice(&mut self, part: &Part<'db>, ty: Ty<'db>) -> Place<'db> {
        let mut pairs = Vec::with_capacity(part.fields.len() * 2);
        for (_, _, name) in &part.fields {
            let text: &'static str = Box::leak(name.clone().into_boxed_str());
            pairs.push(text.as_ptr() as usize);
            pairs.push(text.len());
        }
        let pairs: &'static [usize] = Box::leak(pairs.into_boxed_slice());
        let cell: &'static [usize; 2] = Box::leak(Box::new([pairs.as_ptr() as usize, part.fields.len()]));
        let addr = self.builder.ins().iconst(Type::I64, cell.as_ptr() as i64);
        Place { addr, ty: PlaceTy::from_ty(ty), meta: None }
    }

    fn debug_values(&mut self, part: &Part<'db>, from: Value, dyn_ty: Ty<'db>) -> Result<Value, String> {
        let count = part.fields.len();
                let slot = self.builder.create_stack_slot((count * 16) as u32, 3);
        let array = self.builder.ins().stack_addr(Type::I64, slot, 0);
        for (index, (field, offset, _)) in part.fields.iter().enumerate() {
            let table = self.vtable(*field, dyn_ty)?;
            let at = self.builder.ins().iadd_imm(from, *offset as i64);
            let table = self.builder.ins().iconst(Type::I64, table);
            self.store(at, array, (index * 16) as i32);
            self.store(table, array, (index * 16 + 8) as i32);
        }
        let holder = self.thin_holder(array);
        let count = self.builder.ins().iconst(Type::I64, count as i64);
        self.store(count, holder, 8);
        Ok(holder)
    }
}

#[derive(Clone, Copy)]
enum Compare {
    Eq,
    Order,
}

impl Compare {
    fn same_value(self) -> i64 {
        match self {
            Compare::Eq => 1,
            Compare::Order => 0,
        }
    }
}

fn is_two_param(db: &dyn HirDatabase, func: FunctionId) -> bool {
    let hir_def::ItemContainerId::TraitId(owner) = func.loc(db).container else {
        return false;
    };
    !hir_def::hir::generics::GenericParams::of(db, owner.into()).is_empty()
}

