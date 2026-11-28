//! Semantic analysis of reached function bodies at concrete types, lowered to `haste-codegen` SSA.

pub mod adt;
mod body_scope;
mod borrow;
mod builtin;
pub mod census;
pub mod evaluate;
mod expand;
pub(crate) mod format;
pub mod global_asm;
pub mod impls;
pub mod infer;
pub mod intrinsic;
mod items;
pub mod layout;
mod lookups;
pub mod reach;
mod session;
pub mod substs;
pub mod summary;
pub mod table;
pub mod traits;
pub mod ty;

pub use lookups::{Answer, Cache, Edited};
use lookups::{Naming, SegKey, TypeScope};

use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use crate::body::Body;
use crate::body::sig;
use crate::names::path::Requests;
use crate::names::{ItemRef, ModRef, Names};
use crate::outline::Outline;

use table::Table;
use ty::Types;

pub(super) use crate::sync::{read_lock, write_lock};

/// Everything Sema reads in one pass, with its caches and the lookups that found nothing loaded yet.
pub struct Context<'a> {
    pub outline: &'a Outline,
    pub names: &'a Names,
    pub types: Types,
    pub asks: Requests,
    pub(crate) in_parallel: bool,
    pub(crate) deferred_evals: Vec<(reach::Instance, u32)>,
    store: Cache,
    kids: Table<usize, Arc<Kids>>,
    trait_members: Table<ItemRef, Arc<HashMap<&'a str, ItemRef>>>,
    impl_members: Table<impls::ImplRef, Arc<HashMap<&'a str, ItemRef>>>,
    aliases: Table<ItemRef, Arc<Alias>>,
    pub(super) eval_errors: Table<reach::Instance, String>,
    pub(super) eval_stack: Vec<reach::Instance>,
    pub(super) drop_stack: Vec<ty::Ty>,
    pub(super) drop_cycle_at: usize,
    pub(super) discr_visiting: Vec<ItemRef>,
    pub(super) form_stack: Vec<(infer::Form, usize)>,
    global_asm: Option<Arc<global_asm::Table>>,
    langs: Option<Arc<traits::Langs>>,
    lang_map: std::cell::OnceCell<Arc<lookups::LangMap>>,
    crate_owners: Vec<Option<Arc<HashMap<(u32, Option<u32>), u32>>>>,
    crate_modules: Vec<Option<Arc<HashMap<(u32, u32), u32>>>>,
    crate_heads: Vec<Option<Arc<impls::Index>>>,
    crate_heads_replayed: Vec<bool>,
    module_of_local: HashMap<ItemRef, ModRef>,
    trait_auto_local: std::cell::RefCell<HashMap<ItemRef, bool>>,
    headers_local: HashMap<impls::ImplRef, Arc<sig::ImplHeader>>,
    trait_members_local: std::cell::RefCell<HashMap<ItemRef, Arc<HashMap<&'a str, ItemRef>>>>,
    impl_members_local: std::cell::RefCell<HashMap<impls::ImplRef, Arc<HashMap<&'a str, ItemRef>>>>,
    intrinsics_local: std::cell::RefCell<HashMap<ItemRef, Option<intrinsic::Intrinsic>>>,
    type_defs_local: std::cell::RefCell<HashMap<(TypeScope, SegKey), crate::names::Def>>,
    type_defs_single: std::cell::RefCell<HashMap<(ModRef, crate::names::Sym), crate::names::Def>>,
    selections_local: std::cell::RefCell<
        HashMap<(ItemRef, ty::Ty, substs::Substs), Option<(impls::ImplRef, substs::Substs)>>,
    >,
    scope_traits_local: std::cell::RefCell<HashMap<ModRef, Arc<[ItemRef]>>>,
    pub(super) glues_local: HashMap<ty::Ty, Option<infer::Res>>,
    pub(super) drops_local: HashMap<ty::Ty, bool>,
    pub(super) reveals_local: HashMap<reach::Instance, ty::Ty>,
    pub(crate) ty_facts: HashMap<ty::Ty, u128>,
}

/// A parsed type alias: its type parameters, its generics and where clause, and its target type.
#[derive(Debug)]
pub struct Alias {
    pub params: Vec<sig::Generic>,
    pub generics: Arc<traits::TraitGenerics>,
    pub arena: Body,
    pub ty: crate::body::TyId,
}

type Kids = HashMap<Option<u32>, Arc<[u32]>>;

impl<'a> Context<'a> {
    /// What the body that defines the closure type `ty` inferred for it.
    pub fn closure(&self, ty: ty::Ty) -> Option<Arc<infer::ClosureInfo>> {
        self.store.closures.get(&ty)
    }

    /// Whether the body that defines the closure type `ty` has recorded it.
    pub fn has_closure(&self, ty: ty::Ty) -> bool {
        self.store.closures.contains_key(&ty)
    }

    /// The output of the async fn future `ty` that a call named while its body was still being inferred, from the declared return type.
    pub fn future_output(&self, ty: ty::Ty) -> Option<ty::Ty> {
        self.store.lookups.futures.get(&ty)
    }

    /// Records what the body that defines the closure type `ty` inferred for it.
    pub fn set_closure(&mut self, ty: ty::Ty, info: infer::ClosureInfo) {
        if info.typed.exprs.is_empty()
            && self
                .store
                .closures
                .get(&ty)
                .is_some_and(|cur| !cur.typed.exprs.is_empty())
        {
            return;
        }
        self.store.closures.insert(ty, Arc::new(info));
    }

    /// The tuple of the captures of the closure type `ty`, which is how the closure is laid out and dropped.
    pub fn closure_env(&self, ty: ty::Ty) -> Option<ty::Ty> {
        self.store.closures.get(&ty).map(|info| info.env)
    }

    /// Whether the code of the closure type `ty` takes the address of its environment first, which only a closure that captures something does.
    pub fn closure_takes_env(&self, ty: ty::Ty) -> bool {
        self.store
            .closures
            .get(&ty)
            .is_some_and(|info| !info.closure.captures.is_empty())
    }

    /// The type that `ty` owns when it is a `Box`.
    pub fn boxed(&self, ty: ty::Ty) -> Option<ty::Ty> {
        if ty == ty::Ty::NONE {
            return None;
        }
        match self.types.kind(ty) {
            ty::Kind::Adt(item, args) if self.lang(*item) == Some("owned_box") => {
                self.types.substs(*args).first().copied()
            }
            _ => None,
        }
    }

    /// Whether `ty` is `core::ptr::Unique`, the owning pointer inside `Box` and `Vec`.
    pub fn is_unique(&mut self, ty: ty::Ty) -> bool {
        let ty::Kind::Adt(item, _) = *self.types.kind(ty) else {
            return false;
        };
        self.name(item) == "Unique" && self.qualified(item) == "core::ptr::unique::Unique"
    }

    /// Whether a call of the closure type `ty` consumes the closure, because its body moves a captured value out.
    pub fn closure_consumes(&self, ty: ty::Ty) -> bool {
        self.store
            .closures
            .get(&ty)
            .is_some_and(|info| info.closure.consumes)
    }

    /// A short spelling of `ty`, each type in it first passed through `resolve`.
    pub fn type_text(&self, ty: ty::Ty, resolve: &dyn Fn(ty::Ty) -> ty::Ty) -> String {
        use ty::{Kind, Mut};
        let list = |tys: &[ty::Ty]| {
            tys.iter()
                .map(|&ty| self.type_text(ty, resolve))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match self.types.kind(resolve(ty)).clone() {
            Kind::Adt(item, args) if args.is_empty() => self.name(item).to_owned(),
            Kind::Adt(item, args) => {
                format!("{}<{}>", self.name(item), list(self.types.substs(args)))
            }
            Kind::Ref(Mut::Not, inner) => format!("&{}", self.type_text(inner, resolve)),
            Kind::Ref(Mut::Mut, inner) => format!("&mut {}", self.type_text(inner, resolve)),
            Kind::Tuple(items) => format!("({})", list(self.types.substs(items))),
            Kind::Slice(element) => format!("[{}]", self.type_text(element, resolve)),
            Kind::Prim(primitive) => format!("{primitive:?}").to_lowercase(),
            Kind::Param(index) => format!("Param({index})"),
            Kind::Proj(id) => {
                let def = self.types.proj(id);
                format!(
                    "<{} as {}>::{}",
                    self.type_text(def.self_ty, resolve),
                    self.name(def.of),
                    self.name(def.assoc),
                )
            }
            other => self.callable_text(other, resolve),
        }
    }

    fn callable_text(&self, kind: ty::Kind, resolve: &dyn Fn(ty::Ty) -> ty::Ty) -> String {
        use ty::{Kind, Mut};
        let list = |tys: &[ty::Ty]| {
            tys.iter()
                .map(|&ty| self.type_text(ty, resolve))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let angled = |args: substs::Substs| {
            if self.types.substs(args).is_empty() {
                String::new()
            } else {
                format!("<{}>", list(self.types.substs(args)))
            }
        };
        match kind {
            Kind::FnDef(item, args) => format!("fn {}{}", self.name(item), angled(args)),
            Kind::Closure(item, args, expr) => {
                format!("{{closure#{expr} in {}{}}}", self.name(item), angled(args))
            }
            Kind::FnPtr(params, ret) => {
                format!(
                    "fn({}) -> {}",
                    list(self.types.substs(params)),
                    self.type_text(ret, resolve)
                )
            }
            Kind::UnsafeFnPtr(params, ret) => {
                format!(
                    "unsafe fn({}) -> {}",
                    list(self.types.substs(params)),
                    self.type_text(ret, resolve)
                )
            }
            Kind::Dyn(item, args, ..) => format!("dyn {}{}", self.name(item), angled(args)),
            Kind::Ptr(Mut::Not, inner) => format!("*const {}", self.type_text(inner, resolve)),
            Kind::Ptr(Mut::Mut, inner) => format!("*mut {}", self.type_text(inner, resolve)),
            Kind::Array(element, len) => {
                format!(
                    "[{}; {}]",
                    self.type_text(element, resolve),
                    self.type_text(len, resolve)
                )
            }
            Kind::Never => "!".to_owned(),
            Kind::Value(bits) => bits.to_string(),
            other => format!("{other:?}"),
        }
    }

    /// `FnOnce::call_once`, which every call of a closure value is an instance of.
    pub fn call_once(&mut self) -> Option<ItemRef> {
        let of = self.lang_item("fn_once")?;
        self.trait_member(of, "call_once")
    }

    /// Whether `item` is `FnOnce::call_once` or the `call_once` of an `FnOnce` impl, which take the callee by value.
    pub fn takes_by_value(&mut self, item: ItemRef) -> bool {
        let once = self.lang_item("fn_once");
        self.call_once() == Some(item)
            || self
                .owner_impl(item)
                .and_then(|block| self.impl_trait(block))
                .is_some_and(|of| Some(of) == once)
    }

    /// Whether `instance` names, in its function or its type arguments, an item that a body declared before an edit and declares no longer.
    pub fn undeclared(&self, instance: &reach::Instance) -> bool {
        let gone = |item: ItemRef| {
            item.outer().is_some_and(|(outer, index)| {
                self.expansion(outer)
                    .is_none_or(|found| index as usize >= found.file.items.items.len())
            })
        };
        Naming {
            types: &self.types,
            mine: &gone,
            memo: HashMap::default(),
        }
        .touches(instance)
    }
}
