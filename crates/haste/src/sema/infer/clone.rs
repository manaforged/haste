use anyhow::{Result, bail};

use crate::names::ItemRef;

use super::super::Context;
use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Local, Shape, Typed, detached};

/// The tuple whose fields the compiler-made `Clone::clone` of `ty` clones: a tuple itself, or the captures of a closure.
pub fn cloned_tuple(cx: &Context<'_>, ty: Ty) -> Option<Ty> {
    match cx.types.kind(ty) {
        Kind::Tuple(_) => Some(ty),
        Kind::Closure(..) => cx.closure_env(ty),
        _ => None,
    }
}

/// Whether the compiler implements `Clone` for `ty`, a tuple, a closure, a function item or a function pointer, with no impl to select.
pub fn compiler_clone(cx: &Context<'_>, ty: Ty) -> bool {
    matches!(cx.types.kind(ty), Kind::Tuple(_) | Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..))
}

/// The typed form of the `Clone::clone` that the compiler writes for the tuple, closure, function item or function pointer `ty`: one `&ty` parameter, and the clone of each field.
pub fn clone_typed(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> Result<Typed> {
    let fields = match cloned_tuple(cx, ty).map(|tuple| cx.types.kind(tuple).clone()) {
        Some(Kind::Tuple(fields)) => cx.types.substs(fields).to_vec(),
        None if compiler_clone(cx, ty) => Vec::new(),
        _ => bail!("a compiler-made clone of a type that the compiler does not clone"),
    };
    let source = cx.types.intern(Kind::Ref(Mut::Not, ty));
    let calls = detached(cx, item, |infer| {
        fields.iter().map(|&field| {
            let substs = infer.cx.types.intern_substs(&[field]);
            infer.trait_call(item, substs)
        }).collect::<Result<Vec<_>>>()
    })?;
    Ok(Typed { locals: vec![Local { ty: source }], params: vec![0], ret: Some(ty), shape: Shape::Clone(calls), ..Typed::default() })
}

impl Infer<'_, '_> {
    /// Whether the compiler implements trait `of` for `ty` with no impl to select, as it does `Clone` for a tuple or closure.
    pub(in crate::sema) fn compiler_impl(&self, of: ItemRef, ty: Ty) -> bool {
        match self.cx.lang(of) {
            Some("clone") => compiler_clone(self.cx, self.resolve(ty)),
            Some("future_trait") => self.coroutine_output(ty).is_some(),
            _ => false,
        }
    }
}
