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

/// The typed form of the `Clone::clone` that the compiler writes for the tuple or closure `ty`: one `&ty` parameter, and the clone of each field.
pub fn clone_typed(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> Result<Typed> {
    let Some(Kind::Tuple(fields)) = cloned_tuple(cx, ty).map(|tuple| cx.types.kind(tuple).clone()) else { bail!("a compiler-made clone of a type that is not a tuple or closure") };
    let source = cx.types.intern(Kind::Ref(Mut::Not, ty));
    let calls = detached(cx, item, |infer| fields.iter().map(|&field| infer.trait_call(item, vec![field])).collect::<Result<Vec<_>>>())?;
    Ok(Typed { locals: vec![Local { ty: source, mutable: false }], params: vec![0], ret: Some(ty), shape: Shape::Clone(calls), ..Typed::default() })
}

impl Infer<'_, '_> {
    /// Whether the compiler implements trait `of` for `ty` with no impl to select, as it does `Clone` for a tuple or closure.
    pub(super) fn compiler_impl(&self, of: ItemRef, ty: Ty) -> bool {
        match self.cx.lang(of) {
            Some("clone") => cloned_tuple(self.cx, self.resolve(ty)).is_some(),
            Some("future_trait") => self.coroutine_output(ty).is_some(),
            _ => false,
        }
    }
}
