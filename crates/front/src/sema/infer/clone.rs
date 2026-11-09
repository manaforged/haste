use anyhow::{Result, bail};

use crate::names::ItemRef;

use super::super::Context;
use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Local, Shape, Typed, detached};

/// The typed form of the `Clone::clone` that the compiler writes for the tuple `ty`: one `&ty` parameter, and the clone of each field.
pub fn clone_typed(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> Result<Typed> {
    let Kind::Tuple(fields) = cx.types.kind(ty).clone() else { bail!("a compiler-made clone of a type that is not a tuple") };
    let source = cx.types.intern(Kind::Ref(Mut::Not, ty));
    let calls = detached(cx, item, |infer| fields.iter().map(|&field| infer.trait_call(item, vec![field])).collect::<Result<Vec<_>>>())?;
    Ok(Typed { locals: vec![Local { ty: source, mutable: false }], params: vec![0], ret: Some(ty), shape: Shape::Clone(calls), ..Typed::default() })
}

impl Infer<'_, '_> {
    /// Whether the compiler implements trait `of` for `ty` with no impl to select, as it does `Clone` for a tuple.
    pub(super) fn compiler_impl(&self, of: ItemRef, ty: Ty) -> bool {
        self.cx.lang(of) == Some("clone") && matches!(self.cx.types.kind(self.resolve(ty)), Kind::Tuple(_))
    }
}
