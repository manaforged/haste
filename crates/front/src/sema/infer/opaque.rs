use anyhow::{Result, anyhow, bail};

use crate::names::ItemRef;

use super::super::Context;
use super::super::ty::Ty;
use super::Infer;

/// A return type that holds `impl Trait`: the function, its type arguments, and the type that stands for the return value at the call.
pub(super) type Opaque = (ItemRef, Vec<Ty>, Ty);

impl Infer<'_, '_> {
    /// Ties `ret`, the return type of `item` at `substs` that holds `impl Trait`, to the type that its body returns, once `substs` are known.
    pub(super) fn opaque_ret(&mut self, item: ItemRef, substs: &[Ty], ret: Ty) -> Result<()> {
        self.opaques.push((item, substs.to_vec(), ret));
        self.settle_opaques(false)
    }

    /// Reveals each `impl Trait` return type whose type arguments are known, or every one when `force` is set.
    pub(super) fn settle_opaques(&mut self, force: bool) -> Result<()> {
        let mut at = 0;
        while at < self.opaques.len() {
            let (item, substs, ret) = self.opaques[at].clone();
            if !force && substs.iter().any(|ty| self.has_vars(*ty)) {
                at += 1;
                continue;
            }
            self.opaques.swap_remove(at);
            let substs: Vec<Ty> = substs.into_iter().map(|ty| self.zonk(ty)).collect();
            let found = revealed(self.cx, item, &substs)?;
            self.unify(ret, found)?;
        }
        Ok(())
    }
}

/// The return type of `item` at `substs`, inferring its body once per pass, which also records the closures it defines.
pub(super) fn revealed(cx: &mut Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<Ty> {
    let key = (item, substs.to_vec());
    match cx.reveals.get(&key) {
        Some(Some(ty)) => return Ok(*ty),
        Some(None) => bail!("the `impl Trait` return type of `{}` depends on itself", cx.name(item)),
        None => {}
    }
    cx.reveals.insert(key.clone(), None);
    let ret = super::infer(cx, item, substs).and_then(|typed| typed.ret.ok_or_else(|| anyhow!("`{}` has no return type", cx.name(item))));
    match &ret {
        Ok(ty) => cx.reveals.insert(key, Some(*ty)),
        Err(_) => cx.reveals.remove(&key),
    };
    ret
}
