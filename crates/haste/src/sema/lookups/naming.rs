use super::{ItemRef, Substs, ty};
use ty::Types;

fn memo_hit(memo: &[u8], idx: usize) -> Option<bool> {
    let v = *memo.get(idx)?;
    (v != 0).then_some(v == 2)
}

fn memo_set(memo: &mut Vec<u8>, idx: usize, found: bool) -> bool {
    if memo.len() <= idx {
        memo.resize(idx + 1, 0);
    }
    memo[idx] = u8::from(found) + 1;
    found
}

pub(in super::super) struct Naming<'t, F> {
    pub(in super::super) types: &'t Types,
    pub(super) mine: &'t F,
    memo: Vec<u8>,
    substs_memo: Vec<u8>,
}

impl<'t, F: Fn(ItemRef) -> bool> Naming<'t, F> {
    pub(in super::super) fn new(types: &'t Types, mine: &'t F) -> Self {
        Self {
            types,
            mine,
            memo: Vec::new(),
            substs_memo: Vec::new(),
        }
    }

    pub(in super::super) fn touches(
        &mut self,
        (item, substs): &super::super::reach::Instance,
    ) -> bool {
        !self.fn_kept(*item, *substs)
    }

    pub(in super::super) fn fn_kept(&mut self, item: ItemRef, substs: Substs) -> bool {
        !(self.mine)(item) && !self.substs(substs)
    }

    pub(in super::super) fn trait_kept(
        &mut self,
        item: ItemRef,
        ty: ty::Ty,
        substs: Substs,
    ) -> bool {
        self.fn_kept(item, substs) && !self.names(ty)
    }

    pub(in super::super) fn substs(&mut self, substs: Substs) -> bool {
        if substs.is_empty() {
            return false;
        }
        let idx = substs.0 as usize;
        memo_hit(&self.substs_memo, idx).unwrap_or_else(|| {
            let found = self.types.substs(substs).iter().any(|ty| self.names(*ty));
            memo_set(&mut self.substs_memo, idx, found)
        })
    }

    pub(in super::super) fn names(&mut self, ty: ty::Ty) -> bool {
        if ty == ty::Ty::NONE {
            return false;
        }
        let idx = ty.0 as usize;
        if let Some(known) = memo_hit(&self.memo, idx) {
            return known;
        }
        let found = match *self.types.kind(ty) {
            ty::Kind::Tuple(items) => self.substs(items),
            ty::Kind::Ref(_, i)
            | ty::Kind::Ptr(_, i)
            | ty::Kind::Slice(i)
            | ty::Kind::Pat(i, ..) => self.names(i),
            ty::Kind::Array(i, l) => self.names(i) || self.names(l),
            ty::Kind::Adt(i, a)
            | ty::Kind::FnDef(i, a)
            | ty::Kind::Closure(i, a, ..)
            | ty::Kind::Dyn(i, a, ..) => (self.mine)(i) || self.substs(a),
            ty::Kind::FnPtr(p, r) | ty::Kind::UnsafeFnPtr(p, r) => self.names(r) || self.substs(p),
            ty::Kind::Proj(id) => {
                let d = self.types.proj(id);
                (self.mine)(d.of)
                    || (self.mine)(d.assoc)
                    || self.names(d.self_ty)
                    || self.substs(d.args)
                    || self.substs(d.own)
            }
            _ => false,
        };
        memo_set(&mut self.memo, idx, found)
    }
}
