use std::rc::Rc;

use anyhow::{Result, anyhow, bail};
use rustc_abi::{AddressSpace, BackendRepr, Float, Integer, LayoutCalculator, LayoutData, Primitive as Abi, ReprOptions, Scalar, Size, StructKind, TargetDataLayout, WrappingRange};
use rustc_index::{Idx, IndexVec};

use crate::names::Primitive;

use super::Context;
use super::ty::{Kind, Ty};

/// The x86_64-unknown-linux-gnu data layout, as `rustc --print target-spec-json` reports it.
const DATA_LAYOUT: &str = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128";

/// A field index inside a layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Field(u32);

/// A variant index inside a layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Variant(u32);

impl Idx for Field {
    fn new(index: usize) -> Self {
        Field(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

impl Idx for Variant {
    fn new(index: usize) -> Self {
        Variant(index as u32)
    }
    fn index(self) -> usize {
        self.0 as usize
    }
}

/// The memory layout of one type, computed by rustc's own layout algorithm.
pub type Layout = LayoutData<Field, Variant>;

/// The target's data layout.
pub fn target() -> Result<TargetDataLayout> {
    TargetDataLayout::parse_from_llvm_datalayout_string(DATA_LAYOUT, AddressSpace::ZERO).map_err(|_| anyhow!("bad data layout"))
}

fn scalar(dl: &TargetDataLayout, value: Abi, valid: Option<(u128, u128)>) -> Layout {
    let valid_range = match valid {
        Some((start, end)) => WrappingRange { start, end },
        None => WrappingRange::full(value.size(dl)),
    };
    Layout::scalar(dl, Scalar::Initialized { value, valid_range })
}

fn pointer_scalar(dl: &TargetDataLayout, nonnull: bool) -> Scalar {
    let value = Abi::Pointer(AddressSpace::ZERO);
    let valid_range = if nonnull { WrappingRange { start: 1, end: u64::MAX as u128 } } else { WrappingRange::full(value.size(dl)) };
    Scalar::Initialized { value, valid_range }
}

fn primitive(dl: &TargetDataLayout, primitive: Primitive) -> Result<Layout> {
    use Primitive::*;
    let int = |integer, signed| Abi::Int(integer, signed);
    Ok(match primitive {
        Bool => scalar(dl, int(Integer::I8, false), Some((0, 1))),
        Char => scalar(dl, int(Integer::I32, false), Some((0, 0x10FFFF))),
        I8 | U8 => scalar(dl, int(Integer::I8, primitive == I8), None),
        I16 | U16 => scalar(dl, int(Integer::I16, primitive == I16), None),
        I32 | U32 => scalar(dl, int(Integer::I32, primitive == I32), None),
        I64 | U64 | Isize | Usize => scalar(dl, int(Integer::I64, matches!(primitive, I64 | Isize)), None),
        I128 | U128 => scalar(dl, int(Integer::I128, primitive == I128), None),
        F32 => scalar(dl, Abi::Float(Float::F32), None),
        F64 => scalar(dl, Abi::Float(Float::F64), None),
        other => bail!("no layout for {other:?} yet"),
    })
}

impl Context<'_> {
    /// The layout of `ty`, which must hold no inference variables or parameters.
    pub fn layout(&mut self, ty: Ty) -> Result<Rc<Layout>> {
        if let Some(found) = self.layouts.get(&ty) {
            return Ok(found.clone());
        }
        let computed = Rc::new(self.compute(ty)?);
        self.layouts.insert(ty, computed.clone());
        Ok(computed)
    }

    fn compute(&mut self, ty: Ty) -> Result<Layout> {
        let dl = target()?;
        if let Some(layout) = self.leaf(&dl, ty)? {
            return Ok(layout);
        }
        let calc = LayoutCalculator::new(&dl);
        let fields = match self.types.kind(ty).clone() {
            Kind::Tuple(items) => items,
            Kind::Adt(item, args) => return self.adt_layout(item, &args),
            Kind::Closure(..) => return self.closure_layout(ty),
            Kind::Array(element, count) => {
                let Some(count) = self.types.len(count) else { bail!("layout of an array whose length is not known") };
                let element = self.layout(element)?;
                return calc.array_like::<Field, Variant, ()>(&element, Some(count)).map_err(|error| anyhow!("layout of an array: {error:?}"));
            }
            other => bail!("no layout for {other:?} yet"),
        };
        let layouts = fields.iter().map(|field| self.layout(*field)).collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Layout> = layouts.iter().map(|layout| &**layout).collect();
        let refs: IndexVec<Field, &&Layout> = refs.iter().collect();
        calc.univariant(&refs, &ReprOptions::default(), StructKind::AlwaysSized).map_err(|error| anyhow!("layout: {error:?}"))
    }

    fn leaf(&mut self, dl: &TargetDataLayout, ty: Ty) -> Result<Option<Layout>> {
        Ok(Some(match self.types.kind(ty).clone() {
            Kind::Prim(prim) => primitive(dl, prim)?,
            Kind::Never => Layout::never_type(dl),
            Kind::FnPtr(..) => Layout::scalar(dl, pointer_scalar(dl, true)),
            Kind::Ref(_, pointee) => self.pointer(dl, true, pointee),
            Kind::Ptr(_, pointee) => self.pointer(dl, false, pointee),
            Kind::Pat(base, start, end) => self.pattern(dl, base, WrappingRange { start, end })?,
            _ => return Ok(None),
        }))
    }

    fn pattern(&mut self, dl: &TargetDataLayout, base: Ty, valid_range: WrappingRange) -> Result<Layout> {
        match self.layout(base)?.backend_repr {
            BackendRepr::Scalar(Scalar::Initialized { value, .. }) => Ok(Layout::scalar(dl, Scalar::Initialized { value, valid_range })),
            BackendRepr::ScalarPair(Scalar::Initialized { value, .. }, metadata) => Ok(Layout::scalar_pair(dl, Scalar::Initialized { value, valid_range }, metadata)),
            _ => bail!("a pattern type on a type that is not a scalar or a pointer with metadata"),
        }
    }

    fn closure_layout(&mut self, ty: Ty) -> Result<Layout> {
        let Some(env) = self.closure_env(ty) else { bail!("layout of a closure whose defining body was not inferred") };
        Ok((*self.layout(env)?).clone())
    }

    fn pointer(&self, dl: &TargetDataLayout, nonnull: bool, pointee: Ty) -> Layout {
        let data = pointer_scalar(dl, nonnull);
        if self.thin(pointee) {
            return Layout::scalar(dl, data);
        }
        let metadata = match self.types.kind(pointee) {
            Kind::Dyn(..) => pointer_scalar(dl, true),
            _ => Scalar::Initialized { value: Abi::Int(Integer::I64, false), valid_range: WrappingRange::full(Size::from_bytes(8)) },
        };
        Layout::scalar_pair(dl, data, metadata)
    }

    fn thin(&self, pointee: Ty) -> bool {
        !self.types.kind(pointee).is_unsized()
    }
}
