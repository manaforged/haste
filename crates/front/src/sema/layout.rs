use std::rc::Rc;

use anyhow::{Result, anyhow, bail};
use rustc_abi::{AddressSpace, Float, Integer, LayoutCalculator, LayoutData, Primitive as Abi, ReprOptions, Scalar, StructKind, TargetDataLayout, WrappingRange};
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
        let calc = LayoutCalculator::new(&dl);
        let fields = match self.types.kind(ty).clone() {
            Kind::Prim(prim) => return primitive(&dl, prim),
            Kind::Never => return Ok(Layout::never_type(&dl)),
            Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) if self.thin(pointee) => {
                let nonnull = matches!(self.types.kind(ty), Kind::Ref(..));
                return Ok(scalar(&dl, Abi::Pointer(AddressSpace::ZERO), nonnull.then(|| (1, u64::MAX as u128))));
            }
            Kind::Tuple(items) => items,
            Kind::Array(element, count) => {
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

    fn thin(&self, pointee: Ty) -> bool {
        !matches!(self.types.kind(pointee), Kind::Slice(_) | Kind::Prim(Primitive::Str))
    }
}
