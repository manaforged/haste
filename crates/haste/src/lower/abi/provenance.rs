use haste_codegen::{AbiParam, Type};
use rustc_abi::{FieldsShape, Variants};
use rustc_index::Idx;

use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::impls::Head;
use crate::sema::infer::field_types;
use crate::sema::layout::field_offset;
use crate::sema::substs::Substs;
use crate::sema::table::Table;
use crate::sema::ty::{Kind, Mut, Ty};

pub(super) fn apply_param_provenance(
    cx: &mut Context<'_>,
    abi: AbiParam,
    param: Ty,
    offset: i32,
) -> AbiParam {
    if abi.value_type != Type::I64 {
        return abi;
    }
    let Some(core_ty) = field_ty_at_offset(cx, param, offset, 0) else {
        return abi;
    };
    let exclusive_inner = cx.boxed(core_ty).or_else(|| match *cx.types.kind(core_ty) {
        Kind::Ref(Mut::Mut, inner) => Some(inner),
        _ => None,
    });
    if let Some(inner) = exclusive_inner {
        let abi = if is_unpin_like(cx, inner, 0) {
            abi.with_noalias()
        } else {
            abi
        };
        return match pointee_bytes(cx, inner) {
            Some(bytes) => abi.with_deref_size(bytes),
            None => abi,
        };
    }
    let Kind::Ref(Mut::Not, inner) = *cx.types.kind(core_ty) else {
        return abi;
    };
    if !is_freeze_ty(cx, inner, 0) {
        return abi;
    }
    match pointee_bytes(cx, inner) {
        Some(bytes) => abi.with_deref_bytes(bytes),
        None => abi.with_readonly(),
    }
}

fn field_ty_at_offset(cx: &mut Context<'_>, ty: Ty, off: i32, depth: u32) -> Option<Ty> {
    if depth > 8 || off < 0 || ty == Ty::NONE || cx.types.has_params(ty) {
        return None;
    }
    if off == 0 && (matches!(cx.types.kind(ty), Kind::Ref(..)) || cx.boxed(ty).is_some()) {
        return Some(ty);
    }
    match cx.types.kind(ty).clone() {
        Kind::Pat(inner, ..) => field_ty_at_offset(cx, inner, off, depth + 1),
        Kind::Array(elem, _) => {
            if cx.is_unsized(elem).unwrap_or(true) || cx.layout(ty).ok()?.align.bytes() < 8 {
                return None;
            }
            let stride = i32::try_from(cx.layout(elem).ok()?.size.bytes())
                .ok()
                .filter(|&s| s > 0)?;
            field_ty_at_offset(cx, elem, off % stride, depth + 1)
        }
        Kind::Tuple(items) => {
            let fields = cx.types.substs(items).to_vec();
            find_field_at_offset(cx, ty, 0, &fields, off, depth)
        }
        Kind::Adt(item, args) => {
            if cx.boxed(ty).is_some()
                || cx.types.substs_have_params(args)
                || cx.is_unsized(ty).unwrap_or(true)
            {
                return None;
            }
            let layout = cx.layout(ty).ok()?;
            let Variants::Single { index } = layout.variants else {
                return None;
            };
            let slice = cx.types.substs(args).to_vec();
            let fields = field_types(cx, item, &slice)
                .ok()?
                .get(index.index())?
                .clone();
            find_field_at_offset(cx, ty, index.index() as u32, &fields, off, depth)
        }
        _ => None,
    }
}

fn find_field_at_offset(
    cx: &mut Context<'_>,
    ty: Ty,
    variant: u32,
    fields: &[Ty],
    off: i32,
    depth: u32,
) -> Option<Ty> {
    if cx.is_unsized(ty).unwrap_or(true) {
        return None;
    }
    let layout = cx.layout(ty).ok()?;
    if matches!(layout.fields, FieldsShape::Union(_))
        || !matches!(layout.variants, Variants::Single { .. })
        || layout.align.bytes() < 8
    {
        return None;
    }
    for (idx, &fty) in fields.iter().enumerate() {
        let size = i32::try_from(cx.layout(fty).ok()?.size.bytes()).ok()?;
        if size == 0 {
            continue;
        }
        let f_off = field_offset(&layout, variant, idx as u32);
        if f_off >= 0 && off >= f_off && off - f_off < size {
            return field_ty_at_offset(cx, fty, off - f_off, depth + 1);
        }
    }
    None
}

fn pointee_bytes(cx: &mut Context<'_>, ty: Ty) -> Option<u32> {
    if !cx.is_unsized(ty).is_ok_and(|u| !u) {
        return None;
    }
    let bytes = cx.layout(ty).ok()?.size.bytes();
    u32::try_from(bytes).ok().filter(|&b| b > 0)
}

fn is_unpin_like(cx: &mut Context<'_>, ty: Ty, depth: u32) -> bool {
    let mut seen = Vec::new();
    auto_reach(cx, Auto::Unpin, ty, depth, &mut seen).holds
}

/// Whether `ty` contains no interior mutability (`UnsafeCell` / `UnsafePinned`).
pub(crate) fn is_freeze_ty(cx: &mut Context<'_>, ty: Ty, depth: u32) -> bool {
    let mut seen = Vec::new();
    auto_reach(cx, Auto::Freeze, ty, depth, &mut seen).holds
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Auto {
    Unpin,
    Freeze,
}

impl Auto {
    const fn cap(self) -> u32 {
        match self {
            Auto::Unpin | Auto::Freeze => 16,
        }
    }

    fn memo<'c>(self, cx: &'c Context<'_>) -> &'c Table<Ty, (bool, u32)> {
        match self {
            Auto::Unpin => cx.store().unpins(),
            Auto::Freeze => cx.store().freezes(),
        }
    }
}

#[derive(Clone, Copy)]
struct Reach {
    holds: bool,
    height: u32,
    cut: bool,
    cycled: bool,
}

impl Reach {
    const UNKNOWN: Reach = Reach {
        holds: false,
        height: 0,
        cut: true,
        cycled: false,
    };

    const fn known(holds: bool) -> Self {
        Self {
            holds,
            height: 0,
            cut: false,
            cycled: false,
        }
    }

    fn join(&mut self, part: Reach) -> bool {
        self.height = self.height.max(part.height + 1);
        self.cut |= part.cut;
        self.cycled |= part.cycled;
        self.holds &= part.holds;
        part.holds
    }
}

fn auto_reach(cx: &mut Context<'_>, auto: Auto, ty: Ty, depth: u32, seen: &mut Vec<Ty>) -> Reach {
    if depth > auto.cap() {
        return Reach::UNKNOWN;
    }
    if ty == Ty::NONE || cx.types.has_params(ty) {
        return Reach::known(false);
    }
    if seen.contains(&ty) {
        return Reach {
            holds: true,
            height: 0,
            cut: false,
            cycled: true,
        };
    }
    if let Some((holds, height)) = auto.memo(cx).get(&ty)
        && depth + height <= auto.cap()
    {
        return Reach {
            holds,
            height,
            cut: false,
            cycled: false,
        };
    }
    seen.push(ty);
    let found = auto_parts(cx, auto, ty, depth, seen);
    seen.pop();
    if !found.cut && (!found.cycled || depth == 0) {
        let height = if found.cycled { 0 } else { found.height };
        auto.memo(cx).insert(ty, (found.holds, height));
    }
    found
}

fn auto_parts(cx: &mut Context<'_>, auto: Auto, ty: Ty, depth: u32, seen: &mut Vec<Ty>) -> Reach {
    let mut out = Reach::known(true);
    match *cx.types.kind(ty) {
        Kind::Pat(inner, ..) | Kind::Array(inner, _) => {
            out.join(auto_reach(cx, auto, inner, depth + 1, seen));
        }
        Kind::Slice(inner) if auto == Auto::Freeze => {
            out.join(auto_reach(cx, auto, inner, depth + 1, seen));
        }
        Kind::Tuple(items) => {
            for at in 0..cx.types.substs(items).len() {
                let part = cx.types.substs(items)[at];
                if !out.join(auto_reach(cx, auto, part, depth + 1, seen)) {
                    break;
                }
            }
        }
        Kind::Adt(item, args) => return adt_reach(cx, auto, ty, (item, args), depth, seen),
        Kind::Closure(..) if auto == Auto::Unpin => {
            return cx.ensure_closure(ty).map_or(
                Reach {
                    holds: true,
                    ..Reach::UNKNOWN
                },
                |info| Reach::known(!info.closure.coroutine),
            );
        }
        Kind::Closure(..)
        | Kind::Param(_)
        | Kind::Proj(_)
        | Kind::Dyn(..)
        | Kind::Var(_)
        | Kind::IntVar(_)
        | Kind::FloatVar(_) => return Reach::known(false),
        _ => {}
    }
    out
}

fn adt_reach(
    cx: &mut Context<'_>,
    auto: Auto,
    ty: Ty,
    (item, args): (ItemRef, Substs),
    depth: u32,
    seen: &mut Vec<Ty>,
) -> Reach {
    if cx.types.substs_have_params(args) {
        return Reach::known(false);
    }
    if auto == Auto::Unpin && cx.boxed(ty).is_some() {
        return Reach::known(true);
    }
    let blocked = match auto {
        Auto::Unpin => {
            cx.lang(item) == Some("unsafe_pinned")
                || cx.lang_item("unpin").is_some_and(|unpin| {
                    cx.negative_impl(unpin, Head::Adt(item), &[item.krate, unpin.krate])
                })
        }
        Auto::Freeze => matches!(cx.lang(item), Some("unsafe_cell" | "unsafe_pinned")),
    };
    if blocked {
        return Reach::known(false);
    }
    let slice = cx.types.substs(args).to_vec();
    let Ok(variants) = field_types(cx, item, &slice) else {
        return Reach::UNKNOWN;
    };
    let mut out = Reach::known(true);
    for &field in variants.iter().flatten() {
        if !out.join(auto_reach(cx, auto, field, depth + 1, seen)) {
            break;
        }
    }
    out
}
