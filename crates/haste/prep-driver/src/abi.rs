use rustc_abi::{BackendRepr, Primitive, Reg, RegKind, Scalar, TargetDataLayout};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, TypingEnv};
use rustc_target::callconv::{ArgAbi, ArgAttributes, CastTarget, FnAbi, PassMode};
use serde_json::{Value, json};

use crate::error::{Error, Result};

/// The calling convention rustc lowered `instance` to: its convention, each argument and the return with their type, size, alignment and passing mode, and whether the return goes through a hidden pointer; and the types of the arguments and the return.
pub fn of<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Result<(Value, Vec<Ty<'tcx>>)> {
    let input = TypingEnv::fully_monomorphized().as_query_input((instance, ty::List::empty()));
    tcx.fn_abi_of_instance(input).map(|abi| describe(tcx, abi)).map_err(|error| Error(format!("{error:?}")))
}

/// The calling convention of a call through a function pointer of type `pointer`, as `of` describes one.
pub fn of_pointer<'tcx>(tcx: TyCtxt<'tcx>, pointer: Ty<'tcx>) -> Result<(Value, Vec<Ty<'tcx>>)> {
    let sig = pointer.fn_sig(tcx);
    let input = TypingEnv::fully_monomorphized().as_query_input((sig, ty::List::empty()));
    tcx.fn_abi_of_fn_ptr(input).map(|abi| describe(tcx, abi)).map_err(|error| Error(format!("{error:?}")))
}

fn describe<'tcx>(tcx: TyCtxt<'tcx>, abi: &FnAbi<'tcx, Ty<'tcx>>) -> (Value, Vec<Ty<'tcx>>) {
    let types = abi.args.iter().chain([&abi.ret]).map(|one| one.layout.ty).collect();
    let value = json!({
        "conv": format!("{:?}", abi.conv),
        "variadic": abi.c_variadic,
        "unwind": abi.can_unwind,
        "sret": matches!(abi.ret.mode, PassMode::Indirect { .. }),
        "ret": arg(tcx, &abi.ret),
        "args": abi.args.iter().map(|one| arg(tcx, one)).collect::<Vec<_>>(),
    });
    (value, types)
}

fn arg<'tcx>(tcx: TyCtxt<'tcx>, arg: &ArgAbi<'tcx, Ty<'tcx>>) -> Value {
    let (layout, dl) = (arg.layout, &tcx.data_layout);
    json!({
        "ty": with_no_trimmed_paths!(layout.ty.to_string()),
        "key": crate::key::ty(tcx, layout.ty),
        "size": layout.size.bytes(),
        "align": layout.align.abi.bytes(),
        "pass": mode(dl, arg),
    })
}

fn mode<'tcx>(dl: &TargetDataLayout, arg: &ArgAbi<'tcx, Ty<'tcx>>) -> Value {
    match &arg.mode {
        PassMode::Ignore => json!({ "kind": "ignore" }),
        PassMode::Direct(attrs) => json!({ "kind": "direct", "regs": regs(dl, &arg.layout.backend_repr), "attrs": [flags(attrs)] }),
        PassMode::Pair(first, second) => {
            json!({ "kind": "pair", "regs": regs(dl, &arg.layout.backend_repr), "attrs": [flags(first), flags(second)] })
        }
        PassMode::Cast { pad_i32, cast } => cast_mode(*pad_i32, cast),
        PassMode::Indirect { attrs, meta_attrs, on_stack } => {
            json!({ "kind": "indirect", "on_stack": on_stack, "unsized": meta_attrs.is_some(), "attrs": [flags(attrs)] })
        }
    }
}

fn cast_mode(pad_i32: bool, cast: &CastTarget) -> Value {
    json!({
        "kind": "cast",
        "pad_i32": pad_i32,
        "prefix": cast.prefix.iter().map(reg).collect::<Vec<_>>(),
        "rest": { "unit": reg(&cast.rest.unit), "total": cast.rest.total.bytes() },
        "rest_offset": cast.rest_offset.map(|offset| offset.bytes()),
        "attrs": [flags(&cast.attrs)],
    })
}

fn regs(dl: &TargetDataLayout, repr: &BackendRepr) -> Value {
    match repr {
        BackendRepr::Scalar(scalar) => json!([{ "class": class(dl, scalar), "offset": 0 }]),
        BackendRepr::ScalarPair { a, b, b_offset } => {
            json!([{ "class": class(dl, a), "offset": 0 }, { "class": class(dl, b), "offset": b_offset.bytes() }])
        }
        BackendRepr::SimdVector { .. } | BackendRepr::SimdScalableVector { .. } => json!([{ "class": "vector", "offset": 0 }]),
        BackendRepr::Memory { .. } => json!([]),
    }
}

fn class(dl: &TargetDataLayout, scalar: &Scalar) -> String {
    let bits = scalar.size(dl).bits();
    match scalar.primitive() {
        Primitive::Int(_, signed) => format!("{}{bits}", if signed { "i" } else { "u" }),
        Primitive::Float(_) => format!("f{bits}"),
        Primitive::Pointer(_) => "ptr".to_string(),
    }
}

fn reg(reg: &Reg) -> String {
    let bits = reg.size.bits();
    match reg.kind {
        RegKind::Integer => format!("i{bits}"),
        RegKind::Float => format!("f{bits}"),
        RegKind::Vector { .. } => format!("v{bits}"),
    }
}

fn flags(attrs: &ArgAttributes) -> String {
    format!("{:?}", attrs.regular)
}
