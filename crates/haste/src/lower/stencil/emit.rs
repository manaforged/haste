use std::sync::Arc;

use anyhow::Result;
use haste_codegen::{AbiParam, ArgumentPurpose, Function, FunctionBuilder, Type, Value};

use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{Arena, Shape, Typed};
use crate::sema::reach::Reached;

use super::super::reloc::{RelocPlan, RelocStep, SiteOrigin};
use super::super::{Inline, Lower, Slots, is_static, signature, signature_of};
use super::splice;
use super::{LowerMode, RelocVec, RodataId, SymbolSource};

pub(crate) fn sym_may_unwind(
    cx: &mut Context<'_>,
    reached: &Reached,
    src: Option<&SymbolSource>,
) -> bool {
    match src {
        Some(SymbolSource::Slot(idx)) => reached.may_unwind(cx, *idx),
        Some(SymbolSource::Host(h)) => crate::sema::reach::host_symbol_may_unwind(h),
        _ => true,
    }
}

fn body_arena(cx: &mut Context<'_>, source: ItemRef, shape: &Shape) -> Result<Arena> {
    match shape {
        Shape::Body | Shape::Closure(_) => Ok(Arena::Body(cx.body(source)?)),
        Shape::Anon(anon) => Ok(anon.arena.clone()),
        Shape::Glue(_)
        | Shape::Virtual(_)
        | Shape::Untuple(_)
        | Shape::Ctor(_)
        | Shape::Clone(_)
        | Shape::Asm(_)
        | Shape::GlobalAlloc(_) => Ok(Arena::Body(Arc::default())),
    }
}

pub(super) fn run_with_cycle(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    index: usize,
    mode: LowerMode,
    is_template: bool,
) -> Result<(Function, RelocVec, Arc<RelocPlan>, usize, bool)> {
    let (item, substs) = &reached.list[index].0;
    let concrete = substs.is_empty();
    let typed = reached.typed(index, cx.name(*item))?;
    let typed = typed.scoped(&cx.types);
    let typed: &Typed = &typed;
    let source = match &typed.shape {
        Shape::Closure(closure) => closure.parent,
        _ => *item,
    };
    let arena = body_arena(cx, source, &typed.shape)?;
    let mut sig = if is_static(cx, *item) {
        signature_of(cx, None, [])?
    } else {
        signature(cx, typed)?
    };
    if mode == LowerMode::Dict {
        sig.params
            .push(AbiParam::special(Type::I64, ArgumentPurpose::Dict));
    }
    let mut func = Function::new(sig);
    let abort_on_unwind = matches!(typed.shape, Shape::Body)
        && !cx.fn_abi_may_unwind(*item)
        && reached.may_unwind(cx, index);
    func.abort_on_unwind = abort_on_unwind;
    func.keep_fp = crate::sema::reach::unwind_enabled(cx, reached) || abort_on_unwind;
    let b = FunctionBuilder::new(&mut func);
    let (mut relocs, mut reloc_plan) = (Vec::new(), Vec::new());
    let mut cycle = {
        let mut lower = Lower::new(
            cx,
            (reached, slots, inline),
            (index, source, arena.body(), typed),
            (b, mode),
            (&mut relocs, &mut reloc_plan),
        );
        lower.run()?;
        lower.cycle_root
    };
    let mut simplified = false;
    if splice::should_splice_devirt(slots, &func, mode, &reloc_plan) {
        cycle = splice::splice_devirt_vtable_calls(
            cx,
            (reached, slots),
            inline,
            (&mut func, &mut relocs, &mut reloc_plan),
            (cycle, concrete),
        )?;
        simplified = true;
    }
    if mode == LowerMode::Stencil {
        splice::canonicalize_const_symbols(
            (cx, reached),
            &mut func,
            (&reloc_plan, &relocs),
            (concrete, inline.stack.is_empty()),
        );
        if !is_template {
            if func.is_coroutine() {
                haste_codegen::split_coroutine(&mut func);
            }
            if !func.unwind_drops().is_empty()
                && slots.allow_inline()
                && crate::profile() == crate::outline::tree::Profile::Release
            {
                func.enable_drop_elision();
                for (sym, src) in relocs.iter().enumerate() {
                    if !sym_may_unwind(cx, reached, Some(src)) {
                        func.mark_no_unwind_sym(sym as u32);
                    }
                }
            }
            func.zero_init_unwind_flags();
        }
    }
    Ok((
        func,
        relocs,
        Arc::new(RelocPlan::from_vec(reloc_plan)),
        cycle,
        simplified,
    ))
}

impl Lower<'_, '_> {
    pub(in crate::lower) fn push_reloc(&mut self, src: SymbolSource, step: RelocStep) -> u32 {
        let idx = self.relocs.len() as u32;
        self.relocs.push(src);
        self.reloc_plan.push(step);
        idx
    }

    pub(in crate::lower) fn emit_slot(&mut self, src: SymbolSource, origin: SiteOrigin) -> Value {
        self.emit_slot_with(src, RelocStep::SiteSlot(origin))
    }

    pub(in crate::lower) fn emit_slot_with(&mut self, src: SymbolSource, step: RelocStep) -> Value {
        let self_call = matches!(self.mode, LowerMode::Stencil)
            && matches!(&src, SymbolSource::Slot(i) if *i == self.index);
        let low = self.slots.low_entries();
        self.emit_reloc(src, step, self_call, low)
    }

    pub(in crate::lower) fn emit_symbol(&mut self, src: SymbolSource) -> Value {
        let step = RelocStep::Const(src.clone());
        self.emit_symbol_with(src, step)
    }

    pub(in crate::lower) fn emit_const_rodata(&mut self, bytes: &[u8]) -> Value {
        let id = self.slots.rodata(bytes, &[]);
        self.emit_rodata_id(id, bytes)
    }

    pub(in crate::lower) fn emit_static_rodata(&mut self, index: usize, bytes: &[u8]) -> Value {
        let id = self.slots.static_rodata(index, bytes);
        self.emit_rodata_id(id, bytes)
    }

    pub(in crate::lower) fn emit_rodata_id(&mut self, id: RodataId, bytes: &[u8]) -> Value {
        let idx = self.relocs.len() as u32;
        let val = self.emit_symbol(SymbolSource::Rodata(id));
        if matches!(self.mode, LowerMode::Stencil) && bytes.len() <= 1024 {
            self.b.func.add_rodata_bytes(idx, bytes.to_vec());
        }
        val
    }

    pub(in crate::lower) fn emit_symbol_with(
        &mut self,
        src: SymbolSource,
        step: RelocStep,
    ) -> Value {
        self.emit_reloc(src, step, false, false)
    }

    pub(in crate::lower) fn emit_reloc(
        &mut self,
        src: SymbolSource,
        step: RelocStep,
        self_call: bool,
        low_slot: bool,
    ) -> Value {
        let image = match src {
            SymbolSource::Vtable(_)
            | SymbolSource::Rodata(_)
            | SymbolSource::PromotedCell { .. } => true,
            SymbolSource::StaticCell { index, .. } => {
                !self.cx.foreign(self.reached.list[index].0.0)
            }
            _ => false,
        };
        let idx = self.push_reloc(src, step);
        if self_call {
            self.b.func.add_self_slot(idx);
        }
        match self.mode {
            LowerMode::Dict => {
                let dict = self.dict_param.expect("dictionary parameter is set");
                self.ins().load(Type::I64, dict, (idx * 8) as i32)
            }
            LowerMode::Stencil if low_slot => self.ins().slot_addr(idx),
            LowerMode::Stencil => {
                let addr = self.ins().symbol(idx);
                if image {
                    self.b.func.mark_image_addr(addr);
                }
                addr
            }
        }
    }
}
