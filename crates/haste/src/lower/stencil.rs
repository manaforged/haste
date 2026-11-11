use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Result, bail};
use haste_codegen::{AbiParam, ArgumentPurpose, Function, FunctionBuilder, Type, Value};
use rustc_hash::FxHashMap;

use crate::host::HostSymbol;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{Arena, Shape};
use crate::sema::reach::Reached;

use super::{Lower, Slots, Word, is_static, signature, signature_of};

/// The 128-bit structural footprint digest of a lowered function and its inlined callees.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FootprintId(pub u128);

/// An interned read-only data blob identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RodataId(pub u32);

/// An interned vtable layout identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VtableId(pub u32);

/// The target of an 8-byte pointer relocation inside a read-only data blob.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RodataTarget {
    Rodata(RodataId),
}

/// One 8-byte pointer relocation inside a read-only data blob.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RodataReloc {
    pub offset: u32,
    pub target: RodataTarget,
}

/// One word inside a symbolic vtable definition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VtableEntry {
    Word(Word),
    Super(VtableId),
}

/// A symbolic relocation entry referenced by a function stencil or dictionary.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum SymbolSource {
    Slot(usize),
    Vtable(VtableId),
    Rodata(RodataId),
    StaticCell { index: usize, size: u32 },
    PromotedCell { owner: usize, seq: u32, size: u32 },
    Host(HostSymbol),
    Word(u64),
}

/// The ordered symbolic relocation vector (`D`) of one lowered function instance.
pub type RelocVec = Vec<SymbolSource>;

/// The output of lowering one function into a reusable SSA stencil.
pub struct StencilOutput {
    pub func: Function,
    pub relocs: RelocVec,
}

/// A cached, simplified callee body ready to splice inline into callers.
pub struct InlinedCallee {
    pub item: ItemRef,
    pub footprint: FootprintId,
    pub relocs: RelocVec,
    pub body: Arc<Function>,
    pub returns: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum LowerMode {
    Probe,
    Stencil,
    Dict,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TemplateKey {
    item: ItemRef,
    footprint: FootprintId,
    limit: usize,
    low: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CycleState {
    Acyclic,
    CyclicIncomplete,
    CyclicComplete,
}

impl CycleState {
    fn is_complete(self) -> bool {
        self != Self::CyclicIncomplete
    }
}

/// Cached inlineable callee stencils and footprints across a compilation.
#[derive(Default)]
pub struct Bodies {
    probed: RefCell<FxHashMap<usize, Option<Arc<InlinedCallee>>>>,
    templates: RefCell<FxHashMap<TemplateKey, Option<(Arc<Function>, bool)>>>,
    active: RefCell<Vec<usize>>,
    cyclic: RefCell<FxHashMap<usize, CycleState>>,
}

#[derive(Clone, Copy)]
struct NodeInfo {
    index: u32,
    low: u32,
    on_stack: bool,
    self_loop: bool,
    complete: bool,
}

struct TarjanRun {
    nodes: FxHashMap<usize, NodeInfo>,
    stack: Vec<usize>,
    clock: u32,
}

impl TarjanRun {
    fn enter(&mut self, u: usize, (edges, self_loop, complete): (Vec<usize>, bool, bool)) -> (usize, usize, Vec<usize>) {
        self.clock += 1;
        let info = NodeInfo { index: self.clock, low: self.clock, on_stack: true, self_loop, complete };
        self.nodes.insert(u, info);
        self.stack.push(u);
        (u, 0, edges)
    }

    fn finish(&mut self, u: usize, parent: Option<usize>, cyclic: &mut FxHashMap<usize, CycleState>) -> Option<bool> {
        let u_info = self.nodes[&u];
        let ans = (u_info.low == u_info.index).then(|| self.pop_scc(u, cyclic));
        if let Some(p) = parent {
            let u_now = self.nodes[&u];
            if let Some(p_info) = self.nodes.get_mut(&p) {
                p_info.low = p_info.low.min(u_now.low);
                p_info.complete &= u_now.complete;
            }
        }
        ans
    }

    fn pop_scc(&mut self, u: usize, cyclic: &mut FxHashMap<usize, CycleState>) -> bool {
        let mut scc = Vec::new();
        let mut self_loop = false;
        let mut complete = true;
        while let Some(w) = self.stack.pop() {
            if let Some(info) = self.nodes.get_mut(&w) {
                info.on_stack = false;
                self_loop |= info.self_loop;
                complete &= info.complete;
            }
            scc.push(w);
            if w == u {
                break;
            }
        }
        let is_cyc = scc.len() > 1 || self_loop;
        let state = match (is_cyc, complete) {
            (true, true) => Some(CycleState::CyclicComplete),
            (true, false) => Some(CycleState::CyclicIncomplete),
            (false, true) => Some(CycleState::Acyclic),
            (false, false) => None,
        };
        for &w in &scc {
            if let Some(st) = state {
                cyclic.insert(w, st);
            }
            if !complete && let Some(info) = self.nodes.get_mut(&w) {
                info.complete = false;
            }
        }
        is_cyc
    }
}

impl Bodies {
    /// Drops all cached inlineable bodies and footprints.
    pub fn clear(&self) {
        self.clear_transient();
        self.cyclic.borrow_mut().clear();
    }

    /// Drops cached inline probes and templates while retaining any precomputed cycle entries.
    pub fn clear_transient(&self) {
        self.probed.borrow_mut().clear();
        self.templates.borrow_mut().clear();
        self.active.borrow_mut().clear();
    }

    pub(super) fn is_cyclic(&self, cx: &mut Context<'_>, reached: &Reached, root: usize) -> bool {
        if let Some(&st) = self.cyclic.borrow().get(&root) {
            return st != CycleState::Acyclic;
        }
        if reached.is_untyped(root) {
            return false;
        }
        if !super::may_inline(cx, reached, root) {
            self.cyclic.borrow_mut().insert(root, CycleState::Acyclic);
            return false;
        }
        self.tarjan_from(cx, reached, root)
    }

    fn is_done_callee(&self, v: usize) -> bool {
        self.cyclic.borrow().get(&v).is_some_and(|st| st.is_complete())
    }

    fn tarjan_from(&self, cx: &mut Context<'_>, reached: &Reached, root: usize) -> bool {
        let mut run = TarjanRun { nodes: FxHashMap::default(), stack: Vec::new(), clock: 0 };
        let mut frames = vec![run.enter(root, self.node_edges(cx, reached, root))];
        let mut root_ans = false;
        while let Some(top) = frames.last_mut() {
            let u = top.0;
            if top.1 < top.2.len() {
                let v = top.2[top.1];
                top.1 += 1;
                self.step_edge(cx, reached, &mut run, &mut frames, u, v);
            } else if let Some((done, _, _)) = frames.pop() {
                let popped = run.finish(done, frames.last().map(|f| f.0), &mut self.cyclic.borrow_mut());
                if done == root && let Some(ans) = popped {
                    root_ans = ans;
                }
            }
        }
        root_ans
    }

    fn step_edge(&self, cx: &mut Context<'_>, reached: &Reached, run: &mut TarjanRun, frames: &mut Vec<(usize, usize, Vec<usize>)>, u: usize, v: usize) {
        if self.is_done_callee(v) {
            return;
        }
        if let Some(&v_info) = run.nodes.get(&v) {
            if let Some(u_info) = run.nodes.get_mut(&u) {
                if v_info.on_stack {
                    u_info.low = u_info.low.min(v_info.index);
                } else {
                    u_info.complete &= v_info.complete;
                }
            }
            return;
        }
        frames.push(run.enter(v, self.node_edges(cx, reached, v)));
    }

    fn node_edges(&self, cx: &mut Context<'_>, reached: &Reached, u: usize) -> (Vec<usize>, bool, bool) {
        let mut info = NodeInfo { index: 0, low: 0, on_stack: false, self_loop: false, complete: true };
        let mut edges = Vec::new();
        if let Some(slice) = reached.direct_calls(u) {
            for &v in slice {
                self.scan_callee(cx, reached, u, v, &mut info, &mut edges);
            }
            return (edges, info.self_loop, info.complete);
        }
        let Ok(typed) = &reached.list[u].1 else { return (edges, false, true) };
        for (item, substs) in typed.calls() {
            if cx.intrinsic(item).is_some() {
                continue;
            }
            match reached.index.get(&(item, substs)) {
                Some(&v) => self.scan_callee(cx, reached, u, v, &mut info, &mut edges),
                None => info.complete = false,
            }
        }
        (edges, info.self_loop, info.complete)
    }

    fn scan_callee(&self, cx: &mut Context<'_>, reached: &Reached, u: usize, v: usize, info: &mut NodeInfo, edges: &mut Vec<usize>) {
        if v == u {
            info.self_loop = true;
            return;
        }
        if self.is_done_callee(v) {
            return;
        }
        if reached.is_untyped(v) {
            info.complete = false;
            return;
        }
        if !super::may_inline(cx, reached, v) {
            self.cyclic.borrow_mut().insert(v, CycleState::Acyclic);
            return;
        }
        edges.push(v);
    }

    pub(super) fn lookup_probed(&self, index: usize) -> Option<Option<Arc<InlinedCallee>>> {
        self.probed.borrow().get(&index).cloned()
    }

    pub(super) fn push_active(&self, index: usize) -> Result<usize, usize> {
        let mut active = self.active.borrow_mut();
        if let Some(pos) = active.iter().position(|&at| at == index) {
            return Err(pos);
        }
        let depth = active.len();
        active.push(index);
        Ok(depth)
    }

    pub(super) fn pop_active(&self) {
        self.active.borrow_mut().pop();
    }
}

pub(super) fn item_bits(item: ItemRef) -> u128 {
    (u128::from(item.krate) << 64) | (u128::from(item.file) << 32) | u128::from(item.item)
}

/// Computes the 128-bit footprint and symbolic relocations of reached function `index` without allocating IR instructions.
pub fn probe_footprint(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize) -> Result<(FootprintId, RelocVec)> {
    let (_, footprint, relocs, _) = run_with_cycle(cx, reached, slots, index, LowerMode::Probe)?;
    Ok((footprint, relocs))
}

/// Lowers reached function `index` into an SSA stencil with symbolic relocations.
pub fn lower_stencil(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize) -> Result<StencilOutput> {
    let (func, _, relocs, _) = run_with_cycle(cx, reached, slots, index, LowerMode::Stencil)?;
    Ok(StencilOutput { func, relocs })
}

/// Lowers reached function `index` into a shared dictionary-parameterized SSA body and its relocation vector.
pub fn lower_dict(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize) -> Result<(Function, RelocVec)> {
    let (func, _, relocs, _) = run_with_cycle(cx, reached, slots, index, LowerMode::Dict)?;
    Ok((func, relocs))
}

pub(super) fn resolve_inline(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize, limit: usize) -> Result<(Option<Arc<InlinedCallee>>, usize)> {
    let bodies = slots.bodies();
    if bodies.is_cyclic(cx, reached, index) {
        return Ok((None, usize::MAX));
    }
    if let Some(cached) = bodies.lookup_probed(index) {
        return Ok((cached, usize::MAX));
    }
    let depth = match bodies.push_active(index) {
        Ok(depth) => depth,
        Err(pos) => return Ok((None, pos)),
    };
    let wanted0 = slots.wanted_len();
    let outcome = build_inline(cx, reached, slots, index, limit, wanted0);
    bodies.pop_active();
    let (result, cycle) = outcome?;
    if cycle > depth && slots.wanted_len() == wanted0 {
        bodies.probed.borrow_mut().insert(index, result.clone());
    }
    Ok((result, cycle))
}

fn has_promoted_cell(relocs: &[SymbolSource]) -> bool {
    relocs.iter().any(|r| matches!(r, SymbolSource::PromotedCell { .. }))
}

fn build_inline(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize, limit: usize, wanted0: usize) -> Result<(Option<Arc<InlinedCallee>>, usize)> {
    let item = reached.list[index].0.0;
    let (_, footprint, relocs, probe_cycle) = match run_with_cycle(cx, reached, slots, index, LowerMode::Probe) {
        Ok(out) => out,
        Err(err) if slots.wanted_len() > wanted0 => return Err(err),
        Err(_) => return Ok((None, usize::MAX)),
    };
    if has_promoted_cell(&relocs) {
        return Ok((None, probe_cycle));
    }
    let key = TemplateKey { item, footprint, limit, low: slots.low_entries() };
    if let Some(template) = slots.bodies().templates.borrow().get(&key).cloned() {
        let built = template.map(|(body, returns)| Arc::new(InlinedCallee { item, footprint, relocs, body, returns }));
        return Ok((built, probe_cycle));
    }
    let (func, _, _, ir_cycle) = match run_with_cycle(cx, reached, slots, index, LowerMode::Stencil) {
        Ok(out) => out,
        Err(err) if slots.wanted_len() > wanted0 => return Err(err),
        Err(_) => return Ok((None, probe_cycle)),
    };
    let cycle = probe_cycle.min(ir_cycle);
    let template = simplify_inline(func, limit);
    let depth = slots.bodies().active.borrow().len().saturating_sub(1);
    if cycle > depth && slots.wanted_len() == wanted0 {
        slots.bodies().templates.borrow_mut().insert(key, template.clone());
    }
    let built = template.map(|(body, returns)| Arc::new(InlinedCallee { item, footprint, relocs, body, returns }));
    Ok((built, cycle))
}

fn simplify_inline(mut func: Function, limit: usize) -> Option<(Arc<Function>, bool)> {
    if !func.can_inline() {
        return None;
    }
    haste_codegen::simplify(&mut func);
    (func.size() <= limit).then(|| {
        let returns = func.has_return();
        (Arc::new(func), returns)
    })
}

fn run_with_cycle(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize, mode: LowerMode) -> Result<(Function, FootprintId, RelocVec, usize)> {
    let ((item, _), typed) = &reached.list[index];
    let Ok(typed) = typed else { bail!("{} did not type-check", cx.name(*item)) };
    let source = match &typed.shape {
        Shape::Closure(closure) => closure.parent,
        _ => *item,
    };
    let arena = match &typed.shape {
        Shape::Body | Shape::Closure(_) => Arena::Body(cx.body(source)?),
        Shape::Anon(anon) => anon.arena.clone(),
        Shape::Glue(_) | Shape::Virtual(_) | Shape::Untuple(_) | Shape::Ctor(_) | Shape::Clone(_) => Arena::Body(Rc::default()),
    };
    let src = cx.body_src(source);
    let overflow_checks = cx.outline.crates[source.krate as usize].env.overflow_checks();
    let mut sig = if is_static(cx, *item) { signature_of(cx, None, [])? } else { signature(cx, typed)? };
    if mode == LowerMode::Dict {
        sig.params.push(AbiParam::special(Type::I64, ArgumentPurpose::Dict));
    }
    let mut func = Function::new(sig);
    let b = match mode {
        LowerMode::Probe => FunctionBuilder::new_probe(&mut func),
        LowerMode::Stencil | LowerMode::Dict => FunctionBuilder::new(&mut func),
    };
    let mut relocs = Vec::new();
    let (digest, cycle) = {
        let mut lower = Lower {
            cx,
            reached,
            slots,
            body: arena.body(),
            src,
            typed,
            b,
            locals: vec![None; typed.locals.len()],
            loops: Vec::new(),
            try_exits: Vec::new(),
            open: true,
            dead: None,
            sret: None,
            dict_param: None,
            mode,
            relocs: &mut relocs,
            promoted_seq: 0,
            cycle_root: usize::MAX,
            owned: Vec::new(),
            flags: Vec::new(),
            moved: Default::default(),
            extended: Default::default(),
            pending: Vec::new(),
            index,
            resume: None,
            overflow_checks,
        };
        lower.run()?;
        let digest = if mode == LowerMode::Probe { FootprintId(lower.b.finish_probe()) } else { FootprintId(0) };
        (digest, lower.cycle_root)
    };
    Ok((func, digest, relocs, cycle))
}

impl Lower<'_, '_> {
    pub(super) fn push_reloc(&mut self, src: SymbolSource) -> u32 {
        let idx = self.relocs.len() as u32;
        self.relocs.push(src);
        idx
    }

    pub(super) fn emit_slot(&mut self, src: SymbolSource) -> Value {
        let low = self.slots.low_entries();
        let idx = self.push_reloc(src);
        match self.mode {
            LowerMode::Dict => {
                let dict = self.dict_param.expect("dictionary parameter is set");
                self.ins().load(Type::I64, dict, (idx * 8) as i32)
            }
            LowerMode::Probe | LowerMode::Stencil if low => self.ins().slot_addr(idx),
            LowerMode::Probe | LowerMode::Stencil => self.ins().symbol(idx),
        }
    }

    pub(super) fn emit_symbol(&mut self, src: SymbolSource) -> Value {
        let idx = self.push_reloc(src);
        match self.mode {
            LowerMode::Dict => {
                let dict = self.dict_param.expect("dictionary parameter is set");
                self.ins().load(Type::I64, dict, (idx * 8) as i32)
            }
            LowerMode::Probe | LowerMode::Stencil => self.ins().symbol(idx),
        }
    }
}
