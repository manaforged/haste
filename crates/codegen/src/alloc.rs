mod addr;
pub(crate) mod consts;
mod demote;
mod fold;
mod hints;
pub(crate) mod hoist;
mod preserved;
mod setup;
pub(crate) mod walk;

pub(crate) use addr::{atomic_elidable_addrs, iadd_sub_const};
pub(crate) use hints::{rcpc2_atomic_site, rcpc2_delta_fits};
pub(crate) use preserved::{
    cold_cached_frame, cold_thread_cell, is_preserved_call_op, used_preserved_leaf_regs,
};
pub(crate) use setup::BASE_WEIGHT;

use crate::frame::{Abi, Loc};
use crate::ir::{Block, FloatCC, Function, Inst, InstData, Op, Signature, StackSlot, Type, Value};
use crate::live::{Elided, Live, Sib, live, results};
use crate::park::{Hints, ParkReg, allocate};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AtomicAddrMode {
    None,
    Full,
    Rcpc2,
}

pub(crate) struct Regs<R: 'static> {
    pub saved: &'static [R],
    pub saved_float: &'static [u8],
    pub leaf: &'static [R],
    pub wide_scratch: &'static [R],
    pub op_scratch: &'static [R],
    pub preserved_clobbers: &'static [R],
    pub scratch_mask: fn(&Function, &Op) -> u8,
    pub dict: Option<R>,
    pub float: &'static [u8],
    pub float_branches: &'static [FloatCC],
    pub vectors: bool,
    pub absolute_calls: bool,
    /// The backend calls a runtime helper or a symbol in place instead of loading its address first.
    pub constant_calls: bool,
    pub two_address: bool,
    pub park: bool,
    pub fold_sib: bool,
    pub atomic_addrs: AtomicAddrMode,
    /// The backend zero-extends 8- and 16-bit call results at the call, and 8- and 16-bit narrowing copies between two registers.
    pub call_zext: bool,
    /// The instructions that store a constant into a frame home.
    pub imm_store: fn(i64) -> u64,
    /// The register a 64-bit high product lands in, which the product keeps as its home when nothing else needs it.
    pub high_product: Option<R>,
}

pub(crate) const FOLD_LIMIT: i64 = 1 << 24;

/// Reports whether `data` reads or makes a 128-bit integer.
pub(crate) fn touches_i128(func: &Function, data: &InstData) -> bool {
    let mut wide = data.result.is_some_and(|r| func.ty(r) == Type::I128);
    data.op.for_operands(|v| wide |= func.ty(v) == Type::I128);
    wide
}

impl<R: PartialEq + 'static> Regs<R> {
    /// Reports whether emitting `data` may overwrite `reg` as a wide or op scratch register.
    pub(crate) fn scratches(&self, func: &Function, data: &InstData, reg: R) -> bool {
        let wide = touches_i128(func, data);
        if wide && self.wide_scratch.contains(&reg) {
            return true;
        }
        let mask = if wide {
            u8::MAX
        } else {
            (self.scratch_mask)(func, &data.op)
        };
        self.op_scratch
            .iter()
            .enumerate()
            .any(|(i, r)| (mask >> i) & 1 != 0 && *r == reg)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Location<R> {
    Slot,
    Imm(i64),
    SlotAddr(u32),
    Symbol(u32),
    Reg(R),
    Float(u8),
    Frame(StackSlot, i32),
    Offset(R, i32),
}

impl<R> Location<R> {
    pub(crate) fn is_static(&self) -> bool {
        matches!(
            self,
            Self::Imm(_) | Self::SlotAddr(_) | Self::Symbol(_) | Self::Frame(..)
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CallTarget {
    Abs(i32),
    Slot(u32),
    Runtime(crate::reloc::RuntimeSymbol),
    Symbol(u32),
}

pub(crate) struct Plan<R> {
    pub locations: Vec<Location<R>>,
    pub fused: Vec<bool>,
    pub used: Vec<R>,
    pub used_floats: Vec<u8>,
    pub pairs: Vec<Option<Value>>,
    pub deferred: Vec<bool>,
    pub through: Vec<Option<CallTarget>>,
    pub shared: Vec<Option<u32>>,
    pub parked: Vec<(Inst, Vec<(Value, ParkReg<R>, bool)>)>,
    pub homed: Vec<bool>,
    pub sib: Vec<Option<Sib>>,
    pub elided: Vec<bool>,
    pub zero_ext: Vec<bool>,
    pub clean_bits: Vec<u8>,
    /// The summed block-weighted accesses of the register candidates left in stack slots.
    pub spill_cost: u64,
    pub hoist: hoist::Hoist<R>,
}

fn registered<R>(regs: &Regs<R>, ty: Type) -> bool {
    ty.is_small_int()
        || ((ty.is_float() || (regs.vectors && ty.is_vector())) && !regs.float.is_empty())
}

type Position = Option<(usize, usize)>;

pub(super) struct Uses {
    pub(super) uses: Vec<u32>,
    last_use: Vec<Position>,
    pub(super) def: Vec<Position>,
    def_inst: Vec<Option<Inst>>,
}

impl Uses {
    fn visit(&mut self, inst: Inst, data: &InstData, at: (usize, usize)) {
        data.op.for_operands(|value| {
            self.uses[value.index()] += 1;
            self.last_use[value.index()] = Some(at);
        });
        for result in results(data) {
            self.def[result.index()] = Some(at);
            self.def_inst[result.index()] = Some(inst);
        }
    }
}

fn census(func: &Function) -> Uses {
    let count = func.types.len();
    let mut out = Uses {
        uses: vec![0; count],
        last_use: vec![None; count],
        def: vec![None; count],
        def_inst: vec![None; count],
    };
    for (b, block) in func.blocks.iter().enumerate() {
        for (i, &inst) in block.insts.iter().enumerate() {
            out.visit(inst, &func.insts[inst.index()], (b, i));
        }
    }
    out
}

fn eligible(live: &Live, wanted: &[bool], value: usize) -> bool {
    wanted[value]
        && !live.ranges_of(value).is_empty()
        && !crate::park::crosses_ranges(&live.unwinds, live.ranges_of(value))
}

fn place<R, F>(
    locations: &mut [Location<R>],
    homes: Vec<Option<F>>,
    wrap: impl Fn(F) -> Location<R>,
) {
    for (location, home) in locations.iter_mut().zip(homes) {
        if let Some(reg) = home {
            *location = wrap(reg);
        }
    }
}

fn preserved_tc_mask<R>(
    func: &Function,
    preserve_tc: bool,
    through: &[Option<CallTarget>],
    locations: &[Location<R>],
    elided: &[bool],
) -> Vec<bool> {
    if !preserve_tc {
        return Vec::new();
    }
    let has_tc = func.has_thread_cell();
    let has_fc = through.iter().any(|t| {
        matches!(
            t,
            Some(CallTarget::Runtime(
                crate::reloc::RuntimeSymbol::Malloc | crate::reloc::RuntimeSymbol::Free
            ))
        )
    });
    if has_tc || has_fc {
        func.insts
            .iter()
            .map(|d| is_preserved_call_op(func, &d.op, through, locations, elided))
            .collect()
    } else {
        Vec::new()
    }
}

fn build_candidates<R>(
    func: &Function,
    regs: &Regs<R>,
    live: &Live,
    census: &Uses,
    wanted: &[bool],
    through: &[Option<CallTarget>],
) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
    let count = func.types.len();
    let candidates: Vec<bool> = (0..count).map(|v| eligible(live, wanted, v)).collect();
    let fused: Vec<bool> = (0..count)
        .map(|v| {
            candidates[v] && fold::fuses(func, census, v, regs.float_branches, regs.two_address)
        })
        .collect();
    let ints: Vec<bool> = (0..count)
        .map(|v| candidates[v] && func.types[v].is_small_int() && !fused[v] && through[v].is_none())
        .collect();
    let floats: Vec<bool> = (0..count)
        .map(|v| {
            candidates[v]
                && !fused[v]
                && (func.types[v].is_float() || (regs.vectors && func.types[v].is_vector()))
        })
        .collect();
    (fused, ints, floats)
}

fn restore_promoted<R>(
    locations: &mut [Location<R>],
    promoted_consts: Vec<(usize, i64)>,
    promoted_syms: Vec<(usize, Location<R>)>,
) {
    for (v, loc) in promoted_consts
        .into_iter()
        .map(|(v, i)| (v, Location::Imm(i)))
        .chain(promoted_syms)
    {
        if !matches!(locations[v], Location::Reg(_)) {
            locations[v] = loc;
        }
    }
}

fn zero_vector_consts<R>(func: &Function, locations: &mut [Location<R>]) {
    for data in &func.insts {
        if let (Op::Const(0), Some(res)) = (&data.op, data.result)
            && func.ty(res).is_vector()
            && matches!(locations[res.index()], Location::Slot)
        {
            locations[res.index()] = Location::Imm(0);
        }
    }
}

fn spill_cost<R>(
    weights: Option<&[u64]>,
    ints: &[bool],
    floats: &[bool],
    locations: &[Location<R>],
) -> u64 {
    weights.map_or(0, |weights| {
        (0..locations.len())
            .filter(|&v| (ints[v] || floats[v]) && matches!(locations[v], Location::Slot))
            .map(|v| weights[v])
            .sum()
    })
}

pub(crate) fn plan<R: Copy + PartialEq>(
    func: &Function,
    regs: &Regs<R>,
    (order, copied, weighed, preserve_tc): (&[Block], &[bool], bool, bool),
    params: &[Loc<R>],
    classify: &impl Fn(&Signature) -> Result<Abi<R>, String>,
) -> Plan<R> {
    let count = func.types.len();
    let mut locations = vec![Location::Slot; count];
    fold::consts(func, regs.absolute_calls, &mut locations);
    fold::statics(func, &mut locations);
    let census = census(func);
    let (zero_ext, clean_bits) = hints::zero_ext(func, &locations, &census, regs.call_zext);
    let cfg = weighed.then(|| func.cfg());
    let loops = cfg.as_ref().map(|cfg| cfg.loops());
    let loop_depths = loops.as_ref().map(|loops| &loops.depths[..]);
    let (sib, mut elided) = if weighed && regs.fold_sib {
        addr::sib_addrs(
            func,
            &locations,
            &census,
            &zero_ext,
            loop_depths,
            regs.two_address,
        )
    } else {
        (vec![None; count], vec![false; count])
    };
    let atomic_base = if weighed && regs.atomic_addrs != AtomicAddrMode::None {
        addr::atomic_addrs(
            func,
            regs.atomic_addrs,
            &locations,
            &census,
            &sib,
            &mut elided,
        )
    } else {
        vec![None; count]
    };
    let through = if regs.absolute_calls {
        fold::through(func, &locations, &census, &mut elided, regs.constant_calls)
    } else {
        vec![None; count]
    };
    fold::const_pairs(func, &census, &mut elided);
    let eff = Elided {
        sib: &sib,
        atomic_base: &atomic_base,
        elided: &elided,
    };
    let promoted_consts = loops
        .as_deref()
        .map(|loops| {
            consts::promote_loop_consts(
                func,
                &census,
                loops,
                &elided,
                regs.two_address,
                &mut locations,
            )
        })
        .unwrap_or_default();
    let promoted_syms = (weighed && regs.absolute_calls)
        .then(|| consts::promote_loop_syms(func, &census, loop_depths, &elided, &mut locations))
        .unwrap_or_default();
    let wanted: Vec<bool> = (0..count)
        .map(|value| {
            !elided[value]
                && matches!(locations[value], Location::Slot)
                && registered(regs, func.types[value])
        })
        .collect();
    let tracked: Vec<bool> = (0..count)
        .map(|value| {
            !elided[value]
                && (wanted[value] || (weighed && matches!(locations[value], Location::Slot)))
        })
        .collect();
    let preserved_tc = preserved_tc_mask(func, preserve_tc, &through, &locations, &elided);
    let live = live(func, order, copied, &tracked, eff, &preserved_tc);
    let (fused, ints, floats) = build_candidates(func, regs, &live, &census, &wanted, &through);
    let reuse = if weighed {
        hints::reusable(func, regs, &locations, &live, &elided)
    } else {
        vec![None; count]
    };
    let blocks = cfg
        .as_deref()
        .zip(loops)
        .map(|(cfg, loops)| setup::block_weights(func, cfg, loops));
    let weights = blocks
        .as_deref()
        .map(|blocks| walk::weights(func, blocks, eff));
    let (partners, back_args, edge_args) = if weighed {
        walk::edge_hints(func, cfg.as_deref(), order, blocks.as_deref())
    } else {
        (None, Vec::new(), Vec::new())
    };
    let park = blocks.as_deref().filter(|_| regs.park).map(|blocks| {
        live.sites
            .iter()
            .map(|(_, block)| blocks[block.index()])
            .collect::<Vec<u64>>()
    });
    let copies = if weighed && regs.two_address {
        hints::extend_copies(func, &live, &zero_ext, &ints)
    } else {
        Vec::new()
    };
    let hints = Hints {
        reuse: &reuse,
        weights: weights.as_deref(),
        partners: partners.as_deref(),
        back_args: &back_args,
        edge_args: &edge_args,
        park: park.as_deref(),
        two_address: regs.two_address,
        copies: &copies,
    };
    let (passed_ints, passed_floats) = hints::passed(func, classify, regs.high_product);
    let (homes, used) = allocate(
        func,
        &live,
        &setup::int_class(
            func,
            regs,
            (order, copied, &elided),
            params,
            passed_ints,
            &live.preserved_calls,
        ),
        &ints,
        &hints,
    );
    place(&mut locations, homes, Location::Reg);
    restore_promoted(&mut locations, promoted_consts, promoted_syms);
    let (float_homes, used_floats) = allocate(
        func,
        &live,
        &setup::float_class(regs, params, passed_floats, &live.preserved_calls),
        &floats,
        &hints,
    );
    place(&mut locations, float_homes, Location::Float);
    zero_vector_consts(func, &mut locations);
    let (pairs, deferred) = fold::pairs(func, &locations, &elided);
    let webs = blocks.as_deref().zip(partners.as_deref()).map(|edges| {
        demote::webs(
            func,
            &live,
            edges,
            (eff, &deferred),
            regs.imm_store,
            &mut locations,
        )
    });
    if weighed {
        consts::promote_reg_offsets(
            func,
            regs,
            &live,
            &census,
            (&pairs, &deferred),
            classify,
            &elided,
            &mut locations,
        );
    }
    let parked = if park.is_some() {
        setup::parked(
            func,
            &live,
            &locations,
            (regs.saved, regs.saved_float),
            classify,
        )
    } else {
        Vec::new()
    };
    let mut homed = vec![false; if parked.is_empty() { 0 } else { count }];
    parked
        .iter()
        .flat_map(|(_, values)| values)
        .for_each(|(value, _, _)| homed[value.index()] = true);
    let shared = webs.map_or_else(Vec::new, |webs| {
        crate::color::share(func, &live, &locations, &deferred, &webs)
    });
    let (zero_ext, clean_bits) = if regs.call_zext && hints::has_reg_reduce(func, &locations) {
        hints::zero_ext(func, &locations, &census, true)
    } else {
        (zero_ext, clean_bits)
    };
    if crate::trace::enabled() {
        crate::trace::on_alloc(
            func,
            &live,
            (0..count)
                .filter(|&value| ints[value])
                .map(|value| (value, matches!(locations[value], Location::Slot))),
        );
        if let (Some(loops), Some(blocks)) = (loops, blocks.as_deref()) {
            crate::trace::on_loops(func, &live, (loops, blocks), &ints, |value| {
                matches!(locations[value], Location::Slot)
            });
            crate::trace::on_edge_copies(func, &live, blocks, |value| {
                if let Location::Reg(reg) = locations[value] {
                    Some(reg)
                } else {
                    None
                }
            });
        }
    }
    let spill_cost = spill_cost(weights.as_deref(), &ints, &floats, &locations);
    let hoist = if weighed {
        hoist::Hoist::new(func, &live, &locations)
    } else {
        hoist::Hoist::empty()
    };
    Plan {
        locations,
        fused,
        used,
        used_floats,
        pairs,
        deferred,
        through,
        shared,
        parked,
        homed,
        sib,
        elided,
        zero_ext,
        clean_bits,
        spill_cost,
        hoist,
    }
}

impl<R: Copy> Plan<R> {
    pub(crate) fn location(&self, value: Value) -> Location<R> {
        self.locations[value.index()]
    }

    pub(crate) fn is_static(&self, value: Value) -> bool {
        self.elided[value.index()]
            || self.location(value).is_static()
            || matches!(self.location(value), Location::Offset(..))
    }

    pub(crate) fn const_of(&self, value: Value) -> Option<i64> {
        if let Location::Imm(imm) = self.location(value) {
            Some(imm)
        } else {
            None
        }
    }

    pub(crate) fn reg_of(&self, value: Value) -> Option<R> {
        if let Location::Reg(reg) = self.location(value) {
            Some(reg)
        } else {
            None
        }
    }

    pub(crate) fn parked_at(&self, inst: Inst) -> &[(Value, ParkReg<R>, bool)] {
        self.parked
            .binary_search_by_key(&inst.index(), |(at, _)| at.index())
            .map_or(&[], |at| &self.parked[at].1)
    }

    pub(crate) fn float_of(&self, value: Value) -> Option<u8> {
        if let Location::Float(reg) = self.location(value) {
            Some(reg)
        } else {
            None
        }
    }
}

impl<R: Copy + PartialEq> Plan<R> {
    pub(crate) fn same_home(&self, frame: &crate::frame::Frame<R>, a: Value, b: Value) -> bool {
        crate::frame::same_home(&self.locations, &frame.values, a, b)
    }

    pub(crate) fn moves(
        &self,
        func: &Function,
        frame: &crate::frame::Frame<R>,
        block: Block,
        args: &[Value],
    ) -> bool {
        args.iter()
            .zip(&func.blocks[block.index()].params)
            .any(|(arg, param)| !self.same_home(frame, *arg, *param))
    }
}
