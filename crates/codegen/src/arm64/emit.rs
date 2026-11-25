use super::asm::{Alu, Cond, FP, Label, Mem, Reg, SP, Size, Vreg, ZR};
use super::frame::{Abi, Loc, Packing};
use super::fuse;
use super::{A, C, Emit, IP, LR, V0, V1};
use crate::Site;
use crate::alloc::{Location, Plan};
use crate::emit::{Backend, Tails, emit_goto};
use crate::frame::{
    ArgStep, EdgeScratch, Emitter, Ext, Frame, INCOMING, ShrinkWrap, Step, arg_steps, edge_steps,
    reg_steps,
};
use crate::ir::{ArgumentPurpose, Binary, Block, Convert, IntCC, Op, SigRef, Value};
use crate::park::ParkReg;

const ARG_SCRATCH: Reg = C;

impl Emit<'_> {
    pub(super) fn trap_target(&mut self) -> Label {
        *self.trap_label.get_or_insert_with(|| self.asm.label())
    }

    pub(super) fn trapz(&mut self, condition: Value) {
        if let Some(known) = self.plan.const_of(condition) {
            if crate::fold::extend(known, self.ty(condition), false) == 0 {
                let target = self.trap_target();
                self.asm.b(target);
            }
            return;
        }
        let target = self.trap_target();
        self.branch_cond(condition, true, target);
    }

    fn assign(&mut self, param: Value, value: Value) {
        match self.plan.location(param) {
            Location::Reg(reg) => self.raw(reg, value),
            Location::Float(v) => self.fget(Vreg(v), value),
            _ => self.move_to_slot(
                self.at(param),
                Some(self.frame.values[param.index()]),
                value,
            ),
        }
    }

    pub(super) fn edge(&mut self, block: Block, args: &[Value]) -> Result<(), String> {
        let int_scratch = [A, C];
        let float_scratch = [V0, V1];
        for step in edge_steps(
            self.func,
            &self.plan.locations,
            &self.frame.values,
            args,
            &self.func.blocks[block.index()].params,
        ) {
            match step {
                Step::Move(arg, param) => self.assign(param, arg),
                Step::Save(arg, EdgeScratch::Int(i)) => self.raw(int_scratch[i], arg),
                Step::Restore(EdgeScratch::Int(i), param) => self.put(param, int_scratch[i]),
                Step::Save(arg, EdgeScratch::Float(i)) => self.fget(float_scratch[i], arg),
                Step::Restore(EdgeScratch::Float(i), param) => self.fput(param, float_scratch[i]),
                Step::Save(arg, EdgeScratch::Stack(i)) => {
                    self.move_to(self.frame_mem(self.frame.edge_slot(i)), arg)
                }
                Step::Restore(EdgeScratch::Stack(i), param) => {
                    self.load_home(param, self.frame_mem(self.frame.edge_slot(i)))
                }
            }
        }
        Ok(())
    }

    fn emit_cbz(&mut self, val: Value, nonzero: bool, label: Label) {
        let bytes = self.ty(val).bytes();
        if bytes < 4 && self.tst_small_zero(val) {
            self.clear_flags();
            self.asm
                .bcond(if nonzero { Cond::Ne } else { Cond::Eq }, label);
            return;
        }
        let reg = if bytes >= 4 {
            self.bits(val, A)
        } else {
            self.zext_src(val, A)
        };
        self.asm.cbz(
            nonzero,
            if bytes >= 8 { Size::X } else { Size::W },
            reg,
            label,
        );
    }

    pub(super) fn branch_cond(&mut self, condition: Value, invert: bool, label: Label) {
        if let Some((value, cond)) = self.flags
            && value == condition
        {
            self.asm
                .bcond(if invert { cond.invert() } else { cond }, label);
            return;
        }
        if self.plan.elided[condition.index()]
            && let Some(inst) = self.defs[condition.index()]
        {
            if let Op::Icmp(cc, lhs, rhs) = self.func.insts[inst.index()].op
                && let Some((val, bit, set)) =
                    fuse::sign_bit_test(self.func, &self.plan, (cc, lhs, rhs), &|v: Value| {
                        self.defs[v.index()]
                    })
            {
                let reg = self.bits(val, A);
                self.asm.tbz(set ^ invert, reg, bit, label);
                return;
            }
            let band_branch = match self.func.insts[inst.index()].op {
                Op::Binary(Binary::Band, x, y) => Some((x, y, !invert)),
                Op::Convert(Convert::Ireduce, inner) if self.plan.elided[inner.index()] => {
                    match self.defs[inner.index()].map(|i| &self.func.insts[i.index()].op) {
                        Some(&Op::Binary(Binary::Band, x, y)) => Some((x, y, !invert)),
                        _ => None,
                    }
                }
                Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), lhs, rhs) => self
                    .elided_band_cmp(lhs, rhs)
                    .map(|(x, y)| (x, y, (cc == IntCC::Equal) ^ !invert)),
                _ => None,
            };
            if let Some((x, y, nonzero)) = band_branch {
                if let Some((val, bit)) = fuse::single_bit_mask(self.func, &self.plan, x, y) {
                    let reg = self.bits(val, A);
                    self.asm.tbz(nonzero, reg, bit, label);
                } else {
                    self.emit_tst(x, y);
                    self.clear_flags();
                    self.asm
                        .bcond(if nonzero { Cond::Ne } else { Cond::Eq }, label);
                }
                return;
            }
            if let Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), lhs, rhs) =
                self.func.insts[inst.index()].op
                && let Some((val, one)) = fuse::zero_cmp_operand(&self.plan, self.ty(lhs), lhs, rhs)
            {
                return self.emit_cbz(val, (cc == IntCC::NotEqual) ^ one ^ invert, label);
            }
        }
        self.emit_cbz(condition, !invert, label);
    }

    pub(super) fn br_table(&mut self, index: Value, default: Block, targets: &[Block]) {
        if targets.is_empty() {
            emit_goto(self, default, true);
            return;
        }
        let idx = self.src(index, A, Ext::Zero);
        if let Ok(len) = u32::try_from(targets.len())
            && len < 4096
        {
            self.asm.add_imm(true, true, Size::X, ZR, idx, len);
        } else {
            self.asm.mov_imm(C, targets.len() as i64);
            self.asm.alu(Alu::Subs, Size::X, ZR, idx, C);
        }
        self.asm.bcond(
            Cond::Hs,
            self.blocks[self.func.jump_target(default).index()],
        );
        let table = self.asm.label();
        self.asm.adr(C, table);
        self.asm.ldrsw_scaled(IP, C, idx);
        self.asm.alu(Alu::Add, Size::X, C, C, IP);
        self.asm.br(C);
        self.asm.bind(table);
        for &target in targets {
            self.asm
                .table_entry(table, self.blocks[self.func.jump_target(target).index()]);
        }
    }

    pub(super) fn ret(&mut self, values: &[Value]) -> Result<(), String> {
        let returns = self.abi.returns.clone();
        self.move_locs(values, &returns, None);
        if self.in_fast {
            self.asm.ret();
        } else {
            self.epilogue();
        }
        Ok(())
    }

    fn emit_float_arg_step(&mut self, step: ArgStep<u8>, load: bool) {
        match step {
            ArgStep::Move(src, dst) => {
                self.asm.vmov(Vreg(dst), Vreg(src));
                self.clobber_vreg(Vreg(dst));
            }
            ArgStep::Save(src) => {
                self.asm.add_imm(true, false, Size::X, SP, SP, 16);
                self.sp_bias += 16;
                self.asm.vstore(16, Mem::new(SP, 0), Vreg(src));
            }
            ArgStep::Restore(dst) => {
                self.asm.vload(16, Vreg(dst), Mem::new(SP, 0));
                self.asm.add_imm(false, false, Size::X, SP, SP, 16);
                self.sp_bias -= 16;
                self.clobber_vreg(Vreg(dst));
            }
            ArgStep::Load(val, dst) if load => self.fget(Vreg(dst), val),
            ArgStep::Load(..) => {}
        }
    }

    pub(super) fn move_locs(
        &mut self,
        values: &[Value],
        locs: &[Loc],
        extra: Option<(Value, Reg)>,
    ) {
        let float_targets: Vec<(Value, u8)> = values
            .iter()
            .zip(locs)
            .filter_map(|(&val, &loc)| {
                if let Loc::Float(v) = loc {
                    Some((val, v))
                } else {
                    None
                }
            })
            .collect();
        for step in arg_steps(&float_targets, |v| self.plan.float_of(v)) {
            self.emit_float_arg_step(step, true);
        }
        let mut int_targets: Vec<(Value, Reg)> = values
            .iter()
            .zip(locs)
            .filter_map(|(&val, &loc)| {
                if let Loc::Int(reg) = loc {
                    Some((val, reg))
                } else {
                    None
                }
            })
            .collect();
        int_targets.extend(extra);
        for step in arg_steps(&int_targets, |v| self.plan.reg_of(v)) {
            match step {
                ArgStep::Move(src, dst) => {
                    self.asm.mov(dst, src);
                    self.clobber_reg(dst);
                }
                ArgStep::Save(src) => {
                    self.asm.mov(ARG_SCRATCH, src);
                    self.clobber_reg(ARG_SCRATCH);
                }
                ArgStep::Restore(dst) => {
                    self.asm.mov(dst, ARG_SCRATCH);
                    self.clobber_reg(dst);
                }
                ArgStep::Load(val, dst) => self.raw(dst, val),
            }
        }
        for (&val, &loc) in values.iter().zip(locs) {
            if let Loc::Pair(low, high) = loc {
                self.raw(low, val);
                self.high(high, val);
            }
        }
    }

    pub(super) fn store_locs(&mut self, values: &[Value], locs: &[Loc]) {
        let mut float_pairs = Vec::new();
        let mut int_pairs = Vec::new();
        let mut narrow = Vec::new();
        for (&value, &loc) in values.iter().zip(locs) {
            let bytes = self.ty(value).bytes();
            if let Loc::Int(src) = loc
                && bytes < 8
                && self.plan.zero_ext[value.index()]
            {
                match self.plan.location(value) {
                    Location::Reg(dst) if dst != src => narrow.push((dst, bytes)),
                    _ => self.asm.extend(false, bytes, src, src),
                }
            }
            match (loc, self.plan.location(value)) {
                (Loc::Float(src), Location::Float(dst)) => float_pairs.push((src, dst)),
                (Loc::Int(src), Location::Reg(dst)) => int_pairs.push((src, dst)),
                _ => self.store_loc(value, loc),
            }
        }
        for step in reg_steps(float_pairs) {
            self.emit_float_arg_step(step, false);
        }
        let width = |dst: Reg| {
            narrow
                .iter()
                .find(|&&(reg, _)| reg == dst)
                .map_or(8, |&(_, bytes)| bytes)
        };
        for step in reg_steps(int_pairs) {
            match step {
                ArgStep::Move(src, dst) => {
                    self.asm.extend(false, width(dst), dst, src);
                    self.clobber_reg(dst);
                }
                ArgStep::Save(src) => {
                    self.asm.mov(ARG_SCRATCH, src);
                    self.clobber_reg(ARG_SCRATCH);
                }
                ArgStep::Restore(dst) => {
                    self.asm.extend(false, width(dst), dst, ARG_SCRATCH);
                    self.clobber_reg(dst);
                }
                ArgStep::Load(..) => {}
            }
        }
    }

    pub(super) fn store_loc(&mut self, value: Value, loc: Loc) {
        match loc {
            Loc::Int(reg) => self.put(value, reg),
            Loc::Pair(low, high) => {
                self.put(value, low);
                self.put_high(value, high);
            }
            Loc::Float(index) => self.fput(value, Vreg(index)),
            Loc::Stack(_) | Loc::Struct(..) => {}
        }
    }

    pub(super) fn stack_argument(&mut self, offset: u32, purpose: ArgumentPurpose, value: Value) {
        let to = Mem::new(SP, offset as i32);
        let bytes = self.ty(value).bytes();
        if bytes < 8 && !self.ty(value).in_float_register() {
            let reg = self.src(value, A, Ext::Zero);
            let width = if self.packing == Packing::Natural && purpose != ArgumentPurpose::Variadic
            {
                Size::of(bytes)
            } else {
                Size::X
            };
            self.asm.store(width, to, reg);
            return;
        }
        self.move_to(to, value);
    }

    pub(super) fn clobber_call_spills(&mut self) {
        for r in 0..19 {
            self.spills.clobber_reg(Reg(r));
        }
        self.spills.clobber_reg(LR);
        self.spills.clear_fspill();
        self.asm.clear_anchor();
    }
}

impl<'a> Backend<'a, Reg, Label> for Emit<'a> {
    fn plan(&self) -> &Plan<Reg> {
        &self.plan
    }
    fn plan_mut(&mut self) -> &mut Plan<Reg> {
        &mut self.plan
    }
    fn frame(&self) -> &Frame<Reg> {
        &self.frame
    }
    fn abi(&self) -> &Abi {
        &self.abi
    }
    fn shrink(&self) -> Option<&ShrinkWrap> {
        self.shrink.as_ref()
    }
    fn in_fast(&self) -> bool {
        self.in_fast
    }
    fn ret_fast(&mut self, values: &[Value]) -> Result<(), String> {
        self.in_fast = true;
        self.ret(values)
    }
    fn next_block(&self) -> Option<Block> {
        self.next
    }
    fn block_label(&self, block: Block) -> Label {
        self.blocks[block.index()]
    }
    fn new_label(&mut self) -> Label {
        self.asm.label()
    }
    fn tails(&mut self) -> &mut Tails<'a, Label> {
        &mut self.tails
    }
    fn island(&mut self) -> Result<(), String> {
        self.asm.maybe_island()
    }

    fn bind_label(&mut self, label: Label) {
        self.invalidate_spills();
        self.asm.bind(label);
    }
    fn jump_label(&mut self, label: Label) {
        self.invalidate_spills();
        self.asm.b(label);
    }
    fn branch_cond(&mut self, condition: Value, invert: bool, label: Label) {
        self.branch_cond(condition, invert, label);
    }
    fn edge(&mut self, block: Block, args: &[Value]) -> Result<(), String> {
        self.edge(block, args)
    }
    fn arguments(
        &mut self,
        sig: SigRef,
        callee: Value,
        args: &[Value],
        is_try: bool,
    ) -> Result<Abi, String> {
        self.arguments(sig, callee, args, is_try)
    }
    fn store_locs(&mut self, values: &[Value], locs: &[Loc]) {
        self.store_locs(values, locs);
    }
    fn store_loc(&mut self, value: Value, loc: Loc) {
        self.store_loc(value, loc);
    }
    fn load_incoming(&mut self, value: Value, offset: u32) {
        self.load_home(value, Mem::new(FP, INCOMING + offset as i32));
    }
    fn intake_struct(&mut self, _value: Value, _offset: u32) {}
    fn park_spill(&mut self, value: Value, reg: ParkReg<Reg>) {
        let home = self.at(value);
        let disp = self.frame.values[value.index()];
        match reg {
            ParkReg::Int(r) => {
                self.asm.store(Size::X, home, r);
                self.write_spill(disp, r);
            }
            ParkReg::Float(v) => {
                self.asm.vstore(self.ty(value).bytes(), home, Vreg(v));
                self.spills.write_fspill(disp, Vreg(v), self.ty(value));
            }
        }
    }
    fn park_reload(&mut self, value: Value, reg: ParkReg<Reg>) {
        let home = self.at(value);
        let disp = self.frame.values[value.index()];
        match reg {
            ParkReg::Int(r) => {
                self.asm.load(Size::X, false, r, home);
                self.record_spill(disp, r);
            }
            ParkReg::Float(v) => {
                self.asm.vload(self.ty(value).bytes(), Vreg(v), home);
                self.clobber_vreg(Vreg(v));
                self.spills.fspill = Some((disp, Vreg(v), self.ty(value)));
            }
        }
    }
    fn put_exception(&mut self, value: Value) {
        self.put(value, Reg(0));
    }
    fn push_site(&mut self, site: Site) {
        self.sites.push(site);
    }
    fn spill_cache(&self) -> crate::spills::SpillCache<Reg> {
        self.spills.cache
    }
    fn set_spill_cache(&mut self, cache: crate::spills::SpillCache<Reg>) {
        self.spills.cache = cache;
    }
    fn record_edge_spill(&mut self, target: Block, cache: crate::spills::SpillCache<Reg>) {
        if !self.in_fast {
            self.spills
                .record_edge(self.func.jump_target(target), cache);
        }
    }
}
