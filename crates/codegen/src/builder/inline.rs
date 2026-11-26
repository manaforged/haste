use super::FunctionBuilder;
use crate::ir::{Block, BlockData, Function, Inst, InstData, Op, Value};

fn import_callee_meta(
    dst: &mut Function,
    callee: &Function,
    slot_base: u32,
    sym_base: u32,
    copy_vtables: bool,
) {
    dst.slots.extend(callee.slots.iter().cloned());
    dst.sigs.extend(callee.sigs.iter().cloned());
    if copy_vtables {
        for &(vt_sym, off, slot_sym, low) in &callee.vtable_slots {
            dst.vtable_slots
                .push((vt_sym + sym_base, off, slot_sym + sym_base, low));
        }
    }
    for (sym, bytes) in &callee.rodata_bytes {
        dst.rodata_bytes.push((sym + sym_base, bytes.clone()));
    }
    dst.import_unwind_drops(callee, slot_base, sym_base);
}

impl FunctionBuilder<'_> {
    /// Copies `callee` into the function as if the current block called it with `args`, then continues in a new block whose parameters are its results.
    pub fn inline(&mut self, callee: &Function, args: &[Value]) -> Vec<Value> {
        self.inline_remap(callee, args, 0, None)
    }

    /// Copies `callee` into the function while shifting its relocation indices by `sym_base`.
    pub fn inline_shifted(
        &mut self,
        callee: &Function,
        args: &[Value],
        sym_base: u32,
    ) -> Vec<Value> {
        self.inline_remap(callee, args, sym_base, None)
    }

    /// Copies `callee` into a dictionary-sharing function, loading its symbols from `dict` at `sym_base`.
    pub fn inline_dict(
        &mut self,
        callee: &Function,
        args: &[Value],
        dict: Value,
        sym_base: u32,
    ) -> Vec<Value> {
        self.inline_remap(callee, args, sym_base, Some(dict))
    }

    fn inline_remap(
        &mut self,
        callee: &Function,
        args: &[Value],
        sym_base: u32,
        dict: Option<Value>,
    ) -> Vec<Value> {
        let values: Vec<Value> = self.func.import_values(callee);
        let blocks: Vec<Block> = callee.blocks.iter().map(|_| self.create_block()).collect();
        let after = self.create_block();
        let results: Vec<Value> = callee
            .signature
            .returns
            .iter()
            .map(|ret| self.append_block_param(after, ret.value_type))
            .collect();
        let map = Remap {
            values,
            blocks,
            after,
            slots: self.func.slots.len() as u32,
            sigs: self.func.sigs.len() as u32,
            sym_base,
            dict,
        };
        import_callee_meta(self.func, callee, map.slots, sym_base, dict.is_none());
        for (index, block) in callee.blocks.iter().enumerate() {
            let target = map.blocks[index];
            self.func.blocks[target.index()].params = block
                .params
                .iter()
                .map(|param| map.values[param.index()])
                .collect();
            for inst in &block.insts {
                let data = map.inst(&callee.insts[inst.index()]);
                self.append(target, data);
            }
        }
        let entry = callee
            .entry
            .map_or(after, |entry| map.blocks[entry.index()]);
        self.ins().jump(entry, args);
        self.switch_to_block(after);
        results
    }
}

impl Function {
    /// Replaces the `Op::Call` at `block[inst_pos]` with an inline copy of `callee`, shifting symbols by `sym_base` and splitting the tail into a continuation block.
    pub fn splice_call(&mut self, block: Block, inst_pos: usize, callee: &Function, sym_base: u32) {
        let old_inst = self.blocks[block.index()].insts[inst_pos];
        let Op::Call { args, results, .. } =
            std::mem::replace(&mut self.insts[old_inst.index()].op, Op::Trap)
        else {
            return;
        };
        self.bump_cfg();
        self.insts[old_inst.index()].result = None;
        let values: Vec<Value> = self.import_values(callee);
        let base_block = self.blocks.len() as u32;
        let blocks: Vec<Block> = (0..callee.blocks.len() as u32)
            .map(|i| Block(base_block + i))
            .collect();
        self.blocks
            .resize_with(self.blocks.len() + callee.blocks.len(), BlockData::default);
        let after = Block(self.blocks.len() as u32);
        let tail = self.blocks[block.index()].insts.split_off(inst_pos + 1);
        self.blocks[block.index()].insts.pop();
        self.blocks.push(BlockData {
            params: results,
            insts: tail,
        });
        let map = Remap {
            values,
            blocks,
            after,
            slots: self.slots.len() as u32,
            sigs: self.sigs.len() as u32,
            sym_base,
            dict: None,
        };
        import_callee_meta(self, callee, map.slots, sym_base, true);
        for (index, c_block) in callee.blocks.iter().enumerate() {
            let target = map.blocks[index];
            self.blocks[target.index()].params = c_block
                .params
                .iter()
                .map(|param| map.values[param.index()])
                .collect();
            for inst in &c_block.insts {
                let data = map.inst(&callee.insts[inst.index()]);
                self.insts.push(data);
                let new_inst = Inst(self.insts.len() as u32 - 1);
                self.blocks[target.index()].insts.push(new_inst);
            }
        }
        let entry = callee
            .entry
            .map_or(after, |entry| map.blocks[entry.index()]);
        let jump = self.add(Op::Jump(entry, args), None);
        self.blocks[block.index()].insts.push(jump);
    }

    /// Rewrites non-self symbol operands and rodata metadata using `build(self_slots)`.
    pub fn remap_symbols(&mut self, build: impl FnOnce(&[u32]) -> Option<Vec<u32>>) {
        let Some(remap) = build(&self.self_slots) else {
            return;
        };
        for d in &mut self.insts {
            if let Op::Symbol(s) | Op::SlotAddr(s) = &mut d.op {
                *s = remap.get(*s as usize).copied().unwrap_or(*s);
            }
        }
        for (s, _) in &mut self.rodata_bytes {
            *s = remap.get(*s as usize).copied().unwrap_or(*s);
        }
        for u in &mut self.unwind_drops {
            u.drop_sym = remap
                .get(u.drop_sym as usize)
                .copied()
                .unwrap_or(u.drop_sym);
        }
    }
}

struct Remap {
    values: Vec<Value>,
    blocks: Vec<Block>,
    after: Block,
    slots: u32,
    sigs: u32,
    sym_base: u32,
    dict: Option<Value>,
}

impl Remap {
    fn inst(&self, data: &InstData) -> InstData {
        let mut op = data.op.clone();
        op.map_operands(|value| self.values[value.index()]);
        op.map_targets(|block| self.blocks[block.index()]);
        match &mut op {
            Op::SlotAddr(sym) | Op::Symbol(sym) => match self.dict {
                Some(dict) => op = Op::Load(dict, ((*sym + self.sym_base) * 8) as i32),
                None => *sym += self.sym_base,
            },
            Op::StackAddr(slot, _) | Op::StackLoad(slot, _) | Op::StackStore(_, slot, _) => {
                slot.0 += self.slots;
            }
            Op::Call { sig, results, .. } => {
                sig.0 += self.sigs;
                results
                    .iter_mut()
                    .for_each(|value| *value = self.values[value.index()]);
            }
            Op::TryCall { sig, .. } => sig.0 += self.sigs,
            Op::Return(values) => op = Op::Jump(self.after, std::mem::take(values)),
            _ => {}
        }
        InstData {
            op,
            result: data.result.map(|value| self.values[value.index()]),
            loc: data.loc,
        }
    }
}
