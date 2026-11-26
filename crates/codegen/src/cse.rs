mod key;
mod norm;
mod prove;

use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::fold::extend;
use crate::ir::{
    Binary, Block, Convert, Domain, Function, IntCC, Op, StackSlot, Type, Unary, Value,
};
use crate::merge::{find, identity, rename};
use key::{Key, Operand, canonical_icmp, operand, unwrap_bool_neg};
use prove::narrow_widened_subs;
pub(crate) use prove::{CmpEnv, EdgeVal, cond_proves_want, dom_proves_want};

fn block_tag(block: Block) -> u32 {
    block.0 + 1
}

#[derive(Default)]
struct Table {
    known: HashMap<Key, Value>,
    added: Vec<(Key, Option<Value>)>,
    consts: Vec<Option<i64>>,
    bool_neg: Vec<Option<Value>>,
    icmp_def: Vec<Option<(IntCC, Type, Value, Value)>>,
    def_block: Vec<u32>,
    local_flag_only: Vec<bool>,
    bool_val: Vec<bool>,
    affine: Vec<(Value, i64)>,
    stack_addrs: Vec<Option<(StackSlot, i32)>>,
    assoc: Vec<Option<(Binary, Value, i64)>>,
    unary_src: Vec<Option<(Unary, Value)>>,
    conv_src: Vec<Option<(Convert, Value, Type)>>,
    select_def: Vec<Option<(Value, Value, Value)>>,
    drop_buf: Vec<Key>,
    const_count: usize,
}

impl Table {
    fn alt_select(&self, op: &Op, types: &[Type], subst: &[Value]) -> Option<Value> {
        let Op::Select(c, x, y) = *op else {
            return None;
        };
        let (cond, sx, sy) = unwrap_bool_neg(&self.bool_neg, subst, c, x, y);
        let (cc, x_ty, ix, iy) = self.icmp_def[cond.index()]?;
        let (inv_cc, ia, ib) = canonical_icmp(
            cc.inverse(),
            operand(&self.consts, types, subst, ix),
            operand(&self.consts, types, subst, iy),
        );
        let &inv_c = self.known.get(&Key::Icmp(inv_cc, x_ty, ia, ib))?;
        let key = Key::Select(
            Operand::Val(inv_c),
            operand(&self.consts, types, subst, sy),
            operand(&self.consts, types, subst, sx),
        );
        self.known.get(&key).copied()
    }

    fn simplify_select(
        &mut self,
        c: Value,
        x: Value,
        y: Value,
        types: &[Type],
        result: Value,
        subst: &[Value],
    ) -> (Value, Value, Value, bool) {
        let (cond, raw_x, raw_y) = unwrap_bool_neg(&self.bool_neg, subst, c, x, y);
        let (mut sx, mut sy) = (find(subst, raw_x), find(subst, raw_y));
        let mut flat = false;
        if !types[cond.index()].is_vector() {
            if let Some((c2, a, _)) = self.select_def[sx.index()]
                && find(subst, c2) == cond
            {
                sx = find(subst, a);
                flat = true;
            }
            if let Some((c2, _, b)) = self.select_def[sy.index()]
                && find(subst, c2) == cond
            {
                sy = find(subst, b);
                flat = true;
            }
            self.select_def[result.index()] = Some((cond, sx, sy));
        }
        (cond, sx, sy, flat)
    }

    fn scan_call(
        &mut self,
        func: &Function,
        op: &Op,
        subst: &mut [Value],
        mutated: &mut bool,
    ) -> bool {
        if let Op::Call {
            callee,
            ref args,
            ref results,
            ..
        } = *op
            && (func.is_thread_cell(callee) || func.is_thread_cell(find(subst, callee)))
            && let ([slot, size, init], [result]) = (args.as_slice(), results.as_slice())
        {
            let opnd = |v: Value| operand(&self.consts, &func.types, subst, v);
            let tc_key = Key::ThreadCell(opnd(*slot), opnd(*size), opnd(*init));
            if let Some(&earlier) = self.known.get(&tc_key) {
                subst[result.index()] = earlier;
                *mutated = true;
                return false;
            }
            let prev = self.known.insert(tc_key, *result);
            self.added.push((tc_key, prev));
        }
        if self.const_count > 0 {
            self.drop_buf.clear();
            self.drop_buf.extend(
                self.known
                    .keys()
                    .copied()
                    .filter(|k| matches!(k, Key::Const(..))),
            );
            for k in self.drop_buf.drain(..) {
                let prev = self.known.remove(&k);
                self.const_count -= usize::from(prev.is_some());
                self.added.push((k, prev));
            }
        }
        true
    }

    fn scan(&mut self, func: &mut Function, block: Block, subst: &mut [Value]) -> bool {
        let mut insts = std::mem::take(&mut func.blocks[block.index()].insts);
        let mut mutated = false;
        insts.retain(|&inst| {
            let data = &func.insts[inst.index()];
            if matches!(data.op, Op::Call { .. } | Op::TryCall { .. }) {
                return self.scan_call(func, &data.op, subst, &mut mutated);
            }
            let Some(result) = data.result else {
                return true;
            };
            let Some(norm) =
                self.normalized_key(&data.op, &func.types, func.ty(result), result, subst)
            else {
                return true;
            };
            let key = match norm {
                Ok(key) => key,
                Err(alias) => {
                    subst[result.index()] = alias;
                    mutated = true;
                    return false;
                }
            };
            if let Some(&earlier) = self.known.get(&key) {
                if self.local_flag_only[result.index()]
                    && self.def_block[earlier.index()] != block_tag(block)
                {
                    return true;
                }
                subst[result.index()] = earlier;
                mutated = true;
                false
            } else if let Some(earlier) =
                self.alt_select(&func.insts[inst.index()].op, &func.types, subst)
            {
                subst[result.index()] = earlier;
                mutated = true;
                false
            } else {
                if let Op::Select(c, x, y) = func.insts[inst.index()].op {
                    let (cond, sx, sy, flat) =
                        self.simplify_select(c, x, y, &func.types, result, subst);
                    if flat {
                        func.insts[inst.index()].op = Op::Select(cond, sx, sy);
                        mutated = true;
                    }
                }
                let prev = self.known.insert(key, result);
                self.const_count += usize::from(prev.is_none() && matches!(key, Key::Const(..)));
                self.added.push((key, prev));
                true
            }
        });
        func.blocks[block.index()].insts = insts;
        mutated
    }

    fn forget(&mut self, mark: usize) {
        for (key, prev) in self.added.drain(mark..).rev() {
            let is_const = matches!(key, Key::Const(..));
            match prev {
                Some(v) => {
                    self.const_count +=
                        usize::from(self.known.insert(key, v).is_none() && is_const);
                }
                None => {
                    self.const_count -= usize::from(self.known.remove(&key).is_some() && is_const);
                }
            }
        }
    }
}

enum Visit {
    Enter(Block),
    Leave(usize),
}

struct Facts {
    consts: Vec<Option<i64>>,
    bool_val: Vec<bool>,
    bool_neg: Vec<Option<Value>>,
    icmp_def: Vec<Option<(IntCC, Type, Value, Value)>>,
    def_block: Vec<u32>,
    local_flag_only: Vec<bool>,
    has_narrow_uext: bool,
}

fn record_def_fact(func: &Function, op: &Op, res: Value, facts: &mut Facts) {
    let ri = res.index();
    match *op {
        Op::Const(imm) if func.ty(res).is_small_int() => {
            let z = extend(imm, func.ty(res), false);
            facts.consts[ri] = Some(z);
            facts.bool_val[ri] = z == 0 || z == 1;
        }
        Op::Icmp(cc, x, y) => {
            facts.bool_val[ri] = true;
            facts.icmp_def[ri] = Some((cc, func.ty(x), x, y));
            if cc == IntCC::Equal {
                if facts.consts[x.index()] == Some(0) && facts.bool_val[y.index()] {
                    facts.bool_neg[ri] = Some(y);
                } else if facts.consts[y.index()] == Some(0) && facts.bool_val[x.index()] {
                    facts.bool_neg[ri] = Some(x);
                }
            }
        }
        Op::Fcmp(..) | Op::Fits(..) => facts.bool_val[ri] = true,
        Op::Convert(Convert::Uextend, x) => {
            if facts.bool_val[x.index()] {
                facts.bool_val[ri] = true;
            }
            if matches!(func.ty(x), Type::I8 | Type::I16) {
                facts.has_narrow_uext = true;
            }
        }
        Op::Binary(Binary::Band, x, y)
            if facts.bool_val[x.index()] || facts.bool_val[y.index()] =>
        {
            facts.bool_val[ri] = true;
        }
        Op::Binary(Binary::Bor, x, y) if facts.bool_val[x.index()] && facts.bool_val[y.index()] => {
            facts.bool_val[ri] = true;
        }
        Op::Binary(Binary::Bxor, x, y) => {
            facts.bool_val[ri] = facts.bool_val[x.index()] && facts.bool_val[y.index()];
            if facts.consts[y.index()] == Some(1) && facts.bool_val[x.index()] {
                facts.bool_neg[ri] = Some(x);
            } else if facts.consts[x.index()] == Some(1) && facts.bool_val[y.index()] {
                facts.bool_neg[ri] = Some(y);
            }
        }
        Op::Binary(Binary::Isub, x, y)
            if facts.consts[x.index()] == Some(1) && facts.bool_val[y.index()] =>
        {
            facts.bool_val[ri] = true;
            facts.bool_neg[ri] = Some(y);
        }
        Op::Select(_, x, y) if facts.bool_val[x.index()] && facts.bool_val[y.index()] => {
            facts.bool_val[ri] = true;
        }
        _ => {}
    }
}

fn collect_facts(func: &Function, cfg: &Cfg) -> Facts {
    let n = func.types.len();
    let mut facts = Facts {
        consts: vec![None; n],
        bool_val: vec![false; n],
        bool_neg: vec![None; n],
        icmp_def: vec![None; n],
        def_block: vec![0_u32; n],
        local_flag_only: vec![false; n],
        has_narrow_uext: false,
    };
    let mut uses = vec![0_u32; n];
    let mut flag_cands: Vec<(Value, u32)> = Vec::new();
    for &b in &cfg.rpo {
        let blk = &func.blocks[b.index()];
        let btag = block_tag(b);
        for &inst in &blk.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| uses[v.index()] += 1);
            if let Some(res) = data.result {
                facts.def_block[res.index()] = btag;
                record_def_fact(func, &data.op, res, &mut facts);
            }
            if let Op::Brif(c, ..) | Op::Trapz(c) | Op::Select(c, ..) = data.op
                && facts.def_block[c.index()] == btag
                && facts.icmp_def[c.index()].is_some()
            {
                flag_cands.push((c, btag));
            }
        }
    }
    for (c, _) in flag_cands {
        if uses[c.index()] == 1 {
            facts.local_flag_only[c.index()] = true;
        }
    }
    facts
}

/// Replaces every pure instruction that a dominating instruction already computed with that instruction's result, returning the invalidated IR domain.
pub(crate) fn run(func: &mut Function) -> Domain {
    let Some(entry) = func.entry else {
        return Domain::NONE;
    };
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
    let cfg = func.cfg();
    let children = cfg.children();
    let nvals = func.types.len();
    let mut subst = identity(nvals);
    let facts = collect_facts(func, &cfg);
    let affine = identity(nvals).into_iter().map(|v| (v, 0_i64)).collect();
    let mut table = Table {
        consts: facts.consts,
        bool_neg: facts.bool_neg,
        icmp_def: facts.icmp_def,
        def_block: facts.def_block,
        local_flag_only: facts.local_flag_only,
        bool_val: facts.bool_val,
        affine,
        stack_addrs: vec![None; nvals],
        assoc: vec![None; nvals],
        unary_src: vec![None; nvals],
        conv_src: vec![None; nvals],
        select_def: vec![None; nvals],
        ..Table::default()
    };
    let (mut dropped, mut stack) = (false, vec![Visit::Enter(entry)]);
    while let Some(visit) = stack.pop() {
        match visit {
            Visit::Leave(mark) => table.forget(mark),
            Visit::Enter(block) => {
                stack.push(Visit::Leave(table.added.len()));
                dropped |= table.scan(func, block, &mut subst);
                stack.extend(
                    children[block.index()]
                        .iter()
                        .map(|child| Visit::Enter(*child)),
                );
            }
        }
    }
    if dropped {
        rename(func, &subst);
    }
    let changed =
        facts.has_narrow_uext && narrow_widened_subs(func, &cfg, &table.consts, &table.bool_val);
    if dropped || changed {
        Domain::VALUES
    } else {
        Domain::NONE
    }
}
