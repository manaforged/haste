use crate::fold::extend;
use crate::ir::{Function, Inst, Op, Type, Value};

use super::Info;
use super::bitmask::known_upper_bound;
use super::lut::{ColExpr, build_col_lut, emit_col_expr, uval_of};
use super::tree::{intersect_intervals, match_cond_intervals};

struct SelectWalk<'a> {
    func: &'a Function,
    info: &'a Info,
    b0: usize,
    out_ty: Type,
    val: Option<Value>,
    select_nodes: u32,
    test_ops: u32,
    leaves: Vec<(u64, u64, i64)>,
    dead: Vec<Inst>,
}

impl SelectWalk<'_> {
    fn visit(&mut self, node: Value, domain: Vec<(u64, u64)>, depth: u8, is_root: bool) -> bool {
        if domain.is_empty()
            || domain.len() > 64
            || depth > 24
            || self.select_nodes > 64
            || self.leaves.len() > 256
        {
            return false;
        }
        if let Some(imm) = self.info.consts.get(node.index()).copied().flatten() {
            let val = extend(imm, self.out_ty, true);
            for (lo, hi) in domain {
                self.leaves.push((lo, hi, val));
            }
            return true;
        }
        if self.info.def_block.get(node.index()).copied().flatten() != Some(self.b0) {
            return false;
        }
        if !is_root && self.info.uses.get(node.index()).copied() != Some(1) {
            return false;
        }
        let Some(inst) = self.info.def_inst.get(node.index()).copied().flatten() else {
            return false;
        };
        let Op::Select(cond, t_val, e_val) = self.func.insts[inst.index()].op else {
            return false;
        };
        let Some(ci) =
            match_cond_intervals(self.func, self.info, self.b0, cond, self.val, false, false)
        else {
            return false;
        };
        self.val = Some(ci.val);
        self.select_nodes += 1;
        self.test_ops += u32::try_from(ci.used.len()).unwrap_or(1).max(1);
        if !is_root && !self.dead.contains(&inst) {
            self.dead.push(inst);
        }
        for u_inst in ci.used {
            let Some(res) = self.func.insts[u_inst.index()].result else {
                break;
            };
            if self.info.def_block.get(res.index()).copied().flatten() != Some(self.b0)
                || self.info.uses.get(res.index()).copied() != Some(1)
            {
                break;
            }
            if !self.dead.contains(&u_inst) {
                self.dead.push(u_inst);
            }
        }
        let (t_dom, e_dom) = (
            intersect_intervals(&domain, &ci.true_ivs),
            intersect_intervals(&domain, &ci.false_ivs),
        );
        self.visit(t_val, t_dom, depth + 1, false) && self.visit(e_val, e_dom, depth + 1, false)
    }
}

pub(super) fn try_rewrite_select_trees(func: &mut Function, info: &Info, b0: usize) -> bool {
    let mut changed = false;
    let mut pos = func.blocks[b0].insts.len();
    while pos > 0 {
        pos -= 1;
        let root_inst = func.blocks[b0].insts[pos];
        let data = &func.insts[root_inst.index()];
        let (&Op::Select(cond, ..), Some(root_val)) = (&data.op, data.result) else {
            continue;
        };
        let out_ty = func.ty(root_val);
        if !out_ty.is_small_int() {
            continue;
        }
        let Some(ci0) = match_cond_intervals(func, info, b0, cond, None, false, false) else {
            continue;
        };
        let ty = func.ty(ci0.val);
        let mut walk = SelectWalk {
            func,
            info,
            b0,
            out_ty,
            val: Some(ci0.val),
            select_nodes: 0,
            test_ops: 0,
            leaves: Vec::new(),
            dead: Vec::new(),
        };
        if !walk.visit(root_val, vec![(0, ty.unsigned_max())], 0, true) || walk.leaves.is_empty() {
            continue;
        }
        let Some(v) = walk.val else { continue };
        let mut groups: Vec<(i64, u128, usize)> = Vec::new();
        for &(lo, hi, val) in &walk.leaves {
            let count = u128::from(hi - lo) + 1;
            let uv = uval_of(val, out_ty);
            if let Some((_, tot, cnt)) = groups
                .iter_mut()
                .find(|(gv, _, _)| uval_of(*gv, out_ty) == uv)
            {
                *tot += count;
                *cnt += 1;
            } else {
                groups.push((val, count, 1));
            }
        }
        let Some((default_c, _, _)) = groups.into_iter().max_by_key(|&(_, tot, cnt)| (tot, cnt))
        else {
            continue;
        };
        let def_u = uval_of(default_c, out_ty);
        let mut non_def: Vec<(u64, u64, i64)> = walk
            .leaves
            .into_iter()
            .filter(|&(_, _, val)| uval_of(val, out_ty) != def_u)
            .collect();
        non_def.sort_unstable_by_key(|&(lo, _, _)| lo);
        let mut merged: Vec<(u64, u64, i64)> = Vec::with_capacity(non_def.len());
        for (lo, hi, val) in non_def {
            if let Some(prev) = merged.last_mut()
                && prev.1 < u64::MAX
                && prev.1 + 1 == lo
                && uval_of(prev.2, out_ty) == uval_of(val, out_ty)
            {
                prev.1 = hi;
            } else {
                merged.push((lo, hi, val));
            }
        }
        let known_u_max = known_upper_bound(func, info, v);
        let expr = if merged.is_empty() {
            ColExpr::Same(default_c)
        } else if let Some(expr) = build_col_lut(ty, out_ty, &merged, Some(default_c), known_u_max)
        {
            let words = expr.word_count();
            let profitable = match words {
                0 => true,
                1 => walk.select_nodes >= 3 || (walk.select_nodes >= 2 && walk.test_ops >= 4),
                2 => walk.select_nodes >= 4 || walk.test_ops >= 5,
                _ => false,
            };
            if !profitable {
                continue;
            }
            expr
        } else {
            continue;
        };
        let dead = walk.dead;
        let mut insts = Vec::with_capacity(16);
        let _ = emit_col_expr(func, (v, ty), out_ty, expr, known_u_max, &mut insts);
        let Some(final_inst) = insts.pop() else {
            continue;
        };
        let final_op = func.insts[final_inst.index()].op.clone();
        func.insts[final_inst.index()].op = Op::Trap;
        func.insts[root_inst.index()].op = final_op;
        func.blocks[b0].insts.retain(|i| !dead.contains(i));
        for d in dead {
            func.insts[d.index()].op = Op::Trap;
        }
        if let Some(root_pos) = func.blocks[b0].insts.iter().position(|&i| i == root_inst) {
            func.blocks[b0].insts.splice(root_pos..root_pos, insts);
            pos = root_pos;
        }
        changed = true;
    }
    changed
}
