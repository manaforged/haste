use super::{Before, Fold, extend, for_edges, holds, table_target};
use crate::ir::{Binary, Block, Function, Inst, IntCC, Op, Type, Value};

impl Fold {
    pub(super) fn branches(&self, func: &mut Function) -> bool {
        let mut changed = false;
        let mut cfg_dirty = false;
        for block in 0..func.blocks.len() {
            let mut dropped_after_trap = false;
            if func.blocks[block]
                .insts
                .iter()
                .any(|&i| matches!(func.insts[i.index()].op, Op::Trap | Op::Trapz(_)))
            {
                let mut trapped = false;
                let mut keep = std::mem::take(&mut func.blocks[block].insts);
                keep.retain(|&inst| {
                    if trapped {
                        func.insts[inst.index()].op = Op::Trap;
                        changed = true;
                        dropped_after_trap = true;
                        return false;
                    }
                    match func.insts[inst.index()].op {
                        Op::Trapz(x) => {
                            if let Some(k) = self.known(x) {
                                trapped = extend(k, func.types[x.index()], false) == 0;
                                func.insts[inst.index()].op = Op::Trap;
                                changed = true;
                                return trapped;
                            }
                            if self.non_null(
                                func,
                                Before {
                                    prefix: &[],
                                    block: None,
                                },
                                x,
                            ) {
                                func.insts[inst.index()].op = Op::Trap;
                                changed = true;
                                return false;
                            }
                            true
                        }
                        Op::Trap => {
                            trapped = true;
                            true
                        }
                        _ => true,
                    }
                });
                func.blocks[block].insts = keep;
            }
            if dropped_after_trap {
                cfg_dirty = true;
            }
            if cfg_dirty
                && func.blocks[block]
                    .insts
                    .last()
                    .is_some_and(|&i| matches!(func.insts[i.index()].op, Op::Brif(..)))
            {
                func.bump_cfg();
                cfg_dirty = false;
            }
            let Some((&last, prefix)) = func.blocks[block].insts.split_last() else {
                continue;
            };
            let at = Before {
                prefix,
                block: Some(Block(block as u32)),
            };
            let through = match func.insts[last.index()].op {
                Op::Brif(cond, ..) if self.known(cond).is_none() => match self.def(func, cond) {
                    Some(&Op::Binary(Binary::Bxor, a, b)) => {
                        [(a, b), (b, a)].into_iter().find_map(|(inner, one)| {
                            (func.ty(inner) == Type::I8
                                && self.boolean(func, inner)
                                && self
                                    .known(one)
                                    .is_some_and(|imm| extend(imm, Type::I8, false) == 1))
                            .then_some((inner, true))
                        })
                    }
                    Some(&Op::Select(inner, x, y))
                        if func.ty(inner).is_small_int() && func.ty(cond).is_small_int() =>
                    {
                        let zero = |v: Value| {
                            self.known(v)
                                .is_some_and(|imm| extend(imm, func.ty(v), false) == 0)
                        };
                        let set = |v: Value| {
                            self.known(v).map_or_else(
                                || self.non_null(func, at, v),
                                |imm| extend(imm, func.ty(v), false) != 0,
                            )
                        };
                        if zero(y) && set(x) {
                            Some((inner, false))
                        } else if zero(x) && set(y) {
                            Some((inner, true))
                        } else {
                            None
                        }
                    }
                    _ => None,
                },
                _ => None,
            };
            let non_null_br = through.is_none()
                && matches!(func.insts[last.index()].op, Op::Brif(cond, ..) if self.non_null(func, at, cond));
            if let Some((inner, swap)) = through
                && let Op::Brif(cond, then, then_args, otherwise, else_args) =
                    &mut func.insts[last.index()].op
            {
                *cond = inner;
                if swap {
                    std::mem::swap(then, otherwise);
                    std::mem::swap(then_args, else_args);
                    cfg_dirty = true;
                }
                changed = true;
            }
            if self.decide(&func.types, non_null_br, &mut func.insts[last.index()].op)
                || self.exits(func, last)
            {
                changed = true;
                cfg_dirty = true;
            }
        }
        if cfg_dirty {
            func.bump_cfg();
        }
        changed
    }

    fn decide(&self, types: &[Type], non_null_br: bool, op: &mut Op) -> bool {
        match op {
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                let same = then == otherwise
                    && then_args.len() == else_args.len()
                    && then_args
                        .iter()
                        .zip(else_args.iter())
                        .all(|(a, b)| self.find(*a) == self.find(*b));
                let Some(taken) = self
                    .known(*condition)
                    .map(|k| k != 0)
                    .or((same || non_null_br).then_some(true))
                else {
                    return false;
                };
                let (target, args) = if taken {
                    (*then, std::mem::take(then_args))
                } else {
                    (*otherwise, std::mem::take(else_args))
                };
                *op = Op::Jump(target, args);
                true
            }
            Op::BrTable(index, default, targets) => {
                let target = if let Some(known) = self.known(*index) {
                    table_target(
                        extend(known, types[index.index()], false) as u64,
                        *default,
                        targets,
                    )
                } else if targets.iter().all(|target| target == default) {
                    *default
                } else {
                    return false;
                };
                *op = Op::Jump(target, Vec::new());
                true
            }
            _ => false,
        }
    }

    pub(super) fn thread(&self, func: &mut Function, pinned: &[bool]) -> bool {
        let (mut hop, mut need_uses) = (false, false);
        for block in &func.blocks {
            match block.insts[..] {
                [only] => match func.insts[only.index()].op {
                    Op::Jump(next, _) if !pinned[next.index()] => {
                        hop = true;
                        need_uses |= !block.params.is_empty();
                    }
                    Op::Brif(cond, ..) if block.params.contains(&cond) => {
                        (hop, need_uses) = (true, true);
                    }
                    _ => {}
                },
                [cmp, last] => {
                    let c = &func.insts[cmp.index()];
                    if let (Op::Icmp(_, x, y), Some(res), Op::Brif(cond, ..)) =
                        (&c.op, c.result, &func.insts[last.index()].op)
                        && res == *cond
                        && func.ty(*x).is_small_int()
                        && (!block.params.is_empty()
                            || (self.known(*x).is_some() && self.known(*y).is_some()))
                    {
                        (hop, need_uses) = (true, true);
                    }
                }
                _ => {}
            }
            if hop && need_uses {
                break;
            }
        }
        if !hop {
            return false;
        }
        let mut uses = if need_uses {
            func.use_counts()
        } else {
            Vec::new()
        };
        let mut changed = false;
        for block in 0..func.blocks.len() {
            let Some(&last) = func.blocks[block].insts.last() else {
                continue;
            };
            let mut op = std::mem::replace(&mut func.insts[last.index()].op, Op::Trap);
            match &mut op {
                Op::Jump(target, args) => {
                    changed |= self.retarget(func, (&mut uses, pinned), target, args);
                }
                Op::Brif(_, then, then_args, otherwise, else_args) => {
                    changed |= self.retarget(func, (&mut uses, pinned), then, then_args);
                    changed |= self.retarget(func, (&mut uses, pinned), otherwise, else_args);
                }
                _ => {}
            }
            func.insts[last.index()].op = op;
        }
        changed
    }

    fn retarget(
        &self,
        func: &Function,
        (uses, pinned): (&mut [u32], &[bool]),
        target: &mut Block,
        args: &mut Vec<Value>,
    ) -> bool {
        let Some((first, first_values)) = self.step(func, (uses, pinned), *target, args) else {
            return false;
        };
        if first == *target {
            return false;
        }
        let (mut at, mut values, mut seen) = (first, first_values, vec![*target, first]);
        while let Some((next, next_values)) = self.step(func, (uses, pinned), at, &values) {
            if seen.contains(&next) {
                return false;
            }
            seen.push(next);
            (at, values) = (next, next_values);
        }
        if !self.agrees(func, at, &values) {
            return false;
        }
        if !uses.is_empty() {
            args.iter().for_each(|value| uses[value.index()] -= 1);
            values.iter().for_each(|value| uses[value.index()] += 1);
        }
        (*target, *args) = (at, values);
        true
    }

    fn agrees(&self, func: &Function, block: Block, args: &[Value]) -> bool {
        func.blocks[block.index()]
            .params
            .iter()
            .zip(args)
            .all(|(param, arg)| {
                self.subst.get(param.index()).is_none_or(|&s| s == *param)
                    || self.find(*arg) == self.find(*param)
            })
    }

    fn step(
        &self,
        func: &Function,
        (uses, pinned): (&[u32], &[bool]),
        block: Block,
        args: &[Value],
    ) -> Option<(Block, Vec<Value>)> {
        let data = &func.blocks[block.index()];
        let (compare, last) = match data.insts[..] {
            [only] => (None, only),
            [compare, last] => (Some(compare), last),
            _ => return None,
        };
        if !self.agrees(func, block, args) {
            return None;
        }
        let op = &func.insts[last.index()].op;
        let param = |value: &Value| data.params.iter().position(|param| param == value);
        let (next, next_args) = match op {
            Op::Jump(next, next_args) if compare.is_none() => (*next, next_args),
            Op::Brif(condition, then, then_args, otherwise, else_args) => match self
                .edge_condition(func, (compare, uses), *condition, &|value| {
                    param(&value).map(|at| args[at])
                })? {
                0 => (*otherwise, else_args),
                _ => (*then, then_args),
            },
            _ => return None,
        };
        let reads = |param: Value| {
            let mut count = 0_u32;
            compare
                .iter()
                .map(|inst| &func.insts[inst.index()].op)
                .chain([op])
                .for_each(|op| op.for_operands(|value| count += u32::from(value == param)));
            count
        };
        if pinned[next.index()]
            || data
                .params
                .iter()
                .any(|param| uses[param.index()] != reads(*param))
        {
            return None;
        }
        Some((
            next,
            next_args
                .iter()
                .map(|value| param(value).map_or(*value, |at| args[at]))
                .collect(),
        ))
    }

    fn edge_condition(
        &self,
        func: &Function,
        (compare, uses): (Option<Inst>, &[u32]),
        condition: Value,
        arg: &dyn Fn(Value) -> Option<Value>,
    ) -> Option<i64> {
        let Some(compare) = compare else {
            return self.known(arg(condition)?);
        };
        let data = &func.insts[compare.index()];
        let (Op::Icmp(cc, x, y), Some(result)) = (&data.op, data.result) else {
            return None;
        };
        if result != condition || !func.ty(*x).is_small_int() {
            return None;
        }
        let value = |operand: Value| {
            arg(operand).map_or_else(|| self.known(operand), |given| self.known(given))
        };
        let (vx, vy) = (value(*x)?, value(*y)?);
        if uses[result.index()] != 1 {
            return None;
        }
        Some(i64::from(holds(*cc, func.ty(*x), vx, vy)))
    }

    pub(super) fn params(&mut self, func: &Function, reachable: &[bool], pinned: &[bool]) -> bool {
        let (mut any, mut multi) = (false, false);
        for (b, data) in func.blocks.iter().enumerate() {
            if reachable[b] && !pinned[b] && !data.params.is_empty() {
                any = true;
                multi |= data.params.len() > 1;
            }
        }
        if !any {
            return false;
        }
        if self.incoming.is_empty() {
            (self.incoming, self.mixed) =
                (vec![None; func.types.len()], vec![false; func.types.len()]);
        }
        for param in func.blocks.iter().flat_map(|block| &block.params) {
            (self.incoming[param.index()], self.mixed[param.index()]) = (None, false);
        }
        let mut into: Vec<Vec<&[Value]>> = if multi {
            vec![Vec::new(); func.blocks.len()]
        } else {
            Vec::new()
        };
        for (block, _) in reachable.iter().enumerate().filter(|(_, live)| **live) {
            for_edges(func, block, |target, args| {
                if multi && func.blocks[target.index()].params.len() > 1 {
                    into[target.index()].push(args);
                }
                for (param, arg) in func.blocks[target.index()].params.iter().zip(args) {
                    let arg = self.find(*arg);
                    match self.incoming[param.index()] {
                        _ if arg == *param => {}
                        None => self.incoming[param.index()] = Some(arg),
                        Some(seen) if seen != arg => self.mixed[param.index()] = true,
                        Some(_) => {}
                    }
                }
            });
        }
        let mut changed = false;
        for (block, data) in func
            .blocks
            .iter()
            .enumerate()
            .filter(|(block, _)| reachable[*block] && !pinned[*block])
        {
            for param in &data.params {
                if let (Some(value), false) =
                    (self.incoming[param.index()], self.mixed[param.index()])
                    && self.subst[param.index()] == *param
                {
                    (self.subst[param.index()], changed) = (value, true);
                }
            }
            if multi && data.params.len() > 1 {
                changed |= self.twins(func, &data.params, &into[block]);
            }
        }
        changed
    }

    fn twins(&mut self, func: &Function, params: &[Value], edges: &[&[Value]]) -> bool {
        let mut changed = false;
        for later in 1..params.len() {
            let twin = params[later];
            if edges.is_empty() || self.subst[twin.index()] != twin {
                continue;
            }
            let same = |at: usize| {
                let first = params[at];
                self.subst[first.index()] == first
                    && func.ty(first) == func.ty(twin)
                    && edges
                        .iter()
                        .all(|args| match (args.get(at), args.get(later)) {
                            (Some(x), Some(y)) => {
                                let pair = (self.find(*x), self.find(*y));
                                pair.0 == pair.1
                                    || pair == (first, twin)
                                    || self.stepped(func, pair, (first, twin))
                            }
                            _ => false,
                        })
            };
            if let Some(at) = (0..later).find(|&at| same(at)) {
                self.subst[twin.index()] = params[at];
                changed = true;
            }
        }
        changed
    }

    fn stepped(
        &self,
        func: &Function,
        (x, y): (Value, Value),
        (first, twin): (Value, Value),
    ) -> bool {
        let (Some(&Op::Binary(op, a, c)), Some(&Op::Binary(op2, b, d))) =
            (self.def(func, x), self.def(func, y))
        else {
            return false;
        };
        let (a, b, c, d) = (self.find(a), self.find(b), self.find(c), self.find(d));
        op == op2 && ((a, b) == (first, twin) && c == d || (c, d) == (first, twin) && a == b)
    }

    pub(super) fn entry_guards(
        &mut self,
        func: &mut Function,
        reachable: &[bool],
        pinned: &[bool],
    ) -> bool {
        if func.blocks.len() <= 1 {
            return false;
        }
        let (entry_idx, env) = (
            func.entry.map_or(usize::MAX, |e| e.index()),
            crate::cse::CmpEnv {
                defs: &self.defs,
                consts: &self.consts,
                subst: &self.subst,
                bool_val: None,
            },
        );
        let has_cmp = |b: usize| {
            b != entry_idx
                && reachable[b]
                && !pinned[b]
                && func.blocks[b]
                    .insts
                    .iter()
                    .any(|&inst| match func.insts[inst.index()].op {
                        Op::Trapz(c) => extract_cmp(func, &env, c).is_some(),
                        Op::Brif(c, tb, _, eb, _) if tb != eb => {
                            extract_cmp(func, &env, c).is_some()
                        }
                        _ => false,
                    })
        };
        let want: Vec<bool> = (0..func.blocks.len()).map(has_cmp).collect();
        if !want.iter().any(|&w| w) {
            return false;
        }
        let (cfg, mut edits) = (func.cfg(), Vec::new());
        {
            let mut incoming: Vec<Vec<(Block, &[Value])>> = vec![Vec::new(); func.blocks.len()];
            for (b, _) in reachable.iter().enumerate().filter(|(_, live)| **live) {
                for_edges(func, b, |t, a| {
                    if want[t.index()] {
                        incoming[t.index()].push((Block(b as u32), a));
                    }
                });
            }
            for (b, edges) in incoming.iter().enumerate() {
                if !want[b] || edges.is_empty() {
                    continue;
                }
                let (block, params, insts) = (
                    Block(b as u32),
                    &func.blocks[b].params,
                    &func.blocks[b].insts,
                );
                for &inst in insts {
                    let (cond, is_brif) = match func.insts[inst.index()].op {
                        Op::Trapz(c) => (c, false),
                        Op::Brif(c, tb, _, eb, _) if tb != eb => (c, true),
                        _ => continue,
                    };
                    let Some((cc, ty, x, y)) = extract_cmp(func, &env, cond) else {
                        continue;
                    };
                    let (vx, vy) = (env.find(x), env.find(y));
                    let local = |v: Value| {
                        env.defs
                            .get(v.index())
                            .copied()
                            .flatten()
                            .is_some_and(|di| insts.contains(&di))
                    };
                    let (lx, ly) = (local(vx), local(vy));
                    let same_across_edges = edges.len() >= 2
                        && !lx
                        && !ly
                        && !params.contains(&vx)
                        && !params.contains(&vy);
                    let idom_block = cfg.idom[block.index()].filter(|&d| d != block);
                    let check = |wc: IntCC| {
                        if same_across_edges
                            && let Some(dom) = idom_block
                            && let (Some(mx), Some(my)) = (
                                entry_map_val(&env, params, &[], vx, false),
                                entry_map_val(&env, params, &[], vy, false),
                            )
                            && crate::cse::dom_proves_want(
                                func,
                                &cfg,
                                &env,
                                dom,
                                Some(block),
                                ty,
                                (wc, mx, my),
                            )
                        {
                            return true;
                        }
                        edges.iter().all(|&(pred, args)| {
                            entry_map_val(&env, params, args, vx, lx)
                                .zip(entry_map_val(&env, params, args, vy, ly))
                                .is_some_and(|(mx, my)| {
                                    edge_proves_want(
                                        func,
                                        &cfg,
                                        &env,
                                        (pred, block),
                                        ty,
                                        (wc, mx, my),
                                    )
                                })
                        })
                    };
                    if let Some(t) = check(cc)
                        .then_some(true)
                        .or_else(|| check(cc.inverse()).then_some(false))
                    {
                        edits.push((
                            b,
                            inst,
                            if is_brif {
                                Some(t)
                            } else if t {
                                None
                            } else {
                                Some(false)
                            },
                        ));
                    }
                }
            }
        }
        if edits.is_empty() {
            return false;
        }
        for (b, inst, kind) in edits {
            match (
                kind,
                std::mem::replace(&mut func.insts[inst.index()].op, Op::Trap),
            ) {
                (Some(t), Op::Brif(_, tb, ta, eb, ea)) => {
                    func.insts[inst.index()].op =
                        Op::Jump(if t { tb } else { eb }, if t { ta } else { ea });
                }
                (None, Op::Trapz(_)) => func.blocks[b].insts.retain(|&i| i != inst),
                _ => {}
            }
        }
        true
    }
}

fn edge_proves_want(
    func: &Function,
    cfg: &crate::cfg::Cfg,
    env: &crate::cse::CmpEnv<'_>,
    (pred, block): (Block, Block),
    ty: Type,
    want: (IntCC, crate::cse::EdgeVal, crate::cse::EdgeVal),
) -> bool {
    if let (crate::cse::EdgeVal::Const(a), crate::cse::EdgeVal::Const(b)) = (want.1, want.2) {
        return holds(want.0, ty, a, b);
    }
    if let Some(&Op::Brif(c, tb, _, eb, _)) = func.last(pred)
        && tb != eb
        && (tb == block || eb == block)
        && crate::cse::cond_proves_want(func, env, (c, tb == block, 0), ty, want)
    {
        return true;
    }
    crate::cse::dom_proves_want(func, cfg, env, pred, Some(block), ty, want)
}

fn extract_cmp(
    func: &Function,
    env: &crate::cse::CmpEnv<'_>,
    cond: Value,
) -> Option<(IntCC, Type, Value, Value)> {
    let (op, taken, _) = env.peel_cond(func, cond, true, 0)?;
    let &Op::Icmp(cc, x, y) = op else {
        return None;
    };
    func.ty(x)
        .is_small_int()
        .then_some((if taken { cc } else { cc.inverse() }, func.ty(x), x, y))
}

fn entry_map_val(
    env: &crate::cse::CmpEnv<'_>,
    params: &[Value],
    args: &[Value],
    v: Value,
    is_local: bool,
) -> Option<crate::cse::EdgeVal> {
    if let Some(k) = params.iter().position(|&p| p == v) {
        return Some(env.edge_val(args[k]));
    }
    env.consts
        .get(v.index())
        .copied()
        .flatten()
        .map(crate::cse::EdgeVal::Const)
        .or_else(|| (!is_local).then_some(crate::cse::EdgeVal::Val(v)))
}
