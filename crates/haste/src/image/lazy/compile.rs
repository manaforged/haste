use std::sync::atomic::Ordering;

use rustc_hash::FxHashSet as HashSet;

use crate::image::global_asm::Foreign;
use crate::image::{perf, stub};
use crate::lower::Slots;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::{Instance, calls, infer_declared, infer_quiet};

use super::Jit;

pub(super) enum Stop {
    Asked,
    Failed(String, ItemRef),
}

type Journal = Vec<usize>;

const AHEAD_FUNCTIONS: usize = 64;

impl Jit {
    pub(super) fn answer(&mut self, cx: &mut Context<'_>, index: usize) -> Result<usize, Stop> {
        if index & (stub::TIER_UP_BIT as usize) != 0 {
            let real = index & !(stub::TIER_UP_BIT as usize);
            let item = self.reached.list[real].0.0;
            let code = self
                .image
                .tier_up(real)
                .map_err(|err| Stop::Failed(format!("{}: {err:#}", cx.name(item)), item))?;
            perf::note(cx, &self.reached, (real, code), perf::Origin::Haste);
            return Ok(code);
        }
        let mut journal = Journal::new();
        let answer = self.code(cx, index, &mut journal);
        match answer {
            Err(Stop::Asked) => {
                for &at in &journal {
                    self.reached.forget(at);
                    self.expanded.remove(&at);
                }
                self.tried.clear();
                Err(Stop::Asked)
            }
            Ok((code, fresh)) => {
                if fresh {
                    self.compile_ahead(cx, index);
                }
                Ok(code)
            }
            Err(err) => Err(err),
        }
    }

    fn compile_ahead(&mut self, cx: &mut Context<'_>, index: usize) {
        if !self.ahead {
            return;
        }
        let members = self.frontier(cx, index);
        let (foreign, mut work): (Vec<usize>, Vec<usize>) = members
            .into_iter()
            .filter(|&at| self.callees_typed(cx, at))
            .partition(|&at| cx.foreign(self.reached.list[at].0.0));
        for at in foreign {
            let reached = &self.reached;
            match self.image.with_pass(|image, pass| {
                image.bind_foreign(cx, reached, at, Some(&mut pass.memory))
            }) {
                Ok(Foreign::Asm(code) | Foreign::Host(_, code)) => {
                    self.publish(at, code);
                }
                Err(_) => {
                    self.tried.insert(at);
                }
            }
        }
        work.retain(|&at| crate::prep::bind(cx, &self.reached, at).is_none());
        if work.is_empty() {
            return;
        }
        let compiled = self.image.compile_batch(cx, &self.reached, &work);
        for &(at, code) in &compiled {
            perf::note(cx, &self.reached, (at, code), perf::Origin::Haste);
            self.publish(at, code);
        }
        self.tried
            .extend(work.into_iter().filter(|at| !self.compiled.contains(at)));
    }

    fn frontier(&mut self, cx: &mut Context<'_>, index: usize) -> Vec<usize> {
        let mut seen = HashSet::default();
        seen.insert(index);
        let mut level = self.fresh_callees(cx, index, &mut seen);
        if level.is_empty() {
            return level;
        }
        level.truncate(AHEAD_FUNCTIONS);
        let inferred = infer_quiet(cx, &mut self.reached, &level);
        self.image
            .bind_foreign_statics(cx, &self.reached, &inferred);
        let (mut typed, left): (Vec<usize>, Vec<usize>) = level
            .into_iter()
            .partition(|&at| self.reached.list[at].1.is_ok());
        self.tried.extend(left);
        if !typed.is_empty() && typed.len() < 16 && typed.len() < AHEAD_FUNCTIONS {
            let cap = AHEAD_FUNCTIONS - typed.len();
            let mut hop2 = Vec::new();
            for &at in &typed {
                if hop2.len() >= cap {
                    break;
                }
                for callee in self.fresh_callees(cx, at, &mut seen) {
                    hop2.push(callee);
                    if hop2.len() >= cap {
                        break;
                    }
                }
            }
            if !hop2.is_empty() {
                let inferred2 = infer_quiet(cx, &mut self.reached, &hop2);
                self.image
                    .bind_foreign_statics(cx, &self.reached, &inferred2);
                let (typed2, left2): (Vec<usize>, Vec<usize>) = hop2
                    .into_iter()
                    .partition(|&at| self.reached.list[at].1.is_ok());
                self.tried.extend(left2);
                typed.extend(typed2);
            }
        }
        let mut deps = Vec::new();
        for &at in &typed {
            for callee in calls(cx, &self.reached.list[at].1).collect::<Vec<_>>() {
                let id = self.declare(callee);
                if self.reached.is_untyped(id) && seen.insert(id) {
                    deps.push(id);
                }
            }
        }
        if !deps.is_empty() {
            let kept = infer_quiet(cx, &mut self.reached, &deps);
            self.image.bind_foreign_statics(cx, &self.reached, &kept);
        }
        typed
    }

    fn fresh_callees(
        &mut self,
        cx: &Context<'_>,
        at: usize,
        seen: &mut HashSet<usize>,
    ) -> Vec<usize> {
        let fast_lazy = self.image.is_fast_lazy();
        let callees: Vec<Instance> = calls(cx, &self.reached.list[at].1).collect();
        let mut fresh = Vec::new();
        for callee in callees {
            if cx.has_attr(callee.0, "cold")
                || (fast_lazy && (cx.outline.is_sysroot(callee.0.krate) || cx.foreign(callee.0)))
            {
                continue;
            }
            let callee = self.declare(callee);
            if !self.compiled.contains(&callee)
                && !self.tried.contains(&callee)
                && seen.insert(callee)
            {
                fresh.push(callee);
            }
        }
        fresh
    }

    fn callees_typed(&self, cx: &Context<'_>, at: usize) -> bool {
        calls(cx, &self.reached.list[at].1).all(|callee| {
            self.reached
                .index
                .get(&callee)
                .is_some_and(|&callee| !self.reached.is_untyped(callee))
        })
    }

    fn code(
        &mut self,
        cx: &mut Context<'_>,
        index: usize,
        journal: &mut Journal,
    ) -> Result<(usize, bool), Stop> {
        let item = self.reached.list[index].0.0;
        if cx.foreign(item) {
            return self
                .foreign_code(cx, index, journal)
                .map(|code| (code, false));
        }
        if let Some(code) = self.prepped(cx, index, journal) {
            let origin = if crate::prep::bind(cx, &self.reached, index) == Some(code) {
                perf::Origin::Rustc
            } else {
                perf::Origin::Adapter
            };
            perf::note(cx, &self.reached, (index, code), origin);
            return Ok((self.publish(index, code), false));
        }
        let mut callees = self.infer(cx, index, journal)?;
        let mut fresh = false;
        while !self.compiled.contains(&index) {
            match self.image.compile_one(cx, &self.reached, index) {
                Ok((code, reused)) => {
                    perf::note(cx, &self.reached, (index, code), perf::Origin::Haste);
                    self.publish(index, code);
                    fresh = !reused;
                }
                Err(error) => {
                    let wanted = self.image.take_wanted();
                    if wanted.is_empty() {
                        return Err(self.stop(cx, self.failure(cx, index, &callees, &error), item));
                    }
                    callees.extend(self.reach(cx, wanted, journal)?);
                }
            }
        }
        Ok((self.image.slot(index).load(Ordering::Acquire), fresh))
    }

    fn foreign_code(
        &mut self,
        cx: &mut Context<'_>,
        index: usize,
        journal: &mut Journal,
    ) -> Result<usize, Stop> {
        let item = self.reached.list[index].0.0;
        let wanted: Vec<Instance> = calls(cx, &self.reached.list[index].1).collect();
        if !wanted.is_empty() {
            self.reach(cx, wanted, journal)?;
        }
        let reached = &self.reached;
        let bound = self.image.with_pass(|image, pass| {
            image.bind_foreign(cx, reached, index, Some(&mut pass.memory))
        });
        match bound.map_err(|error| Stop::Failed(error, item))? {
            Foreign::Asm(code) | Foreign::Host(_, code) => Ok(self.publish(index, code)),
        }
    }

    fn publish(&mut self, index: usize, code: usize) -> usize {
        self.image.publish(index, code);
        self.compiled.insert(index);
        code
    }

    fn prepped(
        &mut self,
        cx: &mut Context<'_>,
        index: usize,
        journal: &mut Journal,
    ) -> Option<usize> {
        if infer_declared(cx, &mut self.reached, index) {
            journal.push(index);
        }
        match crate::prep::prepped(cx, &self.reached, index)? {
            crate::prep::Prepped::Direct(address) => Some(address),
            crate::prep::Prepped::Adapter(func) => {
                let code = haste_codegen::compile(func, &crate::host_target()).ok()?;
                self.image
                    .with_pass(|_, pass| pass.memory.write_patched(&code, &[]))
                    .ok()
            }
        }
    }

    fn infer(
        &mut self,
        cx: &mut Context<'_>,
        index: usize,
        journal: &mut Journal,
    ) -> Result<Vec<usize>, Stop> {
        if infer_declared(cx, &mut self.reached, index) {
            journal.push(index);
        }
        if let Err(error) = &self.reached.list[index].1 {
            return Err(self.stop(cx, error.clone(), self.reached.list[index].0.0));
        }
        let wanted: Vec<Instance> = calls(cx, &self.reached.list[index].1).collect();
        let mut callees = self.reach(cx, wanted, journal)?;
        if self.image.allow_inline() {
            let mut seen: HashSet<usize> = callees.iter().copied().collect();
            let mut next = 0;
            while let Some(&at) = callees.get(next) {
                next += 1;
                if !self.expanded.insert(at) || !crate::lower::may_inline(cx, &self.reached, at) {
                    continue;
                }
                let inner: Vec<Instance> = calls(cx, &self.reached.list[at].1).collect();
                let reached = self.reach(cx, inner, journal)?;
                callees.extend(reached.into_iter().filter(|callee| seen.insert(*callee)));
            }
        }
        Ok(callees)
    }

    fn reach(
        &mut self,
        cx: &mut Context<'_>,
        wanted: Vec<Instance>,
        journal: &mut Journal,
    ) -> Result<Vec<usize>, Stop> {
        let mut callees = Vec::with_capacity(wanted.len());
        let mut untyped = Vec::new();
        for instance in wanted {
            let at = self.declare(instance);
            callees.push(at);
            if self.reached.is_untyped(at) {
                untyped.push(at);
            }
        }
        if untyped.len() >= 4 && crate::jobs() > 1 {
            let kept = infer_quiet(cx, &mut self.reached, &untyped);
            if !kept.is_empty() {
                self.image.bind_foreign_statics(cx, &self.reached, &kept);
                journal.extend(kept);
            }
        }
        for at in untyped {
            if infer_declared(cx, &mut self.reached, at) {
                journal.push(at);
                self.image.bind_foreign_statics(cx, &self.reached, &[at]);
            }
        }
        if self.strict && !cx.resolved() {
            return Err(Stop::Asked);
        }
        Ok(callees)
    }

    fn failure(
        &self,
        cx: &Context<'_>,
        index: usize,
        callees: &[usize],
        error: &anyhow::Error,
    ) -> String {
        let broken = callees
            .iter()
            .filter_map(|&at| self.reached.list[at].1.as_ref().err());
        std::iter::once(format!(
            "{}: {error:#}",
            cx.name(self.reached.list[index].0.0)
        ))
        .chain(broken.cloned())
        .collect::<Vec<_>>()
        .join("; ")
    }

    fn stop(&self, cx: &Context<'_>, error: String, item: ItemRef) -> Stop {
        if self.strict && !cx.resolved() {
            Stop::Asked
        } else {
            Stop::Failed(error, item)
        }
    }
}
