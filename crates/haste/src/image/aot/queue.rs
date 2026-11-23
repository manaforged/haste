use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use rustc_hash::FxHashMap as HashMap;

use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sema::ty::Kind;

struct QueueState {
    buckets: VecDeque<Vec<usize>>,
    loose: Vec<usize>,
    warm: Vec<usize>,
    active: usize,
}

pub(in crate::image) fn bucket_by_item(work: &[usize], item_of: impl Fn(usize) -> ItemRef) -> Vec<Vec<usize>> {
    let mut pos_map: HashMap<ItemRef, usize> = HashMap::default();
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    for &index in work {
        let item = item_of(index);
        if let Some(&pos) = pos_map.get(&item) {
            buckets[pos].push(index);
        } else {
            pos_map.insert(item, buckets.len());
            buckets.push(vec![index]);
        }
    }
    buckets
}

pub(in crate::image) fn lpt_chunks(buckets: Vec<Vec<usize>>, max_bucket: usize) -> Vec<Vec<usize>> {
    let mut chunks: Vec<Vec<usize>> = buckets
        .into_iter()
        .flat_map(|b| if b.len() <= max_bucket { vec![b] } else { b.chunks(max_bucket).map(|c| c.to_vec()).collect() })
        .collect();
    chunks.sort_by_key(|b| std::cmp::Reverse(b.len()));
    chunks
}

pub(in crate::image) enum AotWork<'a> {
    Compile(Vec<usize>, TaskGuard<'a>),
    Warm(usize),
}

pub(in crate::image) struct AotQueue {
    claimed: Vec<AtomicBool>,
    warmed: Vec<AtomicBool>,
    items: Box<[ItemRef]>,
    max_bucket: usize,
    state: Mutex<QueueState>,
    wake: Condvar,
}

pub(in crate::image) struct TaskGuard<'a>(&'a AotQueue);

impl Drop for TaskGuard<'_> {
    fn drop(&mut self) {
        self.0.finish_task();
    }
}

impl AotQueue {
    pub(in crate::image) fn new(claimed: Vec<AtomicBool>, items: Box<[ItemRef]>, max_bucket: usize, initial_buckets: Vec<Vec<usize>>) -> Arc<Self> {
        let warmed = (0..claimed.len()).map(|_| AtomicBool::new(false)).collect();
        Arc::new(Self {
            claimed,
            warmed,
            items,
            max_bucket,
            state: Mutex::new(QueueState { buckets: initial_buckets.into(), loose: Vec::new(), warm: Vec::new(), active: 0 }),
            wake: Condvar::new(),
        })
    }

    pub(in crate::image) fn claim(&self, idx: usize) -> bool {
        idx < self.claimed.len() && !self.claimed[idx].swap(true, Ordering::AcqRel)
    }

    pub(in crate::image) fn claim_warm(&self, idx: usize) -> bool {
        idx < self.warmed.len() && !self.warmed[idx].swap(true, Ordering::AcqRel)
    }

    pub(in crate::image) fn want_warm(&self) -> bool {
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.buckets.is_empty() && s.loose.is_empty() && s.warm.len() < 64
    }

    pub(in crate::image) fn push_warm(&self, items: Vec<usize>) {
        if !items.is_empty() {
            self.state.lock().unwrap_or_else(|e| e.into_inner()).warm.extend(items);
            self.wake.notify_all();
        }
    }

    pub(in crate::image) fn push_loose(&self, items: Vec<usize>) {
        if items.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.loose.extend(items);
        self.wake.notify_all();
    }

    pub(in crate::image) fn reseed_buckets(&self, buckets: Vec<Vec<usize>>) {
        if buckets.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.buckets.extend(buckets);
        self.wake.notify_all();
    }

    pub(in crate::image) fn pop(&self) -> Option<AotWork<'_>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(bucket) = state.buckets.pop_front() {
                state.active += 1;
                return Some(AotWork::Compile(bucket, TaskGuard(self)));
            }
            if !state.loose.is_empty() {
                let loose = std::mem::take(&mut state.loose);
                let buckets = lpt_chunks(bucket_by_item(&loose, |idx| self.items[idx]), self.max_bucket);
                state.buckets.extend(buckets);
                if state.buckets.len() > 1 {
                    self.wake.notify_all();
                }
                continue;
            }
            if state.active == 0 {
                self.wake.notify_all();
                return None;
            }
            if let Some(idx) = state.warm.pop() {
                return Some(AotWork::Warm(idx));
            }
            state = self.wake.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    pub(in crate::image) fn take_all_pending(&self) -> Vec<usize> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        while let Some(bucket) = state.buckets.pop_front() {
            out.extend(bucket);
        }
        out.append(&mut state.loose);
        out
    }

    fn finish_task(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.active = state.active.saturating_sub(1);
        if state.active == 0 && state.buckets.is_empty() && state.loose.is_empty() {
            self.wake.notify_all();
        }
    }
}

fn warmable_callee(cx: &Context<'_>, reached: &Reached, idx: usize) -> bool {
    let Some(((item, _), Ok(typed))) = reached.list.get(idx) else { return false };
    !cx.foreign(*item)
        && !cx.has_attr(*item, "cold")
        && cx.attr_value(*item, "inline") != Some("never")
        && !typed.ret.is_some_and(|ret| matches!(cx.types.kind(ret), Kind::Never))
}

pub(in crate::image) fn collect_warm_callees(cx: &Context<'_>, reached: &Reached, queue: &AotQueue, seeds: &[usize]) -> Vec<usize> {
    let mut frontier: VecDeque<(usize, u8)> = seeds.iter().map(|&s| (s, 0)).collect();
    let mut out = Vec::new();
    while let Some((u, depth)) = frontier.pop_front() {
        if depth >= 12 || out.len() >= 1024 {
            break;
        }
        for &callee in reached.direct_calls(u).into_iter().flatten() {
            if warmable_callee(cx, reached, callee) && queue.claim_warm(callee) {
                out.push(callee);
                if out.len() >= 1024 {
                    break;
                }
                frontier.push_back((callee, depth + 1));
            }
        }
    }
    out.reverse();
    out
}
