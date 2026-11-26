use std::collections::VecDeque;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::Result as ThreadResult;

use super::lock;

type Job = Box<dyn FnOnce() -> bool + Send + 'static>;

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    waiting: usize,
}

struct Pool {
    pid: u32,
    queue: Mutex<Queue>,
    ready: Condvar,
}

static POOL: OnceLock<Pool> = OnceLock::new();

fn pool() -> &'static Pool {
    POOL.get_or_init(|| Pool { pid: std::process::id(), queue: Mutex::default(), ready: Condvar::new() })
}

impl Pool {
    /// Queues `job`, waking an idle worker, or starting one when every worker is busy, so each job starts at once as a spawned thread would; a forked child, which has none of the workers, gives each job a thread of its own.
    fn submit(&'static self, job: Job) {
        if std::process::id() != self.pid {
            std::thread::spawn(job);
            return;
        }
        let mut queue = lock(&self.queue);
        queue.jobs.push_back(job);
        if queue.jobs.len() > queue.waiting {
            drop(queue);
            std::thread::spawn(move || self.work());
        } else {
            drop(queue);
            self.ready.notify_one();
        }
    }

    /// Runs queued jobs until one panics, which ends the thread so no thread-local state a panic left behind reaches a later job.
    fn work(&'static self) {
        let mut queue = lock(&self.queue);
        loop {
            if let Some(job) = queue.jobs.pop_front() {
                drop(queue);
                if !job() {
                    return;
                }
                queue = lock(&self.queue);
                continue;
            }
            queue.waiting += 1;
            queue = self.ready.wait(queue).unwrap_or_else(std::sync::PoisonError::into_inner);
            queue.waiting -= 1;
        }
    }
}

#[derive(Default)]
struct State {
    running: usize,
    panics: usize,
    joined_panics: usize,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    done: Condvar,
}

/// A scope whose spawned closures run on the shared pool's threads and all finish before `scope` returns, as with `std::thread::scope`.
pub(crate) struct Scope<'scope, 'env: 'scope> {
    shared: Arc<Shared>,
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
}

struct Slot<T> {
    value: Mutex<Option<ThreadResult<T>>>,
    filled: Condvar,
}

/// The handle to one spawned closure, joined for its value or its panic payload.
pub(crate) struct Handle<'scope, T> {
    slot: Arc<Slot<T>>,
    shared: Arc<Shared>,
    scope: PhantomData<&'scope ()>,
}

impl<'scope, T> Handle<'scope, T> {
    /// Waits for the closure and returns its value, or the payload of its panic.
    pub(crate) fn join(self) -> ThreadResult<T> {
        let mut value = lock(&self.slot.value);
        let result = loop {
            if let Some(result) = value.take() {
                break result;
            }
            value = self.slot.filled.wait(value).unwrap_or_else(std::sync::PoisonError::into_inner);
        };
        if result.is_err() {
            lock(&self.shared.state).joined_panics += 1;
        }
        result
    }
}

impl<'scope, 'env> Scope<'scope, 'env> {
    /// Runs `f` on a pool thread within this scope.
    pub(crate) fn spawn<F, T>(&'scope self, f: F) -> Handle<'scope, T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        let slot = Arc::new(Slot { value: Mutex::new(None), filled: Condvar::new() });
        lock(&self.shared.state).running += 1;
        let (job_slot, shared) = (Arc::clone(&slot), Arc::clone(&self.shared));
        let job = move || {
            let result = catch_unwind(AssertUnwindSafe(f));
            let ok = result.is_ok();
            *lock(&job_slot.value) = Some(result);
            job_slot.filled.notify_all();
            drop(job_slot);
            let mut state = lock(&shared.state);
            state.running -= 1;
            state.panics += usize::from(!ok);
            drop(state);
            shared.done.notify_all();
            ok
        };
        let job: Box<dyn FnOnce() -> bool + Send + 'scope> = Box::new(job);
        // SAFETY: `scope` waits until `running` is zero before it returns or unwinds, and a job lowers `running` only after it dropped `f` and its own hold on the slot, so nothing borrowed for 'scope outlives the scope.
        let job: Job = unsafe { std::mem::transmute::<Box<dyn FnOnce() -> bool + Send + 'scope>, Job>(job) };
        pool().submit(job);
        Handle { slot, shared: Arc::clone(&self.shared), scope: PhantomData }
    }
}

struct Wait(Arc<Shared>);

impl Drop for Wait {
    fn drop(&mut self) {
        let mut state = lock(&self.0.state);
        while state.running > 0 {
            state = self.0.done.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

/// Runs `f` with a scope for spawning borrowing closures onto the shared pool, waits for all of them, and panics when one that was not joined panicked.
pub(crate) fn scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
{
    let scope = Scope { shared: Arc::new(Shared::default()), scope: PhantomData, env: PhantomData };
    let wait = Wait(Arc::clone(&scope.shared));
    let result = catch_unwind(AssertUnwindSafe(|| f(&scope)));
    drop(wait);
    let unjoined_panic = {
        let state = lock(&scope.shared.state);
        state.panics > state.joined_panics
    };
    match result {
        Err(payload) => resume_unwind(payload),
        Ok(_) if unjoined_panic => panic!("a scoped thread panicked"),
        Ok(value) => value,
    }
}
