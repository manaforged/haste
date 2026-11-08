use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock};

use hir_ty::attach_db;
use hir_ty::db::HirDatabase;
use ide_db::RootDatabase;

use super::Backend;

const COMPILE_STACK: usize = 1 << 30;

const WORKERS: usize = 6;

const AHEAD: u8 = 2;

type CompileJob = (u64, std::sync::mpsc::SyncSender<()>);

static COMPILER: Mutex<Option<std::sync::mpsc::Sender<CompileJob>>> = Mutex::new(None);

struct Ahead {
    db: RootDatabase,
    backend: Arc<Backend>,
    slots: std::ops::Range<usize>,
    depth: u8,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Ahead>,
    started: bool,
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue { jobs: VecDeque::new(), started: false });

static READY: Condvar = Condvar::new();

static WORKING: RwLock<()> = RwLock::new(());

fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn compiler() -> std::sync::mpsc::Sender<CompileJob> {
    let mut compiler = locked(&COMPILER);
    compiler
        .get_or_insert_with(|| {
            let (jobs, queue) = std::sync::mpsc::channel::<CompileJob>();
            std::thread::Builder::new()
                .name("compile".to_owned())
                .stack_size(COMPILE_STACK)
                .spawn(move || {
                    for (slot, done) in queue {
                        compile_now(slot);
                        let _ = done.send(());
                    }
                })
                .expect("compile thread starts");
            jobs
        })
        .clone()
}

fn workers() -> usize {
    std::thread::available_parallelism().map_or(1, |cores| cores.get().saturating_sub(2)).min(WORKERS)
}

fn start_workers(queue: &mut Queue) {
    if queue.started {
        return;
    }
    queue.started = true;
    for _ in 0..workers() {
        let started = std::thread::Builder::new().name("ahead".to_owned()).stack_size(COMPILE_STACK).spawn(work);
        if started.is_err() {
            break;
        }
    }
}

fn work() {
    loop {
        let job = {
            let mut queue = locked(&QUEUE);
            loop {
                if let Some(job) = queue.jobs.pop_front() {
                    break job;
                }
                queue = READY.wait(queue).unwrap_or_else(PoisonError::into_inner);
            }
        };
        run_ahead(job);
    }
}

fn run_ahead(job: Ahead) {
    for slot in job.slots.clone() {
        let _working = WORKING.read().unwrap_or_else(PoisonError::into_inner);
        let first = job.backend.slot_count();
        let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let db: &dyn HirDatabase = &job.db;
            attach_db(db, || job.backend.compile_ahead(db, slot));
        }));
        if compiled.is_err() {
            return;
        }
        if job.depth < AHEAD {
            spread(&job.db, &job.backend, first..job.backend.slot_count(), job.depth + 1);
        }
    }
}

fn spread(db: &RootDatabase, backend: &Arc<Backend>, slots: std::ops::Range<usize>, depth: u8) {
    if slots.is_empty() || workers() == 0 {
        return;
    }
    let mut queue = locked(&QUEUE);
    start_workers(&mut queue);
    queue.jobs.push_back(Ahead { db: db.clone(), backend: backend.clone(), slots, depth });
    READY.notify_one();
}

/// Drops the queued work ahead and waits for the work in flight, so the database can change.
pub fn quiesce() {
    drop(std::mem::take(&mut locked(&QUEUE).jobs));
    drop(WORKING.write().unwrap_or_else(PoisonError::into_inner));
}

type Held = (
    MutexGuard<'static, Option<std::sync::mpsc::Sender<CompileJob>>>,
    MutexGuard<'static, Option<crate::patch::Engine>>,
    std::sync::RwLockWriteGuard<'static, ()>,
    MutexGuard<'static, Queue>,
);

std::thread_local! {
    static HELD: std::cell::RefCell<Option<Held>> = const { std::cell::RefCell::new(None) };
}

extern "C" fn before_fork() {
    let compiler = locked(&COMPILER);
    let engine = locked(&crate::patch::ENGINE);
    let working = WORKING.write().unwrap_or_else(PoisonError::into_inner);
    let queue = locked(&QUEUE);
    HELD.set(Some((compiler, engine, working, queue)));
}

extern "C" fn after_fork_parent() {
    HELD.take();
}

extern "C" fn after_fork_child() {
    if let Some((mut compiler, _engine, _working, mut queue)) = HELD.take() {
        std::mem::forget(compiler.take());
        std::mem::forget(std::mem::take(&mut *queue));
    }
}

pub fn watch_forks() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: the three handlers are plain functions that stay valid for the life of the process.
        let failed = unsafe { libc::pthread_atfork(Some(before_fork), Some(after_fork_parent), Some(after_fork_child)) };
        assert_eq!(failed, 0, "fork handlers registered");
    });
}

pub extern "C" fn compile_slot(slot: u64) {
    let (done, wait) = std::sync::mpsc::sync_channel(1);
    compiler().send((slot, done)).expect("compile thread is running");
    wait.recv().expect("compile thread finished the job");
}

fn compile_now(slot: u64) {
    let mut guard = locked(&crate::patch::ENGINE);
    let engine = guard.as_mut().expect("engine is running");
    let (root, backend) = engine.parts();
    let slot = slot as usize;
    if backend.compiled(slot) {
        return;
    }
    let first = backend.slot_count();
    let db: &dyn HirDatabase = root;
    attach_db(db, || backend.compile(db, slot));
    spread(root, backend, first..backend.slot_count(), 1);
}
