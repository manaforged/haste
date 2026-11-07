use hir_ty::db::HirDatabase;
use hir_ty::attach_db;

const COMPILE_STACK: usize = 1 << 30;

type CompileJob = (u64, std::sync::mpsc::SyncSender<()>);

static COMPILER: std::sync::Mutex<Option<std::sync::mpsc::Sender<CompileJob>>> = std::sync::Mutex::new(None);

fn compiler() -> std::sync::mpsc::Sender<CompileJob> {
    let mut compiler = COMPILER.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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

type Held = (
    std::sync::MutexGuard<'static, Option<std::sync::mpsc::Sender<CompileJob>>>,
    std::sync::MutexGuard<'static, Option<crate::patch::Engine>>,
);

std::thread_local! {
    static HELD: std::cell::RefCell<Option<Held>> = const { std::cell::RefCell::new(None) };
}

extern "C" fn before_fork() {
    let compiler = COMPILER.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let engine = crate::patch::ENGINE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    HELD.set(Some((compiler, engine)));
}

extern "C" fn after_fork_parent() {
    HELD.take();
}

extern "C" fn after_fork_child() {
    if let Some((mut compiler, _engine)) = HELD.take() {
        std::mem::forget(compiler.take());
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
    let mut guard = crate::patch::ENGINE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let engine = guard.as_mut().expect("engine is running");
    let (db, backend) = engine.parts();
    let db: &dyn HirDatabase = db;
    if !backend.digests.contains_key(&(slot as usize)) {
        attach_db(db, || backend.compile(db, slot as usize));
    }
}
