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

pub fn forget_compiler() {
    std::mem::forget(COMPILER.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take());
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
