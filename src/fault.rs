use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Mutex;

const FRAMES: usize = 24;
const SIGNALS: [i32; 6] = [libc::SIGABRT, libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGTRAP, libc::SIGALRM];
const LINE_BYTES: usize = 256;
const STACK_BYTES: usize = 1 << 18;

pub use haste_codegen::Site;

struct Code {
    start: usize,
    size: usize,
    name: String,
    sites: Vec<Site>,
}

/// What owns a return address.
pub enum Frame {
    Handler { pc: usize, frame_offset: usize },
    Generated,
    Native(String),
}

static CODE: Mutex<BTreeMap<usize, Code>> = Mutex::new(BTreeMap::new());

const PERF_MAP: &str = "HASTE_PERF_MAP";

fn perf_map_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(PERF_MAP).is_some())
}

fn append_perf_map(items: &[&Code]) {
    use std::io::Write;
    let path = format!("/tmp/perf-{}.map", std::process::id());
    let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) else { return };
    let text: String = items.iter().map(|item| format!("{:x} {:x} {}\n", item.start, item.size, item.name)).collect();
    let _ = file.write_all(text.as_bytes());
}

pub fn remember(start: usize, size: usize, name: String, sites: Vec<Site>) {
    let item = Code { start, size, name, sites };
    if perf_map_on() {
        append_perf_map(&[&item]);
    }
    CODE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(start, item);
}

/// Writes every generated function to this process's perf map when `HASTE_PERF_MAP` is set, for a forked child.
pub fn write_perf_map() {
    if !perf_map_on() {
        return;
    }
    let code = CODE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    append_perf_map(&code.values().collect::<Vec<_>>());
}

pub fn frame_at(ret: usize) -> Frame {
    let code = CODE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(item) = owner(&code, ret) else {
        let name = host_name(ret).map(|name| String::from_utf8_lossy(name).into_owned());
        return Frame::Native(name.unwrap_or_else(|| format!("{ret:#x}")));
    };
    let offset = ret - item.start;
    match item.sites.iter().find(|site| site.ret as usize == offset) {
        Some(site) => Frame::Handler { pc: item.start + site.handler as usize, frame_offset: site.frame_offset as usize },
        None => Frame::Generated,
    }
}

fn owner(code: &BTreeMap<usize, Code>, pc: usize) -> Option<&Code> {
    code.range(..=pc).next_back().map(|(_, item)| item).filter(|item| pc < item.start + item.size)
}

struct Line {
    bytes: [u8; LINE_BYTES],
    len: usize,
}

impl std::fmt::Write for Line {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let room = (LINE_BYTES - 1 - self.len).min(text.len());
        self.bytes[self.len..self.len + room].copy_from_slice(&text.as_bytes()[..room]);
        self.len += room;
        Ok(())
    }
}

impl Line {
    fn emit(&mut self, text: &[u8]) {
        let room = (LINE_BYTES - 1 - self.len).min(text.len());
        self.bytes[self.len..self.len + room].copy_from_slice(&text[..room]);
        self.len += room;
        self.bytes[self.len] = b'\n';
        // SAFETY: write reads len + 1 initialized bytes of this stack buffer and is async-signal-safe.
        unsafe { libc::write(libc::STDERR_FILENO, self.bytes.as_ptr().cast(), self.len + 1) };
        self.len = 0;
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn registers(context: *mut libc::c_void) -> (usize, usize) {
    // SAFETY: context is the ucontext the kernel passed to an SA_SIGINFO handler.
    let state = unsafe { (*(*context.cast::<libc::ucontext_t>()).uc_mcontext).__ss };
    (state.__pc as usize, state.__fp as usize)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn registers(context: *mut libc::c_void) -> (usize, usize) {
    // SAFETY: context is the ucontext the kernel passed to an SA_SIGINFO handler.
    let state = unsafe { &(*context.cast::<libc::ucontext_t>()).uc_mcontext };
    (state.gregs[libc::REG_RIP as usize] as usize, state.gregs[libc::REG_RBP as usize] as usize)
}

fn host_name(pc: usize) -> Option<&'static [u8]> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: dladdr only reads pc as a number and fills the local info.
    if unsafe { libc::dladdr(pc as *const libc::c_void, &mut info) } == 0 {
        return None;
    }
    let text = if info.dli_sname.is_null() { info.dli_fname } else { info.dli_sname };
    // SAFETY: dladdr filled text with a nul-terminated name that lives as long as the image, or left it null.
    (!text.is_null()).then(|| unsafe { std::ffi::CStr::from_ptr(text) }.to_bytes())
}

extern "C" fn on_fault(signal: i32, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // SAFETY: the kernel passes a valid siginfo and ucontext to an SA_SIGINFO handler.
    let (fault, (start_pc, start_frame)) = unsafe { ((*info).si_addr() as usize, registers(context)) };
    let mut line = Line { bytes: [0; LINE_BYTES], len: 0 };
    let _ = write!(line, "fault: signal {signal} address {fault:#x}");
    line.emit(b"");
    if let Ok(code) = CODE.try_lock() {
        let (mut pc, mut frame) = (start_pc, start_frame);
        for _ in 0..FRAMES {
            if let Some(item) = owner(&code, pc) {
                let _ = line.write_str("  at ");
                line.emit(item.name.as_bytes());
            } else if let Some(name) = host_name(pc) {
                let _ = line.write_str("  in ");
                line.emit(name);
            }
            if frame == 0 || frame % 8 != 0 {
                break;
            }
            // SAFETY: generated code and the host keep frame pointers, so frame points at a saved (frame, return address) pair; a bad walk ends in the default handler below.
            let (next, ret) = unsafe { (*(frame as *const usize), *((frame + 8) as *const usize)) };
            if next <= frame {
                break;
            }
            (pc, frame) = (ret, next);
        }
    }
    // SAFETY: the default action is restored and the signal raised again, so it ends the process with its normal status once the handler returns.
    unsafe {
        libc::signal(signal, libc::SIG_DFL);
        libc::raise(signal);
    }
}

pub fn install() {
    let stack = Box::leak(vec![0_u8; STACK_BYTES].into_boxed_slice());
    let alternate = libc::stack_t { ss_sp: stack.as_mut_ptr().cast(), ss_flags: 0, ss_size: STACK_BYTES };
    // SAFETY: the alternate stack is leaked, and every pointer passed to the kernel is valid for the call.
    unsafe {
        libc::sigaltstack(&alternate, std::ptr::null_mut());
        for signal in SIGNALS {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_fault as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
            libc::sigaction(signal, &action, std::ptr::null_mut());
        }
    }
}
